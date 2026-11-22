#!/usr/bin/env python3
"""MTPRETRAIN: retrain the qwen35 MTP draft head (blk.64) on the quantized trunk's own states (trunk frozen).

The torch block mirrors src/models/qwen35.cpp graph_mtp (v0.7.2.1):
  x  = eh_proj( [rms(e)*enorm ; rms(h)*hnorm] )           e = token embedding of t_{p+1}, h = trunk h_nextn of row p
  a  = attn_norm(x); qf = Wq a -> per head [q(256) | gate(256)]; q,k get q_norm/k_norm, NEOX rope on dims 0..63
       (freq base 1e7, imrope sections are all equal for text); GQA 24 q heads / 4 kv heads; scale 1/16
  x += Wo( attn * sigmoid(gate) );  x += down( silu(gate(rms(x)*post_norm)) * up(...) )
  hn = rms(x) * shared_head_norm  -> logits = lm_head hn; hn is also the next chain step's h.
Positions: anchor p (trunk row p, next token t_{p+1}) sits at MTP position p+1; chain step s of anchor p at p+s.
(The runtime uses p + s - 1; rope sees only position differences, so the constant shift is exact.)
Attention (as at runtime): step-1 rows attend causally to step-1 rows (catch-up K/V, trunk h); a step-s row of anchor
p attends to step-1 rows 0..p and to its own chain rows 2..s.

Targets: the trunk's sampled distribution at the predicted position: top_k 20 -> softmax -> top_p 0.95 (production
chat sampler, temp 1), from lm_head(h_nextn) of the same quantized model. Loss = soft cross-entropy of the draft
distribution over the 64K draft shortlist (atx_65536, a superset of the 24,576-token RM24H list the 1-slot 24dense
drafter scores), targets restricted to the shortlist and renormalized; step weights 0.8^(s-1).
Metric (held-out): per-step acceptance a_s = sum_x min(p(x), q(x)) (exact p/q rejection sampling), with q = draft
softmax over the RM24H (or 64K) rows cut to its top 10 and then top_p 0.95 (the draft sampler chain) and p the sampled
target (top_k 20 -> top_p 0.95, renormalized over the full vocab);
E[accepted | depth d] = mean over anchors of sum_{s<=d} prod_{j<=s} a_j, reported for d = 5 and 7 (24dense 1-slot
adaptive depth runs 3..7). All logits are fp32 (bf16 logits move by 0.125 near 16-32 and reorder near-ties).
Checkpoint selection uses the 'heldout_sel' half of the held-out prompts; the go/no-go number is 'heldout_rep'.

usage:
  mtp_train.py targets --w W0 --data D           (teacher targets for every sequence -> D/targets.npz)
  mtp_train.py eval    --w W0 --data D [--head ckpt.pt | --head-gguf overlay.gguf]
  mtp_train.py train   --w W0 --data D --out RUN [--epochs 2 --lr 3e-5 --tokens-per-step 16384 ...]
D holds seq.json, h.bin (+ .idx) from llama-mtp-dump-h, and the shortlists s64.txt / s24.txt."""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PFX = 'blk.64.'
EPS = 1e-6
N_EMBD, N_HEAD, N_KV, HD, N_ROT, ROPE_BASE = 5120, 24, 4, 256, 64, 1e7
TOPK_T, TOPP_T, TOPK_Q = 20, 0.95, 10
DEPTH = 7
STEP_W = [0.8 ** s for s in range(DEPTH)]
MATS = ['nextn.eh_proj.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
        'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']
NORMS = ['nextn.enorm.weight', 'nextn.hnorm.weight', 'attn_norm.weight', 'attn_q_norm.weight', 'attn_k_norm.weight',
         'post_attention_norm.weight', 'nextn.shared_head_norm.weight']
DEV = os.environ.get('MTP_DEV', 'cuda')
LAM_TV = float(os.environ.get('MTP_TV', '0'))
# MTP_QAT=1: every matrix is used through the exact Q5_0 / Q8_0 round trip of the export (straight-through gradient),
# so training and eval see the weights the overlay will ship (round-to-nearest export ate ~2/3 of the fp32 gain)
QAT = os.environ.get('MTP_QAT', '0') == '1'
QTYPE = {'attn_k.weight': 'q8_0', 'attn_v.weight': 'q8_0'}  # every other matrix is Q5_0 (export asserts against meta)
MAT_Q = os.environ.get('MTP_QAT_MATS', 'q5_0')  # q8_0: train for a sidecar whose Q5_0 matrices become Q8_0 (export --retype q5_0=q8_0)


def fake_quant(w, kind):
    """gguf-py reference quantize + dequantize, in torch (blocks of 32 along a row, fp16 scale)."""
    b = w.float().reshape(-1, 32)
    if kind == 'q8_0':
        d = b.abs().amax(-1, keepdim=True) / 127
        idv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
        x = b * idv; ax = x.abs(); f = ax.floor()
        q = x.sign() * (f + (2 * (ax - f)).floor())  # = gguf-py np_roundf: half away from zero (ties occur on Q5 grids)
        out = q * d.half().float()
    else:
        imax = b.abs().argmax(-1, keepdim=True)
        d = torch.gather(b, 1, imax) / -16
        idv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
        q = torch.trunc(b * idv + 16.5).clamp(0, 31)
        out = (q - 16) * d.half().float()
    return out.reshape(w.shape)  # weight of the acceptance term in the training loss (0 = soft CE only)


def rms(x, w):
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w


class MTPHead(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.p = nn.ParameterDict({n.replace('.', '_'): nn.Parameter(sd[PFX + n].float().clone()) for n in MATS + NORMS})
        self.register_buffer('inv_freq', ROPE_BASE ** (-torch.arange(0, N_ROT, 2).float() / N_ROT), persistent=False)

    def w(self, n):
        return self.p[n.replace('.', '_')]

    def state_gguf(self):
        return {PFX + n: self.w(n).detach() for n in MATS + NORMS}

    def rope(self, x, pos):  # x [N, H, 256]; NEOX on dims 0..63: pairs (i, i+32), theta_i = pos * base^(-2i/64)
        ang = pos.float()[:, None] * self.inv_freq[None]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        r = x[..., :N_ROT].float()
        x1, x2 = r[..., :N_ROT // 2], r[..., N_ROT // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., N_ROT:].float()], -1)

    def refresh_q(self):
        """QAT: weights for this forward = fake-quantized copies with a straight-through gradient."""
        self.qw = {}
        if QAT:
            for n in MATS:
                w = self.w(n)
                fq = fake_quant(w.detach(), QTYPE.get(n, MAT_Q))
                if torch.is_grad_enabled():
                    self.qw[n] = (w + (fq - w.detach())).to(torch.bfloat16)
                else:
                    self.qw[n] = fq  # eval: exact fp32 dequantized values

    @torch.no_grad()
    def code_change(self):
        """QAT: fraction of matrix weights whose quantized value differs from the start (is training moving codes?)."""
        if not QAT:
            return None
        if not hasattr(self, 'q0'):
            self.q0 = {n: fake_quant(self.w(n), QTYPE.get(n, MAT_Q)).to(torch.bfloat16) for n in MATS}
        ch = tot = 0
        for n in MATS:
            ch += (fake_quant(self.w(n), QTYPE.get(n, MAT_Q)).to(torch.bfloat16) != self.q0[n]).sum().item(); tot += self.q0[n].numel()
        return round(ch / tot, 5)

    def lin(self, x, n):
        wq = getattr(self, 'qw', {}).get(n)
        if not torch.is_grad_enabled():  # eval: fp32 weights, like llama.cpp's exact dequantized values (no bf16 rounding)
            return F.linear(x.float(), wq.float() if wq is not None else self.w(n).float())
        return F.linear(x.to(torch.bfloat16), wq if wq is not None else self.w(n).to(torch.bfloat16)).float()

    def proj(self, e, h):
        return self.lin(torch.cat([rms(e, self.w('nextn.enorm.weight')), rms(h, self.w('nextn.hnorm.weight'))], -1), 'nextn.eh_proj.weight')

    def qkv(self, x, pos):
        n = x.shape[0]
        a = rms(x, self.w('attn_norm.weight'))
        qf = self.lin(a, 'attn_q.weight').view(n, N_HEAD, 2 * HD)
        q = rms(qf[..., :HD], self.w('attn_q_norm.weight'))
        gate = qf[..., HD:].reshape(n, N_HEAD * HD)
        k = rms(self.lin(a, 'attn_k.weight').view(n, N_KV, HD), self.w('attn_k_norm.weight'))
        v = self.lin(a, 'attn_v.weight').view(n, N_KV, HD)
        return self.rope(q, pos), self.rope(k, pos), v, gate

    def post(self, x, att, gate):
        x = x + self.lin(att.reshape(att.shape[0], N_HEAD * HD) * torch.sigmoid(gate), 'attn_output.weight')
        y = rms(x, self.w('post_attention_norm.weight'))
        x = x + self.lin(F.silu(self.lin(y, 'ffn_gate.weight')) * self.lin(y, 'ffn_up.weight'), 'ffn_down.weight')
        return rms(x, self.w('nextn.shared_head_norm.weight'))


def attend(q, keys, vals, mask):
    """q [R,24,256] f32; keys/vals [M,4,256]; mask [R,M] bool -> [R,24,256] f32 (bf16 compute)."""
    k = keys.to(torch.bfloat16).repeat_interleave(N_HEAD // N_KV, dim=1).transpose(0, 1)[None]
    v = vals.to(torch.bfloat16).repeat_interleave(N_HEAD // N_KV, dim=1).transpose(0, 1)[None]
    with sdpa_ctx():
        o = F.scaled_dot_product_attention(q.to(torch.bfloat16).transpose(0, 1)[None], k, v, attn_mask=mask[None, None])
    return o[0].transpose(0, 1).float()


def sdpa_ctx():  # fail loudly instead of falling back to the O(n^2)-memory math kernel
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]) if DEV.startswith('cuda') else sdpa_kernel([SDPBackend.MATH])


def chain_forward(model, H, E_tok, a0, depth, ckpt):
    """H [L,5120] bf16 trunk rows; E_tok [L,5120] bf16 = embeddings of the sequence tokens; anchors 0..N-1 (N = L-1).
    Step 1 runs every anchor (it is the catch-up K/V every later row reads); steps 2..depth run anchors a0..N-1 only.
    Returns [hn_1[a0:], hn_2, ..., hn_depth], each [R,5120] (R = N - a0); row r of step s = anchor a0+r."""
    L = H.shape[0]; N = L - 1; R = N - a0
    dev = H.device
    model.refresh_q()
    pos1 = torch.arange(1, N + 1, device=dev)
    ar = torch.arange(R, device=dev)

    def step1(H_, E_):
        x = model.proj(E_[1:L], H_[0:N])
        q, k, v, g = model.qkv(x, pos1)
        mask = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
        return model.post(x, attend(q, k, v, mask), g), k, v

    hn, K1, V1 = checkpoint(step1, H, E_tok, use_reentrant=False) if ckpt else step1(H, E_tok)
    outs = [hn[a0:]]
    Kd, Vd = [], []
    hprev = hn[a0:]
    # causal block for the chain rows: anchor a0+r sees step-1 rows 0..a0+r
    base_mask = torch.arange(N, device=dev)[None, :] <= (a0 + ar)[:, None]
    for s in range(2, depth + 1):
        tok_idx = torch.clamp(a0 + ar + s, max=L - 1)
        pos = a0 + ar + s

        # tok_idx/pos are bound as defaults: a checkpoint recompute runs after this loop has moved on
        def stepn(hprev_, K1_, V1_, *kv, _tok_idx=tok_idx, _pos=pos):
            x = model.proj(E_tok[_tok_idx], hprev_)
            q, k, v, g = model.qkv(x, _pos)
            kd = list(kv[:len(kv) // 2]) + [k]; vd = list(kv[len(kv) // 2:]) + [v]
            keys = torch.cat([K1_] + kd, 0); vals = torch.cat([V1_] + vd, 0)
            eye = torch.eye(R, dtype=torch.bool, device=dev)
            mask = torch.cat([base_mask] + [eye] * len(kd), 1)
            return model.post(x, attend(q, keys, vals, mask), g), k, v

        args = (hprev, K1, V1, *Kd, *Vd)
        hn, k, v = checkpoint(stepn, *args, use_reentrant=False) if ckpt else stepn(*args)
        Kd.append(k); Vd.append(v)
        outs.append(hn); hprev = hn
    return outs


class Data:
    def __init__(self, d, w):
        self.d = Path(d)
        self.meta = json.loads((self.d / 'seq.json').read_text())
        self.off = []
        o = 0
        for m in self.meta:
            self.off.append(o); o += m['len']
        self.h = np.memmap(self.d / 'h.bin', dtype=np.uint16, mode='r', shape=(o, N_EMBD))
        idx = [tuple(map(int, l.split())) for l in open(self.d / 'h.bin.idx')]
        assert [(i, self.off[i], m['len']) for i, m in enumerate(self.meta)] == idx, 'seq.json / h.bin.idx mismatch'
        self.tok = None
        self.s64 = torch.tensor([int(x) for x in open(self.d / 's64.txt').read().split()[3:]], dtype=torch.long)
        self.s24 = torch.tensor([int(x) for x in open(self.d / 's24.txt').read().split()[3:]], dtype=torch.long)
        self.tok = json.loads((self.d / 'tokens.json').read_text())

    def H(self, i):
        o, n = self.off[i], self.meta[i]['len']
        return torch.from_numpy(np.ascontiguousarray(self.h[o:o + n])).view(torch.bfloat16)


def load_w(w):
    w = Path(w)
    sd = torch.load(w / 'mtp.pt')
    lm = torch.load(w / 'lm_head.bf16.pt', mmap=True)
    emb = torch.load(w / 'tok_embd.bf16.pt', mmap=True)
    return sd, lm, emb


def teacher(Hrows, lm_full):
    """Hrows [n,5120] bf16 GPU -> (idx [n,20] long, p [n,20] f32): top_k 20 -> softmax -> top_p 0.95, renormalized."""
    h = Hrows.float()
    z = torch.cat([h @ lm_full[c:c + 32768].float().T for c in range(0, lm_full.shape[0], 32768)], -1)
    v, i = z.topk(TOPK_T, -1)
    p = torch.softmax(v, -1)
    c = p.cumsum(-1)
    keep = (c - p) < TOPP_T  # smallest prefix with cumulative >= top_p (the first token is always kept)
    p = p * keep
    return i, p / p.sum(-1, keepdim=True)


def cmd_targets(a):
    sd, lm, emb = load_w(a.w)
    D = Data(a.data, a.w)
    lm_full = lm.to(DEV)
    idx_all, p_all = [], []
    t0 = time.time()
    agree = n_agree = 0
    for i, m in enumerate(D.meta):
        H = D.H(i).to(DEV)
        ii, pp = [], []
        for c in range(0, H.shape[0], 2048):
            ti, tp = teacher(H[c:c + 2048], lm_full)
            ii.append(ti.int().cpu()); pp.append(tp.half().cpu())
        ti, tp = torch.cat(ii), torch.cat(pp)
        # sanity: the trunk's top-1 at row t vs the token actually sampled at t+1 (reply rows)
        toks = torch.tensor(D.tok[i])
        P = m['prompt_len']
        agree += (ti[P - 1:-1, 0].long() == toks[P:]).sum().item(); n_agree += len(toks) - P
        idx_all.append(ti); p_all.append(tp)
        if (i + 1) % 200 == 0:
            print(f'targets {i + 1}/{len(D.meta)} {time.time() - t0:.0f}s', flush=True)
    np.save(Path(a.data) / 'tgt_idx.npy', torch.cat(idx_all).numpy())
    np.save(Path(a.data) / 'tgt_p.npy', torch.cat(p_all).numpy())
    print(f'TARGETS_DONE rows {sum(x.shape[0] for x in idx_all)}; trunk top-1 == sampled next token on reply rows: {agree / max(n_agree, 1):.4f}')


class Ctx:
    """everything a forward pass needs on the GPU."""
    def __init__(self, a, head_sd=None):
        sd, lm, emb = load_w(a.w)
        if head_sd is not None:
            sd = {**sd, **head_sd}
        self.D = Data(a.data, a.w)
        self.model = MTPHead(sd).to(DEV)
        self.emb = emb  # CPU, mmap
        self.W64 = lm[self.D.s64].float().to(DEV)  # fp32 logits (1.3 GB)
        self.full2s64 = torch.full((lm.shape[0],), -1, dtype=torch.long); self.full2s64[self.D.s64] = torch.arange(len(self.D.s64))
        self.full2s64 = self.full2s64.to(DEV)
        in64 = torch.zeros(lm.shape[0], dtype=torch.bool); in64[self.D.s64] = True
        s24_in_64 = self.full2s64.cpu()[self.D.s24]
        assert (s24_in_64 >= 0).all(), 'RM24H must be inside the 64K list'
        self.s24_cols = s24_in_64.to(DEV)
        self.tgt_idx = np.load(Path(a.data) / 'tgt_idx.npy', mmap_mode='r')
        self.tgt_p = np.load(Path(a.data) / 'tgt_p.npy', mmap_mode='r')

    def seq(self, i):
        D = self.D
        o, L = D.off[i], D.meta[i]['len']
        H = D.H(i).to(DEV, non_blocking=True)
        toks = torch.tensor(D.tok[i], dtype=torch.long)
        E = self.emb[toks].to(DEV, non_blocking=True)
        ti = torch.from_numpy(np.asarray(self.tgt_idx[o:o + L])).long().to(DEV)
        tp = torch.from_numpy(np.asarray(self.tgt_p[o:o + L])).float().to(DEV)
        return H, E, ti, tp, L, D.meta[i]['prompt_len']


def seq_rows(L, P, s):
    """anchor rows used for step s: a0 = P-1 (first draft anchor = last prompt row), valid while a0+r+s <= L-1."""
    a0 = max(P - 1, 0)
    R = L - 1 - a0
    valid = (a0 + torch.arange(R, device=DEV) + s) <= L - 1
    return a0, R, valid


def ce_chunk(h, W, tidx, tp, valid):
    z = h.float() @ W.T
    lse = torch.logsumexp(z, -1, keepdim=True)
    logq = torch.gather(z, 1, tidx.clamp(min=0)) - lse
    loss = -((tp * logq).sum(-1) * valid).sum()
    if LAM_TV and torch.is_grad_enabled():  # training only; acceptance-aligned term: -sum_x min(p(x), q(x)) over the target support (= TV distance - 1)
        loss = loss - LAM_TV * (torch.minimum(tp, logq.exp() * (tp > 0)).sum(-1) * valid).sum()
    return loss


def step_loss(hn, ctx, ti, tp, valid):
    """soft CE over the 64K shortlist; targets restricted to the shortlist and renormalized."""
    m = ctx.full2s64[ti]
    tp = tp * (m >= 0)
    ok = tp.sum(-1) > 0
    tp = tp / tp.sum(-1, keepdim=True).clamp(min=1e-9)
    valid = valid & ok
    tot = hn.new_zeros(())
    for c in range(0, hn.shape[0], 1024):
        tot = tot + checkpoint(ce_chunk, hn[c:c + 1024], ctx.W64, m[c:c + 1024], tp[c:c + 1024], valid[c:c + 1024].float(), use_reentrant=False)
    return tot, valid.sum()


@torch.no_grad()
def accept_metrics(hn, ctx, ti, tp, valid):
    """per-row acceptance alpha (RM24H and 64K drafts), top-1 agreement, CE. Returns dict of [R] tensors."""
    out = dict(a24=[], a64=[], top1=[])
    for c in range(0, hn.shape[0], 1024):
        z = hn[c:c + 1024].float() @ ctx.W64.T
        m = ctx.full2s64[ti[c:c + 1024]]
        p = tp[c:c + 1024]
        for key, zz, cols in (('a64', z, None), ('a24', z[:, ctx.s24_cols], ctx.s24_cols)):
            v, i = zz.topk(TOPK_Q, -1)
            q = torch.softmax(v, -1)
            q = q * ((q.cumsum(-1) - q) < TOPP_T)       # draft chain: top 10 -> top_k 20 (no-op) -> top_p 0.95
            q = q / q.sum(-1, keepdim=True)
            s64i = i if cols is None else cols[i]       # 64K-list index of each draft candidate
            hit = (m[:, None, :] == s64i[:, :, None]) & (m[:, None, :] >= 0)  # [n,10,20]
            pq = (hit * p[:, None, :]).sum(-1)           # target prob of each draft candidate
            out[key].append(torch.minimum(pq, q).sum(-1))
        top = z.argmax(-1)
        out['top1'].append((m[:, 0] == top).float())
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def evaluate(ctx, ids, max_seqs=None):
    model = ctx.model
    model.eval()
    acc = {k: [[] for _ in range(DEPTH)] for k in ('a24', 'a64', 'top1')}
    ce = [0.0] * DEPTH; nce = [0] * DEPTH
    chains = {k: [] for k in ('E5_24', 'E7_24', 'E5_64', 'E7_64')}
    for i in ids[:max_seqs]:
        H, E, ti, tp, L, P = ctx.seq(i)
        a0 = max(P - 1, 0)
        if L - 1 - a0 < DEPTH + 1:
            continue
        outs = chain_forward(model, H, E, a0, DEPTH, ckpt=False)
        per = []
        for s in range(1, DEPTH + 1):
            _, R, valid = seq_rows(L, P, s)
            rows = a0 + torch.arange(R, device=DEV) + s
            rows = rows.clamp(max=L - 1)
            l, n = step_loss(outs[s - 1], ctx, ti[rows], tp[rows], valid)
            ce[s - 1] += l.item(); nce[s - 1] += n.item()
            m = accept_metrics(outs[s - 1], ctx, ti[rows], tp[rows], valid)
            for k in m:
                acc[k][s - 1].append(m[k][valid].cpu())
            per.append(m)
        # expected accepted tokens at depth 5 and 7, over anchors where all DEPTH steps are valid
        _, R, vd = seq_rows(L, P, DEPTH)
        for key in ('24', '64'):
            prod = torch.ones(R, device=DEV); e = torch.zeros(R, device=DEV)
            for s in range(DEPTH):
                prod = prod * per[s]['a' + key]; e = e + prod
                if s + 1 in (5, 7):
                    chains[f'E{s + 1}_{key}'].append(e[vd].cpu())
    res = dict(n_seqs=len(ids[:max_seqs]))
    for k in acc:
        res[k] = [round(torch.cat(acc[k][s]).mean().item(), 4) for s in range(DEPTH)]
    res['ce'] = [round(ce[s] / max(nce[s], 1), 4) for s in range(DEPTH)]
    res['rows'] = nce
    for k, v in chains.items():
        res[k] = round(torch.cat(v).mean().item(), 4)
    model.train()
    return res


def split_ids(D, split):
    """train | heldout | heldout_sel (checkpoint selection) | heldout_rep (reported go/no-go; never used to select)."""
    import zlib
    if split in ('heldout_sel', 'heldout_rep'):
        half = 0 if split == 'heldout_sel' else 1
        return [i for i, m in enumerate(D.meta) if m['split'] == 'heldout' and zlib.crc32(m['id'].encode()) % 2 == half]
    return [i for i, m in enumerate(D.meta) if m['split'] == split]


def cmd_eval(a):
    head = None
    if a.head:
        head = torch.load(a.head)
    elif a.head_gguf:
        head = read_overlay(a.head_gguf)
    ctx = Ctx(a, head)
    ids = split_ids(ctx.D, a.split)
    r = evaluate(ctx, ids, a.max_seqs)
    r['head'] = a.head or a.head_gguf or 'original (P76 in-file head)'
    print('EVAL', json.dumps(r), flush=True)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(r, indent=1))


def read_overlay(path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
    import gguf
    from gguf.quants import dequantize
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        ne = [int(x) for x in t.shape]
        a = dequantize(np.asarray(t.data), gguf.GGMLQuantizationType(int(t.tensor_type))).reshape(-1, ne[0]).astype(np.float32)
        out[t.name] = torch.from_numpy(a.reshape(-1) if len(ne) == 1 else a)
    return out


def cmd_train(a):
    torch.manual_seed(a.seed); random.seed(a.seed)
    ctx = Ctx(a, torch.load(a.head) if a.head else None)  # --head: start from a checkpoint (e.g. QAT polish)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = ctx.D
    tr, ho = split_ids(D, 'train'), split_ids(D, 'heldout_sel')
    assert not set(D.meta[i]['id'] for i in tr) & set(D.meta[i]['id'] for i in split_ids(D, 'heldout'))
    assert not set(D.meta[i].get('chash') for i in tr) & set(D.meta[i].get('chash') for i in split_ids(D, 'heldout')) - {None}
    n_tok = sum(D.meta[i]['len'] - D.meta[i]['prompt_len'] for i in tr)
    steps_total = max(1, int(a.epochs * n_tok / a.tokens_per_step))
    model = ctx.model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    sched = lambda st: min(1.0, (st + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(st, steps_total) / steps_total)))
    log = open(out / 'train.log', 'a')

    def P(msg):
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    P(f'train seqs {len(tr)} reply tokens {n_tok}; heldout_sel seqs {len(ho)}; steps {steps_total}; args {vars(a)}')
    model.code_change()  # snapshot the starting codes
    best = evaluate(ctx, ho, a.eval_seqs)
    P('EVAL step 0 ' + json.dumps(best))
    best_e = best['E7_24']
    step, acc_tok, t0 = 0, 0, time.time()
    order = []
    ep = 0
    while step < steps_total:
        if not order:
            order = list(tr); random.shuffle(order); ep += 1
        i = order.pop()
        H, E, ti, tp, L, Pl = ctx.seq(i)
        a0 = max(Pl - 1, 0)
        if L - 1 - a0 < 2:
            continue
        outs = chain_forward(model, H, E, a0, a.depth, ckpt=True)
        loss = 0.0; nrow = 0
        for s in range(1, a.depth + 1):
            _, R, valid = seq_rows(L, Pl, s)
            rows = (a0 + torch.arange(R, device=DEV) + s).clamp(max=L - 1)
            l, n = step_loss(outs[s - 1], ctx, ti[rows], tp[rows], valid)
            loss = loss + STEP_W[s - 1] * l
            if s == 1:
                nrow = n.item()
        (loss / a.tokens_per_step).backward()
        acc_tok += nrow
        if acc_tok >= a.tokens_per_step:
            for g in opt.param_groups:
                g['lr'] = a.lr * sched(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            P(f'step {step}/{steps_total} ep {ep} rows {acc_tok} last-seq loss/row {loss.item() / max(nrow, 1):.4f} gnorm {gn:.3f} lr {a.lr * sched(step - 1):.2e} {time.time() - t0:.0f}s')
            acc_tok = 0
            if step % a.eval_every == 0 or step == steps_total:
                r = evaluate(ctx, ho, a.eval_seqs)
                r['codes_changed'] = model.code_change()
                P(f'EVAL step {step} ' + json.dumps(r))
                if os.environ.get('MTP_SAVE_STEPS') == '1':  # off by default: 1.7 GB per checkpoint
                    torch.save({k: v.float().cpu() for k, v in model.state_gguf().items()}, out / f'head_step{step}.pt')
                if r['E7_24'] > best_e:
                    best_e = r['E7_24']
                    torch.save({k: v.float().cpu() for k, v in model.state_gguf().items()}, out / 'head_best.pt')
                    P(f'BEST step {step} E7_24 {best_e}')
    P('TRAIN_DONE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['targets', 'eval', 'train'])
    ap.add_argument('--w', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--out'); ap.add_argument('--head'); ap.add_argument('--head-gguf'); ap.add_argument('--json-out')
    ap.add_argument('--split', default='heldout'); ap.add_argument('--max-seqs', type=int)
    ap.add_argument('--epochs', type=float, default=2.0); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--warmup', type=int, default=10); ap.add_argument('--tokens-per-step', type=int, default=16384)
    ap.add_argument('--depth', type=int, default=DEPTH); ap.add_argument('--eval-every', type=int, default=40)
    ap.add_argument('--eval-seqs', type=int, default=None); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    dict(targets=cmd_targets, eval=cmd_eval, train=cmd_train)[a.cmd](a)


if __name__ == '__main__':
    main()
