#!/usr/bin/env python3
"""INKSPEC2 DSPARKTRAIN: torch mirror of the Inkling2 DSpark drafter (src/models/dflash.cpp, graph_inkling2) at ship
precision, plus the server's verify-side acceptance math (common/sampling.cpp, common_sampler_verify_pq_impl).

Runtime path mirrored (server default, afc152177):
- context (embd batch, process()): per drafter layer il, mix = softmax(dspark_fuse[il]) . features[k] (k = target layers
  5, 23, 41, 48, the target's llama_get_embeddings_layer_inp rows), c = rms(mix) * dflare_norm; K = rope(rms_head(c Wk_ctx)
  * k_ctx_norm), V = c Wv_ctx written at the token's position. f_rnn = rms(mix[row 3]) * dspark_norm per position.
- block (token batch, draft()): [anchor, mask x7] at positions a..a+7; anchor = rms(tok_embd[t_a]) * token_embd_norm
  (target's), masks = mask_embd. Non-causal; every query uses the anchor's mask row: context cells p < a (SWA layers 0, 2:
  p >= a - 1024, n_swa = 1024 + 1) plus the 8 block cells. q/k per-head RMS norms, NEOX rope base 1e6, GQA 24/8, scale
  1/sqrt(128), SwiGLU FFN, output_norm -> cur. base = target output.weight (AI4 file, Q8_0) . cur.
- markov walk over slots 1..walk: joint = [state, markov_w1[prev], cur_i, f_rnn[a-1]] -> joint_proj -> gate sigmoid,
  cand tanh, state = cand + gate (state - cand), latent tanh; bias = fold . latent + fold_bias (the shipped
  dspark_hidden_out.fold, LLAMA_DSPARK_HEAD_FOLD default on); logits = 30 tanh((base + bias) / 30). prev of slot 1 is the
  anchor, of slot i the token drawn at slot i-1.
- sampled walk (LLAMA_DSPARK_SAMPLE default on, K 20, T = temp 1): q = softmax(top20(logits)/T) over those 20 ids.
- verifier: p = target logits -> top_k 20 -> softmax -> top_p 0.95 (keep while the cum before < 0.95) -> renorm.
  exact: accept d with min(1, p_d/q_d); mb (--spec-mb-mtp default on): a rejected d is still kept when the residual draw
  y ~ (p - q)+ has p_y <= p_d and neither d nor y ends the generation.

Offline acceptance from on-policy data (t_{a+j} sampled from p by the target itself), teacher-forced walk (prev = data
tokens, which is what an accepted chain conditions on):
  A_s = sum_x min(p_s, q_s)                    (exact slot-s acceptance given slots < s accepted)
  c_s = min(1, q_s(t)/p_s(t))                  (P(slot s accepted | its output token is t))
  E_d = sum_{s<=d} prod_{j<s} c_j * A_s         (unbiased expected accepted drafts per round at this anchor)
  mb: J(x) = q(x) a(x), A^mb = sum J, c^mb = J(t)/p(t) for E^mb (importance weight, unbiased), and
      J(t)/p'(t) with p' the mb output law for the round simulation (approximation: data is from p, not p').
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get('GGUF_PY'), os.path.join(HERE, '..', '..', 'gguf-py')):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import gguf  # noqa: E402

Q = gguf.GGMLQuantizationType
N_LAYER = 3
RANK = 256
N_EMBD = 2304
N_HEAD, N_HEAD_KV, HD = 24, 8, 128
BLOCK = 8
SWA_LAYERS = (0, 2)
N_SWA = 1024          # gguf dflash.attention.sliding_window; the loader adds 1 (window includes the anchor)
ROPE_BASE = 1e6
EPS = 1e-6
SOFTCAP = 30.0
V_PAD, V_UNPAD = 201024, 200058
EOG_IDS = (199999, 200006)   # llama_vocab_is_eog of the AI4 target (dump writes PREFIX.eog; DumpSet uses that)

LAYER_MATS = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_k_ctx', 'attn_v_ctx', 'ffn_gate', 'ffn_up', 'ffn_down')
LAYER_VECS = ('dflare_norm', 'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm', 'attn_k_ctx_norm')
GLOBAL_VECS = ('dspark_fuse.weight', 'dspark_norm.weight', 'mask_embd.weight', 'output_norm.weight', 'joint_proj.bias',
               'dspark_hidden_out.bias')
GLOBAL_MATS = ('markov_w1.weight', 'joint_proj.weight')   # Q8_0 in the shipped file
H_NAME = 'dspark_hidden_out.weight'                         # used at runtime only through the fold (F16)


def q8_names():
    return [f'blk.{i}.{m}.weight' for i in range(N_LAYER) for m in LAYER_MATS] + list(GLOBAL_MATS)


def f32_names():
    return [f'blk.{i}.{v}.weight' for i in range(N_LAYER) for v in LAYER_VECS] + list(GLOBAL_VECS)


# ---------------------------------------------------------------- gguf io
def gguf_f32(t):
    """numpy float32 array in torch layout (reversed gguf shape) of a gguf tensor (F32/F16/BF16/Q8_0...)."""
    a = np.asarray(t.data)
    shp = list(reversed([int(x) for x in t.shape]))
    if t.tensor_type == Q.BF16:
        return (a.reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shp)
    if t.tensor_type == Q.F32:
        return a.reshape(shp).astype(np.float32)
    if t.tensor_type == Q.F16:
        return a.reshape(shp).astype(np.float32)
    return gguf.quants.dequantize(a, t.tensor_type).reshape(shp).astype(np.float32)


def q8_codes(t):
    """Q8_0 gguf tensor -> (int8 codes [rows, cols], fp16 scales [rows, cols/32]) without dequantizing."""
    assert t.tensor_type == Q.Q8_0
    shp = list(reversed([int(x) for x in t.shape]))
    raw = np.asarray(t.data).reshape(-1, 34)
    d = raw[:, :2].copy().view(np.float16).reshape(shp[0], shp[1] // 32)
    q = raw[:, 2:].copy().view(np.int8).reshape(shp[0], shp[1])
    return q, d


def load_gguf_tensors(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


# ---------------------------------------------------------------- exact quantizers
def round_half_away(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def q8_0_quant(w):
    """gguf Q8_0 of a [rows, cols] fp32 tensor, bit-exact with quantize_row_q8_0_ref / gguf-py: (codes int8, d fp16)."""
    r, c = w.shape
    b = w.float().reshape(r, c // 32, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    idv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = round_half_away(b * idv)
    return q.to(torch.int8).reshape(r, c), d.squeeze(-1).to(torch.float16)


def q8_0_deq(q, d):
    r, c = q.shape
    return (q.float().reshape(r, c // 32, 32) * d.float().unsqueeze(-1)).reshape(r, c)


def q8_0_fq(w):
    """fake-quant forward value (no grad; the trainer applies the straight-through gradient itself)."""
    with torch.no_grad():
        return q8_0_deq(*q8_0_quant(w))


# ---------------------------------------------------------------- weights
class Frozen:
    """target-side frozen tensors on the device: tok_embd (Q8_0 codes), token_embd_norm, output.weight (Q8_0, AI4
    file) for the base logits, and the BF16 truth lm head for the fold."""

    def __init__(self, w0, device, out_dtype=torch.float32):
        t = torch.load(os.path.join(w0, 'target.pt'), map_location='cpu', weights_only=True)
        self.emb_q, self.emb_d = t['tok_embd_q'].to(device), t['tok_embd_d'].to(device)
        self.tok_norm = t['tok_norm'].float().to(device)
        out = q8_0_deq(t['output_q'][:V_UNPAD].to(device), t['output_d'][:V_UNPAD].to(device))
        self.out = out.to(out_dtype)
        self.out_bf16 = None
        lm = np.fromfile(os.path.join(w0, 'lm_truth.bf16'), dtype=np.uint16).reshape(V_PAD, N_EMBD)
        self.lm = torch.from_numpy(lm.view(np.int16).copy()).view(torch.bfloat16).to(device)  # [V_PAD, N_EMBD]

    def embed(self, ids):
        q = self.emb_q[ids].float()
        d = self.emb_d[ids].float()
        return (q.reshape(*ids.shape, -1, 32) * d.unsqueeze(-1)).reshape(*ids.shape, N_EMBD)

    def base_weight(self, bf16):
        if not bf16:
            return self.out
        if self.out_bf16 is None:
            self.out_bf16 = self.out.to(torch.bfloat16)
        return self.out_bf16

    def fold(self, H, b, chunk=65536):
        """fold = fp16(lm_truth @ bf16(H)) [V_PAD, RANK] and fold_bias = lm_truth @ b [V_PAD], as ink_dspark_fold.py
        (F32 product of the BF16 truth head and the BF16 drafter H). Returned in f32 (fold already fp16-rounded)."""
        Hb = H.to(torch.bfloat16).float()  # [N_EMBD, RANK]
        P = torch.empty(V_PAD, RANK, device=H.device, dtype=torch.float32)
        c = torch.empty(V_PAD, device=H.device, dtype=torch.float32)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for i in range(0, V_PAD, chunk):
                L = self.lm[i:i + chunk].float()
                P[i:i + chunk] = (L @ Hb).half().float()
                c[i:i + chunk] = L @ b.float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return P, c


def load_drafter_masters(q8_path, bf16_path):
    """fp32 masters: Q8_0 matrices from the shipped Q8_0 file (dequantized: re-quantizing them is the identity), F32
    tensors as stored, and H from the BF16 file (the shipped fold was made from it; H is not read at runtime)."""
    tq = load_gguf_tensors(q8_path)
    tb = load_gguf_tensors(bf16_path)
    m = {}
    for n in q8_names():
        assert tq[n].tensor_type == Q.Q8_0, n
        m[n] = torch.from_numpy(gguf_f32(tq[n]))
    for n in f32_names():
        assert tq[n].tensor_type == Q.F32, n
        m[n] = torch.from_numpy(gguf_f32(tq[n]))
    m[H_NAME] = torch.from_numpy(gguf_f32(tb[H_NAME]))
    return m


def effective(masters, frozen, fold_override=None):
    """forward weights at ship precision: Q8_0 fake-quant of the Q8_0 matrices, F32 as is, fold from H."""
    W = {}
    for n, v in masters.items():
        if n in GLOBAL_MATS or n.endswith(tuple(f'{m}.weight' for m in LAYER_MATS)):
            W[n] = q8_0_fq(v)
        else:
            W[n] = v
    if fold_override is not None:
        W['fold'], W['fold_b'] = fold_override
    else:
        W['fold'], W['fold_b'] = frozen.fold(masters[H_NAME], masters['dspark_hidden_out.bias'])
    return W


# ---------------------------------------------------------------- forward
def rms(x, w=None, eps=EPS):
    y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
    return y * w if w is not None else y


_ROPE = {}


def rope_cs(pos, device):
    """cos/sin tables, one per device, regrown (doubling) only when a longer position is needed."""
    need = int(pos.max()) + 1
    t = _ROPE.get(device)
    if t is None or t[0].shape[0] < need:
        n = max(need, 2 * (t[0].shape[0] if t is not None else 0), 8192)
        inv = ROPE_BASE ** (-(torch.arange(0, HD // 2, dtype=torch.float64) * 2) / HD)
        th = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
        t = _ROPE[device] = (torch.cos(th).float().to(device), torch.sin(th).float().to(device))
    return t[0][pos], t[1][pos]


def rope_neox(x, pos):
    """x [..., T, H, HD], pos [..., T] (NEOX: pairs (i, i + HD/2))."""
    c, s = rope_cs(pos, x.device)
    c, s = c.unsqueeze(-2).to(x.dtype), s.unsqueeze(-2).to(x.dtype)
    x0, x1 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], -1)


ACT_Q = True   # emulate the engine's activation rounding: Q8_0 weights get q8 (block 32) activations, F16 weights f16


def act_q8(x):
    """ggml quantizes the activations of a Q8_0 matmul to 8-bit blocks of 32 (CUDA q8_1 / CPU q8_0: d = fp16(amax/127),
    q = round(x/d)); straight-through gradient."""
    if not ACT_Q:
        return x
    sh = x.shape
    xf = x.reshape(-1, sh[-1])
    with torch.no_grad():
        xq = q8_0_deq(*q8_0_quant(xf.float())).to(x.dtype)
    return (xf + (xq - xf).detach()).reshape(sh)


def act_f16(x):
    if not ACT_Q:
        return x
    return x + (x.detach().half().to(x.dtype) - x.detach())


def mm(x, w):
    """x . w^T for a Q8_0 weight (activations q8-rounded like the engine)."""
    return act_q8(x) @ w.t().to(x.dtype)


def context_layers(W, feat, pos):
    """feat [N, 4, N_EMBD] (float) -> per layer (K [N, 8, HD] roped, V [N, 8, HD]) and f_rnn [N, N_EMBD]."""
    fuse = torch.softmax(W['dspark_fuse.weight'].float(), -1)  # [4 rows, 4 targets]
    out = []
    for il in range(N_LAYER):
        mix = torch.einsum('nke,k->ne', feat, fuse[il].to(feat.dtype))
        c = rms(mix, W[f'blk.{il}.dflare_norm.weight'].to(feat.dtype))
        k = mm(c, W[f'blk.{il}.attn_k_ctx.weight']).reshape(-1, N_HEAD_KV, HD)
        v = mm(c, W[f'blk.{il}.attn_v_ctx.weight']).reshape(-1, N_HEAD_KV, HD)
        k = rope_neox(rms(k, W[f'blk.{il}.attn_k_ctx_norm.weight'].to(k.dtype)), pos)
        out.append((k, v))
    mix = torch.einsum('nke,k->ne', feat, fuse[N_LAYER].to(feat.dtype))
    f_rnn = rms(mix, W['dspark_norm.weight'].to(feat.dtype))
    return out, f_rnn


def context_layers_sliced(W, feat, amax, dev, dtype=torch.float32, sl=8192):
    """context_layers over positions 0..amax-1 without a graph, slice by slice into preallocated outputs (no torch.cat
    copy); feat may live on the host (pinned) -- each slice is moved to dev on its own."""
    out, f_rnn = None, None
    for s0 in range(0, amax, sl):
        s1 = min(s0 + sl, amax)
        kv, fr_ = context_layers(W, feat[s0:s1].to(dev, non_blocking=True).to(dtype), torch.arange(s0, s1, device=dev))
        if out is None:
            out = [tuple(torch.empty((amax,) + t.shape[1:], dtype=t.dtype, device=dev) for t in pair) for pair in kv]
            f_rnn = torch.empty((amax,) + fr_.shape[1:], dtype=fr_.dtype, device=dev)
        for (K, V), (k, v) in zip(out, kv):
            K[s0:s1] = k; V[s0:s1] = v
        f_rnn[s0:s1] = fr_
        del kv, fr_
    return out, f_rnn


def block_forward(W, frozen, ctx_kv, anchors, anchor_tok, k0, dtype):
    """Backbone over the blocks of anchors [nA] (positions a..a+7). ctx_kv: per layer (K, V) for context positions
    k0..k0+Lc-1. Returns cur [nA, BLOCK, N_EMBD] (after output_norm)."""
    dev = anchors.device
    nA = anchors.shape[0]
    x = torch.empty(nA, BLOCK, N_EMBD, device=dev, dtype=dtype)
    x[:, 0] = rms(frozen.embed(anchor_tok), frozen.tok_norm).to(dtype)
    x[:, 1:] = W['mask_embd.weight'].to(dtype)
    pos = anchors[:, None] + torch.arange(BLOCK, device=dev)[None, :]          # [nA, 8]
    Lc = ctx_kv[0][0].shape[0]
    cpos = k0 + torch.arange(Lc, device=dev)
    # mask [nA*8, Lc + nA*8]: context cells p < a (and p >= a - N_SWA for SWA layers) + own block
    vis_full = cpos[None, :] < anchors[:, None]                                  # [nA, Lc]
    vis_swa = vis_full & (cpos[None, :] >= anchors[:, None] - N_SWA)
    blk = torch.eye(nA, dtype=torch.bool, device=dev).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    masks = {}
    for kind, vis in (('full', vis_full), ('swa', vis_swa)):
        masks[kind] = torch.cat([vis.repeat_interleave(BLOCK, 0), blk], 1)[None, None]   # [1,1,Lq,Lk]
    for il in range(N_LAYER):
        p = f'blk.{il}.'
        h = rms(x, W[p + 'attn_norm.weight'].to(dtype))
        q = mm(h, W[p + 'attn_q.weight']).reshape(nA, BLOCK, N_HEAD, HD)
        k = mm(h, W[p + 'attn_k.weight']).reshape(nA, BLOCK, N_HEAD_KV, HD)
        v = mm(h, W[p + 'attn_v.weight']).reshape(nA, BLOCK, N_HEAD_KV, HD)
        q = rope_neox(rms(q, W[p + 'attn_q_norm.weight'].to(dtype)), pos)
        k = rope_neox(rms(k, W[p + 'attn_k_norm.weight'].to(dtype)), pos)
        Kc, Vc = ctx_kv[il]
        Kall = torch.cat([Kc.to(dtype), k.reshape(nA * BLOCK, N_HEAD_KV, HD)], 0)
        Vall = torch.cat([Vc.to(dtype), v.reshape(nA * BLOCK, N_HEAD_KV, HD)], 0)
        # GQA without expanding K/V (q head h uses kv head h // G): the G query heads of a kv head are stacked along the
        # query axis, so SDPA sees 8 heads over [G*Lq] queries and the unexpanded [Lk] keys (same math, no [24, Lk] copies)
        G = N_HEAD // N_HEAD_KV; Lq = nA * BLOCK
        qq = q.reshape(Lq, N_HEAD_KV, G, HD).permute(1, 2, 0, 3).reshape(1, N_HEAD_KV, G * Lq, HD)
        kk = Kall.transpose(0, 1)[None]; vv = Vall.transpose(0, 1)[None]                     # [1, 8, Lk, HD]
        m = masks['swa' if il in SWA_LAYERS else 'full'].repeat(1, 1, G, 1)                  # [1, 1, G*Lq, Lk]
        o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=m, scale=1.0 / math.sqrt(HD))
        o = o[0].reshape(N_HEAD_KV, G, Lq, HD).permute(2, 0, 1, 3).reshape(nA, BLOCK, N_HEAD * HD)
        x = x + mm(o, W[p + 'attn_output.weight'])
        h = rms(x, W[p + 'ffn_norm.weight'].to(dtype))
        g = mm(h, W[p + 'ffn_gate.weight'])
        u = mm(h, W[p + 'ffn_up.weight'])
        x = x + mm(F.silu(g) * u, W[p + 'ffn_down.weight'])
    return rms(x, W['output_norm.weight'].to(dtype))


def markov_step(W, prev, state, hidden, context):
    """one markov head step; returns (new state, latent)."""
    w1 = W['markov_w1.weight'][prev].to(hidden.dtype)
    joint = torch.cat([state, w1, hidden, context], -1)
    J = mm(joint, W['joint_proj.weight']) + W['joint_proj.bias'].to(joint.dtype)
    gate = torch.sigmoid(J[..., :RANK])
    cand = torch.tanh(J[..., RANK:2 * RANK])
    state = cand + gate * (state - cand)
    return state, torch.tanh(J[..., 2 * RANK:])


def slot_logits(W, frozen, cur_i, latent, bf16=False):
    """softcapped logits [R, V_UNPAD] of one walked slot."""
    if bf16:   # bf16 GEMM for the [V, E] base (training speed option), fp32 result
        base = (act_q8(cur_i).to(torch.bfloat16) @ frozen.base_weight(True).t()).float()
    else:
        base = mm(cur_i, frozen.out)
    bias = act_f16(latent) @ W['fold'][:V_UNPAD].t().to(latent.dtype) + W['fold_b'][:V_UNPAD].to(latent.dtype)
    l = (base + bias).float()
    return SOFTCAP * torch.tanh(l / SOFTCAP)


def walk_teacher(W, frozen, cur, f_ctx, prevs, walk, bf16=False):
    """Teacher-forced walk. cur [nA, 8, E], f_ctx [nA, E] (f_rnn of a-1), prevs [nA, walk] (anchor token, then the
    data tokens of slots 1..walk-1). Yields (slot index, logits [nA, V_UNPAD])."""
    state = torch.zeros(cur.shape[0], RANK, device=cur.device, dtype=cur.dtype)
    for i in range(1, walk + 1):
        state, latent = markov_step(W, prevs[:, i - 1], state, cur[:, i], f_ctx.to(cur.dtype))
        yield i, slot_logits(W, frozen, cur[:, i], latent, bf16)


# ---------------------------------------------------------------- distributions and acceptance
def target_p(top_logits, k=20, top_p=0.95):
    """server p from the target's top logits [R, >=k] (descending): top_k k -> softmax -> top_p -> renorm."""
    lg = top_logits[:, :k].float()
    pr = torch.softmax(lg, -1)
    cs = torch.cumsum(pr, -1)
    keep = (cs - pr) < top_p
    keep[:, 0] = True
    pr = torch.where(keep, pr, torch.zeros_like(pr))
    return pr / pr.sum(-1, keepdim=True)


def draft_q(logits, k=20, temp=1.0):
    vals, ids = torch.topk(logits, k, -1)
    return torch.softmax(vals / temp, -1), ids, vals


def accept_terms(p, pid, q, qid, t, eog=EOG_IDS, want_mb=True):
    """Per row: exact A, c(t); mb A^mb, J(t)/p(t) (IS) and J(t)/p'(t) (conditional). p [R, Kp] over pid, q [R, Kq] over
    qid, t [R] data token. Differentiable in q."""
    eq = pid[:, :, None] == qid[:, None, :]                         # [R, Kp, Kq]
    q_at_p = (eq.to(q.dtype) * q[:, None, :]).sum(-1)               # q(pid)
    A = torch.minimum(p, q_at_p).sum(-1)
    pt = (p * (pid == t[:, None])).sum(-1)
    qt = (q * (qid == t[:, None])).sum(-1)
    ratio = qt / pt.clamp_min(1e-30)
    c = torch.where(pt > 0, torch.clamp(ratio, max=1.0), torch.zeros_like(qt))   # p(t) = 0: dump/gen numeric mismatch, no credit
    out = dict(A=A, c=c, pt0=(pt <= 0))
    if not want_mb:
        return out
    p_at_q = (eq.to(p.dtype) * p[:, :, None]).sum(1)                # p(qid)
    m = torch.where(q > 0, torch.clamp(p_at_q / q.clamp_min(1e-30), max=1.0), torch.zeros_like(q))
    res = torch.clamp(p - q_at_p, min=0.0)                          # [R, Kp]
    Z = res.sum(-1, keepdim=True)
    eog_t = eog.to(pid.device) if torch.is_tensor(eog) else torch.tensor(list(eog), device=pid.device)
    p_eog = torch.isin(pid, eog_t)
    q_eog = torch.isin(qid, eog_t)
    # R(x) = sum_y res_y [p_y <= p_x][y not EOG] / Z, for x in the draft candidates
    le = (p[:, None, :] <= p_at_q[:, :, None]) & ~p_eog[:, None, :]    # [R, Kq, Kp]
    Rx = (le.to(res.dtype) * res[:, None, :]).sum(-1) / Z.clamp_min(1e-30)
    Rx = torch.where(q_eog | (Z <= 0), torch.zeros_like(Rx), Rx)
    a = m + (1 - m) * Rx
    J = q * a
    Amb = J.sum(-1)
    Jt = (J * (qid == t[:, None])).sum(-1)
    # p'(t): J(t) + res(t)/Z * sum_x q_x (1 - m_x) (1 - [x, t not EOG][p_x >= p_t])
    rest = (res * (pid == t[:, None])).sum(-1)
    t_eog = torch.isin(t, eog_t)
    resc = (p_at_q >= pt[:, None]) & ~q_eog & ~t_eog[:, None]
    rej = (q * (1 - m) * (~resc).to(q.dtype)).sum(-1)
    pprime = Jt + rest / Z.squeeze(-1).clamp_min(1e-30) * rej
    out.update(Amb=Amb, cmb_is=torch.where(pt > 0, Jt / pt.clamp_min(1e-30), torch.zeros_like(Jt)),
               cmb=torch.where(pprime > 0, torch.clamp(Jt / pprime.clamp_min(1e-30), max=1.0), torch.zeros_like(Jt)))
    return out


def chain_E(A, c):
    """A, c [nA, d] -> E [nA, d] cumulative: E_d = sum_{s<=d} prod_{j<s} c_j A_s."""
    reach = torch.cumprod(torch.cat([torch.ones_like(c[:, :1]), c[:, :-1]], 1), 1)
    return torch.cumsum(reach * A, 1), reach


def round_sim(c, valid):
    """Server-like rounds over one sequence's anchors (index = position - first anchor). c [n, d] acceptance
    probability of slot s given its output token. Returns (sum pi, sum pi * tokens) with tokens = 1 + accepted."""
    c = c.double().cpu().numpy()
    n, d = c.shape
    pi = np.zeros(n + d + 2)
    pi[0] = 1.0
    S = T = 0.0
    for a in range(n):
        if pi[a] == 0 or not valid[a]:
            continue
        pr = 1.0
        for k in range(d + 1):
            if k < d:
                stop = pr * (1 - c[a, k])
            else:
                stop = pr
            S += pi[a] * stop
            T += pi[a] * stop * (k + 1)
            pi[a + k + 1] += pi[a] * stop
            if k < d:
                pr *= c[a, k]
    return S, T


# ---------------------------------------------------------------- dump data
class DumpSet:
    """PREFIX.{feat,top,idx} from llama-dspark-dump. seq(i) -> dict(toks, n_prompt, feat [len, 4, E] memmap,
    top_ids/top_vals [len - n_prompt + 1, K] for positions n_prompt-1 .. len-1)."""

    def __init__(self, prefix, seq_bin):
        hdr = open(prefix + '.idx').readline().split()
        h = dict(zip(hdr[1::2], hdr[2::2]))
        self.dtype = h['dtype']; self.nl = int(h['n_layers']); self.ne = int(h['n_embd']); self.K = int(h['topk'])
        self.rows = [tuple(int(x) for x in l.split()) for l in open(prefix + '.idx') if not l.startswith('#')]
        el = {'bf16': np.uint16, 'f16': np.float16, 'f32': np.float32}[self.dtype]
        self.feat = np.memmap(prefix + '.feat', dtype=el, mode='r').reshape(-1, self.nl, self.ne)
        self.top = np.memmap(prefix + '.top', dtype=np.int32, mode='r').reshape(-1, 2 * self.K)
        raw = np.fromfile(seq_bin, dtype=np.int32)
        self.toks, o = [], 1
        for _ in range(int(raw[0])):
            L, npr = int(raw[o]), int(raw[o + 1])
            self.toks.append(raw[o + 2:o + 2 + L]); o += 2 + L
        assert len(self.toks) == len(self.rows)
        self.eog = [int(x) for x in open(prefix + '.eog').read().split()] if os.path.exists(prefix + '.eog') else list(EOG_IDS)

    def __len__(self):
        return len(self.rows)

    def seq(self, i, device):
        s, fo, L, npr, to = self.rows[i]
        f = np.asarray(self.feat[fo:fo + L])
        if self.dtype == 'bf16':
            feat = torch.from_numpy(f.view(np.int16).copy()).to(device).view(torch.bfloat16)
        else:
            feat = torch.from_numpy(f.copy()).to(device)
        nt = L - max(0, npr - 1)
        t = np.asarray(self.top[to:to + nt])
        ids = torch.from_numpy(t[:, :self.K].copy()).to(device).long()
        vals = torch.from_numpy(t[:, self.K:].copy().view(np.float32)).to(device)
        return dict(toks=torch.from_numpy(self.toks[i].astype(np.int64)).to(device), n_prompt=npr, len=L, feat=feat,
                    top_ids=ids, top_vals=vals, row0=max(0, npr - 1), eog=torch.tensor(self.eog, device=device))


def seq_slots(W, frozen, S, anchors, walk, dtype=torch.float32, bf16_base=False, chunk=128, ctx=None):
    """Yield per anchor chunk: (anchor positions [n], list over slots s=1..walk of drafter logits [n, V_UNPAD], p [n, 20],
    pid [n, 20], data tokens [n, walk]). ctx = precomputed context_layers output (reuse across chunks)."""
    if ctx is None:
        with torch.no_grad():
            ctx = context_layers_sliced(W, S['feat'], int(anchors.max()), anchors.device, dtype)
    kv, f_rnn = ctx
    toks = S['toks']
    for i in range(0, anchors.shape[0], chunk):
        a = anchors[i:i + chunk]
        cur = block_forward(W, frozen, kv, a, toks[a], 0, dtype)
        offs = torch.arange(1, walk + 1, device=a.device)
        tdat = toks[a[:, None] + offs[None, :]]                      # data tokens of slots 1..walk
        prevs = torch.cat([toks[a][:, None], tdat[:, :-1]], 1)
        rows = a[:, None] + offs[None, :] - 1 - S['row0']             # p of slot s = target logits at a+s-1
        logits = [l for _, l in walk_teacher(W, frozen, cur, f_rnn[a - 1], prevs, walk, bf16_base)]
        yield a, logits, rows, tdat


def anchors_of(S, walk, stride=1, offset=0):
    lo, hi = S['n_prompt'], S['len'] - 1 - walk
    if hi < lo:
        return None
    return torch.arange(lo + offset, hi + 1, stride, device=S['toks'].device)
