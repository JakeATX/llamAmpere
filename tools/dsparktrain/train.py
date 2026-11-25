#!/usr/bin/env python3
"""DSPARKTRAIN: train / evaluate the Inkling2 DSpark drafter at ship precision on on-policy dumps.

Forward = the server path (dspark.py): every Q8_0 matrix is exact gguf Q8_0 fake-quant of its fp32 master (STE),
the fold is fp16(lm_truth . bf16(H)) (STE through both roundings), F32 tensors as is, base = AI4 output.weight Q8_0.
Init = the shipped drafter (Q8_0 codes are their own fixed point, so step 0 == the shipped file).
Loss (--loss): e4 = -E_d exact (sum_s prod_{j<s} c_j A_s, teacher-forced on-policy walk), mb = -E_d with the mb law
(IS weights), plus --kl * KL(p_s || softmax(l_s)) over the full vocab (gradient for tokens outside the drafter's top 20).
Eval (fp32, no TF32): alpha_s = sum reach_s A_s / sum reach_s, E_1..E_d exact and mb (IS), and the round simulation
(server-like tok/round: exact; mb approximate) over every anchor (stride 1) or a strided subset.
WORD weighting (--word-wt W --word-mask M.npy, default off): slot s of an anchor counts 1 + W in the loss when its
on-policy token t_{a+s} is a whole-word piece (mkwordmask.py): objective sum_s w_s prod_{j<s} c_j A_s, KL_s scaled by w_s.
Selection and every eval number stay unweighted (E_d); evals also print alpha_mb split into WORD and other slots.
Checkpoints: best by --select (default E_d mb on the sel split) -> OUT/best.pt; OUT/log.jsonl.
usage: train.py --w0 W0 --drafter Q8_0-fold.gguf --bf16 BF16.gguf --out OUT [--train P:SEQ] [--sel P:SEQ] [--rep P:SEQ]
                [--steps N] [--eval-only] [--init ckpt.pt] [--profile N]"""
import argparse, gc, json, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark as D  # noqa: E402


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


def dset(spec):
    if not spec:
        return None
    p, s = spec.split(':')
    return D.DumpSet(p, s)


def ste_weights(P, fr, fold_fixed=None):
    W = {}
    for n, v in P.items():
        if n == D.H_NAME:
            continue
        if n in D.GLOBAL_MATS or n.endswith(tuple(f'{m}.weight' for m in D.LAYER_MATS)):
            W[n] = v + (D.q8_0_fq(v) - v).detach()
        else:
            W[n] = v
    if fold_fixed is not None:
        W['fold'], W['fold_b'] = fold_fixed
        return W
    H, b = P[D.H_NAME], P['dspark_hidden_out.bias']
    Hb = H + (H.detach().to(torch.bfloat16).float() - H.detach())
    lm = fr.lm[:D.V_UNPAD]
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # the fold is fp32 like ink_dspark_fold.py (then fp16)
    fold = lm.float() @ Hb
    fold_b = lm.float() @ b
    torch.backends.cuda.matmul.allow_tf32 = prev
    W['fold'] = fold + (fold.detach().half().float() - fold.detach())
    W['fold_b'] = fold_b
    return W


def slot_terms(l, S, rows, t, mb):
    p = D.target_p(S['top_vals'][rows])
    pid = S['top_ids'][rows][:, :p.shape[1]]
    q, qid, _ = D.draft_q(l)
    return D.accept_terms(p, pid, q, qid, t, eog=S['eog'], want_mb=mb), p, pid


MEMDBG = os.environ.get('DSQ_MEMDBG') == '1'
WORD_MASK = None   # bool [n_vocab] on the device (--word-mask): True = whole-word token


def word_of(t):
    return WORD_MASK[t.clamp(0, WORD_MASK.shape[0] - 1)]


def memlog(tag):
    if MEMDBG:
        print('MEM', tag, round(torch.cuda.memory_allocated() / 2**30, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)


CTX_SLICE = 8192   # context positions per slice (train)
LONG_LEN = 16384   # sequences longer than this use the small chunk (train --chunk-long, eval 16)
EVAL_CHUNK = int(os.environ.get('DSPARK_EVAL_CHUNK', '128'))   # anchors per eval chunk (32 for ~100K-token sequences)


@torch.no_grad()
def evaluate(P, fr, ds, walk, stride=1, max_seq=0, fold_fixed=None, roundsim=True, tag=''):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    W = D.effective({k: v.detach() for k, v in P.items()}, fr, fold_fixed)
    acc = dict(n=0, sumA=np.zeros(walk), sumR=np.zeros(walk), sumAmb=np.zeros(walk), sumRmb=np.zeros(walk),
               E=np.zeros(walk), Emb=np.zeros(walk), rs=[0.0, 0.0], rsmb=[0.0, 0.0], pt0=0, n_seq=0, wA=np.zeros((2, walk)), wR=np.zeros((2, walk)))
    idx = list(range(len(ds)))
    if max_seq:
        idx = idx[:max_seq]
    for i in idx:
        S = ds.seq(i, fr.lm.device)
        if S['len'] > LONG_LEN:
            S['feat'] = S['feat'].cpu().pin_memory()   # ~100K-token features stay on host; context slices move on their own
        an = D.anchors_of(S, walk, stride)
        if an is None:
            continue
        cs, cms, valid_all = [], [], []
        ch = EVAL_CHUNK if S['len'] <= LONG_LEN else min(EVAL_CHUNK, 16)   # ~100K-token sequences: small chunks (full-attn layer 1)
        for a, logits, rows, tdat in D.seq_slots(W, fr, S, an, walk, chunk=ch):
            A, C, Am, Cm, Cmc = [], [], [], [], []
            for s, l in enumerate(logits):
                r, _, _ = slot_terms(l, S, rows[:, s], tdat[:, s], True)
                A.append(r['A']); C.append(r['c']); Am.append(r['Amb']); Cm.append(r['cmb_is']); Cmc.append(r['cmb'])
                acc['pt0'] += int(r['pt0'].sum())
            A, C, Am, Cm, Cmc = (torch.stack(x, 1).double() for x in (A, C, Am, Cm, Cmc))
            E, reach = D.chain_E(A, C)
            Em, reachm = D.chain_E(Am, Cm)
            acc['n'] += A.shape[0]
            acc['sumA'] += (reach * A).sum(0).cpu().numpy(); acc['sumR'] += reach.sum(0).cpu().numpy()
            acc['sumAmb'] += (reachm * Am).sum(0).cpu().numpy(); acc['sumRmb'] += reachm.sum(0).cpu().numpy()
            acc['E'] += E.sum(0).cpu().numpy(); acc['Emb'] += Em.sum(0).cpu().numpy()
            if WORD_MASK is not None:   # mb acceptance of WORD vs other slots (class of the slot's on-policy token)
                wd = word_of(tdat)
                for k, sel_ in ((0, wd), (1, ~wd)):   # per slot depth: WORD share changes with depth, so compare per k
                    acc['wA'][k] += ((reachm * Am) * sel_).sum(0).cpu().numpy(); acc['wR'][k] += (reachm * sel_).sum(0).cpu().numpy()
            cs.append(C); cms.append(Cmc)
        acc['n_seq'] += 1
        if roundsim and stride == 1:
            c = torch.cat(cs); cm = torch.cat(cms)
            v = np.ones(c.shape[0], bool)
            for key, cc in (('rs', c), ('rsmb', cm)):
                S_, T_ = D.round_sim(cc, v)
                acc[key][0] += S_; acc[key][1] += T_
    torch.backends.cuda.matmul.allow_tf32 = prev
    n = max(acc['n'], 1)
    out = dict(tag=tag, n_anchors=acc['n'], n_seq=acc['n_seq'], stride=stride,
               alpha=(acc['sumA'] / np.maximum(acc['sumR'], 1e-30)).round(4).tolist(),
               alpha_mb=(acc['sumAmb'] / np.maximum(acc['sumRmb'], 1e-30)).round(4).tolist(),
               E=(acc['E'] / n).round(4).tolist(), E_mb=(acc['Emb'] / n).round(4).tolist(),
               tok_round_sim=acc['rs'][1] / acc['rs'][0] if acc['rs'][0] else None,
               tok_round_sim_mb=acc['rsmb'][1] / acc['rsmb'][0] if acc['rsmb'][0] else None,
               p_t_zero_rows=acc['pt0'], secs=round(time.time() - t0, 2))
    out['anchors_per_s'] = round(acc['n'] / max(out['secs'], 1e-9), 1)
    if WORD_MASK is not None:
        # WORD = the slot's on-policy (target) token is a whole-word piece; per slot k = 1..walk, plus pooled
        out['alpha_mb_word'] = (acc['wA'][0] / np.maximum(acc['wR'][0], 1e-30)).round(4).tolist()
        out['alpha_mb_other'] = (acc['wA'][1] / np.maximum(acc['wR'][1], 1e-30)).round(4).tolist()
        out['alpha_mb_word_pooled'] = round(acc['wA'][0].sum() / max(acc['wR'][0].sum(), 1e-30), 4)
        out['alpha_mb_other_pooled'] = round(acc['wA'][1].sum() / max(acc['wR'][1].sum(), 1e-30), 4)
        out['word_reach_share'] = round(acc['wR'][0].sum() / max(acc['wR'].sum(), 1e-30), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w0', required=True); ap.add_argument('--drafter', required=True); ap.add_argument('--bf16', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train'); ap.add_argument('--sel'); ap.add_argument('--rep')
    ap.add_argument('--init'); ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--shipped-fold', action='store_true', help='eval with the drafter file\'s own fold (step-0 identity check)')
    ap.add_argument('--walk', type=int, default=4)
    ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--warmup', type=int, default=20); ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--loss', default='e4', choices=['e4', 'mb']); ap.add_argument('--kl', type=float, default=0.1)
    ap.add_argument('--seqs-per-step', type=int, default=4); ap.add_argument('--anchors-per-seq', type=int, default=512)
    ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--bf16-base', action='store_true')
    ap.add_argument('--ckpt-every', type=int, default=0, help='also save ckpt_<step>.pt every N steps')
    ap.add_argument('--stop-no-gain', type=int, default=0, help='stop at this step if sel never beat step 0')
    ap.add_argument('--chunk-long', type=int, default=32, help='anchor chunk for sequences longer than LONG_LEN')
    ap.add_argument('--train2', help='second train set P:SEQ (e.g. the long-context dumps), drawn with probability --frac2')
    ap.add_argument('--frac2', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=100); ap.add_argument('--eval-stride', type=int, default=4)
    ap.add_argument('--select', default='E_mb'); ap.add_argument('--max-eval-seq', type=int, default=0)
    ap.add_argument('--profile', type=int, default=0, help='run N train steps, print per-step timings, exit')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-act-q', action='store_true', help='plain fp32 activations (no engine rounding emulation)')
    ap.add_argument('--word-wt', type=float, default=0.0, help='extra loss weight of whole-word slots (0 = off, the unweighted loss)')
    ap.add_argument('--word-mask', help='mkwordmask.py output (bool [n_vocab]); needed by --word-wt, also adds the WORD/other eval split')
    a = ap.parse_args()
    if a.word_wt and not a.word_mask:
        ap.error('--word-wt needs --word-mask')
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = a.device
    D.ACT_Q = not a.no_act_q
    global WORD_MASK
    if a.word_mask:
        WORD_MASK = torch.from_numpy(np.load(a.word_mask)).to(dev)
    fr = D.Frozen(a.w0, dev)
    P = {k: v.to(dev) for k, v in D.load_drafter_masters(a.drafter, a.bf16).items()}
    if a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=True)
        for k in P:
            P[k] = sd[k].to(dev).float()
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')

    def emit(d):
        print(json.dumps(d), flush=True); log.write(json.dumps(d) + '\n'); log.flush()

    fold_fixed = None
    if a.shipped_fold:
        t = D.load_gguf_tensors(a.drafter)
        fold_fixed = (torch.from_numpy(D.gguf_f32(t['dspark_hidden_out.fold'])).to(dev),
                      torch.from_numpy(D.gguf_f32(t['dspark_hidden_out.fold_bias'])).to(dev))
    if a.eval_only:
        for name in ('sel', 'rep'):
            ds = dset(getattr(a, name))
            if ds is not None:
                emit(dict(eval=name, **evaluate(P, fr, ds, a.walk, 1, a.max_eval_seq, fold_fixed, tag=name)))
        return 0

    tr = dset(a.train); tr2 = dset(a.train2); sel = dset(a.sel)
    params = [torch.nn.Parameter(v) for v in P.values()]
    names = list(P.keys())
    for n, p in zip(names, params):
        P[n] = p
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    best = None
    if sel is not None and not a.profile:
        r = evaluate(P, fr, sel, a.walk, a.eval_stride, a.max_eval_seq, tag='sel'); best = r[a.select][-1]; best0 = best
        emit(dict(step=0, eval='sel', **r))
        torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'best.pt'))
    order = list(range(len(tr)))
    random.shuffle(order); oi = 0
    order2 = list(range(len(tr2))) if tr2 is not None else []
    random.shuffle(order2); oi2 = 0
    tstep = []

    def pick_step():
        nonlocal oi, oi2
        picks = []
        for _ in range(a.seqs_per_step):
            if tr2 is not None and random.random() < a.frac2:
                picks.append((tr2, order2[oi2 % len(order2)])); oi2 += 1
                if oi2 % len(order2) == 0:
                    random.shuffle(order2)
            else:
                picks.append((tr, order[oi % len(order)])); oi += 1
                if oi % len(order) == 0:
                    random.shuffle(order)
        return picks

    def load_cpu(picks):   # disk -> pinned host memory (features of ~100K-token sequences are ~1.8 GB each)
        return [{k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in ds.seq(i, 'cpu').items()} for ds, i in picks]

    pool = ThreadPoolExecutor(1); fut = pool.submit(load_cpu, pick_step())
    for step in range(1, (a.profile or a.steps) + 1):
        sync(dev); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        memlog('step%d start' % step)
        if MEMDBG and step > 1:
            import collections
            cnt = collections.Counter((tuple(o.shape), str(o.dtype)) for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda)
            print('LIVE', sorted(((s, d, n) for (s, d), n in cnt.items()), key=lambda x: -int(np.prod(x[0])) * x[2])[:8], flush=True)
        W = ste_weights(P, fr)
        memlog('W')
        n_tot, lsum, esum, ksum = 0, 0.0, 0.0, 0.0
        batches = []
        cpu_seqs = fut.result(); fut = pool.submit(load_cpu, pick_step())   # next step's sequences load while this one trains
        for Sc in cpu_seqs:
            S = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) and k != 'feat' else v) for k, v in Sc.items()}   # feat stays pinned on host; context slices move on their own
            an = D.anchors_of(S, a.walk)
            if an is None:
                continue
            if an.shape[0] > a.anchors_per_seq:
                an = an[torch.randperm(an.shape[0], device=dev)[:a.anchors_per_seq]].sort().values
            batches.append((S, an))
        n_all = sum(an.shape[0] for _, an in batches)
        # gradient leaves: chunks backprop into detached copies of W (and of each sequence's context), which are then
        # backpropagated once into the masters, so the fold / STE / context graphs are not re-traversed per chunk
        Wl = {k: v.detach().requires_grad_(v.requires_grad) for k, v in W.items()}
        for S, an in batches:
            amax = int(an.max())
            # context K/V + f_rnn: computed without a graph (position-wise, in CTX_SLICE slices), used as gradient leaves by
            # the chunks, then recomputed slice by slice WITH a graph and backpropagated with the leaves' gradients (exact;
            # keeps ~100K-token sequences from holding the whole context graph at once)
            # f_rnn is only read at anchor-1: keep just those rows (saves a full [len, N_EMBD] leaf + its grad at ~100K)
            frow = an - 1
            # only the full-attention layer gets full-length leaves; SWA layers compute their window per chunk (LazyCtx)
            FL = [il for il in range(D.N_LAYER) if il not in D.SWA_LAYERS]
            with torch.no_grad():
                full = {}
                for il in FL:
                    K0 = torch.empty(amax, D.N_HEAD_KV, D.HD, device=dev); V0 = torch.empty_like(K0)
                    for s0 in range(0, amax, CTX_SLICE):
                        s1 = min(s0 + CTX_SLICE, amax)
                        K0[s0:s1], V0[s0:s1] = D.ctx_layer(Wl, S['feat'][s0:s1].to(dev, non_blocking=True).float(), torch.arange(s0, s1, device=dev), il)
                    full[il] = (K0.requires_grad_(True), V0.requires_grad_(True))
                fr0 = D.frnn_rows(Wl, S['feat'][frow.cpu() if S['feat'].device.type == 'cpu' else frow].to(dev).float())
            ctx_l = [x for il in FL for x in full[il]] + [fr0.requires_grad_(True)]
            del fr0
            ctx = ([full[il] if il in full else D.LazyCtx(Wl, S['feat'], il, amax, dev) for il in range(D.N_LAYER)], D.RowLookup(frow, ctx_l[-1]))
            del full, K0, V0
            memlog('ctx')
            ch = a.chunk if S['len'] <= LONG_LEN else a.chunk_long
            for i in range(0, an.shape[0], ch):
                sub = an[i:i + ch]
                (aa, logits, rows, tdat), = list(D.seq_slots(Wl, fr, S, sub, a.walk, bf16_base=a.bf16_base, chunk=ch, ctx=ctx))
                A, C, KL = [], [], 0.0
                wt = 1.0 + a.word_wt * word_of(tdat).float() if a.word_wt else None   # [n, walk] slot weights
                for s, l in enumerate(logits):
                    r, p, pid = slot_terms(l, S, rows[:, s], tdat[:, s], a.loss == 'mb')
                    A.append(r['Amb'] if a.loss == 'mb' else r['A']); C.append(r['cmb_is'] if a.loss == 'mb' else r['c'])
                    if a.kl:
                        ls = torch.log_softmax(l, -1).gather(1, pid)
                        kls = (p * (torch.log(p.clamp_min(1e-30)) - ls)).sum(-1)
                        KL = KL + (kls if wt is None else wt[:, s] * kls)
                A = torch.stack(A, 1); C = torch.stack(C, 1)
                E, reach = D.chain_E(A, C)
                Ed = E[:, -1]
                obj = Ed if wt is None else (wt * reach * A).sum(1)   # weighted E_d: WORD slots count 1 + word_wt
                loss = (-obj + (a.kl * KL if a.kl else 0)).sum() / n_all
                loss.backward()
                lsum += float(loss.detach()) * n_all; esum += float(Ed.sum()); ksum += float(KL.sum()) if a.kl else 0.0
                n_tot += sub.shape[0]
                loss = E = Ed = A = C = KL = logits = l = r = p = pid = ls = obj = reach = wt = None   # drop graph refs (they pin Wl/W + grads)
            memlog('chunks')
            for il_i, il in enumerate(FL):
                gK, gV = ctx_l[2 * il_i].grad, ctx_l[2 * il_i + 1].grad
                if gK is None and gV is None:
                    continue
                for s0 in range(0, amax, CTX_SLICE):
                    s1 = min(s0 + CTX_SLICE, amax)
                    k, v = D.ctx_layer(Wl, S['feat'][s0:s1].to(dev, non_blocking=True).float(), torch.arange(s0, s1, device=dev), il)
                    g = [(o, gg[s0:s1]) for o, gg in ((k, gK), (v, gV)) if gg is not None]
                    torch.autograd.backward([o for o, _ in g], [gg for _, gg in g])
                    del k, v, g
            if ctx_l[-1].grad is not None:
                fo = D.frnn_rows(Wl, S['feat'][frow.cpu() if S['feat'].device.type == 'cpu' else frow].to(dev).float())
                torch.autograd.backward([fo], [ctx_l[-1].grad])
                fo = None
            gK = gV = None
            del ctx_l, ctx, S
            gc.collect(); torch.cuda.empty_cache()
            memlog('seq done')
        g = [(W[k], Wl[k].grad) for k in W if Wl[k].grad is not None and W[k].requires_grad]
        torch.autograd.backward([o for o, _ in g], [gg for _, gg in g])
        if step == 1:
            emit(dict(step=1, params_without_grad=[n for n, p in zip(names, params) if p.grad is None or not bool(p.grad.abs().sum() > 0)]))
        del g, Wl, W, batches, cpu_seqs   # free this step's STE weights / leaf copies (+ their grads) before the next step builds new ones
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        memlog('pre-opt')
        opt.step(); sched.step()
        torch.cuda.empty_cache()
        memlog('post-opt')
        sync(dev); dt = time.time() - t0; tstep.append(dt)
        if step % 10 == 0 or a.profile:
            emit(dict(step=step, loss=lsum / max(n_tot, 1), E_train=esum / max(n_tot, 1), kl=ksum / max(n_tot, 1), anchors=n_tot,
                      secs=round(dt, 3), anchors_per_s=round(n_tot / dt, 1), lr=sched.get_last_lr()[0], grad_norm=round(gn, 4),
                      mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == 'cuda' else None))
        if sel is not None and not a.profile and (step % a.eval_every == 0 or step == a.steps):
            r = evaluate(P, fr, sel, a.walk, a.eval_stride, a.max_eval_seq, tag='sel')
            emit(dict(step=step, eval='sel', **r))
            if r[a.select][-1] > best:
                best = r[a.select][-1]
                torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'best.pt.tmp'))
                os.replace(os.path.join(a.out, 'best.pt.tmp'), os.path.join(a.out, 'best.pt'))
                emit(dict(step=step, saved='best.pt', best=best))
            if a.stop_no_gain and step >= a.stop_no_gain and best <= best0:
                emit(dict(step=step, stop='no_gain', best=best, base=best0)); break
        if a.ckpt_every and not a.profile and step % a.ckpt_every == 0:
            torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'ckpt.tmp'))
            os.replace(os.path.join(a.out, 'ckpt.tmp'), os.path.join(a.out, 'ckpt_%04d.pt' % step))
            emit(dict(step=step, saved='ckpt_%04d.pt' % step))
    if a.profile:
        torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'last.pt'))
        emit(dict(profile_steps=len(tstep), mean_step_s=float(np.mean(tstep[1:] if len(tstep) > 1 else tstep))))
        if sel is not None:   # one sel eval with the optimizer state allocated (memory check for the in-train evals)
            r = evaluate(P, fr, sel, a.walk, a.eval_stride, a.max_eval_seq, tag='sel')
            emit(dict(profile_eval='sel', **r, mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == 'cuda' else None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
