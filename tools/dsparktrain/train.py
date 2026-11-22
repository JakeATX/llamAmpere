#!/usr/bin/env python3
"""DSPARKTRAIN: train / evaluate the Inkling2 DSpark drafter at ship precision on on-policy dumps.

Forward = the server path (dspark.py): every Q8_0 matrix is exact gguf Q8_0 fake-quant of its fp32 master (STE),
the fold is fp16(lm_truth . bf16(H)) (STE through both roundings), F32 tensors as is, base = AI4 output.weight Q8_0.
Init = the shipped drafter (Q8_0 codes are their own fixed point, so step 0 == the shipped file).
Loss (--loss): e4 = -E_d exact (sum_s prod_{j<s} c_j A_s, teacher-forced on-policy walk), mb = -E_d with the mb law
(IS weights), plus --kl * KL(p_s || softmax(l_s)) over the full vocab (gradient for tokens outside the drafter's top 20).
Eval (fp32, no TF32): alpha_s = sum reach_s A_s / sum reach_s, E_1..E_d exact and mb (IS), and the round simulation
(server-like tok/round: exact; mb approximate) over every anchor (stride 1) or a strided subset.
Checkpoints: best by --select (default E_d mb on the sel split) -> OUT/best.pt; OUT/log.jsonl.
usage: train.py --w0 W0 --drafter Q8_0-fold.gguf --bf16 BF16.gguf --out OUT [--train P:SEQ] [--sel P:SEQ] [--rep P:SEQ]
                [--steps N] [--eval-only] [--init ckpt.pt] [--profile N]"""
import argparse, json, math, os, random, sys, time
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
    fold = lm.float() @ Hb
    W['fold'] = fold + (fold.detach().half().float() - fold.detach())
    W['fold_b'] = lm.float() @ b
    return W


def slot_terms(l, S, rows, t, mb):
    p = D.target_p(S['top_vals'][rows])
    pid = S['top_ids'][rows][:, :p.shape[1]]
    q, qid, _ = D.draft_q(l)
    return D.accept_terms(p, pid, q, qid, t, eog=S['eog'], want_mb=mb), p, pid


@torch.no_grad()
def evaluate(P, fr, ds, walk, stride=1, max_seq=0, fold_fixed=None, roundsim=True, tag=''):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    W = D.effective({k: v.detach() for k, v in P.items()}, fr, fold_fixed)
    acc = dict(n=0, sumA=np.zeros(walk), sumR=np.zeros(walk), sumAmb=np.zeros(walk), sumRmb=np.zeros(walk),
               E=np.zeros(walk), Emb=np.zeros(walk), rs=[0.0, 0.0], rsmb=[0.0, 0.0], pt0=0, n_seq=0)
    idx = list(range(len(ds)))
    if max_seq:
        idx = idx[:max_seq]
    for i in idx:
        S = ds.seq(i, fr.lm.device)
        an = D.anchors_of(S, walk, stride)
        if an is None:
            continue
        cs, cms, valid_all = [], [], []
        for a, logits, rows, tdat in D.seq_slots(W, fr, S, an, walk):
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
    ap.add_argument('--eval-every', type=int, default=100); ap.add_argument('--eval-stride', type=int, default=4)
    ap.add_argument('--select', default='E_mb'); ap.add_argument('--max-eval-seq', type=int, default=0)
    ap.add_argument('--profile', type=int, default=0, help='run N train steps, print per-step timings, exit')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-act-q', action='store_true', help='plain fp32 activations (no engine rounding emulation)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = a.device
    D.ACT_Q = not a.no_act_q
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

    tr = dset(a.train); sel = dset(a.sel)
    params = [torch.nn.Parameter(v) for v in P.values()]
    names = list(P.keys())
    for n, p in zip(names, params):
        P[n] = p
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    best = None
    if sel is not None and not a.profile:
        r = evaluate(P, fr, sel, a.walk, a.eval_stride, a.max_eval_seq, tag='sel'); best = r[a.select][-1]
        emit(dict(step=0, eval='sel', **r))
        torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'best.pt'))
    order = list(range(len(tr)))
    random.shuffle(order); oi = 0
    tstep = []
    for step in range(1, (a.profile or a.steps) + 1):
        sync(dev); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        W = ste_weights(P, fr)
        n_tot, lsum, esum, ksum = 0, 0.0, 0.0, 0.0
        batches = []
        for _ in range(a.seqs_per_step):
            S = tr.seq(order[oi % len(order)], dev); oi += 1
            if oi % len(order) == 0:
                random.shuffle(order)
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
            kv, f_rnn = D.context_layers(Wl, S['feat'][:amax].float(), torch.arange(amax, device=dev))
            ctx_out = [t for pair in kv for t in pair] + [f_rnn]
            ctx_l = [t.detach().requires_grad_(True) for t in ctx_out]
            ctx = ([(ctx_l[2 * j], ctx_l[2 * j + 1]) for j in range(D.N_LAYER)], ctx_l[-1])
            for i in range(0, an.shape[0], a.chunk):
                sub = an[i:i + a.chunk]
                (aa, logits, rows, tdat), = list(D.seq_slots(Wl, fr, S, sub, a.walk, bf16_base=a.bf16_base, chunk=a.chunk, ctx=ctx))
                A, C, KL = [], [], 0.0
                for s, l in enumerate(logits):
                    r, p, pid = slot_terms(l, S, rows[:, s], tdat[:, s], a.loss == 'mb')
                    A.append(r['Amb'] if a.loss == 'mb' else r['A']); C.append(r['cmb_is'] if a.loss == 'mb' else r['c'])
                    if a.kl:
                        ls = torch.log_softmax(l, -1).gather(1, pid)
                        KL = KL + (p * (torch.log(p.clamp_min(1e-30)) - ls)).sum(-1)
                E, _ = D.chain_E(torch.stack(A, 1), torch.stack(C, 1))
                Ed = E[:, -1]
                loss = (-Ed + (a.kl * KL if a.kl else 0)).sum() / n_all
                loss.backward()
                lsum += float(loss.detach()) * n_all; esum += float(Ed.sum()); ksum += float(KL.sum()) if a.kl else 0.0
                n_tot += sub.shape[0]
            g = [(o, l.grad) for o, l in zip(ctx_out, ctx_l) if l.grad is not None]
            torch.autograd.backward([o for o, _ in g], [gg for _, gg in g])
        g = [(W[k], Wl[k].grad) for k in W if Wl[k].grad is not None and W[k].requires_grad]
        torch.autograd.backward([o for o, _ in g], [gg for _, gg in g])
        if step == 1:
            emit(dict(step=1, params_without_grad=[n for n, p in zip(names, params) if p.grad is None or not bool(p.grad.abs().sum() > 0)]))
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step(); sched.step()
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
    if a.profile:
        torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, 'last.pt'))
        emit(dict(profile_steps=len(tstep), mean_step_s=float(np.mean(tstep[1:] if len(tstep) > 1 else tstep))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
