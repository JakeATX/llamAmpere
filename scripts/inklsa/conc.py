#!/usr/bin/env python3
"""INKLSA attention-concentration analysis on an inklsa-dump directory.

Exact Inkling 2 full-attention logits per sampled query row p (keys 0..p, causal):
  s_hj = scale * q_h . k_{g(h) j} + b_h(min(p - j, E)) ,  b_h(d) = rel_bias[:, d] . r_h (d < E), 0 (d >= E)
plus one sink logit t_h (per query), i.e. softmax over {s_hj} U {t_h}  (== the graph's mask - t, zero sink).

Regions: sinks = first S cells, window = last W cells (W >= E so every biased cell is in the window), middle = rest.
Middle cells are cut into blocks of B cells (aligned to cell 0). A pick keeps `keep` middle cells as whole blocks.

Selectors (per KV head g = the 3 q heads of the group share one pick, or 'shared' = one pick for all 24 heads):
  oracle : exact block mass (sum over the group's heads of the softmax mass)
  mean   : score_blk = logsumexp_h (scale * q_h . mean_k_blk) + log B    (training-free, block-mean keys)
  quest  : score_blk = max_h sum_c max(q_hc * kmax_c, q_hc * kmin_c)      (training-free, channel max/min bound)
Metrics per (visit, fill bucket, selector, keep): missed mass m (share of total incl. window and sink),
  mass-loss KL = -log(1 - m) (renormalised sparse vs dense, per head, mean), output rel. L2 error of o_h.
Also: middle-mass share, cells for 50/90/99 % of middle mass (per head, median), loop-pass pick overlap, step overlap.
"""
import argparse, json, math, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('dump')
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--layers', default='5,11,17,29,35,41,47,23')
ap.add_argument('--sinks', type=int, default=128)
ap.add_argument('--window', default='1024,4096')
ap.add_argument('--block', default='64,128')
ap.add_argument('--keeps', default='1024,2048,4096,8192,16384')
ap.add_argument('--fracs', default='0.02,0.05,0.10,0.25')
ap.add_argument('--qsub', type=int, default=2)  # use every qsub-th stride sample (dense runs always)
args = ap.parse_args()

sys.path.insert(0, '/mnt/data/inklsa/src/gguf-py')
import types; sys.modules.setdefault('yaml', types.ModuleType('yaml'))
import gguf

dev = torch.device('cuda')
NH, NKV, D, NREL = 24, 8, 128, 16
G = NH // NKV
scale = 1.0 / math.sqrt(D)

rd = gguf.GGUFReader(args.model)
E = None
for kv in rd.fields.values():
    if kv.name.endswith('attention.relative_extent'):
        E = int(kv.parts[kv.data[0]][0])
core_start, core_end, n_phys = 6, 18, 24
def phys(il):
    return il if il < n_phys else core_start + (il - n_phys) % (core_end - core_start)
relb = {}
for t in rd.tensors:
    if t.name.endswith('attn_rel_bias.weight'):
        il = int(t.name.split('.')[1])
        a = np.array(t.data, dtype=np.float32)
        relb[il] = a.reshape(-1, NREL)  # [extent, n_rel] (ggml ne = [n_rel, extent])
print('extent', E, 'rel_bias layers', sorted(relb), 'shape', relb[5].shape, flush=True)

meta = dict(l.split(' ', 1) for l in open(os.path.join(args.dump, 'meta.txt')).read().splitlines() if l.startswith('n_tokens'))
N = int(meta['n_tokens'])
layers = [int(x) for x in args.layers.split(',')]
windows = [int(x) for x in args.window.split(',')]
blocks = [int(x) for x in args.block.split(',')]
keeps = [int(x) for x in args.keeps.split(',')]
fracs = [float(x) for x in args.fracs.split(',')]

def bucket(p):
    return f'{(p // 32768) * 32}-{(p // 32768 + 1) * 32}K'

results = []        # dict rows
picks = {}          # (il, pos, W, B, keep, mode) -> oracle block set per group (for loop / step overlap)

for il in layers:
    K = np.fromfile(os.path.join(args.dump, f'K.L{il}.f16'), dtype=np.float16).reshape(-1, NKV, D)
    V = np.fromfile(os.path.join(args.dump, f'V.L{il}.f16'), dtype=np.float16).reshape(-1, NKV, D)
    Q = np.fromfile(os.path.join(args.dump, f'Q.L{il}.f32'), dtype=np.float32).reshape(-1, NH, D)
    R = np.fromfile(os.path.join(args.dump, f'R.L{il}.f32'), dtype=np.float32).reshape(-1, NH, NREL)
    S = np.fromfile(os.path.join(args.dump, f'S.L{il}.f32'), dtype=np.float32).reshape(-1, NH)
    P = np.fromfile(os.path.join(args.dump, f'pos.L{il}.i64'), dtype=np.int64)
    assert K.shape[0] == N and Q.shape[0] == P.shape[0] == R.shape[0] == S.shape[0], (K.shape, Q.shape, P.shape)
    Kt = torch.from_numpy(K).to(dev).float()          # [N, 8, 128]
    Vt = torch.from_numpy(V).to(dev).float()
    rb = torch.from_numpy(relb[phys(il)]).to(dev)       # [E, 16]
    print(f'L{il}: N={N} queries={len(P)}', flush=True)
    # block summaries per B: mean, max, min over cells, [nblk, 8, 128]
    summ = {}
    for B in blocks:
        nb = N // B
        kb = Kt[:nb * B].view(nb, B, NKV, D)
        summ[B] = (kb.mean(1), kb.amax(1), kb.amin(1))
    for qi in range(len(P)):
        p = int(P[qi])
        n = p + 1
        if n < 8192:
            continue
        in_run = p % 32768 >= 32768 - 64
        if not in_run and (p // 256) % args.qsub != 0:
            continue
        q = torch.from_numpy(Q[qi]).to(dev)                  # [24,128]
        r = torch.from_numpy(R[qi]).to(dev)                  # [24,16]
        t = torch.from_numpy(S[qi]).to(dev)                  # [24]
        kk = Kt[:n]                                          # [n,8,128]
        qg = q.view(NKV, G, D)
        logit = torch.einsum('ngd,gjd->gjn', kk, qg).reshape(NH, n) * scale   # [24, n]
        dist = torch.arange(n - 1, -1, -1, device=dev)
        near = dist < E
        bias = (rb @ r.T).T                                   # [24, E]
        logit[:, near] += bias[:, dist[near]]
        mx = torch.maximum(logit.amax(1), t)
        e = torch.exp(logit - mx[:, None])
        z = e.sum(1) + torch.exp(t - mx)
        pr = e / z[:, None]                                    # [24, n] mass per cell (sink share = 1 - sum)
        vv = Vt[:n]                                            # [n, 8, 128]
        # o_h = sum_j pr_hj v_{g(h) j}
        prg = pr.view(NKV, G, n)
        o_full = torch.einsum('ghn,ngd->ghd', prg, vv)          # [8,3,128]
        for W in windows:
            if n < W + args.sinks + 4 * max(blocks):
                continue
            mid0, mid1 = args.sinks, n - W                     # middle = [mid0, mid1)
            mid_mass = pr[:, mid0:mid1].sum(1)                 # [24]
            row = dict(il=il, pos=p, fill=n, bucket=bucket(p), W=W)
            # cells for 50/90/99% of middle mass, per head
            srt = torch.sort(pr[:, mid0:mid1], dim=1, descending=True).values
            cs = torch.cumsum(srt, 1) / mid_mass[:, None].clamp_min(1e-30)
            cells = {f'c{int(x*100)}': (cs < x).sum(1).float().median().item() + 1 for x in (0.5, 0.9, 0.99)}
            results.append(dict(row, kind='mid', mid_share=mid_mass.mean().item(), mid_share_max=mid_mass.max().item(),
                                sink=(1 - pr.sum(1)).mean().item(), n_mid=mid1 - mid0, **cells))
            for B in blocks:
                b0 = (mid0 + B - 1) // B                       # first whole block inside the middle
                b1 = mid1 // B                                 # blocks [b0, b1) are fully in the middle
                if b1 - b0 < 4:
                    continue
                # middle cells outside whole blocks (edges) are always kept (tiny)
                blk_mass = pr[:, b0 * B:b1 * B].view(NH, b1 - b0, B).sum(2)        # [24, nb]
                grp_mass = blk_mass.view(NKV, G, -1).sum(1)                         # [8, nb]
                mean_k, kmax, kmin = (x[b0:b1] for x in summ[B])                   # [nb,8,128]
                ms = torch.einsum('bgd,ghd->ghb', mean_k, qg) * scale               # [8,3,nb]
                s_mean = torch.logsumexp(ms, 1)                                     # [8, nb]
                ub = torch.maximum(torch.einsum('bgd,ghd->ghbd', kmax, qg), torch.einsum('bgd,ghd->ghbd', kmin, qg)).sum(3)
                s_quest = ub.amax(1)                                               # [8, nb]
                # 'vote' = the prototype's shared pick: sum over the 24 heads of softmax over blocks of the mean score
                vote = torch.softmax(ms.reshape(NH, -1), dim=1).sum(0, keepdim=True).expand(NKV, -1)
                sel_scores = {'oracle': grp_mass, 'mean': s_mean, 'quest': s_quest, 'vote': vote,
                              'oracle_sh': grp_mass.sum(0, keepdim=True).expand(NKV, -1),
                              'mean_sh': torch.logsumexp(s_mean, 0, keepdim=True).expand(NKV, -1)}
                # per-block partial outputs, so every selection's output is o_full minus its dropped blocks
                blkout = torch.einsum('ghbj,bjgd->ghbd', prg[:, :, b0 * B:b1 * B].reshape(NKV, G, b1 - b0, B),
                                      vv[b0 * B:b1 * B].reshape(b1 - b0, B, NKV, D))          # [8,3,nb,128]
                budgets = [('k', kc) for kc in keeps] + [('f', fr) for fr in fracs]
                for bt, bv in budgets:
                    kc = bv if bt == 'k' else int(bv * n)
                    nk = min(b1 - b0, max(1, kc // B))
                    if nk >= b1 - b0:
                        continue
                    for name, sc in sel_scores.items():
                        idx = torch.topk(sc, nk, dim=1).indices                     # [8, nk]
                        keepm = torch.zeros(NKV, b1 - b0, device=dev, dtype=torch.bool)
                        keepm.scatter_(1, idx, True)
                        if name == 'oracle' and B == 128 and W == 1024 and bt == 'k' and bv in (4096, 16384):
                            picks[(il, p, bv)] = keepm.cpu().numpy()
                        dropped = (blk_mass.view(NKV, G, -1) * (~keepm)[:, None, :]).sum(2).reshape(NH)   # [24]
                        miss = dropped
                        kl = (-torch.log1p(-miss.clamp(max=1 - 1e-9))).mean().item()
                        # sparse output: renormalised over the kept set (window + sinks + edges + kept blocks + sink logit)
                        dropm = (~keepm).float()                                              # [8, nb]
                        o_sp = (o_full - torch.einsum('ghbd,gb->ghd', blkout, dropm)) / (1 - dropped.view(NKV, G, 1)).clamp_min(1e-9)
                        err = ((o_sp - o_full).norm(dim=2) / o_full.norm(dim=2).clamp_min(1e-9)).mean().item()
                        results.append(dict(row, kind='sel', B=B, budget=f'{bt}{bv}', keep_cells=nk * B, sel=name,
                                            miss=miss.mean().item(), miss_max=miss.max().item(), kl=kl, err=err))
        if qi % 64 == 0:
            print(f'  L{il} q{qi}/{len(P)} pos {p}', flush=True)
    del Kt, Vt, summ
    torch.cuda.empty_cache()

with open(args.out, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
def olap(ka, kb):
    m = min(ka.shape[1], kb.shape[1])
    ka, kb = ka[:, :m], kb[:, :m]
    return float(((ka & kb).sum(1) / np.maximum(ka.sum(1), 1)).mean())
# loop-pass overlap (oracle, B=128, W=1024): same physical layer, same query, other pass
ov = []
for (a, b, c) in ((11, 29, 41), (17, 35, 47)):
    for (il, p, bv), ka in list(picks.items()):
        if il != a:
            continue
        for other in (b, c):
            kb = picks.get((other, p, bv))
            if kb is None:
                continue
            ov.append(dict(kind='loop', phys=a, pass0=a, other=other, pos=p, keep=bv, overlap=olap(ka, kb)))
# step overlap: consecutive sampled positions of the same visit
for (il, p, bv), ka in list(picks.items()):
    kb = picks.get((il, p + 1, bv))
    if kb is not None:
        ov.append(dict(kind='step', il=il, pos=p, keep=bv, overlap=olap(ka, kb)))
    kb = picks.get((il, p + 16, bv))
    if kb is not None:
        ov.append(dict(kind='step16', il=il, pos=p, keep=bv, overlap=olap(ka, kb)))
with open(args.out, 'a') as f:
    for r in ov:
        f.write(json.dumps(r) + '\n')
print('wrote', len(results) + len(ov), 'rows to', args.out)
