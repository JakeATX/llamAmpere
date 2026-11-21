#!/usr/bin/env python3
"""Inkling 2 Mini routing-trace analysis (topk.L<il>.i32 from llama-inklsa-dump INKDUMP_TOPK=1).
Visits il 0..47 -> physical layer phys(il). Cache entries = (phys layer, expert). Reports:
 - distinct experts per token (T=1) and per verify round (union of R consecutive tokens), loop sharing
 - hit rates for a VRAM expert cache of C slots: static (top-C by frequency on the first half, evaluated on the second),
   LRU, and static-half + LRU-half; misses per token / per round
 - per-expert usage histogram per physical layer and per visit -> expert_usage.json"""
import numpy as np, json, sys, os, collections
D = sys.argv[1]; OUT = sys.argv[2]
NE, NU, NPHYS, NV = 128, 8, 24, 48
def phys(il): return il if il < 24 else 6 + (il - 24) % 12
# visit order of a token: 0..17 (core pass 0 = 6..17), then 24..35 (pass 1), 36..47 (pass 2), then 18..23
order = list(range(0, 18)) + list(range(24, 48)) + list(range(18, 24))
tk = {il: np.fromfile(f'{D}/topk.L{il}.i32', dtype=np.int32).reshape(-1, NU) for il in range(NV)}
N = min(v.shape[0] for v in tk.values())
print('tokens', N)
usage_v = {il: np.bincount(tk[il][:N].ravel(), minlength=NE).tolist() for il in range(NV)}
usage_p = collections.defaultdict(lambda: np.zeros(NE, np.int64))
for il in range(NV): usage_p[phys(il)] += np.array(usage_v[il])
json.dump(dict(tokens=int(N), n_expert=NE, n_used=NU, note='counts of selections; visits il 24-47 are loop passes 1-2 of physical layers 6-17 (phys = 6 + (il-24)%12)',
               per_visit={str(k): v for k, v in usage_v.items()}, per_phys={str(k): v.tolist() for k, v in usage_p.items()}),
          open(os.path.join(OUT, 'expert_usage.json'), 'w'))
# keys per token in visit order
keys = np.stack([phys(il) * NE + tk[il][:N] for il in order], 1)          # [N, 48, 8]
flat = keys.reshape(N, -1)
dist_tok = np.array([len(np.unique(r)) for r in flat])
loopshare = []
for p in range(6, 18):
    a = [set(tk[il][:N][t]) for il in (p, 24 + p - 6, 36 + p - 6) for t in [0]]
u = [len(set(tk[p][t]) | set(tk[24 + p - 6][t]) | set(tk[36 + p - 6][t])) for p in range(6, 18) for t in range(0, N, 7)]
res = dict(tokens=int(N), distinct_per_token_mean=float(dist_tok.mean()), expert_visits_per_token=NV * NU,
           loop_union_per_phys_layer_mean=float(np.mean(u)))
for R in (1, 4):
    rounds = [np.unique(flat[i:i + R]) for i in range(0, N - R + 1, R)]
    res[f'distinct_per_round_R{R}'] = float(np.mean([len(r) for r in rounds]))
    half = len(rounds) // 2
    freq = collections.Counter(); [freq.update(r.tolist()) for r in rounds[:half]]
    rank = [k for k, _ in freq.most_common()]
    for C in [int(x) for x in os.environ.get("ROUTE_C", "512,1024,1536,1843,2048,2304,2560,2765,2900").split(",")]:
        st = set(rank[:C]); lru = collections.OrderedDict(); mix_st = set(rank[:C // 2]); mix = collections.OrderedDict()
        ms = ml = mm = 0; tot = 0
        for r in rounds[half:]:
            for k in r.tolist():
                tot += 1
                ms += k not in st
                if k in lru: lru.move_to_end(k)
                else:
                    ml += 1; lru[k] = 1
                    if len(lru) > C: lru.popitem(last=False)
                if k not in mix_st:
                    if k in mix: mix.move_to_end(k)
                    else:
                        mm += 1; mix[k] = 1
                        if len(mix) > C - C // 2: mix.popitem(last=False)
        n = len(rounds) - half
        res[f'R{R}_C{C}'] = dict(frac=C / (NPHYS * NE), static_hit=1 - ms / tot, lru_hit=1 - ml / tot, mix_hit=1 - mm / tot,
                                 miss_per_round_static=ms / n, miss_per_round_lru=ml / n, miss_per_round_mix=mm / n)
        print(R, C, {k: round(v, 3) for k, v in res[f'R{R}_C{C}'].items()}, flush=True)
# usage skew: share of selections covered by the top x% experts per phys layer
sk = {}
for f in (0.1, 0.25, 0.5):
    sk[str(f)] = float(np.mean([np.sort(v)[::-1][:int(f * NE)].sum() / v.sum() for v in usage_p.values()]))
res['top_frac_share'] = sk
print(json.dumps({k: v for k, v in res.items() if not k.startswith('R')}, indent=1))
json.dump(res, open(os.path.join(OUT, 'route_sim.json'), 'w'), indent=1)
