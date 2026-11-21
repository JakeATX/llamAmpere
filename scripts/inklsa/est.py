#!/usr/bin/env python3
"""Est. decode tok/s for the small-card hot-expert cache (no drafter, T=1), from route_sim.json.
All outputs are ESTIMATES. t_base = measured 24 GB P1 LSA keep 4096 at ~128K fill (50.31 tok/s, IQ4_NL, 3090 300 W)."""
import json, sys
R = json.load(open(sys.argv[1]))
T_BASE = 1000 / 50.31                     # ms/token, GPU, all experts resident
EXP_MIB_PER_BPW = 2916.0 / 3072           # MiB per expert per bpw (gate+up+down)
KV_MIB = 14.0                             # LSA host-body fetch per token (pool 2x keep 4096, kvpool sim; est.)
BW = {'x16': 25.0, 'x8': 12.5, 'x4': 6.0} # GB/s effective PCIe 4.0 (est.)
DDR = 40.0                                # GB/s effective DDR5 read for CPU expert matvec (est.)
SYNC_PCIE, SYNC_CPU = 0.5, 2.0            # ms/token: 48 visits of fetch launch / GPU<->CPU handoff (est.)
cases = [  # card, label, bpw, slots
    ('8 GB', 'AIT2_U', 2.125, 2755), ('8 GB', '3.0', 3.0, 1951), ('8 GB', '3.4', 3.4, 1722), ('8 GB', '3.8', 3.8, 1540),
    ('10 GB', '3.0', 3.0, 2671), ('10 GB', '3.4', 3.4, 2356), ('10 GB', '3.8', 3.8, 2108),
    ('12 GB', '3.4', 3.4, 2991), ('12 GB', '3.8', 3.8, 2671 if 'R1_C2671' in R else 2672)]
def get(R, rr, c):
    k = f'R{rr}_C{c}'
    if k not in R:  # nearest simulated size
        cs = sorted(int(x.split('_C')[1]) for x in R if x.startswith(f'R{rr}_C'))
        c = min(cs, key=lambda x: abs(x - c)); k = f'R{rr}_C{c}'
    return R[k]
print('card | bpw | slots | cache frac | hit static/LRU/mix | best miss/token | miss MiB/token | est tok/s PCIe x16/x8/x4 | est tok/s CPU-cold (static set: miss/token, MiB)')
for card, lab, b, c in cases:
    r = get(R, 1, c); best = max(('static', r['static_hit']), ('lru', r['lru_hit']), ('mix', r['mix_hit']), key=lambda x: x[1])
    miss = r['miss_per_round_' + best[0]]; mib = miss * EXP_MIB_PER_BPW * b
    t = {k: T_BASE + KV_MIB * 1.048576 / v + mib * 1.048576 / v + SYNC_PCIE for k, v in BW.items()}
    smib = r['miss_per_round_static'] * EXP_MIB_PER_BPW * b   # CPU-cold: static hot set, misses computed on the CPU
    tc = T_BASE + KV_MIB * 1.048576 / BW['x8'] + smib * 1.048576 / DDR + SYNC_CPU
    print(f"{card} | {lab} | {c} | {c/3072:.2f} | {r['static_hit']:.3f}/{r['lru_hit']:.3f}/{r['mix_hit']:.3f} | {miss:.1f} | {mib:.0f} | "
          + '/'.join(f'{1000/t[k]:.1f}' for k in BW) + f" | {1000/tc:.1f} ({r['miss_per_round_static']:.0f}, {smib:.0f})")
