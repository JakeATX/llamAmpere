#!/usr/bin/env python3
"""Small-card fit for Inkling 2 Mini, Qwen small-card layout: token_embd on host; dense/attn/output at AI bits; drafter on card;
LSA: full-visit KV body on host, on-card = block index + window/sinks + pick pool (2x keep). Sizes in MiB. Est. = estimate."""
import json, sys
EXP_P = 24461.2e6; ATTN_P = 474.2e6 + 169.9e6; OUT_P = 463.2e6; SMALL = 61.6 + 13.5
M = 2**20
def mib(p, b): return p * b / 8 / M
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
KEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
COMPUTE = float(sys.argv[3]) if len(sys.argv) > 3 else 450.0   # compute buffers at dynamic-ub small-card ceiling (est. until measured)
DRAFT = float(sys.argv[4]) if len(sys.argv) > 4 else 432.5     # DSpark Q8_0-fold
dense_b, out_b = 2.33, 2.52                                    # Qwen 8 GB file's attn and output bpw
fix = dict(attn_dense=mib(ATTN_P, dense_b), output=mib(OUT_P, out_b), norms_router=SMALL, token_embd_host=0.0,
           drafter=DRAFT, swa_kv=40 * (1024 + 256) * 4096 / M, lsa_index=8 * (CTX // 64) * 1024 * 2 / M,
           lsa_window_sinks=8 * (1024 + 128 + 16) * 4096 / M, lsa_pool=8 * 2 * KEEP * 4096 / M, compute=COMPUTE)
F = sum(fix.values()); per_bpw = mib(EXP_P, 1)
print(f'ctx {CTX} keep {KEEP}: fixed on-card {F:.0f} MiB; experts {per_bpw:.0f} MiB per bpw; host: token_embd 469 MiB Q8_0 + KV body {8*CTX*4096/M:.0f} MiB')
for k, v in fix.items(): print(f'  {k:18s} {v:7.0f}')
for card in (7168, 9216, 11264):
    left = card - F
    print(f'card {card}: experts budget {left:.0f} MiB -> all-resident {left/per_bpw:.2f} bpw', end='')
    for b in (3.0, 3.4, 3.8):
        print(f' | {b}: need {per_bpw*b:.0f} cache frac {min(1,left/(per_bpw*b)):.2f} slots {int(3072*min(1,left/(per_bpw*b)))}', end='')
    print()
