#!/usr/bin/env python3
"""Compare two llama-hc-fused-check dumps (reference first). Tensors are matched by name and occurrence and the
relative L2 difference of each is printed; the first tensor over the threshold names the layer and stage.

This model amplifies any float change layer by layer (4-bit KV, IQ3 experts): a valid summation reordering alone
reaches ~1e-1 at the logits. So the pass/fail verdict needs a noise dump, a third run with a known valid-math change
(e.g. LLAMA_HC_FUSED=0 GGML_CUDA_SHEXP_GATE=0): the test passes when its final tensors (last l_last, result_norm,
result_output) move at most FACTOR times as far as the noise run's do. Without a noise dump every tensor is held to
the absolute threshold (a diagnostic, not a verdict).

The unfused hc_inject is the raw W_inject product, the fused one is already 2*sigmoid(v/hc); the raw one is
activated before the comparison.

usage: compare.py REF_DIR TEST_DIR [--noise NOISE_DIR] [--factor 2] [--thr 1e-2]"""
import argparse, math, struct, sys
from collections import defaultdict
from pathlib import Path

HC = 4


def load(d):
    out, seen = {}, defaultdict(int)
    for line in (Path(d) / 'index.txt').read_text().splitlines():
        seq, rest = line.split(' ', 1)
        name, op, *ne = rest.rsplit(' ', 5)  # names may hold spaces ("x (reshaped)")
        k = (name, seen[name]); seen[name] += 1
        raw = (Path(d) / f'{seq}.f32').read_bytes()
        out[k] = (op, [int(x) for x in ne], struct.unpack(f'{len(raw)//4}f', raw))
    return out


def rel_diffs(ref, test):
    res = {}
    for k, (op_r, _, a) in ref.items():
        if k not in test:
            continue
        op_t, _, b = test[k]
        if len(a) != len(b):
            continue
        if k[0].startswith('hc_inject') and op_r != op_t:
            act = lambda v: 2.0 / (1.0 + math.exp(-v / HC))
            if op_r == 'MUL_MAT':
                a = [act(v) for v in a]
            if op_t == 'MUL_MAT':
                b = [act(v) for v in b]
        num = sum((x - y) ** 2 for x, y in zip(a, b))
        den = sum(x * x for x in a) or 1.0
        res[k] = (op_r, op_t, math.sqrt(num / den))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('test')
    ap.add_argument('--noise'); ap.add_argument('--factor', type=float, default=2.0)
    ap.add_argument('--thr', type=float, default=1e-2)
    a = ap.parse_args()
    ref = load(a.ref)
    d = rel_diffs(ref, load(a.test))
    dn = rel_diffs(ref, load(a.noise)) if a.noise else {}
    first = None
    for k, (op_r, op_t, r) in d.items():
        over = not math.isfinite(r) or r > a.thr
        if over and first is None:
            first = k
        nz = f'  noise {dn[k][2]:.3e}' if k in dn else ''
        print(f'{k[0]:<24} #{k[1]}  {op_r:>14} / {op_t:<14} rel {r:.3e}{nz}{"  <-- over" if over else ""}')
    print(f'first over {a.thr:g}: {first}')
    if not a.noise:
        sys.exit(1 if first else 0)
    # verdict on the final tensors against the noise run
    lasts = [k for k in d if k[0].startswith('l_last-') and k[0].split('-')[1].isdigit()]
    final = ([max(lasts, key=lambda k: int(k[0].split('-')[1]))] if lasts else []) + \
        [k for k in d if k[0] in ('result_norm', 'result_output')]
    bad = False
    for k in final:
        r, nr = d[k][2], dn.get(k, (0, 0, float('nan')))[2]
        ok = math.isfinite(r) and math.isfinite(nr) and r <= a.factor * nr
        bad |= not ok
        print(f'verdict {k[0]:<16} test {r:.3e}  noise {nr:.3e}  ratio {r / nr if nr else float("inf"):.2f}  {"ok" if ok else "FAIL"}')
    print('PASS' if final and not bad else 'FAIL')
    sys.exit(0 if final and not bad else 1)


if __name__ == '__main__':
    main()
