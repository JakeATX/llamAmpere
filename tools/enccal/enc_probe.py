#!/usr/bin/env python3
"""ENCPROBE: one-pass sensitivity probe for the per-tensor bit allocator.

Instead of encoding every tensor in every candidate format and scoring each, perturb ONE tensor of the reference
model with noise shaped like quantization error, run one scoring pass, and read the KLD increase as that tensor's
damage score. A handful of tensors that do have real encoded measurements calibrate the probe (per role and per
format: measured delta / probe delta, median), which then predicts every other tensor x format. The allocator picks
formats under a byte budget from those predictions (measured values win when present). FFN matrices never go under
--floor-bits (default 2: no 1-bit FFN).

Subcommands
  noise  SRC.gguf TENSOR OUT.gguf [--rel-mse 0.019] [--block 256] [--col-tilt 0 --imatrix I.gguf] [--seed N]
         Writes a one-tensor GGUF: dequantized reference weight (x its per-expert .scale, folded) + noise, stored Q8_0
         (Q8_0's own error is ~1e-4 relative, negligible next to a 2-4 bit code). Assemble it into the reference with
         any tensor-swap tool and score the arm the usual way.
         Noise: per (row, --block columns) the block RMS r, delta = sqrt(rel_mse) * r * z, z ~ N(0,1), deterministic
         from --seed and the tensor name. --col-tilt a > 0 scales column j by (h_j / mean h)^(-a/2) (imatrix h,
         renormalised to the same mean square), i.e. less error on high-activation columns as a weighted encoder does.
  patch  WORK.gguf TENSOR [--rel-mse ...] [--restore ORIG.gguf]
         Same noise, written IN PLACE into a working copy of the reference, in the tensor's own type (NVFP4, Q8_0,
         BF16, F16, F32), so one copy on disk serves every arm (patch, score, --restore from the original, next).
         Re-rounding to the tensor's own grid adds that grid's error on top (uniform over tensors of one type; the
         per-role calibration ratio absorbs it). --restore copies the tensor's original bytes back from ORIG.
  relmse REF.gguf CAND.gguf TENSOR [--experts 3]
         Weight relMSE of CAND vs REF for one tensor (both must be types gguf-py can dequantize) to set --rel-mse.
  rank   --scores DIR [DIR ...] --b0 ARM --probe P{t} [--anchor FORMAT=TEMPLATE ...] [--out probe_rank.csv]
         TEMPLATE names the arm of tensor id {t} (gu<L>, dn<L>, out, emb), e.g. AIT3={t} AIT2=a2{t} Q5_K={t}5.
         Reads <DIR>/S<arm>.ub5.coding.csv style KLD CSVs (column 'kld', per-position or per-bin rows, mean taken),
         prints probe deltas, rank correlation vs each anchor format's measured deltas, the per-role ratio and writes
         one row per tensor with measured + predicted delta for every anchor format.
  alloc  --rank probe_rank.csv --sizes SIZES.json --base-bytes N --budget-gb G [G ...] [--floor-bits 2]
         Multiple-choice knapsack (1 MB grid) minimising the summed predicted delta under each byte budget.
"""
import argparse, csv, hashlib, json, math, os, re, sys

import numpy as np


def _gguf():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.environ.get('ENCCAL_GGUF_PY'), os.path.join(here, '..', '..', 'gguf-py')):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import gguf  # noqa: E402
    return gguf


# ---------------------------------------------------------------------------------------------------------- noise
def _tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    raise SystemExit(f'tensor {name} not found')


def _deq_slices(gguf, t):
    """Yield (index, float32 [rows, cols]) per leading slice (expert) of a 2D/3D tensor, dequantized."""
    data = t.data
    if data.ndim == 2:   # 2D tensor: data is [rows, cols or bytes]
        yield 0, np.asarray(gguf.quants.dequantize(np.asarray(data), t.tensor_type), np.float32).reshape(data.shape[0], -1)
        return
    for e in range(data.shape[0]):
        yield e, np.asarray(gguf.quants.dequantize(np.asarray(data[e]), t.tensor_type), np.float32)


def _col_tilt(imatrix_path, name, n_slices, cols, tilt):
    if not tilt:
        return None
    gguf = _gguf()
    r = gguf.GGUFReader(imatrix_path)
    base = name[:-len('.weight')] if name.endswith('.weight') else name
    want = {f'{name}.in_sum2', f'{base}.in_sum2', f'{base}.weight.in_sum2'}
    s2 = cnt = None
    for t in r.tensors:
        if t.name in want:
            s2 = np.asarray(t.data, np.float64).reshape(-1, cols)
        elif t.name.endswith('.counts') and t.name[:-len('.counts')] + '.in_sum2' in want:
            cnt = np.asarray(t.data, np.float64).reshape(-1)
    if s2 is None:
        raise SystemExit(f'{name}: no in_sum2 in {imatrix_path}')
    if cnt is not None and cnt.size == s2.shape[0]:
        s2 = s2 / np.maximum(cnt, 1.0)[:, None]
    if s2.shape[0] == 1 and n_slices > 1:
        s2 = np.repeat(s2, n_slices, 0)
    h = np.maximum(s2, 1e-12 * np.maximum(s2.mean(1, keepdims=True), 1e-30))
    c = (h / h.mean(1, keepdims=True)) ** (-0.5 * tilt)
    c /= np.sqrt((c * c).mean(1, keepdims=True))
    return c.astype(np.float32)


def add_noise(w, rel_mse, block, rng, colw=None):
    rows, cols = w.shape
    nb = (cols + block - 1) // block
    pad = nb * block - cols
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    cnt = np.full(nb, float(block)); cnt[-1] = cols - (nb - 1) * block
    rms = np.sqrt((wp.reshape(rows, nb, block).astype(np.float64) ** 2).sum(2) / cnt[None, :]).astype(np.float32)
    z = rng.standard_normal((rows, nb * block), dtype=np.float32)
    z = (z.reshape(rows, nb, block) * (math.sqrt(rel_mse) * rms)[:, :, None]).reshape(rows, nb * block)[:, :cols]
    if colw is not None:
        z *= colw[None, :]
    return w + z


def cmd_noise(a):
    gguf = _gguf()
    r = gguf.GGUFReader(a.src)
    t = _tensor(r, a.tensor)
    sc = None
    sn = a.tensor[:-len('.weight')] + '.scale' if a.tensor.endswith('.weight') else None
    if sn and not a.no_fold:
        for u in r.tensors:
            if u.name == sn:
                sc = np.asarray(u.data, np.float32).reshape(-1)
    n_sl = t.data.shape[0] if t.data.ndim == 3 else 1
    cols = int(t.shape[0])
    colw_all = _col_tilt(a.imatrix, a.tensor, n_sl, cols, a.col_tilt) if a.col_tilt else None
    seed = int.from_bytes(hashlib.sha256(f'{a.seed}:{a.tensor}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    out = []; num = den = 0.0
    for e, w in _deq_slices(gguf, t):
        if sc is not None:
            w = w * (sc[e] if sc.size == n_sl else sc[0])
        wn = add_noise(w, a.rel_mse, a.block, rng, None if colw_all is None else colw_all[e])
        num += float(((wn - w).astype(np.float64) ** 2).sum()); den += float((w.astype(np.float64) ** 2).sum())
        out.append(gguf.quants.quantize(wn, gguf.GGMLQuantizationType.Q8_0))
    q = np.stack(out) if t.data.ndim == 3 else out[0]
    w = gguf.GGUFWriter(a.out, 'encprobe')
    w.add_string('encprobe.tensor', a.tensor)
    w.add_string('encprobe.recipe', json.dumps(dict(rel_mse=a.rel_mse, block=a.block, col_tilt=a.col_tilt, seed=a.seed,
                                                      folded_scale=sc is not None, src=os.path.basename(a.src))))
    w.add_tensor(a.tensor, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f'NOISE {a.tensor} slices={n_sl} realised_relmse={num / max(den, 1e-30):.5f} target={a.rel_mse} '
          f'folded_scale={sc is not None} -> {a.out}')


_E2M1x2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def nvfp4_quantize(w):
    """float32 [rows, cols] (cols % 64 == 0) -> NVFP4 block bytes [rows, cols/64*36] (4 x ue4m3 scale + 32 nibble bytes)."""
    gguf = _gguf(); Q = gguf.quants.NVFP4
    rows, cols = w.shape
    x = w.reshape(-1, 4, 16)
    amax = np.abs(x).max(-1)
    db = Q.fp32_to_ue4m3(2.0 * amax / 12.0)          # stored byte decodes to (value * 0.5)
    d = Q.ue4m3_to_fp32(db)
    # bump a scale that rounded down so amax still fits the +-12 grid
    low = (d * 12.0 < amax * (1 - 1e-6)) & (db < 0x7E)
    db = np.where(low, db + 1, db).astype(np.uint8); d = Q.ue4m3_to_fp32(db)
    r = np.where(d[..., None] > 0, x / np.where(d > 0, d, 1)[..., None], 0.0)
    pos = _E2M1x2[:8]
    idx = np.abs(np.abs(r)[..., None] - pos).argmin(-1).astype(np.uint8)
    idx = np.where((r < 0) & (idx > 0), idx + 8, idx).astype(np.uint8)
    qs = (idx[..., :8] | (idx[..., 8:] << 4)).astype(np.uint8)          # byte j: element j low, element j+8 high
    blk = np.concatenate([db.reshape(-1, 4), qs.reshape(-1, 32)], axis=1)
    return blk.reshape(rows, cols // 64 * 36)


def _requant(w, ttype):
    gguf = _gguf(); T = gguf.GGMLQuantizationType
    if ttype == T.NVFP4:
        return nvfp4_quantize(w)
    if ttype in (T.Q8_0, T.BF16, T.F16):
        return gguf.quants.quantize(w, ttype) if ttype != T.F16 else w.astype(np.float16)
    if ttype == T.F32:
        return w.astype(np.float32)
    raise SystemExit(f'patch: no requantizer for {ttype.name}')


def cmd_patch(a):
    gguf = _gguf()
    r = gguf.GGUFReader(a.work)                      # read-only view for offsets + current bytes
    t = _tensor(r, a.tensor)
    nbytes = int(t.n_bytes)
    if a.restore:
        o = _tensor(gguf.GGUFReader(a.restore), a.tensor)
        assert int(o.n_bytes) == nbytes and o.tensor_type == t.tensor_type, 'restore: layout differs'
        src = np.asarray(o.data).view(np.uint8).reshape(-1)
        with open(a.work, 'r+b') as f:
            f.seek(int(t.data_offset)); f.write(src.tobytes())
        print(f'RESTORED {a.tensor} {nbytes} bytes'); return
    seed = int.from_bytes(hashlib.sha256(f'{a.seed}:{a.tensor}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    n_sl = t.data.shape[0] if t.data.ndim == 3 else 1
    num = den = 0.0
    with open(a.work, 'r+b') as f:
        off = int(t.data_offset)
        for e, w in _deq_slices(gguf, t):
            wn = add_noise(w, a.rel_mse, a.block, rng)
            q = np.ascontiguousarray(_requant(wn, t.tensor_type)).view(np.uint8).reshape(-1)
            back = np.asarray(gguf.quants.dequantize(q.reshape(w.shape[0], -1), t.tensor_type), np.float32).reshape(w.shape) \
                if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16) else wn
            num += float(((back - w).astype(np.float64) ** 2).sum()); den += float((w.astype(np.float64) ** 2).sum())
            f.seek(off); f.write(q.tobytes()); off += q.size
        assert off - int(t.data_offset) == nbytes, (off - int(t.data_offset), nbytes)
    print(f'PATCH {a.tensor} {t.tensor_type.name} slices={n_sl} stored_relmse={num / max(den, 1e-30):.5f} (noise {a.rel_mse})')


def cmd_relmse(a):
    gguf = _gguf()
    R, C = gguf.GGUFReader(a.ref), gguf.GGUFReader(a.cand)
    tr, tc = _tensor(R, a.tensor), _tensor(C, a.tensor)
    def scale(rd):
        for u in rd.tensors:
            if u.name == a.tensor[:-len('.weight')] + '.scale':
                return np.asarray(u.data, np.float32).reshape(-1)
    sr, scn = scale(R), scale(C)
    n = tr.data.shape[0] if tr.data.ndim == 3 else 1
    idx = np.linspace(0, n - 1, min(a.experts, n)).round().astype(int)
    num = den = 0.0
    for e in idx:
        wr = np.asarray(gguf.quants.dequantize(np.asarray(tr.data[e] if n > 1 else tr.data), tr.tensor_type), np.float64)
        wc = np.asarray(gguf.quants.dequantize(np.asarray(tc.data[e] if n > 1 else tc.data), tc.tensor_type), np.float64)
        if sr is not None: wr = wr * sr[e % sr.size]
        if scn is not None: wc = wc * scn[e % scn.size]
        num += ((wc - wr) ** 2).sum(); den += (wr ** 2).sum()
    print(f'RELMSE {a.tensor} experts={list(idx)} relmse={num / den:.5f}')


# ----------------------------------------------------------------------------------------------------------- rank
ROLE_RX = [('gu', re.compile(r'^gu(\d+)$')), ('dn', re.compile(r'^dn(\d+)$')), ('out', re.compile(r'^out$')),
           ('emb', re.compile(r'^emb$'))]


def kld_mean(path):
    if not os.path.exists(path):
        return None
    v = [float(r['kld']) for r in csv.DictReader(open(path))]
    return sum(v) / len(v) if v else None


def spearman(x, y):
    if len(x) < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def role_of(tid):
    for role, rx in ROLE_RX:
        if rx.match(tid):
            return role
    return tid


def tensor_ids(n_layer):
    return [f'{p}{L}' for p in ('gu', 'dn') for L in range(n_layer)] + ['out', 'emb']


def cmd_rank(a):
    # each directory is one scoring platform with its own base score; a delta is arm - base of the SAME directory
    bases = {d: kld_mean(os.path.join(d, a.pattern.format(arm=a.b0))) for d in a.scores}
    def f(arm):
        for d in a.scores:
            v = kld_mean(os.path.join(d, a.pattern.format(arm=arm)))
            if v is not None:
                if bases[d] is None:
                    raise SystemExit(f'{d}: has {arm} but no base arm {a.b0}')
                return v - bases[d]
        return None
    b0 = next((v for v in bases.values() if v is not None), None)
    if b0 is None:
        raise SystemExit(f'no score for base arm {a.b0}')
    anchors = [(x.split('=', 1)[0], x.split('=', 1)[1]) for x in a.anchor]   # FORMAT=TEMPLATE, arm = TEMPLATE % tid
    rows = []
    for tid in tensor_ids(a.n_layer):
        p = f(a.probe.format(t=tid))
        row = dict(tensor=tid, role=role_of(tid), probe=p)
        for fmt, pre in anchors:
            m = f(pre.format(t=tid))
            row[f'meas_{fmt}'] = m
        rows.append(row)
    print('base ' + ' '.join(f'{d}:{v:.6f}' for d, v in bases.items() if v is not None) + f'; probe arms scored: {sum(r["probe"] is not None for r in rows)}/{len(rows)}')
    # calibration: per (role, format) median of measured / probe over tensors with both (role pooled when < min_pairs)
    ratio = {}
    for fmt, _ in anchors:
        pool = [(r['role'], r[f'meas_{fmt}'] / r['probe']) for r in rows
                if r['probe'] and r['probe'] > 0 and r[f'meas_{fmt}'] is not None]
        allr = [q for _, q in pool]
        for role in {r['role'] for r in rows}:
            rr = [q for ro, q in pool if ro == role]
            src = rr if len(rr) >= a.min_pairs else (allr if role in ('gu', 'dn') and len(allr) >= a.min_pairs else [])
            ratio[(role, fmt)] = float(np.median(src)) if src else None
        both = [(r['probe'], r[f'meas_{fmt}']) for r in rows if r['probe'] is not None and r[f'meas_{fmt}'] is not None]
        if both:
            x, y = np.array(both).T
            lc = float(np.corrcoef(np.log(np.maximum(x, 1e-9)), np.log(np.maximum(y, 1e-9)))[0, 1]) if len(x) > 2 else float('nan')
            print(f'anchor {fmt}: n={len(x)} spearman={spearman(x, y):.3f} log-pearson={lc:.3f} '
                  f'ratio(meas/probe) by role: ' + ' '.join(f'{ro}={ratio[(ro, fmt)]:.3g}' for ro in ('gu', 'dn', 'out', 'emb')
                                                            if ratio.get((ro, fmt)) is not None))
    order = sorted([r for r in rows if r['probe'] is not None], key=lambda r: -r['probe'])
    for i, r in enumerate(order):
        r['rank'] = i + 1
    for r in rows:
        for fmt, _ in anchors:
            q = ratio.get((r['role'], fmt))
            r[f'pred_{fmt}'] = None if (q is None or r['probe'] is None) else max(0.0, r['probe']) * q
    keys = ['tensor', 'role', 'probe', 'rank'] + [k for fmt, _ in anchors for k in (f'meas_{fmt}', f'pred_{fmt}')]
    with open(a.out, 'w', newline='') as fo:
        w = csv.DictWriter(fo, keys); w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else (f'{r[k]:.7g}' if isinstance(r[k], float) else r[k])) for k in keys})
    print('rank (most damaging first): ' + ' '.join(f'{r["tensor"]}:{r["probe"]:+.6f}' for r in order))
    print(f'wrote {a.out}')


# ---------------------------------------------------------------------------------------------------------- alloc
def cmd_alloc(a):
    S = json.load(open(a.sizes))   # {"elements": {role: n}, "base": {tid-or-role: fmt}, "mb": {role: {fmt: MB}}}
    rows = list(csv.DictReader(open(a.rank)))
    fmts = sorted({k[5:] for k in rows[0] if k.startswith('pred_')})
    groups = []
    for r in rows:
        tid, role = r['tensor'], r['role']
        mb = S['mb'].get(role)
        if not mb:
            continue
        base = S['base'].get(tid, S['base'].get(role))
        opts = [(base, 0.0, 0.0)]
        for fm in fmts:
            if fm == base or fm not in mb:
                continue
            bpw = mb[fm] * 8e6 / S['elements'][role]
            if role in ('gu', 'dn') and bpw < a.floor_bits:
                continue
            val = r.get(f'meas_{fm}') if a.prefer_measured and r.get(f'meas_{fm}') else r.get(f'pred_{fm}')
            if not val:
                continue
            opts.append((fm, mb[fm] - mb[base], float(val)))
        groups.append((tid, opts))
    base_mb = a.base_bytes / 1e6
    for bud in a.budget_gb:
        cap = int(math.ceil(base_mb - bud * 1e3))
        if cap <= 0:
            print(f'budget {bud} GB: base already fits'); continue
        INF = float('inf'); dp = [0.0] + [INF] * cap; back = []
        for _, g in groups:
            nd = [INF] * (cap + 1); nb = [None] * (cap + 1)
            for s in range(cap + 1):
                if dp[s] == INF:
                    continue
                for i, (_, dmb, k) in enumerate(g):
                    t = min(cap, s + int(round(-dmb)))
                    if dp[s] + k < nd[t]:
                        nd[t] = dp[s] + k; nb[t] = (s, i)
            dp = nd; back.append(nb)
        if dp[cap] == INF:
            print(f'budget {bud} GB: infeasible with these options'); continue
        s = cap; sel = []
        for gi in range(len(groups) - 1, -1, -1):
            ps, i = back[gi][s]; sel.append((groups[gi][0], groups[gi][1][i])); s = ps
        sel.reverse()
        saved = -sum(o[1] for _, o in sel)
        print(f'budget {bud} GB: predicted summed delta {dp[cap]:+.6f}, est size {(base_mb - saved) / 1e3:.3f} GB')
        print('  ' + ' '.join(f'{t}={o[0]}' for t, o in sel if o[1] != 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('noise'); p.add_argument('src'); p.add_argument('tensor'); p.add_argument('out')
    p.add_argument('--rel-mse', type=float, default=0.019); p.add_argument('--block', type=int, default=256)
    p.add_argument('--col-tilt', type=float, default=0.0); p.add_argument('--imatrix')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--no-fold', action='store_true')
    p = sp.add_parser('patch'); p.add_argument('work'); p.add_argument('tensor')
    p.add_argument('--rel-mse', type=float, default=0.019); p.add_argument('--block', type=int, default=256)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--restore')
    p = sp.add_parser('relmse'); p.add_argument('ref'); p.add_argument('cand'); p.add_argument('tensor')
    p.add_argument('--experts', type=int, default=3)
    p = sp.add_parser('rank'); p.add_argument('--scores', nargs='+', required=True); p.add_argument('--b0', default='B0')
    p.add_argument('--probe', default='P{t}'); p.add_argument('--anchor', action='append', default=[])
    p.add_argument('--pattern', default='S{arm}.ub5.coding.csv'); p.add_argument('--n-layer', type=int, default=24)
    p.add_argument('--min-pairs', type=int, default=3); p.add_argument('--out', default='probe_rank.csv')
    p = sp.add_parser('alloc'); p.add_argument('--rank', required=True); p.add_argument('--sizes', required=True)
    p.add_argument('--base-bytes', type=int, required=True); p.add_argument('--budget-gb', type=float, nargs='+', required=True)
    p.add_argument('--floor-bits', type=float, default=2.0); p.add_argument('--prefer-measured', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'noise' and a.col_tilt and not a.imatrix:
        ap.error('--col-tilt needs --imatrix')
    dict(noise=cmd_noise, patch=cmd_patch, relmse=cmd_relmse, rank=cmd_rank, alloc=cmd_alloc)[a.cmd](a)


if __name__ == '__main__':
    main()
