#!/usr/bin/env python3
"""aiq_errtab.py (AIQ3, private): numpy twin of llama-ai4-errtab's table mode (no Hessian), same TSV.

  aiq_errtab.py --bf16 S.gguf [...] [--imatrix I.gguf] --cand TAG=F.gguf [...] [--re REGEX] --out T.tsv

Pure Python + numpy: decodes every AI+ type (AI4 70, AI4_U 71, AI3_U 72, AI2_U 73, AI5_U 74, AI6_U 75, and the private
AI6_UO 86 / AI5_UN 87) with a port of ggml-ai4.h's slab decoders, and every type
gguf-py can dequantize (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, ...). Used as an independent cross-check of the C++ tool in the
tests, and as a fallback when no build is available. Columns: tensor tag type bytes n_weights sse ref wsse wref hsse.
"""
import argparse, mmap, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

AI4_MUL = 0x83DCD12D
SIGN_WORDS = [0x9e3779b9, 0x7f4a7c15, 0xf39cc060, 0x5ced2a1b, 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a]
AI_K = {70: 4, 71: 4, 72: 3, 73: 2, 74: 5, 75: 6, 86: 6, 87: 5}
AI_C = {75: 0xA1EEE321, 87: 0x9C971CDD}   # per-type trellis constant (ggml_ai4_trellis_mul); everything else AI4_MUL
CH = 512   # rows per chunk


def _levels(c=AI4_MUL):
    x = (np.arange(65536, dtype=np.uint64) * np.uint64(c)) & 0xFFFFFFFF
    b = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    return (((b + 2) >> 2).astype(np.int32) - 128).astype(np.float32)


_LEV_CACHE = {}


def _lev(c=AI4_MUL):
    if c not in _LEV_CACHE:
        _LEV_CACHE[c] = _levels(c)
    return _LEV_CACHE[c]


LEV = _lev()
SIGN = np.array([-1.0 if (SIGN_WORDS[i >> 5] >> (i & 31)) & 1 else 1.0 for i in range(256)], dtype=np.float32)


def _state_idx(k):
    """per symbol n (0..127) of a k-bit tile: (word w, word w2, shift) of aik_get_state."""
    nb, nw = 128 * k, 4 * k
    b0 = np.array([(k * n + k - 16) % nb for n in range(128)])
    w = b0 >> 5
    return w, (w + 1) % nw, (48 - (b0 & 31)).astype(np.uint64)


def _tiles(words, k, mul=AI4_MUL):
    """words (..., 4k) uint32 -> levels (..., 128) float32 (tile symbol order n = r*8 + c)."""
    w, w2, sh = _state_idx(k)
    c = (words[..., w].astype(np.uint64) << np.uint64(32)) | words[..., w2].astype(np.uint64)
    return _lev(mul)[((c >> sh) & np.uint64(0xFFFF)).astype(np.int64)]


def _unrotate(x, rb):
    """x (..., K) in place-free: blocks of rb: FWHT, then * sign(i)/sqrt(rb) (ai4_unrotate_n)."""
    sh = x.shape
    y = x.reshape(-1, rb).astype(np.float32)
    h = 1
    while h < rb:
        y = y.reshape(-1, rb // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack((a + b, a - b), axis=2)
        h *= 2
    y = y.reshape(-1, rb)
    s = np.float32(1.0 / 16.0) if rb == 256 else np.float32(0.08838834764831845)
    return (y * (SIGN[:rb] * s)).reshape(sh)


def dequant_ai(raw, t, K, nrows):
    """raw uint8 bytes of nrows x K (nrows % 16 == 0) of AI type t -> float32 (nrows, K)."""
    k = AI_K[t]
    mul = AI_C.get(t, AI4_MUL)
    ns = nrows // 16
    if t == 70:   # AI4 1024 layout: superblock = 256 B scales (fp16 [16][8]) + 128 tiles x 64 B
        sb = raw.reshape(ns, K // 1024, 8448)
        d = sb[:, :, :256].copy().view(np.float16).astype(np.float32).reshape(ns, K // 1024, 16, 8)
        lv = _tiles(sb[:, :, 256:].copy().view("<u4").reshape(ns, K // 1024, 128, 16), 4)        # (ns, S, ct, n)
        lv = lv.reshape(ns, K // 1024, 128, 16, 8).transpose(0, 3, 1, 2, 4).reshape(ns, 16, K // 1024, 1024)
        d = np.repeat(d.transpose(0, 2, 1, 3), 128, axis=3)                                     # (ns, 16, S, 1024)
        return _unrotate((lv * d).reshape(nrows, K), 256)
    ub = 32 + 256 * k
    nu = K // 128
    un = raw.reshape(ns, nu, ub)
    d = un[:, :, :32].copy().view(np.float16).astype(np.float32)                                 # (ns, nu, 16)
    lv = _tiles(un[:, :, 32:].copy().view("<u4").reshape(ns, nu, 16, 4 * k), k, mul)                 # (ns, nu, ct, n)
    lv = lv.reshape(ns, nu, 16, 16, 8).transpose(0, 3, 1, 2, 4).reshape(ns, 16, nu, 128)
    x = lv * d.transpose(0, 2, 1)[:, :, :, None]
    return _unrotate(x.reshape(nrows, K), 256 if K % 256 == 0 else 128)


def tensor_dims(t):
    shape = [int(x) for x in t.shape]
    return shape[0], (shape[1] if len(shape) > 1 else 1), (shape[2] if len(shape) > 2 else 1)


def tensor_f32(t, r0=0, r1=None):
    """ReaderTensor rows [r0, r1) of the (E*M, K) row view -> float32 (r1 - r0, K); r0, r1 multiples of 16 for AI types."""
    K, M, E = tensor_dims(t)
    r1 = M * E if r1 is None else r1
    tt = int(t.tensor_type)
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)
    rb = raw.size // (M * E)                      # bytes per row (AI types: per row on average; slabs are 16 rows)
    raw = raw[r0 * rb:r1 * rb]
    if tt in AI_K:
        return dequant_ai(raw, tt, K, r1 - r0)
    if tt == Q.F32:
        return raw.view(np.float32).reshape(r1 - r0, K).copy()
    return np.asarray(dequantize(raw.reshape(r1 - r0, rb), Q(tt)), dtype=np.float32).reshape(r1 - r0, K)


def type_name(t):
    """ggml_type_name spelling (q5_K, ai5u, ...)."""
    names = {70: "ai4", 71: "ai4u", 72: "ai3u", 73: "ai2u", 74: "ai5u", 75: "ai6u", 86: "ai6uo", 87: "ai5un"}
    if t in names:
        return names[t]
    n = Q(t).name.lower()
    return n[:-2] + "_K" if n.endswith("_k") else n


def load_imatrix(path):
    r = GGUFReader(path, "r")
    sums, cnts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-8]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
        elif t.name.endswith(".counts"):
            cnts[t.name[:-7]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
    return {n: (sums[n], cnts[n]) for n in sums if n in cnts}


def get_lam(im, name, K, e, E):
    """same rule as get_lam() in ai4-errtab.cpp / llama-ai4-quantize (GGUF imatrix)."""
    if name not in im:
        return None
    s, c = im[name]
    if s.size != K * E:
        return None
    cnt = float(c[e if c.size == E else 0]) if c.size else 0.0
    if not cnt > 0 and E > 1:
        return None
    sm = s[e * K:(e + 1) * K].astype(np.float64)
    return sm / cnt if cnt > 0 else np.ones(K)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", action="append", required=True)
    ap.add_argument("--imatrix")
    ap.add_argument("--cand", action="append", required=True, help="TAG=FILE")
    ap.add_argument("--re")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="keep finished tensors from OUT.tmp of an interrupted run")
    a = ap.parse_args(argv)
    src, readers = {}, []
    for p in a.bf16:
        readers.append(GGUFReader(p, "r"))
        for t in readers[-1].tensors:
            src[t.name] = t
    cands = []
    for c in a.cand:
        tag, _, p = c.partition("=")
        readers.append(GGUFReader(p, "r"))
        cands.append((tag, {t.name: t for t in readers[-1].tensors}))
    im = load_imatrix(a.imatrix) if a.imatrix else {}
    rx = re.compile(a.re) if a.re else None
    n_rows, done = 0, set()
    if a.resume and os.path.exists(a.out + ".tmp"):   # keep the rows of a killed run (each tensor's rows end with a flush)
        keep = [l for l in open(a.out + ".tmp") if not l.startswith(("#", "tensor"))]
        done = {l.split("\t")[0] for l in keep}
    with open(a.out + ".tmp", "w") as fo:
        fo.write("# aiq_errtab.py: errors vs BF16 source; wsse/wref imatrix-weighted (-1 = no imatrix), hsse -1 (no Hessian)\n")
        fo.write("tensor\ttag\ttype\tbytes\tn_weights\tsse\tref\twsse\twref\thsse\n")
        if done:
            fo.writelines(keep)
            n_rows = len(keep)
            print(f"resume: {len(done)} tensors already done", file=sys.stderr)
        for name in sorted(src):
            if rx and not rx.search(name):
                continue
            if name in done:
                continue
            hits = [(tag, d[name]) for tag, d in cands if name in d]
            if not hits or len(src[name].shape) < 2:
                continue
            K, M, E = tensor_dims(src[name])
            lams = [get_lam(im, name, K, e, E) for e in range(E)] if im else [None] * E
            any_lam = any(l is not None for l in lams)
            rows = []
            for tag, ct in hits:
                if tensor_dims(ct) != (K, M, E):
                    print(f"skip {name} in {tag}: shape differs", file=sys.stderr)
                    continue
                sse = ref = wsse = wref = 0.0
                for r0 in range(0, M * E, CH):   # row chunks (CH % 16 == 0, M % 16 == 0): RSS stays ~CH*K*40 bytes
                    r1 = min(r0 + CH, M * E)
                    W = tensor_f32(src[name], r0, r1).astype(np.float64)
                    D = W - tensor_f32(ct, r0, r1)
                    D2, W2 = D * D, W * W
                    sse += float(D2.sum()); ref += float(W2.sum())
                    if any_lam:
                        for e in range(r0 // M, (r1 - 1) // M + 1):   # rows of expert e inside this chunk
                            a0, a1 = max(r0, e * M) - r0, min(r1, (e + 1) * M) - r0
                            if lams[e] is not None:
                                wsse += float(D2[a0:a1].sum(axis=0) @ lams[e]); wref += float(W2[a0:a1].sum(axis=0) @ lams[e])
                if not any_lam:
                    wsse = wref = -1.0
                rows.append(f"{name}\t{tag}\t{type_name(int(ct.tensor_type))}"
                            f"\t{int(ct.n_bytes)}\t{K * M * E}\t{sse:.9e}\t{ref:.9e}\t{wsse:.9e}\t{wref:.9e}\t{-1.0:.9e}\n")
                print(f"{name:32s} {tag:8s} rel {sse / ref:.4e} wrel {wsse / wref if any_lam else -1:.4e}", file=sys.stderr, flush=True)
            fo.writelines(rows)   # a tensor's rows land together (resume unit)
            fo.flush()
            n_rows += len(rows)
            for r in readers:     # drop the mapped pages we touched: keeps RSS near one tensor's working set
                try:
                    r.data._mmap.madvise(mmap.MADV_DONTNEED)
                except (AttributeError, ValueError, OSError):
                    pass
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {n_rows} rows", file=sys.stderr)
    return 0 if n_rows else 1


if __name__ == "__main__":
    sys.exit(main())
