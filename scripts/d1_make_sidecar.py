#!/usr/bin/env python3
"""[D1] Make the exact-row sidecar for LLAMA_D1_HEAD (design LSA/REAL8GB/DESIGN_d1_head_rescore.md r3, section 1).

Decodes the base model's output.weight (an AI+ type such as AIT3, via ggml_ai4_dequantize_slab from the build's
libggml-base) and writes one tensor `output_exact.weight` [n_embd, n_vocab] as F16 or Q8_0 plus fingerprint KVs:
  d1.rows_type     "f16" | "q8_0"
  d1.source_bytes  size of the base output.weight in bytes
  d1.source_fnv64  FNV-1a 64 over the first and last 1 MiB of those bytes (the loader checks it)
  d1.self_bytes    size of output_exact.weight in bytes
  d1.self_fnv64    FNV-1a 64 over the first and last 1 MiB of output_exact.weight

usage: d1_make_sidecar.py --model base.gguf --lib <bundle>/libggml-base.so --out head-exact-f16.gguf
                          [--rows f16|q8_0] [--threads 8] [--verify 64]
Needs numpy. Peak RAM about the output size plus threads x 16 MiB.

[EXACTHEAD] 8 GB head (the base file's own AIF25_U2 head is the coarse card head):
  d1_make_sidecar.py --model AI2M10-healed.gguf --lib <bundle>/libggml-base.so --out headx.gguf \
      --exact-from <bf16 checkpoint>/model-000NN-of-000NN.safetensors [--exact-tensor lm_head.weight] --stats
--exact-from stores the source BF16 rows unchanged (rows type bf16, d1.rows_type "bf16"). --stats adds
output_exact.gl2, F32 [n_embd/32, n_vocab]: the L2 norm of (decoded base row - exact row) over each 32-column group,
fingerprinted as d1.gl2_fnv64. With it the 6-sigma certificate is on by default (statistically exact: a row that
belongs in the sampler's top 20 stays coarse with probability ~1e-9 per row); LLAMA_D1_CERT=0 turns it off.
"""
import argparse, ctypes, json, os, random, struct, sys, threading, time

import numpy as np

END = 1 << 20


def fnv64_ends(buf):
    """buf: bytes-like of the whole tensor (or of [first k][last k] when pre-sliced with whole=False)."""
    n = len(buf); k = min(END, n)
    h = 0xcbf29ce484222325
    for part in (bytes(buf[:k]), bytes(buf[n - k:])):
        for b in part:
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ---- minimal GGUF reader ----
def _rs(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, f.read(n))


def _str(f):
    (n,) = _rs(f, "Q")
    return f.read(n).decode("utf-8", "replace")


_SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _val(f, t):
    if t in _SC:
        return _rs(f, _SC[t])[0]
    if t == 8:
        return _str(f)
    if t == 9:
        (at,) = _rs(f, "I"); (n,) = _rs(f, "Q")
        return [_val(f, at) for _ in range(n)]
    raise ValueError(f"gguf kv type {t}")


def gguf_tensors(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        _rs(f, "I"); (nt,) = _rs(f, "Q"); (nkv,) = _rs(f, "Q")
        kv = {}
        for _ in range(nkv):
            k = _str(f); (t,) = _rs(f, "I"); kv[k] = _val(f, t)
        info = {}
        for _ in range(nt):
            name = _str(f); (nd,) = _rs(f, "I"); shape = list(_rs(f, "Q" * nd)); (tp,) = _rs(f, "I"); (off,) = _rs(f, "Q")
            info[name] = (tp, shape, off)
        align = kv.get("general.alignment", 32)
        base = (f.tell() + align - 1) // align * align
    return {k: (t, s, base + o) for k, (t, s, o) in info.items()}


# ---- minimal GGUF writer (one tensor) ----
def _wstr(f, s):
    b = s.encode(); f.write(struct.pack("<Q", len(b))); f.write(b)


def write_sidecar(path, tensors, kv):
    """tensors: list of (name, ne0, ne1, ggml_type, data)"""
    align = 32
    offs = []; o = 0
    for (_, _, _, _, d) in tensors:
        offs.append(o); o += (d.nbytes + align - 1) // align * align
    with open(path + ".part", "wb") as f:
        f.write(b"GGUF"); f.write(struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for k, (t, v) in kv.items():
            _wstr(f, k); f.write(struct.pack("<I", t))
            if t == 8: _wstr(f, v)
            elif t == 10: f.write(struct.pack("<Q", v))
            else: raise ValueError(t)
        for (name, ne0, ne1, gt, _), off in zip(tensors, offs):
            _wstr(f, name); f.write(struct.pack("<I", 2)); f.write(struct.pack("<QQ", ne0, ne1))
            f.write(struct.pack("<I", gt)); f.write(struct.pack("<Q", off))
        f.write(b"\0" * ((-f.tell()) % align))
        for (_, _, _, _, d) in tensors:
            mv = memoryview(np.ascontiguousarray(d)).cast("B")
            for a in range(0, len(mv), 1 << 28):
                f.write(mv[a:a + (1 << 28)])
            f.write(b"\0" * ((-len(mv)) % align))
    os.replace(path + ".part", path)


def safetensors_bf16(path, name):
    """memmap of a BF16 [V, K] safetensors tensor as uint16"""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    e = hdr[name]
    assert e["dtype"] == "BF16" and len(e["shape"]) == 2, (name, e["dtype"], e["shape"])
    b0, b1 = e["data_offsets"]
    V, K = e["shape"]
    assert b1 - b0 == V * K * 2
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + b0, shape=(V, K))


def group_l2(screen, exact):
    """[rows, K] f32 x2 -> [rows, K/32] f32: L2 norm of the difference per 32-column group (f64 accumulate)"""
    d = (screen.astype(np.float64) - exact.astype(np.float64))
    r, K = d.shape
    return np.sqrt((d.reshape(r, K // 32, 32) ** 2).sum(axis=2)).astype(np.float32)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def q8_0(x):
    """x: float32 [rows, K] -> uint8 [rows, K/32*34], same as ggml quantize_row_q8_0_ref."""
    r, K = x.shape
    b = x.reshape(r, K // 32, 32)
    amax = np.abs(b).max(axis=2)
    d = amax / 127.0
    idv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0).astype(np.float32)
    v = b * idv[:, :, None]
    q = np.trunc(v + np.copysign(0.5, v)).astype(np.int8)          # roundf: half away from zero
    out = np.empty((r, K // 32, 34), np.uint8)
    out[:, :, :2] = d.astype(np.float16).view(np.uint8).reshape(r, K // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(r, K // 32 * 34)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--lib", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="f16", choices=["f16", "q8_0"]); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verify", type=int, default=64)
    ap.add_argument("--exact-from", default=None, help="BF16 safetensors file with the source head rows")
    ap.add_argument("--exact-tensor", default="lm_head.weight")
    ap.add_argument("--stats", action="store_true", help="add output_exact.gl2 (needs --exact-from)")
    a = ap.parse_args()
    assert not a.stats or a.exact_from, "--stats needs --exact-from"
    if a.exact_from:
        a.rows = "bf16"
    t0 = time.time()
    tp, shape, off = gguf_tensors(a.model)["output.weight"]
    K, V = shape
    lib = ctypes.CDLL(a.lib)
    lib.ggml_row_size.restype = ctypes.c_size_t; lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_ai4_dequantize_slab.restype = ctypes.c_bool
    lib.ggml_ai4_dequantize_slab.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    src_bytes = lib.ggml_row_size(tp, K) * V
    src = np.memmap(a.model, dtype=np.uint8, mode="r", offset=off, shape=(src_bytes,))
    src_fnv = fnv64_ends(src)
    print(f"source output.weight type {tp} [{K}, {V}] {src_bytes} B fnv {src_fnv:016x}", flush=True)
    assert V % 16 == 0 and K % 32 == 0

    ptr = src.ctypes.data   # AIT tile types address the whole tensor: pass its base
    ex = safetensors_bf16(a.exact_from, a.exact_tensor) if a.exact_from else None
    if ex is not None:
        assert ex.shape == (V, K), f"exact rows {ex.shape} != head [{V}, {K}]"
    rb = K * 2 if a.rows in ("f16", "bf16") else K // 32 * 34
    out = np.empty((V, rb), np.uint8)
    gl2 = np.empty((V, K // 32), np.float32) if a.stats else None
    y = np.empty((16, K), np.float32)
    assert lib.ggml_ai4_dequantize_slab(tp, ptr, K, 0, y.ctypes.data), "decode failed (not an AI+ type?)"
    ng = V // 16; nxt = [0]; lk = threading.Lock(); bad = []
    CH = 64

    def conv(w):
        return w.astype(np.float16).view(np.uint8) if a.rows == "f16" else q8_0(w)

    def work():
        buf = np.empty((16 * CH, K), np.float32)
        while True:
            with lk:
                g = nxt[0]; nxt[0] += CH
            if g >= ng:
                return
            g1 = min(g + CH, ng)
            if ex is None or a.stats:
                for gg in range(g, g1):
                    if not lib.ggml_ai4_dequantize_slab(tp, ptr, K, gg, buf[16 * (gg - g):].ctypes.data):
                        bad.append(gg); return
            if ex is None:
                out[16 * g:16 * g1] = conv(buf[:16 * (g1 - g)]).reshape(16 * (g1 - g), rb)
            else:
                rows = np.asarray(ex[16 * g:16 * g1])
                out[16 * g:16 * g1] = rows.view(np.uint8).reshape(16 * (g1 - g), rb)
                if gl2 is not None:
                    gl2[16 * g:16 * g1] = group_l2(buf[:16 * (g1 - g)], bf16_to_f32(rows))
    ts = [threading.Thread(target=work) for _ in range(a.threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not bad, f"decode failed at slab {bad[:3]}"
    print(f"decoded + converted in {time.time() - t0:.1f} s", flush=True)

    # verify: random rows re-decoded and re-converted must be byte-equal
    rng = random.Random(20261113)
    for gg in sorted(rng.sample(range(ng), min(a.verify, ng))):
        assert lib.ggml_ai4_dequantize_slab(tp, ptr, K, gg, y.ctypes.data)
        r = rng.randrange(16)
        if ex is None:
            assert np.array_equal(conv(y[r:r + 1]).reshape(rb), out[16 * gg + r]), f"verify row {16 * gg + r}"
        else:
            src_row = np.asarray(ex[16 * gg + r])
            assert np.array_equal(src_row.view(np.uint8), out[16 * gg + r]), f"verify row {16 * gg + r}"
            if gl2 is not None:
                assert np.array_equal(group_l2(y[r:r + 1], bf16_to_f32(src_row[None])), gl2[16 * gg + r:16 * gg + r + 1]), \
                    f"verify gl2 row {16 * gg + r}"
    print(f"verify: {min(a.verify, ng)} random rows byte-equal to a fresh {'decode' if ex is None else 'source read'}", flush=True)

    self_fnv = fnv64_ends(out.reshape(-1))
    tensors = [("output_exact.weight", K, V, {"f16": 1, "bf16": 30, "q8_0": 8}[a.rows], out)]
    kv = {
        "general.type": (8, "d1-sidecar"),
        "d1.rows_type": (8, a.rows),
        "d1.source_bytes": (10, int(src_bytes)),
        "d1.source_fnv64": (10, src_fnv),
        "d1.self_bytes": (10, int(out.nbytes)),
        "d1.self_fnv64": (10, self_fnv),
    }
    if gl2 is not None:
        kv["d1.gl2_fnv64"] = (10, fnv64_ends(gl2.reshape(-1).view(np.uint8)))
        tensors.append(("output_exact.gl2", K // 32, V, 0, gl2))
        print(f"gl2: max {gl2.max():.5g} mean {gl2.mean():.5g}", flush=True)
    write_sidecar(a.out, tensors, kv)
    print(f"D1_SIDECAR_DONE {a.out} rows={a.rows} bytes={out.nbytes} src_fnv={src_fnv:016x} self_fnv={self_fnv:016x} "
          f"wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
