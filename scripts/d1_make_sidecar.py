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
"""
import argparse, ctypes, os, random, struct, sys, threading, time

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


def write_sidecar(path, K, V, ggml_type, data, kv):
    align = 32
    with open(path + ".part", "wb") as f:
        f.write(b"GGUF"); f.write(struct.pack("<IQQ", 3, 1, len(kv)))
        for k, (t, v) in kv.items():
            _wstr(f, k); f.write(struct.pack("<I", t))
            if t == 8: _wstr(f, v)
            elif t == 10: f.write(struct.pack("<Q", v))
            else: raise ValueError(t)
        _wstr(f, "output_exact.weight"); f.write(struct.pack("<I", 2)); f.write(struct.pack("<QQ", K, V))
        f.write(struct.pack("<I", ggml_type)); f.write(struct.pack("<Q", 0))
        pad = (-f.tell()) % align
        f.write(b"\0" * pad)
        mv = memoryview(data).cast("B")
        for a in range(0, len(mv), 1 << 28):
            f.write(mv[a:a + (1 << 28)])
    os.replace(path + ".part", path)


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
    a = ap.parse_args()
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
    rb = K * 2 if a.rows == "f16" else K // 32 * 34
    out = np.empty((V, rb), np.uint8)
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
            for gg in range(g, g1):
                if not lib.ggml_ai4_dequantize_slab(tp, ptr, K, gg, buf[16 * (gg - g):].ctypes.data):
                    bad.append(gg); return
            out[16 * g:16 * g1] = conv(buf[:16 * (g1 - g)]).reshape(16 * (g1 - g), rb)
    ts = [threading.Thread(target=work) for _ in range(a.threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not bad, f"decode failed at slab {bad[:3]}"
    print(f"decoded + converted in {time.time() - t0:.1f} s", flush=True)

    # verify: random rows re-decoded and re-converted must be byte-equal
    rng = random.Random(20261113)
    for gg in sorted(rng.sample(range(ng), min(a.verify, ng))):
        assert lib.ggml_ai4_dequantize_slab(tp, ptr, K, gg, y.ctypes.data)
        r = rng.randrange(16)
        assert np.array_equal(conv(y[r:r + 1]).reshape(rb), out[16 * gg + r]), f"verify row {16 * gg + r}"
    print(f"verify: {min(a.verify, ng)} random rows byte-equal to a fresh decode", flush=True)

    self_fnv = fnv64_ends(out.reshape(-1))
    kv = {
        "general.type": (8, "d1-sidecar"),
        "d1.rows_type": (8, a.rows),
        "d1.source_bytes": (10, int(src_bytes)),
        "d1.source_fnv64": (10, src_fnv),
        "d1.self_bytes": (10, int(out.nbytes)),
        "d1.self_fnv64": (10, self_fnv),
    }
    write_sidecar(a.out, K, V, 1 if a.rows == "f16" else 8, out, kv)
    print(f"D1_SIDECAR_DONE {a.out} rows={a.rows} bytes={out.nbytes} src_fnv={src_fnv:016x} self_fnv={self_fnv:016x} "
          f"wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
