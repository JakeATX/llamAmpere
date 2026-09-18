#!/usr/bin/env python3
# EXL3 format and trellis layout: Turboderp, exllamav3 (https://github.com/turboderp-org/exllamav3), MIT License,
# Copyright (c) 2025 Turboderp; see licenses/LICENSE-exllamav3. Independent reimplementation.
"""EXL3 CPU reference dequant (M0 of EX2_exl3_port). numpy only; mirrors exllamav3 v1.5.0 kernels:
  window:   weight t of a 256-weight tile = 16-bit window at bit offset ((t+257)*bits - 16) mod (256*bits) of the little-endian
            uint32 stream (exl3_dq.cuh dq/dq2/dq4: b0 = t*bits + bits - 16 + 256*bits, ptr[i % (bits*256/32)]).
  codebook: cb2 "mul1": x = (w16 * 0x83DCD12D) mod 2^32; s = bytesum(x) + 0x6400; half(s) * half(0x1eee) + half(0xc931), one rounding (hfma).
            cb1 "mcg" / cb0 not implemented yet (this checkpoint family is mul1).
  tile map: reconstruct.cu lane layout: t = lane*8 + j ->
            row(k) = 2*(lane%4) + (j&1) + 8*((j>>1)&1);  col(n) = 2*(lane>>3) + ((lane>>2)&1) + 8*(j>>2)
  W_eff[k,n] = svh[n] * (H128 @ (suh[:,None] * (H128 @ C)) @ H128)... precisely: C -> had_l (1/sqrt(128) per 128-row block)
            -> * suh[:,None] -> had_r (per 128-col block) -> * svh[None,:]   (exl3.py reconstruct(), preapply_had_l/r)
usage: exl3_ref.py MODEL_DIR TENSOR_PREFIX [--save W.npy] [--raw C.npy]
"""
import argparse, json, os, struct, sys
import numpy as np

MUL1 = 0x83DCD12D
K_INV = np.frombuffer(np.array([0x1eee], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)
K_BIAS = np.frombuffer(np.array([0xc931], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)

def load_safetensors_tensor(model_dir, name):
    """Minimal safetensors reader (no torch): returns numpy array for `name` from whichever shard holds it."""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    shard = idx["weight_map"][name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16, "F32": np.float32}[meta["dtype"]]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        buf = f.read(b - a)
    arr = np.frombuffer(buf, dtype=dt).reshape(meta["shape"])
    return arr

def tile_windows(words_u32, bits):
    """words_u32: [T, 8*bits] uint32 (the tile's raw little-endian words, ptr[i] in exl3_dq.cuh).
    Returns [T, 256] uint16 codewords in weight order t.
    Bit order (from fshift(b, a, s) = ((a << 32) | b) >> s with a = ptr[i0], b = ptr[i2], i0 < i2): the tile is a
    circular bit string of 256*bits bits in which word i holds stream bits [32i, 32i+32) MSB-first, i.e. stream bit
    p sits at bit (31 - p % 32) of word p // 32. The codeword of weight t is the 16-bit window starting at stream bit
    ((t + 257) * bits - 16) mod 256*bits, read as a big-endian number (earliest stream bit = MSB)."""
    T, nw = words_u32.shape
    assert nw == 8 * bits
    nbits = 256 * bits
    bitarr = np.unpackbits(words_u32.astype('>u4').view(np.uint8).reshape(T, 4 * nw), axis=1)   # [T, nbits] MSB-first
    t = np.arange(256)
    b0 = ((t + 257) * bits - 16) % nbits
    idx = (b0[:, None] + np.arange(16)[None, :]) % nbits                                    # [256, 16]
    win = bitarr[:, idx].astype(np.uint16)                                                  # [T, 256, 16]
    w = (win << (15 - np.arange(16, dtype=np.uint16))[None, None, :]).sum(axis=-1, dtype=np.uint32)
    return w.astype(np.uint16)

def decode_mul1(w16):
    x = (w16.astype(np.uint64) * np.uint64(MUL1)) & np.uint64(0xffffffff)
    s = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff) + 0x6400
    h = np.frombuffer(s.astype(np.uint16).tobytes(), dtype=np.float16).astype(np.float32)
    v = h * K_INV + K_BIAS                            # exact in fp32 (h integer <= 2044, K_INV 11-bit mantissa)
    return v.astype(np.float16)                       # single rounding == hfma

def tile_layout():
    """(row, col) for weight index t in 0..255 within the 16x16 tile."""
    t = np.arange(256); lane = t // 8; j = t % 8
    row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    col = 2 * (lane >> 3) + ((lane >> 2) & 1) + 8 * (j >> 2)
    assert len(set(zip(row.tolist(), col.tolist()))) == 256
    return row, col

def reconstruct_C(trellis_i16):
    """trellis [K/16, N/16, 16*bits] int16 -> C [K, N] float16 (raw codebook values, == exllamav3 ext.reconstruct)."""
    kt, nt, w16 = trellis_i16.shape
    bits = w16 // 16
    words = np.frombuffer(np.ascontiguousarray(trellis_i16).tobytes(), dtype=np.uint32).reshape(kt * nt, 8 * bits)
    # tile_windows expands each tile to [256, 16] bits; do it in chunks so a wide (fused) tensor does not need
    # ~16 GB of temporaries. Chunking is a memory bound only -- the result is bit-identical to one big call.
    CHUNK = 32768
    vals = np.empty((kt * nt, 256), dtype=np.float16)
    for i in range(0, kt * nt, CHUNK):
        vals[i:i + CHUNK] = decode_mul1(tile_windows(words[i:i + CHUNK], bits)).reshape(-1, 256)   # [T, 256]
    row, col = tile_layout()
    C = np.empty((kt * 16, nt * 16), dtype=np.float16)
    tiles = vals.reshape(kt, nt, 256)
    tile = np.empty((kt, nt, 16, 16), dtype=np.float16)
    tile[:, :, row, col] = tiles
    C[:] = tile.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)
    return C, bits

def hadamard128():
    i = np.arange(128)
    H = np.where((np.bitwise_count(i[:, None] & i[None, :]) & 1) == 0, 1.0, -1.0).astype(np.float32)
    return H / np.sqrt(128.0)

def reconstruct_W(C, suh, svh):
    """exl3.py reconstruct(): had_l -> *suh -> had_r -> *svh, each step in fp32 and rounded to fp16 like the reference."""
    H = hadamard128()
    k, n = C.shape
    w = C.astype(np.float32).reshape(k // 128, 128, n)
    w = np.einsum("ij,bjn->bin", H, w).reshape(k, n).astype(np.float16)
    w = (w * suh[:, None]).astype(np.float16)
    w = w.astype(np.float32).reshape(k, n // 128, 128)
    w = np.einsum("bki,ij->bkj", w, H).reshape(k, n).astype(np.float16)
    w = (w * svh[None, :]).astype(np.float16)
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("prefix")
    ap.add_argument("--save"); ap.add_argument("--raw")
    a = ap.parse_args()
    tr = load_safetensors_tensor(a.model_dir, a.prefix + ".trellis")
    suh = load_safetensors_tensor(a.model_dir, a.prefix + ".suh").astype(np.float16)
    svh = load_safetensors_tensor(a.model_dir, a.prefix + ".svh").astype(np.float16)
    C, bits = reconstruct_C(tr)
    W = reconstruct_W(C, suh, svh)
    print(f"{a.prefix}: bits={bits} trellis={tr.shape} C={C.shape} "
          f"C mean={C.astype(np.float32).mean():+.4f} std={C.astype(np.float32).std():.4f} "
          f"min={C.min():+.3f} max={C.max():+.3f} | W std={W.astype(np.float32).std():.5f} "
          f"suh|.|mean={np.abs(suh.astype(np.float32)).mean():.4f} svh|.|mean={np.abs(svh.astype(np.float32)).mean():.4f}")
    if a.raw: np.save(a.raw, C)
    if a.save: np.save(a.save, W)

if __name__ == "__main__":
    main()
