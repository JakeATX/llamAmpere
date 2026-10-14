#!/usr/bin/env python3
"""Private CPU screen: real first-layer Swift rows, actual inputs, common held-out full-H error."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch


def load_rows(model, name, rows=None):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    path = model / index["weight_map"][name]
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[meta["dtype"]]
    a = np.memmap(path, mode="r", offset=8 + n + meta["data_offsets"][0], dtype=dtype, shape=tuple(meta["shape"]))
    a = np.asarray(a if rows is None else a[rows]).copy()
    if meta["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32), meta, str(path)


def metric(W, Q, inputs):
    E = torch.from_numpy(W - Q).double()
    X = inputs.double()
    wr = torch.from_numpy(W).double()
    num = float((E @ X.T).square().sum()) / len(X)
    den = float((wr @ X.T).square().sum()) / len(X)
    return {"mse": float(np.mean((W.astype(np.float64) - Q)**2)),
            "relative_mse": float(np.sum((W.astype(np.float64) - Q)**2) / np.sum(W.astype(np.float64)**2)),
            "h_error": num, "relative_h_error": num / den}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--exl3-library", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--smul", type=float, nargs="+", default=[1.10])
    ap.add_argument("--scale-mode", nargs="+", choices=("group", "row"), default=["group"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    assert args.rows % 128 == 0 and args.offset % 128 == 0
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    # Load the already used local CPU EXL3 implementation; do not substitute another encoder.
    spec = importlib.util.spec_from_file_location("exl3_cpu", args.exl3_library)
    exl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exl)
    exl.NTHREADS = args.threads
    lib = ctypes.CDLL(str(args.library))
    fn = lib.ai4_screen_encode
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    config = json.loads((args.model / "config.json").read_text())
    config = config.get("text_config", config)
    prefix = "model.language_model"
    ids = torch.load(args.calibration, map_location="cpu", weights_only=True).numpy()
    rng = np.random.default_rng(714)
    # Each half contains positions from every calibration row, rather than separate calibration domains.
    positions = np.arange(ids.shape[1])
    pools = [[], []]
    for row in ids:
        order = rng.permutation(positions)
        for half in (0, 1):
            pools[half].append(row[order[half::2]])
    chosen = [rng.choice(np.concatenate(pool), args.samples, replace=False) for pool in pools]
    norm, _, _ = load_rows(args.model, prefix + ".layers.0.input_layernorm.weight")
    inputs = []
    for tokens in chosen:
        emb, _, _ = load_rows(args.model, prefix + ".embed_tokens.weight", tokens)
        x = torch.from_numpy(emb)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
        # Qwen3.5/3.8 stores an offset from one; conversion/qwen.py applies the same +1.
        x *= torch.from_numpy(1.0 + norm)
        inputs.append(x)
    H = (inputs[0].T @ inputs[0]) / args.samples
    lam = np.ascontiguousarray(H.diag().numpy())
    np.save(args.out / "train_input_diagonal.npy", lam)
    report = {"source": str(args.model), "calibration": str(args.calibration),
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "train_samples": args.samples, "heldout_samples": args.samples,
              "split": "disjoint shuffled positions within every calibration row (seed714)",
              "row_selection": [args.offset, args.offset + args.rows],
              "scale_modes": args.scale_mode,
              "exl3_reference": str(args.exl3_library), "exl3_int16": exl.INT16,
              "ai4_accumulator": "CPU fp32; GPU encoder half arithmetic is a separate rental gate",
              "tensors": {}}
    for suffix in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        name = prefix + ".layers.0." + suffix + ".weight"
        W, meta, shard = load_rows(args.model, name, slice(args.offset, args.offset + args.rows))
        W = np.ascontiguousarray(W)
        print(f"TENSOR {name} full_shape={meta['shape']} screen_shape={list(W.shape)}", flush=True)
        entry = {"full_shape": meta["shape"], "screen_shape": list(W.shape), "source_shard": shard,
                 "weight_slice_sha256": hashlib.sha256(W.tobytes()).hexdigest(), "arms": {}}
        report["tensors"][name] = entry
        np.save(args.out / (suffix + ".weights.npy"), W)
        for bits in args.bits:
            # Fresh Hessian, identical calibration and seed per tensor/rate. Ignore EXL3's own normalized proxy.
            hd = {"H": H.clone()*args.samples, "count": args.samples, "k": W.shape[1]}
            t0 = time.monotonic()
            Q, _, tensors, info = exl.quantize_exl3(torch.from_numpy(W.T.copy()), hd, bits, seed=1, share_su=True)
            Q = np.ascontiguousarray(Q.numpy().T)
            arm = f"exl3_k{bits}"
            entry["arms"][arm] = {"train": metric(W, Q, inputs[0]), "heldout": metric(W, Q, inputs[1]),
                                    "cpu_seconds": time.monotonic() - t0, "g_scale": info["g_scale"]}
            np.save(args.out / (suffix + "." + arm + ".npy"), Q)
            print(arm, entry["arms"][arm], flush=True)
            del hd, tensors
            for smul in args.smul:
                for mode in args.scale_mode:
                    for weighted in (0, 1):
                        t0 = time.monotonic()
                        Q = np.empty_like(W)
                        rc = fn(W.ctypes.data, W.shape[1], W.shape[0], lam.ctypes.data, bits, smul, weighted,
                                mode == "row", Q.ctypes.data)
                        if rc: raise RuntimeError(f"AI encoder returned {rc}")
                        tag = "_row" if mode == "row" else ""
                        arm = f"ai_k{bits}_{'weight' if weighted else 'normalized'}{tag}_s{smul:.2f}"
                        entry["arms"][arm] = {"train": metric(W, Q, inputs[0]), "heldout": metric(W, Q, inputs[1]),
                                                "cpu_seconds": time.monotonic() - t0}
                        np.save(args.out / (suffix + "." + arm + ".npy"), Q)
                        print(arm, entry["arms"][arm], flush=True)
            (args.out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
