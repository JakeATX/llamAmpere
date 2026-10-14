#!/usr/bin/env python3
"""Separate weight error from AI+'s rotated int8 activation error, using the same actual Swift inputs and Q."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

from screen_swift import load_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.screen / "screen.json").read_text())
    ids = torch.load(report["calibration"], weights_only=True).numpy()
    rng = np.random.default_rng(714)
    pools = [[], []]
    for row in ids:
        order = rng.permutation(np.arange(ids.shape[1]))
        for half in (0, 1): pools[half].append(row[order[half::2]])
    selected = [rng.choice(np.concatenate(p), report["heldout_samples"], replace=False) for p in pools]
    model = Path(report["source"])
    prefix = "model.language_model"
    x, _, _ = load_rows(model, prefix + ".embed_tokens.weight", selected[1][:2048])
    norm, _, _ = load_rows(model, prefix + ".layers.0.input_layernorm.weight")
    cfg = json.loads((model / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    x *= 1.0/np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + cfg["rms_norm_eps"])
    x *= 1.0 + norm
    lib = ctypes.CDLL(str(args.library))
    rotate = lib.ai4_screen_rotate
    rotate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    def transform(a, inverse=0):
        a = np.ascontiguousarray(a.copy())
        rc = rotate(a.ctypes.data, a.shape[1], a.shape[0], inverse)
        if rc: raise RuntimeError(rc)
        return a
    xr = transform(x)
    rb = 256 if x.shape[1] % 256 == 0 else 128
    grouped = xr.reshape(len(xr), -1, rb)
    amax = np.max(np.abs(grouped), axis=-1, keepdims=True)
    # rq4.cuh:132-137: fp32 scale amax/127, round-nearest-even of float(v*127/amax).
    inv = np.divide(127.0, amax, out=np.zeros_like(amax), where=amax > 0)
    xe = (np.rint(grouped*inv)*(amax/127.0)).reshape(xr.shape)
    result = {"samples": len(x), "group_width": rb,
              "scope": "CPU projection error, not whole-model KL or GPU summation timing", "tensors": {}}
    for name, entry in report["tensors"].items():
        suffix = name.split(".layers.0.")[1].removesuffix(".weight")
        W = np.load(args.screen / (suffix + ".weights.npy"))
        original = torch.from_numpy(x) @ torch.from_numpy(W).T
        den = float(original.double().square().sum())
        items = {}
        for arm in entry["arms"]:
            if not arm.startswith("ai_"): continue
            Q = np.load(args.screen / (suffix + "." + arm + ".npy"))
            qr = torch.from_numpy(transform(Q))
            float_out = torch.from_numpy(xr) @ qr.T
            int8_out = torch.from_numpy(xe) @ qr.T
            weight = float((float_out - original).double().square().sum()) / den
            total = float((int8_out - original).double().square().sum()) / den
            activation = float((int8_out - float_out).double().square().sum()) / den
            items[arm] = {"weight_only_relative_output_error": weight,
                          "weight_and_activation_relative_output_error": total,
                          "activation_only_relative_output_error": activation,
                          "total_over_weight": total/weight}
        result["tensors"][name] = items
    (args.screen / "activation_screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
