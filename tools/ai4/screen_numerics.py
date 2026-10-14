#!/usr/bin/env python3
"""Inspect real pre-feedback targets/scales for GPU clipping and half-cost overflow hazards."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    args = ap.parse_args()
    lib = ctypes.CDLL(str(args.library))
    rotate = lib.ai4_screen_rotate
    rotate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    states = np.arange(65536, dtype=np.uint32)
    x = states * np.uint32(0x83DCD12D)
    summed = sum(((x >> shift) & 255).astype(np.int32) for shift in (0, 8, 16, 24))
    levels = ((summed + 2) >> 2) - 128
    distinct = np.unique(levels)
    lrms = np.float32(np.sqrt(np.mean(levels.astype(np.float64)**2)))
    report = json.loads((args.screen / "screen.json").read_text())
    result = {"scope": "Initial rotated targets only; later LDLQ feedback and selected CUDA paths require GPU diagnostics",
              "level_min": int(levels.min()), "level_max": int(levels.max()), "level_rms": float(lrms), "tensors": {}}
    for name in report["tensors"]:
        suffix = name.split(".layers.0.")[1].removesuffix(".weight")
        W = np.load(args.screen / (suffix + ".weights.npy"))
        wr = np.ascontiguousarray(W.copy())
        if rotate(wr.ctypes.data, wr.shape[1], wr.shape[0], 0): raise RuntimeError("rotation failed")
        groups = wr.reshape(wr.shape[0], -1, 128)
        row = np.sqrt(np.mean(W.astype(np.float64)**2, axis=1)).astype(np.float32)[:, None]
        group = np.sqrt(np.mean(groups.astype(np.float64)**2, axis=-1)).astype(np.float32)
        info = {}
        for mode, rms in (("group", group), ("row", row)):
            scales = ((rms / lrms)*np.float32(1.10)).astype(np.float16).astype(np.float32)
            target = np.divide(groups, scales[..., None], out=np.zeros_like(groups), where=scales[..., None] > 0)
            # A bad candidate may overflow even while the optimum remains finite. Do not equate these events.
            worst = np.maximum(np.abs(target - levels.min()), np.abs(target - levels.max()))
            ix = np.searchsorted(distinct, target)
            near = np.minimum(np.abs(target - distinct[np.maximum(ix - 1, 0)]),
                              np.abs(target - distinct[np.minimum(ix, len(distinct) - 1)]))
            info[mode] = {"values": int(target.size), "max_abs_normalized_target": float(np.max(np.abs(target))),
                          "targets_beyond_gpu_clip250": int(np.sum(np.abs(target) > 250)),
                          "targets_with_some_half_square_overflow": int(np.sum(worst.astype(np.float64)**2 > 65504)),
                          "targets_with_all_levels_half_square_overflow": int(np.sum(near.astype(np.float64)**2 > 65504)),
                          "scale_min": float(scales.min()), "scale_max": float(scales.max()),
                          "zero_scales": int(np.sum(scales == 0)), "nonfinite_scales": int(np.sum(~np.isfinite(scales)))}
        result["tensors"][name] = info
    (args.screen / "numerics_screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
