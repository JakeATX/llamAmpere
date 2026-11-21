#!/usr/bin/env python3
"""MTPRETRAIN: write a trained MTP head as an LLAMA_MODEL_OVERLAY_PATCH sidecar with the SAME tensor names, shapes and
types as the model's own blk.64 tensors (Q5_0 / Q8_0 matrices, F32 norms), so the patched model has the same size and
VRAM fit as today. Plain reference quantization (gguf-py), no imatrix.
usage: export_overlay.py --w W0 --head head.pt --out mtp-otp.gguf"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402
from gguf.quants import quantize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', required=True); ap.add_argument('--head', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = json.loads((Path(a.w) / 'meta.json').read_text())
    head = torch.load(a.head)
    w = gguf.GGUFWriter(a.out, 'qwen35')
    w.add_string('mtpretrain.source', f'MTPRETRAIN on-policy retrained head; {Path(a.head).name}; sha256 {hashlib.sha256(Path(a.head).read_bytes()).hexdigest()[:16]}')
    w.add_string('mtpretrain.base_model', Path(meta['model']).name)
    n = 0
    for name, info in meta['tensors'].items():
        if not name.startswith('blk.'):
            continue
        assert name in head, f'{name} missing from the head checkpoint'
        t = head[name].float().numpy()
        ne = info['ne']
        qt = gguf.GGMLQuantizationType(info['type'])
        shape = tuple(reversed(ne))  # numpy row-major [ne1, ne0]
        t = t.reshape(shape).astype(np.float32)
        assert np.isfinite(t).all(), name
        if qt == gguf.GGMLQuantizationType.F32:
            w.add_tensor(name, t)
        else:
            q = quantize(t, qt)
            assert q.nbytes == info['n_bytes'], (name, q.nbytes, info['n_bytes'])
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=qt)
        n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f'wrote {a.out}: {n} tensors')


if __name__ == '__main__':
    main()
