#!/usr/bin/env python3
"""MTPRETRAIN: dequantize the MTP block (blk.<n_layer>.*), the LM head (output.weight) and the token embedding of a
qwen35 GGUF into torch tensors (bf16 matrices, f32 norms) for offline head training.
usage: extract.py MODEL.gguf OUTDIR
Writes OUTDIR/mtp.pt (dict name -> tensor, GGUF names, torch shape [rows, cols] = [ne1, ne0]),
OUTDIR/lm_head.bf16.pt, OUTDIR/tok_embd.bf16.pt and OUTDIR/meta.json (types, shapes, hparams)."""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

CHUNK_ROWS = 8192


def deq(t):
    """dequantize a GGUFReader tensor to a float32 numpy array [ne1, ne0], row chunks to bound RAM."""
    ne = [int(x) for x in t.shape]  # ne0 first
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    qt = gguf.GGMLQuantizationType(int(t.tensor_type))
    if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        return dequantize(np.asarray(t.data), qt).reshape(rows, ne[0]).astype(np.float32)
    data = np.asarray(t.data).reshape(rows, -1)  # bytes per row
    out = np.empty((rows, ne[0]), dtype=np.float32)
    for r0 in range(0, rows, CHUNK_ROWS):
        out[r0:r0 + CHUNK_ROWS] = dequantize(data[r0:r0 + CHUNK_ROWS], qt).reshape(-1, ne[0])
    return out


def main():
    model, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(model)
    f = {k: v.contents() for k, v in r.fields.items() if k.startswith('qwen35.') or k == 'general.architecture'}
    n_layer = int(f['qwen35.block_count']) - int(f['qwen35.nextn_predict_layers'])
    pre = f'blk.{n_layer}.'
    meta = dict(model=model, n_layer_trunk=n_layer, hparams={k: (v if isinstance(v, (int, float, str)) else list(v)) for k, v in f.items()},
                tensors={})
    mtp = {}
    for t in r.tensors:
        if t.name.startswith(pre):
            a = deq(t)
            if a.shape[0] == 1:
                a = a.reshape(-1)
            mtp[t.name] = torch.from_numpy(a).float() if a.ndim == 1 else torch.from_numpy(a).to(torch.bfloat16)
            meta['tensors'][t.name] = dict(type=int(t.tensor_type), ne=[int(x) for x in t.shape], n_bytes=int(t.n_bytes))
            print(t.name, gguf.GGMLQuantizationType(int(t.tensor_type)).name, tuple(mtp[t.name].shape), flush=True)
        elif t.name in ('output.weight', 'token_embd.weight', 'output_norm.weight'):
            a = deq(t)
            meta['tensors'][t.name] = dict(type=int(t.tensor_type), ne=[int(x) for x in t.shape], n_bytes=int(t.n_bytes))
            if t.name == 'output_norm.weight':
                mtp[t.name] = torch.from_numpy(a.reshape(-1)).float()
            else:
                torch.save(torch.from_numpy(a).to(torch.bfloat16), outdir / ('lm_head.bf16.pt' if t.name == 'output.weight' else 'tok_embd.bf16.pt'))
            print(t.name, gguf.GGMLQuantizationType(int(t.tensor_type)).name, a.shape, flush=True)
            del a
    torch.save(mtp, outdir / 'mtp.pt')
    (outdir / 'meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
