#!/usr/bin/env python3
"""Convert a latent-sparse-attention index map (.pt projection file) to a sidecar GGUF for --attn-sparse-map.

The input is a flat torch file of `<layer>.W_Q_idx` / `<layer>.W_K_idx`, each [n_head_kv*(head_dim-rotary), index_dim]
f32, applied as normalize(x @ W) to the position-free dims of the post-norm keys (side by side over the KV heads)
and to the per-group sums of the queries. The file is read without torch and without running its pickle: a
restricted unpickler only accepts OrderedDict, torch._utils._rebuild_tensor_v2 and torch.*Storage, and the tensor
bytes come straight from the zip's data/<key> entries.

The output pads each map to the full head width with zero rows for the rotary prefix, so the engine applies it to
the cached post-RoPE K and the post-RoPE Q directly (RoPE leaves the other dims untouched):
    lsa.<layer>.wq, lsa.<layer>.wk : f32, numpy shape [index_dim, n_head_kv*head_dim] (ggml ne = [n_head_kv*head_dim, index_dim])

Usage: lsa_convert_map.py IN.pt OUT.gguf [--head-dim 256] [--rotary 64] [--n-head-kv 4] [--model-gguf M.gguf]
With --model-gguf the layer ids are checked against the model's full-attention layers.
With --pca-dump, IN is an LLAMA_LSA_DUMP prefix from a separate calibration prompt. Fit an uncentered PCA basis
on visible keys, with rotary dimensions excluded. K and Q use the same basis; no mean offset is needed at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "BFloat16Storage": "bf16",
}


def read_pca_dump(prefix: Path, head_dim: int, rotary: int, n_head_kv: int, rank: int, samples: int):
    width = head_dim * n_head_kv
    if not (0 <= rotary < head_dim and 1 <= rank <= (head_dim - rotary) * n_head_kv and samples >= rank):
        raise ValueError("invalid PCA dimensions or sample count")
    index_path = Path(f"{prefix}.index.txt")
    index = {}
    sha = hashlib.sha256(index_path.read_bytes())
    for line in index_path.read_text().splitlines():
        p = line.split()
        if len(p) != 7:
            raise ValueError("invalid dump index row")
        if p[1] not in ("k", "mask"):
            continue
        key = (int(p[0]), p[1])
        dims = tuple(int(x) for x in p[2:6])
        if p[6] != "ok" or any(x <= 0 for x in dims) or key in index:
            raise ValueError(f"failed, invalid or duplicate dump entry: {key}")
        index[key] = dims
    layers = sorted({il for il, _ in index})
    if not layers:
        raise ValueError("no key dumps")
    tensors = {}
    for il in layers:
        kd = index.get((il, "k"))
        md = index.get((il, "mask"))
        if kd is None or md is None or kd != (width, kd[1], 1, 1) or md != (kd[1], md[1], 1, 1):
            raise ValueError(f"layer {il}: wrong key/mask dimensions")
        arrays = {}
        for name, dims in (("k", kd), ("mask", md)):
            path = Path(f"{prefix}.L{il}.{name}.f32")
            if path.stat().st_size != int(np.prod(dims, dtype=object)) * 4:
                raise ValueError(f"wrong dump length: {path}")
            with path.open("rb") as f:
                while chunk := f.read(1 << 20):
                    sha.update(chunk)
            arrays[name] = np.memmap(path, dtype="<f4", mode="r", shape=(dims[1], dims[0]))
        mask = arrays["mask"][-1]
        rows = np.flatnonzero(np.isfinite(mask) & (mask > -1e29))
        if rows.size < rank:
            raise ValueError(f"layer {il}: too few visible calibration keys")
        # Sample across the whole visible prefix; never fit to unused padded cache cells.
        rows = rows[np.linspace(0, rows.size - 1, min(samples, rows.size), dtype=np.int64)]
        x = np.asarray(arrays["k"][rows], dtype=np.float64).reshape(-1, n_head_kv, head_dim)
        x = x[:, :, rotary:].reshape(rows.size, -1)
        if not np.isfinite(x).all():
            raise ValueError(f"layer {il}: non-finite calibration keys")
        values, vectors = np.linalg.eigh(x.T @ x / rows.size)
        if values[-rank] <= max(1e-12, values[-1] * 1e-10):
            raise ValueError(f"layer {il}: calibration keys have insufficient rank")
        basis = vectors[:, -rank:][:, ::-1]
        # Fix each eigenvector's sign so saved maps do not depend on the eigensolver's sign choice.
        signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(rank)])
        basis = np.ascontiguousarray(basis * signs, dtype=np.float32)
        for kind in ("W_Q_idx", "W_K_idx"):
            tensors[f"{il}.{kind}"] = basis
        print(f"layer {il}: uncentered PCA rank {rank}, {rows.size} calibration keys")
    return tensors, sha.hexdigest()


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _Storage:
    def __init__(self, key: str, dtype):
        self.key = key
        self.dtype = dtype


class _TensorRef:
    def __init__(self, storage: _Storage, offset: int, size: tuple, stride: tuple):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _TensorRef(storage, storage_offset, size, stride)


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if not isinstance(pid, tuple) or len(pid) < 3 or pid[0] != "storage" or not isinstance(pid[1], _StorageType):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        return _Storage(str(pid[2]), _STORAGE_DTYPES[pid[1].name])


def read_pt(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise ValueError(f"{path}: expected one data.pkl, found {pkl}")
        root = pkl[0][: -len("data.pkl")]
        bo = root + "byteorder"
        if bo in names and z.read(bo).strip() not in (b"little",):
            raise ValueError(f"{path}: big-endian storage is not supported")
        obj = _SafeUnpickler(__import__("io").BytesIO(z.read(pkl[0]))).load()
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: top level is {type(obj).__name__}, not a dict")
        out = {}
        for name, ref in obj.items():
            if not isinstance(ref, _TensorRef):
                raise ValueError(f"{path}: entry {name} is not a tensor")
            raw = z.read(f"{root}data/{ref.storage.key}")
            if ref.storage.dtype == "bf16":
                u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                flat = u16.view(np.float32)
            else:
                flat = np.frombuffer(raw, dtype=np.dtype(ref.storage.dtype).newbyteorder("<"))
            item = flat.itemsize
            arr = np.lib.stride_tricks.as_strided(
                flat[ref.offset:], shape=ref.size, strides=tuple(s * item for s in ref.stride)
            )
            out[str(name)] = np.array(arr, dtype=np.float32)
    return out


def model_full_attn_layers(model_path: Path) -> list[int] | None:
    r = gguf.GGUFReader(str(model_path))
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode()
    if arch is None:
        return None
    n_layer = interval = None
    for f in r.fields.values():
        if f.name == f"{arch}.block_count":
            n_layer = int(f.parts[f.data[0]][0])
        if f.name == f"{arch}.full_attention_interval":
            interval = int(f.parts[f.data[0]][0])
    if n_layer is None or interval is None:
        return None
    return [il for il in range(n_layer) if (il + 1) % interval == 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--rotary", type=int, default=64)
    ap.add_argument("--n-head-kv", type=int, default=4)
    ap.add_argument("--model-gguf", type=Path, default=None)
    ap.add_argument("--name", default=None, help="map name stored in the metadata (default: input file stem)")
    ap.add_argument("--pca-dump", action="store_true", help="fit a map from a dense calibration dump prefix")
    ap.add_argument("--index-dim", type=int, default=64, help="PCA rank (only with --pca-dump)")
    ap.add_argument("--pca-samples", type=int, default=32768, help="maximum keys sampled per layer")
    args = ap.parse_args()

    if args.pca_dump:
        tensors, sha = read_pca_dump(args.inp, args.head_dim, args.rotary, args.n_head_kv, args.index_dim, args.pca_samples)
    else:
        tensors = read_pt(args.inp)
        sha = hashlib.sha256(args.inp.read_bytes()).hexdigest()

    nope = args.head_dim - args.rotary
    d_in = args.n_head_kv * nope
    d_full = args.n_head_kv * args.head_dim

    layers = sorted({int(k.split(".")[0]) for k in tensors})
    index_dim = None
    for il in layers:
        for kind in ("W_Q_idx", "W_K_idx"):
            t = tensors.get(f"{il}.{kind}")
            if t is None:
                raise SystemExit(f"layer {il}: missing {kind}")
            if t.ndim != 2 or t.shape[0] != d_in:
                raise SystemExit(f"layer {il} {kind}: shape {t.shape}, expected [{d_in}, index_dim]")
            if index_dim is None:
                index_dim = t.shape[1]
            elif t.shape[1] != index_dim:
                raise SystemExit(f"layer {il} {kind}: index_dim {t.shape[1]} != {index_dim}")
            if not np.isfinite(t).all():
                raise SystemExit(f"layer {il} {kind}: non-finite values")
    extra = set(tensors) - {f"{il}.{k}" for il in layers for k in ("W_Q_idx", "W_K_idx")}
    if extra:
        raise SystemExit(f"unexpected entries: {sorted(extra)[:5]}")

    if args.model_gguf is not None:
        want = model_full_attn_layers(args.model_gguf)
        if want is None:
            print("warning: could not read the model's full-attention layers; skipping the layer check")
        elif want != layers:
            raise SystemExit(f"map layers {layers} != model full-attention layers {want}")
        else:
            print(f"layer check OK against {args.model_gguf.name}: {layers}")

    w = gguf.GGUFWriter(str(args.out), arch="lsa_map")
    w.add_string("lsa.source_name", args.name or args.inp.stem)
    w.add_string("lsa.source_sha256", sha)
    if args.pca_dump:
        w.add_string("lsa.method", "uncentered-pca")
        w.add_uint32("lsa.pca_samples_max", args.pca_samples)
    w.add_uint32("lsa.index_dim", index_dim)
    w.add_uint32("lsa.head_dim", args.head_dim)
    w.add_uint32("lsa.rotary_dim", args.rotary)
    w.add_uint32("lsa.n_head_kv", args.n_head_kv)
    w.add_array("lsa.layers", layers)

    for il in layers:
        for kind, short in (("W_Q_idx", "wq"), ("W_K_idx", "wk")):
            t = tensors[f"{il}.{kind}"]
            full = np.zeros((d_full, index_dim), dtype=np.float32)
            for h in range(args.n_head_kv):
                full[h * args.head_dim + args.rotary:(h + 1) * args.head_dim] = t[h * nope:(h + 1) * nope]
            # numpy [index_dim, d_full] -> ggml ne = [d_full, index_dim]
            w.add_tensor(f"lsa.{il}.{short}", np.ascontiguousarray(full.T))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {len(layers)} layers {layers[0]}..{layers[-1]}, index_dim {index_dim}, "
          f"[{d_in} -> {d_full}] padded, sha256(in) {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
