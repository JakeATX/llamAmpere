#!/usr/bin/env python3
"""Streaming GGUF splicer for Inkling 2 Mini attention surgery overlays.

Copies an existing Inkling 2 Mini GGUF (e.g. NVFP4-native-q8d) byte-for-byte for all
untouched tensors (preserving NVFP4 experts, scales, tokenizer, loop gates, etc.) while
replacing ONLY the adapted attention block tensors (with Q8_0 matching the source quant
for 2D weights and F32 for norms/clips/convs/extended (16, 4096) attn_rel_bias) and
writing the Option 1A+1B + lsa_far_bias GGUF metadata keys.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GGUF_PY = SCRIPT_DIR.parent.parent / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType  # noqa: E402
from gguf.quants import quantize  # noqa: E402


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def pack_kv_u32(key: str, val: int) -> bytes:
    return pack_str(key) + struct.pack("<II", int(GGUFValueType.UINT32), int(val))


def pack_kv_f32(key: str, val: float) -> bytes:
    return pack_str(key) + struct.pack("<If", int(GGUFValueType.FLOAT32), float(val))


def pack_kv_u32_arr(key: str, vals: List[int]) -> bytes:
    buf = bytearray()
    buf += pack_str(key)
    buf += struct.pack("<IIQ", int(GGUFValueType.ARRAY), int(GGUFValueType.UINT32), len(vals))
    for v in vals:
        buf += struct.pack("<I", int(v))
    return bytes(buf)


def pack_kv_f32_arr(key: str, vals: List[float]) -> bytes:
    buf = bytearray()
    buf += pack_str(key)
    buf += struct.pack("<IIQ", int(GGUFValueType.ARRAY), int(GGUFValueType.FLOAT32), len(vals))
    for v in vals:
        buf += struct.pack("<f", float(v))
    return bytes(buf)


def build_replacement_map(overlay: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, GGMLQuantizationType]]:
    """Map GGUF tensor names -> (float32 numpy array in PyTorch/NumPy row-major shape, target quant type)."""
    repl: Dict[str, Tuple[np.ndarray, GGMLQuantizationType]] = {}
    layers_dict = overlay["layers"]
    for il_key, tensors in layers_dict.items():
        il = int(il_key)
        # 2D weights -> Q8_0 (matching Inkling-2-Mini-NVFP4-native-q8d.gguf)
        for src_k, gguf_suffix in (
            ("wq", "attn_q.weight"),
            ("wk", "attn_k.weight"),
            ("wv", "attn_v.weight"),
            ("wo", "attn_output.weight"),
            ("w_gate", "attn_gate.weight"),
            ("w_rel", "attn_rel.weight"),
            ("w_sink", "attn_sink_proj.weight"),
        ):
            arr = tensors[src_k].detach().float().cpu().numpy()
            repl[f"blk.{il}.{gguf_suffix}"] = (np.ascontiguousarray(arr, dtype=np.float32), GGMLQuantizationType.Q8_0)

        # F32 tensors (norms, clips, convs, rel_bias)
        repl[f"blk.{il}.attn_norm.weight"] = (
            np.ascontiguousarray(tensors["attn_norm"].detach().float().cpu().numpy(), dtype=np.float32),
            GGMLQuantizationType.F32,
        )
        repl[f"blk.{il}.attn_q_clip.weight"] = (
            np.ascontiguousarray(tensors["q_clip"].detach().float().cpu().numpy(), dtype=np.float32),
            GGMLQuantizationType.F32,
        )
        repl[f"blk.{il}.attn_rel_clip.weight"] = (
            np.ascontiguousarray(tensors["rel_clip"].detach().float().cpu().numpy(), dtype=np.float32),
            GGMLQuantizationType.F32,
        )
        # HF rel_bias is [16, E]; GGUF stores transposed [E, 16] (ne0=16, ne1=E)
        rb = tensors["rel_bias"].detach().float().cpu().numpy().T
        repl[f"blk.{il}.attn_rel_bias.weight"] = (
            np.ascontiguousarray(rb, dtype=np.float32),
            GGMLQuantizationType.F32,
        )
        for src_k, gguf_suffix in (
            ("k_sconv", "attn_k_conv.weight"),
            ("v_sconv", "attn_v_conv.weight"),
            ("attn_sconv", "attn_post_conv.weight"),
        ):
            # HF [C, 1, 4] -> squeeze(1) -> [C, 4] (GGUF ne0=4, ne1=C)
            c_arr = tensors[src_k].detach().float().cpu().numpy()
            if c_arr.ndim == 3:
                c_arr = c_arr[:, 0, :]
            repl[f"blk.{il}.{gguf_suffix}"] = (
                np.ascontiguousarray(c_arr, dtype=np.float32),
                GGMLQuantizationType.F32,
            )
    return repl


def align_up(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def splice_gguf(src_path: Path, overlay_path: Path, dst_path: Path, dry_run: bool = False) -> None:
    t0 = time.time()
    overlay = torch.load(overlay_path, map_location="cpu", weights_only=False)
    repl_specs = build_replacement_map(overlay)

    reader = GGUFReader(str(src_path))
    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8") if arch_field else "inkling2_loop"
    align_field = reader.fields.get("general.alignment")
    alignment = int(align_field.parts[align_field.data[0]][0]) if align_field else 32

    # Prepare new metadata entries
    new_kvs: Dict[str, bytes] = {
        f"{arch}.attention.layer_modes": pack_kv_u32_arr(f"{arch}.attention.layer_modes", [int(x) for x in overlay["layer_modes"]]),
        f"{arch}.attention.sliding_window_mid": pack_kv_u32(f"{arch}.attention.sliding_window_mid", int(overlay["swa_mid"])),
        f"{arch}.attention.relative_extent_mid": pack_kv_u32(f"{arch}.attention.relative_extent_mid", int(overlay["relative_extent_mid"])),
        f"{arch}.attention.lsa_keep_min": pack_kv_u32(f"{arch}.attention.lsa_keep_min", int(overlay["lsa_keep_min"])),
        f"{arch}.attention.lsa_keep_frac": pack_kv_f32(f"{arch}.attention.lsa_keep_frac", float(overlay["lsa_keep_frac"])),
        f"{arch}.attention.lsa_keep_max": pack_kv_u32(f"{arch}.attention.lsa_keep_max", int(overlay["lsa_keep_max"])),
        f"{arch}.attention.lsa_block": pack_kv_u32(f"{arch}.attention.lsa_block", int(overlay["lsa_block"])),
        f"{arch}.attention.lsa_sinks": pack_kv_u32(f"{arch}.attention.lsa_sinks", int(overlay["lsa_sinks"])),
        f"{arch}.attention.lsa_far_bias": pack_kv_f32_arr(f"{arch}.attention.lsa_far_bias", [float(x) for x in overlay["lsa_far_bias"]]),
    }

    # Serialize preserved KV fields raw byte-for-byte
    kv_blobs: List[bytes] = []
    skip_fields = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"} | set(new_kvs.keys())
    for name, field in reader.fields.items():
        if name in skip_fields:
            continue
        kv_blobs.append(b"".join(p.tobytes() for p in field.parts))
    for k, blob in new_kvs.items():
        kv_blobs.append(blob)

    # Prepare quantized / F32 bytes for replaced tensors and verify every replaced tensor exists
    src_names = {t.name for t in reader.tensors}
    missing = sorted(set(repl_specs.keys()) - src_names)
    if missing:
        raise ValueError(f"Replacement tensors missing in source GGUF: {missing[:5]}")

    encoded_repl: Dict[str, Tuple[List[int], GGMLQuantizationType, bytes]] = {}
    for name, (arr_f32, qtype) in repl_specs.items():
        # GGUF dims are reversed relative to numpy/PyTorch shape (ne0 = innermost dim)
        gguf_dims = [int(d) for d in reversed(arr_f32.shape)]
        if qtype == GGMLQuantizationType.F32:
            raw = arr_f32.astype("<f4", copy=False).tobytes()
        elif qtype == GGMLQuantizationType.Q8_0:
            q_arr = quantize(arr_f32, qtype)
            raw = q_arr.tobytes()
        else:
            raise ValueError(f"Unsupported target quant type {qtype}")
        encoded_repl[name] = (gguf_dims, qtype, raw)

    # Validate shapes and quant types of all replaced tensors against source GGUF
    for t in reader.tensors:
        if t.name in encoded_repl:
            new_dims, new_qt, _ = encoded_repl[t.name]
            old_dims = [int(d) for d in t.shape]
            if new_qt != t.tensor_type:
                raise ValueError(f"Type mismatch for {t.name}: old={t.tensor_type} new={new_qt}")
            if t.name.endswith(".attn_rel_bias.weight"):
                if len(new_dims) != len(old_dims) or new_dims[0] != old_dims[0]:
                    raise ValueError(f"rel_bias shape mismatch for {t.name}: old={old_dims} new={new_dims}")
            elif new_dims != old_dims:
                raise ValueError(f"Shape mismatch for {t.name}: old={old_dims} new={new_dims}")

    # Build tensor info table and compute new relative offsets
    tensor_entries = []
    cur_rel_off = 0
    for t in reader.tensors:
        if t.name in encoded_repl:
            dims, qtype, raw = encoded_repl[t.name]
            nbytes = len(raw)
            src_abs_off = -1
        else:
            dims = [int(d) for d in t.shape]
            qtype = t.tensor_type
            nbytes = int(t.n_bytes)
            src_abs_off = int(t.data_offset)
            raw = None
        tensor_entries.append((t.name, dims, qtype, cur_rel_off, nbytes, src_abs_off, raw))
        cur_rel_off = align_up(cur_rel_off + nbytes, alignment)

    print(
        f"[splice] arch={arch} alignment={alignment} kv_count={len(kv_blobs)} tensors={len(tensor_entries)} "
        f"replaced={len(encoded_repl)} total_data={cur_rel_off/(1024**3):.2f} GiB",
        flush=True,
    )
    if dry_run:
        print("[splice] dry-run OK", flush=True)
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".gguf.tmp")
    buf_size = 16 * 1024 * 1024

    with open(src_path, "rb") as fin, open(tmp_path, "wb", buffering=buf_size) as fout:
        # GGUF v3 header: magic "GGUF", version=3, n_tensors, n_kv
        fout.write(b"GGUF")
        fout.write(struct.pack("<IQQ", 3, len(tensor_entries), len(kv_blobs)))
        for blob in kv_blobs:
            fout.write(blob)
        for name, dims, qtype, rel_off, _nbytes, _src_off, _raw in tensor_entries:
            fout.write(pack_str(name))
            fout.write(struct.pack("<I", len(dims)))
            for d in dims:
                fout.write(struct.pack("<Q", d))
            fout.write(struct.pack("<IQ", int(qtype), rel_off))

        header_end = fout.tell()
        data_start = align_up(header_end, alignment)
        if data_start > header_end:
            fout.write(b"\x00" * (data_start - header_end))

        for name, _dims, _qtype, rel_off, nbytes, src_abs_off, raw in tensor_entries:
            expected_pos = data_start + rel_off
            cur_pos = fout.tell()
            if cur_pos < expected_pos:
                fout.write(b"\x00" * (expected_pos - cur_pos))
            elif cur_pos != expected_pos:
                raise RuntimeError(f"Offset desync at {name}: cur={cur_pos} expected={expected_pos}")

            if raw is not None:
                fout.write(raw)
            else:
                fin.seek(src_abs_off)
                rem = nbytes
                while rem > 0:
                    chunk = fin.read(min(buf_size, rem))
                    if not chunk:
                        raise EOFError(f"Unexpected EOF reading {name} from {src_path}")
                    fout.write(chunk)
                    rem -= len(chunk)

        final_end = align_up(fout.tell(), alignment)
        if final_end > fout.tell():
            fout.write(b"\x00" * (final_end - fout.tell()))

    tmp_path.replace(dst_path)
    sz_gib = dst_path.stat().st_size / (1024**3)
    print(f"[splice] wrote {dst_path} ({sz_gib:.2f} GiB) in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-gguf", type=str, default="/mnt/data/inkling2/Inkling-2-Mini-NVFP4-native-q8d.gguf")
    ap.add_argument("--overlay", type=str, required=True)
    ap.add_argument("--dst-gguf", type=str, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    splice_gguf(Path(args.src_gguf), Path(args.overlay), Path(args.dst_gguf), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
