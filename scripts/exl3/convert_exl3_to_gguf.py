#!/usr/bin/env python3
# EXL3 format and trellis layout: Turboderp, exllamav3 (https://github.com/turboderp-org/exllamav3), MIT License,
# Copyright (c) 2025 Turboderp; see licenses/LICENSE-exllamav3. Independent reimplementation.
"""EX2 M1: bit-exact repack of an exllamav3 EXL3 checkpoint (Qwen3.5/3.8 text model) into GGUF.

No re-quantization: every `<P>.trellis` (int16 [K/16, N/16, 16*bits]) becomes GGUF tensor `<gguf(P)>.weight` of type
GGML_TYPE_EXL3_{bits} with logical shape [N, K] (ggml ne = [K, N]) and raw bytes = the trellis array in its native
exllamav3 order (kt-major, then nt, then 16*bits int16 words). `<P>.suh` (f16 [K]) and `<P>.svh` (f16 [N]) are written as
`<gguf(P)>.suh` / `<gguf(P)>.svh`. `<P>.mul1` is consumed (asserted) and recorded once as KV `<arch>.exl3.codebook = mul1`.
Everything else (embeddings, norms, GDN small tensors, biases, mtp.* non-linear tensors) goes through the stock converter
(conversion/qwen.py Qwen3_5TextModel), so tensor naming, +1 norm folding, A_log/dt_bias/conv1d handling and the mtp.* →
blk.<n>.nextn.* remap are exactly what the fork's own GGUFs use.

The stock converter permutes GDN V heads from grouped to tiled order (in_proj_qkv V rows, in_proj_z rows, out_proj input
columns, and the small per-head tensors). For EXL3 tensors the same permutation is applied at 128-feature granularity:
the effective weight is diag(suh) · H128_blockdiag · C · H128_blockdiag · diag(svh), so permuting whole 128-blocks of
output features (or input features) is exactly permuting the corresponding 8-tile groups of C and the matching 128
entries of svh (or suh). The permutation is derived from the converter's own _reorder_v_heads on an index vector and
asserted to be 128-block-constant, so any drift in the stock rule fails loudly instead of silently corrupting weights.

--fuse writes every shared-input group of EXL3 linears as ONE wider EXL3 tensor, so llama.cpp runs one matmul instead of
two or three. Groups (all members read the same activation, so they share `suh` and differ only in output columns):
  GDN layers       blk.N.attn_qkv = [in_proj_qkv | in_proj_z]        (attn_gate.* is then not written)
  full-attn layers blk.N.attn_qkv = [attn_q | attn_k | attn_v]       (attn_q/k/v.* are then not written)
  every layer      blk.N.ffn_up   = [ffn_gate | ffn_up]              (ffn_gate.* is then not written)
The fused tensor gets ONE `.suh` (asserted byte-identical across the members) and `.svh` = the members' svh concatenated in
member order; the trellis is concatenated on the n-tile axis, i.e. for each k-tile row the members' n-tile rows in member
order (each member reshaped to [K/16, N_m/16, 16*bits] and concatenated on axis 1) -- exactly the tile order the fused
[K, sum(N_m)] tensor needs. A group whose members disagree on bit width or on suh is written unfused with a warning.

usage: convert_exl3_to_gguf.py MODEL_DIR OUT.gguf [--outtype q8_0|f16|bf16] [--llama-dir DIR] [--dry-run]
                               [--fuse] [--fuse-test] [--layers 0,3] [--no-fuse-mtp]
"""
import argparse, logging, os, re, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out")
ap.add_argument("--outtype", default="q8_0", help="type for the non-EXL3 2-D tensors (token_embd, in_proj_a/b): q8_0|f16|bf16|f32")
ap.add_argument("--llama-dir", default=str(Path(__file__).resolve().parents[2]), help="llama.cpp checkout whose conversion/ and gguf-py/ to use (default: this repo)")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--fuse", action="store_true",
                help="write shared-input EXL3 groups as one wider tensor: GDN [attn_qkv|attn_gate] -> attn_qkv, "
                     "full-attn [attn_q|attn_k|attn_v] -> attn_qkv, [ffn_gate|ffn_up] -> ffn_up. The members' suh must "
                     "be byte-identical; a group that fails that (or mixes bit widths) is written unfused with a warning.")
ap.add_argument("--fuse-test", action="store_true",
                help="TEST ONLY -- PRODUCES A NUMERICALLY WRONG MODEL. Like --fuse, but fuses groups whose members do "
                     "NOT share suh, keeping the FIRST member's suh: every later member's outputs are then wrong. "
                     "Exists only to exercise the fused trellis layout / kernels on checkpoints with per-member suh. "
                     "Combine with --layers to keep the test file cheap. Never ship a GGUF built with this flag.")
ap.add_argument("--layers", default=None,
                help="comma-separated block indices to fuse (default: all); every other tensor is written unfused")
ap.add_argument("--no-fuse-mtp", action="store_true",
                help="leave the MTP/nextn block (the mtp.* tensors, written as the last blk.N) unfused")
args = ap.parse_args()
sys.path.insert(0, args.llama_dir); sys.path.insert(0, os.path.join(args.llama_dir, "gguf-py"))
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
logger = logging.getLogger("convert_exl3")
import numpy as np, torch, gguf
from conversion.base import ModelBase, LazyTorchTensor
from conversion.qwen import Qwen3_5TextModel

FTYPES = {"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "f16": gguf.LlamaFileType.MOSTLY_F16,
          "bf16": gguf.LlamaFileType.MOSTLY_BF16, "f32": gguf.LlamaFileType.ALL_F32}
EXL3_TYPES = {b: getattr(gguf.GGMLQuantizationType, f"EXL3_{b}") for b in range(2, 9)}
HAD = 128   # Hadamard block on both sides; permutations must be block-constant at this size
TILE = 16

# (fused gguf tensor short name, member short names in concatenation order). All members of a group read the same
# activation, so the fused tensor is a pure output-column concatenation: one suh, svh/trellis concatenated on N.
FUSE_GROUPS = (
    ("attn_qkv", ("attn_qkv", "attn_gate")),        # GDN layers: in_proj_qkv | in_proj_z
    ("attn_qkv", ("attn_q", "attn_k", "attn_v")),   # full-attention layers (attn_q already carries the output gate)
    ("ffn_up",   ("ffn_gate", "ffn_up")),           # MLP: gate | up
)
FUSE = args.fuse or args.fuse_test


class Exl3Qwen35Model(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    def _block_perm(self, n_feat: int, kind: str) -> np.ndarray | None:
        """Feature permutation the stock converter applies along one dim (new[j] = old[perm[j]]), or None."""
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)
        if not (num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads):
            return None
        head_k_dim = self.hparams["linear_key_head_dim"]; head_v_dim = self.hparams["linear_value_head_dim"]
        num_v_per_k = num_v_heads // num_k_heads
        idx = torch.arange(n_feat, dtype=torch.long)
        if kind == "qkv_out":      # in_proj_qkv rows: q, k untouched, v reordered
            q_dim = k_dim = head_k_dim * num_k_heads
            v = self._reorder_v_heads(idx[q_dim + k_dim:], 0, num_k_heads, num_v_per_k, head_v_dim)
            perm = torch.cat([idx[:q_dim + k_dim], v])
        elif kind in ("z_out", "out_proj_in"):   # in_proj_z rows / out_proj input columns: all V heads
            perm = self._reorder_v_heads(idx, 0, num_k_heads, num_v_per_k, head_v_dim)
        else:
            return None
        perm = perm.numpy()
        if np.array_equal(perm, idx.numpy()):
            return None
        assert n_feat % HAD == 0
        blk = perm.reshape(-1, HAD)
        assert np.array_equal(blk, blk[:, :1] + np.arange(HAD)[None, :]) and np.all(blk[:, 0] % HAD == 0), \
            f"{kind}: stock V-head permutation is not {HAD}-block-constant; EXL3 repack cannot apply it bit-exactly"
        return perm

    def _exl3_kind(self, prefix: str):
        if "linear_attn." not in prefix:
            return None
        if prefix.endswith(".in_proj_qkv"): return "qkv_out"
        if prefix.endswith(".in_proj_z"):   return "z_out"
        if prefix.endswith(".out_proj"):    return "out_proj_in"
        return None

    def _eager(self, name):
        return LazyTorchTensor.to_eager(self.model_tensors.pop(name)())

    def dequant_model(self):
        # EXL3 tensors are repacked in prepare_tensors() before the stock loop; nothing to dequantize.
        return

    def _prep_exl3(self, prefix: str, force_eager: bool = False):
        """Pop one EXL3 linear's tensors, check them and apply the stock V-head permutation.
        Returns (trellis torch int16 [K/16, N/16, 16*bits], suh np.float16 [K], svh np.float16 [N], bits, K, N).
        force_eager=False keeps the trellis lazy (materialized by the writer); fusion needs it eager to concatenate."""
        tr = self.model_tensors.pop(prefix + ".trellis")()                      # lazy torch tensor, materialized at write time
        if force_eager:
            tr = LazyTorchTensor.to_eager(tr)
        suh = self._eager(prefix + ".suh"); svh = self._eager(prefix + ".svh")
        assert tr.dtype == torch.int16 and suh.dtype == torch.float16 and svh.dtype == torch.float16, (prefix, tr.dtype, suh.dtype, svh.dtype)
        mul1 = self._eager(prefix + ".mul1") if prefix + ".mul1" in self.model_tensors else None
        MUL1 = 0x83DCD12D    # the .mul1 tensor stores the codebook multiplier itself (int32 wrap of 0x83DCD12D)
        assert mul1 is not None and (int(mul1.reshape(-1)[0]) & 0xFFFFFFFF) == MUL1, f"{prefix}: expected mul1 codebook multiplier {MUL1:#x} (got {mul1})"
        assert prefix + ".mcg" not in self.model_tensors, f"{prefix}: mcg codebook not supported"
        kt, nt, w = tr.shape; bits = w // TILE
        assert w == TILE * bits and bits in EXL3_TYPES, (prefix, tr.shape)
        K, N = kt * TILE, nt * TILE
        assert suh.shape == (K,) and svh.shape == (N,), (prefix, tr.shape, suh.shape, svh.shape)
        suh = suh.numpy(); svh = svh.numpy()
        kind = self._exl3_kind(prefix)
        if kind in ("qkv_out", "z_out"):
            perm = self._block_perm(N, kind)
            if perm is not None:
                tile_perm = torch.from_numpy(perm.reshape(-1, TILE)[:, 0] // TILE)   # new tile column t <- old tile column tile_perm[t]
                tr = torch.index_select(tr, 1, tile_perm); svh = svh[perm]
                logger.info(f"{prefix}: applied V-head output permutation ({N // HAD} blocks of {HAD})")
        elif kind == "out_proj_in":
            perm = self._block_perm(K, kind)
            if perm is not None:
                tile_perm = torch.from_numpy(perm.reshape(-1, TILE)[:, 0] // TILE)
                tr = torch.index_select(tr, 0, tile_perm); suh = suh[perm]
                logger.info(f"{prefix}: applied V-head input permutation ({K // HAD} blocks of {HAD})")
        return tr, suh, svh, bits, K, N

    def _emit_exl3(self, new_name: str, tr, suh, svh, bits: int, K: int, N: int, note: str = ""):
        raw = tr.contiguous().view(torch.uint8).reshape(N, K * bits // 8).numpy()   # byte-shape label only; bytes stay in trellis order
        self.gguf_writer.add_tensor(new_name, raw, raw_dtype=EXL3_TYPES[bits])
        # side vectors as F32 (exact upcast of the fp16 values): llama applies them with ggml_mul on f32 activations
        self.gguf_writer.add_tensor(new_name.replace(".weight", ".suh"), np.ascontiguousarray(suh.astype(np.float16).astype(np.float32)))
        self.gguf_writer.add_tensor(new_name.replace(".weight", ".svh"), np.ascontiguousarray(svh.astype(np.float16).astype(np.float32)))
        self._exl3_stats[bits] = self._exl3_stats.get(bits, 0) + 1
        logger.info(f"{new_name:40s} EXL3_{bits}  [K={K}, N={N}]  ({K * N * bits / 8 / 2**20:.1f} MiB) + suh[{K}] svh[{N}]{note}")

    def _write_exl3(self, prefix: str):
        tr, suh, svh, bits, K, N = self._prep_exl3(prefix)
        self._emit_exl3(str(self.map_tensor_name(prefix + ".weight")), tr, suh, svh, bits, K, N)

    def _write_exl3_fused(self, fused_name: str, prefixes: list):
        """Write one wider EXL3 tensor for a group of linears that share their input (and therefore their suh)."""
        parts = [(p,) + self._prep_exl3(p, force_eager=True) for p in prefixes]
        members = ", ".join(str(self.map_tensor_name(p + ".weight")) for p, *_ in parts)
        suh0 = parts[0][2]
        reason = None
        if len({pt[4] for pt in parts}) != 1:
            reason = f"bit widths differ ({sorted({pt[4] for pt in parts})})"
        elif len({pt[5] for pt in parts}) != 1:
            reason = f"input dims differ ({sorted({pt[5] for pt in parts})})"
        elif any(pt[2].tobytes() != suh0.tobytes() for pt in parts[1:]):
            if args.fuse_test:
                logger.warning(f"{fused_name}: --fuse-test: members do NOT share suh; keeping {parts[0][0]}'s suh -- "
                               f"the outputs of every later member are NUMERICALLY WRONG (layout test only)")
            else:
                reason = "members do not share suh (per-member input scales)"
        if reason is not None:
            logger.warning(f"{fused_name}: NOT fusing -- {reason}; writing {len(parts)} tensors unfused ({members})")
            for p, tr, suh, svh, bits, K, N in parts:
                self._emit_exl3(str(self.map_tensor_name(p + ".weight")), tr, suh, svh, bits, K, N)
            self._fuse_stats["unfused"] += 1
            return
        bits, K = parts[0][4], parts[0][5]
        tr = torch.cat([pt[1] for pt in parts], dim=1)                 # n-tile axis: per k-tile row, members' n-tile rows in order
        svh = np.concatenate([pt[3] for pt in parts])
        N = sum(pt[6] for pt in parts)
        assert tuple(tr.shape) == (K // TILE, N // TILE, TILE * bits), (fused_name, tuple(tr.shape), K, N, bits)
        assert svh.shape == (N,), (fused_name, svh.shape, N)
        self._emit_exl3(fused_name, tr, suh0, svh, bits, K, N, note=f"  FUSED <- {members}")
        self._fuse_stats["fused"] += 1

    def _fuse_plan(self, prefixes: list) -> dict:
        """prefix -> ("solo",) | ("head", fused_gguf_name, [member prefixes]) | ("member",)."""
        plan = {p: ("solo",) for p in prefixes}
        if not FUSE:
            return plan
        only = None if not args.layers else {int(x) for x in args.layers.replace(" ", "").split(",") if x}
        # the stock converter renames mtp.layers.0.* to model.layers.<n_layer>.*, i.e. the MTP block is any blk index
        # at or beyond num_hidden_layers (blk.64 here); nothing in the prefix still says "mtp" at this point.
        n_layer = int(self.find_hparam(["num_hidden_layers", "n_layers", "n_layer"]))
        blocks = {}
        for p in prefixes:
            m = re.fullmatch(r"blk\.(\d+)\.(.+)\.weight", str(self.map_tensor_name(p + ".weight")))
            if not m:
                continue
            blocks.setdefault(int(m.group(1)), {})[m.group(2)] = p
        for bi in sorted(blocks):
            if only is not None and bi not in only:
                continue
            if args.no_fuse_mtp and bi >= n_layer:
                logger.info(f"blk.{bi}: MTP/nextn block (>= num_hidden_layers={n_layer}), not fused (--no-fuse-mtp)")
                continue
            for out_short, shorts in FUSE_GROUPS:
                if not all(s in blocks[bi] for s in shorts):
                    continue
                mp = [blocks[bi][s] for s in shorts]
                head = min(mp)                      # earliest in the sorted() write order, so file order stays deterministic
                plan[head] = ("head", f"blk.{bi}.{out_short}.weight", mp)
                for p in mp:
                    if p != head:
                        plan[p] = ("member",)
        return plan

    def prepare_tensors(self):
        self._exl3_stats = {}
        self._fuse_stats = {"fused": 0, "unfused": 0}
        prefixes = sorted({n[:-len(".trellis")] for n in self.model_tensors if n.endswith(".trellis")})
        logger.info(f"EXL3 linear tensors: {len(prefixes)}")
        if args.fuse_test:
            logger.warning("--fuse-test is a LAYOUT TEST FLAG: the resulting GGUF is NOT a valid model")
        plan = self._fuse_plan(prefixes)
        for p in prefixes:
            act = plan[p]
            if act[0] == "solo":
                self._write_exl3(p)
            elif act[0] == "head":
                self._write_exl3_fused(act[1], act[2])
            # "member": already written as part of its group
        leftovers = [n for n in self.model_tensors if n.endswith((".trellis", ".suh", ".svh", ".mul1", ".mcg"))]
        assert not leftovers, leftovers
        qc = self.hparams.get("quantization_config", {})
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        self.gguf_writer.add_string(f"{arch}.exl3.codebook", str(qc.get("codebook", "mul1")))
        self.gguf_writer.add_string(f"{arch}.exl3.quant_version", str(qc.get("version", "")))
        self.gguf_writer.add_uint32(f"{arch}.exl3.hadamard_block", HAD)
        # shared 128-block Hadamard: Sylvester H[i,j] = (-1)^popcount(i&j) / sqrt(128) == exllamav3 get_hadamard(128)
        idx = np.arange(HAD)
        pc = np.array([bin(v).count("1") for v in range(HAD)], dtype=np.int64)
        H = ((-1.0) ** pc[idx[:, None] & idx[None, :]]).astype(np.float32) / np.float32(np.sqrt(HAD))
        self.gguf_writer.add_tensor("exl3_had128.weight", np.ascontiguousarray(H, dtype=np.float32))
        super().prepare_tensors()
        logger.info(f"EXL3 tensors by bits: {self._exl3_stats}")
        if FUSE:
            logger.info(f"fusion: {self._fuse_stats['fused']} groups fused, {self._fuse_stats['unfused']} groups left unfused")


def main():
    dir_model = Path(args.model_dir); out = Path(args.out)
    hparams = ModelBase.load_hparams(dir_model, False)
    qc = hparams.get("quantization_config") or {}
    assert qc.get("quant_method") == "exl3", f"not an EXL3 checkpoint: quantization_config={qc}"
    hparams["quantization_config"] = dict(qc, quant_method="exl3")     # keep the stock NVFP4/MXFP4 detection off
    with torch.inference_mode():
        m = Exl3Qwen35Model(dir_model, FTYPES[args.outtype], out, hparams=hparams, dry_run=args.dry_run, eager=False)
        m.write()
    logger.info(f"wrote {out}")

if __name__ == "__main__":
    main()
