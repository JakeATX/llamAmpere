# EXL3 weights as GGUF-native types

v0.3.1 loads and serves EXL3 (trellis-coded) weights from a GGUF file, with a CUDA decode path written
for SM86. No exllamav3 runtime, no Python, no re-quantization: an existing exllamav3 checkpoint is
repacked bit for bit into a GGUF and served by `llama-server` like any other quant.

## Attribution

The EXL3 format, the "mul1" codebook arithmetic, the 16x16 tile layout and the trellis GEMV design are
Turboderp's, from [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT License, Copyright (c)
2025 Turboderp; the license text is in `licenses/LICENSE-exllamav3`). EXL3 is Turboderp's streamlined
variant of QTIP (Tseng, Sun, Hou, De Sa, "QTIP: Quantization with Trellises and Incoherence Processing",
NeurIPS 2024, arXiv:2406.11235).

The code in this fork is an independent ggml/CUDA reimplementation written with the exllamav3 sources
open as the reference. It is not clean-room and it is not a copy: a line-level audit of
`ggml/src/ggml-cuda/exl3.cu`, `exl3-gemv.cu`, `exl3.cuh`, the `ggml-quants.c` EXL3 code, the llama
loader and graph branches and the converter against the exllamav3 CUDA sources found zero verbatim or
trivially renamed functions and six places where the same algorithm is re-expressed (mul1 decode, tile
reconstruct, WHT128, glue in/out, GEMV kernel architecture, cuBLAS fallback). Shared constants are all
format-defining. Block, unroll, prefetch and k-split choices differ entirely. No other implementation
was consulted, and no other project has any claim on this format.

## What the types are

Seven ggml types, `GGML_TYPE_EXL3_2` through `GGML_TYPE_EXL3_8` (ids 51 to 57), one per bit width. A
block is one 16x16 tile, 256 weights in `32 * bits` bytes, so a tensor's byte size equals the trellis
array's byte size exactly. `ggml_exl3_bits()` returns the bit width for these types and 0 otherwise.
The ggml file type is `LLAMA_FTYPE_MOSTLY_EXL3` (47).

Tensor `ne` is `[K, N]` and the bytes are the native exllamav3 order: k-tile major, `[K/16][N/16][16*bits]`
int16 words. These types are usable only as `src0` of a `MUL_MAT`. There is no `get_rows`, so embeddings
and any non-linear tensor stay in an ordinary type (the converter writes `token_embd`, `ssm_alpha` and
`ssm_beta` as Q8_0 and the norms as F32).

A trellis tensor alone is not the weight. The effective matrix is

```
y = svh * H128( W_trellis . H128( suh * x ) )
```

where `H128` is the 128-point Walsh-Hadamard transform applied per 128-element block, and `suh` / `svh`
are per-input and per-output f16 vectors. The converter writes them next to the weight as
`<tensor>.suh` (length K) and `<tensor>.svh` (length N). The loader attaches them to the matmul node
itself: `llm_graph_context::build_lora_mm` builds the `GGML_OP_MUL_MAT` and then sets
`res->src[2] = suh` and `res->src[3] = svh`, and the backend applies the scales and both Hadamards
inside its own kernels. That keeps the whole matmul at one graph node.

GGUF metadata written by the converter: `<arch>.exl3.codebook` (only `mul1` is implemented),
`<arch>.exl3.quant_version` and `<arch>.exl3.hadamard_block` (128). The file also carries
`exl3_had128.weight`, a 64 KB f32 Sylvester Hadamard matrix used only by the generic-op A/B path below.

## Converting an exllamav3 checkpoint

`scripts/exl3/convert_exl3_to_gguf.py` repacks a checkpoint. It re-quantizes nothing: every
`<P>.trellis` becomes a GGUF tensor of the matching EXL3 type with its bytes unchanged, and every other
tensor goes through the stock converter (`conversion/qwen.py`), so tensor naming, `+1` norm folding,
GDN handling and the `mtp.*` to `blk.<n>.nextn.*` remap are identical to the fork's own GGUFs.

```
usage: convert_exl3_to_gguf.py [-h] [--outtype OUTTYPE] [--llama-dir LLAMA_DIR] [--dry-run] [--verbose]
                               [--fuse] [--fuse-test] [--layers LAYERS] [--no-fuse-mtp]
                               model_dir out
```

- `--outtype` (default `q8_0`) is the type for the non-EXL3 2-D tensors (`token_embd`, `in_proj_a/b`).
  `f16`, `bf16` and `f32` are also accepted.
- `--llama-dir` (default: the repo the script lives in) is the llama.cpp checkout whose `conversion/`
  and `gguf-py/` are imported. Only set it to run the script from outside its own tree.
- `--fuse` writes the fused-group layout described below.
- `--layers 0,3` limits fusing to those block indices; `--no-fuse-mtp` leaves the MTP/nextn block
  unfused.
- `--dry-run` walks the tensors without writing the file.
- `--fuse-test` exists to exercise the fused kernels on checkpoints whose group members do not share
  `suh`. It produces a numerically wrong model. Never ship a GGUF built with it.

The stock converter permutes GDN V heads from grouped to tiled order. For EXL3 tensors the same
permutation is applied at 128-feature granularity, which is exact because the effective weight is
`diag(suh) . H128_blockdiag . C . H128_blockdiag . diag(svh)`: permuting whole 128-blocks of features
permutes whole 8-tile groups of the trellis and the matching 128 entries of `suh` or `svh`. The script
derives the permutation from the stock rule and asserts it is 128-block-constant, so drift in the stock
converter fails loudly instead of corrupting weights.

Example (Qwen3.8-27B, 4.0 bpw):

```bash
python3 scripts/exl3/convert_exl3_to_gguf.py \
  /path/to/Qwen3.8-27B-EXL3-4.0bpw models/Qwen3.8-27B-EXL3-4.0bpw.gguf --fuse
```

The unfused 4.0 bpw file is 14,741,371,648 bytes, 1,684 tensors: 409 EXL3 (`output.weight` EXL3_6 at
K=5120 N=248320, every other linear EXL3_4), 818 `.suh`/`.svh` f16 side vectors, the rest Q8_0 and F32.
Conversion takes under a minute because the trellis is copied lazily and never cast to float.

## Verifying a converted file

`scripts/exl3/verify_exl3_gguf.py` reconstructs the effective weight from the GGUF (EXL3 bytes plus
`.suh` and `.svh`) with the numpy reference decoder and compares it, bit for bit as float32, against the
weight reconstructed from the original safetensors with the stock V-head permutation applied.

```bash
python3 scripts/exl3/verify_exl3_gguf.py models/Qwen3.8-27B-EXL3-4.0bpw.gguf /path/to/Qwen3.8-27B-EXL3-4.0bpw
```

With no further arguments it checks five tensors that cover all four permutation kinds (a GDN gate, a
GDN output projection, a fused QKV, a full-attention K and an MTP projection). Extra tensors are given
as `gguf_name=hf_prefix[:kind]` with `kind` in `none`, `qkv_out`, `z_out`, `out_proj_in`. It is CPU only
and prints `VERIFY PASS` or `VERIFY FAIL`.

`scripts/exl3/exl3_ref.py` is the numpy reference decoder those checks are built on. It reads a
safetensors checkpoint directly and can dump the reconstructed matrix (`--save W.npy`) or the raw
codebook matrix (`--raw C.npy`). It implements the `mul1` codebook only.

On the Qwen3.8-27B 4.0 bpw conversion, all five default tensors reconstruct with `max|Δ| = 0`.

## Running

Same flags as any other model in [QWEN_AMPERE.md](../QWEN_AMPERE.md#build-and-run), with the EXL3 GGUF
in place of the ATX file:

```bash
./build-sm86/bin/llama-server -m models/Qwen3.8-27B-EXL3-4.0bpw.gguf \
  -c 32768 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 \
  --parallel 1 --jinja --fit off \
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
```

The MTP flags are the ones the measurements below used: `--spec-draft-n-max 4 --spec-draft-p-min 0`, the
model's own MTP head, exact p/q verification on by default. `--spec-draft-vocab-map` is not used with
this file: the shortlist in `docs/mtp-vocab/` was built for the ATX quant's head. Raise `-c` for longer
contexts; the decode flags do not change.

Environment switches, all off by default:

- `GGML_CUDA_EXL3_GEMV=0` forces every matmul onto the reconstruct-plus-cuBLAS path. Debug only.
- `GGML_CUDA_EXL3_FUSED=1` runs the cooperative kernel that does the Hadamard prologue and epilogue
  in-kernel behind two grid barriers. It was measured slower than the separate glue kernels on a 1008-block
  persistent grid, so it is opt-in. `EXL3_GEMV_BPS=n` caps that grid at n blocks per SM.
- `LLAMA_EXL3_GLUE_GRAPH` builds the glue from generic ops instead of attaching `suh`/`svh` to the
  matmul node: `ggml_mul` by `suh`, a `mul_mat` against `exl3_had128.weight`, the weight matmul, a second
  Hadamard matmul, `ggml_mul` by `svh`. It exists as an A/B check against the fused form. The code tests
  for the variable's presence, so `LLAMA_EXL3_GLUE_GRAPH=1` and any other value both enable it; unset it
  to get the default path.

## The fused-group layout

exllamav3 quantizes Q/K/V, `in_proj_qkv`/`in_proj_z` and gate/up as one group each, so every member of a
group shares `suh`. `--fuse` exploits that: the members are written as one wider EXL3 tensor, and the
graph runs one GEMV per group instead of two or three. Fewer launches, one `glue_in` per group, one read
of the activation.

| GGUF tensor | fused contents | absent tensors |
|---|---|---|
| `blk.N.attn_qkv.weight` (GDN layer) | `[in_proj_qkv \| in_proj_z]` | `attn_gate.*` |
| `blk.N.attn_qkv.weight` (full-attention layer) | `[attn_q \| attn_k \| attn_v]` | `attn_q/k/v.*` |
| `blk.N.ffn_up.weight` (every layer, MTP block too) | `[ffn_gate \| ffn_up]` | `ffn_gate.*` |

`.suh` is shared and the converter asserts it is byte-identical across members. `.svh` is the members'
vectors concatenated in member order. The trellis tiles are concatenated along the n-tile axis per
k-tile row, which is exactly the tile order a fused `[K, sum(N_m)]` tensor needs. A group whose members
disagree on bit width or on `suh` is written unfused with a warning.

The loader reads the fused width from the tensor metadata, so one binary serves both layouts and nothing
is guessed: `src/models/qwen35.cpp` detects the wider `attn_qkv` and leaves `attn_gate` null, detects
`ffn_up` at `2 * n_ff` and makes `ffn_gate` not required, then splits the outputs with views and runs
SwiGLU over the halves. A non-fused GGUF takes the same code paths it always did.

## Kernels, and what is not done yet

Decode and small verify widths run a trellis-direct GEMV in `ggml/src/ggml-cuda/exl3-gemv.cu`. Persistent
blocks of four warps walk (n-tile, k-split) work items. At 4 bits and under, a 16x16 tile is at most 32
words, so every lane loads one word (one coalesced 128 B request per warp per tile) and the second word
covering its eight bit windows comes from a shuffle; at 5 to 8 bits each lane loads its three words
directly. The eight codes decode through the mul1 codebook into four half2 row pairs. At width 1 those
are FMAed against the activation in half2. At widths 2 to 16 the tile's lane order is already the B
fragment layout of `mma.sync.m16n8k16`, so the pairs feed two tensor-core instructions with the
activation rows as the A operand and an fp16 accumulator folded to fp32 every four tiles, which makes the
per-tile cost independent of the width. Summation order is fixed, so the kernel is deterministic.

Everything wider than 16 activation columns, which in practice means prefill, goes to the M1 path:
reconstruct the whole `[K, N]` weight to f16 in pool memory, run one `cublasGemmEx`, with the `suh`
Hadamard applied by `glue_in` before and the `svh` Hadamard by `glue_out` after.

Known limits:

- **Prefill is the slow half.** Reconstructing every weight per matmul is 2.8x slower than the ATX
  IQ4_XS path. A trellis-direct GEMM for T > 16 is v0.4 work.
- **Widths 9 to 16 are compiled and parity-checked but unexercised in production.** At MTP depth 4 the
  graph only ever issues widths 1, 2 and 5.
- **No `test-backend-ops` coverage.** EXL3 has no case in the suite; the gates are the numpy verifier,
  the standalone parity tool and the KL run below.
- **No MoE.** `supports_op` accepts 2-D weights with contiguous f32 `src1`/`dst` only, and
  `GGML_OP_MUL_MAT_ID` already uses `src[2]` for its ids, so EXL3 experts would collide with the side
  tensors.
- **The side tensors ride on `src[2]`/`src[3]` of a `MUL_MAT` node.** `supports_op` does not inspect
  them, and graph reuse and the allocator were not extended for the extra sources. A backend that falls
  back for some other reason would silently drop the scale and Hadamard glue.
- **`mul1` is the only codebook implemented.** exllamav3's `mcg` and the older `cb0` are not.

Deferred to v0.4 with the rest of the release backlog: the prefill GEMM, `test-backend-ops` cases, and
the width 5 to 9 MMVQ tuning that the n-gram speculative path is waiting on.

## Measured

One RTX 3090 Ti at 350 W, whole-card VRAM budget 23,552 MiB, Qwen3.8-27B EXL3 4.0 bpw
(`output.weight` at 6 bits), CUDA build at `-DCMAKE_CUDA_ARCHITECTURES=86`.

**Correctness.** Converter verify: `max|Δ| = 0` on all five default tensors. Kernel parity against the
numpy reference (tolerance 3e-3 relative RMS): PASS at T = 1, 2, 5, 7, 8, 9, 12, 16, worst 4.9e-4. This
is a numerical parity gate against the same weights decoded a second way, not a quality ranking: over
16 x 2048 wikitext chunks (16,368 scored tokens, q8_0 KV) our port's perplexity is 6.0971 ± 0.115 and
the KL divergence between the reference implementation and our port is 0.000221 ± 0.000057, median
2.5e-5, same top token on 99.609 ± 0.049 % of positions. That residual is 124x smaller than the distance
between two different quantizations of this model, and a f16-KV control attributes it to activation and
KV precision rather than to the decode path.

**Single-token decode.** `llama-completion -ngl 99 -fa on -c 2048 -n 128 --temp 0`, prompt
"The capital of France is", greedy, default environment:

| path | ms/token | tok/s |
|---|---:|---:|
| reconstruct + cuBLAS (M1) | 49.8 | 20.1 |
| trellis GEMV, split glue | 25.4 | 39.3 |
| shipped (tensor-core widths, four-row `glue_out` reduction) | 24.2 | 41.3 |

Output text is byte-identical across all three. Peak whole-card VRAM at width 1: 14,783 MiB.

**MTP smoke.** 512 generated tokens, greedy, one coding prompt, `-c 32768 --spec-draft-n-max 4
--spec-draft-p-min 0`, verify width 5. This is a smoke test, not a ship measurement: single seed, well
under the 5,000-token floor.

| build | tok/s | draft / accepted | prompt tok/s (62 tok) | peak VRAM |
|---|---:|---|---:|---:|
| trellis GEMV, split glue | 56.7 | 663 / 344 | 127 | 16.4 GB |
| shipped | 76.1 | 649 / 348 | 119 | 16,459 MiB |

**Depth benchmark.** Three seeds (7300, 7301, 7302), temperature 1.0, `n_predict` 20,480, EOS honoured,
MTP depth 4, one fresh server boot per measured request, prompts templated from real fork source:

| prompt depth | seeds | mean tok/s | sd | peak VRAM | stop |
|---|---|---:|---:|---:|---|
| 20,469 tokens | 3 | 81.27 | 0.39 | 17,080 MiB | natural (EOS) on all 3, 15,283 to 19,423 generated |
| 51,196 tokens | 3 | 73.72 | 0.59 | 17,893 MiB | cap (20,480) on all 3 |

Acceptance is flat across depth (cell means 0.608 to 0.642, about 3.5 accepted draft tokens per pass),
so the drop from 20K to 50K is per-step engine overhead, not drafting.

Re-measured on the release build (`build-v031`, same fixture, seeds and cap): the 20,469-token row gives
81.90 tok/s (sd 0.82, peak 17,154 MiB, acceptance 0.61 to 0.63) with MTP depth 4, and 40.56 tok/s (sd 0.15,
peak 16,030 MiB) with no drafter, all six runs ending on EOS between 17,188 and 20,084 generated tokens.
Per-seed files: `EX5_depth_bench/results_v031/A.20k.s*.json` and `A1.20k.s*.json`.

## How it compares to our other formats

On the same card and the same binary, the ATX IQ4_XS quant is still the fast choice for this model. In
the MTP smoke above, ATX-4-XS runs 101.1 tok/s against EXL3's 76.1 on the same prompt with the same
draft settings, and 359 prompt tok/s against 119. The gap is almost entirely prefill and verify-width
compute: the trellis GEMV reads a smaller file (13.7 GiB against 14.5 GiB for ATX at 4.56 bpw) but it
decodes 256 codewords per tile through a multiply chain, where IQ4_XS reaches roughly 950 GB/s on the
tuned MMVQ path. EXL3 pays for that with quality per byte, which is what the format is for.

Against the ternary containers documented in [bonsai2.md](bonsai2.md), size is the whole story and the
speed cells are not directly comparable: PTQ1_0 (1.75 bpw, 5.54 GiB) and PQ2_0 (2.125 bpw, 6.71 GiB)
are half the size of this file and were timed on a different model on a 16K fixture, at 55 and 65 tok/s
single-token and 60 and 100 tok/s with an MTP drafter. Take those as an order of magnitude, not as a
ladder against the EXL3 figures above.

Pick EXL3 when you have an exllamav3 checkpoint you want to serve with this stack, or when you want
4.0 bpw weights at 13.7 GiB. Pick ATX for maximum tokens per second on this card. Pick a ternary
container when the card has to hold something else as well.
