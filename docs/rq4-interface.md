# RQ4 kernel interface (track B; for AI4 (Ampere INT4, formerly D2) and other weight codecs)

The W4A8 kernels are in `ggml/src/ggml-cuda/rq4.cuh`. That header does not depend on ggml, so the standalone bench `tools/rq4-m1` and the ggml backend (`rq4-mm.cu`) build the same code. The CPU reference is `ggml/src/ggml-rq4.h`.

## Data formats

### Weights (RQ4)
- **Rotation.** Each run of 256 weights along K is rotated by `(1/16) H256 D` before quantization. `D` is the frozen sign diagonal `RQ4_SIGN_WORDS`.
- **Super block.** One super block covers 1024 weights. It holds an fp16 scale header (`NG = 1024/GS` halves) followed by 512 nibble bytes.

  | Type | Group size | Bytes per super block | bits per weight | Type id |
  |---|---|---|---|---|
  | `RQ4_G128` | 128 | 528 | 4.125 | 64 |
  | `RQ4_G64` | 64 | 544 | 4.25 | 65 |
  | `RQ4NL_G128` | 128 | 528 | 4.125 | 62 |
  | `RQ4NL_G64` | 64 | 544 | 4.25 | 63 |

- **Codes.** For RQ4, codes are signed 4-bit, `n` in [-8, 7], and a weight is `d_group * n`. For RQ4NL, the nibble `n` (0..15) indexes the int8 table `rq4_nl_values` (the IQ4_NL table `RQ4_NL_VALUES`), and a weight is `d_group * rq4_nl_values[n]`. RQ4NL scales may be negative.
- **Nibble order (m16n8k32 s8 A-fragment order).** Inside each 128-wide chunk `c` (64 bytes per row), byte `t*16 + j*4 + b` holds:
  - high nibble: `w[128c + 32j + 4t + b]`
  - low nibble: `w[128c + 32j + 16 + 4t + b]`

  So `word & 0xF0F0F0F0` is the s8 fragment of `16*n`, and the kernel multiplies the output by 1/16.
- **Deviation from the spec (RESEARCH.md §8.B).** The spec asks for a per-row fp16 scale plus an optional u8 sub-scale. This format stores one fp16 scale per group instead.

### Activations
- **Layout.** `q` is `[N][K]` int8 and `s` is `[N][K/256]` fp32: one scale per token per 256 block, rotated the same way as the weights.
- **Byte order inside a 128 chunk (B-fragment order).** Byte `t*32 + j*8 + h*4 + b` = `x[32j + 16h + 4t + b]`.
- **Producer.** `rq4::launch_quantize_act(x, sx, q, s, K, N, stream)` takes `x` as f32 `[N][K]` with row stride `sx`.

## Kernel entry points (namespace `rq4`)
- `launch_gemm<Codec, BN, STAGES>(W, row_bytes, Xq, Xs, Y, ldy, M, N, K, stream)`
  - Computes `Y[n*ldy + m] = sum_k W[m][k] x[n][k]`.
  - Requirements: `M % 128 == 0`, `K % 1024 == 0`; any `N`.
  - `BN` is 64 or 128. `STAGES` is 2 or 3.
  - It sets the dynamic shared-memory attribute once per device.
- `launch_mmv_pair<Codec>(...)` has the same arguments and is for `N <= 8` (decode and verify); the backend default since 924bee85f. Each warp takes whole 256-wide chunk pairs, and the kernel is capped at 128 registers (4 blocks per SM). `launch_mmv<Codec>` is the older per-chunk kernel, selected with `GGML_CUDA_RQ4_MMV=1`. `GGML_CUDA_RQ4_MMV=3` selects mmv_pair without the register cap.
- `launch_gemm<Codec, BN, STAGES, WMW>`: `WMW` is the number of warps along M. The default 2 gives 2 x 4 warps with 64-row warp tiles. 4 gives 4 x 2 warps, with twice the IMMAs per A fragment; the backend selects it with `GGML_CUDA_RQ4_GEMM_WM=4`.
  - One m16n8k32 tile per warp.
  - 4 warps per 16 rows, splitting the K chunks between them.
  - Weights come straight from global memory.
- **ggml dispatch** (`ggml_cuda_mul_mat_rq4`):
  - `N <= 8` uses mmv.
  - `N >= GGML_CUDA_RQ4_BN128_MIN` (default 512) uses GEMM BN128 with 3 stages.
  - Otherwise it uses GEMM BN64 with 3 stages.

## Weight codec hook (template parameter `Codec`)
A different weight format, such as an EXL3 trellis or another codebook, plugs into the same GEMM and mmv by providing a struct with the members below. `rq4_codec_nibble<GS>` is the reference implementation. `rq4_codec_nl<GS>` is a second example: it inherits the loaders and scales and replaces only `a_regs` (a 16-entry `__byte_perm` table lookup, about 13 ALU ops per 32-bit word) and `out_mul` (1).

| Member | Contract |
|---|---|
| `GS`, `GPC` | Group size, and groups per 128-wide K chunk (`GPC = 128/GS`, 1 or 2). The kernel flushes the int32 accumulator into fp32 once per group, so `GS` must divide 128. |
| `SCB`, `SBB` | Used only by the reference codec's own loaders. Not required by the kernels. |
| `template<int BM> smem_bytes()` | Shared-memory bytes per pipeline stage for one 128-wide K chunk of BM rows. |
| `template<int BM, int NTHREADS> load(sm, W, row_bytes, m0, kc, tid)` | Issues `cp.async` (or plain stores) for rows `[m0, m0+BM)`, chunk `kc`, into stage memory `sm`. Every thread calls it; the kernel commits and waits. |
| `frag_t` | Per-lane register payload for one m16 tile (rows r and r+8) and one chunk. |
| `load_frag(sm, r, t)` | Builds `frag_t` from stage memory; `r` is the tile's first row inside the CTA tile and `t` is lane%4. |
| `load_frag_global(rlo, rhi, kc, t)` | Same as `load_frag`, but from global memory, given row pointers for rows r and r+8. Used by mmv. |
| `a_regs(frag, j, a[4])` | The four s8 A registers of IMMA `j` (the k32 step `j` in 0..3 of the chunk), in m16n8k32 row-major A-fragment order. Values must be `out_mul^-1 * code`, with `|value| <= 128`. |
| `scale(sm, BM, r, cin, g)` / `scale_global(rowptr, kc, g)` | fp32 scale of group `g` (0..GPC-1) of chunk `cin = kc & 7` (shared-memory path) or chunk `kc` (global path), for row `r`. |
| `scales_pair_global(rowptr, pp, d[2*GPC])` | The 2*GPC fp32 group scales of chunk pair `pp` (chunks 2pp and 2pp+1), with `d[c*GPC + g]` = group `g` of chunk `2pp + c`. Used by mmv_pair, the default `N <= 8` kernel. RQ4 does it with one 4- or 8-byte load per row. |
| `out_mul` | Constant factor applied to the output (1/16 for RQ4, 1 for RQ4NL). |

**Exactness.** The int32 partial sum of one group is converted with `magic_i2f`, which is exact only below 2^22. The bound is GS × max|a| × 127 < 2^22, so `|a| <= 128` at GS = 128 holds it.

**What a codec cannot change.** The activation format above is fixed. Any codec must therefore use the same rotation, 256-block activation scales, and B-fragment order.

## ggml integration
- **Types.** `GGML_TYPE_RQ4_G128` = 64, `GGML_TYPE_RQ4_G64` = 65, `GGML_TYPE_RQ4NL_G128` = 62 and `GGML_TYPE_RQ4NL_G64` = 63. On the CPU, all four are dequantized and dot-multiplied in f32 (reference only).
- **CUDA routing.** `ggml_cuda_mul_mat` returns early to `ggml_cuda_mul_mat_rq4`. RQ4 is excluded from the MMVQ fusions and the add+rms q8 producer.
- **`supports_op`** accepts only:
  - MUL_MAT with f32 activations and output;
  - 2-D weights;
  - `K % 1024 == 0` and `M % 128 == 0`;
  - contiguous tensors.
- **llama-quantize.** Use ftype `RQ4_G128` (48), `RQ4_G64` (49), `RQ4NL_G128` (50) or `RQ4NL_G64` (51).
  - It follows the IQ4_XS mixture rules and swaps every IQ4_XS tensor for RQ4 where the shape fits.
  - Token embeddings stay IQ4_XS.
  - If a tensor does not fit the RQ4 shape rules, it falls back to IQ4_XS.
  - An imatrix is optional. When given, each group scale is chosen to minimise the imatrix-weighted error in the original (unrotated) basis (`rq4_block_scales`). The codes are then rounded GPTQ-style against the rotated imatrix Hessian, and the scales are refit by least squares (`rq4_block_quant`).
  - Tables named `*embed_tokens*` also stay IQ4_XS.
