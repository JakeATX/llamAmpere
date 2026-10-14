# AI+ weights (AI4 = Ampere INT4)

AI+ is the family name of these trellis-coded weight formats (Ampere and newer). AI4 / AI3 / AI2 name the rates
(bits per weight per trellis step); only the 4-bit rate is implemented. Two layouts exist:
`GGML_TYPE_AI4` (70, the original 1024-wide layout for dense models) and `GGML_TYPE_AI4_U` (71, "ai4u", a 128-wide
unit layout that also covers MoE expert stacks, see [AI4_U](#ai4_u-unit-128-layout-and-moe)).

AI4 is a 4.125 bits-per-weight trellis-coded weight format with an int8 tensor-core (IMMA) decode path written for
SM86 (RTX 30-series). It replaces the IQ4_XS tensors of an IQ4_XS quant and keeps every other tensor as it was.
It was called D2 during development. Its GGML type id (70) and its bytes did not change in the rename, so files
written as D2 load as AI4.

## Measured results

All numbers are from Qwen3.8-27B. Speed was measured on an RTX 3090 at a 350 W power limit.

In short: **EXL3-class quality, faster at long context with MTP.** AI4 has clearly lower KL than IQ4_XS at a similar
size. It does not beat EXL3 at equal size: EXL3 encoded at AI4's byte budget has lower KL than AI4.

Sizes are GPU weight bytes (every tensor except `token_embd`, which llama.cpp keeps on the host) unless marked as a
file size. The IQ4_XS template is the IQ4_XS quant AI4 was made from (same imatrix); ATX-4-XS is the published
IQ4_XS-mix quant. KL is vs BF16 on the generated region of 3 texts, pooled, with a paired 95% CI; a ratio below 1
means AI4 is closer to BF16.

### Quality (KL vs BF16)

| AI4 (13,648 MiB GPU weights) vs | its size | KL ratio AI4 / reference [95% CI] |
|---|---|---|
| IQ4_XS template (D-006 D2_FAST vs D-005) | 14,633 MiB file (AI4 file: 14,303 MiB) | 0.645x [0.596, 0.693] |
| ATX-4-XS, the shipped IQ4_XS quant | 14,174 MiB (+3.9% vs AI4) | 0.725x [0.689, 0.764] |
| EXL3 4.0 bpw (turboderp), our port | 12,773 MiB (AI4 is +6.9%) | 0.907x [0.863, 0.952] |
| EXL3 SC 4.0 bpw H5 (turboderp), exllamav3 | 12,416 MiB (AI4 is +9.9%) | 1.046x [0.974, 1.114] (tie) |
| EXL3 4.27 bpw, encoded at AI4's byte budget (body 12,559 vs AI4 12,572.5 MiB), exllamav3 | 13,349 MiB (AI4 is +2.2%) | 1.237x [1.173, 1.304] (EXL3 better) |

The 0.907x row compares AI4 with a smaller file: AI4 keeps `attn_qkv`/`attn_v` at Q5_K and the output head at Q6_K
from its template. Giving EXL3 the same +6.75% body bits takes its KL to 0.718x of EXL3 4.0, below AI4. At equal body
bytes EXL3's KL is 1.237x lower than AI4's (median 1.222x, p99 1.107x).
Source: EXL3_VS_OURS_20261011 (rental 3090, same box for every arm).

### Speed

| measurement | AI4 | reference |
|---|---|---|
| decode, 100K ship corpus, MTP adaptive 3/4, tq5_0/turbo4 KV, 3 seeds (D-008) | +7.52% G (SEM 1.65%) | IQ4_XS template, 14,633 MiB file |
| prefill at 100K, same run (D-008) | +4.74% G (SEM 0.21%) | IQ4_XS template |
| plain decode (width 1, no drafter) at 100K (D-007) | -2.0% | IQ4_XS template |
| plain decode (width 1, no drafter) at 4K (D-007) | +2.0% | IQ4_XS template |
| decode at 100K, each engine's best settings (rental, 1 seed, indication only) | 1.173x [1.024, 1.343] | exllamav3 EXL3 4.0 bpw (12,568 MiB), MTP, fp16 cache |
| same, both with f16 KV and MTP | 1.050x [0.917, 1.202] (not significant) | exllamav3 EXL3 4.0 bpw |

Most of the speed gain is at verify widths 2-8, where the IMMA kernel runs well. Width-1 decode at long
context is slightly slower than IQ4_XS. With MTP drafting, the per-round cost was 3.9% lower (43.60 vs 45.42 ms)
and acceptance was higher (0.697 vs 0.674), because the weights are closer to BF16. Against exllamav3, most of the
100K lead comes from the compressed tq5_0/turbo4 KV cache and our MTP path; with no drafting exllamav3 is 6% faster at
100K, while our prefill is 1.28x-1.37x faster.

## What the type is

`GGML_TYPE_AI4` = 70. A block is 1024 weights in 528 bytes. It is a whole-tensor type like EXL3: rows come in slabs
of 16, so the type is not row-addressable (`to_float` is NULL). A tensor must be 2-D with `ne0 % 1024 == 0` and
`ne1 % 16 == 0`. Embeddings (`token_embd`) and `output` stay in their template type. For MoE expert tensors and
shapes that are not a multiple of 1024, use AI4_U (below).

- **Rotation.** Each 256-wide K block is rotated offline by `(1/16) * H256 * D`: H256 is a 256-point Hadamard
  transform and D is a fixed sign diagonal. The rotation is the same as RQ4's (`ggml-rq4.h`), so at run time the
  activations go through RQ4's rotate-and-quantize kernel (int8, one fp32 scale per token and 256 block).
- **Trellis.** A tile is 16 rows x 8 K-columns (128 weights), coded as a tail-biting bitshift trellis with 4 bits
  per weight and a 16-bit state. The level of a state is EXL3's "mul1" byte-sum codebook reduced to signed 8-bit:
  `level = ((bytesum(state * 0x83DCD12D) + 2) >> 2) - 128`.
- **Scales.** One fp16 scale per (row, 128 K).
- **Layout.** A 16 x 1024 slab super block is 256 B of scales followed by 128 tiles of 64 B, 8448 B in all.

The full layout is in `ggml/src/ggml-ai4.h`. The trellis follows QTIP (Tseng et al., NeurIPS 2024,
arXiv:2406.11235), and the level map is the mul1 codebook of Turboderp's exllamav3 (see `docs/exl3.md`).

## Backends

- **CUDA, NVIDIA compute capability 8.0 or newer:** the fast path. The trellis decodes straight into s8 m16n8k32 A
  fragments in registers.
  - Batches of up to 8 tokens use a per-warp cp.async mmv kernel. This covers decode and MTP verify.
  - Larger batches use a slab-per-warp GEMM.
- **CUDA, other devices, or `GGML_CUDA_AI4_REF=1`:** a reference path. It rebuilds the whole tensor to f32 and runs
  a cuBLAS SGEMM, and it needs M x K x 4 bytes of scratch per matmul.
- **CPU:** a reference `MUL_MAT`. It dequantizes one slab at a time and computes f32 dot products. It exists for
  tests and is far too slow for serving.
- **Other backends:** none. Any AI4 matmul on another backend fails to schedule.

Kernel switches. Each is read once per process, and each `GGML_CUDA_AI4_*` name also accepts the old
`GGML_CUDA_D2_*` spelling:

| variable | default | effect |
|---|---|---|
| `GGML_CUDA_AI4_MMV_STAGES` | 2 | cp.async ring depth of the batch <= 8 kernel (2, 3 or 4). The ship gate ran with 2 |
| `GGML_CUDA_AI4_MMV_XSMEM` | 0 | 1 = the first mmv version, which stages activations in shared memory |
| `GGML_CUDA_AI4_BN128_MIN` | 512 | batch size at which the GEMM switches from 64- to 128-token tiles |
| `GGML_CUDA_AI4_REF` | 0 | 1 = reference path (f32 rebuild + SGEMM) |

## Making an AI4 GGUF

`llama-quantize` cannot write AI4, because the encoder is a GPU Viterbi search. `llama-ai4-quantize` is built with
`-DGGML_CUDA=ON`. It takes three inputs:

1. **A template:** an IQ4_XS quant of the model, made with `llama-quantize --imatrix`. Every IQ4_XS tensor of the
   template that fits the shape rules becomes AI4, and every other tensor and all metadata are copied byte for byte.
2. **The BF16 GGUF of the same model.** For a split model, pass `--bf16` once per shard.
3. **An imatrix,** normally the one used for the template.

```
llama-ai4-quantize --template model-IQ4_XS.gguf --bf16 model-BF16-00001-of-00002.gguf \
    --bf16 model-BF16-00002-of-00002.gguf --imatrix imatrix.gguf --out model-AI4.gguf
```

- `--selftest` (with the same inputs and no `--out`) encodes 32 rows of three tensors on the GPU and on the CPU
  reference and compares them.
- The output is written as `model-AI4.gguf.tmp` and renamed when complete.
- New files carry `general.file_type` = 52 (`LLAMA_FTYPE_MOSTLY_AI4`), `ai4.format_version` = 2 and `ai4.note`.
- Encoding Qwen3.8-27B took 12,321 s (3.4 h) on one RTX 3090 (D-003).

## AI4_U: unit-128 layout and MoE

`GGML_TYPE_AI4_U` = 71 (`ai4u`) has the same trellis, level map, fp16 scale per (row, 128 K), int8 activations and IMMA
kernels as AI4. Only the byte layout and the rotation block differ:

- **Unit.** 16 rows x 128 K = 16 fp16 scales (32 B) followed by 16 tiles of 64 B, 1056 B in all (4.125 bpw, the same
  as AI4). Any `ne0 % 128 == 0` and `ne1 % 16 == 0` tiles, e.g. Qwen3.8-Flash-Next experts (K = 2560 and 640).
  ggml sees a 128-weight block of 66 bytes.
- **Rotation.** H256 blocks when `ne0 % 256 == 0`; then every weight decodes to exactly the value AI4 gives (same
  encode, reordered bytes), so on dense tensors AI4_U costs nothing in quality against AI4. Otherwise (K = 640) H128
  blocks with norm 1/sqrt(128) and the activations are rotated and quantized per 128.
- **Expert stacks.** 3-D tensors `[K, M, n_expert]` are n_expert AI4_U matrices back to back. `MUL_MAT_ID` is
  supported on CUDA (fast path: a graph-capturable routing kernel, the mmv kernel for up to 8 tokens and the GEMM
  above that) and on the CPU (reference). Only the experts that the router selects are read.
- **Rates.** The rate k (bits per trellis step) is a layout parameter: tile = 16k bytes, unit = 32 + 256k bytes. Type
  ids 72 (k = 3), 73 (k = 2), 74 (k = 5) and 75 (k = 6) are reserved; they are not implemented.

The encoder picks tensors by rule, so AI+ tensors can sit next to Q6_K / Q8_0 / K-quant tensors in one file:

```
llama-ai4-quantize --template model-IQ4_XS.gguf --bf16 model-BF16.gguf --imatrix imatrix.gguf \
    --tensor-type 'ffn_(gate|up|down)_exps=ai4u' --out model-AI4U-experts.gguf
```

- `--tensor-type REGEX=TYPE` (repeatable, first match wins) and `--tensor-list FILE` (lines `tensor_name TYPE`)
  select tensors whatever their template type; TYPE is `ai4`, `ai4u` or `keep`. Every other tensor is copied byte for
  byte from the template. Without rules the tool behaves as before (`--replace iq4_xs`, `--replace-type ai4`).
- Expert stacks are encoded expert by expert with that expert's imatrix row. An expert that the imatrix run never
  routed to is encoded without LDLQ (a warning gives the count).
- Because unselected tensors are copied, runs can be chained (the output of one run as the template of the next).

## Tests and tools

- `test-backend-ops -o MUL_MAT -p "type_a=ai4,"` checks a backend against the CPU reference; `-p "type_a=ai4u"` the
  AI4_U cases, and `-o MUL_MAT_ID -p "type_a=ai4"` the expert cases (512 experts / top-10 and small expert counts).
  `GGML_TEST_AI4_MOE=1 test-backend-ops perf -o MUL_MAT_ID` times AI4_U against IQ4_XS / IQ4_NL at Flash-Next shapes. With
  `GGML_TEST_RQ4_M1=1`, `test-backend-ops perf` adds the six Qwen3.8-27B projection shapes.
- `llama-ai4-kern-bench check` checks the CUDA kernels exactly against an integer reference.
- `llama-ai4-kern-bench gguf FILE` checks the kernels against the f32 reconstruction on the tensors of a real file.
- `tools/ai4/test-ai4.c` is a standalone CPU check of the format: rotation parity with RQ4, the tile round trip and
  slab error. To build it:

  ```
  cc -O3 -march=native -fopenmp tools/ai4/test-ai4.c -Iggml/src -lm -DAI4_RQ4_HEADER='"ggml-rq4.h"'
  ```

- In gguf-py, `GGMLQuantizationType.AI4` = 70, with `D2` kept as an alias, and `LlamaFileType.MOSTLY_AI4` = 52.
  gguf-py can read AI4 files but has no AI4 dequantizer.
