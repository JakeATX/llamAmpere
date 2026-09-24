# KV Cache Quantization with TurboQuant

TurboQuant adds three runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2 bits/value    | 6.4x               |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.25 bits/value | 4.9x               |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.25 bits/value | 3.8x               |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`, `q8_0`, `turbo3`, `turbo4`, `tq5_0`, `tq6_0` for K and V is
supported, and `turbo2` for V only (a `turbo2` K request is rejected at context
creation); mixing a low-bit V with a higher-bit K is the common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed). A quantized V cache with flash attention explicitly disabled is an
error, matching upstream behavior for all quantized V types.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## Tiered TQ

`--tiered-tq` selects a single-sequence experimental cache with a 128-token FP16 sink, a sealed Turbo4 K/V body, and a fixed 8192-token TQ6 K/V tail. It requires CUDA or CPU attention, flash attention, and 256-channel heads. The configured tail is a minimum: sealing rounds to complete 128-token groups and allocation includes input-batch and rollback headroom.

```bash
llama-server -m model.gguf -c 65536 -ngl 99 -fa on --tiered-tq
```

All three regions share the standard signed TurboQuant WHT128 basis. Q, K, and V rotate once before storage or attention; the attention output receives one inverse WHT128. The sink stores FP16 values in this basis. The tail stores the native 98-byte TQ6 blocks; sealing decodes TQ6, rounds to FP16, and quantizes native 66-byte Turbo4 blocks without an additional rotation. Each 128-value body block has 64 packed nibble bytes and a corrected FP16 norm. The body uses the native 16-centroid codebook and no KVarN balancing, residual correction, or outlier table.

The implementation reuses the region lifetime and maintenance machinery internally. `--kvarn-body-type turbo4` selects the same basis and codec; `--tiered-tq` supplies the complete recommended experiment configuration. To compare against a KVarN body with an identical sink and tail policy:

```bash
llama-server -m model.gguf -c 65536 -ngl 99 -fa on \
    -ctk kvarn4 -ctv kvarn4 --kvarn-body-type kvarn4 \
    --kvarn-staging-type tq6_0 --kvarn-sink-type f16 \
    --kvarn-sink 128 --kvarn-tail 8192 --kvarn-tail-max 0
```

The comparator retains KVarN's Hadamard256 basis; identical region sizes do not imply identical quantization errors. These are separately named codecs, and quality and speed need matched measurements.

Narrow CUDA attention directly stages packed Turbo4 strips and decodes their centroids into half MMA operands. It supports one through eight query rows, including MTP verification widths four, five, and six. Large prefills reuse bounded FP16 expansion and ordinary MMA attention; packed MMA remains the fallback. The leading FP16 sink occupies an appended region, with a small unused packed sink reservation and row-alignment padding in the allocation. Full-cache FP16 copies are not retained.

## Hybrid KVarN4/4 with TQ6 staging

`--kvarn-staging-type tq6_0` stores both the leading sink and the unsealed tail in TQ6 while keeping sealed K and V body records at 4 bits. This experimental mode requires `-ctk kvarn4 -ctv kvarn4`, flash attention, 256-channel K/V heads, and a single sequence. CPU and CUDA implement the stored-domain writer and KVarN attention path. Other backends must reject unsupported operations.

```bash
llama-cli -m model.gguf -c 32768 -ngl 99 -fa on \
    -ctk kvarn4 -ctv kvarn4 --kvarn-staging-type tq6_0 \
    --kvarn-sink 128 --kvarn-tail 2048 --kvarn-tail-max 8192
```

These are experiment settings; the default staging type remains `f16`. The tail grows toward 8192 positions during active generation, then complete 128-token groups are compressed toward the 2048-position floor. The allocation includes group and batch headroom. Server idle maintenance can compress toward the floor after resolving speculative rollback and discarding prompt checkpoints. See [KVarN CUDA attention](development/kvarn-cuda.md) for the maintenance contract.

KVarN rotates Q, K, and V once with its 256-point Hadamard transform. TQ6 staging applies its 64-entry codebook, norm correction, and 6-bit packing separately to each 128-value block in that existing basis, without another WHT128 or InnerQ scaling. Attention reads that same basis and applies only the inverse KVarN256 transform to its output. KVarN ignores `TURBO_LAYER_ADAPTIVE` and the optional `LLAMA_ATTN_ROT_K/V_OVERRIDE` rotations. Ordinary `-ctk tq6_0 -ctv tq6_0` retains its existing WHT128 behavior.

Each 128-value TQ6 block occupies 98 bytes, including its FP16 norm; a 256-channel head occupies 196 bytes. Sealing dequantizes TQ6, rounds the reconstruction to FP16, then compresses it into KVarN4 records. This composed TQ6-to-KVarN4 path is lossy. Codec and kernel agreement do not establish model quality: compare matched-token KLD and real generation against the frozen Q8 baseline before drawing quality or performance conclusions.

## KVarN low-bit bodies and the trellis codec

Three KVarN body configurations are the production paths, all with the staged sink (`--kvarn-sink 128`) and the
adaptive tq6_0 tail (`--kvarn-tail 4096 --kvarn-tail-max 8192`):

| Path        | Flags                                                  | Body bits per element | Sealed body codec                     |
|-------------|--------------------------------------------------------|-----------------------|---------------------------------------|
| 4/4         | `-ctk kvarn4 -ctv kvarn4`                              | 4.28                  | scalar records                        |
| 3/3 trellis | `-ctk kvarn4 -ctv kvarn4 --kvarn-bits-k 3 --kvarn-bits-v 3 --kvarn-body-type auto` | 3.28 | trellis (L=9, 512-entry codebook) |
| 3/2 trellis | `-ctk kvarn4 -ctv kvarn4 --kvarn-bits-k 3 --kvarn-bits-v 2 --kvarn-body-type auto` | 2.78 | trellis (K L=9, V L=8, 256-entry codebook) |

`--kvarn-body-type` selects the sealed-body codec: `kvarn4` (scalar records, the default), `kvarn4t` (trellis for every
supported pair), `turbo4` (Tiered TQ) or `auto`, which picks the trellis body for the 3/3, 3/2 and 2/2 pairs and scalar
records for everything else. The trellis pairs are 4/4, 3/3, 3/2 and 2/2; a 4-bit side inside a low-bit pair is rejected
because the low-bit tile loader would read it as scalar. The 2/2 pair works but is data only, not a recommendation.

The trained trellis codebooks are compiled into the binary (`ggml/src/ggml-kvarn-cb-lowbits.h`, generated from
`cb3_trained_mse.bin` and `cb2_trained_mse.bin`); no files or environment variables are needed. The per-group norm refit
(`GGML_KVARN_TRELLIS_REFIT=norm`) is the default. Overrides, for experiments only:

| Variable                     | Default   | Effect                                                                 |
|------------------------------|-----------|------------------------------------------------------------------------|
| `GGML_KVARN_TRELLIS_CB3`     | built-in  | `identity` (scalar codes on the trellis decode path), `trained`, or a path to a 512-entry fp16 K+V codebook file |
| `GGML_KVARN_TRELLIS_CB2`     | built-in  | same for the 2-bit codebook (256 entries)                              |
| `GGML_KVARN_TRELLIS_REFIT`   | `norm`    | `none` (codebook reconstruction as is), `fit` (least-squares scale), `norm` (match the group norm) |

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them.

## Environment knobs

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | `0`     | Layer-adaptive KV precision; `7` = Boundary V (first/last layers in `q8_0`, middle in turbo) |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Rewrite a symmetric turbo3 K+V request to q8_0 K on models with GQA ratio >= 6 (`0` disables); turbo4, tq5_0 and tq6_0 K are never rewritten, turbo2 is V-only |
| `TURBO_SPARSE_V`                | `1`     | Sparse-V dequant skip in flash attention (`0` disables)        |
| `LLAMA_ATTN_ROT_K_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for K                |
| `LLAMA_ATTN_ROT_V_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for V                |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. MoE models disable CUDA graphs
for TQ `MUL_MAT_ID` automatically.
