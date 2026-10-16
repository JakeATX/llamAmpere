# KV Cache Quantization with TurboQuant

TurboQuant adds five runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization. They are named by bit width, `turbo2` to `turbo6`:

| Type               | Also accepted     | Enum (name in logs)                 | Size             | Compression vs f16 |
|--------------------|-------------------|-------------------------------------|------------------|--------------------|
| `turbo2`           | `tq2`             | `GGML_TYPE_TURBO2_0` (43, `turbo2`) | 2.125 bits/value | 7.5x               |
| `turbo3`           | `tq3`, `tq3_0`    | `GGML_TYPE_TURBO3_0` (44, `turbo3`) | 3.125 bits/value | 5.1x               |
| `turbo4`           | `tq4`, `tq4_0`    | `GGML_TYPE_TURBO4_0` (47, `turbo4`) | 4.125 bits/value | 3.9x               |
| `turbo5`           | `tq5`, `tq5_0`    | `GGML_TYPE_TQ5_0` (59, `tq5_0`)     | 5.125 bits/value | 3.1x               |
| `turbo6`           | `tq6`, `tq6_0`    | `GGML_TYPE_TQ6_0` (58, `tq6_0`)     | 6.125 bits/value | 2.6x               |

Every flag that takes a cache type (`-ctk`/`-ctv`, `-ctkd`/`-ctvd`, `llama-bench -ctk/-ctv`)
accepts both spellings, case-insensitively. Logs and `llama-bench` output print the ggml
name in the third column (for example `K (tq5_0)` for `-ctk turbo5`). `tq2_0` is not an
alias: it is ggml's ternary weight type, and `-ctk tq2_0` is rejected.

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`, `q8_0`, `turbo3`, `turbo4`, `turbo5`, `turbo6` for K and V is
supported, and `turbo2` for V only (a `turbo2` K request is rejected at context
creation); mixing a low-bit V with a higher-bit K is the common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed). A quantized V cache with flash attention explicitly disabled is an
error, matching upstream behavior for all quantized V types.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Tiered TQ

`--tiered-tq` selects a single-sequence experimental cache with a 128-token FP16 sink, a sealed Turbo4 K/V body, and a fixed 8192-token TQ6 K/V tail. It requires CUDA or CPU attention, flash attention, and 256-channel heads. The configured tail is a minimum: sealing rounds to complete 128-token groups and allocation includes input-batch and rollback headroom.

```bash
llama-server -m model.gguf -c 65536 -ngl 99 -fa on --tiered-tq
```

All three regions share the standard signed TurboQuant WHT128 basis. Q, K, and V rotate once before storage or attention; the attention output receives one inverse WHT128. The sink stores FP16 values in this basis. The tail stores the native 98-byte TQ6 blocks; sealing decodes TQ6, rounds to FP16, and quantizes native 66-byte Turbo4 blocks without an additional rotation. Each 128-value body block has 64 packed nibble bytes and a corrected FP16 norm. The body uses the native 16-centroid codebook and no SJ-KVaRN balancing, residual correction, or outlier table.

The implementation reuses the region lifetime and maintenance machinery internally. `--sjkvarn-body-type turbo4` selects the same basis and codec; `--tiered-tq` supplies the complete recommended experiment configuration. To compare against a SJ-KVaRN body with an identical sink and tail policy:

```bash
llama-server -m model.gguf -c 65536 -ngl 99 -fa on \
    -ctk sjkvarn4 -ctv sjkvarn4 --sjkvarn-body-type sjkvarn4 \
    --sjkvarn-staging-type tq6_0 --sjkvarn-sink-type f16 \
    --sjkvarn-sink 128 --sjkvarn-tail 8192 --sjkvarn-tail-max 0
```

The comparator retains SJ-KVaRN's Hadamard256 basis; identical region sizes do not imply identical quantization errors. These are separately named codecs, and quality and speed need matched measurements.

Narrow CUDA attention directly stages packed Turbo4 strips and decodes their centroids into half MMA operands. It supports one through eight query rows, including MTP verification widths four, five, and six. Large prefills reuse bounded FP16 expansion and ordinary MMA attention; packed MMA remains the fallback. The leading FP16 sink occupies an appended region, with a small unused packed sink reservation and row-alignment padding in the allocation. Full-cache FP16 copies are not retained.

## Hybrid SJ-KVaRN4/4 with TQ6 staging

`--sjkvarn-staging-type tq6_0` stores both the leading sink and the unsealed tail in TQ6 while keeping sealed K and V body records at 4 bits. This experimental mode requires `-ctk sjkvarn4 -ctv sjkvarn4`, flash attention, 256-channel K/V heads, and a single sequence. CPU and CUDA implement the stored-domain writer and SJ-KVaRN attention path. Other backends must reject unsupported operations.

```bash
llama-cli -m model.gguf -c 32768 -ngl 99 -fa on \
    -ctk sjkvarn4 -ctv sjkvarn4 --sjkvarn-staging-type tq6_0 \
    --sjkvarn-sink 128 --sjkvarn-tail 2048 --sjkvarn-tail-max 8192
```

The default staging is `tq6_0` with a separate F16 sink; `q8_0` staging stores the sink in Q8_0 as well, and `f16` staging is opt-in only. The tail grows toward 8192 positions during active generation, then complete 128-token groups are compressed toward the 2048-position floor. The allocation includes group and batch headroom. Server idle maintenance can compress toward the floor after resolving speculative rollback and discarding prompt checkpoints. See [SJ-KVaRN CUDA attention](development/sjkvarn-cuda.md) for the maintenance contract.

SJ-KVaRN rotates Q, K, and V once with its 256-point Hadamard transform. TQ6 staging applies its 64-entry codebook, norm correction, and 6-bit packing separately to each 128-value block in that existing basis, without another WHT128 or InnerQ scaling. Attention reads that same basis and applies only the inverse SJ-KVaRN256 transform to its output. SJ-KVaRN ignores `TURBO_LAYER_ADAPTIVE` and the optional `LLAMA_ATTN_ROT_K/V_OVERRIDE` rotations. Ordinary `-ctk tq6_0 -ctv tq6_0` retains its existing WHT128 behavior.

Each 128-value TQ6 block occupies 98 bytes, including its FP16 norm; a 256-channel head occupies 196 bytes. Sealing dequantizes TQ6, rounds the reconstruction to FP16, then compresses it into SJ-KVaRN4 records. This composed TQ6-to-SJ-KVaRN4 path is lossy. Codec and kernel agreement do not establish model quality: compare matched-token KLD and real generation against the frozen Q8 baseline before drawing quality or performance conclusions.

`GGML_CUDA_SJ_FILL=1` opts into a split-cost estimate for scalar SJ-KVaRN 4/4 attention with TQ6 K/V staging on sm_80 and sm_86. It uses the card's SM count and kernel occupancy to balance attention work against partial-result reduction cost at query widths 1-8, with enough blocks to cover every SM when the KV tile limit permits. This coverage constraint can increase estimated cost at short contexts. The default is off; other values also leave the original split selection unchanged. Width 9 keeps the general MMA Stream-K path. Multiple sequences still run as separate attention nodes. The existing `GGML_SJKVARN_DIRECT_STABLE` split override takes precedence. Changing splits changes floating-point reduction order; validate correctness and off/on performance on the target card before using it in production.

## SJ-KVaRN low-bit bodies and the trellis codec

`--kv-resident 32768` opts QSA layers into a resident window of 32,768 KV cells, with sealed SJ-KVaRN body blocks in pinned RAM and selected blocks fetched for attention. Dense and linear layers keep their existing allocation. The server default is 32768; `--kv-resident 0` disables it. The C API `llama_context_default_params` still leaves the field at 0 until the caller sets it. `LLAMA_ARG_KV_RESIDENT` sets the same cell count. N reserves the sink and maximum live tail first, then divides the remaining cells into 128-cell sealed body groups. The existing extra staging space for prefill batches stays on the card. The current CUDA implementation supports a single stream, 256-channel QSA heads, and scalar SJ-KVaRN 4/4. Saved states retain the same logical sealed bytes and can move between split and full-card caches.

The existing `--moe-cache auto` uses the free CUDA memory after this allocation; keep the same cache reserve and do not subtract the KV saving a second time. Matched load-only tests on an RTX 3090 Ti with Swift Flash-Next IQ3_XXS, SJ-KVaRN 4/4, and 12 QSA layers measured 432 MiB less whole-card memory at 100,096 context cells and 1,446 MiB less at 262,144 cells. These tests disabled the expert cache to isolate the allocation change. The sealed-body allocation calculation saves 447.92 MiB and 1,464.18 MiB respectively; driver allocation and mapping overhead reduce the measured headroom. Pinned body storage is 601.25 MiB and 1,617.51 MiB respectively. These are memory measurements, not inference speed results.

In matched 100,000-token inference checks with 131,072 cells allocated and the same 3,072 MiB reserve, the expert allocator received 626.25 MiB more with the split. This uses the existing free-minus-reserve policy; no new expert-cache budget flag or manual credit is needed.

`test-sjkvarn-attn --resident` and `--resident-tq6` compare split and full-card attention exactly, including cold and warm selected blocks. `test-sjkvarn-state --resident` checks split/full state transfers and continuation. A saved 100,000-token prefix must survive a cold server restart: the main state, `.ckpt` checkpoint sidecar, and `.dft` MTP draft state keep the same size and SHA-256 hash, and the restored continuation reuses the saved prefix.

Three SJ-KVaRN body configurations are the production paths, all with the staged sink (`--sjkvarn-sink 128`) and the
adaptive tq6_0 tail (`--sjkvarn-tail 4096 --sjkvarn-tail-max 8192`). The adaptive tail is the default for every SJ-KVaRN body (`--sjkvarn-tail 4096 --sjkvarn-tail-max 8192`); `--sjkvarn-tail-max 0` selects a fixed tail. A larger `--sjkvarn-tail` without `--sjkvarn-tail-max` raises the ceiling to match:

| Path        | Flags                                                  | Body bits per element | Sealed body codec                     |
|-------------|--------------------------------------------------------|-----------------------|---------------------------------------|
| 4/4         | `-ctk sjkvarn4 -ctv sjkvarn4`                              | 4.28                  | scalar records                        |
| 3/3 trellis | `-ctk sjkvarn4 -ctv sjkvarn4 --sjkvarn-bits-k 3 --sjkvarn-bits-v 3 --sjkvarn-body-type auto` | 3.28 | trellis (L=9, 512-entry codebook) |
| 3/2 trellis | `-ctk sjkvarn4 -ctv sjkvarn4 --sjkvarn-bits-k 3 --sjkvarn-bits-v 2 --sjkvarn-body-type auto` | 2.78 | trellis (K L=9, V L=8, 256-entry codebook) |

`--sjkvarn-body-type` selects the sealed-body codec: `sjkvarn4` (scalar records, the default), `sjkvarn4t` (trellis for every
supported pair), `turbo4` (Tiered TQ) or `auto`, which picks the trellis body for the 3/3, 3/2 and 2/2 pairs and scalar
records for everything else. The trellis pairs are 4/4, 3/3, 3/2 and 2/2; a 4-bit side inside a low-bit pair is rejected
because the low-bit tile loader would read it as scalar. The 2/2 pair works but is data only, not a recommendation.

The trained trellis codebooks are compiled into the binary (`ggml/src/ggml-sjkvarn-cb-lowbits.h`, generated from
`cb3_trained_mse.bin` and `cb2_trained_mse.bin`); no files or environment variables are needed. The per-group norm refit
(`GGML_SJKVARN_TRELLIS_REFIT=norm`) is the default. Overrides, for experiments only:

| Variable                     | Default   | Effect                                                                 |
|------------------------------|-----------|------------------------------------------------------------------------|
| `GGML_SJKVARN_TRELLIS_CB3`     | built-in  | `identity` (scalar codes on the trellis decode path), `trained`, or a path to a 512-entry fp16 K+V codebook file |
| `GGML_SJKVARN_TRELLIS_CB2`     | built-in  | same for the 2-bit codebook (256 entries)                              |
| `GGML_SJKVARN_TRELLIS_REFIT`   | `norm`    | `none` (codebook reconstruction as is), `fit` (least-squares scale), `norm` (match the group norm) |

## MTP draft memory experiments

`GGML_MTP_PREFILL_DEVICE=1` retains unmasked target hidden rows in one device buffer and feeds the MTP KV-only catch-up at each logical-batch boundary with device copies. The original draft microbatch geometry is preserved. The buffer holds `n_batch + n_seq_max` FP32 hidden rows: 80.020 MiB for width 5120, batch 4096 and one slot, independent of prompt length. It is allocated on the first captured row, so pre-decode auto-fit estimates need this additional headroom. A new prompt, rewind or restored speculative sidecar can upload one carry row per affected sequence; ordinary prompt rows are never re-uploaded. The existing host hidden outputs remain available for verification and cache state.

`GGML_MTP_PREFILL_ASYNC=1` additionally gives the MTP context private compute and scratch storage and retains small input uploads until their completion event. Its prompt reservation uses the KV-only graph, plus a small full graph for generation and deferred tails (72 rows, limited by microbatch capacity and rounded for sequence count). Target-ready and draft-consumed events order device copies and staging reuse. The next target microbatch can execute while draft catch-up is outstanding; graph allocation, cache maintenance and unsupported backend splits can still synchronize. The same device staging is included, so combining the two switches needs no second hidden buffer. Startup logs report compute and hidden staging allocations separately; private CUDA scratch usage is additional and workload-dependent.

Both switches are off by default and require the exact value `1`. They apply to the single-head Qwen35 MTP path on a single device with events, or to the synchronous CPU reference. Other architectures keep the original path. Small deferred updates, generation sampling, KV codecs and prompt-cache formats retain their existing behavior. Both switches include the coalesced hidden getter behavior of `GGML_MTP_PREFILL_OVERLAP=1`. GPU token identity, peak VRAM and wall-time improvements must be validated before enabling them in production.

`LLAMA_MTP_DRAFT_COMPUTE_LEAN=1` caps a single-sequence MTP draft context's physical microbatch at 64 rows. The logical batch stays unchanged, so prompt catch-up is split by the existing decoder. This reduces the full-context attention mask and intermediate graph storage. The switch is off by default; prefill geometry changes, so fixed-seed draft acceptance and output identity still require validation. Separate draft models and multiple sequences keep their existing microbatch size.

`--spec-draft-ubatch N` (`-ubd`, env `LLAMA_ARG_SPEC_DRAFT_UBATCH`) sets the physical microbatch of an MTP draft loaded from its own file (`-md` with `--spec-type draft-mtp` or `draft-mtp-adaptive`). The default, -1, caps it at 2,048 rows; 0 keeps the main `-ub`; N caps it at N. The draft's compute buffer grows with its microbatch (on Flash-Next about 0.8 MiB per row: 1,672 MiB at 2,048 and 6,634 MiB at 8,192), and the fit counts it, so with a large main `-ub` an uncapped draft buffer pushes layers off the GPU. The cap only changes how the draft's prompt pass is split; decode-time draft calls work on a few rows at a time (at most 64 deferred catch-up rows) and stay below it.

`--spec-draft-sjkvarn` gives the MTP context its own SJ-KVaRN cache with the trunk's body bits, body codec, sink, staging type and tail policy. It requires a SJ-KVaRN trunk, one sequence, and no separate draft model or explicit `-ctkd`/`-ctvd`. The draft layer uses the trunk's main body precision, without trunk edge tiers. Existing SJ-KVaRN head-dimension and flash-attention restrictions apply. For example:

```bash
LLAMA_MTP_DRAFT_COMPUTE_LEAN=1 llama-server -m model.gguf -c 204800 -ngl 99 -fa on --parallel 1 \
    -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type sjkvarn4t \
    --sjkvarn-staging-type tq6_0 --sjkvarn-sink-type f16 \
    --sjkvarn-sink 128 --sjkvarn-tail 4096 --sjkvarn-tail-max 8192 \
    --spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-sjkvarn
```

The tail must exceed the maximum draft width so speculative rollback stays in unsealed rows. The server reprocesses an edited prompt if it reaches sealed rows in either context. Draft sealing runs during decode; server idle compression still applies only to the trunk. State serialization retains the existing SJ-KVaRN restrictions. Server slot saves include a `.dft` draft-state file and a `.ckpt` sidecar that binds both state files with SHA-256. Draft saves made before this binding must be saved again with the current server before they can be restored; non-draft version-2 checkpoint sidecars remain readable. Cache compression changes draft probabilities and needs acceptance/quality validation separately from the lean microbatch switch.

### Small-card overhead switches

These cut CUDA memory that is not part of the model, KV or compute buffers. The numbers are from Qwen3.8-27B EXL3 2.0 bpw
at `-c 204800`, ub 512, with the MTP drafter, on an RTX 3090 Ti (84 SMs); the token streams were byte-identical with and
without each switch.

| Switch | Default | Effect | Measured |
|--------|---------|--------|----------|
| `GGML_CUDA_EXL3_GLUE_INPLACE` | `1` | EXL3 prefill GEMM applies its output transform in place when there is no split-K, instead of through a T x N f32 scratch buffer | scratch pool 102 -> 52 MiB |
| `LLAMA_SHARED_POOL` | `1` (`0` opts out) | the MTP draft context uses the trunk context's CUDA scratch pool instead of its own (the two contexts already run one after the other) | scratch pool 52 -> 28 MiB |
| `GGML_CUDA_STACK_LIMIT=<bytes>` | 0 (set at backend init) | per-thread stack reservation; the driver reserves bytes x 1536 threads x SMs and grows it at launch to the largest stack of any kernel that runs | 1024 -> 0 (regrown to 112): -112 MiB on 84 SMs |

The stack reservation scales with the SM count. The CUDA backend sets the limit to 0 when it creates a device's
first backend; the driver then grows the reservation to the largest per-thread stack of any kernel that runs. With the
12 GB command that is 112 B (the SJ-KVaRN 3/2 lowbits prefill attention kernel, which spills registers), reached at the
first warm-up launch: 13.8 MiB on 84 SMs, 13.1 MiB on 80 SMs, 4.6 MiB on 28 SMs, instead of 126 / 120 / 42 MiB at the
driver's 1024 B default. `GGML_CUDA_STACK_LIMIT=1024` restores the driver default.

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
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Rewrite a symmetric turbo3 K+V request to q8_0 K on models with GQA ratio >= 6 (`0` disables); turbo4, turbo5 and turbo6 K are never rewritten, turbo2 is V-only |
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
