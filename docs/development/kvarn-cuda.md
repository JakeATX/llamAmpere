# KVarN CUDA attention

The cache math follows [Huawei KVarN](https://github.com/huawei-csl/KVarN/tree/7586257f1c632e63187bfacbbe21ccb51540f7b3), pinned to `7586257f1c632e63187bfacbbe21ccb51540f7b3`. The native codec and fragment record layout are defined in [`ggml-kvarn.h`](../../ggml/src/ggml-kvarn.h). This codec uses fixed balancing iterations and the final scales, rather than the research implementation's best-so-far scale selection.

## Supported inputs

The CUDA KVarN attention route supports K4/V4, 256-channel K and V heads, one query batch, an f16 attention mask, no ALiBi, and a KV position count padded to 256. Query heads must divide into complete KV-head groups. K and V source tensors contain F16, Q8_0 or stored-domain TQ6_0 sink and ring rows; sealed body records and the device descriptor are separate inputs. Region and ring alignment must satisfy the existing cache invariants.

The cache sink is distinct from an attention sink input. Direct decode excludes attention sinks and nonzero logit softcap; those cases use the packed MMA fallback. Prefill reuse retains the mask, attention sink input, and logit softcap on inputs supported by the outer KVarN route. It does not enable maskless attention, ALiBi, or multiple query batches.

## Sink and tail precision

`--kvarn-staging-type q8_0` stores both the sink and ring in Q8_0; the default is `tq6_0` with a separate F16 sink. For example, use `-ctk kvarn4 -ctv kvarn4 --kvarn-staging-type q8_0 --kvarn-sink 256 --kvarn-tail 4096`. The adaptive tail is the default (`--kvarn-tail 4096 --kvarn-tail-max 8192`); with `--kvarn-tail-max 0` the tail keeps its fixed minimum and seals mature groups as they become available.

Q8 rows use the existing quantized SET_ROWS writer after the KVarN rotation. Attention dequantizes only the tile being read to FP16. Compression also reads Q8, rounds its reconstruction to FP16, and seals the resulting values into K4/V4 records. There is no persistent FP16 copy of the Q8 sink or tail. Q8 staging therefore introduces an additional quantization step before body compression; a larger retained window must be evaluated against this cost using matched-token KLD.

Each 32-value Q8_0 block occupies 34 bytes including its scale, versus 64 bytes for FP16. A 256-token Q8 sink occupies 6.25% more storage than a 128-token FP16 sink. These are storage calculations, not speed or quality guarantees.

`--kvarn-staging-type tq6_0` uses TQ6 for both sink and tail, with the K4/V4 body unchanged. Its SET_ROWS flag selects packing of already KVarN256-rotated values: there is no second WHT128 or InnerQ transform. Each 128-value block uses 98 bytes, so a D256 head is 196 bytes and is only 4-byte aligned. Packed attention loads use byte pitches and scalar or 16-bit reads; the F16 paths retain their 16-byte alignment requirement. Both CPU and CUDA sealers reconstruct staging values through FP16 before applying KVarN. Byte parity remains the sealer test gate; attention compares numerical outputs to the CPU reference. TQ6-to-KVarN4 quality must be measured separately.

The next experiment uses `--kvarn-sink 128 --kvarn-tail 2048 --kvarn-tail-max 8192`. See the [usage documentation](../KV-cache-quantization.md#hybrid-kvarn44-with-tq6-staging) for the full command. The default staging type and tail policy are unchanged.

## Delayed compression

Compression runs in a separate cached maintenance graph before a model pass. Ordinary model graphs contain no seal nodes. The host computes whether a group is mature; an empty plan returns without launching a maintenance graph or synchronizing the GPU. A nonempty plan completes compression for every cache layer before publishing the new sealed boundary. The initial implementation synchronizes at these boundaries for correctness.

Set `--kvarn-tail-max 4096` with `--kvarn-tail 1024` to enable a tail that grows and then shrinks. While generation continues, unsealed rows accumulate until the earliest query reaches the upper limit. The runtime then compresses complete 128-token groups toward the minimum. This changes compression frequency; it does not skip model layers or omit their distinct K/V data. A higher minimum, such as 2048, retains more recent Q8 rows but compresses smaller batches more often. Group rounding and the current input batch add headroom beyond the configured limits.

For a single server slot, after a request finishes, the server can compress toward the minimum while idle. It requires an accepted, materialized frontier and no unresolved speculative transaction. Before compression, the server discards saved prompt checkpoints that could require the old unsealed rows. The live prompt and recurrent state remain available for continuation; older checkpoint-based prefix reuse is lost. A new request, cache clear, or unresolved decode invalidates the pending idle work. Applications using libllama directly can call `llama_kvarn_compress_idle` under the same contract. This API only operates on the attention cache; it does not advance or reset recurrent state.

The ring reserves space for the upper limit at context creation. Idle compression does not release that allocation. Once older rows are compressed, increasing the desired tail cannot restore their original Q8 values; the tail grows as new tokens arrive. Batching delays work rather than eliminating the codec work per converted group, and larger flushes can cause a larger individual pause. Measure end-to-end generation and boundary latency in addition to compression-event counts.

## Decode

[`fattn-kvarn-stream.cuh`](../../ggml/src/ggml-cuda/fattn-kvarn-stream.cuh) serves one through eight query rows with GQA ratios one through eight on devices with `cp.async` support. Each query has its own warp and reads its own mask. Warps for different queries share packed K/V bytes and record metadata in a per-KV-split shared-memory ring. Asynchronous loads overlap with tensor-core work.

The default uses half-precision tensor-core operands and float accumulators at every supported direct-decode width. K scales are absorbed into query operands, and V scales into attention operands and output rescaling. This avoids the additional integer rounding used by the older windowed path when verification crosses four query rows. It does not remove K4/V4 storage loss or half-precision rounding.

The launcher evaluates two-, four-, and six-stage rings with CUDA occupancy queries. It prefers the greatest resident-block count, then the deeper pipeline on ties. Choices are cached per device and query width within each kernel specialization. Forced stage choices that cannot launch fall back to this selection.

On SM86 with 24 query heads and four KV heads, five-query verification uses the generic half kernel through 5376 padded KV positions, based on the measured short-context crossover. Larger contexts and other shapes retain the dedicated specialization. Both paths use half operands.

Five-query verification has a dedicated specialization with a compile-time query count and one KV split per block. Its `__launch_bounds__(160, 2)` guides the compiler's register budget toward two resident blocks, while the occupancy query still accounts for actual register and shared-memory use. This specialization changes scheduling and compiler optimization; it retains half-MMA operands and adds no int8 query or attention rounding.

The diagnostic stable mode uses one KV split per block and a fixed six-stage generic kernel, bypassing the five-query specialization. Its block partition is chosen from a fixed four-query reference geometry, then capped by the available KV tiles. This keeps partition and reduction order independent of query width at the same cache extent. It sacrifices width-specific scheduling and can change throughput; it is not a general bit-exactness guarantee.

## Prefill reuse

[`fattn-kvarn-prefill.cu`](../../ggml/src/ggml-cuda/fattn-kvarn-prefill.cu) handles at least 128 query rows on Ampere or newer compiled CUDA architectures. It expands the active padded cache once into contiguous f16 K/V scratch, then invokes existing f16 MMA Flash Attention. The expansion reuses the native half2 tile decoder, copies exact rows with their source strides, resolves ring wrap from the device descriptor, and zeroes positions beyond the visible end. It does not read the descriptor back to the host.

Scratch uses the existing CUDA pool and stream. Its size is `2 * 256 * KV_heads * padded_positions * sizeof(half)`, covering both K and V. The default cap is 256 MiB: 65,536 positions with four KV heads. This is a byte cap, so larger head counts lower the position limit. The pool may retain the allocation for reuse; the cap does not bound other attention workspace or total GPU memory.

When the full active cache exceeds the cap, dispatch splits it into independent KV-head groups and their matching query heads. Each group runs ordinary f16 attention over the complete token range, then scatters its contiguous temporary output into the destination's original query/head strides. Scratch for both expanded K/V and the output temporary counts against the cap and is reused on the same stream. A smaller final group is allowed. Attention sink logits are sliced with their query heads; grouped expansion requires a mask shared across heads (`ne[2] == 1`).

The all-head path stays unchanged when its expanded K/V fit. If one KV head plus its output cannot fit, grouped expansion is disabled, or the mask is head-specific, an over-budget request keeps packed MMA. This implementation does not split the token softmax domain or merge window outputs. Small query batches also keep the packed path to avoid expansion overhead.

## Experimental controls

Set these before process startup; values are cached. These controls select implementations, not quality or performance guarantees.

| Variable | Default | Effect |
| --- | --- | --- |
| `GGML_KVARN_PREFILL_MIB` | `256` | Expansion scratch cap in MiB; `0` disables reuse. Accepted range is 0 through 16384; invalid input uses the default. |
| `GGML_KVARN_PREFILL_TRACE` | unset | `1` logs each expanded prefill's query count, position count, KV heads, and scratch size. |
| `GGML_KVARN_PREFILL_HEAD_GROUPS` | `1` | `0` disables grouped expansion above the scratch cap, restoring packed fallback there. The all-head path is unaffected. |
| `GGML_KVARN_NO_DIRECT` | unset | Any defined value disables direct decode, including `0`. It does not disable prefill reuse. |
| `GGML_KVARN_DIRECT_STREAM_MAX` | `8` | Largest direct-decode width using half-MMA streaming. `4` selects the legacy integer windowed path for widths 5 through 8; `0` selects it for all direct widths. |
| `GGML_KVARN_DIRECT_NSTAGE` | `0` | `0` selects ring depth by occupancy; use `2`, `4`, or `6` to request a depth. An unlaunchable request falls back to selection. |
| `GGML_KVARN_DIRECT_KW8` | unset | Any defined value targets up to eight warps per block for query widths up to four, instead of four. |
| `GGML_KVARN_DIRECT_STABLE` | unset | A nonzero integer enables the fixed-partition diagnostic mode for streaming decode, overriding its stage and KV-split controls. |
| `GGML_KVARN_DIRECT_NT5_GENERIC` | unset | Unset selects five-query half kernels by shape; `0` forces the specialization and `1` forces the generic kernel. Stable mode takes precedence. |
| `GGML_KVARN_DIRECT_W8` | unset | A nonzero integer requests eight-strip windows for the legacy integer path at widths up to seven. |
| `GGML_KVARN_DIRECT_NWIN4` | unset | A nonzero integer requests four-strip, four-stage windows for the legacy integer path; takes priority over `W8`. |

To isolate prefill reuse, compare identical requests with `GGML_KVARN_PREFILL_MIB=0` and an adequate scratch cap. To isolate grouped expansion, keep the cap fixed below the all-head requirement and compare `GGML_KVARN_PREFILL_HEAD_GROUPS=0` with the default. To isolate the decode arithmetic change, compare the default streaming width with `GGML_KVARN_DIRECT_STREAM_MAX=4`. Record every override, actual context depth, generated length, model, and binary revision.

## Numerical validation

Reusing the native tile decoder preserves its body reconstruction rounding. Attention scheduling, scale absorption, and reduction order can still change results across implementations or query widths. Compare attention outputs numerically against the CPU reference and separately measure generated-region quality. Bit identity and identical sampled text are not assumed.

Use the existing KVarN cases in `test-backend-ops`, `test-kvarn-codec`, and `test-kvarn-attn`. Check that the intended cases and dispatches actually ran. Exercise query widths 1 through 8, large and ragged prefills, GQA classes, different masks per query, wholly masked strips, ring wrap, padded positions, source strides, and supported attention-sink/softcap fallback cases. Test prefill disabled and over-budget dispatch as well as the expanded path. Keep speed claims in measured result artifacts with matching settings.
