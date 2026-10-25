# CPU expert math

These experimental controls are off by default. Set them before process startup. They use the existing CPU thread pool and preserve the packed model weights.

- `GGML_CPU_EXPERT_REUSE=1`: on AVX2, decode each IQ2_XS, IQ3_XXS, IQ3_S, or IQ4_NL weight block once for a group of up to eight routed activation rows. Single-row work keeps the existing dot product. IQ2_XS and IQ3 formats use Q8_K activations; IQ4_NL keeps its existing Q8_0 activations.
- `GGML_CPU_EXPERT_SCHEDULE=1`: for verification batches of up to eight tokens, distribute 32-row tasks across all selected CPU experts with one shared queue. The queue skips experts served by the GPU cache. Longer batches keep the existing tiled scheduling.
- `GGML_CPU_EXPERT_Q8_REUSE=1`: share a single input conversion between neighboring routed gate/up operations using the same F32 input and routing IDs. An intervening F32 clamp is supported. The existing cache gate/up fusion also shares its converted input. Each down projection converts its distinct intermediate input as usual.
- `GGML_CPU_EXPERT_PIN=1`: on Linux, select one allowed logical CPU per physical core, reserve the first physical core for host/GPU coordination, and cap the pool at the remaining physical core count. Explicit thread-pool CPU masks take precedence. If topology is unavailable, keep the caller's thread-pool configuration.

The grouped kernels retain the existing integer accumulation lanes, block scales, and floating-point reduction order. The CPU reference path bypasses grouped dispatch and the new scheduling. Input conversion uses the existing quantizer unchanged. Expert-wide scheduling uses vector dots instead of the existing eight-token tiled kernel; that choice can change rounding relative to a previous tiled run. Full-model KL and throughput belong to the integration gate.

## Verification

`test-quantize-fns` compares grouped dots byte for byte with the existing CPU dots, including arbitrary codebook entries, signs, scales, zero activations, all group sizes, odd IQ4_NL block counts, and the full signed Q8 byte range.

`test-backend-ops -b CPU -o MUL_MAT_ID,MUL_MAT_ID_FUSION,MUL_MAT_ID_SHARED -p '(^type_a=(iq2_xs|iq3_xxs|iq3_s|iq4_nl),.*m=65,|^type=(iq2_xs|iq3_xxs|iq3_s|iq4_nl),n=[1-8],clamped=)'` exercises routed math, shared gate/up inputs, and strided expert weights. Enable each control for the candidate run. The routed cases require zero numerical error with the new scheduling or with `GGML_CPU_TILED_MM=0`; use the latter for isolated-control comparisons against stock vector dots.

`test-quantize-perf --op iq_grouped --size 4096 -i 25` compares the current dots with grouped dots in alternating order over 256 weight rows. It reports kernel timings for one through eight tokens. These are synthetic CPU timings, not model throughput measurements.

## CUDA host split executor

The Flash-Next CPU split uses its own pool rather than MUL_MAT_ID. The REUSE, SCHEDULE, and Q8_REUSE controls above do not select its kernels. Its existing ROWS control already reuses IQ weights for four tokens, and it already shares gate/up input conversion and schedules row tasks across experts. PIN can indirectly change which physical cores it reserves.

Two additional split controls are off by default:

- `GGML_MOE_CPU_SPLIT_Q8_REUSE=1`: quantize each distinct input row once per layer job, sharing that image across routed experts. Both device publishers supply input IDs; per-slot inputs keep distinct IDs. Hidden inputs remain separate per route.
- `GGML_MOE_CPU_SPLIT_REUSE8=1`: with `GGML_MOE_CPU_SPLIT_ROWS=1`, group up to eight tokens per IQ up/gate weight decode instead of four. AVX-512 single-token dispatch still takes precedence when requested.

`test-moe-cs-bw 0 0 4 check` runs 96 CPU-only pool jobs at widths one through eight with shared and distinct inputs, and prints a hash of every output byte. Compare the four combinations of these controls from one binary. `test-moe-iq-rows 0 1` checks the grouped kernels against the stock dots through width eight. Run CPU tests on the build rental, not on an inference host. These checks are not model throughput qualification.
