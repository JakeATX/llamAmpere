# Qwen3.8-27B SM86 optimization handover

Updated: 2026-09-03 (see the 2026-09-03 addendum at the end for the current state; the sections above it are kept as the historical record and corrected only where they would mislead)

## Read this first

The production target is Qwen3.8-27B on one RTX 3090/3090 Ti (`sm_86`), full
GPU offload, `--parallel 1`, Flash Attention, Q8_0 K cache, Turbo3 V cache, and
native MTP3. As of 2026-09-03 the product quant is **ATX-IQ4_XS-M**
(`jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF` on Hugging Face; renamed from the working
name ATX-4-XS on 2026-09-03, which the GGUF filename, tensor map and every log below keep: IQ4_XS base, M-pattern upgrades, 4.56 bpw): measured populated 200K at
20.9 GiB ready and a 245,760-token window with a 240K prompt at 22.1 GiB ready,
on the `main` branch of the renamed repo. Q3_K_XL remains the reference quant
and still fits its full 262,144 window.

The accepted product is stable. The next high-priority research task is a
**single fused CUDA chunked Gated DeltaNet (GDN) prefill kernel using the
existing WY/triangular algorithm**, followed by the larger Q3_K_XL replacement-
layout/narrow-integer-MMA decode project.

Do not conflate the two objectives. Chunked WY/GDN is a prompt-ingestion/TTFT
project: multi-token chunking is unavailable during ordinary singleton decode,
where GDN has measured only about 1--2% of GPU time. It will not by itself fix
the smaller 200K decode gain. Long-context generated-token throughput still
requires Q3-specific MMVQ work and/or a new attention dataflow that reduces the
ever-growing KV scan.

Do not repeat the graph-level chunked-GDN screen, generic large-N MMQ routing,
or the exact local Q3 unpack-reuse family. They have already been measured.

## GitHub and branch map

Fork remote for v0.2 (2026-09-07): `https://github.com/JakeATX/llamAmpere`. The v0.1 remote
(renamed 2026-09-03; the old `JakeATX/llama.cpp` URL redirects) is kept as the historical record:

`https://github.com/JakeATX/llama-cpp-qwen-ampere`

The important branches are:

- `main` (default): the product. `fdfea8123` = the accepted product merged with
  TheTom's `feature/turboquant-kv-cache` as of 2026-09-03, validated
  output-identical; later commits on `main` are docs plus the prompt-cache disk
  tier (`b8c4f3298`). Build from here;
- `perf/qwen38-sm86-decode-product` and `perf/qwen38-sm86-prefill`: both at
  `fdfea8123`, kept for existing links (historical tips: `c387d38b2`, `dd66db0cc`);
- `research/qwen38-sm86-100k-decode-frontier`: decode experiments and this handover; source tip before the handover commit was `2795680c47ce02f83b225ef60c658994d6495c28`;
- `research/qwen38-sm86-ub512-1024-prefill-frontier`: prefill frontier experiments, tip `685d66faefe1dd754561776042c4c0ead08e0107`;
- `archive/qwen38-sm86-fast-mmvq`: rejected optional non-exact MMVQ path, `aef2a435527ce4761bbde3218bae4093899b9c81`.

All experiment commits are isolated and bisectable. Rejected candidates retain
default-off flags or a subsequent revert.

## Local machine map

Source worktrees:

- accepted/main development tree: `/home/jake-k/TheTom-llama-cpp-turboquant`;
- **current product worktree (branch `main`, validated `build-sm86/`)**:
  `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-merge`;
- clean pre-merge product binary (3612acca6) used as the A/B baseline:
  `/home/jake-k/TheTom-llama-cpp-turboquant-atx4xs-eval/build-sm86/bin/llama-server`;
- GDN research tree (branch `research/qwen38-sm86-fused-gdn-chunk`, NOT merged with upstream yet):
  `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-fused-gdn`;
- decode frontier: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-decode`;
- prefill frontier: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-prefill`;
- archived FAST diagnostic: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-fast-diag`.

Models:

- **ATX-4-XS (product)**: `/home/jake-k/qwen38-bench/models/atx4xs/Qwen3.8-27B-ATX-4-XS.gguf`
  (sha256 `5cf05ad9...963b`, 15,588,551,712 bytes; map `tensor_types_ATX-4-XS.txt` in the same directory;
  previous versions `.v1.txt`/`.v2.txt` of the map, and `Qwen3.8-27B-ATX-4-XS-q8ab-fixed-7a92b152.gguf`
  is the other agent's copy of the intermediate build);
- BF16 source + Unsloth imatrix for rebuilding: `models/qwen38_27b_bf16/`, `models/unsloth_qwen38_27b_ud_iq4_xs/imatrix_unsloth.gguf`;
- Q4_K_M: `/home/jake-k/qwen38-bench/models/unsloth_qwen38_27b_ud_q4_k_m/Qwen3.8-27B-UD-Q4_K_M.gguf`;
- Q3_K_XL: `/home/jake-k/qwen38-bench/models/unsloth_qwen38_27b_ud_q3_k_xl/Qwen3.8-27B-UD-Q3_K_XL.gguf`.

Publication:

- article (markdown): `/home/jake-k/qwen38-bench/ARTICLE_3090_200K_ATX4XS.md`; the web version with charts is a private
  claude.ai artifact (`f6394cdb-37e7-450f-89d2-e61ee0461011`); its HTML source lives in the session scratchpad, not in Git;
- HF model card = the artifact's "Run it" section; keep the three in sync.

Benchmark and audit root:

`/home/jake-k/qwen38-bench/kernel_cache_investigation`

Start with these local documents:

- `frontier/QWEN38_SM86_FRONTIER_FINAL.md`: consolidated frontier explanation and next-work ordering;
- `frontier/FRONTIER_LEDGER.csv`: every frontier experiment, commit, flag, result, and artifact directory;
- `frontier/FRONTIER_EXECUTION_PLAN.md`: lifecycle, correctness rules, and paper-trail requirements;
- `frontier/P3_gdn_chunk/`: chunked-GDN hypothesis, raw results, analysis, decision, and commands;
- `frontier/D2_q3_narrow_mmvq/reopen/INITIAL_DEV_REPORT.md`: most recent Q3 initial-development result;
- `frontier/DF_decode_portfolio/ANALYSIS.md`: final combined Q3 portfolio;
- `Q3_K_XL_MMVQ_FOLLOWUP.md`: Q3-specific follow-up specification;
- `QWEN38_SM86_DECODE_FINAL.md`: accepted decode qualification;
- `QWEN38_SM86_PREFILL_FINAL.md`: accepted prefill qualification;
- `prefill/CONTEXT_CAPACITY_UB1024.md`: direct maximum-context results;
- `prefill/runtime_200k_100tok_ab/RUNTIME_200K_100TOK_AB.md`: existing 200K runtime canary.

Raw Nsight reports and JSON results are intentionally not copied into Git.
They remain under the paths above on this machine.

## Reproduce the build

From either frontier worktree:

```bash
cmake -S . -B build-sm86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_NATIVE=ON

cmake --build build-sm86 -j8 \
  --target llama-server llama-bench llama-cli test-backend-ops
```

The decode frontier's working build is:

`/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-decode/build-sm86`

The prefill worktree has historically used `build-cuda`; either name is fine if
the CMake architecture is explicitly 86. Do not use a stale build directory
after switching between candidate commits without rebuilding `libggml-cuda.so`.

Before timing, check for competing CUDA jobs with:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Desktop compositor/browser allocations are normal. Do not kill unrelated user
processes. Keep clocks, power, model, prompt, sampler, seed, and output length
fixed for A/B tests.

## What is already banked

Original comparison baseline means TheTom's TurboQuant+ commit
`f97400563641837efeeb2c6b9b45badfc5d35530`, not upstream stock llama.cpp.

The accepted product at `c387d38b` includes grouped Q8/Turbo3 MTP verification
attention and exact Q4_K/Q5_K narrow-verification reuse. Directly measured
results include:

- final EXACT versus original TurboQuant+ across ten bounded three-way cells:
  +17.92% median, +16.41% geometric mean, ten wins in ten cells;
- accepted narrow-MMVQ contribution against the accepted attention stack:
  +4.46% equal-cell median at 16K, +4.37% in the 2K/64K expansion, and +3.91%
  on held-out 100K RAG;
- accepted UB1024 prefill configuration versus UB512: +4.40% median;
- production-shaped prompt throughput confirmation: +5.92% to +7.72%.

MTP3 remains the product default. MTP3 and MTP4 were effectively tied in the
matched aggregate; MTP3 was selected for lower round latency, waste, and
workspace. MTP4 remains useful on explicitly qualified high-acceptance tasks.

## Why the 200K speedup looks smaller

The existing 200K number is not an apples-to-apples estimate of the complete
64K gain:

1. It is one 100-output-token canary on a highly repetitive prompt, not the
   multi-workload, multi-seed qualification used for aggregate claims.
2. It measures accepted `c387d38b` against original `f9740056` at one MTP3
   schedule. A single near-ceiling acceptance trajectory has low statistical
   power and different weighting from the broader cells.
3. The accepted narrow-MMVQ work removes a mostly context-independent amount of
   repeated weight unpacking per verification round. Attention, by contrast,
   scans a KV history whose cost grows with context.
4. Measured attention share rose from about 26--28% at 64K to 35.9% at 100K
   and 40.7--43.8% at 140K. The fixed MMVQ saving is therefore diluted at 200K.

The 200K canary still passed: 51.538 versus 50.027 tok/s, +3.02%, identical
output and acceptance, and identical 22,104 MiB peak VRAM. Treat +3.02% as a
capacity/performance canary, not a precise 200K aggregate. A proper estimate
would need the same prompt/workload manifest and 512--1024 output tokens at 64K
and 200K, paired and alternated. Do not rerun it merely to seek a larger number.

Also distinguish the accepted 64K stack from the historical maximum stack. The
accepted safe 64K coding measurements were approximately +10.94% for MTP3 and
+8.54% for MTP4. The often-cited +21.44% 64K/MTP3 result included the archived
fast MMVQ implementation that changed the continuation and verification-round
count and was removed from the product. Comparing that historical +21.44%
directly with the accepted Q3 200K +3.02% mixes different kernels, quants,
workloads, and qualification standards.

## Capacity facts

With Q8_0 K, Turbo3 V, MTP3, and UB1024:

- Q3_K_XL passed a 200,000-token prompt at about 22.1 GiB peak;
- Q3_K_XL passed 215,000 tokens at 22,764 MiB peak;
- Q3_K_XL failed the 225K request after 206,848 tokens;
- Q4_K_M passed 149,000 tokens and failed at 155,000;
- Q4_K_M cannot initialize the full 200K target/MTP configuration.

The bullets above predate the V1/V2 VRAM fixes (2026-09-02) and the merge. Current facts on `main`:

- Q3_K_XL: 200K populated at 19,204 MiB ready; the full 262,144 window with a 200K prompt at 20,922 MiB;
- ATX-4-XS: 200K populated at 21,372 MiB ready / 21,422 peak (prefill 694 tok/s, decode 65 tok/s with the window full);
  245,760 window with a 240K prompt at 22,600 MiB ready / 22,634 peak. 262K not probed;
- Q4_K_M: 200K loads at 22,208 MiB ready; its max-context cell in the article is still an extrapolation (~230K).

Any replacement/repack design must preserve the Q3 200K tier. Replace the
device layout of hot tensors rather than retaining a multi-GiB duplicate.

## Completed frontier experiments

### Decode attention

`b6d539ceb` implemented warp-owned packed-byte-sharing singleton GQA attention.
It was correct and spill-free but made the kernel 81--87% slower. `b73c8eb4e`
split each query head across two consumer warps; it remained correct/spill-free
but increased 100K attention time by 26.2% and regressed the request 2.42%.

Do not repeat “put more query heads in one CTA” without a new occupancy/traffic
argument. The existing accepted grouped verification path remains enabled.

### Q3 exact narrow reuse

The relaxed initial-development round implemented/restored:

- IQ3_S: `d9b9b0f0f`, flag `GGML_CUDA_SM86_IQ_REUSE`;
- IQ4_XS: `ca2b07ad6`, flag `GGML_CUDA_SM86_IQ4_REUSE`;
- IQ3_XXS: `a04f21195`, flag `GGML_CUDA_SM86_IQXXS_REUSE`;
- IQ2_S: `25ad45d27`, flag `GGML_CUDA_SM86_IQ2S_REUSE`.

IQ3_XXS and IQ2_S passed 32/32 CUDA-versus-reference cases each. Width-five
microkernels improved about 1.1% and 1.45%, but the clean combined 64K/MTP4
test measured -0.17% mean and -0.11% median with identical output, acceptance,
and VRAM. Keep all four flags off.

`2bb855846` routed Q3 width four/five through the existing Ampere large-N MMQ
engine. It passed its reference cases but regressed 20--25% at width four and
6--13% at width five. `47b3ef28f` reverts it. The next path needs a genuine
narrow N=8 layout/kernel, not a dispatch change.

### Prefill MMQ and attention

- async activation-stage MMQ `1c235461c`: correct, approximately -0.3% E2E;
- exact-48 GQA attention `20c3bc440`: correct, only 0.25% attention reduction;
- direct Q8/Turbo3 prefill attention `27bc49fb6`: correct canary but 2.97x
  slower attention and -12.33% prompt throughput because compressed KV was
  decoded repeatedly for successive query tiles;
- paired prepared-Q8_1 `af87de28a`: +0.36% Q4 / +0.26% Q3 at 16K, then -0.32%
  median prompt throughput in the 64K portfolio. Leave it off.

### GDN graph reference

`26f20a32d` exposed the existing graph-level chunk algorithm behind
`GGML_CUDA_SM86_GDN_GRAPH_REFERENCE`. A Q/K head-broadcast assumption was fixed
so the path could run Qwen. The canary was numerically/visibly correct, but Q4
16K UB1024 prompt throughput fell 13.44%. `9e3b868ca` reverts it.

This rejects the graph realization, not chunked GDN mathematics.

## Priority 1: fused CUDA WY/triangular GDN

### Exact starting points

Work on a new branch from accepted prefill `dd66db0cc`, not from a tree with an
enabled frontier experiment. Suggested branch:

`research/qwen38-sm86-fused-gdn-chunk`

Inspect these files first:

- `ggml/src/ggml-cuda/gated_delta_net.cu`: current fused operator; it loops
  serially over `t < n_tokens` and contains `TODO: Add chunked kernel for even
  faster pre-fill`;
- `ggml/src/ggml-cuda/gated_delta_net.cuh`: CUDA operator interface;
- `src/models/delta-net-base.cpp`, especially `build_delta_net_chunking()`:
  working graph expression of the decay mask, lower-triangular solve, WY-like
  transformed values/keys, outputs, and final state;
- `src/models/delta-net-base.cpp`, `build_delta_net()`: singleton versus
  multi-token dispatch;
- `src/models/qwen35.cpp`: Qwen broadcast and fused-GDN integration;
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch and recurrent-state snapshot-copy
  fusion.

The Qwen path has 48 recurrent GDN layers. Preserve FP32 recurrent state and
leave singleton decode/rollback untouched.

### Implementation sequence

1. Add `GGML_CUDA_SM86_GDN_CHUNK=0|32|64`; default zero must reproduce the
   accepted product exactly.
2. Dispatch only multi-token prefill on SM86 and only the qualified Qwen shape.
   Keep the serial fused and generic graph fallbacks.
3. Implement chunk 32 first. One CTA should own a sequence/head/chunk unit and
   keep the compact triangular/WY factors and state tile on chip as far as the
   SM86 register/shared-memory budget permits.
4. Fuse the operations currently materialized by the graph: decay-prefix
   preparation, lower-triangular system/factor construction, transformed K/V,
   within-chunk output, and final FP32 state update.
5. Use Tensor Core-friendly small matrix operations where they reduce total
   work, but do not force every triangular operation through MMA if packing and
   padding dominate.
6. Avoid writing full `[chunk, chunk, head, chunks]` intermediates to global
   memory. This is the principal difference from the rejected graph path.
7. Prove chunk tails and state boundaries before measuring. UB512 has 16
   chunk-32 groups and UB1024 has 32 per microbatch.
8. Try chunk 64 only after chunk 32 is correct and faster. It is the one bounded
   refinement, not a broad tile sweep.

### Correctness and quality

Strict token identity is diagnostic, not the product gate. Preserve represented
model values and FP32 state, but tolerate floating-point scheduling differences
if all of the following pass:

- fixed-token replay at actual chunk dispatch with cross-entropy/PPL ratio,
  mean/p95/p99/max logit KL, and target-token log-probability deltas;
- final recurrent-state comparison at every chunk boundary;
- ragged lengths 1, 31, 32, 33, 63, 64, 65 and actual UB tails;
- frozen singleton decode and every MTP partial-rejection/rollback canary;
- Compute Sanitizer memcheck, racecheck, and synccheck;
- independent coding, code-edit, reasoning, and retrieval canaries under the
  production sampler.

### Performance gates

Start with Q4 and Q3 at 16K, UB1024, five alternating repetitions. Record GDN
kernel time as well as prompt tok/s. Advance if the first correct fused kernel
is consistently positive or reduces total GDN time materially. Product
promotion still requires at least 25% total-GDN reduction and approximately
2.5% end-to-end prefill improvement, or a directly dependent portfolio that
clears the ordinary aggregate gate.

Qualify survivors at UB512 and UB1024, 16K/64K/100K, both quants. Include TTFT,
short fixed decode tail, peak VRAM, and the 200K Q3 capacity canary. Do not add
component percentages; measure the combined runtime directly.

### Reproduction templates

Baseline/screen:

```bash
GGML_CUDA_SM86_GDN_CHUNK=0 build-sm86/bin/llama-bench \
  -m Q3_MODEL -p 16384 -n 0 -b 4096 -ub 1024 \
  -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 -fa on \
  -r 5 -o json --progress -fitt 0

GGML_CUDA_SM86_GDN_CHUNK=32 build-sm86/bin/llama-bench \
  -m Q3_MODEL -p 16384 -n 0 -b 4096 -ub 1024 \
  -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 -fa on \
  -r 5 -o json --progress -fitt 0
```

Profile only after the screen proves dispatch and correctness:

```bash
GGML_CUDA_SM86_GDN_CHUNK=32 nsys profile \
  --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
  --stats=false --cuda-flush-interval=1000 --cuda-graph-trace=node \
  -o /tmp/qwen38_gdn_chunk32 \
  build-sm86/bin/llama-bench -m Q3_MODEL -p 16384 -n 0 \
  -b 4096 -ub 1024 -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 \
  -fa on -r 1 -o json --progress -fitt 0
```

Use `/home/jake-k/qwen38-bench/kernel_cache_investigation/analyze_nsys.py`
to normalize the Nsight report.

## Priority 2: Q3 replacement layout plus narrow IMMA

After or independently from fused GDN:

1. Reprofile Q3 at 64K, 100K, and 200K to rank exact IQ type/tensor/shape calls.
2. Repack one hottest width-five tensor family at model load into an equivalent
   SM86 consumption order. Replace its device representation; do not duplicate.
3. Implement a dedicated physical N=8 integer-MMA kernel, width five first.
   Eight derives from SM86 MMA fragment geometry; width five uses five of eight
   output columns and width four uses four.
4. Producer warps stage/prepare packed weights and Q8_1 activations; consumer
   warps execute MMA and apply the exact scales/mins in FP32.
5. Extend to width four/MTP3 only if width five proves the dataflow.
6. Under the relaxed development rule, any reproducible positive E2E canary may
   receive one refinement. Final product promotion remains +2% aggregate or
   +3% at 64K--200K with no important regression and preserved 200K capacity.

Use the existing reference shapes/tests introduced in `f002298a5` and
`268d2547a`, and the raw microkernel evidence under
`frontier/D2_q3_narrow_mmvq/reopen/`.

## Paper-trail requirements

For every new experiment create an artifact directory under:

`/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/`

Include:

- `HYPOTHESIS.md` before coding;
- exact baseline and candidate SHAs;
- `commands.txt`;
- raw benchmark/profile/sanitizer output;
- normalized measurements;
- compiler resource report: registers, shared memory, occupancy, spills;
- `ANALYSIS.md` explaining the mechanism in plain English;
- `DECISION.md` stating accept/refine/reject;
- one append-only row in `FRONTIER_LEDGER.csv`.

Do not erase unfavorable data or repeatedly rerun until a result improves.
Measure the final portfolio directly. Preserve rejected source in an isolated
commit followed by a revert or keep it behind a default-off flag.

## Current safe recommendation

(Superseded 2026-09-03; see the addendum for the current command.) Use accepted
EXACT with MTP3, Q8_0 K/Turbo3 V, and UB1024. Use Q3_K_XL for 200K. Leave
these frontier flags off:

- `GGML_CUDA_SM86_Q8_T3_GQA_WARP`;
- `GGML_CUDA_SM86_IQ_REUSE`;
- `GGML_CUDA_SM86_IQ4_REUSE`;
- `GGML_CUDA_SM86_IQXXS_REUSE`;
- `GGML_CUDA_SM86_IQ2S_REUSE`;
- `GGML_CUDA_PREPARED_Q8_1`;
- `GGML_CUDA_SM86_GDN_GRAPH_REFERENCE`.

The handoff objective is not to retune these flags. It is to build a genuinely
fused chunked-GDN CUDA implementation and, secondarily, a genuinely narrow Q3
layout/IMMA dataflow.

## 2026-09-01 addendum: P3C serial-GDN result and revised priorities

Before the fused WY kernel was started, the serial fused GDN kernel was
characterized instead of assumed. It was latency-bound (about 4,400 GPU cycles
per token step for roughly 65 instructions per warp) and launched 1.83 waves of
fixed-length serial blocks. Giving each warp four recurrent-state columns,
with the per-column FP32 operation order unchanged, fixed both. Full record:
`qwen38-bench/kernel_cache_investigation/frontier/P3C_gdn_serial_ilp/`
(hypothesis, commands, raw data, analysis, decision, qualification plan).

Measured against `dd66db0cc` binaries, Q8_0 K / Turbo3 V, `-b 4096`:

| cell | baseline tok/s | P3C tok/s | delta |
|---|---:|---:|---:|
| Q3 UB1024 16K | ~1405 | 1483 | +5.6% |
| Q3 UB1024 64K | ~1106 | 1153 | +4.3% |
| Q3 UB1024 100K | 958 | 987 | +3.0% |
| Q3 UB512 16K / 64K | 1353 / 1059 | 1409 / 1098 | +4.1% / +3.7% |
| Q4 UB1024 16K | 1427 | 1499 | +5.1% |
| Q4 UB512 16K / 64K | 1368 / 1022 | 1434 / 1113 | +4.9% / +8.9% |
| Q3 200K server canary (prompt) | 652 | 679 | +4.1% |

GDN kernel time per 1024-token launch: 2219 us to 1582 us (-28.7%). Decode
tails, output hashes, 200K peak VRAM (22,104 MiB), and MTP rollback state
hashes are unchanged; perplexity and saved-logit KL are identical; SASS
per-column FFMA/FADD/FMUL counts match the original. Compute Sanitizer is
clean. `GGML_CUDA_SM86_GDN_COLS=1` restores the original kernel.

Consequences for the priorities above:

1. The fused chunk-32 WY kernel (Priority 1) now has a 1.58 ms baseline and
   GDN is about 11% of 16K prefill kernel time, so its end-to-end ceiling is
   roughly half of what this document implied. It remains worthwhile, but the
   open design question is numerical policy: chunked WY only wins on tensor
   cores with reduced-precision operands against the FP32-state rule.
2. Nsight Compute works under `sudo`; `ERR_NVGPUCTRPERM` only blocks
   unprivileged users. Capture stall reasons before any further GDN
   refinement or attention pipeline work.
3. Decode attention at 100K: the grouped verification kernel moves about 18%
   of DRAM bandwidth (812 us per layer for ~149 MB of KV) with 8 warps per SM
   and single-stage synchronous tile loads; the singleton kernel about 10%.
   The q8_0 K tile loader does a runtime division and three sub-word loads
   per element pair. Decide load-bound versus issue-bound first
   (`frontier/A0_attention_discriminator/`), then fix loaders (exact) before
   any pipeline or head-ownership redesign.
4. `tests/test-backend-ops.cpp` now covers the production GDN shape
   (16 to 48 head broadcast, head size 128, multi-token, K=4 snapshots).

## 2026-09-02 addendum: A1 decode attention result

With Nsight Compute now usable (passwordless `sudo ncu` for the user, env
passed through an `env` target), the compressed-KV decode attention was
characterized instead of guessed at: the GQA-packed MMA path was
latency-bound (load latency 47% of stalls, 8 warps/SM, DRAM 20%), the vector
singleton path issue-bound (6x per-query-head rescan absorbed by L2 but 5.5x
the instructions). Two exact changes landed as product defaults:
`88be730f5` (vectorized q8_0/turbo3 tile loaders) and `a0a53e354` (cp.async
staging of packed tiles one tile ahead). Outputs are identical; decode at
MTP3 improved +2.3% (Q3 64K) to +10.9% (Q3 140K), Q4 +3.6% to +7.4%, and
+9.6% at the 200K canary. Record: `frontier/A1_decode_attention_mma/`.

Routing widths 1-2 through the (2,8) MMA instance (`GGML_Q8_TURBO3_MMA_MIN_Q=1
GGML_Q8_TURBO3_MMA_NCOLS1_MIN=2`) removes a further ~8 points of GPU time per
round at 100K but is not bit-identical; it stays opt-in until the replay
KL/PPL and rollback gates pass. Measurement lesson: back-to-back server
runs show a 4-7% order effect; alternate order and cool down between runs.

## 2026-09-02 addendum: A2 decode idle (CUDA graphs, output buffer)

Nsight traces with OS-runtime calls showed ~13% of decode wall time idle at
100K: one pinned output-buffer reallocation per request (~60 ms) when the
first verify batch needed 4 logits rows, and CUDA graph capture plus
instantiate for every graph shape on every request, because ggml-cuda kept
one graph per graph identity (verify width and accepted-draft count vary)
and evicted graphs after 10 s idle. `7296ede65` keys graphs by shape,
retains them for 300 s (`GGML_CUDA_GRAPH_EVICT_S`), and reserves 8 output
rows up front. Exact; +5.4% decode at Q3 100K, +1.7% at Q4 64K, 200K within
noise. `GGML_CUDA_GRAPH_DEBUG=1` prints the first changed graph node per
call. Record: `frontier/A2_decode_graphs/`.

Also settled: MTP4 gives nothing over MTP3 even with the cheaper verify
launch; the routing candidate (widths 1-2 on the MMA path) fails the
generated-task quality protocol by a small consistent margin and stays
opt-in; the exact path to its speed is an FP32-accumulating (2,8) instance.

## 2026-09-03 addendum: upstream merge, ATX-4-XS corrections, capacity, agent-session curve, cache tiers

### Runtime: merged with TurboQuant+ (accepted, pushed)

`fdfea8123` on `main` = product `3612acca6` + `origin/feature/turboquant-kv-cache`
`1208c5956` (117 upstream commits: MMVQ shared-quantize cache and elementwise
chain fusion (#343, both default-on), persistent q8/TQ pre-rotation buffers,
`--gdn-replay` ingredient replay (opt-in), turbo4 vec fix; the rest is
TQ4_1S/MoE/AMD/Metal/Vulkan work this project does not use). Two real
conflicts: `gated_delta_net.cu` (our ILP kernel kept, gained upstream's
`emit_ingredients_t` template parameter, and is skipped when `emit_mode==1`
because it writes full state snapshots, not ingredients; the snapshot layout
for `emit_mode==0` is unchanged upstream) and `test-backend-ops.cpp` (both
sides' cases kept). Validation on the merged build: GDN 48/48, FLASH_ATTN_EXT
7664/7664, MUL_MAT 1697/1697; Q3 200K capacity canary identical
(`944b399e...`, ready 19,204 MiB); ATX-4-XS decode A/B old vs merged, 100K two
alternated pairs and 64K one pair: identical output hashes and draft
acceptance, +0.16% / +0.29% tok/s (noise). Record:
`frontier/R1_turboquant_merge_20260903/`. The research branch
`research/qwen38-sm86-fused-gdn-chunk` was NOT merged with upstream.

### ATX-4-XS corrections (same name, overwritten on HF)

1. The 96 GDN `ssm_alpha`/`ssm_beta` vectors were IQ4_XS; both Unsloth
   references keep them Q8_0. Fixed (0.01 GiB).
2. The eight attention K/V tensors Q4_K_M keeps at Q8_0 (V in layers 11, 27, 31,
   51, 55, 59, 63; K in 31) were Q6_K. Now Q8_0 (+17.6 MiB). ATX-4-XS now holds
   exactly the same 106 tensors at Q8_0 as Q4_K_M.
3. The only remaining Q4_K tensor is `token_embd` (1.27 B params, 682 MiB). It
   never touches the GPU (llama.cpp keeps the input layer on the host), has no
   imatrix data (lookups are not matmuls), and Q4_K_M keeps it at Q4_K too.
   Moving it to IQ4_XS would be a small quality downgrade for 38 MiB; Q8_0
   would be the quality-maximal choice at +606 MiB of host RAM/file size.

Final file: 14.52 GiB, IQ4_XS 71.4% / Q5_0 16.1% / Q6_K 7.4% / Q4_K 4.6% /
Q8_0 0.5% of weight bytes. Verified by header diff: only the intended tensors
changed between builds. **Gotcha:** this fork's `llama-quantize` takes the
FIRST matching regex in `--tensor-type-file`, not the last; overrides must
replace existing lines. The `-mtp8` variant was not rebuilt (user decision).

### Long-context measurements on the final file (merged runtime)

- Populated 200K: 21,372 MiB ready, 21,422 peak, prefill 694 tok/s, decode 65 tok/s.
- 245,760 window with a 240K prompt: 22,600 MiB ready, 22,634 peak, output identical.
- Agent session, 99,922-token agentic prompt (OpenHands SWE trajectory) to
  245,686 tokens of context, 120,186 generated over 288 ChatML turns (260
  tool-result turns, 28 follow-up tasks), EOS never suppressed: cumulative
  decode 51.8 tok/s; 10K-window tok/s 61.5 at 114K context, 55.6 at 175K,
  47.3 at 198K, 47.1 at 245K; draft acceptance 76% overall (65-88% per window);
  first prefill 982 tok/s; peak VRAM 23,678 MiB. Harness: session scratchpad
  `longgen2.py` (checkpoints every 1K tokens with per-window acceptance from
  `timings_per_token`), results under
  `frontier/R1_turboquant_merge_20260903/raw/longgen_agentic_multiturn_100k_140k/`.
- Two harness lessons. (a) `ignore_eos=true` on a single answer collapses into
  a repeated sign-off within ~12K tokens and reports ~97-100% draft
  acceptance; such curves are not workload data and were discarded. (b) On the
  agentic fixtures the model behaves as an agent step machine (reason + tool
  call, then EOS); a multi-turn harness must answer tool calls with `[TOOL]`
  result stubs and use real chat turns (ChatML, append-only so the KV prefix
  stays valid), not raw transcript continuation.

### 5K/5K comparison against vLLM (other agent's run, kept in the article)

Same prompts, native MTP-3 on both: llama.cpp ATX-4-XS 70.8 / 80.2 / 81.9 tok/s
(agentic / coding / RAG) at 17.1 GiB VRAM vs vLLM W4A16 AWQ 81.8 / 86.2 / 88.2
at 22.1 GiB with a 12K window and 19.6% more effective bits. Different
questions: vLLM wins single-request decode at 5K, cannot hold 200K on this card.

### Prompt cache, checkpoints, and the new disk tier

- `--cache-prompt` (default on) keeps the conversation's KV in the slot; a
  new turn on a 200K conversation costs only the new tokens.
- Context checkpoints (`--ctx-checkpoints N --checkpoint-min-step S`) are
  recurrent-state snapshots in host RAM, taken at every user turn and
  otherwise no closer than S tokens; they are the only way to "rewind" the 48
  GDN layers after an edit, regenerate, or harness compaction. **Measured size
  on this model: 150 MiB + 1.47 KiB per token of position** (the MTP drafter's
  single-layer KV is saved with the recurrent state; `PARTIAL_ONLY` flag), so
  ~495 MiB at 240K. Full coverage of the 245,760 window under 8 GiB is 24
  checkpoints at 10,240 spacing (worst replay ~10 s); denser spacing
  multiplies RAM (60 at 4,096 is ~20 GiB at the deep end).
- `--cache-ram` is a separate host-RAM budget for parking a displaced
  conversation (KV + its checkpoints). A conversation larger than the budget
  is skipped, not parked, and re-prefills on return; 200K is ~7.4 GB of KV,
  245K ~9.1 GB. Parking transiently duplicates the state in host RAM.
- **New on `main` (`b8c4f3298`): `--cache-disk-path DIR [--cache-disk-limit MiB]`**,
  a disk tier under the RAM prompt cache. RAM evictions and prompts too large
  for the RAM budget are serialized (tokens, target/draft state, checkpoints;
  format `LCPC` v1) and restored on a later matching request under the same
  prefix-similarity rule; the directory is indexed at startup. Tested with
  three rotating conversations at RAM budgets below and equal to one
  conversation: spill, restore with `cache_n` = full prefix (~360 ms for
  539 MiB, ~1.5 GB/s), restart re-index, greedy output identical to an
  uninterrupted control; flag-off path unchanged. Manual test:
  `tools/server/tests/test_prompt_cache_disk_tier_manual.py`. Multimodal
  prompts are not spilled.

### Current recommended server command (single user)

```bash
GGML_Q8_TURBO3_MMA_FUSED=1 ./build-sm86/bin/llama-server -m Qwen3.8-27B-ATX-4-XS.gguf \
  -c 245760 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 \
  --parallel 1 --jinja --fit off \
  --cache-prompt --cache-ram 8192 --cache-disk-path /fast-nvme/llama-cache --cache-disk-limit 65536 \
  --ctx-checkpoints 24 --checkpoint-min-step 10240 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
  --spec-draft-type-k q8_0 --spec-draft-type-v turbo3
```

`-c 204800` for 200K at 20.9 GiB ready if anything else needs the card. Host
RAM for the caches at the deep end: up to ~8 GiB of checkpoints plus the 8 GiB
park space. `--parallel 1` is structural: a second slot needs its own ~7 GiB
KV at 200K and its own MTP drafter, and none of the kernel tuning covers it.

### Operations lessons from 2026-09-03

- The host has 30 GB RAM and 8 GB swap. `llama-quantize` from BF16 peaks ~10 GB
  anon RSS; run it under `systemd-run --user --scope -p MemoryMax=10G` and
  never concurrently with a vLLM job or the desktop above ~15 GB. Two OOM
  incidents today: one killed the terminal scope (Claude session included), one
  filled swap completely (PSI full ~20%) when quantize + vLLM load overlapped.
- Another agent (hermes) starts GPU jobs from systemd user units without
  warning (`qwen38-atx4xs-q8fix-eval`, `qwen38-compact-compare-trace`). Gate
  every GPU chain on `nvidia-smi` memory < 3 GB and no `VLLM::EngineCore`.
  Use bracketed `pgrep`/`pkill -f "[c]hain.sh"` patterns; a bare pattern
  matches the calling shell and kills it (happened twice).
- Long jobs: `setsid nohup ... & disown`, log to a file, watch with a
  persistent monitor; back-to-back server A/Bs still need alternation and a
  30-60 s cooldown.

### Heretic abliteration LoRA (audited, not changed)

`models/atx4xs-heretic-adapter{-hf,.gguf}`: the GGUF is a faithful export (all
128 A/B pairs bit-identical to the safetensors after the grouped-to-tiled
column permutation on `ssm_out` lora_a; 83 B tensors are exactly zero because
trial 93's layer windows are o_proj 35-63, down_proj 48-62, out_proj 34-62).
It derives from the HF bf16 source, targets only ffn_down/attn_output/ssm_out,
and is unaffected by the quant corrections. The real problem is efficacy: the
136-trial study never got below 54/100 keyword refusals (baseline 84); the
pipeline's speed gate found it 5% slower and selected base. A new study, not a
re-export, is what would change that.

### Open items

- Merge `research/qwen38-sm86-fused-gdn-chunk` with upstream (same two
  conflict files expected, plus the research knobs).
- Q4_K_M max-context cell is still an extrapolation; ATX-4-XS at 262K not probed.
- The other agent's eval report lists the pre-correction ATX hash; its
  `run_vllm_single_request_compare.py` fails on a `NoneType` rounding.
- Optional: `token_embd` at Q8_0 as a quality-maximal ATX variant.
- Optional: rebuild the `-mtp8` variant with both corrections.

