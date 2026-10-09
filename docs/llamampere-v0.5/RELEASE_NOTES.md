# llamAmpere v0.5 release notes

llamAmpere is a llama.cpp fork tuned for the RTX 3090 / 3090 Ti (Ampere, SM86). It targets Qwen3.8-27B on one card and
uses the model's own MTP draft head for speculative decoding. v0.5 is a KV cache, prompt cache and speed release.

- **SJ-KVaRN** is a new compressed KV cache, in 4/4, 3/3 and 3/2 bit configurations. At 4/4 it has 41% lower KL than
  v0.4's tq5_0/turbo4 cache for 2% more cache memory.
- **Faster decode at 100K.** On our task benchmark at 100K depth, v0.5 decodes 8.4% faster than v0.4.
- **Prompt caching is on by default.** It has a RAM tier and a disk tier, and it works with every KV type, SJ-KVaRN
  included.
- **12 GB cards (RTX 3060)** get a 204,800-token context in about 11 GB, with the MTP drafter.

## What's new in v0.5

**KV cache**

- **SJ-KVaRN**, a new compressed KV cache (`-ctk sjkvarnN -ctv sjkvarnN`). There are three configurations:
  - 4/4: the lowest-KL cache at about tq5_0/turbo4 size.
  - 3/3t: for small cards.
  - 3/2t: for the most context per GB.

  It keeps a short full-precision window of recent tokens, and its built-in defaults need no tuning.
  `--sjkvarn-body-type auto` picks a trellis-coded body for 3/3 and 3/2 (the "t").
- SJ-KVaRN caches can be saved and restored, so they work with the prompt cache, slot save/restore and conversation
  resume.
- The MTP drafter runs over an SJ-KVaRN main cache with its own tq5_0/turbo4 cache, chosen automatically.
- tq6_0 keys and values decode faster.

**Speed**

- Faster decode at long context with tq5_0/turbo4 and with SJ-KVaRN 4/4.
- Faster SJ-KVaRN writes, and faster decode for 3/3t and 3/2t.
- Faster MTP verify decode on EXL3 models.
- Removed blocking host-to-device copies that an upstream change had added to every decode step (see Bug fixes).

**Prompt cache and server**

- **The prompt cache is on by default.** It has a RAM tier sized from the host's memory, and a 16 GiB disk tier. A
  conversation that comes back after another one used the slot, or after a server restart, resumes without a full
  re-prefill.
- **Draft depth.** The MTP drafter now runs at adaptive depth 3-4 by default. v0.4 used a fixed depth of 4.

**Small cards**

- **12 GB cards.** There is a tested command for 12 GB cards: a 2.3 bpw Swift 1.5 model, SJ-KVaRN 3/3t, a
  204,800-token context and the MTP drafter, in about 11 GB.
- New options that cut memory on small cards:
  - An 8-bit or 16-bit Gated DeltaNet recurrent state (`--cache-type-s q8_0|bf16|f16`).
  - A capped draft window for the drafter's cache (`--spec-draft-window N`).
  - A smaller draft micro-batch (`LLAMA_MTP_DRAFT_COMPUTE_LEAN=1`).

**Maintenance**

- A catch-up to upstream llama.cpp (0.6.0, ggml 0.26.0) and to TheTom's TurboQuant fork. New upstream features include:
  - an opt-in MoE expert cache (`--moe-cache-mib`);
  - KV rotation metadata in saved state;
  - greedy speculative decoding at temperature 0.

## Recommended settings for 24 GB cards (RTX 3090 / 3090 Ti)

Needs Linux, an NVIDIA card with 24 GB (RTX 3090 / 3090 Ti), the CUDA toolkit (tested with 12.4), CMake, git and a C++
compiler. Paste into a terminal; the model download is 15.6 GB.

```bash
git clone -b v0.5 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
```

`-DLLAMA_BUILD_BORINGSSL=ON` builds HTTPS support from source so `-hf` can download the model. Without it, a machine
that has no OpenSSL development files gets a build whose `-hf` fails with "HTTPS is not supported".

Then pick one of the four configurations below.

**Fastest decode (tq5_0 K / turbo4 V):**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk tq5_0 -ctv turbo4 -b 4096 -ub 1024 --parallel 1
```

**Lowest KL at about the same speed (SJ-KVaRN 4/4):**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1
```

**Smaller cache (SJ-KVaRN 3/3t):**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn3 -ctv sjkvarn3 --sjkvarn-body-type auto -b 4096 -ub 1024 --parallel 1
```

**Smallest cache (SJ-KVaRN 3/2t):**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto -b 4096 -ub 1024 --parallel 1
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). The MTP drafter, its vocabulary shortlist, its
cache types and the prompt cache are all on by default, so they need no flags.

Release check: on an RTX 3090 Ti that also drives a desktop, each of the four commands booted at 262,144 and generated
512 tokens. The whole-card peaks were:

| configuration | whole-card peak |
|---|---:|
| tq5_0/turbo4 | 21,934 MiB |
| 4/4 | 21,734 MiB |
| 3/3t | 20,732 MiB |
| 3/2t | 20,228 MiB |

These were short prompts. Full-depth numbers are under "Long context and memory".

## Recommended settings for 12 GB cards (RTX 3060 12 GB)

Same build as above. Paste the command below from the `llamAmpere` directory; the model download is 9.0 GB.

```bash
LLAMA_MTP_DRAFT_COMPUTE_LEAN=1 ./build-sm86/bin/llama-server \
  -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF \
  -hff ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP.gguf \
  -c 204800 --parallel 1 -ngl 99 -fa on -fit off -b 4096 -ub 512 \
  --no-context-shift --jinja \
  -ctk sjkvarn3 -ctv sjkvarn3 --sjkvarn-body-type sjkvarn4t \
  --sjkvarn-sink 128 --sjkvarn-sink-type f16 --sjkvarn-staging-type tq6_0 \
  --sjkvarn-tail 4096 --sjkvarn-tail-max 8192 \
  --cache-type-s q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --spec-draft-vocab-map docs/mtp-vocab/atx_65536.txt \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-window 8192 \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0
```

- **Fit.** With 3/3t at 204,800 context, the server uses 11,064 MiB after boot and a short reply. With a 203,568-token
  prompt plus 256 generated tokens, the whole card peaked at 12,052 MiB, including about 1 GB used by the desktop.
- **3/2t option.** With 3/2t (`-ctv sjkvarn2`), the same command uses 10,672 MiB after boot. It peaked at 10,690 MiB
  with a 203,568-token prompt plus 256 generated tokens. Use 3/2t if you need about 390 MiB more room. Its KL is about
  twice that of 3/3t.
- **Quality.** The 2.3 bpw model scores 77.71% on LiveCodeBench v6 (100 tasks, reasoning effort xhigh, temperature
  1.0, seven runs on rented 12 GB cards, 95% CI 75.74-79.69). That is 86% of Qwen3.8-27B BF16's 90.3%. Swift 1.5
  IQ4_XS-M scores 89.25% on our engine. Those LiveCodeBench runs used 3/2t.
- **Bigger cards.** On a 24 GB card, the IQ4_XS-M model with one of the commands above is the stronger choice.

## Terms used below

- **MTP draft head.** Qwen3.8 ships a small extra layer that predicts the next few tokens. llamAmpere uses it as a
  built-in drafter. The main model then checks the drafted tokens in one forward pass and keeps the ones it agrees with.
  Checking is exact, so the output follows the main model's distribution.
- **KV cache.** The per-token attention memory the model keeps for the context. Its size grows with the context, so
  compressing it is what makes long contexts fit.
- **SJ-KVaRN a/b.** a bits per key value, b bits per value. A "t" (3/3t, 3/2t) means the trellis-coded body.
- **GDN state.** Qwen3.8 is a hybrid model. Most of its layers are gated delta-net (GDN) layers, which carry a
  fixed-size recurrent state instead of a KV cache.
- **Ship corpus.** Our task benchmark: real coding, agentic and RAG task histories at temperature 1.0, with a fresh
  server per request. "Weighted gain" is 0.4 x coding + 0.4 x agentic + 0.2 x rag. "±" is two standard errors of the
  mean across seeds.
- **KL (nats).** The KL divergence of the output token distribution against a reference, measured on generated tokens.
  Lower is closer to the reference.

All speed numbers come from an RTX 3090 Ti at 350 W, unless a row says it ran on a rented RTX 3090. Decode tok/s counts
generation time only; prefill is reported separately.

## Headline numbers

Every release comparison runs each release with its own published settings, in one session on one card. Each cell is
a 100,000-token depth (context 110,592) followed by 5,120 generated tokens.

| release and cache | ship corpus at 100K, weighted tok/s | peak VRAM |
|---|---:|---:|
| v0.3.1 | 88.45 | 19,505 MiB |
| v0.4 (tq5_0/turbo4) | 87.43 | 18,883 MiB |
| **v0.5 tq5_0/turbo4** | **94.47** | 18,936 MiB |
| **v0.5 SJ-KVaRN 4/4** | **93.31** | 19,707 MiB |

| comparison | weighted gain | seeds |
|---|---|---|
| v0.5 tq5_0/turbo4 vs v0.4, ship corpus at 100K | **+8.41% ± 5.76%** | 2 |
| v0.5 SJ-KVaRN 4/4 vs v0.4, ship corpus at 100K | **+6.77% ± 2.68%** | 2 |
| v0.5 tq5_0/turbo4 vs v0.3.1, ship corpus at 100K | +6.78% ± 3.31% | 2 |
| v0.5 tq5_0/turbo4 vs v0.4, short fixtures | +4.09% ± 3.03% | 3 |
| v0.5 SJ-KVaRN 4/4 vs v0.5 tq5_0/turbo4, ship corpus at 100K | −0.80% ± 6.32% (within noise) | 2 |

- **v0.4's single-prompt headline cell.** v0.4 published a single-prompt 100K cell, which we re-ran in the same
  session. v0.4 reproduced its published 102.19 tok/s at 103.33. On this one-prompt cell, v0.5 against v0.4 was within
  noise.
- **SJ-KVaRN 4/4 vs tq5_0/turbo4 on that cell.** Over 5 clean seeds, SJ-KVaRN 4/4 read 101.3 tok/s against
  97.1 for tq5_0/turbo4. That difference is not significant.
- **KL.** At 100K depth, against a q8_0/q8_0 cache:
  - SJ-KVaRN 4/4 has 0.00066 nats against 0.00113 for tq5_0/turbo4, the same 41% reduction as above (CI 39-43%).
  - 3/3t has 0.00130 nats and 3/2t has 0.00256.

## SJ-KVaRN

SJ-KVaRN builds on the KVarN method (huawei-csl/KVarN). The table below comes from the published SJ-KVaRN paper.
"KL" is mean KL on generated tokens; "memory" is KV cache bytes.

| comparison | KL | memory |
|---|---|---|
| SJ-KVaRN 4/4 vs tq5_0/turbo4 | 41% lower (CI 39-43%) | +2% |
| SJ-KVaRN 4/4 vs turbo4/turbo4 | 68% lower | +14% |
| SJ-KVaRN 3/3t vs turbo4/turbo4 | 18% lower | −9% |
| SJ-KVaRN 4/4 vs KVarN as published (1,024-token tail) | 38% lower (CI 27-47%) | |
| SJ-KVaRN 3/3 vs KVarN as published | 48% lower | |
| SJ-KVaRN 3/2 vs KVarN as published | 76% lower | |

The comparison against KVarN as published covers nine task histories, and SJ-KVaRN had the lower KL on all nine. Those
histories are short, so the two caches did not use the same bytes per token in that comparison. Higher-bit cache types
(q8_0, q5_1) have lower KL than SJ-KVaRN 4/4.

Full method, measurements and caveats: the [SJ-KVaRN paper](https://claude.ai/artifact/GFj87G99yEcgnMe2e1MDjV). The
standalone codec (single-header C99 library plus CUDA reference kernels, MIT) is at
[JakeATX/sj-kvarn-codec](https://github.com/JakeATX/sj-kvarn-codec).

Bits per stored value, and decode speed relative to 4/4 on the ship corpus at 100K:

| configuration | bits per value | KL at 100K vs q8_0/q8_0 (nats) | decode vs 4/4 |
|---|---:|---:|---:|
| tq5_0/turbo4 (for reference) | 4.63 | 0.00113 | within noise |
| SJ-KVaRN 4/4 | 4.72 | 0.00066 | 1.00 |
| SJ-KVaRN 3/3t | 3.76 | 0.00130 | 0.79x |
| SJ-KVaRN 3/2t | 3.27 | 0.00256 | 0.77x |

- **Trellis prefill.** Prefill with a trellis body (3/3t, 3/2t) runs at about 575-613 tok/s at 100K, against about
  990-1,018 tok/s for 4/4.
- **Built-in defaults** (no flags needed):
  - a 128-token f16 sink;
  - 6-bit (tq6_0) staging;
  - an adaptive full-precision tail of 4,096 to 8,192 tokens;
  - a scalar body at 4/4.
- **Override flags.** `--sjkvarn-tail-max 0` fixes the tail at `--sjkvarn-tail`. `--sjkvarn-sink`,
  `--sjkvarn-sink-type` and `--sjkvarn-staging-type` override the other defaults.
- **Off by default.** A fused-rotation build of the SJ-KVaRN kernels (CMake option `GGML_SJKVARN_FUSED_ROT`). It gives
  the same output with no measured speed gain.

## Long context and memory

- **262,144 tokens with SJ-KVaRN 4/4, full depth.** At 262,144 context with a full-depth prompt, v0.5 SJ-KVaRN 4/4
  peaked at 21,924 MiB. v0.4 with tq5_0/turbo4 peaked at 22,211 MiB on the same test. So 4/4 holds the same context
  with lower KL and 287 MiB less memory.
- **252K-deep decode on a rented RTX 3090 (350 W, 5,120 generated):**

  | cache | decode | peak |
  |---|---:|---:|
  | tq5_0/turbo4 | 61.58 tok/s | 21,274 MiB |
  | 4/4 | 60.37 tok/s | 22,182 MiB |

- **3/3t on 24 GB, full depth.** At 262,144 context with a 260,839-token prompt plus 256 generated tokens, SJ-KVaRN
  3/3t peaked at 20,941 MiB (whole card, desktop included).
- **3/2t on 24 GB.** At 262,144, only the allocation is measured: 19,309 MiB after boot. A full 262K-deep run is not
  measured.
- **Two slots.** Two slots of 131,072 tokens each fit with tq5_0/turbo4. The full-depth peak was 23,407 MiB on v0.5 and
  23,486 on v0.4. SJ-KVaRN runs one slot only (see Known limits).
- **12 GB cards.** 204,800 tokens in about 11 GB (see the 12 GB settings).
- **Bounded prefill.** SJ-KVaRN prefill uses the same 256 MiB bounded workspace as the other cache types.

## Prompt cache

The prompt cache is on by default and needs no flags.

- **RAM tier.** When another conversation takes the slot, the old one is parked in host RAM. The size follows the
  host's RAM, capped at half of the memory available at startup. The log says what it chose.

  | host RAM | RAM tier |
  |---|---:|
  | under 16 GB | 2,048 MiB |
  | 16 GB | 4,096 MiB |
  | 32 GB | 8,192 MiB |
  | 64 GB | 12,288 MiB |
  | 96 GB | 16,384 MiB |
  | 128 GB or more | 20,480 MiB |

- **Disk tier.** Entries evicted from RAM, and everything cached at shutdown, go to a 16,384 MiB disk tier in
  `~/.cache/llamampere/prompt-cache` (`$XDG_CACHE_HOME` if set).
  - The oldest entries are deleted first.
  - Writes pause while the filesystem has less than the larger of 10% and 8 GiB free.
- **Restores.** An entry is restored only into the same model with the same KV, drafter-KV and GDN state settings.
- **Overrides:**
  - `--cache-ram N`, in MiB: `-1` = half of free memory, `0` = no prompt cache.
  - `--cache-disk-path DIR`.
  - `--cache-disk-limit N`, in MiB: `-1` = no limit, `0` = off.
  - `--no-cache-disk`.
- **Measured on a 20K-token conversation.** We tested three interruptions: a RAM restore, a disk restore and a server
  restart. In each case the next reply matched the reply of an uninterrupted run. They saved 13.1, 11.6 and 12.7 s
  against a 14.3 s cold prefill.
- **SJ-KVaRN.** A reused prompt restarts at the 128-token group boundary at or below the first changed token.
  - On an agent's second turn, the server re-prefilled 150 tokens instead of 27,670. That took 0.30-0.37 s instead of
    about 21-30 s.
  - Parking a 17K-token conversation in RAM and bringing it back saved 12.5-17.2 s per resume.
- **Tests.** All 174 cache-type combinations pass save and restore on CUDA and CPU. The disk tier has 125 unit tests.

## Model formats

- **IQ4_XS-M (24 GB).**
  [`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF`](https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF),
  15.6 GB, with the MTP head. This is the reference model for every 24 GB number in these notes.
- **2.3 bpw (12 GB).**
  [`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF`](https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF).
  - 9.0 GB: EXL3 trellis tensor types plus the MTP head, encoded from Swift 1.5 BF16.
  - The Q8_0 token embedding stays in host RAM.
  - Total output, reasoning plus answer, was 0.73x that of stock Qwen3.8-27B at 4 bits (24 fixed questions, greedy;
    90% CI 0.54-0.98).
  - LiveCodeBench runs, grader and audit:
    [`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-LiveCodeBench-Pagoda`](https://huggingface.co/datasets/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-LiveCodeBench-Pagoda).
- **EXL3.** MTP verify decode on EXL3 models is faster in v0.5.

- **Coming in v0.5.1:** a Swift 1.5 EXL3 4.0 bpw model, with its own fit and KL numbers.

## Speculative decoding (drafter)

- **Adaptive depth 3-4 is the default.** The drafter is on by default for Qwen3.8 GGUFs that carry the MTP head, and it
  uses the built-in 65,536-token draft vocabulary.
- **Drafter cache types.** The drafter's KV cache takes the main model's types. Over an SJ-KVaRN main cache, it uses
  tq5_0/turbo4. We gated an SJ-KVaRN drafter cache and it did not ship (see "Tried, measured, not shipped").
- **Faster drafter.** The drafter's per-round cost is lower.
- **Small-card options.** `--spec-draft-window N` caps the drafter's cache at the last N tokens.
  `LLAMA_MTP_DRAFT_COMPUTE_LEAN=1` caps the draft micro-batch at 64 tokens. The 12 GB command uses both.

## Server and usability: defaults that changed

| default | now | how to get the old behaviour |
|---|---|---|
| Prompt cache | on: an auto-sized RAM tier plus a 16 GiB disk tier | `--cache-ram 0 --no-cache-disk` |
| MTP draft depth | adaptive 3-4 | `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0` (fixed depth 4) |
| Drafter KV over an SJ-KVaRN main cache | tq5_0/turbo4 | `--spec-draft-type-k` / `--spec-draft-type-v` |

New flags:

- **SJ-KVaRN:**
  - `-ctk sjkvarn4|sjkvarn3`, `-ctv sjkvarn4|sjkvarn3|sjkvarn2`;
  - `--sjkvarn-body-type auto`;
  - `--sjkvarn-tail`, `--sjkvarn-tail-max`;
  - `--sjkvarn-sink`, `--sjkvarn-sink-type`;
  - `--sjkvarn-staging-type`.
- **Prompt cache:** `--cache-disk-path`, `--cache-disk-limit`, `--no-cache-disk`.
- **Small cards:** `--cache-type-s`, `--spec-draft-window`.

**Opt-in, off by default.** `GGML_CUDA_FA_I8QK=1` gives faster prefill at a small KL cost (+0.0013 nats). It is not in
any recommended command.

## Bug fixes

- **Per-decode host copies.** An upstream change to mixed token/embedding batches added three blocking host-to-device
  copies to every decode step. They now run only for batches that really mix both.
- **Upstream attention refactor.** The refactor broke SJ-KVaRN 4/4 prefill on our tree, and we fixed it before release.
  For speed, the refactor itself is not in this release (see "Tried, measured, not shipped").

## Tried, measured, not shipped

- **A dedicated SJ-KVaRN 4/4 attention path for verify widths 3 to 5.** It gave the same output.
  - Per-round cost: −1.97% (± 0.17%) on identical token streams.
  - Weighted gain: +3.33% ± 3.89% at 2 seeds and +5.33% ± 10.23% at 4 seeds. Both are below the ship rule.
  - It was reverted.
- **SJ-KVaRN 4/4 as the drafter's cache.** Against a tq5_0/turbo4 drafter cache it read −0.83% ± 1.18% on the ship
  corpus at 100K and −1.93% at 262K (rented RTX 3090). The drafter stays on tq5_0/turbo4.
- **Double-buffered GDN replay.** Decode −4.00% ± 0.25% and prefill −3.66% against the production snapshot ring.
- **Extending one SJ-KVaRN 4/4 decode speedup to 3/3t and 3/2t.** −0.67% ± 2.30% and +0.64% ± 6.73%. It stays on 4/4
  only.
- **A revised trellis weight kernel for the 2.3 bpw model.** It was 1.006x in a microbenchmark. End to end it read
  +3.78% with a standard error of 9.06% (2 seeds), so it is not significant.
- **Upstream's flash-attention swizzle refactor.** Weighted −0.93% and −0.59% in two runs, with attention kernel time
  +3.18%. It is reverted in our tree.
- **An SJ-KVaRN 4/4 trellis body.** No KL gain once the decode path is held fixed.
- **Holding the first and last 12.5% of layers at 4/4 under 3/3t and 3/2t.** The KL ratios were 0.928 and 0.850 of the
  uniform trellis at 32K, but the confidence intervals include 1. It is not the default.
- **f16 staging instead of tq6_0 for SJ-KVaRN.**
  - Speed: up to 2.8% faster, within noise.
  - KL: no measurable gain.
  - Memory: about 400 MiB more.
- **q8_0 staging instead of tq6_0.** No measurable KL difference at 3/2t on nine histories.
- **SageAttention-style int8 Q·K for prefill.** 4-14% slower per operation at 512 query tokens. Parked.
- **Tiled top-k sampling.** Sampling is only 0.9-2.5% of a decode round, which is under the 3% gate. Parked.
- **A post-trellis scale refit for the 2.3 bpw encoder.** KL got worse.
- **INT4 weight-format variants:**
  - One variant was 6-8% slower on every weight type it touched.
  - Another was about 1.2x the KL of IQ4_XS at the same speed tier.

  Neither ships.
- **tq6_0 keys and values.** They decode faster than in v0.4. Per attention operation they are
  still 1.06-1.46x slower than tq5_0/turbo4 and use 1.32x the bytes. tq5_0/turbo4 stays the speed recommendation.

## Known limits

- **SJ-KVaRN is single-slot.** It refuses `--parallel` above 1. Multi-slot SJ-KVaRN is in progress. Use tq5_0/turbo4
  for two slots.
- **Context shift** is not supported with SJ-KVaRN. Use `--no-context-shift`.
- **SJ-KVaRN 4/4 above 262,144 context.** 262,144 boots, but 327,680 fails with a 1,126 MiB compute buffer allocation.
  tq5_0/turbo4 boots 311,296. 262,144 is the model's trained context.
- **Full depth not measured:** 3/3t and 3/2t at 262K on 24 GB, and 3/3t at 204,800 on 12 GB. 3/2t was measured at full
  depth on 12 GB (10,690 MiB).
- **Cached replies are not byte-identical.**
  - At temperature 0, a reply after an SJ-KVaRN cache restore is not byte-identical to a fresh run.
  - The first-token KL is in the same range as for plain q8_0 caching.
- **Prompt cache edge cases:**
  - An edit deep inside a long prompt falls back to the nearest checkpoint (`--checkpoint-min-step`, default 8192).
  - Deep edits to multimodal prompts re-process the whole prompt.
  - Block KV streaming (`--kv-stream-arena-mib`) cannot be cached.
- **Platforms not run.**
  - The prompt cache's memory sizing on Windows and macOS has not been compiled.
  - SJ-KVaRN is tested on CUDA and CPU only.
- **12 GB quality.** The 2.3 bpw model reaches 86% of BF16's LiveCodeBench score. Expect weaker answers than on the
  24 GB configurations.

## Attribution and licenses

- llama.cpp and ggml are ggml-org's (MIT).
- SJ-KVaRN builds on the KVarN method by huawei-csl (https://github.com/huawei-csl/KVarN).
- The TurboQuant KV cache types and TurboQuant weight types come from TheTom's TurboQuant fork of llama.cpp.
- The EXL3 format, the trellis codebooks and their arithmetic are Turboderp's exllamav3.
