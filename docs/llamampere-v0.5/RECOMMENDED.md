# llamAmpere v0.5: recommended settings

v0.5 adds **SJ-KVaRN**, a compressed KV cache. It stores keys and values at 4, 3 or 2 bits, with a short
full-precision recent window, and it can use a trellis-coded body. You select it with `-ctk sjkvarnN -ctv sjkvarnN`.
 This page lists the configurations we recommend for Qwen3.8-27B, with the
measurements behind each one. Unless a row says otherwise:

- Model: [ATX-Swift 1.5 IQ4_XS-M](https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF).
- Card: RTX 3090 Ti at 350 W.
- Workload: the coding / agentic / rag ship corpus (real task histories), temperature 1.0, reasoning effort medium,
  5,120 generated tokens per cell.
- Speed is timed on generated tokens only. VRAM is the whole card, held at or under 23,552 MiB.

## Which configuration

| option | use it for | KV flags |
|---|---|---|
| 1. Pure speed | 24 GB cards, fastest decode | `-ctk tq5_0 -ctv turbo4` |
| 2. Balanced | 24 GB cards, lowest KL at about the same speed | `-ctk sjkvarn4 -ctv sjkvarn4` |
| 3. Small-card fit | the least KL that still frees memory | `-ctk sjkvarn3 -ctv sjkvarn3 --sjkvarn-body-type auto` |
| 4. Maximum context (acceptable, not first class) | the most context per GB | `-ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto` |

You do not need to set any other SJ-KVaRN flag. These are the built-in defaults:

- a 128-token f16 sink;
- tq6_0 staging;
- an adaptive full-precision tail of 4,096 to 8,192 tokens (`--sjkvarn-tail-max 0` fixes it at `--sjkvarn-tail`);
- a scalar body at 4/4.

`--sjkvarn-body-type auto` selects the trellis body for 3/3 and 3/2.

The MTP drafter is on by default for Qwen3.8 GGUFs that carry the MTP head. It runs at adaptive depth 3-4 with the
built-in 65,536-token draft vocabulary, and its KV cache takes the trunk's types. Over an SJ-KVaRN trunk the drafter
cache is tq5_0/turbo4.

## Measurements

| option | KV bits per value | KL at 100K (nats) | decode speed at 100K | max context measured (24 GB) |
|---|---:|---:|---|---|
| tq5_0 / turbo4 | 4.63 | 0.00113 | 94.5 tok/s (paired with 4/4) | 262,144 at 252K deep, peak 21,274 MiB |
| SJ-KVaRN 4/4 | 4.72 | 0.00066 | 93.3 tok/s (paired with tq5_0/turbo4) | 262,144 at 252K deep, peak 22,182 MiB |
| SJ-KVaRN 3/3t | 3.76 | 0.00130 | 0.79x of 4/4 (paired) | 262,144 boots; full depth not measured |
| SJ-KVaRN 3/2t | 3.27 | 0.00256 | 0.77x of 4/4 (paired) | 262,144 boots; full depth not measured |
| 12 GB: SJ-KVaRN 3/2t, 2.3 bpw model | 3.27 | not measured on this model | not a speed cell | 204,800 at 203,568 deep, peak 10,690 MiB |

How each column was measured:

- **KV bits per value.** Trunk KV at a 110,592-token context, including the sink, tail and staging, divided by the
  number of K and V values. In MiB: 1,998.13 for tq5_0/turbo4, 2,038 for 4/4, 1,622.5 for 3/3t and 1,414.75 for 3/2t.
  tq5_0/turbo4 is K 5.125 and V 4.125.
- **KL.** Mean KL divergence on the generated tokens against a q8_0/q8_0 cache. Six ship-corpus histories at 100K
  depth, 37,502 positions. 95% intervals:
  - 4/4: [0.00053, 0.00081]
  - 3/3t: [0.00103, 0.00160]
  - 3/2t: [0.00207, 0.00313]

  The tq5_0/turbo4 and 4/4 cells share one reference run; at that depth 4/4 is 0.58x tq5_0/turbo4. The 3/3t and 3/2t
  cells come from a codec audit that regenerated the q8_0 reference, which moves KL by up to about 10%.
- **Decode speed.** Speeds are only compared within one run, because the card moves a few percent between sessions.
  Ship-corpus speeds are means at a 100,000-token depth (context 110,592), seeds 7300 and 7301, 5,120 generated.
  - tq5_0/turbo4 vs 4/4, same run:

    | measurement | tq5_0 / turbo4 | 4/4 | difference |
    |---|---:|---:|---|
    | ship corpus at 100K (tok/s) | 94.5 | 93.3 | within noise |
    | one 100K prompt, v0.4 headline protocol, 5 clean seeds (tok/s) | 97.1 | 101.3 | not significant |
    | short fixtures (tok/s) | 107.5 | 103.3 | tq5_0/turbo4 faster, significant |
    | prefill at 100K (tok/s) | 1,016 | 986 | |
    | peak VRAM at 110,592 context (MiB) | 18,936 | 19,707 | |

  - 3/3t and 3/2t vs 4/4, a second run: 3/3t decodes at 0.79x and 3/2t at 0.77x of 4/4.
    - The 3/3t cost is per round (−20.8% per-round speed); acceptance is unchanged.
    - 3/2t vs 3/3t is not significant (−3.1% ± 5.5%).
    - Trellis prefill runs at about 575-613 tok/s, against about 990-1,018 for 4/4.
- **Max context.**
  - 262,144 tokens is the model's trained context.
  - The 252K-deep cells ran on a rented RTX 3090 at 350 W with 5,120 generated tokens: 61.58 tok/s for tq5_0/turbo4 and
    60.37 for 4/4.
  - For 3/3t and 3/2t on 24 GB, only the allocation at 262,144 is measured: 19,813 and 19,309 MiB after boot, against
    20,888 for tq5_0/turbo4 and 20,820 for 4/4. A full 262K-deep run is not measured.
  - The release check of the four commands below ran on an RTX 3090 Ti that also drives a desktop (about 790 MiB idle).
    Each command booted at 262,144 and generated 512 tokens. Whole-card peaks were 22,031 MiB for tq5_0/turbo4,
    22,693 for 4/4, 21,669 for 3/3t and 21,157 for 3/2t. This is a short prompt, not a full-depth run.

### KL against an f16 cache

These numbers come from a paper run with an f16 KV reference. It covers one agentic and one coding history at
seed 7300. Values are mean KL in nats on the generated tokens.

| cache | agentic | coding | bits per value (agentic) |
|---|---:|---:|---:|
| q8_0 / turbo4 | 0.000732 | | 6.31 |
| SJ-KVaRN 3/3t | 0.001145 | 0.001120 | 3.88 |
| SJ-KVaRN 3/2t | 0.002427 | 0.002129 | 3.48 |
| q4_0 / q4_0 | 0.002544 | 0.002336 | |
| q8_0 / turbo3 | 0.002683 | | |
| turbo3 / turbo3 | 0.007333 | 0.007175 | |

The rag history of this run is still in progress. 4/4 and tq5_0/turbo4 were not arms of this run. On a shorter
nine-history set (964 positions, f16 reference) the KL was:

| cache | KL (nats) |
|---|---:|
| 4/4 | 0.000979 |
| 3/3t | 0.001542 |
| 3/2t | 0.002241 |

## 24 GB cards (RTX 3090 / 3090 Ti)

### Build

You need:

- Linux;
- an NVIDIA card with 24 GB;
- the CUDA toolkit (tested with 12.4);
- CMake, git and a C++ compiler.

`-DLLAMA_BUILD_BORINGSSL=ON` builds HTTPS support from source so that `-hf` can download models. Without it, a
machine that has no OpenSSL development files gets a build whose `-hf` fails with "HTTPS is not supported".

```bash
git clone -b v0.5 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
```

### Run

Each command downloads the 15.6 GB model the first time it runs (`-hf`). The server listens on
http://127.0.0.1:8080 and serves an OpenAI-compatible API.

**1. Pure speed: tq5_0 K / turbo4 V**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk tq5_0 -ctv turbo4 -b 4096 -ub 1024 --parallel 1
```

**2. Balanced: SJ-KVaRN 4/4**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1
```

**3. Small-card fit: SJ-KVaRN 3/3t**

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn3 -ctv sjkvarn3 --sjkvarn-body-type auto -b 4096 -ub 1024 --parallel 1
```

**4. Maximum context: SJ-KVaRN 3/2t** (acceptable, not first class)

```bash
./build-sm86/bin/llama-server -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto -b 4096 -ub 1024 --parallel 1
```

The speed cells used the same KV flags, `-b 4096 -ub 1024` and the drafter flags, spelled out explicitly
(`--spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-n-min-adaptive 3 --spec-draft-p-min 0` and the
`docs/mtp-vocab/atx_65536.txt` map). These are the same values as the defaults.

v0.5 builds in these defaults:

- the 6-bit staging-group kernel;
- the fused SJ-KVaRN write path;
- 256 MiB bounded-prefill workspaces.

Two items are off by default:

- int8 Q·K attention for prefill (`GGML_CUDA_FA_I8QK=1` turns it on);
- the fused-rotation SJ-KVaRN kernels (CMake option `GGML_SJKVARN_FUSED_ROT`).

None of these need an environment variable. With tq5_0/turbo4, v0.5 measured +8.41% ± 5.76% G over v0.4 on the ship
corpus and +4.09% ± 3.03% on the fixtures. G = 0.4 coding + 0.4 agentic + 0.2 rag.

## 12 GB cards (RTX 3060 12 GB, 3080 12 GB, 3080 Ti)

**Token efficiency.** Total output, reasoning plus answer, was 0.73x that of stock Qwen3.8-27B at 4 bits (ATX-4-XS).

- Measured on 24 fixed questions run greedy to the end of the answer; 90% CI 0.54-0.98.
- Swift 1.5 IQ4_XS on the same questions: 0.81x stock. This build keeps some, but not all, of Swift 1.5's shorter
  output.
- At temperature 1.0 its reasoning was about 1.22x as long as Swift 1.5 IQ4_XS's (90% CI 1.11-1.33).

**Model:** [`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF`](https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF).

- 9.0 GB, EXL3 trellis tensor types plus the MTP head.
- Encoded from Swift 1.5 BF16.
- The Q8_0 token embedding stays in host RAM.

**Command** (SJ-KVaRN 3/2t, 204,800 context, MTP), with the flags of the measured runs:

```bash
LLAMA_MTP_DRAFT_COMPUTE_LEAN=1 ./build-sm86/bin/llama-server \
  -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP-GGUF \
  -hff ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-MTP.gguf \
  -c 204800 --parallel 1 -ngl 99 -fa on -fit off -b 4096 -ub 512 \
  --no-context-shift --cache-ram 0 --jinja \
  -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type sjkvarn4t \
  --sjkvarn-sink 128 --sjkvarn-sink-type f16 --sjkvarn-staging-type tq6_0 \
  --sjkvarn-tail 4096 --sjkvarn-tail-max 8192 \
  --cache-type-s q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --spec-draft-vocab-map docs/mtp-vocab/atx_65536.txt \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-window 8192 \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0
```

What the 12 GB-specific settings do:

- `--cache-type-s q8_0` stores the Gated DeltaNet recurrent state at 8 bits.
- The fixed depth-4 drafter has a q8_0/q8_0 cache and an 8,192-token draft window.
- `LLAMA_MTP_DRAFT_COMPUTE_LEAN=1` caps the draft context's micro-batch at 64 tokens.
- `-ub 512` keeps the compute buffers inside the budget.

The measured runs also set `GGML_CUDA_PREFILL_KV_MIB=256 GGML_SJKVARN_PREFILL_MIB=256`. These only restate the built-in
default. `--spec-draft-vocab-map auto:65536` selects the same built-in list as the file.

**Fit.** At context 204,800, a 203,568-token prompt plus 256 generated tokens peaked at 10,690 MiB whole-card memory.
That is under the 11,000 MiB budget we use for 12 GB cards. The counted agent runs peaked at 10,728-10,744 MiB.

**Quality.** The model reaches about 85% of the BF16 model's LiveCodeBench score, so expect weaker answers than the
24 GB configurations.

- LiveCodeBench v6, 100 pinned tasks, reasoning effort xhigh, temperature 1.0, seven runs on rented 12 GB cards:
  **77.71%** pooled (95% CI 75.74-79.69).
  - Qwen3.8-27B BF16: 90.3%, so 86% of it.
  - Swift 1.5 IQ4_XS-M on our engine: 89.25%.
- These runs used the configuration above, with the context calibrated on each card: 229,376 on the RTX 3060 and 221,184
  on the RTX 3080 / 3080 Ti. MTP acceptance was 0.513 and generation 34.7 tok/s pooled.
- The model was also tested as a Hermes agent backend at xhigh with the 3/2t configuration above.

Runs, grader, audit and the agent test outputs:
[`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-LiveCodeBench-Pagoda`](https://huggingface.co/datasets/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-2.3bpw-LiveCodeBench-Pagoda).

On a 24 GB card the IQ4_XS-M model with one of the four configurations above is the stronger choice.
