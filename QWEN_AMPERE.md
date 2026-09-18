# llamAmpere

Qwen3.8-27B on one Ampere card (RTX 3090 / 3090 Ti, 24 GB): a fork of
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
(TurboQuant+ KV cache, native MTP speculative decoding) carrying SM86-specific
kernel and memory work, plus the quantization recipe it was tuned with.

v0.3.1 (2026-09-18) is a format release on top of v0.3: EXL3 (Turboderp's exllamav3 trellis
format) as GGUF-native types with an SM86 decode kernel ([docs/exl3.md](docs/exl3.md)), Prism ML's
Ternary Bonsai 2 27B with SM86 decode kernels for both of its ternary containers
([docs/bonsai2.md](docs/bonsai2.md)), an opt-in shared-memory codebook for IQ3 decode, and a full upstream
catch-up (llama.cpp master `b49650adb` and TurboQuant `407f3237b`, 772 commits ahead of the v0.3 base). The ATX IQ4_XS configuration below is unchanged; see
[docs/llamampere-v0.3.1/RELEASE_NOTES.md](docs/llamampere-v0.3.1/RELEASE_NOTES.md).

This is v0.3 (2026-09-13), the successor of v0.2 (2026-09-07) and of
[llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere) (v0.1, 2026-09-03).
v0.3 replaces the draft verification rule with exact p/q verification, defaults the fused MMA
attention path for q8_0-K/turbo3-V and extends it to q8_0-V, and bounds recurrent rollback to the
snapshots a ubatch actually wrote. It is 1.28x v0.2 and 1.46x stock llama.cpp on clean fixtures, and
1.70x upstream TurboQuant at 100K KV depth. The write-up with figures is
[docs/llamampere-v0.3/ARTICLE.md](docs/llamampere-v0.3/ARTICLE.md); v0.2's is
[docs/llamampere-v0.2/ARTICLE.md](docs/llamampere-v0.2/ARTICLE.md).

Measured on a 3090 Ti at 350 W with the ATX-IQ4_XS-M quant (working name ATX-4-XS, which the filenames keep; IQ4_XS base, M-pattern upgrades, 4.56 bpw):

| | |
|---|---|
| clean-fixture decode, temperature 1, MTP-3 (v0.3) | 99.4 tok/s; v0.2 77.9, upstream TurboQuant 62.4, stock llama.cpp 67.9 |
| 100K KV depth, temperature 1, MTP-3 (v0.3) | 93.16 tok/s decode; upstream TurboQuant 54.96 (1.70x) |
| deepest context that loads with MTP-3 (v0.3) | 229,376; upstream TurboQuant on UD-Q3_K_XL 262,144, stock llama.cpp on UD-Q3_K_XL 212,992 |
| 100K-token generation, temperature 1, MTP-3 (v0.2) | 75.3 tok/s cumulative over 102,400 generated tokens; v0.1 as documented 66.1, upstream TurboQuant+ of 2026-09-03 56.0 |
| 220K prompt populated with the shortlist (v0.2) | 22,174 MiB peak, +150 MiB over v0.1, no OOM |
| populated 200K context (v0.1 measurement) | 20.9 GiB ready VRAM, 694 tok/s prefill, 65 tok/s decode |
| long session from 56K to 207K KV, temperature 1, MTP-3 (v0.3) | 95.1 tok/s at 56K, 85.2 at 207K; -9.6% first five windows to last five (TurboQuant -21.6%) |
| single stream vs vLLM / SGLang at 32K / 64K (v0.3) | 112.4 / 100.7 tok/s; tuned vLLM MTP-4 101.7 / 90.3, tuned SGLang 68.6 / 64.4 |
| largest measured fit | 245,760-token window with a 240K prompt, 22.1 GiB ready for the server (measured before v0.3; P6 uses 170-260 MiB less). Counting the whole card, a desktop leaves room for 237,568 |
| context by KV cache type under a 23 GB card budget | measured at full depth: q8_0/turbo3 237,568 (shipped), q8_0/q5_1 204,800, q8_0/q8_0 180,224, see [KV cache options](#kv-cache-options) |
| agent session 100K -> 245K context | 120K generated tokens, 52 tok/s cumulative (60 at 110K, 47 at 245K), 76% draft acceptance |
| per speculative round vs Q3_K_XL / Q4_K_M | +9-10% / +23% |

Single-user configuration: `--parallel 1`, one request at a time.

## Model

Every number in this file was measured with
[`jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF`](https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF),
which carries the GGUF, the per-tensor type map, and the recipe. 14.5 GiB file,
13.9 GiB on the GPU. Bulk tensors IQ4_XS (the fastest format on SM86 at
speculative verification widths), Q5_0 on the tensors Unsloth's tier ladder
upgrades first, Q6_K on attention K/V, and Q8_0 on the GDN alpha/beta vectors and
the attention K/V tensors Q4_K_M keeps at Q8_0.

### Recommended

Two options for Swift-Qwen3.8-27B, both IQ4_XS-based and within 0.1 GiB of each
other, so both fit a 24 GB card the same way:

| | file | size | what it is |
|---|---|---|---|
| **ATX-Swift uncensored** | [`jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF`](https://huggingface.co/jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF) | 14.52 GiB (4.56 bpw) | The ATX type map above, applied to the uncensored Swift weights with the same imatrix procedure. |
| **Swift stock** | [`ukisai/Swift-Qwen3.8-27B-GGUF`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF), file `Swift-Qwen3.8-27B-IQ4_XS.gguf` | 14.61 GiB | The Swift authors' own IQ4_XS, with their published [per-tensor layout](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF/blob/main/layouts/Swift-Qwen3.8-27B-IQ4_XS.tensor-types.txt). |

Both are per-tensor maps, and they upgrade the same structural tensors — attention
K/V, attention output, `ssm_out`, `output.weight`. They differ in the ladder they
spend on those upgrades: ATX puts 138 tensors at Q8_0 and uses Q5_0 as the middle
rung, keeps the GDN alpha/beta vectors at Q8_0, and never drops a tensor below
IQ4_XS. The stock map leads with Q6_K (36 at Q8_0, 31 at Q6_K), leaves the GDN
alpha/beta vectors at F32, and puts seven tensors on Q4_K/Q5_K. The two have not
been benchmarked head to head on this card; the numbers in this file are the
flagship ATX quant, not either Swift build.

## Branches

| branch | what it is |
|---|---|
| `main` | the product (v0.3): upstream TurboQuant+ as of 2026-09-03 plus all accepted SM86 work through 2026-09-13. Build from here. |
| `release/v0.3` | the exact commit (`36a6bca81`) every v0.3 number in this file was measured on |
| `ws/AGN-agnes` | the Agnes 3.0 Flash loader work before it merged into `main` |

The v0.1 release branches (`perf/qwen38-sm86-decode-product`, `perf/qwen38-sm86-prefill`), the
`research/qwen38-sm86-*` experiment branches and the `archive/*` rejected experiments live on
[llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere).

The experiment log, with hypotheses, results, and what did not work, is
`QWEN38_SM86_FRONTIER_HANDOVER.md` in this tree.

## Build and run

```bash
git clone -b main https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
cmake --build build-sm86 -j8 --target llama-server

GGML_Q8_TURBO3_MMA_FUSED=1 ./build-sm86/bin/llama-server -m Qwen3.8-27B-ATX-4-XS.gguf \
  -c 245760 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 \
  --parallel 1 --jinja --fit off \
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --spec-draft-vocab-map docs/mtp-vocab/atx_65536.txt
```

Since 2026-09-10 the MTP drafter is verified by exact p/q rejection sampling instead of identity match
(`common/speculative.cpp`): each draft token is sampled from the drafter's own distribution under the request's
temperature / top-k / top-p, and the target accepts it with probability min(1, p/q), drawing from the residual on
rejection. The output distribution is the target's exactly (chi-square against the id-match verifier p = 1.00 / 0.98
/ 0.99 per position; greedy output byte-identical), and expected acceptance rises from p(argmax q) to Σ min(p, q).
It is default-on; `LLAMA_SPEC_PQ=0` restores identity match. With it the draft gate goes to `--spec-draft-p-min 0`
and the drafter cache to `--spec-draft-type-v q8_0` (the fused q8_0/q8_0 drafter kernel): +7.5% tokens/s over v0.2
at temperature 1.0, weighted 0.4 coding / 0.4 agentic / 0.2 RAG over three seeds (per-fixture means +7.5% / +8.0%
/ +6.3%), same 245,760 fit.

`--spec-draft-vocab-map` (new in v0.2) restricts the MTP drafter's output head to a 65,536-token
shortlist built from this model's generations (`docs/mtp-vocab/atx_65536.txt`; a 32K map is next to
it). The target model's head is untouched, so verification and the sampled output distribution are
the target's; only the draft proposals change. It is the largest single contributor to v0.2's decode
gain and costs 150 MiB of VRAM at 220K. See `docs/mtp-vocabulary-shortlist.md`. The adaptive
`--spec-draft-vocab-hot` tail that the same code supports measured 2.7-4.0% slower than the static
map and is left off.

### KV cache options

`-ctk q8_0 -ctv turbo3` is the shipped cache and the one every number in this file was measured with. The attention
layers are the only ones with a KV cache (16 of 64, 4 KV heads, head dim 256); the other 48 are recurrent and
their state does not grow with context. Two alternatives keep the value cache at a conventional llama.cpp format for
anyone who would rather not run turbo3. The drafter cache stays `--spec-draft-type-k/v q8_0` (2,176 B/token) in
all three.

| `-ctk` / `-ctv` | KV bytes per token, incl. drafter | largest context under a 23 GB card budget | suggested `-c` | decode speed vs shipped | build |
|---|---:|---|---:|---|---|
| `q8_0` / `turbo3` (shipped) | 25,984 | 237,568 whole-card; 245,760 on a card with nothing else on it | 237568 | reference | default |
| `q8_0` / `q8_0` | 36,992 (+42%) | 180,224 | 180224 | -0.3% weighted, within 1.2% at every depth | default |
| `q8_0` / `q5_1` | 31,872 (+23%) | 204,800 | 204800 | -23% weighted, and it gets worse with depth | `-DGGML_CUDA_FA_ALL_QUANTS=ON` |

Those three ceilings are measured, not scaled: each was found by a ladder in 8,192-token steps that loads the server
at `-c`, fills it with a C-2048 prompt, decodes 256 tokens, and requires the whole card (`nvidia-smi`, so the desktop
counts) to stay at or under 23,552 MiB the whole time. The largest passing window and the first failing one were
180,224 / 188,416 for q8_0, 204,800 / 212,992 for q5_1, and 237,568 / 245,760 for turbo3.

The turbo3 row needs a word, because 245,760 is the window quoted elsewhere in this file. The two measurements do not
disagree: 22,634 MiB at 245,760 is the server's own footprint, and the ladder measures the whole card, which on this
machine carries about 800 MiB of desktop. Server-side the ladder agrees, at roughly 22.8 GiB for 245,760; add the
desktop and the card peaks at 23,565 MiB, 13 MiB over the budget, so the run is killed during load. Keep 245,760 if
the card is headless or nearly so, and use 237,568 (23,349 MiB peak with the desktop up) if anything else is drawing
on it. Nothing regressed between the two numbers.

Decode speed was measured on the same four fixtures as the rest of this file, temperature 1.0, three seeds, 256
tokens, at 32,768 / 73,728 / 106,496 / 147,456 context. **q8_0 V costs nothing**: -0.3% weighted 0.4 coding / 0.4
agentic / 0.2 RAG, and no fixture moves more than 1.2%, which matches the kernel timing (321 µs per call at 100K KV
depth against 273 µs for turbo3 V). It runs on the same fused MMA attention kernel as the shipped cache (extended to
q8_0 V in v0.3). Raw tokens/s for that arm scatters more than that (-1.3% weighted, +7.1% to -6.9% by fixture), but
the scatter is draft acceptance drifting between arms as sampling diverges at temperature 1.0, not kernel speed: hold
acceptance fixed by counting verify steps per second instead of tokens and the arms sit on top of each other.

**q5_1 has no fused kernel, and it is expensive.** It runs on the generic flash-attention path, and the measured
cost is -23% weighted against the shipped cache, worsening with depth: -18% at 32K, -22% at 73K, -29% at 106K and
-36% at 147K (verify steps per second, so acceptance is held fixed; raw tokens/s gives -20% weighted). It buys
24,576 tokens of context over turbo3 and gives up about a quarter of the decode rate to get them, which is a bad
trade unless the extra window is the point. The default build also does not compile flash-attention kernels for q5_1.
Without `-DGGML_CUDA_FA_ALL_QUANTS=ON` the attention op is not supported on the GPU and falls back to the CPU. Use
that flag in the cmake line above (the build takes much longer). A fused q8_0-K/q5_1-V kernel is on the backlog.

The three cache flags are what make a long conversation usable rather than merely possible. `--cache-prompt` keeps the conversation's KV cache in the slot between turns, so a new turn on a 200K conversation pays only for the new tokens instead of a 100-second re-prefill. `--ctx-checkpoints` matters specifically for this model: 48 of its layers are recurrent, and a recurrent state cannot be rewound, so when you edit or regenerate a turn the server needs a saved state from before the edit point; it keeps up to 24 of them, at least 10,240 tokens apart (and one at every user turn regardless), in host RAM. Measured on this model, a snapshot is 150 MiB plus 1.5 KiB per token of position, because the MTP drafter's own single-layer KV cache is saved with the recurrent state: about 165 MiB at 10K, 495 MiB at 240K. 24 at 10,240 spacing covers the whole 245,760 window for about 7.8 GiB with at most ten seconds of replay after an edit; denser spacing multiplies that RAM (60 at 4,096 is about 20 GiB at the deep end). `--cache-ram` is a separate host-RAM budget for parking a whole conversation's KV (with its checkpoints) when another conversation takes the slot; a populated 200K conversation is about 7.4 GB, so 8 GiB holds one, and it only does work when you switch between chats. None of this touches VRAM. Budget about 16 GB of host RAM for it at the deep end (up to 8 GiB of snapshots plus the 8 GiB park space) on top of the model's own mapping; on a 32 GB machine keep the desktop light, or drop the count to 12 at 20,480 spacing for half the snapshot RAM.

**Disk tier for the prompt cache (this fork).** `--cache-disk-path DIR [--cache-disk-limit MiB]`
adds a second tier under the RAM prompt cache: a conversation evicted from RAM, or one too large
for the RAM budget in the first place (a full 200K-245K session is 7-9 GB), is written to `DIR`
instead of being dropped, and a later request that matches it is restored from the file with only
the new tokens processed. The index survives a server restart. Restores run at roughly 1.5 GB/s,
so a 200K conversation comes back in a few seconds instead of a 100-second re-prefill, and the
restored state is exact (greedy outputs match an uninterrupted session). Multimodal prompts stay
RAM-only. Off unless the path is given.

```bash
--cache-prompt --cache-ram 8192 --cache-disk-path /fast-nvme/llama-cache --cache-disk-limit 65536 \
--ctx-checkpoints 24 --checkpoint-min-step 10240
```

Everything the project adds is on by default. `LLAMA_SHARED_COMPUTE=0` disables
the shared compute arena; the research knobs (`GGML_Q8_TURBO3_MMA_MIN_Q`,
`GGML_CUDA_SM86_MMQ_POLICY`, `GGML_CUDA_SM86_MMVQ_WARP_ROWS`) stay off unless
you are reproducing a rejected experiment.

W2 research control: `LLAMA_OUTPUT_BUFFER_REUSE=0` restores the original combined output/sampling buffer, row reservation, and lazy allocation policy. The default keeps sampling storage separate, retains the largest row reservation, and reserves added embeddings during setup. Compare this control with GPU verification disabled on both sides to isolate allocation behavior. Avoided setup allocations do not imply a sustained tokens-per-second gain.

## v0.3 release notes (2026-09-13)

Same card and model as above. Four arms, one protocol: **stock llama.cpp** is upstream master
`8ea290247`, **TurboQuant** is `1208c5956` built clean, **v0.2** is `44233f009`, **v0.3** is
`36a6bca81`. Each arm is launched with only the flags its own tree understands. Decode tok/s at
temperature 1, top-k 20, top-p 0.95, MTP depth 3, mean of three runs:

| workload | stock llama.cpp | TurboQuant | v0.2 | **v0.3** | vs stock | vs TurboQuant | vs v0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| agentic (3 fixtures) | 68.6 | 66.1 | 81.3 | 100.9 | 1.47x | 1.53x | 1.24x |
| coding (1 fixture) | 65.7 | 51.3 | 67.8 | 94.9 | 1.45x | 1.85x | 1.40x |
| **all clean fixtures** | 67.9 | 62.4 | 77.9 | **99.4** | **1.46x** | **1.61x** | **1.28x** |

Retrieval fixtures are excluded, and so are two coding cells -- not missing, excluded. On those
prompts every arm except v0.3 degenerates into repetition before the generation ends. Degenerate text
is trivially draftable, so a looping arm posts an inflated tok/s. The contamination does not run in
one direction (on one shard it understates our margin, on another it flatters us) and cannot be
corrected arithmetically, so those cells are dropped rather than adjusted. The full argument is in
[docs/llamampere-v0.3/ARTICLE.md](docs/llamampere-v0.3/ARTICLE.md).

**The decomposition matters more than the ratio.** At a fixed draft depth every run reports
`predicted_n` and `draft_n_accepted`, and those split a speedup exactly, with nothing estimated:
`passes = predicted_n - draft_n_accepted`, so `tok/pass` isolates acceptance and `passes/s` isolates
kernel speed.

| v0.3 over | tok/s | = acceptance | x kernel | kernel's share of the log gain |
|---|---:|---:|---:|---:|
| v0.2 (`44233f009`) | 1.281 | 1.082 | 1.184 | 68% |
| TurboQuant (`1208c5956`) | 1.608 | 1.082 | 1.483 | 83% |
| stock llama.cpp (`8ea290247`) | 1.464 | 1.048 | 1.399 | 88% |

Exact p/q verification is this release's headline change, and p/q is an acceptance mechanism -- yet
acceptance contributes 1.048-1.082 and the kernels contribute 1.399-1.483. The ordering explains it:
p/q measured +7.45% G against v0.2's rule on v0.2's kernels, but it ships on top of P5b, P5c and P6,
and once the per-pass cost has fallen a faster pass is worth more than a fuller one. Writing this
release up as "acceptance improved" would be false.

**The margin widens with depth.** The same decomposition at 100K KV depth (100,000-token prompt,
2,048 generated, three runs, temperature 1):

| build at 100K depth | decode tok/s | tok/pass | passes/s | acceptance |
|---|---:|---:|---:|---:|
| TurboQuant (`1208c5956`) | 54.96 | 3.368 | 16.49 | 0.792 |
| **v0.3 (`36a6bca81`)** | **93.16** | 3.568 | 27.22 | 0.858 |

1.70x at depth against 1.61x on short fixtures, and the kernel's share rises from 83% to 95%: at
100K the attention stack is almost the whole story.

**Reachable depth is itself a result.** Probing each arm down a ladder to the deepest context it can
actually allocate on a 24,564 MiB card with MTP enabled -- the draft context allocates a *second* KV
cache, exactly 4,096 B/token (f16, one layer), on top of the main one:

| arm | model | V-cache | deepest context that loads |
|---|---|---|---:|
| stock llama.cpp | UD-Q3_K_XL | q8_0 | 212,992 |
| TurboQuant | UD-Q3_K_XL | turbo3 | 262,144 |
| v0.2 | ATX-4-XS | turbo3 | 229,376 |
| v0.3 | ATX-4-XS | turbo3 | 229,376 |

The first two rows are the same weights and differ only in V-cache format, so the +49,152 tokens
between them is what turbo3 buys on this card. The llamAmpere rows carry a larger model file
(14.52 GiB against 12.24 GiB) and spend some of that headroom on quality.

Read those ceilings with their condition attached: the probe left the drafter's own cache at its f16
default, 4,096 B/token. The run command above sets `--spec-draft-type-k/v q8_0`, which is smaller,
and that is the configuration the 245,760 window elsewhere in this file was measured in. The ladder
was run at the f16 default so that all four arms sat on an identical drafter-cache footprint, which
is what makes the four ceilings comparable to each other; it is not the configuration to deploy.

**The speed holds over a long session.** One server per arm, never restarted, prompt cache on,
starting from a 51,223-token agentic prompt and taking 5,000-token turns that each inject four fresh
SWE-bench cases, until the next turn would not fit in `-c 208,896`. Decode tok/s per window,
temperature 1:

| KV depth after window | v0.3 | v0.2 | TurboQuant | stock llama.cpp |
|---:|---:|---:|---:|---:|
| 56,222 | 95.09 | 75.34 | 63.87 | 68.86 |
| 100,201 | 100.45 | 75.01 | 56.94 | 59.11 |
| 149,743 | 87.94 | 78.35 | 54.37 | 50.09 |
| 187,185 | 90.82 | 73.49 | 48.30 (last) | -- (last at 156,515: 49.83) |
| 206,851 | 85.16 | -- (last at 197,174: 71.57) | -- | -- |

v0.3 falls 9.6% from its first five windows to its last five, and TurboQuant 21.6%. The decay is the
kernel, not the drafter: v0.3's passes/s falls 18.9% while its tok/pass rises 11.4%. The v0.3/v0.2
passes/s ratio stays between 1.166 and 1.183 from 56K to 197K, so what P5b, P5c and P6 buy at 20K
they still buy at 197K. The stock arms stop earlier because their KV costs more memory (stock at
`-c 159,744`, TurboQuant at 188,416, both chosen to stay under the 23 GB cap). Details, including
host-noisy and looped windows, are in the write-up.

**Against serving engines.** vLLM 0.29.0 and SGLang 0.5.9, each tuned and each with its own MTP
speculation, serving `Qwen3.8-27B-W4A16-AWQ` (neither loads our GGUF), single stream, 2,048 generated
tokens, mean of three, all under 23,552 MiB at peak:

| context | **v0.3, MTP-3** | vLLM, MTP-4 | vLLM, MTP-3 | SGLang, NEXTN | v0.3 over best other |
|---|---:|---:|---:|---:|---:|
| 32K | **112.4** (18,812 MiB) | 101.7 (23,295) | 90.2 (23,187) | 68.6 (23,028) | 1.10x |
| 64K | **100.7** (19,847 MiB) | 90.3 (23,356) | 83.1 (23,365) | 64.4 (23,348) | 1.11x |

Past 64K neither engine fits under the cap in any configuration we found. This is the single-user
case only; nothing here speaks to batched throughput, which is what those engines are built for.

What changed since v0.2 (`44233f009`), in merge order -- 17 commits, and excluding documentation and
the vocabulary map, 36 files with 1,802 insertions and 130 deletions:

- `b6d29742b` W5 instrumentation: donor-arena adoption log and fattn path census.
- `2d41505a1` cuda: route q8_0-K/turbo3-V verify attention through the fused MMA kernel by default (P5b).
- `05b49181d` cuda: conversion-free q8_0 K and turbo3 V tile loaders for the fused MMA verify kernel (P5c).
- `52c723bb0` cuda: fused MMA attention for q8_0-K / q8_0-V caches (P6) -- gives the drafter's own cache the same kernel, and hands back 170-260 MiB.
- `ca1aec06f` spec: port upstream DFlash2 support (PR #27342) onto P6.
- `ba4c86d8e` cuda: `GGML_Q8_TURBO3_MMA_MAX_Q` routes widths 6..8 onto the fused (8,8) instance (DF3); built and measured, off by default.
- `628b065e1`, `3207e7fd7` speculative: exact p/q draft verification, including the sequential MTP drafter (PQ1); on by default, `LLAMA_SPEC_PQ=0` restores the old identity-match rule.
- `24aafbd9c` docs: run commands use the shipped p/q settings (`p_min 0`, q8_0/q8_0 drafter cache).
- `3772c377e` delta-net: only write the conv snapshots a ubatch can produce (RB1).
- `d2b16841f` cuda: SM86 cross-column reuse for Q5_0, and make the MMVQ launch table tunable.
- `b2a5cbd0f` recurrent: bound rollback by snapshots that were actually written (RB1b).
- `c2d647f79` cuda: raise MMVQ `rows_per_cuda_block` to 8 for the SM86 verify widths.
- `c3421dfe0` cuda: add `GGML_CUDA_QC4_NW1` launch-shape flag for MMVQ (off by default).
- `8feb74710` tests: reader-bound rollback gates for the recurrent state ring.
- `1c2811776` docs: `p_min` is a target-mirrored emission gate, and it stays off by default.
- `36a6bca81` merge RB1b into the QC kernel stack: bounded recurrent rollback + p/q docs.

RB1 and RB1b buy no speed. They fix recurrent rollback writing over history it should not have
touched -- the class of bug that surfaces as an unreproducible generation months later.

Two defaults changed from v0.2's run command: `--spec-draft-p-min` is now **0** (p/q wants the full
draft every round, where the old identity-match rule wanted a confidence gate), and
`--spec-draft-type-v` is now **q8_0** rather than turbo3, because P6 gave the drafter's own cache a
fused kernel. Draft depth stays at 3.

## v0.2 release notes (2026-09-07)

Measured on the same card and model as above, 102,400 generated tokens after a 685-token prompt
(EOS ignored, context 112,640, seed 6100, checkpoints every 5K tokens), server-reported decode rate.
Temperature 1, top-k 20, top-p 0.95, MTP depth 3:

| build | tok/s over 100K generated | at 25K / 50K / 75K / 100K (cumulative) | draft acceptance |
|---|---:|---|---:|
| upstream TurboQuant+ 2026-09-03 (`1208c5956`) | 55.98 | 59.0 / 58.7 / 57.0 / 55.8 | 0.660 |
| v0.1 as documented (`26e7bc523`, `GGML_Q8_TURBO3_MMA_FUSED=1`) | 66.09 | 64.8 / 66.2 / 65.7 / 65.8 | 0.660 |
| v0.1 + v0.2 exact kernels only (`43651b47e`) | 66.38 | 64.2 / 65.6 / 65.4 / 66.0 | 0.660 |
| v0.2 (kernels + 64K shortlist + runtime, `4017a1af4`) | 75.29 | 71.5 / 74.0 / 74.0 / 74.9 | 0.680 |

The first three rows decoded byte-identical text. v0.2 is +13.9% over v0.1 and +34.5% over upstream;
the exact kernel set alone is +0.4% over this run, so nearly all of the gain is the shortlist and the
runtime fixes. Temperature 0 (greedy, MTP depth 4, exactness runs for greedy users, not the benchmark
configuration): upstream 71.84, v0.2 98.54, v0.2 with `LLAMA_MTP_GPU_VERIFY=greedy` 99.10 with
byte-identical text over all 102,400 tokens.

What changed since v0.1, in merge order (each commit is exact unless noted):

- `f2fdec42f` cuda: Turbo3 verify loader selects centroids in registers instead of a constant table.
- `7b690c279` cuda: guard input access for source-free graphs.
- `485961968` cuda: IQ4_XS activation layout and cross-column weight reuse in MMVQ.
- `43651b47e` cuda: 16-byte cp.async staging for flat aligned Turbo3 V tiles.
- `ef86f905c`, `f7da4730d` speculative: draft-only vocabulary shortlist for the draft-mtp head (`--spec-draft-vocab-map`); changes draft proposals, never target verification.
- `e3696efb3`, `076c90eda`, `4017a1af4` llama, server: adaptive hot tail for the shortlist (`--spec-draft-vocab-hot`), measured slower, off by default.
- `f0a913279` llama-context: pinned output buffer kept stable across sampler changes (no per-request host realloc).
- `1ae96377c`, `82b355e86`, `13e022e0a`, `d8a35a9ad`, `0c5098b98`, `bd64685fc`, `1b9b70be1` GPU verification of MTP drafts, opt-in via `LLAMA_MTP_GPU_VERIFY` (greedy path exact; sampled path not a default, see the knob table).

Depth 3 remains the default: on real prompts at temperature 1 depth 4 was 1.8% slower with the map
(and 10.7% slower on a 140K retrieval prompt); depth 4 wins only at temperature 0. Draft KV in f16
was not evaluated in this round.

## Blackwell (RTX 50 series, RTX PRO 6000): what carries over

This tree was tuned and validated on an RTX 3090 Ti (sm_86). Nothing in it is compiled or tested for
Blackwell yet, but an audit of every CUDA change against upstream (2026-09-04) found no code path that
excludes it: the kernels select on capability helpers (`turing_mma_available`, `CP_ASYNC_AVAILABLE`,
device warp size), never on `cc == 86`, and upstream's CMake adds `120a-real` automatically when the CUDA
toolkit is 12.8 or newer. The one sm_86-only path is the opt-in MMVQ exact-reuse kernel
(`GGML_CUDA_SM86_EXACT_REUSE=1`), which stays off elsewhere by design.

Build (CUDA 12.8 or newer is required for sm_120; 12.4 cannot target it):

```
cmake -B build-sm120 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON
cmake --build build-sm120 -j --target llama-server llama-cli test-backend-ops llama-quantize llama-imatrix
```

First hour on the card, in this order:

1. `build-sm120/bin/test-backend-ops -o GATED_DELTA_NET`, then `-o FLASH_ATTN_EXT`, then `-o MUL_MAT_VEC`
   and `-o MUL_MAT`. These cover the GDN ILP kernel, the q8_0/Turbo3 attention loaders and the fused MMA
   path, and the MMVQ changes. All must pass before any timing is trusted.
2. Run the server with the same flags as the Ampere command above (the `GGML_Q8_TURBO3_MMA_FUSED=1` fused
   attention path is opt-in; try with and without it) and confirm output is identical to the sm_86 build on a
   greedy 16K prompt.
3. Only then measure. Expect the ranking of formats to shift: Blackwell has about 1.8x the memory bandwidth
   of the 3090 Ti but roughly 2.6x its integer instruction throughput, so the instruction-bound MMVQ work that
   made IQ4_XS and Q5_0 the fast choices here shrinks relative to weight streaming. IQ4_XS should still lead on
   bytes; the Q5_0-over-Q5_K margin may vanish; the int8 tensor-core (MMQ) crossover at speculative widths 3-5
   that was a wash on the 3090 is worth re-testing.

Kill switches if something regresses on the new card, each restoring the upstream kernel for that stage:

| knob | default | effect |
|---|---|---|
| `GGML_CUDA_SM86_GDN_COLS=1` | 4 | falls back to the original gated-delta-net kernel (the ILP kernel is qualified on sm_86 only) |
| `GGML_Q8_TURBO3_MMA_FUSED` unset | unset | keeps the fused q8_0/Turbo3 MMA attention off (it is opt-in) |
| `GGML_Q8_TURBO3_MMA_MIN_Q`, `GGML_Q8_TURBO3_MMA_NCOLS1_MIN` | 3, 1 | research knobs for routing narrow queries to the MMA path; leave at defaults |
| `GGML_CUDA_GRAPH_NO_SHAPE_KEY=1` | unset | disables the per-shape CUDA-graph cache |
| `GGML_CUDA_GRAPH_EVICT_S` | 300 | seconds an unused CUDA graph is kept |
| `LLAMA_MTP_GPU_VERIFY=greedy` or `=1` | unset (off) | enables GPU verification of MTP drafts; unset or `0` verifies every draft on the CPU. `=greedy` turns on the greedy path only: argmax on the GPU for `temperature 0`, `top_k 0`, `top_p 1`, `min_p 0` requests without grammar/n_probs/active penalties. Request logit biases (which is how `ignore_eos` is expressed) and model suppress tokens are honoured on the GPU: they are added to the logit rows in front of the argmax by the same logit-bias sampler the CPU chain uses, so no request is pushed back to the CPU for them. Active truncation must fall back because it can reorder tied logits and change the CPU-selected token. `=1` additionally turns on the sampled path: the whole default chain (top-k/top-p/min-p/temperature/dist) samples every verification row on the GPU and downloads only token ids; falls back to the CPU for grammar, active penalties, DRY, XTC, typical, top-n-sigma, mirostat, n_probs. The sampled path is not a default because its `[n_vocab, n_rows]` per-sequence working set can exhaust the device at 220K context. Same-seed sampled outputs can differ between GPU and CPU verification (RNG consumption order); GPU distribution/correctness gates remain required. Log lines: `GPU greedy verification enabled` / `GPU sampled verification enabled` / `GPU verification ineligible for sampler settings; using CPU verification` |

What needs no porting at all: the prompt cache, the recurrent checkpoints, the disk tier, the graph shape
cache and the shared compute arena are host-side or arch-independent. What will want retuning: the GDN ILP
kernel's grid (sized for 84 SMs; on 170-188 SMs a single sequence underfills the chip more), and the
attention staging, which uses cp.async where Blackwell's TMA would be the native primitive. Blackwell's FP8
and NVFP4 tensor-core paths are not used by this tree; a Blackwell-native quant on those would be the next
step there, not a port of the Ampere integer path.

## Credits

Qwen team for Qwen3.8; TheTom for TurboQuant+; Unsloth for the BF16 GGUF, the
importance matrix, and the tier ladder the recipe follows; ggml-org for
llama.cpp.
