# llamAmpere v0.6.2

Release checks: the saved-prompt save, erase and restore test gave three identical runs that match the release candidate exactly (30,455 tokens restored); the 2.3 bpw model loads and answers correctly; 20 of 22 CPU tests pass, and the other two (`test-mmap-prefetch` and `test-sjkvarn-state --scalar-only`) fail the same way in the unmodified candidate.

## Quick start: Swift 1.5 27B at 2.3 bpw, 200K context, 24 GB RTX 30-series card

Requirements: Linux, an NVIDIA driver with the CUDA 12 or newer toolkit (`nvcc`), CMake, git and a C++ compiler; the model is a ~9.1 GB download on first start.

This one paste clones llamAmpere at v0.6.2 (or updates an existing `llamAmpere` folder to v0.6.2), builds `llama-server` with HTTPS support so `-hf` can download the model, and starts the server with the recommended flags below. It downloads [jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2) on the first start.

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.6.2
else
  git clone -b v0.6.2 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
./build-sm86/bin/llama-server -hf jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2 \
  -c 204800 -ngl 99 -fa on -b 4096 -ub 1024 --parallel 1 --jinja --no-context-shift \
  -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 3 \
  --spec-draft-p-min 0 --spec-mb-mtp
```

The server listens on http://127.0.0.1:8080 and serves an OpenAI-compatible chat API there. This block was tested on an RTX 30-series card (`sm_86`) only. For another NVIDIA card generation, change `86` in `-DCMAKE_CUDA_ARCHITECTURES=86` to your card's CUDA compute capability number without the dot (NVIDIA lists it for every card); other generations have not been tested.

## Recommended flags

For the Swift 1.5 27B dense model (the 2.3 bpw file above) at 200K context on a 24 GB RTX 30-series card, with the SJ-KVaRN 3/2 KV cache:

| Flag | What it does |
|---|---|
| `-hf jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2` | Downloads the model's single GGUF file from Hugging Face on first start and reuses the saved copy afterward. |
| `-c 204800` | Gives the model 204,800 tokens (200K) of context. |
| `-ngl 99` | Puts every model layer on the graphics card. |
| `-fa on` | Turns on flash attention, the attention path that reads the SJ-KVaRN cache. |
| `-b 4096 -ub 1024` | Reads prompts in 4,096-token batches, 1,024 tokens per card step, which speeds up long prompts. |
| `--parallel 1` | Uses one conversation slot so the whole 200K context belongs to one conversation (the server otherwise picks 4 slots that share it). |
| `--jinja` | Formats chats with the chat template stored in the model file (already the default). |
| `--no-context-shift` | Stops a reply at the context limit instead of silently dropping the oldest text (already the default). |
| `-ctk sjkvarn3 -ctv sjkvarn2` | Stores the attention memory (KV cache) in our compressed SJ-KVaRN format with 3 bits per key and 2 bits per value, the 3/2 setting (2.78 bits per element in the compressed cache body, versus 4.28 for 4/4). |
| `--sjkvarn-body-type auto` | Picks the trellis encoding that the 3/2 setting needs. |
| `--spec-type draft-mtp-adaptive` | Turns on speculative decoding with the draft layer built into the model file, which guesses tokens ahead so the full model can check several at once. |
| `--spec-draft-n-min-adaptive 3 --spec-draft-n-max 3` | Drafts 3 tokens per round, the setting used for the model card's results. |
| `--spec-draft-p-min 0` | Always drafts the full 3 tokens instead of stopping early when the draft layer is unsure. |
| `--spec-mb-mtp` | Turns on mb-MTP, which also keeps a draft token the full model would have rejected when that token is at least as likely as the replacement; this is faster but changes sampling slightly, so use `--no-spec-mb-mtp` instead for exact sampling (mb-MTP is already on by default with MTP). |

## Changes

`scripts/flash-next-server.sh -m MODEL` selects the Flash-Next launch defaults:

- SJ-KVaRN 4-bit keys and values, adaptive MTP depth 3 to 4, and mb-MTP enabled.
- Our Q2S drafter and bundled `data/moe/flash-next-experts.strp` expert list (our v2-r-pure-skip15 list) from `data/moe/flash-next.env`; override their paths with `FLASH_NEXT_DRAFTER` and `FLASH_NEXT_PROFILE`.
- `--moe-skip`: skip low-weight experts that are off the card within a 15% cumulative router-weight budget, protecting the first five routes. This changes the model's answers. Resident experts and prompt processing keep their normal computation.
- Two-worker PLE page-touch prefetch and the CPU expert split. Set `LLAMA_PLE_PREFETCH=0` to disable prefetch; existing environment overrides are kept.

Use `scripts/flash-next-server.sh --exact -m MODEL` to turn off expert skipping and mb-MTP in one step. This launcher option must come first. It expands to `--moe-skip off --moe-skip-budget 0 --moe-skip-rank-floor 0 --no-spec-mb-mtp`. For the bare server, `--moe-skip off` disables skipping; `--no-spec-mb-mtp` controls exact speculative sampling separately. Later ordinary server arguments override the launch defaults; the leading `--exact` wins over skip and mb-MTP options.

The bare server keeps expert skipping off. `--moe-skip` (or `LLAMA_ARG_MOE_SKIP=on`) selects budget 0.15 and rank floor 5; `off` disables the preset. Detailed budget and floor flags override the preset in either argument order. `--moe-skip-rank-floor N` accepts 0 through 10 and can also be set with `LLAMA_ARG_MOE_SKIP_RANK_FLOOR`. A floor of 0 protects no routes; 10 protects all ten. `--moe-skip-budget` accepts 0 through 0.25; 0 disables the budget. Budget mode works independently of a route-rank cutoff.

The AI+ decode correction adds eligible warp partials in a fixed order inside each block (`GGML_CUDA_AIT_CTA=2`, default). Ineligible shapes keep the ordered scratch-buffer reduction. This fixes repeatability on the tested issue62 inputs but changes floating-point grouping relative to the prior ordered path. `GGML_CUDA_AIT_CTA=0` keeps that prior path; `GGML_CUDA_AIT_ORDERED=0` selects the older unordered comparison. Existing experimental kernel switches stay opt-in.

AI+ buffer retention, new replay kernels, int8 verify, register RMS, and unqualified CPU split experiments are not enabled by this release configuration.

Small verification batches now use one expert-sum reduction by default; set `LLAMA_MOE_SMALL_REDUCE=0` to use the previous add chain. Direct AI+ verification uses compact partial sums and writes skipped outputs in the same kernel by default. Set `GGML_CUDA_AI23_VERIFY_COMPACT=0` to disable it. The server also disables compact verification when the memory fit fails, is disabled, or requests less than 300 MiB of free memory on any device. This uses the fit's minimum guaranteed margin, so a smaller requested margin disables compact even if the actual free space is larger. Both reductions use valid floating-point math and can change rounding and generated text.

The bank/cache joint reduction is included with `LLAMA_MOE_BANK_JOINT_REDUCE=1`, but remains off pending its CUDA check. Bank staging and streamed-copy checks are included; pageable bank retention still requires `LLAMA_B1_KEEP_BANK_RAM=1`.

Mapped expert sources require a complete tracked host registration; endpoint-only borrowing is removed. Table rebuilds reuse session-owned mappings through their normal retention accounting so pruning cannot drop a borrowed source registration. `GGML_MOE_CACHE_STRICT_ALIAS=0` no longer weakens this check.

Additional candidate code includes the short-sink saved-state fix and per-token streamed byte fix. The launcher places `token_embd.weight` on CUDA0 with `-ot`. New prompt variants `LLAMA_PLE_RAW_ROWS`, `GGML_CUDA_HC_POST_V4`, and `GGML_CUDA_LID_SPLIT16` remain off unless enabled. `GGML_CUDA_MOE_REDUCE_VEC4=1` enables vector reduction; `GGML_CUDA_MOE_REDUCE_THREADS=128` selects its smaller block size (default 256). CTA2 neighbor sharing stays off. Joint reduction also stays off pending its corrected CUDA checks.

New 2.5-bit fractional AI+ tensor type (type id 118, `aif25_u`) so the published ATX AI2 2.3 bpw 27B model loads and runs. It packs two weights into 5 bits with one fp16 scale per 128 weights and runs through a CUDA matrix-multiply path on Ampere and newer cards. Files that use it must carry the `ait.fractional.contract` metadata key; files without it are refused at load.

Speculative decoding now keeps each slot's verification state when a draft is only partly accepted and the slot rolls back to a checkpoint, with or without mb-MTP. The adaptive depth controller sees the draft width that was actually tried, and a stop string inside an accepted draft ends the round.
