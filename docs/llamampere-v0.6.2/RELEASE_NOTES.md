# v0.6.2 candidate

This candidate still needs its combined build, correctness, saved-prompt, quality, and product-speed checks. Earlier measurements do not qualify this binary.

`scripts/flash-next-server.sh -m MODEL` selects the Flash-Next launch defaults:

- SJ-KVaRN 4-bit keys and values, adaptive MTP depth 3 to 4, and mb-MTP enabled.
- Our Q2S drafter and bundled `data/moe/flash-next-experts.strp` expert list (our v2-r-pure list) from `data/moe/flash-next.env`; override their paths with `FLASH_NEXT_DRAFTER` and `FLASH_NEXT_PROFILE`.
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
