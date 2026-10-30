# v0.6.2 candidate

This candidate still needs its combined build, correctness, saved-prompt, quality, and product-speed checks. Earlier measurements do not qualify this binary.

`scripts/flash-next-server.sh -m MODEL` selects the Flash-Next launch defaults:

- SJ-KVaRN 4-bit keys and values, adaptive MTP depth 3 to 4, and mb-MTP enabled.
- The Q2S drafter and bundled `data/moe/flash-next-experts.bin` expert profile from `data/moe/flash-next.env`; override their paths with `FLASH_NEXT_DRAFTER` and `FLASH_NEXT_PROFILE`.
- `--moe-skip`: skip low-weight experts that are off the card within a 15% cumulative router-weight budget, protecting the first five routes. This changes the model's answers. Resident experts and prompt processing keep their normal computation.
- Two-worker PLE page-touch prefetch and the CPU expert split. Set `LLAMA_PLE_PREFETCH=0` to disable prefetch; existing environment overrides are kept.

Use `scripts/flash-next-server.sh --exact -m MODEL` to turn off expert skipping and mb-MTP in one step. This launcher option must come first. It expands to `--moe-skip off --moe-skip-budget 0 --moe-skip-rank-floor 0 --no-spec-mb-mtp`. For the bare server, `--moe-skip off` disables skipping; `--no-spec-mb-mtp` controls exact speculative sampling separately. Later ordinary server arguments override the launch defaults; the leading `--exact` wins over skip and mb-MTP options.

The bare server keeps expert skipping off. `--moe-skip` (or `LLAMA_ARG_MOE_SKIP=on`) selects budget 0.15 and rank floor 5; `off` disables the preset. Detailed budget and floor flags override the preset in either argument order. `--moe-skip-rank-floor N` accepts 0 through 10 and can also be set with `LLAMA_ARG_MOE_SKIP_RANK_FLOOR`. A floor of 0 protects no routes; 10 protects all ten. `--moe-skip-budget` accepts 0 through 0.25; 0 disables the budget. Budget mode works independently of a route-rank cutoff.

The AI+ decode correction adds eligible warp partials in a fixed order inside each block (`GGML_CUDA_AIT_CTA=2`, default). Ineligible shapes keep the ordered scratch-buffer reduction. This fixes repeatability on the tested issue62 inputs but changes floating-point grouping relative to the prior ordered path. `GGML_CUDA_AIT_CTA=0` keeps that prior path; `GGML_CUDA_AIT_ORDERED=0` selects the older unordered comparison. Existing experimental kernel switches stay opt-in.

AI+ buffer retention, new replay kernels, int8 verify, register RMS, and unqualified CPU split experiments are not enabled by this release configuration.

Mapped expert sources require a complete tracked host registration; endpoint-only borrowing is removed. Table rebuilds reuse session-owned mappings through their normal retention accounting so pruning cannot drop a borrowed source registration. `GGML_MOE_CACHE_STRICT_ALIAS=0` no longer weakens this check.
