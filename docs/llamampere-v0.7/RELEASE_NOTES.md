# llamAmpere v0.7 release candidate

Final release validation is pending.

llamAmpere is based on mainline llama.cpp (v1.0). With 1.0 we prioritize keeping up with mainline. This release includes TheTom TurboQuant work, credited in [CREDITS.md](../../CREDITS.md).

Dense 27B card presets: `--preset 24dense`, `12dense`, `10dense`, or `8dense` selects the best known settings for the card size. Small-card profiles support one user. Explicit flags override the profile.

TODO: Re-check / update every preset against final 0.7 measurements before ship. See [the preset recheck list](../../PRESETS_RECHECK.md).

All four presets enable dynamic microbatch sizing with checks at most every 2 seconds of prompt processing and 10 seconds of decode. Explicit `-ub` sets the ceiling; `LLAMA_DEEPCTX_LADDER=0` keeps static sizing. Small-card profiles still require one user.

8dense uses the accepted F2 fit profile: reserve 7 rows, stage strips on, CUDA graphs off, 1% buffer growth, and mean chunks on. This build includes the F2 reserve code. Copy mode follows the engine default (off); explicit environment values still override preset values.

Speculative copy mode is off by default. Set `LLAMA_SPEC_COPY=N` to opt in with a copy tail of N tokens.

Explicit environment values keep their underlying engine meaning. `GGML_CUDA_DISABLE_GRAPHS` is presence-based: even `0` disables graphs. The 8dense preset sets it, so its environment override cannot turn graphs back on.

The accepted 8dense measurements used 6 CPU threads; the preset currently inherits the engine thread default. Recheck that difference before ship.

Eligible single-sequence internal MTP paths with a SJ-KVaRN trunk and draft window 0 select DSJ/RM24H automatically. Pass `--no-spec-draft-sjkvarn` to opt out. The 24dense preset uses draft window 4096 and does not select this automatic path.
