# 0.7 release notes (draft)

Dense 27B card presets: `--preset 24dense`, `12dense`, `10dense`, or `8dense` selects the best known settings for the card size. Small-card profiles support one user. Explicit flags override the profile.

TODO: Re-check / update every preset against final 0.7 measurements before ship. See [the preset recheck list](../../PRESETS_RECHECK.md).

8dense uses the accepted F2 fit profile: reserve 7 rows, stage strips on, CUDA graphs off, 1% buffer growth, and mean chunks on. The final build must include the F2 reserve code. Copy mode follows the engine default (off); explicit environment values still override preset values.
