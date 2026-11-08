# llamAmpere v0.6.3.5

Flash-Next leaves more temporary card memory for the first long prompt on small cards. This makes the v0.6.3.4 small-card memory workaround the launcher default.

## Small-card memory cushion

The launcher sets `GGML_CUDA_MOE_CACHE_RESERVE_MB=1536` and `LLAMA_PROMPT_TEMP_HEADROOM=1` when a selected card has less than 23,552 MiB total memory or less than 14,336 MiB free. A normally free 24 GB card keeps its previous settings.

Explicit values already in your environment still win, including zero or an empty value. Remove old overrides to use the new defaults. The larger cushion also applies to 16 GB cards and busy 24 GB cards in this tier; it leaves about 608 MiB less space for the expert cache than the previous default.

## CUDA allocation fix

Includes the fix from #71 that keeps GDN convolution-normalization allocation dependencies stable across batch widths. This avoids repeated memory planning and full synchronization on pipelined multi-GPU splits. Thanks to bordman20 for the contribution.

## Optional decode diagnostic setting

`GGML_MOE_CACHE_DEVICE_STATS_EVERY=0` disables periodic device-route diagnostic snapshots. The route counters and resident expert mask stay enabled. The default interval remains 64; this release makes no speed claim for changing it.

## Hardware limits

Small-card checks use an RTX 3090 with memory held aside to exercise the 12 GB launcher path. Native 12 GB and 16 GB cards remain unmeasured, and 64 GB of system RAM is not a fit promise. The launcher still uses `--fit off`; automatic multi-card placement remains outside this release.
