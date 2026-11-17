# Preset checks before release

Each preset needs validation on the final combined build with its matching model and sidecar files. Historical results do not validate a new build.

| Preset | Required checks |
| --- | --- |
| 24dense | Load and decode below 23,552 MiB; one to four users, asymmetric context reservations, populated prompt-cache save/restore. The preset uses draft window 4096; automatic DSJ/RM24H applies to eligible window-0 paths. Packed masks remain opt-in. |
| 12dense | Launch, decode and full-context fit below 12,288 MiB on the matching hardware; model, map and index discovery; populated prompt-cache save/restore. |
| 10dense | Launch, decode and full-context fit below 10,240 MiB on the matching hardware; model, map and index discovery; populated prompt-cache save/restore; confirm the inherited `LLAMA_SPEC_MB_KL=8` setting. |
| 8dense | Launch, decode and full-context fit below 8,192 MiB on the matching hardware; model, map, overlay and head discovery; populated prompt-cache save/restore. Use the accepted F2 profile with reserve 7 rows, stage strips on, graphs off, 1% buffer growth and mean chunks on. Remeasure with copy off and check the CPU thread count: earlier measurements used six threads. |

All four presets enable `LLAMA_DEEPCTX_LADDER=1`, cleared by ASYMCTX/QC_READY_asymctx_24gb.md under the November 16 23:09 UTC owner order. Checks run at most every 2 seconds of prompt processing and 10 seconds of decode; unchanged sizes keep the existing allocation. The configured ceilings remain 1024/512/512/256 for 24/12/10/8dense. Explicit `-ub` wins, and `LLAMA_DEEPCTX_LADDER=0` keeps static sizing. The HOME smoke does not establish final small-card fit. SMALLPORT owns one rc2 8 GB startup plus short-chat check; SMALLUB owns the separately reopened real-card recheck. No full B8 rerun or extra dynamic-ub fit matrix is required.

Small-card presets support one user. TODO: multi-user for small cards (future).

Explicit flags and shell environment values override profile defaults. Presets do not download models. A launch with explicit sidecar paths does not validate default path discovery. Check natural server and queue exits, actual peak memory, and completed token output.

Copy defaults off. `LLAMA_SPEC_COPY=N` is an explicit opt-in. `GGML_CUDA_DISABLE_GRAPHS` is presence-based: even `0` disables graphs; the 8dense profile sets it.
