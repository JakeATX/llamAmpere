# Preset checks before release

Current owner order (2026-11-17T04:48Z): 8dense is headless-only, with 114688 configured context (0.7.1, REL071 RTX 3070 peaks) and 500 MiB free. `8dense headed` notes this and uses the same defaults. Headed 10dense uses 180224 context; headed 12dense uses 196608 with CUDA graphs off. Headless 10/12 retain 212992/262144; 12dense headless full-context fit is unverified. Explicit CLI flags and shell environment retain precedence. Sources: common/preset.cpp and the SMALLUB real-card results in V07/PRESETS/SOL_HANDOFF.md; owner COORD-bc. Re-check / update every preset against final 0.7 measurements before ship.

Owner ruling 2026-11-17T05:24Z cancels the small-card RM24H change and its smoke test. All small presets retain auto:65536, explicit q8_0 draft K/V and their existing draft windows to preserve memory fit. RM24H needs full-context draft KV, estimated to add at least 150 MiB at 204800, beyond the measured small-card margins. The 24dense profile is unchanged. No speed or final fit claim.

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
