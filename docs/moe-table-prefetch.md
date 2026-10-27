# Experimental expert transition-table prefetch

This path is disabled by default and is not yet qualified as a speed improvement.
It predicts future cold experts from the current layer's actual routed IDs. It
does not run another router or evict experts from the resident cache.

Required settings:

```sh
GGML_MOE_CACHE_DEVICE=1
GGML_MOE_CACHE_DEVICE_STATS=1
GGML_MOE_CACHE_DEVICE_FUSED=0
GGML_MOE_MISSJOIN=stage
GGML_MOE_MISSJOIN_OVERLAP=0
GGML_MOE_TABLE_PREFETCH=on
GGML_MOE_TABLE_MODEL=/absolute/path/to/calibrated-table.bin
GGML_MOE_TABLE_CAP=20
GGML_MOE_TABLE_BUDGET_BYTES=57600000
```

Use `--moe-prefetch off` and leave the legacy asynchronous prefetch disabled.
The initial implementation requires one CUDA device on Linux, CUDA 11.7 or
newer, one compute stream, 512 experts, and at most 128 actual routes per
dispatch. Concurrent scopes are rejected. The cap is 15 or 20; the byte budget
applies to each target layer's missing projections. Replay qualification uses
28,800,000 bytes times the artifact's lead distance (one, two or three layers).

`pred` performs prediction and snapshots without copying. `stub` replaces the
transition score with a fixed ranking derived from training coefficients and
performs real copies. `on` performs prediction and copies. All three reserve the
same workspace before the expert pools, so an off-versus-pred timing includes
both prediction work and the effect of that reservation on cache capacity.

The signed, little-endian `MTPFTB02` artifact contains projection geometry,
per-target int16 coefficients/biases, scoring flags and a payload SHA256. The
loader rejects malformed files, unsupported dimensions and active final-layer
targets. The predeclared `exclude_final_v1` policy avoids predicting a target
whose output rows may be cropped. Its missed bytes still count in full-model
replay recall. Artifacts are model-specific and must pass held-out useful-byte
precision above 50 percent before a speed comparison.

Prediction is folded into the existing route-count dispatch. Missing projections
are pulled from their mapped host aliases into separate ring slots on a low
priority stream. Slots publish only after their complete payload is visible;
the compute stream never waits for a target copy. Late work is canceled between
bounded slices, and the copy stream rejoins at the graph tail before plan or
workspace reuse. MISSJOIN consumes the resulting pointer/hot-mask snapshot.

Shutdown statistics count unique useful projection bytes, actual copied bytes
(including canceled partial copies), served routes and copy launches. The
reported kernel span excludes FIFO waits and is not copy-engine busy time.

Full model/output, SJ-KVaRN save/restore and CUDA operator gates remain required;
compilation or replay alone does not qualify the runtime.

## Run-to-run variation with expert offload

When some experts run on the CPU and others on the GPU, the two do the same math but round slightly differently, because they add numbers in a different order and at different precision. Which experts are on the GPU at a given moment depends on what the expert cache holds, and that depends on earlier requests and on the timing of background swaps. As a result, the same prompt and seed can produce slightly different text from one run to the next. The output is still correct, and this is normal for expert offload in any mixture-of-experts engine. If you need bit-for-bit repeatability, keep every expert on the GPU or every expert on the CPU.
