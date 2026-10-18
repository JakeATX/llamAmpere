# Decode expert prefetch validation

The code uses GPU matrix multiplication, the model's gate activation and bias, and GPU top-k sorting. Only ranked expert IDs return to the host. It never changes routing IDs used by the model, drops experts, changes weights, or runs router prediction on the CPU.

`--moe-prefetch off|auto|1|2|3` defaults to `off`. `auto` evaluates up to three later routers, then admits the shortest measured lead that covers a complete expert transfer. Fixed depths consider every later layer up to that depth. The current graph hook covers single-token trunk decode. Prompt batches and MTP verification batches keep their regular graphs.

Admission waits for measured layer intervals and at least four completed demand transfers. The transfer rate includes RAM-to-pinned staging and completion of the card copy. A 25% deadline margin and a 16 MiB per-issue cap bound traffic. Already resident or copying entries are skipped. Queued demand fills precede predictions; a demand for a queued prediction promotes it. A transfer already in flight completes before the worker takes the next job.

Up to four slots per format pool serve as staging. They use the cache's existing slab, generations, source invalidation, and reader pins. Reserving staging can displace cold demand entries once. Predictions reuse only closed, unpinned staging entries. GPU results are counted as used only after successful collection. Logs include issue, start, completion, target entry, actual selection, cancellation, and used/wasted outcomes. Canceled work has `done_us=-1`; it did not complete a card transfer.

`test-moe-cache --prefetch-policy-only` runs deterministic CPU budget checks. `--prefetch-cpu-only` also checks router graphs on the CPU. The normal test checks exact top-32 IDs on CPU and CUDA for sigmoid, softmax, and square-root softplus routers, plus the existing cache tests.

The validation harness at `/mnt/models/fnprefetch/protocol/golden.cpp` feeds identical token IDs, writes all vocabulary logits for 32 decode positions, and checks exact state-save/state-restore bytes. `/mnt/models/fnprefetch/compare-golden.py` reports full-distribution KL for each flag mode against `off`. The guarded runners require the shared GPU lock and active system guard. Full speed acceptance still requires interleaved 100K + 5,120-token runs at temperature 1.0 against the current committed baseline.

The device-dispatch table path must consume these staging slots through the FNMISS interface before prefetch is enabled together with that path. Stable device tables must not point at a slot that prefetch can recycle.
