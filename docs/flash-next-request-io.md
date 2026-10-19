# Flash-Next request read accounting

The home shipping protocol permits n-gram/PLE reads. Every other model tensor must remain in RAM or VRAM. Unknown bytes reject a completed request; live unknown bytes above 1 GiB stop it.

`scripts/fnarena/trace-ship-nonweight.c` is the exact home diagnostic probe used by FNSHIP. Its owned output directory is intentionally fixed to `/mnt/models/fnship/output`. It is a benchmark tool, not a runtime default. Build with `cc -shared -fPIC -O2 -Wall -Wextra -Werror ... -ldl` and the installed CUDA headers. The extra syscall probes do not automatically grant read exemptions.

`activation_device_copies` accepts only successful CUDA device-to-owned-output operations. Before its counters start, the probe verifies a whole shared read/write deleted output mapping and checks that the source has CUDA device memory type. Registered host memory is excluded. The measured operation is one CUDA copy, without context/graph synchronization or host callback. The engine's only CUDA host callback is disabled by `#if 0`.

The four supported copy APIs are synchronous, asynchronous, 2D and 2D asynchronous. The 2D check includes every row and the pitch gaps. Counter failures, malformed supported records, request-boundary crossings, duplicate/overlapping records, incomplete final records and counts above process reads refuse attribution. Callers must merge the returned per-thread intervals with every other attribution category and enforce total-byte conservation. H2D, flush, close and other probe records remain unknown unless separately proved.

This identifies a nonweight output-copy operation. It does not claim which filesystem extent or driver code page supplied the physical read. It cannot authorize a host model-weight source. The actual CUDA fixture checks exact bytes through all four APIs and ordinary host-source delegation; the CPU fixture checks mapped-file contents, syscall binding and errno behavior.

New forensic analysis of the completed `short1510-ours8k` capture identifies only 12,288 of its 38,592,512 residual bytes in these operations. The remaining 38,580,224 bytes are unknown. The original FAILED_CONFIG result is retained; this is not a new speed measurement. Prospective captures may use the strict parser after source/fixture verification. The frozen engine remains `4f5282e14`; adding this benchmark parser does not create new engine build evidence.
