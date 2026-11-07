# llamAmpere v0.6.3.2

This hotfix addresses a Flash-Next crash and Windows model loading.

## Fixes

- Flash-Next could abort on a follow-up request that reused a cached prompt. A fused GPU operation could cross the point where the main stream must wait for a shared expert on another stream. Fused operations now stop at those stream boundaries. Shared-expert overlap stays enabled for one slot. Fixes [#88](https://github.com/JakeATX/llamAmpere/issues/88), reported by @justin025.
- Windows model loading could fail with "cannot allocate the direct-read bounce buffer" since v0.6.2. Aligned buffers now use matching Windows allocation and free calls, and Windows buffered file handles report their read mode correctly. Fixes [#82](https://github.com/JakeATX/llamAmpere/issues/82), reported by @fchamieh.
- Guards optional Windows thread-power APIs and Linux-only host-cache code so they compile on Windows. Expert-overlay reads use 64-bit file offsets on Windows.

## Limits

No speed gain is claimed. This release does not address the slower writing speed also reported in #88.

The Windows gate cross-compiles and links the CPU server and tests with MinGW. This is not a native Windows CUDA model-load test.

Fresh and restored conversations can choose different words where model scores are close. The prior release also showed run-to-run differences; this hotfix does not claim exact repeatability.

Linux installs build from source. This is a source-only release, with no prebuilt Linux binaries.
