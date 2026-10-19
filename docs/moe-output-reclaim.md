# Completed MTP output pages

`LLAMA_MOE_OUTPUT_RECLAIM=1` opts into releasing completed unmasked MTP output batches when `LLAMA_MOE_ARENA_OUTPUT_DIR` selects file-backed activation storage. It does not change weights, quantization, output floats, or the expert arena.

After each batch's device-to-host copy, the scheduler is synchronized before any pages are released. The completed byte range is flushed synchronously, unmapped from the process working set with `MADV_DONTNEED`, and dropped from the owned file's page cache with `POSIX_FADV_DONTNEED` on Linux. The mapping and its unlinked file remain alive. Later helper reads refault the original activation bytes from this temporary file. This adds activation I/O and a synchronization boundary; speed must be measured separately. Activation reads are not model-weight reads and must be identified separately by request I/O instrumentation.

Partial pages can overlap adjacent output rows. All scheduled writes finish before the page-aligned flush and release, so neighboring bytes are preserved. The usual final output flush remains in place. The opt-in is off by default, and allocations smaller than the existing file-backed threshold are unchanged.

The MTP driver's `process_range` also releases consumed rows after all helper heads and carry copies complete. This bounds refaulted activation pages during helper prefill. The verification indices remain valid: a later accept reads the same bytes from the retained file mapping.

`test-moe-arena` covers exact arbitrary float bits after unaligned release, empty and invalid ranges, subsequent writes, and absence of resident file pages before refault. `test-moe-arena --cuda-output` additionally covers asynchronous CUDA output copying and completion before release. These operator checks do not establish full-model quality, saved-cache reload, memory fit, or speed.
