# Completed MTP output pages

`LLAMA_MOE_OUTPUT_RECLAIM=1` opts into releasing completed unmasked MTP output batches when `LLAMA_MOE_ARENA_OUTPUT_DIR` selects file-backed activation storage. It does not change weights, quantization, output floats, or the expert arena.

After each batch's device-to-host copy, the scheduler is synchronized before any pages are released. The completed byte range is flushed synchronously, unmapped from the process working set with `MADV_DONTNEED`, and dropped from the owned file's page cache with `POSIX_FADV_DONTNEED` on Linux. The mapping and its unlinked file remain alive. Later helper reads refault the original activation bytes from this temporary file. This adds activation I/O and a synchronization boundary; speed must be measured separately. Activation reads are not model-weight reads and must be identified separately by request I/O instrumentation.

Partial pages can overlap adjacent output rows. All scheduled writes finish before the page-aligned flush and release, so neighboring bytes are preserved. The usual final output flush remains in place. The opt-in is off by default, and allocations smaller than the existing file-backed threshold are unchanged.

The MTP driver's `process_range` also releases consumed rows after all helper heads and carry copies complete. This bounds refaulted activation pages during helper prefill. The verification indices remain valid: a later accept reads the same bytes from the retained file mapping.

`test-moe-arena` covers exact arbitrary float bits after unaligned release, empty and invalid ranges, subsequent writes, and absence of resident file pages before refault. `test-moe-arena --cuda-output` additionally covers asynchronous CUDA output copying and completion before release. These operator checks do not establish full-model quality, saved-cache reload, memory fit, or speed.

For request I/O accounting, `llama_copy_embeddings_nextn` synchronizes the context before staging. Its entire duration must **not** be attributed to activation reads. Only the separate exported `llama_copy_moe_output` function performs the pure `memcpy`, after synchronization, output reordering, flushing, and source-range validation. The optional preload tracer verifies the entire source span in `/proc/self/maps` against the canonical owned output directory and deleted activation filename before sampling per-thread `read_bytes`. The parser requires that same directory and rejects unknown sources, malformed/overlapping intervals, boundary crossings, and nonconserving counts. The cold-file fixture proves reads performed before this narrow boundary remain unattributed. This does not retroactively identify any old request's unknown bytes.

The separate metadata parser counts successful `mkstemp` calls only for a new
empty activation file in the canonical owned output directory, and successful
`posix_fallocate` calls only for its deleted-file name. Creation cannot read model
tensor payload. Both scopes expose their thread intervals so a combined harness
can reject overlap with tensor or activation-copy attribution. Unknown request
bytes still reject acceptance; a parser fix is not a new model measurement.
