# B1 allocator page diagnostic

`LLAMA_B1_MALLOC_TRIM=1` opts into glibc `malloc_trim(0)` after B1 preparation and after each completed layer range. It is off by default and does nothing on other allocators. `LLAMA_MALLOC_TRIM=1` remains the separate startup-only option.

This releases only pages that the allocator considers unused. Live model weights, pinned buffers, activations and checkpoint contents are preserved. The call needs no GPU synchronization. It can add allocator lock contention and page faults on subsequent allocations, so measure end-to-end performance before enabling it for speed runs.

On glibc 2.33 or later the log includes allocator arena, allocated, free and mmap bytes before and after trimming, plus call duration. Allocator free bytes can include pages that are already nonresident. The `released` field is the allocator's return value, not a byte count or proof of lower RSS. Use independent `/proc/<pid>/smaps_rollup` and `status` samples to measure physical memory. Counts can change concurrently with other threads.

`python3 scripts/fnarena/test-b1-malloc-trim.py` compiles the actual production helper into a controlled CPU allocation fixture. It checks default-off behavior, exact enabled value, errno preservation, live bytes, live worker allocations, and physical release of freed heap pages. Its RSS saving is synthetic and must not be reported as a model saving.
