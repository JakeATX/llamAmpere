# Experimental GPU expert-cache dispatch

`GGML_MOE_CACHE_DEVICE=1` opts into device-side dispatch for the runtime MoE cache. It is off by default. Use the normal expert-cache configuration as well. This path preserves every selected expert and the original compressed weights; it does not skip experts or substitute a lower precision weight format.

At an idle boundary, `ggml_backend_sched_moe_cache_device_prepare` publishes one stable address table per expert tensor. Each entry points to a resident cache slot or an exact CUDA-addressable host copy. The CUDA MUL_MAT_ID kernel reads expert IDs and resolves addresses on the card. Hits need no host routing readback, per-expert stream synchronization, activation download or result upload. Cold entries currently read mapped pinned RAM; asynchronous miss staging is a separate feature.

The initial implementation supports one CUDA device on Linux. Preparation validates tensor shapes, types, allocation sizes and host/card headroom. Unsupported shapes and IQ1_M keep the existing exact CPU/cache path. Preparation can be expensive on drivers that cannot register a read-only model mapping: exact pinned copies are allocated for nonresident weights. That setup cost and host memory pressure must be included in request timing.

`GGML_MOE_CACHE_DEVICE_CAPTURE=1` separately opts into CUDA graph capture. A globally unique preparation generation invalidates cached graph executables when a table is released, replaced or belongs to another session. All table and slot addresses remain fixed while graphs reference them. The scheduler rejects stale device assignments rather than copying a complete host expert bank to the card.

Adaptation runs only when no graph scope or expert node is active. `GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL` sets the boundary interval (1-1024 scopes, default 64). `GGML_MOE_CACHE_DEVICE_ADAPT_MB` caps attempted copy bytes at each boundary (1-1024 MiB, default 64). Evicted entries first receive a stable exact host fallback. When available, promotion reads the retained pinned copy instead of faulting the original file mapping back into RAM. Trimming, warming, teardown and source invalidation retire prepared tables before changing ownership.

`GGML_MOE_CACHE_DEVICE_STATS=1` enables device hit/miss/dispatch counters and periodic diagnostic snapshots. Snapshots synchronize and can affect timing. Legacy host-dispatch counters exclude device-routed work. A cache hit rate must use device counters when device dispatch is active.

The cache tests cover supported compressed types, mixed resident/mapped routes, strided routing IDs, read-only source mappings, host budgets, adaptation, source reuse and graph lifetime across preparation generations. Passing those checks does not establish a model speed improvement. Keep this feature opt-in until full-protocol comparisons pass.
