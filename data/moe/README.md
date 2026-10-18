# Flash-Next startup expert profile

`flash-next.profile` uses the MOEPROF1 usage-count format from fn/fast. Its model identity and checksum are verified before any expert is warmed. The loader does not change a single quantized weight block.

Generate it with `scripts/profile-moe-arena.py` against the checked-in coding, agentic, and retrieval fixtures. The script records the measured counts and fixture hashes. Startup orders each tensor's measured use rates per expert byte, with a boost of `1 + 4/(layer + 1)` for early layers, and fills unused capacity in that order. A user-supplied `--moe-profile FILE` takes precedence over this default.

The arena uses one pinned host allocation and three strided tensor views per layer. Set `--moe-arena off` to retain the original loading path, or `--moe-arena on` to require the arena and fail clearly when the startup memory check does not pass. With no flag it is enabled only when the complete host arena fits with 12 GiB of startup headroom. Loading stops below 10 GiB available host memory. Automatic in-memory prompt caching preserves 11 GiB after load; disk prompt-cache serialization remains supported.

The automatic card budget is free VRAM after model weights, target and draft KV, and MTP allocations, less 2,560 MiB for compute workspaces and a reserve. An explicit `GGML_CUDA_MOE_CACHE_RESERVE_MB` setting overrides that reserve. This value follows the home card's short-prompt workspace gate; a 1,536 MiB reserve exceeded the required whole-card VRAM cap.
