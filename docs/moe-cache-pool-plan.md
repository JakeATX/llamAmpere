# Expert-cache pool limits

`GGML_MOE_CACHE_POOL_PLAN=/absolute/path/plan.bin` opts into per-format slot limits for the global profile planner. Use it with a loaded expert profile. The planner still selects whole experts, honors the current memory budget and may fit fewer experts under allocation pressure. The limits affect adaptive cache pools, not permanent resident banks. Unset the variable for normal allocation.

The file contains little-endian unsigned 64-bit integers:

- Magic `0x314c4f4f50454f4d` (the eight bytes `MOEPOOL1`).
- Number of format pools.
- One `(ggml_type, expert_bytes, slots)` row per pool, in any order.

Every runtime pool must appear exactly once. The type and logical bytes per expert must match the model. Slot counts must be at least 64 and no greater than the pool's model capacity. Truncated files, extra bytes, duplicate or unknown pools, and invalid capacities stop execution. The file is checked again when the global profile is rebuilt; keep it unchanged for a run.

The `pool plan loaded` log confirms that the file was accepted. The following `global profile capacity` lines report planned and actual slots. Actual allocations can be smaller under memory pressure. Pool allocation and resize enforce the limits, including post-prompt replanning and memory returned after a prompt loan. If the global planner is unavailable, execution stops instead of falling back to unconstrained pools. Prompt-cache save and restore still need a runtime check on the candidate build.

A pool plan controls capacities. It does not encode which experts are selected, change the loaded profile's ranking, freeze experts in place or establish a speed prediction. Match the ranking and adaptive policy to the simulator before comparing route counts.

Repeated warm calls reuse slabs when every planned pool capacity is unchanged. Sessions without a loaded profile update the whole-expert plan from the latest nonzero prompt counts; saved-profile sessions retain their profile order. Prompt lending uses those counts to choose survivors.
