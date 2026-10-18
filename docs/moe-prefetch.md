# GPU expert prefetch

`--moe-prefetch off|auto|1|2|3` controls speculative expert copies for
Qwen4Exp decode. The default is `off`. `LLAMA_ARG_MOE_PREFETCH` accepts the
same values. This option requires the CUDA expert cache; it does not enable
expert skipping or change router choices.

After attention, the current layer's mixed FFN input is evaluated by
the next one to three routers on the GPU. Routers use the model’s softmax activation and produce up to 32
ranked candidates. For verification batches of up to eight tokens, ranking
uses each expert's largest score across the batch. The draft context and
larger batches do not predict.

The copy worker admits only uncached candidates, in descending router
order. Its byte allowance uses observed layer intervals and completed
host-to-device copy throughput, with 25% time headroom and a 16 MiB limit.
`auto` selects the shortest lead that can cover a complete expert. No copies
are admitted before timing samples exist. Real demand work takes priority;
queued predictions are canceled when demand arrives or their deadline
expires. An already running transfer completes normally.

Copies use at most four reserved slots per format pool inside the cache's
existing allocation. Demand eviction cannot overwrite these slots. Device
dispatch publishes a completed staging address only after the real router
selects that expert. The address is restored before the slot is reused.
Legacy dispatch pins the staged entry through its normal collection path.
Both paths retire speculative entries at the end of the graph scope. The optional
fused gate/up provider uses both staging tables and accounts for each copy. CUDA graph launches also account for staged
projections consumed during replay, which bypasses host dispatch.

Each `[moe-prefetch]` log row describes one tensor projection copy:
`source`, `target`, `expert`, `rank`, `bytes`, `issue_us`, `start_us`,
`done_us`, `target_start_us`, `selected`, `canceled`, and `outcome`.
Times use the process monotonic clock in microseconds. A canceled copy has
`done_us=-1`. `outcome=used` means an expert kernel consumed the copied
weights; selection alone does not count as use. Gate, up, and down copies
are logged separately.

The backend interface appends `prefetch_configure`, `prefetch`, and
`prefetch_observe` to `ggml_moe_cache_api`. Configuration happens before
pool sizing. A null observation ID array marks layer entry; real routed
IDs mark selection. Prediction IDs are side outputs and never feed model
math. Prefetch has no serialized state and is drained before cache reset
or source-buffer teardown.

`test-moe-cache --prefetch-cpu-only` checks budget arithmetic and router
ranking without discovering a GPU. The normal cache test also checks CUDA
ranking and four rounds of copy, use, and retirement for legacy, device, and
fused device dispatch.
`test-moe-prefetch-model` is a manual model gate: set `FN_GOLDEN_IDS` to
2,080 fixed int32 token IDs and `FN_GOLDEN_LOGITS` to an output path. It
writes 32 full-vocabulary logit rows and checks an exact saved-state round
trip. Model and full-protocol performance acceptance are separate gates.
