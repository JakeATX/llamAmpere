# Startup PLE row cache

`LLAMA_PLE_PROFILE=/absolute/path/to/profile.rows` opts into a lossless compact RAM cache for a lazily mapped Qwen4Exp per-layer word table. Rows listed in the profile are copied at model startup and locked in RAM. Quantized blocks are unchanged. The original model tensor, its shape, and model-file identity remain unchanged; the gather uses a separate compact tensor and translates row IDs before running.

Every row in each actual graph input is checked. A missing or invalid row logs its count and stops the process before the gather. There is no request-time weight read, zero row, approximation, or disk fallback. The page-prefetch path is disabled for this table. Successful lookup counts are logged at debug level; miss counts are always logged. This protects PLE only; the run must still audit experts and every other model-weight source.

Create profiles from saved tokenizer IDs with:

```sh
python3 scripts/profile-ple-rows.py --model MODEL.gguf --prompt prompt-tokens.json --generated generated-tokens.json --out profiles
```

The tool reads GGUF metadata, not weight payloads. `prompt.rows` contains only prompt rows; the report measures misses on the saved continuation with the prompt's preceding context. `replay.rows` includes that known continuation and is a replay fixture, not a profile with demonstrated coverage of future output. Rejected MTP drafts, a different sample, another prompt, images, or tool turns can need additional rows. A zero-miss replay does not authorize a timed run with an incomplete profile.

The text profile header is `LLAMA_PLE_ROWS_V1 TYPE WIDTH FULL_ROWS ROW_BYTES COUNT`, followed by exactly COUNT distinct, increasing row IDs. Type and original table geometry must match. Profiles contain indices only, so the current model supplies its own weight bytes at startup.

`MALLOC_ARENA_MAX=2` is a glibc allocator setting applied before process startup. `LLAMA_MALLOC_TRIM=1` additionally calls `malloc_trim(0)` after the loader and its temporary metadata are destroyed. Both are opt-in. The trim return code says whether glibc released pages, not how many bytes it saved. Use smaps and MemAvailable measurements for that claim.

Run `test-ple-cache` on CPU and `test-ple-cache --cuda` under the shared GPU lock. The tests protect the original mapped source before graph execution, compare IQ4_NL gathers byte for byte, check fail-closed misses without partially rewriting inputs, and reject malformed profiles.
