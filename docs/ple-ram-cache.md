# Startup PLE row cache

`LLAMA_PLE_PROFILE=/absolute/path/to/profile.rows` opts into a lossless compact RAM cache for a lazily mapped Qwen4Exp per-layer word table. Rows listed in the profile are copied at model startup and locked in RAM. Quantized blocks are unchanged. The original model tensor, its shape, and model-file identity remain unchanged; the gather uses a separate compact tensor and translates row IDs before running.

Every row in each actual graph input is checked. A missing or invalid row logs its count and stops the process before the gather. There is no request-time weight read, zero row, approximation, or disk fallback. The page-prefetch path is disabled for this table. Successful lookup counts are logged at debug level; miss counts are always logged. This protects PLE only; the run must still audit experts and every other model-weight source.

Create profiles from saved tokenizer IDs with:

```sh
python3 scripts/profile-ple-rows.py --model MODEL.gguf --prompt prompt-tokens.json --generated generated-tokens.json --out profiles
```

The tool reads GGUF metadata, not weight payloads. `prompt.rows` contains only prompt rows; the report measures misses on the saved continuation with the prompt's preceding context. `replay.rows` includes that known continuation and is a replay fixture, not a profile with demonstrated coverage of future output. `startup-prompt.rows` and `startup-replay.rows` additionally cover the server's standard BOS/EOS warmup; use one of those for a normal server launch. Warmup rows remain separate from the held-out coverage counts. Rejected MTP drafts, a different sample, another prompt, images, or tool turns can need additional rows. A zero-miss replay does not authorize a timed run with an incomplete profile.

The text profile header is `LLAMA_PLE_ROWS_V1 TYPE WIDTH FULL_ROWS ROW_BYTES COUNT`, followed by exactly COUNT distinct, increasing row IDs. Type and original table geometry must match. Profiles contain indices only, so the current model supplies its own weight bytes at startup.

`MALLOC_ARENA_MAX=2` is a glibc allocator setting applied before process startup. `LLAMA_MALLOC_TRIM=1` additionally calls `malloc_trim(0)` after the loader and its temporary metadata are destroyed. Both are opt-in. The trim log reports the glibc return code and process RSS immediately before and after the call. The RSS decrease can include concurrent process activity; use smaps and MemAvailable measurements before attributing a sustained saving.

Run `test-ple-cache` on CPU and `test-ple-cache --cuda` under the shared GPU lock. The tests protect the original mapped source before graph execution, compare IQ4_NL gathers byte for byte, check fail-closed misses without partially rewriting inputs, and reject malformed profiles.

`LLAMA_PLE_MISS_PROFILE=/absolute/path/to/next-startup.rows` optionally writes a new profile when a lookup fails. It contains the loaded profile plus valid rows from the failed graph input. Invalid indices are excluded. The request still stops before gathering; rows are never loaded during the request. Use the exported profile only at a later startup. It covers the observed failed batch, not later output or rejected helper drafts that have not run. The output file is replaced, so use a dedicated path distinct from the input profile.

The optional diagnostic tracer `scripts/fnarena/trace-activation-metadata.c` records thread-local physical read counters around activation-file allocation. Build it as a shared library with `cc -shared -fPIC -O2 scripts/fnarena/trace-activation-metadata.c -ldl -o trace-activation-metadata.so` and add its absolute path to `LD_PRELOAD` before startup. These reads can come from filesystem allocation metadata; the call does not read model weights or activation payloads. Keep the original process counter and model-weight tracing.

`scripts/fnarena/nonweight_io.py` validates only successful, fully request-contained `posix_fallocate` records for that run's owned, deleted `llama-moe-output-XXXXXX` files. It rejects duplicate intervals, boundary overlaps and attribution exceeding process reads. Any remaining reads still need attribution; this diagnostic is not a blanket exception for positive I/O. A metadata-only stopped interval is not acceptance of the complete request. Run `python3 scripts/fnarena/test-nonweight-io.py` for the CPU policy fixtures.
