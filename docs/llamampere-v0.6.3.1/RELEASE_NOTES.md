# llamAmpere v0.6.3.1

Flash-Next fixes and optional prompt memory planning. Use the same model and Q2S-B drafter as v0.6.3. Linux installation builds from source; this release includes no prebuilt Linux binaries.

## Changes

- On 24 GB cards, prompt graph sizing leaves a larger cushion for temporary CUDA allocations. The short 4K-token warm-up and 100K-token prompt checks stayed within 23,552 MiB of whole-card memory.
- Cards under 24 GB keep 32K-token prompt passes and the prior graph cushion. Their launcher microbatch is 4096 instead of 6144: in the measured 12 GB setup, 6144 could run out of layer scratch and fall back to much slower 512-token chunks. Explicit user microbatch settings still win. This does not guarantee that every prompt pass fits the available RAM.
- The Flash-Next launcher supports startup with `--parallel` greater than one, using per-slot GPU KV storage and the main stream for shared experts. No parallel throughput improvement is claimed.
- Unsupported Flash-Next `--split-mode tensor` now returns a clear error instead of asserting during startup. Other MTP plus tensor-split failures are not claimed fixed.
- Small-card prompt messages explain the pass limit instead of suggesting a larger prompt batch.

## Optional adaptive prompt passes

`LLAMA_B1_ADAPTIVE_PASSES=1` chooses prompt pass sizes within the RAM budget and can retry a smaller microbatch when the layer graph does not fit. It stays off by default. The 5 GiB RAM floor remains unchanged. This release does not claim a completed 12 GB speed or correctness gate for this optional mode.

## Output and performance limits

Greedy reruns can choose different words when the leading candidates are close. The normal-launch 24 GB check found both selected words in both runs' top two at each first difference, meeting the release check. It does not establish the cause of rerun differences.

The same-card four-pair graph-cushion check measured 122.79 versus 123.59 writing tokens/s for the old and new cushions. The paired ratio was 1.0063, with a 95% interval of 0.9703 to 1.0437; this small test does not prove a speed gain or exclude a loss greater than 1%.

Fresh and reloaded conversations can also take different prompt paths, as documented in the v0.6.3 notes. This update does not promise word-for-word equality between those paths.
