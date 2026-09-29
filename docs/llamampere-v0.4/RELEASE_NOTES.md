# llamAmpere v0.4 release notes

## What's new in v0.4

**Speed**

- Verify kernels for 5 to 8 tokens per pass, plus an Ampere routing table that picks the best kernel per weight type
  and width.
- MTP draft depth 4 by default. Adaptive depth 3-4 is available as an option.
- The MTP drafter turns on by itself for Qwen3.8 GGUFs that carry the MTP head, so no drafter flags are needed.
- The drafter's KV cache now uses the same cache types as the main model.
- A built-in draft vocabulary shortlist (`--spec-draft-vocab-map auto`), now on EXL3 too.
- The fused attention tile for q8_0 keys now covers 6 to 8 tokens per pass.
- GDN recurrent-state copies now run as one copy kernel instead of memcpy nodes in the CUDA graph.
- The GDN kernel reads the recurrent state in place instead of gathering a copy every round.
- Fewer copies per decode round: an activation cache for the matrix-vector kernels, a strided copy folded into the
  next kernel, a skipped identity gather and logits pad, and a better launch shape for short concatenations.
- The IQ3 and IQ2 codebook grids are staged in shared memory by default on SM86.
- EXL3 weight-major tensor-core kernels and a fused FFN bridge, both on by default.
- New prefill tiles and a verify-width kernel for PTQ1_0 ternary weights (Ternary Bonsai 2).

**KV cache**

- A new 5-bit key cache type, `turbo5` (tq5_0), with a sign-magnitude encoding.
- Fused attention for turbo4 values paired with turbo5, turbo6 or q8_0 keys, with staged key and value tiles.
- One flat loader for turbo4 keys and values.
- One naming family by bit width, `turbo2` to `turbo6`, with the `tq` spellings accepted in every cache-type flag.
- turbo2 is enforced as values-only.
- Unsupported cache types on Metal, Vulkan and SYCL now fail with a clear error message.

**Long context and memory**

- A 262,144-token context on one 24 GB card, with IQ4_XS and with EXL3 4.0 bpw.
- The prefill attention workspace is capped at 256 MiB by default, and prefill runs in groups.
- The attention scratch buffer is sized for the kernel that actually runs.
- `-fit` accounting fixes: the recurrent state is no longer counted twice, and the adaptive drafter is now budgeted. A
  launch with only `-m` now gets a much larger context.
- An n-gram drafter for cards where the MTP head does not fit, built on a new recurrent-state snapshot ring
  (`--spec-n-rs-seq`).

**Fixes and maintenance**

- tq4_1s weights: fixed NaN from views of weights converted at load time, and the copy abort.
- EXL3: output widths that are not a multiple of 128 no longer abort, and the half-precision accumulation bound is
  corrected.
- HIP: fixed tq5_0 sign packing on 64-wide AMD waves.
- A stress check of the server's cache paths: prompt reuse, branching, restore, cancel, regenerate and slot recycling.
- A catch-up to upstream llama.cpp.

## Recommended settings for 24 GB cards (RTX 3090 / 3090 Ti)

Needs Linux, an NVIDIA card with 24 GB (RTX 3090 / 3090 Ti), the CUDA toolkit (tested with 12.4), CMake, git and a C++
compiler. Paste into a terminal; the model download is 15.6 GB.

```bash
git clone -b v0.4 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j8 --target llama-server
curl -L -o ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf \
  https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF/resolve/main/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf
./build-sm86/bin/llama-server -m ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 262144 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). The MTP drafter (draft depth 4), the vocabulary
shortlist and the drafter's cache types are on by default, so no drafter flags are needed. Tested exactly as written from
a fresh clone of v0.4: 262,144-token context, 67.8 tok/s after a 250,000-token prompt (5,120 generated), peak
22,346 MiB on an RTX 3090 Ti.
