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

Build v0.4:

```bash
git clone -b v0.4 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j8 --target llama-server
```

The MTP drafter (fixed depth 4), the vocabulary shortlist and the drafter's cache types are all defaults, so none of
the commands below pass drafter flags. The model is
[jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF](https://huggingface.co/jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF)
unless the row says EXL3.

| use | model | `-c` | measured decode tok/s | peak whole-card VRAM |
|---|---|---|---|---|
| Everyday coding, agent and RAG work | ATX-Swift IQ4_XS-M | 49152 | 124.2 coding, 119.6 agentic, 115.4 RAG | 17,765-17,790 MiB |
| 100K-token prompts | ATX-Swift IQ4_XS-M | 110592 | 98.2 after a 100K prompt | 19,154 MiB |
| Long multi-turn sessions | ATX-Swift IQ4_XS-M | 208896 | 102.5 at 100K, 86.0 at 206,851 tokens | 21,256 MiB |
| Largest context | ATX-Swift IQ4_XS-M | 262144 | 72.4 after a 250K prompt | 22,588 MiB |
| Largest context, EXL3 | Qwen3.8-27B EXL3 4.0 bpw | 262144 | 61.9 after a 250K prompt | 21,632 MiB |
| No tuning | ATX-Swift IQ4_XS-M | only `-m`; `-fit` picks 61,952 | 112.1 on a 5K request | 23,168 MiB |

Everyday coding, agent and RAG work:

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 49152 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1
```

100K-token prompts:

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 110592 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1
```

Long multi-turn sessions:

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 208896 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1 \
  --cache-ram 0 --ctx-checkpoints 4
```

Largest context:

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 262144 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1
```

Largest context, EXL3 4.0 bpw:

```bash
./build-sm86/bin/llama-server -m Qwen3.8-27B-EXL3-4.0bpw.gguf -c 262144 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 -t 8 -tb 8 --parallel 1
```

No tuning (`-fit` sizes the context to the card):

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf
```

- **Where the numbers come from.** RTX 3090 Ti at 350 W, temperature 1.0, 5,120 or more generated tokens. The peaks are
  whole-card, so they include about 1.1 GB that other processes already held on the card. The 49,152 row is the
  release build at its default fixed depth 4. The other rows ran on v0.4 builds from before that default, when it was
  adaptive depth 3-4, with the same cache types and flags as the commands above.
- **RTX 3090.** It has the same 24 GB, so the same contexts fit. We measured speed on the 3090 Ti only.
- **Cache types.** `-ctk turbo5 -ctv turbo4` is the recommended pair. On 2,400 GPQA and LiveCodeBench answers, its task
  accuracy was not distinguishable from `q8_0/q8_0`. Every command above except the no-tuning one uses it.
- **Keep `--parallel 1`.** Every number here is one slot.
