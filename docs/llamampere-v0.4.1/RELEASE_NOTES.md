# llamAmpere v0.4.1 release notes

## What's new in v0.4.1

v0.4.1 brings llamAmpere even with TheTom's TurboQuant release `tqp-v0.4.0` (`bcb85fc3a`, 2026-09-28). It adds 7
upstream commits on top of v0.4.

**CUDA**

- The host-memory KV stream (`--kv-stream-arena-mib`, off by default) now keeps its KV in mapped host memory that is
  not write-combined, on every OS. Before, only native Windows dropped the write-combined flag. Upstream found that GPU
  writes to write-combined mapped memory can fault under WDDM, and that includes CUDA running through WSL.
- The default path is unchanged: same kernels, same routing, same MTP drafter settings. Identical logits to
  v0.4 at 25,600 and 76,800 tokens of KV (q8_0/q8_0 and turbo5/turbo4), and identical text on the ship corpus and at
  100K depth.

**Vulkan** (from upstream; not tested by us, since llamAmpere is tuned and measured on CUDA / RTX 3090)

- The coopmat2 flash attention path decodes turbo2, turbo3 and turbo4 K and V caches directly.
- turbo3 on Vulkan now uses the same codebook as the C, CUDA, Metal and SYCL code. Before this, Vulkan quantized and
  dequantized turbo3 with an older set of centroids.
- MoE expert-cache uploads take the device's queue lock, so they no longer race the backend's own submissions.
- Flash attention block sizes for Q1_0, Q2_0 and turbo2/3/4 are restored in `fa_types.glsl`. The llama.cpp catch-up in
  v0.4 had dropped them (llamAmpere fix).

**Maintenance**

- CI: the Vulkan workflow skips the MoE-cache integration test on the llvmpipe software driver.
- The Vulkan accessors used by the MoE expert cache (device, queue, physical device, queue family) are back in
  `ggml-vulkan.cpp`. The header already declared them.

## Speed

- Ship corpus (coding / agentic / rag, 3 seeds, temperature 1.0, turbo5/turbo4, fixed MTP depth 4 with the vocab map,
  ATX-Swift 1.5 IQ4_XS-M): G +0.09% ± 1.53% against v0.4. This is no change.
- 100K KV depth (3 seeds, 5,120 generated tokens each): 100.39 tok/s for v0.4.1 vs 100.35 for v0.4, identical text.

## Recommended settings for 24 GB cards (RTX 3090 / 3090 Ti)

These are unchanged from v0.4. Needs Linux, an NVIDIA card with 24 GB (RTX 3090 / 3090 Ti), the CUDA toolkit (tested
with 12.4), CMake, git and a C++ compiler. Paste into a terminal; the model download is 15.6 GB.

```bash
git clone -b v0.4.1 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j8 --target llama-server
./build-sm86/bin/llama-server \
  --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 262144 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 --parallel 1
```

- The server listens on http://127.0.0.1:8080 (OpenAI-compatible API).
- The MTP drafter (draft depth 4), the vocabulary shortlist and the drafter's cache types are on by default, so no
  drafter flags are needed.
- Tested exactly as written from a fresh clone of the release branch: 116.3 tok/s on a 5,120-token request, peak
  21,858 MiB on an RTX 3090 Ti.

Everything else in the [v0.4 release notes](../llamampere-v0.4/RELEASE_NOTES.md) still applies.
