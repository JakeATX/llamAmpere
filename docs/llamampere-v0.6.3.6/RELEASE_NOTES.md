# llamAmpere v0.6.3.6

Flash-Next can now run AI4, a four-bit AI+ model file (qwen3.8-flash-next-Swift1.5-ATX-AI4). This release adds native card kernels and expert-cache support for the four-bit row-scale layout (type 124). AI4 is 84.97 GiB, 2.27 GiB smaller than UD-IQ4_XS, and closer to the Swift-Q8 reference (KL divergence 0.068353 vs 0.080886 over 2,286 positions). On one RTX 3090, single-seed screens at a 100K prompt wrote 15-22% faster than Strata v0.2.12 with UD-IQ4_XS, with 5-16% slower prompt reading. On the 89 GiB test machine the server peaked at 58.6-75.3 GiB of RAM. AI4 is not recommended on 64 GB RAM machines unless you pin a large share of experts on the GPU and move backup copies out of VRAM to cut RAM use; otherwise keep using AI3.

AI4 model file: https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI4
All Flash-Next files: https://huggingface.co/collections/jakeatx/qwen38-flash-next-6ae148b9e743c8905b00c70d

AI3 users can update now; AI3 behaves as before.

Fixes:
- Linux build failure reported in #98 (moe-iq-rows.h: "the value of 'ksigns_iq2xs' is not usable in a constant expression" with Clang and some other compilers). The sign table is now computed from a formula, checked at compile time and by a new CPU test. Thanks to francoispda for the report and the fix idea.
- Saved prompt caches are now keyed by expert-routing settings and all GGML_ environment settings, so a cache saved under one setup is not reused under another.
- The startup probe now works when the batch size is set to 1.

Validation:
- Passed: Linux CPU and CUDA suites, the README-flags AVX2 check, MinGW/Wine (including the new sign-table test), and the tensor-split checks.
- Compilers: g++ 11 and 13 and clang 22 all build this release.
- Dense 27B: the same greedy tokens and top-10 probabilities as v0.6.3.5.
- Card smokes on an RTX 3090 at 350 W with the shipped launcher: AI4 and AI3 both load and answer on the full 24 GB card and when limited to 12 GB. Peak card use was 22,691-23,515 MiB, under the 23,552 MiB limit. At 24 GB this included reusing a saved prompt after a fresh restart.
- Still running, added here when done: the batch-size-1 smoke and AI3 speed against v0.6.3.5.

Source-only release.
