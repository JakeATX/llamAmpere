# llamAmpere changelog

The "What's new" sections of earlier releases, moved here from the README as they were written. The newest release is in the [README](../README.md); full notes for each version are on the [releases page](https://github.com/JakeATX/llamAmpere/releases).

## What's new in v0.7.5

- Inkling 2 Mini is faster on 24 GB cards: prompts are read about 40% faster at 100K tokens and decode is 4 to 12%
  faster, with the same files and commands; 8, 10 and 12 GB cards are the same speed or up to 4% faster. Long reasoning
  is no longer cut off at 3/4 of `max_tokens`, `--preset 24ink` finds the published drafter on its own, and images work
  on every preset, including `--preset 24dense`. [Release notes](https://github.com/JakeATX/llamAmpere/releases/tag/v0.7.5).

## What's new in v0.7.2

- Small MoE models stay on the card: Mellum2.1-12B-A2.5B-Thinking IQ3_XS writes 155.1 tokens/s with default flags (v0.7.1: 10.8, RTX 3090, 8,072-token prompt). Flash-Next no longer drops to 0.63 tokens/s when there is too little RAM to pin its experts (89.82 in the same state). [Release notes](llamampere-v0.7.2/RELEASE_NOTES.md).
- Thinking replies keep room for the answer, a second user no longer forces a full prompt re-read (`-np 2`, 28,301-token follow-up: 0.6 s instead of 17.9 s), and earlier requests no longer change later replies. 27B and Flash-Next speed is the same as v0.7.1.

## What's new in v0.7.1

- Faster for one user on 24 GB cards: `--preset 24dense` with one slot writes 115.6 tokens/s at a 98,766-token prompt (v0.7.0.1: 99.8, RTX 3090, P76 AI4 model, +15.8%) and 138.8 at 8,000 (v0.7.0.1: 123.1).
- The 8 GB model: `--preset 8dense` runs [the 8 GB file](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB) with the small `mtp-dm.gguf` MTP drafter and mb-MTP on, at a 204,800-token default context: 41.4 tokens/s at a 99,984-token prompt on an RTX 3070 (v0.7.0.1 with the same settings: 39.6). Prompt reading and writing are faster on the small-card weight types. [Release notes](llamampere-v0.7.1/RELEASE_NOTES.md).
- Claude Code works with Qwen3.8 (before, every request failed with HTTP 500). A Linux build for Ubuntu 22.04 and newer comes with the release; a Windows build (made by CI, not yet run on a GPU by us) follows on the release page.

## What's new in v0.7.0.1

- Fix: the 10 GB headed preset (`--preset 10dense`) failed to start on RTX 3080 cards. [Release notes](llamampere-v0.7.0.1/RELEASE_NOTES.md).

## What's new in v0.7

- Built on mainline llama.cpp v1.0, with card presets for Swift 1.5 Qwen3.8-27B: `--preset 24dense`, `12dense`, `10dense` and `8dense` pick the context length, memory settings and sampling for each card size. The 24 GB, 12 GB and 10 GB models ship with this release; the 8 GB model file comes in v0.7.1. [Release notes](llamampere-v0.7/RELEASE_NOTES.md).

## What's new in v0.6.4

- Fix: Flash-Next ran out of memory or stalled on machines with two or more NVIDIA cards, or on a card that another program was already using (#96). v0.6.4 runs Flash-Next on one card, the one with the most free memory (`--device` picks another); splitting it across two cards is not supported yet. On a busy card with less than 8 GiB free, the launcher uses the memory planner, which runs but is slow. One idle card: unchanged. [Release notes](llamampere-v0.6.4/RELEASE_NOTES.md).

## What's new in v0.6.3.11

- Fix: faster first token on short follow-up messages (up to 256 new tokens) after a long first message, on Flash-Next. Longer follow-ups are unchanged. [Release notes](llamampere-v0.6.3.11/RELEASE_NOTES.md).

## What's new in v0.6.3.10

- Fix: slow first token on short follow-up messages after a long conversation. [Release notes](llamampere-v0.6.3.10/RELEASE_NOTES.md).

## What's new in v0.6.3.9

- Flash-Next on native Windows builds (MSVC + CUDA) can use the fast card-side expert tables, which were Linux-only. The contributor measured 0.26 -> 60 tokens/s on code on an RTX 3090 with 64 GB RAM. Thanks to huhichao (#98). We have no Windows machine: built and tested under MinGW/Wine, a CPU-only build that cannot compile the CUDA and MSVC parts; Windows speed figures are from the contributor. Linux is unchanged. [Release notes](llamampere-v0.6.3.9/RELEASE_NOTES.md).

## What's new in v0.6.3.8

- The DFlash2 drafter on dense Swift 1.5 27B writes about 3.4 times faster (27 -> 93 tokens/s at temperature 1.0 on an RTX 3090 Ti, 100K prompt). A mixing step in the drafter ran twice per layer; it now runs once. The big model still checks every guess, so output quality is unchanged. MTP and Flash-Next are not affected. [Release notes](llamampere-v0.6.3.8/RELEASE_NOTES.md).

## What's new in v0.6.3.6

- Flash-Next runs AI4, a four-bit AI+ model file, [qwen3.8-flash-next-Swift1.5-ATX-AI4](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI4) (84.97 GiB, closer to the Swift-Q8 reference than UD-IQ4_XS). Not recommended on 64 GB RAM machines unless you pin a large share of experts on the GPU and move the backup copies out of VRAM to cut RAM use.
- Fixes the Linux build failure with Clang and some other compilers (#98). Saved prompt caches are keyed by expert-routing and GGML_ settings. [Release notes](llamampere-v0.6.3.6/RELEASE_NOTES.md).

## What's new in v0.6.3.5

- Flash-Next leaves more temporary card memory for long prompts on small cards. Explicit memory settings still win. [Release notes and hardware limits](llamampere-v0.6.3.5/RELEASE_NOTES.md).

## What's new in v0.6.3.3

- Small cards use memory-bounded prompt passes and shorter prompt-tail chunks by default. Normally free 24 GB cards keep their previous plan.
- Fixes Windows private model mappings and MSVC atomics (#94, #93, #82). Linux launchers count physical performance cores on hybrid Intel CPUs.
- Adds CPU expert-split diagnostics and `--moe-cpu-share` for troubleshooting (#88). [Release notes and hardware limits](llamampere-v0.6.3.3/RELEASE_NOTES.md).

## What's new in v0.6.3.2

- Fixes a Flash-Next crash when a follow-up request reuses a cached prompt with shared experts on a side stream (#88).
- Fixes Windows model loading, Windows build errors, and reads from large expert overlays (#82). [Release notes](llamampere-v0.6.3.2/RELEASE_NOTES.md).

## What's new in v0.6.3.1

- Flash-Next leaves more temporary card memory on 24 GB cards and handles `--parallel` startup. Unsupported Flash-Next tensor split now gives a clear error.
- Small cards keep 32K prompt passes, with a 4096-token microbatch to avoid a measured prompt fallback. Adaptive prompt passes remain opt-in. [Release notes](llamampere-v0.6.3.1/RELEASE_NOTES.md).

## What's new in v0.6.3

- **New Flash-Next model file, [qwen3.8-flash-next-Swift1.5-ATX-AI3](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3):** 80.11 GB (74.61 GiB), compared with 75.97 GB for IQ3_XXS. On an RTX 3090 with ample system RAM, this file wrote 121–125 tokens/s across coding, agentic and RAG tasks. These are observations on the new file, not a same-engine comparison with the standard file. [Measured results and limits](llamampere-v0.6.3/RELEASE_NOTES.md).
- The expert kernels our speed tests use are now on by default; whole-file page read-ahead when the model fits in RAM.

## What's new in v0.6.1

- **Faster Flash-Next:** about 3% faster writing and prompt reading than v0.6 on an RTX 3090 Ti (113.15 vs 109.65 writing tokens/s, 2,751 vs 2,672 prompt tokens/s, 100K prompt; one-seed observation), and room for more experts on the card. Fixes the 12 GB startup crash. [v0.6.1 release notes](llamampere-v0.6.1/RELEASE_NOTES.md).

## What's new in v0.6

- **mb-MTP is on by default** with the MTP draft head: more guessed words are kept, so writing is faster (`--no-spec-mb-mtp` gives exact sampling).
- **Faster prompt reading on Swift 1.5 27B** (+5.3% at 100K tokens) with the same output.
- **AI+ model file for Swift 1.5 27B**, our own weight format for Ampere and newer cards: 14.4 GB, 10.4% lower KL (a measure of drift from the full-precision model) than IQ4_XS-M. Older AI+ files no longer load.
- **Qwen3.8 Flash-Next** support: experts split between the card and RAM, with `--kv-gpu-tokens N` to keep only part of the context memory on the card.

Details, speed tables and everything older: [v0.6 release notes](llamampere-v0.6/RELEASE_NOTES.md).

## Older release notes

[v0.7.0.1](llamampere-v0.7.0.1/RELEASE_NOTES.md), [v0.7](llamampere-v0.7/RELEASE_NOTES.md), [v0.6.4](llamampere-v0.6.4/RELEASE_NOTES.md), [v0.6.3.11](llamampere-v0.6.3.11/RELEASE_NOTES.md), [v0.6.3.10](llamampere-v0.6.3.10/RELEASE_NOTES.md), [v0.6.3.9](llamampere-v0.6.3.9/RELEASE_NOTES.md), [v0.6.3.8](llamampere-v0.6.3.8/RELEASE_NOTES.md), [v0.6.3.6](llamampere-v0.6.3.6/RELEASE_NOTES.md), [v0.6.3.5](llamampere-v0.6.3.5/RELEASE_NOTES.md), [v0.6.3.3](llamampere-v0.6.3.3/RELEASE_NOTES.md), [v0.6.3.2](llamampere-v0.6.3.2/RELEASE_NOTES.md), [v0.6.3.1](llamampere-v0.6.3.1/RELEASE_NOTES.md), [v0.6.3](llamampere-v0.6.3/RELEASE_NOTES.md), [v0.6.2](llamampere-v0.6.2/RELEASE_NOTES.md), [v0.6.1](llamampere-v0.6.1/RELEASE_NOTES.md), [v0.6](llamampere-v0.6/RELEASE_NOTES.md), [v0.5](llamampere-v0.5/RELEASE_NOTES.md), [v0.4](llamampere-v0.4/RELEASE_NOTES.md), [v0.3.1](llamampere-v0.3.1/RELEASE_NOTES.md); recommended settings for each card size: [v0.5 RECOMMENDED.md](llamampere-v0.5/RECOMMENDED.md)
