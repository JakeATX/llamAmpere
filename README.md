# llamAmpere

llamAmpere is based on mainline llama.cpp (v1.0), tuned for NVIDIA RTX 30-series Ampere cards. With 1.0 we prioritize keeping up with mainline. It includes TheTom TurboQuant work, credited below.

It runs Qwen3.8 models, including Qwen3.8-27B with its built-in MTP draft head (a small helper that guesses several words ahead so the model writes faster) and Qwen3.8 Flash-Next.

## What's new in v0.7.1

- Faster for one user on 24 GB cards: `--preset 24dense` with one slot writes 115.6 tokens/s at a 98,766-token prompt (v0.7.0.1: 99.8, RTX 3090, P76 AI4 model, +15.8%) and 138.8 at 8,000 (v0.7.0.1: 123.1).
- The 8 GB model: `--preset 8dense` runs [the 8 GB file](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB) with the small `mtp-dm.gguf` MTP drafter and mb-MTP on, at a 204,800-token default context: 41.4 tokens/s at a 99,984-token prompt on an RTX 3070 (v0.7.0.1 with the same settings: 39.6). Prompt reading and writing are faster on the small-card weight types. [Release notes](docs/llamampere-v0.7.1/RELEASE_NOTES.md).
- Claude Code works with Qwen3.8 (before, every request failed with HTTP 500). A Linux build for Ubuntu 22.04 and newer and a Windows build (made by CI, not yet run on a GPU by us) come with the release.

## What's new in v0.7.0.1

- Fix: the 10 GB headed preset (`--preset 10dense`) failed to start on RTX 3080 cards. [Release notes](docs/llamampere-v0.7.0.1/RELEASE_NOTES.md).

## What's new in v0.7

- Built on mainline llama.cpp v1.0, with card presets for Swift 1.5 Qwen3.8-27B: `--preset 24dense`, `12dense`, `10dense` and `8dense` pick the context length, memory settings and sampling for each card size. The 24 GB, 12 GB and 10 GB models ship with this release; the 8 GB model file comes in v0.7.1. [Release notes](docs/llamampere-v0.7/RELEASE_NOTES.md).

## What's new in v0.6.4

- Fix: Flash-Next ran out of memory or stalled on machines with two or more NVIDIA cards, or on a card that another program was already using (#96). v0.6.4 runs Flash-Next on one card, the one with the most free memory (`--device` picks another); splitting it across two cards is not supported yet. On a busy card with less than 8 GiB free, the launcher uses the memory planner, which runs but is slow. One idle card: unchanged. [Release notes](docs/llamampere-v0.6.4/RELEASE_NOTES.md).

## What's new in v0.6.3.11

- Fix: faster first token on short follow-up messages (up to 256 new tokens) after a long first message, on Flash-Next. Longer follow-ups are unchanged. [Release notes](docs/llamampere-v0.6.3.11/RELEASE_NOTES.md).

## What's new in v0.6.3.10

- Fix: slow first token on short follow-up messages after a long conversation. [Release notes](docs/llamampere-v0.6.3.10/RELEASE_NOTES.md).

## What's new in v0.6.3.9

- Flash-Next on native Windows builds (MSVC + CUDA) can use the fast card-side expert tables, which were Linux-only. The contributor measured 0.26 -> 60 tokens/s on code on an RTX 3090 with 64 GB RAM. Thanks to huhichao (#98). We have no Windows machine: built and tested under MinGW/Wine, a CPU-only build that cannot compile the CUDA and MSVC parts; Windows speed figures are from the contributor. Linux is unchanged. [Release notes](docs/llamampere-v0.6.3.9/RELEASE_NOTES.md).

## What's new in v0.6.3.8

- The DFlash2 drafter on dense Swift 1.5 27B writes about 3.4 times faster (27 -> 93 tokens/s at temperature 1.0 on an RTX 3090 Ti, 100K prompt). A mixing step in the drafter ran twice per layer; it now runs once. The big model still checks every guess, so output quality is unchanged. MTP and Flash-Next are not affected. [Release notes](docs/llamampere-v0.6.3.8/RELEASE_NOTES.md).

## What's new in v0.6.3.6

- Flash-Next runs AI4, a four-bit AI+ model file, [qwen3.8-flash-next-Swift1.5-ATX-AI4](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI4) (84.97 GiB, closer to the Swift-Q8 reference than UD-IQ4_XS). Not recommended on 64 GB RAM machines unless you pin a large share of experts on the GPU and move the backup copies out of VRAM to cut RAM use.
- Fixes the Linux build failure with Clang and some other compilers (#98). Saved prompt caches are keyed by expert-routing and GGML_ settings. [Release notes](docs/llamampere-v0.6.3.6/RELEASE_NOTES.md).

## What's new in v0.6.3.5

- Flash-Next leaves more temporary card memory for long prompts on small cards. Explicit memory settings still win. [Release notes and hardware limits](docs/llamampere-v0.6.3.5/RELEASE_NOTES.md).

## What's new in v0.6.3.3

- Small cards use memory-bounded prompt passes and shorter prompt-tail chunks by default. Normally free 24 GB cards keep their previous plan.
- Fixes Windows private model mappings and MSVC atomics (#94, #93, #82). Linux launchers count physical performance cores on hybrid Intel CPUs.
- Adds CPU expert-split diagnostics and `--moe-cpu-share` for troubleshooting (#88). [Release notes and hardware limits](docs/llamampere-v0.6.3.3/RELEASE_NOTES.md).

## What's new in v0.6.3.2

- Fixes a Flash-Next crash when a follow-up request reuses a cached prompt with shared experts on a side stream (#88).
- Fixes Windows model loading, Windows build errors, and reads from large expert overlays (#82). [Release notes](docs/llamampere-v0.6.3.2/RELEASE_NOTES.md).

## What's new in v0.6.3.1

- Flash-Next leaves more temporary card memory on 24 GB cards and handles `--parallel` startup. Unsupported Flash-Next tensor split now gives a clear error.
- Small cards keep 32K prompt passes, with a 4096-token microbatch to avoid a measured prompt fallback. Adaptive prompt passes remain opt-in. [Release notes](docs/llamampere-v0.6.3.1/RELEASE_NOTES.md).

## What's new in v0.6.3

- **New Flash-Next model file, [qwen3.8-flash-next-Swift1.5-ATX-AI3](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3):** 80.11 GB (74.61 GiB), compared with 75.97 GB for IQ3_XXS. On an RTX 3090 with ample system RAM, this file wrote 121–125 tokens/s across coding, agentic and RAG tasks. These are observations on the new file, not a same-engine comparison with the standard file. [Measured results and limits](docs/llamampere-v0.6.3/RELEASE_NOTES.md).
- The expert kernels our speed tests use are now on by default; whole-file page read-ahead when the model fits in RAM.

## What's new in v0.6.1

- **Faster Flash-Next:** about 3% faster writing and prompt reading than v0.6 on an RTX 3090 Ti (113.15 vs 109.65 writing tokens/s, 2,751 vs 2,672 prompt tokens/s, 100K prompt; one-seed observation), and room for more experts on the card. Fixes the 12 GB startup crash. [v0.6.1 release notes](docs/llamampere-v0.6.1/RELEASE_NOTES.md).

## What's new in v0.6

- **mb-MTP is on by default** with the MTP draft head: more guessed words are kept, so writing is faster (`--no-spec-mb-mtp` gives exact sampling).
- **Faster prompt reading on Swift 1.5 27B** (+5.3% at 100K tokens) with the same output.
- **AI+ model file for Swift 1.5 27B**, our own weight format for Ampere and newer cards: 14.4 GB, 10.4% lower KL (a measure of drift from the full-precision model) than IQ4_XS-M. Older AI+ files no longer load.
- **Qwen3.8 Flash-Next** support: experts split between the card and RAM, with `--kv-gpu-tokens N` to keep only part of the context memory on the card.

Details, speed tables and everything older: [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).


## Quick start

**Let your coding agent set it up:** paste this into Claude Code, Codex, Cursor or Copilot: *"Set up llamAmpere on this PC for me: https://github.com/JakeATX/llamAmpere - follow docs/AI_SETUP.md in that repository. Ask me before you install system packages or use sudo."* ([what it does](docs/AI_SETUP.md))

### Quick start: pick your card

Card presets assume a display is attached and reserve 1,024 MiB of GPU memory, except 8dense, which is headless-only and reserves 500 MiB. The reserve comes off the card's total memory (a 10 GB card stays at or under 9,216 MiB headed); memory already in use, such as a desktop, counts toward it. An explicit `--fit-target` keeps its free-memory meaning. `8dense headed` prints a note and uses the same headless settings. Append `headless` (`--preset 12dense headless`) or use `12dense-headless` for a 500 MiB margin on 10dense, 12dense, and 24dense. Headed defaults are 180,224 tokens for 10dense and 196,608 for 12dense, with CUDA graphs off for headed 12dense. Headless defaults remain 212,992 for 10dense, 262,144 for 12dense, and 204,800 for 8dense (with `mtp-dm.gguf` the 0.7.1 file peaks at 7,607 MiB on an RTX 3070, under the 7,692 MiB headless line). Headless 12dense at the full 262,144 tokens is tested on an RTX 3080 Ti 12 GB: a 262,142-token fill completed with 11,642 MiB peak (500 MiB margin line 11,788), 1,009 prompt tokens/s and 70.8 writing tokens/s at a 100K prompt. Fitting is on: the dynamic microbatch ladder remains enabled, and fitting may reduce a preset context to honor the margin. An explicit `-c`, `-fitt`, or `-fit` wins in either order. Startup reports the chosen context. Small cards still support one user only; multi-user support is future work.

Card profiles target dense Swift 1.5 Qwen 27B. All four set sampler defaults to temperature 1.0, top-k 20, top-p 0.95, and min-p 0; explicit CLI flags and per-request sampler fields override them. All four profiles keep mb-MTP on with MTP; `--no-spec-mb-mtp` turns it off. Choose the matching model file and use one profile. The 24 GB file (P76 AI4, 14.4 GB) is [qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4).

```bash
llama-server --preset 24dense -m /path/to/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf
llama-server --preset 24dense -m /path/to/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -np 3 -c 352256 --slot-ctx 262144,0,0
llama-server --preset 12dense -m /path/to/L0.gguf
llama-server --preset 12dense headless -m /path/to/L0.gguf   # no display on the card: full 262,144 context
llama-server --preset 10dense -m /path/to/public-AI2.gguf
llama-server --preset 8dense -m /path/to/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB.gguf   # mtp-dm.gguf + LSA map in the same folder
```

Bare names work too: `llama-server 24dense -m MODEL`. `llama-cli -p TEXT` still takes a prompt; the server does not accept `-p`. Explicit flags override profile values in either order. Card profiles override config.ini defaults; environment values and CLI flags override the card profile. For adaptive MTP, lowering the maximum draft depth also lowers the profile's minimum depth unless you set both. Raising the minimum also raises the profile's maximum unless you set both. Changing a K/V cache type or SJ body type drops the profile's related KV/LSA defaults; supply the full cache configuration you want. A smaller batch caps the default microbatch. Profiles do not download models. Place `lsa-qwen38-27b-d2048.gguf` beside the 8/10/12 GB model, or set `LLAMA_LSA_MAP` / `--attn-sparse-map` to its path. 8dense also needs `mtp-dm.gguf` (127 MB MTP drafter) beside the model; startup stops and names the path if it is missing. Set `LLAMA_MODEL_OVERLAY_PATCH=` (empty) with `-c 114688` to use the model file's own MTP layer instead. Shell environment values override profile environment defaults. Startup prints the profile version and expanded defaults; a different model family gets a warning.

24dense and 8dense were re-measured for v0.7.1; 10dense and 12dense keep their v0.7.0.1 measurements; see [the preset checks](PRESETS_RECHECK.md). These are not full-context fit guarantees. The 24 GB profile keeps f32 recurrent state and supports one to four slots with asymmetric context reservations; `0` gives a slot a share of the remaining pool. The 8/10/12 GB profiles require `-np 1`. TODO: multi-user for small cards (future). All four profiles enable dynamic microbatch sizing, with checks at most every 2 seconds of prompt processing and 10 seconds of decode. Set `LLAMA_DEEPCTX_LADDER=0` for static sizing. The microbatch ceilings are 2048 for 24dense with one slot (1024 with `-np 2` to `4`) and 512/512/256 for 12/10/8dense; an explicit `-ub` or `LLAMA_ARG_UBATCH` overrides the ceiling. The 8 GB profile uses the accepted F2 settings: reserve 7 rows, stage strips on, CUDA graphs off, 1% buffer growth, and mean chunks on. This build includes the reserve reader. Copy defaults off; earlier 8 GB measurements used copy on, so final measurements must be repeated with copy off. Packed masks remain opt-in. Eligible window-0 paths select DSJ/RM24H automatically, with `--no-spec-draft-sjkvarn` as the opt-out; 24dense with one slot drafts with window 0, gdn-replay off and adaptive depth 3 to 7, so it selects that path; with `-np 2` to `4` it uses the multi-user values (window 4096, gdn-replay on, depth 4 to 5). Context and memory fit depend on the model and overrides.

**Build and run (v0.6.3.11, Qwen3.8 Flash-Next, Linux).** For one RTX 30-series card with enough system RAM for the model and prompt workspace. Small-card tests used a 3090 limited to 12 GB with about 90 GiB of system RAM; native 12/16 GB cards remain unmeasured, and 64 GB of system RAM is not a fit promise. Small cards use memory-bounded prompt passes; see the [hardware limits](docs/llamampere-v0.6.3.5/RELEASE_NOTES.md). The launcher keeps the tested `--fit off` setting; its own card planner still chooses the memory plan and keeps at least 5 GiB of RAM free. Automatic `--fit on` has not been measured on this file. Needs the CUDA 12 or newer toolkit (`nvcc`), CMake, git, a C++
compiler, the `hf` download tool (`pip install -U huggingface_hub`) and 81.63 GB for the model and drafter, plus space for the build, prompt scratch and saved prompts. The launcher uses a 5 GiB free-RAM floor. This
one paste clones or updates llamAmpere at v0.6.3.11, builds `llama-server`, downloads the model and its drafter and starts the server:

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.6.3.11
else
  git clone -b v0.6.3.11 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
hf download jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3 --include '*.gguf' --local-dir models/flash-next && \
hf download jakeatx/Swift-1.5-Qwen3.8-Flash-Next-MTP-Drafter-GGUF mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf --local-dir models/flash-next && \
LLAMA_SERVER_BIN=build-sm86/bin/llama-server \
FLASH_NEXT_DRAFTER=models/flash-next/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf \
scripts/flash-next-server.sh -m models/flash-next/qwen3.8-flash-next-Swift1.5-ATX-AI3-00001-of-00002.gguf
```

To start it again later, run the last three lines from the `llamAmpere` folder. The server listens on http://127.0.0.1:8080 (OpenAI-compatible API).

**Build and run (v0.7.1, Swift 1.5 27B, one 24 GB RTX 30-series card, Linux).** Needs the CUDA 12 or newer
toolkit (`nvcc`), CMake, git and a C++ compiler. Get the model file for 24 GB cards first:
[qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4) (P76 AI4, 14.4 GB), for example with the `hf` tool (`pip install -U huggingface_hub`):
`hf download jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4 qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf --local-dir models`. Then replace `/path/to/` below with the full path of the folder you saved it in. This one paste clones or updates llamAmpere at v0.7.1, builds
`llama-server` and starts it with the 24 GB preset:

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.7.1
else
  git clone -b v0.7.1 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
./build-sm86/bin/llama-server --preset 24dense -m /path/to/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf
```

The preset sets the context, cache types, MTP drafting and the microbatch ladder; do not add `-b` or `-ub`.
For 12, 10 and 8 GB cards, swap in the matching preset and model file from **Quick start: pick your card**.

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). Add `--no-spec-mb-mtp` for exact drafting.
To use the IQ4_XS-M file (15.6 GB, also runs in mainline llama.cpp; not re-measured with the v0.7.1 preset) instead,
replace `-m /path/to/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf` with
`--hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf`.

## Headline results

Numbers for this release (Swift 1.5 27B, v0.6 against v0.5) are in the [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).

## Links

- Release notes: [v0.7.0.1](docs/llamampere-v0.7.0.1/RELEASE_NOTES.md), [v0.7](docs/llamampere-v0.7/RELEASE_NOTES.md), [v0.6.4](docs/llamampere-v0.6.4/RELEASE_NOTES.md), [v0.6.3.11](docs/llamampere-v0.6.3.11/RELEASE_NOTES.md), [v0.6.3.10](docs/llamampere-v0.6.3.10/RELEASE_NOTES.md), [v0.6.3.9](docs/llamampere-v0.6.3.9/RELEASE_NOTES.md), [v0.6.3.8](docs/llamampere-v0.6.3.8/RELEASE_NOTES.md), [v0.6.3.6](docs/llamampere-v0.6.3.6/RELEASE_NOTES.md), [v0.6.3.5](docs/llamampere-v0.6.3.5/RELEASE_NOTES.md), [v0.6.3.3](docs/llamampere-v0.6.3.3/RELEASE_NOTES.md), [v0.6.3.2](docs/llamampere-v0.6.3.2/RELEASE_NOTES.md), [v0.6.3.1](docs/llamampere-v0.6.3.1/RELEASE_NOTES.md), [v0.6.3](docs/llamampere-v0.6.3/RELEASE_NOTES.md), [v0.6.2](docs/llamampere-v0.6.2/RELEASE_NOTES.md), [v0.6.1](docs/llamampere-v0.6.1/RELEASE_NOTES.md), [v0.6](docs/llamampere-v0.6/RELEASE_NOTES.md), [v0.5](docs/llamampere-v0.5/RELEASE_NOTES.md), [v0.4](docs/llamampere-v0.4/RELEASE_NOTES.md), [v0.3.1](docs/llamampere-v0.3.1/RELEASE_NOTES.md); recommended settings for each card size: [v0.5 RECOMMENDED.md](docs/llamampere-v0.5/RECOMMENDED.md)
- Models: [huggingface.co/jakeatx](https://huggingface.co/jakeatx), [Qwen3.8 Flash-Next collection](https://huggingface.co/collections/jakeatx/qwen38-flash-next-6ae148b9e743c8905b00c70d)
- Background: [QWEN_AMPERE.md](QWEN_AMPERE.md), [write-up](docs/llamampere-v0.3/ARTICLE.md), flags in [docs/speculative.md](docs/speculative.md) and [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md)
- SJ-KVaRN paper: [paper](https://huggingface.co/spaces/jakeatx/sj-kvarn-paper), codec [JakeATX/sj-kvarn-codec](https://github.com/JakeATX/sj-kvarn-codec) (MIT)
- Upstream llama.cpp README (quick start, backends, docs index): [docs/llama.cpp-upstream-README.md](docs/llama.cpp-upstream-README.md)

## License and credits

llamAmpere is MIT licensed (see [LICENSE](LICENSE)) and is based on mainline [llama.cpp](https://github.com/ggml-org/llama.cpp) by the ggml authors. It includes TheTom's TurboQuant KV-cache work and contributions from that fork. Original authors and license headers are retained. It is the successor of [llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere). See [CREDITS.md](CREDITS.md) for contribution links. Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

Our attention selector is derived from Distilled Latent Sparse Attention (D-LSA), introduced by Ales Kubicek and colleagues in [Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention](https://arxiv.org/abs/2610.42147v2). The paper projects queries and keys into a learned index space, groups keys by similarity, searches group summaries, and applies attention to the selected original keys plus a local window and sink tokens. We use the authors' [released Qwen d2048 projection map](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1). Our implementation adapts this approach with fixed blocks of 128 or 16 tokens, finer selection, and host KV offload; these are our adaptations, rather than a reproduction of the paper's semantic clustering or tree search. Attention on selected keys uses the stored KV representation; selection, KV quantization, and compensation still introduce approximation. The map's actual index width is 64; d2048 is its release name.

See the [published evaluation](https://huggingface.co/datasets/jakeatx/long-context-lsa-qsa-evaluation) and [measured quality and limits](docs/KV-cache-quantization.md#lsa-attribution-and-measured-quality).

Third-party libraries (from upstream llama.cpp):


- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
