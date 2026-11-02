# llamAmpere

llamAmpere is a fast version of llama.cpp (the open-source program that runs AI language models on your own computer),
tuned for NVIDIA RTX 30-series "Ampere" graphics cards, especially the RTX 3090 and 3090 Ti. It runs Qwen3.8 models,
including Qwen3.8-27B with its built-in MTP draft head (a small helper that guesses several words ahead so the model
writes faster) and Qwen3.8 Flash-Next.

## What's new in v0.6.2

- **Support for the new Swift 1.5 27B at 2.3 bits per weight** ([ATX AI2](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2), 9.14 GB). It uses a new 2.5-bit tensor type (id 118, `aif25_u`) that **only v0.6.2 or newer can load**; v0.6.1.1 and earlier, and stock llama.cpp, refuse the file. On our 16,200-position test it drifts less from the full-precision model than Q2_K and every 2-bit IQ quant we built (KL 0.0594), and it scored 84.3% on LiveCodeBench v6 (100 problems, 3 runs, strict grader). That is within noise of our previous best 2.3-bit file (82.7%); the gain over our first one (75.3%) is real. Numbers, method and what was not done are on the [model card](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2).
- **Speculative decoding keeps each slot's verification state** when a draft is only partly accepted and the slot rolls back, and the adaptive depth controller now sees the draft width actually tried.
- **Flash-Next launcher defaults** (`scripts/flash-next-server.sh`) with `--moe-skip`, which changes answers; `--exact` turns it off.

Full list, flags and checks: [v0.6.2 release notes](https://github.com/JakeATX/llamAmpere/releases/tag/v0.6.2) and [docs](docs/llamampere-v0.6.2/RELEASE_NOTES.md).

## What's new in v0.6.1

- **Faster Flash-Next:** about 3% faster writing and prompt reading than v0.6 on an RTX 3090 Ti (113.15 vs 109.65 writing tokens/s, 2,751 vs 2,672 prompt tokens/s, 100K prompt; one-seed observation), and room for more experts on the card. Fixes the 12 GB startup crash. [v0.6.1 release notes](docs/llamampere-v0.6.1/RELEASE_NOTES.md).

## What's new in v0.6

- **mb-MTP is on by default** with the MTP draft head: more guessed words are kept, so writing is faster (`--no-spec-mb-mtp` gives exact sampling).
- **Faster prompt reading on Swift 1.5 27B** (+5.3% at 100K tokens) with the same output.
- **AI+ model file for Swift 1.5 27B**, our own weight format for Ampere and newer cards: 14.4 GB, 10.4% lower KL (a measure of drift from the full-precision model) than IQ4_XS-M. Older AI+ files no longer load.
- **Qwen3.8 Flash-Next** support: experts split between the card and RAM, with `--kv-gpu-tokens N` to keep only part of the context memory on the card.

Details, speed tables and everything older: [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).


## Quick start

**Build and run (v0.6.2, Swift 1.5 27B at 2.3 bpw, 200K context, one 24 GB RTX 30-series card, Linux).** Needs the CUDA 12 or newer
toolkit (`nvcc`), CMake, git and a C++ compiler. The model (about 9.1 GB) downloads on first start. This one paste clones or
updates llamAmpere at v0.6.2, builds `llama-server` and starts it:

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.6.2
else
  git clone -b v0.6.2 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
./build-sm86/bin/llama-server -hf jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2 \
  -c 204800 -ngl 99 -fa on -b 4096 -ub 1024 --parallel 1 --jinja --no-context-shift \
  -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 4 \
  --spec-draft-p-min 0 --spec-mb-mtp
```

Commands for 12 and 10 GB cards (not yet run on this model) are in the [v0.6.2 release notes](https://github.com/JakeATX/llamAmpere/releases/tag/v0.6.2).

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). Add `--no-spec-mb-mtp` for exact drafting.
To use the IQ4_XS-M file (15.6 GB, also runs in mainline llama.cpp) instead, use
`--hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf`.

## Headline results

Numbers for this release (Swift 1.5 27B, v0.6 against v0.5) are in the [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).

## Links

- Release notes: [v0.6.2](docs/llamampere-v0.6.2/RELEASE_NOTES.md), [v0.6.1](docs/llamampere-v0.6.1/RELEASE_NOTES.md), [v0.6](docs/llamampere-v0.6/RELEASE_NOTES.md), [v0.5](docs/llamampere-v0.5/RELEASE_NOTES.md), [v0.4](docs/llamampere-v0.4/RELEASE_NOTES.md), [v0.3.1](docs/llamampere-v0.3.1/RELEASE_NOTES.md); recommended settings for each card size: [v0.5 RECOMMENDED.md](docs/llamampere-v0.5/RECOMMENDED.md)
- Models: [huggingface.co/jakeatx](https://huggingface.co/jakeatx)
- Background: [QWEN_AMPERE.md](QWEN_AMPERE.md), [write-up](docs/llamampere-v0.3/ARTICLE.md), flags in [docs/speculative.md](docs/speculative.md) and [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md)
- SJ-KVaRN paper: [paper](https://huggingface.co/spaces/jakeatx/sj-kvarn-paper), codec [JakeATX/sj-kvarn-codec](https://github.com/JakeATX/sj-kvarn-codec) (MIT)
- Upstream llama.cpp README (quick start, backends, docs index): [docs/llama.cpp-upstream-README.md](docs/llama.cpp-upstream-README.md)

## License and credits

llamAmpere is MIT licensed (see [LICENSE](LICENSE)). It is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
by the ggml authors and includes work from TheTom's TurboQuant fork. It is the successor of
[llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

Third-party libraries (from upstream llama.cpp):


- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
