<div style="display: flex;">
  <div style="flex: 1;"></div>
  <div style="flex: 1;"></div>
</div>

<table><tr>
    <td width="167" valign="top"><img width="167" height="250" alt="llamaAmpre_mascot_250h" src="https://github.com/user-attachments/assets/9c6dfd5f-6575-40f4-9d97-868c429631fc" /></td>
    <td valign="top">
<h1>llamAmpere</h1>
llamAmpere is a fast version of llama.cpp (the open-source program that runs AI language models on your own computer),
tuned for NVIDIA RTX 30-series "Ampere" graphics cards, especially the RTX 3090 and 3090 Ti. It runs Qwen3.8 models,
including Qwen3.8-27B with its built-in MTP draft head (a small helper that guesses several words ahead so the model
writes faster) and Qwen3.8 Flash-Next.
    </td>
  </tr>
</table>

## What's new in v0.6

- **mb-MTP is on by default** with the MTP draft head: more guessed words are kept, so writing is faster (`--no-spec-mb-mtp` gives exact sampling).
- **Faster prompt reading on Swift 1.5 27B** (+5.3% at 100K tokens) with the same output.
- **AI+ model file for Swift 1.5 27B**, our own weight format for Ampere and newer cards: 14.4 GB, 10.4% lower KL (a measure of drift from the full-precision model) than IQ4_XS-M. Older AI+ files no longer load.
- **Qwen3.8 Flash-Next** support: experts split between the card and RAM, with `--kv-gpu-tokens N` to keep only part of the context memory on the card.

Details, speed tables and everything older: [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).

## Quick start

**Build and run (v0.6, one RTX 3090 / 3090 Ti, 262,144-token context).** Needs Linux, an NVIDIA card with 24 GB, the
CUDA toolkit (tested with 12.4), CMake, git and a C++ compiler. The model download is 14.4 GB.

```bash
git clone -b v0.6 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
./build-sm86/bin/llama-server \
  --hf-repo jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4 \
  --hf-file qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 262144 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). Add `--no-spec-mb-mtp` for exact drafting.
To use the IQ4_XS-M file (15.6 GB, also runs in mainline llama.cpp) instead, use
`--hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf`.

## Headline results

Numbers for this release (Swift 1.5 27B, v0.6 against v0.5) are in the [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).

## Links

- Release notes: [v0.6](docs/llamampere-v0.6/RELEASE_NOTES.md), [v0.5](docs/llamampere-v0.5/RELEASE_NOTES.md), [v0.4](docs/llamampere-v0.4/RELEASE_NOTES.md), [v0.3.1](docs/llamampere-v0.3.1/RELEASE_NOTES.md); recommended settings for each card size: [v0.5 RECOMMENDED.md](docs/llamampere-v0.5/RECOMMENDED.md)
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
