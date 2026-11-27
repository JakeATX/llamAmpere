# llamAmpere

llamAmpere is llama.cpp (the open-source program that runs AI language models on your own PC) tuned for NVIDIA RTX
30-series "Ampere" cards, from the 8 GB RTX 3070 to the 24 GB RTX 3090. It tracks mainline llama.cpp v1.0 and adds
faster GPU code, model files sized for each card, and one `--preset` per card that sets everything up. Linux and
Windows, text and images, OpenAI-compatible server.

Two model families are tuned and measured on every card size: **Inkling 2 Mini** and **Qwen3.8 27B** (Swift 1.5). It
also runs Qwen3.8 Flash-Next and ordinary GGUF files. Three terms used below:

- **Drafter**: a small helper model that guesses the next few tokens. The main model checks the guesses in one pass and keeps the ones it agrees with, so replies come faster.
- **MTP** (multi-token prediction): the drafter built into Qwen3.8 27B. Nothing extra to download on 24 GB.
- **KLD** (KL divergence): how far a compressed model's word choices drift from the full-size model. Lower is closer.

## Which file for my card

Decode = tokens per second written after a 100K-token prompt, chat template, temperature 1. Each number's card, power
limit and prompt are in the notes under the table.

| Card | Model | Preset | File | Decode at 100K | Context (tokens) |
|---|---|---|---|---:|---:|
| 24 GB | Inkling 2 Mini | `24ink` | `Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf` + drafter | 141.8 ¹ | 262,144 (up to 385,024) |
| 24 GB | Qwen3.8 27B | `24dense` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf` | 126.2 ² | 262,144 |
| 12 GB | Inkling 2 Mini | `12ink` | `Inkling-2-Mini-ATX-12GB-4.2bpw.gguf` | 60.0 ³ | 262,144 |
| 12 GB | Qwen3.8 27B | `12dense` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf` | 70.8 ⁴ | 196,608 (262,144 headless) |
| 10 GB | Inkling 2 Mini | `10ink` | `Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf` | 58.0 ³ | 262,144 |
| 10 GB | Qwen3.8 27B | `10dense` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2.gguf` | 60.8 ⁵ | 180,224 (212,992 headless) |
| 8 GB | Inkling 2 Mini | `8ink` | `Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf` | 38.8 ³ | 212,992 |
| 8 GB | Qwen3.8 27B | `8dense` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB.gguf` | 41.4 ⁶ | 204,800 (headless only) |

1. RTX 3090 Ti at 350 W, shown 2% lower as an RTX 3090 figure; 100K-token prompt, v0.7.5, 6 seeds. The smaller `Inkling-2-Mini-ATX-AI4-4.4bpw.gguf` writes 144.9 and reaches 614,400 tokens of context.
2. RTX 3090 Ti at 350 W, as measured; 98,766-token coding prompt already cached, v0.7.5, 6 seeds.
3. One RTX 3090 at 390 W limited to that card's memory, PCIe 4.0 x16, drafter off, 100K-token prompt, v0.7.5, 6 seeds.
4. RTX 3080 Ti 12 GB, no display on the card, 100K-token prompt, v0.7 (power limit not published).
5. RTX 3080 10 GB at 320 W, no display on the card, 98,842-token prompt, v0.7.0.1, 6 seeds.
6. RTX 3070 8 GB at 220 W, PCIe 4.0 x16, 99,984-token prompt, v0.7.1, 6 seeds, on the file before its final tuning pass (same size and layout). A PCIe 3.0 host measured 35.7 (v0.7.2.1).

Model downloads: [Inkling 24 GB](https://huggingface.co/jakeatx/Inkling-2-Mini-ATX-GGUF), [Inkling 8/10/12 GB](https://huggingface.co/jakeatx/Inkling-2-Mini-ATX-SmallCard-GGUF), Qwen3.8 27B for [24 GB](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4), [12 GB](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3), [10 GB](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2) and [8 GB](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB).

On 8, 10 and 12 GB cards part of the work is done from system RAM, so speed depends on your PCIe slot and RAM. The
Qwen3.8 27B small-card presets serve one user.

## Speed against stock llama.cpp

One 24 GB card, one user, v0.7.5 (v0.7.5.1 is the same speed). Decode in tokens per second after a prompt of the given
length; prefill = how fast the prompt is read.

| Engine and file | Size | Decode 3K | Decode 32K | Decode 100K | Prefill 100K | Max context |
|---|---:|---:|---:|---:|---:|---:|
| **llamAmpere**, Inkling 2 Mini 5.4 bpw + drafter | 17.6 GB | **185.2** | **179.0** | **141.8** | **2,462** | 385,024 |
| **llamAmpere**, Inkling 2 Mini 4.4 bpw + drafter | 14.4 GB | **188.7** | **186.9** | **144.9** | **2,692** | 614,400 |
| stock llama.cpp b12627, Inkling 2 Mini `UD-Q4_K_XL` | 17.6 GB | 51.3 | 48.4 | 33.3 | 249 | 278,528 |
| **llamAmpere**, Qwen3.8 27B AI4 + MTP | 14.4 GB | see notes | see notes | **126.2** | see notes | 262,144 |

Setup: chat template, temperature 1, 5,120-token replies, 6 seeds per build. Inkling rows: RTX 3090 Ti at 350 W, shown
2% lower as RTX 3090 figures, 3K / 31,985 / 100K-token prompts, our 4-bit KV cache; stock llama.cpp uses its 8-bit KV
cache and no drafter. Qwen3.8 27B row: RTX 3090 Ti at 350 W as measured, 98,766-token coding prompt. No stock llama.cpp
row is published for the current 27B file.

Inkling 2 Mini is 3.6x stock decode at a 3K prompt and 4.3x at 100K with the same 17.6 GB file size and the same
quality (KLD 0.0032 for both; the 4.4 bpw file scores 0.0060). Method and every other row:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quick start

1. Download the build from [Releases](https://github.com/JakeATX/llamAmpere/releases): the Linux tar (Ubuntu 22.04 or newer) or the Windows zip. Both need an RTX 30-series card and NVIDIA driver R550 or newer; no CUDA toolkit.
2. Download the file for your card from the table above (`hf download <repo> --local-dir models`).
3. Start the server with the preset for your card:

```bash
# Inkling 2 Mini, 24 GB (keep the drafter in the same folder)
./llama-server --preset 24ink -m models/Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf -md models/Inkling-2-Mini-ATX-Drafter-Q8_0.gguf

# Qwen3.8 27B, 24 GB (MTP is inside the file)
./llama-server --preset 24dense -m models/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf

# 8, 10 or 12 GB: the same command with your card's preset and file
./llama-server --preset 12ink -m models/Inkling-2-Mini-ATX-12GB-4.2bpw.gguf
```

The server listens on http://127.0.0.1:8080 with an OpenAI-compatible API and a web UI. The preset sets context,
memory and sampling; any flag you add wins. Do not add `-b` or `-ub`.

- **Images:** add `--mmproj mmproj-Inkling-2-Mini-F16.gguf` (in both Inkling repos) and send `image_url` parts in a
  chat request. Images also work with `--preset 24dense` and a Qwen3.8 27B vision file.
- **Small-card extras:** Qwen3.8 27B on 8, 10 and 12 GB needs `lsa-qwen38-27b-d2048.gguf` beside the model, and 8 GB
  also `mtp-dm.gguf` (both in the 8 GB repo). Inkling on small cards needs free RAM of at least the file size.
- **No display on the card?** Add `headless` (`--preset 12dense headless`, `--preset 8ink-headless`) for more room.
- **More:** every preset's defaults, several users on one card, building from source and Docker are in
  [docs/SETUP.md](docs/SETUP.md). A coding agent can do the setup for you: [docs/AI_SETUP.md](docs/AI_SETUP.md).

## Windows

Every release has a Windows zip (`llama-server.exe` with the CUDA 12.4 runtime inside, RTX 30-series, driver R550 or
newer). Flags, presets and model files are the same as on Linux; run `llama-server.exe` in place of `./llama-server`:
`llama-server.exe --preset 24ink -m C:\models\Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf -md C:\models\Inkling-2-Mini-ATX-Drafter-Q8_0.gguf`

## What's new in v0.7.5.1

- **Fix: Inkling 2 Mini with two or more requests at once** (#140). v0.7.5 could return an HTTP 500 ("non-finite
  logits") or a wrong reply when requests were decoded together. Treat v0.7.5 multi-request Inkling output as suspect.
- Checked with 2, 3 and 4 parallel requests: 0 failed. The same 3-request test failed 3 of 3 on v0.7.5
  (`Inkling-2-Mini-ATX-AI4-4.4bpw.gguf`, 28.5K-token prompts, RTX 3090 Ti at 350 W).
- One request at a time is unchanged: same tokens as v0.7.5 and the same speed (144.81 → 144.89 tokens/s, 5.4 bpw +
  drafter, 100K-token prompt, RTX 3090 Ti at 350 W, 6 seeds per build).
- Staying on v0.7.5? Set `LLAMA_NK_DEV_MASK=0` in the server's environment for correct multi-request output.
- Models, commands, images and Windows are the same as v0.7.5. Qwen3.8 27B was not affected.
  [Release notes](https://github.com/JakeATX/llamAmpere/releases/tag/v0.7.5.1).

## Links

- [Releases](https://github.com/JakeATX/llamAmpere/releases) (downloads and notes), [docs/CHANGELOG.md](docs/CHANGELOG.md) (what was new in every earlier version), [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (method and all measured rows), [docs/SETUP.md](docs/SETUP.md) (presets, source builds, Docker)
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
