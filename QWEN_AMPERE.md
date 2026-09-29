# llamAmpere

llamAmpere runs Qwen3.8-27B on one 24 GB Ampere card (RTX 3090 / 3090 Ti), using the model's own MTP head to draft
tokens. It is a fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (llama.cpp
with the TurboQuant KV cache types and native MTP speculative decoding), plus kernel and memory work tuned for SM86
and the quantization recipe behind the ATX models.

This file describes v0.4 (2026-09-28). The full list of changes is in the
[v0.4 release notes](docs/llamampere-v0.4/RELEASE_NOTES.md).

## Which model

| model | size | notes |
|---|---|---|
| [`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF`](https://huggingface.co/jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF) | 14.52 GiB | Recommended. The command below downloads it. |
| [`ukisai/Swift-Qwen3.8-27B-GGUF`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF), file `Swift-Qwen3.8-27B-IQ4_XS.gguf` | 14.61 GiB | The Swift authors' own IQ4_XS. Fits the card the same way; not benchmarked head to head. |
| [`jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF`](https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF) | 14.5 GiB | The original ATX quant, used for all v0.3 numbers. |

The ATX quants keep the bulk tensors at IQ4_XS, the fastest format on SM86 at speculative verification widths, and
upgrade the structural tensors (attention K/V and output, `ssm_out`, `output.weight`, the GDN alpha/beta vectors) to
higher-precision types, 4.56 bpw overall. No tensor goes below IQ4_XS. The per-tensor map and the recipe are in the
model repos.

Other formats run too: EXL3 ([docs/exl3.md](docs/exl3.md)), Ternary Bonsai 2 27B ([docs/bonsai2.md](docs/bonsai2.md))
and Agnes 3.0 Flash ([docs/agnes-3.0-flash.md](docs/agnes-3.0-flash.md)).

## Build and run

Needs Linux, an NVIDIA card with 24 GB (RTX 3090 / 3090 Ti), the CUDA toolkit (tested with 12.4), CMake, git and a C++
compiler. Paste into a terminal; the model download is 15.6 GB.

```bash
git clone -b v0.4 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j8 --target llama-server
./build-sm86/bin/llama-server \
  --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 262144 \
  -ngl 99 -fa on -ctk turbo5 -ctv turbo4 -b 4096 -ub 1024 --parallel 1
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). The MTP drafter (draft depth 4), the vocabulary
shortlist and the drafter's cache types are on by default, so no drafter flags are needed. Tested exactly as written from
a fresh clone of v0.4: 262,144-token context, 67.8 tok/s after a 250,000-token prompt (5,120 generated), peak
22,346 MiB on an RTX 3090 Ti.

## Options worth knowing

- `--spec-type none` turns the MTP drafter off. For a GGUF without the MTP head, the n-gram drafter
  (`--spec-type ngram-cache --spec-ngram-cache-n-max 3 --lookup-cache-dynamic FILE --lookup-cache-dynamic-save`)
  measured +16.40% ± 1.57% over no drafter on an IQ3_S quant.
- `--spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-n-min-adaptive 3` gives adaptive draft depth
  3-4. The fixed depth 4 default measured +1.50% ± 1.16% over it.
- Long conversations: `--cache-prompt` keeps the KV cache between turns, so a new turn only pays for its new tokens.
  48 of the model's 64 layers are recurrent and cannot be rewound, so `--ctx-checkpoints N --checkpoint-min-step T`
  keeps state snapshots for editing or regenerating a turn. `--cache-ram MiB` parks a whole conversation when another
  takes the slot, and `--cache-disk-path DIR --cache-disk-limit MiB` adds a disk tier under it that survives a
  restart. All of these live in host RAM or on disk, not VRAM. With `--cache-ram 8192 --ctx-checkpoints 24
  --checkpoint-min-step 10240`, v0.3 needed about 16 GB of host RAM at the deep end.

## Numbers

RTX 3090 Ti at 350 W, temperature 1.0, whole-card VRAM at or under 23 GB.

| | tok/s |
|---|---|
| decode on the four fixtures behind v0.3.1's published 99.4 | 104.28 (+4.9%) |
| decode at 100K KV depth | 103.09 (v0.3.1 published 93.16, +10.7%) |
| coding / agentic / RAG task histories, ATX-Swift, 3 seeds | 124.19 / 119.62 / 115.42 |
| v0.3 vs upstream TurboQuant at 100K KV depth (ATX IQ4_XS-M) | 93.16 vs 54.96 (1.70x) |

The first two rows were measured on the release tree when the default was adaptive depth 3-4, one run per cell. The
width 5-8 verify kernels alone are +6.70% ± 0.42% over v0.3.1 running the same depth-4 flags, and the draft
vocabulary shortlist is +8.45% ± 0.94% over no shortlist.

## KV cache

- `-ctk turbo5 -ctv turbo4` (the command above): 9.25 bits per K+V element, KL divergence 0.00155-0.00241 nats
  on generated tokens at 10K-77K depth.
- `-ctk q8_0 -ctv turbo3` (the v0.3 default): 11.625 bits, 0.00362-0.00455 nats. No speed difference from turbo5/turbo4
  was measured.
- On GPQA and LiveCodeBench neither differs from a q8_0/q8_0 cache at 95% confidence.

The drafter's KV cache follows `-ctk`/`-ctv`. Details and the other cache types are in
[docs/KV-cache-quantization.md](docs/KV-cache-quantization.md).

## Other GPUs

Only SM86 is tuned and tested. Blackwell (sm_120) needs CUDA 12.8 or newer; a v0.3 audit found no code path that
excludes it, but run `test-backend-ops` before trusting any timing. The v0.3 porting notes are in the archive below.

## Older releases

Release notes for every version: https://github.com/JakeATX/llamAmpere/releases. The v0.3 and v0.2 text that used to
be in this file, including the v0.3 command, KV cache ladder, benchmarks and Blackwell notes, is in
[docs/llamampere-v0.3/RELEASE_NOTES_v0.3_v0.2.md](docs/llamampere-v0.3/RELEASE_NOTES_v0.3_v0.2.md). Write-ups with
figures: [v0.3](docs/llamampere-v0.3/ARTICLE.md), [v0.2](docs/llamampere-v0.2/ARTICLE.md),
[v0.3.1](docs/llamampere-v0.3.1/RELEASE_NOTES.md). v0.1 was
[llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere).

## Credits

Qwen team for Qwen3.8; TheTom for TurboQuant+; Unsloth for the BF16 GGUF, the importance matrix, and the tier ladder
the recipe follows; ggml-org for llama.cpp.
