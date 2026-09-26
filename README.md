> **llamAmpere** (v0.4): this fork runs Qwen3.8-27B on one RTX 3090 / 3090 Ti with the model's own MTP head. New kernels make verifying 5 to 8 tokens per step cheaper, so the MTP drafter now proposes 4 tokens per step, and an adaptive depth drops to 3 when drafts miss: +8.98% tokens/s over the v0.3.1 build running the same depth-4 flags. The drafter is now on by default for Qwen3.8 GGUFs that carry the MTP head, and its KV cache follows `-ctk`/`-ctv`. v0.4 also adds a 5-bit key cache type (`turbo5`) with fused attention for turbo4 values, an n-gram drafter for cards where the MTP head does not fit, faster prefill for ternary PTQ1_0 models, and a catch-up to llama.cpp master `a25c9865f`. Against the numbers v0.3.1 published, the release tree decodes 104.28 tok/s on the same fixtures (99.4, +4.9%) and 103.09 tok/s at 100K KV depth (93.16, +10.7%), and runs a 262,144-token context under the 23 GB cap. Unless noted, v0.4 numbers are from an RTX 3090 Ti at 350 W on the coding / agentic / rag ship corpus (real task histories): temperature 1.0, reasoning effort medium, 3 seeds, 10K-27K generated tokens per answer, whole-card VRAM at or under 23 GB. G is the weighted tokens/s gain 0.4 coding + 0.4 agentic + 0.2 rag, and ± is two standard errors over the seeds.

**From v0.3.1, all still in this tree** (numbers measured on v0.3.1; [release notes](docs/llamampere-v0.3.1/RELEASE_NOTES.md)): up to 245K context; 99 tok/s on clean agentic and coding fixtures at temperature 1 (1.46x stock llama.cpp, 1.28x v0.2), 93 tok/s at 100K KV depth, and 1.10x tuned vLLM single-stream at 32K. **EXL3** (Turboderp's exllamav3 trellis format) as GGUF-native types with an SM86 decode kernel: 82 tok/s at a 20K prompt and 74 at 50K with the MTP head on Qwen3.8-27B at 4.0 bpw, and the same 81-82 tok/s at 3.5 bpw and 80.7 at 3.0 bpw from files of 12.3 and 10.9 GiB ([docs/exl3.md](docs/exl3.md)). **Ternary Bonsai 2 27B** from Prism ML at 1.75 and 2.125 bits per weight, 105 tok/s with the MTP drafter at 16K on the 2.125-bit container ([docs/bonsai2.md](docs/bonsai2.md)). A shared-memory codebook for IQ3 decode (opt-in in v0.3.1, the SM86 default in v0.4). A full upstream catch-up (llama.cpp master `b49650adb` and TurboQuant `407f3237b`, 772 commits ahead of the v0.3 base). **Agnes 3.0 Flash** loads with its MTP head, which upstream llama.cpp does not ([docs/agnes-3.0-flash.md](docs/agnes-3.0-flash.md)).

**Fastest configuration (v0.4, one RTX 3090 / 3090 Ti).** Build as in [QWEN_AMPERE.md](QWEN_AMPERE.md#build-and-run), with the GGUF from [jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF](https://huggingface.co/jakeatx/ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF), the model the v0.4 numbers cite (same ATX-IQ4_XS-M recipe and MTP head as [jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF](https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF); G +1.14% ± 0.56% over it on the same build), then:

```bash
./build-sm86/bin/llama-server -m ATX-Swift-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf \
  -c 49152 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk turbo5 -ctv turbo4 --parallel 1
```

The drafter needs no flags. For a Qwen3.8 GGUF with the MTP head, `llama-server` and `llama-cli` apply `--spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-n-min-adaptive 3 --spec-draft-p-min 0 --spec-draft-vocab-map auto` (`auto` is the built-in 65,536-token list, the same shortlist as `docs/mtp-vocab/atx_65536.txt`), and the drafter's KV cache takes the trunk's `-ctk`/`-ctv` (here turbo5/turbo4). `--spec-type none` turns the drafter off; explicit `--spec-draft-*` flags override the default values. Against the same flags spelled out, the default gave identical text and acceptance on 3 seeds.

On the release tree (`809014b75`) this configuration decodes 104.28 tok/s on the four fixtures behind v0.3.1's published 99.4 (+4.9%; agentic shards 1-3 at `-c 73728` 107.77 / 104.82 / 103.06 and coding at `-c 106496` 101.48, against 104.52 / 99.29 / 98.99 / 94.90) and 103.09 tok/s at 100K KV depth against the published 93.16 (+10.7%): temperature 1.0, seed 6100, one run per cell, peaks 18,594-19,453 MiB. The gain over the earlier q8_0/q8_0 drafter cache comes from matching it to the trunk: draft acceptance 0.682 to 0.763 on agentic shard 1, 0.735 to 0.798 at 100K. At `-c 262144` with a 250K-token prompt (5,120 generated, one seed) it fits under the 23,552 MiB cap: ATX-Swift 72.4 tok/s and plain ATX 64.4 at a 22,588 MiB peak, EXL3 4.0 bpw 56.0 at 21,674 MiB. Launched with only `-m`, `-fit` picks 61,952 tokens of context with the drafter (112.1 tok/s on a 5,120-token request, 101.7 at 60,652 deep, peak 23,168 MiB) or 108,544 with `--spec-type none` (50.2 tok/s, 37.5 at 100,000 deep, peak 23,162 MiB).

The ship-corpus 3-seed gate has not been run with the inherited drafter cache; with a q8_0/q8_0 drafter cache on the pre-merge build (`1ee8729c7`) ATX-Swift measured coding 118.6, agentic 115.7 and rag 108.8 tok/s (pooled over 3 seeds). The ship-corpus gates in the other bullets ran without a vocabulary map, with `-ctk q8_0 -ctv turbo3` unless the bullet says otherwise. All ran at `-c 49152`. For multi-turn chats add the prompt-cache and checkpoint flags from [QWEN_AMPERE.md](QWEN_AMPERE.md#build-and-run).

What each part buys:

- **Width 5-8 verify kernels** (built in). New matrix-vector kernels for 5-8 tokens per verify pass, plus an Ampere routing table that sends q4_K / q5_K / q6_K to the tile kernel above width 4. Same fixed depth-4 flags on both builds, identical tokens: coding 98.2 to 104.1, agentic 95.0 to 101.4, rag 93.4 to 101.1 tok/s, G +6.70% ± 0.42% over v0.3.1. After a 50K- and a 64K-token prompt: +3.93% and +3.37% (these deep cells used the vocab map and a q8_0 drafter cache on both builds).
- **Adaptive depth 3-4** (`--spec-type draft-mtp-adaptive --spec-draft-n-max 4`, now the default for Qwen3.8 with the MTP head). Starts at depth 3, climbs to 4 after a streak of fully accepted drafts and drops back after misses (22-31% of rounds ran at depth 3). Verification stays exact p/q, so the output distribution is the target model's. Same day, corpus and seeds as the W58 run: coding 105.6, agentic 105.1, rag 102.0 tok/s; G +2.15% ± 1.92% over fixed depth 4 on the v0.4 build and +8.98% ± 2.12% over the v0.3.1 build at fixed depth 4. Three later runs of the same command on the same corpus measured coding 100.9-102.0, agentic 98.7-100.3 and rag 95.8-97.3 tok/s.
- **Fused attention up to width 8** (built in). The fused q8_0-K attention tile now covers 6-8 tokens per verify pass (it stopped at 5). At 100K KV, width 8 takes 429 µs per attention call instead of 1,434 µs (test-backend-ops, q8_0 K / turbo3 V). Widths 1-5 are unchanged, so the depth 3-4 default does not use it; drafts of 5-7 tokens do.
- **KV cache: `-ctk turbo5 -ctv turbo4`** (in the command) for quality and space, or `-ctk q8_0 -ctv turbo3`. The TurboQuant cache types are `turbo2` to `turbo6` by bit width; the `tq` spellings `tq2` to `tq6` (and `tq3_0` to `tq6_0`) are accepted too, and the load log prints the ggml names `turbo2`, `turbo3`, `turbo4`, `tq5_0`, `tq6_0`. No speed difference was measured between them: turbo5/turbo4 G +0.23% ± 1.98% on the ship corpus, and 85.8 vs 85.9 tok/s after a 64K prompt (fixed depth 4 with the vocab map; peak VRAM 19,114 vs 19,506 MiB). turbo5/turbo4 stores 9.25 bits per K+V element against 11.625 and is closer to an f16 cache: KL divergence 0.00155-0.00241 vs 0.00362-0.00455 nats on generated tokens at 10K-77K depth. On GPQA and LiveCodeBench (shallow context, 2,400 answers on rented GPUs) neither differs from a q8_0/q8_0 cache at 95% confidence: +1.2 points [-2.6, +5.2] for turbo5/turbo4, -1.0 [-4.8, +2.8] for q8_0/turbo3. Details in [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md).
- **Draft vocabulary map** (`--spec-draft-vocab-map auto`, the default; for the Qwen3.8-27B tokenizer it picks the built-in 65,536-token list, the same shortlist as `docs/mtp-vocab/atx_65536.txt`). Limits the MTP draft head to a 65,536-token shortlist, so each draft step scores 65,536 rows of the output head instead of 248,320. The target still verifies every draft against its full vocabulary, so the output distribution does not change. On the ship corpus (turbo5/turbo4, adaptive depth 3-4): coding 104.3 to 111.1, agentic 97.9 to 108.5, rag 98.0 to 104.6 tok/s, G +8.45% ± 0.94% over no map. The 32,768 list (`atx_32768.txt`) measured +7.59% ± 1.10%. Draft acceptance moves by 2 points or less (coding 0.76 to 0.74, agentic 0.70 to 0.71), while verify passes per second rise 9-10%. `--spec-draft-vocab-map none` drafts over the full vocabulary, as does a model with no built-in list. The draft silently uses the full head under `--split-mode tensor`, with the chained drafter, or when the head is off the main CUDA device or of an unsupported type ([docs/mtp-vocabulary-shortlist.md](docs/mtp-vocabulary-shortlist.md)).
- **MTP drafter on by default** (Qwen3.8 GGUFs with the MTP head; `llama-server` and `llama-cli`). A per-family table (`common/spec-defaults.cpp`) applies the drafter flags above when `--spec-type` is not given. Any `--spec-type` (including `none`), a draft model, `--eagle3` or `--dflash` turns it off; explicit `--spec-draft-*` values are kept on top of it. The drafter's KV cache takes the trunk's `-ctk`/`-ctv` unless `--spec-draft-type-k/-v` are given (before, it defaulted to f16). Other tools (`llama-bench`, `llama-perplexity`, ...) are unchanged. Bare launch on ATX-Swift: 112.1 tok/s on a 5,120-token request with the drafter, 50.2 without.
- **`-fit` memory accounting**. `-fit` counted the recurrent (GDN) state twice (2,992.5 MiB on Qwen3.8-27B with the drafter), and counted the MTP draft context for `draft-mtp` but not `draft-mtp-adaptive`. It now sizes the state without allocating it, measures the draft context once at 4,096 tokens and scales its KV cache, and counts the draft's compute buffers only when they do not fit inside the main context's (`LLAMA_SHARED_COMPUTE=0` restores the old draft-context accounting). Launched with only `-m`, the context it picks went from 15,616 to 61,952 tokens with the drafter and from 99,072 to 108,544 without; peaks 23,168 / 23,162 MiB, under the 23,552 MiB cap.
- **Without the MTP head: `--spec-type ngram-cache --spec-ngram-cache-n-max 3 --lookup-cache-dynamic FILE --lookup-cache-dynamic-save`**, for cards where the head does not fit. On the 3090 Ti with an IQ3_S GGUF of Qwen3.8-27B, starting with no cache file: G +16.40% ± 1.57% over no drafter (n-max 7: +9.66% ± 1.70%). It relies on v0.4's recurrent-state snapshot ring, which is on by default.
- **Staged IQ3 / IQ2 codebook grids** (SM86 default): the grid is copied to shared memory; IQ3 matmuls at widths 1-4 run 1.8% to 14.6% faster in test-backend-ops.
- **PTQ1_0 prefill tiles** (built in): Ternary-Bonsai-2-27B prefill 846 to 1,148 tok/s at 512 tokens and 815 to 1,126 at 16K (llama-bench, v0.3.1 to v0.4).
- **`--gdn-replay`** (opt-in, off by default) rebuilds rolled-back recurrent state by replay instead of snapshots: 84 MiB less VRAM at one slot, G -4.31% ± 0.39% against the default.

Start with [QWEN_AMPERE.md](QWEN_AMPERE.md) (its numbers are v0.3) and the write-up in [docs/llamampere-v0.3/ARTICLE.md](docs/llamampere-v0.3/ARTICLE.md) (v0.2: [docs/llamampere-v0.2/ARTICLE.md](docs/llamampere-v0.2/ARTICLE.md)); the flags are documented in [docs/speculative.md](docs/speculative.md) and [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md). Successor of [llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere) (v0.1). The rest of this README is upstream llama.cpp's.

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
