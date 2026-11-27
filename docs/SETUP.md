# Setup details: card presets, building from source, Docker

Moved here from the README as written. For the short version, see the [README quick start](../README.md#quick-start).

## Quick start

**Let your coding agent set it up:** paste this into Claude Code, Codex, Cursor or Copilot: *"Set up llamAmpere on this PC for me: https://github.com/JakeATX/llamAmpere - follow docs/AI_SETUP.md in that repository. Ask me before you install system packages or use sudo."* ([what it does](AI_SETUP.md))

### Quick start: Inkling 2 Mini

Download the release for your system from [Releases](https://github.com/JakeATX/llamAmpere/releases) (Linux tar or
Windows zip, RTX 30-series, NVIDIA driver R550 or newer) and the model files from Hugging Face.

**24 GB cards** ([jakeatx/Inkling-2-Mini-ATX-GGUF](https://huggingface.co/jakeatx/Inkling-2-Mini-ATX-GGUF)): put the
model and the drafter `Inkling-2-Mini-ATX-Drafter-Q8_0.gguf` in the same folder. `--preset 24ink` finds the drafter beside
the model (with `-hf` it fetches it from the same repo); without it decode is much slower, and startup prints a warning
with the fix. Add `--mmproj mmproj-Inkling-2-Mini-F16.gguf` for images.

```bash
# Linux
./llama-server --preset 24ink -m /path/to/Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf -md /path/to/Inkling-2-Mini-ATX-Drafter-Q8_0.gguf
```

```bat
:: Windows
llama-server.exe --preset 24ink -m C:\models\Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf -md C:\models\Inkling-2-Mini-ATX-Drafter-Q8_0.gguf
```

**8, 10 and 12 GB cards**
([jakeatx/Inkling-2-Mini-ATX-SmallCard-GGUF](https://huggingface.co/jakeatx/Inkling-2-Mini-ATX-SmallCard-GGUF)): the
experts stay in system RAM with a VRAM cache, so plan for at least the model file size in free RAM and a PCIe 4.0 x16
slot. These presets run without the drafter (it measured slower on small cards).

```bash
llama-server --preset 12ink -m /path/to/Inkling-2-Mini-ATX-12GB-4.2bpw.gguf
llama-server --preset 10ink -m /path/to/Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf
llama-server --preset 8ink  -m /path/to/Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf
```

On Windows use `llama-server.exe` with the same flags. Speeds and quality are in the release notes of each release.

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

24dense and 8dense were re-measured for v0.7.1; 10dense and 12dense keep their v0.7.0.1 measurements; see [the preset checks](../PRESETS_RECHECK.md). These are not full-context fit guarantees. The 24 GB profile keeps f32 recurrent state and supports one to four slots with asymmetric context reservations; `0` gives a slot a share of the remaining pool. The 8/10/12 GB profiles require `-np 1`. TODO: multi-user for small cards (future). All four profiles enable dynamic microbatch sizing, with checks at most every 2 seconds of prompt processing and 10 seconds of decode. Set `LLAMA_DEEPCTX_LADDER=0` for static sizing. The microbatch ceilings are 2048 for 24dense with one slot (1024 with `-np 2` to `4`) and 512/512/256 for 12/10/8dense; an explicit `-ub` or `LLAMA_ARG_UBATCH` overrides the ceiling. The 8 GB profile uses the accepted F2 settings: reserve 7 rows, stage strips on, CUDA graphs off, 1% buffer growth, and mean chunks on. This build includes the reserve reader. Copy defaults off; earlier 8 GB measurements used copy on, so final measurements must be repeated with copy off. Packed masks remain opt-in. Eligible window-0 paths select DSJ/RM24H automatically, with `--no-spec-draft-sjkvarn` as the opt-out; 24dense with one slot drafts with window 0, gdn-replay off and adaptive depth 3 to 7, so it selects that path; with `-np 2` to `4` it uses the multi-user values (window 4096, gdn-replay on, depth 4 to 5). Context and memory fit depend on the model and overrides.

**Build and run (v0.6.3.11, Qwen3.8 Flash-Next, Linux).** For one RTX 30-series card with enough system RAM for the model and prompt workspace. Small-card tests used a 3090 limited to 12 GB with about 90 GiB of system RAM; native 12/16 GB cards remain unmeasured, and 64 GB of system RAM is not a fit promise. Small cards use memory-bounded prompt passes; see the [hardware limits](llamampere-v0.6.3.5/RELEASE_NOTES.md). The launcher keeps the tested `--fit off` setting; its own card planner still chooses the memory plan and keeps at least 5 GiB of RAM free. Automatic `--fit on` has not been measured on this file. Needs the CUDA 12 or newer toolkit (`nvcc`), CMake, git, a C++
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

## Docker (Linux)

From the next release on, each release also ships a Linux image at `ghcr.io/jakeatx/llamampere` (tags: the release tag, e.g. `v0.7.5`, and `latest`). It is the Linux release tar on a slim Ubuntu 22.04 base: CUDA 12.4 runtime, cuBLAS and the GCC runtime are inside, so the host only needs an NVIDIA driver R550 or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Windows stays the zip.

```bash
docker run --gpus all -v /path/to/models:/models -p 8080:8080 \
  ghcr.io/jakeatx/llamampere:latest -m /models/model.gguf --preset 24dense
```

The entrypoint is `llama-server`, listening on `0.0.0.0:8080` inside the container; everything after the image name is passed to it, so the flags and card presets are the same as in the quick start above. Then open `http://localhost:8080`. To build the image yourself from a release tar, see [.devops/llamampere-server.Dockerfile](../.devops/llamampere-server.Dockerfile).
