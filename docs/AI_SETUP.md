# Set up llamAmpere with your coding agent

If you use an AI coding assistant (Claude Code, Codex, Cursor, Copilot, ...), paste this into it:

```text
Set up llamAmpere on this PC for me: https://github.com/JakeATX/llamAmpere - follow docs/AI_SETUP.md in that repository. Ask me before you install system packages or use sudo.
```

The rest of this page is written for the agent.

---

## Instructions for the agent

Work step by step. Show the user each command before it changes the system. Never use `sudo` or install system
packages without asking. Stop and explain if a check fails; do not work around it.

### 1. Check the machine

Run these and keep the answers:

```bash
uname -s                                   # must be Linux (Windows: see "Windows" at the end)
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
free -g | head -2                          # system RAM
df -h .                                    # free disk where you will build and keep models
nvcc --version || ls /usr/local/cuda*/bin/nvcc   # CUDA toolkit 12 or newer
cmake --version; git --version
```

- The card should be an NVIDIA RTX 30-series (Ampere: RTX 3060, 3070, 3080, 3090, A-series workstation cards).
  Other NVIDIA cards may work, but the presets are tuned and measured on Ampere only. Say so to the user.
- If `nvcc` is missing, ask the user to install the CUDA toolkit (12 or newer) for their distribution. Do not
  install it yourself without permission.
- Note whether a display is connected to this card (`memory.used` above a few hundred MiB with no other program
  running usually means a desktop is using it).

### 2. Pick the card preset

Use the card's total memory from step 1:

| Card memory | Preset | Notes |
|---|---|---|
| 24 GB (3090, 3090 Ti, A5000) | `24dense` | up to four users |
| 12 GB (3060 12 GB, 3080 12 GB) | `12dense` | one user; add `headless` if no display uses the card |
| 10 GB (3080 10 GB) | `10dense` | one user |
| 8 GB (3060 Ti, 3070, 3070 Ti) | `8dense` | one user, headless only (no display on the card) |

Other sizes (16 GB or 20 GB cards such as the A4000 or A4500) have no preset: ask the user which one to try rather
than picking one yourself.

Each preset needs its own model file. Take the file for the chosen preset from the **Quick start: pick your
card** section of the repository README for the release you build. Do not mix a preset with another preset's file.
If the README gives no file name for the chosen preset (for example "name TBD"), stop and tell the user that this
card size has no released model file yet.
The model files are on https://huggingface.co/jakeatx.

### 3. Get the code and build

Use the newest release tag (not `master`):

```bash
git clone https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
tag=$(git tag --list 'v0.[0-9]*' --sort=-v:refname | head -1); echo "$tag"   # llamAmpere tags are v0.x
git checkout "$tag"
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
```

If the build fails, show the user the first error, not the last lines.

### 4. Download the model file

```bash
python3 -m venv .venv && . .venv/bin/activate   # never use pip --break-system-packages
pip install -U huggingface_hub                   # gives the `hf` command
hf download jakeatx/<repository> <file>.gguf --local-dir models
# 8/10/12 GB presets: also the LSA map, into the same folder
hf download jakeatx/<repository> lsa-qwen38-27b-d2048.gguf --local-dir models
# 8 GB preset: also the small MTP drafter, into the same folder
hf download jakeatx/<repository> mtp-dm.gguf --local-dir models
```

The 8, 10 and 12 GB presets stop at start and name the missing file if `lsa-qwen38-27b-d2048.gguf` (and, for 8 GB,
`mtp-dm.gguf`) is not beside the model file. Use the files listed on the model page.

Check the free disk from step 1 against the file size on the model page first.

### 5. Start the server

```bash
build-sm86/bin/llama-server --preset <preset> -m models/<file>.gguf --host 127.0.0.1 --port 8080
```

- The first start loads the model; wait for `server is listening`.
- Keep `--host 127.0.0.1` unless the user asks for access from other machines. If they do, also set
  `--api-key <secret>` and tell them the key.
- If it stops with an out-of-memory error, check that no other program holds card memory (`nvidia-smi`), then
  tell the user. Do not lower settings by guessing; the presets are measured.

### 6. Check it works

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

The log line `eval time` gives the writing speed in tokens per second. Report it to the user together with the
card name, the preset and the prompt length.

### 7. Connect the user's apps

- OpenAI-compatible apps: base URL `http://127.0.0.1:8080/v1`, any model name, any key (or the `--api-key` value).
- Anthropic-compatible apps such as Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. The server thinks
  only when the request asks for thinking, as the Anthropic API does (`--anthropic-thinking template` keeps the
  model's default instead). Older builds lack this option; check `llama-server --help` for `--anthropic-thinking`.

### 8. Tell the user

Summarize: card, preset, model file, build tag, the start command, the measured speed, and how to stop the server
(Ctrl+C in its terminal). Give the start command as one line they can paste later.

### Windows

The presets are measured on Linux. A native Windows build (Visual Studio + CUDA toolkit) works for
experienced users; WSL2 with the NVIDIA driver is the simpler path: follow the Linux steps inside WSL2.
