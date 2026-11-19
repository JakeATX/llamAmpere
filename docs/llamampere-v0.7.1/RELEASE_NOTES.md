# llamAmpere v0.7.1

Faster writing for one user on 24 GB cards, the 8 GB model file, and faster small-card kernels.

8 GB model: [`jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB`](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB):
`qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB.gguf` (8,393,828,704 bytes), `lsa-qwen38-27b-d2048.gguf` and
`mtp-dm.gguf`. Put all three in one folder.

## What changed

- **24dense with one user writes faster.** With one slot (the default), `--preset 24dense` now drafts with full
  attention (draft window 0), rolls back without GDN replay, and lets the adaptive draft depth go from 3 to 7. Full
  attention makes the drafter eligible for its faster KV cache and the ordered RM24H vocabulary shortlist, which turn
  on automatically. With `-np 2` to `4` the preset keeps the v0.7 multi-user values (window 4096, GDN replay on,
  depth 4 to 5). Explicit flags and environment values still win. To get the v0.7.0.1 one-user behaviour back:
  `--spec-draft-window 4096 --gdn-replay 1 --spec-draft-n-min-adaptive 4 --spec-draft-n-max 5`; `--no-spec-draft-sjkvarn`
  turns off only the faster draft cache and shortlist. Saved slot files from v0.7.0.1 one-user runs do not reload,
  because the draft settings changed.
- **24dense with one user reads long prompts faster.** The prompt-batch ceiling is 2048 tokens with one slot (it was
  1024); the batch ladder still steps down near full context. A cold 98,766-token prompt reads at 1,350 tokens/s instead
  of 1,277 (+5.8%, RTX 3090, P76 AI4); writing speed is unchanged within noise. Peak card memory rises to 21,354 MiB at
  that prompt. With `-np 2` to `4` the ceiling stays 1024. `-ub 1024` (or `LLAMA_ARG_UBATCH=1024`) restores the old
  ceiling.
- **8dense needs three files in one folder:** the model file, the LSA map (`lsa-qwen38-27b-d2048.gguf`) and the small
  MTP drafter `mtp-dm.gguf` (127,489,024 bytes, sha256 `4a8e9a5cfb6b1395088bd88b26400b504e47a10fad5a109e4cbee3276f9d5cae`).
  The preset loads `mtp-dm.gguf` in place of the model file's own MTP layer and no longer looks for `head-ait2.gguf`
  or `head-exact-f16.gguf`. If `mtp-dm.gguf` is missing, startup stops and names the path. In v0.7.0.1,
  `--preset 8dense` with this file stopped with "failed to load overlay patch".
- **The 8dense default context is 204,800 tokens.** With `mtp-dm.gguf` the peak on an RTX 3070 is 7,607 MiB, under the
  7,692 MiB headless line (8,192 minus 500). To use the model file's own MTP layer instead, set
  `LLAMA_MODEL_OVERLAY_PATCH=` (empty) and `-c 114688` (Linux; on Windows an empty variable cannot be set, so keep
  `mtp-dm.gguf`). `-c` still sets any context you want. 8dense stays headless-only: `8dense headed` prints a note and
  uses the same settings.
- **mb-MTP is on for 8dense**, as on the other three presets. `--no-spec-mb-mtp` turns it off.
- **Faster prompt reading on the small-card weight types** (AIF175_U2, AIF25_U2, AIF25_U2_512). These ran the
  one-token kernel even for prompt batches, so each weight was decoded again for every 8 tokens. They now use the
  batch kernel with the same arithmetic (34 of 34 checked calls bit-identical). Off switch: `GGML_CUDA_AIF_RS_MMQ=0`.
- **Faster writing on the same types:** the kernel loads the next block of weights while it works on the current one.
  Off switch: `GGML_CUDA_AIF_MMV_PF=0`.
- **Anthropic API requests think only when they ask.** `/v1/messages` without `"thinking"` (or with
  `"type": "disabled"`) now runs with thinking off, as the Anthropic API does; `"enabled"` or `"adaptive"` turns it
  on. Before, Qwen3.8 thought on every request, and Claude Code's short helper calls (session titles) spent their
  tokens thinking. Apps that never send `"thinking"` now get no thinking: `--anthropic-thinking template` restores
  the old behaviour, and an explicit `--reasoning` or `enable_thinking` in `--chat-template-kwargs` still wins. The
  OpenAI endpoints are unchanged.
- **Claude Code works with Qwen3.8.** Claude Code (2.1.327) got HTTP 500 on every request: it puts a system message
  inside the conversation, which the Qwen3.8 template rejects, and asks for effort "high", which the template does not
  have. System messages before the first reply now join the leading system message (a later one becomes a
  `<system-reminder>` user turn, so the cached prompt start stays the same), and an effort level a template does not
  take maps to its nearest level (high and max to xhigh, minimal to low; logged once). Templates that take any value
  are unchanged. v0.7.0.1 has the same failure.
- **Windows: the Flash-Next CPU split threads ask not to be power-throttled**, like ggml's own CPU threads already
  did, so Windows does not move them to efficiency cores or park them.
- **docs/AI_SETUP.md**: a setup guide to paste into a coding agent (checks the card, picks the preset, downloads the
  files, builds, starts and tests the server).
- Smaller fixes: `GGML_CUDA_DISABLE_GRAPHS=0` now turns CUDA graphs back on (before, any value turned them off);
  the 8/10/12 GB presets no longer set the diagnostic `LLAMA_SPEC_MB_KL=8`; the fit step warns when other programs
  already use the whole preset margin; the prompt-batch ladder no longer fails on non-CUDA GPUs;
  `LLAMA_MTP_DRAFT_COMPUTE_LEAN=1` keeps an explicit smaller `--spec-draft-n-ubatch`; the UI accepts the prebuilt
  service worker (#120); building from source works again with the current upstream prebuilt UI (the build now
  fetches the UI build that matches the server, b12359, #121); a Windows CI build fix (#116).

## Measured

All speed runs: chat template, a coding assignment that asks for a reply over 5,120 tokens, temperature 1, top-k 20,
top-p 0.95, 5,120 tokens written per reply, builds alternated (A B B A), the disk prompt cache off in every run.

**24 GB, one user.** RTX 3090 (Taichi, 350 W), P76 AI4 model, `--preset 24dense`, 6 seeds per build. "Best earlier
setup" is the fastest one-user setup measured before this release (R16 flags, 110,592-token context).

| Prompt | v0.7.0.1 | Best earlier setup | v0.7.1 | v0.7.1 vs v0.7.0.1 |
|---|---|---|---|---|
| 98,766 tokens | 99.83 tokens/s | 111.81 tokens/s | 115.62 tokens/s | +15.8%, 6 of 6 seeds faster |
| 8,000 tokens | 123.13 tokens/s | 130.29 tokens/s | 138.84 tokens/s | +12.8%, 6 of 6 seeds faster |
| Peak card memory | 19,564 MiB | 17,450 MiB (smaller context) | 20,708 MiB | under the 23,552 MiB line |

At a 256,696-token prompt (full 262,144 context) v0.7.1 wrote 89.56 tokens/s with a 21,440 MiB peak (one seed).

The table above was measured with a 1024 prompt-batch ceiling. With the final 2048 ceiling, same card, model and prompt,
6 seeds alternated against the 1024 build: writing 112.03 vs 115.58 tokens/s (per-seed ratio 0.97, 95% range 0.84 to
1.13: no difference shown; a separate 12-seed run measured 110.90 vs 111.05), prompt reading 1,350 vs 1,277 tokens/s,
peak 21,354 vs 20,708 MiB. At the full 256,696-token fill the 2048 ceiling reads the prompt 3.3% faster.

**24 GB, three users.** Same card and model, the shipped multi-user command (`--preset 24dense -np 3 -c 352256
--slot-ctx 262144,0,0`), prompts of 99,984, 32,766 and 32,766 tokens written at the same time, 21 runs per build over
18 seeds. The multi-user settings did not change. Per seed, v0.7.1 / v0.7.0.1 = 0.993 (95% range 0.980 to 1.007, faster
on 7 of 18 seeds): no difference shown.

| | v0.7.0.1 | v0.7.1 |
|---|---|---|
| All three users together | 196.1 tokens/s | 194.9 tokens/s |
| Peak card memory | 22,474 MiB | 22,452 MiB |

**8 GB.** Rented RTX 3070 8 GB (220 W, PCIe 4 x16, Core i5-12400F), AI2M10-iw test file (the unhealed parent of the
release file), `--preset 8dense`. v0.7.0.1 was run with the same settings (`mtp-dm.gguf`, 204,800 context, mb-MTP on),
because its bare preset refuses this file.

| Check | v0.7.0.1 | v0.7.1 |
|---|---|---|
| Bare `--preset 8dense` start with the release file | stops: "failed to load overlay patch" | starts at 204,800 tokens with `mtp-dm.gguf` |
| Writing speed, 99,984-token prompt, 6 seeds | 39.59 tokens/s | 41.41 tokens/s (+4.6%, 6 of 6 seeds faster) |
| Prompt reading, 99,984 tokens | 123 tokens/s | 137 tokens/s (+12%) |
| Writing speed, 199,387-token prompt, 3 seeds | - | 37.24 tokens/s |
| Peak card memory (headless line 7,692) | 7,607 MiB | 7,607 MiB |

The release file itself (healed) on the same card, v0.7.1: 42.66 tokens/s at a 99,984-token prompt (one seed),
peak 7,607 MiB.

The 8 GB card still misses the 40 tokens/s writing goal at deep prompts (37.24 at 199,387).

**10 and 12 GB.** The presets start and write on v0.7.1 (smoke test on an RTX 3090 limited to 9,920 and 11,910 MiB,
one 512-token chat reply, 120 to 146 tokens/s). Their settings are unchanged from v0.7.0.1 except that
`LLAMA_SPEC_MB_KL=8` is no longer set; they were not re-measured on 10 and 12 GB cards for this release.

**Claude Code check** (bare `--preset 8dense`, the 8 GB release file, RTX 3090 Ti): captured Claude Code requests
(three sessions, streamed and not, three turns each, about 11,400-token prompts) get 18 of 18 answers (thinking, text
and Bash tool calls) on v0.7.1; on the build before this fix the first request of each session fails (HTTP 500).

**Linux build:** `llamampere-v0.7.1-linux-x86_64-cuda12-server.tar` runs on Ubuntu 22.04 or newer (glibc 2.35+) with
an NVIDIA driver and the CUDA 12 runtime and cuBLAS. It brings its own GCC runtime libraries (libstdc++, libgcc_s,
libgomp; GPL with the GCC Runtime Library Exception, licence and source notes inside), so no newer compiler packages
are needed; checked on stock Ubuntu 22.04 with an RTX 3060 Ti. Building from source needs GCC 13 or newer. The source
of those GCC runtime libraries (gcc-16 16-20260315-1ubuntu1~22~ppa1, Ubuntu Toolchain PPA) is attached to the release.

**Windows build:** `llamampere-v0.7.1-windows-x86_64-cuda12-server.zip` holds `llama-server.exe` with the CUDA 12
runtime and cuBLAS DLLs and the web UI; it needs an NVIDIA driver. Windows build made by CI; not yet run on a GPU by
us, report issues.
