# llamAmpere v0.7.2.1

A small update to v0.7.2 for 8 GB cards, plus the Windows link fix. `--preset 8dense` reads long prompts faster and
keeps your conversations in a prompt cache, so switching between chats or restarting the server no longer means
reading the whole chat again. Same model files as v0.7.2. The 24dense, 12dense and 10dense presets are unchanged.

## What changed

- **8 GB cards read prompts faster.** `--preset 8dense` now processes prompts in batches of 512 tokens instead of 256
  and still starts at the full 204,800-token context. To pay for the larger batch it keeps a smaller staging buffer for
  the sparse-attention cache (`GGML_CUDA_LSA_RESIDENT_STAGE=128`, was 256; the output is bit-identical). On the 8 GB
  file, reading a 99,984-token prompt from cold went from 441.8 to 507.4 tokens/s (+14.8%), and filling the rest of the
  context up to 204,640 tokens went 27% faster (RTX 3090 limited to a 3070's free memory).
  @@PENDING UB512: real RTX 3070 numbers (prefill tok/s at 99,984, decode tok/s v0.7.2 vs v0.7.2.1, peak MiB vs the
  7,692 MiB line)@@
  An explicit `-ub` or `GGML_CUDA_LSA_RESIDENT_STAGE` still overrides the preset.
- **8 GB cards keep a prompt cache.** Before, `--preset 8dense` turned the prompt cache off. Now it uses the normal
  defaults: a RAM cache sized from your host memory (12 GiB on a 64 GB machine, at most half of the free memory) and a
  disk cache of up to 16 GiB in `~/.cache/llamampere/prompt-cache`. On an 11.6K-token agent chat, switching back from
  another chat or resending after a server restart now waits about 0.5 to 1.9 s for the first token instead of about
  17 to 18 s (RTX 3090 Ti and RTX 3090, both limited to a 3070's free memory). A 100K-token chat comes back in 0.89 s
  instead of 225 s (RTX 3090 limited to a 3070's memory). The cache lives in host RAM and on disk, not on the card:
  card memory is the same, and host memory grows by about 0.65 GB for each cached 12K-token chat. A chat restored
  from the cache continues with exactly the same tokens as one that never left the card.
- **Warm prefill on 8 GB cards.** `--preset 8dense` also turns on `--warm-prefill`: after a reply, the server reads the
  part of the next request it already knows while you type. It helps clients that send the model's thinking back with
  the history: there the first token of the next turn came after 0.16 to 0.25 s instead of 1.75 to 2.26 s. With clients
  that drop the thinking it neither helps nor costs time. Because it reads the same tokens in a different batch split,
  a reply can differ from v0.7.2 at a token where two choices are almost equally likely.
- **Kill switches:** `--cache-ram 0` (no prompt cache), `--no-cache-disk` (RAM only), `--no-warm-prefill`. Explicit
  flags and `LLAMA_ARG_*` environment values always win over the preset, in either order.
- **Quieter log.** The MTP drafter printed "verify reserve skipped: non-default context" as a warning at every memory
  reserve. That is expected, so it is now a debug line.
- **Windows build from the tag.** The v0.7.2 source did not link on Windows (the server called a function the Windows
  DLL does not export), so the v0.7.2 Windows zip was built from a one-line fix on top of the tag. That fix is part of
  v0.7.2.1, and its Windows zip is built from the v0.7.2.1 tag itself.

## Measured

Speed gate, v0.7.2.1 against v0.7.2: chat template, temperature 1, 6 seeds per build in ABBA order.

| card | model and setup | prompt | reply | v0.7.2 tokens/s (sd) | v0.7.2.1 tokens/s (sd) | v0.7.2.1 / v0.7.2 |
|---|---|---:|---:|---:|---:|---:|
| RTX 3090 | 27B AI4, `--preset 24dense`, disk cache off | 98,766 | 5,120 | @@PENDING G24 A@@ | @@PENDING G24 B@@ | @@PENDING@@ |
| RTX 3090, limited to a 3070's memory | 8 GB file, `--preset 8dense` | 99,984 | 5,120+ | @@PENDING G8 A@@ | @@PENDING G8 B@@ | @@PENDING@@ |
| RTX 3070 | 8 GB file, `--preset 8dense` | 99,984 | 5,120+ | @@PENDING UB512 real@@ | @@PENDING UB512 real@@ | @@PENDING@@ |

- `--preset 8dense` with the 8 GB file starts at 204,800 tokens of context with batch 512; peak card memory at a
  204,640-token fill: @@PENDING smoke peak@@ MiB (v0.7.2: @@PENDING@@; headless line 7,692 MiB).
- The GPU libraries (`libggml*`) are byte-identical to v0.7.2.

**Linux build:** `llamampere-v0.7.2.1-linux-x86_64-cuda12-server.tar`, same as v0.7.2: Ubuntu 22.04 or newer (glibc
2.35+), NVIDIA driver for CUDA 12.4 or newer (R550+), no CUDA toolkit needed, built for RTX 30-series cards (sm_86).
It brings the CUDA 12.4 runtime and cuBLAS (NVIDIA redistributables, CUDA-EULA.txt inside) and its own GCC runtime
libraries (licence and source notes inside; the GCC source is attached to the release).

**Ada and Blackwell build:** `llamampere-v0.7.2.1-linux-x86_64-cuda12.8-sm86-89-120-server-fat.tar` is built for RTX 30
(sm_86), RTX 40 (sm_89) and RTX 50 (sm_120) cards with CUDA 12.8 and needs an NVIDIA driver 570 or newer. Its GPU
libraries are the v0.7.2 ones. @@PENDING fat smoke: checked on an RTX 3090 (sm_86)@@

**Windows build:** `llamampere-v0.7.2.1-windows-x86_64-cuda12-server.zip`, built by CI like v0.7.2 (CUDA 12.4, sm_86,
driver R550+); not run on a GPU by us.
