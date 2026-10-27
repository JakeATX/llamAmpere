# llamAmpere v0.6

v0.6 makes Swift 1.5 (Qwen3.8 27B) faster on one RTX 3090 / 3090 Ti and adds the engine for Qwen3.8 Flash-Next, a
large mixture-of-experts model whose experts live partly in system RAM and partly on the card.

- **Swift 1.5 27B:** faster prompt reading, a faster MTP drafter setting on by default (mb-MTP, see below), and
  support for AI+ model files (our 4-bit weight format for Ampere and newer cards).
- **Qwen3.8 Flash-Next:** expert cache on the card with the rest of the experts in RAM, a much faster long-prompt read,
  and a KV cache that keeps only the recently used part of the context on the card (`--kv-gpu-tokens`). The launch
  command and measured numbers for Flash-Next are published separately (see "Flash-Next" below).

## Swift 1.5 27B: v0.6 against v0.5

**Measurement in progress.** The table is filled in once the runs finish; this note will be updated without a new
version number.

Home RTX 3090 Ti at 350 W, 100K-token prompt (real coding, agentic and RAG task histories), 5,120 written tokens per
answer, temperature 1.0, writing speed in tokens per second. G = 0.4 coding + 0.4 agentic + 0.2 RAG.

| Setup | Coding | Agentic | RAG | G |
|---|---|---|---|---|
| v0.5 release, IQ4_XS-M, published command (tq5_0/turbo4, adaptive draft depth 3-4) | pending | pending | pending | pending |
| v0.6, IQ4_XS-M, SJ-KVaRN 4/4, exact drafting (`--no-spec-mb-mtp`) | pending | pending | pending | pending |
| v0.6, AI+ 4-bit, SJ-KVaRN 4/4, exact drafting | pending | pending | pending | pending |
| v0.6, AI+ 4-bit, SJ-KVaRN 4/4, mb-MTP, draft depth 6 | pending | pending | pending | pending |

## Run it (Swift 1.5 27B, one RTX 3090 / 3090 Ti, 262,144-token context)

Needs Linux, an NVIDIA card with 24 GB, the CUDA toolkit (tested with 12.4), CMake, git and a C++ compiler. The model
download is 15.6 GB.

```bash
git clone -b v0.6 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
./build-sm86/bin/llama-server \
  --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -c 262144 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). The drafter uses the built-in 65,536-token
vocabulary shortlist. For exact (lossless) drafting add `--no-spec-mb-mtp`. The v0.5 commands in the README still work.

## What is new

### mb-MTP, on by default with the MTP drafter

The MTP drafter guesses several tokens ahead and the main model checks them. Normally a guessed token the model would
not have picked exactly is thrown away. mb-MTP ("mode-biased MTP") keeps a guessed token when the model rates it at
least as likely as the token it would put in its place (never for end-of-text or stop tokens). More guesses survive,
so each step writes more tokens and longer draft depths pay off (depth 6 instead of 3-4).

The trade: the output leans slightly toward the model's most likely tokens. Measured on Swift 1.5 at draft depth 6 the
difference from exact sampling is 0.009 KL divergence per token. That measures how much sharper the text gets, not
errors, so it is not comparable to quantization KL. `--no-spec-mb-mtp` gives exact sampling, the same output
distribution as running the model without a drafter.

### Faster prompt reading for Swift 1.5

A new kernel for the GDN layers (the layers that carry a running memory instead of attention) shares work between
heads that use the same key. On a 100K prompt with the AI+ file, prompt reading went from about 1,204 to 1,267 tokens
per second (+5.3%, two back-to-back pairs) with the same written text.

### AI+ model files

AI+ is our weight format for Ampere and newer cards (4-bit, also 2- and 3-bit rates), with its own decode kernels.
v0.6 can run AI+ files. A Swift 1.5 AI+ 4-bit file will be posted on Hugging Face; the README will link it.

### Qwen3.8 Flash-Next

Flash-Next does not fit on one 24 GB card, so v0.6 keeps the most used experts on the card and runs the rest from RAM:

- **Expert cache on the card**, filled from a shipped usage profile (`data/moe/flash-next.profile`) and kept as whole
  experts.
- **Fast whole-prompt read:** the prompt is read layer by layer in large batches, with the drafter handed off during the
  read. On a 100K prompt this took prompt reading from about 1,450 to about 2,330 tokens per second on our earlier
  runs.
- **`--kv-gpu-tokens N`:** keeps N tokens' worth of the KV cache on the card and the rest in pinned RAM, which leaves
  more card memory for experts. Which pages stay on the card is now chosen by recent use (a CLOCK page cache, on by
  default when `--kv-gpu-tokens` is set): +5.65% writing speed over the previous choice on Flash-Next (4 seeds).
- **2-bit MTP drafter file** with the output head shared with the main model: about 390 MiB less card memory than the
  3-bit drafter, which leaves room for about 200 more experts on the card at the same speed.

**Flash-Next numbers and the launch command are pending** and will be added to this page without a new version
number.

**Run-to-run variation with expert offload.** When experts are split between the card and the CPU, which experts sit
on the card can change from run to run, and the CPU and the card round slightly differently. So even at temperature 0
the text can differ between runs. That is expected, not a bug. Details: "Run-to-run variation with expert offload" in
`docs/backend/MOE-CACHE.md`.

### Also in v0.6

- **EXL3:** small helper steps around EXL3 matrix products are folded into their neighbors (same output bits).
- **Nemotron 3.5 Bolt and Super** load and run.
- **Packed attention masks** for cards with 12 GB or less (`LLAMA_KQ_MASK_PACKED=1`, off by default; see
  `docs/llamampere-v0.6/CHANGELOG.md`).
- **Fix:** SJ-KVaRN works with image prompts on models that use M-RoPE (#39, #49).

## Gates run for this release

Filled in as each gate finishes: full build, CPU tests, a Swift 1.5 smoke run of the command above on an RTX 3090 Ti,
and a Flash-Next smoke run.
