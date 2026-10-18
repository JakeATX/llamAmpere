# llamAmpere v0.5.5

One new opt-in feature: **`--eval-cache DIR`**, a prompt cache on disk for prompts you send again and again, such as
eval and benchmark prompts. Nothing changes unless you pass the flag.

## Why it matters

- **About 6-7x faster long-context evals with everything on.** Eval cache + MTP drafting + parallel slots stack. The cache removes the wait before the answer; the MTP drafter speeds up the
  answer itself. Approximate typical speed-up per 100K eval prompt, as arithmetic from measured parts (not one
  end-to-end run): about **6-7x** for a 2,000-token answer with eval cache + MTP + `--parallel 2`, about 17x for a
  500-token answer, about 3x for a 10,000-token reasoning answer. Parts: 105.1 s vs 1.65 s prefill; 91 tok/s
  decode with MTP at 100K; MTP 2.19x and two slots 1.06x (with MTP) in a same-card A/B with tq5_0/turbo4 KV.
- **Long-context evals stop waiting on prefill.** Swift 1.5 (Qwen3.8 27B) on a ~102K-token prompt: 1.6-1.7 s to load
  the saved prompt against 104.6-105.1 s to read it fresh, about **64x less prefill** (RTX 3090 Ti, 350 W). Time to
  first answer goes from 105.7 s to 4.4 s.
- **Hours back per run.** About 101 s saved per 100K prompt: a 100-prompt long-context eval saves about 2.8 hours per
  pass, and every re-run (new settings, new seeds, a new sampler) saves it again. The folder is built once.
- **Same answers.** At temperature 0 the text is byte-for-byte the same as a fresh prefill, so scores stay comparable
  with runs that don't use the cache.
- **Download instead of building.** Ready-made folders for AA-LCR, LongCodeQA 32K/64K/128K, MRCR, RULER 128K and
  LongBench v2 (being completed) for Swift 1.5 IQ4_XS-M with SJ-KVaRN 4/4:
  https://huggingface.co/datasets/jakeatx/llamampere-evalcache
- **No SSD wear at eval time.** The server only reads the folder.
- **Safe by default.** Opt-in; a folder that doesn't match the model or cache settings is refused at startup.

## What it does

Before a model writes its first word, the server has to read the whole prompt. This step is called **prefill**. A
100,000-token prompt takes about two minutes of prefill on a 3090 Ti, and an eval pays that again for every prompt on
every run.

`--eval-cache DIR` points the server at a folder of saved prompt states. A new tool,
`tools/eval-cache/llama-eval-cache.py`, builds the folder once. After that, a request whose prompt starts with a
saved prompt loads the saved state from disk and only reads the rest. Loading is a plain file read.

- **Same output.** At temperature 0 the text is byte-for-byte the same as without the cache. The saved state ends
  exactly where a normal prefill would pause between two batches, and the server continues from there in the same way.
- **Shared documents are stored once.** When several prompts ask different questions about the same long document,
  the tool stores the document once. Each prompt then reads only its own question.
- **Longest match.** The server picks the longest saved prompt that the request starts with.
- **Refuses a wrong folder at startup.** A folder records the model file, the KV-cache types (including SJ-KVaRN
  settings), the recurrent-state type, the MTP drafter setup and the cache format version. If the server's settings
  differ, it refuses to start and names the setting that differs. It never loads a state that does not match.
  A folder written in an older cache format is not an error: the server logs a warning, ignores the folder and
  prefills normally (the folder is left untouched). Rebuild it into a new folder to use it again.
- **Works with every KV cache type we ship.** Tested: q8_0/q8_0, tq5_0/turbo4, SJ-KVaRN 4/4, 3/3t and 3/2t, on
  plain-attention, sliding-window (Gemma) and hybrid (Qwen3.5/3.8) models, with and without the MTP drafter, with
  `--parallel 1` and `--parallel 2`.
- **Read-only at eval time.** The server never changes the folder, so an eval adds no SSD writes.
- **Checkable.** `verify` checks every file against its checksum and can re-run prompts with and without the cache
  and compare the text.

## Measured

Home RTX 3090 Ti at its 350 W limit. Each prompt is a long document plus a short question; four questions share
one document, so the folder stores the document once. "Same text" = temperature-0 output compared byte for byte
with and without the folder (64 tokens per prompt). Sources: the v0.5.5 measurement notes.

| Model | Prompt length | Folder | Prefill per prompt, with / without the folder | Same text |
|---|---|---|---|---|
| Swift 1.5 (Qwen3.8 27B) IQ4_XS-M, SJ-KVaRN 4/4 | 102,273-102,400 tokens | 6.8 GiB, built in 260 s | 1.6-1.7 s / 104.6-105.1 s (about 64x less prefill; whole request 4.2-4.5 s vs 105.5-106.1 s) | 4/4 (`--parallel 1`) |
| Qwen3.8 Flash-Next IQ3_XXS, experts in RAM, `-ub 8192` | 100,021 tokens | 2.6 GiB, built in 562 s | 128-143 s / 453-478 s | 1/1 (`--parallel 1`), 1/1 (`--parallel 2`) |
| Qwen3.8 Flash-Next IQ3_XXS, experts in RAM, `-ub 8192` | 32,830 tokens | 0.5 GiB (8,205 tokens stored) | 85-118 s / 113-154 s | 2/2 (`--parallel 1`), 2/2 (`--parallel 2`) |
| DeepSeek V4.1 Flash Q2_K, CPU only (rented 104-thread box, test build) | 8,145 tokens | 60 MiB, built in 592 s | 94-153 s / 292-362 s | 3/3 (`--parallel 1`), 3/3 (`--parallel 2`) |
| Gemma 4 31B IQ4_XS, q8_0 KV | 99,200 tokens | 4.5 GiB, built in 155 s | 8.3-10.1 s / 130.1-130.5 s | 4/4 (`--parallel 1`), 2/2 (`--parallel 2`) |
| Gemma 4 31B IQ4_XS, q8_0 KV | 32,745 tokens | 1.9 GiB, built in 37 s | 2.9-3.1 s / 28.7-29.0 s | 4/4 (`--parallel 1`), 2/2 (`--parallel 2`) |

"With the folder" includes loading the saved state from disk (Flash-Next at 100K: 1.4-1.7 s for 2.6 GiB; Gemma at
100K: 0.9-3.7 s for 4.5 GiB). Flash-Next saves less at 32K because with `-ub 8192` the folder can only store the first
8,205 of the 32,830 tokens (see Limits); built with `build --margin 256` it stored 16,397 tokens and the prefill took
74 s instead of 116 s (1/1 same text). The Gemma 100K `--parallel 2` cell ran with `--kv-unified` and a folder built
with `build --parallel 2` (see Limits).

Whole eval runs (server start not included): four 100K Gemma prompts took 43.8 s with the folder and 528.7 s
without; four 32K prompts 15.6 s and 118.7 s.

Every KV cache type was checked end to end on small models (build, restart, load, same text): q8_0/q8_0,
tq5_0/turbo4, SJ-KVaRN 4/4, 3/3t and 3/2t, and a hybrid model with the MTP drafter, at `--parallel 1` and
`--parallel 2`: 21/21 pass on CPU and 24/24 on the GPU (the GPU set adds two hybrid + MTP cells). A folder built for other settings or another model is refused: 3/3.

## How to use it

Build v0.5.5 as usual (or reuse your build directory):

```sh
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 -j8 --target llama-server
```

1. Write your eval prompts as JSONL, one request per line, exactly as the eval sends them (`messages` lines go to
   `/v1/chat/completions`, `prompt` lines to `/completion`). Settings that change the prompt text, such as
   `reasoning_effort`, `chat_template_kwargs` or the system prompt, must match the eval. Sampling settings do not
   matter.

   ```json
   {"id": "q1", "messages": [{"role": "user", "content": "<long document>\n\nQuestion 1"}]}
   {"id": "q2", "messages": [{"role": "user", "content": "<long document>\n\nQuestion 2"}]}
   ```

2. Build the folder with the same model and flags you will run the eval with (here Swift 1.5, our Qwen3.8-27B
   model, with the SJ-KVaRN 4/4 KV cache):

   ```sh
   python3 tools/eval-cache/llama-eval-cache.py build --dir ./evc-swift --prompts prompts.jsonl \
       --server ./build-sm86/bin/llama-server -- \
       --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
       --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -c 110592
   ```

3. Run the eval server with the folder:

   ```sh
   ./build-sm86/bin/llama-server --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
       --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 \
       -c 110592 --eval-cache ./evc-swift
   ```

   Each hit prints `eval cache: restored entry ..., N of M prompt tokens` in the server log.

4. Optional: check the folder, and compare outputs with and without it at temperature 0:

   ```sh
   python3 tools/eval-cache/llama-eval-cache.py verify --dir ./evc-swift
   python3 tools/eval-cache/llama-eval-cache.py verify --dir ./evc-swift --identity 4 --prompts prompts.jsonl \
       --server ./build-sm86/bin/llama-server -- \
       --hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
       --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -c 110592
   ```

Qwen3.8 Flash-Next with experts in system RAM (24 GB card + 64 GB RAM):

```sh
python3 tools/eval-cache/llama-eval-cache.py build --dir ./evc-flashnext --prompts prompts.jsonl \
    --server ./build-sm86/bin/llama-server -- \
    -hf ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_XXS --no-mmproj -ngl 99 --cpu-moe --moe-cache auto \
    -lzm on -fa on -ub 8192 -b 8192 -c 131072
./build-sm86/bin/llama-server -hf ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_XXS --no-mmproj -ngl 99 \
    --cpu-moe --moe-cache auto -lzm on -fa on -ub 8192 -b 8192 -c 131072 --eval-cache ./evc-flashnext
```

Free to change between build and eval: `-c`, `--parallel`, `--kv-unified`, GPU layers, threads and sampling. Keep
`-b`/`-ub` the same if you want byte-identical output. The full guide is `tools/eval-cache/README.md`.

## Ready-made folders

[jakeatx/llamampere-evalcache](https://huggingface.co/datasets/jakeatx/llamampere-evalcache) has eval-cache folders
for Swift 1.5 IQ4_XS-M with SJ-KVaRN 4/4 for long-context benchmarks (AA-LCR, LongCodeQA 32K/64K/128K, MRCR and
more as they finish uploading), plus the exact prompt files they were built from. Download a folder, start the server
with the flags listed on the dataset card and `--eval-cache <folder>`, and send the prompt files as they are.

## Limits

- An entry is about the size of the prompt's KV cache: 4.5 GiB for 96,270 Gemma 4 31B tokens with q8_0 KV,
  2.6 GiB for 81,933 Flash-Next tokens. Smaller KV types give smaller files.
- A shared entry stops a little before the end of the shortest prompt that uses it (two batches plus 260 tokens) and
  at the start of a batch, so with very large batches (`-ub 8192`) less of each prompt is stored: 82% of a 100K-token
  prompt and 25% of a 32K-token prompt on Flash-Next. `build --margin 256` shortens the gap to one batch plus 260
  tokens (Flash-Next 32K: 50% stored, same text 1/1).
- Gemma and other sliding-window models with `--kv-unified` and `--parallel N` above 1: build the folder with the same
  `--parallel N` (`build --parallel N ... -- ... --kv-unified`) to get the same text. Built with one slot, 1 of 2 Gemma
  prompts at 100K came out different.
- Servers started with a multimodal projector (`--mmproj`) do not use the eval cache.
- Byte-identical output is per prompt with one request running at a time. With several requests running at once,
  batches mix prompts, which can change the last digits of the math with or without the cache.

## Other changes

- Fixed: the MTP drafter (multi-token prediction, the small built-in model that guesses the next few tokens) kept
  the last request's state when a slot was reused for a new prompt, so a reused server could draft differently from a
  freshly started one. It now starts clean at the beginning of every prompt. Its state is also restored correctly
  when a prompt is loaded from a prompt cache (slot, RAM, disk or eval cache).
- Fixed: a request naming an empty slot with `id_slot` skipped the prompt cache and read a cached prompt again.
- Fixed: when no placement of the model fits in GPU memory, the automatic fit now stops before loading and says
  which device, how much memory it needs and how much it has, instead of failing later during allocation.
- Fixed: loading a Qwen3.8 Flash-Next MTP drafter (`-md mtp-...gguf --spec-type draft-mtp-adaptive`) crashed on
  v0.5.4. The drafter now uses the same hybrid memory as the main model. Checked on the home card: Flash-Next with the
  MTP drafter and an eval-cache folder, 3/3 prompts the same text with and without the folder, 3/3 hits.

- `--parallel N` can now load a prompt state saved with a different `--parallel` (one-sequence restores accept any
  number of KV streams, including the indexer cache of Qwen3.8 Flash-Next). This also helps `/slots` save and restore.
- New library calls `llama_state_seq_save_file_ext` / `llama_state_seq_load_file_ext` with the flag
  `LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT`: a saved sequence goes back into the same KV cells it came from. On
  sliding-window models (Gemma) this keeps the math of a restored prompt exactly the same as a fresh prefill. The eval
  cache uses it; existing save/restore calls are unchanged.
