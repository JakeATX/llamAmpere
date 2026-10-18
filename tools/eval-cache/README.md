# Eval cache: skip the prefill of prompts you send again and again

Evals and benchmarks send the same long prompts every run. Before the model writes its first token, the server has
to read the whole prompt ("prefill"). At 100K tokens that is minutes per prompt, every run.

An **eval-cache folder** stores the server's state after reading each prompt. A server started with
`--eval-cache DIR` finds the longest stored prompt that a request starts with, loads it from disk, and reads only
the rest of the prompt. Loading is a sequential file read, much faster than recomputing the prompt. The folder is
read-only at eval time: entries are never evicted or changed.

At temperature 0 the output is the same, byte for byte, as without the cache: the stored state ends exactly where a
normal prefill of that prompt would pause between two batches, and the server continues from there the same way.

## Recipe

1. Write your prompts as JSONL, one request per line, exactly as your eval sends them. Anything that changes the
   prompt text (`reasoning_effort`, `chat_template_kwargs`, `tools`, the system prompt) must be the same as in the
   eval. Sampling fields do not matter.

   ```json
   {"id": "q1", "messages": [{"role": "user", "content": "<long document> ... Question 1"}], "reasoning_effort": "medium"}
   {"id": "q2", "messages": [{"role": "user", "content": "<long document> ... Question 2"}], "reasoning_effort": "medium"}
   {"id": "c1", "prompt": "<plain completion prompt>"}
   ```

   `messages` lines go to `/v1/chat/completions`, `prompt` lines to `/completion`. Use the same endpoint in the eval.

2. Build the folder with the **same model and KV flags** you will run the eval with:

   ```sh
   python3 tools/eval-cache/llama-eval-cache.py build --dir ./evc-mybench --prompts prompts.jsonl -- \
       -m model.gguf -ngl 99 -fa on -ctk tq5_0 -ctv turbo4
   ```

   The tool starts `llama-server` (from `build/bin`, or `--server PATH`) with your flags plus
   `--eval-cache DIR --eval-cache-build --parallel 1 --cache-ram 0` (the builder's own `--parallel N` changes the
   slot count), sends every prompt once with `max_tokens` 1 from an empty cache, and writes `manifest.json` (model
   sha256, flags, which entry covers which prompt, file checksums).

3. Run the eval with the folder:

   ```sh
   llama-server -m model.gguf -ngl 99 -fa on -ctk tq5_0 -ctv turbo4 --eval-cache ./evc-mybench
   ```

   The log shows `eval cache: restored entry ..., N of M prompt tokens` for each hit. `/props` shows the folder.

4. Check a folder (files and checksums; `--identity N` also runs the first N prompts at temperature 0 with and
   without the cache and compares the text):

   ```sh
   python3 tools/eval-cache/llama-eval-cache.py verify --dir ./evc-mybench
   python3 tools/eval-cache/llama-eval-cache.py verify --dir ./evc-mybench --identity 3 --prompts prompts.jsonl -- \
       -m model.gguf -ngl 99 -fa on -ctk tq5_0 -ctv turbo4
   python3 tools/eval-cache/llama-eval-cache.py ls --dir ./evc-mybench
   ```

   `--report timings.json` writes, per prompt, the prompt tokens read, the tokens loaded from the folder, the
   prompt time and the request time of both runs.

## Which settings must match

The folder records a **key**: the model file, the KV-cache types (including SJ-KVaRN bits, body codec, tail and
edge settings), the recurrent-state type (`--cache-type-s`), the drafter / MTP setup and its KV types, flash
attention and rope settings, and the cache-format version. A server with a different key **refuses to start** and
names the first setting that differs, for example:

```
eval cache: refusing to start: ./evc-mybench was built for another model or KV configuration
(this server has 'ctk=q8_0', the folder was built with 'ctk=tq5_0'); ...
```

Folders written with cache format 1 must be rebuilt to reuse their state. Read-only mode warns, skips an unsupported cache version and serves with fresh prefill without changing the folder. Build mode refuses an incompatible folder; build into a new folder instead. Format 2 also saves the MTP carry row, deferred catch-up rows and adaptive-depth state. The adaptive depth range and chain settings must match when building and loading a folder.

The CPU HTTP compatibility regression uses a generated model and a CPU-only server build:

```sh
build-cpu/bin/test-mtp-state --save-model /tmp/eval-cache-fixture.gguf
LLAMA_SERVER_BIN_PATH="$PWD/build-cpu/bin/llama-server" LLAMA_TEST_MTP_MODEL_FILE=/tmp/eval-cache-fixture.gguf \
    python3 tools/eval-cache/test_eval_cache.py -v
```

Not part of the key, so free to change: `--ctx-size`, `--parallel`, `--kv-unified`, the number of GPU layers, threads, sampling.

Keep `-b` / `-ub` (batch sizes) the same as at build time if you want byte-identical output: the cache is correct
with any batch size, but the identity guarantee relies on the same batch layout. `manifest.json` records both.

Sliding-window models (Gemma) with `--kv-unified` and `--parallel N` (N > 1): the slots share one window cache whose
size grows with N, so pass the same `--parallel N` to the builder (`build --parallel N ... -- ... --kv-unified`) for
byte-identical output. Without `--kv-unified` every slot has its own window cache of the same size, and a folder built
with the default `--parallel 1` gives identical output at any `--parallel`.

## Prompts that share a long document

Many evals ask several questions about one document. The builder groups prompts that share at least
`--min-shared` leading tokens (default 1024), or those with the same `"group"` field, and stores the shared part
**once**. Each prompt then prefills only its own question. The shared entry ends a little before the shortest
member's end (the last `n_ubatch + 4` tokens of a prompt are always processed in their own batches;
`--margin` adds more room, default `n_ubatch + 256`), and at the start of a batch.

`--full` also stores every grouped prompt on its own (more disk, slightly less prefill).

## Disk use

An entry is about the size of the prompt's KV cache (plus the recurrent state on hybrid models), so smaller KV
types give smaller files. `llama-eval-cache.py ls` prints the size of every entry. The builder writes each entry once; the eval only reads, so a folder
costs no SSD wear after it is built.

## Folder layout

```
DIR/eval-cache.json         index (format version, key, entries); written by the server
DIR/manifest.json           written by the builder: model hash, flags, prompt -> entry, checksums
DIR/entries/<id>.evh        entry header: prompt tokens, context checkpoints, checksum
DIR/entries/<id>.tgt        model state (streamed from disk on load)
DIR/entries/<id>.dft        drafter state, when the drafter has its own KV cache
DIR/build-server.log        server log of the build
```

You can copy or share the folder as a whole (for example as a dataset). It works for anyone with the same model
file and settings.

## Limitations

- Servers with a multimodal projector (`--mmproj`) do not use the eval cache.
- The server only loads an entry when it covers more of the prompt than the slot already has in memory. The
  regular RAM / disk prompt cache keeps working alongside it.
- Requests must use `cache_prompt` (the default).
- Temperature-0 identity holds per prompt with one request running at a time. With several slots busy at once, batches mix
  prompts, which can change the last digits of the math with or without the cache.
