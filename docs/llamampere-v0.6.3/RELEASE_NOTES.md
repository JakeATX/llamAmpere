# llamAmpere v0.6.3

## New model file: [qwen3.8-flash-next-Swift1.5-ATX-AI3](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3)

The two model shards total 80,114,445,888 bytes: 80.11 GB, or 74.61 GiB. The standard IQ3_XXS download is 75.97 GB, so this file is a larger download. The public Q2S-B MTP drafter is a separate download. The file requires v0.6.3 or newer.

## Hardware support

The new Flash-Next file supports one 24 GB RTX 30-series card, or a 12 GB card with 64 GB of system RAM. The launcher chooses a plan for the card and keeps at least 5 GiB of RAM free.

### Why exact output can differ after reloading a saved conversation

A fresh conversation reads its prompt in one layer-major pass. After a saved conversation is reloaded, the remaining prompt tokens are read in normal chunks. These paths use a different order for some math, so their numbers can differ slightly; when two words are nearly tied, the selected word can differ too. Our 24 GB check matched across all 45 outputs; the 12 GB checks varied at one near-tie word position.

On cards under 24 GB the launcher reads long prompts in 32K-token passes, about 2.3x faster prompt reading at 100K on 12 GB cards with 64 GB RAM.


### Measured writing and prompt reading on this file

RTX 3090, 350 W, ample system RAM: 100K-token prompts, temperature 1.0, 5,120 generated tokens per run. The new file uses the release-candidate engine, public Q2S-B drafter and product settings. These are observations on this file; a same-engine comparison with the standard IQ3_XXS file was not measured in this test. The actual 64 GB home fit checks below have different prompt speeds.

| Task | Timed runs | Writing tok/s | Prompt tok/s |
|---|---|---|---|
| Coding | 6 | 121.32 | 2,944.23 |
| Agentic | 3 | 124.89 | 2,931.60 |
| RAG | 6 | 124.26 | 2,878.55 |

The table uses arithmetic mean rates. All 45 timed replies across the three product arms completed 5,120 tokens and passed loop rule v3. The other two product arms used another engine; they are not a same-build control for this table. Coding and RAG were extended after an initial three-seed look.

On the actual 64 GB home machine with a 24 GB RTX 3090 Ti, the file completed all three 100K fit checks: prompt 1,105-1,933 tok/s, writing 116.28-120.56 tok/s. These are fit observations, not a paired speed result or a comparison with the RTX 3090. Both supported card profiles passed their RAM-fit checks.

### Quality of the new file

On the release-candidate engine, KLD against the Q8 teacher was **0.086541** on 2,286 positions in 18 chunks (top-1024 probability ruler; lower is better). The same-job earlier AI+ patch scored 0.155914. The older Swift IQ3_XXS read was 0.1744/0.1753, and a Swift IQ3_XXS file made with another published quantization map scored 0.118003 on the same Q8 reference. Those last two are historical cross-build comparisons, not paired release-build quality gates. KLD measures probability drift, not task accuracy. The chosen-file result is the selected weight recipe scored as overlays on the release candidate; the baked file was separately checked against that recipe.

Run it: see the
[quick start](https://github.com/JakeATX/llamAmpere/blob/v0.6.3/README.md#quick-start).

## Existing AI+ patch support

The following patch-file measurements predate the new model above. They describe a different file and configuration and must not be read as the new file's speed or quality. Engine support remains available; this release does not announce a patch download.

v0.6.3 can load an AI+ copy of every routed expert of Qwen3.8 Flash-Next on top of the normal model file. AI+ is our own
expert format: each expert is smaller than its standard (3XS) copy, so the graphics card holds more of them (a 24 GB card:
about 10,800-11,000 experts instead of 9,500-9,700) and fewer have to run on the CPU. The AI+ copy also drifts less from the
reference model than the standard file (lower KLD, see below).

Turn it on with one setting (the standard model file stays the same):

```bash
LLAMA_MODEL_OVERLAY_PATCH=/path/to/patch.gguf scripts/flash-next-server.sh -m MODEL
```

Experts that still run on the CPU use 8-bit integer math by default on all CPUs (`GGML_MOE_AI_HOST_Q8=0` selects the
older floating-point path). Without the patch file nothing changes: the standard model gives the same output as before.

### Measured speed, AI+ vs the standard model

Flash-Next, 100K-token prompt, 5,120 tokens written, temperature 1.0, 350 W, the two models run back to back with one
program build. "Writing" is what you see while the answer is generated; "prompt" is reading the 100K prompt. p is the
chance the gap is luck (two-tailed, paired); p <= .05 counts as real.

| Card | Task | Writing tok/s, standard -> AI+ | Prompt tok/s, standard -> AI+ |
|---|---|---|---|
| RTX 3090 Ti, 24 GB | coding | 113.0 -> 122.4 (+8.4%, p = .022) | 2,418 -> 2,662 (+10.1%, p = .007) |
| RTX 3090, 24 GB | agentic | 134.7 -> 150.5 (+11.7%, p < .001) | 2,745 -> 2,953 (+7.6%, p < .001) |

Earlier AI+ runs with the floating-point CPU path on the RTX 3090 also wrote faster on coding (+5.3%, p < .001) and RAG
(+6.0%, p = .017). Not measured yet with the 8-bit CPU path: RTX 3090 coding and RAG, RTX 3090 Ti agentic.

### Quality

KLD (how far the word probabilities drift from an 8-bit reference; lower is better), 2,286 positions, RTX 3090:
standard 0.1744 / 0.1753 (two runs), AI+ 0.1537 (12% lower, p = .0007).

### 12 GB cards: fast 100K prompt with host carry, on by itself for AI+ files

On a 12 GB card the prompt used to fall back to small 512-token pieces, because the in-between results of the
layer-by-layer prompt pass did not fit on the card. Host carry keeps those results in pinned system memory (about
3.9 GiB for a 100K prompt) and copies them to the card one layer at a time, so the fast one-pass prompt works again.

With an AI+ expert file, baked in or loaded as a patch, this now turns on by itself when the card has under 14 GB free.
`LLAMA_SMALL_CARD_AUTO=0` turns the automatic plan off. Host carry still leaves 5 GiB of RAM free or stays off. With the
standard model it stays opt-in, because at this prompt size a fresh run and a restored run gave different greedy text in
our check and we have not found out why:

```bash
LLAMA_B1_HOST_CARRY=1 LLAMA_B1_HOST_CARRY_ASYNC=1 scripts/flash-next-server.sh -m MODEL ...
```

`LLAMA_B1_HOST_CARRY_ASYNC=1` overlaps the copies with compute. When two expert layers do not fit next to the carry
slots, it falls back to one slot by itself (`LLAMA_B1_HOST_CARRY_ONE_SLOT=0` disables the fallback). The checks below used earlier files and builds; they do not establish exact repeatability for the new file on 12 GB cards.

If the first layer-by-layer plan does not fit after the model loads, the prompt now re-plans with less card memory
(async carry, then one slot) instead of retrying the same plan and dropping to 512-token pieces.
`LLAMA_B1_REPLAN_SHRINK=0` restores the old behavior. Measured on the RTX 3090 limited to about 11 GB, AI+, host carry
and async on, 3 seeds: prompt 258 -> 2,595 tok/s (p = .0001). That
test build also reused the pinned carry memory across the re-plan, which this release does not, so the re-planned
prompt here may be slightly slower than that number.

Measured on an RTX 3090 with the card limited to about 11 GB for the server, 100K coding prompt, 5,120 tokens written,
temperature 1.0, one program build, runs interleaved:

| Model | Prompt tok/s, off -> host carry -> host carry + async |
|---|---|
| AI+ | 269-273 -> 1,974-1,991 -> 2,510-2,545 (4 runs each) |
| standard | 242 -> 1,832 (one-slot fallback, 2 runs each) |

## Other changes

- **New AI+ expert formats are ready for the card cache.** Experts in the U2 formats (`ait2_u2`, `ait3_u2`) are read in place
  and can be cached on the card like the older formats.
- **Baked AI+ files get the full expert budget.** The launcher now plans for up to 24,576 ranked experts with every file; the
  card budget still decides how many fit, so the standard model is not changed.
- **About 4% faster 100K prompt reading on small cards with an AI+ file.** A 64-route work tile is now used when a prompt batch
  routes few rows per expert (`GGML_CUDA_AI23_ID_BN64_MAXR=0` turns it off). RTX 3090 limited to 10 GB: prompt +4.3% on runs
  without the slow page-fault class (p = .0004), writing unchanged. Counting all 32 runs the gap is not significant, because
  that slow class hit 10 of the 32 runs, 5 per arm (0.94x, p = .38). Output unchanged at 10 and 12 GB on our check prompt.
- **Faster 100K prompt with the launcher.** The v0.6.2 launcher forced a slow page-fault prefetch for the per-layer
  embedding table, so the prompt waited about 15 s on disk reads. The launcher now leaves this to the engine, which reads
  the table ahead with 4 threads (on by default, skipped when under 8 GiB of RAM is free). RTX 3090 Ti, AI+, 100K coding
  prompt, 4 interleaved pairs: launcher prompt 2,083 -> 3,023 tok/s (+45%, p < .0001), writing unchanged (132.9 vs 134.0,
  p = .49). Against no prefetch at all, the new default is +2.8% on the prompt (p = .004). Output is unchanged.
  `LLAMA_PLE_PREFETCH=0` turns it off.
- **The launcher now runs the measured setup without extra flags.** `scripts/flash-next-server.sh -m MODEL` sets the batch
  size, threads and card settings itself; anything you pass still wins. It also keeps a long prompt's temporary output
  (about 4 GiB at `-c 110592`) in a hidden file under `~/.cache/llamampere/moe-output` that is removed automatically. Set
  `LLAMA_MOE_ARENA_OUTPUT_DIR` to use another folder. With under 16 GiB free, or on a RAM disk, it stays in RAM and prompts
  are slower.
- **Steady speed on a busy desktop.** The thread that drives the graphics card no longer shares a CPU core with the CPU
  expert threads. Measured on the 3090 Ti with three busy programs on that core: 33.9 -> 24.6 ms per round; quiet desktop
  27.8 -> 24.8 ms per round (2 blocks per case, no p-value). Output is unchanged. `GGML_CPU_CARD_FENCE=0` turns it off.
- **Faster expert kernels on by default.** The expert kernels and prompt memory settings that our speed tests always used
  are now the Flash-Next defaults. Output of the standard model is unchanged (checked bit for bit). `NAME=0` turns one off;
  `LLAMA_FLASHNEXT_DEFAULTS=0` turns all of them off.
- **Page read-ahead covers the whole model file when it fits** above the 5 GiB free-RAM floor (an explicit flag still wins).
  RTX 3090, standard model: 27.72 -> 26.64 ms per draft round (-3.9%), writing 122.0 -> 124.3 tok/s, major page faults
  about 20-51K -> 2.4-3.0K.
- **Expert swaps weighted by router weight when skipping is on:** -0.96% time per round over 29 pairs (p = .026).
- **Faster CPU side:** AVX2 input quantizer for CPU-computed experts (byte-identical, about 2.9x faster per row), and the
  AVX-512 path for AI+ experts on the CPU is chosen at run time (AVX2 otherwise; bit-identical).
- **Expert list files can no longer be misread.** `LLAMA_MOE_BANK_KEEP` reads both list formats and refuses any other file
  with a message that names it.
- **`-md` with `--spec-type none` no longer crashes**; an unused draft model is skipped with a warning.
- **Retired AI+ formats are refused** at load, and the encoder will not write them.
- **A draft model with zero layers no longer crashes startup.** Two routine prompt messages moved to debug level.
- **CPU attention mask fix** (also released alone as v0.6.2.1): with `-ngl 0` or any KV-cache layer kept in system memory,
  output was wrong. Fixed.

## Notes

- The new file's 24 GB diagnostic produced identical output in all 45 replies, including 15 prompt restores. See the hardware-support section for why exact output can differ after reloading a saved conversation on 12 GB cards.
- The launcher currently passes `--fit off`; its independent cache planner still respects the 5 GiB RAM floor. Automatic `--fit on` has not been measured on the chosen file; the published launch command is the tested configuration.
