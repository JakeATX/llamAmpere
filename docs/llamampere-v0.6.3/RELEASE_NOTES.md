# llamAmpere v0.6.3

## AI+ for Flash-Next: every expert in AI+

v0.6.3 can load an AI+ copy of every routed expert of Qwen3.8 Flash-Next on top of the normal model file. AI+ is our own
expert format: each expert is smaller than its standard (3XS) copy, so the graphics card holds more of them (a 24 GB card:
about 10,800-11,000 experts instead of 9,500-9,700) and fewer have to run on the CPU. The AI+ copy also drifts less from the
reference model than the standard file (lower KLD, see below).

Turn it on with one setting (the standard model file stays the same):

```bash
LLAMA_MODEL_OVERLAY_PATCH=/path/to/AIPLUS_PATCH_FILE.gguf scripts/flash-next-server.sh -m MODEL
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
| RTX 3090 Ti limited to 12 GB | coding | 65.0 -> 81.6 (+25.5%, p = .011) | 242.8 -> 272.6 (+12.3%, p = .001) |
| RTX 3090 Ti limited to 12 GB | agentic | 65.7 -> 77.7 (+18.4%, p = .033) | 240.2 -> 268.3 (+11.7%, p = .005) |

The 12 GB rows kept the CPU expert workers off core 0 (`GGML_MOE_CPU_SPLIT_CORE0=0`) for both models; that is not the
default in this release.

Earlier AI+ runs with the floating-point CPU path on the RTX 3090 also wrote faster on coding (+5.3%, p < .001) and RAG
(+6.0%, p = .017). Not measured yet with the 8-bit CPU path: RTX 3090 coding and RAG, RTX 3090 Ti agentic.

### Quality

KLD (how far the word probabilities drift from an 8-bit reference; lower is better), 2,286 positions, RTX 3090:
standard 0.1744 / 0.1753 (two runs), AI+ 0.1537 (12% lower, p = .0007).

### 12 GB cards: fast 100K prompt with host carry (opt-in)

On a 12 GB card the prompt used to fall back to small 512-token pieces, because the in-between results of the
layer-by-layer prompt pass did not fit on the card. Host carry keeps those results in pinned system memory (about
3.9 GiB for a 100K prompt) and copies them to the card one layer at a time, so the fast one-pass prompt works again.
It is off by default in this release; turn it on with:

```bash
LLAMA_B1_HOST_CARRY=1 LLAMA_B1_HOST_CARRY_ASYNC=1 scripts/flash-next-server.sh -m MODEL ...
```

`LLAMA_B1_HOST_CARRY_ASYNC=1` overlaps the copies with compute. When two expert layers do not fit next to the carry
slots, it falls back to one slot by itself (`LLAMA_B1_HOST_CARRY_ONE_SLOT=0` disables the fallback). Output is the same
bytes as without it (checked: same greedy text, prompt-cache save/restore OK).

If the first layer-by-layer plan does not fit after the model loads, the prompt now re-plans with less card memory
(async carry, then one slot) instead of retrying the same plan and dropping to 512-token pieces.
`LLAMA_B1_REPLAN_SHRINK=0` restores the old behavior. Measured on the RTX 3090 limited to about 11 GB, AI+, host carry
and async on, 3 seeds: prompt 258 -> 2,595 tok/s (p = .0001), writing 87.9 -> 86.9 tok/s (p = .48, no change). That
test build also reused the pinned carry memory across the re-plan, which this release does not, so the re-planned
prompt here may be slightly slower than that number.

Measured on an RTX 3090 with the card limited to about 11 GB for the server, 100K coding prompt, 5,120 tokens written,
temperature 1.0, one program build, runs interleaved:

| Model | Prompt tok/s, off -> host carry -> host carry + async | Writing tok/s, off -> on |
|---|---|---|
| AI+ | 269-273 -> 1,974-1,991 -> 2,510-2,545 (4 runs each) | 88.0-88.9 -> 87.8-90.0 (async) |
| standard | 242 -> 1,832 (one-slot fallback, 2 runs each) | 68.5-68.6 -> 67.0-68.0 (not enough runs to call) |

## Other changes

- **Faster 100K prompt with the launcher.** The v0.6.2 launcher forced a slow page-fault prefetch for the per-layer
  embedding table, so the prompt waited about 15 s on disk reads. The launcher now leaves this to the engine, which reads
  the table ahead with 4 threads (on by default, skipped when under 8 GiB of RAM is free). RTX 3090 Ti, AI+, 100K coding
  prompt, 4 interleaved pairs: launcher prompt 2,083 -> 3,023 tok/s (+45%, p < .0001), writing unchanged (132.9 vs 134.0,
  p = .49). Against no prefetch at all, the new default is +2.8% on the prompt (p = .004). Output is unchanged.
  `LLAMA_PLE_PREFETCH=0` turns it off.
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

- With an AI+ file, the same prompt can give a slightly different answer from run to run. The engine copies experts to the
  card in the background, and timing decides whether a given expert is computed on the card or on the CPU. Both use the
  same weights; only the order of additions differs. This is expected, not a bug. `LLAMA_MOE_BANK_DET=1` narrows it.
- `--fit off` is no longer needed for full page read-ahead.
