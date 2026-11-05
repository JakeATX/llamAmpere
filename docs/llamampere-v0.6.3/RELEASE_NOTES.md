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

## Other changes

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
