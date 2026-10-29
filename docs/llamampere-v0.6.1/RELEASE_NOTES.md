# llamAmpere v0.6.1

v0.6.1 integrates Flash-Next prompt kernels, router and workspace memory savings, and two correctness fixes.

## Flash-Next integration measurements

On one RTX 3090 Ti at 350 W, three adjacent v0.6/candidate pairs used a 100,000-token coding prompt, 5,120 output tokens, temperature 1.0, and seed 23. Model pages were evicted before each arm. CPU expert splitting was on; mode-biased MTP and the shared drafter cache were off. The candidate used a 928 MiB expert-cache reserve, host-backed raw indexer history, the BF16 router, and paired prompt matmuls.

- Writing: v0.6 averaged **109.65 tokens/s**; the candidate averaged **113.15 tokens/s**, an observed **3.2%** increase. The three control/candidate pairs were 107.57/112.82, 111.06/112.73, and 110.32/113.89 tokens/s.
- Prompt processing: v0.6 averaged **2,671.61 tokens/s**; the candidate averaged **2,751.07 tokens/s**, an observed **3.0%** increase. The pairs were 2,613.72/2,716.78, 2,653.06/2,773.00, and 2,748.06/2,763.44 tokens/s.
- Approximate prompt major page faults, which count disk-backed pages brought into memory: 3,349/2,845, 2,830/2,830, and 2,830/2,830. These use one-second samples bracketing prompt processing.
- Device-route hit rates were 91.66%/92.25%, 91.85%/92.95%, and 91.07%/92.54%. A lower hit rate alone does not invalidate a run.
- All six arms completed without repetition loops or detected CPU contention. Candidate whole-card memory peaked at **23,341 MiB**, under the 23,552 MiB limit. The recorded census was **10,026 complete experts in layers 0-47**, versus 9,720 for the control; expert count is descriptive, not a release gate.

The candidate was faster in all three pairs. With only three pairs from one seed, the exact two-sided paired test gives p=0.25 for both speeds. These observations do not establish a statistically significant speedup or equivalence. The release comparison passed the declared no-significant-slowdown criterion and was accepted by the release coordinator. These numbers apply to this coding fixture and configuration, not every workload or another card.

## Changes and checks

- BF16 router storage is on by default when the stored values can be represented exactly. The router retains the F32 reduction order.
- Raw indexer history uses host memory by default when compact indexer storage is enabled. `LLAMA_QSA_RAW_HOST=0` keeps that history on its original backend. Flash-Next's default expert-cache reserve is 928 MiB; `GGML_CUDA_MOE_CACHE_RESERVE_MB` overrides it.
- Prompt normalization, helper-thread placement, paired matmuls, and the shared-expert combination kernel are integrated. They leave more CPU time and card memory for useful work.
- Dense BF16 prompt math stays opt-in: `GGML_CUDA_DENSE_TC=1` forces it on and `=-1` selects automatic detection. It can change temperature-zero text when a restored prompt is processed with a different batch shape, so it is off by default.
- Prompt read-ahead stays off by default to avoid page-cache pressure on 64 GB hosts. `LLAMA_PLE_PREFETCH=N` opts in with N readers.
- The changed CUDA operators passed all 560 exercised cases. The measured reserve-928 configuration produced four byte-identical temperature-zero outputs across the control and candidate, including save/erase/restore. A three-prompt matching-prefix probability check passed its declared limits; it does not cover the full continuation after text diverges.
- The 12 GB command with `LLAMA_KQ_MASK_PACKED=1` and SJ-KVaRN no longer stops at startup. The attention mask check now uses the attended cache length.
- `--gdn-replay` on Flash-Next now selects the convolution history for the pending rollback after rejected drafts.

## Build and run

Build with HTTPS support so `-hf` can download the model. This Swift 1.5 27B command is separate from the Flash-Next measurements above.

```bash
git clone -b v0.6.1 https://github.com/JakeATX/llamAmpere.git
cd llamAmpere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build-sm86 -j8 --target llama-server
./build-sm86/bin/llama-server \
  -hf jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF \
  -hff ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf \
  -c 262144 -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0
```

The server listens on http://127.0.0.1:8080. Add `--no-spec-mb-mtp` for exact sampling.

The measured Flash-Next configuration also needs our Q2S drafter and v2-r-pure expert profile; a public download command for those two exact assets is not supplied here. The [public main-model card](https://huggingface.co/taurusduan/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) lists the two IQ3_XXS shards. Substituting a different drafter or profile does not reproduce the figures above.
