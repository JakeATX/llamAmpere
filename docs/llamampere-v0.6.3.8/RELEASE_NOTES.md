# llamAmpere v0.6.3.8

The DFlash2 drafter on dense Swift 1.5 27B now writes about 3.4 times faster. A drafter is a small helper model that guesses several words ahead; the big model checks every guess, so the text you get comes from the big model either way, only faster.

## The fix

In v0.6.3.7 and earlier, one mixing step inside the DFlash2 drafter (the attention output convolution) ran twice in every layer instead of once. That spoiled the drafter's guesses, so the big model rejected almost all of them. The step now runs once, before the attention output norm, as in upstream llama.cpp. Five lines removed in `src/models/dflash.cpp`.

Only DFlash2 drafters (`--spec-type draft-dflash` with a DFlash2 drafter file) are affected. The built-in MTP drafter, Flash-Next and the models themselves use different code and are unchanged.

## Measured

RTX 3090 Ti at 350 W. Dense Swift 1.5 27B IQ4_XS-M with a Qwen3.8-27B DFlash2 Q4_K_M drafter, 7 guesses per round (`--spec-draft-n-max 7 --spec-draft-p-min 0`), 100K-token coding prompt, 5,120 tokens written, v0.6.3.7 and v0.6.3.8 run alternately (old, new, old, new):

| sampling | v0.6.3.7 tokens/s | v0.6.3.8 tokens/s | guesses accepted |
|---|---|---|---|
| temperature 1.0 | 27.26, 27.27 | 92.69, 93.19 | 2.0% -> 42.0% |
| greedy | 26.84, 27.03 | 102.75, 103.93 | 2.0% -> 48.6% |

- Greedy output: all 5,120 tokens identical between v0.6.3.7 and v0.6.3.8 (4 of 4 runs).
- Card memory peak: 20,821 MiB.
- Flash-Next with its default MTP drafter loads and writes 5,120 tokens on both versions, within the same card limits.

Validation: Linux CPU suites, the README-flags AVX2 check, example builds (lookup, speculative, speculative-simple), MinGW/Wine and tensor-split checks all passed.

Source-only release.
