# llamAmpere v0.7.1 (draft, not tagged)

The 8 GB release. `--preset 8dense` now runs the 0.7.1 8 GB model file as it is, and two GPU kernel speedups are on
by default.

Model file: TBD (AI2M10-iw or its healed version from HEALM10; the file name, size and quality numbers go here).

## What changed

- **8dense needs three files in one folder:** the model file, the LSA map (`lsa-qwen38-27b-d2048.gguf`) and the small
  MTP drafter `mtp-dm.gguf` (127,489,024 bytes, sha256 `4a8e9a5cfb6b1395088bd88b26400b504e47a10fad5a109e4cbee3276f9d5cae`;
  download link TBD). The preset loads `mtp-dm.gguf` in place of the model file's own MTP layer. It no longer looks for
  `head-ait2.gguf` or `head-exact-f16.gguf`. If `mtp-dm.gguf` is missing, the server stops at start and names the
  path. In v0.7.0.1, `--preset 8dense` with this file stopped with "failed to load overlay patch".
- **The 8dense default context is 204,800 tokens.** With `mtp-dm.gguf` the peak on an RTX 3070 is 7,607 MiB, under the
  7,692 MiB headless line (8,192 minus 500). To use the model file's own MTP layer instead, set
  `LLAMA_MODEL_OVERLAY_PATCH=` (empty) and `-c 114688`: that layer peaks at 7,819 MiB at 204,800 (over the line) and
  7,663 MiB at 114,688. `-c` still sets any context you want. 8dense stays headless-only: `8dense headed` prints a
  note and uses the same settings.
- **mb-MTP is on for 8dense**, as on the other three presets. `--no-spec-mb-mtp` turns it off.
- **Faster prompt reading on the small-card weight types** (AIF175_U2, AIF25_U2, AIF25_U2_512). These ran the
  one-token kernel even for prompt batches, so each weight was decoded again for every 8 tokens. They now use the
  batch kernel with the same arithmetic (34 of 34 checked calls bit-identical). Off switch: `GGML_CUDA_AIF_RS_MMQ=0`.
- **Faster writing on the same types:** the kernel loads the next block of weights while it works on the current one.
  Off switch: `GGML_CUDA_AIF_MMV_PF=0`.

## Measured

Rented RTX 3070 8 GB (PCIe 4 x16, 270 W, Ryzen 9 3950X), AI2M10-iw test file, `--preset 8dense`, chat prompt,
temperature 1, 5,120-token replies, 6 seeds, builds alternated (v0.7.0.1 run with the same settings: no sidecars, mb-MTP on):

| Check | v0.7.0.1 | v0.7.1 |
|---|---|---|
| Bare `--preset 8dense` start with this file | stops: "failed to load overlay patch" | starts (own MTP layer at 114,688; with mtp-dm.gguf at 204,800) |
| Peak card memory, 100K and 109K prompts (line 7,692) | 7,663 MiB | 7,663 MiB |
| Writing speed, 99,984-token prompt | 39.30 tokens/s | 41.35 tokens/s (+5.2%, every seed faster) |
| Writing speed, 109,288-token prompt | 37.16 tokens/s | 36.32 tokens/s (within seed spread, see below) |
| Prompt reading, 99,984 / 109,288 tokens | 112 / 127 tokens/s | 121 / 144 tokens/s |

At the 109,288-token prompt the replies stay in thinking mode and the draft hit rate swings a lot from run to run: the
same build and seed gives different text on a second run, and single seeds range from 33 to 42 tokens/s. The earlier
kernel test with fixed text (same card type, 199,495-token prompt) measured +5.5% on every seed.

24 GB check on an RTX 3090 (`--preset 24dense`, 99,984-token prompt, 6 seeds alternated): v0.7.0.1 99.39, v0.7.1 99.23
tokens/s (-0.16%, noise; 5 of 6 replies identical), same 19.5 GiB peak.

Context ladder for this file on the 3070 (new build, one seed, prompt about 5,400 tokens under the context):

| Context | Peak MiB | Writing tokens/s (prompt) |
|---|---|---|
| 65,536 | 7,629 | 43.4 (60,136) |
| 98,304 | 7,663 | 39.1 (92,904) |
| 114,688 | 7,663 | 36.3 to 41.4 (see above) |
| 131,072 | 7,697 (over) | 38.5 (125,661) |
| 204,800 | 7,819 (over) | 37.0 (199,387) |

The rows above, and the v0.7.0.1 vs v0.7.1 table, use the file's own MTP layer. The 0.7.1 default, `mtp-dm.gguf` at
204,800 tokens, same 3070, new build, mb-MTP on:

| Prompt | Peak MiB (line 7,692) | Writing tokens/s |
|---|---|---|
| 100,000 tokens, 6 seeds | 7,607 | 41.85 |
| 199,400 tokens, 3 seeds | 7,607 | 36.53 |

The 8 GB card still misses the 40 tokens/s writing goal at deep prompts (36.53 at 199,400).

Source-only release.
