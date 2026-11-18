# llamAmpere v0.7.0.1

Fix: the 10 GB headed preset (`--preset 10dense`) failed to start on RTX 3080 cards.

The preset kept its 1,024 MiB reserve from the memory the card reported as free, so memory already in use at start was
counted twice and the fit check refused to load. The reserve now comes off the card's total memory: on a 10 GB card the
server stays at or under 9,216 MiB (total minus 1,024). Memory already in use, such as a desktop, counts toward that
reserve. An explicit `--fit-target` keeps its old meaning.

Measured on a rented RTX 3080 10 GB (320 W), public AI2 model, chat prompts, temperature 1:

| Check | v0.7 | v0.7.0.1 |
|---|---|---|
| `--preset 10dense` (headed) start | refuses to start | starts at 180,224 tokens, 3 of 3 cold starts |
| Headed peak card memory | - | 9,131 MiB (line 9,216) |
| Headed writing speed, 98,842-token prompt | - | 56.6 to 64.8 tokens/s |
| `--preset 10dense headless` writing speed, 98,842-token prompt (6 seeds, alternating builds) | 60.68 tokens/s | 60.76 tokens/s |
| Headless peak card memory | 9,455 MiB | 9,455 MiB (line 9,740) |

The same fix lets the headed 12 GB preset keep its 196,608-token context instead of shrinking it. The 24 GB presets and
all headless presets start with the same context and memory as v0.7. Source-only release.
