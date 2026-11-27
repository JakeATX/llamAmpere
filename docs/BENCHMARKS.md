# llamAmpere benchmarks

Every speed and quality number we publish for the current files, with the card, power limit and prompt it was measured
on. Numbers come from the release notes of the version named in each section; nothing here is estimated. The short
version is in the [README](../README.md).

## How we measure

- **Decode** is tokens per second written in a chat reply after a prompt of the stated length. **Prefill** (prompt
  reading) is tokens per second while the prompt is read.
- Chat template, temperature 1, replies of 5,120 tokens, never greedy sampling. A task that asks for a long answer, so
  the reply is not cut short.
- Two builds are compared on the same card in A B B A order, 6 seeds per build unless a row says otherwise.
- Every row names its card and power limit. Rows measured on an RTX 3090 Ti at 350 W are shown 2% lower when they are
  labelled "RTX 3090"; rows labelled "RTX 3090 Ti" are as measured.
- Small-card rows marked "limited" ran on one RTX 3090 held to that card's memory. Real 8, 10 and 12 GB cards are
  named where we used them. On small cards part of the model or context is served from system RAM, so speed depends
  on the PCIe link and RAM speed.
- llamAmpere rows use our 4-bit KV cache (SJ-KVaRN). Stock llama.cpp rows use its 8-bit KV cache and no drafter.
- **Max context** is the largest context that starts and finishes a 100K-token prompt plus a reply within 23,552 MiB
  on a 24 GB card.
- **KLD** (KL divergence) compares a file's next-token choices with the full-precision model: lower is closer.
  **Top-1** is how often both pick the same token. Test: coding, agent and document-question chats, 16,200 scored
  tokens.
- A **drafter** guesses several tokens ahead and the main model checks them in one pass. **MTP** is the drafter built
  into Qwen3.8 27B.

## Inkling 2 Mini, 24 GB, against stock llama.cpp (v0.7.5)

One RTX 3090 (measured on an RTX 3090 Ti at 350 W, shown 2% lower), `--preset 24ink` with the drafter, 262,144-token
context, one user. 3K, 31,985 and 100K-token prompts. In parentheses: v0.7.4.1 measured the same way.

| Engine and file | Size | Decode 3K | Decode 32K | Decode 100K | Prefill 100K | Max context |
|---|---:|---:|---:|---:|---:|---:|
| llamAmpere 0.7.5, `Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf` + drafter | 17.6 GB | 185.2 (168.2) | 179.0 (160.0) | 141.8 (128.2) | 2,462 (1,697) | 385,024 |
| llamAmpere 0.7.5, `Inkling-2-Mini-ATX-AI4-4.4bpw.gguf` + drafter | 14.4 GB | 188.7 (181.7) | 186.9 (171.4) | 144.9 (139.1) | 2,692 (1,954) | 614,400 |
| stock llama.cpp b12627, `UD-Q4_K_XL` | 17.6 GB | 51.3 | 48.4 | 33.3 | 249 | 278,528 |

That is 3.6x stock decode at a 3K prompt and 4.3x at 100K with the same 17.6 GB file size. Stock llama.cpp reaches
its max context only at 32-token prompt batches; with a 16-bit KV cache it reaches 147,456.

The same runs as measured on the RTX 3090 Ti at 350 W, v0.7.4.1 → v0.7.5, 6 seeds per build:

| Model | Prompt | Decode | Prefill |
|---|---:|---:|---:|
| 5.4 bpw + drafter | 100K | 130.8 → 144.7 tok/s (+10.6%) | 1,732 → 2,512 tok/s (+45%) |
| 5.4 bpw + drafter | 32K | 163.3 → 182.6 tok/s (+11.8%) | 2,094 → 2,758 tok/s (+32%) |
| 5.4 bpw + drafter | 3K | 171.6 → 189.0 tok/s (+10.1%) | 2,028 → 2,488 tok/s (+23%) |
| 4.4 bpw + drafter | 100K | 141.9 → 147.9 tok/s (+4.2%) | 1,994 → 2,747 tok/s (+38%) |
| 4.4 bpw + drafter | 32K | 174.9 → 190.7 tok/s (+9.0%) | 2,494 → 3,065 tok/s (+23%) |
| 4.4 bpw + drafter | 3K | 185.4 → 192.6 tok/s (+3.9%) | 2,445 → 2,787 tok/s (+14%) |

## Inkling 2 Mini, 8, 10 and 12 GB (v0.7.5)

One RTX 3090 at 390 W, PCIe 4.0 x16, limited to the card's memory, drafter off, 262,144-token context (8 GB: 212,992).
v0.7.4.1 → v0.7.5, 6 seeds per build.

| Card and preset | File | Prompt | Decode | Prefill |
|---|---|---:|---:|---:|
| 12 GB, `--preset 12ink` | `Inkling-2-Mini-ATX-12GB-4.2bpw.gguf` (13.6 GB) | 100K | 57.8 → 60.0 tok/s | 622 → 638 tok/s |
| 12 GB, `--preset 12ink` | same | 3K | 61.6 → 62.2 tok/s | 403 → 403 tok/s |
| 10 GB, `--preset 10ink` | `Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf` (12.2 GB) | 100K | 56.3 → 58.0 tok/s | 404 → 403 tok/s |
| 10 GB, `--preset 10ink` | same | 3K | 59.5 → 60.4 tok/s | 353 → 355 tok/s |
| 8 GB, `--preset 8ink` | `Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf` (12.2 GB) | 100K | 38.2 → 38.8 tok/s | 405 → 404 tok/s |
| 8 GB, `--preset 8ink` | same | 3K | 39.1 → 39.5 tok/s | 362 → 362 tok/s |

## Inkling 2 Mini quality

KLD and top-1 against the full-precision model, 16,200 scored tokens, 4-bit KV cache (v0.7.4 and v0.7.5 notes).

| File | Card | KLD | Top-1 |
|---|---|---:|---:|
| `Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf` | 24 GB | 0.0032 | 98.1% |
| `Inkling-2-Mini-ATX-AI4-4.4bpw.gguf` | 24 GB | 0.0060 | 97.1% |
| `Inkling-2-Mini-ATX-12GB-4.2bpw.gguf` | 12 GB | 0.0080 | 96.5% |
| `Inkling-2-Mini-ATX-8GB-10GB-3.7bpw.gguf` | 8 / 10 GB | 0.011 | 96.1% |
| `UD-Q4_K_XL` on stock llama.cpp, for comparison | 24 GB | 0.0032 | 97.9% |

## Qwen3.8 27B, 24 GB (`--preset 24dense`)

`qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf` (14.4 GB) with its built-in MTP drafter, one user.

v0.7.5, RTX 3090 Ti at 350 W, 110,592-token context, prompt already cached, v0.7.4.1 → v0.7.5, 6 seeds per build:

| Prompt | Decode |
|---:|---:|
| 98,766 tokens (coding) | 126.2 → 126.2 tok/s |
| 95,678 tokens (agent) | 127.5 → 127.4 tok/s |
| 103,390 tokens (document questions) | 113.6 → 113.6 tok/s |

Both builds wrote the same replies token for token. Two agent seeds copy long passages from the prompt and run at
about 187 tok/s; without them the agent row is 110.0 tok/s.

v0.7.1, RTX 3090 at 350 W, 6 seeds per build unless noted:

| Check | Result |
|---|---:|
| Decode, 8,000-token prompt | 138.84 tok/s |
| Decode, 98,766-token prompt | 115.62 tok/s |
| Decode, 256,696-token prompt (full 262,144 context, 1 seed) | 89.56 tok/s, peak 21,440 MiB |
| Prefill, 98,766-token prompt read cold | 1,350 tok/s |
| Three users at once (99,984 + 32,766 + 32,766-token prompts, `-np 3 -c 352256 --slot-ctx 262144,0,0`), all together, 21 runs | 194.9 tok/s, peak 22,452 MiB |

Quality (model card): 10.4% lower KLD than the IQ4_XS-M file of the same model (ratio 0.896, 95% range 0.860 to 0.932,
32,400 positions) in a file 1.2 GB smaller. There is no published stock llama.cpp speed row for this file; it does not
load in stock llama.cpp.

## Qwen3.8 27B, 8, 10 and 12 GB

One user. Speeds are on real cards where a card is named.

| Card and preset | File | Card, power limit | Prompt | Decode | Context | Source |
|---|---|---|---:|---:|---:|---|
| 12 GB, `--preset 12dense headless` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf` (11.24 GB) | RTX 3080 Ti 12 GB, power limit not published | 100K | 70.8 tok/s | 262,144, peak 11,642 MiB | v0.7 |
| 12 GB, `--preset 12dense` | same | RTX 3090 limited to 12 GB, power limit not published | 100K | 70.30 → 74.49 tok/s (v0.7.4 change) | - | v0.7.4 |
| 10 GB, `--preset 10dense headless` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2.gguf` (9.14 GB) | RTX 3080 10 GB, 320 W | 98,842 | 60.76 tok/s (6 seeds) | 212,992, peak 9,455 MiB | v0.7.0.1 |
| 10 GB, `--preset 10dense` (display attached) | same | RTX 3080 10 GB, 320 W | 98,842 | 56.6 to 64.8 tok/s | 180,224, peak 9,131 MiB | v0.7.0.1 |
| 8 GB, `--preset 8dense` | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2-8GB.gguf` (8.39 GB) | RTX 3070 8 GB, 220 W, PCIe 4.0 x16 | 99,984 | 41.41 tok/s (6 seeds) | 204,800, peak 7,607 MiB | v0.7.1 |
| 8 GB, `--preset 8dense` | same | RTX 3070 8 GB, 220 W, PCIe 4.0 x16 | 199,387 | 37.24 tok/s (3 seeds) | 204,800 | v0.7.1 |
| 8 GB, `--preset 8dense` | same | RTX 3070 8 GB, 220 W, PCIe 3.0 x16 host | 99,984 | 35.70 tok/s (6 seeds) | 204,800, peak 7,657 MiB | v0.7.2.1 |
| 8 GB, `--preset 8dense` | same | RTX 3090 limited to 8 GB, power limit not published | 100K | 56.87 → 60.22 tok/s (v0.7.4 change) | - | v0.7.4 |

Notes: the v0.7.1 8 GB rows used the file before its final tuning pass (same size and layout); the released file
measured 42.66 tok/s at the 99,984-token prompt on the same card (1 seed). 8 GB prefill at 99,984 tokens was 137 tok/s
in v0.7.1; v0.7.2.1 raised the 8 GB prompt batch from 256 to 512 tokens. The 12 GB and 10 GB presets have not been
re-measured on real 12 and 10 GB cards since v0.7 and v0.7.0.1.

Quality against the full-precision model (model cards, 16,200 positions): 12 GB file KLD 0.0270, top-1 95.98%;
10 GB file KLD 0.0594, top-1 94.0%; 8 GB file KLD 0.0868, top-1 92.52%.

## Images (v0.7.4.1 and v0.7.5.1)

Inkling 2 Mini, one RTX 3090 at 350 W, PCIe 4.0 x16, limited to each card's memory less 1 GB for a desktop. A
100K-token chat turn, then a question about a chart image in the same conversation (v0.7.4.1):

| Card and preset | Text decode at 100K, no vision / vision loaded | Image question (whole turn) | Answer |
|---|---:|---:|---|
| 24 GB, `--preset 24ink` + drafter | 124.2 / 125.4 tok/s | 1.9 s | correct |
| 12 GB, `--preset 12ink` | 54.8 / 54.5 tok/s | 9.2 s | correct |
| 8 GB, `--preset 8ink` | 37.4 / 37.3 tok/s | 10.1 s | correct |

v0.7.5.1: `--preset 24ink`, 5.4 bpw file, drafter and `--mmproj mmproj-Inkling-2-Mini-F16.gguf`, 262,144-token
context, RTX 3090 at 300 W: 6 of 6 image questions correct, peak 22,271 MiB.

## Several requests at once

Inkling 2 Mini 5.4 bpw with the drafter, rented RTX 3090 at 270 to 280 W, total tokens over wall time with a warm
prompt cache, 6 seeds (v0.7.4):

| Users x prompt each | Total decode |
|---|---:|
| 4 x 16K | 175.3 tok/s (slowest user 42.8 tok/s) |
| 2 x 100K | 94.6 tok/s |

v0.7.5.1 fix check (`Inkling-2-Mini-ATX-AI4-4.4bpw.gguf`, 3 parallel requests, 3K-token prompts then 28.5K-token
prompts, chat template, temperature 1, 4-bit KV cache):

| Build | Card | Power limit | Failed requests |
|---|---|---:|---:|
| v0.7.5 | RTX 3090 Ti | 350 W | 3 of 3 at 28.5K (HTTP 500) |
| v0.7.5 | RTX 3090 | 300 W | 3 of 3 at 28.5K (HTTP 500) |
| v0.7.5 + `LLAMA_NK_DEV_MASK=0` | RTX 3090 | 300 W | 0 |
| v0.7.5.1 | RTX 3090 Ti | 350 W | 0 |

v0.7.5.1 also ran clean with 2 parallel requests (3K-token prompts) and with 4 (3K and 28.5K-token prompts), same
model, RTX 3090 Ti at 350 W.

## v0.7.5.1 against v0.7.5, one request

`--preset 24ink`, `Inkling-2-Mini-ATX-NVFP4Q8-5.4bpw.gguf` with the drafter, 262,144-token context, 100K-token prompt,
5,120-token replies, RTX 3090 Ti at 350 W, 6 seeds per build in A B B A order: decode 144.81 → 144.89 tok/s (+0.06%),
prefill 2,522 → 2,526 tok/s. The same tokens were written at 3K and 28.5K-token prompts (4.4 bpw file, 1 seed).

Older measurements are in each version's release notes: see [CHANGELOG.md](CHANGELOG.md).
