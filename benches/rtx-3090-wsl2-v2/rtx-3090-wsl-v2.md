# RTX 3090 on Windows 11 + WSL2 - Qwen3.8-27B on llamAmpere v0.7.2.1

## System info

```bash
uname -a
Linux bench-wsl 6.18.58.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC \
  Tue Oct 27 00:24:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

nvidia-smi
command not found          # not installed inside WSL; NVML is used instead

llama-server --version
version: 1.0.0-dev (build 15466, commit 857ccba7d)
built with GNU 13.3.0 for Linux x86_64
```

Engine: **llamAmpere v0.7.2.1**, build **15466**, commit **857ccba7d**, built as
`build-sm86` (Release, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`).

Host: Windows 11 + WSL2 (Ubuntu 24.04 guest, `microsoft-standard-WSL2` kernel), GPU
paravirtualised through `/dev/dxg`. Ryzen 5 3600, 73.6 GiB RAM, RTX 3090 24 576 MiB.
GPU memory is read through NVML (`libnvidia-ml.so.1`), sampled at 0.5 s.

Model: `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf` (14.4 GB).

## What this measures

- `--preset 24dense` with its default microbatch: `-b 4096`, `-ub 2048` for one
  slot (the preset raises 1 024 to 2 048 when `--parallel 1`), then the dynamic
  ladder resizes it while prefilling.
- The largest context that still loads, lowering the microbatch when the default
  does not fit.
- Prefill (pp) and generation (tg) at 16k, 64k, 128k, 160k, 200k, 250k, 300k, 350k,
  400k where the prompt fits, plus one `max_ctx` request at 98 % of the window.

No `-b` or `-ub` is passed unless a row says so.

## Method

One server start per record: load, warm-up request, prompt ladder, `max_ctx`. Every
request goes to `POST /completion` with `cache_prompt:false`, `ignore_eos:true`, a
unique salt per prompt and 512 generated tokens (64 for `max_ctx`), so the prompt
cache cannot skip prefill. Prompts are coding-style fillers; earlier closed-loop
testing of the same generator measured a 1.13 % spread between coding, RE and
agentic prompts.

VRAM gate: peak <= **23 552 MiB** for the headed preset (24 576 total minus the
1 024 MiB display reserve) and <= **24 076 MiB** for `24dense-headless` (500 MiB
reserve). A row passes when `max_ctx` completes, no OOM line appears and the peak is
inside the line. Spare = line minus peak.

Two full repetitions at 262 144 with the default microbatch. Every other row is one
start and is a fit check, not a speed comparison.

## Headed preset (1 024 MiB reserve) - context and microbatch

| context | b/ub | peak MiB | spare | verdict |
|---|---|---:|---:|---|
| 262 144 | 4096/2048 (default) | 21 528 | +2 024 | PASS |
| 327 680 | 4096/2048 (default) | 22 574 | +978 | PASS |
| 393 216 | 4096/2048 (default) | - | - | FAIL: startup fit refused |
| 393 216 | 4096/2048, `--fit off` | 24 145 | -593 | FAIL: over the line |
| 393 216 | 4096/1024 | 23 040 | +512 | PASS, 385 351-token fill |
| 432 538 | 4096/2048 (default) | - | - | FAIL: startup fit refused |
| 432 538 | 4096/512 | 23 443 | +109 | PASS, 423 272-token fill |
| 458 752 | 4096/256 | 24 137 | -585 | FAIL: over the line |

## Headless preset (500 MiB reserve) - context and microbatch

| context | b/ub | peak MiB | spare | verdict |
|---|---|---:|---:|---|
| 262 144 | 4096/2048 (default) | 21 673 | +2 403 | PASS |
| 327 680 | 4096/2048 (default) | 22 719 | +1 357 | PASS |
| 393 216 | 4096/2048 (default) | - | - | FAIL: startup fit refused |
| 393 216 | 2048/1024 | 23 212 | +864 | PASS, 385 351-token fill |
| 432 538 | 4096/2048 (default) | - | - | FAIL: startup fit refused |
| 432 538 | 2048/1024 | 23 953 | +123 | PASS, 423 144-token fill |
| 432 538 | 1024/512 | 23 626 | +450 | PASS, 423 987-token fill |
| 458 752 | 4096/2048 (default) | - | - | FAIL: startup fit refused |
| 458 752 | 2048/1024 | - | - | FAIL: startup fit refused |
| 458 752 | 1024/512 | 24 010 | +66 | PASS (nominal, see caveats) |

## Why the default microbatch stops at 327 680

Above 327 680 the startup fit cannot prove the margin, so the server exits before a
single request is served:

```
preset line 24075 MiB = total 24575 - margin 500 on CUDA0; free target 0 MiB
other programs already use 1151 MiB of CUDA0, at or over the preset margin of 500 MiB
card preset fit could not prove the memory margin; adjust -c/-ub or use -fit off
```

The 1 151 MiB is llama's own CUDA context plus the desktop, counted before any
allocation. It does not move when the card is empty, so it is not a leftover from a
previous run: the same figure appears on a fresh start with 556 MiB reported used.

Two things follow.

1. The default `-ub 2048` really is too large here. At 393 216 with the fit disabled
   the run loads, but peaks at 24 145 MiB, over both lines. Lowering the microbatch
   is what makes the context possible, not a larger card budget.
2. Once the microbatch is lowered, the contexts load and complete a full-window
   request: 385 k at 393 216, 423 k at 432 538, 449 k at 458 752.

The cost of the lower microbatch is visible in generation at the full window: 42.2
tok/s at 432 538 with `-ub 1024` against 41.0 with `-ub 512`, and 41.0 against 38.4
at 458 752. Prefill at long prompts is unchanged within noise (222 against 219 at
432 538), because the context, not the batch, dominates there.

## Prefill, passing configurations (tok/s)

| context | preset | b/ub | 16k | 64k | 128k | 160k | 200k | 250k | 300k | 350k | 400k | max_ctx |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 262 144 | headed | 4096/2048 | 1270 | 1405 | 1110 | 1001 | 877 | 543 | - | - | - | 541 |
| 327 680 | headed | 4096/2048 | 1509 | 1354 | 958 | 795 | 669 | 461 | 360 | - | - | 340 |
| 393 216 | headed | 4096/1024 | 1359 | 1121 | 674 | 534 | 429 | 350 | 296 | 257 | - | 241 |
| 432 538 | headed | 4096/512 | 1364 | 1121 | 675 | 533 | 429 | 350 | 296 | 258 | 228 | 221 |
| 262 144 | headless | 4096/2048 | 1262 | 1389 | 1082 | 973 | 849 | 527 | - | - | - | 527 |
| 327 680 | headless | 4096/2048 | 1464 | 1328 | 932 | 775 | 654 | 452 | 353 | - | - | 334 |
| 393 216 | headless | 2048/1024 | 1395 | 1125 | 670 | 529 | 426 | 347 | 294 | 256 | - | 241 |
| 432 538 | headless | 2048/1024 | 1350 | 1096 | 658 | 521 | 421 | 343 | 291 | 253 | 225 | 222 |
| 432 538 | headless | 1024/512 | 1338 | 1098 | 661 | - | 422 | - | 291 | - | 225 | 219 |
| 458 752 | headless | 1024/512 | 1359 | 1109 | 666 | - | 428 | - | 296 | - | 228 | 210 |

## Generation, passing configurations (tok/s)

| context | preset | b/ub | 16k | 64k | 128k | 160k | 200k | 250k | 300k | 350k | 400k | max_ctx |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 262 144 | headed | 4096/2048 | 141 | 140 | 108 | 106 | 110 | 94 | - | - | - | 68 |
| 327 680 | headed | 4096/2048 | 147 | 112 | 114 | 102 | 105 | 92 | 79 | - | - | 68 |
| 393 216 | headed | 4096/1024 | 123 | 118 | 104 | 87 | 75 | 70 | 81 | 75 | - | 45 |
| 432 538 | headed | 4096/512 | 105 | 102 | 86 | 101 | 85 | 80 | 68 | 69 | 68 | 38 |
| 262 144 | headless | 4096/2048 | 153 | 138 | 97 | 99 | 90 | 84 | - | - | - | 71 |
| 327 680 | headless | 4096/2048 | 160 | 108 | 121 | 96 | 95 | 83 | 82 | - | - | 70 |
| 393 216 | headless | 2048/1024 | 114 | 128 | 109 | 95 | 85 | 71 | 64 | 77 | - | 41 |
| 432 538 | headless | 2048/1024 | 118 | 107 | 98 | 95 | 92 | 78 | 69 | 82 | 65 | 42 |
| 432 538 | headless | 1024/512 | 115 | 120 | 93 | - | 86 | - | 70 | - | 72 | 41 |
| 458 752 | headless | 1024/512 | 115 | 125 | 88 | - | 97 | - | 70 | - | 73 | 38 |

## Headed against headless

Both presets run the same engine and model; the headless one keeps 500 MiB of the
card instead of 1 024, so its line is 24 076 MiB instead of 23 552.

- Up to 327 680 the two are the same: the default microbatch fits both.
- From 393 216 the default microbatch fits neither.
- At the top, the extra 524 MiB is worth one rung: 458 752 completes headless with
  66 MiB to spare, while headed is 585 MiB over the line.
- Speeds at the shared contexts are within run-to-run noise; the reserve changes
  what fits, not how fast it runs.

## Caveats

- WSL2 is not a bare-metal target. The Windows host shares the card, so the
  free-memory baseline moves between runs. Margins under about 500 MiB are
  statistical, and the 458 752 headless result (66 MiB spare) is nominal.
- These are informational measurements of what is possible inside WSL, not a strict
  benchmark. Windows applications were closed for the headless runs, which is why
  24dense-headless reached every context in the table. The headed rows were taken
  with the desktop attached to the same GPU.
- WSL released memory inconsistently between runs. Two starts had to be repeated
  after the previous server was reloaded; those repeats are not listed separately,
  one record per configuration is kept.
- The 432 538 headed run needed the fit disabled to get a ladder; the row that
  reports PASS is the `-ub 512` rung. Its `max_ctx` row comes from a separate
  max-only start at the same batch.
- No other engine was measured in this set.

## Raw records

`results-v2/` holds one JSON per server start: preset, arguments, chosen context,
effective `batch=/ubatch=`, peak VRAM, load and probe seconds, and every ladder row
with prompt tokens, pp and tg.
