## System info

```bash
uname -a
Linux bench-wsl 6.18.58.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Tue Oct 27 00:24:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.

cat /proc/version
Linux version 6.18.58.1-microsoft-standard-WSL2 (gcc (GCC) 13.2.0, GNU ld (GNU Binutils) 2.41) #1 SMP PREEMPT_DYNAMIC Tue Oct 27 00:24:11 UTC 2026

nvidia-smi
command not found                       # not installed inside WSL; NVML is used instead
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24575 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24575 MiB
```

```
llama-server --version
version: 0.6.0-dev (build 13457, commit a5d8684ce)
built with GNU 13.3.0 for Linux x86_64
```

Engine: **llamAmpere** - the CUDA/sm86 fork of llama.cpp with SJ-KVaRN KV cache and MTP
speculative decoding - `0.6.0-dev`, build **13457**, commit **a5d8684ce**, built with GNU 13.3.0.

Running on **Windows 11 + WSL2** (Ubuntu 24.04 guest, kernel `microsoft-standard-WSL2`), with GPU
paravirtualisation through `/dev/dxg`. The GPU is shared with the Windows host compositor, which
moves the VRAM baseline by ±150-900 MiB between runs - the peak-VRAM numbers below are therefore
per-run measurements, not constants.

Measured with `llama-server` from the **llamAmpere** fork, using its own features
(SJ-KVaRN `sjkvarn4` KV cache and MTP adaptive speculative decoding). The upstream
`llama-bench` / `llama-batched-bench` tools in this tree do not expose those options, so each
row below is one full server start driven through `POST /completion` with
`cache_prompt:false`, `ignore_eos:true` and a unique salted prompt - the prefill times are
genuinely computed, never restored from the prompt cache.

> `margin` = **23 650 - peak VRAM**: positive means the run stayed under the VRAM gate, negative means it exceeded it by that many MiB. (The raw JSON also reports `headroom_mib`, which is measured against 23 976 MiB = 24 576 total - ~600 MiB host-compositor reserve; the two numbers differ.)

`PP` = prompt tokens, `TG` = generated tokens, `T_PP`/`T_TG` = measured wall time for each phase,
`S_PP`/`S_TG` = tokens/s, `S` = end-to-end. Rows at ~98 % of the window are the `max_ctx` gate.
Two repetitions per configuration; values are the mean.

## Summary

Benchmark of **Qwen3.8-27B (ATX Swift 1.5 Uncensored, IQ4_XS)** served by **llamAmpere**
(`0.6.0-dev`, build 13457, commit a5d8684ce) on an **RTX 3090 (24 576 MiB)** for the following
questions:

* which `-b` / `-ub` pair maximises prefill (`pp`) and generation (`tg`) throughput;
* how that choice changes with the context window (262k native, then YaRN x1.5 / x1.545 / x1.65 /
  x1.75 / x1.8);
* what is the **maximum context that actually fits** in GPU memory at a 23 650 MiB peak-VRAM
  budget (24 576 total - ~600 MiB display reserve - 300 MiB safety margin).

All measurements were taken on **Windows 11 + WSL2** (Ubuntu guest, GPU paravirtualisation via
`/dev/dxg`). 116 server starts, one model load each, strictly sequential.

### Main findings

* **Best overall: `-b 4864 -ub 4864` at the native 262 144 context** - 1660 tok/s prefill at 16k
  prompts, 1441 at 64k, 1116 at 160k, 549 at 250k, at 23 486 MiB peak VRAM.
* **`-ub` is a memory gate, not a speed dial.** Past the VRAM budget prefill collapses 4-20x or
  the request returns HTTP 503. The safe ceiling shrinks with context:
  **4864 @262k -> 1408 @393k -> 1024 @405k -> 512 @432k -> 128 @458k -> <64 @471k.**
* **`-b` does not matter.** `-b 4864` vs `-b 19456` at `-ub 4864` differ by **-0.62 %**, so
  `-b = -ub` is the correct default.
* **Above ~160k of prompt the 393k tier is faster than the 262k tier** (966 vs 828 tok/s at 200k),
  because its narrower ubatch suits long prompts better than the 262k tier's `-ub 4864`.
* **There is a step change between 200k and 250k prompts** - prefill drops 31-42 % at every
  context size (393k: 966 -> 565 tok/s; 262k: 828 -> 549).
* **432 538 is the practical maximum context.** 458 752 passes an isolated gate probe but peaks
  24 039 MiB in a full run; 471 859 cannot fit at any `-ub` (23 788 MiB even at `-ub 64`).
* **Generation is not tunable by batching** - tg stays 54-120 tok/s across every setting, bound
  by the MTP draft/verify loop.
* **VRAM drifts ±150-900 MiB run-to-run** on WSL2 GPU-PV because the Windows host compositor
  shares the device, so margins below ~1 GiB are statistical rather than guarantees.

## Test matrix

* **Batch settings:** `-b 8192 -ub 2048`, `-b 4096 -ub 1024`, `-b 9216 -ub 3072`,
  `-b 7168 -ub 1536`, `-b 2048 -ub 512`, `-b 512 -ub 128`, plus a 128-grid `-ub` ceiling search
  at full context (up to `-ub 4864`).
* **Context tiers:** 262 144 native - 393 216 (YaRN x1.5) - 405 000 (x1.545) - 432 538 (x1.65) -
  458 752 (x1.75) - 471 859 (x1.8).
* **Prompt ladder:** 16k - 64k - 128k - 160k - 200k - 250k - 300k - 350k - 400k - 450k (whichever
  fit the window) plus a `max_ctx` gate at ~98 % of the window.
* **Repetitions:** 2 per winning cell.
* **Method:** every row is one full server start driven through `POST /completion` with
  `cache_prompt:false`, `ignore_eos:true` and a **unique salted prompt**, so prefill is always
  genuinely computed and never restored from the prompt cache. Peak VRAM is sampled at 0.5 s
  through NVML.

---

## jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4

Model: `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf` ([jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4))
IQ4_XS - 13.43 GiB - 27 B parameters

### 262 144 ctx - native (no YaRN) PASS

```bash
llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 262144 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4864 -ub 4864 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fitt 600 --metrics
```

main: n_ctx = 262144, n_batch = 4864, n_ubatch = 4864, flash_attn = 1, n_gpu_layers = 99,
n_threads = 6, n_threads_batch = 6, kv = sjkvarn4/sjkvarn4, spec = draft-mtp-adaptive, parallel = 1

|    PP |  TG |  N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s |   T s |  S t/s |
|-------|-----|-------|--------|----------|--------|----------|-------|--------|
| 16476 | 512 | 16988 |  9.927 |  **1659.89** |  4.508 |   **113.44** | 14.435 | 1176.87 |
| 66066 | 512 | 66578 | 45.861 |  1440.58 |  4.935 |   103.60 | 50.795 | 1310.70 |
| 130992 | 512 | 131504 | 108.433 |  1208.05 |  5.481 |    93.28 | 113.913 | 1154.42 |
| 163729 | 512 | 164241 | 146.696 |  1116.12 |  5.961 |    85.87 | 152.657 | 1075.88 |
| 204720 | 512 | 205232 | 247.179 |   828.22 |  6.459 |    79.34 | 253.638 | 809.15 |
| 255829 | 512 | 256341 | 466.243 |   548.70 |  6.918 |    73.91 | 473.160 | 541.76 |
| 249800 |  64 | 249864 | 439.112 |   568.88 |  1.082 |    58.41 | 440.193 | 567.62 |

- `.sh` default `-b 4096 -ub 1024`, `-ub 2048`, `-ub 3072`, `-ub 4096`, `-ub 4608`, `-ub 4864`,
  `-ub 4992`, `-ub 5120`, `-ub 6144`, `-ub 8192` were all measured at 98 % fill:

| `-b` / `-ub` | peak VRAM | margin vs 23650 | pp @~250k | verdict |
|---|---:|---:|---:|---|
| 8192 / 2048 | 22 444 | +1 206 | 543 | PASS ok, slower at depth |
| 12288 / 3072 | 22 036 | +1 614 | 561 | PASS ok |
| 16384 / 4096 | 22 856 | +794 | 545 | PASS ok |
| 18432 / 4608 | 23 276 | +374 | 534 | PASS ok |
| **19456 / 4864** | **23 486** | **+164** | **550** | **PASS recommended** |
| 19968 / 4992 | 23 590 | +60 | - (HTTP 503) | FAIL fail |
| 20480 / 5120 | 23 698 | -48 | 548 | FAIL fail |
| 24576 / 6144 | 24 498 | -848 | 532 | FAIL fail |
| 32768 / 8192 | 24 496 | -846 | 78 | FAIL fail |

build: 13457 (a5d8684ce)

### 393 216 ctx - YaRN x1.5 PASS

```bash
llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 393216 \
  --rope-scaling yarn --rope-scale 1.5 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 5632 -ub 1408 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fitt 600 --metrics
```

main: n_ctx = 393216, n_batch = 5632, n_ubatch = 1408, flash_attn = 1, n_gpu_layers = 99,
n_threads = 6, n_threads_batch = 6, kv = sjkvarn4/sjkvarn4, rope = yarn x1.5, parallel = 1

|    PP |  TG |  N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s |    T s |  S t/s |
|-------|-----|-------|--------|----------|--------|----------|--------|--------|
| 16426 | 512 | 16938 |  9.624 |  **1706.78** |  4.372 |   **116.93** | 13.996 | 1210.22 |
| 65712 | 512 | 66224 | 45.728 |  1437.01 |  4.612 |   110.80 | 50.340 | 1315.52 |
| 131005 | 512 | 131517 | 110.739 |  1183.01 |  5.695 |    89.95 | 116.433 | 1129.55 |
| 163725 | 512 | 164237 | 151.247 |  1082.51 |  6.280 |    81.37 | 157.527 | 1042.60 |
| 204667 | 512 | 205179 | 211.803 |   966.31 |  6.661 |    76.72 | 218.464 | 939.19 |
| 255819 | 512 | 256331 | 452.773 |   565.00 |  6.996 |    73.36 | 459.769 | 557.52 |
| 307010 | 512 | 307522 | 740.527 |   414.59 |  7.419 |    69.06 | 747.946 | 411.16 |
| 358142 | 512 | 358654 | 1073.892 |   333.50 |  7.530 |    67.99 | 1081.422 | 331.65 |
| 384507 |  64 | 384571 | 1263.078 |   304.42 |  1.107 |    56.97 | 1264.185 | 304.20 |

| `-b` / `-ub` | peak VRAM (2 runs) | margin | pp @~385k | verdict |
|---|---:|---:|---:|---|
| 8192 / 2048 | 24 462 | -812 | 302 | FAIL fail |
| 6144 / 1536 | 23 740 | -90 | 323 | FAIL fail (90 MiB over) |
| **5632 / 1408** | **23 612 / 23 514** | +38 ... +136 | 304 | PASS recommended (WARN borderline on the second run) |

build: 13457 (a5d8684ce)

### 405 000 ctx - YaRN x1.545 PASS

```bash
llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 405000 \
  --rope-scaling yarn --rope-scale 1.545 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 4096 -ub 1024 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fitt 600 --metrics
```

main: n_ctx = 405000, n_batch = 4096, n_ubatch = 1024, flash_attn = 1, n_gpu_layers = 99,
n_threads = 6, n_threads_batch = 6, kv = sjkvarn4/sjkvarn4, rope = yarn x1.545, parallel = 1

|    PP |  TG |  N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s |    T s |  S t/s |
|-------|-----|-------|--------|----------|--------|----------|--------|--------|
| 16400 | 512 | 16912 |  9.520 |  **1722.65** |  4.293 |   **119.36** | 13.813 | 1224.30 |
| 66183 | 512 | 66695 | 46.090 |  1435.94 |  4.696 |   109.05 | 50.786 | 1313.25 |
| 130987 | 512 | 131499 | 118.886 |  1101.80 |  5.602 |    91.22 | 124.487 | 1056.33 |
| 163762 | 512 | 164274 | 165.347 |   990.41 |  5.798 |    88.23 | 171.145 | 959.85 |
| 204701 | 512 | 205213 | 231.500 |   884.24 |  6.694 |    76.34 | 238.194 | 861.54 |
| 255796 | 512 | 256308 | 487.831 |   524.36 |  7.260 |    70.39 | 495.092 | 517.70 |
| 306982 | 512 | 307494 | 795.055 |   386.12 |  7.461 |    68.53 | 802.516 | 383.16 |
| 358105 | 512 | 358617 | 1152.633 |   310.69 |  8.011 |    64.22 | 1160.644 | 308.98 |
| 396574 |  64 | 396638 | 1453.412 |   272.86 |  1.116 |    56.44 | 1454.528 | 272.69 |

| `-b` / `-ub` | peak VRAM | margin | pp @~397k | verdict |
|---|---:|---:|---:|---|
| 4608 / 1152 | 23 608 | +42 | 296 | WARN passes, thin margin |
| **4096 / 1024** | **23 478 / 23 515** | +172 ... +135 | 273 | **PASS recommended** |

build: 13457 (a5d8684ce)

### 432 538 ctx - YaRN x1.65 PASS

```bash
llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 432538 \
  --rope-scaling yarn --rope-scale 1.65 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 2048 -ub 512 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fitt 600 --metrics
```

main: n_ctx = 432538, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99,
n_threads = 6, n_threads_batch = 6, kv = sjkvarn4/sjkvarn4, rope = yarn x1.65, parallel = 1

|    PP |  TG |  N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s |    T s |  S t/s |
|-------|-----|-------|--------|----------|--------|----------|--------|--------|
| 16471 | 512 | 16983 | 10.119 |  **1627.80** |  4.260 |   **120.12** | 14.379 | 1181.11 |
| 65512 | 512 | 66024 | 49.405 |  1326.03 |  4.856 |   105.24 | 54.261 | 1216.78 |
| 131018 | 512 | 131530 | 138.388 |   946.74 |  5.508 |    92.78 | 143.895 | 914.06 |
| 163780 | 512 | 164292 | 206.096 |   794.68 |  5.934 |    86.25 | 212.030 | 774.85 |
| 204652 | 512 | 205164 | 305.438 |   670.03 |  6.213 |    82.25 | 311.652 | 658.31 |
| 255790 | 512 | 256302 | 552.455 |   463.00 |  6.630 |    77.10 | 559.085 | 458.43 |
| 306943 | 512 | 307455 | 854.246 |   359.31 |  7.389 |    69.27 | 861.635 | 356.83 |
| 358014 | 512 | 358526 | 1207.340 |   296.53 |  7.534 |    68.01 | 1214.874 | 295.11 |
| 409306 | 512 | 409818 | 1606.198 |   254.83 |  8.571 |    59.64 | 1614.769 | 253.79 |
| 423668 |  64 | 423732 | 1722.325 |   245.99 |  1.091 |    58.00 | 1723.415 | 245.87 |

| `-b` / `-ub` | peak VRAM | margin | pp @~424k | verdict |
|---|---:|---:|---:|---|
| 4096 / 1024 | 24 048 | -398 | 252 | FAIL fail |
| 3584 / 896 | 24 358 | -708 | 277 | FAIL fail |
| 2560 / 640 | 23 626 ... 23 655 | +24 ... -5 | 285 | FAIL marginal (fails in a full ladder run) |
| **2048 / 512** | **23 490 / 23 527 ... 23 669** | +160 ... -19 | 246 | PASS recommended (WARN borderline on rep 2) |

build: 13457 (a5d8684ce)

### 458 752 ctx - YaRN x1.75 FAIL

```bash
llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf -c 458752 \
  --rope-scaling yarn --rope-scale 1.75 \
  -ngl 99 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -b 512 -ub 128 --parallel 1 \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fitt 600 --metrics
```

main: n_ctx = 458752, n_batch = 512, n_ubatch = 128, flash_attn = 1, n_gpu_layers = 99,
n_threads = 6, n_threads_batch = 6, kv = sjkvarn4/sjkvarn4, rope = yarn x1.75, parallel = 1

|    PP |  TG |  N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s |   T s |  S t/s |
|-------|-----|-------|--------|----------|--------|----------|-------|--------|
| 16458 |  512 | 16970 |      - |  **1127.97** |      - |   **109.41** |     - |      - |
| 65535 |  512 | 66047 |      - |   796.31 |      - |   103.80 |     - |      - |
| 130987 |  512 | 131499 |      - |   412.63 |      - |    92.60 |     - |      - |
| 163719 |  512 | 164231 |      - |   315.03 |      - |    91.50 |     - |      - |
| 204697 |  512 | 205209 |      - |   246.12 |      - |    80.12 |     - |      - |
| 255848 |  512 | 256360 |      - |   234.06 |      - |    76.37 |     - |      - |
| 306982 |  512 | 307494 |      - |   221.23 |      - |    70.58 |     - |      - |
| 358118 |  512 | 358630 |      - |   208.22 |      - |    66.83 |     - |      - |
| 409346 |  512 | 409858 |      - |   194.52 |      - |    61.87 |     - |      - |

Times are not recorded for this tier: the `max_ctx` gate was cancelled at 98 % of its prefill, and
the ladder data was recovered from the run's progress log (rates only).

| `-b` / `-ub` | peak VRAM | margin | pp @~450k | verdict |
|---|---:|---:|---:|---|
| 1536 / 384 | 23 870 | -220 | 237 | FAIL fail |
| 1024 / 256 | 23 728 | -78 | 208 | FAIL fail |
| 512 / 128 | 23 608 (gate probe) / **24 039 (full ladder)** | +42 / **-389** | 180 | **FAIL fails in a full run - not deployable** |

build: 13457 (a5d8684ce)

### 471 859 ctx - YaRN x1.8 FAIL not viable

```bash
# -c 471859 --rope-scaling yarn --rope-scale 1.8
```

| `-b` / `-ub` | peak VRAM | margin | verdict |
|---|---:|---:|---|
| 512 / 128 | 23 864 | -214 | FAIL fail |
| 256 / 64 | 23 788 | -138 | FAIL fail - `-ub` floor reached |

Even at the smallest ubatch the KV cache for 471 859 tokens plus the compute buffers exceed the
23 650 MiB budget, so this tier cannot be served on this GPU.

## Summary by context

| ctx | YaRN | `-b` / `-ub` | pp @64k | pp @max_ctx | tg @64k | peak VRAM | status |
|---|---|---:|---:|---:|---:|---:|---|
| 262 144 | - | 4864 / 4864 | 1441 | 569 | 104 | 23 514 | PASS deploy |
| 393 216 | x1.5 | 5632 / 1408 | 1437 | 304 | 111 | 23 612 | PASS deploy, WARN VRAM borderline |
| 405 000 | x1.545 | 4096 / 1024 | 1436 | 273 | 109 | 23 478 | PASS deploy |
| 432 538 | x1.65 | 2048 / 512 | 1326 | 246 | 105 | 23 490 | PASS practical max context |
| 458 752 | x1.75 | 512 / 128 | 796 | - | 104 | 24 039 (fails) | FAIL over budget |
| 471 859 | x1.8 | - | - | - | - | 23 788 (fails) | FAIL infeasible |

`-b` itself is irrelevant to throughput: `-b`=`-ub` vs `-b`=4x`-ub` differ by -0.62 %. Prefill has a
step change between 200k and 250k prompts (-31...-42 %), and above ~160k the 393k tier beats the
262k tier despite the larger window.
