#pragma once

#include "common.cuh"

// FATTN-SJKVARN-FAST (fork): SM 8.6 optimized SJ-KVaRN 4/4 stream decode kernel for M = 1..8 (compact & rows).
// Targets the measured 2.629 ms/round (8 x 328.7 us, ~0.33 TB/s) far-attention bottleneck at 100K context:
//   GGML_SJKVARN_FAST=0 : off, kill switch (reverts 100% to baseline fattn-sjkvarn-stream.cuh)
//   GGML_SJKVARN_FAST=1 : (default) FP32-accum HMMA + branchless issue() (595 -> ~35 insns) +
//                         raw_pairs K (eliminates 64 FP16 decode insns/strip, NMSE ~3.9e-10) +
//                         __expf() softmax + templated record_change() with 0.0625 cancellation
//   GGML_SJKVARN_FAST=2 : (opt-in) FP32-accum KQ (LSE err = 0) + per-strip FP16 HMMA.16816.F16 on V -> FP32 acc
//                         (NMSE ~2.1e-7; cuts V TC cycles 512 -> 256/strip, total TC 1024 -> 768 cycles/strip)
//   GGML_SJKVARN_FAST=3 : (opt-in) FP16 HMMA.16816.F16 on BOTH KQ and per-strip V -> FP32 acc
//                         (NMSE ~3.2e-5; cuts total TC 1024 -> 512 cycles/strip < 680-cycle HBM floor!)
bool ggml_cuda_fattn_sj_kvarn_fast_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool rot);
