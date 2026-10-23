#pragma once

#include "common.cuh"

// [FROWS] Float-weight row kernel for narrow F32 / F16 / BF16 weights at 2..8 columns (GGML_CUDA_FROWS=1, default off).
//
// Why: on Ampere a 2-D float weight at 4..8 columns goes to the tensor-core GEMM (mul_mat_f), which tiles 32 output rows
// per block. The Flash-Next router (2560 x 512, F32, 48 layers) then runs on 16 blocks of an 82-84 SM card: 28.9 us per
// launch at width 4 on the Taichi (about 180 GB/s, 1.32 ms per verify round). The vector kernel (mul_mat_vec_f, also
// what GGML_CUDA_NARROW_MMVF=1 picks) puts one block on every row, but then every block reads the whole activation
// again: at 4 columns that is 4x the weight bytes through L2, and it only reached 12.3 us.
//
// This kernel gives each block R consecutive rows, so one read of the activation serves R rows, and every thread issues
// all its loads for one K step (R weight vectors + the columns' activation vectors) before the multiply-adds.
// 16-byte loads: 4 floats of an F32 row, 8 halves of an F16/BF16 row.
//
// Math: fp32 multiply-add of the exact weight value (F16/BF16 widen exactly) and the f32 activation, in a fixed order
// (per thread along K, then a butterfly across the warp, then warps 0..NW-1 in order), so results are deterministic run
// to run. It is valid math, not bit-identical to mul_mat_f (which rounds both operands to TF32 for F32 weights, or the
// activation to the weight type for F16/BF16).
//
// Env: GGML_CUDA_FROWS=1 on; GGML_CUDA_FROWS_MAX_ROWS (default 1024) widest weight it takes; GGML_CUDA_FROWS_MIN_N /
// GGML_CUDA_FROWS_MAX_N (default 2 / 8) the widths it takes; GGML_CUDA_FROWS_R=1|2|4|8
// and GGML_CUDA_FROWS_NT=128|256 force rows per block / threads per block (default: the most rows per block that still
// leaves at least one block per SM, 128 threads).

bool ggml_cuda_frows_wants(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_frows_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

extern int64_t ggml_cuda_frows_count; // launches, for test-backend-ops' required-kernel check
