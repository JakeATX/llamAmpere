#pragma once

#include "common.cuh"

// [KQTC] Q4_K / Q5_K / Q6_K / IQ3_S x f32 at 2..8 columns (speculative verify widths) on int8 tensor cores
// (Ampere mma.sync). Each weight block is decoded once for all columns instead of once per column. The
// activations use the same q8_1 quantization as MMVQ, so only the order of the sums changes (valid-math).
// Default off: GGML_CUDA_KQ_TC=1 (all four types) or a comma list of type names (q4_K,q5_K,q6_K,iq3_s).
// GGML_CUDA_KQ_TC_MIN_N (default 2) is the narrowest width taken; GGML_CUDA_KQ_TC_WARPS = 4 | 8 | auto (default);
// GGML_CUDA_KQ_TC_MAX_M skips matrices with more rows (0 = none); GGML_CUDA_KQ_TC_Q8CACHE=0 stops sharing the q8_1 cache.
// Returns false (caller keeps its route) when the type, shape, layout or device rules it out.
bool ggml_cuda_mul_mat_kq_tc_try(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
