#pragma once
#include "common.cuh"

// [K126-4] GGML_CUDA_AIF_STAGED_GEMM=1 (default 0 = off): the fractional AI+ prompt matmul (AIF25_U, AIF25_U2,
// AIF25_U2_512, AIF175_U2) on a cp.async-staged GEMM in place of magic2::mmq. Bit-identical to magic2::mmv.
// GGML_CUDA_AIF_STAGED_CFG picks the CTA shape (0 = 64 rows x 64 tokens, the default; 1 = 128 x 64; 2 = 64 x 128).
bool ggml_cuda_aif_staged_gemm_enabled();

// Returns false, without launching, for an unsupported type / shape (K % 512 != 0, N > INT32_MAX); the caller then runs
// its usual kernel.
bool ggml_cuda_aif_staged_gemm(ggml_type type, int xsh, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                               int M, int K, int64_t N, cudaStream_t stream);
