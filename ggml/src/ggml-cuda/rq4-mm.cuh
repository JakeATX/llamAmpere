#pragma once
// RQ4 (Hadamard-256 rotated 4-bit weights, W4A8 on int8 MMA) mul_mat for ggml. Kernels: rq4.cuh.
#include "common.cuh"

static inline bool ggml_cuda_is_rq4(const ggml_type type) {
    return type == GGML_TYPE_RQ4_G128 || type == GGML_TYPE_RQ4_G64 ||
           type == GGML_TYPE_RQ4NL_G128 || type == GGML_TYPE_RQ4NL_G64;
}

// shapes the kernels take: K % 1024, M % 128, 2-D weights, f32 contiguous activations and output; cc >= SM80
bool ggml_cuda_rq4_supported(const ggml_tensor * op, int cc);

// dst[n][m] = sum_k W[m][k] x[n][k]: activations rotated + int8-quantized into pool memory, then the
// N <= 8 tile kernel (decode / verify) or the GEMM (prefill)
void ggml_cuda_mul_mat_rq4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
