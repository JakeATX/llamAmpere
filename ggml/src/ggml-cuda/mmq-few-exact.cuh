#pragma once

#include "common.cuh"

// Bit-exact row-parallel few-column MMQ for 2D contiguous Q4_K/Q6_K weights at 2..8 columns.
// Matches regular MMQ (quantize_mmq_q8_1 + mul_mat_q + mul_mat_q_stream_k_fixup) to 0 ULP on Ampere+.
bool ggml_cuda_should_use_mmq_few_exact(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

size_t ggml_cuda_mmq_few_exact_pool_bytes(const ggml_tensor * src0, const ggml_tensor * src1);

void ggml_cuda_mul_mat_q_few_exact(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        bool self_check = false);
