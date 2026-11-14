#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

// bytes ggml_cuda_mul_mat_q_few takes from the pool for the quantized src1 (sizes the borrowed compute pool)
size_t ggml_cuda_mmq_few_pool_bytes(const ggml_tensor * src0, const ggml_tensor * src1);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
