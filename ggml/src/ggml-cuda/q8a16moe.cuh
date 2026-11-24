#pragma once
// INKNVFP4K Q8A16 MoE: Q8_0 MUL_MAT_ID at verify width T = 2..8 on f32 activations, one weight read per distinct expert
// (kernel in q8a16moe-kern.cuh). Kill switch GGML_CUDA_Q8A16_MOE=0.
#include "common.cuh"

bool ggml_cuda_q8a16_moe_enabled();

// dst [N, n_used, T] = expert ids[k, t] of src0 (Q8_0 [K, N, E]) x src1 column (k or 0, t). Returns false (nothing
// launched) when the layout is not supported.
bool ggml_cuda_q8a16_moe_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst);
