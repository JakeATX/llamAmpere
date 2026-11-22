#pragma once
// INKNVFP4K Q8A16: Q8_0 decode matvec on f32 activations (kernel in q8a16-kern.cuh). Kill switch GGML_CUDA_Q8A16=0,
// widest batch GGML_CUDA_Q8A16_MAXT (default 8). Hooks: ggml_cuda_mul_mat_vec_q (plain 2D Q8_0, no ids / fusion /
// convrot) and the INKGLUE grouped single-column Q8_0 launch.
#include "common.cuh"

bool ggml_cuda_q8a16_enabled();

// plain MUL_MAT: dst = src0 (Q8_0) x src1 (f32, 1..MAXT columns). Returns false (nothing launched) when not taken.
bool ggml_cuda_q8a16_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// INKGLUE group: dsts[k] = MUL_MAT(src0_k, src1_k), one column each, same ne10. check_only: support test, no launch.
// Reads each member's f32 src1 (dsts[k]->src[1]->data), so the caller's q8_1 copies are not used.
bool ggml_cuda_q8a16_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n, bool check_only);

// MUL_MAT_ID at one token: dst[:, k] = expert ids[k] of src0 (Q8_0 [K, N, E]) x src1 column k (or 0), k < n_used <= 8
bool ggml_cuda_q8a16_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst);
