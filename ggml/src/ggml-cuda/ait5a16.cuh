#pragma once
// AIT5_A16: High-precision FP32-activation decode kernel for GGML_TYPE_AIT5_U (dense MUL_MAT & MoE MUL_MAT_ID).
//
// Eliminates int8 (a8) activation quantization error by rotating src1 in FP32 (H256 if K % 256 == 0, else H128,
// exact match to ggml-ai4.h ai4_sign_bits + 1/sqrt(n) FWHT) and dotting decoded AIT5_U trellis weights directly
// against FP32 rotated activations with FP32 accumulation (ZERO q8_1 error, ZERO fp16 overflow risk).
//
// Kill switches (ALL default 0 = opt-in until RTX 3090 KLD + speed gate):
//   GGML_CUDA_AIT5_A16=0|1      (default 0) dense MUL_MAT T = 1..MAXT
//   GGML_CUDA_AIT5_A16_MAXT=1..8 (default 8) widest batch width T
//   GGML_CUDA_AIT5_A16_ID=0|1   (default 0) MoE MUL_MAT_ID T = 1..MAXT (n_used * T <= 64, zero ids_helper)

#include "common.cuh"

#include <cstdint>

bool ggml_cuda_ait5a16_enabled();
bool ggml_cuda_ait5a16_id_enabled();
int  ggml_cuda_ait5a16_max_t();

// Dense MUL_MAT: dst = src0 (AIT5_U) x src1 (F32, 1..MAXT columns). Returns false (nothing launched) when declined.
bool ggml_cuda_ait5a16_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

// MoE MUL_MAT_ID: dst = src0 (AIT5_U [K, M, E]) x src1 (F32) routed by ids (I32, n_used * n_tok <= 64).
// y_into != nullptr: write routed outputs to y_into and leave invalid-id columns untouched (matches ai4.cu contract).
// Returns false (nothing launched) when declined.
bool ggml_cuda_ait5a16_mul_mat_id(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        float * y_into = nullptr);

extern "C" {
// Fire counters for benchmark/gate engagement assertions (which: 0 = dense MUL_MAT, 1 = MoE MUL_MAT_ID).
GGML_BACKEND_API uint64_t ggml_cuda_ait5a16_fire_count(int which);
GGML_BACKEND_API void     ggml_cuda_ait5a16_reset_fire_count(void);
// Returns the active cudaStream_t (as void*) of a CUDA ggml_backend_t for exact GPU cudaEvent timing.
GGML_BACKEND_API void *   ggml_cuda_ait5_backend_stream(ggml_backend_t backend);
}
