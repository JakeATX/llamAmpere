#pragma once
#include "common.cuh"

// AI+ unit weights (AITk_U): row-slab reconstruct to f32 (original basis) + cuBLAS SGEMM on the backend's
// handle (TF32 math). The matching weight-only control for a quantized type is GGML_CUDA_FORCE_CUBLAS_RT=1 plus
// GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (dequant to f32 + the same cublasSgemm); without the second flag sm_86 runs the
// quantized cuBLAS path in fp16 compute. No mmvq/mmq/mmf.
void ggml_cuda_mul_mat_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// AI+ unit types: shapes and layouts the CUDA backend runs (MUL_MAT and MUL_MAT_ID)
bool ggml_cuda_ai4_supported(const ggml_tensor * op);

// MUL_MAT_ID over an AI+ unit expert stack (device-side routing, only the selected experts are read)
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// true when ggml_cuda_mul_mat_id_ai4 would take the host-routed reference path (GGML_CUDA_AI4_REF=1 or pre-Ampere)
bool ggml_cuda_ai4_id_needs_sync(int cc, ggml_type t);

bool ggml_cuda_moe_mixed_supported(const ggml_tensor * bank, const ggml_tensor * base);
const ggml_tensor * ggml_cuda_moe_mixed_base(const ggml_tensor * joined);
// The caller retains these buffers through the mixed GEMV; IQ4_NL banks need only Q8 activations.
void ggml_cuda_moe_mixed_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        ggml_cuda_pool_alloc<int8_t> & q, ggml_cuda_pool_alloc<float> & s);
void ggml_cuda_mul_mat_id_mixed(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        const ggml_tensor * base, ggml_tensor * dst, const char * act_q8 = nullptr,
        const void * const * table = nullptr, const uint8_t * skip = nullptr, const uint8_t * resident = nullptr,
        const ggml_tensor * route_map = nullptr, ggml_tensor * route_mask = nullptr,
        const int8_t * act_ai_q = nullptr, const float * act_ai_s = nullptr);
