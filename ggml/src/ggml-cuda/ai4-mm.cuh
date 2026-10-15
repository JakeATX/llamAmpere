#pragma once
#include "common.cuh"

// AI4 (Ampere INT4, formerly D2) weights: row-slab reconstruct to f32 (original basis) + cuBLAS SGEMM on the backend's
// handle (TF32 math). The matching weight-only control for a quantized type is GGML_CUDA_FORCE_CUBLAS_RT=1 plus
// GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (dequant to f32 + the same cublasSgemm); without the second flag sm_86 runs the
// quantized cuBLAS path in fp16 compute. No mmvq/mmq/mmf.
void ggml_cuda_mul_mat_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// AI4 / AI4_U: shapes and layouts the CUDA backend runs (MUL_MAT and MUL_MAT_ID)
bool ggml_cuda_ai4_supported(const ggml_tensor * op);

// MUL_MAT_ID over an AI4 / AI4_U expert stack (device-side routing, only the selected experts are read)
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// true when ggml_cuda_mul_mat_id_ai4 would take the host-routed reference path (GGML_CUDA_AI4_REF=1 or pre-Ampere)
bool ggml_cuda_ai4_id_needs_sync(int cc, ggml_type t);
