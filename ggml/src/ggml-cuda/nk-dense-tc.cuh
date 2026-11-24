#pragma once

#include "common.cuh"

// GGML_CUDA_NK_DENSE_TC=1: opt-in Ampere int8 tensor-core kernel for dense Q8_0
// decode/verify projections (T=1..8). Default OFF (unset or 0 = base path).
bool    ggml_cuda_nk_dense_tc_enabled();
int64_t ggml_cuda_nk_dense_tc_count();

bool ggml_cuda_nk_dense_tc_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor       * dst,
    bool                force_enable = false);

bool ggml_cuda_nk_dense_tc_group(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * const       * dsts,
    int                         n,
    bool                        check_only,
    bool                        force_enable = false);
