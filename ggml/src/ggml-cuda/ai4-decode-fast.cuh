#pragma once
// SM 8.6 (GA102 / RTX 3090) instruction-wall breakthrough decode kernels for ALL 6 AI+ bit-rates:
//   - AIT4_U / AIT5_U / AIT3_U / AIT2_U / AIT2P_U (dense N <= 8 + MoE MUL_MAT_ID N <= 8, zero ids_helper)
//   - AIF25_U / AIF25_U2 / AIF25_U2_512 / AIF175_U2 (dense N <= 8, cp.async.16 + whole-row 4-state quad decode)
// ZERO weight re-encoding. 100% bit-exact int32 per-chunk sums (exact 1MAD / pair-table math).
// Kill switch: GGML_CUDA_AI4_DECODE_FAST=0 (= exact rc7 behavior); default ON since 0.7.4.

#include "common.cuh"
#include "ai4-kern.cuh"

bool ggml_cuda_ai4_decode_fast_enabled();

// Unit-layout AI+ decode fast path (AIT4_U, AIT5_U, AIT3_U, AIT2_U, AIT2P_U):
//   - Dense (ids == nullptr): N = 1..8
//   - MoE MUL_MAT_ID (ids != nullptr): N = n_tok = 1..8, n_used * N <= 64 (ZERO ids_helper launch!)
bool ggml_cuda_ai4_decode_fast_unit_try(
        ggml_backend_cuda_context & ctx,
        ggml_type type,
        const uint8_t * W,
        int64_t nb02,
        const ggml_tensor * ids,     // nullptr for dense MUL_MAT
        const int8_t * xq,
        const float  * xs,
        float        * y_out,
        int M, int N, int K,
        int n_used, int ne11, int n_expert,
        int xsh,
        bool has_y_into);

// Fractional AI+ decode fast path (AIF175_U2, AIF25_U, AIF25_U2, AIF25_U2_512) for dense N = 1..8.
bool ggml_cuda_ai4_decode_fast_frac_try(
        ggml_backend_cuda_context & ctx,
        ggml_type type,
        int xsh,
        const uint8_t * w,
        const int8_t  * q,
        const float   * scales,
        float         * y,
        int M, int K, int64_t N,
        const uint8_t * group_mask = nullptr);
