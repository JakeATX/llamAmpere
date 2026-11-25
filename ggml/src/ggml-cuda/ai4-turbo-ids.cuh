#pragma once

#include "common.cuh"

#include <cstdint>

// [TURBO-IDS] opt-in AI+ expert MUL_MAT_ID for small token counts: the turbo split-K matvec (ai4-turbo.cuh
// ampere_mmv_turbo, BN 8, 4 warps) with direct routing ids. Each CTA owns one (16-row slab, slot, token) pair and
// reads its expert id from ids itself, so there is no ids_helper routing launch and no routing table.
// Unit types AIT4_U (MODE_AIT4_S8) and AIT3_U (MODE_AIT3_S8_WROW, PERM B-load on the standard activation layout).
//
// Env: GGML_CUDA_AI4_TURBO_IDS=n (default 0 = off) takes n_tok 1..n (n capped at 8). Unset or 0: never taken, the
// caller's dispatch is unchanged.
bool ggml_cuda_ai4_turbo_ids_try(
        ggml_type type,
        const uint8_t * W,       // expert stack, expert e at W + e*nb02
        int64_t nb02,
        const ggml_tensor * ids, // int32 [n_used, n_tok]
        const int8_t * xq,       // ai4_quantize_x layout [ne11*n_tok, K]
        const float * xs,        // [ne11*n_tok, K >> (7 + xsh)]
        float * y_out,           // [M, n_used, n_tok]
        int M,
        int K,
        int n_tok,
        int n_used,
        int ne11,
        int n_expert,
        int xsh,
        bool has_y_into,         // true: skipped pairs keep what y_out holds; false: they are zeroed here
        cudaStream_t stream);
