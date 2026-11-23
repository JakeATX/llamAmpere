#pragma once

#include "common.cuh"

struct ggml_cuda_gated_delta_net_state_src;

// Opt-in bit-identical R16 GDN kernel for verify/replay (GGML_CUDA_GDN_V16=1, F32 state).
// Returns true when launched; false falls back to the existing dispatch path.
bool ggml_cuda_gated_delta_net_v16_try(
    bool keep_rs, bool emit_ingr,
    const float * q_d, const float * k_d, const float * v_d,
    const float * g_d, const float * b_d, const float * s_d,
    float * dst_d, void * state_d,
    int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    int64_t neqk1, int64_t rq3,
    float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
    int ingr_w, int64_t ingr_region,
    const ggml_cuda_gated_delta_net_state_src & state_src, cudaStream_t stream);
