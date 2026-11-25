#pragma once

#include "gated_delta_net.cuh"

#include <climits>

// Shared conflict check for in-place GDN replay direct writes (used by gated_delta_net.cu and gated_delta_net_v16.cu).
static __device__ __forceinline__ bool gdn_direct_rows_conflict(
        const float * base, const ggml_cuda_gated_delta_net_state_src & ss,
        const int64_t n_seqs, const int64_t D) {
    if (ss.guard_in_place == 2) {
        return true; // GGML_CUDA_GDN_DIRECT_FORCE_FALLBACK=1 (test only)
    }
    for (int64_t t = 0; t < n_seqs; ++t) {
        const float * r0 = base + (int64_t) ss.rows[t] * ss.row_stride;
        for (int64_t s = 0; s < n_seqs; ++s) {
            for (int j = 0; j < 2; ++j) {
                const float * w = j == 0 ? ss.final_dst : ss.ckpt_dst;
                if (w == nullptr) {
                    continue;
                }
                const float * w0 = w + s * D;
                if (r0 < w0 + D && w0 < r0 + D && !(t == s && r0 == w0)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Shared word-level bit comparator for diagnostic checks (R16 mode 2 and V16 mode 3).
static __global__ void gdn_r16_diff(
        const uint32_t * __restrict__ a, const uint32_t * __restrict__ b, const int64_t n,
        unsigned long long * __restrict__ ndiff,
        unsigned long long * __restrict__ first_idx = nullptr) {
    unsigned long long c = 0;
    unsigned long long first = ULLONG_MAX;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        if (a[i] != b[i]) {
            c++;
            if ((unsigned long long) i < first) {
                first = (unsigned long long) i;
            }
        }
    }
    if (c) {
        atomicAdd(ndiff, c);
        if (first_idx != nullptr) {
            atomicMin(first_idx, first);
        }
    }
}

// Reference scalar-warp F32 GDN launch (S_v=128, !KDA) for diagnostic bit-identity verification.
void ggml_cuda_gated_delta_net_ref_f32(
    bool keep_rs, bool emit_ingr,
    const float * q_d, const float * k_d, const float * v_d,
    const float * g_d, const float * b_d, const float * s_d,
    float * dst_d, void * state_d,
    int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    int64_t neqk1, int64_t rq3,
    float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
    int ingr_w, int64_t ingr_region,
    const ggml_cuda_gated_delta_net_state_src & state_src, cudaStream_t stream);

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
    const ggml_cuda_gated_delta_net_state_src & state_src, ggml_cuda_pool * pool, cudaStream_t stream);
