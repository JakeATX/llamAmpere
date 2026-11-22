#pragma once

#include "common.cuh"
#include "convert.cuh"

#include <cstdlib>

// INKGLUE (Inkling 2 decode glue fusions). Each fusion is on by default; GGML_CUDA_INKGLUE=0 turns all of them off,
// and its own variable (=0) turns one off.
static inline bool ggml_cuda_inkglue_on(const char * name) {
    static const bool all = [] { const char * e = getenv("GGML_CUDA_INKGLUE"); return !e || atoi(e) != 0; }();
    if (!all) {
        return false;
    }
    const char * e = getenv(name);
    return !e || atoi(e) != 0;
}

struct ink_situ_params {
    float ga_s, ga_b;   // SCALE of g before the tanh
    float gb_s, gb_b;   // SCALE after the tanh
    float ua_s, ua_b;   // SCALE of u before the tanh
    float ub_s, ub_b;   // SCALE after the tanh
    int   sig_first;    // MUL(sigmoid, act) instead of MUL(act, sigmoid)
    int   act_first;    // MUL(act, u) instead of MUL(u, act)
};

// one SITU element (k_ink_situ; INKSITUQ's producer kernel runs the same expression)
static __device__ __forceinline__ float ink_situ_elem(const float gv, const float uv, const ink_situ_params & p) {
    const float a1 = p.ga_s*gv + p.ga_b;
    const float a2 = tanhf(a1);
    const float a3 = p.gb_s*a2 + p.gb_b;
    const float sg = 1.0f / (1.0f + expf(-gv));
    const float a4 = p.sig_first ? sg*a3 : a3*sg;
    const float u1 = p.ua_s*uv + p.ua_b;
    const float u2 = tanhf(u1);
    const float u3 = p.ub_s*u2 + p.ub_b;
    // round-to-nearest product: never contracted into a following add (INKSITUQ quantizes this value in registers)
    return p.act_first ? __fmul_rn(a4, u3) : __fmul_rn(u3, a4);
}

void ggml_cuda_ink_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * dst,
        const ink_situ_params & p);
void ggml_cuda_ink_moe_wsum(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w, ggml_tensor * dst);
void ggml_cuda_ink_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * m, ggml_tensor * dst,
        float eps, float s, float b);
// tab != nullptr: the MUL_MAT table [E + 1, H, T] (bias, r unused); tab == nullptr: computes it from bias [n_rel, E] and r [n_rel, H, T]
void ggml_cuda_ink_rel_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * tab, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst);

// LOOP_GATE x G (2..4) of the same n, R, T and weight type in one down + one apply launch; per gate the kernels of
// loop-gate.cu term for term (bit-identical). Returns false when unsupported.
#define INK_MULTI_MAX 4
bool ggml_cuda_ink_loop_gate_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G);
// RMS_NORM -> MUL(gamma) x G (1..4) rows of the same width in one launch (rms_norm_f32's scalar arithmetic and block size);
// q8[k] != nullptr also writes row k as q8_1 exactly as quantize_q8_1 would (fills the MMVQ shared-quantize cache)
bool ggml_cuda_ink_rms_norm_mul_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * x,
        const ggml_tensor * const * gamma, ggml_tensor * const * dst, void * const * q8, const float * eps, int G);
// (mmvq.cu) grouped Q8_0 matvec, 1..8 members of one width, member k reading the prequantized q8_1 activation vy[k];
// vy == nullptr: only answers whether the group is supported
bool ggml_cuda_mul_mat_vec_q_group_pre(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n, const void * const * vy);

// LGTUNE launch shape of the loop-gate kernels (GGML_CUDA_LG_RB 1/2/4, GGML_CUDA_LG_ROWS 2/4/8), see inkglue.cu
int ggml_cuda_lg_tune_rb();
int ggml_cuda_lg_tune_rows();
