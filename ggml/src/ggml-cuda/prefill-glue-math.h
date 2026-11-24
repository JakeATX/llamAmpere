#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// Pure C++ / CUDA shared helpers for the prefill activation fusion.
// Must remain includable from pure-CPU test harnesses without any CUDA headers or runtime calls.

#if defined(__CUDACC__)
#define PF_HD __host__ __device__ __forceinline__
#else
#define PF_HD inline
#endif

enum pf_glu_kind {
    PF_GLU_SITU         = -1,
    PF_GLU_REGLU        = 0,  // matches GGML_GLU_OP_REGLU
    PF_GLU_GEGLU        = 1,  // matches GGML_GLU_OP_GEGLU
    PF_GLU_SWIGLU       = 2,  // matches GGML_GLU_OP_SWIGLU
    PF_GLU_SWIGLU_OAI   = 3,  // matches GGML_GLU_OP_SWIGLU_OAI
    PF_GLU_GEGLU_ERF    = 4,  // matches GGML_GLU_OP_GEGLU_ERF
    PF_GLU_GEGLU_QUICK  = 5,  // matches GGML_GLU_OP_GEGLU_QUICK
    PF_GLU_SWIGLU_CLAMP = 6,  // note: runtime caller maps enum ggml_glu_op explicitly
};

struct pf_situ_desc {
    float ga_s, ga_b;
    float gb_s, gb_b;
    float ua_s, ua_b;
    float ub_s, ub_b;
    int   sig_first;
    int   act_first;
};

PF_HD float pf_silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

PF_HD float pf_gelu_f32(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

PF_HD float pf_gelu_erf_f32(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f * x * (1.0f + std::erf(x * SQRT_2_INV));
}

PF_HD float pf_gelu_quick_f32(float x) {
    const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + std::exp(GELU_QUICK_COEF * x)));
}

PF_HD float pf_relu_f32(float x) {
    return x > 0.0f ? x : 0.0f;
}

#if !defined(__CUDA_ARCH__)
// Host reference version matching ink_situ_elem bit-for-bit on IEEE FP32.
inline float pf_situ_eval_host(float gv, float uv, const pf_situ_desc & p) {
    const float a1 = p.ga_s * gv + p.ga_b;
    const float a2 = std::tanh(a1);
    const float a3 = p.gb_s * a2 + p.gb_b;
    const float sg = 1.0f / (1.0f + std::exp(-gv));
    const float a4 = p.sig_first ? sg * a3 : a3 * sg;
    const float u1 = p.ua_s * uv + p.ua_b;
    const float u2 = std::tanh(u1);
    const float u3 = p.ub_s * u2 + p.ub_b;
    return p.act_first ? a4 * u3 : u3 * a4;
}

inline float pf_glu_eval_host(int kind, float gate, float up, float limit, float oai_alpha, float oai_limit, const pf_situ_desc & situ) {
    switch (kind) {
        case PF_GLU_SITU:         return pf_situ_eval_host(gate, up, situ);
        case PF_GLU_REGLU:        return pf_relu_f32(gate) * up;
        case PF_GLU_GEGLU:        return pf_gelu_f32(gate) * up;
        case PF_GLU_SWIGLU:       return pf_silu_f32(gate) * up;
        case PF_GLU_GEGLU_ERF:    return pf_gelu_erf_f32(gate) * up;
        case PF_GLU_GEGLU_QUICK:  return pf_gelu_quick_f32(gate) * up;
        case PF_GLU_SWIGLU_OAI: {
            const float x = gate < oai_limit ? gate : oai_limit;
            const float g = up > -oai_limit ? (up < oai_limit ? up : oai_limit) : -oai_limit;
            const float o = x / (1.0f + std::exp(-x * oai_alpha));
            return o * (1.0f + g);
        }
        case PF_GLU_SWIGLU_CLAMP: {
            const float g = gate < limit ? gate : limit;
            const float u = up > -limit ? (up < limit ? up : limit) : -limit;
            return pf_silu_f32(g) * u;
        }
        default:
            return up;
    }
}
#endif

#undef PF_HD
