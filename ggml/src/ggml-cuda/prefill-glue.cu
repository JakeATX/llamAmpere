// [PREFILL-GLUE] Prefill FP32 Glue Fusion (SITU/GLU -> Rotated INT8/NVFP4PF/MMQ Q8_1,
// RMSNorm(+ADD) -> cached INT8/NVFP4PF/MMQ Q8_1) & Opt-In BF16 Transient MLP Buffer Narrowing.
// All features default OFF (0) until verified via GPU gates on SM 8.6.

#include "prefill-glue.cuh"
#include "prefill-glue-dev.cuh"
#include "ggml-backend-impl.h"
#include "ai4-mm.cuh"
#include "unary.cuh"
#include "mmq.cuh"
#include "mmq-prefill.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "nvfp4pf.cuh"
#include "nvfp4pf-kern.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static bool pf_env_on(const char * name) {
    const char * e = std::getenv(name);
    return e && std::atoi(e) != 0;
}

template <typename T>
static __device__ __forceinline__ float pf_to_f32(T v) {
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return v;
    }
}

template <typename T>
static __device__ __forceinline__ T pf_from_f32(float v) {
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return v;
    }
}

enum pf_write_mode {
    PF_WRITE_NONE = 0,
    PF_WRITE_F32  = 1,
    PF_WRITE_BF16 = 2,
};

static __device__ __forceinline__ float pf_eval_glu_dev(
        int glu_kind, float gate, float up, const ink_situ_params & situ,
        float glu_limit, float oai_alpha, float oai_limit) {
    switch (glu_kind) {
        case PF_GLU_SITU:
            return ink_situ_elem(gate, up, situ);
        case PF_GLU_REGLU:
            return fmaxf(gate, 0.0f) * up;
        case PF_GLU_GEGLU:
            return ggml_cuda_op_gelu_single(gate) * up;
        case PF_GLU_SWIGLU:
            return ggml_cuda_op_silu_single(gate) * up;
        case PF_GLU_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate, up, oai_alpha, oai_limit);
        case PF_GLU_SWIGLU_CLAMP:
            return ggml_cuda_op_swiglu_clamp_single(gate, up, glu_limit);
        case PF_GLU_GEGLU_ERF: {
            const float SQRT_2_INV = 0.70710678118654752440084436210484f;
            return (0.5f * gate * (1.0f + erff(gate * SQRT_2_INV))) * up;
        }
        case PF_GLU_GEGLU_QUICK: {
            const float GELU_QUICK_COEF = -1.702f;
            return (gate * (1.0f / (1.0f + expf(GELU_QUICK_COEF * gate)))) * up;
        }
        default:
            return up;
    }
}

template <typename T>
static __device__ __forceinline__ void pf_load8(const T * p, float (&dst)[8]) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 a = *(const float4 *) p;
        const float4 b = *(const float4 *) (p + 4);
        dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
        dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
    } else {
        const uint4 raw = *(const uint4 *) p;
        const nv_bfloat162 * v = reinterpret_cast<const nv_bfloat162 *>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v[i]);
            dst[2 * i + 0] = f.x;
            dst[2 * i + 1] = f.y;
        }
    }
}

template <typename T>
static __device__ __forceinline__ void pf_load4(const T * p, float (&dst)[4]) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 a = *(const float4 *) p;
        dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
    } else {
        const uint2 raw = *(const uint2 *) p;
        const nv_bfloat162 * v = reinterpret_cast<const nv_bfloat162 *>(&raw);
        const float2 f0 = __bfloat1622float2(v[0]);
        const float2 f1 = __bfloat1622float2(v[1]);
        dst[0] = f0.x; dst[1] = f0.y; dst[2] = f1.x; dst[3] = f1.y;
    }
}

template <int WRITE_MODE>
static __device__ __forceinline__ void pf_store8(void * out_base, int64_t idx, const float (&v)[8]) {
    if constexpr (WRITE_MODE == PF_WRITE_F32) {
        float * yr = (float *) out_base + idx;
        *(float4 *) yr       = make_float4(v[0], v[1], v[2], v[3]);
        *(float4 *) (yr + 4) = make_float4(v[4], v[5], v[6], v[7]);
    } else if constexpr (WRITE_MODE == PF_WRITE_BF16) {
        nv_bfloat162 packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
        }
        *(uint4 *) ((nv_bfloat16 *) out_base + idx) = *reinterpret_cast<const uint4 *>(packed);
    }
}

template <int WRITE_MODE>
static __device__ __forceinline__ void pf_store4(void * out_base, int64_t idx, const float (&v)[4]) {
    if constexpr (WRITE_MODE == PF_WRITE_F32) {
        *((float4 *) ((float *) out_base + idx)) = make_float4(v[0], v[1], v[2], v[3]);
    } else if constexpr (WRITE_MODE == PF_WRITE_BF16) {
        const nv_bfloat162 p0 = __floats2bfloat162_rn(v[0], v[1]);
        const nv_bfloat162 p1 = __floats2bfloat162_rn(v[2], v[3]);
        const uint2 u = make_uint2(*reinterpret_cast<const uint32_t *>(&p0), *reinterpret_cast<const uint32_t *>(&p1));
        *((uint2 *) ((nv_bfloat16 *) out_base + idx)) = u;
    }
}

// =================================================================================================
// 1. Fused Prefill GLU/SITU -> AI+ (H256 / H128)
// =================================================================================================
template <typename SRC_T, int XSH, bool SWAP8, bool XPERM, bool LINEAR, int WRITE_MODE>
static __global__ void __launch_bounds__(128) k_prefill_glu_ai4(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, void * __restrict__ out,
        int8_t * __restrict__ q, float * __restrict__ s,
        int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        ink_situ_params situ, float glu_limit, float oai_alpha, float oai_limit) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if constexpr (XSH == 1) {
        const int hb = blockIdx.x * 4 + warp;
        if (hb * RQ4_HB >= K) {
            return;
        }
        for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
            const int64_t i1 = tok % ne1;
            const int64_t i2 = tok / ne1;
            const int64_t elem_off = (int64_t) hb * RQ4_HB + lane * 8;
            const SRC_T * gr = (const SRC_T *) (g + (i2 * g_nb2 + i1 * g_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
            const SRC_T * ur = (const SRC_T *) (u + (i2 * u_nb2 + i1 * u_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
            float gv[8], uv[8], v[8];
            pf_load8(gr, gv);
            pf_load8(ur, uv);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
            }
            if constexpr (WRITE_MODE != PF_WRITE_NONE) {
                pf_store8<WRITE_MODE>(out, tok * K + elem_off, v);
            }
            rq4::quantize_act_regs<SWAP8, XPERM, LINEAR>(v, hb, q, s, K, (int) tok);
        }
    } else {
        const int l  = lane & 15;
        const int cb = (blockIdx.x * 4 + warp) * 2 + (lane >> 4);
        const bool live_cb = cb * 128 < K;
        for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
            const bool live = live_cb && tok < rows;
            float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (live) {
                const int64_t i1 = tok % ne1;
                const int64_t i2 = tok / ne1;
                const int64_t elem_off = (int64_t) cb * 128 + l * 8;
                const SRC_T * gr = (const SRC_T *) (g + (i2 * g_nb2 + i1 * g_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
                const SRC_T * ur = (const SRC_T *) (u + (i2 * u_nb2 + i1 * u_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
                float gv[8], uv[8];
                pf_load8(gr, gv);
                pf_load8(ur, uv);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
                }
                if constexpr (WRITE_MODE != PF_WRITE_NONE) {
                    pf_store8<WRITE_MODE>(out, tok * K + elem_off, v);
                }
            }
            ai4k::quantize_act128_regs<SWAP8, XPERM, LINEAR>(v, live, l, cb, q, s, K, (int) tok);
        }
    }
}

// =================================================================================================
// 2. Fused Prefill GLU/SITU -> NVFP4PF (KF = 64 or 32)
// =================================================================================================
template <typename SRC_T, int KF, int WRITE_MODE>
static __global__ void __launch_bounds__(128) k_prefill_glu_nvfp4pf(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, void * __restrict__ out,
        int8_t * __restrict__ xq, float * __restrict__ xs,
        int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        ink_situ_params situ, float glu_limit, float oai_alpha, float oai_limit) {
    static_assert(KF == 32 || KF == 64, "KF 32 or 64");
    constexpr int V = KF / 32;
    const int lane = threadIdx.x & 31;
    const int kb   = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nb   = K / KF;
    if (kb >= nb) {
        return;
    }
    for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
        const int64_t i1 = tok % ne1;
        const int64_t i2 = tok / ne1;
        const int64_t elem_off = (int64_t) kb * KF + V * lane;
        const SRC_T * gr = (const SRC_T *) (g + (i2 * g_nb2 + i1 * g_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
        const SRC_T * ur = (const SRC_T *) (u + (i2 * u_nb2 + i1 * u_nb1) / (sizeof(float) / sizeof(SRC_T))) + elem_off;
        float v[V];
        bool bad = false;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            const float gv = pf_to_f32(gr[i]);
            const float uv = pf_to_f32(ur[i]);
            const float val = pf_eval_glu_dev(glu_kind, gv, uv, situ, glu_limit, oai_alpha, oai_limit);
            if constexpr (WRITE_MODE == PF_WRITE_F32) {
                ((float *) out)[tok * K + elem_off + i] = val;
            } else if constexpr (WRITE_MODE == PF_WRITE_BF16) {
                ((nv_bfloat16 *) out)[tok * K + elem_off + i] = __float2bfloat16(val);
            }
            bad  = bad || !isfinite(val);
            v[i] = isfinite(val) ? val : 0.0f;
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o, 32));
        }
        bad = __any_sync(0xffffffff, bad);
        const bool  ok  = amax >= PF_NVFP4PF_MIN_SCALE;
        const float inv = ok ? 127.0f / amax : 0.0f;
        int8_t * q = xq + tok * K + elem_off;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            q[i] = (int8_t) max(-127, min(127, pf_nvfp4pf_f2i(v[i] * inv)));
        }
        if (lane == 0) {
            xs[tok * nb + kb] = bad ? __int_as_float(0x7fc00000) : ok ? amax * (1.0f / 127.0f) : 0.0f;
        }
    }
}

// =================================================================================================
// 3. Fused Prefill GLU/SITU -> MMQ block_q8_1_mmq (dense & MoE sorted ids_src1)
// =================================================================================================
template <mmq_q8_1_ds_layout DS_LAYOUT, bool SWIZZLE_IQ4>
static __device__ __forceinline__ void pf_emit_mmq_q8_1(
        const float (&xf)[4], block_q8_1_mmq * y, int64_t ib, int iqs) {
    constexpr int vals_per_scale = DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    float amax = fmaxf(fmaxf(fabsf(xf[0]), fabsf(xf[1])), fmaxf(fabsf(xf[2]), fabsf(xf[3])));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if constexpr (DS_LAYOUT != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xf[0] + xf[1] + xf[2] + xf[3];
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);
        }
    }
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    char4 q;
    q.x = roundf(xf[0] * d_inv);
    q.y = roundf(xf[1] * d_inv);
    q.z = roundf(xf[2] * d_inv);
    q.w = roundf(xf[3] * d_inv);
    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;

    if constexpr (SWIZZLE_IQ4) {
        int packed;
        std::memcpy(&packed, &q, sizeof(packed));
        const int t = threadIdx.x & 7;
        const int lo = __shfl_sync(0xffffffff, packed, t / 2, 8);
        const int hi = __shfl_sync(0xffffffff, packed, t / 2 + 4, 8);
        const int permuted = __byte_perm(lo, hi, (t & 1) ? 0x7362 : 0x5140);
        std::memcpy(&q, &permuted, sizeof(q));
    }

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs / 4] = q;

    if constexpr (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs / 64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if constexpr (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

template <typename SRC_T, mmq_q8_1_ds_layout DS_LAYOUT, bool SWIZZLE_IQ4, bool HAS_IDS, int WRITE_MODE>
static __global__ void k_prefill_glu_mmq(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, void * __restrict__ out,
        const int32_t * __restrict__ ids, void * __restrict__ vy,
        int K, int K_padded, int64_t rows_dst, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        ink_situ_params situ, float glu_limit, float oai_alpha, float oai_limit) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= K_padded) {
        return;
    }
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int     iqs     = (int) (i0 % QK8_1_MMQ);
    block_q8_1_mmq * y    = (block_q8_1_mmq *) vy;
    const int64_t blocks_per_col = K_padded / QK8_1_MMQ;
    GGML_UNUSED(blocks_per_col);

    for (int64_t col = blockIdx.x; col < rows_dst; col += gridDim.x) {
        const int64_t tok = HAS_IDS ? (int64_t) ids[col] : col;
        float xf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (i0 < K) {
            const int64_t i1 = tok % ne1;
            const int64_t i2 = tok / ne1;
            const SRC_T * gr = (const SRC_T *) (g + (i2 * g_nb2 + i1 * g_nb1) / (sizeof(float) / sizeof(SRC_T))) + i0;
            const SRC_T * ur = (const SRC_T *) (u + (i2 * u_nb2 + i1 * u_nb1) / (sizeof(float) / sizeof(SRC_T))) + i0;
            float gv[4], uv[4];
            pf_load4(gr, gv);
            pf_load4(ur, uv);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                xf[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
            }
            if constexpr (WRITE_MODE != PF_WRITE_NONE && !HAS_IDS) {
                pf_store4<WRITE_MODE>(out, tok * K + i0, xf);
            }
        }
        const int64_t ib = k_block * rows_dst + col;
        pf_emit_mmq_q8_1<DS_LAYOUT, SWIZZLE_IQ4>(xf, y, ib, iqs);
    }
}

// =================================================================================================
// 4. Standalone BF16 -> ActQ quantizers (used when PREFILL_BF16_ACT=1 & GLU fusion OFF)
// =================================================================================================
template <int XSH, bool SWAP8, bool XPERM, bool LINEAR>
static __global__ void __launch_bounds__(128) k_prefill_bf16_to_ai4(
        const nv_bfloat16 * __restrict__ x, int8_t * __restrict__ q, float * __restrict__ s,
        int K, int64_t rows) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if constexpr (XSH == 1) {
        const int hb = blockIdx.x * 4 + warp;
        if (hb * RQ4_HB >= K) {
            return;
        }
        for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
            float v[8];
            pf_load8(x + tok * K + (int64_t) hb * RQ4_HB + lane * 8, v);
            rq4::quantize_act_regs<SWAP8, XPERM, LINEAR>(v, hb, q, s, K, (int) tok);
        }
    } else {
        const int l  = lane & 15;
        const int cb = (blockIdx.x * 4 + warp) * 2 + (lane >> 4);
        const bool live_cb = cb * 128 < K;
        for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
            const bool live = live_cb && tok < rows;
            float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (live) {
                pf_load8(x + tok * K + (int64_t) cb * 128 + l * 8, v);
            }
            ai4k::quantize_act128_regs<SWAP8, XPERM, LINEAR>(v, live, l, cb, q, s, K, (int) tok);
        }
    }
}

template <int KF>
static __global__ void __launch_bounds__(128) k_prefill_bf16_to_nvfp4pf(
        const nv_bfloat16 * __restrict__ x, int8_t * __restrict__ xq, float * __restrict__ xs,
        int K, int64_t rows) {
    constexpr int V = KF / 32;
    const int lane = threadIdx.x & 31;
    const int kb   = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nb   = K / KF;
    if (kb >= nb) {
        return;
    }
    for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
        const int64_t elem_off = (int64_t) kb * KF + V * lane;
        const nv_bfloat16 * xp = x + tok * K + elem_off;
        float v[V];
        bool bad = false;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            const float val = __bfloat162float(xp[i]);
            bad  = bad || !isfinite(val);
            v[i] = isfinite(val) ? val : 0.0f;
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o, 32));
        }
        bad = __any_sync(0xffffffff, bad);
        const bool  ok  = amax >= PF_NVFP4PF_MIN_SCALE;
        const float inv = ok ? 127.0f / amax : 0.0f;
        int8_t * q = xq + tok * K + elem_off;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            q[i] = (int8_t) max(-127, min(127, pf_nvfp4pf_f2i(v[i] * inv)));
        }
        if (lane == 0) {
            xs[tok * nb + kb] = bad ? __int_as_float(0x7fc00000) : ok ? amax * (1.0f / 127.0f) : 0.0f;
        }
    }
}

template <mmq_q8_1_ds_layout DS_LAYOUT, bool SCATTER, bool SWIZZLE_IQ4>
static __global__ void k_prefill_bf16_to_mmq(
        const nv_bfloat16 * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int ne1, int ne2, int n_expert_used) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }
    int64_t base_idx;
    if constexpr (SCATTER) {
        base_idx = (int64_t) blockIdx.x * s02;
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    }
    float xf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (i0 < ne00) {
        pf_load4(x + base_idx + i0, xf);
    }

    constexpr int vals_per_scale = DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    float amax = fmaxf(fmaxf(fabsf(xf[0]), fabsf(xf[1])), fmaxf(fabsf(xf[2]), fabsf(xf[3])));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if constexpr (DS_LAYOUT != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xf[0] + xf[1] + xf[2] + xf[3];
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);
        }
    }
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    char4 q;
    q.x = roundf(xf[0] * d_inv);
    q.y = roundf(xf[1] * d_inv);
    q.z = roundf(xf[2] * d_inv);
    q.w = roundf(xf[3] * d_inv);
    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;
    if constexpr (SWIZZLE_IQ4) {
        int packed;
        std::memcpy(&packed, &q, sizeof(packed));
        const int t = threadIdx.x % 8;
        const int lo = __shfl_sync(0xffffffff, packed, t / 2, 8);
        const int hi = __shfl_sync(0xffffffff, packed, t / 2 + 4, 8);
        const int permuted = __byte_perm(lo, hi, (t % 2) ? 0x7362 : 0x5140);
        std::memcpy(&q, &permuted, sizeof(q));
    }

    block_q8_1_mmq * y    = (block_q8_1_mmq *) vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;
    const int     nwrite  = SCATTER ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (SCATTER) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block * ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / QK8_1);
            ib = ib0 + k_block * ne1 + blockIdx.x;
        }
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs / 4] = q;
        if constexpr (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs / 16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs / 64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if constexpr (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs / 32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs / 32] = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

static __global__ void k_prefill_bf16_to_f32(const nv_bfloat16 * __restrict__ src, float * __restrict__ dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// =================================================================================================
// 5. Fused Prefill RMSNorm (+ ADD) + MUL(gamma) -> Cached AI4 / NVFP4PF / MMQ via dynamic SMEM
//    Exact scalar reduction order matching norm.cu rms_norm_f32 (0 ULP vs unfused).
// =================================================================================================
template <int BS, bool HAS_ADD, bool WRITE_F32>
static __device__ __forceinline__ void pf_rms_norm_to_smem(
        const float * __restrict__ x_row, const float * __restrict__ r_row,
        float * __restrict__ add_row, const float * __restrict__ w, float * __restrict__ mul_row,
        float * __restrict__ s_norm, int K, float eps) {
    __shared__ float s_red[32];
    float tmp = 0.0f;
    for (int col = threadIdx.x; col < K; col += BS) {
        const float xi = HAS_ADD ? (x_row[col] + r_row[col]) : x_row[col];
        if constexpr (HAS_ADD) {
            add_row[col] = xi;
        }
        s_norm[col] = xi;
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_red);
    const float sc = rsqrtf(tmp / (float) K + eps);
    __syncthreads();
    for (int col = threadIdx.x; col < K; col += BS) {
        const float val = __fmul_rn(__fmul_rn(sc, s_norm[col]), w[col]);
        s_norm[col] = val;
        if constexpr (WRITE_F32) {
            mul_row[col] = val;
        }
    }
    __syncthreads();
}

template <int BS, int XSH, bool SWAP8, bool XPERM, bool LINEAR, bool HAS_ADD, bool WRITE_F32>
static __global__ void k_prefill_norm_ai4(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        int8_t * __restrict__ q, float * __restrict__ s,
        int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    const int nwarps = BS / 32;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD, WRITE_F32>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            WRITE_F32 ? (mul_dst + row * K) : nullptr,
            s_norm, K, eps);
        pf_quantize_row_ai4_from_ptr<XSH, SWAP8, XPERM, LINEAR>(s_norm, q, s, K, (int) row, nwarps);
        __syncthreads();
    }
}

template <int BS, int KF, bool HAS_ADD, bool WRITE_F32>
static __global__ void k_prefill_norm_nvfp4pf(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        int8_t * __restrict__ xq, float * __restrict__ xs,
        int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    const int nwarps = BS / 32;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD, WRITE_F32>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            WRITE_F32 ? (mul_dst + row * K) : nullptr,
            s_norm, K, eps);
        pf_quantize_row_nvfp4pf_from_ptr<KF>(s_norm, xq, xs, K, (int) row, nwarps);
        __syncthreads();
    }
}

template <int BS, mmq_q8_1_ds_layout DS_LAYOUT, bool SWIZZLE_IQ4, bool HAS_ADD, bool WRITE_F32>
static __global__ void k_prefill_norm_mmq(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        void * __restrict__ vy,
        int64_t n_rows, int K, int K_padded, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD, WRITE_F32>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            WRITE_F32 ? (mul_dst + row * K) : nullptr,
            s_norm, K, eps);
        for (int i0 = threadIdx.x * 4; i0 < K_padded; i0 += BS * 4) {
            float xf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (i0 < K) {
                const float4 v = *(const float4 *) (s_norm + i0);
                xf[0] = v.x; xf[1] = v.y; xf[2] = v.z; xf[3] = v.w;
            }
            const int k_block = i0 / QK8_1_MMQ;
            const int iqs     = i0 % QK8_1_MMQ;
            const int64_t ib  = (int64_t) k_block * n_rows + row;
            pf_emit_mmq_q8_1<DS_LAYOUT, SWIZZLE_IQ4>(xf, y, ib, iqs);
        }
        __syncthreads();
    }
}

// =================================================================================================
// Host-side cache, lifetime safety & graph matchers
// =================================================================================================

static const ggml_tensor * pf_unwrap_views(const ggml_tensor * t, int64_t * byte_off = nullptr) {
    int64_t off = 0;
    while (t && ((t->op == GGML_OP_VIEW && t->view_src) || (t->op == GGML_OP_RESHAPE && !t->view_src && t->src[0]))) {
        if (t->op == GGML_OP_VIEW) {
            off += (int64_t) t->view_offs;
            t = t->view_src;
        } else {
            t = t->src[0];
        }
    }
    if (byte_off) {
        *byte_off = off;
    }
    return t;
}

static bool pf_same_backing(const ggml_tensor * a, const ggml_tensor * b) {
    return a && b && (a == b || pf_unwrap_views(a) == pf_unwrap_views(b));
}

static bool pf_is_view_or_noop(const ggml_tensor * node) {
    return node && (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                    node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                    node->op == GGML_OP_TRANSPOSE);
}

static bool pf_cuda_buf(const ggml_tensor * t, int device) {
    return t && t->buffer && t->buffer->buft == ggml_backend_cuda_buffer_type(device);
}

static bool pf_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const char * pa = (const char *) a->data;
    const char * pb = (const char *) b->data;
    return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
}

static int pf_map_glu_op(ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_REGLU:        return PF_GLU_REGLU;
        case GGML_GLU_OP_GEGLU:        return PF_GLU_GEGLU;
        case GGML_GLU_OP_SWIGLU:       return PF_GLU_SWIGLU;
        case GGML_GLU_OP_SWIGLU_OAI:   return PF_GLU_SWIGLU_OAI;
        case GGML_GLU_OP_GEGLU_ERF:    return PF_GLU_GEGLU_ERF;
        case GGML_GLU_OP_GEGLU_QUICK:  return PF_GLU_GEGLU_QUICK;
        case GGML_GLU_OP_SWIGLU_CLAMP: return PF_GLU_SWIGLU_CLAMP;
        default: return -99;
    }
}

static bool pf_alloc_entry_buf(ggml_backend_cuda_context & ctx, ggml_backend_cuda_context::prefill_act_entry * e, size_t bytes) {
    if (e->ptr && e->dev == ctx.device && e->cap >= bytes) {
        return true;
    }
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(ctx.stream(), &cap_st) == cudaSuccess && cap_st != cudaStreamCaptureStatusNone) {
        return false; // never allocate raw CUDA memory during CUDA graph capture
    }
    if (e->ptr) {
        e->retired.push_back({ e->ptr, e->cap, e->dev });
        e->ptr = nullptr;
        e->cap = 0;
        e->dev = -1;
    }
    ggml_cuda_set_device(ctx.device);
    char * new_ptr = nullptr;
    if (cudaMalloc(&new_ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    e->ptr = new_ptr;
    e->cap = bytes;
    e->dev = ctx.device;
    return true;
}

static ggml_backend_cuda_context::prefill_act_entry * pf_pick_slot(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int kind) {
    auto & pc = ctx.prefill_act_cache;
    // Reuse an entry matching (src1, kind) first, then any invalid/stale slot, else LRU
    for (auto & c : pc.entries) {
        if (c.src1 == src1 && c.kind == kind && c.dev == ctx.device) {
            return &c;
        }
    }
    for (auto & c : pc.entries) {
        if (c.epoch != ctx.graph_epoch || c.kind == ggml_backend_cuda_context::PREFILL_ACT_NONE) {
            return &c;
        }
    }
    ggml_backend_cuda_context::prefill_act_entry * best = &pc.entries[0];
    for (auto & c : pc.entries) {
        if (c.last_use < best->last_use) {
            best = &c;
        }
    }
    return best;
}

template <typename LAUNCHER>
static void pf_dispatch_ai4_variants(int variant, int xsh, LAUNCHER && fn) {
    if (xsh == 1) {
        switch (variant) {
            case 1:  fn(std::integral_constant<int, 1>{}, std::true_type{}, std::false_type{}, std::false_type{}); break;
            case 2:  fn(std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{}, std::false_type{}); break;
            case 3:  fn(std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{}, std::true_type{}); break;
            default: fn(std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{}, std::false_type{}); break;
        }
    } else {
        switch (variant) {
            case 1:  fn(std::integral_constant<int, 0>{}, std::true_type{}, std::false_type{}, std::false_type{}); break;
            case 2:  fn(std::integral_constant<int, 0>{}, std::false_type{}, std::true_type{}, std::false_type{}); break;
            case 3:  fn(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{}, std::true_type{}); break;
            default: fn(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{}, std::false_type{}); break;
        }
    }
}

// Consumer route descriptor predicted from graph nodes for a single downstream matmul.
struct pf_consumer_route {
    int  kind       = ggml_backend_cuda_context::PREFILL_ACT_NONE;
    int  param      = 0;
    bool mmq_id     = false;  // true if MUL_MAT_ID taking MMQ (needs compact ids_src1)
    bool valid      = false;
    bool cublas_bf16 = false;
};

static pf_consumer_route pf_inspect_consumer(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * src_act) {
    pf_consumer_route r;
    if (!mm || (mm->op != GGML_OP_MUL_MAT && mm->op != GGML_OP_MUL_MAT_ID)) {
        return r;
    }
    if (pf_unwrap_views(mm->src[1]) != src_act && mm->src[1] != src_act) {
        return r;
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const bool is_id = (mm->op == GGML_OP_MUL_MAT_ID);
    const int64_t K  = src0->ne[0];
    const int64_t M  = src0->ne[1];
    const int dev    = ctx.device;
    const int cc     = ggml_cuda_info().devices[dev].cc;

    if (!src0 || !src1 || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || src0->ne[3] != 1) {
        return r;
    }
    if (!pf_cuda_buf(src0, dev) || !pf_cuda_buf(mm, dev) || ((uintptr_t) src1->data & 15) != 0 || K % 128 != 0) {
        return r;
    }
    if (!ggml_is_contiguous(mm)) {
        return r;
    }

    // 1. AI+ (dense & MoE)
    if (ggml_ai4_rate(src0->type) != 0 && !is_id && ggml_cuda_ai4_supported(mm) && M % 16 == 0) {
        const int64_t n_rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        if (n_rows <= 65535 && src0->ne[2] == 1 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
            const int xsh     = (K % 256 == 0) ? 1 : 0;
            const int variant = (src0->type == GGML_TYPE_AIT2P_U) ? 0 :
                                ggml_cuda_ai4_prefill_variant(ctx, src0->type, (int) M, (int) n_rows, (int) K, false);
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_AI4;
            r.param = xsh | (variant << 1);
            r.valid = true;
            return r;
        }
    }
    if (ggml_ai4_rate(src0->type) != 0 && is_id && ggml_cuda_ai4_supported(mm) && !ggml_cuda_ai4_id_needs_sync(cc, src0->type)) {
        const int64_t n_rows = src1->ne[1] * src1->ne[2];
        if (n_rows <= 65535 && src1->ne[3] == 1 && ggml_is_contiguous(src1)) {
            const int xsh     = (K % 256 == 0) ? 1 : 0;
            const int variant = ggml_cuda_ai4_prefill_variant(ctx, src0->type, (int) M, (int) n_rows, (int) K, true);
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_AI4;
            r.param = xsh | (variant << 1);
            r.valid = true;
            return r;
        }
    }
    // 2. AIF*
    if ((!is_id && ggml_cuda_aif25_supported(mm)) || (is_id && ggml_cuda_aif_rows_supported(mm))) {
        const int64_t n_rows = src1->ne[1] * src1->ne[2];
        if (n_rows <= 65535 && ggml_is_contiguous(src1)) {
            const int xsh = (K % 256 == 0) ? 1 : 0;
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_AI4;
            r.param = xsh | (3 << 1); // AI4_PREP_LINEAR
            r.valid = true;
            return r;
        }
    }
    // 3. NVFP4PF (MoE >= 512 tokens on SM 8.0 / 8.6)
    if (is_id && src0->type == GGML_TYPE_NVFP4) {
        const ggml_tensor * ids = mm->src[2];
        const int KF = ggml_cuda_nvfp4pf_mode();
        if (KF > 0 && ids && ids->type == GGML_TYPE_I32 && (cc == GGML_CUDA_CC_AMPERE || cc == 860) &&
                M % 16 == 0 && src0->ne[2] <= NVFP4PF_MAX_EXPERTS && ids->ne[1] >= 512 &&
                src1->ne[1] * src1->ne[2] <= 65535 && src1->ne[3] == 1 && ggml_is_contiguous(src1) &&
                src0->nb[0] == 36 && src0->nb[1] % 36 == 0 && src0->nb[1] >= (size_t) (K / 64) * 36 &&
                src0->nb[2] % 4 == 0 && (uintptr_t) src0->data % 4 == 0) {
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF;
            r.param = KF;
            r.valid = true;
            return r;
        }
    }
    // 4. MMQ (quantized weights above MMVQ threshold)
    if (ggml_is_quantized(src0->type) && src0->type != GGML_TYPE_TQ4_1S && src0->type != GGML_TYPE_TQ3_1S) {
#if !defined(BLACKWELL_MMA_AVAILABLE)
        const int64_t ncols_mm = is_id ? mm->src[2]->ne[1] : src1->ne[1];
        const bool use_mmvq = is_id ? (ncols_mm <= 8) : (ncols_mm <= 8 || ggml_cuda_should_use_mmvq(src0->type, cc, ncols_mm));
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                       ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
        if (!use_mmvq && !bad_padding_clear && ggml_cuda_should_use_mmq(src0->type, cc, ncols_mm, is_id ? src0->ne[2] : 0) &&
                ggml_cuda_mmq_has_config(src0->type, M, cc, ggml_cuda_info().devices[dev].smpbo) &&
                ggml_is_contiguous(src1) && src1->ne[3] == 1) {
            if (!is_id && src1->ne[2] == 1) {
                mmq_args a{};
                a.x = (const char *) src0->data; a.type_x = src0->type;
                a.dst = (float *) mm->data; a.ncols_x = K; a.nrows_x = M;
                a.ncols_dst = src1->ne[1]; a.stride_row_x = src0->nb[1] / ggml_type_size(src0->type);
                a.ncols_y = src1->ne[1]; a.nrows_dst = mm->ne[1];
                a.nchannels_x = 1; a.nchannels_y = 1; a.nsamples_x = 1; a.nsamples_y = 1;
                a.ncols_max = src1->ne[1];
                const bool swizzle_iq4 = ggml_cuda_should_use_mmq_prefill(a);
                const int dsl = (int) mmq_get_q8_1_ds_layout(src0->type);
                r.kind  = ggml_backend_cuda_context::PREFILL_ACT_MMQ;
                r.param = dsl | (swizzle_iq4 ? 0x10 : 0);
                r.valid = true;
                return r;
            }
            if (is_id) {
                const bool dedup_bcast = (src1->ne[1] == 1 && mm->src[2]->ne[0] > 1);
                if (!dedup_bcast) {
                    const int dsl = (int) mmq_get_q8_1_ds_layout(src0->type);
                    r.kind   = ggml_backend_cuda_context::PREFILL_ACT_MMQ;
                    r.param  = dsl;
                    r.mmq_id = true;
                    r.valid  = true;
                    return r;
                }
            }
        }
#endif
    }
    // 5. BF16 native cuBLAS weights (eligible for BF16 buffer narrowing, not INT8 ActQ)
    if (!is_id && src0->type == GGML_TYPE_BF16) {
        r.cublas_bf16 = true;
    }
    return r;
}

static bool pf_collect_consumers(
        const ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int prod_idx, const ggml_tensor * prod_out,
        std::vector<int> & consumer_idxs, std::vector<pf_consumer_route> & routes, bool & all_quant_same, bool & has_any_view_leak) {
    consumer_idxs.clear();
    routes.clear();
    all_quant_same = false;
    has_any_view_leak = false;

    const int use_cnt = ggml_node_get_use_count(cgraph, prod_idx);
    if (use_cnt <= 0 || (prod_out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    // Check if any graph node reads prod_out directly or via a contiguous VIEW/RESHAPE alias
    for (int j = prod_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!node->src[s]) {
                continue;
            }
            if (node->src[s] == prod_out || pf_unwrap_views(node->src[s]) == prod_out) {
                if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE) {
                    if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(node) || node->data != prod_out->data) {
                        has_any_view_leak = true;
                        return false;
                    }
                    continue;
                }
                if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && s == 1) {
                    consumer_idxs.push_back(j);
                    routes.push_back(pf_inspect_consumer(ctx, node, prod_out));
                } else {
                    has_any_view_leak = true;
                    return false;
                }
            }
        }
    }
    if (consumer_idxs.empty()) {
        return false;
    }
    all_quant_same = routes[0].valid;
    for (size_t k = 1; k < routes.size(); ++k) {
        if (!routes[k].valid || routes[k].kind != routes[0].kind || routes[k].param != routes[0].param || routes[k].mmq_id != routes[0].mmq_id) {
            all_quant_same = false;
            break;
        }
    }
    return true;
}

} // namespace

// =================================================================================================
// Public Env Switches
// =================================================================================================

bool ggml_cuda_prefill_fuse_glu_actq_on() {
    static const bool on = pf_env_on("GGML_CUDA_PREFILL_FUSE_GLU_ACTQ");
    return on;
}

bool ggml_cuda_prefill_fuse_norm_actq_on() {
    static const bool on = pf_env_on("GGML_CUDA_PREFILL_FUSE_NORM_ACTQ");
    return on;
}

bool ggml_cuda_prefill_bf16_act_on() {
    static const bool on = pf_env_on("GGML_CUDA_PREFILL_BF16_ACT");
    return on;
}

bool ggml_cuda_prefill_any_on() {
    return ggml_cuda_prefill_fuse_glu_actq_on() ||
           ggml_cuda_prefill_fuse_norm_actq_on() ||
           ggml_cuda_prefill_bf16_act_on();
}

// =================================================================================================
// Cache Lifecycle & Invalidation
// =================================================================================================

void ggml_cuda_prefill_glue_reset(ggml_backend_cuda_context * ctx) {
    if (!ctx) {
        return;
    }
    for (auto & c : ctx->prefill_act_cache.entries) {
        c.src1     = nullptr;
        c.data     = nullptr;
        c.kind     = ggml_backend_cuda_context::PREFILL_ACT_NONE;
        c.rem_uses = 0;
    }
    ctx->prefill_glu_mmq_id.active = false;
    ctx->prefill_glu_mmq_id.dst    = nullptr;
    ctx->prefill_bf16_acts.clear();
}

void ggml_cuda_prefill_glue_free(ggml_backend_cuda_context * ctx) {
    if (!ctx) {
        return;
    }
    auto free_buf = [](char * ptr, int dev) {
        if (ptr != nullptr) {
            ggml_cuda_set_device(dev);
            CUDA_CHECK(cudaFree(ptr));
        }
    };
    for (auto & c : ctx->prefill_act_cache.entries) {
        free_buf(c.ptr, c.dev);
        for (const auto & r : c.retired) {
            free_buf(r.ptr, r.dev);
        }
        c.retired.clear();
        c.ptr      = nullptr;
        c.cap      = 0;
        c.dev      = -1;
        c.src1     = nullptr;
        c.data     = nullptr;
        c.kind     = ggml_backend_cuda_context::PREFILL_ACT_NONE;
        c.rem_uses = 0;
    }
    ctx->prefill_glu_mmq_id.active = false;
    ctx->prefill_glu_mmq_id.dst    = nullptr;
    ctx->prefill_bf16_acts.clear();
}

void ggml_cuda_prefill_invalidate_ptr(ggml_backend_cuda_context & ctx, const void * ptr, size_t nbytes) {
    if (!ptr || nbytes == 0 || !ggml_cuda_prefill_any_on()) {
        return;
    }
    const char * p = (const char *) ptr;
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind != ggml_backend_cuda_context::PREFILL_ACT_NONE && c.data != nullptr) {
            const char * d = (const char *) c.data;
            const size_t db = (size_t) c.n_rows * (size_t) c.k * sizeof(float);
            if (p < d + db && d < p + nbytes) {
                c.src1     = nullptr;
                c.data     = nullptr;
                c.kind     = ggml_backend_cuda_context::PREFILL_ACT_NONE;
                c.rem_uses = 0;
            }
        }
    }
    for (size_t i = 0; i < ctx.prefill_bf16_acts.size();) {
        const auto & e = ctx.prefill_bf16_acts[i];
        const char * d = (const char *) e.data;
        if (d && p < d + e.nbytes_f32 && d < p + nbytes) {
            ctx.prefill_bf16_acts[i] = ctx.prefill_bf16_acts.back();
            ctx.prefill_bf16_acts.pop_back();
        } else {
            ++i;
        }
    }
    if (ctx.prefill_glu_mmq_id.active) {
        auto overlaps = [&](const ggml_tensor * t) {
            if (!t || !t->data) return false;
            const char * td = (const char *) t->data;
            return p < td + ggml_nbytes(t) && td < p + nbytes;
        };
        if (overlaps(ctx.prefill_glu_mmq_id.dst) || overlaps(ctx.prefill_glu_mmq_id.g) || overlaps(ctx.prefill_glu_mmq_id.u)) {
            ctx.prefill_glu_mmq_id.active = false;
        }
    }
}

// =================================================================================================
// Persistent ActQ Alloc / Take Helpers
// =================================================================================================

bool ggml_cuda_prefill_ai4_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, int rem_uses, int8_t * & q, float * & s) {
    if (!src1 || !src1->data || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const size_t ns    = (size_t) n_rows * (size_t) (K >> (7 + xsh));
    const size_t q_off = GGML_PAD(ns * sizeof(float), 256);
    const size_t bytes = q_off + (size_t) n_rows * (size_t) K;
    auto * e = pf_pick_slot(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_AI4);
    if (!pf_alloc_entry_buf(ctx, e, bytes)) {
        return false;
    }
    e->src1     = src1;
    e->data     = src1->data;
    e->epoch    = ctx.graph_epoch;
    e->n_rows   = n_rows;
    e->k        = K;
    e->k_padded = K;
    e->kind     = ggml_backend_cuda_context::PREFILL_ACT_AI4;
    e->param    = xsh | (variant << 1);
    e->aux_off  = q_off;
    e->rem_uses = rem_uses;
    e->last_use = ++ctx.prefill_act_cache.tick;
    s = (float *) e->ptr;
    q = (int8_t *) (e->ptr + q_off);
    return true;
}

bool ggml_cuda_prefill_ai4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, const int8_t * & q, const float * & s) {
    if (!ggml_cuda_prefill_any_on() || !src1 || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const int param = xsh | (variant << 1);
    const ggml_tensor * base = pf_unwrap_views(src1);
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4 &&
                c.epoch == ctx.graph_epoch &&
                c.dev == ctx.device &&
                c.n_rows == n_rows && c.k == K && c.param == param &&
                (c.src1 == src1 || c.src1 == base) && c.data == src1->data) {
            s = (const float *) c.ptr;
            q = (const int8_t *) (c.ptr + c.aux_off);
            c.last_use = ++ctx.prefill_act_cache.tick;
            if (c.rem_uses > 0 && --c.rem_uses == 0) {
                c.src1 = nullptr;
                c.kind = ggml_backend_cuda_context::PREFILL_ACT_NONE;
            }
            ctx.fusion_stats.prefill_norm_actq_hits++;
            return true;
        }
    }
    return false;
}

bool ggml_cuda_prefill_nvfp4_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, int rem_uses, int8_t * & xq, float * & xs) {
    if (!src1 || !src1->data || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const size_t xq_bytes = GGML_PAD((size_t) n_rows * (size_t) K, 256);
    const size_t xs_bytes = (size_t) n_rows * (size_t) (K / kf) * sizeof(float);
    const size_t bytes    = xq_bytes + xs_bytes;
    auto * e = pf_pick_slot(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF);
    if (!pf_alloc_entry_buf(ctx, e, bytes)) {
        return false;
    }
    e->src1     = src1;
    e->data     = src1->data;
    e->epoch    = ctx.graph_epoch;
    e->n_rows   = n_rows;
    e->k        = K;
    e->k_padded = K;
    e->kind     = ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF;
    e->param    = kf;
    e->aux_off  = xq_bytes;
    e->rem_uses = rem_uses;
    e->last_use = ++ctx.prefill_act_cache.tick;
    xq = (int8_t *) e->ptr;
    xs = (float *) (e->ptr + xq_bytes);
    return true;
}

bool ggml_cuda_prefill_nvfp4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, const int8_t * & xq, const float * & xs) {
    if (!ggml_cuda_prefill_any_on() || !src1 || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const ggml_tensor * base = pf_unwrap_views(src1);
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF &&
                c.epoch == ctx.graph_epoch &&
                c.dev == ctx.device &&
                c.n_rows == n_rows && c.k == K && c.param == kf &&
                (c.src1 == src1 || c.src1 == base) && c.data == src1->data) {
            xq = (const int8_t *) c.ptr;
            xs = (const float *) (c.ptr + c.aux_off);
            c.last_use = ++ctx.prefill_act_cache.tick;
            if (c.rem_uses > 0 && --c.rem_uses == 0) {
                c.src1 = nullptr;
                c.kind = ggml_backend_cuda_context::PREFILL_ACT_NONE;
            }
            ctx.fusion_stats.prefill_norm_actq_hits++;
            return true;
        }
    }
    return false;
}

bool ggml_cuda_prefill_mmq_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, int rem_uses, char * & q8) {
    if (!src1 || !src1->data || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const size_t bytes = (size_t) (k_padded / QK8_1_MMQ) * (size_t) n_rows * sizeof(block_q8_1_mmq) + 32u * 1024u;
    auto * e = pf_pick_slot(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_MMQ);
    if (!pf_alloc_entry_buf(ctx, e, bytes)) {
        return false;
    }
    e->src1     = src1;
    e->data     = src1->data;
    e->epoch    = ctx.graph_epoch;
    e->n_rows   = n_rows;
    e->k        = K;
    e->k_padded = k_padded;
    e->kind     = ggml_backend_cuda_context::PREFILL_ACT_MMQ;
    e->param    = ds_layout | (swizzle_iq4 ? 0x10 : 0);
    e->aux_off  = 0;
    e->rem_uses = rem_uses;
    e->last_use = ++ctx.prefill_act_cache.tick;
    q8 = e->ptr;
    return true;
}

bool ggml_cuda_prefill_mmq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, const char * & q8) {
    if (!ggml_cuda_prefill_any_on() || !src1 || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const int param = ds_layout | (swizzle_iq4 ? 0x10 : 0);
    const ggml_tensor * base = pf_unwrap_views(src1);
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ &&
                c.epoch == ctx.graph_epoch &&
                c.dev == ctx.device &&
                c.n_rows == n_rows && c.k == K && c.k_padded == k_padded && c.param == param &&
                (c.src1 == src1 || c.src1 == base) && c.data == src1->data) {
            q8 = c.ptr;
            c.last_use = ++ctx.prefill_act_cache.tick;
            if (c.rem_uses > 0 && --c.rem_uses == 0) {
                c.src1 = nullptr;
                c.kind = ggml_backend_cuda_context::PREFILL_ACT_NONE;
            }
            ctx.fusion_stats.prefill_norm_actq_hits++;
            return true;
        }
    }
    return false;
}

// =================================================================================================
// Public Kernel Launchers (also called by bench-prefill-glue)
// =================================================================================================

void ggml_cuda_prefill_launch_glu_ai4(
        int glu_kind, const void * g, const void * u, void * out,
        int8_t * q, float * s, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode, int xsh, int variant,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const dim3 grid(xsh ? (unsigned) ((K / RQ4_HB + 3) / 4) : (unsigned) ((K / 128 + 7) / 8),
                    (unsigned) std::min<int64_t>(rows, 65535));
    auto go = [&](auto src_tag, auto wm_tag) {
        using SRC_T = decltype(src_tag);
        constexpr int WM = decltype(wm_tag)::value;
        pf_dispatch_ai4_variants(variant, xsh, [&](auto xsh_tag, auto sw_tag, auto xp_tag, auto ln_tag) {
            k_prefill_glu_ai4<SRC_T, decltype(xsh_tag)::value, decltype(sw_tag)::value, decltype(xp_tag)::value, decltype(ln_tag)::value, WM>
                <<<grid, 128, 0, stream>>>(
                    glu_kind, (const char *) g, (const char *) u, out, q, s, K, rows, ne1,
                    g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
        });
    };
    auto pick_wm = [&](auto src_tag) {
        if (write_mode == PF_WRITE_F32)       go(src_tag, std::integral_constant<int, PF_WRITE_F32>{});
        else if (write_mode == PF_WRITE_BF16) go(src_tag, std::integral_constant<int, PF_WRITE_BF16>{});
        else                                  go(src_tag, std::integral_constant<int, PF_WRITE_NONE>{});
    };
    if (src_bf16) pick_wm(nv_bfloat16{});
    else          pick_wm(float{});
}

void ggml_cuda_prefill_launch_glu_nvfp4pf(
        int glu_kind, const void * g, const void * u, void * out,
        int8_t * xq, float * xs, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode, int kf,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const dim3 grid((unsigned) ((K / kf + 3) / 4), (unsigned) std::min<int64_t>(rows, 65535));
    auto go = [&](auto src_tag, auto kf_tag, auto wm_tag) {
        using SRC_T = decltype(src_tag);
        k_prefill_glu_nvfp4pf<SRC_T, decltype(kf_tag)::value, decltype(wm_tag)::value>
            <<<grid, 128, 0, stream>>>(
                glu_kind, (const char *) g, (const char *) u, out, xq, xs, K, rows, ne1,
                g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    };
    auto pick_wm = [&](auto src_tag, auto kf_tag) {
        if (write_mode == PF_WRITE_F32)       go(src_tag, kf_tag, std::integral_constant<int, PF_WRITE_F32>{});
        else if (write_mode == PF_WRITE_BF16) go(src_tag, kf_tag, std::integral_constant<int, PF_WRITE_BF16>{});
        else                                  go(src_tag, kf_tag, std::integral_constant<int, PF_WRITE_NONE>{});
    };
    auto pick_kf = [&](auto src_tag) {
        if (kf == 32) pick_wm(src_tag, std::integral_constant<int, 32>{});
        else          pick_wm(src_tag, std::integral_constant<int, 64>{});
    };
    if (src_bf16) pick_kf(nv_bfloat16{});
    else          pick_kf(float{});
}

void ggml_cuda_prefill_launch_glu_mmq(
        int glu_kind, const void * g, const void * u, void * out,
        const int32_t * ids, void * vy, int ds_layout, bool swizzle_iq4,
        int K, int K_padded, int64_t rows_dst, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const unsigned by = (unsigned) ((K_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ));
    const dim3 grid((unsigned) std::min<int64_t>(rows_dst, 65535), by, 1);
    auto go = [&](auto src_tag, auto ds_tag, auto sw_tag, auto id_tag, auto wm_tag) {
        using SRC_T = decltype(src_tag);
        k_prefill_glu_mmq<SRC_T, decltype(ds_tag)::value, decltype(sw_tag)::value, decltype(id_tag)::value, decltype(wm_tag)::value>
            <<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
                glu_kind, (const char *) g, (const char *) u, out, ids, vy,
                K, K_padded, rows_dst, ne1, g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    };
    auto pick_wm = [&](auto src_tag, auto ds_tag, auto sw_tag, auto id_tag) {
        if (ids != nullptr || write_mode == PF_WRITE_NONE) {
            go(src_tag, ds_tag, sw_tag, id_tag, std::integral_constant<int, PF_WRITE_NONE>{});
        } else if (write_mode == PF_WRITE_BF16) {
            go(src_tag, ds_tag, sw_tag, id_tag, std::integral_constant<int, PF_WRITE_BF16>{});
        } else {
            go(src_tag, ds_tag, sw_tag, id_tag, std::integral_constant<int, PF_WRITE_F32>{});
        }
    };
    auto pick_id = [&](auto src_tag, auto ds_tag, auto sw_tag) {
        if (ids != nullptr) pick_wm(src_tag, ds_tag, sw_tag, std::true_type{});
        else                pick_wm(src_tag, ds_tag, sw_tag, std::false_type{});
    };
    auto pick_ds = [&](auto src_tag) {
        if (swizzle_iq4) {
            pick_id(src_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::true_type{});
        } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            pick_id(src_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_DS4>{}, std::false_type{});
        } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            pick_id(src_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D2S6>{}, std::false_type{});
        } else {
            pick_id(src_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::false_type{});
        }
    };
    if (src_bf16) pick_ds(nv_bfloat16{});
    else          pick_ds(float{});
}

void ggml_cuda_prefill_launch_norm_ai4(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        int8_t * q, float * s, int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, int xsh, int variant, cudaStream_t stream) {
    GGML_UNUSED(w_ne1);
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto add_tag, auto wf_tag) {
        constexpr int BS      = decltype(bs_tag)::value;
        constexpr bool ADD    = decltype(add_tag)::value;
        constexpr bool WF     = decltype(wf_tag)::value;
        pf_dispatch_ai4_variants(variant, xsh, [&](auto xsh_tag, auto sw_tag, auto xp_tag, auto ln_tag) {
            auto * kfn = k_prefill_norm_ai4<BS, decltype(xsh_tag)::value, decltype(sw_tag)::value, decltype(xp_tag)::value, decltype(ln_tag)::value, ADD, WF>;
            if (smem > 48u * 1024u) {
                CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            }
            kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, q, s, n_rows, K, x_s1, r_s1, eps);
        });
    };
    auto pick_wf = [&](auto bs_tag, auto add_tag) {
        if (write_f32) launch(bs_tag, add_tag, std::true_type{});
        else           launch(bs_tag, add_tag, std::false_type{});
    };
    auto pick_add = [&](auto bs_tag) {
        if (r != nullptr) pick_wf(bs_tag, std::true_type{});
        else              pick_wf(bs_tag, std::false_type{});
    };
    if (K < 1024) pick_add(std::integral_constant<int, 256>{});
    else          pick_add(std::integral_constant<int, 1024>{});
}

void ggml_cuda_prefill_launch_norm_nvfp4pf(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        int8_t * xq, float * xs, int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, int kf, cudaStream_t stream) {
    GGML_UNUSED(w_ne1);
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto kf_tag, auto add_tag, auto wf_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        auto * kfn = k_prefill_norm_nvfp4pf<BS, decltype(kf_tag)::value, decltype(add_tag)::value, decltype(wf_tag)::value>;
        if (smem > 48u * 1024u) {
            CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        }
        kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, xq, xs, n_rows, K, x_s1, r_s1, eps);
    };
    auto pick_wf = [&](auto bs_tag, auto kf_tag, auto add_tag) {
        if (write_f32) launch(bs_tag, kf_tag, add_tag, std::true_type{});
        else           launch(bs_tag, kf_tag, add_tag, std::false_type{});
    };
    auto pick_add = [&](auto bs_tag, auto kf_tag) {
        if (r != nullptr) pick_wf(bs_tag, kf_tag, std::true_type{});
        else              pick_wf(bs_tag, kf_tag, std::false_type{});
    };
    auto pick_kf = [&](auto bs_tag) {
        if (kf == 32) pick_add(bs_tag, std::integral_constant<int, 32>{});
        else          pick_add(bs_tag, std::integral_constant<int, 64>{});
    };
    if (K < 1024) pick_kf(std::integral_constant<int, 256>{});
    else          pick_kf(std::integral_constant<int, 1024>{});
}

void ggml_cuda_prefill_launch_norm_mmq(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        void * vy, int ds_layout, bool swizzle_iq4, int64_t n_rows, int K, int K_padded, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, cudaStream_t stream) {
    GGML_UNUSED(w_ne1);
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto ds_tag, auto sw_tag, auto add_tag, auto wf_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        auto * kfn = k_prefill_norm_mmq<BS, decltype(ds_tag)::value, decltype(sw_tag)::value, decltype(add_tag)::value, decltype(wf_tag)::value>;
        if (smem > 48u * 1024u) {
            CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        }
        kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, vy, n_rows, K, K_padded, x_s1, r_s1, eps);
    };
    auto pick_wf = [&](auto bs_tag, auto ds_tag, auto sw_tag, auto add_tag) {
        if (write_f32) launch(bs_tag, ds_tag, sw_tag, add_tag, std::true_type{});
        else           launch(bs_tag, ds_tag, sw_tag, add_tag, std::false_type{});
    };
    auto pick_add = [&](auto bs_tag, auto ds_tag, auto sw_tag) {
        if (r != nullptr) pick_wf(bs_tag, ds_tag, sw_tag, std::true_type{});
        else              pick_wf(bs_tag, ds_tag, sw_tag, std::false_type{});
    };
    auto pick_ds = [&](auto bs_tag) {
        if (swizzle_iq4) {
            pick_add(bs_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::true_type{});
        } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            pick_add(bs_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_DS4>{}, std::false_type{});
        } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            pick_add(bs_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D2S6>{}, std::false_type{});
        } else {
            pick_add(bs_tag, std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::false_type{});
        }
    };
    if (K < 1024) pick_ds(std::integral_constant<int, 256>{});
    else          pick_ds(std::integral_constant<int, 1024>{});
}

// =================================================================================================
// BF16 Standalone Converters & Tracker
// =================================================================================================

void ggml_cuda_prefill_bf16_to_ai4(const nv_bfloat16 * x, int64_t rows, int K, int xsh, int variant,
        int8_t * q, float * s, cudaStream_t stream) {
    const int64_t ns = (int64_t) (K >> (7 + xsh));
    for (int64_t r0 = 0; r0 < rows; r0 += 65535) {
        const int64_t nr = std::min<int64_t>(65535, rows - r0);
        const dim3 grid(xsh ? (unsigned) ((K / RQ4_HB + 3) / 4) : (unsigned) ((K / 128 + 7) / 8), (unsigned) nr);
        pf_dispatch_ai4_variants(variant, xsh, [&](auto xsh_tag, auto sw_tag, auto xp_tag, auto ln_tag) {
            k_prefill_bf16_to_ai4<decltype(xsh_tag)::value, decltype(sw_tag)::value, decltype(xp_tag)::value, decltype(ln_tag)::value>
                <<<grid, 128, 0, stream>>>(x + r0 * K, q + r0 * K, s + r0 * ns, K, nr);
        });
    }
}

void ggml_cuda_prefill_bf16_to_nvfp4pf(const nv_bfloat16 * x, int64_t rows, int K, int kf,
        int8_t * xq, float * xs, cudaStream_t stream) {
    const dim3 grid((unsigned) ((K / kf + 3) / 4), (unsigned) std::min<int64_t>(rows, 65535));
    if (kf == 32) {
        k_prefill_bf16_to_nvfp4pf<32><<<grid, 128, 0, stream>>>(x, xq, xs, K, rows);
    } else {
        k_prefill_bf16_to_nvfp4pf<64><<<grid, 128, 0, stream>>>(x, xq, xs, K, rows);
    }
}

void ggml_cuda_prefill_bf16_to_mmq(const nv_bfloat16 * x, const int32_t * ids, void * vy, ggml_type w_type,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        bool scatter, int n_expert_used, bool swizzle_iq4, cudaStream_t stream) {
    const int ds_layout = (int) mmq_get_q8_1_ds_layout(w_type);
    const int64_t by = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 grid((unsigned) ne1, (unsigned) by, scatter ? 1u : (unsigned) (ne2 * ne3));
    auto go = [&](auto ds_tag, auto sc_tag, auto sw_tag) {
        k_prefill_bf16_to_mmq<decltype(ds_tag)::value, decltype(sc_tag)::value, decltype(sw_tag)::value>
            <<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, (int) ne1, (int) ne2, n_expert_used);
    };
    auto pick_sc = [&](auto ds_tag, auto sw_tag) {
        if (scatter) go(ds_tag, std::true_type{}, sw_tag);
        else         go(ds_tag, std::false_type{}, sw_tag);
    };
    if (swizzle_iq4) {
        pick_sc(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::true_type{});
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        pick_sc(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_DS4>{}, std::false_type{});
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        pick_sc(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D2S6>{}, std::false_type{});
    } else {
        pick_sc(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::false_type{});
    }
}

void ggml_cuda_prefill_bf16_to_f32(const nv_bfloat16 * src, float * dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    k_prefill_bf16_to_f32<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(src, dst, n);
}

void ggml_cuda_prefill_bf16_mark(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst || !dst->data) return;
    for (auto & e : ctx.prefill_bf16_acts) {
        if (e.tensor == dst && e.data == dst->data) {
            e.nbytes_f32 = ggml_nbytes(dst);
            e.epoch      = ctx.graph_epoch;
            return;
        }
    }
    ctx.prefill_bf16_acts.push_back({ dst, dst->data, ggml_nbytes(dst), ctx.graph_epoch });
}

bool ggml_cuda_prefill_bf16_is_active(const ggml_backend_cuda_context & ctx, const ggml_tensor * src, int64_t * elem_off_out) {
    if (!ggml_cuda_prefill_bf16_act_on() || !src || ctx.prefill_bf16_acts.empty()) {
        if (elem_off_out) *elem_off_out = 0;
        return false;
    }
    int64_t byte_off = 0;
    const ggml_tensor * base = pf_unwrap_views(src, &byte_off);
    if (!elem_off_out && (byte_off != 0 || !ggml_is_contiguous(src))) {
        return false;
    }
    if (elem_off_out) {
        *elem_off_out = byte_off / (int64_t) sizeof(float);
    }
    for (const auto & e : ctx.prefill_bf16_acts) {
        if (e.epoch == ctx.graph_epoch && e.tensor == base && e.data == base->data) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_prefill_bf16_want_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_prefill_bf16_act_on() || !cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes ||
            ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const ggml_tensor * dst = cgraph->nodes[node_idx];
    if (!dst || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            (dst->flags & GGML_TENSOR_FLAG_OUTPUT) || ((uintptr_t) dst->data & 15) != 0) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const int64_t n_tok = dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2]->ne[1] : dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (n_tok < 64 || dst->ne[0] < 1024 || dst->ne[0] % 128 != 0) {
        return false;
    }
    // Check whether producer backend supports native BF16 epilogue
    const ggml_tensor * w = dst->src[0];
    const bool is_id = dst->op == GGML_OP_MUL_MAT_ID;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool producer_ok =
        (ggml_ai4_rate(w->type) != 0 && (!is_id ? ggml_cuda_ai4_supported(dst) : (ggml_cuda_ai4_supported(dst) && !ggml_cuda_ai4_id_needs_sync(cc, w->type)))) ||
        (!is_id && ggml_cuda_aif25_supported(dst)) ||
        (is_id && w->type == GGML_TYPE_NVFP4 && ggml_cuda_nvfp4pf_mode() > 0 && n_tok >= 512 && (cc == GGML_CUDA_CC_AMPERE || cc == 860)) ||
        (ggml_is_quantized(w->type) && w->type != GGML_TYPE_TQ4_1S && w->type != GGML_TYPE_TQ3_1S) ||
        (!is_id && w->type == GGML_TYPE_BF16);
    if (!producer_ok) {
        return false;
    }

    // Verify EVERY consumer of `dst` (directly or via VIEW) is GLU or an Inkling SITU subgraph
    bool saw_glu = false;
    int situ_start = -1;
    int situ_end   = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!node->src[s]) continue;
            if (node->src[s] == dst || pf_unwrap_views(node->src[s]) == dst) {
                if (pf_is_view_or_noop(node)) {
                    continue;
                }
                if (node->op == GGML_OP_GLU && pf_map_glu_op(ggml_get_glu_op(node)) >= 0) {
                    saw_glu = true;
                } else if (j >= situ_start && j <= situ_end) {
                    saw_glu = true;
                } else {
                    const int last_s = ggml_cuda_ink_situ_spans(const_cast<ggml_cgraph *>(cgraph), j, dst);
                    if (last_s >= j) {
                        situ_start = j;
                        situ_end   = last_s;
                        saw_glu = true;
                    } else {
                        return false;
                    }
                }
            }
        }
    }
    return saw_glu;
}

// =================================================================================================
// MMQ_ID Deferred Consumer Hook (runs inside mmq.cu after ids_helper builds ids_src1)
// =================================================================================================

bool ggml_cuda_prefill_glu_mmq_id_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        const int32_t * ids_src1, void * vy, ggml_type w_type, int64_t ne10, int64_t ne10_padded,
        int64_t ne11_flat, int64_t ne12_flat, int64_t ne13_flat, cudaStream_t stream) {
    auto & d = ctx.prefill_glu_mmq_id;
    if (!d.active || d.epoch != ctx.graph_epoch || d.dst != src1 || ne12_flat != 1 || ne13_flat != 1 ||
            ne10 != d.dst->ne[0] || !ids_src1) {
        return false;
    }
    ink_situ_params situ{};
    std::memcpy(&situ, d.situ_params, sizeof(situ));
    const int dsl = (int) mmq_get_q8_1_ds_layout(w_type);
    ggml_cuda_prefill_launch_glu_mmq(
        d.glu_op, d.g_ptr, d.u_ptr, nullptr, ids_src1, vy,
        dsl, false, (int) ne10, (int) ne10_padded, ne11_flat, d.ne1,
        d.g_nb1, d.g_nb2, d.u_nb1, d.u_nb2,
        d.src_bf16, PF_WRITE_NONE, situ, d.glu_limit, d.oai_alpha, d.oai_limit, stream);
    d.active = false;
    ctx.fusion_stats.prefill_norm_actq_hits++;
    return true;
}

// =================================================================================================
// Shared GLU / SITU Execution Core + Graph Matchers
// =================================================================================================

static bool pf_exec_glu_or_situ(
        ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int prod_idx,
        int glu_kind, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit) {
    const bool fuse_actq = ggml_cuda_prefill_fuse_glu_actq_on();
    const bool bf16_act  = ggml_cuda_prefill_bf16_act_on();
    if (!fuse_actq && !bf16_act) {
        return false;
    }
    if (ctx.curr_stream_no != 0 || !ctx.stream_context().concurrent_events.empty() || ctx.scratch_borrowed) {
        return false;
    }
    if (!out || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[3] != 1) {
        return false;
    }
    if (!pf_cuda_buf(g, ctx.device) || !pf_cuda_buf(u, ctx.device) || !pf_cuda_buf(out, ctx.device)) {
        return false;
    }

    int64_t g_off_el = 0, u_off_el = 0;
    const bool g_bf16 = ggml_cuda_prefill_bf16_is_active(ctx, g, &g_off_el);
    const bool u_bf16 = ggml_cuda_prefill_bf16_is_active(ctx, u, &u_off_el);
    if (g_bf16 != u_bf16) {
        return false;
    }
    const bool src_bf16 = g_bf16;
    const int64_t elem_sz = sizeof(float);
    if (g->nb[0] != elem_sz || u->nb[0] != elem_sz ||
            (g->nb[1] | g->nb[2] | u->nb[1] | u->nb[2]) % elem_sz != 0) {
        return false;
    }
    const int K = (int) out->ne[0];
    const int64_t rows = out->ne[1] * out->ne[2];
    const int64_t ne1  = out->ne[1];
    const int64_t n_tok_logical = out->ne[2] > 1 ? out->ne[2] : out->ne[1];
    if (K < 128 || K % 128 != 0 || ((uintptr_t) out->data & 15) != 0) {
        return false;
    }
    const char * g_ptr = (const char *) (src_bf16 ? ((const nv_bfloat16 *) pf_unwrap_views(g)->data + g_off_el) : (const void *) g->data);
    const char * u_ptr = (const char *) (src_bf16 ? ((const nv_bfloat16 *) pf_unwrap_views(u)->data + u_off_el) : (const void *) u->data);
    if (((uintptr_t) g_ptr & 15) != 0 || ((uintptr_t) u_ptr & 15) != 0) {
        return false;
    }

    std::vector<int> cidxs;
    std::vector<pf_consumer_route> croutes;
    bool all_same = false, view_leak = false;
    pf_collect_consumers(ctx, cgraph, prod_idx, out, cidxs, croutes, all_same, view_leak);

    // Check BF16 output eligibility if ActQ doesn't consume it exclusively (or if #1 is OFF & #3 is ON)
    bool want_bf16_out = false;
    if (bf16_act && !view_leak && !cidxs.empty() && n_tok_logical >= 64 && K >= 1024 && !(out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        bool all_support_bf16 = true;
        for (const auto & cr : croutes) {
            if (!cr.valid && !cr.cublas_bf16) {
                all_support_bf16 = false;
                break;
            }
        }
        want_bf16_out = all_support_bf16;
    }

    // Attempt ActQ fusion when #1 is ON and prefill token/row count fits (decode <= 8 keeps INKSITUQ / MMVQ paths)
    const bool allow_actq = fuse_actq && n_tok_logical > 8 && !croutes.empty() && croutes[0].valid && all_same;
    const cudaStream_t stream = ctx.stream();

    if (allow_actq && croutes[0].mmq_id && cidxs.size() == 1 && !view_leak && !(out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        // Verify neither g nor u is written before consumer node cidxs[0]
        const int cons_i = cidxs[0];
        bool safe_deferred = !pf_overlap(out, g) && !pf_overlap(out, u);
        for (int j = prod_idx + 1; safe_deferred && j <= cons_i; ++j) {
            const ggml_tensor * nj = cgraph->nodes[j];
            if (j < cons_i && !pf_is_view_or_noop(nj) && (pf_overlap(nj, g) || pf_overlap(nj, u))) {
                safe_deferred = false;
            }
            if (j == cons_i && (pf_overlap(nj, g) || pf_overlap(nj, u))) {
                safe_deferred = false;
            }
        }
        if (safe_deferred) {
            auto & d = ctx.prefill_glu_mmq_id;
            d.active    = true;
            d.epoch     = ctx.graph_epoch;
            d.dst       = out;
            d.g         = pf_unwrap_views(g);
            d.u         = pf_unwrap_views(u);
            d.g_ptr     = g_ptr;
            d.u_ptr     = u_ptr;
            d.glu_op    = glu_kind;
            d.glu_limit = glu_limit;
            d.oai_alpha = oai_alpha;
            d.oai_limit = oai_limit;
            std::memcpy(d.situ_params, &situ, sizeof(situ));
            d.ne1       = ne1;
            d.ne2       = out->ne[2];
            d.g_nb1     = (int64_t) g->nb[1];
            d.g_nb2     = (int64_t) g->nb[2];
            d.u_nb1     = (int64_t) u->nb[1];
            d.u_nb2     = (int64_t) u->nb[2];
            d.src_bf16  = src_bf16;
            ctx.fusion_stats.prefill_glu_actq++;
            ctx.fusion_stats.prefill_glu_actq_elided++;
            if (src_bf16) ctx.fusion_stats.prefill_bf16_glu++;
            return true;
        }
    }

    if (allow_actq && !croutes[0].mmq_id) {
        const pf_consumer_route & cr = croutes[0];
        const bool can_elide = !view_leak && !(out->flags & GGML_TENSOR_FLAG_OUTPUT);
        const int rem_uses   = (int) cidxs.size();
        const int write_mode = can_elide ? PF_WRITE_NONE : (want_bf16_out ? PF_WRITE_BF16 : PF_WRITE_F32);

        if (cr.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4) {
            const int xsh = cr.param & 1;
            const int var = cr.param >> 1;
            int8_t * q = nullptr;
            float  * s = nullptr;
            if (ggml_cuda_prefill_ai4_alloc(ctx, out, rows, K, xsh, var, rem_uses, q, s)) {
                ggml_cuda_prefill_launch_glu_ai4(
                    glu_kind, g_ptr, u_ptr, out->data, q, s, K, rows, ne1,
                    (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
                    src_bf16, write_mode, xsh, var, situ, glu_limit, oai_alpha, oai_limit, stream);
                if (write_mode == PF_WRITE_BF16) ggml_cuda_prefill_bf16_mark(ctx, out);
                ctx.fusion_stats.prefill_glu_actq++;
                if (can_elide) ctx.fusion_stats.prefill_glu_actq_elided++;
                if (src_bf16 || write_mode == PF_WRITE_BF16) ctx.fusion_stats.prefill_bf16_glu++;
                return true;
            }
        } else if (cr.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
            const int kf = cr.param;
            int8_t * xq = nullptr;
            float  * xs = nullptr;
            if (ggml_cuda_prefill_nvfp4_alloc(ctx, out, rows, K, kf, rem_uses, xq, xs)) {
                ggml_cuda_prefill_launch_glu_nvfp4pf(
                    glu_kind, g_ptr, u_ptr, out->data, xq, xs, K, rows, ne1,
                    (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
                    src_bf16, write_mode, kf, situ, glu_limit, oai_alpha, oai_limit, stream);
                if (write_mode == PF_WRITE_BF16) ggml_cuda_prefill_bf16_mark(ctx, out);
                ctx.fusion_stats.prefill_glu_actq++;
                if (can_elide) ctx.fusion_stats.prefill_glu_actq_elided++;
                if (src_bf16 || write_mode == PF_WRITE_BF16) ctx.fusion_stats.prefill_bf16_glu++;
                return true;
            }
        } else if (cr.kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ && out->ne[2] == 1) {
            const int dsl = cr.param & 0xf;
            const bool swz = (cr.param & 0x10) != 0;
            const int K_pad = (int) GGML_PAD(K, 512);
            char * q8 = nullptr;
            if (ggml_cuda_prefill_mmq_alloc(ctx, out, rows, K, K_pad, dsl, swz, rem_uses, q8)) {
                ggml_cuda_prefill_launch_glu_mmq(
                    glu_kind, g_ptr, u_ptr, out->data, nullptr, q8, dsl, swz,
                    K, K_pad, rows, ne1,
                    (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
                    src_bf16, write_mode, situ, glu_limit, oai_alpha, oai_limit, stream);
                if (write_mode == PF_WRITE_BF16) ggml_cuda_prefill_bf16_mark(ctx, out);
                ctx.fusion_stats.prefill_glu_actq++;
                if (can_elide) ctx.fusion_stats.prefill_glu_actq_elided++;
                if (src_bf16 || write_mode == PF_WRITE_BF16) ctx.fusion_stats.prefill_bf16_glu++;
                return true;
            }
        }
    }

    // Pillar #3 fallback handled by caller pf_exec_glu_or_situ_full
    return false;
}

template <typename SRC_T, typename DST_T>
static __global__ void __launch_bounds__(256) k_prefill_glu_elem(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, DST_T * __restrict__ out,
        int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        ink_situ_params situ, float glu_limit, float oai_alpha, float oai_limit) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) {
        return;
    }
    for (int64_t tok = blockIdx.y; tok < rows; tok += gridDim.y) {
        const int64_t i1 = tok % ne1;
        const int64_t i2 = tok / ne1;
        const SRC_T * gr = (const SRC_T *) (g + (i2 * g_nb2 + i1 * g_nb1) / (sizeof(float) / sizeof(SRC_T)));
        const SRC_T * ur = (const SRC_T *) (u + (i2 * u_nb2 + i1 * u_nb1) / (sizeof(float) / sizeof(SRC_T)));
        const float val = pf_eval_glu_dev(glu_kind, pf_to_f32(gr[idx]), pf_to_f32(ur[idx]), situ, glu_limit, oai_alpha, oai_limit);
        out[tok * K + idx] = pf_from_f32<DST_T>(val);
    }
}

static bool pf_exec_glu_or_situ_full(
        ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int prod_idx,
        int glu_kind, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit) {
    if (pf_exec_glu_or_situ(ctx, cgraph, prod_idx, glu_kind, g, u, out, situ, glu_limit, oai_alpha, oai_limit)) {
        return true;
    }
    if (!ggml_cuda_prefill_bf16_act_on()) {
        return false;
    }
    if (!out || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[3] != 1) {
        return false;
    }
    int64_t g_off_el = 0, u_off_el = 0;
    const bool g_bf16 = ggml_cuda_prefill_bf16_is_active(ctx, g, &g_off_el);
    const bool u_bf16 = ggml_cuda_prefill_bf16_is_active(ctx, u, &u_off_el);
    if (g_bf16 != u_bf16) {
        return false;
    }
    const bool src_bf16 = g_bf16;
    const int K = (int) out->ne[0];
    const int64_t rows = out->ne[1] * out->ne[2];
    const int64_t ne1  = out->ne[1];
    const int64_t n_tok_logical = out->ne[2] > 1 ? out->ne[2] : out->ne[1];

    std::vector<int> cidxs;
    std::vector<pf_consumer_route> croutes;
    bool all_same = false, view_leak = false;
    pf_collect_consumers(ctx, cgraph, prod_idx, out, cidxs, croutes, all_same, view_leak);
    bool want_bf16_out = false;
    if (!view_leak && !cidxs.empty() && n_tok_logical >= 64 && K >= 1024 && K % 128 == 0 && !(out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        want_bf16_out = true;
        for (const auto & cr : croutes) {
            if (!cr.valid && !cr.cublas_bf16) {
                want_bf16_out = false;
                break;
            }
        }
    }
    if (!src_bf16 && !want_bf16_out) {
        return false;
    }
    if (g->nb[0] != sizeof(float) || u->nb[0] != sizeof(float) ||
            (g->nb[1] | g->nb[2] | u->nb[1] | u->nb[2]) % sizeof(float) != 0) {
        return false;
    }
    const char * g_ptr = (const char *) (src_bf16 ? ((const nv_bfloat16 *) pf_unwrap_views(g)->data + g_off_el) : (const void *) g->data);
    const char * u_ptr = (const char *) (src_bf16 ? ((const nv_bfloat16 *) pf_unwrap_views(u)->data + u_off_el) : (const void *) u->data);
    const dim3 grid((unsigned) ((K + 255) / 256), (unsigned) std::min<int64_t>(rows, 65535));
    const cudaStream_t stream = ctx.stream();
    if (src_bf16 && want_bf16_out) {
        k_prefill_glu_elem<nv_bfloat16, nv_bfloat16><<<grid, 256, 0, stream>>>(
            glu_kind, g_ptr, u_ptr, (nv_bfloat16 *) out->data, K, rows, ne1,
            (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
            situ, glu_limit, oai_alpha, oai_limit);
    } else if (src_bf16) {
        k_prefill_glu_elem<nv_bfloat16, float><<<grid, 256, 0, stream>>>(
            glu_kind, g_ptr, u_ptr, (float *) out->data, K, rows, ne1,
            (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
            situ, glu_limit, oai_alpha, oai_limit);
    } else {
        k_prefill_glu_elem<float, nv_bfloat16><<<grid, 256, 0, stream>>>(
            glu_kind, g_ptr, u_ptr, (nv_bfloat16 *) out->data, K, rows, ne1,
            (int64_t) g->nb[1], (int64_t) g->nb[2], (int64_t) u->nb[1], (int64_t) u->nb[2],
            situ, glu_limit, oai_alpha, oai_limit);
    }
    if (want_bf16_out) {
        ggml_cuda_prefill_bf16_mark(ctx, out);
    }
    ctx.fusion_stats.prefill_bf16_glu++;
    return true;
}

bool ggml_cuda_prefill_situ_launch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out, const ink_situ_params & p) {
    return pf_exec_glu_or_situ_full(ctx, cgraph, last, PF_GLU_SITU, g, u, out, p, 0.0f, 0.0f, 0.0f);
}

int ggml_cuda_try_fuse_prefill_glu(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (!ctx || (!ggml_cuda_prefill_fuse_glu_actq_on() && !ggml_cuda_prefill_bf16_act_on())) {
        return 0;
    }
    ggml_tensor * glu = cgraph->nodes[i];
    if (glu->op != GGML_OP_GLU || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || glu->ne[3] != 1) {
        return 0;
    }
    const int kind = pf_map_glu_op(ggml_get_glu_op(glu));
    if (kind < 0) {
        return 0;
    }
    const ggml_tensor * src0 = glu->src[0];
    const ggml_tensor * src1 = glu->src[1];
    if (!src0 || !ggml_is_contiguous_1(src0) || src0->type != GGML_TYPE_F32 || src0->ne[3] != 1) {
        return 0;
    }
    if (src1 && (!ggml_is_contiguous_1(src1) || src1->type != GGML_TYPE_F32 || src1->ne[0] != glu->ne[0])) {
        return 0;
    }
    const int32_t swapped = ((const int32_t *) glu->op_params)[1];
    const float limit     = ggml_get_op_params_f32(glu, 2);
    float oai_alpha = 1.702f, oai_limit = 7.0f;
    if (kind == PF_GLU_SWIGLU_OAI) {
        oai_alpha = ggml_get_op_params_f32(glu, 2);
        oai_limit = ggml_get_op_params_f32(glu, 3);
    }

    // If unsplit src0 (src1 == nullptr), create stack descriptors pointing to the gate/up halves cleanly without mutating graph
    ggml_tensor g_tmp = *src0;
    ggml_tensor u_tmp = src1 ? *src1 : *src0;
    const ggml_tensor * g_ptr_t = src1 ? (swapped ? src1 : src0) : &g_tmp;
    const ggml_tensor * u_ptr_t = src1 ? (swapped ? src0 : src1) : &u_tmp;
    if (!src1) {
        const size_t half_bytes = (size_t) glu->ne[0] * sizeof(float);
        const ggml_tensor * src0_base = pf_unwrap_views(src0);
        int64_t src0_boff = 0;
        pf_unwrap_views(src0, &src0_boff);
        g_tmp.ne[0] = glu->ne[0];
        u_tmp.ne[0] = glu->ne[0];
        g_tmp.op = GGML_OP_VIEW;
        g_tmp.view_src = const_cast<ggml_tensor *>(src0_base);
        u_tmp.op = GGML_OP_VIEW;
        u_tmp.view_src = const_cast<ggml_tensor *>(src0_base);
        if (!swapped) {
            g_tmp.view_offs = (size_t) src0_boff;
            u_tmp.data      = (void *) ((char *) src0->data + half_bytes);
            u_tmp.view_offs = (size_t) src0_boff + half_bytes;
        } else {
            u_tmp.view_offs = (size_t) src0_boff;
            g_tmp.data      = (void *) ((char *) src0->data + half_bytes);
            g_tmp.view_offs = (size_t) src0_boff + half_bytes;
        }
    }
    ink_situ_params dummy{};
    if (pf_exec_glu_or_situ_full(*ctx, cgraph, i, kind, g_ptr_t, u_ptr_t, glu, dummy, limit, oai_alpha, oai_limit)) {
        return 1; // consumed node i (GLU)
    }
    return 0;
}

// =================================================================================================
// Fused Prefill RMSNorm (+ ADD) -> ActQ Matcher
// =================================================================================================

int ggml_cuda_try_fuse_prefill_norm(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (!ctx || !ggml_cuda_prefill_fuse_norm_actq_on() || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    if (ctx->curr_stream_no != 0 || !ctx->stream_context().concurrent_events.empty() || ctx->scratch_borrowed) {
        return 0;
    }

    ggml_tensor * add_node  = nullptr;
    ggml_tensor * norm_node = cgraph->nodes[i];
    int mul_idx = i + 1;

    if (norm_node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RMS_NORM) {
        add_node  = norm_node;
        norm_node = cgraph->nodes[i + 1];
        mul_idx   = i + 2;
        if (norm_node->src[0] != add_node || (add_node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        // Only handle residual ADD when add_node shapes match cleanly without broadcasting
        if (!add_node->src[0] || !add_node->src[1] ||
                !ggml_are_same_shape(add_node->src[0], add_node) || !ggml_are_same_shape(add_node->src[1], add_node) ||
                add_node->type != GGML_TYPE_F32 || add_node->src[0]->type != GGML_TYPE_F32 || add_node->src[1]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(add_node) || !ggml_is_contiguous(add_node->src[0]) || !ggml_is_contiguous(add_node->src[1])) {
            return 0;
        }
    }

    ggml_tensor * mul_node = cgraph->nodes[mul_idx];
    if (norm_node->op != GGML_OP_RMS_NORM || mul_node->op != GGML_OP_MUL ||
            ggml_node_get_use_count(cgraph, mul_idx - 1) != 1 || (norm_node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * x = add_node ? add_node->src[0] : norm_node->src[0];
    const ggml_tensor * r = add_node ? add_node->src[1] : nullptr;
    const ggml_tensor * w = mul_node->src[0] == norm_node ? mul_node->src[1] :
                            mul_node->src[1] == norm_node ? mul_node->src[0] : nullptr;
    if (!w || !x || norm_node->type != GGML_TYPE_F32 || mul_node->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, norm_node) || !ggml_are_same_shape(norm_node, mul_node) ||
            !ggml_is_contiguous(mul_node) || !ggml_is_contiguous(w) ||
            w->ne[0] != norm_node->ne[0] || ggml_nrows(w) != 1 ||
            x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous_2(x) || x->ne[3] != 1) {
        return 0;
    }
    if (add_node && !ggml_is_contiguous(x)) {
        return 0;
    }
    const int K = (int) mul_node->ne[0];
    const int64_t n_rows = ggml_nrows(mul_node);
    // SMEM holds one row of float[K] (up to 96 KiB -> K <= 24576 on Ampere SM 8.6); only prefill N >= 16
    if (K < 128 || K % 128 != 0 || K > 24576 || n_rows < 16 || ((uintptr_t) mul_node->data & 15) != 0 ||
            !pf_cuda_buf(mul_node, ctx->device) || !pf_cuda_buf(x, ctx->device) || !pf_cuda_buf(w, ctx->device)) {
        return 0;
    }
    if (pf_overlap(mul_node, w) ||
            (mul_node->data != x->data && pf_overlap(mul_node, x)) ||
            (r && (pf_overlap(mul_node, r) || (add_node->data != r->data && pf_overlap(add_node, r)) ||
                   (add_node->data != x->data && pf_overlap(add_node, x)) || pf_overlap(add_node, mul_node)))) {
        return 0;
    }

    float eps = 0.0f;
    std::memcpy(&eps, norm_node->op_params, sizeof(float));

    std::vector<int> cidxs;
    std::vector<pf_consumer_route> croutes;
    bool all_same = false, view_leak = false;
    if (!pf_collect_consumers(*ctx, cgraph, mul_idx, mul_node, cidxs, croutes, all_same, view_leak)) {
        return 0;
    }
    // Check if there is at least one valid prefill quantized consumer (all sharing same format OR first valid)
    const pf_consumer_route * tgt = nullptr;
    int match_uses = 0;
    for (const auto & cr : croutes) {
        if (cr.valid && !cr.mmq_id) {
            if (!tgt) tgt = &cr;
            if (cr.kind == tgt->kind && cr.param == tgt->param && !cr.mmq_id) {
                match_uses++;
            }
        }
    }
    if (!tgt || match_uses == 0) {
        return 0;
    }

    const bool write_f32 = view_leak || !all_same || (match_uses != (int) croutes.size()) || (mul_node->flags & GGML_TENSOR_FLAG_OUTPUT);
    const int64_t x_s1 = (int64_t) (x->nb[1] / sizeof(float));
    const int64_t r_s1 = r ? (int64_t) (r->nb[1] / sizeof(float)) : 0;
    const cudaStream_t stream = ctx->stream();

    if (tgt->kind == ggml_backend_cuda_context::PREFILL_ACT_AI4) {
        const int xsh = tgt->param & 1;
        const int var = tgt->param >> 1;
        int8_t * q = nullptr;
        float  * s = nullptr;
        if (!ggml_cuda_prefill_ai4_alloc(*ctx, mul_node, n_rows, K, xsh, var, match_uses, q, s)) {
            return 0;
        }
        ggml_cuda_prefill_launch_norm_ai4(
            (const float *) x->data, r ? (const float *) r->data : nullptr,
            add_node ? (float *) add_node->data : nullptr,
            (const float *) w->data, 1, (float *) mul_node->data,
            q, s, n_rows, K, x_s1, r_s1, eps, write_f32, xsh, var, stream);
    } else if (tgt->kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
        const int kf = tgt->param;
        int8_t * xq = nullptr;
        float  * xs = nullptr;
        if (!ggml_cuda_prefill_nvfp4_alloc(*ctx, mul_node, n_rows, K, kf, match_uses, xq, xs)) {
            return 0;
        }
        ggml_cuda_prefill_launch_norm_nvfp4pf(
            (const float *) x->data, r ? (const float *) r->data : nullptr,
            add_node ? (float *) add_node->data : nullptr,
            (const float *) w->data, 1, (float *) mul_node->data,
            xq, xs, n_rows, K, x_s1, r_s1, eps, write_f32, kf, stream);
    } else if (tgt->kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ && mul_node->ne[2] == 1) {
        const int dsl = tgt->param & 0xf;
        const bool swz = (tgt->param & 0x10) != 0;
        const int K_pad = (int) GGML_PAD(K, 512);
        char * q8 = nullptr;
        if (!ggml_cuda_prefill_mmq_alloc(*ctx, mul_node, n_rows, K, K_pad, dsl, swz, match_uses, q8)) {
            return 0;
        }
        ggml_cuda_prefill_launch_norm_mmq(
            (const float *) x->data, r ? (const float *) r->data : nullptr,
            add_node ? (float *) add_node->data : nullptr,
            (const float *) w->data, 1, (float *) mul_node->data,
            q8, dsl, swz, n_rows, K, K_pad, x_s1, r_s1, eps, write_f32, stream);
    } else {
        return 0;
    }

    ctx->fusion_stats.prefill_norm_actq++;
    if (!write_f32) {
        ctx->fusion_stats.prefill_norm_actq_elided++;
    }
    return mul_idx - i; // additional nodes after i skipped by eval_graph
}

bool ggml_cuda_prefill_try_mix_norm_actq(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        ggml_tensor * mix, const ggml_tensor * w, ggml_tensor * mul, float eps,
        int & out_kind, void ** out0, void ** out1, bool & write_f32, const ggml_tensor ** out_mm) {
    GGML_UNUSED(mix);
    GGML_UNUSED(w);
    GGML_UNUSED(eps);
    out_kind = ggml_backend_cuda_context::PREFILL_ACT_NONE;
    *out0 = nullptr;
    *out1 = nullptr;
    write_f32 = true;
    if (out_mm) *out_mm = nullptr;

    if (!ggml_cuda_prefill_fuse_norm_actq_on() || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return false;
    }
    const int K = (int) mul->ne[0];
    const int64_t n_rows = ggml_nrows(mul);
    if (K < 128 || K % 128 != 0 || K > 4096 || ((uintptr_t) mul->data & 15) != 0) {
        return false;
    }
    std::vector<int> cidxs;
    std::vector<pf_consumer_route> croutes;
    bool all_same = false, view_leak = false;
    if (!pf_collect_consumers(ctx, cgraph, last, mul, cidxs, croutes, all_same, view_leak)) {
        return false;
    }
    const pf_consumer_route * tgt = nullptr;
    int match_uses = 0;
    for (size_t idx = 0; idx < croutes.size(); ++idx) {
        const auto & cr = croutes[idx];
        if (cr.valid && !cr.mmq_id && (cr.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4 || cr.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF)) {
            if (!tgt) {
                tgt = &cr;
                if (out_mm) *out_mm = cgraph->nodes[cidxs[idx]];
            }
            if (cr.kind == tgt->kind && cr.param == tgt->param) {
                match_uses++;
            }
        }
    }
    if (!tgt || match_uses == 0) {
        return false;
    }
    write_f32 = view_leak || !all_same || (match_uses != (int) croutes.size()) || (mul->flags & GGML_TENSOR_FLAG_OUTPUT);
    if (tgt->kind == ggml_backend_cuda_context::PREFILL_ACT_AI4 && (tgt->param >> 1) == 0 && K % 256 == 0) {
        int8_t * q = nullptr;
        float  * s = nullptr;
        if (!ggml_cuda_prefill_ai4_alloc(ctx, mul, n_rows, K, 1, 0, match_uses, q, s)) {
            return false;
        }
        out_kind = ggml_backend_cuda_context::PREFILL_ACT_AI4;
        *out0 = q;
        *out1 = s;
        return true;
    }
    if (tgt->kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
        int8_t * xq = nullptr;
        float  * xs = nullptr;
        if (!ggml_cuda_prefill_nvfp4_alloc(ctx, mul, n_rows, K, tgt->param, match_uses, xq, xs)) {
            return false;
        }
        out_kind = (tgt->param == 32) ? 232 : 264;
        *out0 = xq;
        *out1 = xs;
        return true;
    }
    return false;
}

void ggml_cuda_prefill_mix_norm_commit(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, int64_t n_rows, int K, int kind, int param, void * p0, void * p1) {
    GGML_UNUSED(mul);
    GGML_UNUSED(n_rows);
    GGML_UNUSED(K);
    GGML_UNUSED(kind);
    GGML_UNUSED(param);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    ctx.fusion_stats.prefill_mix_actq++;
}
