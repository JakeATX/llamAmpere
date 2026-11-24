// [PREFILL-GLUE] Prefill glue fusion: GLU / SITU -> AI+ / NVFP4PF / MMQ q8_1 activation, RMS_NORM(+ADD)*w -> the same.
// Opt-in (GGML_CUDA_PREFILL_FUSE_GLU_ACTQ / GGML_CUDA_PREFILL_FUSE_NORM_ACTQ, default 0). A fused producer always writes
// its normal f32 output as well; the quantized copy is only a cache a consumer matmul may take when its src1, shape and
// layout match exactly, so any consumer that does not take it reads the f32 tensor as before.

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

static __device__ __forceinline__ void pf_load8(const float * p, float (&dst)[8]) {
    const float4 a = *(const float4 *) p;
    const float4 b = *(const float4 *) (p + 4);
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
    dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
}

static __device__ __forceinline__ void pf_load4(const float * p, float (&dst)[4]) {
    const float4 a = *(const float4 *) p;
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
}

static __device__ __forceinline__ void pf_store8(float * yr, const float (&v)[8]) {
    *(float4 *) yr       = make_float4(v[0], v[1], v[2], v[3]);
    *(float4 *) (yr + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// =================================================================================================
// 1. Fused Prefill GLU/SITU -> AI+ (H256 / H128)
// =================================================================================================
template <int XSH, bool SWAP8, bool XPERM, bool LINEAR>
static __global__ void __launch_bounds__(128) k_prefill_glu_ai4(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, float * __restrict__ out,
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
            const float * gr = (const float *) (g + i2 * g_nb2 + i1 * g_nb1) + elem_off;
            const float * ur = (const float *) (u + i2 * u_nb2 + i1 * u_nb1) + elem_off;
            float gv[8], uv[8], v[8];
            pf_load8(gr, gv);
            pf_load8(ur, uv);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
            }
            pf_store8(out + tok * K + elem_off, v);
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
                const float * gr = (const float *) (g + i2 * g_nb2 + i1 * g_nb1) + elem_off;
                const float * ur = (const float *) (u + i2 * u_nb2 + i1 * u_nb1) + elem_off;
                float gv[8], uv[8];
                pf_load8(gr, gv);
                pf_load8(ur, uv);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
                }
                pf_store8(out + tok * K + elem_off, v);
            }
            ai4k::quantize_act128_regs<SWAP8, XPERM, LINEAR>(v, live, l, cb, q, s, K, (int) tok);
        }
    }
}

// =================================================================================================
// 2. Fused Prefill GLU/SITU -> NVFP4PF (KF = 64 or 32)
// =================================================================================================
template <int KF>
static __global__ void __launch_bounds__(128) k_prefill_glu_nvfp4pf(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, float * __restrict__ out,
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
        const float * gr = (const float *) (g + i2 * g_nb2 + i1 * g_nb1) + elem_off;
        const float * ur = (const float *) (u + i2 * u_nb2 + i1 * u_nb1) + elem_off;
        float v[V];
        bool bad = false;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            const float val = pf_eval_glu_dev(glu_kind, gr[i], ur[i], situ, glu_limit, oai_alpha, oai_limit);
            out[tok * K + elem_off + i] = val;
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

template <mmq_q8_1_ds_layout DS_LAYOUT, bool SWIZZLE_IQ4>
static __global__ void __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE_MMQ) k_prefill_glu_mmq(
        int glu_kind, const char * __restrict__ g, const char * __restrict__ u, float * __restrict__ out,
        void * __restrict__ vy,
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

    for (int64_t col = blockIdx.x; col < rows_dst; col += gridDim.x) {
        const int64_t tok = col;
        float xf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (i0 < K) {
            const int64_t i1 = tok % ne1;
            const int64_t i2 = tok / ne1;
            const float * gr = (const float *) (g + i2 * g_nb2 + i1 * g_nb1) + i0;
            const float * ur = (const float *) (u + i2 * u_nb2 + i1 * u_nb1) + i0;
            float gv[4], uv[4];
            pf_load4(gr, gv);
            pf_load4(ur, uv);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                xf[e] = pf_eval_glu_dev(glu_kind, gv[e], uv[e], situ, glu_limit, oai_alpha, oai_limit);
            }
            *(float4 *) (out + tok * K + i0) = make_float4(xf[0], xf[1], xf[2], xf[3]);
        }
        const int64_t ib = k_block * rows_dst + col;
        pf_emit_mmq_q8_1<DS_LAYOUT, SWIZZLE_IQ4>(xf, y, ib, iqs);
    }
}

// =================================================================================================
// 4. Fused Prefill RMSNorm (+ ADD) + MUL(gamma) -> Cached AI4 / NVFP4PF / MMQ via dynamic SMEM
//    Exact scalar reduction order matching norm.cu rms_norm_f32 (0 ULP vs unfused).
// =================================================================================================
template <int BS, bool HAS_ADD>
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
        mul_row[col] = val;
    }
    __syncthreads();
}

template <int BS, int XSH, bool SWAP8, bool XPERM, bool LINEAR, bool HAS_ADD>
static __global__ void __launch_bounds__(BS) k_prefill_norm_ai4(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        int8_t * __restrict__ q, float * __restrict__ s,
        int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    const int nwarps = BS / 32;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            mul_dst + row * K,
            s_norm, K, eps);
        pf_quantize_row_ai4_from_ptr<XSH, SWAP8, XPERM, LINEAR>(s_norm, q, s, K, (int) row, nwarps);
        __syncthreads();
    }
}

template <int BS, int KF, bool HAS_ADD>
static __global__ void __launch_bounds__(BS) k_prefill_norm_nvfp4pf(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        int8_t * __restrict__ xq, float * __restrict__ xs,
        int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    const int nwarps = BS / 32;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            mul_dst + row * K,
            s_norm, K, eps);
        pf_quantize_row_nvfp4pf_from_ptr<KF>(s_norm, xq, xs, K, (int) row, nwarps);
        __syncthreads();
    }
}

template <int BS, mmq_q8_1_ds_layout DS_LAYOUT, bool SWIZZLE_IQ4, bool HAS_ADD>
static __global__ void __launch_bounds__(BS) k_prefill_norm_mmq(
        const float * __restrict__ x, const float * __restrict__ r, float * __restrict__ add_dst,
        const float * __restrict__ w, float * __restrict__ mul_dst,
        void * __restrict__ vy,
        int64_t n_rows, int K, int K_padded, int64_t x_s1, int64_t r_s1, float eps) {
    extern __shared__ __align__(16) float s_norm[];
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        pf_rms_norm_to_smem<BS, HAS_ADD>(
            x + row * x_s1,
            HAS_ADD ? (r + row * r_s1) : nullptr,
            HAS_ADD ? (add_dst + row * K) : nullptr,
            w,
            mul_dst + row * K,
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


// one step is enough: ggml sets view_src to the root tensor and accumulates view_offs
static const ggml_tensor * pf_unwrap_views(const ggml_tensor * t, int64_t * byte_off = nullptr) {
    int64_t off = 0;
    if (t && t->view_src && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
                             t->op == GGML_OP_TRANSPOSE)) {
        off = (int64_t) t->view_offs;
        t   = t->view_src;
    }
    if (byte_off) {
        *byte_off = off;
    }
    return t;
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

static bool pf_producer_ok(const ggml_backend_cuda_context & ctx) {
    return ctx.curr_stream_no == 0 && !ctx.scratch_borrowed &&
           const_cast<ggml_backend_cuda_context &>(ctx).stream_context().concurrent_events.empty();
}

static bool pf_alloc_entry_buf(ggml_backend_cuda_context & ctx, ggml_backend_cuda_context::prefill_act_entry * e, size_t bytes) {
    if (e->ptr && e->dev == ctx.device && e->cap >= bytes) {
        return true;
    }
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(ctx.stream(), &cap_st) != cudaSuccess || cap_st != cudaStreamCaptureStatusNone) {
        return false; // never allocate raw CUDA memory during CUDA graph capture
    }
    if (e->ptr) {
        // a captured graph may still replay kernels that use the old buffer: keep it until teardown
        e->retired.push_back({ e->ptr, e->cap, e->dev });
        e->ptr = nullptr;
        e->cap = 0;
        e->dev = -1;
    }
    ggml_cuda_set_device(ctx.device);
    char * new_ptr = nullptr;
    if (cudaMalloc(&new_ptr, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    e->ptr = new_ptr;
    e->cap = bytes;
    e->dev = ctx.device;
    return true;
}

static void pf_clear(ggml_backend_cuda_context::prefill_act_entry & c) {
    c.src1     = nullptr;
    c.data     = nullptr;
    c.kind     = ggml_backend_cuda_context::PREFILL_ACT_NONE;
    c.rem_uses = 0;
}

// Claims an entry for src1's activation (reusing its buffer when it is large enough). An entry that is overwritten
// here simply stops matching, so its consumers quantize the f32 tensor themselves.
static ggml_backend_cuda_context::prefill_act_entry * pf_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        int kind, int param, int64_t n_rows, int K, int64_t k_padded, size_t bytes, size_t aux_off, int rem_uses) {
    if (!src1 || !src1->data || !pf_producer_ok(ctx)) {
        return nullptr;
    }
    auto & pc = ctx.prefill_act_cache;
    ggml_backend_cuda_context::prefill_act_entry * e = nullptr;
    for (auto & c : pc.entries) {
        if (c.src1 == src1 && c.dev == ctx.device) {
            e = &c;
            break;
        }
    }
    for (auto & c : pc.entries) {
        if (!e && (c.epoch != ctx.graph_epoch || c.kind == ggml_backend_cuda_context::PREFILL_ACT_NONE)) {
            e = &c;
        }
    }
    if (!e) {
        e = &pc.entries[0];
        for (auto & c : pc.entries) {
            if (c.last_use < e->last_use) {
                e = &c;
            }
        }
    }
    pf_clear(*e);
    if (!pf_alloc_entry_buf(ctx, e, bytes)) {
        return nullptr;
    }
    e->src1     = src1;
    e->data     = src1->data;
    e->epoch    = ctx.graph_epoch;
    e->n_rows   = n_rows;
    e->k        = K;
    e->k_padded = k_padded;
    e->kind     = kind;
    e->param    = param;
    e->aux_off  = aux_off;
    e->rem_uses = rem_uses;
    e->born     = ++pc.tick;
    e->last_use = e->born;
    return e;
}

static ggml_backend_cuda_context::prefill_act_entry * pf_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        int kind, int param, int64_t n_rows, int K, int64_t k_padded) {
    if (!ggml_cuda_prefill_any_on() || !src1 || !src1->data || ctx.curr_stream_no != 0 || ctx.scratch_borrowed) {
        return nullptr;
    }
    const ggml_tensor * base = pf_unwrap_views(src1);
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind != kind || c.epoch != ctx.graph_epoch || c.dev != ctx.device || c.data != src1->data ||
                c.n_rows != n_rows || c.k != K || c.k_padded != k_padded || c.param != param) {
            continue;
        }
        // the producer's own tensor, or a view of it that reads the very same bytes in the same layout
        const bool same = c.src1 == src1 ||
            (c.src1 == base && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[0] == K &&
             ggml_nrows(src1) == n_rows);
        if (!same) {
            continue;
        }
        c.last_use = ++ctx.prefill_act_cache.tick;
        ctx.fusion_stats.prefill_actq_hits++;
        if (c.rem_uses > 0 && --c.rem_uses == 0) {
            // the buffer stays valid for the launch the caller is about to enqueue; the entry just stops matching
            pf_clear(c);
        }
        return &c;
    }
    return nullptr;
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

// The activation layout a consumer matmul is expected to quantize src1 into. Only a guess about which entry to build:
// the consumer takes the entry only when its real dispatch asks for exactly this layout, else it reads the f32 tensor.
struct pf_consumer_route {
    int  kind  = ggml_backend_cuda_context::PREFILL_ACT_NONE;
    int  param = 0;
    bool valid = false;
};

static pf_consumer_route pf_inspect_consumer(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm) {
    pf_consumer_route r;
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (!src0 || !src1 || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || src0->ne[3] != 1) {
        return r;
    }
    const bool is_id = (mm->op == GGML_OP_MUL_MAT_ID);
    const int64_t K  = src0->ne[0];
    const int64_t M  = src0->ne[1];
    const int dev    = ctx.device;
    const int cc     = ggml_cuda_info().devices[dev].cc;
    if (!pf_cuda_buf(src0, dev) || !pf_cuda_buf(mm, dev) || ((uintptr_t) src1->data & 15) != 0 || K % 128 != 0 ||
            !ggml_is_contiguous(mm) || !ggml_is_contiguous(src1)) {
        return r;
    }
    const int xsh = (K % 256 == 0) ? 1 : 0;

    // AI+ dense / MoE
    if (ggml_ai4_rate(src0->type) != 0 && !is_id && ggml_cuda_ai4_supported(mm) && M % 16 == 0) {
        const int64_t n_rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        if (n_rows <= 65535 && src0->ne[2] == 1 && ggml_is_contiguous(src0)) {
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
        if (n_rows <= 65535 && src1->ne[3] == 1) {
            const int variant = ggml_cuda_ai4_prefill_variant(ctx, src0->type, (int) M, (int) n_rows, (int) K, true);
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_AI4;
            r.param = xsh | (variant << 1);
            r.valid = true;
            return r;
        }
    }
    // AIF* (linear AI+ activation layout)
    if ((!is_id && ggml_cuda_aif25_supported(mm)) || (is_id && ggml_cuda_aif_rows_supported(mm))) {
        if (src1->ne[1] * src1->ne[2] <= 65535) {
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_AI4;
            r.param = xsh | (3 << 1);
            r.valid = true;
            return r;
        }
    }
    // NVFP4PF (MoE >= 512 tokens on SM 8.0 / 8.6)
    if (is_id && src0->type == GGML_TYPE_NVFP4) {
        const ggml_tensor * ids = mm->src[2];
        const int KF = ggml_cuda_nvfp4pf_mode();
        if (KF > 0 && ids && ids->type == GGML_TYPE_I32 && (cc == GGML_CUDA_CC_AMPERE || cc == 860) &&
                M % 16 == 0 && src0->ne[2] <= NVFP4PF_MAX_EXPERTS && ids->ne[1] >= 512 &&
                src1->ne[1] * src1->ne[2] <= 65535 && src1->ne[3] == 1 &&
                src0->nb[0] == 36 && src0->nb[1] % 36 == 0 && src0->nb[1] >= (size_t) (K / 64) * 36 &&
                src0->nb[2] % 4 == 0 && (uintptr_t) src0->data % 4 == 0) {
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF;
            r.param = KF;
            r.valid = true;
        }
        return r;
    }
    // dense MMQ (quantized weights above the MMVQ width); MUL_MAT_ID reorders rows by expert, so it is never a target
#if !defined(BLACKWELL_MMA_AVAILABLE)
    if (!is_id && ggml_is_quantized(src0->type) && src0->type != GGML_TYPE_TQ4_1S && src0->type != GGML_TYPE_TQ3_1S &&
            src1->ne[2] == 1 && src1->ne[3] == 1) {
        const int64_t ncols = src1->ne[1];
        if (ncols > 8 && !ggml_cuda_should_use_mmvq(src0->type, cc, ncols) && ggml_cuda_should_use_mmq(src0->type, cc, ncols, 0) &&
                ggml_cuda_mmq_has_config(src0->type, M, cc, ggml_cuda_info().devices[dev].smpbo)) {
            mmq_args a{};
            a.x = (const char *) src0->data; a.type_x = src0->type;
            a.dst = (float *) mm->data; a.ncols_x = K; a.nrows_x = M;
            a.ncols_dst = ncols; a.stride_row_x = src0->nb[1] / ggml_type_size(src0->type);
            a.ncols_y = ncols; a.nrows_dst = mm->ne[1];
            a.nchannels_x = 1; a.nchannels_y = 1; a.nsamples_x = 1; a.nsamples_y = 1;
            a.ncols_max = ncols;
            const bool swizzle_iq4 = ggml_cuda_should_use_mmq_prefill(a);
            r.kind  = ggml_backend_cuda_context::PREFILL_ACT_MMQ;
            r.param = (int) mmq_get_q8_1_ds_layout(src0->type) | (swizzle_iq4 ? 0x10 : 0);
            r.valid = true;
        }
    }
#endif
    return r;
}

// The first predicted layout among the matmuls that read prod_out as src1 (directly or through a view), and how many
// of them want that same layout. Other readers need nothing: the f32 output is always written.
static bool pf_pick_target(const ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int prod_idx,
        const ggml_tensor * prod_out, pf_consumer_route & tgt, int & uses) {
    uses = 0;
    tgt  = pf_consumer_route{};
    for (int j = prod_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        const ggml_tensor * src1 = node->src[1];
        if (!src1 || (src1 != prod_out && pf_unwrap_views(src1) != prod_out)) {
            continue;
        }
        const pf_consumer_route r = pf_inspect_consumer(ctx, node);
        if (!r.valid) {
            continue;
        }
        if (!tgt.valid) {
            tgt = r;
        }
        if (r.kind == tgt.kind && r.param == tgt.param) {
            uses++;
        }
    }
    return tgt.valid;
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

bool ggml_cuda_prefill_any_on() {
    static const bool on = ggml_cuda_prefill_fuse_glu_actq_on() || ggml_cuda_prefill_fuse_norm_actq_on();
    return on;
}

// =================================================================================================
// Cache Lifecycle & Invalidation
// =================================================================================================

void ggml_cuda_prefill_glue_reset(ggml_backend_cuda_context * ctx) {
    if (!ctx) {
        return;
    }
    for (auto & c : ctx->prefill_act_cache.entries) {
        pf_clear(c);
    }
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
        c.ptr = nullptr;
        c.cap = 0;
        c.dev = -1;
        pf_clear(c);
    }
}

void ggml_cuda_prefill_invalidate_ptr(ggml_backend_cuda_context & ctx, const void * ptr, size_t nbytes, uint64_t keep_after) {
    if (!ptr || nbytes == 0 || !ggml_cuda_prefill_any_on()) {
        return;
    }
    const char * p = (const char *) ptr;
    for (auto & c : ctx.prefill_act_cache.entries) {
        if (c.kind == ggml_backend_cuda_context::PREFILL_ACT_NONE || c.data == nullptr || c.born > keep_after) {
            continue;
        }
        const char * d = (const char *) c.data;
        const size_t db = (size_t) c.n_rows * (size_t) c.k * sizeof(float);
        if (p < d + db && d < p + nbytes) {
            pf_clear(c);
        }
    }
}

// =================================================================================================
// Take side (called by the consumer matmuls)
// =================================================================================================

bool ggml_cuda_prefill_ai4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, const int8_t * & q, const float * & s) {
    const auto * c = pf_find(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_AI4, xsh | (variant << 1), n_rows, K, K);
    if (!c) {
        return false;
    }
    s = (const float *) c->ptr;
    q = (const int8_t *) (c->ptr + c->aux_off);
    return true;
}

bool ggml_cuda_prefill_nvfp4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, const int8_t * & xq, const float * & xs) {
    const auto * c = pf_find(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF, kf, n_rows, K, K);
    if (!c) {
        return false;
    }
    xq = (const int8_t *) c->ptr;
    xs = (const float *) (c->ptr + c->aux_off);
    return true;
}

bool ggml_cuda_prefill_mmq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, const char * & q8) {
    const auto * c = pf_find(ctx, src1, ggml_backend_cuda_context::PREFILL_ACT_MMQ, ds_layout | (swizzle_iq4 ? 0x10 : 0),
                             n_rows, K, k_padded);
    if (!c) {
        return false;
    }
    q8 = c->ptr;
    return true;
}

// Reserve side: one entry for the predicted target layout of prod_out; false = no entry (nothing to launch fused)
namespace {
struct pf_buffers {
    int8_t * q  = nullptr;  // AI+ / NVFP4PF int8 values, or the MMQ block_q8_1_mmq buffer
    float  * s  = nullptr;  // AI+ / NVFP4PF scales
    int      k_padded = 0;  // MMQ only
};

static bool pf_reserve_for(ggml_backend_cuda_context & ctx, const ggml_tensor * out, const pf_consumer_route & tgt, int uses,
        int64_t n_rows, int K, pf_buffers & b) {
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4) {
        const int xsh = tgt.param & 1;
        const size_t ns    = (size_t) n_rows * (size_t) (K >> (7 + xsh));
        const size_t q_off = GGML_PAD(ns * sizeof(float), 256);
        auto * e = pf_reserve(ctx, out, tgt.kind, tgt.param, n_rows, K, K, q_off + (size_t) n_rows * K, q_off, uses);
        if (!e) {
            return false;
        }
        b.s = (float *) e->ptr;
        b.q = (int8_t *) (e->ptr + q_off);
        return true;
    }
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
        const size_t xq_bytes = GGML_PAD((size_t) n_rows * K, 256);
        const size_t xs_bytes = (size_t) n_rows * (size_t) (K / tgt.param) * sizeof(float);
        auto * e = pf_reserve(ctx, out, tgt.kind, tgt.param, n_rows, K, K, xq_bytes + xs_bytes, xq_bytes, uses);
        if (!e) {
            return false;
        }
        b.q = (int8_t *) e->ptr;
        b.s = (float *) (e->ptr + xq_bytes);
        return true;
    }
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ) {
        b.k_padded = (int) GGML_PAD(K, MATRIX_ROW_PADDING);
        const size_t bytes = (size_t) (b.k_padded / QK8_1_MMQ) * (size_t) n_rows * sizeof(block_q8_1_mmq) + 32u * 1024u;
        auto * e = pf_reserve(ctx, out, tgt.kind, tgt.param, n_rows, K, b.k_padded, bytes, 0, uses);
        if (!e) {
            return false;
        }
        b.q = (int8_t *) e->ptr;
        return true;
    }
    return false;
}
} // namespace

// =================================================================================================
// Kernel Launchers (also called by bench-prefill-glue)
// =================================================================================================

void ggml_cuda_prefill_launch_glu_ai4(
        int glu_kind, const void * g, const void * u, float * out,
        int8_t * q, float * s, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2, int xsh, int variant,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const dim3 grid(xsh ? (unsigned) ((K / RQ4_HB + 3) / 4) : (unsigned) ((K / 128 + 7) / 8),
                    (unsigned) std::min<int64_t>(rows, 65535));
    pf_dispatch_ai4_variants(variant, xsh, [&](auto xsh_tag, auto sw_tag, auto xp_tag, auto ln_tag) {
        k_prefill_glu_ai4<decltype(xsh_tag)::value, decltype(sw_tag)::value, decltype(xp_tag)::value, decltype(ln_tag)::value>
            <<<grid, 128, 0, stream>>>(
                glu_kind, (const char *) g, (const char *) u, out, q, s, K, rows, ne1,
                g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    });
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_launch_glu_nvfp4pf(
        int glu_kind, const void * g, const void * u, float * out,
        int8_t * xq, float * xs, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2, int kf,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const dim3 grid((unsigned) ((K / kf + 3) / 4), (unsigned) std::min<int64_t>(rows, 65535));
    if (kf == 32) {
        k_prefill_glu_nvfp4pf<32><<<grid, 128, 0, stream>>>(glu_kind, (const char *) g, (const char *) u, out, xq, xs, K, rows, ne1,
            g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    } else {
        k_prefill_glu_nvfp4pf<64><<<grid, 128, 0, stream>>>(glu_kind, (const char *) g, (const char *) u, out, xq, xs, K, rows, ne1,
            g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_launch_glu_mmq(
        int glu_kind, const void * g, const void * u, float * out,
        void * vy, int ds_layout, bool swizzle_iq4,
        int K, int K_padded, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream) {
    const unsigned by = (unsigned) ((K_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ));
    const dim3 grid((unsigned) std::min<int64_t>(rows, 65535), by, 1);
    auto go = [&](auto ds_tag, auto sw_tag) {
        k_prefill_glu_mmq<decltype(ds_tag)::value, decltype(sw_tag)::value><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
            glu_kind, (const char *) g, (const char *) u, out, vy,
            K, K_padded, rows, ne1, g_nb1, g_nb2, u_nb1, u_nb2, situ, glu_limit, oai_alpha, oai_limit);
    };
    if (swizzle_iq4) {
        go(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::true_type{});
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        go(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_DS4>{}, std::false_type{});
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        go(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D2S6>{}, std::false_type{});
    } else {
        go(std::integral_constant<mmq_q8_1_ds_layout, MMQ_Q8_1_DS_LAYOUT_D4>{}, std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_launch_norm_ai4(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        int8_t * q, float * s, int64_t n_rows, int K, float eps, int xsh, int variant, cudaStream_t stream) {
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto add_tag) {
        constexpr int  BS  = decltype(bs_tag)::value;
        constexpr bool ADD = decltype(add_tag)::value;
        pf_dispatch_ai4_variants(variant, xsh, [&](auto xsh_tag, auto sw_tag, auto xp_tag, auto ln_tag) {
            auto * kfn = k_prefill_norm_ai4<BS, decltype(xsh_tag)::value, decltype(sw_tag)::value, decltype(xp_tag)::value, decltype(ln_tag)::value, ADD>;
            if (smem > 48u * 1024u) {
                CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            }
            kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, q, s, n_rows, K, K, K, eps);
        });
    };
    auto pick_add = [&](auto bs_tag) {
        if (r != nullptr) launch(bs_tag, std::true_type{});
        else              launch(bs_tag, std::false_type{});
    };
    if (K < 1024) pick_add(std::integral_constant<int, 256>{});
    else          pick_add(std::integral_constant<int, 1024>{});
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_launch_norm_nvfp4pf(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        int8_t * xq, float * xs, int64_t n_rows, int K, float eps, int kf, cudaStream_t stream) {
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto kf_tag, auto add_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        auto * kfn = k_prefill_norm_nvfp4pf<BS, decltype(kf_tag)::value, decltype(add_tag)::value>;
        if (smem > 48u * 1024u) {
            CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        }
        kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, xq, xs, n_rows, K, K, K, eps);
    };
    auto pick_add = [&](auto bs_tag, auto kf_tag) {
        if (r != nullptr) launch(bs_tag, kf_tag, std::true_type{});
        else              launch(bs_tag, kf_tag, std::false_type{});
    };
    auto pick_kf = [&](auto bs_tag) {
        if (kf == 32) pick_add(bs_tag, std::integral_constant<int, 32>{});
        else          pick_add(bs_tag, std::integral_constant<int, 64>{});
    };
    if (K < 1024) pick_kf(std::integral_constant<int, 256>{});
    else          pick_kf(std::integral_constant<int, 1024>{});
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prefill_launch_norm_mmq(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        void * vy, int ds_layout, bool swizzle_iq4, int64_t n_rows, int K, int K_padded, float eps, cudaStream_t stream) {
    const size_t smem = (size_t) K * sizeof(float);
    const dim3 grid((unsigned) std::min<int64_t>(n_rows, 65535));
    auto launch = [&](auto bs_tag, auto ds_tag, auto sw_tag, auto add_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        auto * kfn = k_prefill_norm_mmq<BS, decltype(ds_tag)::value, decltype(sw_tag)::value, decltype(add_tag)::value>;
        if (smem > 48u * 1024u) {
            CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        }
        kfn<<<grid, BS, smem, stream>>>(x, r, add_dst, w, mul_dst, vy, n_rows, K, K_padded, K, K, eps);
    };
    auto pick_add = [&](auto bs_tag, auto ds_tag, auto sw_tag) {
        if (r != nullptr) launch(bs_tag, ds_tag, sw_tag, std::true_type{});
        else              launch(bs_tag, ds_tag, sw_tag, std::false_type{});
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
    CUDA_CHECK(cudaGetLastError());
}

// =================================================================================================
// Fused GLU / SITU -> f32 output + quantized activation
// =================================================================================================

static bool pf_exec_glu(
        ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int prod_idx,
        int glu_kind, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit) {
    if (!ggml_cuda_prefill_fuse_glu_actq_on() || !pf_producer_ok(ctx)) {
        return false;
    }
    if (!out || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[3] != 1 ||
            g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32) {
        return false;
    }
    if (!pf_cuda_buf(g, ctx.device) || !pf_cuda_buf(u, ctx.device) || !pf_cuda_buf(out, ctx.device)) {
        return false;
    }
    const int K = (int) out->ne[0];
    const int64_t rows = out->ne[1] * out->ne[2];
    const int64_t ne1  = out->ne[1];
    const int64_t n_tok_logical = out->ne[2] > 1 ? out->ne[2] : out->ne[1];
    // rows of K floats, every row start 16-byte aligned (float4 loads / stores), no aliasing with the output
    if (K < 128 || K % 128 != 0 || n_tok_logical <= 8 || ((uintptr_t) out->data & 15) != 0 ||
            ((uintptr_t) g->data & 15) != 0 || ((uintptr_t) u->data & 15) != 0 ||
            g->ne[0] != K || u->ne[0] != K || g->ne[1] != out->ne[1] || u->ne[1] != out->ne[1] ||
            g->ne[2] != out->ne[2] || u->ne[2] != out->ne[2] || g->ne[3] != 1 || u->ne[3] != 1 ||
            g->nb[0] != sizeof(float) || u->nb[0] != sizeof(float) ||
            (g->nb[1] | g->nb[2] | u->nb[1] | u->nb[2]) % 16 != 0 ||
            pf_overlap(out, g) || pf_overlap(out, u)) {
        return false;
    }

    pf_consumer_route tgt;
    int uses = 0;
    if (!pf_pick_target(ctx, cgraph, prod_idx, out, tgt, uses)) {
        return false;
    }
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ && out->ne[2] != 1) {
        return false;
    }
    pf_buffers b;
    if (!pf_reserve_for(ctx, out, tgt, uses, rows, K, b)) {
        return false;
    }
    const cudaStream_t stream = ctx.stream();
    const int64_t gn1 = (int64_t) g->nb[1], gn2 = (int64_t) g->nb[2], un1 = (int64_t) u->nb[1], un2 = (int64_t) u->nb[2];
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4) {
        ggml_cuda_prefill_launch_glu_ai4(glu_kind, g->data, u->data, (float *) out->data, b.q, b.s, K, rows, ne1,
            gn1, gn2, un1, un2, tgt.param & 1, tgt.param >> 1, situ, glu_limit, oai_alpha, oai_limit, stream);
    } else if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
        ggml_cuda_prefill_launch_glu_nvfp4pf(glu_kind, g->data, u->data, (float *) out->data, b.q, b.s, K, rows, ne1,
            gn1, gn2, un1, un2, tgt.param, situ, glu_limit, oai_alpha, oai_limit, stream);
    } else {
        ggml_cuda_prefill_launch_glu_mmq(glu_kind, g->data, u->data, (float *) out->data, b.q, tgt.param & 0xf,
            (tgt.param & 0x10) != 0, K, b.k_padded, rows, ne1, gn1, gn2, un1, un2, situ, glu_limit, oai_alpha, oai_limit, stream);
    }
    ctx.fusion_stats.prefill_glu_actq++;
    return true;
}

bool ggml_cuda_prefill_situ_launch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out, const ink_situ_params & p) {
    return pf_exec_glu(ctx, cgraph, last, PF_GLU_SITU, g, u, out, p, 0.0f, 0.0f, 0.0f);
}

bool ggml_cuda_try_fuse_prefill_glu(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (!ctx || !ggml_cuda_prefill_fuse_glu_actq_on()) {
        return false;
    }
    ggml_tensor * glu = cgraph->nodes[i];
    if (glu->op != GGML_OP_GLU || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || glu->ne[3] != 1) {
        return false;
    }
    const int kind = pf_map_glu_op(ggml_get_glu_op(glu));
    if (kind < 0) {
        return false;
    }
    const ggml_tensor * src0 = glu->src[0];
    const ggml_tensor * src1 = glu->src[1];
    if (!src0 || src0->type != GGML_TYPE_F32 || src0->ne[0] != (src1 ? glu->ne[0] : 2 * glu->ne[0])) {
        return false;
    }
    if (src1 && (src1->type != GGML_TYPE_F32 || !ggml_are_same_shape(src0, src1))) {
        return false;
    }
    const int32_t swapped = ((const int32_t *) glu->op_params)[1];
    // same op_params slots the stock CUDA GLU ops read: SWIGLU_CLAMP limit = [3], SWIGLU_OAI alpha / limit = [2] / [3]
    const float limit     = ggml_get_op_params_f32(glu, 3);
    float oai_alpha = 1.702f, oai_limit = 7.0f;
    if (kind == PF_GLU_SWIGLU_OAI) {
        oai_alpha = ggml_get_op_params_f32(glu, 2);
        oai_limit = ggml_get_op_params_f32(glu, 3);
    }

    // one src0 [2K, ...]: describe the gate / up halves as stack copies (only ne, data and nb are read)
    ggml_tensor g_tmp = *src0;
    ggml_tensor u_tmp = src1 ? *src1 : *src0;
    const ggml_tensor * g = src1 ? (swapped ? src1 : src0) : &g_tmp;
    const ggml_tensor * u = src1 ? (swapped ? src0 : src1) : &u_tmp;
    if (!src1) {
        const size_t half_bytes = (size_t) glu->ne[0] * sizeof(float);
        g_tmp.ne[0] = glu->ne[0];
        u_tmp.ne[0] = glu->ne[0];
        (swapped ? g_tmp : u_tmp).data = (void *) ((char *) src0->data + half_bytes);
    }
    ink_situ_params none{};
    return pf_exec_glu(*ctx, cgraph, i, kind, g, u, glu, none, limit, oai_alpha, oai_limit);
}

// =================================================================================================
// Fused RMS_NORM(+ADD) * w -> f32 outputs + quantized activation
// =================================================================================================

int ggml_cuda_try_fuse_prefill_norm(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i) {
    if (!ctx || !ggml_cuda_prefill_fuse_norm_actq_on() || i + 1 >= cgraph->n_nodes || !pf_producer_ok(*ctx)) {
        return 0;
    }

    ggml_tensor * add_node  = nullptr;
    ggml_tensor * norm_node = cgraph->nodes[i];
    int mul_idx = i + 1;

    if (norm_node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RMS_NORM) {
        add_node  = norm_node;
        norm_node = cgraph->nodes[i + 1];
        mul_idx   = i + 2;
        // residual ADD without broadcasting, everything contiguous (the kernel reads and writes rows of K)
        if (norm_node->src[0] != add_node || !add_node->src[0] || !add_node->src[1] ||
                !ggml_are_same_shape(add_node->src[0], add_node) || !ggml_are_same_shape(add_node->src[1], add_node) ||
                add_node->type != GGML_TYPE_F32 || add_node->src[0]->type != GGML_TYPE_F32 || add_node->src[1]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(add_node) || !ggml_is_contiguous(add_node->src[0]) || !ggml_is_contiguous(add_node->src[1]) ||
                !pf_cuda_buf(add_node, ctx->device) || !pf_cuda_buf(add_node->src[1], ctx->device)) {
            return 0;
        }
    }

    ggml_tensor * mul_node = cgraph->nodes[mul_idx];
    // the RMS_NORM output itself is not written: its only reader must be the MUL
    if (norm_node->op != GGML_OP_RMS_NORM || mul_node->op != GGML_OP_MUL ||
            ggml_node_get_use_count(cgraph, mul_idx - 1) != 1 || (norm_node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * x = add_node ? add_node->src[0] : norm_node->src[0];
    const ggml_tensor * r = add_node ? add_node->src[1] : nullptr;
    const ggml_tensor * w = mul_node->src[0] == norm_node ? mul_node->src[1] :
                            mul_node->src[1] == norm_node ? mul_node->src[0] : nullptr;
    if (!w || !x || norm_node->type != GGML_TYPE_F32 || mul_node->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            x->type != GGML_TYPE_F32 || !ggml_are_same_shape(add_node ? (const ggml_tensor *) add_node : x, norm_node) ||
            !ggml_are_same_shape(norm_node, mul_node) || !ggml_is_contiguous(x) || !ggml_is_contiguous(mul_node) ||
            !ggml_is_contiguous(w) || w->ne[0] != norm_node->ne[0] || ggml_nrows(w) != 1) {
        return 0;
    }
    const int K = (int) mul_node->ne[0];
    const int64_t n_rows = ggml_nrows(mul_node);
    // shared memory holds one row of float[K] (<= 96 KiB on SM 8.6); prompt rows only
    if (K < 128 || K % 128 != 0 || K > 24576 || n_rows < 16 || ((uintptr_t) mul_node->data & 15) != 0 ||
            !pf_cuda_buf(mul_node, ctx->device) || !pf_cuda_buf(x, ctx->device) || !pf_cuda_buf(w, ctx->device)) {
        return 0;
    }
    if (pf_overlap(mul_node, w) || (mul_node->data != x->data && pf_overlap(mul_node, x)) ||
            (r && (pf_overlap(mul_node, r) || (add_node->data != r->data && pf_overlap(add_node, r)) ||
                   (add_node->data != x->data && pf_overlap(add_node, x)) || pf_overlap(add_node, mul_node)))) {
        return 0;
    }

    pf_consumer_route tgt;
    int uses = 0;
    if (!pf_pick_target(*ctx, cgraph, mul_idx, mul_node, tgt, uses)) {
        return 0;
    }
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_MMQ && mul_node->ne[2] != 1) {
        return 0;
    }
    pf_buffers b;
    if (!pf_reserve_for(*ctx, mul_node, tgt, uses, n_rows, K, b)) {
        return 0;
    }

    float eps = 0.0f;
    std::memcpy(&eps, norm_node->op_params, sizeof(float));
    const float * xd  = (const float *) x->data;
    const float * rd  = r ? (const float *) r->data : nullptr;
    float * add_d     = add_node ? (float *) add_node->data : nullptr;
    const float * wd  = (const float *) w->data;
    float * mul_d     = (float *) mul_node->data;
    const cudaStream_t stream = ctx->stream();
    if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4) {
        ggml_cuda_prefill_launch_norm_ai4(xd, rd, add_d, wd, mul_d, b.q, b.s, n_rows, K, eps, tgt.param & 1, tgt.param >> 1, stream);
    } else if (tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF) {
        ggml_cuda_prefill_launch_norm_nvfp4pf(xd, rd, add_d, wd, mul_d, b.q, b.s, n_rows, K, eps, tgt.param, stream);
    } else {
        ggml_cuda_prefill_launch_norm_mmq(xd, rd, add_d, wd, mul_d, b.q, tgt.param & 0xf, (tgt.param & 0x10) != 0,
            n_rows, K, b.k_padded, eps, stream);
    }
    ctx->fusion_stats.prefill_norm_actq++;
    return mul_idx - i; // additional nodes after i skipped by eval_graph
}

bool ggml_cuda_prefill_try_mix_norm_actq(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * mul, int & kf, int8_t ** q, float ** s) {
    kf = 0;
    if (!ggml_cuda_prefill_fuse_norm_actq_on() || !pf_producer_ok(ctx)) {
        return false;
    }
    const int K = (int) mul->ne[0];
    const int64_t n_rows = ggml_nrows(mul);
    if (K < 128 || K % 128 != 0 || K > 32 * 256 || n_rows < 16 || !ggml_is_contiguous(mul) || ((uintptr_t) mul->data & 15) != 0) {
        return false;
    }
    pf_consumer_route tgt;
    int uses = 0;
    if (!pf_pick_target(ctx, cgraph, last, mul, tgt, uses)) {
        return false;
    }
    // depth-mix.cu writes the AI+ H256 plain layout (xsh 1, variant 0) or the NVFP4PF one
    const bool ai4 = tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_AI4 && tgt.param == 1 && K % 256 == 0;
    const bool nv  = tgt.kind == ggml_backend_cuda_context::PREFILL_ACT_NVFP4PF;
    if (!ai4 && !nv) {
        return false;
    }
    pf_buffers b;
    if (!pf_reserve_for(ctx, mul, tgt, uses, n_rows, K, b)) {
        return false;
    }
    kf = nv ? tgt.param : 0;
    *q = b.q;
    *s = b.s;
    return true;
}
