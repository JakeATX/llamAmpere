#pragma once

#include "common.cuh"
#include "ggml-cuda/mma.cuh"
#include "convert.cuh"
#include "cp-async.cuh"

#include <cstdint>
#include <cstdlib>

// [PREFILLFAST] Argon chunked tensor-core GDN prompt kernel, ported from bl/gdn-tc-argon-fix b6aa8e51c
// (code unchanged; macros renamed GDN_TC_* -> GDNA_* and wrapped in a namespace so it builds beside
// the #112 GGML_CUDA_GDN_TC kernel). Included by gated_delta_net.cu after gdn_store_state.
// OPT-IN here: GGML_CUDA_GDN_CHUNKED=1 turns it on (default off in this branch).
namespace gdn_argon {

#define GDNA_S_V             128
#define GDNA_CHUNK           32
#define GDNA_WARPS           4
#define GDNA_COLS_PER_WARP   8
#define GDNA_COLS_PER_BLOCK  (GDNA_WARPS * GDNA_COLS_PER_WARP)
#define GDNA_LM_STRIDE       36

struct gdn_tc_chunk_prep {
    half2 k_swz[GDNA_CHUNK * (GDNA_S_V / 2)];
    half2 q_swz[GDNA_CHUNK * (GDNA_S_V / 2)];
    float lm[GDNA_CHUNK * GDNA_CHUNK];
    float k_scale[GDNA_CHUNK];
    float q_scale[GDNA_CHUNK];
};

static_assert(sizeof(gdn_tc_chunk_prep) % 256 == 0, "gdn_tc_chunk_prep must be 256-byte aligned");

static __device__ __forceinline__ void gdn_tc_pow2_scale(float max_abs, float & scale, float & inv_scale) {
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        scale     = 1.0f;
        inv_scale = 1.0f;
        return;
    }
    uint32_t exp_bits = __float_as_uint(max_abs) & 0x7f800000u;
    if (exp_bits < 0x08800000u) {
        exp_bits = 0x08800000u;
    } else if (exp_bits > 0x77000000u) {
        exp_bits = 0x77000000u;
    }
    scale     = __uint_as_float(exp_bits);
    inv_scale = __uint_as_float(0x7f000000u - exp_bits);
}

// Precision bits (GGML_CUDA_GDN_CHUNKED_PREC). 0 = original single FP16 products.
//   1: Q*K^T / K*K^T (the solve matrix) from split FP16 Q/K (three products)
//   2: state operand of K*S / Q*S split (two products)
//   4: Delta operand of M*Delta / K^T*Delta split, and M split (two-three products)
//   8: low FP16 planes of Q/K also in K*S / Q*S / K^T*Delta (needs 16 KiB more shared memory)
#define GDNA_P_LM 1
#define GDNA_P_S  2
#define GDNA_P_D  4  // M*Delta: split Delta and M (output path)
#define GDNA_P_QK 8
#define GDNA_P_KLO 16 // low FP16 plane of K only, used in K*S and K^T*Delta (+8 KiB shared, keeps 3 CTAs/SM)
#define GDNA_P_DU 32  // K^T*Delta (state update): split Delta
#define GDNA_P_KLOU 64 // low FP16 plane of K used in K^T*Delta only
#define GDNA_P_DMD 128 // M*Delta: split Delta only (half of bit 4)
#define GDNA_P_DMM 256 // M*Delta: split M only (half of bit 4)

// x * exp(g) without the intermediate exp(g) underflowing: expf (__expf under fast-math) flushes to zero below
// about -87 even when the product (for example a large carried state) is still representable.
// g < -200 (including -inf) returns a signed zero: e^-200 times any finite float is zero, and it bounds the loop.
static __device__ __forceinline__ float gdn_tc_mul_exp(float x, float g) {
    if (!(g >= -200.0f)) {
        return x * 0.0f;
    }
    while (g < -60.0f) {
        x *= 8.75651076e-27f; // expf(-60)
        g += 60.0f;
    }
    return x * expf(g);
}

static __device__ __forceinline__ float gdn_tc_f16_residual(const float x) {
    return x - __half2float(__float2half_rn(x));
}

static __device__ __forceinline__ ggml_cuda_mma::tile<16, 8, float> gdn_tc_residual_tile(
        const ggml_cuda_mma::tile<16, 8, float> & t) {
    ggml_cuda_mma::tile<16, 8, float> r;
#pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        r.x[l] = gdn_tc_f16_residual(t.x[l]);
    }
    return r;
}

static __device__ __forceinline__ void gdn_load_ldmatrix_8x8_swz(
        ggml_cuda_mma::tile<8, 8, half2> & t,
        const half2 * __restrict__ tile_base,
        const int i0, const int j0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
    const int i = i0 + (threadIdx.x % 8);
    const int j = j0 + ((threadIdx.x / 8) % 2) * 4;
    int * xi = (int *) t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(ggml_cuda_mma::swizzle<true>(tile_base, i, j, stride)));
#else
    GGML_UNUSED_VARS(t, tile_base, i0, j0, stride);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void gdn_load_ldmatrix_trans_swz(
        ggml_cuda_mma::tile<16, 8, half2> & t,
        const half2 * __restrict__ tile_base,
        const int i0, const int j0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
    const int i = i0 + (threadIdx.x % 16);
    const int j = j0 + (threadIdx.x / 16) * 4;
    int * xi = (int *) t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(ggml_cuda_mma::swizzle<true>(tile_base, i, j, stride)));
#else
    GGML_UNUSED_VARS(t, tile_base, i0, j0, stride);
    NO_DEVICE_CODE;
#endif
}

// smem_lo / gmem_lo (nullable): the FP16 residual plane, same layout and row scale as the high plane.
static __device__ __forceinline__ void gdn_tc_load_qk_chunk(
        const float * __restrict__ src_chunk,
        const int64_t stride_tok,
        half2 * __restrict__ smem_qk,
        half2 * __restrict__ gmem_qk_swz,
        float * __restrict__ smem_scale,
        float * __restrict__ gmem_scale,
        half2 * __restrict__ smem_lo = nullptr,
        half2 * __restrict__ gmem_lo = nullptr) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int row = tid / 4;
    const int sub = tid % 4;
    const float * row_ptr = src_chunk + row * stride_tok;

    float vals[4][8];
    float max_abs = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int c8 = sub + 4 * p;
        const float4 v0 = *reinterpret_cast<const float4 *>(row_ptr + c8 * 8 + 0);
        const float4 v1 = *reinterpret_cast<const float4 *>(row_ptr + c8 * 8 + 4);
        vals[p][0] = v0.x; vals[p][1] = v0.y; vals[p][2] = v0.z; vals[p][3] = v0.w;
        vals[p][4] = v1.x; vals[p][5] = v1.y; vals[p][6] = v1.z; vals[p][7] = v1.w;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            max_abs = fmaxf(max_abs, fabsf(vals[p][k]));
        }
    }

    max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, 1));
    max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, 2));

    float scale, inv_scale;
    gdn_tc_pow2_scale(max_abs, scale, inv_scale);

    if (sub == 0) {
        smem_scale[row] = scale;
        gmem_scale[row] = scale;
    }

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int c8 = sub + 4 * p;
        half2 h2[4];
        h2[0] = make_half2(vals[p][0] * inv_scale, vals[p][1] * inv_scale);
        h2[1] = make_half2(vals[p][2] * inv_scale, vals[p][3] * inv_scale);
        h2[2] = make_half2(vals[p][4] * inv_scale, vals[p][5] * inv_scale);
        h2[3] = make_half2(vals[p][6] * inv_scale, vals[p][7] * inv_scale);
        const int swz_bytes = ggml_cuda_mma::swizzle_bytes<true, half2>(row, c8 * 4, GDNA_S_V / 2);
        const int4 packed = *reinterpret_cast<const int4 *>(h2);
        *reinterpret_cast<int4 *>(reinterpret_cast<char *>(smem_qk) + swz_bytes) = packed;
        *reinterpret_cast<int4 *>(reinterpret_cast<char *>(gmem_qk_swz) + swz_bytes) = packed;
        if (smem_lo != nullptr || gmem_lo != nullptr) {
            half2 l2[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                l2[e] = make_half2(gdn_tc_f16_residual(vals[p][2 * e + 0] * inv_scale),
                                   gdn_tc_f16_residual(vals[p][2 * e + 1] * inv_scale));
            }
            const int4 packed_lo = *reinterpret_cast<const int4 *>(l2);
            if (smem_lo != nullptr) {
                *reinterpret_cast<int4 *>(reinterpret_cast<char *>(smem_lo) + swz_bytes) = packed_lo;
            }
            if (gmem_lo != nullptr) {
                *reinterpret_cast<int4 *>(reinterpret_cast<char *>(gmem_lo) + swz_bytes) = packed_lo;
            }
        }
    }
}

// Per-(chunk, key_head, seq) preparation: swizzles row-scaled Q and K and computes packed Q*K^T (lower+diag) and K*K^T (upper).
template <int prec_t>
__global__ void __launch_bounds__(WARP_SIZE * GDNA_WARPS, 4)
gdn_tc_prepare_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        gdn_tc_chunk_prep * __restrict__ prep_all,
        half2 * __restrict__ lo_all,
        int64_t n_full_chunks,
        int64_t neqk1,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        const uint3 rq3_magic) {
#if defined(TURING_MMA_AVAILABLE)
    const int64_t  ch       = blockIdx.x;
    const uint32_t iq1      = blockIdx.y;
    const uint32_t sequence = blockIdx.z;
    const uint32_t iq3      = fastdiv(sequence, rq3_magic);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    __shared__ half2 smem_k[GDNA_CHUNK * (GDNA_S_V / 2)];
    __shared__ half2 smem_q[GDNA_CHUNK * (GDNA_S_V / 2)];
    __shared__ float smem_k_scale[GDNA_CHUNK];
    __shared__ float smem_q_scale[GDNA_CHUNK];
    constexpr bool lm_split = (prec_t & GDNA_P_LM) != 0;
    constexpr bool qk_lo    = (prec_t & GDNA_P_QK) != 0;
    constexpr int  LO_N     = lm_split ? GDNA_CHUNK * (GDNA_S_V / 2) : 1;
    __shared__ half2 smem_k_lo[LO_N];
    __shared__ half2 smem_q_lo[LO_N];

    ggml_cuda_pdl_sync();

    const int64_t t0 = ch * GDNA_CHUNK;
    const float * q_chunk = q + iq3 * sq3 + t0 * sq2 + iq1 * sq1;
    const float * k_chunk = k + iq3 * sq3 + t0 * sq2 + iq1 * sq1;
    const int64_t prep_idx = ((int64_t) sequence * neqk1 + iq1) * n_full_chunks + ch;
    gdn_tc_chunk_prep * prep = prep_all + prep_idx;
    constexpr bool k_lo_on  = qk_lo || (prec_t & (GDNA_P_KLO | GDNA_P_KLOU)) != 0;
    constexpr int  lo_stride = (qk_lo ? 2 : 1) * GDNA_CHUNK * (GDNA_S_V / 2);
    half2 * k_lo_g = k_lo_on ? lo_all + prep_idx * lo_stride : nullptr;
    half2 * q_lo_g = qk_lo ? k_lo_g + GDNA_CHUNK * (GDNA_S_V / 2) : nullptr;

    gdn_tc_load_qk_chunk(k_chunk, sq2, smem_k, prep->k_swz, smem_k_scale, prep->k_scale,
        lm_split ? smem_k_lo : nullptr, k_lo_g);
    gdn_tc_load_qk_chunk(q_chunk, sq2, smem_q, prep->q_swz, smem_q_scale, prep->q_scale,
        lm_split ? smem_q_lo : nullptr, q_lo_g);

    __syncthreads();

    ggml_cuda_mma::tile<16, 8, float> qk_acc[2];
    ggml_cuda_mma::tile<16, 8, float> kk_acc[2];

    // lm_split: hi*hi + hi*lo + lo*hi (the lo*lo term is below FP32 rounding of the result).
#pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        ggml_cuda_mma::tile<8, 8, half2> k_b;
        gdn_load_ldmatrix_8x8_swz(k_b, smem_k, warp * 8, kd * 8, GDNA_S_V / 2);
        ggml_cuda_mma::tile<8, 8, half2> k_b_lo;
        if constexpr (lm_split) {
            gdn_load_ldmatrix_8x8_swz(k_b_lo, smem_k_lo, warp * 8, kd * 8, GDNA_S_V / 2);
        }

        if (warp < 2) {
            ggml_cuda_mma::tile<16, 8, half2> q_a0;
            ggml_cuda_mma::load_ldmatrix<true>(q_a0, smem_q, 0, kd * 8, GDNA_S_V / 2);
            ggml_cuda_mma::mma(qk_acc[0], q_a0, k_b);
            if constexpr (lm_split) {
                ggml_cuda_mma::tile<16, 8, half2> q_a0_lo;
                ggml_cuda_mma::load_ldmatrix<true>(q_a0_lo, smem_q_lo, 0, kd * 8, GDNA_S_V / 2);
                ggml_cuda_mma::mma(qk_acc[0], q_a0, k_b_lo);
                ggml_cuda_mma::mma(qk_acc[0], q_a0_lo, k_b);
            }
        }
        {
            ggml_cuda_mma::tile<16, 8, half2> q_a1;
            ggml_cuda_mma::load_ldmatrix<true>(q_a1, smem_q, 16, kd * 8, GDNA_S_V / 2);
            ggml_cuda_mma::mma(qk_acc[1], q_a1, k_b);

            ggml_cuda_mma::tile<16, 8, half2> k_a0;
            ggml_cuda_mma::load_ldmatrix<true>(k_a0, smem_k, 0, kd * 8, GDNA_S_V / 2);
            ggml_cuda_mma::mma(kk_acc[0], k_a0, k_b);
            if constexpr (lm_split) {
                ggml_cuda_mma::tile<16, 8, half2> q_a1_lo;
                ggml_cuda_mma::load_ldmatrix<true>(q_a1_lo, smem_q_lo, 16, kd * 8, GDNA_S_V / 2);
                ggml_cuda_mma::mma(qk_acc[1], q_a1, k_b_lo);
                ggml_cuda_mma::mma(qk_acc[1], q_a1_lo, k_b);
                ggml_cuda_mma::tile<16, 8, half2> k_a0_lo;
                ggml_cuda_mma::load_ldmatrix<true>(k_a0_lo, smem_k_lo, 0, kd * 8, GDNA_S_V / 2);
                ggml_cuda_mma::mma(kk_acc[0], k_a0, k_b_lo);
                ggml_cuda_mma::mma(kk_acc[0], k_a0_lo, k_b);
            }
        }
        if (warp >= 2) {
            ggml_cuda_mma::tile<16, 8, half2> k_a1;
            ggml_cuda_mma::load_ldmatrix<true>(k_a1, smem_k, 16, kd * 8, GDNA_S_V / 2);
            ggml_cuda_mma::mma(kk_acc[1], k_a1, k_b);
            if constexpr (lm_split) {
                ggml_cuda_mma::tile<16, 8, half2> k_a1_lo;
                ggml_cuda_mma::load_ldmatrix<true>(k_a1_lo, smem_k_lo, 16, kd * 8, GDNA_S_V / 2);
                ggml_cuda_mma::mma(kk_acc[1], k_a1, k_b_lo);
                ggml_cuda_mma::mma(kk_acc[1], k_a1_lo, k_b);
            }
        }
    }

    const int g_lane = lane / 4;
    const int t_lane = lane % 4;
    const int c0     = warp * 8 + 2 * t_lane;
    const int c1     = c0 + 1;
    const float ks_c0 = smem_k_scale[c0];
    const float ks_c1 = smem_k_scale[c1];

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int m0 = 0; m0 < 2; ++m0) {
            const int r = mt * 16 + m0 * 8 + g_lane;
            const float qs_r = smem_q_scale[r];
            const float ks_r = smem_k_scale[r];

            const float val0 = (r >= c0)
                ? (qk_acc[mt].x[2 * m0 + 0] * (qs_r * ks_c0))
                : (kk_acc[mt].x[2 * m0 + 0] * (ks_r * ks_c0));
            const float val1 = (r >= c1)
                ? (qk_acc[mt].x[2 * m0 + 1] * (qs_r * ks_c1))
                : (kk_acc[mt].x[2 * m0 + 1] * (ks_r * ks_c1));

            *reinterpret_cast<float2 *>(&prep->lm[r * GDNA_CHUNK + c0]) = make_float2(val0, val1);
        }
    }
#else
    GGML_UNUSED_VARS(q, k, prep_all, lo_all, n_full_chunks, neqk1, sq1, sq2, sq3, rq3_magic);
    NO_DEVICE_CODE;
#endif
}

template <bool keep_rs_t, bool emit_ingredients_t, typename state_t, int prec_t>
__global__ void __launch_bounds__(WARP_SIZE * GDNA_WARPS, 3)
gdn_tc_carry_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        const gdn_tc_chunk_prep * __restrict__ prep_all,
        const half2 * __restrict__ lo_all,
        float * __restrict__ dst,
        void * __restrict__ state,
        int64_t H,
        int64_t n_tokens,
        int64_t n_full_chunks,
        int64_t n_seqs,
        int64_t neqk1,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        const uint3 neqk1_magic,
        const uint3 rq3_magic,
        float scale,
        int64_t state_slot_stride,
        int K,
        const int32_t * __restrict__ slot_rows,
        int ingr_w,
        int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src state_src,
        const int pipe) {
#if defined(TURING_MMA_AVAILABLE)
    static_assert(!emit_ingredients_t || std::is_same_v<state_t, float>, "emit_mode 1/2 writes F32 only");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const uint32_t col_grp  = blockIdx.z;

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int tid    = warp * WARP_SIZE + lane;
    const int g_lane = lane / 4;
    const int t_lane = lane % 4;

    const int col_base = col_grp * GDNA_COLS_PER_BLOCK + warp * GDNA_COLS_PER_WARP;
    const int col0     = col_base + 2 * t_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    __shared__ __align__(16) half2 smem_k[GDNA_CHUNK * (GDNA_S_V / 2)];
    __shared__ __align__(16) half2 smem_q[GDNA_CHUNK * (GDNA_S_V / 2)];
    __shared__ float smem_lm[GDNA_CHUNK][GDNA_LM_STRIDE];
    __shared__ float smem_g_cum[GDNA_CHUNK];
    __shared__ float smem_beta[GDNA_CHUNK];
    __shared__ __align__(16) float smem_k_scale[GDNA_CHUNK];
    __shared__ __align__(16) float smem_q_scale[GDNA_CHUNK];
    __shared__ float smem_m_scale[GDNA_CHUNK];
    constexpr bool s_split  = (prec_t & GDNA_P_S)  != 0;
    constexpr bool d_split  = (prec_t & (GDNA_P_D | GDNA_P_DMD)) != 0; // Delta lo in M*Delta
    constexpr bool m_split  = (prec_t & (GDNA_P_D | GDNA_P_DMM)) != 0; // M lo in M*Delta
    constexpr bool du_split = (prec_t & GDNA_P_DU) != 0;
    constexpr bool qk_lo    = (prec_t & GDNA_P_QK) != 0;
    constexpr bool k_lo_s   = qk_lo || (prec_t & GDNA_P_KLO) != 0;               // K lo in K*S
    constexpr bool k_lo_on  = k_lo_s || (prec_t & GDNA_P_KLOU) != 0;             // K lo in K^T*Delta (and loaded)
    constexpr int  lo_stride = (qk_lo ? 2 : 1) * GDNA_CHUNK * (GDNA_S_V / 2);
    __shared__ __align__(16) half2 smem_k_lo[k_lo_on ? GDNA_CHUNK * (GDNA_S_V / 2) : 4];
    __shared__ __align__(16) half2 smem_q_lo[qk_lo   ? GDNA_CHUNK * (GDNA_S_V / 2) : 4];

    ggml_cuda_pdl_sync();

    const int64_t state_in_offset = (sequence * H + h_idx) * (GDNA_S_V * GDNA_S_V);
    const int64_t state_out_e     = emit_ingredients_t
        ? (sequence * H + h_idx) * (int64_t) ingr_w
        : (sequence * H + h_idx) * (GDNA_S_V * GDNA_S_V);
    float * const state_f = (float *) state;

    const float * s_in = curr_state + (state_src.rows != nullptr
        ? (int64_t) state_src.rows[sequence] * state_src.row_stride + h_idx * (GDNA_S_V * GDNA_S_V)
        : state_in_offset);

    // Persistent in-register state: 8 tiles of 16x8 floats per warp (128x8 slice of S).
    ggml_cuda_mma::tile<16, 8, float> s_frag[8];
#pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        const int d0 = kd * 16 + g_lane;
        const int d1 = d0 + 8;
        s_frag[kd].x[0] = s_in[(col0 + 0) * GDNA_S_V + d0];
        s_frag[kd].x[1] = s_in[(col0 + 1) * GDNA_S_V + d0];
        s_frag[kd].x[2] = s_in[(col0 + 0) * GDNA_S_V + d1];
        s_frag[kd].x[3] = s_in[(col0 + 1) * GDNA_S_V + d1];
    }

    const float * v_base = v    + sequence * sv3 + h_idx * sv1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + sequence * sb3 + h_idx * sb1;
    float * attn_base    = dst  + (sequence * n_tokens * H + h_idx) * GDNA_S_V;
    const gdn_tc_chunk_prep * prep_seq = prep_all + ((int64_t) sequence * neqk1 + iq1) * n_full_chunks;

#ifdef CP_ASYNC_AVAILABLE
    // GGML_CUDA_GDN_CHUNKED_PIPE: 1 = cp.async chunk planes (no register staging) + L2 prefetch of chunk ch+1;
    // 2 = additionally stream the next chunk's Q plane into smem_q as soon as K*S/Q*S is done (one extra barrier).
    // +4 = skip the L2 prefetch (A/B).
    const bool pipe_on = (pipe & 3) != 0;
    const bool pipe_q  = (pipe & 3) >= 2;
    const bool pipe_l2 = (pipe & 4) == 0;
#else
    constexpr bool pipe_on = false;
    constexpr bool pipe_q  = false;
    GGML_UNUSED(pipe);
#endif

    for (int64_t ch = 0; ch < n_full_chunks; ++ch) {
        const int64_t t0 = ch * GDNA_CHUNK;
        const gdn_tc_chunk_prep * prep = prep_seq + ch;

#ifdef CP_ASYNC_AVAILABLE
        if (pipe_on) {
            // L2 prefetch of everything chunk ch+1 reads from global memory (prep/lo shared by the CTAs of a q/k head).
            if (pipe_l2 && ch + 1 < n_full_chunks) {
                const char * pn = (const char *) (prep + 1);
                constexpr int prep_lines = (int) (sizeof(gdn_tc_chunk_prep) / 128);
                for (int i = tid; i < prep_lines; i += WARP_SIZE * GDNA_WARPS) {
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(pn + 128 * i));
                }
                if constexpr (k_lo_on) {
                    const char * ln = (const char *) (lo_all + (prep + 1 - prep_all) * lo_stride);
                    constexpr int lo_lines = (int) (lo_stride * sizeof(half2) / 128);
                    for (int i = tid; i < lo_lines; i += WARP_SIZE * GDNA_WARPS) {
                        asm volatile("prefetch.global.L2 [%0];" : : "l"(ln + 128 * i));
                    }
                }
                const int64_t tn = t0 + GDNA_CHUNK;
                if (warp == 0) {
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(v_base + (tn + lane) * sv2 + col_grp * GDNA_COLS_PER_BLOCK));
                } else if (warp == 1) {
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(g_base + (tn + lane) * sb2));
                } else if (warp == 2) {
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(b_base + (tn + lane) * sb2));
                }
            }
            const bool load_q = !pipe_q || ch == 0;
#pragma unroll
            for (int idx = tid * 4; idx < GDNA_CHUNK * (GDNA_S_V / 2); idx += (WARP_SIZE * GDNA_WARPS) * 4) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_k + idx), prep->k_swz + idx);
                if (load_q) {
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_q + idx), prep->q_swz + idx);
                }
                if constexpr (k_lo_on) {
                    const half2 * lo = lo_all + (prep - prep_all) * lo_stride;
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_k_lo + idx), lo + idx);
                }
                if constexpr (qk_lo) {
                    const half2 * lo = lo_all + (prep - prep_all) * lo_stride;
                    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_q_lo + idx), lo + GDNA_CHUNK * (GDNA_S_V / 2) + idx);
                }
            }
            if (tid < 8) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_k_scale + 4 * tid), prep->k_scale + 4 * tid);
            } else if (tid < 16) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_q_scale + 4 * (tid - 8)), prep->q_scale + 4 * (tid - 8));
            }
        } else
#endif
        {
#pragma unroll
        for (int idx = tid * 4; idx < GDNA_CHUNK * (GDNA_S_V / 2); idx += (WARP_SIZE * GDNA_WARPS) * 4) {
            *reinterpret_cast<int4 *>(smem_k + idx) = *reinterpret_cast<const int4 *>(prep->k_swz + idx);
            *reinterpret_cast<int4 *>(smem_q + idx) = *reinterpret_cast<const int4 *>(prep->q_swz + idx);
            if constexpr (k_lo_on) {
                const half2 * lo = lo_all + (prep - prep_all) * lo_stride;
                *reinterpret_cast<int4 *>(smem_k_lo + idx) = *reinterpret_cast<const int4 *>(lo + idx);
            }
            if constexpr (qk_lo) {
                const half2 * lo = lo_all + (prep - prep_all) * lo_stride;
                *reinterpret_cast<int4 *>(smem_q_lo + idx) =
                    *reinterpret_cast<const int4 *>(lo + GDNA_CHUNK * (GDNA_S_V / 2) + idx);
            }
        }
        if (tid < GDNA_CHUNK) {
            smem_k_scale[tid] = prep->k_scale[tid];
            smem_q_scale[tid] = prep->q_scale[tid];
        }
        }

        float g_cum_lane = g_base[(t0 + lane) * sb2];
        const float b_lane = b_base[(t0 + lane) * sb2];
#pragma unroll
        for (int offset = 1; offset < GDNA_CHUNK; offset <<= 1) {
            const float other = __shfl_up_sync(0xffffffff, g_cum_lane, offset);
            if (lane >= offset) {
                g_cum_lane += other;
            }
        }
        if (warp == 0) {
            smem_g_cum[lane] = g_cum_lane;
            smem_beta[lane]  = b_lane;
        }

#pragma unroll
        for (int k_row = 0; k_row < 8; ++k_row) {
            const int r = warp * 8 + k_row;
            const float g_r   = __shfl_sync(0xffffffff, g_cum_lane, r);
            const float raw   = prep->lm[r * GDNA_CHUNK + lane];
            const float m_val = (r >= lane) ? (raw * expf(g_r - g_cum_lane)) : 0.0f;
            const float m_max = warp_reduce_max<WARP_SIZE>(fabsf(m_val));
            float m_sc, m_inv;
            gdn_tc_pow2_scale(m_max, m_sc, m_inv);
            if (lane == 0) {
                smem_m_scale[r] = m_sc;
            }
            smem_lm[r][lane] = (r >= lane)
                ? (m_val * m_inv)
                : (raw * b_lane * expf(g_cum_lane - g_r));
        }

#ifdef CP_ASYNC_AVAILABLE
        if (pipe_on) {
            cp_async_wait_all();
        }
#endif
        // Barrier 1: all chunk shared memory is ready and strictly read-only until Barrier 2.
        __syncthreads();

        float s_max0 = 0.0f;
        float s_max1 = 0.0f;
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            s_max0 = fmaxf(s_max0, fmaxf(fabsf(s_frag[kd].x[0]), fabsf(s_frag[kd].x[2])));
            s_max1 = fmaxf(s_max1, fmaxf(fabsf(s_frag[kd].x[1]), fabsf(s_frag[kd].x[3])));
        }
#pragma unroll
        for (int mask = 4; mask <= 16; mask <<= 1) {
            s_max0 = fmaxf(s_max0, __shfl_xor_sync(0xffffffff, s_max0, mask));
            s_max1 = fmaxf(s_max1, __shfl_xor_sync(0xffffffff, s_max1, mask));
        }
        float s_scale0, s_inv0, s_scale1, s_inv1;
        gdn_tc_pow2_scale(s_max0, s_scale0, s_inv0);
        gdn_tc_pow2_scale(s_max1, s_scale1, s_inv1);

        ggml_cuda_mma::tile<16, 8, float> delta_frag[2];
        ggml_cuda_mma::tile<16, 8, float> qs_frag[2];
        {
            ggml_cuda_mma::tile<16, 8, float> ks_frag[2];
#pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                ggml_cuda_mma::tile<16, 8, float> s_scaled;
                s_scaled.x[0] = s_frag[kd].x[0] * s_inv0;
                s_scaled.x[1] = s_frag[kd].x[1] * s_inv1;
                s_scaled.x[2] = s_frag[kd].x[2] * s_inv0;
                s_scaled.x[3] = s_frag[kd].x[3] * s_inv1;
                const ggml_cuda_mma::tile<8, 8, half2> s_b =
                    ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(s_scaled));
                ggml_cuda_mma::tile<8, 8, half2> s_b_lo;
                if constexpr (s_split) {
                    s_b_lo = ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(gdn_tc_residual_tile(s_scaled)));
                }

#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    ggml_cuda_mma::tile<16, 8, half2> k_a;
                    ggml_cuda_mma::tile<16, 8, half2> q_a;
                    ggml_cuda_mma::load_ldmatrix<true>(k_a, smem_k, mt * 16, kd * 8, GDNA_S_V / 2);
                    ggml_cuda_mma::load_ldmatrix<true>(q_a, smem_q, mt * 16, kd * 8, GDNA_S_V / 2);
                    ggml_cuda_mma::mma(ks_frag[mt], k_a, s_b);
                    ggml_cuda_mma::mma(qs_frag[mt], q_a, s_b);
                    if constexpr (s_split) {
                        ggml_cuda_mma::mma(ks_frag[mt], k_a, s_b_lo);
                        ggml_cuda_mma::mma(qs_frag[mt], q_a, s_b_lo);
                    }
                    if constexpr (k_lo_s) {
                        ggml_cuda_mma::tile<16, 8, half2> k_a_lo;
                        ggml_cuda_mma::load_ldmatrix<true>(k_a_lo, smem_k_lo, mt * 16, kd * 8, GDNA_S_V / 2);
                        ggml_cuda_mma::mma(ks_frag[mt], k_a_lo, s_b);
                    }
                    if constexpr (qk_lo) {
                        ggml_cuda_mma::tile<16, 8, half2> q_a_lo;
                        ggml_cuda_mma::load_ldmatrix<true>(q_a_lo, smem_q_lo, mt * 16, kd * 8, GDNA_S_V / 2);
                        ggml_cuda_mma::mma(qs_frag[mt], q_a_lo, s_b);
                    }
                }
            }

#ifdef CP_ASYNC_AVAILABLE
            if (pipe_q) {
                // smem_q (and smem_q_lo) are dead for this chunk once every warp is past K*S/Q*S.
                __syncthreads();
                if (ch + 1 < n_full_chunks) {
                    const gdn_tc_chunk_prep * prep_n = prep + 1;
#pragma unroll
                    for (int idx = tid * 4; idx < GDNA_CHUNK * (GDNA_S_V / 2); idx += (WARP_SIZE * GDNA_WARPS) * 4) {
                        cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(smem_q + idx), prep_n->q_swz + idx);
                    }
                }
            }
#endif

#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int m0 = 0; m0 < 2; ++m0) {
                    const int r = mt * 16 + m0 * 8 + g_lane;
                    // exp(G_r) is applied after the power-of-two scales so a large carried state cannot hide
                    // behind an exp underflow (the "carry" fixture).
                    const float g_r    = smem_g_cum[r];
                    const float beta_r = smem_beta[r];
                    const float k_sc0  = gdn_tc_mul_exp(smem_k_scale[r] * s_scale0, g_r);
                    const float k_sc1  = gdn_tc_mul_exp(smem_k_scale[r] * s_scale1, g_r);
                    const float q_sc0  = gdn_tc_mul_exp(smem_q_scale[r] * s_scale0, g_r);
                    const float q_sc1  = gdn_tc_mul_exp(smem_q_scale[r] * s_scale1, g_r);

                    const float2 v_pair = *reinterpret_cast<const float2 *>(v_base + (t0 + r) * sv2 + col0);
                    const float ks0 = ks_frag[mt].x[2 * m0 + 0] * k_sc0;
                    const float ks1 = ks_frag[mt].x[2 * m0 + 1] * k_sc1;

                    delta_frag[mt].x[2 * m0 + 0] = beta_r * (v_pair.x - ks0);
                    delta_frag[mt].x[2 * m0 + 1] = beta_r * (v_pair.y - ks1);

                    qs_frag[mt].x[2 * m0 + 0] *= q_sc0;
                    qs_frag[mt].x[2 * m0 + 1] *= q_sc1;
                }
            }
        }

        // In-register warp-shuffle forward substitution for (I + L) * Delta = B (0 barriers).
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int src_lane = 4 * p + t_lane;
            const float piv0 = __shfl_sync(0xffffffff, delta_frag[0].x[0], src_lane);
            const float piv1 = __shfl_sync(0xffffffff, delta_frag[0].x[1], src_lane);
            if (g_lane > p) {
                const float l0 = smem_lm[p][g_lane];
                delta_frag[0].x[0] -= l0 * piv0;
                delta_frag[0].x[1] -= l0 * piv1;
            }
            {
                const float l1 = smem_lm[p][g_lane + 8];
                delta_frag[0].x[2] -= l1 * piv0;
                delta_frag[0].x[3] -= l1 * piv1;
            }
            {
                const float l2 = smem_lm[p][g_lane + 16];
                delta_frag[1].x[0] -= l2 * piv0;
                delta_frag[1].x[1] -= l2 * piv1;
                const float l3 = smem_lm[p][g_lane + 24];
                delta_frag[1].x[2] -= l3 * piv0;
                delta_frag[1].x[3] -= l3 * piv1;
            }
        }
#pragma unroll
        for (int p = 8; p < 16; ++p) {
            const int src_lane = 4 * (p - 8) + t_lane;
            const float piv0 = __shfl_sync(0xffffffff, delta_frag[0].x[2], src_lane);
            const float piv1 = __shfl_sync(0xffffffff, delta_frag[0].x[3], src_lane);
            if (g_lane + 8 > p) {
                const float l1 = smem_lm[p][g_lane + 8];
                delta_frag[0].x[2] -= l1 * piv0;
                delta_frag[0].x[3] -= l1 * piv1;
            }
            {
                const float l2 = smem_lm[p][g_lane + 16];
                delta_frag[1].x[0] -= l2 * piv0;
                delta_frag[1].x[1] -= l2 * piv1;
                const float l3 = smem_lm[p][g_lane + 24];
                delta_frag[1].x[2] -= l3 * piv0;
                delta_frag[1].x[3] -= l3 * piv1;
            }
        }
#pragma unroll
        for (int p = 16; p < 24; ++p) {
            const int src_lane = 4 * (p - 16) + t_lane;
            const float piv0 = __shfl_sync(0xffffffff, delta_frag[1].x[0], src_lane);
            const float piv1 = __shfl_sync(0xffffffff, delta_frag[1].x[1], src_lane);
            if (g_lane + 16 > p) {
                const float l2 = smem_lm[p][g_lane + 16];
                delta_frag[1].x[0] -= l2 * piv0;
                delta_frag[1].x[1] -= l2 * piv1;
            }
            {
                const float l3 = smem_lm[p][g_lane + 24];
                delta_frag[1].x[2] -= l3 * piv0;
                delta_frag[1].x[3] -= l3 * piv1;
            }
        }
#pragma unroll
        for (int p = 24; p < 31; ++p) {
            const int src_lane = 4 * (p - 24) + t_lane;
            const float piv0 = __shfl_sync(0xffffffff, delta_frag[1].x[2], src_lane);
            const float piv1 = __shfl_sync(0xffffffff, delta_frag[1].x[3], src_lane);
            if (g_lane + 24 > p) {
                const float l3 = smem_lm[p][g_lane + 24];
                delta_frag[1].x[2] -= l3 * piv0;
                delta_frag[1].x[3] -= l3 * piv1;
            }
        }

        // Intra-chunk output: Out = scale * (exp(G) * (Q * S_init) + M * Delta).
        {
            float d_max0 = fmaxf(
                fmaxf(fabsf(delta_frag[0].x[0]), fabsf(delta_frag[0].x[2])),
                fmaxf(fabsf(delta_frag[1].x[0]), fabsf(delta_frag[1].x[2])));
            float d_max1 = fmaxf(
                fmaxf(fabsf(delta_frag[0].x[1]), fabsf(delta_frag[0].x[3])),
                fmaxf(fabsf(delta_frag[1].x[1]), fabsf(delta_frag[1].x[3])));
#pragma unroll
            for (int mask = 4; mask <= 16; mask <<= 1) {
                d_max0 = fmaxf(d_max0, __shfl_xor_sync(0xffffffff, d_max0, mask));
                d_max1 = fmaxf(d_max1, __shfl_xor_sync(0xffffffff, d_max1, mask));
            }
            float d_scale0, d_inv0, d_scale1, d_inv1;
            gdn_tc_pow2_scale(d_max0, d_scale0, d_inv0);
            gdn_tc_pow2_scale(d_max1, d_scale1, d_inv1);

            ggml_cuda_mma::tile<8, 8, half2> d_b[2];
            ggml_cuda_mma::tile<8, 8, half2> d_b_lo[2];
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                ggml_cuda_mma::tile<16, 8, float> d_scaled;
                d_scaled.x[0] = delta_frag[kt].x[0] * d_inv0;
                d_scaled.x[1] = delta_frag[kt].x[1] * d_inv1;
                d_scaled.x[2] = delta_frag[kt].x[2] * d_inv0;
                d_scaled.x[3] = delta_frag[kt].x[3] * d_inv1;
                d_b[kt] = ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(d_scaled));
                if constexpr (d_split) {
                    d_b_lo[kt] = ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(gdn_tc_residual_tile(d_scaled)));
                }
            }

            ggml_cuda_mma::tile<16, 8, float> md_frag[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r0 = mt * 16 + g_lane;
                const int r1 = r0 + 8;
#pragma unroll
                for (int kt = 0; kt <= mt; ++kt) {
                    const int c0_m = kt * 16 + 2 * t_lane;
                    const int c1_m = c0_m + 8;
                    ggml_cuda_mma::tile<16, 8, half2> m_a;
                    float mf[8];
                    if (mt == kt) {
                        const float m00 = (r0 >= c0_m + 0) ? smem_lm[r0][c0_m + 0] : 0.0f;
                        const float m01 = (r0 >= c0_m + 1) ? smem_lm[r0][c0_m + 1] : 0.0f;
                        const float m10 = (r1 >= c0_m + 0) ? smem_lm[r1][c0_m + 0] : 0.0f;
                        const float m11 = (r1 >= c0_m + 1) ? smem_lm[r1][c0_m + 1] : 0.0f;
                        const float m20 = (r0 >= c1_m + 0) ? smem_lm[r0][c1_m + 0] : 0.0f;
                        const float m21 = (r0 >= c1_m + 1) ? smem_lm[r0][c1_m + 1] : 0.0f;
                        const float m30 = (r1 >= c1_m + 0) ? smem_lm[r1][c1_m + 0] : 0.0f;
                        const float m31 = (r1 >= c1_m + 1) ? smem_lm[r1][c1_m + 1] : 0.0f;
                        mf[0] = m00; mf[1] = m01; mf[2] = m10; mf[3] = m11;
                        mf[4] = m20; mf[5] = m21; mf[6] = m30; mf[7] = m31;
                    } else {
                        const float2 f0 = *reinterpret_cast<const float2 *>(&smem_lm[r0][c0_m]);
                        const float2 f1 = *reinterpret_cast<const float2 *>(&smem_lm[r1][c0_m]);
                        const float2 f2 = *reinterpret_cast<const float2 *>(&smem_lm[r0][c1_m]);
                        const float2 f3 = *reinterpret_cast<const float2 *>(&smem_lm[r1][c1_m]);
                        mf[0] = f0.x; mf[1] = f0.y; mf[2] = f1.x; mf[3] = f1.y;
                        mf[4] = f2.x; mf[5] = f2.y; mf[6] = f3.x; mf[7] = f3.y;
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        m_a.x[e] = make_half2(mf[2 * e + 0], mf[2 * e + 1]);
                    }
                    ggml_cuda_mma::mma(md_frag[mt], m_a, d_b[kt]);
                    if constexpr (d_split) {
                        ggml_cuda_mma::mma(md_frag[mt], m_a, d_b_lo[kt]);
                    }
                    if constexpr (m_split) {
                        ggml_cuda_mma::tile<16, 8, half2> m_a_lo;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            m_a_lo.x[e] = make_half2(gdn_tc_f16_residual(mf[2 * e + 0]), gdn_tc_f16_residual(mf[2 * e + 1]));
                        }
                        ggml_cuda_mma::mma(md_frag[mt], m_a_lo, d_b[kt]);
                    }
                }

#pragma unroll
                for (int m0 = 0; m0 < 2; ++m0) {
                    const int r = mt * 16 + m0 * 8 + g_lane;
                    const float m_sc = smem_m_scale[r];
                    const float out0 = (qs_frag[mt].x[2 * m0 + 0] + md_frag[mt].x[2 * m0 + 0] * (m_sc * d_scale0)) * scale;
                    const float out1 = (qs_frag[mt].x[2 * m0 + 1] + md_frag[mt].x[2 * m0 + 1] * (m_sc * d_scale1)) * scale;
                    *reinterpret_cast<float2 *>(attn_base + (t0 + r) * (GDNA_S_V * H) + col0) = make_float2(out0, out1);
                }
            }
        }

        // State recurrence update: S = exp(G_last) * S + K^T * Delta_tilde.
        {
            const float g_last = smem_g_cum[GDNA_CHUNK - 1];
            float td_max0 = 0.0f;
            float td_max1 = 0.0f;
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
#pragma unroll
                for (int m0 = 0; m0 < 2; ++m0) {
                    const int j = kt * 16 + m0 * 8 + g_lane;
                    const float factor = gdn_tc_mul_exp(smem_k_scale[j], g_last - smem_g_cum[j]);
                    delta_frag[kt].x[2 * m0 + 0] *= factor;
                    delta_frag[kt].x[2 * m0 + 1] *= factor;
                    td_max0 = fmaxf(td_max0, fabsf(delta_frag[kt].x[2 * m0 + 0]));
                    td_max1 = fmaxf(td_max1, fabsf(delta_frag[kt].x[2 * m0 + 1]));
                }
            }
#pragma unroll
            for (int mask = 4; mask <= 16; mask <<= 1) {
                td_max0 = fmaxf(td_max0, __shfl_xor_sync(0xffffffff, td_max0, mask));
                td_max1 = fmaxf(td_max1, __shfl_xor_sync(0xffffffff, td_max1, mask));
            }
            float td_scale0, td_inv0, td_scale1, td_inv1;
            gdn_tc_pow2_scale(td_max0, td_scale0, td_inv0);
            gdn_tc_pow2_scale(td_max1, td_scale1, td_inv1);

            ggml_cuda_mma::tile<8, 8, half2> td_b[2];
            ggml_cuda_mma::tile<8, 8, half2> td_b_lo[2];
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                ggml_cuda_mma::tile<16, 8, float> td_scaled;
                td_scaled.x[0] = delta_frag[kt].x[0] * td_inv0;
                td_scaled.x[1] = delta_frag[kt].x[1] * td_inv1;
                td_scaled.x[2] = delta_frag[kt].x[2] * td_inv0;
                td_scaled.x[3] = delta_frag[kt].x[3] * td_inv1;
                td_b[kt] = ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(td_scaled));
                if constexpr (du_split) {
                    td_b_lo[kt] = ggml_cuda_mma::get_transposed(ggml_cuda_mma::get_half2(gdn_tc_residual_tile(td_scaled)));
                }
            }

            // exp(G_last) can underflow while the decayed state is still representable: decay in steps.
            const bool  big_decay  = g_last < -80.0f;
            const float exp_g_last = big_decay ? 0.0f : expf(g_last);
#pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                ggml_cuda_mma::tile<16, 8, float> kt_d_acc;
#pragma unroll
                for (int kt = 0; kt < 2; ++kt) {
                    ggml_cuda_mma::tile<16, 8, half2> kt_a;
                    gdn_load_ldmatrix_trans_swz(kt_a, smem_k, kt * 16, kd * 8, GDNA_S_V / 2);
                    ggml_cuda_mma::mma(kt_d_acc, kt_a, td_b[kt]);
                    if constexpr (du_split) {
                        ggml_cuda_mma::mma(kt_d_acc, kt_a, td_b_lo[kt]);
                    }
                    if constexpr (k_lo_on) {
                        ggml_cuda_mma::tile<16, 8, half2> kt_a_lo;
                        gdn_load_ldmatrix_trans_swz(kt_a_lo, smem_k_lo, kt * 16, kd * 8, GDNA_S_V / 2);
                        ggml_cuda_mma::mma(kt_d_acc, kt_a_lo, td_b[kt]);
                    }
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float td_s = (l & 1) ? td_scale1 : td_scale0;
                    const float s_old = big_decay ? gdn_tc_mul_exp(s_frag[kd].x[l], g_last) : exp_g_last * s_frag[kd].x[l];
                    s_frag[kd].x[l] = s_old + kt_d_acc.x[l] * td_s;
                }
            }
        }

        // Barrier 2: protect shared memory before next chunk load.
        __syncthreads();
    }

    // Sequential in-register tail for remaining tokens (ragged n_tokens or keep_rs checkpoint window).
    const int64_t tail_start = n_full_chunks * GDNA_CHUNK;
    if (tail_start < n_tokens) {
        const float * q_base = q + iq3 * sq3 + iq1 * sq1;
        const float * k_base = k + iq3 * sq3 + iq1 * sq1;
        float * smem_k_tail  = &smem_lm[0][0];
        float * smem_q_tail  = &smem_lm[4][0];

        for (int64_t t = tail_start; t < n_tokens; ++t) {
            smem_k_tail[tid] = k_base[t * sq2 + tid];
            smem_q_tail[tid] = q_base[t * sq2 + tid];
            if (tid == 0) {
                smem_g_cum[0] = g_base[t * sb2];
                smem_beta[0]  = b_base[t * sb2];
            }
            __syncthreads();

            const float g_raw_t = smem_g_cum[0];
            const float g_val_t = expf(g_raw_t);
            const float beta_t  = smem_beta[0];
            const float2 v_pair = *reinterpret_cast<const float2 *>(v_base + t * sv2 + col0);

            float kv0 = 0.0f;
            float kv1 = 0.0f;
#pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                const int d0 = kd * 16 + g_lane;
                const int d1 = d0 + 8;
                const float k0 = smem_k_tail[d0];
                const float k1 = smem_k_tail[d1];
                kv0 += s_frag[kd].x[0] * k0 + s_frag[kd].x[2] * k1;
                kv1 += s_frag[kd].x[1] * k0 + s_frag[kd].x[3] * k1;
            }
#pragma unroll
            for (int mask = 4; mask <= 16; mask <<= 1) {
                kv0 += __shfl_xor_sync(0xffffffff, kv0, mask);
                kv1 += __shfl_xor_sync(0xffffffff, kv1, mask);
            }

            const float delta0 = (v_pair.x - g_val_t * kv0) * beta_t;
            const float delta1 = (v_pair.y - g_val_t * kv1) * beta_t;

            float attn0 = 0.0f;
            float attn1 = 0.0f;
#pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                const int d0 = kd * 16 + g_lane;
                const int d1 = d0 + 8;
                const float k0 = smem_k_tail[d0];
                const float k1 = smem_k_tail[d1];
                const float q0 = smem_q_tail[d0];
                const float q1 = smem_q_tail[d1];

                s_frag[kd].x[0] = g_val_t * s_frag[kd].x[0] + k0 * delta0;
                s_frag[kd].x[1] = g_val_t * s_frag[kd].x[1] + k0 * delta1;
                s_frag[kd].x[2] = g_val_t * s_frag[kd].x[2] + k1 * delta0;
                s_frag[kd].x[3] = g_val_t * s_frag[kd].x[3] + k1 * delta1;

                attn0 += s_frag[kd].x[0] * q0 + s_frag[kd].x[2] * q1;
                attn1 += s_frag[kd].x[1] * q0 + s_frag[kd].x[3] * q1;
            }
#pragma unroll
            for (int mask = 4; mask <= 16; mask <<= 1) {
                attn0 += __shfl_xor_sync(0xffffffff, attn0, mask);
                attn1 += __shfl_xor_sync(0xffffffff, attn1, mask);
            }

            if (g_lane == 0) {
                *reinterpret_cast<float2 *>(attn_base + t * (GDNA_S_V * H) + col0) =
                    make_float2(attn0 * scale, attn1 * scale);
            }

            if constexpr (keep_rs_t) {
                const int target_slot = emit_ingredients_t ? ((int) t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - (int) t);
                if (target_slot >= 0 && target_slot < K) {
                    if constexpr (emit_ingredients_t) {
                        float * ingr_slot = state_f + state_out_e + (int64_t) target_slot * state_slot_stride;
                        const bool compact = ingr_w != 4 * GDNA_S_V;
                        if (col_grp == 0) {
                            ingr_slot[tid] = smem_k_tail[tid];
                            if (!compact) {
                                ingr_slot[2 * GDNA_S_V + tid] = g_raw_t;
                                ingr_slot[3 * GDNA_S_V + tid] = beta_t;
                            } else if (tid == 0) {
                                ingr_slot[2 * GDNA_S_V]     = g_raw_t;
                                ingr_slot[2 * GDNA_S_V + 1] = beta_t;
                            }
                        }
                        if (g_lane == 0) {
                            ingr_slot[GDNA_S_V + col0 + 0] = v_pair.x;
                            ingr_slot[GDNA_S_V + col0 + 1] = v_pair.y;
                        }
                    } else {
                        const int64_t snap_e = slot_rows == nullptr
                            ? state_out_e + (int64_t) target_slot * state_slot_stride
                            : (int64_t) slot_rows[target_slot * n_seqs + sequence] * (H * GDNA_S_V * GDNA_S_V) + h_idx * (GDNA_S_V * GDNA_S_V);
#pragma unroll
                        for (int kd = 0; kd < 8; ++kd) {
                            const int d0 = kd * 16 + g_lane;
                            const int d1 = d0 + 8;
                            gdn_store_state<state_t, WARP_SIZE>(state, snap_e + (col0 + 0) * GDNA_S_V + d0, s_frag[kd].x[0]);
                            gdn_store_state<state_t, WARP_SIZE>(state, snap_e + (col0 + 1) * GDNA_S_V + d0, s_frag[kd].x[1]);
                            gdn_store_state<state_t, WARP_SIZE>(state, snap_e + (col0 + 0) * GDNA_S_V + d1, s_frag[kd].x[2]);
                            gdn_store_state<state_t, WARP_SIZE>(state, snap_e + (col0 + 1) * GDNA_S_V + d1, s_frag[kd].x[3]);
                        }
                    }
                }

                if constexpr (emit_ingredients_t) {
                    const int t_ckpt = (int) n_tokens - K - 1;
                    if (t_ckpt >= 0 && (int) t == t_ckpt) {
                        float * ckpt_slot = state_f + ingr_region +
                                            (int64_t) GDNA_S_V * GDNA_S_V * H * n_seqs + (sequence * H + h_idx) * (GDNA_S_V * GDNA_S_V);
#pragma unroll
                        for (int kd = 0; kd < 8; ++kd) {
                            const int d0 = kd * 16 + g_lane;
                            const int d1 = d0 + 8;
                            ckpt_slot[(col0 + 0) * GDNA_S_V + d0] = s_frag[kd].x[0];
                            ckpt_slot[(col0 + 1) * GDNA_S_V + d0] = s_frag[kd].x[1];
                            ckpt_slot[(col0 + 0) * GDNA_S_V + d1] = s_frag[kd].x[2];
                            ckpt_slot[(col0 + 1) * GDNA_S_V + d1] = s_frag[kd].x[3];
                        }
                    }
                }
            }

            __syncthreads();
        }
    }

    if constexpr (emit_ingredients_t) {
        float * final_slot = state_f + ingr_region + (sequence * H + h_idx) * (GDNA_S_V * GDNA_S_V);
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            const int d0 = kd * 16 + g_lane;
            const int d1 = d0 + 8;
            final_slot[(col0 + 0) * GDNA_S_V + d0] = s_frag[kd].x[0];
            final_slot[(col0 + 1) * GDNA_S_V + d0] = s_frag[kd].x[1];
            final_slot[(col0 + 0) * GDNA_S_V + d1] = s_frag[kd].x[2];
            final_slot[(col0 + 1) * GDNA_S_V + d1] = s_frag[kd].x[3];
        }
    }

    if constexpr (!keep_rs_t) {
        // Final-only writes use the selected cache row just like retained snapshots.
        const int64_t final_out_e = slot_rows == nullptr ? state_out_e
            : (int64_t) slot_rows[sequence] * (H * GDNA_S_V * GDNA_S_V)
                + h_idx * (GDNA_S_V * GDNA_S_V);
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            const int d0 = kd * 16 + g_lane;
            const int d1 = d0 + 8;
            gdn_store_state<state_t, WARP_SIZE>(state, final_out_e + (col0 + 0) * GDNA_S_V + d0, s_frag[kd].x[0]);
            gdn_store_state<state_t, WARP_SIZE>(state, final_out_e + (col0 + 1) * GDNA_S_V + d0, s_frag[kd].x[1]);
            gdn_store_state<state_t, WARP_SIZE>(state, final_out_e + (col0 + 0) * GDNA_S_V + d1, s_frag[kd].x[2]);
            gdn_store_state<state_t, WARP_SIZE>(state, final_out_e + (col0 + 1) * GDNA_S_V + d1, s_frag[kd].x[3]);
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, curr_state, prep_all, lo_all, dst, state, H, n_tokens, n_full_chunks,
        n_seqs, neqk1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
        scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, pipe);
    NO_DEVICE_CODE;
#endif
}

static bool ggml_cuda_gdn_chunked_enabled() {
    static const bool on = [] {
        // [PREFILLFAST] opt-in on this branch (upstream bl/main-gdn-chunked defaults it on): GGML_CUDA_GDN_CHUNKED=1
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED");
        return s != nullptr && atoi(s) != 0;
    }();
    return on;
}

static int ggml_cuda_gdn_chunked_pipe() {
    static const int pipe = [] {
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED_PIPE");
        return s != nullptr ? atoi(s) : 0;
    }();
    return pipe;
}

static int ggml_cuda_gdn_chunked_prec() {
    static const int prec = [] {
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED_PREC");
        // Default 100 = M*Delta split + K^T*Delta Delta split + K low plane in K^T*Delta: the cheapest set that passes
        // the full GDN TC suite (1e-6 NMSE) with ~20x margin. 0 = single FP16 products (fails "correlated").
        const int p = s != nullptr ? atoi(s) : 100;
        switch (p) {
            case 0: case 54: case 100: case 224:
                return p;
            default:
                GGML_LOG_WARN("%s: GGML_CUDA_GDN_CHUNKED_PREC=%d is not built, using 100\n", __func__, p);
                return 100;
        }
    }();
    return prec;
}

template <bool keep_rs_t, bool emit_ingredients_t, typename state_t, int prec_t>
static void launch_gdn_chunked_prec(
        ggml_cuda_pool & pool, const dim3 & prep_grid, const dim3 & carry_grid, const dim3 & block_dims,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        gdn_tc_chunk_prep * prep, size_t n_prep,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_full_chunks, int64_t n_seqs, int64_t neqk1,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows, int ingr_w, int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    constexpr int lo_planes = (prec_t & GDNA_P_QK) ? 2 : ((prec_t & (GDNA_P_KLO | GDNA_P_KLOU)) ? 1 : 0);
    ggml_cuda_pool_alloc<half2> lo_buf(pool, lo_planes > 0 ? n_prep * lo_planes * GDNA_CHUNK * (GDNA_S_V / 2) : 1);
    const ggml_cuda_kernel_launch_params prep_params = ggml_cuda_kernel_launch_params(prep_grid, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gdn_tc_prepare_kernel<prec_t>, prep_params,
        q_d, k_d, prep, lo_buf.get(), n_full_chunks, neqk1, sq1, sq2, sq3, rq3_magic);
    const ggml_cuda_kernel_launch_params carry_params = ggml_cuda_kernel_launch_params(carry_grid, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gdn_tc_carry_kernel<keep_rs_t, emit_ingredients_t, state_t, prec_t>, carry_params,
        q_d, k_d, v_d, g_d, b_d, s_d, prep, (const half2 *) lo_buf.get(), dst_d, state_d,
        H, n_tokens, n_full_chunks, n_seqs, neqk1,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src,
        ggml_cuda_gdn_chunked_pipe());
}

template <bool keep_rs_t, bool emit_ingredients_t, typename state_t>
static bool launch_gated_delta_net_chunked_tc(
        ggml_cuda_pool & pool,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    if constexpr (std::is_same_v<state_t, block_q8_0>) {
        return false;
    }
    if (!ggml_cuda_gdn_chunked_enabled()) {
        return false;
    }
    if (S_v != GDNA_S_V || n_tokens < 2 * GDNA_CHUNK || n_seqs <= 0 || H <= 0 || neqk1 <= 0) {
        return false;
    }
    const auto & dev_info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int cc = dev_info.cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE ||
            dev_info.warp_size != WARP_SIZE) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(q_d) % 16 != 0) ||
        (reinterpret_cast<uintptr_t>(k_d) % 16 != 0) ||
        (reinterpret_cast<uintptr_t>(v_d) % 8  != 0) ||
        (sq1 % 4 != 0) || (sq2 % 4 != 0) || (sq3 % 4 != 0) ||
        (sv1 % 2 != 0) || (sv2 % 2 != 0) || (sv3 % 2 != 0)) {
        return false;
    }

    const int64_t n_tc_tokens   = keep_rs_t ? (n_tokens - K - (emit_ingredients_t ? 1 : 0)) : n_tokens;
    const int64_t n_full_chunks = (n_tc_tokens > 0) ? (n_tc_tokens / GDNA_CHUNK) : 0;
    if (n_full_chunks < 2) {
        return false;
    }

    const size_t n_prep = (size_t) n_seqs * (size_t) neqk1 * (size_t) n_full_chunks;
    ggml_cuda_pool_alloc<gdn_tc_chunk_prep> prep_buf(pool, n_prep);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const dim3 prep_grid(n_full_chunks, neqk1, n_seqs);
    const dim3 block_dims(WARP_SIZE, GDNA_WARPS, 1);
    const dim3 carry_grid(H, n_seqs, GDNA_S_V / GDNA_COLS_PER_BLOCK);

#define GDNA_CHUNKED_LAUNCH(P_) launch_gdn_chunked_prec<keep_rs_t, emit_ingredients_t, state_t, P_>( \
        pool, prep_grid, carry_grid, block_dims, q_d, k_d, v_d, g_d, b_d, s_d, prep_buf.get(), n_prep, dst_d, state_d, \
        H, n_tokens, n_full_chunks, n_seqs, neqk1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, \
        neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream)
    switch (ggml_cuda_gdn_chunked_prec()) {
        case 0:   GDNA_CHUNKED_LAUNCH(0);   break;
        case 54:  GDNA_CHUNKED_LAUNCH(54);  break;
        case 224: GDNA_CHUNKED_LAUNCH(224); break;
        default:  GDNA_CHUNKED_LAUNCH(100); break;
    }
#undef GDNA_CHUNKED_LAUNCH

    return true;
}

} // namespace gdn_argon
