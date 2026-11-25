// NEONKERN NK_DENSE_TC: Ampere int8 dense Q8_0 decode/verify head kernels (T=1..8).
//
// Default policy (default ON; GGML_CUDA_NK_DENSE_TC=0 = kill switch):
//   - MIN_ROWS = 16385 (head-only: output.weight 201024x2304; =0 restores r11 attention/grouped)
//   - MINT     = 4     (T=1..3 stay on base q8a16, which streams 201024x2304 at ~876 GB/s at T=1)
//   - HEAD_PATH = auto | tc | grid (default auto, selects per T from k_head_auto_table):
//       * grid (default for T=4..8, any K % 128 == 0): k_nk_quant_x_dp4a (~2-3 us pre-quant into
//         L1-resident qx_buf via ctx.pool()) + k_nk_head_grid_dp4a (K % 256 == 0) or
//         k_nk_head_grid_dp4a_tail4 in nk-dense-tc-tail.cu (K % 256 == 128).
//       * tc (r12 persistent, K % 256 == 0 and K <= 3072): k_nk_head_tc (NW=8 independent warps/CTA,
//         2-stage cross-tile cp.async, x quantized once per SM in SMEM, mma.m16n8k32.s32.s8).
//   - Fallback / MIN_ROWS=0 (attention or grouped, K % 256 == 0): k_nk_dense_tc (r11 4-warp 3-stage ring).

#include "nk-dense-tc.cuh"
#include "nk-dense-tc-tail.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

#define NK_DTC_MAX_GROUP      8
#define NK_DTC_STAGES         3
#define NK_DTC_ROW_WORDS      68
#define NK_DTC_STAGE_WORDS    (16 * NK_DTC_ROW_WORDS)
#define NK_DTC_STAGE_BYTES    (NK_DTC_STAGE_WORDS * 4)
#define NK_DTC_RED_FLOATS     (3 * 32 * 4)

#define NK_HEAD_MIN_ROWS_DEF  16385
#define NK_HEAD_MINT_DEF      4
#define NK_HEAD_NW            8
#define NK_HEAD_MAX_NCH       12
#define NK_HEAD_W_WORDS       (NK_HEAD_NW * 2 * NK_DTC_STAGE_WORDS)
#define NK_HEAD_W_BYTES       (NK_HEAD_W_WORDS * 4)
#define NK_HEAD_X_BLK_BYTES   (32 * 8 + 4 * 8)
#define NK_HEAD_MAX_SMEM      (NK_HEAD_W_BYTES + (NK_HEAD_MAX_NCH * 8) * NK_HEAD_X_BLK_BYTES)

static std::atomic<int64_t> g_nk_dense_tc_fires{0};
static std::atomic<int64_t> g_nk_dense_tc_head_grid_fires{0};
static std::atomic<int64_t> g_nk_dense_tc_head_tc_fires{0};
static std::atomic<int64_t> g_nk_dense_tc_dense_fires{0};

static int nk_parse_nonneg_env(const char * name, const int def_val) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return def_val;
    }
    char * end = nullptr;
    const long v = strtol(e, &end, 10);
    if (end == e || *end != '\0') {
        return def_val;
    }
    if (v < 0) {
        return 0;
    }
    return v > INT_MAX ? INT_MAX : (int) v;
}

bool ggml_cuda_nk_dense_tc_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC");
        return e == nullptr || strcmp(e, "0") != 0; // default ON; GGML_CUDA_NK_DENSE_TC=0 = kill switch
    }();
    return on;
}

int ggml_cuda_nk_dense_tc_min_rows() {
    static const int v = nk_parse_nonneg_env("GGML_CUDA_NK_DENSE_TC_MIN_ROWS", NK_HEAD_MIN_ROWS_DEF);
    return v;
}

static int nk_dense_tc_max_rows() {
    static const int v = nk_parse_nonneg_env("GGML_CUDA_NK_DENSE_TC_MAX_ROWS", 0);
    return v;
}

bool ggml_cuda_nk_dense_tc_takes_rows(int64_t nrows) {
    if (!ggml_cuda_nk_dense_tc_enabled()) {
        return false;
    }
    if (nrows < ggml_cuda_nk_dense_tc_min_rows()) {
        return false;
    }
    const int max_r = nk_dense_tc_max_rows();
    if (max_r > 0 && nrows > max_r) {
        return false;
    }
    return true;
}

int64_t ggml_cuda_nk_dense_tc_count() {
    return g_nk_dense_tc_fires.load(std::memory_order_relaxed);
}

int64_t ggml_cuda_nk_dense_tc_head_grid_count() {
    return g_nk_dense_tc_head_grid_fires.load(std::memory_order_relaxed);
}

int64_t ggml_cuda_nk_dense_tc_head_tc_count() {
    return g_nk_dense_tc_head_tc_fires.load(std::memory_order_relaxed);
}

int64_t ggml_cuda_nk_dense_tc_dense_count() {
    return g_nk_dense_tc_dense_fires.load(std::memory_order_relaxed);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static int nk_dense_tc_min_t() {
    static const int v = [] {
        const int x = nk_parse_nonneg_env("GGML_CUDA_NK_DENSE_TC_MINT", NK_HEAD_MINT_DEF);
        return x < 1 ? 1 : x > 8 ? 8 : x;
    }();
    return v;
}

static nk_dense_tc_head_path nk_dense_tc_head_path_env() {
    static const nk_dense_tc_head_path p = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC_HEAD_PATH");
        if (e == nullptr) {
            return NK_HEAD_PATH_AUTO;
        }
        if (strcmp(e, "tc") == 0) {
            return NK_HEAD_PATH_TC;
        }
        if (strcmp(e, "grid") == 0) {
            return NK_HEAD_PATH_GRID;
        }
        return NK_HEAD_PATH_AUTO;
    }();
    return p;
}

static const char * nk_head_path_name(const nk_dense_tc_head_path p) {
    switch (p) {
        case NK_HEAD_PATH_TC:   return "tc";
        case NK_HEAD_PATH_GRID: return "grid";
        default:                return "auto";
    }
}

// Per-T auto selection table for head-sized tensors (M >= 16385).
// T=1..3 stay on base q8a16 via default MINT=4; T=4..8 pick grid (0-SMEM DP4A).
static constexpr nk_dense_tc_head_path k_head_auto_table[9] = {
    NK_HEAD_PATH_GRID, // T=0 (unused)
    NK_HEAD_PATH_GRID, // T=1 (base by default MINT=4)
    NK_HEAD_PATH_GRID, // T=2 (base by default MINT=4)
    NK_HEAD_PATH_GRID, // T=3 (base by default MINT=4)
    NK_HEAD_PATH_GRID, // T=4
    NK_HEAD_PATH_GRID, // T=5 (headline)
    NK_HEAD_PATH_GRID, // T=6
    NK_HEAD_PATH_GRID, // T=7
    NK_HEAD_PATH_GRID, // T=8
};

static bool nk_dense_tc_l2_evict_first_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC_L2");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool nk_dense_tc_trace() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return on;
}

struct nk_dense_tc_args {
    const char  * vx[NK_DTC_MAX_GROUP];
    const float * x[NK_DTC_MAX_GROUP];
    float       * dst[NK_DTC_MAX_GROUP];
    int64_t       srow[NK_DTC_MAX_GROUP];
    int64_t       sx[NK_DTC_MAX_GROUP];
    int64_t       sd[NK_DTC_MAX_GROUP];
    uint32_t      blk_end[NK_DTC_MAX_GROUP];
    int           nrows[NK_DTC_MAX_GROUP];
    int           n;
};

static __device__ __forceinline__ void nk_dtc_cp16(uint32_t smem_addr, const void * gmem) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
#else
    GGML_UNUSED_VARS(smem_addr, gmem);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ uint64_t nk_head_l2_evict_first() {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    uint64_t p = 0;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
#else
    return 0;
#endif
}

template <bool L2_EF>
static __device__ __forceinline__ void nk_head_cp16(uint32_t smem_addr, const void * gmem, uint64_t l2_pol) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    if constexpr (L2_EF) {
        asm volatile("cp.async.cg.shared.global.L2::128B.L2::cache_hint [%0], [%1], 16, %2;\n"
            :: "r"(smem_addr), "l"(gmem), "l"(l2_pol));
    } else {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"(gmem));
    }
#else
    GGML_UNUSED_VARS(smem_addr, gmem, l2_pol);
    NO_DEVICE_CODE;
#endif
}

// Non-volatile read-only streaming 64-bit load so ptxas can hoist and overlap across rows.
// Contiguous 32-lane uint2 loads read 256 B (8 sectors) + 2-lane tail 16 B (1 sector) = 272 B
// with zero duplicate L2->SM sector fetches and zero L1 pollution.
template <bool L2_EF>
static __device__ __forceinline__ uint2 nk_ldg_na_u64(const uint2 * p, uint64_t l2_pol) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    uint2 v;
    if constexpr (L2_EF) {
        asm("ld.global.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;\n"
            : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(l2_pol));
    } else {
        asm("ld.global.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n"
            : "=r"(v.x), "=r"(v.y) : "l"(p));
    }
    return v;
#else
    GGML_UNUSED_VARS(l2_pol);
    return *p;
#endif
}

static __device__ __forceinline__ void nk_dtc_mma_i8(
        int (&d)[4],
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// Round 4 floats scaled into [-127, 127] to signed int8 via 1.5*2^23 magic bias and pack.
static __device__ __forceinline__ uint32_t nk_dtc_pack4_i8(const float4 v, const float inv_dx) {
    const uint32_t i0  = __float_as_uint(fmaf(v.x, inv_dx, 12582912.0f));
    const uint32_t i1  = __float_as_uint(fmaf(v.y, inv_dx, 12582912.0f));
    const uint32_t i2  = __float_as_uint(fmaf(v.z, inv_dx, 12582912.0f));
    const uint32_t i3  = __float_as_uint(fmaf(v.w, inv_dx, 12582912.0f));
    const uint32_t p01 = __byte_perm(i0, i1, 0x0040);
    const uint32_t p23 = __byte_perm(i2, i3, 0x0040);
    return __byte_perm(p01, p23, 0x5410);
}

static __device__ __forceinline__ void nk_dtc_quant_x_blk(
        const float4 v0, const float4 v1, const int tig,
        uint32_t & b0, uint32_t & b1, float & dx0, float & dx1) {
    float m = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const float dx     = m * (1.0f / 127.0f);
    const float inv_dx = (m > 1e-30f) ? __fdividef(127.0f, m) : 0.0f;
    b0  = nk_dtc_pack4_i8(v0, inv_dx);
    b1  = nk_dtc_pack4_i8(v1, inv_dx);
    dx0 = __shfl_sync(0xffffffffu, dx, 8 * tig);
    dx1 = __shfl_sync(0xffffffffu, dx, 8 * tig + 4);
}

static __device__ __forceinline__ void nk_dtc_quant_x_pair(
        const float * xp, const bool colok, const int tig,
        uint32_t & b0_0, uint32_t & b1_0, float & dx0_0, float & dx1_0,
        uint32_t & b0_1, uint32_t & b1_1, float & dx0_1, float & dx1_1) {
    const float4 z   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 v00 = colok ? __ldg((const float4 *) (xp +  0)) : z;
    const float4 v01 = colok ? __ldg((const float4 *) (xp + 16)) : z;
    const float4 v10 = colok ? __ldg((const float4 *) (xp + 32)) : z;
    const float4 v11 = colok ? __ldg((const float4 *) (xp + 48)) : z;
    nk_dtc_quant_x_blk(v00, v01, tig, b0_0, b1_0, dx0_0, dx1_0);
    nk_dtc_quant_x_blk(v10, v11, tig, b0_1, b1_1, dx0_1, dx1_1);
}

// Step 1 of improved head grid path: quantize [K, T] f32 activation ONCE into L1-resident
// per-32 int8 + fp32-scale layout: qx_q[nb*T*4] (int2 per 8 elems) and qx_d[nb*T] (float).
// 4 lanes per (ib, t) block -> 32 blocks per 128-thread CTA (~2-3 us for K=2304, T=5).
static __global__ void __launch_bounds__(128) k_nk_quant_x_dp4a(
        const float * __restrict__ x,
        int2        * __restrict__ qx_q,
        float       * __restrict__ qx_d,
        const int64_t sx,
        const int     nb,
        const int     T) {
    const int idx   = (int) blockIdx.x * 32 + ((int) threadIdx.x >> 2);
    const int iq    = (int) threadIdx.x & 3;
    const int total = nb * T;
    const bool valid = idx < total;

    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v0 = z;
    float4 v1 = z;
    if (valid) {
        const int ib = idx / T;
        const int t  = idx - ib * T;
        const float * xp = x + (int64_t) t * sx + (int64_t) ib * QK8_0 + iq * 8;
        v0 = __ldg((const float4 *) (xp + 0));
        v1 = __ldg((const float4 *) (xp + 4));
    }

    // All 32 lanes in the warp stay converged through the 4-lane amax butterfly.
    float m = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));

    const float dx     = __fmul_rn(m, 1.0f / 127.0f);
    const float inv_dx = (m > 1e-30f) ? __fdiv_rn(127.0f, m) : 0.0f;
    const uint32_t q0  = nk_dtc_pack4_i8(v0, inv_dx);
    const uint32_t q1  = nk_dtc_pack4_i8(v1, inv_dx);

    if (valid) {
        qx_q[idx * 4 + iq] = make_int2((int) q0, (int) q1);
        if (iq == 0) {
            qx_d[idx] = dx;
        }
    }
}

// Step 2 of improved head grid path: 0-SMEM, 0-barrier, high-occupancy streaming DP4A kernel.
// Each warp owns RW rows; 32 lanes read 8 consecutive block_q8_0 (272 contiguous bytes = 34 uint2)
// once per iteration via coalesced LDG.E.NA.64 (256 B + 16 B tail) + L2::evict_first, leaving the
// full 128 KB L1 dedicated to the 12-21 KB pre-quantized activation buffer (qx_q, qx_d).
template <int T, int RW, int NW, int MINB, bool L2_EF>
static __global__ void __launch_bounds__(32 * NW, MINB) k_nk_head_grid_dp4a(
        const char  * __restrict__ vx,
        const int2  * __restrict__ qx_q,
        const float * __restrict__ qx_d,
        float       * __restrict__ dst,
        const int64_t srow,
        const int64_t sd,
        const int     nr,
        const int     nb) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    static_assert(T >= 1 && T <= 8 && RW * T <= 32, "head_grid_dp4a shape");
    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int row0 = ((int) blockIdx.x * NW + warp) * RW;
    if (row0 >= nr) {
        return;
    }

    const int pair_in_iter = lane >> 3;
    const int g            = lane & 7;
    const int is_odd       = g >> 2;
    const int ib_lane      = lane >> 2;
    const int iq           = lane & 3;
    const int pair_base_w  = pair_in_iter * 17 + (is_odd ? (9 + 2 * iq) : (2 * iq));
    const int s_lo         = (lane < 30) ? (pair_base_w >> 1) : 0;
    const int s_hi         = (s_lo + 1) & 31;
    const int parity       = pair_base_w & 1;
    const int d_src_lane   = (lane & ~7) + (is_odd ? 3 : 0);

    const char * wp0 = vx + (int64_t) row0 * srow;
    int64_t row_off[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        row_off[r] = (row0 + r < nr) ? (int64_t) r * srow : 0;
    }

    const int2  * pq = qx_q + (ib_lane * T) * 4 + iq;
    const float * pd = qx_d + (ib_lane * T);

    const uint64_t l2_pol = L2_EF ? nk_head_l2_evict_first() : 0ull;

    float acc[RW][T];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (int ib0 = 0; ib0 < nb; ib0 += 8) {
        int2  xq[T];
        float xd[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xq[t] = __ldg(pq + t * 4);
            xd[t] = __ldg(pd + t);
        }
        pq += 8 * T * 4;
        pd += 8 * T;

        uint2 v0[RW], v1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint2 * rp = (const uint2 *) (wp0 + row_off[r]);
            v0[r] = nk_ldg_na_u64<L2_EF>(rp + lane, l2_pol);
            v1[r] = (lane >= 30) ? nk_ldg_na_u64<L2_EF>(rp + 2 + lane, l2_pol) : make_uint2(0u, 0u);
        }
        wp0 += 272;

#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint32_t a      = __shfl_sync(0xffffffffu, v0[r].x, s_lo);
            const uint32_t b      = __shfl_sync(0xffffffffu, v0[r].y, s_lo);
            const uint32_t c      = __shfl_sync(0xffffffffu, v0[r].x, s_hi);
            const uint32_t d      = __shfl_sync(0xffffffffu, v0[r].y, s_hi);
            const uint32_t u0     = (lane >= 30) ? v1[r].x : (parity ? b : a);
            const uint32_t u1     = (lane >= 30) ? v1[r].y : (parity ? c : b);
            const uint32_t next_w = parity ? d : c;
            const uint32_t q0     = is_odd ? u0 : __byte_perm(u0, u1, 0x5432);
            const uint32_t q1     = is_odd ? u1 : __byte_perm(u1, next_w, 0x5432);
            const uint32_t d_carry = (g == 3) ? next_w : u0;
            const uint32_t d_word  = __shfl_sync(0xffffffffu, d_carry, d_src_lane);
            const float dw = __half2float(__ushort_as_half(
                (uint16_t) (is_odd ? (d_word >> 16) : (d_word & 0xFFFFu))));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                int dot = __dp4a((int) q0, xq[t].x, 0);
                dot     = __dp4a((int) q1, xq[t].y, dot);
                acc[r][t] = fmaf(__int2float_rn(dot), dw * xd[t], acc[r][t]);
            }
        }
    }

    float out = 0.0f;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, off);
            }
            if (lane == r * T + t) {
                out = s;
            }
        }
    }
    if (lane < RW * T) {
        const int r = lane / T;
        const int t = lane - r * T;
        if (row0 + r < nr) {
            dst[(int64_t) t * sd + row0 + r] = out;
        }
    }
#else
    GGML_UNUSED_VARS(vx, qx_q, qx_d, dst, srow, sd, nr, nb);
    NO_DEVICE_CODE;
#endif
}

// Persistent 1-CTA/SM head kernel (r12 tc path): NW=8 independent warps, 2-stage cross-tile cp.async ring,
// activation quantized once per CTA into SMEM (exact MMA lane layout), zero loop/epilogue barriers.
template <bool L2_EF>
static __global__ void __launch_bounds__(32 * NK_HEAD_NW, 1) k_nk_head_tc(
        const char  * __restrict__ vx,
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int64_t srow,
        const int64_t sx,
        const int64_t sd,
        const int     nr,
        const int     nch,
        const int     T) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    extern __shared__ __align__(16) uint32_t smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    const int nb   = nch * 8;

    uint32_t * sw    = smem;
    uint2    * s_xb  = (uint2 *)  (sw + NK_HEAD_W_WORDS);
    float2   * s_xdx = (float2 *) (s_xb + nb * 32);

    const bool colok = gid < T;
    const float * x_col = x + (int64_t) (colok ? gid : 0) * sx + 4 * tig;
    for (int b = w; b < nb; b += NK_HEAD_NW) {
        const float * xp = x_col + (int64_t) b * QK8_0;
        const float4 z   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 v0  = colok ? __ldg((const float4 *) (xp +  0)) : z;
        const float4 v1  = colok ? __ldg((const float4 *) (xp + 16)) : z;
        uint32_t b0, b1;
        float dx0, dx1;
        {
            float m = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            const float dx     = __fmul_rn(m, 1.0f / 127.0f);
            const float inv_dx = (m > 1e-30f) ? __fdiv_rn(127.0f, m) : 0.0f;
            b0  = nk_dtc_pack4_i8(v0, inv_dx);
            b1  = nk_dtc_pack4_i8(v1, inv_dx);
            dx0 = __shfl_sync(0xffffffffu, dx, 8 * tig);
            dx1 = __shfl_sync(0xffffffffu, dx, 8 * tig + 4);
        }
        s_xb[b * 32 + lane] = make_uint2(b0, b1);
        if (gid == 0) {
            s_xdx[b * 4 + tig] = make_float2(dx0, dx1);
        }
    }
    __syncthreads();

    const int ntiles     = (nr + 15) / 16;
    const int nfull      = nr / 16;
    const int base_tiles = ntiles / (int) gridDim.x;
    const int rem_tiles  = ntiles % (int) gridDim.x;
    const int sm_start   = (int) blockIdx.x * base_tiles + min((int) blockIdx.x, rem_tiles);
    const int sm_end     = sm_start + base_tiles + ((int) blockIdx.x < rem_tiles ? 1 : 0);

    int tile = sm_start + w;
    if (tile >= sm_end) {
        return;
    }

    uint32_t * S_warp = sw + w * (2 * NK_DTC_STAGE_WORDS);
    int32_t  g_off[9];
    uint32_t sa_base[9];
#pragma unroll
    for (int i0 = 0; i0 < 8; ++i0) {
        const int i = i0 * 32 + lane;
        const int r = i / 17;
        const int c = i - r * 17;
        g_off[i0]   = (int32_t) ((int64_t) r * srow + c * 16);
        sa_base[i0] = (uint32_t) __cvta_generic_to_shared(&S_warp[r * NK_DTC_ROW_WORDS + c * 4]);
    }
    g_off[8]   = (lane < 16) ? (int32_t) ((int64_t) 15 * srow + (lane + 1) * 16) : 0;
    sa_base[8] = (lane < 16) ? (uint32_t) __cvta_generic_to_shared(&S_warp[15 * NK_DTC_ROW_WORDS + (lane + 1) * 4]) : 0u;

    const uint64_t l2_pol = L2_EF ? nk_head_l2_evict_first() : 0ull;

    auto issue_tile_chunk = [&](const int t_idx, const int c_idx, const uint32_t stage_bytes) {
        const char * cb = vx + (int64_t) t_idx * 16 * srow + (int64_t) c_idx * 272;
        if (t_idx < nfull) {
#pragma unroll
            for (int i0 = 0; i0 < 8; ++i0) {
                nk_head_cp16<L2_EF>(sa_base[i0] + stage_bytes, cb + g_off[i0], l2_pol);
            }
            if (lane < 16) {
                nk_head_cp16<L2_EF>(sa_base[8] + stage_bytes, cb + g_off[8], l2_pol);
            }
        } else {
            const int rmax = nr - 1 - t_idx * 16;
#pragma unroll
            for (int i0 = 0; i0 < 8; ++i0) {
                const int i = i0 * 32 + lane;
                const int r = i / 17;
                const int c = i - r * 17;
                nk_head_cp16<L2_EF>(sa_base[i0] + stage_bytes, cb + (int64_t) min(r, rmax) * srow + c * 16, l2_pol);
            }
            if (lane < 16) {
                nk_head_cp16<L2_EF>(sa_base[8] + stage_bytes, cb + (int64_t) min(15, rmax) * srow + (lane + 1) * 16, l2_pol);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    int ch = 0;
    uint32_t stage_b = 0u;
    uint32_t stage_w = 0u;
    issue_tile_chunk(tile, 0, 0u);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const uint32_t * S_base0 = S_warp + gid * NK_DTC_ROW_WORDS;
    const int c0_out = 2 * tig;
    const int c1_out = c0_out + 1;

    for (;;) {
        const bool last_ch = (ch + 1 == nch);
        const int nxt_tile = last_ch ? (tile + NK_HEAD_NW) : tile;
        const int nxt_ch   = last_ch ? 0 : (ch + 1);
        const bool has_nxt = (nxt_tile < sm_end);

        if (has_nxt) {
            issue_tile_chunk(nxt_tile, nxt_ch, stage_b ^ NK_DTC_STAGE_BYTES);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncwarp();

        const uint32_t * S0 = S_base0 + stage_w;
        const uint32_t * S1 = S0 + 8 * NK_DTC_ROW_WORDS;
        const int blk0 = ch * 8;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t * W0 = S0 + 17 * j;
            const uint32_t * W1 = S1 + 17 * j;

            const uint32_t u0_a = W0[tig],     u0_b = W0[tig + 1];
            const uint32_t u0_c = W0[tig + 4], u0_d = W0[tig + 5];
            const uint32_t u1_a = W1[tig],     u1_b = W1[tig + 1];
            const uint32_t u1_c = W1[tig + 4], u1_d = W1[tig + 5];

            const uint32_t a0_e0 = __byte_perm(u0_a, u0_b, 0x5432);
            const uint32_t a2_e0 = __byte_perm(u0_c, u0_d, 0x5432);
            const uint32_t a1_e0 = __byte_perm(u1_a, u1_b, 0x5432);
            const uint32_t a3_e0 = __byte_perm(u1_c, u1_d, 0x5432);

            const uint32_t a0_e1 = W0[9 + tig],  a2_e1 = W0[13 + tig];
            const uint32_t a1_e1 = W1[9 + tig],  a3_e1 = W1[13 + tig];

            const uint32_t h0_e0 = __shfl_sync(0xffffffffu, u0_a, lane & ~3);
            const uint32_t h0_e1 = __shfl_sync(0xffffffffu, u0_d, (lane & ~3) + 3);
            const uint32_t h1_e0 = __shfl_sync(0xffffffffu, u1_a, lane & ~3);
            const uint32_t h1_e1 = __shfl_sync(0xffffffffu, u1_d, (lane & ~3) + 3);

            const float dw0_0 = __half2float(__ushort_as_half((uint16_t) (h0_e0 & 0xFFFFu)));
            const float dw1_0 = __half2float(__ushort_as_half((uint16_t) (h1_e0 & 0xFFFFu)));
            const float dw0_1 = __half2float(__ushort_as_half((uint16_t) (h0_e1 >> 16)));
            const float dw1_1 = __half2float(__ushort_as_half((uint16_t) (h1_e1 >> 16)));

            const int b_even = blk0 + 2 * j;
            const int b_odd  = b_even + 1;
            const uint2  xb0  = s_xb[b_even * 32 + lane];
            const uint2  xb1  = s_xb[b_odd  * 32 + lane];
            const float2 xdx0 = s_xdx[b_even * 4 + tig];
            const float2 xdx1 = s_xdx[b_odd  * 4 + tig];

            int d0[4] = { 0, 0, 0, 0 };
            nk_dtc_mma_i8(d0, a0_e0, a1_e0, a2_e0, a3_e0, xb0.x, xb0.y);

            int d1[4] = { 0, 0, 0, 0 };
            nk_dtc_mma_i8(d1, a0_e1, a1_e1, a2_e1, a3_e1, xb1.x, xb1.y);

            acc[0] = fmaf(__int2float_rn(d0[0]), dw0_0 * xdx0.x, acc[0]);
            acc[1] = fmaf(__int2float_rn(d0[1]), dw0_0 * xdx0.y, acc[1]);
            acc[2] = fmaf(__int2float_rn(d0[2]), dw1_0 * xdx0.x, acc[2]);
            acc[3] = fmaf(__int2float_rn(d0[3]), dw1_0 * xdx0.y, acc[3]);

            acc[0] = fmaf(__int2float_rn(d1[0]), dw0_1 * xdx1.x, acc[0]);
            acc[1] = fmaf(__int2float_rn(d1[1]), dw0_1 * xdx1.y, acc[1]);
            acc[2] = fmaf(__int2float_rn(d1[2]), dw1_1 * xdx1.x, acc[2]);
            acc[3] = fmaf(__int2float_rn(d1[3]), dw1_1 * xdx1.y, acc[3]);
        }

        __syncwarp();

        if (last_ch) {
            const int r0_out = tile * 16 + gid;
            const int r1_out = r0_out + 8;
            if (r0_out < nr) {
                if (c0_out < T) { dst[(int64_t) c0_out * sd + r0_out] = acc[0]; }
                if (c1_out < T) { dst[(int64_t) c1_out * sd + r0_out] = acc[1]; }
            }
            if (r1_out < nr) {
                if (c0_out < T) { dst[(int64_t) c0_out * sd + r1_out] = acc[2]; }
                if (c1_out < T) { dst[(int64_t) c1_out * sd + r1_out] = acc[3]; }
            }
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
        }

        if (!has_nxt) {
            break;
        }
        tile = nxt_tile;
        ch   = nxt_ch;
        stage_b ^= NK_DTC_STAGE_BYTES;
        stage_w ^= NK_DTC_STAGE_WORDS;
    }
#else
    GGML_UNUSED_VARS(vx, x, dst, srow, sx, sd, nr, nch, T);
    NO_DEVICE_CODE;
#endif
}

// Fallback / MIN_ROWS=0 kernel (r11 4-warp cooperative 3-stage ring for grouped or attention).
static __global__ void __launch_bounds__(128, 6) k_nk_dense_tc(
        const nk_dense_tc_args a, const int nch, const int T) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    __shared__ __align__(16) uint32_t sw[NK_DTC_STAGES * NK_DTC_STAGE_WORDS + NK_DTC_RED_FLOATS];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    int k = 0;
    while (k < a.n - 1 && blockIdx.x >= a.blk_end[k]) {
        ++k;
    }
    const int tile  = (int) (blockIdx.x - (k == 0 ? 0u : a.blk_end[k - 1]));
    const int nr    = a.nrows[k];
    const int64_t srow = a.srow[k];
    const char * vx = a.vx[k] + (int64_t) tile * 16 * srow;
    const int rmax  = nr - 1 - tile * 16;

    const int r0 = tid / 17;
    const int c0 = tid - r0 * 17;
    const int i1 = tid + 128;
    const int r1 = i1 / 17;
    const int c1 = i1 - r1 * 17;

    const char * g0 = vx + (int64_t) min(r0, rmax) * srow + c0 * 16;
    const char * g1 = vx + (int64_t) min(r1, rmax) * srow + c1 * 16;
    const char * g2 = (tid < 16) ? (vx + (int64_t) min(15, rmax) * srow + (tid + 1) * 16) : nullptr;

    const uint32_t sa0_base = (uint32_t) __cvta_generic_to_shared(&sw[r0 * NK_DTC_ROW_WORDS + c0 * 4]);
    const uint32_t sa1_base = (uint32_t) __cvta_generic_to_shared(&sw[r1 * NK_DTC_ROW_WORDS + c1 * 4]);
    const uint32_t sa2_base = (tid < 16)
        ? (uint32_t) __cvta_generic_to_shared(&sw[15 * NK_DTC_ROW_WORDS + (tid + 1) * 4])
        : 0u;

    auto issue_chunk = [&](const uint32_t stage_bytes) {
        nk_dtc_cp16(sa0_base + stage_bytes, g0); g0 += 272;
        nk_dtc_cp16(sa1_base + stage_bytes, g1); g1 += 272;
        if (tid < 16) {
            nk_dtc_cp16(sa2_base + stage_bytes, g2); g2 += 272;
        }
        asm volatile("cp.async.commit_group;\n");
    };

    issue_chunk(0u);
    if (nch > 1) {
        issue_chunk(NK_DTC_STAGE_BYTES);
    }

    const bool colok = gid < T;
    const float * x_ptr = a.x[k] + (int64_t) (colok ? gid : 0) * a.sx[k] + (2 * w) * QK8_0 + 4 * tig;

    uint32_t xb0_0, xb1_0, xb0_1, xb1_1;
    float    dx0_0, dx1_0, dx0_1, dx1_1;
    nk_dtc_quant_x_pair(x_ptr, colok, tig, xb0_0, xb1_0, dx0_0, dx1_0, xb0_1, xb1_1, dx0_1, dx1_1);
    x_ptr += 8 * QK8_0;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t cur_w = 0u;
    uint32_t nxt_b = 2u * NK_DTC_STAGE_BYTES;

    for (int ch = 0; ch < nch; ++ch) {
        if (ch + 1 < nch) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        if (ch + 2 < nch) {
            issue_chunk(nxt_b);
        }

        const uint32_t * W0 = sw + cur_w + gid * NK_DTC_ROW_WORDS + 17 * w;
        const uint32_t * W1 = W0 + 8 * NK_DTC_ROW_WORDS;

        const uint32_t u0_a = W0[tig],     u0_b = W0[tig + 1];
        const uint32_t u0_c = W0[tig + 4], u0_d = W0[tig + 5];
        const uint32_t u1_a = W1[tig],     u1_b = W1[tig + 1];
        const uint32_t u1_c = W1[tig + 4], u1_d = W1[tig + 5];

        const uint32_t a0_e0 = __byte_perm(u0_a, u0_b, 0x5432);
        const uint32_t a2_e0 = __byte_perm(u0_c, u0_d, 0x5432);
        const uint32_t a1_e0 = __byte_perm(u1_a, u1_b, 0x5432);
        const uint32_t a3_e0 = __byte_perm(u1_c, u1_d, 0x5432);

        const uint32_t a0_e1 = W0[9 + tig],  a2_e1 = W0[13 + tig];
        const uint32_t a1_e1 = W1[9 + tig],  a3_e1 = W1[13 + tig];

        const uint32_t h0_e0 = __shfl_sync(0xffffffffu, u0_a, lane & ~3);
        const uint32_t h0_e1 = __shfl_sync(0xffffffffu, u0_d, (lane & ~3) + 3);
        const uint32_t h1_e0 = __shfl_sync(0xffffffffu, u1_a, lane & ~3);
        const uint32_t h1_e1 = __shfl_sync(0xffffffffu, u1_d, (lane & ~3) + 3);

        const float dw0_0 = __half2float(__ushort_as_half((uint16_t) (h0_e0 & 0xFFFFu)));
        const float dw1_0 = __half2float(__ushort_as_half((uint16_t) (h1_e0 & 0xFFFFu)));
        const float dw0_1 = __half2float(__ushort_as_half((uint16_t) (h0_e1 >> 16)));
        const float dw1_1 = __half2float(__ushort_as_half((uint16_t) (h1_e1 >> 16)));

        int d0[4] = { 0, 0, 0, 0 };
        nk_dtc_mma_i8(d0, a0_e0, a1_e0, a2_e0, a3_e0, xb0_0, xb1_0);

        int d1[4] = { 0, 0, 0, 0 };
        nk_dtc_mma_i8(d1, a0_e1, a1_e1, a2_e1, a3_e1, xb0_1, xb1_1);

        acc[0] = fmaf(__int2float_rn(d0[0]), dw0_0 * dx0_0, acc[0]);
        acc[1] = fmaf(__int2float_rn(d0[1]), dw0_0 * dx1_0, acc[1]);
        acc[2] = fmaf(__int2float_rn(d0[2]), dw1_0 * dx0_0, acc[2]);
        acc[3] = fmaf(__int2float_rn(d0[3]), dw1_0 * dx1_0, acc[3]);

        acc[0] = fmaf(__int2float_rn(d1[0]), dw0_1 * dx0_1, acc[0]);
        acc[1] = fmaf(__int2float_rn(d1[1]), dw0_1 * dx1_1, acc[1]);
        acc[2] = fmaf(__int2float_rn(d1[2]), dw1_1 * dx0_1, acc[2]);
        acc[3] = fmaf(__int2float_rn(d1[3]), dw1_1 * dx1_1, acc[3]);

        if (ch + 1 < nch) {
            nk_dtc_quant_x_pair(x_ptr, colok, tig, xb0_0, xb1_0, dx0_0, dx1_0, xb0_1, xb1_1, dx0_1, dx1_1);
            x_ptr += 8 * QK8_0;
        }

        cur_w = (cur_w == 2u * NK_DTC_STAGE_WORDS) ? 0u : (cur_w + NK_DTC_STAGE_WORDS);
        nxt_b = (nxt_b == 2u * NK_DTC_STAGE_BYTES) ? 0u : (nxt_b + NK_DTC_STAGE_BYTES);
    }

    float4 * red = (float4 *) &sw[NK_DTC_STAGES * NK_DTC_STAGE_WORDS];
    if (w > 0) {
        red[(w - 1) * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    __syncthreads();
    if (w > 0) {
        return;
    }
#pragma unroll
    for (int v = 0; v < 3; ++v) {
        const float4 p = red[v * 32 + lane];
        acc[0] += p.x;
        acc[1] += p.y;
        acc[2] += p.z;
        acc[3] += p.w;
    }

    float * dst = a.dst[k];
    const int64_t sd = a.sd[k];
    const int r0_out = tile * 16 + gid;
    const int r1_out = r0_out + 8;
    const int c0_out = 2 * tig;
    const int c1_out = c0_out + 1;
    if (r0_out < nr) {
        if (c0_out < T) { dst[(int64_t) c0_out * sd + r0_out] = acc[0]; }
        if (c1_out < T) { dst[(int64_t) c1_out * sd + r0_out] = acc[1]; }
    }
    if (r1_out < nr) {
        if (c0_out < T) { dst[(int64_t) c0_out * sd + r1_out] = acc[2]; }
        if (c1_out < T) { dst[(int64_t) c1_out * sd + r1_out] = acc[3]; }
    }
#else
    GGML_UNUSED_VARS(a, nch, T);
    NO_DEVICE_CODE;
#endif
}

static bool nk_dense_tc_device_ok(const ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) &&
        ggml_cuda_info().devices[ctx.device].warp_size == 32;
}

static bool nk_dense_tc_member_ok(
        const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * d,
        const int64_t ne10, const int64_t T, const int min_rows, const int max_rows) {
    if (!w || !x || !d || w->type != GGML_TYPE_Q8_0 || !w->buffer ||
            ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || w->ne[0] != ne10 ||
            w->ne[1] <= 0 || w->ne[1] > INT_MAX || w->ne[1] < min_rows ||
            (max_rows > 0 && w->ne[1] > max_rows) ||
            w->ne[2] != 1 || w->ne[3] != 1 ||
            w->nb[1] != (size_t) (ne10 / QK8_0) * sizeof(block_q8_0) ||
            (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || x->ne[0] != ne10 || x->ne[1] != T ||
            x->ne[2] != 1 || x->ne[3] != 1 ||
            x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || (uintptr_t) x->data % 16 != 0) {
        return false;
    }
    if (d->type != GGML_TYPE_F32 || d->ne[0] != w->ne[1] || d->ne[1] != T ||
            d->ne[2] != 1 || d->ne[3] != 1 ||
            d->nb[0] != sizeof(float) || d->nb[1] % sizeof(float) != 0) {
        return false;
    }
    return true;
}

template <bool L2_EF>
static bool nk_head_tc_ensure_smem(const int dev, const size_t smem_bytes, cudaStream_t stream) {
    static std::atomic<uint32_t> done_mask{0};
    if (dev < 0 || dev >= 32) {
        return false;
    }
    const size_t smpbo = ggml_cuda_info().devices[dev].smpbo;
    if (smem_bytes > smpbo) {
        return false;
    }
    const uint32_t bit = 1u << (unsigned) dev;
    if ((done_mask.load(std::memory_order_acquire) & bit) != 0u) {
        return true;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
        return false;
    }
    const int req = (int) std::min<size_t>(smpbo, NK_HEAD_MAX_SMEM);
    if (cudaFuncSetAttribute(k_nk_head_tc<L2_EF>, cudaFuncAttributeMaxDynamicSharedMemorySize, req) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaFuncSetAttribute(k_nk_head_tc<L2_EF>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared) != cudaSuccess) {
        (void) cudaGetLastError();
    }
    done_mask.fetch_or(bit, std::memory_order_release);
    return true;
}

static void nk_dense_tc_maybe_set_carveout(const int dev, cudaStream_t stream) {
    static std::atomic<uint32_t> done_mask{0};
    if (dev < 0 || dev >= 32) {
        return;
    }
    const uint32_t bit = 1u << (unsigned) dev;
    if ((done_mask.load(std::memory_order_acquire) & bit) != 0u) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
        return;
    }
    if (cudaFuncSetAttribute(k_nk_dense_tc, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared) == cudaSuccess) {
        done_mask.fetch_or(bit, std::memory_order_release);
    } else {
        (void) cudaGetLastError();
    }
}

static void nk_dense_tc_log_active() {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        GGML_LOG_WARN("[NK_DENSE_TC] active (head_path=%s, l2_ef=%d, min_rows=%d, min_t=%d, max_rows=%d)\n",
            nk_head_path_name(nk_dense_tc_head_path_env()),
            (int) nk_dense_tc_l2_evict_first_enabled(),
            ggml_cuda_nk_dense_tc_min_rows(), nk_dense_tc_min_t(), nk_dense_tc_max_rows());
    }
}

template <int T, int RW, int NW, int MINB, bool L2_EF>
static void nk_head_grid_launch_t(
        const char * vx, const int2 * qx_q, const float * qx_d, float * dst,
        const int64_t srow, const int64_t sd, const int nr, const int nb, cudaStream_t stream) {
    const uint32_t grid = (uint32_t) ((nr + RW * NW - 1) / (RW * NW));
    k_nk_head_grid_dp4a<T, RW, NW, MINB, L2_EF><<<grid, 32 * NW, 0, stream>>>(
        vx, qx_q, qx_d, dst, srow, sd, nr, nb);
}

template <bool L2_EF>
static void nk_head_grid_dispatch(
        const char * vx, const int2 * qx_q, const float * qx_d, float * dst,
        const int64_t srow, const int64_t sd, const int nr, const int nb, const int T, cudaStream_t stream) {
    switch (T) {
        case 1: nk_head_grid_launch_t<1, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 2: nk_head_grid_launch_t<2, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 3: nk_head_grid_launch_t<3, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 4: nk_head_grid_launch_t<4, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 5: nk_head_grid_launch_t<5, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 6: nk_head_grid_launch_t<6, 3, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 7: nk_head_grid_launch_t<7, 2, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 8: nk_head_grid_launch_t<8, 2, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        default: GGML_ABORT("nk_head_grid_dp4a: invalid T");
    }
}

static bool nk_head_grid_try_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst,
        const int           nb,
        const int           T) {
    const int nr = (int) src0->ne[1];
    if (nr < NK_HEAD_MIN_ROWS_DEF || nb < 4 || nb % 4 != 0 || nb > INT_MAX / 32 || T < 1 || T > 8) {
        return false;
    }

    const size_t q_bytes  = (size_t) nb * (size_t) T * 4u * sizeof(int2);
    const size_t d_bytes  = (size_t) nb * (size_t) T * sizeof(float);
    const size_t tot_bytes = (q_bytes + d_bytes + 15u) & ~size_t(15u);
    ggml_cuda_pool_alloc<char> qx_buf(ctx.pool(), tot_bytes);

    int2  * qx_q = (int2 *)  qx_buf.get();
    float * qx_d = (float *) (qx_buf.get() + q_bytes);

    nk_dense_tc_log_active();

    cudaStream_t stream = ctx.stream();
    const char  * vx   = (const char *)  src0->data;
    const float * x    = (const float *) src1->data;
    float       * out  = (float *)       dst->data;
    const int64_t srow = (int64_t) src0->nb[1];
    const int64_t sx   = (int64_t) (src1->nb[1] / sizeof(float));
    const int64_t sd   = (int64_t) (dst->nb[1] / sizeof(float));

    const uint32_t q_grid = (uint32_t) (((int64_t) nb * T + 31) / 32);
    k_nk_quant_x_dp4a<<<q_grid, 128, 0, stream>>>(x, qx_q, qx_d, sx, nb, T);

    const bool l2_ef = nk_dense_tc_l2_evict_first_enabled();
    if ((nb & 7) == 0) {
        if (l2_ef) {
            nk_head_grid_dispatch<true>(vx, qx_q, qx_d, out, srow, sd, nr, nb, T, stream);
        } else {
            nk_head_grid_dispatch<false>(vx, qx_q, qx_d, out, srow, sd, nr, nb, T, stream);
        }
    } else {
        ggml_cuda_nk_head_grid_tail4_launch(vx, qx_q, qx_d, out, srow, sd, nr, nb, T, l2_ef, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    g_nk_dense_tc_head_grid_fires.fetch_add(1, std::memory_order_relaxed);
    const int64_t fire = g_nk_dense_tc_fires.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dense_tc_trace() && fire <= 16) {
        GGML_LOG_WARN("[NK_DENSE_TC] trace #%lld (head_grid%s l2_ef=%d): rows=%d K=%d T=%d qx=%zu B\n",
            (long long) fire, ((nb & 7) == 0) ? "" : "_tail4", (int) l2_ef, nr, nb * QK8_0, T, tot_bytes);
    }
    return true;
}

static bool nk_head_tc_try_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst,
        const int           nch,
        const int           T) {
    if (nch < 1 || nch > NK_HEAD_MAX_NCH || src0->ne[1] < NK_HEAD_MIN_ROWS_DEF) {
        return false;
    }
    const size_t smem_bytes = (size_t) NK_HEAD_W_BYTES + (size_t) (nch * 8) * NK_HEAD_X_BLK_BYTES;
    cudaStream_t stream = ctx.stream();
    const bool l2_ef = nk_dense_tc_l2_evict_first_enabled();
    const bool ready = l2_ef
        ? nk_head_tc_ensure_smem<true>(ctx.device, smem_bytes, stream)
        : nk_head_tc_ensure_smem<false>(ctx.device, smem_bytes, stream);
    if (!ready) {
        return false;
    }

    const int nr     = (int) src0->ne[1];
    const int ntiles = (nr + 15) / 16;
    const int nsm    = ggml_cuda_info().devices[ctx.device].nsm;
    const int grid   = std::max(1, std::min(nsm, (ntiles + NK_HEAD_NW - 1) / NK_HEAD_NW));

    nk_dense_tc_log_active();

    const char  * vx   = (const char *)  src0->data;
    const float * x    = (const float *) src1->data;
    float       * out  = (float *)       dst->data;
    const int64_t srow = (int64_t) src0->nb[1];
    const int64_t sx   = (int64_t) (src1->nb[1] / sizeof(float));
    const int64_t sd   = (int64_t) (dst->nb[1] / sizeof(float));

    if (l2_ef) {
        k_nk_head_tc<true><<<(uint32_t) grid, 32 * NK_HEAD_NW, smem_bytes, stream>>>(
            vx, x, out, srow, sx, sd, nr, nch, T);
    } else {
        k_nk_head_tc<false><<<(uint32_t) grid, 32 * NK_HEAD_NW, smem_bytes, stream>>>(
            vx, x, out, srow, sx, sd, nr, nch, T);
    }
    CUDA_CHECK(cudaGetLastError());

    g_nk_dense_tc_head_tc_fires.fetch_add(1, std::memory_order_relaxed);
    const int64_t fire = g_nk_dense_tc_fires.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dense_tc_trace() && fire <= 16) {
        GGML_LOG_WARN("[NK_DENSE_TC] trace #%lld (head_tc l2_ef=%d): rows=%d K=%d T=%d grid=%d smem=%zu\n",
            (long long) fire, (int) l2_ef, nr, nch * 8 * QK8_0, T, grid, smem_bytes);
    }
    return true;
}

static bool nk_dense_tc_launch(
        ggml_backend_cuda_context & ctx, nk_dense_tc_args & a, const int nch, const int T) {
    uint64_t blk64 = 0;
    for (int k = 0; k < a.n; ++k) {
        blk64 += (uint64_t) ((a.nrows[k] + 15) / 16);
        if (blk64 > (uint64_t) INT_MAX) {
            return false;
        }
        a.blk_end[k] = (uint32_t) blk64;
    }
    if (blk64 == 0) {
        return false;
    }

    nk_dense_tc_log_active();

    cudaStream_t stream = ctx.stream();
    nk_dense_tc_maybe_set_carveout(ctx.device, stream);
    k_nk_dense_tc<<<(uint32_t) blk64, 128, 0, stream>>>(a, nch, T);
    CUDA_CHECK(cudaGetLastError());

    g_nk_dense_tc_dense_fires.fetch_add(1, std::memory_order_relaxed);
    const int64_t fire = g_nk_dense_tc_fires.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dense_tc_trace() && fire <= 16) {
        int total_rows = 0;
        for (int k = 0; k < a.n; ++k) {
            total_rows += a.nrows[k];
        }
        GGML_LOG_WARN("[NK_DENSE_TC] trace #%lld (dense_tc): n=%d rows=%d K=%d T=%d grid=%u\n",
            (long long) fire, a.n, total_rows, nch * 8 * QK8_0, T, (unsigned) blk64);
    }
    return true;
}

bool ggml_cuda_nk_dense_tc_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst,
        bool                force_enable,
        int                 path_override,
        bool                respect_mint) {
    if ((!force_enable && !ggml_cuda_nk_dense_tc_enabled()) || !nk_dense_tc_device_ok(ctx)) {
        return false;
    }
    if (!src0 || !src1 || !dst) {
        return false;
    }
    const int64_t ne10 = src0->ne[0];
    const int64_t T    = src1->ne[1];
    const int min_t    = respect_mint ? nk_dense_tc_min_t() : 1;
    const int min_rows = force_enable ? 1 : ggml_cuda_nk_dense_tc_min_rows();
    const int max_rows = force_enable ? 0 : nk_dense_tc_max_rows();
    if (T < min_t || T > 8 || ne10 < 4 * QK8_0 || ne10 % (4 * QK8_0) != 0 ||
            ne10 / QK8_0 > INT_MAX / 32) {
        return false;
    }
    if (!nk_dense_tc_member_ok(src0, src1, dst, ne10, T, min_rows, max_rows)) {
        return false;
    }

    const int  nb    = (int) (ne10 / QK8_0);
    const bool mult8 = (nb % 8 == 0);
    const int  nch   = nb / 8;

    if (src0->ne[1] >= NK_HEAD_MIN_ROWS_DEF) {
        const nk_dense_tc_head_path mode = (path_override >= 0)
            ? (nk_dense_tc_head_path) path_override
            : nk_dense_tc_head_path_env();
        if (mode == NK_HEAD_PATH_TC) {
            return mult8 && nk_head_tc_try_launch(ctx, src0, src1, dst, nch, (int) T);
        }
        if (mode == NK_HEAD_PATH_GRID) {
            return nk_head_grid_try_launch(ctx, src0, src1, dst, nb, (int) T);
        }
        const nk_dense_tc_head_path chosen = k_head_auto_table[T];
        if (chosen == NK_HEAD_PATH_GRID) {
            if (nk_head_grid_try_launch(ctx, src0, src1, dst, nb, (int) T)) {
                return true;
            }
            if (mult8 && nk_head_tc_try_launch(ctx, src0, src1, dst, nch, (int) T)) {
                return true;
            }
        } else {
            if (mult8 && nk_head_tc_try_launch(ctx, src0, src1, dst, nch, (int) T)) {
                return true;
            }
            if (nk_head_grid_try_launch(ctx, src0, src1, dst, nb, (int) T)) {
                return true;
            }
        }
    }

    if (!mult8) {
        return false;
    }

    nk_dense_tc_args a{};
    a.n        = 1;
    a.vx[0]    = (const char *) src0->data;
    a.x[0]     = (const float *) src1->data;
    a.dst[0]   = (float *) dst->data;
    a.srow[0]  = (int64_t) src0->nb[1];
    a.sx[0]    = (int64_t) (src1->nb[1] / sizeof(float));
    a.sd[0]    = (int64_t) (dst->nb[1] / sizeof(float));
    a.nrows[0] = (int) src0->ne[1];
    return nk_dense_tc_launch(ctx, a, nch, (int) T);
}

bool ggml_cuda_nk_dense_tc_group(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * const       * dsts,
        int                         n,
        bool                        check_only,
        bool                        force_enable) {
    if ((!force_enable && !ggml_cuda_nk_dense_tc_enabled()) || n < 1 || n > NK_DTC_MAX_GROUP ||
            !dsts || !dsts[0] || !dsts[0]->src[0] || !dsts[0]->src[1] ||
            !nk_dense_tc_device_ok(ctx)) {
        return false;
    }
    const int64_t ne10 = dsts[0]->src[0]->ne[0];
    const int64_t T    = dsts[0]->src[1]->ne[1];
    const int min_t    = force_enable ? 1 : nk_dense_tc_min_t();
    const int min_rows = force_enable ? 1 : ggml_cuda_nk_dense_tc_min_rows();
    const int max_rows = force_enable ? 0 : nk_dense_tc_max_rows();
    if (T < min_t || T > 8 || ne10 < 8 * QK8_0 || ne10 % (8 * QK8_0) != 0) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        if (!dsts[k] || !nk_dense_tc_member_ok(dsts[k]->src[0], dsts[k]->src[1], dsts[k], ne10, T, min_rows, max_rows)) {
            return false;
        }
    }
    if (check_only) {
        return true;
    }
    nk_dense_tc_args a{};
    a.n = n;
    for (int k = 0; k < n; ++k) {
        a.vx[k]    = (const char *) dsts[k]->src[0]->data;
        a.x[k]     = (const float *) dsts[k]->src[1]->data;
        a.dst[k]   = (float *) dsts[k]->data;
        a.srow[k]  = (int64_t) dsts[k]->src[0]->nb[1];
        a.sx[k]    = (int64_t) (dsts[k]->src[1]->nb[1] / sizeof(float));
        a.sd[k]    = (int64_t) (dsts[k]->nb[1] / sizeof(float));
        a.nrows[k] = (int) dsts[k]->src[0]->ne[1];
    }
    return nk_dense_tc_launch(ctx, a, (int) (ne10 / (8 * QK8_0)), (int) T);
}

#else

bool ggml_cuda_nk_dense_tc_mul_mat(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, bool, int, bool) {
    return false;
}

bool ggml_cuda_nk_dense_tc_group(
        ggml_backend_cuda_context &, ggml_tensor * const *, int, bool, bool) {
    return false;
}

#endif
