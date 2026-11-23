// SM 8.6 (GA102 / RTX 3090) AI+ Trellis Decode Instruction-Wall Breakthrough (ALL 6 bit-rates: 1.75/2/2.5/3/4/5-bit).
// Standalone TU for fast incremental builds and clean SASS auditing. ONLY wired, reachable kernels are instantiated.
// Kill switch: GGML_CUDA_AI4_DECODE_FAST=0 (= exact rc7 behavior); default ON since 0.7.4.

#include "ai4-decode-fast.cuh"
#include "../ggml-aif25.h"
#include "../ggml-aif175.h"

#include <cstdlib>

bool ggml_cuda_ai4_decode_fast_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AI4_DECODE_FAST");
        return e == nullptr || atoi(e) != 0; // default ON (0.7.4, INKA4PLUS kpa4 truth4 paired KLD gate); =0 restores rc7
    }();
    return on;
}

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_ai4_decode_fast_unit_try(
        ggml_backend_cuda_context &, ggml_type, const uint8_t *, int64_t, const ggml_tensor *,
        const int8_t *, const float *, float *, int, int, int, int, int, int, int, bool) {
    return false;
}

bool ggml_cuda_ai4_decode_fast_frac_try(
        ggml_backend_cuda_context &, ggml_type, int, const uint8_t *, const int8_t *,
        const float *, float *, int, int, int64_t, const uint8_t *) {
    return false;
}

#else

namespace ai4_fast {

using rq4::cp_async16;
using rq4::cp_async_commit;
using rq4::cp_async_wait;
using rq4::mma_s8;
using rq4::magic_i2f;

// =================================================================================================
// 1. Optimal Whole-Row 8-Weight Decoders (100% BIT-EXACT to ggml-ai4.h / ggml-aif25.h / ggml-aif175.h)
// =================================================================================================

template <uint32_t MULT>
static __device__ __forceinline__ uint32_t lv1_u(const uint32_t s) {
    return __dp4a(s * MULT, 0x40404040u, 128u - 32768u);
}

static __device__ __forceinline__ uint32_t pack4_b1(
        const uint32_t r0, const uint32_t r1, const uint32_t r2, const uint32_t r3) {
    return __byte_perm(__byte_perm(r0, r1, 0x0051), __byte_perm(r2, r3, 0x0051), 0x5410);
}

// AIT4_U whole-row decoder (8 weights from w = ta[R] and wp = ta[(R+15)&15]):
// 10 extract ops per 8 weights (1.25 ops/w vs 1.50 in levels4 and 2.38 in mmvr levels4_kr<4>!) + HALF the LDS reads!
template <uint32_t MULT>
static __device__ __forceinline__ void levels8_k4(const uint32_t w, const uint32_t wp, uint32_t & lo, uint32_t & hi) {
    const uint32_t f20 = __funnelshift_r(w, wp, 20);
    const uint32_t w4  = w >> 4;
    const uint32_t s0  = __byte_perm(f20, 0u, 0x4421);
    const uint32_t s1  = __byte_perm(w, wp, 0x0043) & 0xFFFFu;
    const uint32_t s2  = f20 & 0xFFFFu;
    const uint32_t s3  = w >> 16;
    const uint32_t s4  = __byte_perm(w4, 0u, 0x4421);
    const uint32_t s5  = __byte_perm(w,  0u, 0x4421);
    const uint32_t s6  = w4 & 0xFFFFu;
    const uint32_t s7  = w  & 0xFFFFu;
    lo = pack4_b1(lv1_u<MULT>(s0), lv1_u<MULT>(s1), lv1_u<MULT>(s2), lv1_u<MULT>(s3));
    hi = pack4_b1(lv1_u<MULT>(s4), lv1_u<MULT>(s5), lv1_u<MULT>(s6), lv1_u<MULT>(s7));
}

// Fractional pair whole-row decoders (4 states -> 8 weights, 7 extract ops total = 0.875 ops/weight!):
//   - AIF25 (B=5, V=2, 10 words = 320 bits): shifts 15, 10, 5, 0 of 32-bit window ending at 20*(R+1)
//   - AIF175 (B=7, V=4, 7 words = 224 bits): shifts 11, 7, 4, 0 of 32-bit window ending at 14*(R+1)
template <int B, int V>
static __device__ __forceinline__ ai4k::k5win frac_row_win(const int R) {
    constexpr int BITS  = 128 * B / V;
    constexpr int WORDS = 4 * B / V;
    int s = (B * 8 / V) * (R + 1) - 32;
    if (s < 0) s += BITS;
    ai4k::k5win w;
    w.w0 = s >> 5;
    w.sh = s & 31;
    w.w1 = (w.w0 + 1 == WORDS) ? 0 : (w.w0 + 1);
    return w;
}

template <int B, int V, uint32_t C>
static __device__ __forceinline__ void levels8_frac(const uint32_t f, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t C2 = (uint32_t) ((uint64_t) C * C);
    constexpr int sh0 = (B == 7 && V == 4) ? 11 : 15;
    constexpr int sh1 = (B == 7 && V == 4) ?  7 : 10;
    constexpr int sh2 = (B == 7 && V == 4) ?  4 :  5;
    const uint32_t s0 = (f >> sh0) & 0xFFFFu;
    const uint32_t s1 = (f >> sh1) & 0xFFFFu;
    const uint32_t s2 = (f >> sh2) & 0xFFFFu;
    const uint32_t s3 =  f         & 0xFFFFu;
    lo = pack4_b1(lv1_u<C>(s0), lv1_u<C2>(s0), lv1_u<C>(s1), lv1_u<C2>(s1));
    hi = pack4_b1(lv1_u<C>(s2), lv1_u<C2>(s2), lv1_u<C>(s3), lv1_u<C2>(s3));
}

// =================================================================================================
// 2. Unit-Layout High-Throughput Decode Kernel (AIT4_U, AIT5_U, AIT3_U, AIT2_U, AIT2P_U)
//    - Supports BOTH dense (ID == false, N = 1..8) AND MoE MUL_MAT_ID (ID == true, N = 1..8, 0 ids_helper!)
//    - Whole-row PERM decode for AIT4_U / AIT2_U / AIT2P_U (reads unmodified rc7 Xq directly!)
//    - Hoisted SMEM word base pointers (pa0, pa1, pb0, pb1) -> ZERO per-j IMAD indexing in inner loop!
//    - Compile-time stage unroll + zero-static-SMEM reduction + WARPS=3/4 -> ZERO spills!
// =================================================================================================

template <int LAY>
constexpr __host__ __device__ bool lay_use_perm() {
    return LAY == ai4k::LAY_AIT4_U || LAY == ai4k::LAY_AIT2_U || LAY == ai4k::LAY_AIT2P_U;
}

template <int LAY>
constexpr __host__ __device__ int unit_stage_bytes() {
    return ai4k::ch_bytes<LAY>() + 64;
}

template <int LAY, int WARPS>
constexpr __host__ __device__ int unit_smem_bytes() {
    constexpr int ring_b = WARPS * 2 * unit_stage_bytes<LAY>();
    constexpr int red_b  = (WARPS - 1) * 4 * 32 * 4;
    return ring_b > red_b ? ring_b : red_b;
}

template <int LAY, int XSH, bool ID, int KT, int WARPS, int MINB>
static __global__ void __launch_bounds__(WARPS * 32, MINB) k_ai4_decode_fast_unit(
        const uint8_t * __restrict__ W,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y,
        int64_t ldy, int M, int N_in, int Kr, const ai4k::idmap idm) {
    static_assert(LAY == ai4k::LAY_AIT4_U || LAY == ai4k::LAY_AIT5_U || LAY == ai4k::LAY_AIT3_U ||
                  LAY == ai4k::LAY_AIT2_U || LAY == ai4k::LAY_AIT2P_U, "unit layout");
    static_assert(WARPS == 3 || WARPS == 4, "warps");
    constexpr int  KR   = ai4k::lay_rate<LAY>();
    constexpr bool PERM = lay_use_perm<LAY>();
    constexpr int  CHB  = ai4k::ch_bytes<LAY>();
    constexpr int  UB   = ai4k::u_bytes<LAY>();
    constexpr int  ST   = unit_stage_bytes<LAY>();
    constexpr int  CHP  = CHB / 16;
    constexpr int  TW   = 4 * KR;

    extern __shared__ __align__(16) uint8_t smem[];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int slab = (int) blockIdx.x;

    int N = N_in;
    int e = 0;
    int xrow_g = 0;
    int xrow_s = 0;
    bool has_n = false;
    bool s_live = false;
    int col0 = 0, col1 = 0;

    if constexpr (ID) {
        const int u      = (int) blockIdx.y;
        const int tok    = (int) blockIdx.z;
        const int n_used = idm.direct_n_used;
        const int own    = tok * n_used + u;
        const bool contig_ids = (idm.direct_nb0 == (int64_t) sizeof(int32_t)) &&
                                (idm.direct_nb1 == (int64_t) n_used * (int64_t) sizeof(int32_t));
        e = contig_ids
            ? ((const int32_t *) idm.direct_ids)[own]
            : *(const int32_t *) (idm.direct_ids + u * idm.direct_nb0 + tok * idm.direct_nb1);
        if (e < 0 || e >= idm.direct_n_expert) {
            if (idm.direct_zero && tid < 16) {
                Y[(int64_t) own * ldy + slab * 16 + tid] = 0.0f;
            }
            return;
        }
        const int routes = (int) gridDim.z * n_used; // <= 64
        int id0 = -1, id1 = -1;
        if (contig_ids) {
            const int32_t * idp = (const int32_t *) idm.direct_ids;
            if (lane < routes) id0 = idp[lane];
            if (routes > 32 && (32 + lane) < routes) id1 = idp[32 + lane];
        } else {
            if (lane < routes) {
                const int t0 = lane / n_used, u0 = lane - t0 * n_used;
                id0 = *(const int32_t *) (idm.direct_ids + u0 * idm.direct_nb0 + t0 * idm.direct_nb1);
            }
            if (routes > 32 && (32 + lane) < routes) {
                const int r1 = 32 + lane;
                const int t1 = r1 / n_used, u1 = r1 - t1 * n_used;
                id1 = *(const int32_t *) (idm.direct_ids + u1 * idm.direct_nb0 + t1 * idm.direct_nb1);
            }
        }
        const uint32_t m0 = __ballot_sync(0xffffffffu, (lane < routes) && (id0 == e));
        const uint32_t m1 = (routes > 32) ? __ballot_sync(0xffffffffu, ((32 + lane) < routes) && (id1 == e)) : 0u;
        const uint32_t prior0 = (own < 32) ? (m0 & ((1u << own) - 1u)) : m0;
        const uint32_t prior1 = (own < 32) ? 0u : (m1 & ((1u << (own - 32)) - 1u));
        const int before = __popc(prior0) + __popc(prior1);
        if ((before & 7) != 0) {
            return;
        }
        const uint32_t f0 = m0 & ~prior0;
        const uint32_t f1 = m1 & ~prior1;
        const int n0 = __popc(f0);
        N = min(8, n0 + __popc(f1));

        uint8_t * ring0 = smem + warp * 2 * ST;
        int * wcols = (int *) ring0;
        const uint32_t lower = (1u << lane) - 1u;
        if (f0 & (1u << lane)) {
            const int idx = __popc(f0 & lower);
            if (idx < 8) wcols[idx] = lane;
        }
        if (f1 & (1u << lane)) {
            const int idx = n0 + __popc(f1 & lower);
            if (idx < 8) wcols[idx] = 32 + lane;
        }
        __syncwarp();

        const int ne11 = idm.direct_ne11;
        const bool pow2_used = (n_used & (n_used - 1)) == 0;
        const int  used_sh   = __ffs(n_used) - 1;
        auto route_to_xrow = [&](int r) -> int {
            if (ne11 == n_used) return r;
            if (ne11 == 1) return pow2_used ? (r >> used_sh) : (r / n_used);
            return (r / n_used) * ne11 + (r % n_used) % ne11;
        };

        has_n  = (g < N);
        xrow_g = has_n ? route_to_xrow(wcols[g]) : 0;
        const int sn = lane - 8;
        s_live = (lane >= 8 && lane < 16 && sn < N);
        xrow_s = s_live ? route_to_xrow(wcols[sn]) : 0;
        const int c_out = 2 * t;
        col0 = (warp == 0 && c_out     < N) ? wcols[c_out]     : 0;
        col1 = (warp == 0 && c_out + 1 < N) ? wcols[c_out + 1] : 0;
        __syncwarp();
    } else {
        has_n  = (g < N);
        xrow_g = has_n ? g : 0;
        const int sn = lane - 8;
        s_live = (lane >= 8 && lane < 16 && sn < N);
        xrow_s = s_live ? sn : 0;
        col0 = 2 * t;
        col1 = 2 * t + 1;
    }

    if constexpr (ai4k::lay_pair<LAY>()) {
        ai4k::pair_tab_async(tid, WARPS * 32);
    }

    const int K   = KT > 0 ? KT : Kr;
    const int nk  = K / 128;
    const int xsn = K >> (7 + XSH);
    constexpr bool EXACT_NMY = (KT > 0) && ((KT / 128) % WARPS == 0);
    const int nmy = EXACT_NMY ? (KT / (128 * WARPS)) : ((nk - warp + WARPS - 1) / WARPS);

    const uint8_t * ws = W + (ID ? (int64_t) e * idm.nb02 : 0) + (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    const uint8_t * wp_cur = ws + (int64_t) warp * UB;
    const int8_t  * xr_cur = Xq + (int64_t) xrow_g * K + (int64_t) warp * 128 +
                             (PERM ? (64 * (t & 1) + 4 * (t >> 1)) : (t * 32));
    const float   * xsr    = Xs + (int64_t) xrow_s * xsn;
    uint8_t * ring = smem + warp * 2 * ST;

    // Precompute hoisted SMEM word pointers for stage 0 (stage 1 = +ST/4 constant immediate!) -> ZERO inner IMADs!
    const uint32_t * tl0 = (const uint32_t *) ring + (PERM ? (t * TW) : ((t >> 1) * TW));
    const ai4k::k5win wa = (KR == 4) ? ai4k::k5win{g, (g + 15) & 15, 0}
                         : (KR == 2) ? ai4k::kr_win<2>(g, 1)
                         : (KR == 5) ? ai4k::k5_win(g, t & 1)
                                     : ai4k::kr_win<3>(g, t & 1);
    const ai4k::k5win wb = (KR == 4) ? ai4k::k5win{g + 8, g + 7, 0}
                         : (KR == 2) ? ai4k::kr_win<2>(g + 8, 1)
                         : (KR == 5) ? ai4k::k5_win(g + 8, t & 1)
                                     : ai4k::kr_win<3>(g + 8, t & 1);
    const uint32_t * pa0_s0 = tl0 + wa.w0;
    const uint32_t * pa1_s0 = tl0 + wa.w1;
    const uint32_t * pb0_s0 = tl0 + wb.w0;
    const uint32_t * pb1_s0 = tl0 + wb.w1;

    auto load_chunk = [&](const uint8_t * uptr, int kc, int slot) {
        uint8_t * st = ring + slot * ST;
        const uint8_t * tiles = uptr + ai4k::U_SC;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        const bool is_wsc = (lane < 8);
        const bool is_xsc = (lane >= 8 && lane < 16);
        const void * src = is_wsc ? (const void *) (uptr + lane * 4)
                                  : (s_live ? (const void *) (xsr + (kc >> XSH)) : (const void *) Xs);
        ai4k::cp_async4_guard(st + CHB + lane * 4, src, is_wsc || s_live, is_wsc || is_xsc);
    };

    auto ldx = [&](const int8_t * xp, uint32_t (&xv)[8]) {
        if (has_n) {
            if constexpr (PERM) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    xv[j]     = __ldg((const uint32_t *) (xp + 8 * j));
                    xv[4 + j] = __ldg((const uint32_t *) (xp + 32 + 8 * j));
                }
            } else {
                const uint4 v0 = __ldg((const uint4 *) xp);
                const uint4 v1 = __ldg((const uint4 *) xp + 1);
                xv[0] = v0.x; xv[1] = v0.y; xv[2] = v0.z; xv[3] = v0.w;
                xv[4] = v1.x; xv[5] = v1.y; xv[6] = v1.z; xv[7] = v1.w;
            }
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) { xv[k] = 0u; }
        }
    };

    if (EXACT_NMY || nmy > 0) {
        load_chunk(wp_cur, warp, 0);
    }
    cp_async_commit();

    uint32_t xa[8], xb[8];
    if (EXACT_NMY || nmy > 0) {
        ldx(xr_cur, xa);
    }
    if constexpr (ai4k::lay_pair<LAY>()) {
        cp_async_wait<0>();
        __syncthreads();
    }

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int kc_cur = warp;

    auto body = [&](int i, int slot, const uint32_t (&xcur)[8], uint32_t (&xnxt)[8]) {
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            load_chunk(wp_cur + (int64_t) WARPS * UB, kc_cur + WARPS, slot ^ 1);
        }
        cp_async_commit();
        if (i + 1 < nmy) {
            ldx(xr_cur + (int64_t) WARPS * 128, xnxt);
        }
        wp_cur += (int64_t) WARPS * UB;
        xr_cur += (int64_t) WARPS * 128;
        kc_cur += WARPS;

        const uint8_t  * st  = ring + slot * ST;
        const int        soff = slot * (ST / 4);
        const half     * sdw = (const half *) (st + CHB);
        const float    * sdx = (const float *) (st + CHB + 32);
        const float d0  = __half2float(sdw[g]);
        const float d1  = __half2float(sdw[g + 8]);
        const float dx0 = sdx[2 * t];
        const float dx1 = sdx[2 * t + 1];

        int p[4] = { 0, 0, 0, 0 };
        int q[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int joff = soff + 4 * j * TW;
            uint32_t a[4];
            uint32_t b0, b1;
            if constexpr (LAY == ai4k::LAY_AIT4_U) {
                levels8_k4<ai4k::lay_mul<LAY>()>(pa0_s0[joff], pa1_s0[joff], a[0], a[2]);
                levels8_k4<ai4k::lay_mul<LAY>()>(pb0_s0[joff], pb1_s0[joff], a[1], a[3]);
                b0 = xcur[j]; b1 = xcur[4 + j];
            } else if constexpr (LAY == ai4k::LAY_AIT2_U) {
                ai4k::levels8_k2<ai4k::lay_mul<LAY>()>(__funnelshift_l(pa1_s0[joff], pa0_s0[joff], wa.sh), a[0], a[2]);
                ai4k::levels8_k2<ai4k::lay_mul<LAY>()>(__funnelshift_l(pb1_s0[joff], pb0_s0[joff], wb.sh), a[1], a[3]);
                b0 = xcur[j]; b1 = xcur[4 + j];
            } else if constexpr (LAY == ai4k::LAY_AIT2P_U) {
                ai4k::pair_levels8(__funnelshift_l(pa1_s0[joff], pa0_s0[joff], wa.sh), a[0], a[2]);
                ai4k::pair_levels8(__funnelshift_l(pb1_s0[joff], pb0_s0[joff], wb.sh), a[1], a[3]);
                b0 = xcur[j]; b1 = xcur[4 + j];
            } else if constexpr (LAY == ai4k::LAY_AIT3_U) {
                const int boff = joff + 2 * TW;
                a[0] = ai4k::levels4_kr<3, ai4k::lay_mul<LAY>()>(__funnelshift_l(pa1_s0[joff], pa0_s0[joff], wa.sh));
                a[1] = ai4k::levels4_kr<3, ai4k::lay_mul<LAY>()>(__funnelshift_l(pb1_s0[joff], pb0_s0[joff], wb.sh));
                a[2] = ai4k::levels4_kr<3, ai4k::lay_mul<LAY>()>(__funnelshift_l(pa1_s0[boff], pa0_s0[boff], wa.sh));
                a[3] = ai4k::levels4_kr<3, ai4k::lay_mul<LAY>()>(__funnelshift_l(pb1_s0[boff], pb0_s0[boff], wb.sh));
                b0 = xcur[2 * j]; b1 = xcur[2 * j + 1];
            } else {
                const int boff = joff + 2 * TW;
                a[0] = ai4k::levels4_k5<ai4k::lay_mul<LAY>()>(__funnelshift_l(pa1_s0[joff], pa0_s0[joff], wa.sh));
                a[1] = ai4k::levels4_k5<ai4k::lay_mul<LAY>()>(__funnelshift_l(pb1_s0[joff], pb0_s0[joff], wb.sh));
                a[2] = ai4k::levels4_k5<ai4k::lay_mul<LAY>()>(__funnelshift_l(pa1_s0[boff], pa0_s0[boff], wa.sh));
                a[3] = ai4k::levels4_k5<ai4k::lay_mul<LAY>()>(__funnelshift_l(pb1_s0[boff], pb0_s0[boff], wb.sh));
                b0 = xcur[2 * j]; b1 = xcur[2 * j + 1];
            }
            if (j < 2) {
                mma_s8(p, a, b0, b1);
            } else {
                mma_s8(q, a, b0, b1);
            }
        }
        acc[0] += magic_i2f(p[0] + q[0]) * (d0 * dx0);
        acc[1] += magic_i2f(p[1] + q[1]) * (d0 * dx1);
        acc[2] += magic_i2f(p[2] + q[2]) * (d1 * dx0);
        acc[3] += magic_i2f(p[3] + q[3]) * (d1 * dx1);
    };

#pragma unroll 1
    for (int i0 = 0; i0 < nmy; i0 += 2) {
        if (i0 < nmy)     body(i0,     0, xa, xb);
        if (i0 + 1 < nmy) body(i0 + 1, 1, xb, xa);
    }
    cp_async_wait<0>();

    __syncthreads();
    float * red = (float *) smem;
    const int c_out = 2 * t;
    if (warp > 0 && c_out < N) {
        red[((warp - 1) * 4 + 0) * 32 + lane] = acc[0];
        red[((warp - 1) * 4 + 2) * 32 + lane] = acc[2];
        if (c_out + 1 < N) {
            red[((warp - 1) * 4 + 1) * 32 + lane] = acc[1];
            red[((warp - 1) * 4 + 3) * 32 + lane] = acc[3];
        }
    }
    __syncthreads();
    if (warp == 0 && c_out < N) {
#pragma unroll
        for (int w = 0; w < WARPS - 1; ++w) {
            acc[0] += red[(w * 4 + 0) * 32 + lane];
            acc[2] += red[(w * 4 + 2) * 32 + lane];
            if (c_out + 1 < N) {
                acc[1] += red[(w * 4 + 1) * 32 + lane];
                acc[3] += red[(w * 4 + 3) * 32 + lane];
            }
        }
        const int r = slab * 16 + g;
        Y[(int64_t) col0 * ldy + r]     = acc[0];
        Y[(int64_t) col0 * ldy + r + 8] = acc[2];
        if (c_out + 1 < N) {
            Y[(int64_t) col1 * ldy + r]     = acc[1];
            Y[(int64_t) col1 * ldy + r + 8] = acc[3];
        }
    }
    (void) M;
}

// =================================================================================================
// 3. Fractional AI+ High-Throughput Decode Kernel (AIF175_U2, AIF25_U, AIF25_U2, AIF25_U2_512)
//    - 2-stage unrolled cp.async.ca.16 warp ring + hoisted SMEM pointers (ZERO inner IMADs!)
//    - Whole-row 4-state quad decode (7 extract ops / 8 weights!) + single LDG.64 per IMMA!
//    - 100.000% BIT-EXACT to magic2::mmv!
// =================================================================================================

template <int B, int V>
constexpr __host__ __device__ int frac_payload_bytes() {
    return 256 * B / V;
}

template <int B, int V>
constexpr __host__ __device__ int frac_stage_bytes() {
    return frac_payload_bytes<B, V>() + 32;
}

template <int B, int V, uint32_t C, int XSH, bool ROW_SCALE, int SCALE_CHUNKS, bool MASK>
static __global__ void __launch_bounds__(128, 6) k_aif_decode_fast(
        const uint8_t * __restrict__ weights,
        const int8_t  * __restrict__ x,
        const float   * __restrict__ xs,
        float         * __restrict__ y,
        int M, int K, int N,
        const uint8_t * __restrict__ group_mask) {
    if constexpr (MASK) {
        if (group_mask[blockIdx.x] == 0) {
            return;
        }
    }
    constexpr int PB    = frac_payload_bytes<B, V>();
    constexpr int ST    = frac_stage_bytes<B, V>();
    constexpr int CHP   = PB / 16;
    constexpr int TW    = 4 * B / V;
    constexpr int BLK_B = ROW_SCALE ? (32 + SCALE_CHUNKS * PB) : (32 + PB);

    __shared__ __align__(16) uint8_t ring_all[4 * 2 * ST];
    __shared__ __align__(16) float   partial[4][8][16];

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int g     = lane >> 2;
    const int t     = lane & 3;
    const int nu    = K / 128;
    const int64_t nbase = (int64_t) blockIdx.y * 8;
    const int64_t mbase = (int64_t) blockIdx.x * 16;

    const ai4k::k5win wa = frac_row_win<B, V>(g);
    const ai4k::k5win wb = frac_row_win<B, V>(g + 8);

    uint8_t * ring = ring_all + warp * (2 * ST);
    const uint32_t * tl0    = (const uint32_t *) ring + t * TW;
    const uint32_t * pa0_s0 = tl0 + wa.w0;
    const uint32_t * pa1_s0 = tl0 + wa.w1;
    const uint32_t * pb0_s0 = tl0 + wb.w0;
    const uint32_t * pb1_s0 = tl0 + wb.w1;

    const uint8_t * slab_base = ROW_SCALE
        ? (weights + (int64_t) blockIdx.x * (nu / SCALE_CHUNKS) * BLK_B)
        : (weights + (int64_t) blockIdx.x * nu * BLK_B);

    const uint8_t * blk_ptr = ROW_SCALE
        ? (slab_base + (int64_t) (SCALE_CHUNKS == 4 ? 0 : (warp / SCALE_CHUNKS)) * BLK_B)
        : (slab_base + (int64_t) warp * BLK_B);
    int cur_sub = ROW_SCALE ? (SCALE_CHUNKS == 4 ? warp : (warp % SCALE_CHUNKS)) : 0;

    auto load_cur = [&](int slot) {
        uint8_t * st = ring + slot * ST;
        const uint8_t * pay = ROW_SCALE ? (blk_ptr + 32 + cur_sub * PB) : (blk_ptr + 32);
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, pay + r * 512 + lane * 16, true);
            }
        }
        if (lane < 2) {
            cp_async16(st + PB + lane * 16, blk_ptr + lane * 16, true);
        }
    };

    auto adv4 = [&]() {
        if constexpr (ROW_SCALE) {
            if constexpr (SCALE_CHUNKS == 4) {
                blk_ptr += BLK_B;
            } else {
                if (cur_sub >= 1) {
                    cur_sub -= 1;
                    blk_ptr += BLK_B;
                } else {
                    cur_sub = 4;
                }
            }
        } else {
            blk_ptr += 4 * BLK_B;
        }
    };

    const bool has_row = (nbase + g < N);
    const int8_t * xu = x + (has_row ? (nbase + g) : 0) * K + (int64_t) warp * 128 + t * 8;

    const int nmy = (nu - warp + 3) >> 2;
    const int xsn = nu >> XSH;
    const bool live_s0 = (nbase + 2 * t < N);
    const bool live_s1 = (nbase + 2 * t + 1 < N);
    const float * xs0 = live_s0 ? (xs + (nbase + 2 * t) * xsn + (warp >> XSH)) : nullptr;
    const float * xs1 = live_s1 ? (xs + (nbase + 2 * t + 1) * xsn + (warp >> XSH)) : nullptr;
    constexpr int XS_STEP = 4 >> XSH;

    if (nmy > 0) {
        load_cur(0);
    }
    cp_async_commit();

    float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    auto step = [&](int i, int slot) {
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            adv4();
            load_cur(slot ^ 1);
        }
        cp_async_commit();

        const int soff = slot * (ST / 4);
        const __half * sc = (const __half *) (ring + slot * ST + PB);

        int c[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int joff = soff + 4 * j * TW;
            uint32_t a[4];
            levels8_frac<B, V, C>(__funnelshift_l(pa1_s0[joff], pa0_s0[joff], wa.sh), a[0], a[2]);
            levels8_frac<B, V, C>(__funnelshift_l(pb1_s0[joff], pb0_s0[joff], wb.sh), a[1], a[3]);
            uint2 bv = make_uint2(0u, 0u);
            if (has_row) {
                bv = __ldg((const uint2 *) (xu + j * 32));
            }
            mma_s8(c, a, bv.x, bv.y);
        }

        const float d0 = __half2float(sc[g]);
        const float d1 = __half2float(sc[g + 8]);
        const float s0 = live_s0 ? __ldg(xs0) : 0.0f;
        const float s1 = live_s1 ? __ldg(xs1) : 0.0f;
        sum[0] += c[0] * d0 * s0;
        sum[1] += c[1] * d0 * s1;
        sum[2] += c[2] * d1 * s0;
        sum[3] += c[3] * d1 * s1;
        xu  += 4 * 128;
        xs0 += XS_STEP;
        xs1 += XS_STEP;
    };

#pragma unroll 1
    for (int i0 = 0; i0 < nmy; i0 += 2) {
        if (i0 < nmy)     step(i0,     0);
        if (i0 + 1 < nmy) step(i0 + 1, 1);
    }
    cp_async_wait<0>();

    partial[warp][2 * t][g]         = sum[0];
    partial[warp][2 * t + 1][g]     = sum[1];
    partial[warp][2 * t][g + 8]     = sum[2];
    partial[warp][2 * t + 1][g + 8] = sum[3];
    __syncthreads();
    const int n = threadIdx.x >> 4;
    const int r = threadIdx.x & 15;
    if (nbase + n < N) {
        y[(nbase + n) * M + mbase + r] =
            (partial[0][n][r] + partial[1][n][r]) + (partial[2][n][r] + partial[3][n][r]);
    }
}

// =================================================================================================
// 4. Dispatch Helpers (ONLY reachable templates instantiated; MINB tuned for ZERO spills!)
// =================================================================================================

template <int LAY, int XSH, bool ID, int KT, int WARPS, int MINB>
static void launch_unit(
        const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y,
        int M, int N, int K, int n_used, const ai4k::idmap & idm, cudaStream_t stream) {
    constexpr int smem = unit_smem_bytes<LAY, WARPS>();
    const dim3 grid = ID ? dim3(M / 16, n_used, N) : dim3(M / 16, 1, 1);
    k_ai4_decode_fast_unit<LAY, XSH, ID, KT, WARPS, MINB><<<grid, WARPS * 32, smem, stream>>>(
            W, Xq, Xs, Y, M, M, N, K, idm);
}

template <int LAY, int XSH, bool ID>
static void dispatch_unit_kt(
        const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y,
        int M, int N, int K, int n_used, const ai4k::idmap & idm, cudaStream_t stream) {
    const bool div3 = ((K / 128) % 3) == 0;
    if constexpr (LAY == ai4k::LAY_AIT4_U && XSH == 1) {
        if (K == 2304) { launch_unit<LAY, 1, ID, 2304, 3, 8>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream); return; }
    }
    if constexpr (LAY == ai4k::LAY_AIT4_U && XSH == 0) {
        if (K == 1152) { launch_unit<LAY, 0, ID, 1152, 3, 8>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream); return; }
    }
    if constexpr (LAY == ai4k::LAY_AIT5_U && XSH == 1) {
        if (K == 2304) { launch_unit<LAY, 1, ID, 2304, 3, 8>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream); return; }
    }
    if constexpr (LAY == ai4k::LAY_AIT5_U && XSH == 0) {
        if (K == 1152) { launch_unit<LAY, 0, ID, 1152, 3, 8>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream); return; }
    }
    if (div3) {
        launch_unit<LAY, XSH, ID, 0, 3, 8>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream);
    } else {
        launch_unit<LAY, XSH, ID, 0, 4, 6>(W, Xq, Xs, Y, M, N, K, n_used, idm, stream);
    }
}

template <int B, int V, uint32_t C, int XSH, bool ROW_SCALE, int SCALE_CHUNKS>
static void launch_frac(
        const uint8_t * w, const int8_t * q, const float * scales, float * y,
        int M, int K, int N, const uint8_t * group_mask, cudaStream_t stream) {
    const dim3 grid(M / 16, (N + 7) / 8);
    if (group_mask != nullptr) {
        k_aif_decode_fast<B, V, C, XSH, ROW_SCALE, SCALE_CHUNKS, true><<<grid, 128, 0, stream>>>(
                w, q, scales, y, M, K, N, group_mask);
    } else {
        k_aif_decode_fast<B, V, C, XSH, ROW_SCALE, SCALE_CHUNKS, false><<<grid, 128, 0, stream>>>(
                w, q, scales, y, M, K, N, nullptr);
    }
}

} // namespace ai4_fast

bool ggml_cuda_ai4_decode_fast_unit_try(
        ggml_backend_cuda_context & ctx,
        ggml_type t,
        const uint8_t * W,
        int64_t nb02,
        const ggml_tensor * ids,
        const int8_t * xq,
        const float  * xs,
        float        * y_out,
        int M, int N, int K,
        int n_used, int ne11, int n_expert,
        int xsh,
        bool has_y_into) {
    if (!ggml_cuda_ai4_decode_fast_enabled()) {
        return false;
    }
    if (N < 1 || N > 8 || M <= 0 || M % 16 != 0 || K <= 0 || K % 128 != 0) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    if (!GGML_CUDA_CC_IS_NVIDIA(dev.cc) || dev.cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if (t != GGML_TYPE_AIT4_U && t != GGML_TYPE_AIT5_U && t != GGML_TYPE_AIT3_U &&
        t != GGML_TYPE_AIT2_U && t != GGML_TYPE_AIT2P_U) {
        return false;
    }
    const bool is_id = (ids != nullptr);
    ai4k::idmap idm{};
    if (is_id) {
        if (n_used <= 0 || n_expert <= 0 || (int64_t) n_used * N > 64) {
            return false;
        }
        idm.nb02            = nb02;
        idm.direct_ids      = (const char *) ids->data;
        idm.direct_nb0      = (int64_t) ids->nb[0];
        idm.direct_nb1      = (int64_t) ids->nb[1];
        idm.direct_n_used   = n_used;
        idm.direct_ne11     = ne11;
        idm.direct_n_expert = n_expert;
        idm.direct_zero     = !has_y_into;
    }

    cudaStream_t stream = ctx.stream();

#define AI4_FAST_DISPATCH(LAY_VAL) \
    do { \
        if (is_id) { \
            if (xsh) ai4_fast::dispatch_unit_kt<LAY_VAL, 1, true>(W, xq, xs, y_out, M, N, K, n_used, idm, stream); \
            else     ai4_fast::dispatch_unit_kt<LAY_VAL, 0, true>(W, xq, xs, y_out, M, N, K, n_used, idm, stream); \
        } else { \
            if (xsh) ai4_fast::dispatch_unit_kt<LAY_VAL, 1, false>(W, xq, xs, y_out, M, N, K, 1, idm, stream); \
            else     ai4_fast::dispatch_unit_kt<LAY_VAL, 0, false>(W, xq, xs, y_out, M, N, K, 1, idm, stream); \
        } \
    } while (0)

    switch (t) {
        case GGML_TYPE_AIT4_U:  AI4_FAST_DISPATCH(ai4k::LAY_AIT4_U);  break;
        case GGML_TYPE_AIT5_U:  AI4_FAST_DISPATCH(ai4k::LAY_AIT5_U);  break;
        case GGML_TYPE_AIT3_U:  AI4_FAST_DISPATCH(ai4k::LAY_AIT3_U);  break;
        case GGML_TYPE_AIT2_U:  AI4_FAST_DISPATCH(ai4k::LAY_AIT2_U);  break;
        case GGML_TYPE_AIT2P_U: AI4_FAST_DISPATCH(ai4k::LAY_AIT2P_U); break;
        default: return false;
    }
#undef AI4_FAST_DISPATCH
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_ai4_decode_fast_frac_try(
        ggml_backend_cuda_context & ctx,
        ggml_type type,
        int xsh,
        const uint8_t * w,
        const int8_t  * q,
        const float   * scales,
        float         * y,
        int M, int K, int64_t N,
        const uint8_t * group_mask) {
    if (!ggml_cuda_ai4_decode_fast_enabled()) {
        return false;
    }
    if (N < 1 || N > 8 || M <= 0 || M % 16 != 0 || K <= 0 || K % 128 != 0) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    if (!GGML_CUDA_CC_IS_NVIDIA(dev.cc) || dev.cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    if (type == GGML_TYPE_AIF175_U2) {
        if (K % 512 != 0 || xsh != 1) return false;
        ai4_fast::launch_frac<7, 4, AIF175_MUL, 1, true, 4>(w, q, scales, y, M, K, (int) N, group_mask, stream);
    } else if (type == GGML_TYPE_AIF25_U2_512) {
        if (K % 512 != 0 || xsh != 1) return false;
        ai4_fast::launch_frac<5, 2, AIF25_MUL, 1, true, 4>(w, q, scales, y, M, K, (int) N, group_mask, stream);
    } else if (type == GGML_TYPE_AIF25_U2) {
        if (K % 640 != 0) return false;
        if (xsh) ai4_fast::launch_frac<5, 2, AIF25_MUL, 1, true, 5>(w, q, scales, y, M, K, (int) N, group_mask, stream);
        else     ai4_fast::launch_frac<5, 2, AIF25_MUL, 0, true, 5>(w, q, scales, y, M, K, (int) N, group_mask, stream);
    } else if (type == GGML_TYPE_AIF25_U) {
        if (xsh) ai4_fast::launch_frac<5, 2, AIF25_MUL, 1, false, 1>(w, q, scales, y, M, K, (int) N, group_mask, stream);
        else     ai4_fast::launch_frac<5, 2, AIF25_MUL, 0, false, 1>(w, q, scales, y, M, K, (int) N, group_mask, stream);
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif
