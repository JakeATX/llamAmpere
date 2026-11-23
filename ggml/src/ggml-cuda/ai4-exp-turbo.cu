// [INK200 T1] SM 8.0+ (Ampere RTX 3090/Ti) SpecDecode (M = 2..32) grouped MoE expert kernels.
// Standalone translation unit for fast incremental builds and clean SASS auditing.
#include "ai4-exp-turbo.cuh"
#include "ai4-kern.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_ai4_exp_turbo_try(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
        const int8_t *, const float *, float *, int, int, int, int, int, int, int, bool, bool) {
    return false;
}

void ggml_cuda_ai4_exp_turbo_free(const ggml_backend_cuda_context *) {}

#else

namespace ai4_exp_turbo {

using rq4::cp_async16;
using rq4::cp_async4;
using rq4::cp_async_commit;
using rq4::cp_async_wait;
using rq4::mma_s8;
using rq4::magic_i2f;

static int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

static bool turbo_enabled() {
    static const bool v = env_int("GGML_CUDA_AI4_EXP_TURBO", 1) != 0;
    return v;
}

static int turbo_max_n() {
    static const int v = std::max(0, std::min(32, env_int("GGML_CUDA_AI4_EXP_TURBO_MAX_N", 32)));
    return v;
}

static bool turbo_idcache_enabled() {
    static const bool v = env_int("GGML_CUDA_AI4_EXP_TURBO_IDCACHE", 1) != 0;
    return v;
}

enum DirectCfg {
    DCFG_AUTO = 0,
    DCFG_3x2  = 1,
    DCFG_3x3  = 2,
    DCFG_4x2  = 3,
};

static int turbo_direct_cfg() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_AI4_EXP_TURBO_CFG");
        if (!e || !strcmp(e, "auto")) return (int) DCFG_AUTO;
        if (!strcmp(e, "3x2")) return (int) DCFG_3x2;
        if (!strcmp(e, "3x3")) return (int) DCFG_3x3;
        if (!strcmp(e, "4x2")) return (int) DCFG_4x2;
        GGML_LOG_WARN("GGML_CUDA_AI4_EXP_TURBO_CFG='%s' invalid (expected auto, 3x2, 3x3, 4x2); using auto\n", e);
        return (int) DCFG_AUTO;
    }();
    return v;
}

// =================================================================================================
// 1. Direct-Grouped Split-K Expert Kernel (M = 2..8, n_pairs <= 64)
//    - ZERO ids_helper launch
//    - ZERO static SMEM (reduction aliases dead cp.async ring)
//    - ZERO __syncthreads() for routing (warp-private 32B scatter into own ring[0..31])
//    - Dead CTAs exit in ~15 insns with ZERO BAR.SYNC
//    - Native a_regs_smem (levels4 PRMT) for AIT4_U instead of levels4_kr<4>
//    - Skips empty LDG.128 on inactive columns (g >= N)
//    - WARPS=3 gives 100% K-balance on Inkling-2-Mini K=2304 (18/3=6) AND K=1152 (9/3=3)
// =================================================================================================

template <int LAY>
constexpr __host__ __device__ int direct_stage_bytes() {
    // CHB tiles (16-B aligned) + 32B weight scales (16 fp16) + 32B act scales (8 fp32)
    return ai4k::ch_bytes<LAY>() + 64;
}

template <int LAY, int WARPS, int STAGES>
constexpr __host__ __device__ int direct_smem_bytes() {
    constexpr int ring_b = WARPS * STAGES * direct_stage_bytes<LAY>();
    constexpr int red_b  = (WARPS - 1) * 4 * 32 * 4;
    return ring_b > red_b ? ring_b : red_b;
}

template <int LAY, int XSH, int KT, int WARPS, int STAGES, int MINB>
static __global__ void __launch_bounds__(WARPS * 32, MINB) exp_turbo_direct_s8(
        const uint8_t * __restrict__ W,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y,
        int64_t ldy, int M, int Kr, const ai4k::idmap idm) {
    static_assert(LAY == ai4k::LAY_AIT4_U || LAY == ai4k::LAY_AIT3_U || LAY == ai4k::LAY_AIT2_U, "unit layout");
    static_assert(WARPS == 3 || WARPS == 4, "warps");
    static_assert(STAGES == 2 || STAGES == 3, "stages");

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int u      = (int) blockIdx.y;
    const int tok    = (int) blockIdx.z;
    const int n_used = idm.direct_n_used;
    const int own    = tok * n_used + u;
    const int slab   = (int) blockIdx.x;

    // Fast path when ids [n_used, n_tok] is contiguous (true for Inkling-2-Mini & standard MoE)
    const bool contig_ids = (idm.direct_nb0 == (int64_t) sizeof(int32_t)) &&
                            (idm.direct_nb1 == (int64_t) n_used * (int64_t) sizeof(int32_t));
    const int e = contig_ids
        ? ((const int32_t *) idm.direct_ids)[own]
        : *(const int32_t *) (idm.direct_ids + u * idm.direct_nb0 + tok * idm.direct_nb1);

    if (e < 0 || e >= idm.direct_n_expert) {
        if (idm.direct_zero && threadIdx.x < 16) {
            Y[(int64_t) own * ldy + slab * 16 + threadIdx.x] = 0.0f;
        }
        return;
    }

    // Register-only 64-route warp ballot (ZERO SMEM, ZERO __syncthreads())
    const int routes = (int) gridDim.z * n_used; // <= 64
    int id0 = -1, id1 = -1;
    if (contig_ids) {
        const int32_t * idp = (const int32_t *) idm.direct_ids;
        if (lane < routes) id0 = idp[lane];
        if (routes > 32 && (32 + lane) < routes) id1 = idp[32 + lane];
    } else {
        if (lane < routes) {
            const int u0 = lane % n_used, t0 = lane / n_used;
            id0 = *(const int32_t *) (idm.direct_ids + u0 * idm.direct_nb0 + t0 * idm.direct_nb1);
        }
        if (routes > 32 && (32 + lane) < routes) {
            const int r1 = 32 + lane;
            const int u1 = r1 % n_used, t1 = r1 / n_used;
            id1 = *(const int32_t *) (idm.direct_ids + u1 * idm.direct_nb0 + t1 * idm.direct_nb1);
        }
    }

    const uint32_t m0 = __ballot_sync(0xffffffffu, (lane < routes) && (id0 == e));
    const uint32_t m1 = (routes > 32) ? __ballot_sync(0xffffffffu, ((32 + lane) < routes) && (id1 == e)) : 0u;

    const uint32_t prior0 = (own < 32) ? (m0 & ((1u << own) - 1u)) : m0;
    const uint32_t prior1 = (own < 32) ? 0u : (m1 & ((1u << (own - 32)) - 1u));
    const int before = __popc(prior0) + __popc(prior1);
    // Dead CTA instant exit (0 BAR.SYNC, 0 SMEM)
    if ((before & 7) != 0) {
        return;
    }

    const uint32_t f0 = m0 & ~prior0;
    const uint32_t f1 = m1 & ~prior1;
    const int n0 = __popc(f0);
    const int N  = min(8, n0 + __popc(f1));

    // Warp-private 32-byte scatter into own ring[0..31] (ZERO static SMEM, ZERO __syncthreads())
    constexpr int CHB = ai4k::ch_bytes<LAY>();
    constexpr int ST  = direct_stage_bytes<LAY>();
    constexpr int CHP = CHB / 16;
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t * ring = smem + warp * STAGES * ST;
    int * wcols = (int *) ring;
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

    const int K    = KT > 0 ? KT : Kr;
    const int nk   = K / 128;
    const int xsn  = K >> (7 + XSH);
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int ne11 = idm.direct_ne11;
    const bool pow2_used = (n_used & (n_used - 1)) == 0;
    const int  used_sh   = __ffs(n_used) - 1;

    auto route_to_xrow = [&](int r) -> int {
        if (ne11 == n_used) return r; // down projection: 1:1 with route index (0 div/mod!)
        if (ne11 == 1) return pow2_used ? (r >> used_sh) : (r / n_used); // gate_up: token index
        return (r / n_used) * ne11 + (r % n_used) % ne11;   // 1 < ne11 < n_used: slot j reads src1 row j % ne11 (ggml MUL_MAT_ID)
    };

    const bool has_n  = (g < N);
    const int xrow_g  = has_n ? route_to_xrow(wcols[g]) : 0;
    const int8_t * xr = Xq + (int64_t) xrow_g * K + t * 32;

    const int sn      = lane - 8;
    const bool s_live = (lane >= 8 && lane < 16 && sn < N);
    const int xrow_s  = s_live ? route_to_xrow(wcols[sn]) : 0;
    const float * xsr = Xs + (int64_t) xrow_s * xsn;

    // Save output column indices in 2 registers (only used by warp 0 at epilogue)
    const int c_out = 2 * t;
    const int col0  = (warp == 0 && c_out < N)     ? wcols[c_out]     : 0;
    const int col1  = (warp == 0 && c_out + 1 < N) ? wcols[c_out + 1] : 0;
    __syncwarp(); // all reads from wcols finished before cp.async writes into ring

    const uint8_t * ws = W + (int64_t) e * idm.nb02 + (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    constexpr bool EXACT_NMY = (KT > 0) && ((KT / 128) % WARPS == 0);
    const int nmy = EXACT_NMY ? (KT / (128 * WARPS)) : ((nk - warp + WARPS - 1) / WARPS);

    const ai4k::k5win k5a = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win k5b = ai4k::lay_win<LAY>(g + 8, t & 1);
    (void) k5a; (void) k5b;

    auto load = [&](int i, int slot) {
        const int kc = warp + i * WARPS;
        uint8_t * st = ring + slot * ST;
        const uint8_t * tiles = ws + ai4k::chunk_off<LAY>(kc);
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        const bool is_wsc = (lane < 8);
        const bool is_xsc = (lane >= 8 && lane < 16);
        const void * src = is_wsc ? (const void *) (ws + ai4k::sc_off<LAY>(kc) + lane * 4)
                                  : (s_live ? (const void *) (xsr + (kc >> XSH)) : (const void *) Xs);
        ai4k::cp_async4_guard(st + CHB + lane * 4, src, is_wsc || s_live, is_wsc || is_xsc);
    };

    auto ldx = [&](int i, uint4 & v0, uint4 & v1) {
        if (has_n && (EXACT_NMY || i < nmy)) {
            const uint4 * src = (const uint4 *) (xr + (int64_t) (warp + i * WARPS) * 128);
            v0 = __ldg(src);
            v1 = __ldg(src + 1);
        } else {
            v0 = make_uint4(0, 0, 0, 0);
            v1 = make_uint4(0, 0, 0, 0);
        }
    };

    // Prologue
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (EXACT_NMY ? (s < KT / (128 * WARPS)) : (s < nmy)) {
            load(s, s);
        }
        cp_async_commit();
    }
    uint4 nx0, nx1;
    if (EXACT_NMY || nmy > 0) {
        ldx(0, nx0, nx1);
    }

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    auto body = [&](int i, int slot) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        if (i + STAGES - 1 < nmy) {
            load(i + STAGES - 1, (slot + STAGES - 1) % STAGES);
        }
        cp_async_commit();

        const uint8_t * st = ring + slot * ST;
        const uint32_t * tw = (const uint32_t *) st;
        uint4 x0 = nx0, x1 = nx1;
        if (i + 1 < nmy) {
            ldx(i + 1, nx0, nx1);
        }

        const half  * sdw = (const half *) (st + CHB);
        const float * sdx = (const float *) (st + CHB + 32);
        const float d0  = __half2float(sdw[g]);
        const float d1  = __half2float(sdw[g + 8]);
        const float dx0 = sdx[2 * t];
        const float dx1 = sdx[2 * t + 1];

        int p[4] = { 0, 0, 0, 0 };
        int q[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            ai4k::a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a);
            const uint4 & xv = (j < 2) ? x0 : x1;
            if (j < 2) {
                mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
            } else {
                mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
            }
        }
        acc[0] += magic_i2f(p[0] + q[0]) * (d0 * dx0);
        acc[1] += magic_i2f(p[1] + q[1]) * (d0 * dx1);
        acc[2] += magic_i2f(p[2] + q[2]) * (d1 * dx0);
        acc[3] += magic_i2f(p[3] + q[3]) * (d1 * dx1);
    };

    // Unroll only by STAGES: stage offsets are compile-time immediates while staying compact in L1 I-cache
#pragma unroll 1
    for (int i0 = 0; i0 < nmy; i0 += STAGES) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            if (i0 + s < nmy) {
                body(i0 + s, s);
            }
        }
    }

    cp_async_wait<0>();
    // ZERO static SMEM reduction: barrier guarantees cp.async ring is dead before reuse
    __syncthreads();
    float * red = (float *) smem; // [WARPS - 1][4][32]
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

template <int LAY, int XSH, int KT, int WARPS, int STAGES, int MINB>
static void launch_direct(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y,
                          int M, int K, int n_used, int n_tok, const ai4k::idmap & idm, cudaStream_t stream) {
    constexpr int smem = direct_smem_bytes<LAY, WARPS, STAGES>();
    const dim3 grid(M / 16, n_used, n_tok);
    exp_turbo_direct_s8<LAY, XSH, KT, WARPS, STAGES, MINB><<<grid, WARPS * 32, smem, stream>>>(
            W, Xq, Xs, Y, M, M, K, idm);
}

template <int LAY, int XSH>
static void dispatch_direct_lay(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y,
                                int M, int K, int n_used, int n_tok, const ai4k::idmap & idm, cudaStream_t stream) {
    const int cfg = turbo_direct_cfg();
    const bool div3 = ((K / 128) % 3) == 0;
    const bool use_3x3 = (cfg == DCFG_3x3);
    const bool use_4x2 = (cfg == DCFG_4x2) || (cfg == DCFG_AUTO && !div3);

    if constexpr (LAY == ai4k::LAY_AIT4_U && XSH == 1) {
        if (K == 2304) {
            if (use_4x2)      launch_direct<LAY, 1, 2304, 4, 2, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else if (use_3x3) launch_direct<LAY, 1, 2304, 3, 3, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else              launch_direct<LAY, 1, 2304, 3, 2, 10>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            return;
        }
    }
    if constexpr (LAY == ai4k::LAY_AIT4_U && XSH == 0) {
        if (K == 1152) {
            if (use_4x2)      launch_direct<LAY, 0, 1152, 4, 2, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else if (use_3x3) launch_direct<LAY, 0, 1152, 3, 3, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else              launch_direct<LAY, 0, 1152, 3, 2, 10>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            return;
        }
    }
    if constexpr (LAY == ai4k::LAY_AIT3_U && XSH == 1) {
        if (K == 2304) {
            if (use_4x2) launch_direct<LAY, 1, 2304, 4, 2, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else         launch_direct<LAY, 1, 2304, 3, 2, 10>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            return;
        }
    }
    if constexpr (LAY == ai4k::LAY_AIT3_U && XSH == 0) {
        if (K == 1152) {
            if (use_4x2) launch_direct<LAY, 0, 1152, 4, 2, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            else         launch_direct<LAY, 0, 1152, 3, 2, 10>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
            return;
        }
    }
    // Generic K fallback
    if (use_4x2) {
        launch_direct<LAY, XSH, 0, 4, 2, 8>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
    } else {
        launch_direct<LAY, XSH, 0, 3, 2, 10>(W, Xq, Xs, Y, M, K, n_used, n_tok, idm, stream);
    }
}

// =================================================================================================
// 2. Grouped Split-K Expert Kernel (M = 9..32, or n_pairs > 64)
//    - Refined v2 ampere_mmv_turbo: ZERO static SMEM, A_UPFRONT 2x LDG.128, early nt skip
//    - Derives xrow from cols[] using (ne11, n_used) so ids_helper output is 100% IDENTICAL between
//      gate_up (ne11=1) and down (ne11=n_used) -> ONE ids_helper launch per MoE visit (-50%)!
// =================================================================================================

template <int BN, int LAY>
constexpr __host__ __device__ int grouped_stage_bytes() {
    return ai4k::ch_bytes<LAY>() + 32 + ((BN * 4 + 15) & ~15);
}

template <int BN, int LAY, int WARPS>
constexpr __host__ __device__ int grouped_smem_bytes() {
    constexpr int ring_b = WARPS * 2 * grouped_stage_bytes<BN, LAY>();
    constexpr int red_b  = (WARPS - 1) * (BN / 8) * 4 * 32 * 4;
    return ring_b > red_b ? ring_b : red_b;
}

template <int BN, int LAY, int XSH, int WARPS, int MINB>
static __global__ void __launch_bounds__(WARPS * 32, MINB) exp_turbo_grouped_s8(
        const uint8_t * __restrict__ W,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y,
        int64_t ldy, int M, int K, int n_used, int ne11, const ai4k::idmap idm) {
    static_assert(BN == 8 || BN == 16 || BN == 32, "grouped verify width");
    static_assert(WARPS == 3 || WARPS == 4, "warps");
    constexpr int NTI = BN / 8;
    constexpr int CHB = ai4k::ch_bytes<LAY>();
    constexpr int ST  = grouped_stage_bytes<BN, LAY>();
    constexpr int CHP = CHB / 16;

    if ((int) blockIdx.y >= idm.hdr[1]) {
        return;
    }

    extern __shared__ __align__(16) uint8_t smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int slab = (int) blockIdx.x;
    const int nk   = K / 128;
    const int xsn  = K >> (7 + XSH);

    const int active = idm.tiles[2 * blockIdx.y];
    const int off    = idm.off[active] + idm.tiles[2 * blockIdx.y + 1];
    const int N      = min(BN, idm.off[active + 1] - off);
    const int * cols = idm.cols + off;
    const uint8_t * ws = W + (int64_t) idm.exps[active] * idm.nb02 + (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    uint8_t * ring = smem + warp * 2 * ST;
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    const bool pow2_used = (n_used & (n_used - 1)) == 0;
    const int  used_sh   = __ffs(n_used) - 1;
    auto col_to_xrow = [&](int r) -> int {
        if (ne11 == n_used) return r;
        if (ne11 == 1) return pow2_used ? (r >> used_sh) : (r / n_used);
        return (r / n_used) * ne11 + (r % n_used) % ne11;   // 1 < ne11 < n_used: slot j reads src1 row j % ne11 (ggml MUL_MAT_ID)
    };

    int xr_idx[NTI];
    bool live_c[NTI];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nt * 8 + g;
        live_c[nt] = (c < N);
        xr_idx[nt] = live_c[nt] ? col_to_xrow(cols[c]) : 0;
    }

    const bool s_live = (lane < BN) && (lane < N);
    const int xrow_s  = s_live ? col_to_xrow(cols[lane]) : 0;
    const float * xsr = Xs + (int64_t) xrow_s * xsn;

    const ai4k::k5win k5a = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win k5b = ai4k::lay_win<LAY>(g + 8, t & 1);
    (void) k5a; (void) k5b;

    auto load_stage = [&](int i, int slot) {
        const int kc = warp + i * WARPS;
        uint8_t * st = ring + slot * ST;
        const uint8_t * tiles = ws + ai4k::chunk_off<LAY>(kc);
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            cp_async4(st + CHB + lane * 4, ws + ai4k::sc_off<LAY>(kc) + lane * 4, true);
        }
        if (lane < BN) {
            cp_async4(st + CHB + 32 + lane * 4, s_live ? (const void *) (xsr + (kc >> XSH)) : (const void *) Xs, s_live);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    if (nmy > 0) load_stage(0, 0);
    cp_async_commit();

    for (int i = 0; i < nmy; ++i) {
        const int slot = i & 1;
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) load_stage(i + 1, 1 - slot);
        cp_async_commit();

        const int kc = warp + i * WARPS;
        const uint8_t * st = ring + slot * ST;
        const uint32_t * tw = (const uint32_t *) st;
        const half  * sd = (const half *) (st + CHB);
        const float * dx = (const float *) (st + CHB + 32);
        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);

        // Decode A once per 128-K chunk, reuse across all active nt groups
        uint32_t a[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            ai4k::a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a[j]);
        }

#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (nt * 8 >= N) break;
            uint32_t xv[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (live_c[nt]) {
                const uint4 * xp = (const uint4 *) (Xq + (int64_t) xr_idx[nt] * K + kc * 128 + t * 32);
                const uint4 x0 = __ldg(xp);
                const uint4 x1 = __ldg(xp + 1);
                xv[0] = x0.x; xv[1] = x0.y; xv[2] = x0.z; xv[3] = x0.w;
                xv[4] = x1.x; xv[5] = x1.y; xv[6] = x1.z; xv[7] = x1.w;
            }
            int p[4] = { 0, 0, 0, 0 };
            int q[4] = { 0, 0, 0, 0 };
            mma_s8(p, a[0], xv[0], xv[1]);
            mma_s8(p, a[1], xv[2], xv[3]);
            mma_s8(q, a[2], xv[4], xv[5]);
            mma_s8(q, a[3], xv[6], xv[7]);
            const float dx0 = dx[nt * 8 + 2 * t];
            const float dx1 = dx[nt * 8 + 2 * t + 1];
            acc[nt][0] += magic_i2f(p[0] + q[0]) * (d0 * dx0);
            acc[nt][1] += magic_i2f(p[1] + q[1]) * (d0 * dx1);
            acc[nt][2] += magic_i2f(p[2] + q[2]) * (d1 * dx0);
            acc[nt][3] += magic_i2f(p[3] + q[3]) * (d1 * dx1);
        }
    }

    cp_async_wait<0>();
    // ZERO static SMEM reduction: reuse dead cp.async ring after barrier
    __syncthreads();
    float * red = (float *) smem; // [WARPS - 1][NTI][4][32]
    if (warp > 0) {
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (nt * 8 >= N) break;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                red[(((warp - 1) * NTI + nt) * 4 + e) * 32 + lane] = acc[nt][e];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int r = slab * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (nt * 8 >= N) break;
#pragma unroll
            for (int w = 0; w < WARPS - 1; ++w) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[nt][e] += red[((w * NTI + nt) * 4 + e) * 32 + lane];
                }
            }
            const int c = nt * 8 + 2 * t;
            if (c < N) {
                const int64_t y0 = cols[c];
                Y[y0 * ldy + r]     = acc[nt][0];
                Y[y0 * ldy + r + 8] = acc[nt][2];
            }
            if (c + 1 < N) {
                const int64_t y1 = cols[c + 1];
                Y[y1 * ldy + r]     = acc[nt][1];
                Y[y1 * ldy + r + 8] = acc[nt][3];
            }
        }
    }
    (void) M;
}

template <int BN, int LAY, int XSH, int WARPS, int MINB>
static void launch_grouped(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y,
                           int M, int K, int n_used, int ne11, const ai4k::idmap & idm, int max_tiles, cudaStream_t stream) {
    constexpr int smem = grouped_smem_bytes<BN, LAY, WARPS>();
    const dim3 grid(M / 16, max_tiles);
    exp_turbo_grouped_s8<BN, LAY, XSH, WARPS, MINB><<<grid, WARPS * 32, smem, stream>>>(
            W, Xq, Xs, Y, M, M, K, n_used, ne11, idm);
}

// =================================================================================================
// 3. Per-Context 1-Entry CUDA-Graph-Safe ids_helper Cache (for M = 9..32)
//    Because exp_turbo_grouped_s8 computes xrow from cols[], gate_up (ne11=1) and down (ne11=n_used)
//    consume the EXACT SAME routing buffer -> cuts ids_helper launches by 50%!
// =================================================================================================

constexpr size_t IDCACHE_MAX_INTS = 16384; // 64 KiB fixed device buffer (never freed early -> CUDA-graph safe)

struct idcache_entry {
    int *              ptr      = nullptr;
    int                dev      = -1;
    uint64_t           epoch    = 0;
    const ggml_tensor * ids     = nullptr;
    const void *       ids_data = nullptr;
    size_t             nb0      = 0;
    size_t             nb1      = 0;
    int                n_used   = 0;
    int                n_tok    = 0;
    int                n_expert = 0;
    int                bn       = 0;
    int                max_tiles = 0;
    cudaStream_t       stream   = nullptr;
};

static std::mutex g_idcache_mu;
static std::unordered_map<const ggml_backend_cuda_context *, idcache_entry> g_idcache;

static ai4k::idmap get_or_build_idmap(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * ids,
        int n_used, int n_tok, int ne11, int n_expert, int bn, int max_tiles, int64_t nb02,
        ggml_cuda_pool_alloc<int> & fallback_buf,
        float * zero_y, int M) {
    cudaStream_t stream = ctx.stream();
    const int64_t n_pairs = (int64_t) n_used * n_tok;
    const size_t need_ints = (size_t) ai4k::ids_helper_ints(n_expert, n_pairs, max_tiles);

    const bool cacheable = turbo_idcache_enabled() && zero_y == nullptr && !ctx.scratch_borrowed &&
                           ctx.curr_stream_no == 0 && need_ints <= IDCACHE_MAX_INTS;
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_idcache_mu);
        idcache_entry & c = g_idcache[&ctx];
        if (c.ptr != nullptr && c.dev == ctx.device && c.epoch == ctx.graph_epoch &&
                c.ids == ids && c.ids_data == ids->data && c.nb0 == ids->nb[0] && c.nb1 == ids->nb[1] &&
                c.n_used == n_used && c.n_tok == n_tok && c.n_expert == n_expert &&
                c.bn == bn && c.max_tiles == max_tiles && c.stream == stream) {
            ai4k::idmap m{};
            m.hdr   = c.ptr;
            m.exps  = c.ptr + 2;
            m.off   = m.exps + n_expert;
            // rows[] was built with the first consumer's ne11 (gate_up: 1) and ne11 is not in the key: on a hit only exps/off/cols/tiles
            // are valid (the turbo kernels map cols -> src1 row themselves); never read rows[] from here.
            m.rows  = m.off + n_expert + 1;
            m.cols  = m.rows + n_pairs;
            m.tiles = m.cols + n_pairs;
            m.nb02  = nb02;
            return m;
        }
        if (c.ptr == nullptr || c.dev != ctx.device) {
            CUDA_CHECK(ggml_cuda_device_malloc((void **) &c.ptr, IDCACHE_MAX_INTS * sizeof(int), ctx.device));
            c.dev = ctx.device;
        }
        c.epoch     = ctx.graph_epoch;
        c.ids       = ids;
        c.ids_data  = ids->data;
        c.nb0       = ids->nb[0];
        c.nb1       = ids->nb[1];
        c.n_used    = n_used;
        c.n_tok     = n_tok;
        c.n_expert  = n_expert;
        c.bn        = bn;
        c.max_tiles = max_tiles;
        c.stream    = stream;
        return ai4k::launch_ids_helper(ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, ne11,
                                       n_expert, bn, max_tiles, nb02, c.ptr, stream, nullptr, 0, false, n_expert <= 1024);
    }

    fallback_buf.alloc(need_ints);
    return ai4k::launch_ids_helper(ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, ne11,
                                   n_expert, bn, max_tiles, nb02, fallback_buf.get(), stream,
                                   zero_y, M, false, n_expert <= 1024);
}

template <int LAY, int XSH>
static void dispatch_grouped_lay(
        const uint8_t * W, const int8_t * xq, const float * xs, float * Y,
        int M, int K, int n_used, int ne11, int bn, const ai4k::idmap & idm, int max_tiles, cudaStream_t stream) {
    const bool div3 = ((K / 128) % 3) == 0;
    if (bn == 8) {
        if (div3) launch_grouped<8,  LAY, XSH, 3, 8>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
        else      launch_grouped<8,  LAY, XSH, 4, 6>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
    } else if (bn == 16) {
        if (div3) launch_grouped<16, LAY, XSH, 3, 8>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
        else      launch_grouped<16, LAY, XSH, 4, 6>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
    } else {
        if (div3) launch_grouped<32, LAY, XSH, 3, 6>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
        else      launch_grouped<32, LAY, XSH, 4, 5>(W, xq, xs, Y, M, K, n_used, ne11, idm, max_tiles, stream);
    }
}

} // namespace ai4_exp_turbo

// Called from ~ggml_backend_cuda_context: frees the 64 KiB buffer and drops the entry, so a context re-created at the same
// address (graph_epoch restarts at 1) can never match stale routing.
void ggml_cuda_ai4_exp_turbo_free(const ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(ai4_exp_turbo::g_idcache_mu);
    auto it = ai4_exp_turbo::g_idcache.find(ctx);
    if (it == ai4_exp_turbo::g_idcache.end()) {
        return;
    }
    if (it->second.ptr != nullptr) {
        ggml_cuda_set_device(it->second.dev);
        cudaFree(it->second.ptr);
    }
    ai4_exp_turbo::g_idcache.erase(it);
}

bool ggml_cuda_ai4_exp_turbo_try(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * ids,
        const int8_t * xq,
        const float * xs,
        float * y_out,
        int M,
        int K,
        int n_tok,
        int n_used,
        int ne11,
        int n_expert,
        int xsh,
        bool has_y_into,
        bool skip_zero) {
    if (!ai4_exp_turbo::turbo_enabled()) {
        return false;
    }
    // Preserve N == 1 (rc's specialized 56.6 us single-token mmvr) and prefill N > max_n
    if (n_tok < 2 || n_tok > ai4_exp_turbo::turbo_max_n()) {
        return false;
    }
    const ggml_type t = src0->type;
    if (t != GGML_TYPE_AIT4_U && t != GGML_TYPE_AIT3_U && t != GGML_TYPE_AIT2_U) {
        return false;
    }
    if (M <= 0 || M % 16 != 0 || K <= 0 || K % 128 != 0 || n_used <= 0 || n_expert <= 0) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    if (!GGML_CUDA_CC_IS_NVIDIA(dev.cc) || dev.cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }

    const uint8_t * W = (const uint8_t *) src0->data;
    cudaStream_t stream = ctx.stream();
    const int64_t n_pairs = (int64_t) n_used * n_tok;

    // Regime 1: M = 2..8 AND n_pairs <= 64 -> ZERO ids_helper direct-grouped Split-K!
    if (n_tok <= 8 && n_pairs <= 64) {
        ai4k::idmap idm{};
        idm.nb02            = src0->nb[2];
        idm.direct_ids      = (const char *) ids->data;
        idm.direct_nb0      = ids->nb[0];
        idm.direct_nb1      = ids->nb[1];
        idm.direct_n_used   = n_used;
        idm.direct_ne11     = ne11;
        idm.direct_n_expert = n_expert;
        idm.direct_zero     = !has_y_into;

        if (t == GGML_TYPE_AIT4_U) {
            if (xsh) ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT4_U, 1>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
            else     ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT4_U, 0>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
        } else if (t == GGML_TYPE_AIT3_U) {
            if (xsh) ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT3_U, 1>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
            else     ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT3_U, 0>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
        } else {
            if (xsh) ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT2_U, 1>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
            else     ai4_exp_turbo::dispatch_direct_lay<ai4k::LAY_AIT2_U, 0>(W, xq, xs, y_out, M, K, n_used, n_tok, idm, stream);
        }
        return true;
    }

    // Regime 2: M = 9..32 (or M = 2..8 with n_pairs > 64) -> zero-static-SMEM grouped Split-K + cached ids_helper
    const int bn = (n_tok <= 8) ? 8 : (n_tok <= 16 ? 16 : 32);
    const int max_tiles = ai4k::ids_max_tiles(n_expert, n_pairs, bn, true);
    if (max_tiles <= 0 || max_tiles > 65535) {
        return false;
    }
    ggml_cuda_pool_alloc<int> fallback_routing(ctx.pool());
    float * zero_y = (skip_zero && !has_y_into) ? y_out : nullptr;
    const ai4k::idmap idm = ai4_exp_turbo::get_or_build_idmap(
            ctx, ids, n_used, n_tok, ne11, n_expert, bn, max_tiles, src0->nb[2], fallback_routing, zero_y, M);

    if (t == GGML_TYPE_AIT4_U) {
        if (xsh) ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT4_U, 1>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
        else     ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT4_U, 0>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
    } else if (t == GGML_TYPE_AIT3_U) {
        if (xsh) ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT3_U, 1>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
        else     ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT3_U, 0>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
    } else {
        if (xsh) ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT2_U, 1>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
        else     ai4_exp_turbo::dispatch_grouped_lay<ai4k::LAY_AIT2_U, 0>(W, xq, xs, y_out, M, K, n_used, ne11, bn, idm, max_tiles, stream);
    }
    return true;
}

#endif
