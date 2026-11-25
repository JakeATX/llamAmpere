// [TURBO-IDS] AI+ expert MUL_MAT_ID with the turbo split-K matvec and direct routing ids (opt-in, default off).
// Own translation unit: nothing here is instantiated or reachable unless GGML_CUDA_AI4_TURBO_IDS=n (n >= 1) is set.
//
// The kernel body is the BN = 8, WARPS = 4 A_UPFRONT path of ai4_turbo::ampere_mmv_turbo (MODE_AIT4_S8 with the
// activation prefetch, MODE_AIT3_S8_WROW with the PERM B-load on the standard activation layout). Only the ID prologue
// differs: instead of (active expert, token tile) from ai4k::ids_helper, CTA (slab, u, tok) reads ids[u, tok] itself
// (ai4-turbo-ids-route.h) and computes the single column of that pair. Same int8 products and per-chunk scale order
// as the turbo ID kernel at one column; the four warps' split-K partials are summed in warp order 0, 1, 2, 3.

#include "ai4-turbo-ids.cuh"

#include <atomic>
#include <cstdlib>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_ai4_turbo_ids_try(
        ggml_type, const uint8_t *, int64_t, const ggml_tensor *, const int8_t *, const float *, float *,
        int, int, int, int, int, int, int, bool, cudaStream_t) {
    return false;
}

#else

#include "ai4-turbo.cuh"
#include "ai4-turbo-ids-route.h"

namespace ai4_turbo_ids {

using ai4_turbo::MODE_AIT4_S8;
using ai4_turbo::MODE_AIT3_S8_WROW;
using rq4::cp_async16;
using rq4::cp_async4;
using rq4::cp_async_commit;
using rq4::cp_async_wait;
using rq4::mma_s8;
using rq4::magic_i2f;

constexpr int BN    = 8;   // one MMA column tile; only column 0 (the pair) is live
constexpr int WARPS = 4;
constexpr int MINB  = 6;

template <int MODE>
constexpr int ids_smem_bytes() {
    return ai4_turbo::turbo_mmv_smem_bytes<BN, MODE, WARPS, 2, 1>();
}

template <int MODE, int XSH>
static __global__ void __launch_bounds__(WARPS * 32, MINB) turbo_ids_mmv(
        const uint8_t * __restrict__ W, int64_t nb02,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y, int64_t ldy, int K,
        const char * __restrict__ ids, int64_t ids_nb0, int64_t ids_nb1,
        int n_used, int ne11, int n_expert, bool zero_skipped) {
    static_assert(MODE == MODE_AIT4_S8 || MODE == MODE_AIT3_S8_WROW, "s8 unit modes only");
    constexpr int CHB = ai4_turbo::turbo_ch_bytes<MODE>();
    constexpr int U_B = CHB + 32;
    constexpr int ST  = ai4_turbo::turbo_mmv_stage_bytes<BN, MODE>();
    constexpr bool USE_PERM_B = MODE == MODE_AIT3_S8_WROW;   // whole-row tiles on the standard (un-permuted) Xq
    constexpr bool PREFETCH_X = !USE_PERM_B;                  // turbo: s8 BN == 8 prefetch only without PERM
    static_assert(U_B == ai4k::u_bytes<MODE == MODE_AIT4_S8 ? ai4k::LAY_AIT4_U : ai4k::LAY_AIT3_U>(), "unit bytes");

    extern __shared__ __align__(16) uint8_t smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int slab = blockIdx.x;

    // direct ids prologue (block-uniform: depends on blockIdx only, so the early exit precedes every barrier)
    const ai4_tids_pair pr = ai4_tids_route(ids, ids_nb0, ids_nb1, (int) blockIdx.y, (int) blockIdx.z,
                                            n_used, ne11, n_expert);
    if (!pr.live) {
        if (zero_skipped && tid < 16) {
            Y[pr.ycol * ldy + slab * 16 + tid] = 0.0f;
        }
        return;
    }
    W += (int64_t) pr.expert * nb02;

    const int nk  = K / 128;
    const int xsn = K >> (7 + XSH);
    const uint8_t * ws = W + (int64_t) slab * (int64_t) nk * U_B;
    uint8_t * ring = smem + warp * 2 * ST;
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    const ai4_turbo::k3row k3_ra = ai4_turbo::k3_row_win(g);
    const ai4_turbo::k3row k3_rb = ai4_turbo::k3_row_win(g + 8);
    (void) k3_ra; (void) k3_rb;

    // MMA column g holds the pair when g == 0; the other seven columns are zero
    const bool live_c = g == 0;
    const int64_t xr = pr.xrow;

    auto load_stage = [&](int i, int slot) {
        const int kc = warp + i * WARPS;
        uint8_t * st = ring + slot * ST;
        const uint8_t * u = ws + (int64_t) kc * U_B;
        const uint8_t * tiles = u + 32;
        constexpr int CHP = CHB / 16;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            cp_async4(st + CHB + lane * 4, u + lane * 4, true);
        }
        if (lane < BN) {
            const bool live = lane == 0;
            const float * src = live ? (Xs + xr * xsn + (kc >> XSH)) : Xs;
            cp_async4(st + CHB + 64 + lane * 4, src, live);   // dead columns zero-filled
        }
    };

    uint4 nx[2];
    auto prefetch_x = [&](int i) {
        if constexpr (PREFETCH_X) {
            const int kc = warp + i * WARPS;
            if (i < nmy && live_c) {
                const uint4 * p = (const uint4 *) ((const uint32_t *) Xq + (xr * nk + kc) * 32 + t * 8);
                nx[0] = __ldg(p);
                nx[1] = __ldg(p + 1);
            } else {
                nx[0] = make_uint4(0, 0, 0, 0);
                nx[1] = make_uint4(0, 0, 0, 0);
            }
        }
    };

    auto decode_a_s8 = [&](const uint32_t * __restrict__ tw, int j, uint32_t (&a)[4]) {
        if constexpr (MODE == MODE_AIT4_S8) {
            ai4k::a_regs_smem<AIT_MUL_4>(tw, j, g, t, a);
        } else {
            const uint32_t * tile = tw + (4 * j + t) * 12;
            ai4_turbo::levels8_k3_from_3words<AIT_MUL_3>(tile, k3_ra, a[0], a[2]);
            ai4_turbo::levels8_k3_from_3words<AIT_MUL_3>(tile, k3_rb, a[1], a[3]);
        }
    };

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (nmy > 0) load_stage(0, 0);
    cp_async_commit();
    prefetch_x(0);

    for (int i = 0; i < nmy; ++i) {
        const int slot = i & 1;
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) load_stage(i + 1, 1 - slot);
        cp_async_commit();

        const int kc = warp + i * WARPS;
        const uint8_t * st = ring + slot * ST;
        const uint32_t * tw = (const uint32_t *) st;
        const half * sd = (const half *) (st + CHB);
        const float * dx = (const float *) (st + CHB + 64);

        uint32_t xv[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        if constexpr (PREFETCH_X) {
            xv[0] = nx[0].x; xv[1] = nx[0].y; xv[2] = nx[0].z; xv[3] = nx[0].w;
            xv[4] = nx[1].x; xv[5] = nx[1].y; xv[6] = nx[1].z; xv[7] = nx[1].w;
            prefetch_x(i + 1);
        } else if (live_c) {
            // exact PERM B-fragment remap for whole-row tiles (4j + t) on standard un-reordered Xq
            const int8_t * x = Xq + xr * K + kc * 128 + 64 * (t & 1) + 4 * (t >> 1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xv[2 * j]     = __ldg((const uint32_t *) (x + 8 * j));
                xv[2 * j + 1] = __ldg((const uint32_t *) (x + 32 + 8 * j));
            }
        }

        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);

        uint32_t a[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) decode_a_s8(tw, j, a[j]);

        int p[4] = { 0, 0, 0, 0 };
        int q[4] = { 0, 0, 0, 0 };
        mma_s8(p, a[0], xv[0], xv[1]);
        mma_s8(p, a[1], xv[2], xv[3]);
        mma_s8(q, a[2], xv[4], xv[5]);
        mma_s8(q, a[3], xv[6], xv[7]);
        const float dx0 = dx[2 * t];
        const float dx1 = dx[2 * t + 1];
        acc[0] += magic_i2f(p[0] + q[0]) * (d0 * dx0);
        acc[1] += magic_i2f(p[1] + q[1]) * (d0 * dx1);
        acc[2] += magic_i2f(p[2] + q[2]) * (d1 * dx0);
        acc[3] += magic_i2f(p[3] + q[3]) * (d1 * dx1);
    }
    cp_async_wait<0>();
    // zero static SMEM reduction: the barrier makes the cp.async ring dead before it is reused
    __syncthreads();
    float * red = (float *) smem;   // [WARPS - 1][4][32]
    if (warp > 0) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            red[((warp - 1) * 4 + e) * 32 + lane] = acc[e];
        }
    }
    __syncthreads();
    if (warp == 0 && t == 0) {   // column 2t == 0 is the pair; acc[0] = row g, acc[2] = row g + 8
#pragma unroll
        for (int w = 0; w < WARPS - 1; ++w) {
            acc[0] += red[(w * 4 + 0) * 32 + lane];
            acc[2] += red[(w * 4 + 2) * 32 + lane];
        }
        const int r = slab * 16 + g;
        Y[pr.ycol * ldy + r]     = acc[0];
        Y[pr.ycol * ldy + r + 8] = acc[2];
    }
}

template <int MODE, int XSH>
static void launch(const uint8_t * W, int64_t nb02, const int8_t * xq, const float * xs, float * Y, int M, int K,
                   const ggml_tensor * ids, int n_used, int n_tok, int ne11, int n_expert, bool zero_skipped,
                   cudaStream_t stream) {
    constexpr int smem = ids_smem_bytes<MODE>();
    static_assert(smem <= 48 * 1024, "no dynamic SMEM opt-in needed");
    const dim3 grid(M / 16, n_used, n_tok);
    turbo_ids_mmv<MODE, XSH><<<grid, WARPS * 32, smem, stream>>>(
        W, nb02, xq, xs, Y, (int64_t) M, K, (const char *) ids->data, ids->nb[0], ids->nb[1],
        n_used, ne11, n_expert, zero_skipped);
}

static int max_n() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_AI4_TURBO_IDS");
        const int n = e ? atoi(e) : 0;
        return n < 0 ? 0 : (n > 8 ? 8 : n);
    }();
    return v;
}

} // namespace ai4_turbo_ids

bool ggml_cuda_ai4_turbo_ids_try(
        ggml_type type, const uint8_t * W, int64_t nb02, const ggml_tensor * ids, const int8_t * xq, const float * xs,
        float * y_out, int M, int K, int n_tok, int n_used, int ne11, int n_expert, int xsh, bool has_y_into,
        cudaStream_t stream) {
    const int max_n = ai4_turbo_ids::max_n();
    if (max_n == 0 || n_tok < 1 || n_tok > max_n) {
        return false;
    }
    if (type != GGML_TYPE_AIT4_U && type != GGML_TYPE_AIT3_U) {
        return false;
    }
    if (M <= 0 || M % 16 != 0 || K <= 0 || K % 128 != 0 || n_used <= 0 || n_used > 65535 ||
            n_expert <= 0 || (ne11 != 1 && ne11 != n_used)) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != (int64_t) sizeof(int32_t)) {
        return false;
    }
    // 16-B cp.async on unit tiles and 16-B LDG on activation rows (K % 128 == 0 keeps every row aligned)
    if (((uintptr_t) W & 15) != 0 || nb02 % 16 != 0 || ((uintptr_t) xq & 15) != 0 || ((uintptr_t) xs & 3) != 0) {
        return false;
    }
    static std::atomic<bool> logged[2];   // once per unit type, so a test run shows both kernels fired
    if (!logged[type == GGML_TYPE_AIT4_U ? 0 : 1].exchange(true)) {
        GGML_LOG_INFO("%s: direct-ids turbo expert matvec on (GGML_CUDA_AI4_TURBO_IDS=%d): first call %s M %d K %d n_tok %d\n",
                      __func__, max_n, ggml_type_name(type), M, K, n_tok);
    }
    const bool zero_skipped = !has_y_into;
    if (type == GGML_TYPE_AIT4_U) {
        if (xsh) ai4_turbo_ids::launch<ai4_turbo::MODE_AIT4_S8, 1>(W, nb02, xq, xs, y_out, M, K, ids, n_used, n_tok, ne11, n_expert, zero_skipped, stream);
        else     ai4_turbo_ids::launch<ai4_turbo::MODE_AIT4_S8, 0>(W, nb02, xq, xs, y_out, M, K, ids, n_used, n_tok, ne11, n_expert, zero_skipped, stream);
    } else {
        if (xsh) ai4_turbo_ids::launch<ai4_turbo::MODE_AIT3_S8_WROW, 1>(W, nb02, xq, xs, y_out, M, K, ids, n_used, n_tok, ne11, n_expert, zero_skipped, stream);
        else     ai4_turbo_ids::launch<ai4_turbo::MODE_AIT3_S8_WROW, 0>(W, nb02, xq, xs, y_out, M, K, ids, n_used, n_tok, ne11, n_expert, zero_skipped, stream);
    }
    return true;
}

#endif
