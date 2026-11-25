#pragma once
// AI+ cooperative 2-warp-per-slab BN=128 expert prefill GEMM for AIT4_U (SM 8.0 / 8.6):
//   gemm_s8_coop128<LAY, XSH, ID> (BM = 64 [M_SLABS = 4], BN = 128 [N_WARPS = 2], STAGES = 2, MINB = 2)
//   - one weight read per 128-route expert tile (the 64-route lowreg kernel reads it twice at >= 128 routes),
//   - the two warps of a slab each decode 2 of the 4 j-steps and swap them through a 64-thread named barrier
//     (bar.sync mw+1, 64), halving the per-warp decode ALU of gemm<128>,
//   - <= 128 regs, 2 CTAs/SM; same int32 IMMA sums and float accumulation order as gemm<128> (bitwise equal),
//   - warps with no routed column in their half skip the IMMAs (those columns are never stored), always: the
//     GGML_CUDA_AI4_TAIL_SKIP kill switch does not apply here (no output difference either way).
//   - SM 8.6 build: ID kernels REG 112, non-ID REG 128, STACK 0 / LOCAL 0 (no spills).
// Opt-in: GGML_CUDA_AI4_COOP128=1 (default 0). Only engages where gemm<128> would run for AIT4_U expert tiles,
// i.e. bn == 128: n_tok >= GGML_CUDA_AI4_BN128_MIN and the lowreg-ID path not taken (it forces bn 64 up to
// GGML_CUDA_AI4_LOWREG_ID_MAX_N = 1024 by default), and GGML_CUDA_AI4_ID_SHAPE unset.
// Dense (ID = false): opt-in GGML_CUDA_AI4_COOP128_DENSE=1 (default 0, independent of GGML_CUDA_AI4_COOP128), only where
// the dense gemm<128, 3, 4> would run for AIT4_U in ai4_mm_kernels (N >= GGML_CUDA_AI4_BN128_MIN, no prefill_v2 /
// Ampere-mmv route). It reads the PLAIN activation prep (row-major [N][K] int8, [N][K >> (7 + XSH)] scales), the only
// prep rate-4 types get (SWAP8 needs rate 2/3, XPERM rate 2), including a PREFILL-GLUE cached one.

#include "ai4-kern.cuh"
#include "ai4-ampere.cuh"

namespace ai4_coop {

static __device__ __forceinline__ void bar_sync_64(const int bar_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("bar.sync %0, 64;\n" :: "r"(bar_id) : "memory");
#else
    (void) bar_id;
    __syncthreads();
#endif
}

template <int LAY, int M_SLABS = 4, int BN = 128>
constexpr __host__ __device__ int coop128_stage_bytes() {
    // M_SLABS slabs of (ch_bytes + 64B scales) + BN * 128B XOR-swizzled Xq + BN * 4B Xs
    return M_SLABS * (ai4k::ch_bytes<LAY>() + 64) + BN * 128 + BN * 4;
}

template <int LAY, int STAGES = 2, int M_SLABS = 4, int BN = 128>
constexpr __host__ __device__ int coop128_smem_bytes() {
    // STAGES pipeline buffers + M_SLABS * 2 slots * 32 lanes * 16B cooperative A-fragment exchange (4,096 B)
    return STAGES * coop128_stage_bytes<LAY, M_SLABS, BN>() + M_SLABS * 2 * 32 * 16;
}

template <int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false, int STAGES = 2, int MINB = 2>
static __global__ void __launch_bounds__(256, MINB) gemm_s8_coop128(
        const uint8_t * __restrict__ W,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y,
        const int64_t ldy, const int M, int N, const int K,
        const ai4k::idmap idm) {
    static_assert(LAY == ai4k::LAY_AIT4_U || LAY == ai4k::LAY_AIT3_U || LAY == ai4k::LAY_AIT2_U, "unit layout");
    static_assert(STAGES == 2 || STAGES == 3, "STAGES 2 or 3");
    constexpr int M_SLABS = 4;
    constexpr int N_WARPS = 2;
    constexpr int BN      = 128;
    constexpr int WN      = BN / N_WARPS; // 64 cols/warp
    constexpr int NTI     = WN / 8;       // 8
    constexpr int THREADS = 256;
    constexpr int CHB     = ai4k::ch_bytes<LAY>();
    constexpr int CHP     = CHB / 16;
    constexpr int WSB     = M_SLABS * (CHB + 64);
    constexpr int SB      = coop128_stage_bytes<LAY, M_SLABS, BN>();

    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int srow[ID ? BN : 1];
    __shared__ int scol[ID ? BN : 1];

    const int tid   = threadIdx.x;
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int mw    = warp >> 1; // 0..3 slab within CTA
    const int nw    = warp & 1;  // 0..1 N-half within slab
    const int g     = lane >> 2; // 0..7
    const int t     = lane & 3;  // 0..3

    const int slab0 = (int) blockIdx.x * M_SLABS;
    const int nslab = M >> 4;
    int n0          = (int) blockIdx.y * BN;
    const int nk    = K >> 7;
    const int xsn   = K >> (7 + XSH);
    const int64_t sbytes = ai4k::slab_bytes_l<LAY>(K);

    if constexpr (ID) {
        if ((int) blockIdx.y >= idm.hdr[1]) {
            return;
        }
        const int a  = idm.tiles[2 * blockIdx.y];
        const int o  = idm.off[a];
        const int nt = idm.tiles[2 * blockIdx.y + 1];
        W += (int64_t) idm.exps[a] * idm.nb02;
        N  = idm.off[a + 1] - o - nt;
        n0 = 0;
        for (int i = tid; i < BN; i += THREADS) {
            srow[i] = (i < N) ? idm.rows[o + nt + i] : 0;
            scol[i] = (i < N) ? idm.cols[o + nt + i] : 0;
        }
        __syncthreads();
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) return srow[c]; else return n0 + c; };
    auto ycol = [&](int c) -> int64_t { if constexpr (ID) return scol[c]; else return n0 + c; };

    const ai4k::k5win k5a = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win k5b = ai4k::lay_win<LAY>(g + 8, t & 1);
    (void) k5a; (void) k5b;

    // Cooperative decoded-A exchange buffer at end of SMEM: [M_SLABS][2][32] uint4 (4,096 B)
    uint4 * acoop = (uint4 *) (smem + STAGES * SB) + mw * 64;
    const int my_bar = mw + 1; // dedicated 64-thread named barrier per slab (barriers 1..4)

    auto load_stage = [&](int kc, int slot) {
        uint8_t * st = smem + slot * SB;
        const int64_t coff = ai4k::chunk_off<LAY>(kc);
        const int64_t soff = ai4k::sc_off<LAY>(kc);

        // 1) Weights: 4 slabs * CHP x 16B (AIT4_U: 4 * 64 = 256 -> EXACTLY 1 cp.async16/thread!)
        constexpr int TOTAL_W16 = M_SLABS * CHP;
        constexpr int W_ITERS   = (TOTAL_W16 + THREADS - 1) / THREADS;
#pragma unroll
        for (int r = 0; r < W_ITERS; ++r) {
            const int i = tid + r * THREADS;
            if (TOTAL_W16 % THREADS == 0 || i < TOTAL_W16) {
                const int s = i / CHP;
                const int p = i % CHP;
                const bool ok = (slab0 + s) < nslab;
                const uint8_t * src = W + (ok ? ((int64_t) (slab0 + s) * sbytes + coff + p * 16) : 0);
                rq4::cp_async16(st + s * CHB + p * 16, src, ok);
            }
        }
        // 2) Weight scales: 4 slabs * 8 words = 32 x cp.async4
        if (tid < M_SLABS * 8) {
            const int s = tid >> 3;
            const int w = tid & 7;
            const bool ok = (slab0 + s) < nslab;
            const uint8_t * src = W + (ok ? ((int64_t) (slab0 + s) * sbytes + soff + w * 4) : 0);
            rq4::cp_async4(st + M_SLABS * CHB + s * 64 + w * 4, src, ok);
        }
        // 3) Activations: 128 cols * 8 x 16B = 1024 x 16B -> 4 iters/thread (XOR-swizzled!)
        uint8_t * sx = st + WSB;
        float * sxd  = (float *) (sx + BN * 128);
        constexpr int XPIECES = BN * 8; // 1024
#pragma unroll
        for (int r = 0; r < XPIECES / THREADS; ++r) {
            const int i = tid + r * THREADS;
            const int c = i >> 3;
            const int p = i & 7;
            const bool ok = (n0 + c) < N;
            const int xp = p ^ (c & 7);
            const int8_t * src = Xq + (ok ? xrow(c) : 0) * K + kc * 128 + p * 16;
            rq4::cp_async16(sx + c * 128 + xp * 16, src, ok);
        }
        // 4) Activation scales: 128 x cp.async4
        if (tid < BN) {
            const bool ok = (n0 + tid) < N;
            rq4::cp_async4(sxd + tid, Xs + (ok ? xrow(tid) : 0) * xsn + (kc >> XSH), ok);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s, s);
        rq4::cp_async_commit();
    }

    int cur_slot = 0;
    int nxt_slot = STAGES - 1;
    const int n_warp_cols = N - (n0 + nw * WN); // <= 0 means this warp has 0 active columns (still participates in decode!)

    for (int kc = 0; kc < nk; ++kc) {
        rq4::cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < nk) load_stage(kc + STAGES - 1, nxt_slot);
        rq4::cp_async_commit();

        const uint8_t  * st  = smem + cur_slot * SB;
        if constexpr (STAGES == 2) { cur_slot ^= 1; nxt_slot ^= 1; }
        else { if (++cur_slot == STAGES) cur_slot = 0; if (++nxt_slot == STAGES) nxt_slot = 0; }

        const uint32_t * tw  = (const uint32_t *) (st + mw * CHB);
        const half     * sd  = (const half *) (st + M_SLABS * CHB + mw * 64);
        const uint8_t  * sx  = st + WSB;
        const float    * sxd = (const float *) (sx + BN * 128);

        int p[NTI][4];
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            p[nt][0] = 0; p[nt][1] = 0; p[nt][2] = 0; p[nt][3] = 0;
        }

        // Cooperative 2-step decode: each warp decodes ONLY 2 of the 4 j-steps (HALVES PRMT & IDP ALU!)
        // and exchanges with its slab partner via dedicated 64-thread named barrier (no CTA-wide stall!).
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            const int j_own = jp * 2 + nw;
            uint32_t a_own[4];
            ai4k::a_regs_lay<LAY>(tw, j_own, g, t, k5a, k5b, a_own);
            acoop[nw * 32 + lane] = make_uint4(a_own[0], a_own[1], a_own[2], a_own[3]);
            bar_sync_64(my_bar);

            const uint4 a_par_v = acoop[(1 - nw) * 32 + lane];
            const uint32_t a_par[4] = { a_par_v.x, a_par_v.y, a_par_v.z, a_par_v.w };
            // Ensure both warps finished reading acoop before next jp overwrites it (only needed after jp == 0)
            if (jp == 0) {
                bar_sync_64(my_bar);
            }

            // Compute IMMAs for both j = 2*jp (j0) and j = 2*jp + 1 (j1) — skip if warp has 0 active MoE columns!
            if (!ID || n_warp_cols > 0) {
                const int j0 = jp * 2;
                const int j1 = jp * 2 + 1;
                const uint32_t aj0[4] = {
                    (nw == 0) ? a_own[0] : a_par[0], (nw == 0) ? a_own[1] : a_par[1],
                    (nw == 0) ? a_own[2] : a_par[2], (nw == 0) ? a_own[3] : a_par[3]
                };
                const uint32_t aj1[4] = {
                    (nw == 0) ? a_par[0] : a_own[0], (nw == 0) ? a_par[1] : a_own[1],
                    (nw == 0) ? a_par[2] : a_own[2], (nw == 0) ? a_par[3] : a_own[3]
                };

                const int ob0 = t * 32 + j0 * 8;
                const int p16_0 = ob0 >> 4, rem0 = ob0 & 15;
                const int ob1 = t * 32 + j1 * 8;
                const int p16_1 = ob1 >> 4, rem1 = ob1 & 15;

#pragma unroll
                for (int nt = 0; nt < NTI; ++nt) {
                    if constexpr (ID) {
                        if (nt * 8 >= n_warp_cols) {
                            continue;
                        }
                    }
                    const int c = nw * WN + nt * 8 + g;
                    const uint2 xv0 = *(const uint2 *) (sx + c * 128 + ((p16_0 ^ (c & 7)) << 4) + rem0);
                    rq4::mma_s8(p[nt], aj0, xv0.x, xv0.y);
                    const uint2 xv1 = *(const uint2 *) (sx + c * 128 + ((p16_1 ^ (c & 7)) << 4) + rem1);
                    rq4::mma_s8(p[nt], aj1, xv1.x, xv1.y);
                }
            }
        }

        if (!ID || n_warp_cols > 0) {
            const float d0 = __half2float(sd[g]);
            const float d1 = __half2float(sd[g + 8]);
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if constexpr (ID) {
                    if (nt * 8 >= n_warp_cols) {
                        continue;
                    }
                }
                const float dx0 = sxd[nw * WN + nt * 8 + 2 * t];
                const float dx1 = sxd[nw * WN + nt * 8 + 2 * t + 1];
                acc[nt][0] += rq4::magic_i2f(p[nt][0]) * (d0 * dx0);
                acc[nt][1] += rq4::magic_i2f(p[nt][1]) * (d0 * dx1);
                acc[nt][2] += rq4::magic_i2f(p[nt][2]) * (d1 * dx0);
                acc[nt][3] += rq4::magic_i2f(p[nt][3]) * (d1 * dx1);
            }
        }
    }
    rq4::cp_async_wait<0>();

    if (slab0 + mw >= nslab) {
        return;
    }
    const int r = (slab0 + mw) * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nw * WN + nt * 8 + 2 * t;
        if (n0 + c < N) {
            Y[ycol(c) * ldy + r]     = acc[nt][0];
            Y[ycol(c) * ldy + r + 8] = acc[nt][2];
        }
        if (n0 + c + 1 < N) {
            Y[ycol(c + 1) * ldy + r]     = acc[nt][1];
            Y[ycol(c + 1) * ldy + r + 8] = acc[nt][3];
        }
    }
}

template <int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false, int STAGES = 2, int MINB = 2>
static void launch_gemm_s8_coop128(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    constexpr int smem = coop128_smem_bytes<LAY, STAGES, 4, 128>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(
                gemm_s8_coop128<LAY, XSH, ID, STAGES, MINB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((M / 16 + 3) / 4, ID ? max_tiles : (N + 127) / 128);
    gemm_s8_coop128<LAY, XSH, ID, STAGES, MINB><<<grid, 256, smem, stream>>>(
        W, Xq, Xs, Y, ldy, M, N, K, idm);
}

// Public non-templated entrypoints (defined in ai4-coop128.cu)
bool env_coop128_enabled();
bool env_coop128_dense_enabled();

void dispatch_coop128_ait4(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, int xsh, bool is_id,
        const ai4k::idmap & idm, int max_tiles, cudaStream_t stream);

} // namespace ai4_coop
