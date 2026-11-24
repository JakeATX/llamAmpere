// Standalone CUDA translation unit for AIT5_U Cooperative 2-Warp-per-Slab BN=128 MoE Prefill GEMM.
// ONLY instantiates LAY_AIT5_U (XSH = 0 and XSH = 1, ID = true) — ZERO unused dead kernels!

#include "ai5-coop128.cuh"

#include <atomic>
#include <cstdlib>

namespace {
std::atomic<uint64_t> g_ai5_coop128_fires{0};
}

extern "C" GGML_BACKEND_API uint64_t ggml_cuda_ai5_coop128_fire_count(void) {
    return g_ai5_coop128_fires.load(std::memory_order_relaxed);
}

extern "C" GGML_BACKEND_API void ggml_cuda_ai5_coop128_reset_fire_count(void) {
    g_ai5_coop128_fires.store(0, std::memory_order_relaxed);
}

namespace ai5_coop {

bool env_coop128_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AI5_COOP128");
        return e != nullptr && atoi(e) != 0; // default OFF (opt-in)
    }();
    return on;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ void bar_sync_64(const int bar_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("bar.sync %0, 64;\n" :: "r"(bar_id) : "memory");
#else
    (void) bar_id;
    __syncthreads();
#endif
}

constexpr int M_SLABS = 4;
constexpr int N_WARPS = 2;
constexpr int BN      = 128;
constexpr int WN      = BN / N_WARPS; // 64 cols/warp
constexpr int NTI     = WN / 8;       // 8
constexpr int THREADS = 256;
constexpr int CHB     = ai4k::ch_bytes<ai4k::LAY_AIT5_U>(); // 1280 B
constexpr int CHP     = CHB / 16;                           // 80 x 16B
constexpr int SC_SLOT = 32;                                 // 16 x fp16 = 32 B (16B aligned)
constexpr int WSB     = M_SLABS * (CHB + SC_SLOT);          // 5,248 B
constexpr int SB      = WSB + BN * 128 + BN * 4;            // 22,144 B
constexpr int ACOOP_B = M_SLABS * 2 * 32 * 16;              // 4,096 B

template <int STAGES>
constexpr __host__ __device__ int coop128_smem_bytes() {
    return STAGES * SB + ACOOP_B; // STAGES=2 -> 48,384 B (+ 1,024 B static + 1,024 B reserved = 50,432 B -> 2 CTAs/SM!)
}

static_assert(CHB == 1280 && WSB % 16 == 0 && SB % 16 == 0 && coop128_smem_bytes<2>() == 48384, "ai5-coop128 SMEM geometry");

template <int XSH, int STAGES = 2, int MINB = 2>
static __global__ void __launch_bounds__(THREADS, MINB) gemm_s8_coop128_ait5_id(
        const uint8_t * __restrict__ W,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y,
        const int64_t ldy, const int M, const int K,
        const ai4k::idmap idm) {
    constexpr int LAY = ai4k::LAY_AIT5_U;

    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int srow[BN];
    __shared__ int scol[BN];

    const int tid   = threadIdx.x;
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int mw    = warp >> 1; // 0..3 slab within CTA
    const int nw    = warp & 1;  // 0..1 N-half within slab
    const int g     = lane >> 2; // 0..7
    const int t     = lane & 3;  // 0..3

    if ((int) blockIdx.y >= idm.hdr[1]) {
        return;
    }

    const int slab0 = (int) blockIdx.x * M_SLABS;
    const int nslab = M >> 4;
    const int nk    = K >> 7;
    const int xsn   = K >> (7 + XSH);
    const int64_t sbytes = ai4k::slab_bytes_l<LAY>(K);

    const int a  = idm.tiles[2 * blockIdx.y];
    const int o  = idm.off[a];
    const int nt_off = idm.tiles[2 * blockIdx.y + 1];
    W += (int64_t) idm.exps[a] * idm.nb02;
    const int N  = idm.off[a + 1] - o - nt_off;
    for (int i = tid; i < BN; i += THREADS) {
        srow[i] = (i < N) ? idm.rows[o + nt_off + i] : 0;
        scol[i] = (i < N) ? idm.cols[o + nt_off + i] : 0;
    }
    __syncthreads();

    const ai4k::k5win k5a = ai4k::k5_win(g,     t & 1);
    const ai4k::k5win k5b = ai4k::k5_win(g + 8, t & 1);

    uint4 * acoop = (uint4 *) (smem + STAGES * SB) + mw * 64;
    const int my_bar = mw + 1; // dedicated 64-thread named barrier per slab (barriers 1..4)

    auto load_stage = [&](int kc, int slot) {
        uint8_t * st = smem + slot * SB;
        const int64_t coff = ai4k::chunk_off<LAY>(kc);
        const int64_t soff = ai4k::sc_off<LAY>(kc);

        // 1) Weights: 4 slabs * 80 x 16B = 320 x 16B (2 iters/thread)
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
        // 2) Weight scales: 4 slabs * 8 words (32 B each) = 32 x cp.async4
        if (tid < M_SLABS * 8) {
            const int s = tid >> 3;
            const int w = tid & 7;
            const bool ok = (slab0 + s) < nslab;
            const uint8_t * src = W + (ok ? ((int64_t) (slab0 + s) * sbytes + soff + w * 4) : 0);
            rq4::cp_async4(st + M_SLABS * CHB + s * SC_SLOT + w * 4, src, ok);
        }
        // 3) Activations: 128 cols * 8 x 16B = 1024 x 16B -> 4 iters/thread (XOR-swizzled)
        uint8_t * sx = st + WSB;
        float * sxd  = (float *) (sx + BN * 128);
        constexpr int XPIECES = BN * 8;
#pragma unroll
        for (int r = 0; r < XPIECES / THREADS; ++r) {
            const int i = tid + r * THREADS;
            const int c = i >> 3;
            const int p = i & 7;
            const bool ok = c < N;
            const int xp = p ^ (c & 7);
            const int8_t * src = Xq + (ok ? (int64_t) srow[c] : 0) * K + kc * 128 + p * 16;
            rq4::cp_async16(sx + c * 128 + xp * 16, src, ok);
        }
        // 4) Activation scales: 128 x cp.async4
        if (tid < BN) {
            const bool ok = tid < N;
            rq4::cp_async4(sxd + tid, Xs + (ok ? (int64_t) srow[tid] : 0) * xsn + (kc >> XSH), ok);
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
    const int  n_warp_cols = N - nw * WN;
    const bool warp_active = !idm.tail_skip || (n_warp_cols > 0);

    for (int kc = 0; kc < nk; ++kc) {
        rq4::cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < nk) load_stage(kc + STAGES - 1, nxt_slot);
        rq4::cp_async_commit();

        const uint8_t  * st  = smem + cur_slot * SB;
        if constexpr (STAGES == 2) { cur_slot ^= 1; nxt_slot ^= 1; }
        else { if (++cur_slot == STAGES) cur_slot = 0; if (++nxt_slot == STAGES) nxt_slot = 0; }

        const uint32_t * tw  = (const uint32_t *) (st + mw * CHB);
        const half     * sd  = (const half *) (st + M_SLABS * CHB + mw * SC_SLOT);
        const uint8_t  * sx  = st + WSB;
        const float    * sxd = (const float *) (sx + BN * 128);

        int p[NTI][4];
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            p[nt][0] = 0; p[nt][1] = 0; p[nt][2] = 0; p[nt][3] = 0;
        }

        // Cooperative 2-step decode: both warps of each slab participate unconditionally in bar_sync_64 (deadlock-free!)
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            const int j_own = jp * 2 + nw;
            uint32_t a_own[4];
            ai4k::a_regs_smem_k5<LAY>(tw, j_own, t, k5a, k5b, a_own);
            acoop[nw * 32 + lane] = make_uint4(a_own[0], a_own[1], a_own[2], a_own[3]);
            bar_sync_64(my_bar);

            const uint4 a_par_v = acoop[(1 - nw) * 32 + lane];
            const uint32_t a_par[4] = { a_par_v.x, a_par_v.y, a_par_v.z, a_par_v.w };
            if (jp == 0) {
                bar_sync_64(my_bar);
            }

            if (warp_active) {
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
                    if (idm.tail_skip && nt * 8 >= n_warp_cols) {
                        continue;
                    }
                    const int c = nw * WN + nt * 8 + g;
                    const uint2 xv0 = *(const uint2 *) (sx + c * 128 + ((p16_0 ^ (c & 7)) << 4) + rem0);
                    rq4::mma_s8(p[nt], aj0, xv0.x, xv0.y);
                    const uint2 xv1 = *(const uint2 *) (sx + c * 128 + ((p16_1 ^ (c & 7)) << 4) + rem1);
                    rq4::mma_s8(p[nt], aj1, xv1.x, xv1.y);
                }
            }
        }

        if (warp_active) {
            const float d0 = __half2float(sd[g]);
            const float d1 = __half2float(sd[g + 8]);
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if (idm.tail_skip && nt * 8 >= n_warp_cols) {
                    continue;
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
        if (c < N) {
            Y[(int64_t) scol[c] * ldy + r]     = acc[nt][0];
            Y[(int64_t) scol[c] * ldy + r + 8] = acc[nt][2];
        }
        if (c + 1 < N) {
            Y[(int64_t) scol[c + 1] * ldy + r]     = acc[nt][1];
            Y[(int64_t) scol[c + 1] * ldy + r + 8] = acc[nt][3];
        }
    }
}

template <int XSH>
static void launch_coop128_ait5_id(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int K,
        const ai4k::idmap & idm, int max_tiles, cudaStream_t stream) {
    constexpr int smem = coop128_smem_bytes<2>();
    // dynamic (48,384 B) + static (1,024 B) = 49,408 B > 48 KiB default opt-in threshold: ALWAYS set max dynamic SMEM!
    static bool attr_set[64] = {};
    int dev = 0;
    CUDA_CHECK_AI4K(cudaGetDevice(&dev));
    if (!attr_set[dev & 63]) {
        CUDA_CHECK_AI4K(cudaFuncSetAttribute(
            gemm_s8_coop128_ait5_id<XSH, 2, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev & 63] = true;
    }
    const dim3 grid((M / 16 + M_SLABS - 1) / M_SLABS, max_tiles);
    gemm_s8_coop128_ait5_id<XSH, 2, 2><<<grid, THREADS, smem, stream>>>(
        W, Xq, Xs, Y, ldy, M, K, idm);
}

void dispatch_coop128_ait5_id(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, int xsh,
        const ai4k::idmap & idm, int max_tiles, cudaStream_t stream) {
    (void) N;
    if (xsh) {
        launch_coop128_ait5_id<1>(W, Xq, Xs, Y, ldy, M, K, idm, max_tiles, stream);
    } else {
        launch_coop128_ait5_id<0>(W, Xq, Xs, Y, ldy, M, K, idm, max_tiles, stream);
    }
    g_ai5_coop128_fires.fetch_add(1, std::memory_order_relaxed);
}

#else

void dispatch_coop128_ait5_id(
        const uint8_t *, const int8_t *, const float *, float *, int64_t,
        int, int, int, int, const ai4k::idmap &, int, cudaStream_t) {}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

} // namespace ai5_coop
