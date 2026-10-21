#pragma once

#include "ai4-kern.cuh"

namespace ai4k {

template <int LAY>
static __global__ void __launch_bounds__(256) prefill_decode(
        const uint8_t * __restrict__ W, int8_t * __restrict__ decoded, int M, int K) {
    static_assert(LAY == LAY_SB || LAY == LAY_U, "ai4 prefill layout");
    const int lane = threadIdx.x & 31;
    const int64_t chunk = (int64_t) blockIdx.x*8 + (threadIdx.x >> 5);
    const int nk = K/128;
    if (chunk >= (int64_t) (M/16)*nk) {
        return;
    }
    const int slab = chunk/nk;
    const int kc = chunk % nk;
    const uint32_t * tw = (const uint32_t *) (W + slab*slab_bytes_l<LAY>(K) + chunk_off<LAY>(kc));
    // Store [chunk][IMMA step][lane][A register] for direct vector loads in GEMM.
    uint4 * out = (uint4 *) decoded + chunk*128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t a[4];
        a_regs_smem(tw, j, lane >> 2, lane & 3, a);
        out[j*32 + lane] = make_uint4(a[0], a[1], a[2], a[3]);
    }
}

// Decode each weight once across all N tiles. The caller owns M*K scratch bytes until this stream completes.
template <int LAY = LAY_SB, int XSH = 1>
static void launch_prefill_v2(const uint8_t * W, const int8_t * Xq, const float * Xs, int8_t * decoded,
                              float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream) {
    const int64_t chunks = (int64_t) (M/16)*(K/128);
    prefill_decode<LAY><<<(chunks + 7)/8, 256, 0, stream>>>(W, decoded, M, K);
    if (M <= 1024 && N < 2048) {
        launch_gemm<128, 2, 2, LAY, XSH, false, 1, 4, true>(W, Xq, Xs, Y, ldy, M, N, K, stream, idmap{}, 0, decoded);
    } else {
        launch_gemm<256, 2, 2, LAY, XSH, false, 1, 8, true>(W, Xq, Xs, Y, ldy, M, N, K, stream, idmap{}, 0, decoded);
    }
}

template <int BN, int LAY>
constexpr __host__ __device__ int ampere_mmv_stage_bytes() {
    return ch_bytes<LAY>() + 64 + BN*4;
}

template <int BN, int LAY, int WARPS>
constexpr __host__ __device__ int ampere_mmv_smem_bytes() {
    return WARPS*2*ampere_mmv_stage_bytes<BN, LAY>();
}

// One slab per block keeps narrow batches distributed across the SMs. Each decoded A fragment serves all token groups.
template <int BN, int LAY, int XSH, bool ID, int WARPS, int KT = 0, bool PERM = false>
static __global__ void __launch_bounds__(WARPS*32, 1) ampere_mmv(
        const uint8_t * __restrict__ W, const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int Kr, const idmap idm) {
    static_assert(BN == 8 || BN == 16 || BN == 32 || BN == 64, "ai4 narrow width");
    static_assert(WARPS == 2 || WARPS == 4 || WARPS == 8, "ai4 narrow warps");
    static_assert(LAY == LAY_SB || LAY == LAY_U || lay_rate<LAY>() == 2 || lay_rate<LAY>() == 3, "ai4 narrow layout");
    static_assert(!PERM || lay_rate<LAY>() == 2, "whole-row rate-2 decode");
    const int K = KT > 0 ? KT : Kr;
    constexpr int NTI = BN/8;
    constexpr int CHB = ch_bytes<LAY>();
    constexpr int ST = ampere_mmv_stage_bytes<BN, LAY>();
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ float red[WARPS - 1][NTI][4][32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const k5win wa = lay_win<LAY>(g, t & 1);
    const k5win wb = lay_win<LAY>(g + 8, t & 1);
    const int slab = blockIdx.x;
    const int nk = K/128;
    const int xsn = K >> (7 + XSH);
    const int * rows = nullptr;
    const int * cols = nullptr;
    int n0 = blockIdx.y*BN;
    if constexpr (ID) {
        if ((int) blockIdx.y >= idm.hdr[1]) {
            return;
        }
        const int active = idm.tiles[2*blockIdx.y];
        const int off = idm.off[active] + idm.tiles[2*blockIdx.y + 1];
        N = min(BN, idm.off[active + 1] - off);
        W += (int64_t) idm.exps[active]*idm.nb02;
        rows = idm.rows + off;
        cols = idm.cols + off;
        n0 = 0;
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) { return rows[c]; } else { return n0 + c; } };
    auto ycol = [&](int c) -> int64_t { if constexpr (ID) { return cols[c]; } else { return n0 + c; } };
    const uint8_t * ws = W + (int64_t) slab*slab_bytes_l<LAY>(K);
    uint8_t * ring = smem + warp*2*ST;
    const int nmy = KT > 0 && (KT/128) % WARPS == 0 ? KT/(128*WARPS) : (nk - warp + WARPS - 1)/WARPS;

    auto load = [&](int i, int slot) {
        const int kc = warp + i*WARPS;
        uint8_t * st = ring + slot*ST;
        const uint8_t * tiles = ws + chunk_off<LAY>(kc);
#pragma unroll
        for (int r = 0; r < (CHB/16 + 31)/32; ++r) {
            if (CHB/16 % 32 == 0 || lane + 32*r < CHB/16) {
                cp_async16(st + r*512 + lane*16, tiles + r*512 + lane*16, true);
            }
        }
        if constexpr (LAY == LAY_SB) {
            if (lane < 16) {
                cp_async4(st + CHB + lane*4, ws + sc_off<LAY>(kc) + lane*16 + (kc & 6)*2, true);
            }
        } else {
            if (lane < 8) {
                cp_async4(st + CHB + lane*4, ws + sc_off<LAY>(kc) + lane*4, true);
            }
        }
        for (int c = lane; c < BN; c += 32) {
            const bool live = n0 + c < N;
            const float * src = live ? Xs + xrow(c)*xsn + (kc >> XSH) : Xs;
            cp_async4(st + CHB + 64 + c*4, src, live);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc[nt][e] = 0.0f;
        }
    }
    if (nmy > 0) {
        load(0, 0);
    }
    cp_async_commit();
    auto step = [&](int i, int slot) {
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            load(i + 1, 1 - slot);
        }
        cp_async_commit();

        const int kc = warp + i*WARPS;
        const uint8_t * st = ring + slot*ST;
        const uint32_t * tw = (const uint32_t *) st;
        const half * sd = (const half *) (st + CHB);
        const float * dx = (const float *) (st + CHB + 64);
        const float d0 = __half2float(LAY == LAY_SB ? sd[2*g + (kc & 1)] : sd[g]);
        const float d1 = __half2float(LAY == LAY_SB ? sd[2*(g + 8) + (kc & 1)] : sd[g + 8]);
        uint32_t a[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if constexpr (PERM) {
                const k5win pa = kr_win<2>(g, 1), pb = kr_win<2>(g + 8, 1);
                const uint32_t * tile = tw + (4*j + t)*8;
                levels8_k2(__funnelshift_l(tile[pa.w1], tile[pa.w0], pa.sh), a[j][0], a[j][2]);
                levels8_k2(__funnelshift_l(tile[pb.w1], tile[pb.w0], pb.sh), a[j][1], a[j][3]);
            } else {
                a_regs_lay<LAY>(tw, j, g, t, wa, wb, a[j]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (n0 + nt*8 >= N) {
                continue;
            }
            const int c = nt*8 + g;
            uint32_t xv[8] = {};
            if (n0 + c < N) {
                const int8_t * x = Xq + xrow(c)*K + kc*128;
                if constexpr (PERM) {
                    x += 64*(t & 1) + 4*(t >> 1);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xv[2*j] = __ldg((const uint32_t *) (x + 8*j));
                        xv[2*j + 1] = __ldg((const uint32_t *) (x + 32 + 8*j));
                    }
                } else {
                    const uint4 x0 = __ldg((const uint4 *) (x + t*32));
                    const uint4 x1 = __ldg((const uint4 *) (x + t*32) + 1);
                    xv[0] = x0.x; xv[1] = x0.y; xv[2] = x0.z; xv[3] = x0.w;
                    xv[4] = x1.x; xv[5] = x1.y; xv[6] = x1.z; xv[7] = x1.w;
                }
            }
            int p[4] = {0, 0, 0, 0};
            int q[4] = {0, 0, 0, 0};
            mma_s8(p, a[0], xv[0], xv[1]);
            mma_s8(p, a[1], xv[2], xv[3]);
            mma_s8(q, a[2], xv[4], xv[5]);
            mma_s8(q, a[3], xv[6], xv[7]);
            const float dx0 = dx[nt*8 + 2*t];
            const float dx1 = dx[nt*8 + 2*t + 1];
            acc[nt][0] += magic_i2f(p[0] + q[0]) * (d0*dx0);
            acc[nt][1] += magic_i2f(p[1] + q[1]) * (d0*dx1);
            acc[nt][2] += magic_i2f(p[2] + q[2]) * (d1*dx0);
            acc[nt][3] += magic_i2f(p[3] + q[3]) * (d1*dx1);
        }
    };
    if constexpr (KT > 0) {
        for (int i0 = 0; i0 < (KT/128 + WARPS - 1)/WARPS; i0 += 2) {
#pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                if (i0 + slot < nmy) step(i0 + slot, slot);
            }
        }
    } else {
        for (int i = 0; i < nmy; ++i) step(i, i & 1);
    }
    cp_async_wait<0>();
    if (warp > 0) {
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                red[warp - 1][nt][e][lane] = acc[nt][e];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
            for (int w = 0; w < WARPS - 1; ++w) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[nt][e] += red[w][nt][e][lane];
                }
            }
            const int r = slab*16 + g;
            const int c = nt*8 + 2*t;
            if (n0 + c < N) {
                Y[ycol(c)*ldy + r] = acc[nt][0];
                Y[ycol(c)*ldy + r + 8] = acc[nt][2];
            }
            if (n0 + c + 1 < N) {
                Y[ycol(c + 1)*ldy + r] = acc[nt][1];
                Y[ycol(c + 1)*ldy + r + 8] = acc[nt][3];
            }
        }
    }
    (void) M;
}

template <int BN, int LAY = LAY_SB, int XSH = 1, bool ID = false, int WARPS = 4, int KT = 0, bool PERM = false>
static void launch_ampere_mmv(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y, int64_t ldy,
                               int M, int N, int K, cudaStream_t stream, const idmap & idm = idmap{}, int max_tiles = 1) {
    constexpr int smem = ampere_mmv_smem_bytes<BN, LAY, WARPS>();
    constexpr int red_bytes = (WARPS - 1)*(BN/8)*4*32*4;
    if constexpr (smem + red_bytes > 48*1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(ampere_mmv<BN, LAY, XSH, ID, WARPS, KT, PERM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid(M/16, ID ? max_tiles : (N + BN - 1)/BN);
    ampere_mmv<BN, LAY, XSH, ID, WARPS, KT, PERM><<<grid, WARPS*32, smem, stream>>>(W, Xq, Xs, Y, ldy, M, N, K, idm);
}

} // namespace ai4k
