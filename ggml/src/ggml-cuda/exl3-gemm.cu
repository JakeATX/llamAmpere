// EXL3 format, "mul1" codebook arithmetic, tile layout and trellis GEMV/GEMM design: Turboderp, exllamav3
// (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp; see
// licenses/LICENSE-exllamav3. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou, De Sa,
// "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235).
// This file is an independent ggml/CUDA reimplementation written with the exllamav3 source as reference;
// no QTIP code was used.
#include "exl3.cuh"

#include <cstring>

// [#13] EXL3 prefill GEMM (T > EXL3_GEMV_MAX_T activation columns) straight from the trellis stream: no f16 copy of
// the weight. Replaces reconstruct-to-f16 + cuBLAS (the M1 path, still reachable with GGML_CUDA_EXL3_GEMM=0).
//
// y[T][N] = x[T][K] . W[K][N], x = the glued fp16 activations (exl3_glue_in, scale 1, or 1/16 for ACC16).
// Block tile BM tokens x BN = 128 outputs (8 n-tiles), k-step BK = 64 (4 k-tiles), 8 warps as 2 (M) x 4 (N), warp
// tile (16*MT) x 32. Per k-step a block stages BM x 64 fp16 activations (cp.async 16 B, 16-byte chunks XOR-swizzled
// by row so ldmatrix is conflict-free) and the 4 x 8 compressed 16x16 weight tiles (cp.async 16 B, 32*BITS bytes
// per tile, contiguous along n) in a STAGES-deep ring. Each warp decodes its two n-tiles of a k-tile from shared
// memory into registers with the GEMV decode (lane order == mma.m16n8k16 B fragment, see exl3-gemv.cu) and issues
// 2*MT mma.m16n8k16 per tile with x as the A operand (ldmatrix.x4). Accumulation is fp32 (mma .f32), or with
// ACC16 fp16 inside one k-step (64 products, the GEMV fold period) folded into fp32 after every k-step.
// Ragged tails: tokens >= T, k-tiles >= kt and n-tiles >= nt are zero-filled (cp.async src-size 0) and never stored,
// so any T and any K, N that are multiples of 16 are exact. Split-K (grid z) writes partials [ks][T][N] that
// glue_out reduces in a fixed order (deterministic); without split-K and svh the block stores y directly.

#define EXL3_GEMM_BN      128
#define EXL3_GEMM_BK      64
#define EXL3_GEMM_KT      (EXL3_GEMM_BK / 16)   // k-tiles per k-step
#define EXL3_GEMM_NTB     (EXL3_GEMM_BN / 16)   // n-tiles per block
#define EXL3_GEMM_WARPS_M 2
#define EXL3_GEMM_WARPS_N 4
#define EXL3_GEMM_THREADS (32 * EXL3_GEMM_WARPS_M * EXL3_GEMM_WARPS_N)
#ifndef EXL3_GEMM_ACC16_MT4_MINB
#define EXL3_GEMM_ACC16_MT4_MINB 2   // 2 blocks/SM (128-register cap; <= 16 B spill in 2 of 21 ACC16 instantiations): +1.0% pp4096, +1.8% pp20480 vs 1
#endif
#define EXL3_GEMM_X_SCALE_ACC16 0.0625f         // == EXL3_GEMV_X_SCALE (exl3-gemv.cu): same fp16 envelope

template <int BITS, int MT, int STAGES>
struct exl3_gemm_cfg {
    static constexpr int BM        = EXL3_GEMM_WARPS_M * 16 * MT;
    static constexpr int A_BYTES   = BM * EXL3_GEMM_BK * 2;                              // per stage
    static constexpr int W_BYTES   = EXL3_GEMM_KT * EXL3_GEMM_NTB * 32 * BITS;            // per stage
    static constexpr int STAGE     = A_BYTES + W_BYTES;
    static constexpr int SMEM      = STAGES * STAGE;
    static constexpr int A_CHUNKS  = A_BYTES / 16;
    static constexpr int W_CHUNKS  = W_BYTES / 16;
    static constexpr int W_ROW_CHUNKS  = EXL3_GEMM_NTB * 2 * BITS;   // 16-B chunks of one k-tile row (8 tiles)
    static constexpr int TILE_CHUNKS   = 2 * BITS;                   // 16-B chunks per tile
};

static __device__ __forceinline__ uint32_t exl3_gemm_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

// 16-byte global -> shared copy; src_bytes = 0 zero-fills the destination (src must still be a valid address)
static __device__ __forceinline__ void exl3_gemm_cp16(const uint32_t dst, const void * src, const int src_bytes) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes));
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(src_bytes);
#endif
}
static __device__ __forceinline__ void exl3_gemm_cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
static __device__ __forceinline__ void exl3_gemm_cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

static __device__ __forceinline__ void exl3_gemm_ldmatrix_x4(uint32_t (&a)[4], const uint32_t addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
#else
    GGML_UNUSED(addr); a[0] = a[1] = a[2] = a[3] = 0u;
#endif
}

// d[16x8] (fp32) += a[16x16] . b[16x8]
static __device__ __forceinline__ void exl3_gemm_mma_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}
// d[16x8] (fp16) += a[16x16] . b[16x8]
static __device__ __forceinline__ void exl3_gemm_mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

// one 16x16 tile (NW = 8*BITS words at w) -> the lane's four half2 (b0, b1 of n8 half 0, b0, b1 of n8 half 1);
// the window arithmetic of exl3-gemv.cu (k_exl3_gemv decode), words read from shared memory
template <int BITS>
static __device__ __forceinline__ void exl3_gemm_decode(const uint32_t * __restrict__ w, const int (&woff)[BITS <= 4 ? 2 : 3],
                                                        const int rel0, uint32_t (&b)[4]) {
    constexpr int NWORDS = BITS <= 4 ? 2 : 3;
    uint32_t ww[NWORDS];
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        ww[m] = w[woff[m]];
    }
    uint32_t a[NWORDS];
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        a[m] = __funnelshift_l(m + 1 < NWORDS ? ww[m + 1] : 0u, ww[m], rel0);
    }
    const half2 cb_mul = __halves2half2(__ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0x1eee));
    const half2 cb_add = __halves2half2(__ushort_as_half((unsigned short) 0xc931), __ushort_as_half((unsigned short) 0xc931));
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {
        uint32_t sum[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int o  = (2 * jp + e) * BITS;
            const int wi = o >> 5;
            const int sh = o & 31;
            const uint32_t hi = a[wi];
            const uint32_t lo = (wi + 1 < NWORDS) ? a[wi + 1] : 0u;
            uint32_t code;
            if (sh == 0) {
                code = hi >> 16;
            } else if (sh == 8) {
                code = __byte_perm(hi, 0u, 0x4421);
            } else if (sh == 16) {
                code = hi & 0xffffu;
            } else {
                code = __funnelshift_l(lo, hi, sh) >> 16;
            }
            sum[e] = __dp4a(code * 0x83DCD12Du, 0x01010101u, 0x6400u);
        }
        const half2 v = __hfma2(__halves2half2(__ushort_as_half((unsigned short) sum[0]),
                                               __ushort_as_half((unsigned short) sum[1])), cb_mul, cb_add);
        b[jp] = *reinterpret_cast<const uint32_t *>(&v);
    }
}

// x: fp16 [T][K]; trellis [kt][nt][8*BITS]; out [gridDim.z][T][N] (gridDim.z == 1: y itself); ksteps_per_split
// k-steps of BK per z slice
template <int BITS, int MT, int STAGES, bool ACC16>
static __global__ void __launch_bounds__(EXL3_GEMM_THREADS, (MT >= 4 && ACC16) ? EXL3_GEMM_ACC16_MT4_MINB : 2)
k_exl3_gemm(const uint32_t * __restrict__ trellis, const half * __restrict__ x, float * __restrict__ out,
            const int T, const int K, const int N, const int ksteps_per_split, const float out_scale) {
#if __CUDA_ARCH__ >= 800
    using cfg = exl3_gemm_cfg<BITS, MT, STAGES>;
    constexpr int BM     = cfg::BM;
    constexpr int NW     = 8 * BITS;
    constexpr int NBITS  = 256 * BITS;
    constexpr int NWORDS = BITS <= 4 ? 2 : 3;
    extern __shared__ __align__(16) uint8_t exl3_gemm_smem[];

    const int kt = K / 16;
    const int nt = N / 16;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / EXL3_GEMM_WARPS_N;   // 0..1
    const int wn   = warp % EXL3_GEMM_WARPS_N;   // 0..3
    const int g    = lane >> 2;
    const int q    = lane & 3;

    const int nb0 = blockIdx.x * EXL3_GEMM_NTB;   // first n-tile of the block
    const int m0  = blockIdx.y * BM;              // first token of the block
    const int nsteps_total = (kt + EXL3_GEMM_KT - 1) / EXL3_GEMM_KT;
    const int s_begin = blockIdx.z * ksteps_per_split;
    const int s_end   = min(nsteps_total, s_begin + ksteps_per_split);
    const int nsteps  = max(0, s_end - s_begin);

    // decode window offsets of this lane (compile-time BITS, lane-dependent)
    const int p0   = ((lane * 8 + 1) * BITS - 16 + NBITS) % NBITS;
    const int ws   = p0 >> 5;
    const int rel0 = p0 & 31;
    int woff[NWORDS];
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        woff[m] = (ws + m) % NW;
    }

    // ---- stage loader: k-step s (absolute) into ring slot ----
    auto load_stage = [&](const int s, const int slot) {
        uint8_t * st = exl3_gemm_smem + slot * cfg::STAGE;
        const uint32_t a_base = exl3_gemm_smem_u32(st);
        const uint32_t w_base = exl3_gemm_smem_u32(st + cfg::A_BYTES);
        const int k0 = s * EXL3_GEMM_BK;
        // activations: BM rows x 8 chunks of 8 halves; chunk c of row r lands at chunk c ^ (r & 7)
#pragma unroll
        for (int i = 0; i < (cfg::A_CHUNKS + EXL3_GEMM_THREADS - 1) / EXL3_GEMM_THREADS; ++i) {
            const int c = tid + i * EXL3_GEMM_THREADS;
            if (cfg::A_CHUNKS % EXL3_GEMM_THREADS == 0 || c < cfg::A_CHUNKS) {
                const int r  = c >> 3;
                const int ch = c & 7;
                const int t  = m0 + r;
                const int k  = k0 + ch * 8;
                const bool ok = t < T && k < K;
                const half * src = ok ? x + (size_t) t * K + k : x;
                exl3_gemm_cp16(a_base + (uint32_t) (r * 128 + ((ch ^ (r & 7)) << 4)), src, ok ? 16 : 0);
            }
        }
        // weights: EXL3_GEMM_KT k-tile rows, each 8 contiguous tiles (W_ROW_CHUNKS chunks) in global memory
#pragma unroll
        for (int i = 0; i < (cfg::W_CHUNKS + EXL3_GEMM_THREADS - 1) / EXL3_GEMM_THREADS; ++i) {
            const int c = tid + i * EXL3_GEMM_THREADS;
            if (cfg::W_CHUNKS % EXL3_GEMM_THREADS == 0 || c < cfg::W_CHUNKS) {
                const int kk = c / cfg::W_ROW_CHUNKS;
                const int cr = c - kk * cfg::W_ROW_CHUNKS;
                const int nl = cr / cfg::TILE_CHUNKS;
                const int ki = s * EXL3_GEMM_KT + kk;
                const int ni = nb0 + nl;
                const bool ok = ki < kt && ni < nt;
                const uint8_t * src = ok ? reinterpret_cast<const uint8_t *>(trellis + ((size_t) ki * nt + nb0) * NW) + cr * 16
                                         : reinterpret_cast<const uint8_t *>(trellis);
                exl3_gemm_cp16(w_base + (uint32_t) (c * 16), src, ok ? 16 : 0);
            }
        }
    };

    float acc[MT][2][2][4];   // [m16 tile][n-tile of the warp][n8 half][c0..c3]
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][h][e] = 0.0f;

    // ---- prologue ----
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nsteps) {
            load_stage(s_begin + s, s);
        }
        exl3_gemm_cp_commit();
    }

    // ldmatrix.x4 source of lane: row (lane & 15) of the m16 tile, k chunk (lane >> 4) of the k-tile
    const int lrow = lane & 15;
    const int lchk = lane >> 4;

    for (int it = 0; it < nsteps; ++it) {
        exl3_gemm_cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nx = it + STAGES - 1;
            if (nx < nsteps) {
                load_stage(s_begin + nx, nx % STAGES);
            }
            exl3_gemm_cp_commit();
        }
        const int slot = it % STAGES;
        const uint8_t * st = exl3_gemm_smem + slot * cfg::STAGE;
        const uint32_t a_base = exl3_gemm_smem_u32(st);
        const uint32_t * wsm = reinterpret_cast<const uint32_t *>(st + cfg::A_BYTES);

        uint32_t d16[ACC16 ? MT : 1][2][2][2];
        if constexpr (ACC16) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) { d16[i][j][h][0] = 0u; d16[i][j][h][1] = 0u; }
        }

#pragma unroll
        for (int kk = 0; kk < EXL3_GEMM_KT; ++kk) {
            uint32_t b[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                exl3_gemm_decode<BITS>(wsm + (kk * EXL3_GEMM_NTB + wn * 2 + j) * NW, woff, rel0, b[j]);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r  = (wm * MT + i) * 16 + lrow;
                const int ch = kk * 2 + lchk;
                uint32_t a[4];
                exl3_gemm_ldmatrix_x4(a, a_base + (uint32_t) (r * 128 + ((ch ^ (r & 7)) << 4)));
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if constexpr (ACC16) {
                        exl3_gemm_mma_f16(d16[i][j][0], a, b[j][0], b[j][1]);
                        exl3_gemm_mma_f16(d16[i][j][1], a, b[j][2], b[j][3]);
                    } else {
                        exl3_gemm_mma_f32(acc[i][j][0], a, b[j][0], b[j][1]);
                        exl3_gemm_mma_f32(acc[i][j][1], a, b[j][2], b[j][3]);
                    }
                }
            }
        }
        if constexpr (ACC16) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 lo = __half22float2(*reinterpret_cast<const half2 *>(&d16[i][j][h][0]));
                        const float2 hi = __half22float2(*reinterpret_cast<const half2 *>(&d16[i][j][h][1]));
                        acc[i][j][h][0] += lo.x; acc[i][j][h][1] += lo.y;
                        acc[i][j][h][2] += hi.x; acc[i][j][h][3] += hi.y;
                    }
        }
    }
    exl3_gemm_cp_wait<0>();

    // ---- epilogue: D rows = tokens (g, g + 8), D columns = outputs (2q, 2q + 1) of each n8 half ----
    float * o = out + (size_t) blockIdx.z * T * N;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int ni = nb0 + wn * 2 + j;
        if (ni >= nt) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const int t0 = m0 + (wm * MT + i) * 16 + g;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = ni * 16 + h * 8 + 2 * q;
                if (t0 < T) {
                    *reinterpret_cast<float2 *>(o + (size_t) t0 * N + n) =
                        make_float2(acc[i][j][h][0] * out_scale, acc[i][j][h][1] * out_scale);
                }
                if (t0 + 8 < T) {
                    *reinterpret_cast<float2 *>(o + (size_t) (t0 + 8) * N + n) =
                        make_float2(acc[i][j][h][2] * out_scale, acc[i][j][h][3] * out_scale);
                }
            }
        }
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(x); GGML_UNUSED(out); GGML_UNUSED(T); GGML_UNUSED(K); GGML_UNUSED(N);
    GGML_UNUSED(ksteps_per_split); GGML_UNUSED(out_scale);
    NO_DEVICE_CODE;
#endif
}

static int exl3_gemm_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env != nullptr ? atoi(env) : def;
}

bool ggml_cuda_exl3_gemm_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_GEMM");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

// ACC16 is the default: same fp16 envelope as the shipped GEMV mma path (x/16, 64 products per fold); generated-token
// KLD vs the base build equals the fp32-acc kernel's (q8 KV 0.000021 vs 0.000017 at 25.6K) at +19% prefill (pp4096).
// GGML_CUDA_EXL3_GEMM_ACC16=0 selects fp32 accumulation.
static bool exl3_gemm_acc16() {
    static const bool on = exl3_gemm_env_int("GGML_CUDA_EXL3_GEMM_ACC16", 1) != 0;
    return on;
}

#define EXL3_GEMM_STAGES 2
#define EXL3_GEMM_MIN_T  16   // = EXL3_GEMV_MAX_T (exl3-gemv.cu): T <= 16 never reaches this GEMM

// probe == true: return the resident blocks per SM of the instantiation instead of launching
template <int BITS, int MT, bool ACC16>
static int exl3_gemm_launch_bm(const bool probe, const uint32_t * trellis, const half * x, float * out, const int T, const int K,
                               const int N, const int ksplit, const int ksteps_per_split, const float out_scale, cudaStream_t stream) {
    using cfg = exl3_gemm_cfg<BITS, MT, EXL3_GEMM_STAGES>;
    static int occ[GGML_CUDA_MAX_DEVICES] = {0};
    const int id = ggml_cuda_get_device();
    if (occ[id] == 0) {
        CUDA_CHECK(cudaFuncSetAttribute(k_exl3_gemm<BITS, MT, EXL3_GEMM_STAGES, ACC16>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, cfg::SMEM));
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_exl3_gemm<BITS, MT, EXL3_GEMM_STAGES, ACC16>,
                                                                 EXL3_GEMM_THREADS, cfg::SMEM));
        occ[id] = std::max(1, nb);
    }
    if (probe) {
        return occ[id];
    }
    const dim3 grid((N / 16 + EXL3_GEMM_NTB - 1) / EXL3_GEMM_NTB, (T + cfg::BM - 1) / cfg::BM, ksplit);
    k_exl3_gemm<BITS, MT, EXL3_GEMM_STAGES, ACC16><<<grid, EXL3_GEMM_THREADS, cfg::SMEM, stream>>>(
        trellis, x, out, T, K, N, ksteps_per_split, out_scale);
    return occ[id];
}

template <int MT, bool ACC16>
static int exl3_gemm_launch_bits(const bool probe, const int bits, const uint32_t * trellis, const half * x, float * out, const int T,
                                 const int K, const int N, const int ksplit, const int kps, const float out_scale, cudaStream_t stream) {
    switch (bits) {
        case 2: return exl3_gemm_launch_bm<2, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 3: return exl3_gemm_launch_bm<3, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 4: return exl3_gemm_launch_bm<4, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 5: return exl3_gemm_launch_bm<5, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 6: return exl3_gemm_launch_bm<6, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 7: return exl3_gemm_launch_bm<7, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 8: return exl3_gemm_launch_bm<8, MT, ACC16>(probe, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        default: GGML_ABORT("exl3 gemm: unsupported bits");
    }
}

static int exl3_gemm_launch(const bool probe, const int mt, const bool acc16, const int bits, const uint32_t * trellis, const half * x,
                            float * out, const int T, const int K, const int N, const int ksplit, const int kps, const float out_scale,
                            cudaStream_t stream) {
    switch (mt) {
        case 1:  return acc16 ? exl3_gemm_launch_bits<1, true>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream)
                              : exl3_gemm_launch_bits<1, false>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        case 2:  return acc16 ? exl3_gemm_launch_bits<2, true>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream)
                              : exl3_gemm_launch_bits<2, false>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
        default: return acc16 ? exl3_gemm_launch_bits<4, true>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream)
                              : exl3_gemm_launch_bits<4, false>(probe, bits, trellis, x, out, T, K, N, ksplit, kps, out_scale, stream);
    }
}

static int exl3_gemm_mt(const int64_t T) {
    static const int forced = exl3_gemm_env_int("GGML_CUDA_EXL3_GEMM_MT", 0);
    if (forced == 1 || forced == 2 || forced == 4) {
        return forced;
    }
    return T <= 32 ? 1 : (T <= 64 ? 2 : 4);
}

// split-K so that the grid fills the device: the smallest ksplit whose last wave is >= 85% full (or the best one up to
// the cap), each slice keeping >= 4 k-steps; only for grids of <= 2 waves (beyond that the tail is a small fraction and
// the split costs a ksplit*T*N fp32 partial buffer plus a reduction pass)
static int exl3_gemm_ksplit(const int blocks, const int resident, const int nsteps) {
    static const int forced = exl3_gemm_env_int("GGML_CUDA_EXL3_GEMM_KSPLIT", 0);
    if (forced > 0) {
        return std::max(1, std::min(forced, nsteps));
    }
    if (blocks > 2 * resident) {
        return 1;
    }
    const int ks_max = std::max(1, std::min(8, nsteps / 4));
    int best = 1;
    double best_eff = 0.0;
    for (int ks = 1; ks <= ks_max; ++ks) {
        const int64_t items = (int64_t) blocks * ks;
        const int64_t waves = (items + resident - 1) / resident;
        const double eff = (double) items / (double) (waves * resident);
        if (eff >= 0.85) {
            return ks;
        }
        if (eff > best_eff + 1e-9) {
            best_eff = eff;
            best = ks;
        }
    }
    return best;
}

bool ggml_cuda_exl3_gemm_supported(const ggml_tensor * src0, const int64_t T) {
    // T <= EXL3_GEMV_MAX_T stays with the GEMV or, with GGML_CUDA_EXL3_GEMV=0, the M1 reconstruct + cuBLAS reference
    if (!ggml_cuda_exl3_gemm_enabled() || T <= EXL3_GEMM_MIN_T) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_MTHREADS(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    // 16-byte cp.async of whole tiles: the tensor base must be 16-byte aligned (tiles are 32*bits bytes)
    // glue_in (x -> fp16, optional suh Hadamard) needs K % 128
    return ((uintptr_t) src0->data % 16) == 0 && src0->ne[0] % 128 == 0 && src0->ne[1] % 16 == 0 &&
           T * src0->ne[0] < INT32_MAX && T * src0->ne[1] < INT32_MAX;
}

void ggml_cuda_exl3_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         const int64_t T, const float * suh, const float * svh) {
    const int bits = ggml_exl3_bits(src0->type);
    GGML_ASSERT(bits >= 2 && bits <= 8);
    const int K  = (int) src0->ne[0];
    const int N  = (int) src0->ne[1];
    const int kt = K / 16;
    const int nt = N / 16;
    // pools and device info by the context's (virtual) device id, as GGML_CUDA_DEVICES emulation maps several contexts to
    // one physical device; the per-device occupancy/attribute cache inside exl3_gemm_launch keys on the physical id
    const int id = ctx.device;
    cudaStream_t stream = ctx.stream();

    const bool  acc16   = exl3_gemm_acc16();
    const float x_scale = acc16 ? EXL3_GEMM_X_SCALE_ACC16 : 1.0f;
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), (size_t) T * K);
    ggml_cuda_exl3_glue_in_f16((const float *) src1->data, suh, xh.get(), K, T, x_scale, stream);

    const int mt = exl3_gemm_mt(T);
    const int bm = EXL3_GEMM_WARPS_M * 16 * mt;
    const int blocks = ((nt + EXL3_GEMM_NTB - 1) / EXL3_GEMM_NTB) * (int) ((T + bm - 1) / bm);
    const int nsteps = (kt + EXL3_GEMM_KT - 1) / EXL3_GEMM_KT;
    const int resident = exl3_gemm_launch(true, mt, acc16, bits, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0.0f, stream) *
                         ggml_cuda_info().devices[id].nsm;
    const int ksplit_req = exl3_gemm_ksplit(blocks, resident, nsteps);
    const int kps    = (nsteps + ksplit_req - 1) / ksplit_req;
    const int ksplit = (nsteps + kps - 1) / kps;   // no empty z slices

    const bool post = ksplit > 1 || svh != nullptr;
    // [FIT12 overhead] without split-K the svh/Hadamard epilogue runs in place on dst (bit-identical), so the T x N f32
    // partial buffer is not taken from the pool (68 MiB for the fused gate+up at T = 512). GGML_CUDA_EXL3_GLUE_INPLACE=0
    // restores the out-of-place epilogue.
    static const bool inplace_ok = [] { const char * e = getenv("GGML_CUDA_EXL3_GLUE_INPLACE"); return !(e && e[0] == '0'); }();
    const bool inplace = post && ksplit == 1 && svh != nullptr && N % 128 == 0 && inplace_ok;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    float * y   = (float *) dst->data;
    float * out = post && !inplace ? part.alloc((size_t) ksplit * T * N) : y;
    const float out_scale = 1.0f / x_scale;

    exl3_gemm_launch(false, mt, acc16, bits, (const uint32_t *) src0->data, xh.get(), out, (int) T, K, N, ksplit, kps, out_scale, stream);
    if (inplace) {
        ggml_cuda_exl3_glue_out_inplace(y, svh, N, T, 1.0f, stream);
    } else if (post) {
        ggml_cuda_exl3_glue_out(out, svh, y, N, T, ksplit, 1.0f, stream);
    }
}
