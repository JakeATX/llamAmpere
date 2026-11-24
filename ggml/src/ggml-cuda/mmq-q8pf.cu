// PFG-4: dense Q8_0 prompt GEMM (GGML_CUDA_MMQ_Q8PF=1, default off).
//
// The stock MMQ Q8_0 path on sm_86 spends most of its issue slots outside the tensor cores: each m16n8k32 product
// (16 clocks of int8 tensor work per sub-partition) is followed by four I2F conversions, which run at 16 per clock per
// SM on sm_86 (32 clocks per mma), and its tile loads are synchronous (load, barrier, math, barrier).
// This kernel:
//   - seeds the mma accumulator with the bits of 1.5*2^23, so the int32 result already is a biased float; one exact
//     FADD then gives float(dot) (|dot| <= 32*128*128 = 2^19 < 2^22), so no I2F. Per output the fp32 ops are the
//     MMQ vec dot's: sum = fma(float(dot)*dA, dB, sum), blocks in K order, no K split.
//   - streams X (raw 34-byte Q8_0 blocks, 4 per stage row) and the q8_1 activation tile with cp.async, 2-4 stages.
//   - tile 128 rows x BN columns (BN 128/64/32 picked for wave efficiency), 8 warps of 32 x BN/2.
// Gate: Q8_0 weights, rows % 128 == 0 and <= GGML_CUDA_MMQ_Q8PF_MAXROWS (2304), K % 128 == 0, ncols >= GGML_CUDA_MMQ_Q8PF_MIN (default 64, never below 16),
// no ids / fusion / channels. Everything else keeps the default MMQ.

#include "mmq-q8pf.cuh"
#include "mmq.cuh"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int q8pf_bm       = 128;
constexpr int q8pf_nthreads = 256;
constexpr int q8pf_xrow     = 144;                    // smem bytes per x row and stage: 4 blocks (136) + 8 pad
constexpr int q8pf_ycol     = sizeof(block_q8_1_mmq); // 144: 4 scales + 128 int8
constexpr int q8pf_magic    = 0x4B400000;             // bits of 1.5*2^23
constexpr float q8pf_magicf = 12582912.0f;

static_assert(sizeof(block_q8_0) == 34, "q8pf assumes the 34-byte Q8_0 block");
static_assert(sizeof(block_q8_1_mmq) == 144, "q8pf assumes the 144-byte q8_1 MMQ block");

template <int BN> static constexpr __host__ __device__ int q8pf_nstages() { return BN == 128 ? 2 : BN == 64 ? 3 : 4; }
template <int BN> static constexpr __host__ __device__ int q8pf_stage_bytes() { return q8pf_bm*q8pf_xrow + BN*q8pf_ycol; }
template <int BN> static constexpr __host__ __device__ int q8pf_smem() { return q8pf_nstages<BN>()*q8pf_stage_bytes<BN>(); }

static int q8pf_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

static bool q8pf_enabled() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_MMQ_Q8PF"); return e && strcmp(e, "1") == 0; }();
    return v;
}

static int q8pf_min_cols() {
    static const int v = std::max(16, q8pf_env_int("GGML_CUDA_MMQ_Q8PF_MIN", 64));
    return v;
}

// Weights with more rows stay on stock MMQ: its stream-K partition beats this kernel there (RTX 3090, n=1024, kernel
// only: 3072 x 2304 198 us vs 155 + 20 us fixup; 1024 x 2304 60 vs 56 + 19; 384 x 2304 40 vs 27 + 17).
// GGML_CUDA_MMQ_Q8PF_MAXROWS=N (default 2304, 0 = no cap).
static int64_t q8pf_max_rows() {
    static const int64_t v = std::max(0, q8pf_env_int("GGML_CUDA_MMQ_Q8PF_MAXROWS", 2304));
    return v;
}

static int q8pf_forced_bn() {
    static const int v = [] { const int b = q8pf_env_int("GGML_CUDA_MMQ_Q8PF_BN", 0); return b == 32 || b == 64 || b == 128 ? b : 0; }();
    return v;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ void q8pf_cp_async(void * dst, const void * src, const int bytes, const bool valid) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    const uint32_t addr = __cvta_generic_to_shared(dst);
    if (bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr), "l"(src), "r"(valid ? 16 : 0) : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(addr), "l"(src), "r"(valid ? 8 : 0) : "memory");
    }
#else
    GGML_UNUSED_VARS(dst, src, bytes, valid);
#endif
}

static __device__ __forceinline__ void q8pf_commit() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// One stage: 128 x rows of 4 Q8_0 blocks (136 bytes, 17 x 8-byte copies) and BN q8_1 columns of 128 values (9 x 16 bytes).
template <int BN>
static __device__ __forceinline__ void q8pf_load_stage(
        char * stage, const char * x, const block_q8_1_mmq * y, const int kk, const int m0, const int n0, const int n,
        const size_t row_bytes, const int64_t stride_y) {
    const char * xb = x + size_t(m0)*row_bytes + size_t(kk)*(4*sizeof(block_q8_0));
#pragma unroll 1
    for (int t = threadIdx.x; t < q8pf_bm*17; t += q8pf_nthreads) {
        const int row = t / 17;
        const int c   = t % 17;
        q8pf_cp_async(stage + row*q8pf_xrow + c*8, xb + size_t(row)*row_bytes + c*8, 8, true);
    }
    char * ys = stage + q8pf_bm*q8pf_xrow;
    const block_q8_1_mmq * yb = y + int64_t(kk)*stride_y;
#pragma unroll 1
    for (int t = threadIdx.x; t < BN*9; t += q8pf_nthreads) {
        const int col = t / 9;
        const int c   = t % 9;
        const int cg  = n0 + col;
        const char * src = (const char *) (yb + min(cg, n - 1)) + c*16;
        q8pf_cp_async(ys + col*q8pf_ycol + c*16, src, 16, cg < n);
    }
    q8pf_commit();
}

// int at byte p of a block's qs: odd blocks are 4-byte aligned in the stage, even ones sit 2 bytes off.
template <bool aligned>
static __device__ __forceinline__ int q8pf_ld_qs(const char * p) {
    if constexpr (aligned) {
        return *(const int *) p;
    } else {
        const int * w = (const int *) (p - 2);
        return __byte_perm(w[0], w[1], 0x5432);
    }
}

static __device__ __forceinline__ void q8pf_mma(int (&d)[4], const int (&a)[4], const int b0, const int b1) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    // not volatile: a pure register op, so the compiler may batch independent products ahead of their epilogues
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(q8pf_magic));
#else
    GGML_UNUSED_VARS(d, a, b0, b1);
#endif
}

// B fragments of two consecutive sub-blocks (sp, sp + 1) for 8 columns: matrix m = lane/8 is sub sp + m/2, k half m%2
static __device__ __forceinline__ void q8pf_ldmatrix_b(int (&r)[4], const char * ys, const int c0, const int sp, const int lane) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    const char * p = ys + (c0 + lane % 8)*q8pf_ycol + 16 + (sp + lane/16)*32 + ((lane/8) % 2)*16;
    const uint32_t addr = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, ys, c0, sp, lane);
#endif
}

// One 128-deep K step (4 Q8_0 blocks) of the warp's 32 x BN/2 tile, two blocks at a time. Per output the blocks are
// added in K order.
template <int BN>
static __device__ __forceinline__ void q8pf_kstep(
        const char * xs, const char * ys, const int wm, const int wn, const int lane, float (&acc)[2][BN/16][4]) {
    constexpr int nb = BN/16;
    const int q = lane % 4;
#pragma unroll
    for (int sp = 0; sp < 4; sp += 2) {
        int   a[2][2][4];
        float da[2][2][2];
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const char * rp = xs + (wm*32 + t*16 + h*8 + lane/4)*q8pf_xrow + (sp + s2)*int(sizeof(block_q8_0));
                    da[s2][t][h] = __half2float(*(const half *) rp);
                    if (s2 == 1) { // odd block: 4-byte aligned
                        a[s2][t][h]     = q8pf_ld_qs<true>(rp + 2 + 4*q);
                        a[s2][t][2 + h] = q8pf_ld_qs<true>(rp + 2 + 16 + 4*q);
                    } else {
                        a[s2][t][h]     = q8pf_ld_qs<false>(rp + 2 + 4*q);
                        a[s2][t][2 + h] = q8pf_ld_qs<false>(rp + 2 + 16 + 4*q);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < nb; ++j) {
            const int c0 = wn*(BN/2) + j*8;
            const float2 d0 = *(const float2 *) (ys + (c0 + 2*q + 0)*q8pf_ycol + sp*4);
            const float2 d1 = *(const float2 *) (ys + (c0 + 2*q + 1)*q8pf_ycol + sp*4);
            int bf[4];
            q8pf_ldmatrix_b(bf, ys, c0, sp, lane);
            int d[2][2][4];
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    q8pf_mma(d[s2][t], a[s2][t], bf[2*s2], bf[2*s2 + 1]);
                }
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                const float db0 = s2 == 0 ? d0.x : d0.y;
                const float db1 = s2 == 0 ? d1.x : d1.y;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    // float(dot) exactly, then the MMQ vec dot's (float(dot)*dA)*dB + sum; blocks in K order
                    const float f0 = __fsub_rn(__int_as_float(d[s2][t][0]), q8pf_magicf);
                    const float f1 = __fsub_rn(__int_as_float(d[s2][t][1]), q8pf_magicf);
                    const float f2 = __fsub_rn(__int_as_float(d[s2][t][2]), q8pf_magicf);
                    const float f3 = __fsub_rn(__int_as_float(d[s2][t][3]), q8pf_magicf);
                    acc[t][j][0] = __fmaf_rn(__fmul_rn(f0, da[s2][t][0]), db0, acc[t][j][0]);
                    acc[t][j][1] = __fmaf_rn(__fmul_rn(f1, da[s2][t][0]), db1, acc[t][j][1]);
                    acc[t][j][2] = __fmaf_rn(__fmul_rn(f2, da[s2][t][1]), db0, acc[t][j][2]);
                    acc[t][j][3] = __fmaf_rn(__fmul_rn(f3, da[s2][t][1]), db1, acc[t][j][3]);
                }
            }
        }
    }
}

template <int BN>
__launch_bounds__(q8pf_nthreads, 1)
static __global__ void k_mmq_q8pf(
        const char * __restrict__ x, const block_q8_1_mmq * __restrict__ y, float * __restrict__ dst,
        const int n, const int ksteps, const size_t row_bytes, const int64_t stride_y, const int64_t stride_dst) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    constexpr int nst = q8pf_nstages<BN>();
    constexpr int sb  = q8pf_stage_bytes<BN>();
    constexpr int nb  = BN/16;
    extern __shared__ __align__(16) char q8pf_stages[];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int wm   = warp % 4;
    const int wn   = warp / 4;
    const int n0   = blockIdx.x*BN;
    const int m0   = blockIdx.y*q8pf_bm;

    float acc[2][nb][4];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int j = 0; j < nb; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[t][j][l] = 0.0f;
            }
        }
    }

    // every stage owns one async group, empty groups at the K tail keep the wait count fixed
#pragma unroll
    for (int p = 0; p < nst - 1; ++p) {
        if (p < ksteps) {
            q8pf_load_stage<BN>(q8pf_stages + p*sb, x, y, p, m0, n0, n, row_bytes, stride_y);
        } else {
            q8pf_commit();
        }
    }

#pragma unroll 1
    for (int k = 0; k < ksteps; ++k) {
        asm volatile("cp.async.wait_group %0;" :: "n"(nst - 2) : "memory");
        __syncthreads();
        const int next = k + nst - 1;
        if (next < ksteps) {
            q8pf_load_stage<BN>(q8pf_stages + (next % nst)*sb, x, y, next, m0, n0, n, row_bytes, stride_y);
        } else {
            q8pf_commit();
        }
        const char * xs = q8pf_stages + (k % nst)*sb;
        const char * ys = xs + q8pf_bm*q8pf_xrow;
        q8pf_kstep<BN>(xs, ys, wm, wn, lane, acc);
    }

    const int q = lane % 4;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int j = 0; j < nb; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = m0 + wm*32 + t*16 + (l/2)*8 + lane/4;
                const int col = n0 + wn*(BN/2) + j*8 + 2*q + l%2;
                if (col < n) {
                    dst[int64_t(col)*stride_dst + row] = acc[t][j][l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, n, ksteps, row_bytes, stride_y, stride_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template <int BN>
static void q8pf_launch(const mmq_args & args, cudaStream_t stream) {
    constexpr int smem = q8pf_smem<BN>();
    CUDA_SET_SHARED_MEMORY_LIMIT(k_mmq_q8pf<BN>, smem);
    const dim3 grid((args.ncols_dst + BN - 1)/BN, args.nrows_x/q8pf_bm, 1);
    k_mmq_q8pf<BN><<<grid, q8pf_nthreads, smem, stream>>>(
        args.x, (const block_q8_1_mmq *) args.y, args.dst, (int) args.ncols_dst, (int) (args.ncols_x/128),
        size_t(args.stride_row_x)*sizeof(block_q8_0), args.ncols_y, args.nrows_dst);
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA

// Smallest modelled time over the tile widths that fit: waves of one CTA per SM, each costing BN + 24 (fixed per-tile
// overhead in column units). GGML_CUDA_MMQ_Q8PF_BN=32|64|128 forces a width.
static int q8pf_pick_bn(const int64_t m, const int64_t n, const int nsm, const size_t smpbo) {
    if (const int f = q8pf_forced_bn(); f != 0) {
        return f;
    }
    int best = 0;
    int64_t best_cost = INT64_MAX;
    for (const int bn : {128, 64, 32}) {
        const size_t smem = bn == 128 ? q8pf_smem<128>() : bn == 64 ? q8pf_smem<64>() : q8pf_smem<32>();
        if (smem > smpbo) {
            continue;
        }
        const int64_t tiles = (m/q8pf_bm)*((n + bn - 1)/bn);
        const int64_t cost  = ((tiles + nsm - 1)/nsm)*(bn + 24);
        if (cost < best_cost) {
            best_cost = cost;
            best = bn;
        }
    }
    return best;
}

} // namespace

bool ggml_cuda_should_use_mmq_q8pf(const mmq_args & args) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!q8pf_enabled() || args.type_x != GGML_TYPE_Q8_0 || args.ids_dst || args.tile_spec || args.y_scale ||
            args.fusion.gate || args.fusion.x_scale || args.fusion.x_bias || args.fusion.shared_up ||
            args.nchannels_x != 1 || args.nchannels_y != 1 || args.nsamples_x != 1 || args.nsamples_y != 1 ||
            args.ncols_dst < q8pf_min_cols() || args.ncols_dst != args.ncols_y || args.ncols_max != args.ncols_dst ||
            args.ncols_x <= 0 || args.ncols_x % 128 != 0 || args.ncols_x/128 > INT_MAX ||
            args.nrows_x <= 0 || args.nrows_x % q8pf_bm != 0 || (q8pf_max_rows() > 0 && args.nrows_x > q8pf_max_rows()) || args.nrows_x/q8pf_bm > 65535 ||
            args.ncols_dst > INT_MAX - 128 ||
            args.stride_row_x < args.ncols_x/QK8_0 || args.stride_row_x % 4 != 0 ||
            args.nrows_dst < args.nrows_x || uintptr_t(args.x) % 8 != 0 || uintptr_t(args.y) % 16 != 0) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int cc = dev.cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_HOPPER || !ampere_mma_available(cc)) {
        return false;
    }
    const int bn = q8pf_pick_bn(args.nrows_x, args.ncols_dst, dev.nsm, dev.smpbo);
    const size_t smem = bn == 128 ? q8pf_smem<128>() : bn == 64 ? q8pf_smem<64>() : bn == 32 ? q8pf_smem<32>() : SIZE_MAX;
    return bn != 0 && smem <= dev.smpbo;
#else
    GGML_UNUSED(args);
    return false;
#endif
}

void ggml_cuda_mul_mat_q8pf(const mmq_args & args, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    switch (q8pf_pick_bn(args.nrows_x, args.ncols_dst, dev.nsm, dev.smpbo)) {
        case 128: q8pf_launch<128>(args, stream); break;
        case 64:  q8pf_launch<64>(args, stream);  break;
        default:  q8pf_launch<32>(args, stream);  break;
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(args, stream);
    GGML_ABORT("MMQ Q8PF requires Ampere CUDA");
#endif
}
