// [OPT10] cp.async-staged fractional AI+ matvec for decode / verify widths (see aif-async-mmv.cuh). Own TU so the
// default build path (ai4.cu, aif25-mmq.cu) is unchanged; ai4.cu reaches it only with GGML_CUDA_AIF_ASYNC=1.
#include "aif-async-mmv.cuh"
#include "aif-async-addr.h"
#include "magic2-v2.cuh"
#include "../ggml-aif25.h"
#include "../ggml-aif175.h"

#include <algorithm>
#include <cstdlib>

bool ggml_cuda_aif_async_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AIF_ASYNC");
        return e != nullptr && atoi(e) != 0;   // default off
    }();
    return on;
}

namespace aif_async {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define AIF_ASYNC_PRE_AMPERE 1
#endif

static __device__ __forceinline__ void cp_async_ca16(void * smem, const void * gmem) {
#ifdef AIF_ASYNC_PRE_AMPERE
    (void) smem; (void) gmem;
    __trap();
#else
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#ifndef AIF_ASYNC_PRE_AMPERE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N_WAIT>
static __device__ __forceinline__ void cp_async_wait() {
#ifndef AIF_ASYNC_PRE_AMPERE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
#endif
}

// exact for |x| < 2^22; a unit's IMMA sum is 128 int8 x int8 products, |c| <= 2^21, so (float) c == magic_i2f(c)
static __device__ __forceinline__ float magic_i2f(int x) {
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

template <int NB> constexpr int min_blocks = NB <= 2 ? 6 : NB == 3 ? 5 : 4;

// One block = one 16-row strip (blockIdx.x) x TN = 8 NB tokens (blockIdx.y), 4 warps; warp w takes units w, w + 4, ...
// The arithmetic per (row, token) is magic2::mmv's: the same int32 IMMA sum per unit (fragment + mma of
// magic2-v2.cuh on the same tile words), sum += c * d * s in the same order, then (p0 + p1) + (p2 + p3).
template <int B, int V, uint32_t C, int XSH, int NB, bool ROW_SCALE, int SC>
static __global__ void __launch_bounds__(128, min_blocks<NB>) mm(const uint8_t * __restrict__ weights,
        const int8_t * __restrict__ x, const float * __restrict__ xs, float * __restrict__ y, int M, int K, int N) {
    static_assert(XSH == 0 || XSH == 1, "activation scale covers 128 or 256 columns");
    static_assert(!ROW_SCALE || (B == 5 && V == 2 && (SC == 5 || SC == 4)) || (B == 7 && V == 4 && SC == 4),
        "row-scale pair layout");
    using cur_t = cursor<B, V, ROW_SCALE, SC>;
    static_assert(cur_t::UB == magic2::unit_bytes<B, V>, "same unit size as magic2");
    constexpr int UW = cur_t::UB / 4, TW = magic2::tile_words<B, V>, TN = 8 * NB;
    constexpr int PASSES = (cur_t::CHUNKS + 31) / 32;
    __shared__ __align__(16) union {
        uint32_t tiles[2][4][UW];
        float    partial[4][TN][16];
    } smem;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = lane >> 2, col = lane & 3, nu = K / 128;
    const int64_t ms = blockIdx.x, nbase = int64_t(blockIdx.y) * TN;
    float sum[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        sum[nb][0] = sum[nb][1] = sum[nb][2] = sum[nb][3] = 0;
    }
    cur_t cur;
    cur.init(ms, nu, warp);
    auto load = [&](uint32_t * dst) {
#pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            const int i = lane + 32 * p;
            if (i < cur_t::CHUNKS) {
                cp_async_ca16(dst + 4 * i, weights + cur.chunk(i));
            }
        }
    };
    if (warp < nu) {
        load(smem.tiles[0][warp]);
    }
    cp_async_commit();
    int stage = 0;
    for (int u = warp; u < nu; u += 4) {
        if (u + 4 < nu) {
            cur.advance();
            load(smem.tiles[stage ^ 1][warp]);   // read by this warp two steps ago; the __syncwarp below closed it
        }
        cp_async_commit();
        cp_async_wait<1>();                      // this unit's group has landed (the next one may be in flight)
        __syncwarp();
        const uint32_t * tile = smem.tiles[stage][warp];
        int c[NB][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            c[nb][0] = c[nb][1] = c[nb][2] = c[nb][3] = 0;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t * a0 = tile + 8 + (4 * j + (col >> 1)) * TW;
            const uint32_t * a1 = a0 + 2 * TW;
            const uint32_t a[4] = {magic2::fragment<B,V,C>(a0, row, col & 1), magic2::fragment<B,V,C>(a0, row + 8, col & 1),
                magic2::fragment<B,V,C>(a1, row, col & 1), magic2::fragment<B,V,C>(a1, row + 8, col & 1)};
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const int64_t t = nbase + 8 * nb + row;
                uint32_t b0 = 0, b1 = 0;
                if (t < N) {
                    const int8_t * p = x + t * K + int64_t(u) * 128 + j * 32 + col * 4;
                    b0 = __ldg(reinterpret_cast<const uint32_t *>(p));
                    b1 = __ldg(reinterpret_cast<const uint32_t *>(p + 16));
                }
                magic2::mma(c[nb], a, b0, b1);
            }
        }
        const __half * scales = reinterpret_cast<const __half *>(tile);
        const float d0 = __half2float(scales[row]), d1 = __half2float(scales[row + 8]);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int64_t t = nbase + 8 * nb + 2 * col;
            const float s0 = t < N ? xs[t * (nu >> XSH) + (u >> XSH)] : 0;
            const float s1 = t + 1 < N ? xs[(t + 1) * (nu >> XSH) + (u >> XSH)] : 0;
            sum[nb][0] += magic_i2f(c[nb][0]) * d0 * s0; sum[nb][1] += magic_i2f(c[nb][1]) * d0 * s1;
            sum[nb][2] += magic_i2f(c[nb][2]) * d1 * s0; sum[nb][3] += magic_i2f(c[nb][3]) * d1 * s1;
        }
        __syncwarp();
        stage ^= 1;
    }
    cp_async_wait<0>();
    __syncthreads();   // every warp is done with the tiles before the partial sums overwrite them
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        smem.partial[warp][8 * nb + 2 * col][row] = sum[nb][0]; smem.partial[warp][8 * nb + 2 * col + 1][row] = sum[nb][1];
        smem.partial[warp][8 * nb + 2 * col][row + 8] = sum[nb][2]; smem.partial[warp][8 * nb + 2 * col + 1][row + 8] = sum[nb][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TN * 16; i += 128) {
        const int n = i / 16, r = i % 16;
        const int64_t m = ms * 16 + r, t = nbase + n;
        if (m < M && t < N) y[t * M + m] =
            (smem.partial[0][n][r] + smem.partial[1][n][r]) + (smem.partial[2][n][r] + smem.partial[3][n][r]);
    }
}

template <int XSH, int B = 5, int V = 2, uint32_t C = AIF25_MUL, bool ROW_SCALE = false, int SC = 5>
static bool launch(int nb, const uint8_t * w, const int8_t * q, const float * scales, float * y, int M, int K, int64_t N,
                   cudaStream_t stream) {
    if (nb < 1 || nb > 4 || (nb > 1 && N > 8 * nb)) {
        return false;
    }
    // grid.y <= 65535: nb > 1 is one tile; nb = 1 splits N like aif25_mmv in ai4.cu
    const int64_t step = int64_t(8) * nb * 65535;
    for (int64_t n0 = 0; n0 < N; n0 += step) {
        const int nr = (int) std::min<int64_t>(N - n0, step);
        const dim3 grid((unsigned) (M / 16), (unsigned) ((nr + 8 * nb - 1) / (8 * nb)));
        const float * xs = scales + (size_t) n0 * (K >> (7 + XSH));
        const int8_t * qn = q + (size_t) n0 * K;
        float * yn = y + (size_t) n0 * M;
#define AIF_ASYNC_LAUNCH(NB) mm<B,V,C,XSH,NB,ROW_SCALE,SC><<<grid,128,0,stream>>>(w, qn, xs, yn, M, K, nr)
        switch (nb) {
            case 1: AIF_ASYNC_LAUNCH(1); break;
            case 2: AIF_ASYNC_LAUNCH(2); break;
            case 3: AIF_ASYNC_LAUNCH(3); break;
            default: AIF_ASYNC_LAUNCH(4); break;
        }
#undef AIF_ASYNC_LAUNCH
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

} // namespace aif_async

bool ggml_cuda_aif_async_mm(ggml_type type, int xsh, int nb, const uint8_t * w, const int8_t * q, const float * scales,
                            float * y, int M, int K, int64_t N, cudaStream_t stream) {
    if (N < 1 || N > INT32_MAX || M % 16 != 0 || K % 128 != 0) {
        return false;
    }
    // the template arguments of each type's mmv call in ai4.cu (aif25_mmv_type)
    switch (type) {
        case GGML_TYPE_AIF25_U:
            return xsh ? aif_async::launch<1>(nb, w, q, scales, y, M, K, N, stream)
                       : aif_async::launch<0>(nb, w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2:
            if (K % 640 != 0) return false;
            return xsh ? aif_async::launch<1, 5, 2, AIF25_MUL, true, 5>(nb, w, q, scales, y, M, K, N, stream)
                       : aif_async::launch<0, 5, 2, AIF25_MUL, true, 5>(nb, w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2_512:
            return K % 512 == 0 && xsh && aif_async::launch<1, 5, 2, AIF25_MUL, true, 4>(nb, w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF175_U2:
            return K % 512 == 0 && xsh && aif_async::launch<1, 7, 4, AIF175_MUL, true, 4>(nb, w, q, scales, y, M, K, N, stream);
        default:
            return false;
    }
}
