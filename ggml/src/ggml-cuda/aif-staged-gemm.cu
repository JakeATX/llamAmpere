// [K126-4] cp.async-staged prompt GEMM for the fractional AI+ types (AIF25_U, AIF25_U2, AIF25_U2_512, AIF175_U2),
// laid out like ai4k::gemm: a CTA covers MS 16-row slabs x BN tokens; each 128-column weight unit of the MS slabs, the
// BN x 128 activation bytes and the BN activation scales go through a STAGES-deep cp.async ring in shared memory.
// The NW warps of one slab split the unit's A-fragment decode (4/NW of the 4 k-steps each) and share it through shared
// memory, so every weight unit is decoded once per CTA; activations are read from shared memory (padded 144-byte rows,
// conflict-free) instead of global memory per IMMA.
//
// Exactness: magic2::mmv gives each (row, token) four float partial sums, one per warp w, over the units u = w, w + 4, ...
// in increasing u, each step sum += c * d * s (c = the exact int32 IMMA sum of the unit), then (p0 + p1) + (p2 + p3).
// Here every thread keeps the same four partials in registers (acc[u & 3], the unit loop unrolled by 4, so K % 512 == 0),
// adds the same int32 unit sums with the same expression in the same unit order, and ends with the same (p0 + p1) +
// (p2 + p3). The weight decode is magic2::fragment on the same unit bytes. So y is bit-identical to mmv (and to mmq).
#include "aif-staged-gemm.cuh"
#include "magic2-v2.cuh"
#include "../ggml-aif25.h"
#include "../ggml-aif175.h"

#include <atomic>
#include <cstdlib>

namespace aifsg {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define AIFSG_PRE_AMPERE 1
#endif

static __device__ __forceinline__ void cp16(void * smem, const void * gmem, bool pred) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
#ifdef AIFSG_PRE_AMPERE
    __trap(); (void) s; (void) n; (void) gmem;
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}

static __device__ __forceinline__ void cp4(void * smem, const void * gmem, bool pred) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = pred ? 4 : 0;
#ifdef AIFSG_PRE_AMPERE
    __trap(); (void) s; (void) n; (void) gmem;
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}

static __device__ __forceinline__ void commit() {
#ifndef AIFSG_PRE_AMPERE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int n> static __device__ __forceinline__ void wait() {
#ifndef AIFSG_PRE_AMPERE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
#endif
}

constexpr int XST = 144;   // padded activation row (128 B + 16): lanes (g, t) hit 32 distinct banks

template <int B, int V, int MS, int BN>
constexpr __host__ __device__ int stage_bytes() { return MS*magic2::unit_bytes<B, V> + BN*XST + BN*4; }

template <int B, int V, int MS, int NW, int BN, int STAGES>
constexpr __host__ __device__ int smem_bytes() { return STAGES*stage_bytes<B, V, MS, BN>() + (NW > 1 ? MS*4*32*16 : 0); }

template <int B, int V, uint32_t C, int XSH, bool RS, int SC, int MS, int NW, int BN, int STAGES>
static __global__ void __launch_bounds__(MS*NW*32) gemm(const uint8_t * __restrict__ W, const int8_t * __restrict__ X,
        const float * __restrict__ XS, float * __restrict__ Y, int M, int K, int N) {
    constexpr int UB = magic2::unit_bytes<B, V>, TW = magic2::tile_words<B, V>, UP = UB/16;
    constexpr int NT = MS*NW*32, BNW = BN/NW, NTI = BNW/8;
    constexpr int SB = stage_bytes<B, V, MS, BN>();
    static_assert(UB % 16 == 0 && BN % (8*NW) == 0 && 4 % NW == 0, "tile shape");
    static_assert(!RS || (B == 5 && V == 2 && (SC == 5 || SC == 4)) || (B == 7 && V == 4 && SC == 4), "row-scale layout");
    extern __shared__ __align__(16) uint8_t smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int mw = warp / NW, nw = warp % NW;
    const int nslab = M/16, nu = K/128, xsn = nu >> XSH;
    const int slab0 = blockIdx.x*MS;
    const int64_t n0 = int64_t(blockIdx.y)*BN;

    auto load_stage = [&](int u) {
        uint8_t * st = smem + (u % STAGES)*SB;
        for (int i = tid; i < MS*UP; i += NT) {
            const int s = i / UP, p = i % UP;
            const int64_t ms = slab0 + s;
            const bool ok = ms < nslab;
            const uint8_t * src = W;
            if (ok) {
                if constexpr (RS) {
                    // block = [16 row scales, 32 B][SC payloads of UB - 32 B]; the unit is the header + payload u % SC
                    const uint8_t * block = W + (ms*(nu/SC) + u/SC)*(int64_t) (32 + SC*(UB - 32));
                    src = p < 2 ? block + p*16 : block + 32 + (u % SC)*(UB - 32) + (p - 2)*16;
                } else {
                    src = W + (ms*nu + u)*(int64_t) UB + p*16;
                }
            }
            cp16(st + s*UB + p*16, src, ok);
        }
        uint8_t * sx = st + MS*UB;
        for (int i = tid; i < BN*8; i += NT) {
            const int c = i >> 3, p = i & 7;
            const bool ok = n0 + c < N;
            cp16(sx + c*XST + p*16, ok ? (const void *) (X + (n0 + c)*K + u*128 + p*16) : (const void *) X, ok);
        }
        float * sxs = (float *) (sx + BN*XST);
        for (int i = tid; i < BN; i += NT) {
            const bool ok = n0 + i < N;
            cp4(sxs + i, ok ? (const void *) (XS + (n0 + i)*xsn + (u >> XSH)) : (const void *) XS, ok);
        }
    };

    float acc[4][NTI][4];
#pragma unroll
    for (int ph = 0; ph < 4; ++ph)
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[ph][nt][i] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nu) load_stage(s);
        commit();
    }

    uint4 * dec = (uint4 *) (smem + STAGES*SB);   // NW > 1: [MS][4 k-steps][32 lanes] decoded A fragments
    (void) dec;

    for (int u0 = 0; u0 < nu; u0 += 4) {
#pragma unroll
        for (int ph = 0; ph < 4; ++ph) {
            const int u = u0 + ph;
            wait<STAGES - 2>();
            __syncthreads();
            if (u + STAGES - 1 < nu) load_stage(u + STAGES - 1);
            commit();

            const uint8_t  * st   = smem + (u % STAGES)*SB;
            const uint32_t * tile = (const uint32_t *) (st + mw*UB);
            const uint8_t  * sx   = st + MS*UB;
            const float    * sxs  = (const float *) (sx + BN*XST);

            uint32_t a[4][4];
            if constexpr (NW == 1) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t * a0 = tile + 8 + (4*j + (t >> 1))*TW;
                    const uint32_t * a1 = a0 + 2*TW;
                    a[j][0] = magic2::fragment<B,V,C>(a0, g, t & 1);
                    a[j][1] = magic2::fragment<B,V,C>(a0, g + 8, t & 1);
                    a[j][2] = magic2::fragment<B,V,C>(a1, g, t & 1);
                    a[j][3] = magic2::fragment<B,V,C>(a1, g + 8, t & 1);
                }
            } else {
#pragma unroll
                for (int jj = 0; jj < 4/NW; ++jj) {
                    const int j = jj*NW + nw;
                    const uint32_t * a0 = tile + 8 + (4*j + (t >> 1))*TW;
                    const uint32_t * a1 = a0 + 2*TW;
                    uint4 v;
                    v.x = magic2::fragment<B,V,C>(a0, g, t & 1);
                    v.y = magic2::fragment<B,V,C>(a0, g + 8, t & 1);
                    v.z = magic2::fragment<B,V,C>(a1, g, t & 1);
                    v.w = magic2::fragment<B,V,C>(a1, g + 8, t & 1);
                    dec[(mw*4 + j)*32 + lane] = v;
                }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4 v = dec[(mw*4 + j)*32 + lane];
                    a[j][0] = v.x; a[j][1] = v.y; a[j][2] = v.z; a[j][3] = v.w;
                }
            }
            const __half * hs = (const __half *) tile;
            const float d0 = __half2float(hs[g]), d1 = __half2float(hs[g + 8]);
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                const int cb = nw*BNW + nt*8;
                const uint8_t * xr = sx + (cb + g)*XST + t*4;
                int c[4] = {0, 0, 0, 0};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t b0 = *(const uint32_t *) (xr + j*32);
                    const uint32_t b1 = *(const uint32_t *) (xr + j*32 + 16);
                    magic2::mma(c, a[j], b0, b1);
                }
                const float s0 = sxs[cb + 2*t], s1 = sxs[cb + 2*t + 1];
                // the magic2::mmv expression, term for term
                acc[ph][nt][0] += c[0] * d0 * s0; acc[ph][nt][1] += c[1] * d0 * s1;
                acc[ph][nt][2] += c[2] * d1 * s0; acc[ph][nt][3] += c[3] * d1 * s1;
            }
        }
    }
    wait<0>();

    if (slab0 + mw >= nslab) {
        return;
    }
    const int64_t r = int64_t(slab0 + mw)*16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int64_t tk = n0 + nw*BNW + nt*8 + 2*t;
        if (tk < N) {
            Y[tk*M + r]     = (acc[0][nt][0] + acc[1][nt][0]) + (acc[2][nt][0] + acc[3][nt][0]);
            Y[tk*M + r + 8] = (acc[0][nt][2] + acc[1][nt][2]) + (acc[2][nt][2] + acc[3][nt][2]);
        }
        if (tk + 1 < N) {
            Y[(tk + 1)*M + r]     = (acc[0][nt][1] + acc[1][nt][1]) + (acc[2][nt][1] + acc[3][nt][1]);
            Y[(tk + 1)*M + r + 8] = (acc[0][nt][3] + acc[1][nt][3]) + (acc[2][nt][3] + acc[3][nt][3]);
        }
    }
}

template <int B, int V, uint32_t C, int XSH, bool RS, int SC, int MS, int NW, int BN, int STAGES>
static bool launch(const uint8_t * w, const int8_t * q, const float * scales, float * y, int M, int K, int64_t N,
                   cudaStream_t stream) {
    constexpr int smem = smem_bytes<B, V, MS, NW, BN, STAGES>();
    auto kern = gemm<B, V, C, XSH, RS, SC, MS, NW, BN, STAGES>;
    static std::atomic<bool> attr_set[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    if (!attr_set[dev].load()) {
        CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev].store(true);
    }
    const int64_t gy = (N + BN - 1)/BN;
    if (gy > 65535) {
        return false;
    }
    const dim3 grid((unsigned) ((M/16 + MS - 1)/MS), (unsigned) gy);
    kern<<<grid, MS*NW*32, smem, stream>>>(w, q, scales, y, M, K, (int) N);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int B, int V, uint32_t C, int XSH, bool RS, int SC>
static bool launch_cfg(const uint8_t * w, const int8_t * q, const float * scales, float * y, int M, int K, int64_t N,
                       cudaStream_t stream) {
    static const int cfg = [] { const char * e = getenv("GGML_CUDA_AIF_STAGED_CFG"); return e ? atoi(e) : 0; }();
    switch (cfg) {
        case 1:  return launch<B, V, C, XSH, RS, SC, 8, 2,  64, 3>(w, q, scales, y, M, K, N, stream);
        case 2:  return launch<B, V, C, XSH, RS, SC, 4, 4, 128, 3>(w, q, scales, y, M, K, N, stream);
        default: return launch<B, V, C, XSH, RS, SC, 4, 2,  64, 4>(w, q, scales, y, M, K, N, stream);
    }
}

} // namespace aifsg

bool ggml_cuda_aif_staged_gemm_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AIF_STAGED_GEMM");
        const bool v = e != nullptr && atoi(e) != 0;
        if (v) {
            GGML_LOG_INFO("[AIF_STAGED] GGML_CUDA_AIF_STAGED_GEMM=1: fractional AI+ prompt widths take the staged GEMM\n");
        }
        return v;
    }();
    return on;
}

bool ggml_cuda_aif_staged_gemm(ggml_type type, int xsh, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                               int M, int K, int64_t N, cudaStream_t stream) {
    // the unit loop is unrolled by the four mmv phases: K % 512 == 0 (every 27B dense shape)
    // (so xsh == 1: one activation scale per 256 columns)
    if (K % 512 != 0 || M % 16 != 0 || N < 1 || N > INT32_MAX || !xsh) {
        return false;
    }
    // same template arguments as each type's magic2::mmv call in ggml_cuda_mul_mat_aif25
    switch (type) {
        case GGML_TYPE_AIF25_U:
            return aifsg::launch_cfg<5, 2, AIF25_MUL, 1, false, 5>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2:
            return K % 640 == 0 && aifsg::launch_cfg<5, 2, AIF25_MUL, 1, true, 5>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF25_U2_512:
            return aifsg::launch_cfg<5, 2, AIF25_MUL, 1, true, 4>(w, q, scales, y, M, K, N, stream);
        case GGML_TYPE_AIF175_U2:
            return aifsg::launch_cfg<7, 4, AIF175_MUL, 1, true, 4>(w, q, scales, y, M, K, N, stream);
        default:
            return false;
    }
}
