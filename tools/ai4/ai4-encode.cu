// AI4 (Ampere INT4, formerly D2) GPU encoder: the same algorithm as ai4_quantize_slab (ggml/src/ggml-ai4.h) with fp16 path costs.
//
// One CTA per (slab of 16 rows, 256-wide K block); 512 threads. The 32 column steps (tiles of 16 rows x 8 K) run in
// order, each tile through two tail-biting Viterbi passes over the 65536-state trellis, then the block-LDLQ update
// of the later columns.
//
// Trellis layout: thread j owns the states t | hi << 12 for t in 8j .. 8j+7 and hi in 0..15 (128 states, fp16 costs
// as 64 half2 registers). The predecessor of state s is any state with low 12 bits s >> 4, so the min over hi is
// local to the thread that owns t, and the new cost of s' = hi' << 12 | t' needs m[hi' << 8 | t' >> 4]: for one
// thread t' >> 4 = j >> 1 for all its 8 t', so it reads 16 m values from shared memory per step. Levels are constant
// per register slot (biased bytes, 32 registers). Costs are renormalized by the block min every step.
//
// Decisions (4-bit argmin hi per t) go to a 32-step shared window (64 KB). Pass 1 (free start, on the tile rolled by
// AI4_ROLL = 31 so original index 127 sits at rolled 96) keeps steps 96..127 and traces back to rolled 96, which gives
// the wrap-around state t0. Pass 2 (start and end forced to t0) traces back by checkpointed re-runs: forward 0..127
// keeps 96..127, then 0..95 keeps 64..95, 0..63, 0..31 (320 steps). A tile costs 448 forward steps.
#include "../../ggml/src/ggml-cuda/ai4.cuh"
#include "ai4-encode.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define ENC_NT  512
#define ENC_WIN 32
#define ENC_ROLL 31

struct enc_smem {
    uint32_t dec[ENC_WIN][ENC_NT];   // 64 KB
    __half2  mb[2][2048];            // 2 x 4096 halves (double-buffered min over hi)
    float    y[16][256];             // rotated weights of the slab block, LDLQ-updated in place
    float    tgt[128];               // current tile targets in level units (row-major n = r*8 + c)
    float    ebuf[128];              // LDLQ errors of the current tile
    float    wmin[2][16];
    float    redv[16];
    int      redi[16];
    uint16_t st[128];
    int      bests;
    int      t0;
};

static __device__ __forceinline__ __half2 enc_lev2(const uint32_t (&Lp)[16][2], const int hi, const int p) {
    const uint32_t x = Lp[hi][p >> 1];
    const uint32_t b = __byte_perm(x, 0x64u, (p & 1) ? 0x4342u : 0x4140u);   // 0x64XX: fp16 1024 + byte
    const __half2 h = *reinterpret_cast<const __half2 *>(&b);
    return __hsub2(h, __float2half2_rn(1152.0f));                              // byte - 128 = level, exact
}

// Forward Viterbi over steps 0..E of the current tile targets (rolled by R). t0 >= 0 forces the start state's top 12
// bits to t0. Decisions of steps W0..E go to dec[i - W0]. endmode 1: S.bests = global argmin state at step E (lowest
// state on ties); endmode 2: S.bests = argmin state among those with low 12 bits == t0.
static __device__ __forceinline__ void enc_viterbi(enc_smem & S, const uint32_t (&Lp)[16][2], __half2 (&c)[16][4],
                                   const int R, const int t0, const int E, const int W0, const int endmode) {
    const int j    = threadIdx.x;
    const int lane = j & 31;
    const int warp = j >> 5;
    const __half2 hinf = __float2half2_rn(INFINITY);

    {
        const __half2 y2 = __float2half2_rn(S.tgt[R & 127]);
#pragma unroll
        for (int hi = 0; hi < 16; ++hi) {
            const bool off = t0 >= 0 && (((hi << 8) | (j >> 1)) != t0);
            const __half2 mv = off ? hinf : __float2half2_rn(0.0f);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const __half2 e = __hsub2(enc_lev2(Lp, hi, p), y2);
                c[hi][p] = __hfma2(e, e, mv);
            }
        }
    }

    for (int i = 1; i <= E; ++i) {
        const int buf = i & 1;
        // A: min over hi with argmin, per t pair
        float    lmin = INFINITY;
        uint32_t dw   = 0;
        uint32_t mw[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            __half2  mn  = c[0][p];
            uint32_t arg = 0;
#pragma unroll
            for (int hi = 1; hi < 16; ++hi) {
                const uint32_t mask = __hlt2_mask(c[hi][p], mn);
                mn  = __hmin2(mn, c[hi][p]);
                arg = (arg & ~mask) | ((uint32_t) hi * 0x00010001u & mask);
            }
            dw |= ((arg & 0xFu) << (8 * p)) | (((arg >> 16) & 0xFu) << (8 * p + 4));
            mw[p] = *reinterpret_cast<const uint32_t *>(&mn);
            lmin = fminf(lmin, fminf(__low2float(mn), __high2float(mn)));
        }
        reinterpret_cast<uint4 *>(S.mb[buf])[j] = make_uint4(mw[0], mw[1], mw[2], mw[3]);
        if (i >= W0) {
            S.dec[i - W0][j] = dw;
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            lmin = fminf(lmin, __shfl_xor_sync(0xffffffffu, lmin, m));
        }
        if (lane == 0) {
            S.wmin[buf][warp] = lmin;
        }
        __syncthreads();
        // C: new costs
        float gmin = S.wmin[buf][0];
#pragma unroll
        for (int w = 1; w < 16; ++w) {
            gmin = fminf(gmin, S.wmin[buf][w]);
        }
        const __half  gminh = __float2half_rn(gmin);   // exact: gmin is an fp16 value
        const __half2 y2    = __float2half2_rn(S.tgt[(i + R) & 127]);
        const __half * mbh  = reinterpret_cast<const __half *>(S.mb[buf]);
#pragma unroll
        for (int hi = 0; hi < 16; ++hi) {
            const __half2 mv = __half2half2(__hsub(mbh[(hi << 8) | (j >> 1)], gminh));
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const __half2 e = __hsub2(enc_lev2(Lp, hi, p), y2);
                c[hi][p] = __hfma2(e, e, mv);
            }
        }
    }

    if (endmode == 1) {
        float bv = INFINITY;
        int   bi = 0x7fffffff;
#pragma unroll
        for (int hi = 0; hi < 16; ++hi) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float a = __low2float(c[hi][p]), b = __high2float(c[hi][p]);
                const int   s = (hi << 12) | (8 * j + 2 * p);
                if (a < bv) { bv = a; bi = s; }
                if (b < bv) { bv = b; bi = s + 1; }
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            S.redv[warp] = bv;
            S.redi[warp] = bi;
        }
        __syncthreads();
        if (j == 0) {
            float v = S.redv[0];
            int   s = S.redi[0];
            for (int w = 1; w < 16; ++w) {
                if (S.redv[w] < v || (S.redv[w] == v && S.redi[w] < s)) { v = S.redv[w]; s = S.redi[w]; }
            }
            S.bests = s & 0xFFFF;   // defensive: all-inf costs leave s = 0x7fffffff
        }
    } else if (endmode == 2) {
        if (j == (t0 >> 3)) {
            const int u = t0 & 7;
            float bv = INFINITY;
            int   bh = 0;
#pragma unroll
            for (int hi = 0; hi < 16; ++hi) {
                float v = 0.0f;
#pragma unroll
                for (int p = 0; p < 4; ++p) {   // static register indexing only
                    if (p == (u >> 1)) {
                        v = (u & 1) ? __high2float(c[hi][p]) : __low2float(c[hi][p]);
                    }
                }
                if (v < bv) { bv = v; bh = hi; }
            }
            S.bests = (bh << 12) | t0;
        }
    }
    __syncthreads();
}

// trace back from state s at step E through dec[E - W0 .. lo - W0], writing st[i] for i = E .. lo when st != null;
// returns the state at step lo - 1
static __device__ __forceinline__ int enc_trace(const enc_smem & S, int s, const int E, const int lo, const int W0,
                                                uint16_t * st) {
    for (int i = E; i >= lo; --i) {
        if (st) {
            st[i] = (uint16_t) s;
        }
        const int t  = s >> 4;
        const int hi = (int) ((S.dec[i - W0][t >> 3] >> (4 * (t & 7))) & 0xFu);
        s = t | (hi << 12);
    }
    return s;
}

// In-warp RB-point FWHT, lane holds elements lane*8 + q (RB = 256: the whole warp; RB = 128: each half-warp independently).
template <int RB>
static __device__ __forceinline__ void enc_fwht(float (&v)[8]) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            if ((q & h) == 0) {
                const float a = v[q], b = v[q + h];
                v[q] = a + b;
                v[q + h] = a - b;
            }
        }
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < RB / 8; m <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float o = __shfl_xor_sync(0xffffffffu, v[q], m);
            v[q] = (lane & m) ? (o - v[q]) : (v[q] + o);
        }
    }
}

// RB: rotation / LDLQ block (256 or 128). UNIT: 0 = AI4 1024 super-block layout (RB must be 256), 1 = AI4_U unit-128 layout.
// RB = 256, UNIT = 0 is the original AI4 encoder, arithmetic unchanged.
template <int RB, int UNIT>
static __global__ void __launch_bounds__(ENC_NT, 1) k_ai4_encode(const float * __restrict__ W, const int64_t K,
        const float * __restrict__ G, const uint8_t * __restrict__ gok, const float lrms, const float smul,
        uint8_t * __restrict__ out, const int hb0, const int prerot, float * __restrict__ qr) {
    // hb0 / prerot / qr (AIQ full-K LDLQ, RB = 256 only): with prerot W already holds rotated (and cross-block
    // LDLQ-updated) weights and is not rotated again; hb = hb0 + blockIdx.x; G is indexed by blockIdx.x; qr (optional)
    // receives the rotated dequantized values of the encoded block, [M][RB] row-major.
    extern __shared__ __align__(16) uint8_t enc_smem_raw[];
    enc_smem & S = *reinterpret_cast<enc_smem *>(enc_smem_raw);
    constexpr int NCS = RB / 8;   // column steps (tiles) per block

    const int j = threadIdx.x, r = j >> 5, l = j & 31;
    const int hb = hb0 + blockIdx.x, slab = blockIdx.y;
    const int gb = prerot ? blockIdx.x : hb;
    const bool lact = l < NCS;    // lane holds 8 columns of this block

    // byte offset of 128-K chunk kc of this slab
    auto chunk = [&](const int64_t kc) -> uint8_t * {
        if (UNIT) {
            return out + ((int64_t) slab * (K / 128) + kc) * 1056;
        }
        return out + ((int64_t) slab * (K / 1024) + (kc >> 3)) * AI4_DEV_SB_BYTES;
    };

    // load + rotate (w' = (1/sqrt(RB)) H D w), one warp per row
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    {
        if (lact) {
            const float * wr = W + ((int64_t) slab * 16 + r) * K + (int64_t) hb * RB + l * 8;
            const float4 a = reinterpret_cast<const float4 *>(wr)[0];
            const float4 b = reinterpret_cast<const float4 *>(wr)[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        }
        if (!prerot) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (ai4_dev_sign_neg((l * 8 + q) % RB)) {
                    v[q] = -v[q];
                }
            }
            enc_fwht<RB>(v);
        }
        const float nrm = prerot ? 1.0f : RB == 256 ? 1.0f / 16.0f : 0.08838834764831845f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            v[q] *= nrm;
            if (lact) {
                S.y[r][l * 8 + q] = v[q];
            }
        }
        if (!lact) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                v[q] = 0.0f;
            }
        }
    }
    // scale of this thread's 128-group (lanes 0-15: group 0, 16-31: group 1)
    float ss = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        ss += v[q] * v[q];
    }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, m);
    }
    float dv = __half2float(__float2half_rn(sqrtf(ss / 128.0f) / lrms * smul));
    if (!(dv >= 0.0f) || !isfinite(dv)) {
        dv = 0.0f;
    }
    if ((l & 15) == 0 && lact) {
        const int64_t kc = (int64_t) hb * (RB / 128) + (l >> 4);
        if (UNIT) {
            reinterpret_cast<uint16_t *>(chunk(kc))[r] = __half_as_ushort(__float2half_rn(dv));
        } else {
            reinterpret_cast<uint16_t *>(chunk(kc))[r * 8 + (kc & 7)] = __half_as_ushort(__float2half_rn(dv));
        }
    }

    // levels of the owned states, biased bytes
    uint32_t Lp[16][2];
#pragma unroll
    for (int hi = 0; hi < 16; ++hi) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            uint32_t x = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint32_t s = ((uint32_t) hi << 12) | (uint32_t) (8 * j + 4 * w + u);
                x |= (uint32_t) (ai4_dev_level(s) + 128) << (8 * u);
            }
            Lp[hi][w] = x;
        }
    }
    const bool useG = G != nullptr && (gok == nullptr || gok[gb] != 0);
    __half2 c[16][4];
    __syncthreads();

    for (int cs = 0; cs < NCS; ++cs) {
        if (l == cs) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float t = dv > 0.0f ? S.y[r][cs * 8 + q] / dv : 0.0f;
                S.tgt[r * 8 + q] = fminf(fmaxf(t, -250.0f), 250.0f);   // keeps every fp16 cost of the best path finite
            }
        }
        __syncthreads();

        // pass 1: rolled by ENC_ROLL, free start/end -> wrap state t0 (low 12 bits of the state at original 127)
        enc_viterbi(S, Lp, c, ENC_ROLL, -1, 127, 96, 1);
        if (j == 0) {
            const int s96 = enc_trace(S, S.bests, 127, 97, 96, nullptr);
            S.t0 = s96 & 0xFFF;
        }
        __syncthreads();
        const int t0 = S.t0;

        // pass 2: tail-biting, checkpointed traceback
        enc_viterbi(S, Lp, c, 0, t0, 127, 96, 2);
        if (j == 0) {
            S.st[95] = (uint16_t) enc_trace(S, S.bests, 127, 96, 96, S.st);
        }
        __syncthreads();
#pragma unroll 1
        for (int E = 95; E > 0; E -= 32) {
            const int W0 = E - 31;
            enc_viterbi(S, Lp, c, 0, t0, E, W0, 0);
            if (j == 0) {
                const int lo = W0 > 0 ? W0 : 1;
                S.st[lo - 1] = (uint16_t) enc_trace(S, S.st[E], E, lo, W0, S.st);
            }
            __syncthreads();
        }

        if (qr != nullptr && l == cs) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                qr[((int64_t) slab * 16 + r) * RB + cs * 8 + q] = dv * (float) ai4_dev_level(S.st[r * 8 + q]);
            }
        }
        if (j < 16) {
            uint32_t w = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                w |= (uint32_t) (S.st[j * 8 + q] & 0xF) << (4 * (7 - q));
            }
            const int64_t kc = (int64_t) hb * (RB / 128) + cs / 16;
            uint8_t * cb = chunk(kc) + (UNIT ? 32 : AI4_DEV_SCALE_BYTES + (kc & 7) * 1024);
            reinterpret_cast<uint32_t *>(cb)[(cs % 16) * 16 + j] = w;
        }

        if (useG && cs < NCS - 1) {
            if (l == cs) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    S.ebuf[r * 8 + q] = S.y[r][cs * 8 + q] - dv * (float) ai4_dev_level(S.st[r * 8 + q]);
                }
            }
            __syncthreads();
            if (l > cs && lact) {
                const float * g = G + (((int64_t) gb * NCS + cs) * 8) * RB + l * 8;
                float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    const float ea = S.ebuf[r * 8 + a];
                    const float4 g0 = reinterpret_cast<const float4 *>(g + a * RB)[0];
                    const float4 g1 = reinterpret_cast<const float4 *>(g + a * RB)[1];
                    acc[0] += ea * g0.x; acc[1] += ea * g0.y; acc[2] += ea * g0.z; acc[3] += ea * g0.w;
                    acc[4] += ea * g1.x; acc[5] += ea * g1.y; acc[6] += ea * g1.z; acc[7] += ea * g1.w;
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    S.y[r][l * 8 + q] -= acc[q];
                }
            }
        }
        __syncthreads();
    }
}

#define AI4_CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "ai4-encode: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

int ai4_cuda_encode(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                   uint8_t * out, float * ms) {
    return ai4_cuda_encode_layout(W, K, M, G, gok, lrms, smul, 0, out, ms);
}

int ai4_cuda_encode_layout(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                          int unit, uint8_t * out, float * ms) {
    if (K % (unit ? 128 : 1024) != 0 || M % 16 != 0) {
        return 2;
    }
    const int rb = unit ? (K % 256 == 0 ? 256 : 128) : 256;
    const size_t nw = (size_t) K * M, nq = (size_t) (M / 16) * (K / 128) * 1056;
    const size_t ng = (size_t) K * rb;   // [K/rb][rb/8][8][rb] = K * rb floats
    float * dW = nullptr, * dG = nullptr;
    uint8_t * dok = nullptr, * dq = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMemcpy(dW, W, nw * sizeof(float), cudaMemcpyHostToDevice));
    AI4_CU(cudaMemset(dq, 0, nq));
    if (G) {
        AI4_CU(cudaMalloc(&dG, ng * sizeof(float)));
        AI4_CU(cudaMalloc(&dok, K / rb));
        AI4_CU(cudaMemcpy(dG, G, ng * sizeof(float), cudaMemcpyHostToDevice));
        AI4_CU(cudaMemcpy(dok, gok, K / rb, cudaMemcpyHostToDevice));
    }
    const int smem = (int) sizeof(enc_smem);
    auto kern = unit ? (rb == 256 ? k_ai4_encode<256, 1> : k_ai4_encode<128, 1>) : k_ai4_encode<256, 0>;
    AI4_CU(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaEvent_t e0, e1;
    AI4_CU(cudaEventCreate(&e0));
    AI4_CU(cudaEventCreate(&e1));
    AI4_CU(cudaEventRecord(e0));
    const dim3 grid((unsigned) (K / rb), (unsigned) (M / 16));
    kern<<<grid, ENC_NT, smem>>>(dW, K, dG, dok, lrms, smul, dq, 0, 0, nullptr);
    AI4_CU(cudaGetLastError());
    AI4_CU(cudaEventRecord(e1));
    AI4_CU(cudaEventSynchronize(e1));
    if (ms) {
        AI4_CU(cudaEventElapsedTime(ms, e0, e1));
    }
    AI4_CU(cudaMemcpy(out, dq, nq, cudaMemcpyDeviceToHost));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dW);
    cudaFree(dq);
    cudaFree(dG);
    cudaFree(dok);
    return 0;
}

int ai4_cuda_reconstruct(const uint8_t * q, int64_t K, int64_t M, float * W) {
    return ai4_cuda_reconstruct_layout(q, K, M, 0, W);
}

int ai4_cuda_reconstruct_layout(const uint8_t * q, int64_t K, int64_t M, int unit, float * W) {
    const size_t nw = (size_t) K * M, nq = (size_t) (M / 16) * (K / 128) * 1056;
    float * dW = nullptr;
    uint8_t * dq = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMemcpy(dq, q, nq, cudaMemcpyHostToDevice));
    if (unit) {
        ai4u_reconstruct_launch<float>(dq, dW, K, M, 0);
    } else {
        ai4_reconstruct_launch<float>(dq, dW, K, M, 0);
    }
    AI4_CU(cudaGetLastError());
    AI4_CU(cudaDeviceSynchronize());
    AI4_CU(cudaMemcpy(W, dW, nw * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dW);
    cudaFree(dq);
    return 0;
}

// ---------------- AIQ: full-K LDLQ with a full input Hessian (private experiment path) ----------------

// rotate every 256 block of every row in place: x' = (1/16) H256 D x (the encoder's rotation). One warp per (row, block).
static __global__ void k_aiq_rot(float * __restrict__ A, const int64_t K) {
    const int l = threadIdx.x;
    float * p = A + (int64_t) blockIdx.y * K + (int64_t) blockIdx.x * 256 + l * 8;
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] = ai4_dev_sign_neg(l * 8 + q) ? -p[q] : p[q];
    }
    ai4_dev_fwht256(v);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        p[q] = v[q] * (1.0f / 16.0f);
    }
}

// mode 1: zero everything outside the 256 diagonal blocks; mode 2: keep the diagonal only; mode 3: zero the
// strictly-lower part of a row-major matrix (row-major index i*K + j, j < i).
static __global__ void k_aiq_mask(float * __restrict__ A, const int64_t K, const int mode) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * K) return;
    const int64_t i = idx / K, j = idx % K;
    bool z = false;
    if (mode == 1) z = (i >> 8) != (j >> 8);
    else if (mode == 2) z = i != j;
    else if (mode == 3) z = j < i;
    if (z) A[idx] = 0.0f;
}

static __global__ void k_aiq_adddiag(float * __restrict__ A, const int64_t K, const float v) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) A[i * K + i] += v;
}

// D[r][c] = Y[r][b*256 + c] - Qr[r][c]; also Q[r][b*256 + c] = Qr[r][c] (rotated reconstruction, for the error report)
static __global__ void k_aiq_diff(const float * __restrict__ Y, const float * __restrict__ Qr, float * __restrict__ D,
                                  float * __restrict__ Q, const int64_t K, const int b, const int64_t M) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * 256) return;
    const int64_t r = idx / 256, c = idx % 256;
    D[idx] = Y[r * K + (int64_t) b * 256 + c] - Qr[idx];
    Q[r * K + (int64_t) b * 256 + c] = Qr[idx];
}

#define AIQ_CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "aiq: %s failed: %d (%s:%d)\n", #x, (int) e_, __FILE__, __LINE__); return 1; } } while (0)
#define AIQ_CS(x) do { cusolverStatus_t e_ = (x); if (e_ != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "aiq: %s failed: %d (%s:%d)\n", #x, (int) e_, __FILE__, __LINE__); return 1; } } while (0)

static void aiq_inv_upper8(const double * A, double * X) {
    for (int i = 0; i < 64; ++i) X[i] = 0.0;
    for (int j = 0; j < 8; ++j) {
        X[j * 8 + j] = 1.0 / A[j * 8 + j];
        for (int i = j - 1; i >= 0; --i) {
            double s = 0.0;
            for (int k = i + 1; k <= j; ++k) s += A[i * 8 + k] * X[k * 8 + j];
            X[i * 8 + j] = -s / A[i * 8 + i];
        }
    }
}

// H-weighted relative error tr((W - Wh) H (W - Wh)^T) / tr(W H W^T), original basis, all on the GPU.
int ai4_cuda_werr(const float * W, const float * Wh, const float * H, int64_t K, int64_t M, double * werr) {
    const size_t nw = (size_t) K * M, nh = (size_t) K * K;
    float * dW = nullptr, * dE = nullptr, * dH = nullptr, * dT = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dE, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dT, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dH, nh * sizeof(float)));
    AI4_CU(cudaMemcpy(dW, W, nw * sizeof(float), cudaMemcpyHostToDevice));
    AI4_CU(cudaMemcpy(dE, Wh, nw * sizeof(float), cudaMemcpyHostToDevice));
    AI4_CU(cudaMemcpy(dH, H, nh * sizeof(float), cudaMemcpyHostToDevice));
    cublasHandle_t cb;
    AIQ_CB(cublasCreate(&cb));
    const float one = 1.0f, mone = -1.0f, zero = 0.0f;
    AIQ_CB(cublasSgeam(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, &one, dW, (int) K, &mone, dE, (int) K, dE, (int) K));
    float num = 0.0f, den = 0.0f;
    AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH, (int) K, dE, (int) K, &zero, dT, (int) K));
    AIQ_CB(cublasSdot(cb, (int) nw, dE, 1, dT, 1, &num));
    AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH, (int) K, dW, (int) K, &zero, dT, (int) K));
    AIQ_CB(cublasSdot(cb, (int) nw, dW, 1, dT, 1, &den));
    *werr = (double) num / (double) den;
    cublasDestroy(cb);
    cudaFree(dW); cudaFree(dE); cudaFree(dT); cudaFree(dH);
    return 0;
}

// =====================================================================================================================
// AI+ rate-KB encoder (unit-128 layout; AI5_U: KB = 5). The AI4 encoder generalized to 2^KB predecessors per state:
// the predecessor of state s is any state with low TB = 16 - KB bits s >> KB. Thread j owns the states t | hi << TB for
// t in TPT*j .. TPT*j + TPT - 1 (TPT = 2^TB / 512) and hi in 0..2^KB - 1 (always 128 states, 64 half2 cost registers).
// The new cost of s' = hi' << TB | t' needs m[hi' << (TB - KB) | t' >> KB], and t' >> KB is constant per thread, so a thread
// reads 2^KB m values per step. Decisions: KB bits per t, TPT*KB <= 32 bits per thread and step. KB = 4 reproduces
// k_ai4_encode<RB, 1> byte for byte (checked by --selftest-rate).
// =====================================================================================================================
template <int KB>
struct aik_geom {
    static constexpr int TB  = 16 - KB;
    static constexpr int NT  = 1 << TB;      // predecessor classes
    static constexpr int NHI = 1 << KB;
    static constexpr int TPT = NT / ENC_NT;  // t values per thread (32/16/8/4/2 for KB = 2/3/4/5/6)
    static constexpr int NP  = TPT / 2;      // half2 per hi (NHI*NP == 64 at every rate)
    static constexpr int NL  = (TPT + 3) / 4;// level words per hi (4 biased bytes each; KB = 6 uses 2 of the 4)
    static constexpr int UB  = 32 + 256 * KB;
    static constexpr int TW  = 4 * KB;       // tile words
    // Decisions: KB bits per t, SPW symbols per 32-bit word, DW words per thread and step (KB <= 3 needs two).
    static constexpr int SPW = 32 / KB;
    static constexpr int DW  = (TPT + SPW - 1) / SPW;
    // Decision window. The min-over-hi buffer is 2*NT halves, which grows as 2^(16-KB), so the window shrinks at the low
    // rates to stay inside the 100 KB SM86 dynamic shared-memory limit (all rates land at <= 99,852 B, the KB = 4 size).
    static constexpr int WIN  = KB >= 4 ? 32 : (KB == 3 ? 8 : 4);
    static constexpr int ROLL = WIN - 1;     // tail-biting roll: original index 127 at rolled index 128 - WIN (== aik_roll)
};

// Shared memory of one rate-KB encoder CTA (the AI4 enc_smem generalized; KB = 4 has the same layout and size).
template <int KB>
struct aik_smem {
    uint32_t dec[aik_geom<KB>::WIN][aik_geom<KB>::DW][ENC_NT];
    __half2  mb[2][aik_geom<KB>::NT / 2];   // NT halves per buffer (double-buffered min over hi)
    float    y[16][256];
    float    tgt[128];
    float    ebuf[128];
    float    wmin[2][16];
    float    redv[16];
    int      redi[16];
    uint16_t st[128];
    int      bests;
    int      t0;
};

template <int KB>
static __device__ __forceinline__ __half2 aik_lev2(const uint32_t (&Lp)[aik_geom<KB>::NHI][aik_geom<KB>::NL], const int hi, const int p) {
    const uint32_t x = Lp[hi][p >> 1];
    const uint32_t b = __byte_perm(x, 0x64u, (p & 1) ? 0x4342u : 0x4140u);
    const __half2 h = *reinterpret_cast<const __half2 *>(&b);
    return __hsub2(h, __float2half2_rn(1152.0f));
}

template <int KB>
static __device__ __forceinline__ void aik_viterbi(aik_smem<KB> & S, const uint32_t (&Lp)[aik_geom<KB>::NHI][aik_geom<KB>::NL],
                                   __half2 (&c)[aik_geom<KB>::NHI][aik_geom<KB>::NP],
                                   const int R, const int t0, const int E, const int W0, const int endmode) {
    using G = aik_geom<KB>;
    const int j    = threadIdx.x;
    const int lane = j & 31;
    const int warp = j >> 5;
    const __half2 hinf = __float2half2_rn(INFINITY);
    // t' >> KB of this thread's t' = TPT*j + 2*p. At KB >= 4, TPT <= 2^KB and the range is aligned, so it is the same
    // value for every p (what the KB = 4 encoder used); at KB = 2/3 a thread spans TPT/2^KB of these groups, so it has
    // to be recomputed per half2 pair (both t of a pair always share it, since 2^KB >= 4).
    const int tg0 = (G::TPT * j) >> KB;
#define AIK_TG(p) (KB >= 4 ? tg0 : (int) ((G::TPT * j + 2 * (p)) >> KB))

    {
        const __half2 y2 = __float2half2_rn(S.tgt[R & 127]);
#pragma unroll
        for (int hi = 0; hi < G::NHI; ++hi) {
#pragma unroll
            for (int p = 0; p < G::NP; ++p) {
                const bool off = t0 >= 0 && (((hi << (G::TB - KB)) | AIK_TG(p)) != t0);
                const __half2 mv = off ? hinf : __float2half2_rn(0.0f);
                const __half2 e = __hsub2(aik_lev2<KB>(Lp, hi, p), y2);
                c[hi][p] = __hfma2(e, e, mv);
            }
        }
    }

    for (int i = 1; i <= E; ++i) {
        const int buf = i & 1;
        float    lmin = INFINITY;
        uint32_t dw[G::DW] = { 0 };
        uint32_t mw[G::NP];
#pragma unroll
        for (int p = 0; p < G::NP; ++p) {
            __half2  mn  = c[0][p];
            uint32_t arg = 0;
#pragma unroll
            for (int hi = 1; hi < G::NHI; ++hi) {
                const uint32_t mask = __hlt2_mask(c[hi][p], mn);
                mn  = __hmin2(mn, c[hi][p]);
                arg = (arg & ~mask) | ((uint32_t) hi * 0x00010001u & mask);
            }
            // symbol for this thread's t index u goes to word u/SPW at bit KB*(u%SPW) (KB = 4/5: one word, bit KB*u)
            dw[(2 * p) / G::SPW]     |= (arg & (G::NHI - 1))          << (KB * ((2 * p)     % G::SPW));
            dw[(2 * p + 1) / G::SPW] |= ((arg >> 16) & (G::NHI - 1))  << (KB * ((2 * p + 1) % G::SPW));
            mw[p] = *reinterpret_cast<const uint32_t *>(&mn);
            lmin = fminf(lmin, fminf(__low2float(mn), __high2float(mn)));
        }
        uint32_t * mbw = reinterpret_cast<uint32_t *>(S.mb[buf]) + G::NP * j;
#pragma unroll
        for (int p = 0; p < G::NP; ++p) {
            mbw[p] = mw[p];
        }
        if (i >= W0) {
#pragma unroll
            for (int w = 0; w < G::DW; ++w) {
                S.dec[i - W0][w][j] = dw[w];
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            lmin = fminf(lmin, __shfl_xor_sync(0xffffffffu, lmin, m));
        }
        if (lane == 0) {
            S.wmin[buf][warp] = lmin;
        }
        __syncthreads();
        float gmin = S.wmin[buf][0];
#pragma unroll
        for (int w = 1; w < 16; ++w) {
            gmin = fminf(gmin, S.wmin[buf][w]);
        }
        const __half  gminh = __float2half_rn(gmin);
        const __half2 y2    = __float2half2_rn(S.tgt[(i + R) & 127]);
        const __half * mbh  = reinterpret_cast<const __half *>(S.mb[buf]);
#pragma unroll
        for (int hi = 0; hi < G::NHI; ++hi) {
#pragma unroll
            for (int p = 0; p < G::NP; ++p) {
                const __half2 mv = __half2half2(__hsub(mbh[(hi << (G::TB - KB)) | AIK_TG(p)], gminh));
                const __half2 e = __hsub2(aik_lev2<KB>(Lp, hi, p), y2);
                c[hi][p] = __hfma2(e, e, mv);
            }
        }
    }

    if (endmode == 1) {
        float bv = INFINITY;
        int   bi = 0x7fffffff;
#pragma unroll
        for (int hi = 0; hi < G::NHI; ++hi) {
#pragma unroll
            for (int p = 0; p < G::NP; ++p) {
                const float a = __low2float(c[hi][p]), b = __high2float(c[hi][p]);
                const int   s = (hi << G::TB) | (G::TPT * j + 2 * p);
                if (a < bv) { bv = a; bi = s; }
                if (b < bv) { bv = b; bi = s + 1; }
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            S.redv[warp] = bv;
            S.redi[warp] = bi;
        }
        __syncthreads();
        if (j == 0) {
            float v = S.redv[0];
            int   s = S.redi[0];
            for (int w = 1; w < 16; ++w) {
                if (S.redv[w] < v || (S.redv[w] == v && S.redi[w] < s)) { v = S.redv[w]; s = S.redi[w]; }
            }
            S.bests = s & 0xFFFF;
        }
    } else if (endmode == 2) {
        if (j == t0 / G::TPT) {
            const int u = t0 % G::TPT;
            float bv = INFINITY;
            int   bh = 0;
#pragma unroll
            for (int hi = 0; hi < G::NHI; ++hi) {
                float v = 0.0f;
#pragma unroll
                for (int p = 0; p < G::NP; ++p) {
                    if (p == (u >> 1)) {
                        v = (u & 1) ? __high2float(c[hi][p]) : __low2float(c[hi][p]);
                    }
                }
                if (v < bv) { bv = v; bh = hi; }
            }
            S.bests = (bh << G::TB) | t0;
        }
    }
    __syncthreads();
}

template <int KB>
static __device__ __forceinline__ int aik_trace(const aik_smem<KB> & S, int s, const int E, const int lo, const int W0, uint16_t * st) {
    using G = aik_geom<KB>;
    for (int i = E; i >= lo; --i) {
        if (st) {
            st[i] = (uint16_t) s;
        }
        const int t  = s >> KB;
        const int u  = t % G::TPT;
        const int hi = (int) ((S.dec[i - W0][u / G::SPW][t / G::TPT] >> (KB * (u % G::SPW))) & (G::NHI - 1));
        s = t | (hi << G::TB);
    }
    return s;
}

template <int RB, int KB>
static __global__ void __launch_bounds__(ENC_NT, 1) k_aik_encode(const float * __restrict__ W, const int64_t K,
        const float * __restrict__ Gm, const uint8_t * __restrict__ gok, const float lrms, const float smul,
        uint8_t * __restrict__ out, const int hb0, const int prerot, float * __restrict__ qr, const uint32_t lmul,
        const float * __restrict__ sfix) {
    // hb0 / prerot / qr: as for k_ai4_encode (EXLGAP full-K LDLQ at rate KB, RB = 256 only): with prerot W holds rotated,
    // cross-block LDLQ-updated weights; hb = hb0 + blockIdx.x; G / gok are indexed by blockIdx.x; qr gets [M][RB].
    using G = aik_geom<KB>;
    extern __shared__ __align__(16) uint8_t enc_smem_raw[];
    aik_smem<KB> & S = *reinterpret_cast<aik_smem<KB> *>(enc_smem_raw);
    constexpr int NCS = RB / 8;

    const int j = threadIdx.x, r = j >> 5, l = j & 31;
    const int hb = hb0 + blockIdx.x, slab = blockIdx.y;
    const int gb = prerot ? blockIdx.x : hb;
    const bool lact = l < NCS;

    auto chunk = [&](const int64_t kc) -> uint8_t * {
        return out + ((int64_t) slab * (K / 128) + kc) * G::UB;
    };

    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    {
        if (lact) {
            const float * wr = W + ((int64_t) slab * 16 + r) * K + (int64_t) hb * RB + l * 8;
            const float4 a = reinterpret_cast<const float4 *>(wr)[0];
            const float4 b = reinterpret_cast<const float4 *>(wr)[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        }
        if (!prerot) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (ai4_dev_sign_neg((l * 8 + q) % RB)) {
                    v[q] = -v[q];
                }
            }
            enc_fwht<RB>(v);
        }
        const float nrm = prerot ? 1.0f : RB == 256 ? 1.0f / 16.0f : 0.08838834764831845f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            v[q] *= nrm;
            if (lact) {
                S.y[r][l * 8 + q] = v[q];
            }
        }
        if (!lact) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                v[q] = 0.0f;
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        ss += v[q] * v[q];
    }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, m);
    }
    // sfix (private A/B control, do not ship): externally supplied per-128-group scale in place of this group's RMS,
    // [M][K/128] row-major; nullptr = the group RMS, which is byte-identical to the shipped encoder.
    float srms = sqrtf(ss / 128.0f);
    if (sfix && lact) {
        srms = sfix[(int64_t) (slab * 16 + r) * (K / 128) + (int64_t) hb * (RB / 128) + (l >> 4)];
    }
    float dv = __half2float(__float2half_rn(srms / lrms * smul));
    if (!(dv >= 0.0f) || !isfinite(dv)) {
        dv = 0.0f;
    }
    if ((l & 15) == 0 && lact) {
        const int64_t kc = (int64_t) hb * (RB / 128) + (l >> 4);
        reinterpret_cast<uint16_t *>(chunk(kc))[r] = __half_as_ushort(__float2half_rn(dv));
    }

    uint32_t Lp[G::NHI][G::NL];
#pragma unroll
    for (int hi = 0; hi < G::NHI; ++hi) {
#pragma unroll
        for (int w = 0; w < G::NL; ++w) {
            uint32_t x = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (4 * w + u >= G::TPT) {
                    continue;       // KB = 6: only 2 of the 4 bytes of the single level word are used
                }
                const uint32_t s = ((uint32_t) hi << G::TB) | (uint32_t) (G::TPT * j + 4 * w + u);
                x |= (uint32_t) (aik_dev_level_c(s, lmul) + 128) << (8 * u);
            }
            Lp[hi][w] = x;
        }
    }
    const bool useG = Gm != nullptr && (gok == nullptr || gok[gb] != 0);
    __half2 c[G::NHI][G::NP];
    __syncthreads();

    for (int cs = 0; cs < NCS; ++cs) {
        if (l == cs) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float t = dv > 0.0f ? S.y[r][cs * 8 + q] / dv : 0.0f;
                S.tgt[r * 8 + q] = fminf(fmaxf(t, -250.0f), 250.0f);
            }
        }
        __syncthreads();

        constexpr int WB = 128 - G::WIN;   // first step of the last decision window
        aik_viterbi<KB>(S, Lp, c, G::ROLL, -1, 127, WB, 1);
        if (j == 0) {
            const int sw = aik_trace<KB>(S, S.bests, 127, WB + 1, WB, nullptr);
            S.t0 = sw & (G::NT - 1);
        }
        __syncthreads();
        const int t0 = S.t0;

        aik_viterbi<KB>(S, Lp, c, 0, t0, 127, WB, 2);
        if (j == 0) {
            S.st[WB - 1] = (uint16_t) aik_trace<KB>(S, S.bests, 127, WB, WB, S.st);
        }
        __syncthreads();
#pragma unroll 1
        for (int E = WB - 1; E > 0; E -= G::WIN) {
            const int W0 = E - (G::WIN - 1);
            aik_viterbi<KB>(S, Lp, c, 0, t0, E, W0, 0);
            if (j == 0) {
                const int lo = W0 > 0 ? W0 : 1;
                S.st[lo - 1] = (uint16_t) aik_trace<KB>(S, S.st[E], E, lo, W0, S.st);
            }
            __syncthreads();
        }

        if (qr != nullptr && l == cs) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                qr[((int64_t) slab * 16 + r) * RB + cs * 8 + q] = dv * (float) aik_dev_level_c(S.st[r * 8 + q], lmul);
            }
        }
        if (j < G::TW) {
            // word j of the tile: stream bits 32j .. 32j + 31, symbol n at bits [KB*n, KB*n + KB), MSB first
            uint32_t w = 0;
#pragma unroll
            for (int b = 0; b < 32; ++b) {
                const int bit = 32 * j + b, n = bit / KB, bp = bit % KB;
                w |= (uint32_t) ((S.st[n] >> (KB - 1 - bp)) & 1) << (31 - b);
            }
            const int64_t kc = (int64_t) hb * (RB / 128) + cs / 16;
            reinterpret_cast<uint32_t *>(chunk(kc) + 32)[(cs % 16) * G::TW + j] = w;
        }

        if (useG && cs < NCS - 1) {
            if (l == cs) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    S.ebuf[r * 8 + q] = S.y[r][cs * 8 + q] - dv * (float) aik_dev_level_c(S.st[r * 8 + q], lmul);
                }
            }
            __syncthreads();
            if (l > cs && lact) {
                const float * g = Gm + (((int64_t) gb * NCS + cs) * 8) * RB + l * 8;
                float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    const float ea = S.ebuf[r * 8 + a];
                    const float4 g0 = reinterpret_cast<const float4 *>(g + a * RB)[0];
                    const float4 g1 = reinterpret_cast<const float4 *>(g + a * RB)[1];
                    acc[0] += ea * g0.x; acc[1] += ea * g0.y; acc[2] += ea * g0.z; acc[3] += ea * g0.w;
                    acc[4] += ea * g1.x; acc[5] += ea * g1.y; acc[6] += ea * g1.z; acc[7] += ea * g1.w;
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    S.y[r][l * 8 + q] -= acc[q];
                }
            }
        }
        __syncthreads();
    }
}

// Launch table: the rate-KB encoder at RB = 256 / 128, and its shared-memory size (grows as 2^(16-KB), see aik_geom).
struct aik_kern_info {
    void (*fn)(const float *, int64_t, const float *, const uint8_t *, float, float, uint8_t *, int, int, float *, uint32_t,
               const float *);
    int  smem;
};

template <int KB>
static aik_kern_info aik_kern_of(int rb) {
    return { rb == 256 ? k_aik_encode<256, KB> : k_aik_encode<128, KB>, (int) sizeof(aik_smem<KB>) };
}

static bool aik_kern(int kbits, int rb, aik_kern_info & ki) {
    switch (kbits) {
        case 2: ki = aik_kern_of<2>(rb); return true;
        case 3: ki = aik_kern_of<3>(rb); return true;
        case 4: ki = aik_kern_of<4>(rb); return true;
        case 5: ki = aik_kern_of<5>(rb); return true;
        case 6: ki = aik_kern_of<6>(rb); return true;
        default: return false;
    }
}

int ai4_cuda_encode_rate(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                         int kbits, uint32_t lmul, uint8_t * out, float * ms, const float * sfix) {
    if (K % 128 != 0 || M % 16 != 0 || kbits < 2 || kbits > 6) {
        return 2;
    }
    const int rb = K % 256 == 0 ? 256 : 128;
    const size_t nw = (size_t) K * M, nq = (size_t) (M / 16) * (K / 128) * (32 + 256 * kbits);
    const size_t ng = (size_t) K * rb;
    float * dW = nullptr, * dG = nullptr, * dsf = nullptr;
    uint8_t * dok = nullptr, * dq = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMemcpy(dW, W, nw * sizeof(float), cudaMemcpyHostToDevice));
    AI4_CU(cudaMemset(dq, 0, nq));
    if (G) {
        AI4_CU(cudaMalloc(&dG, ng * sizeof(float)));
        AI4_CU(cudaMalloc(&dok, K / rb));
        AI4_CU(cudaMemcpy(dG, G, ng * sizeof(float), cudaMemcpyHostToDevice));
        AI4_CU(cudaMemcpy(dok, gok, K / rb, cudaMemcpyHostToDevice));
    }
    if (sfix) {
        const size_t nsf = (size_t) M * (K / 128);
        AI4_CU(cudaMalloc(&dsf, nsf * sizeof(float)));
        AI4_CU(cudaMemcpy(dsf, sfix, nsf * sizeof(float), cudaMemcpyHostToDevice));
    }
    aik_kern_info ki;
    if (!aik_kern(kbits, rb, ki)) {
        return 2;
    }
    auto kern = ki.fn;
    const int smem = ki.smem;
    AI4_CU(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaEvent_t e0, e1;
    AI4_CU(cudaEventCreate(&e0));
    AI4_CU(cudaEventCreate(&e1));
    AI4_CU(cudaEventRecord(e0));
    const dim3 grid((unsigned) (K / rb), (unsigned) (M / 16));
    kern<<<grid, ENC_NT, smem>>>(dW, K, dG, dok, lrms, smul, dq, 0, 0, nullptr, lmul, dsf);
    AI4_CU(cudaGetLastError());
    AI4_CU(cudaEventRecord(e1));
    AI4_CU(cudaEventSynchronize(e1));
    if (ms) {
        AI4_CU(cudaEventElapsedTime(ms, e0, e1));
    }
    AI4_CU(cudaMemcpy(out, dq, nq, cudaMemcpyDeviceToHost));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dW);
    cudaFree(dq);
    cudaFree(dG);
    cudaFree(dok);
    cudaFree(dsf);
    return 0;
}

int ai4_cuda_reconstruct_rate(const uint8_t * q, int64_t K, int64_t M, int kbits, uint32_t lmul, int rotated, float * W) {
    if (kbits == 4 && lmul == AI4_DEV_MUL && !rotated) {
        return ai4_cuda_reconstruct_layout(q, K, M, 1, W);
    }
    if (kbits < 2 || kbits > 6) {
        return 2;
    }
    const size_t nw = (size_t) K * M, nq = (size_t) (M / 16) * (K / 128) * (32 + 256 * kbits);
    float * dW = nullptr;
    uint8_t * dq = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMemcpy(dq, q, nq, cudaMemcpyHostToDevice));
    if (!aik_reconstruct_rate<float>(kbits, dq, dW, K, M, !rotated, 0, lmul)) {
        cudaFree(dW); cudaFree(dq);
        return 2;
    }
    AI4_CU(cudaGetLastError());
    AI4_CU(cudaDeviceSynchronize());
    AI4_CU(cudaMemcpy(W, dW, nw * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dW);
    cudaFree(dq);
    return 0;
}

int ai4_cuda_reconstruct_span(const uint8_t * q, int64_t K, int64_t M, int kbits, uint32_t lmul, int rotated, float * W) {
    if (kbits < 2 || kbits > 6 || K % 1024 != 0) {
        return 2;
    }
    const size_t nw = (size_t) K * M, nq = (size_t) (M / 16) * (K / 128) * (4 + 256 * kbits);
    float * dW = nullptr;
    uint8_t * dq = nullptr;
    AI4_CU(cudaMalloc(&dW, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMemcpy(dq, q, nq, cudaMemcpyHostToDevice));
    if (!aik_reconstruct_rate<float>(kbits, dq, dW, K, M, !rotated, 0, lmul, true)) {
        cudaFree(dW); cudaFree(dq);
        return 2;
    }
    AI4_CU(cudaGetLastError());
    AI4_CU(cudaDeviceSynchronize());
    AI4_CU(cudaMemcpy(W, dW, nw * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dW);
    cudaFree(dq);
    return 0;
}

size_t ai4_cuda_encode_smem_bytes(void) {
    return sizeof(enc_smem);
}

#undef AIK_TG

// ---------------- AIQ / EXLGAP full-K LDLQ driver (after k_aik_encode so both kernel families are defined) ----------------

// kbits 0: AI4 1024 layout (the original AIQ path). kbits 4 / 5 (EXLGAP): AI+ unit-128 layout at that rate (AI4_U / AI5_U),
// rotation block 256 (K % 256), same LDLQ math; out is (M/16)*(K/128)*(32 + 256*kbits) bytes.
// c_r = <Y0_r, T_r> / <Q_r, T_r> (T = H' Q^T, rows of the col-major K x M views), clamped to [0.5, 2]; 1 when degenerate.
// EXLGAP --refit: the closed-form per-row scale that minimizes tr(E H' E^T) with the trellis held fixed (EXL3 refit_scales,
// output-channel half). One block per row.
static __global__ void k_aiq_rowfit(const float * __restrict__ Y0, const float * __restrict__ Q, const float * __restrict__ T,
                                    const int64_t K, float * __restrict__ c) {
    __shared__ double sn[256], sd[256];
    const int64_t r = blockIdx.x;
    double n = 0.0, d = 0.0;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        const double t = T[r * K + k];
        n += (double) Y0[r * K + k] * t;
        d += (double) Q[r * K + k] * t;
    }
    sn[threadIdx.x] = n; sd[threadIdx.x] = d;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if ((int) threadIdx.x < h) { sn[threadIdx.x] += sn[threadIdx.x + h]; sd[threadIdx.x] += sd[threadIdx.x + h]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double v = sd[0] > 1e-30 ? sn[0] / sd[0] : 1.0;
        if (!(v == v)) v = 1.0;
        c[r] = (float) fmin(fmax(v, 0.5), 2.0);
    }
}

static __global__ void k_aiq_rowscale(float * __restrict__ Q, const float * __restrict__ c, const int64_t K, const int64_t M) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K * M) Q[idx] *= c[idx / K];
}

static int aiq_encode_fullk_impl(const float * W, const float * H, int64_t K, int64_t M, int mode, double damp, float lrms,
                                 float smul, int kbits, uint32_t lmul, int refit, uint8_t * out, float * ms, double * werr_out,
                                 double * werr_pre) {
    if (M % 16 != 0) return 2;
    if (kbits == 0 ? K % 1024 != 0 : ((kbits < 2 || kbits > 6) || K % 256 != 0)) return 2;
    const int NB = (int) (K / 256);
    const size_t nw = (size_t) K * M, nh = (size_t) K * K;
    const size_t nq = kbits == 0 ? (size_t) (M / 16) * (K / 1024) * AI4_DEV_SB_BYTES
                                 : (size_t) (M / 16) * (K / 128) * (32 + 256 * kbits);
    float * dY = nullptr, * dY0 = nullptr, * dU = nullptr, * dH0 = nullptr, * dT = nullptr, * dG = nullptr, * dQr = nullptr,
          * dD = nullptr, * dQ = nullptr;
    uint8_t * dq = nullptr;
    int * dinfo = nullptr;
    cudaEvent_t e0, e1;
    AI4_CU(cudaEventCreate(&e0));
    AI4_CU(cudaEventCreate(&e1));
    AI4_CU(cudaEventRecord(e0));
    cublasHandle_t cb;
    cusolverDnHandle_t cs;
    AIQ_CB(cublasCreate(&cb));
    AIQ_CB(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
    AIQ_CS(cusolverDnCreate(&cs));
    AI4_CU(cudaMalloc(&dY, nw * sizeof(float)));
    AI4_CU(cudaMalloc(&dU, nh * sizeof(float)));
    AI4_CU(cudaMalloc(&dT, nh * sizeof(float)));
    AI4_CU(cudaMalloc(&dq, nq));
    AI4_CU(cudaMalloc(&dG, (size_t) 32 * 8 * 256 * sizeof(float)));
    AI4_CU(cudaMalloc(&dQr, (size_t) M * 256 * sizeof(float)));
    AI4_CU(cudaMalloc(&dD, (size_t) M * 256 * sizeof(float)));
    AI4_CU(cudaMalloc(&dinfo, sizeof(int)));
    AI4_CU(cudaMemset(dq, 0, nq));
    AI4_CU(cudaMemcpy(dY, W, nw * sizeof(float), cudaMemcpyHostToDevice));
    k_aiq_rot<<<dim3((unsigned) NB, (unsigned) M), 32>>>(dY, K);
    // H' = R H R^T: rotate rows, transpose, rotate rows
    AI4_CU(cudaMemcpy(dU, H, nh * sizeof(float), cudaMemcpyHostToDevice));
    const unsigned nbk = (unsigned) ((nh + 255) / 256);
    if (mode == 2) k_aiq_mask<<<nbk, 256>>>(dU, K, 2);
    k_aiq_rot<<<dim3((unsigned) NB, (unsigned) K), 32>>>(dU, K);
    {
        const float one = 1.0f, zero = 0.0f;
        AIQ_CB(cublasSgeam(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) K, (int) K, &one, dU, (int) K, &zero, dT, (int) K, dT, (int) K));
    }
    k_aiq_rot<<<dim3((unsigned) NB, (unsigned) K), 32>>>(dT, K);
    if (mode == 1) k_aiq_mask<<<nbk, 256>>>(dT, K, 1);
    // keep the undamped H' for the error report
    if (werr_out || refit) {
        AI4_CU(cudaMalloc(&dH0, nh * sizeof(float)));
        AI4_CU(cudaMemcpy(dH0, dT, nh * sizeof(float), cudaMemcpyDeviceToDevice));
        AI4_CU(cudaMalloc(&dY0, nw * sizeof(float)));
        AI4_CU(cudaMemcpy(dY0, dY, nw * sizeof(float), cudaMemcpyDeviceToDevice));
        AI4_CU(cudaMalloc(&dQ, nw * sizeof(float)));
    }
    {
        std::vector<float> dg(K);
        AI4_CU(cudaMemcpy2D(dg.data(), sizeof(float), dT, (K + 1) * sizeof(float), sizeof(float), K, cudaMemcpyDeviceToHost));
        double mean = 0.0;
        for (int64_t i = 0; i < K; ++i) mean += dg[i];
        mean /= (double) K;
        if (!(mean > 0.0)) { fprintf(stderr, "aiq: non-positive Hessian trace\n"); return 3; }
        k_aiq_adddiag<<<(unsigned) ((K + 255) / 256), 256>>>(dT, K, (float) (damp * mean));
    }
    // U = chol(H'^-1) (upper, row-major view): potrf(L) -> potri(L) -> potrf(L) on the column-major buffer
    {
        int lw = 0, lw2 = 0, info = 0;
        AIQ_CS(cusolverDnSpotrf_bufferSize(cs, CUBLAS_FILL_MODE_LOWER, (int) K, dT, (int) K, &lw));
        AIQ_CS(cusolverDnSpotri_bufferSize(cs, CUBLAS_FILL_MODE_LOWER, (int) K, dT, (int) K, &lw2));
        lw = lw > lw2 ? lw : lw2;
        float * wk = nullptr;
        AI4_CU(cudaMalloc(&wk, (size_t) lw * sizeof(float)));
        AIQ_CS(cusolverDnSpotrf(cs, CUBLAS_FILL_MODE_LOWER, (int) K, dT, (int) K, wk, lw, dinfo));
        AI4_CU(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) { fprintf(stderr, "aiq: potrf(H) info %d\n", info); return 4; }
        AIQ_CS(cusolverDnSpotri(cs, CUBLAS_FILL_MODE_LOWER, (int) K, dT, (int) K, wk, lw, dinfo));
        AI4_CU(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) { fprintf(stderr, "aiq: potri info %d\n", info); return 4; }
        AIQ_CS(cusolverDnSpotrf(cs, CUBLAS_FILL_MODE_LOWER, (int) K, dT, (int) K, wk, lw, dinfo));
        AI4_CU(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) { fprintf(stderr, "aiq: potrf(Hinv) info %d\n", info); return 4; }
        cudaFree(wk);
        k_aiq_mask<<<nbk, 256>>>(dT, K, 3);
        std::swap(dU, dT);   // dU = U (row-major upper)
        cudaFree(dT); dT = nullptr;
    }
    aik_kern_info ki = { nullptr, 0 };
    if (kbits != 0 && !aik_kern(kbits, 256, ki)) return 2;
    const int smem = (int) sizeof(enc_smem);
    AI4_CU(cudaFuncSetAttribute(k_ai4_encode<256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (ki.fn) AI4_CU(cudaFuncSetAttribute(ki.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ki.smem));
    std::vector<float> Ub((size_t) 256 * 256), G((size_t) 32 * 8 * 256);
    for (int b = 0; b < NB; ++b) {
        AI4_CU(cudaMemcpy2D(Ub.data(), 256 * sizeof(float), dU + (size_t) b * 256 * K + (size_t) b * 256, K * sizeof(float),
                            256 * sizeof(float), 256, cudaMemcpyDeviceToHost));
        for (int c = 0; c < 32; ++c) {
            const int c0 = c * 8;
            double A8[64], Ai[64];
            for (int a = 0; a < 8; ++a) for (int bb = 0; bb < 8; ++bb) A8[a * 8 + bb] = Ub[(size_t) (c0 + a) * 256 + c0 + bb];
            aiq_inv_upper8(A8, Ai);
            for (int a = 0; a < 8; ++a) {
                for (int k = 0; k < 256; ++k) {
                    double s = 0.0;
                    for (int bb = a; bb < 8; ++bb) s += Ai[a * 8 + bb] * Ub[(size_t) (c0 + bb) * 256 + k];
                    G[((size_t) c * 8 + a) * 256 + k] = (float) s;
                }
            }
        }
        AI4_CU(cudaMemcpy(dG, G.data(), G.size() * sizeof(float), cudaMemcpyHostToDevice));
        const dim3 g1(1, (unsigned) (M / 16));
        if (kbits == 0) k_ai4_encode<256, 0><<<g1, ENC_NT, smem>>>(dY, K, dG, nullptr, lrms, smul, dq, b, 1, dQr);
        else { auto kf = ki.fn; kf<<<g1, ENC_NT, ki.smem>>>(dY, K, dG, nullptr, lrms, smul, dq, b, 1, dQr, lmul, nullptr); }
        AI4_CU(cudaGetLastError());
        // D = Y_b - Qr ; F = D U_bb^-1 (TRSM) ; Y_later -= F U_b,later
        if (dQ) {
            k_aiq_diff<<<(unsigned) ((M * 256 + 255) / 256), 256>>>(dY, dQr, dD, dQ, K, b, M);
        }
        if (b + 1 < NB) {
            if (!dQ) {
                const float one = 1.0f, mone = -1.0f;
                AIQ_CB(cublasSgeam(cb, CUBLAS_OP_N, CUBLAS_OP_N, 256, (int) M, &one, dY + (size_t) b * 256, (int) K, &mone, dQr, 256, dD, 256));
            }
            const float one = 1.0f, mone = -1.0f;
            AIQ_CB(cublasStrsm(cb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 256, (int) M, &one,
                               dU + (size_t) b * 256 * K + (size_t) b * 256, (int) K, dD, 256));
            const int Lc = (int) (K - (int64_t) (b + 1) * 256);
            AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, Lc, (int) M, 256, &mone, dU + (size_t) b * 256 * K + (size_t) (b + 1) * 256,
                               (int) K, dD, 256, &one, dY + (size_t) (b + 1) * 256, (int) K));
        }
    }
    AI4_CU(cudaDeviceSynchronize());
    std::vector<float> cfit;
    if (refit) {
        float * dTf = nullptr, * dc = nullptr;
        AI4_CU(cudaMalloc(&dTf, nw * sizeof(float)));
        AI4_CU(cudaMalloc(&dc, (size_t) M * sizeof(float)));
        const float one = 1.0f, zero = 0.0f;
        if (werr_pre) {   // error before the refit (same formula as below)
            float num = 0.0f, den = 0.0f, mone = -1.0f;
            AIQ_CB(cublasSgeam(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, &one, dY0, (int) K, &mone, dQ, (int) K, dY, (int) K));
            AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH0, (int) K, dY, (int) K, &zero, dTf, (int) K));
            AIQ_CB(cublasSdot(cb, (int) nw, dY, 1, dTf, 1, &num));
            AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH0, (int) K, dY0, (int) K, &zero, dTf, (int) K));
            AIQ_CB(cublasSdot(cb, (int) nw, dY0, 1, dTf, 1, &den));
            *werr_pre = (double) num / (double) den;
        }
        AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH0, (int) K, dQ, (int) K, &zero, dTf, (int) K));
        k_aiq_rowfit<<<(unsigned) M, 256>>>(dY0, dQ, dTf, K, dc);
        k_aiq_rowscale<<<(unsigned) ((nw + 255) / 256), 256>>>(dQ, dc, K, M);
        AI4_CU(cudaGetLastError());
        cfit.resize(M);
        AI4_CU(cudaMemcpy(cfit.data(), dc, (size_t) M * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dTf); cudaFree(dc);
    }
    if (werr_out) {
        // H'-weighted relative error tr(E H' E^T) / tr(Y0 H' Y0^T), E = Y0 - Q (rotated domain, undamped H')
        float * dE = dY;   // reuse
        const float one = 1.0f, mone = -1.0f, zero = 0.0f;
        AIQ_CB(cublasSgeam(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, &one, dY0, (int) K, &mone, dQ, (int) K, dE, (int) K));
        // T1 = H' E^T (col-major: K x M) ; num = sum(E^T .* T1)
        float * dT1 = dQ;
        AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH0, (int) K, dE, (int) K, &zero, dT1, (int) K));
        float num = 0.0f, den = 0.0f;
        // dot over all K*M entries
        AIQ_CB(cublasSdot(cb, (int) nw, dE, 1, dT1, 1, &num));
        AIQ_CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) M, (int) K, &one, dH0, (int) K, dY0, (int) K, &zero, dT1, (int) K));
        AIQ_CB(cublasSdot(cb, (int) nw, dY0, 1, dT1, 1, &den));
        *werr_out = (double) num / (double) den;
    }
    AI4_CU(cudaEventRecord(e1));
    AI4_CU(cudaEventSynchronize(e1));
    if (ms) AI4_CU(cudaEventElapsedTime(ms, e0, e1));
    AI4_CU(cudaMemcpy(out, dq, nq, cudaMemcpyDeviceToHost));
    if (refit) {   // rescale every stored fp16 group scale of row r by c_r
        for (int64_t row = 0; row < M; ++row) {
            const int64_t slab = row / 16, r = row % 16;
            for (int64_t kc = 0; kc < K / 128; ++kc) {
                uint16_t * sp = kbits == 0
                    ? reinterpret_cast<uint16_t *>(out + (slab * (K / 1024) + (kc >> 3)) * AI4_DEV_SB_BYTES) + r * 8 + (kc & 7)
                    : reinterpret_cast<uint16_t *>(out + (slab * (K / 128) + kc) * (32 + 256 * kbits)) + r;
                __half h; memcpy(&h, sp, 2);
                const __half n = __float2half_rn(__half2float(h) * cfit[row]);
                memcpy(sp, &n, 2);
            }
        }
    }
    cudaFree(dY); cudaFree(dY0); cudaFree(dU); cudaFree(dH0); cudaFree(dG); cudaFree(dQr); cudaFree(dD); cudaFree(dQ);
    cudaFree(dq); cudaFree(dinfo);
    cublasDestroy(cb);
    cusolverDnDestroy(cs);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}

int ai4_cuda_encode_fullk(const float * W, const float * H, int64_t K, int64_t M, int mode, double damp, float lrms,
                          float smul, uint8_t * out, float * ms, double * werr_out) {
    return aiq_encode_fullk_impl(W, H, K, M, mode, damp, lrms, smul, 0, AI4_DEV_MUL, 0, out, ms, werr_out, nullptr);
}

int ai4_cuda_encode_fullk_rate(const float * W, const float * H, int64_t K, int64_t M, int mode, double damp, float lrms,
                               float smul, int kbits, uint32_t lmul, int refit, uint8_t * out, float * ms, double * werr_out,
                               double * werr_pre) {
    return aiq_encode_fullk_impl(W, H, K, M, mode, damp, lrms, smul, kbits, lmul, refit, out, ms, werr_out, werr_pre);
}

