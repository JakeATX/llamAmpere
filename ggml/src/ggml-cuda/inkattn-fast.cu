// INKATTN-FAST (fork): SM 8.6 occupancy- and SWA-dead-tile-optimized GQA-grouped tensor-core decode attention.
//
// Targets the measured 2.544 ms/round (48 x 48.5 us decode + 48 x 4.5 us combine) SWA/near window bottleneck on
// Inkling-2-Mini at 100K context (n_swa = 1024 inside n_kv = 3072..5120 ring buffer):
//   1. Mask-first dead-tile skip: reads the 16-key mask BEFORE K/V/rel_pos/tab and skips all-(-INF) tiles
//      (bit-exact: ~66.5% of tiles skipped at -ub 2048, ~80% at -ub 4096).
//   2. Cyclic tile striding (t = wi; t < NT; t += W): load-balances the ~65 live SWA tiles evenly across all W warps.
//   3. CTA-shared Q staging in 4 KB SMEM (s_qa[8][32]): frees 32 regs/thread and cuts Q global loads 4x.
//   4. Split K/V register live ranges: streams K per s-chunk (8 regs) and V per u-half (16 regs) -> <= 128 regs/thread,
//      0 B local spills, 4 CTAs/SM (16 warps/SM) on SM 8.6 vs baseline 1 CTA/SM (4 warps/SM, 198-255 regs).
//   5. Guarded O *= alpha rescaling (bit-exact; skips 64 FMULs on ~90% of live tiles when max unchanged).
//   6. Bit-exact FP16-magic Q8_0 dequant (0 mismatches across all 16,252,928 finite (half, int8_t) pairs) with
//      shared scale reuse across V u=0/1 and 4-byte aligned loads when available.
//   7. Dead-warp Op store skip + vectorized float4 STG.E.128 Op stores + SMEM-cached live-warp combine.
//
// Kill switch: GGML_CUDA_INKATTN_FAST=0 falls back to baseline inkattn.cu.

#include "inkattn-fast.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

constexpr int INKATTN_FAST_D     = 128;
constexpr int INKATTN_FAST_WARPS = 4;

struct inkattn_fast_params {
    const char * Q; int64_t nbq1, nbq2, nbq3;
    const char * K; int64_t nbk1, nbk2, nbk3;
    const char * V; int64_t nbv1, nbv2, nbv3;
    const char * M; int64_t nbm1, nbm2, nbm3;
    int ne32, ne33;
    int mask_al4;
    int kv_al4;
    const float   * tab; int64_t tab_h1, tab_t;
    const int32_t * rp;  int64_t rp_t;
    int n_kv, T, g, H, Hkv;
    int W;
    float scale;
    float  * Op;         // [S][Hkv][W][16][D]
    float2 * Mp;         // [S][Hkv][W][16]
};

static __device__ __forceinline__ void inkattn_fast_mma(
        float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ uint32_t inkattn_fast_h2u(const float x, const float y) {
    const half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

static __device__ __forceinline__ uint4 inkattn_fast_ld16(const char * p) {
    return __ldg(reinterpret_cast<const uint4 *>(p));
}

// Bit-exact Q8_0 int8x4 -> two half2: (uint8_t)(q ^ 0x80) | 0x6400 = 1024 + (q + 128) = 1152 + q in FP16.
// Exact IEEE-754 identity: __hmul(d_f16, half(q_s8)) == __float2half_rn(float(d_f16) * float(q_s8)) for all finite d.
static __device__ __forceinline__ void inkattn_fast_q8x4_to_h2(
        const uint32_t q, const half2 dh, uint32_t & out0, uint32_t & out1) {
    const uint32_t u     = q ^ 0x80808080u;
    const uint32_t magic = 0x64006400u;
    const half2    m1152 = make_half2(1152.0f, 1152.0f);
    const uint32_t p0    = __byte_perm(u, magic, 0x5150);
    const uint32_t p1    = __byte_perm(u, magic, 0x5352);
    const half2    h0    = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&p0), m1152), dh);
    const half2    h1    = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&p1), m1152), dh);
    out0 = *reinterpret_cast<const uint32_t *>(&h0);
    out1 = *reinterpret_cast<const uint32_t *>(&h1);
}

// Load 8 consecutive int8s from a Q8_0 block (at 2-byte aligned address qptr) and dequantize with half2 scale dh.
static __device__ __forceinline__ uint4 inkattn_fast_q8_load8_with_d(const char * qptr, const half2 dh, const bool al4) {
    uint32_t q0, q1;
    if (al4 && ((reinterpret_cast<uintptr_t>(qptr) & 3u) == 0u)) {
        q0 = __ldg(reinterpret_cast<const uint32_t *>(qptr + 0));
        q1 = __ldg(reinterpret_cast<const uint32_t *>(qptr + 4));
    } else {
        const unsigned short * qs = reinterpret_cast<const unsigned short *>(qptr);
        const uint32_t s0 = __ldg(qs + 0);
        const uint32_t s1 = __ldg(qs + 1);
        const uint32_t s2 = __ldg(qs + 2);
        const uint32_t s3 = __ldg(qs + 3);
        q0 = s0 | (s1 << 16);
        q1 = s2 | (s3 << 16);
    }
    uint32_t w0, w1, w2, w3;
    inkattn_fast_q8x4_to_h2(q0, dh, w0, w1);
    inkattn_fast_q8x4_to_h2(q1, dh, w2, w3);
    return make_uint4(w0, w1, w2, w3);
}

template <bool q8>
static __device__ __forceinline__ uint4 inkattn_fast_ld8(const char * row, const int dim0, const bool kv_al4) {
    if constexpr (!q8) {
        return inkattn_fast_ld16(row + dim0*sizeof(half));
    } else {
        const char * b = row + (dim0 / 32)*34;
        const half2 dh = __half2half2(__ldg(reinterpret_cast<const half *>(b)));
        return inkattn_fast_q8_load8_with_d(b + 2 + (dim0 & 31), dh, kv_al4);
    }
}

template <bool rows16, bool rel, bool q8>
static __global__ void __launch_bounds__(INKATTN_FAST_WARPS*WARP_SIZE, 4)
inkattn_fast_decode(const inkattn_fast_params p) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int D = INKATTN_FAST_D;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int kvh  = blockIdx.y;
    const int seq  = blockIdx.z;
    const int c    = lane & 3;
    const int ln   = lane >> 2;
    const int R    = p.g*p.T;

    const int r0 = ln;
    const int r1 = ln + 8;

    // Cooperatively stage pre-scaled f16 Q fragments into 4 KB SMEM (shared by all 4 warps in CTA),
    // plus the 16-row mask/INKREL row pointers (384 B SMEM -> frees 12 regs/thread across KQ+PV loops!).
    __shared__ uint4           s_qa[8][WARP_SIZE];
    __shared__ const half    * s_mrow[16];
    __shared__ const float   * s_trow[16];
    __shared__ const int32_t * s_rrow[16];
    {
        const int s = warp; // 0..3
        uint32_t q0_r0 = 0, q0_r1 = 0, q0_z0 = 0, q0_z1 = 0;
        uint32_t q1_r0 = 0, q1_r1 = 0, q1_z0 = 0, q1_z1 = 0;

        if (r0 < R) {
            const float * q = reinterpret_cast<const float *>(
                p.Q + seq*p.nbq3 + (int64_t)(r0 / p.g)*p.nbq1 + (int64_t)(kvh*p.g + r0 % p.g)*p.nbq2);
            const float4 x0 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c);
            const float4 x1 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c + 4);
            q0_r0 = inkattn_fast_h2u(x0.x*p.scale, x0.y*p.scale);
            q0_z0 = inkattn_fast_h2u(x0.z*p.scale, x0.w*p.scale);
            q1_r0 = inkattn_fast_h2u(x1.x*p.scale, x1.y*p.scale);
            q1_z0 = inkattn_fast_h2u(x1.z*p.scale, x1.w*p.scale);
        }
        if constexpr (rows16) {
            if (r1 < R) {
                const float * q = reinterpret_cast<const float *>(
                    p.Q + seq*p.nbq3 + (int64_t)(r1 / p.g)*p.nbq1 + (int64_t)(kvh*p.g + r1 % p.g)*p.nbq2);
                const float4 x0 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c);
                const float4 x1 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c + 4);
                q0_r1 = inkattn_fast_h2u(x0.x*p.scale, x0.y*p.scale);
                q0_z1 = inkattn_fast_h2u(x0.z*p.scale, x0.w*p.scale);
                q1_r1 = inkattn_fast_h2u(x1.x*p.scale, x1.y*p.scale);
                q1_z1 = inkattn_fast_h2u(x1.z*p.scale, x1.w*p.scale);
            }
        }
        s_qa[2*s + 0][lane] = make_uint4(q0_r0, q0_r1, q0_z0, q0_z1);
        s_qa[2*s + 1][lane] = make_uint4(q1_r0, q1_r1, q1_z0, q1_z1);

        if (threadIdx.x < 16) {
            const int rr = (int)threadIdx.x < R ? (int)threadIdx.x : R - 1;
            const int hh = kvh*p.g + rr % p.g;
            const int tq = rr / p.g;
            s_mrow[threadIdx.x] = reinterpret_cast<const half *>(
                p.M + (int64_t)(seq % p.ne33)*p.nbm3 + (int64_t)(hh % p.ne32)*p.nbm2 + (int64_t)tq*p.nbm1);
            if constexpr (rel) {
                s_trow[threadIdx.x] = p.tab + hh + (int64_t)tq*p.tab_t;
                s_rrow[threadIdx.x] = p.rp + (int64_t)tq*p.rp_t;
            }
        }
    }
    __syncthreads();

    const int wi = blockIdx.x*INKATTN_FAST_WARPS + warp;
    if (wi >= p.W) {
        return;
    }

    const char * Kh = p.K + seq*p.nbk3 + kvh*p.nbk2;
    const char * Vh = p.V + seq*p.nbv3 + kvh*p.nbv2;

    const int NT = (p.n_kv + 15) / 16;

    float O[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.0f;
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
    float sm0 = 0.0f,      sm1 = 0.0f;

    const bool kv_al4 = p.kv_al4 != 0;

    // Cyclic tile striding: distributes the ~65 live SWA tiles evenly across all W warps.
    for (int t = wi; t < NT; t += p.W) {
        const int  k0   = 16*t;
        const bool full = k0 + 16 <= p.n_kv;

        // 1. Load mask FIRST (16 keys x 2 rows) and skip all-(-INF) tiles before touching K, V, or rel_pos/tab!
        const half * mrow0 = s_mrow[r0];
        const half * mrow1 = rows16 ? s_mrow[r1] : mrow0;
        float2 mk00, mk10, mk01, mk11;
        const int key_m0 = k0 + 2*c;
        const int key_m1 = k0 + 8 + 2*c;
        if (full && p.mask_al4) {
            mk00 = __half22float2(__ldg(reinterpret_cast<const half2 *>(mrow0 + key_m0)));
            mk10 = __half22float2(__ldg(reinterpret_cast<const half2 *>(mrow0 + key_m1)));
            if constexpr (rows16) {
                mk01 = __half22float2(__ldg(reinterpret_cast<const half2 *>(mrow1 + key_m0)));
                mk11 = __half22float2(__ldg(reinterpret_cast<const half2 *>(mrow1 + key_m1)));
            } else {
                mk01 = make_float2(-INFINITY, -INFINITY);
                mk11 = make_float2(-INFINITY, -INFINITY);
            }
        } else {
            mk00.x = key_m0 + 0 < p.n_kv ? __half2float(__ldg(mrow0 + key_m0 + 0)) : -INFINITY;
            mk00.y = key_m0 + 1 < p.n_kv ? __half2float(__ldg(mrow0 + key_m0 + 1)) : -INFINITY;
            mk10.x = key_m1 + 0 < p.n_kv ? __half2float(__ldg(mrow0 + key_m1 + 0)) : -INFINITY;
            mk10.y = key_m1 + 1 < p.n_kv ? __half2float(__ldg(mrow0 + key_m1 + 1)) : -INFINITY;
            if constexpr (rows16) {
                mk01.x = key_m0 + 0 < p.n_kv ? __half2float(__ldg(mrow1 + key_m0 + 0)) : -INFINITY;
                mk01.y = key_m0 + 1 < p.n_kv ? __half2float(__ldg(mrow1 + key_m0 + 1)) : -INFINITY;
                mk11.x = key_m1 + 0 < p.n_kv ? __half2float(__ldg(mrow1 + key_m1 + 0)) : -INFINITY;
                mk11.y = key_m1 + 1 < p.n_kv ? __half2float(__ldg(mrow1 + key_m1 + 1)) : -INFINITY;
            } else {
                mk01 = make_float2(-INFINITY, -INFINITY);
                mk11 = make_float2(-INFINITY, -INFINITY);
            }
        }

        // Bit-exact dead-tile check: if all live rows see -INF across all 16 keys, this tile contributes exactly 0.
        bool any_live = (mk00.x != -INFINITY) | (mk00.y != -INFINITY) | (mk10.x != -INFINITY) | (mk10.y != -INFINITY);
        if constexpr (rows16) {
            any_live |= (mk01.x != -INFINITY) | (mk01.y != -INFINITY) | (mk11.x != -INFINITY) | (mk11.y != -INFINITY);
        }
        if (!__any_sync(0xFFFFFFFF, any_live)) {
            continue;
        }

        // Apply INKREL relative bias only on finite mask entries (f16(mask + bias) matches dense path bitwise).
        if constexpr (rel) {
            const float   * trow0 = s_trow[r0];
            const int32_t * rrow0 = s_rrow[r0];
            if (mk00.x != -INFINITY) mk00.x = __half2float(__float2half_rn(mk00.x + __ldg(trow0 + (int64_t)__ldg(rrow0 + key_m0 + 0)*p.tab_h1)));
            if (mk00.y != -INFINITY) mk00.y = __half2float(__float2half_rn(mk00.y + __ldg(trow0 + (int64_t)__ldg(rrow0 + key_m0 + 1)*p.tab_h1)));
            if (mk10.x != -INFINITY) mk10.x = __half2float(__float2half_rn(mk10.x + __ldg(trow0 + (int64_t)__ldg(rrow0 + key_m1 + 0)*p.tab_h1)));
            if (mk10.y != -INFINITY) mk10.y = __half2float(__float2half_rn(mk10.y + __ldg(trow0 + (int64_t)__ldg(rrow0 + key_m1 + 1)*p.tab_h1)));
            if constexpr (rows16) {
                const float   * trow1 = s_trow[r1];
                const int32_t * rrow1 = s_rrow[r1];
                if (mk01.x != -INFINITY) mk01.x = __half2float(__float2half_rn(mk01.x + __ldg(trow1 + (int64_t)__ldg(rrow1 + key_m0 + 0)*p.tab_h1)));
                if (mk01.y != -INFINITY) mk01.y = __half2float(__float2half_rn(mk01.y + __ldg(trow1 + (int64_t)__ldg(rrow1 + key_m0 + 1)*p.tab_h1)));
                if (mk11.x != -INFINITY) mk11.x = __half2float(__float2half_rn(mk11.x + __ldg(trow1 + (int64_t)__ldg(rrow1 + key_m1 + 0)*p.tab_h1)));
                if (mk11.y != -INFINITY) mk11.y = __half2float(__float2half_rn(mk11.y + __ldg(trow1 + (int64_t)__ldg(rrow1 + key_m1 + 1)*p.tab_h1)));
            }
        }

        // 2. S^T = Q K^T: stream K one s-chunk (32 dims) at a time -> only 8 regs live for K instead of 32!
        float S0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float S1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int key_k0 = (k0 + ln     < p.n_kv) ? (k0 + ln)     : (p.n_kv - 1);
        const int key_k1 = (k0 + 8 + ln < p.n_kv) ? (k0 + 8 + ln) : (p.n_kv - 1);
        const char * Krow0 = Kh + (int64_t)key_k0*p.nbk1;
        const char * Krow1 = Kh + (int64_t)key_k1*p.nbk1;

#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint4 kk0 = inkattn_fast_ld8<q8>(Krow0, 32*s + 8*c, kv_al4);
            const uint4 kk1 = inkattn_fast_ld8<q8>(Krow1, 32*s + 8*c, kv_al4);
            const uint4 qa0 = s_qa[2*s + 0][lane];
            const uint4 qa1 = s_qa[2*s + 1][lane];
            inkattn_fast_mma(S0, qa0.x, qa0.y, qa0.z, qa0.w, kk0.x, kk0.y);
            inkattn_fast_mma(S0, qa1.x, qa1.y, qa1.z, qa1.w, kk0.z, kk0.w);
            inkattn_fast_mma(S1, qa0.x, qa0.y, qa0.z, qa0.w, kk1.x, kk1.y);
            inkattn_fast_mma(S1, qa1.x, qa1.y, qa1.z, qa1.w, kk1.z, kk1.w);
        }

        S0[0] += mk00.x; S0[1] += mk00.y;
        S1[0] += mk10.x; S1[1] += mk10.y;
        if constexpr (rows16) {
            S0[2] += mk01.x; S0[3] += mk01.y;
            S1[2] += mk11.x; S1[3] += mk11.y;
        }

        // 3. Online softmax with guarded O *= alpha rescaling (bit-exact: alpha == 1.0 when max unchanged).
        float m0 = fmaxf(fmaxf(S0[0], S0[1]), fmaxf(S1[0], S1[1]));
        m0 = fmaxf(m0, __shfl_xor_sync(0xFFFFFFFF, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xFFFFFFFF, m0, 2));
        const float mn0 = fmaxf(mx0, m0);
        const float mu0 = mn0 == -INFINITY ? 0.0f : mn0;

        float mn1 = -INFINITY, mu1 = 0.0f;
        if constexpr (rows16) {
            float m1 = fmaxf(fmaxf(S0[2], S0[3]), fmaxf(S1[2], S1[3]));
            m1 = fmaxf(m1, __shfl_xor_sync(0xFFFFFFFF, m1, 1));
            m1 = fmaxf(m1, __shfl_xor_sync(0xFFFFFFFF, m1, 2));
            mn1 = fmaxf(mx1, m1);
            mu1 = mn1 == -INFINITY ? 0.0f : mn1;
        }

        const bool chg = (mn0 != mx0) || (rows16 && mn1 != mx1);
        if (__any_sync(0xFFFFFFFF, chg)) {
            const float a0 = __expf(mx0 - mu0);
            mx0 = mn0;
            sm0 *= a0;
            if constexpr (rows16) {
                const float a1 = __expf(mx1 - mu1);
                mx1 = mn1;
                sm1 *= a1;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    O[j][0] *= a0; O[j][1] *= a0;
                    O[j][2] *= a1; O[j][3] *= a1;
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    O[j][0] *= a0; O[j][1] *= a0;
                }
            }
        }

        S0[0] = __expf(S0[0] - mu0); S0[1] = __expf(S0[1] - mu0);
        S1[0] = __expf(S1[0] - mu0); S1[1] = __expf(S1[1] - mu0);
        sm0 += (S0[0] + S0[1] + S1[0] + S1[1]);
        if constexpr (rows16) {
            S0[2] = __expf(S0[2] - mu1); S0[3] = __expf(S0[3] - mu1);
            S1[2] = __expf(S1[2] - mu1); S1[3] = __expf(S1[3] - mu1);
            sm1 += (S0[2] + S0[3] + S1[2] + S1[3]);
        }

        const uint32_t pa0 = inkattn_fast_h2u(S0[0], S0[1]);
        const uint32_t pa1 = rows16 ? inkattn_fast_h2u(S0[2], S0[3]) : 0u;
        const uint32_t pa2 = inkattn_fast_h2u(S1[0], S1[1]);
        const uint32_t pa3 = rows16 ? inkattn_fast_h2u(S1[2], S1[3]) : 0u;

        // 4. O += P V: stream V one u-half (8 dims) at a time -> only 16 regs live for V, never overlapping K!
        const int kv0 = (k0 + 2*c + 0 < p.n_kv) ? (k0 + 2*c + 0) : (p.n_kv - 1);
        const int kv1 = (k0 + 2*c + 1 < p.n_kv) ? (k0 + 2*c + 1) : (p.n_kv - 1);
        const int kv2 = (k0 + 2*c + 8 < p.n_kv) ? (k0 + 2*c + 8) : (p.n_kv - 1);
        const int kv3 = (k0 + 2*c + 9 < p.n_kv) ? (k0 + 2*c + 9) : (p.n_kv - 1);
        const char * Vrow0 = Vh + (int64_t)kv0*p.nbv1;
        const char * Vrow1 = Vh + (int64_t)kv1*p.nbv1;
        const char * Vrow2 = Vh + (int64_t)kv2*p.nbv1;
        const char * Vrow3 = Vh + (int64_t)kv3*p.nbv1;

        // For Q8_0, dv = 16*ln + 0..7 (u=0) and 16*ln + 8..15 (u=1) belong to the SAME 32-elem Q8_0 block!
        // Load the 4 scales ONCE and reuse for both u = 0 and u = 1.
        half2 vd0, vd1, vd2, vd3;
        const int v_blk_off = (16*ln / 32)*34;
        const int v_in_blk  = (16*ln) & 31;
        if constexpr (q8) {
            vd0 = __half2half2(__ldg(reinterpret_cast<const half *>(Vrow0 + v_blk_off)));
            vd1 = __half2half2(__ldg(reinterpret_cast<const half *>(Vrow1 + v_blk_off)));
            vd2 = __half2half2(__ldg(reinterpret_cast<const half *>(Vrow2 + v_blk_off)));
            vd3 = __half2half2(__ldg(reinterpret_cast<const half *>(Vrow3 + v_blk_off)));
        }

#pragma unroll
        for (int u = 0; u < 2; ++u) {
            uint4 vr0, vr1, vr2, vr3;
            if constexpr (!q8) {
                const int dv = (16*ln + 8*u)*(int)sizeof(half);
                vr0 = inkattn_fast_ld16(Vrow0 + dv);
                vr1 = inkattn_fast_ld16(Vrow1 + dv);
                vr2 = inkattn_fast_ld16(Vrow2 + dv);
                vr3 = inkattn_fast_ld16(Vrow3 + dv);
            } else {
                const int qoff = v_blk_off + 2 + v_in_blk + 8*u;
                vr0 = inkattn_fast_q8_load8_with_d(Vrow0 + qoff, vd0, kv_al4);
                vr1 = inkattn_fast_q8_load8_with_d(Vrow1 + qoff, vd1, kv_al4);
                vr2 = inkattn_fast_q8_load8_with_d(Vrow2 + qoff, vd2, kv_al4);
                vr3 = inkattn_fast_q8_load8_with_d(Vrow3 + qoff, vd3, kv_al4);
            }
            const uint32_t * va = reinterpret_cast<const uint32_t *>(&vr0);
            const uint32_t * vb = reinterpret_cast<const uint32_t *>(&vr1);
            const uint32_t * vc = reinterpret_cast<const uint32_t *>(&vr2);
            const uint32_t * vd = reinterpret_cast<const uint32_t *>(&vr3);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int j = 8*u + 2*w;
                inkattn_fast_mma(O[j + 0], pa0, pa1, pa2, pa3, __byte_perm(va[w], vb[w], 0x5410), __byte_perm(vc[w], vd[w], 0x5410));
                inkattn_fast_mma(O[j + 1], pa0, pa1, pa2, pa3, __byte_perm(va[w], vb[w], 0x7632), __byte_perm(vc[w], vd[w], 0x7632));
            }
        }
    }

    // Finalize lane-partial sums.
    sm0 += __shfl_xor_sync(0xFFFFFFFF, sm0, 1);
    sm0 += __shfl_xor_sync(0xFFFFFFFF, sm0, 2);
    if constexpr (rows16) {
        sm1 += __shfl_xor_sync(0xFFFFFFFF, sm1, 1);
        sm1 += __shfl_xor_sync(0xFFFFFFFF, sm1, 2);
    }

    const int64_t base = ((int64_t)(seq*p.Hkv + kvh)*p.W + wi)*16;

    // Vectorized float4 Op stores, skipped completely for rows that saw no finite tiles (mx == -INF).
    if (r0 < R) {
        if (c == 0) {
            p.Mp[base + r0] = make_float2(mx0, sm0);
        }
        if (mx0 != -INFINITY) {
            float * o = p.Op + (base + r0)*D;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int j = 4*k;
                *reinterpret_cast<float4 *>(o + 32*c + j)      = make_float4(O[j+0][0], O[j+1][0], O[j+2][0], O[j+3][0]);
                *reinterpret_cast<float4 *>(o + 32*c + 16 + j) = make_float4(O[j+0][1], O[j+1][1], O[j+2][1], O[j+3][1]);
            }
        }
    }
    if constexpr (rows16) {
        if (r1 < R) {
            if (c == 0) {
                p.Mp[base + r1] = make_float2(mx1, sm1);
            }
            if (mx1 != -INFINITY) {
                float * o = p.Op + (base + r1)*D;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int j = 4*k;
                    *reinterpret_cast<float4 *>(o + 32*c + j)      = make_float4(O[j+0][2], O[j+1][2], O[j+2][2], O[j+3][2]);
                    *reinterpret_cast<float4 *>(o + 32*c + 16 + j) = make_float4(O[j+0][3], O[j+1][3], O[j+2][3], O[j+3][3]);
                }
            }
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif
}

// Fast combine: caches Mp in shared memory (single coalesced pass) and only touches live warps in Op.
static __global__ void __launch_bounds__(INKATTN_FAST_D, 4)
inkattn_fast_combine(
        const float * __restrict__ Op, const float2 * __restrict__ Mp, const float * __restrict__ sinks,
        char * __restrict__ dst, const int64_t nbd1, const int64_t nbd2, const int64_t nbd3,
        float * __restrict__ lse, const float * __restrict__ gate, const int64_t gate_t,
        const int W, const int g, const int Hkv) {
    constexpr int D = INKATTN_FAST_D;
    const int h   = blockIdx.x;
    const int j   = blockIdx.y;
    const int seq = blockIdx.z;
    const int H   = gridDim.x;
    const int T   = gridDim.y;
    const int tid = threadIdx.x;
    const int kvh = h / g;
    const int r   = j*g + h % g;

    const int64_t base = (int64_t)(seq*Hkv + kvh)*W*16 + r;

    extern __shared__ float smem_c[];
    float2 * s_mp = reinterpret_cast<float2 *>(smem_c); // W entries

    // Coalesced cooperative load of Mp[0..W-1] + parallel max reduction.
    float my_max = -INFINITY;
    for (int w = tid; w < W; w += D) {
        const float2 ms = __ldg(Mp + base + (int64_t)w*16);
        s_mp[w] = ms;
        my_max = fmaxf(my_max, ms.x);
    }
    // Warp + block reduction for global M across W warps.
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 1));
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 2));
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 4));
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 8));
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, 16));
    __shared__ float warp_max[4];
    if ((tid & 31) == 0) {
        warp_max[tid >> 5] = my_max;
    }
    __syncthreads();
    float M = fmaxf(fmaxf(warp_max[0], warp_max[1]), fmaxf(warp_max[2], warp_max[3]));

    float L   = 0.0f;
    float acc = 0.0f;
    if (M != -INFINITY) {
        for (int w = 0; w < W; ++w) {
            const float2 ms = s_mp[w];
            if (ms.x == -INFINITY) {
                continue;
            }
            const float f = __expf(ms.x - M);
            L   += f*ms.y;
            acc += f*__ldg(Op + (base + (int64_t)w*16)*D + tid);
        }
    }
    if (sinks) {
        const float s  = sinks[h];
        const float M2 = fmaxf(M, s);
        const float f  = M == -INFINITY ? 0.0f : __expf(M - M2);
        L   = L*f + __expf(s - M2);
        acc = acc*f;
        M   = M2;
    }
    float * d = reinterpret_cast<float *>(dst + seq*nbd3 + j*nbd2 + h*nbd1);
    float res = L > 0.0f ? acc / L : 0.0f;
    if (gate) {
        res = res * (1.0f / (1.0f + expf(-gate[(int64_t)j*gate_t + h*D + tid])));
    }
    d[tid] = res;
    if (lse && tid == 0) {
        lse[((int64_t)seq*T + j)*H + h] = L > 0.0f ? M + logf(L) : -INFINITY;
    }
}

static int inkattn_fast_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

static bool inkattn_fast_aligned(const void * ptr, const int64_t a, const int64_t b, const int64_t c, const int64_t n) {
    return (reinterpret_cast<uintptr_t>(ptr) % n) == 0 && a % n == 0 && b % n == 0 && c % n == 0;
}

} // namespace

bool ggml_cuda_inkattn_fast_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_INKATTN_FAST");
        return !(e && *e && atoi(e) == 0);
    }();
    return on;
}

void ggml_cuda_inkattn_fast_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * mask,
        const ggml_tensor * snk,
        ggml_tensor * dst,
        ggml_tensor * lse,
        const ggml_tensor * tab,
        const ggml_tensor * rp,
        const ggml_tensor * gate,
        float scale,
        bool inkrel,
        bool q8) {
    const int T    = (int) Q->ne[1];
    const int H    = (int) Q->ne[2];
    const int Hkv  = (int) K->ne[2];
    const int S    = (int) Q->ne[3];
    const int g    = H / Hkv;
    const int n_kv = (int) K->ne[1];

    const bool rows16 = g*T > 8;
    const int  nsm    = ggml_cuda_info().devices[ctx.device].nsm;

    // With 4 CTAs/SM (16 warps/SM) and cyclic striding, choose W so all SMs stay busy without bloating combine.
    // Default WPS = 16, TPW = 4 (at n_kv=3072, NT=192 -> W=48 -> 96 blocks = 1 single wave on 84 SMs at 4 CTAs/SM!).
    static const int tgt_wps = inkattn_fast_env_int("GGML_CUDA_INKATTN_WPS", 16);
    static const int min_tpw = inkattn_fast_env_int("GGML_CUDA_INKATTN_TPW", 4);
    const int NT = (n_kv + 15) / 16;
    int W = (nsm*tgt_wps + S*Hkv - 1) / (S*Hkv);
    W = std::min(W, std::max(1, NT / std::max(1, min_tpw)));
    W = std::max(W, 1);

    ggml_cuda_pool_alloc<float>  Op(ctx.pool(), (size_t) S*Hkv*W*16*INKATTN_FAST_D);
    ggml_cuda_pool_alloc<float2> Mp(ctx.pool(), (size_t) S*Hkv*W*16);

    inkattn_fast_params p;
    p.Q = (const char *) Q->data; p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.K = (const char *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.V = (const char *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.M = (const char *) mask->data; p.nbm1 = mask->nb[1]; p.nbm2 = mask->nb[2]; p.nbm3 = mask->nb[3];
    p.ne32 = (int) mask->ne[2]; p.ne33 = (int) mask->ne[3];
    p.mask_al4 = inkattn_fast_aligned(mask->data, mask->nb[1], mask->nb[2], mask->nb[3], 4) ? 1 : 0;
    p.kv_al4   = (inkattn_fast_aligned(K->data, K->nb[1], K->nb[2], K->nb[3], 4) &&
                  inkattn_fast_aligned(V->data, V->nb[1], V->nb[2], V->nb[3], 4)) ? 1 : 0;
    p.n_kv = n_kv; p.T = T; p.g = g; p.H = H; p.Hkv = Hkv; p.W = W; p.scale = scale;
    p.Op = Op.get(); p.Mp = Mp.get();
    p.tab = nullptr; p.tab_h1 = 0; p.tab_t = 0; p.rp = nullptr; p.rp_t = 0;
    if (inkrel && tab && rp) {
        p.tab = (const float *) tab->data; p.tab_h1 = tab->nb[1]/sizeof(float); p.tab_t = tab->nb[2]/sizeof(float);
        p.rp  = (const int32_t *) rp->data; p.rp_t = rp->nb[1]/sizeof(int32_t);
    }

    cudaStream_t stream = ctx.stream();
    const dim3 grid((W + INKATTN_FAST_WARPS - 1)/INKATTN_FAST_WARPS, Hkv, S);
    auto launch = [&](auto r16, auto rel, auto kq8) {
        inkattn_fast_decode<decltype(r16)::value, decltype(rel)::value, decltype(kq8)::value>
            <<<grid, INKATTN_FAST_WARPS*WARP_SIZE, 0, stream>>>(p);
    };
    using T_ = std::true_type;
    using F_ = std::false_type;
    if (q8) {
        if (inkrel) { rows16 ? launch(T_{}, T_{}, T_{}) : launch(F_{}, T_{}, T_{}); }
        else        { rows16 ? launch(T_{}, F_{}, T_{}) : launch(F_{}, F_{}, T_{}); }
    } else {
        if (inkrel) { rows16 ? launch(T_{}, T_{}, F_{}) : launch(F_{}, T_{}, F_{}); }
        else        { rows16 ? launch(T_{}, F_{}, F_{}) : launch(F_{}, F_{}, F_{}); }
    }
    CUDA_CHECK(cudaGetLastError());

    const size_t combine_smem = (size_t) W * sizeof(float2);
    inkattn_fast_combine<<<dim3(H, T, S), INKATTN_FAST_D, combine_smem, stream>>>(
        Op.get(), Mp.get(), snk ? (const float *) snk->data : nullptr,
        (char *) dst->data, dst->nb[1], dst->nb[2], dst->nb[3],
        lse ? (float *) lse->data : nullptr,
        gate ? (const float *) gate->data : nullptr,
        gate ? (int64_t)(gate->nb[1]/sizeof(float)) : 0,
        W, g, Hkv);
    CUDA_CHECK(cudaGetLastError());
}
