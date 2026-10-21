// RQ4 CUDA kernels: rotate+quantize activations, W4A8 GEMM on mma.m16n8k32.s8 (SM80+).
// Format description: ggml/src/ggml-rq4.h. This header has no ggml dependencies so the standalone
// microbenchmark (tools/rq4-m1) and the ggml backend use the same kernels.
//
// Weight-unpack hook (for trellis / other codebooks, see docs/rq4-interface.md):
//   The GEMM takes a WeightCodec template parameter that owns (a) the per-stage shared-memory layout
//   of the weight tile, (b) the cp.async loader from global memory and (c) the producer of the four
//   s8 A-fragment registers for one m16 tile and one k32 step, plus the per-group scale lookup.
//   rq4_codec_nibble<GS> is the RQ4 implementation, rq4_codec_nl<GS> the RQ4NL one (table lookup).

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#include "../ggml-rq4.h"

namespace rq4 {

static __device__ __constant__ uint32_t d_sign_bits[8] = { RQ4_SIGN_WORDS };

// ---------------------------------------------------------------------------------------------
// small PTX helpers (cp.async and IMMA m16n8k32 need SM80+; older targets compile to a trap and the
// ggml backend never routes RQ4 to them, see ggml_cuda_rq4_supported)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define RQ4_PRE_AMPERE 1
#endif

static __device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool pred) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
#ifdef RQ4_PRE_AMPERE
    __trap(); (void) s; (void) n;
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}

static __device__ __forceinline__ void cp_async4(void * smem, const void * gmem, bool pred) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = pred ? 4 : 0;
#ifdef RQ4_PRE_AMPERE
    __trap(); (void) s; (void) n;
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#ifndef RQ4_PRE_AMPERE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int n>
static __device__ __forceinline__ void cp_async_wait() {
#ifndef RQ4_PRE_AMPERE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
#endif
}

static __device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef RQ4_PRE_AMPERE
    __trap(); (void) d; (void) a; (void) b0; (void) b1;
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// exact int -> float for |x| < 2^22 with one IADD + one FADD (I2F is 16/clk/SM on SM86)
static __device__ __forceinline__ float magic_i2f(int x) {
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

// ---------------------------------------------------------------------------------------------
// Activations: rotate (D, H256/16) + int8 quantize, one fp32 scale per (token, 256 block).
// x: [N][K] f32 with row stride sx (elements). q: [N][K] int8 (fragment byte order), s: [N][K/256].
// One warp per (token, 256 block), 4 warps per CTA along K.

// Optional scratch layout swaps the two 8-byte halves of each vector for token bit 1.
// The default fragment layout remains unchanged.
template <bool SWAP8 = false>
static __global__ void __launch_bounds__(128) quantize_act(
        const float * __restrict__ x, int64_t sx, int8_t * __restrict__ q, float * __restrict__ s, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int hb   = blockIdx.x*4 + (threadIdx.x >> 5);
    const int tok  = blockIdx.y;
    if (hb*RQ4_HB >= K || tok >= N) {
        return;
    }
    const float * xr = x + (int64_t) tok*sx + hb*RQ4_HB + lane*8;
    const float4 u0 = *(const float4 *) xr;
    const float4 u1 = *(const float4 *) (xr + 4);
    float v[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };

    const uint32_t sw = d_sign_bits[lane >> 2] >> ((lane & 3)*8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1) {
            v[e] = -v[e];
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2*h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffff, v[e], m);
            v[e] = (lane & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= 1.0f/16.0f;
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    }
    const float d  = amax/127.0f;
    const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
    uint32_t w[2] = { 0, 0 };
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int qi = __float2int_rn(v[e]*id);
        w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3)*8);
    }
    // element index i = lane*8 + e inside the 256 block: chunk c = lane/16, j = (lane%16)/4,
    // h = (lane%4)/2, t = (lane%2)*2 + e/4, b = e%4 -> byte c*128 + t*32 + j*8 + h*4 + b
    const int c  = lane >> 4;
    const int j  = (lane & 15) >> 2;
    const int h  = (lane & 3) >> 1;
    const int t0 = (lane & 1)*2;
    int8_t * qr = q + (int64_t) tok*K + hb*RQ4_HB + c*RQ4_CHUNK + (j ^ (SWAP8 ? ((tok >> 1) & 1) : 0))*8 + h*4;
    *(uint32_t *) (qr + (t0 + 0)*32) = w[0];
    *(uint32_t *) (qr + (t0 + 1)*32) = w[1];
    if (lane == 0) {
        s[(int64_t) tok*(K/RQ4_HB) + hb] = d;
    }
}

// ---------------------------------------------------------------------------------------------
// Weight codec: RQ4 nibbles. Per stage (one 128-wide K chunk) for BM rows:
//   nib: BM x 64 bytes (row stride 64, LDS.128 per lane is conflict-free per 8-lane phase)
//   dsc: BM x SCB bytes, the whole fp16 scale header of the current 1024 super block

template <int GS_>
struct rq4_codec_nibble {
    static constexpr int GS   = GS_;
    static constexpr int NG   = RQ4_SB/GS;          // groups per super block
    static constexpr int SCB  = 2*NG;               // scale header bytes (16 or 32)
    static constexpr int SBB  = SCB + RQ4_SB/2;     // super block bytes
    static constexpr int GPC  = RQ4_CHUNK/GS;       // groups per 128 chunk (1 or 2)

    template <int BM>
    static constexpr __host__ __device__ int smem_bytes() { return BM*64 + BM*SCB; }

    // issue cp.async for rows [m0, m0+BM) and chunk kc into stage memory sm
    template <int BM, int NTHREADS>
    static __device__ __forceinline__ void load(uint8_t * sm, const uint8_t * __restrict__ W, int64_t row_bytes,
                                                int m0, int kc, int tid) {
        const int sb  = kc >> 3;
        const int cin = kc & 7;
        uint8_t * snib = sm;
        uint8_t * sdsc = sm + BM*64;
#pragma unroll
        for (int i = tid; i < BM*4; i += NTHREADS) {
            const int r = i >> 2, p = i & 3;
            const uint8_t * src = W + (int64_t) (m0 + r)*row_bytes + (int64_t) sb*SBB + SCB + cin*64 + p*16;
            cp_async16(snib + r*64 + p*16, src, true);
        }
        {
            // scale header: reloaded every chunk (BM*SCB bytes) so each stage is self-contained
            for (int i = tid; i < BM*(SCB/16); i += NTHREADS) {
                const int r = i/(SCB/16), p = i%(SCB/16);
                const uint8_t * src = W + (int64_t) (m0 + r)*row_bytes + (int64_t) sb*SBB + p*16;
                cp_async16(sdsc + r*SCB + p*16, src, true);
            }
        }
    }

    // packed words for rows r (lo) and r+8 (hi) of one m16 tile, lane thread t
    struct frag_t { uint4 lo, hi; };

    static __device__ __forceinline__ frag_t load_frag(const uint8_t * sm, int r, int t) {
        frag_t f;
        f.lo = *(const uint4 *) (sm + r*64 + t*16);
        f.hi = *(const uint4 *) (sm + (r + 8)*64 + t*16);
        return f;
    }

    // same fragment straight from global memory (decode/verify): rlo/rhi = rows r and r+8, chunk kc
    static __device__ __forceinline__ frag_t load_frag_global(const uint8_t * rlo, const uint8_t * rhi, int kc, int t) {
        const int64_t off = (int64_t) (kc >> 3)*SBB + SCB + (kc & 7)*64 + t*16;
        frag_t f;
        f.lo = __ldg((const uint4 *) (rlo + off));
        f.hi = __ldg((const uint4 *) (rhi + off));
        return f;
    }

    static __device__ __forceinline__ float scale_global(const uint8_t * r, int kc, int g) {
        const half * d = (const half *) (r + (int64_t) (kc >> 3)*SBB);
        return __half2float(d[(kc & 7)*GPC + g]);
    }

    // scales of the 2*GPC groups of chunk pair pp (chunks kc0 = 2*pp, kc0 + 1) for one row: a single 4-byte (G128) or 8-byte
    // (G64) load. d[c*GPC + g] = group g of chunk kc0 + c. Alignment: SBB % 16 == 0, offset (kc0 & 7)*GPC*2 % (4*GPC) == 0.
    static __device__ __forceinline__ void scales_pair_global(const uint8_t * r, int pp, float (&d)[2*GPC]) {
        const int kc0 = 2*pp;   // first chunk of pair pp (even by construction)
        const uint8_t * p = r + (int64_t) (kc0 >> 3)*SBB + (kc0 & 7)*GPC*2;
        if constexpr (GPC == 2) {
            const uint2 u = __ldg((const uint2 *) p);
            const float2 a = __half22float2(*(const half2 *) &u.x);
            const float2 b = __half22float2(*(const half2 *) &u.y);
            d[0] = a.x; d[1] = a.y; d[2] = b.x; d[3] = b.y;
        } else {
            static_assert(GPC == 1, "GPC");
            const unsigned int u = __ldg((const unsigned int *) p);
            const float2 a = __half22float2(*(const half2 *) &u);
            d[0] = a.x; d[1] = a.y;
        }
    }

    static __device__ __forceinline__ uint32_t word(const uint4 & u, int j) {
        return j == 0 ? u.x : j == 1 ? u.y : j == 2 ? u.z : u.w;
    }

    // A registers of IMMA j (k32 step j of the chunk); values are 16*code
    static __device__ __forceinline__ void a_regs(const frag_t & f, int j, uint32_t (&a)[4]) {
        const uint32_t wl = word(f.lo, j), wh = word(f.hi, j);
        a[0] = wl & 0xF0F0F0F0u;
        a[1] = wh & 0xF0F0F0F0u;
        a[2] = (wl << 4) & 0xF0F0F0F0u;
        a[3] = (wh << 4) & 0xF0F0F0F0u;
    }

    // scale of group g (0..GPC-1 inside the chunk) for row r; output is multiplied by 1/16 later
    static __device__ __forceinline__ float scale(const uint8_t * sm, int BMv, int r, int cin, int g) {
        const half * d = (const half *) (sm + BMv*64 + r*SCB);
        return __half2float(d[cin*GPC + g]);
    }

    static constexpr float out_mul = 1.0f/16.0f;
};

// ---------------------------------------------------------------------------------------------
// Weight codec: RQ4NL. Same bytes, loads and scales as rq4_codec_nibble; each nibble indexes the int8 table
// rq4_nl_values, so the A registers are table bytes (byte-permute lookup, ~13 ALU ops per 32-bit word) and the
// output factor is 1. |table| <= 127 and |act| <= 127 keep a 128-wide group partial sum below 2^21.

static constexpr uint32_t rq4_pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (uint32_t) (uint8_t) a | ((uint32_t) (uint8_t) b << 8) | ((uint32_t) (uint8_t) c << 16) | ((uint32_t) (uint8_t) d << 24);
}

template <int GS_>
struct rq4_codec_nl : rq4_codec_nibble<GS_> {
    using base = rq4_codec_nibble<GS_>;
    using typename base::frag_t;

    static constexpr int8_t T[16] = { RQ4_NL_VALUES };
    static constexpr uint32_t T0 = rq4_pack4(T[ 0], T[ 1], T[ 2], T[ 3]);
    static constexpr uint32_t T1 = rq4_pack4(T[ 4], T[ 5], T[ 6], T[ 7]);
    static constexpr uint32_t T2 = rq4_pack4(T[ 8], T[ 9], T[10], T[11]);
    static constexpr uint32_t T3 = rq4_pack4(T[12], T[13], T[14], T[15]);

    // 8 nibble indices in q4 -> x: table bytes of the low nibbles of bytes 0..3, y: of the high nibbles
    // (same scheme as ggml get_int_from_table_16: __byte_perm uses 3-bit selectors, bit 3 picks the table half)
    static __device__ __forceinline__ uint2 lut(uint32_t q4) {
        const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
        const uint32_t lo0 = __byte_perm(T0, T1, q4);
        const uint32_t hi0 = __byte_perm(T2, T3, q4);
        const uint32_t lo1 = __byte_perm(T0, T1, q4 >> 16);
        const uint32_t hi1 = __byte_perm(T2, T3, q4 >> 16);
        const uint32_t t0  = __byte_perm(lo0, hi0, sel);
        const uint32_t t1  = __byte_perm(lo1, hi1, sel >> 16);
        return make_uint2(__byte_perm(t0, t1, 0x6420), __byte_perm(t0, t1, 0x7531));
    }

    // A registers of IMMA j: high nibbles -> a[0]/a[1], low nibbles -> a[2]/a[3] (as rq4_codec_nibble)
    static __device__ __forceinline__ void a_regs(const frag_t & f, int j, uint32_t (&a)[4]) {
        const uint2 l = lut(base::word(f.lo, j));
        const uint2 h = lut(base::word(f.hi, j));
        a[0] = l.y;
        a[1] = h.y;
        a[2] = l.x;
        a[3] = h.x;
    }

    static constexpr float out_mul = 1.0f;
};

// ---------------------------------------------------------------------------------------------
// GEMM: Y[n][m] = sum_k W'[m][k] * X'[n][k], M % BM == 0, K % 1024 == 0, any N.
// CTA tile BM=128 x BN, 8 warps as WMW (M) x 8/WMW (N). WMW = 2 (default): warp tile 64 x BN/4. WMW = 4: warp tile
// 32 x BN/2, so each A fragment (and an RQ4NL table lookup) feeds twice as many IMMAs, at the cost of twice the
// activation fragment loads per warp. One 128-wide K chunk per stage.

constexpr int GEMM_BM      = 128;
constexpr int GEMM_THREADS = 256;
constexpr int GEMM_STAGES_DEFAULT = 3;
constexpr int X_STRIDE     = RQ4_CHUNK + 16;   // padded token row in smem (conflict-free LDS.128)

template <class Codec, int BN>
constexpr __host__ __device__ int gemm_stage_bytes() {
    return Codec::template smem_bytes<GEMM_BM>() + BN*X_STRIDE + BN*4;
}

template <class Codec, int BN, int STAGES>
constexpr __host__ __device__ int gemm_smem_bytes() {
    return STAGES*gemm_stage_bytes<Codec, BN>();
}

template <class Codec, int BN, int GEMM_STAGES, int WMW = 2>
static __global__ void __launch_bounds__(GEMM_THREADS, (BN <= 64 && GEMM_STAGES <= 2) ? 2 : 1) gemm(
        const uint8_t * __restrict__ W, int64_t row_bytes,
        const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K) {
    constexpr int BM  = GEMM_BM;
    constexpr int WNW = 8/WMW;    // warps along N
    constexpr int WMR = BM/WMW;   // rows per warp (64 or 32)
    constexpr int WN  = BN/WNW;
    constexpr int NTI = WN/8;     // n8 tiles per warp
    constexpr int MTI = WMR/16;   // m16 tiles per warp
    static_assert(WMW == 2 || WMW == 4, "WMW");
    constexpr int SB  = gemm_stage_bytes<Codec, BN>();
    constexpr int WSB = Codec::template smem_bytes<BM>();
    static_assert(NTI >= 1, "BN too small");

    extern __shared__ __align__(16) uint8_t smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int wm = warp / WNW, wn = warp % WNW;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;
    const int nk = K/RQ4_CHUNK;
    const int KH = K/RQ4_HB;

    auto load_stage = [&](int kc) {
        uint8_t * st = smem + (kc % GEMM_STAGES)*SB;
        Codec::template load<BM, GEMM_THREADS>(st, W, row_bytes, m0, kc, tid);
        uint8_t * sx  = st + WSB;
        float   * sxd = (float *) (sx + BN*X_STRIDE);
#pragma unroll
        for (int i = tid; i < BN*8; i += GEMM_THREADS) {
            const int c = i >> 3, p = i & 7;
            const bool ok = n0 + c < N;
            const int8_t * src = Xq + (int64_t) (ok ? n0 + c : 0)*K + kc*RQ4_CHUNK + p*16;
            cp_async16(sx + c*X_STRIDE + p*16, src, ok);
        }
        for (int i = tid; i < BN; i += GEMM_THREADS) {
            const bool ok = n0 + i < N;
            cp_async4(sxd + i, Xs + (int64_t) (ok ? n0 + i : 0)*KH + (kc*RQ4_CHUNK)/RQ4_HB, ok);
        }
    };

    float acc[MTI][NTI][4];
#pragma unroll
    for (int mt = 0; mt < MTI; ++mt)
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;

#pragma unroll
    for (int s = 0; s < GEMM_STAGES - 1; ++s) {
        if (s < nk) {
            load_stage(s);
        }
        cp_async_commit();
    }

    for (int kc = 0; kc < nk; ++kc) {
        cp_async_wait<GEMM_STAGES - 2>();
        __syncthreads();
        if (kc + GEMM_STAGES - 1 < nk) {
            load_stage(kc + GEMM_STAGES - 1);
        }
        cp_async_commit();

        const uint8_t * st  = smem + (kc % GEMM_STAGES)*SB;
        const uint8_t * sx  = st + WSB;
        const float   * sxd = (const float *) (sx + BN*X_STRIDE);
        const int cin = kc & 7;

        typename Codec::frag_t wf[MTI];
#pragma unroll
        for (int mt = 0; mt < MTI; ++mt) {
            wf[mt] = Codec::load_frag(st, wm*WMR + mt*16 + g, t);
        }
        uint4 xf[NTI][2];
        float dx[NTI][2];
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const int c = wn*WN + nt*8 + g;
            xf[nt][0] = *(const uint4 *) (sx + c*X_STRIDE + t*32);
            xf[nt][1] = *(const uint4 *) (sx + c*X_STRIDE + t*32 + 16);
            dx[nt][0] = sxd[wn*WN + nt*8 + 2*t + 0];
            dx[nt][1] = sxd[wn*WN + nt*8 + 2*t + 1];
        }

        int p[MTI][NTI][4];
#pragma unroll
        for (int mt = 0; mt < MTI; ++mt)
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) p[mt][nt][i] = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int mt = 0; mt < MTI; ++mt) {
                uint32_t a[4];
                Codec::a_regs(wf[mt], j, a);
#pragma unroll
                for (int nt = 0; nt < NTI; ++nt) {
                    const uint4 & u = xf[nt][j >> 1];
                    const uint32_t b0 = (j & 1) ? u.z : u.x;
                    const uint32_t b1 = (j & 1) ? u.w : u.y;
                    mma_s8(p[mt][nt], a, b0, b1);
                }
            }
            constexpr int JPG = 4/Codec::GPC;   // k32 steps per group
            if ((j % JPG) == JPG - 1) {
                const int gi = j/JPG;
#pragma unroll
                for (int mt = 0; mt < MTI; ++mt) {
                    const int r = wm*WMR + mt*16 + g;
                    const float d0 = Codec::scale(st, BM, r,     cin, gi);
                    const float d1 = Codec::scale(st, BM, r + 8, cin, gi);
#pragma unroll
                    for (int nt = 0; nt < NTI; ++nt) {
                        acc[mt][nt][0] += magic_i2f(p[mt][nt][0]) * (d0*dx[nt][0]);
                        acc[mt][nt][1] += magic_i2f(p[mt][nt][1]) * (d0*dx[nt][1]);
                        acc[mt][nt][2] += magic_i2f(p[mt][nt][2]) * (d1*dx[nt][0]);
                        acc[mt][nt][3] += magic_i2f(p[mt][nt][3]) * (d1*dx[nt][1]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) p[mt][nt][i] = 0;
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mt = 0; mt < MTI; ++mt) {
        const int r = m0 + wm*WMR + mt*16 + g;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const int c = n0 + wn*WN + nt*8 + 2*t;
            if (c < N) {
                Y[(int64_t) c*ldy + r]           = acc[mt][nt][0]*Codec::out_mul;
                Y[(int64_t) c*ldy + r + 8]       = acc[mt][nt][2]*Codec::out_mul;
            }
            if (c + 1 < N) {
                Y[(int64_t) (c + 1)*ldy + r]     = acc[mt][nt][1]*Codec::out_mul;
                Y[(int64_t) (c + 1)*ldy + r + 8] = acc[mt][nt][3]*Codec::out_mul;
            }
        }
    }
    (void) M;
}

template <class Codec, int BN, int STAGES = GEMM_STAGES_DEFAULT, int WMW = 2>
static void launch_gemm(const uint8_t * W, int64_t row_bytes, const int8_t * Xq, const float * Xs,
                        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream) {
    constexpr int smem = gemm_smem_bytes<Codec, BN, STAGES>();
    static bool attr_set[64] = {};   // per template instance and device
    int dev = 0;
    cudaGetDevice(&dev);
    if (!attr_set[dev & 63]) {
        cudaFuncSetAttribute(gemm<Codec, BN, STAGES, WMW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set[dev & 63] = true;
    }
    const dim3 grid(M/GEMM_BM, (N + BN - 1)/BN);
    gemm<Codec, BN, STAGES, WMW><<<grid, GEMM_THREADS, smem, stream>>>(W, row_bytes, Xq, Xs, Y, ldy, M, N, K);
}

// ---------------------------------------------------------------------------------------------
// Decode / verify (N <= 8 tokens): one m16n8k32 tile per warp, weights straight from global memory
// (fragment order means a lane's 16 bytes are already its A registers), activations from L1/L2.
// Block = MMV_WARPS warps on the same 16 rows, splitting the 128-wide K chunks round-robin; the
// partial tiles are reduced through shared memory. Tokens n >= N contribute zero B columns.

constexpr int MMV_WARPS  = 4;
constexpr int MMV_UNROLL = 4;   // chunks in flight per warp

template <class Codec>
static __global__ void __launch_bounds__(MMV_WARPS*32) mmv(
        const uint8_t * __restrict__ W, int64_t row_bytes,
        const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x*16;
    const int nk = K/RQ4_CHUNK;
    const int KH = K/RQ4_HB;

    const uint8_t * w_lo = W + (int64_t) (m0 + g)*row_bytes;
    const uint8_t * w_hi = w_lo + 8*row_bytes;
    const bool  has_n  = g < N;
    const int8_t * xr  = Xq + (int64_t) (has_n ? g : 0)*K + t*32;
    const bool  has_c0 = 2*t < N, has_c1 = 2*t + 1 < N;
    const float * xs0  = Xs + (int64_t) (has_c0 ? 2*t     : 0)*KH;
    const float * xs1  = Xs + (int64_t) (has_c1 ? 2*t + 1 : 0)*KH;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int kb = warp; kb < nk; kb += MMV_WARPS*MMV_UNROLL) {
        typename Codec::frag_t wf[MMV_UNROLL];
        uint4 xf[MMV_UNROLL][2];
        float dx[MMV_UNROLL][2];
        float dw[MMV_UNROLL][Codec::GPC][2];
#pragma unroll
        for (int u = 0; u < MMV_UNROLL; ++u) {
            const int kc = kb + u*MMV_WARPS;
            if (kc < nk) {
                wf[u] = Codec::load_frag_global(w_lo, w_hi, kc, t);
#pragma unroll
                for (int gi = 0; gi < Codec::GPC; ++gi) {
                    dw[u][gi][0] = Codec::scale_global(w_lo, kc, gi);
                    dw[u][gi][1] = Codec::scale_global(w_hi, kc, gi);
                }
                const uint4 z = make_uint4(0, 0, 0, 0);
                xf[u][0] = has_n ? *(const uint4 *) (xr + (int64_t) kc*RQ4_CHUNK)      : z;
                xf[u][1] = has_n ? *(const uint4 *) (xr + (int64_t) kc*RQ4_CHUNK + 16) : z;
                dx[u][0] = has_c0 ? xs0[kc >> 1] : 0.0f;
                dx[u][1] = has_c1 ? xs1[kc >> 1] : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < MMV_UNROLL; ++u) {
            const int kc = kb + u*MMV_WARPS;
            if (kc >= nk) {
                break;
            }
            int p[4] = { 0, 0, 0, 0 };
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t a[4];
                Codec::a_regs(wf[u], j, a);
                const uint4 & xv = xf[u][j >> 1];
                mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
                constexpr int JPG = 4/Codec::GPC;
                if ((j % JPG) == JPG - 1) {
                    const int gi = j/JPG;
                    acc[0] += magic_i2f(p[0]) * (dw[u][gi][0]*dx[u][0]);
                    acc[1] += magic_i2f(p[1]) * (dw[u][gi][0]*dx[u][1]);
                    acc[2] += magic_i2f(p[2]) * (dw[u][gi][1]*dx[u][0]);
                    acc[3] += magic_i2f(p[3]) * (dw[u][gi][1]*dx[u][1]);
                    p[0] = p[1] = p[2] = p[3] = 0;
                }
            }
        }
    }

    __shared__ float red[MMV_WARPS - 1][32][4];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[warp - 1][lane][i] = acc[i];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < MMV_WARPS - 1; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += red[w][lane][i];
            }
        }
        const int r = m0 + g;
        const int c = 2*t;
        if (c < N) {
            Y[(int64_t) c*ldy + r]           = acc[0]*Codec::out_mul;
            Y[(int64_t) c*ldy + r + 8]       = acc[2]*Codec::out_mul;
        }
        if (c + 1 < N) {
            Y[(int64_t) (c + 1)*ldy + r]     = acc[1]*Codec::out_mul;
            Y[(int64_t) (c + 1)*ldy + r + 8] = acc[3]*Codec::out_mul;
        }
    }
    (void) M;
}

// Pair variant (default): each warp takes whole 256-wide chunk pairs (one activation scale block), so a row's
// group scales for the pair come in one 4/8-byte load instead of GPC 2-byte loads per chunk, the activation scale
// once per pair instead of once per chunk, and a row's 128 weight bytes per pair are contiguous. Pairs per row =
// K/256 is a multiple of MMV_WARPS (K % 1024 == 0), so the warps stay balanced.

// MINB = minimum resident blocks per SM. 4 caps the kernel at 128 registers (no spills on SM86 for any codec) so
// 4 blocks fit per SM: for M = 5120 that is one wave on an 82-SM card (320 blocks <= 328) instead of 1.3 at 3 blocks
// (larger M stays multi-wave either way).
constexpr int MMV_PAIRS = 2;   // chunk pairs in flight per warp (4 chunks, as MMV_UNROLL)

template <class Codec, int MINB = 4>
static __global__ void __launch_bounds__(MMV_WARPS*32, MINB) mmv_pair(
        const uint8_t * __restrict__ W, int64_t row_bytes,
        const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K) {
    constexpr int GPC = Codec::GPC;
    constexpr int JPG = 4/GPC;   // k32 steps per group
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x*16;
    const int np = K/RQ4_HB;     // chunk pairs = activation scale blocks

    const uint8_t * w_lo = W + (int64_t) (m0 + g)*row_bytes;
    const uint8_t * w_hi = w_lo + 8*row_bytes;
    const bool  has_n  = g < N;
    const int8_t * xr  = Xq + (int64_t) (has_n ? g : 0)*K + t*32;
    const bool  has_c0 = 2*t < N, has_c1 = 2*t + 1 < N;
    const float * xs0  = Xs + (int64_t) (has_c0 ? 2*t     : 0)*np;
    const float * xs1  = Xs + (int64_t) (has_c1 ? 2*t + 1 : 0)*np;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int pb = warp; pb < np; pb += MMV_WARPS*MMV_PAIRS) {
        typename Codec::frag_t wf[MMV_PAIRS][2];
        uint4 xf[MMV_PAIRS][2][2];
        float dx[MMV_PAIRS][2];
        float dw[MMV_PAIRS][2][2*GPC];
#pragma unroll
        for (int u = 0; u < MMV_PAIRS; ++u) {
            const int pp = pb + u*MMV_WARPS;
            if (pp < np) {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int kc = 2*pp + c;
                    wf[u][c] = Codec::load_frag_global(w_lo, w_hi, kc, t);
                    xf[u][c][0] = has_n ? *(const uint4 *) (xr + (int64_t) kc*RQ4_CHUNK)      : z;
                    xf[u][c][1] = has_n ? *(const uint4 *) (xr + (int64_t) kc*RQ4_CHUNK + 16) : z;
                }
                Codec::scales_pair_global(w_lo, pp, dw[u][0]);
                Codec::scales_pair_global(w_hi, pp, dw[u][1]);
                dx[u][0] = has_c0 ? xs0[pp] : 0.0f;
                dx[u][1] = has_c1 ? xs1[pp] : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < MMV_PAIRS; ++u) {
            if (pb + u*MMV_WARPS >= np) {
                break;
            }
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                int p[4] = { 0, 0, 0, 0 };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    Codec::a_regs(wf[u][c], j, a);
                    const uint4 & xv = xf[u][c][j >> 1];
                    mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
                    if ((j % JPG) == JPG - 1) {
                        const int gi = c*GPC + j/JPG;
                        acc[0] += magic_i2f(p[0]) * (dw[u][0][gi]*dx[u][0]);
                        acc[1] += magic_i2f(p[1]) * (dw[u][0][gi]*dx[u][1]);
                        acc[2] += magic_i2f(p[2]) * (dw[u][1][gi]*dx[u][0]);
                        acc[3] += magic_i2f(p[3]) * (dw[u][1][gi]*dx[u][1]);
                        p[0] = p[1] = p[2] = p[3] = 0;
                    }
                }
            }
        }
    }

    __shared__ float red[MMV_WARPS - 1][32][4];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[warp - 1][lane][i] = acc[i];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < MMV_WARPS - 1; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += red[w][lane][i];
            }
        }
        const int r = m0 + g;
        const int c = 2*t;
        if (c < N) {
            Y[(int64_t) c*ldy + r]           = acc[0]*Codec::out_mul;
            Y[(int64_t) c*ldy + r + 8]       = acc[2]*Codec::out_mul;
        }
        if (c + 1 < N) {
            Y[(int64_t) (c + 1)*ldy + r]     = acc[1]*Codec::out_mul;
            Y[(int64_t) (c + 1)*ldy + r + 8] = acc[3]*Codec::out_mul;
        }
    }
    (void) M;
}

template <class Codec, int MINB = 4>
static void launch_mmv_pair(const uint8_t * W, int64_t row_bytes, const int8_t * Xq, const float * Xs,
                            float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream) {
    mmv_pair<Codec, MINB><<<M/16, MMV_WARPS*32, 0, stream>>>(W, row_bytes, Xq, Xs, Y, ldy, M, N, K);
}

template <class Codec>
static void launch_mmv(const uint8_t * W, int64_t row_bytes, const int8_t * Xq, const float * Xs,
                       float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream) {
    mmv<Codec><<<M/16, MMV_WARPS*32, 0, stream>>>(W, row_bytes, Xq, Xs, Y, ldy, M, N, K);
}

template <bool SWAP8 = false>
static void launch_quantize_act(const float * x, int64_t sx, int8_t * q, float * s, int K, int N, cudaStream_t stream) {
    const dim3 grid((K/RQ4_HB + 3)/4, N);
    quantize_act<SWAP8><<<grid, 128, 0, stream>>>(x, sx, q, s, K, N);
}

} // namespace rq4
