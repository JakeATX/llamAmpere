// AI+ tile layout (AITk, ggml-ait.h): CUDA reference path = rotated-basis reconstruct (level x row scale / 64) x the
// emulated activation path of the tile kernel (H256 rotation, x column scale, int8 round trip with one scale per
// (token, slice)), then cuBLAS SGEMM in row slabs. This is the W4A8-style arithmetic of the fast kernel up to summation
// order, so KL measured on it is the KL of the format. N <= 8 tokens take the tile-chain IMMA kernel (k_ait_mmv) unless
// GGML_CUDA_AIT_ORDERED=0 restores the legacy unordered sum for comparison.
// GGML_CUDA_AIT_REF=1; GGML_CUDA_AIT_WARPS = warps per block of it (default 12),
// GGML_CUDA_AIT_FULL = 1 (default): fully unrolled chunk loop, 0 = rolled 4 x 4 groups (Gate 2 10-23: unrolled 8-13% faster at N=1, every rate). GGML_CUDA_AIT_EXACT=1: f32 activations (no int8 round trip; the
// A/B for the activation quantizer). GGML_CUDA_AIT_ASLICE = 512 (default) or 256: activation scale slice.
// AIT2P (code 1, k = 2 bytes, pair-table levels): the same kernels with the table in shared memory, one 32-bit read per
// weight pair; GGML_CUDA_AIT_PLDS = 1 (default): u32 entries, 16 lane copies (64 KB, one SHF + LOP3 per address, lanes
// l and l + 16 share a copy), 2: u16 entries, one copy per bank (64 KB, conflict-free, two more ALU ops per address).
// GGML_CUDA_AIT_OCC = 0 (default), 1 or 2: AIT4 low-register experiment with a two-group or one-group prefetch ring.
// Modes 3/4 cap registers; GGML_CUDA_AIT_OCC_GRID sets their blocks per SM (1..8, default 4).
// GGML_CUDA_AIT3_PACK: 1 packs each AIT3 row tile with a tight register cap; 2 keeps the base register budget.
// GGML_CUDA_AIT3_RING2=1 keeps two AIT3 weight groups live with the base register budget.
// GGML_CUDA_AIT3_RING1=1 keeps one AIT3 weight group live; it takes precedence over RING2 and PACK.
// GGML_CUDA_AIT4_NEIGHBOR=1 loads one weight vector per lane and shuffles its neighbor instead of loading it twice.
// GGML_CUDA_AIT4_NEIGHBOR_PACK=1 packs one MMA tile at a time in the neighbor path.
// GGML_CUDA_AIT4_NEIGHBOR_TAIL=1 refills the neighbor ring after its IMMA work.
// GGML_CUDA_AIT4_NEIGHBOR_EARLY_X=1 reads activations before the neighbor group's decode.
// GGML_CUDA_AIT4_NEIGHBOR_BOUNDS=1 gives the neighbor kernel a 384-thread compile bound.
// GGML_CUDA_AIT4_NEIGHBOR_RING1=1 keeps one weight group live in the neighbor path.
// GGML_CUDA_AIT4_NEIGHBOR_UNROLL2=1 unrolls two four-group steps in the neighbor path.
// GGML_CUDA_AIT4_NEIGHBOR_OCC=1 launches two register-capped blocks per SM without tile packing.
// GGML_CUDA_AIT4_RING2=1 keeps two AIT4 weight groups live with the base register budget (OCC=0, NEIGHBOR=0).
// GGML_CUDA_AIT4_RING1=1 keeps one AIT4 weight group live and takes precedence over RING2.
// GGML_CUDA_AIT4_TAIL=1 refills the four-group weight ring after each group's IMMA work.
// GGML_CUDA_AIT4_PACK=1 packs each AIT4 MMA tile at the base register budget and four-group ring.
// GGML_CUDA_AIT4_UNROLL2=1 unrolls two four-group steps instead of all four steps.
// GGML_CUDA_AIT4_EARLY_X=1 loads each activation vector before decoding its weight group.
// GGML_CUDA_AIT4_BOUNDS=1 sets the plain decode thread bound to 384 for a larger register budget.
#include "ait.cuh"
#include "ait-mm.cuh"
#include "ai4.cuh"
#include "ggml.h"

#include <algorithm>
#include <cstdlib>

static int ait_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

// W[mm][K] f32 (rotated basis, no col scale: that rides on the activations): level * row scale / 64.
// src = first byte of a 32-row-aligned slab. grid (K/512, mm/32), 256 threads.
static __global__ void __launch_bounds__(256) k_ait_reconstruct(const uint8_t * __restrict__ src, float * __restrict__ W,
        const int64_t K, const int k, const int code) {
    const int64_t S = K / 512, ks = blockIdx.x, rt = blockIdx.y;
    const uint8_t * chunk = src + (rt * S + ks) * ait_dev_chunk_bytes(k);
    __shared__ float rsc[32];
    if (threadIdx.x < 32) {
        rsc[threadIdx.x] = ait_dev_row_scale(src, K, k, rt * 32 + threadIdx.x, ks * 512);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 32 * 512; e += 256) {
        const int rr = e >> 9, kk = e & 511;
        const int lev = ait_dev_weight_level(chunk, k, rr, kk, code);
        W[(rt * 32 + rr) * K + ks * 512 + kk] = (float) lev * rsc[rr] * (1.0f / 64.0f);
    }
}

// cs[K] from the side slots (rows 0..1023 of the tensor hold every column once)
static __global__ void k_ait_cols(const uint8_t * __restrict__ src, float * __restrict__ cs, const int64_t K, const int k) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        cs[c] = ait_dev_col_scale(src, K, k, c);
    }
}

// X [N][K] -> Y: rotated per 256 block, times cs[k], then (emul) int8 round trip with amax/127 per (token, SL slice).
// grid (K/256, N), 32 threads; lane holds 8 consecutive elements of the 256 block.
template <int SL>
static __global__ void __launch_bounds__(32) k_ait_act(const float * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int64_t K, const int emul) {
    const int l = threadIdx.x;
    const int64_t kb = (int64_t) blockIdx.x * 256 + l * 8;
    const int64_t off = (int64_t) blockIdx.y * K + kb;
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] = ai4_dev_sign_neg(l * 8 + q) ? -X[off + q] : X[off + q];
    }
    ai4_dev_fwht256(v);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] *= (1.0f / 16.0f) * cs[kb + q];
    }
    if (!emul) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            Y[off + q] = v[q];
        }
        return;
    }
    float amax = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        amax = fmaxf(amax, fabsf(v[q]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    if (SL == 512) {
        // the partner 256 block of the same 512 slice (K % 1024 == 0 keeps slices aligned): recompute its amax here
        const int64_t pb = (int64_t) (blockIdx.x ^ 1) * 256 + l * 8;
        const int64_t poff = (int64_t) blockIdx.y * K + pb;
        float w[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            w[q] = ai4_dev_sign_neg(l * 8 + q) ? -X[poff + q] : X[poff + q];
        }
        ai4_dev_fwht256(w);
        float am2 = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            am2 = fmaxf(am2, fabsf(w[q] * (1.0f / 16.0f) * cs[pb + q]));
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            am2 = fmaxf(am2, __shfl_xor_sync(0xffffffffu, am2, m));
        }
        amax = fmaxf(amax, am2);
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        Y[off + q] = d * (float) __float2int_rn(v[q] * id);
    }
}

// ---------------------------------------------------------------- fast path (N <= 8 tokens): tile-chain IMMA decode
// The AIT trellis bytes are the tile layout of the AIV2 Gate-2 kernel (kern/tmmv.cu, 1MAD, B = 8, TR = 32, BQ = 8k bits per
// lane per tile), with 64 side bytes after each 2048k-byte chunk. Activations: rotated, x col scale, int8 per (token, 512 K)
// in the m16n8k32 B-fragment slot order ([K/32][NT][32], byte 8q + 4hs + e), scale SA = amax / 127 / 64 (the format's 1/64).
namespace aitk {
static __device__ __forceinline__ uint4 ldw16(const char * p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint32_t comp(const uint4 & v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// 1MAD level as an s8 in byte 1 of the dp4a result: 64 (bytesum + 2) - 32768
template <int KB> static __device__ __forceinline__ uint32_t lv1(uint32_t st) { return __dp4a(st * AIT_MUL_FOR_K(KB), 0x40404040u, 128u - 32768u); }
static __device__ __forceinline__ uint32_t pack4(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    return __byte_perm(__byte_perm(r0, r1, 0x0051), __byte_perm(r2, r3, 0x0051), 0x5410);
}
static __device__ __forceinline__ void mma_s8(int * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 8 states of one lane in one tile: weight j's state = 16 bits at j k of the lane's window (BASE: second window for j >= 4
// when the 8 states do not fit one 32-bit window)
template <int KB, int J> static __device__ __forceinline__ uint32_t st1(uint32_t A, uint32_t F1) {
    constexpr bool ONE = 7 * KB + 16 <= 32;
    constexpr int base = (ONE || J < 4) ? 0 : 4 * KB, d = J * KB - base;
    const uint32_t f = base == 0 ? A : F1;
    if constexpr (d == 0) { return lv1<KB>(f & 0xFFFFu); } else { return lv1<KB>((f >> d) & 0xFFFFu); }
}
// AIT2P pair entry (int8 levels in bytes 0, 1) of window w from the shared table. tsm = table base, lo = lane offset
// (PC 1: 4 (lane % NC), PC 2: 4 lane). PC 1: u32 entries, NC copies, byte (e NC + c) 4. PC 2: u16 entries, one copy per
// bank: byte 128 (e / 2) + 4 lane + 2 (e % 2).
template <int PC, int NC> static __device__ __forceinline__ uint32_t pent(const char * tsm, const uint32_t lo, const uint32_t w) {
    const uint32_t h = w * w + w;
    if constexpr (PC == 1) {
        constexpr int LC = NC == 16 ? 4 : NC == 8 ? 3 : NC == 4 ? 2 : NC == 2 ? 1 : 0;
        static_assert((1 << LC) == NC, "NC");
        const uint32_t a = ((h >> (8 - LC)) & (1023u << (LC + 2))) | lo;
        return *reinterpret_cast<const uint32_t *>(tsm + a);
    } else {
        const uint32_t a = ((h >> 4) & 0xFF80u) | ((h >> 9) & 2u) | lo;
        return *reinterpret_cast<const uint16_t *>(tsm + a);
    }
}
// fill the shared table (all threads of the block, then a barrier); bytes PC 1: 4096 NC, PC 2: 65536
template <int PC, int NC> static __device__ __forceinline__ void pfill(uint32_t * t32) {
    if constexpr (PC == 1) {
        for (int i = threadIdx.x; i < 1024 * NC; i += blockDim.x) {
            const int e = i / NC;
            t32[i] = (uint32_t) (uint8_t) ait2p_dev_tab[2 * e] | (uint32_t) (uint8_t) ait2p_dev_tab[2 * e + 1] << 8;
        }
    } else {
        for (int i = threadIdx.x; i < 16384; i += blockDim.x) {
            const int e = 2 * (i >> 5);
            t32[i] = (uint32_t) (uint8_t) ait2p_dev_tab[2 * e] | (uint32_t) (uint8_t) ait2p_dev_tab[2 * e + 1] << 8 |
                     (uint32_t) (uint8_t) ait2p_dev_tab[2 * e + 2] << 16 | (uint32_t) (uint8_t) ait2p_dev_tab[2 * e + 3] << 24;
        }
    }
    __syncthreads();
}
template <int PC, int NC> static constexpr int ptab_bytes() { return PC == 0 ? 0 : PC == 1 ? 4096 * NC : 65536; }
template <int PC, int NC> static __device__ __forceinline__ uint32_t plo(const int lane) {
    return PC == 1 ? 4u * (uint32_t) (lane & (NC - 1)) : 4u * (uint32_t) lane;
}
// one tile's 4 pair entries of a lane (window A from the lane's first bit) -> P[0..3]
template <int PC, int NC> static __device__ __forceinline__ void ptile(uint32_t (&P)[4], const uint32_t A, const char * tsm, const uint32_t lo) {
    P[0] = pent<PC, NC>(tsm, lo, A);       P[1] = pent<PC, NC>(tsm, lo, A >> 4);
    P[2] = pent<PC, NC>(tsm, lo, A >> 8);  P[3] = pent<PC, NC>(tsm, lo, A >> 12);
}
} // namespace aitk

template <int NT>
static __global__ void __launch_bounds__(64) k_ait_actq(const float * __restrict__ X, const uint8_t * __restrict__ W,
        int8_t * __restrict__ XQ, float * __restrict__ SA, float * __restrict__ Y, const int M, const int K, const int k) {
    const int S = K / 512, n = blockIdx.x / S, ks = blockIdx.x - n * S;
    const int h = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int k0 = ks * 512 + h * 256 + l * 8;
    const float * x = X + (int64_t) n * K + k0;
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] = ai4_dev_sign_neg(l * 8 + q) ? -x[q] : x[q];
    }
    ai4_dev_fwht256(v);
    float a = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] *= (1.0f / 16.0f) * ait_dev_col_scale(W, K, k, k0 + q);
        a = fmaxf(a, fabsf(v[q]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    }
    __shared__ float wm[2];
    if (l == 0) {
        wm[h] = a;
    }
    __syncthreads();
    a = fmaxf(wm[0], wm[1]);
    const float id = a > 0.0f ? 127.0f / a : 0.0f;
    if (threadIdx.x == 0) {
        SA[n * S + ks] = a / 127.0f * (1.0f / 64.0f);
    }
    // lane's 8 columns = one 8-col tile tb = l % 4 of 32-K block k0 / 32: column c = 2q + ce -> byte 8q + 4 (tb / 2) + 2 (tb % 2) + ce
    const int tb = l & 3;
    int8_t * dst = XQ + ((int64_t) (k0 / 32) * NT + n) * 32 + 4 * (tb >> 1) + 2 * (tb & 1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t lo = (uint8_t) (int8_t) __float2int_rn(v[2 * q] * id), hi = (uint8_t) (int8_t) __float2int_rn(v[2 * q + 1] * id);
        *reinterpret_cast<uint16_t *>(dst + 8 * q) = (uint16_t) (lo | (hi << 8));
    }
    for (int i = blockIdx.x * 64 + threadIdx.x; Y && i < NT * M; i += gridDim.x * 64) {
        Y[i] = 0.0f;
    }
}

template <int KB, int U>
static __device__ __forceinline__ void ait_group_mma(int * c, const uint32_t (&a)[4], const uint32_t (&f)[4], const uint2 b) {
    using namespace aitk;
    uint32_t p[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = 2 * h;
        p[i] = pack4(st1<KB, 4 * U>(a[i], f[i]), st1<KB, 4 * U + 1>(a[i], f[i]),
                     st1<KB, 4 * U>(a[i + 1], f[i + 1]), st1<KB, 4 * U + 1>(a[i + 1], f[i + 1]));
        p[i + 1] = pack4(st1<KB, 4 * U + 2>(a[i], f[i]), st1<KB, 4 * U + 3>(a[i], f[i]),
                         st1<KB, 4 * U + 2>(a[i + 1], f[i + 1]), st1<KB, 4 * U + 3>(a[i + 1], f[i + 1]));
    }
    mma_s8(c, p[0], p[1], p[2], p[3], b.x, b.y);
}

// One group decodes four tiles and refills its ring slot before IMMA, or after IMMA with TAIL.
template <int KB, int NT, int Gi, int PC = 0, bool CO = false, int DEPTH = 4, bool LOW = false, bool SHARED_X = false, bool NEIGHBOR = false, bool TAIL = false, bool EARLY_X = false>
static __device__ __forceinline__ void ait_group(uint4 (&buf)[DEPTH][2], const int G, const bool pf, const char * gp, const int (&off)[2],
        const int r, const char * xs, const bool live, int (&c)[2][4], const char * tsm = nullptr, const uint32_t lo = 0) {
    static_assert(!TAIL || (KB == 4 && PC == 0 && !CO && !LOW && !SHARED_X), "tail refill requires plain AIT4");
    static_assert(!EARLY_X || (KB == 4 && PC == 0 && !CO && !LOW && !SHARED_X && !TAIL), "early activation requires plain AIT4");
    using namespace aitk;
    constexpr int GB = 128 * KB;
    constexpr bool ONE = 7 * KB + 16 <= 32;
    const uint4 cur0 = buf[Gi][0];
    uint4 cur1;
    if constexpr (NEIGHBOR) {
        static_assert(KB == 4 && PC == 0, "neighbor vectors require one 16-byte vector per lane");
        const int src = (threadIdx.x + 1) & 31;
        cur1 = make_uint4(__shfl_sync(0xffffffffu, cur0.x, src), __shfl_sync(0xffffffffu, cur0.y, src),
                         __shfl_sync(0xffffffffu, cur0.z, src), __shfl_sync(0xffffffffu, cur0.w, src));
    } else {
        cur1 = buf[Gi][1];
    }
    if (pf && !TAIL) {
        // groups past this chunk continue in the next one, after its 64 side bytes
        const char * p = gp + (G + DEPTH) * GB + (G + DEPTH >= 16 ? 64 : 0);
        buf[Gi][0] = ldw16(p + off[0]);
        if constexpr (!NEIGHBOR) { buf[Gi][1] = ldw16(p + off[1]); }
    }
    if constexpr (PC != 0) {
        static_assert(KB == 2, "AIT2P is k = 2");
        uint32_t P[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ptile<PC, 16>(P[i], __funnelshift_r(comp(cur0, i), comp(cur1, i), r), tsm, lo);
        }
        uint2 b = make_uint2(0, 0);
        if (NT == 8 || live) {
            b = CO ? __ldcg((const uint2 *) (xs + G * NT * 32)) : __ldg((const uint2 *) (xs + G * NT * 32));
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const uint32_t a0 = __byte_perm(P[0][2 * u], P[1][2 * u], 0x5410), a1 = __byte_perm(P[0][2 * u + 1], P[1][2 * u + 1], 0x5410);
            const uint32_t a2 = __byte_perm(P[2][2 * u], P[3][2 * u], 0x5410), a3 = __byte_perm(P[2][2 * u + 1], P[3][2 * u + 1], 0x5410);
            mma_s8(c[u], a0, a1, a2, a3, b.x, b.y);
        }
        return;
    }
    if constexpr (LOW) {
        static_assert((KB == 3 || KB == 4) && PC == 0, "low-register path requires AIT3 or AIT4");
        uint32_t a[4], f[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t hi = comp(cur1, i);
            a[i] = __funnelshift_r(comp(cur0, i), hi, r);
            f[i] = __funnelshift_r(a[i], hi >> r, 4 * KB);
        }
        uint2 b = make_uint2(0, 0);
        if (NT == 8 || live) {
            if constexpr (SHARED_X) {
                b = *reinterpret_cast<const uint2 *>(xs + G * NT * 32);
            } else {
                b = __ldg((const uint2 *) (xs + G * NT * 32));
            }
        }
        ait_group_mma<KB, 0>(c[0], a, f, b);
        ait_group_mma<KB, 1>(c[1], a, f, b);
        return;
    }
    uint2 b = make_uint2(0, 0);
    if constexpr (EARLY_X) {
        if (NT == 8 || live) {
            b = __ldg((const uint2 *) (xs + G * NT * 32));
        }
    }
    uint32_t Lv[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t R0 = comp(cur0, i), R1 = comp(cur1, i);
        const uint32_t A = __funnelshift_r(R0, R1, r);
        uint32_t F1 = A;
        if constexpr (!ONE) {
            F1 = __funnelshift_r(A, R1 >> r, 4 * KB);
        }
        Lv[i][0] = st1<KB, 0>(A, F1); Lv[i][1] = st1<KB, 1>(A, F1); Lv[i][2] = st1<KB, 2>(A, F1); Lv[i][3] = st1<KB, 3>(A, F1);
        Lv[i][4] = st1<KB, 4>(A, F1); Lv[i][5] = st1<KB, 5>(A, F1); Lv[i][6] = st1<KB, 6>(A, F1); Lv[i][7] = st1<KB, 7>(A, F1);
    }
    if constexpr (!EARLY_X) {
        if (NT == 8 || live) {
            b = CO ? __ldcg((const uint2 *) (xs + G * NT * 32)) : __ldg((const uint2 *) (xs + G * NT * 32));
        }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        uint32_t a[4];
#pragma unroll
        for (int hs = 0; hs < 2; ++hs) {
#pragma unroll
            for (int h8 = 0; h8 < 2; ++h8) {
                const int j = (2 * u + h8) * 2;
                a[2 * hs + h8] = pack4(Lv[2 * hs][j], Lv[2 * hs][j + 1], Lv[2 * hs + 1][j], Lv[2 * hs + 1][j + 1]);
            }
        }
        mma_s8(c[u], a[0], a[1], a[2], a[3], b.x, b.y);
    }
    if constexpr (TAIL) {
        if (pf) {
            const char * p = gp + (G + DEPTH) * GB + (G + DEPTH >= 16 ? 64 : 0);
            buf[Gi][0] = ldw16(p + off[0]);
            if constexpr (!NEIGHBOR) { buf[Gi][1] = ldw16(p + off[1]); }
        }
    }
}

// grid barrier of the fused kernel (V & 2): arrival counter + generation word, reused across calls without a reset kernel
// (the last arriver zeroes the counter before it bumps the generation; the next call starts after this one ends)
static __device__ unsigned int ait_bar_cnt = 0;
static __device__ unsigned int ait_bar_gen = 0;
static __device__ __forceinline__ unsigned int ait_ld_acq(const unsigned int * p) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void ait_st_rel(unsigned int * p, unsigned int v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// FULL: the 16 groups of a chunk fully unrolled (largest code); else a rolled loop of 4 x (4 unrolled groups) (about 1/4 the
// code: the Gate-2 trace showed the unrolled 1MAD loop losing 8-11 us when another kernel ran before it)
// V (experiment, GGML_CUDA_AIT_V): bit 0 = load the item's row scales and SA at item start (they were a cold DRAM read at the
// item's end, not prefetched); bit 1 = fused activation prep: the kernel itself does k_ait_actq's work (one warp per (token,
// 512-K slice)) and zeroes Y, behind a grid barrier (cooperative launch, 1 CTA per SM); the weight ring and l2pf lines
// (prefetch.global.L2, 128 B per lane each) are issued before the barrier; activations are then read with ld.global.cg.
template <int KB, int NT, bool FULL, int PC = 0, int V = 0, int DEPTH = 4, bool LOW = false, bool CAP = false, bool PACK = LOW, bool NEIGHBOR = false, bool TAIL = false, bool UNROLL2 = false, bool EARLY_X = false, bool BOUNDS = false>
static __global__ void __launch_bounds__(CAP ? 256 : LOW || BOUNDS ? 384 : 512, CAP ? 4 : LOW ? 2 : 1) k_ait_mmv(const char * __restrict__ W, const int8_t * X,
        const float * SA, float * __restrict__ Y, const int M, const int K, const float * __restrict__ XF = nullptr,
        const int l2pf = 0, float * __restrict__ partial = nullptr) {
    static_assert(!NEIGHBOR || (KB == 4 && FULL && PC == 0 && (V == 0 || V == 1)), "neighbor path requires plain full AIT4");
    static_assert(!TAIL || (KB == 4 && FULL && PC == 0 && V == 0 && !PACK), "tail refill requires plain full AIT4");
    static_assert(!UNROLL2 || (KB == 4 && FULL && PC == 0 && V == 0 && DEPTH == 4 && !LOW && !CAP && !PACK && !TAIL), "two-step unroll requires plain full AIT4");
    static_assert(!EARLY_X || (KB == 4 && FULL && PC == 0 && V == 0 && DEPTH == 4 && !LOW && !CAP && !PACK && !TAIL && !UNROLL2), "early activation requires plain full AIT4");
    static_assert(!BOUNDS || (KB == 4 && FULL && PC == 0 && V == 0 && DEPTH == 4 && !LOW && !CAP && !PACK && !TAIL && !UNROLL2 && !EARLY_X), "thread bound requires plain full AIT4");
    constexpr bool CO = (V & 2) != 0;
    using namespace aitk;
    constexpr int BQ = 8 * KB, GB = 16 * BQ, CS = 2048 * KB + 64, MU = 2;
    const int S = K / 512, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    extern __shared__ uint32_t ait_mv_tab[];
    if constexpr (PC != 0) {
        pfill<PC, 16>(ait_mv_tab);
    }
    const char * tsm = (const char *) ait_mv_tab;
    const uint32_t lo = plo<PC, 16>(lane);
    const int items = (M / 32) * S, W_ = gridDim.x * nw, w = blockIdx.x * nw + (threadIdx.x >> 5);
    const int i0 = (int) ((int64_t) w * items / W_), i1 = (int) ((int64_t) (w + 1) * items / W_);
    if (!CO && i0 >= i1) {
        return;
    }
    const int wq = (lane * BQ) >> 5, r = (lane * BQ) & 31;
    const int off[2] = { 16 * (wq % BQ), 16 * ((wq + 1) % BQ) };
    const char * gp = W + (int64_t) i0 * CS;
    uint4 buf[DEPTH][2];
    if (i0 < i1) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            buf[d][0] = ldw16(gp + d * GB + off[0]);
            if constexpr (!NEIGHBOR) { buf[d][1] = ldw16(gp + d * GB + off[1]); }
        }
    }
    if constexpr (CO) {
        __shared__ unsigned int gen_s;
        if (threadIdx.x == 0) {
            gen_s = ait_ld_acq(&ait_bar_gen);   // before this CTA arrives, so it is the generation this barrier releases
        }
        // L2 prefetch of the weight bytes after the ring (the DRAM would idle during the barrier otherwise)
        const int64_t lim = (int64_t) (i1 - i0) * CS;
        for (int j = 0; j < l2pf; ++j) {
            const int64_t o = DEPTH * GB + 128 * (lane + 32 * j);
            if (o < lim) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + o));
            }
        }
        // k_ait_actq's work: unit = (token n, slice ks), one warp, 512 columns = 2 x 256 (FWHT256 each), lane cols 8 per half
        int8_t * XQ = (int8_t *) X;
        float * SAW = (float *) SA;
        const int units = NT * S, wl = threadIdx.x >> 5;
        for (int un = blockIdx.x + wl * gridDim.x; un < units; un += gridDim.x * nw) {
            const int n = un / S, kq = un - n * S;
            float v[2][8];
            float a = 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k0 = kq * 512 + h * 256 + lane * 8;
                const float4 x0 = __ldg((const float4 *) (XF + (int64_t) n * K + k0));
                const float4 x1 = __ldg((const float4 *) (XF + (int64_t) n * K + k0 + 4));
                const float xx[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    v[h][q] = ai4_dev_sign_neg(lane * 8 + q) ? -xx[q] : xx[q];
                }
                ai4_dev_fwht256(v[h]);
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    v[h][q] *= (1.0f / 16.0f) * ait_dev_col_scale((const uint8_t *) W, K, KB, k0 + q);
                    a = fmaxf(a, fabsf(v[h][q]));
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
            }
            const float id = a > 0.0f ? 127.0f / a : 0.0f;
            if (lane == 0) {
                SAW[n * S + kq] = a / 127.0f * (1.0f / 64.0f);
            }
            const int tb = lane & 3;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k0 = kq * 512 + h * 256 + lane * 8;
                int8_t * dst = XQ + ((int64_t) (k0 / 32) * NT + n) * 32 + 4 * (tb >> 1) + 2 * (tb & 1);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t lo8 = (uint8_t) (int8_t) __float2int_rn(v[h][2 * q] * id);
                    const uint32_t hi8 = (uint8_t) (int8_t) __float2int_rn(v[h][2 * q + 1] * id);
                    *reinterpret_cast<uint16_t *>(dst + 8 * q) = (uint16_t) (lo8 | (hi8 << 8));
                }
            }
        }
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; !partial && i < NT * M; i += gridDim.x * blockDim.x) {
            Y[i] = 0.0f;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            const unsigned int gen = gen_s;
            if (atomicAdd(&ait_bar_cnt, 1u) == gridDim.x - 1) {
                atomicExch(&ait_bar_cnt, 0u);
                __threadfence();
                ait_st_rel(&ait_bar_gen, gen + 1);
            } else {
                while (ait_ld_acq(&ait_bar_gen) == gen) {
                }
            }
            __threadfence();
        }
        __syncthreads();
        if (i0 >= i1) {
            return;
        }
    }
    const int g = lane >> 2, q = lane & 3, n0 = q * 2;
    const bool live = g < NT;
    const char * xb = (const char *) X + g * 32 + q * 8;
    int t = i0 / S, ks = i0 - t * S;
    const char * xs = xb + ks * (16 * NT * 32);
    float cf[MU][4];
#pragma unroll
    for (int u = 0; u < MU; ++u) {
        cf[u][0] = cf[u][1] = cf[u][2] = cf[u][3] = 0.0f;
    }
#pragma unroll 1
    for (int it = i0; it < i1; ++it) {
        const bool more = it + 1 < i1;
        int c[MU][4];
#pragma unroll
        for (int u = 0; u < MU; ++u) {
            c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0;
        }
        float s0 = 0.0f, s1 = 0.0f;
        uint32_t rp[MU];
        if constexpr ((V & 1) != 0) {
            // row scales of rows 32t + 16u + g (+8) for K block ks / 2: chunk (t, 2 (ks / 2) + u), slots g and g + 8
            if (n0 < NT)     { s0 = CO ? __ldcg(SA + n0 * S + ks) : __ldg(SA + n0 * S + ks); }
            if (n0 + 1 < NT) { s1 = CO ? __ldcg(SA + (n0 + 1) * S + ks) : __ldg(SA + (n0 + 1) * S + ks); }
#pragma unroll
            for (int u = 0; u < MU; ++u) {
                const unsigned short * sb = (const unsigned short *) (W + ((int64_t) t * S + (ks & ~1) + u) * CS + 2048 * KB);
                rp[u] = (uint32_t) __ldg(sb + g) | ((uint32_t) __ldg(sb + g + 8) << 16);
            }
        }
        if constexpr (UNROLL2) {
#pragma unroll 2
            for (int G4 = 0; G4 < 16; G4 += 4) {
                ait_group<KB, NT, 0, PC, CO, DEPTH, PACK, false, NEIGHBOR>(buf, G4 + 0, G4 + 4 < 16 || more, gp, off, r, xs, live, c);
                ait_group<KB, NT, 1, PC, CO, DEPTH, PACK, false, NEIGHBOR>(buf, G4 + 1, G4 + 5 < 16 || more, gp, off, r, xs, live, c);
                ait_group<KB, NT, 2, PC, CO, DEPTH, PACK, false, NEIGHBOR>(buf, G4 + 2, G4 + 6 < 16 || more, gp, off, r, xs, live, c);
                ait_group<KB, NT, 3, PC, CO, DEPTH, PACK, false, NEIGHBOR>(buf, G4 + 3, G4 + 7 < 16 || more, gp, off, r, xs, live, c);
            }
        } else if constexpr (FULL) {
#pragma unroll
            for (int G4 = 0; G4 < 16; G4 += 4) {
                ait_group<KB, NT, 0 % DEPTH, PC, CO, DEPTH, PACK, false, NEIGHBOR, TAIL, EARLY_X>(buf, G4 + 0, G4 + DEPTH < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 1 % DEPTH, PC, CO, DEPTH, PACK, false, NEIGHBOR, TAIL, EARLY_X>(buf, G4 + 1, G4 + DEPTH + 1 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 2 % DEPTH, PC, CO, DEPTH, PACK, false, NEIGHBOR, TAIL, EARLY_X>(buf, G4 + 2, G4 + DEPTH + 2 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 3 % DEPTH, PC, CO, DEPTH, PACK, false, NEIGHBOR, TAIL, EARLY_X>(buf, G4 + 3, G4 + DEPTH + 3 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
            }
        } else {
#pragma unroll 1
            for (int G4 = 0; G4 < 16; G4 += 4) {
                ait_group<KB, NT, 0, PC>(buf, G4 + 0, G4 + 4 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 1, PC>(buf, G4 + 1, G4 + 5 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 2, PC>(buf, G4 + 2, G4 + 6 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
                ait_group<KB, NT, 3, PC>(buf, G4 + 3, G4 + 7 < 16 || more, gp, off, r, xs, live, c, tsm, lo);
            }
        }
        gp += CS;
        if constexpr ((V & 1) == 0) {
            // row scales of rows 32t + 16u + g (+8) for K block ks / 2: chunk (t, 2 (ks / 2) + u), slots g and g + 8
            s0 = n0 < NT ? __ldg(SA + n0 * S + ks) : 0.0f;
            s1 = n0 + 1 < NT ? __ldg(SA + (n0 + 1) * S + ks) : 0.0f;
#pragma unroll
            for (int u = 0; u < MU; ++u) {
                const unsigned short * sb = (const unsigned short *) (W + ((int64_t) t * S + (ks & ~1) + u) * CS + 2048 * KB);
                rp[u] = (uint32_t) __ldg(sb + g) | ((uint32_t) __ldg(sb + g + 8) << 16);
            }
        }
#pragma unroll
        for (int u = 0; u < MU; ++u) {
            const float r0 = __half2float(__ushort_as_half((unsigned short) (rp[u] & 0xFFFFu)));
            const float r1 = __half2float(__ushort_as_half((unsigned short) (rp[u] >> 16)));
            cf[u][0] += s0 * r0 * (float) c[u][0]; cf[u][1] += s1 * r0 * (float) c[u][1];
            cf[u][2] += s0 * r1 * (float) c[u][2]; cf[u][3] += s1 * r1 * (float) c[u][3];
        }
        xs += 16 * NT * 32;
        ++ks;
        if (it + 1 == i1 || ks == S) {
#pragma unroll
            for (int u = 0; u < MU; ++u) {
                const int m0 = t * 32 + 16 * u + g, m1 = m0 + 8;
                if (partial) {
                    const int first = (int) ((int64_t) t * S * W_ / items);
                    float * out = partial + (int64_t) (w - first) * NT * M;
                    if (n0 < NT)     { out[n0 * M + m0] = cf[u][0];       out[n0 * M + m1] = cf[u][2]; }
                    if (n0 + 1 < NT) { out[(n0 + 1) * M + m0] = cf[u][1]; out[(n0 + 1) * M + m1] = cf[u][3]; }
                } else {
                    if (n0 < NT)     { atomicAdd(Y + n0 * M + m0, cf[u][0]);       atomicAdd(Y + n0 * M + m1, cf[u][2]); }
                    if (n0 + 1 < NT) { atomicAdd(Y + (n0 + 1) * M + m0, cf[u][1]); atomicAdd(Y + (n0 + 1) * M + m1, cf[u][3]); }
                }
                cf[u][0] = cf[u][1] = cf[u][2] = cf[u][3] = 0.0f;
            }
            ks = 0;
            ++t;
            xs = xb;
        }
    }
}

// Only nonempty warp ranges write scratch. Reduce them in warp order without reading unused slots.
template <int NT>
static __global__ void k_ait_mmv_sum(const float * partial, float * Y, int M, int S, int warps) {
    const int tile = blockIdx.x, row = tile * 32 + (threadIdx.x & 31), token = threadIdx.x >> 5;
    const int items = (M / 32) * S, begin = tile * S, end = begin + S;
    const int first = (int) ((int64_t) begin * warps / items);
    const int last = (int) (((int64_t) end * warps + items - 1) / items);
    float sum = 0.0f;
    for (int w = first; w < last; ++w) {
        const int lo = (int) ((int64_t) w * items / warps);
        const int hi = (int) ((int64_t) (w + 1) * items / warps);
        if (lo < hi && lo < end && hi > begin) {
            sum += partial[(int64_t) (w - first) * NT * M + token * M + row];
        }
    }
    Y[token * M + row] = sum;
}

// Default-off N=1 experiment: a CTA owns all K slices of 32 output rows.
// Activation preparation is repeated per row tile but stays in shared memory;
// the block combines its warp partials, so no output-zero launch or atomics are needed.
template <int NW>
static __global__ void __launch_bounds__(32 * NW, NW == 16 ? 2 : 4) k_ait4_row_owned(
        const char * __restrict__ W, const float * __restrict__ X, float * __restrict__ Y, const int K) {
    using namespace aitk;
    constexpr int CS = 8256, DEPTH = 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, S = K / 512;
    extern __shared__ uint32_t row_storage[];
    int8_t * xq = reinterpret_cast<int8_t *>(row_storage);
    float * sa = reinterpret_cast<float *>(xq + K);
    float * partial = sa + S;
    for (int ks = warp; ks < S; ks += NW) {
        float v[2][8], amax = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k0 = ks * 512 + h * 256 + lane * 8;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                v[h][q] = ai4_dev_sign_neg(lane * 8 + q) ? -X[k0 + q] : X[k0 + q];
            }
            ai4_dev_fwht256(v[h]);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                v[h][q] *= (1.0f / 16.0f) * ait_dev_col_scale(reinterpret_cast<const uint8_t *>(W), K, 4, k0 + q);
                amax = fmaxf(amax, fabsf(v[h][q]));
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        }
        if (lane == 0) { sa[ks] = amax / 127.0f * (1.0f / 64.0f); }
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        const int tb = lane & 3;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k0 = ks * 512 + h * 256 + lane * 8;
            int8_t * dst = xq + (k0 / 32) * 32 + 4 * (tb >> 1) + 2 * (tb & 1);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t lo = (uint8_t) (int8_t) __float2int_rn(v[h][2 * q] * inverse);
                const uint32_t hi = (uint8_t) (int8_t) __float2int_rn(v[h][2 * q + 1] * inverse);
                *reinterpret_cast<uint16_t *>(dst + 8 * q) = (uint16_t) (lo | (hi << 8));
            }
        }
    }
    __syncthreads();
    const int first = warp * S / NW, last = (warp + 1) * S / NW;
    const int g = lane >> 2, q = lane & 3;
    const int off[2] = {16 * lane, 16 * ((lane + 1) & 31)};
    const int row_tile = blockIdx.x;
    const char * gp = W + ((int64_t) row_tile * S + first) * CS;
    uint4 buf[DEPTH][2];
    if (first < last) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            buf[d][0] = ldw16(gp + d * 512 + off[0]);
            buf[d][1] = ldw16(gp + d * 512 + off[1]);
        }
    }
    const char * xs = reinterpret_cast<const char *>(xq) + first * 512 + g * 32 + q * 8;
    float sum[2][2] = {};
#pragma unroll 1
    for (int ks = first; ks < last; ++ks) {
        int c[2][4] = {};
        const bool more = ks + 1 < last;
#pragma unroll
        for (int G4 = 0; G4 < 16; G4 += 4) {
            ait_group<4, 1, 0, 0, false, DEPTH, true, true>(buf, G4,     G4 + DEPTH < 16 || more,     gp, off, 0, xs, g == 0, c);
            ait_group<4, 1, 1, 0, false, DEPTH, true, true>(buf, G4 + 1, G4 + DEPTH + 1 < 16 || more, gp, off, 0, xs, g == 0, c);
            ait_group<4, 1, 0, 0, false, DEPTH, true, true>(buf, G4 + 2, G4 + DEPTH + 2 < 16 || more, gp, off, 0, xs, g == 0, c);
            ait_group<4, 1, 1, 0, false, DEPTH, true, true>(buf, G4 + 3, G4 + DEPTH + 3 < 16 || more, gp, off, 0, xs, g == 0, c);
        }
        if (q == 0) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const unsigned short * scales = reinterpret_cast<const unsigned short *>(
                    W + ((int64_t) row_tile * S + (ks & ~1) + u) * CS + 8192);
                const float r0 = __half2float(__ushort_as_half(__ldg(scales + g)));
                const float r1 = __half2float(__ushort_as_half(__ldg(scales + g + 8)));
                sum[u][0] += sa[ks] * r0 * (float) c[u][0];
                sum[u][1] += sa[ks] * r1 * (float) c[u][2];
            }
        }
        gp += CS;
        xs += 512;
    }
    if (q == 0) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            partial[warp * 32 + u * 16 + g] = sum[u][0];
            partial[warp * 32 + u * 16 + g + 8] = sum[u][1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) { value += partial[w * 32 + threadIdx.x]; }
        Y[row_tile * 32 + threadIdx.x] = value;
    }
}

template <int NT, bool FULL, int PC>
static void ait_mmv_p(const dim3 grid, const dim3 block, cudaStream_t stream, const char * w, const int8_t * xq, const float * sa,
        float * Y, int M, int K, float * partial) {
    constexpr int SMEM = aitk::ptab_bytes<PC, 16>();
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_ait_mmv<2, NT, FULL, PC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        attr = true;
    }
    k_ait_mmv<2, NT, FULL, PC><<<grid, block, SMEM, stream>>>(w, xq, sa, Y, M, K, nullptr, 0, partial);
}

// GGML_CUDA_AIT_V experiment launcher (FULL kernel; NT 1/4/5/8 only, others keep V = 0): V & 2 = cooperative fused launch
template <int KB, int NT, int PC, int V>
static void ait_mmv_v(const dim3 grid, const dim3 block, cudaStream_t stream, const char * w, const int8_t * xq, const float * sa,
        float * Y, int M, int K, const float * X, int l2pf, float * partial) {
    constexpr int SMEM = aitk::ptab_bytes<PC, 16>();
    static bool attr = false;
    if (!attr && SMEM > 0) {
        CUDA_CHECK(cudaFuncSetAttribute(k_ait_mmv<KB, NT, true, PC, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    }
    attr = true;
    if constexpr ((V & 2) != 0) {
        void * args[] = { (void *) &w, (void *) &xq, (void *) &sa, (void *) &Y, (void *) &M, (void *) &K, (void *) &X, (void *) &l2pf, (void *) &partial };
        CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) k_ait_mmv<KB, NT, true, PC, V>, grid, block, args, SMEM, stream));
    } else {
        k_ait_mmv<KB, NT, true, PC, V><<<grid, block, SMEM, stream>>>(w, xq, sa, Y, M, K, X, l2pf, partial);
    }
}

template <int NT, int V>
static void ait_mmv_vk(const dim3 grid, const dim3 block, cudaStream_t stream, const char * w, const int8_t * xq, const float * sa,
        float * Y, int M, int K, const float * X, int l2pf, int k, int code, float * partial) {
    if (code == 1) { ait_mmv_v<2, NT, 1, V>(grid, block, stream, w, xq, sa, Y, M, K, X, l2pf, partial); return; }
    switch (k) {
        case 2:  ait_mmv_v<2, NT, 0, V>(grid, block, stream, w, xq, sa, Y, M, K, X, l2pf, partial); break;
        case 3:  ait_mmv_v<3, NT, 0, V>(grid, block, stream, w, xq, sa, Y, M, K, X, l2pf, partial); break;
        default: ait_mmv_v<4, NT, 0, V>(grid, block, stream, w, xq, sa, Y, M, K, X, l2pf, partial); break;
    }
}

static int ait_plds() {
    static const int v = ait_env_int("GGML_CUDA_AIT_PLDS", 1) == 2 ? 2 : 1;
    return v;
}

template <int NT>
static void ait_fast_nt(ggml_backend_cuda_context & ctx, const uint8_t * W, const float * X, float * Y, int M, int K, int k, int warps,
        int code) {
    const int id = ggml_cuda_get_device(), S = K / 512;
    cudaStream_t stream = ctx.stream();
    if constexpr (NT == 1) {
        static const int row_owned = ait_env_int("GGML_CUDA_AIT_ROWOWN", 0);
        const bool wide = row_owned == 16 || (row_owned != 8 && K >= 8192);
        const int nw = wide ? 16 : 8;
        const size_t shared = K + sizeof(float) * (S + nw * 32);
        if (row_owned != 0 && k == 4 && code == 0 && shared <= 48 * 1024) {
            if (wide) {
                k_ait4_row_owned<16><<<M / 32, 512, shared, stream>>>((const char *) W, X, Y, K);
            } else {
                k_ait4_row_owned<8><<<M / 32, 256, shared, stream>>>((const char *) W, X, Y, K);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(id), (size_t) NT * K);
    ggml_cuda_pool_alloc<float>  sa(ctx.pool(id), (size_t) NT * S);
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 grid(nsm), block(32 * warps);
    static const bool full = ait_env_int("GGML_CUDA_AIT_FULL", 1) != 0;
    const char * w = (const char *) W;
    static const int vexp = ait_env_int("GGML_CUDA_AIT_V", 0) & 3;
    static const int l2pf = std::max(0, std::min(32, ait_env_int("GGML_CUDA_AIT_L2PF", 0)));
    static const int occ = ait_env_int("GGML_CUDA_AIT_OCC", 0);
    static const bool neighbor4_occ = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_OCC", 0) == 1;
    static const int grid_factor = std::max(1, std::min(8, ait_env_int("GGML_CUDA_AIT_OCC_GRID", 4)));
    static const bool ordered = ait_env_int("GGML_CUDA_AIT_ORDERED", 1) != 0;
    const bool use_v = (NT == 1 || NT == 4 || NT == 5 || NT == 8) && vexp != 0 && (code != 1 || ait_plds() == 1);
    int total_warps = nsm * warps;
    if (!use_v && k == 4 && code == 0 && full && occ > 0) {
        total_warps = occ >= 3 ? grid_factor * nsm * 8 : 2 * nsm * std::min(warps, 12);
    }
    const bool reserve_neighbor_occ = !use_v && k == 4 && code == 0 && full && warps <= 12 && occ == 0 &&
        neighbor4_occ && ait_env_int("GGML_CUDA_AIT4_NEIGHBOR", 0) == 1;
    const int scratch_warps = reserve_neighbor_occ ? 2 * total_warps : total_warps;
    const size_t slots = (scratch_warps + M / 32 - 1) / (M / 32) + 2;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(id));
    if (ordered) {
        scratch.alloc(slots * NT * (size_t) M);
    }
    float * partial = ordered ? scratch.get() : nullptr;
    auto finish = [&]() {
        if (ordered) {
            k_ait_mmv_sum<NT><<<M / 32, 32 * NT, 0, stream>>>(partial, Y, M, S, total_warps);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (NT == 1 || NT == 4 || NT == 5 || NT == 8) {
        if (vexp != 0 && (code != 1 || ait_plds() == 1)) {
            if (vexp & 2) {
                ait_mmv_vk<NT, 3>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, X, l2pf, k, code, partial);
            } else {
                k_ait_actq<NT><<<NT * S, 64, 0, stream>>>(X, W, xq.get(), sa.get(), ordered ? nullptr : Y, M, K, k);
                ait_mmv_vk<NT, 1>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, X, l2pf, k, code, partial);
            }
            finish();
            return;
        }
    }
    k_ait_actq<NT><<<NT * S, 64, 0, stream>>>(X, W, xq.get(), sa.get(), ordered ? nullptr : Y, M, K, k);
    static const bool bounds4 = ait_env_int("GGML_CUDA_AIT4_BOUNDS", 0) == 1;
    if (bounds4 && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, false, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool early4_x = ait_env_int("GGML_CUDA_AIT4_EARLY_X", 0) == 1;
    if (early4_x && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool unroll4_two = ait_env_int("GGML_CUDA_AIT4_UNROLL2", 0) == 1;
    if (unroll4_two && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool pack4 = ait_env_int("GGML_CUDA_AIT4_PACK", 0) == 1;
    if (pack4 && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 4, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool tail4 = ait_env_int("GGML_CUDA_AIT4_TAIL", 0) == 1;
    if (tail4 && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool neighbor4 = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR", 0) == 1;
    if (neighbor4 && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        static const bool neighbor4_pack = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_PACK", 0) == 1;
        static const bool neighbor4_ring2 = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_RING2", 0) == 1;
        static const bool neighbor4_ring1 = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_RING1", 0) == 1;
        static const bool neighbor4_tail = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_TAIL", 0) == 1;
        static const bool neighbor4_early_x = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_EARLY_X", 0) == 1;
        static const bool neighbor4_bounds = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_BOUNDS", 0) == 1;
        static const bool neighbor4_unroll2 = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_UNROLL2", 0) == 1;
        static const bool neighbor4_scales = ait_env_int("GGML_CUDA_AIT4_NEIGHBOR_SCALES", 0) == 1;
        if (neighbor4_pack) {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, true, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_ring2) {
            k_ait_mmv<4, NT, true, 0, 0, 2, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_ring1) {
            k_ait_mmv<4, NT, true, 0, 0, 1, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_tail) {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, true, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_early_x) {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, true, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_bounds) {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, true, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_unroll2) {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, true, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_scales) {
            k_ait_mmv<4, NT, true, 0, 1, 4, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else if (neighbor4_occ) {
            const dim3 grid_occ(2 * nsm);
            total_warps = 2 * nsm * warps;
            k_ait_mmv<4, NT, true, 0, 0, 4, true, false, false, true><<<grid_occ, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else {
            k_ait_mmv<4, NT, true, 0, 0, 4, false, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        }
        finish();
        return;
    }
    static const bool ring4_one = ait_env_int("GGML_CUDA_AIT4_RING1", 0) == 1;
    if (ring4_one && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 1><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool ring4 = ait_env_int("GGML_CUDA_AIT4_RING2", 0) == 1;
    if (ring4 && k == 4 && code == 0 && full && warps <= 12 && occ == 0) {
        k_ait_mmv<4, NT, true, 0, 0, 2><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool ring1 = ait_env_int("GGML_CUDA_AIT3_RING1", 0) == 1;
    if (ring1 && k == 3 && code == 0 && full && warps <= 12) {
        k_ait_mmv<3, NT, true, 0, 0, 1><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const bool ring2 = ait_env_int("GGML_CUDA_AIT3_RING2", 0) == 1;
    if (ring2 && k == 3 && code == 0 && full && warps <= 12) {
        k_ait_mmv<3, NT, true, 0, 0, 2><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        finish();
        return;
    }
    static const int pack3 = ait_env_int("GGML_CUDA_AIT3_PACK", 0);
    if ((pack3 == 1 || pack3 == 2) && k == 3 && code == 0 && full && warps <= 12) {
        if (pack3 == 2) {
            k_ait_mmv<3, NT, true, 0, 0, 4, false, false, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else {
            k_ait_mmv<3, NT, true, 0, 0, 4, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        }
        finish();
        return;
    }
    if (k == 4 && code == 0 && full && occ > 0) {
        // Modes 3/4 cap registers for four 256-thread blocks per SM. Keep
        // both prefetch depths; profiles must check for local-memory spills.
        if (occ >= 3) {
            const dim3 grid_cap(grid_factor * nsm), block_cap(256);
            if (occ == 4) {
                k_ait_mmv<4, NT, true, 0, 0, 1, true, true><<<grid_cap, block_cap, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
            } else {
                k_ait_mmv<4, NT, true, 0, 0, 2, true, true><<<grid_cap, block_cap, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
            }
            finish();
            return;
        }
        const dim3 grid_occ(2 * nsm), block_occ(32 * std::min(warps, 12));
        if (occ == 2) {
            k_ait_mmv<4, NT, true, 0, 0, 1, true><<<grid_occ, block_occ, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        } else {
            k_ait_mmv<4, NT, true, 0, 0, 2, true><<<grid_occ, block_occ, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial);
        }
        finish();
        return;
    }
    if (code == 1) {
        const int pl = ait_plds();
        if (full) {
            if (pl == 2) { ait_mmv_p<NT, true, 2>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, partial); }
            else         { ait_mmv_p<NT, true, 1>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, partial); }
        } else {
            if (pl == 2) { ait_mmv_p<NT, false, 2>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, partial); }
            else         { ait_mmv_p<NT, false, 1>(grid, block, stream, w, xq.get(), sa.get(), Y, M, K, partial); }
        }
    } else if (full) {
        switch (k) {
            case 2:  k_ait_mmv<2, NT, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
            case 3:  k_ait_mmv<3, NT, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
            default: k_ait_mmv<4, NT, true><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
        }
    } else {
        switch (k) {
            case 2:  k_ait_mmv<2, NT, false><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
            case 3:  k_ait_mmv<3, NT, false><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
            default: k_ait_mmv<4, NT, false><<<grid, block, 0, stream>>>(w, xq.get(), sa.get(), Y, M, K, nullptr, 0, partial); break;
        }
    }
    finish();
}

static bool ait_fast(ggml_backend_cuda_context & ctx, const uint8_t * W, const float * X, float * Y, int64_t T, int M, int K, int k, int code) {
    static const int warps = std::max(1, std::min(16, ait_env_int("GGML_CUDA_AIT_WARPS", 12)));
    switch (T) {
        case 1: ait_fast_nt<1>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 2: ait_fast_nt<2>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 3: ait_fast_nt<3>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 4: ait_fast_nt<4>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 5: ait_fast_nt<5>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 6: ait_fast_nt<6>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 7: ait_fast_nt<7>(ctx, W, X, Y, M, K, k, warps, code); return true;
        case 8: ait_fast_nt<8>(ctx, W, X, Y, M, K, k, warps, code); return true;
        default: return false;
    }
}

// ---------------------------------------------------------------- batch path (N > 8 tokens): tile-chain IMMA, token tiles
// CTA = 8 warps = 8 row tiles (256 rows) x TN = 8 NB tokens. Per 512-K slice the CTA stages the TN-token int8 activation
// slice (fragment order, [16 groups][TN][32 B]) in shared memory (double buffered); each warp decodes its chunk once and runs
// the IMMA for its NB token blocks. Exact int32 per slice, then the float scaling of k_ait_mmv: the same arithmetic as the
// N <= 8 kernel up to float summation order, and no atomics (each warp owns its 32 rows x TN outputs).
// blockIdx.x = token tile (fastest), so the CTAs that run together share their row tiles' weights in L2.

// activations for any T: k_ait_actq without the Y zeroing, T at run time ([K/32][T][32] fragment order)
static __global__ void __launch_bounds__(64) k_ait_actq_t(const float * __restrict__ X, const uint8_t * __restrict__ W,
        int8_t * __restrict__ XQ, float * __restrict__ SA, const int K, const int k, const int T) {
    const int S = K / 512, n = blockIdx.x / S, ks = blockIdx.x - n * S;
    const int h = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int k0 = ks * 512 + h * 256 + l * 8;
    const float * x = X + (int64_t) n * K + k0;
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] = ai4_dev_sign_neg(l * 8 + q) ? -x[q] : x[q];
    }
    ai4_dev_fwht256(v);
    float a = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        v[q] *= (1.0f / 16.0f) * ait_dev_col_scale(W, K, k, k0 + q);
        a = fmaxf(a, fabsf(v[q]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    }
    __shared__ float wm[2];
    if (l == 0) {
        wm[h] = a;
    }
    __syncthreads();
    a = fmaxf(wm[0], wm[1]);
    const float id = a > 0.0f ? 127.0f / a : 0.0f;
    if (threadIdx.x == 0) {
        SA[(int64_t) n * S + ks] = a / 127.0f * (1.0f / 64.0f);
    }
    const int tb = l & 3;
    int8_t * dst = XQ + ((int64_t) (k0 / 32) * T + n) * 32 + 4 * (tb >> 1) + 2 * (tb & 1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t lo = (uint8_t) (int8_t) __float2int_rn(v[2 * q] * id), hi = (uint8_t) (int8_t) __float2int_rn(v[2 * q + 1] * id);
        *reinterpret_cast<uint16_t *>(dst + 8 * q) = (uint16_t) (lo | (hi << 8));
    }
}

template <int KB, int NB, int Gi, int PC = 0, int NC = 16>
static __device__ __forceinline__ void ait_group_mq(uint4 (&buf)[4][2], const int G, const bool pf, const char * gp, const int (&off)[2],
        const int r, const char * bs, int (&c)[2][NB][4], const char * tsm = nullptr, const uint32_t lo = 0) {
    using namespace aitk;
    constexpr int GB = 128 * KB;
    constexpr bool ONE = 7 * KB + 16 <= 32;
    const uint4 cur0 = buf[Gi][0], cur1 = buf[Gi][1];
    if (pf) {
        const char * p = gp + (G + 4) * GB + (G + 4 >= 16 ? 64 : 0);
        buf[Gi][0] = ldw16(p + off[0]);
        buf[Gi][1] = ldw16(p + off[1]);
    }
    if constexpr (PC != 0) {
        static_assert(KB == 2, "AIT2P is k = 2");
        uint32_t P[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ptile<PC, NC>(P[i], __funnelshift_r(comp(cur0, i), comp(cur1, i), r), tsm, lo);
        }
        uint32_t a[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            a[u][0] = __byte_perm(P[0][2 * u], P[1][2 * u], 0x5410); a[u][1] = __byte_perm(P[0][2 * u + 1], P[1][2 * u + 1], 0x5410);
            a[u][2] = __byte_perm(P[2][2 * u], P[3][2 * u], 0x5410); a[u][3] = __byte_perm(P[2][2 * u + 1], P[3][2 * u + 1], 0x5410);
        }
        const char * bg = bs + G * (NB * 8 * 32);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const uint2 b = *reinterpret_cast<const uint2 *>(bg + nb * 8 * 32);
            mma_s8(c[0][nb], a[0][0], a[0][1], a[0][2], a[0][3], b.x, b.y);
            mma_s8(c[1][nb], a[1][0], a[1][1], a[1][2], a[1][3], b.x, b.y);
        }
        return;
    }
    uint32_t Lv[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t R0 = comp(cur0, i), R1 = comp(cur1, i);
        const uint32_t A = __funnelshift_r(R0, R1, r);
        uint32_t F1 = A;
        if constexpr (!ONE) {
            F1 = __funnelshift_r(A, R1 >> r, 4 * KB);
        }
        Lv[i][0] = st1<KB, 0>(A, F1); Lv[i][1] = st1<KB, 1>(A, F1); Lv[i][2] = st1<KB, 2>(A, F1); Lv[i][3] = st1<KB, 3>(A, F1);
        Lv[i][4] = st1<KB, 4>(A, F1); Lv[i][5] = st1<KB, 5>(A, F1); Lv[i][6] = st1<KB, 6>(A, F1); Lv[i][7] = st1<KB, 7>(A, F1);
    }
    uint32_t a[2][4];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
        for (int hs = 0; hs < 2; ++hs) {
#pragma unroll
            for (int h8 = 0; h8 < 2; ++h8) {
                const int j = (2 * u + h8) * 2;
                a[u][2 * hs + h8] = pack4(Lv[2 * hs][j], Lv[2 * hs][j + 1], Lv[2 * hs + 1][j], Lv[2 * hs + 1][j + 1]);
            }
        }
    }
    const char * bg = bs + G * (NB * 8 * 32);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const uint2 b = *reinterpret_cast<const uint2 *>(bg + nb * 8 * 32);
        mma_s8(c[0][nb], a[0][0], a[0][1], a[0][2], a[0][3], b.x, b.y);
        mma_s8(c[1][nb], a[1][0], a[1][1], a[1][2], a[1][3], b.x, b.y);
    }
}

template <int KB, int NB, int PC = 0, int NC = 16>
static __global__ void __launch_bounds__(256, 1) k_ait_mmq(const char * __restrict__ W, const int8_t * __restrict__ X,
        const float * __restrict__ SA, float * __restrict__ Y, const int M, const int K, const int T) {
    using namespace aitk;
    constexpr int TN = 8 * NB, BQ = 8 * KB, GB = 16 * BQ, CS = 2048 * KB + 64;
    constexpr int SLB = 16 * TN * 32;            // bytes of one staged slice
    constexpr int NV = SLB / 16 / 256;           // uint4 per thread per slice
    extern __shared__ uint4 ait_mq_smem[];
    char * sm = (char *) ait_mq_smem;            // [2 staged slices][AIT2P table]
    const int S = K / 512, lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const char * tsm = sm + 2 * SLB;
    const uint32_t lo = plo<PC, NC>(lane);
    if constexpr (PC != 0) {
        pfill<PC, NC>((uint32_t *) (sm + 2 * SLB));
    }
    const int n0 = blockIdx.x * TN, rt = blockIdx.y * 8 + wid;
    const bool active = rt < M / 32;

    // staging with cp.async (no registers held across the compute): element e = tid + 256 i -> group G = e / (2 TN),
    // token e % (2 TN) / 2, 16-byte half e % 2; tokens past T are zero-filled (src size 0)
    const uint32_t sm0 = (uint32_t) __cvta_generic_to_shared(sm);
    auto issue_slice = [&](int ks, int buf_i) {
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = threadIdx.x + 256 * i, G = e / (2 * TN), w = e - G * 2 * TN, n = n0 + (w >> 1);
            const int8_t * src = X + ((int64_t) (16 * ks + G) * T + (n < T ? n : 0)) * 32 + 16 * (w & 1);
            const uint32_t dst = sm0 + buf_i * SLB + 16 * e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n < T ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
    };
    issue_slice(0, 0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    const int wq = (lane * BQ) >> 5, r = (lane * BQ) & 31;
    const int off[2] = { 16 * (wq % BQ), 16 * ((wq + 1) % BQ) };
    const char * gp = W + (int64_t) (active ? rt : 0) * S * CS;
    uint4 buf[4][2];
    if (active) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            buf[d][0] = ldw16(gp + d * GB + off[0]);
            buf[d][1] = ldw16(gp + d * GB + off[1]);
        }
    }
    const int g = lane >> 2, q = lane & 3;
    float cf[2][NB][4];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            cf[u][nb][0] = cf[u][nb][1] = cf[u][nb][2] = cf[u][nb][3] = 0.0f;
        }
    }
#pragma unroll 1
    for (int ks = 0; ks < S; ++ks) {
        const bool more = ks + 1 < S;
        if (more) {
            issue_slice(ks + 1, (ks + 1) & 1);
        }
        if (active) {
            const char * bs = sm + (ks & 1) * SLB + g * 32 + q * 8;
            int c[2][NB][4];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    c[u][nb][0] = c[u][nb][1] = c[u][nb][2] = c[u][nb][3] = 0;
                }
            }
#pragma unroll
            for (int G4 = 0; G4 < 16; G4 += 4) {
                ait_group_mq<KB, NB, 0, PC, NC>(buf, G4 + 0, G4 + 4 < 16 || more, gp, off, r, bs, c, tsm, lo);
                ait_group_mq<KB, NB, 1, PC, NC>(buf, G4 + 1, G4 + 5 < 16 || more, gp, off, r, bs, c, tsm, lo);
                ait_group_mq<KB, NB, 2, PC, NC>(buf, G4 + 2, G4 + 6 < 16 || more, gp, off, r, bs, c, tsm, lo);
                ait_group_mq<KB, NB, 3, PC, NC>(buf, G4 + 3, G4 + 7 < 16 || more, gp, off, r, bs, c, tsm, lo);
            }
            float rs[2][2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const unsigned short * sb = (const unsigned short *) (W + ((int64_t) rt * S + (ks & ~1) + u) * CS + 2048 * KB);
                rs[u][0] = __half2float(__ushort_as_half(__ldg(sb + g)));
                rs[u][1] = __half2float(__ushort_as_half(__ldg(sb + g + 8)));
            }
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const int na = n0 + nb * 8 + 2 * q;
                const float s0 = na < T ? __ldg(SA + (int64_t) na * S + ks) : 0.0f;
                const float s1 = na + 1 < T ? __ldg(SA + (int64_t) (na + 1) * S + ks) : 0.0f;
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    cf[u][nb][0] += s0 * rs[u][0] * (float) c[u][nb][0]; cf[u][nb][1] += s1 * rs[u][0] * (float) c[u][nb][1];
                    cf[u][nb][2] += s0 * rs[u][1] * (float) c[u][nb][2]; cf[u][nb][3] += s1 * rs[u][1] * (float) c[u][nb][3];
                }
            }
            gp += CS;
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    if (!active) {
        return;
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int na = n0 + nb * 8 + 2 * q;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int m0 = rt * 32 + 16 * u + g, m1 = m0 + 8;
            if (na < T)     { Y[(int64_t) na * M + m0] = cf[u][nb][0];       Y[(int64_t) na * M + m1] = cf[u][nb][2]; }
            if (na + 1 < T) { Y[(int64_t) (na + 1) * M + m0] = cf[u][nb][1]; Y[(int64_t) (na + 1) * M + m1] = cf[u][nb][3]; }
        }
    }
}

// AIT2P (PC != 0): the table after the two staged slices; TN 32: 32 + 64 KB, TN 64: 64 KB + the 8-copy u32 table (32 KB)
template <int KB, int NB, int PC = 0, int NC = 16>
static void ait_mmq_launch(const char * W, const int8_t * xq, const float * sa, float * Y, int M, int K, int T, cudaStream_t stream) {
    constexpr int TN = 8 * NB, SMEM = 2 * 16 * TN * 32 + aitk::ptab_bytes<PC, NC>();
    static_assert(SMEM <= 99 * 1024, "smem");
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_ait_mmq<KB, NB, PC, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        attr = true;
    }
    const dim3 grid((unsigned) ((T + TN - 1) / TN), (unsigned) ((M / 32 + 7) / 8)), block(256);
    k_ait_mmq<KB, NB, PC, NC><<<grid, block, SMEM, stream>>>(W, xq, sa, Y, M, K, T);
}

// GGML_CUDA_AIT_MMQ = token tile of the batch kernel: 64 (default; mmq1 2026-10-23: 512 tokens ait4 729 vs 1052 us at 32, ait2 651 vs 880) or 32; 0 = the reference path (reconstruct + SGEMM)
static bool ait_mmq(ggml_backend_cuda_context & ctx, const uint8_t * W, const float * X, float * Y, int64_t T, int M, int K, int k, int code) {
    static const int tn = ait_env_int("GGML_CUDA_AIT_MMQ", 64);
    if (tn != 32 && tn != 64) {
        return false;
    }
    if (T > 65535 * (int64_t) tn || (int64_t) T * (K / 512) > INT32_MAX) {
        return false;
    }
    const int id = ggml_cuda_get_device(), S = K / 512;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(id), (size_t) T * K);
    ggml_cuda_pool_alloc<float>  sa(ctx.pool(id), (size_t) T * S);
    k_ait_actq_t<<<(unsigned) (T * S), 64, 0, stream>>>(X, W, xq.get(), sa.get(), K, k, (int) T);
    const char * w = (const char *) W;
    if (code == 1) {
        if (tn == 32) {
            if (ait_plds() == 2) { ait_mmq_launch<2, 4, 2>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); }
            else                 { ait_mmq_launch<2, 4, 1, 16>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); }
        } else {
            ait_mmq_launch<2, 8, 1, 8>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream);
        }
    } else if (tn == 32) {
        switch (k) {
            case 2:  ait_mmq_launch<2, 4>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
            case 3:  ait_mmq_launch<3, 4>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
            default: ait_mmq_launch<4, 4>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
        }
    } else {
        switch (k) {
            case 2:  ait_mmq_launch<2, 8>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
            case 3:  ait_mmq_launch<3, 8>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
            default: ait_mmq_launch<4, 8>(w, xq.get(), sa.get(), Y, M, K, (int) T, stream); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_ait_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return op->op == GGML_OP_MUL_MAT && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           a->ne[0] % 1024 == 0 && a->ne[1] % 32 == 0 && a->ne[1] >= 1024 && a->ne[2] == 1 && a->ne[3] == 1 &&
           ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op);
}

void ggml_cuda_mul_mat_ait(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int k = ggml_ait_rate(src0->type), code = ggml_ait_code(src0->type);
    GGML_ASSERT(k >= 2 && k <= 4 && (code == 0 || (code == 1 && k == 2)));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    const int64_t K = src0->ne[0], M = src0->ne[1];
    GGML_ASSERT(K % 1024 == 0 && M % 32 == 0 && M >= 1024 && src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == M);
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (T == 0) {
        return;
    }
    GGML_ASSERT(M <= INT32_MAX && T <= INT32_MAX && K <= INT32_MAX);
    static const bool exact = ait_env_int("GGML_CUDA_AIT_EXACT", 0) != 0;
    static const bool ref   = ait_env_int("GGML_CUDA_AIT_REF", 0) != 0;
    if (!exact && !ref && ait_fast(ctx, (const uint8_t *) src0->data, (const float *) src1->data, (float *) dst->data, T, (int) M, (int) K, k, code)) {
        return;
    }
    if (!exact && !ref && ait_mmq(ctx, (const uint8_t *) src0->data, (const float *) src1->data, (float *) dst->data, T, (int) M, (int) K, k, code)) {
        return;
    }
    static const int  slice = ait_env_int("GGML_CUDA_AIT_ASLICE", 512);
    static const int  mib   = ait_env_int("GGML_CUDA_AIT_SLAB_MIB", 256);
    GGML_ASSERT(slice == 512 || slice == 256);

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    const uint8_t * base = (const uint8_t *) src0->data;
    const int64_t row_bytes = (int64_t) ggml_row_size(src0->type, K);
    GGML_ASSERT(row_bytes == (int64_t) src0->nb[1]);

    int64_t rows = ((int64_t) mib << 20) / (K * (int64_t) sizeof(float)) / 32 * 32;
    rows = std::max<int64_t>(32, std::min<int64_t>(rows, std::min<int64_t>(M, (int64_t) 65535 * 32)));

    ggml_cuda_pool_alloc<float> w(ctx.pool(id), (size_t) (rows * K));
    ggml_cuda_pool_alloc<float> xe(ctx.pool(id), (size_t) (T * K));
    ggml_cuda_pool_alloc<float> cs(ctx.pool(id), (size_t) K);

    // col scales from the whole tensor (rows 0..1023 hold every column once)
    k_ait_cols<<<(unsigned) ((K + 255) / 256), 256, 0, stream>>>(base, cs.get(), K, k);
    const float * X = (const float *) src1->data;
    for (int64_t r0 = 0; r0 < T; r0 += 65535) {
        const unsigned nr = (unsigned) std::min<int64_t>(65535, T - r0);
        if (slice == 512) {
            k_ait_act<512><<<dim3((unsigned) (K / 256), nr), 32, 0, stream>>>(X + r0 * K, cs.get(), xe.get() + r0 * K, K, !exact);
        } else {
            k_ait_act<256><<<dim3((unsigned) (K / 256), nr), 32, 0, stream>>>(X + r0 * K, cs.get(), xe.get() + r0 * K, K, !exact);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    float * Y = (float *) dst->data;
    for (int64_t m0 = 0; m0 < M; m0 += rows) {
        const int64_t mm = std::min<int64_t>(rows, M - m0);
        k_ait_reconstruct<<<dim3((unsigned) (K / 512), (unsigned) (mm / 32)), 256, 0, stream>>>(base + m0 * row_bytes, w.get(), K, k, code);
        CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int) mm, (int) T, (int) K,
                                 &alpha, w.get(), (int) K, xe.get(), (int) K, &beta, Y + m0, (int) M));
    }
    CUDA_CHECK(cudaGetLastError());
}
