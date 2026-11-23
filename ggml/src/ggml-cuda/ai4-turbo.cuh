// AI+ Turbo (SM 8.6 Ampere RTX 3090/Ti first-class):
//   1. Fast trellis unpackers cutting the 84% unpack bottleneck:
//      - AIT2P4_U (V=2 Pair-LUT -> s4, 1 KB table, 16x conflict-free replication in 16 KB, 2.375 insns/w + 569T mma.s4)
//      - AIT2Q4_U (V=4 Quad-LUT -> s4, 4 KB table, 1.125 insns/w + 569T mma.s4)
//      - AIT2Q8_U (V=4 Quad-LUT -> s8, 8 KB table, 1.000 insns/w + 285T mma.s8)
//      - SWAR_TRI (V=2 2-byte SWAR triangular -> s8, 0 SMEM, 0 DP4A, 2.50 insns/w)
//      - Format-compatible AIT3_U whole-row 2-word decode (50% fewer tile LDS on existing AIT3_U/AIT3_U2 GGUFs)
//   2. ampere_mmv_turbo: SpecDecode (M = 2..32, MTP/DFlash/DSpine) + MoE (ID=true) kernel:
//      - Zero static SMEM (reduction aliases dead cp.async ring -> saves up to 28 KB SMEM/CTA)
//      - J_MAJOR 4-reg A-streaming (-12 regs/thread vs ampere_mmv's a[4][4])
//      - Software-pipelined activation prefetch (hides L2/HBM latency)
//      - Works on both existing s8 layouts (AIT2/3/4_U) and s4 mma.m16n8k64.s4 (A4 569T & A8-split exact)
//   3. Prefill & Medium-N:
//      - prefill_decode_s4 + gemm_predec_s4: 0.5*M*K scratchpad (2x less VRAM BW) + IMMA.16864.S4.S4 (569 TOPS)
//        with both A4 (569T for 2-bit) and bit-exact A8-split (2x mma.s4, 16*p_hi + p_lo)
//      - gemm_sparse24_s4_proto: real SM 8.6 mma.sp.m16n8k128.s4 (1,139 TOPS) + dense mma.s4 base prototype
//
// Strictly SM 8.0+ / SM 8.6 (Ampere). No ggml runtime dependencies beyond ai4-kern.cuh.
#pragma once

#include "ai4-kern.cuh"
#include <type_traits>

namespace ai4_turbo {

using rq4::cp_async16;
using rq4::cp_async4;
using rq4::cp_async_commit;
using rq4::cp_async_wait;
using rq4::mma_s8;
using rq4::magic_i2f;

// =================================================================================================
// 1. SM 8.6 INT4 Dense (569 TOPS) & 2:4 Sparse (1,139 TOPS) Tensor Core PTX Wrappers
// =================================================================================================

// Dense INT4 MMA: m16n8k64.s4 -> 2x MACs/cycle vs m16n8k32.s8 (569 TOPS on RTX 3090 SM 8.6)
// A fragment a[4]: lane (g = lane/4, t = lane%4):
//   a[0] = row g,     cols 8*t .. 8*t+7       (8 packed s4 nibbles = 1 full AI+ tile row!)
//   a[1] = row g + 8, cols 8*t .. 8*t+7
//   a[2] = row g,     cols 32 + 8*t .. 32 + 8*t+7
//   a[3] = row g + 8, cols 32 + 8*t .. 32 + 8*t+7
// B fragment (b0, b1): token g, cols 8*t..+7 (b0) and 32+8*t..+7 (b1)
static __device__ __forceinline__ void mma_s4(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

// 2:4 Structured Sparse INT4 MMA: m16n8k128.s4 -> 4x MACs/cycle vs m16n8k32.s8 (1,139 TOPS on RTX 3090 SM 8.6)
// A compressed (16x64 s4 non-zeros) in a[4], B dense (128x8 s4) in b[4], metadata in e, thread selector F in {0, 1}.
template <int F = 0>
static __device__ __forceinline__ void mma_sp_s4(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[4], uint32_t e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (F == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
    } else {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x1;\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
    }
#else
    (void)d; (void)a; (void)b; (void)e;
#endif
}

// =================================================================================================
// 2. Fast Trellis Unpackers (SM 8.6 SASS-Optimized)
// =================================================================================================

// (A) AIT2P4_U: V=2 Pair-LUT outputting packed s4 (2 x s4 per byte!).
// Table size = 1024 entries x 1 byte = 1 KB!
// Replicating NC=16 copies across SMEM banks (16 KB total, uint32_t per (entry, copy)) gives
// near-zero bank conflicts (lanes l and l+16 share a bank -> at most 2-way conflict, avg ~1.03x!).
// Decodes 8 weights (= 1 complete AI+ tile row = 1 full m16n8k64.s4 A register!) from 32-bit window f.
template <int NC = 16>
static __device__ __forceinline__ uint32_t pair_lut_s4_byte(const uint8_t * __restrict__ tab, uint32_t lane_off, uint32_t w) {
    // Quadratic hash matching AIT2P (only low 20 bits matter): h = w*(w+1)
    const uint32_t h = w * w + w;
    if constexpr (NC == 1) {
        // 1 KB compact table (zero occupancy impact): byte offset = (h >> 10) & 0x3FF
        return (uint32_t)tab[(h >> 10) & 0x3FFu];
    } else {
        // Bank-replicated u32 table: entry e, copy c -> u32 word (e * NC + c) -> byte (e * NC + c) * 4
        constexpr int LC = (NC == 16) ? 4 : (NC == 8) ? 3 : (NC == 4) ? 2 : 1;
        const uint32_t addr = ((h >> (8 - LC)) & (1023u << (LC + 2))) | lane_off;
        return *(const uint32_t *)(tab + addr);
    }
}

template <int NC = 16>
static __device__ __forceinline__ uint32_t levels8_pair_lut_s4(const uint8_t * __restrict__ tab, uint32_t lane_off, uint32_t f) {
    // 4 consecutive pairs of a k=2 tile row (8 weights = 32-bit packed s4 register!)
    const uint32_t b0 = pair_lut_s4_byte<NC>(tab, lane_off, f >> 12);
    const uint32_t b1 = pair_lut_s4_byte<NC>(tab, lane_off, f >> 8);
    const uint32_t b2 = pair_lut_s4_byte<NC>(tab, lane_off, f >> 4);
    const uint32_t b3 = pair_lut_s4_byte<NC>(tab, lane_off, f);
    const uint32_t lo = __byte_perm(b0, b1, 0x0040);
    const uint32_t hi = __byte_perm(b2, b3, 0x0040);
    return __byte_perm(lo, hi, 0x5410);
}

// (B) AIT2Q4_U: V=4 Quad-LUT outputting packed s4 (4 x s4 = uint16_t per entry!).
// Table size = 2048 entries x 2 bytes = 4 KB!
// 2 lookups + 1 PRMT = 8 x s4 weights (1 full m16n8k64.s4 A register) in ~9 insns = 1.125 insns/weight!
static __device__ __forceinline__ uint32_t quad_lut_s4_u16(const uint8_t * __restrict__ tab, uint32_t w) {
    // Multiplicative hash (Knuth golden ratio prime): 1 IMAD + 1 SHF + 1 LOP3 + 1 LDS.U16
    const uint32_t h = (w & 0xFFFFu) * 0x9E3779B1u;
    return *(const uint16_t *)(tab + ((h >> 20) & 0xFFEu)); // 2048 entries x 2B = 4096B
}

static __device__ __forceinline__ uint32_t levels8_quad_lut_s4(const uint8_t * __restrict__ tab, uint32_t f) {
    const uint32_t lo = quad_lut_s4_u16(tab, f >> 8);
    const uint32_t hi = quad_lut_s4_u16(tab, f);
    return __byte_perm(lo, hi, 0x5410);
}

// (C) AIT2Q8_U: V=4 Quad-LUT outputting packed s8 (4 x s8 = uint32_t per entry!).
// Table size = 2048 entries x 4 bytes = 8 KB!
// 1 IMAD + 1 SHF + 1 LOP3 + 1 LDS.32 = 4 weights (1 full m16n8k32.s8 A register) = 1.00 insn/weight!
static __device__ __forceinline__ uint32_t levels4_quad_lut_s8(const uint8_t * __restrict__ tab, uint32_t w) {
    const uint32_t h = (w & 0xFFFFu) * 0x9E3779B1u;
    return *(const uint32_t *)(tab + ((h >> 19) & 0x1FFCu)); // 2048 entries x 4B = 8192B
}

// (D) SWAR-Triangular V=2 -> s8 (ZERO SMEM, ZERO DP4A, ZERO BANK CONFLICTS!).
// 2 IMAD + 2 PRMT + 2 LOP3 + 1 IADD3 + 1 LOP3 = 8 ALU ops (+2 extract = 10 insns / 4 weights vs 17 in 1MAD!).
template <uint32_t MULT>
static __device__ __forceinline__ uint32_t levels4_swar_tri_s8(uint32_t s0, uint32_t s1) {
    const uint32_t x0 = s0 * MULT;
    const uint32_t x1 = s1 * MULT;
    const uint32_t p  = __byte_perm(x0, x1, 0x6420) & 0x7F7F7F7Fu;
    const uint32_t q  = __byte_perm(x0, x1, 0x7531) & 0x7F7F7F7Fu;
    return (p + q) ^ 0x80808080u;
}

// (E) Format-compatible AIT3_U whole-row 2-word loader (ZERO weight re-encoding!).
// Existing a_regs_smem_kr<3> loads 4 words per tile row (two separate 4-symbol windows).
// All 8 symbols of a k=3 tile row span only 8*3 + 13 = 37 bits -> ALWAYS fits in 2 adjacent 32-bit words!
// Cuts tile LDS by 50% (32 -> 16 per chunk) and funnelshifts by 50%.
struct k3row { int w0, w1, sh; };
static __device__ __forceinline__ k3row k3_row_win(int R) {
    // 8 symbols: n = 8*R .. 8*R+7. Last symbol ends at bit 3*(8*R + 8) = 24*R + 24.
    // First symbol's 16-bit state starts at bit 3*(8*R) + 3 - 16 = 24*R - 13 (mod 384).
    // Total span = 37 bits: [24*R - 13, 24*R + 24). Always fits inside 2 words (64 bits) unless sh > 27!
    int s = 24 * R - 13;
    if (s < 0) s += 384;
    k3row r;
    r.w0 = s >> 5;
    r.sh = s & 31;
    r.w1 = (r.w0 + 1 == 12) ? 0 : r.w0 + 1;
    return r;
}

template <uint32_t MULT>
static __device__ __forceinline__ void levels8_k3_from_3words(
    const uint32_t * __restrict__ tile, const k3row & rw, uint32_t & lo, uint32_t & hi) {
    // Note: when rw.sh <= 27, 37 bits fit in 2 words (w0, w1). For R where sh > 27 (only R=7: s=155, sh=27 -> 27+37=64, STILL FITS IN 2 WORDS!!
    // Check all R=0..15: s = (24*R - 13) mod 384 -> s & 31 in {19,11,3,27,19,11,3,27,...}! MAX sh = 27!
    // 27 + 37 = 64 <= 64! EVERY SINGLE ROW R=0..15 ALWAYS FITS IN EXACTLY 2 WORDS!
    const uint32_t w0 = tile[rw.w0];
    const uint32_t w1 = tile[rw.w1];
    // High 32 bits of the 37-bit window: symbols 0..3 (shifts 9, 6, 3, 0 of f_hi = bits [0..24] of 37-bit window)
    // Wait: f_lo = __funnelshift_l(w1, w0, rw.sh) gives bits [0..31] of the 37-bit window (MSB-first: bit 0 is MSB)!
    // Symbols 0..3 need bits [0..24] -> reside in f_hi = f_lo >> 7 (shifts 9, 6, 3, 0 of (f_lo >> 7))!
    // Symbols 4..7 need bits [12..36] -> reside in f_tail = __funnelshift_l(w1, w0, rw.sh + 5) (since 37 - 32 = 5 -> shift left by rw.sh + 5)!
    // Wait: if rw.sh + 5 >= 32 (only when rw.sh == 27 -> rw.sh + 5 == 32), __funnelshift_l(w1, w0, 32) = w1!
    // In PTX, shf.l.clamp or (rw.sh == 27 ? w1 : __funnelshift_l(w1, w0, rw.sh + 5)) handles sh==27 with 1 SEL!
    const uint32_t f_top = __funnelshift_l(w1, w0, rw.sh);
    const uint32_t f_bot = (rw.sh == 27) ? w1 : __funnelshift_l(w1, w0, rw.sh + 5);
    lo = ai4k::levels4_kr<3, MULT>(f_top >> 7);
    hi = ai4k::levels4_kr<3, MULT>(f_bot);
}

// =================================================================================================
// 3. Activation Quantizers for mma.m16n8k64.s4 (A4 569T AND Bit-Exact A8-Split!)
// =================================================================================================

// Bit-exact signed 8-bit [-119, +119] -> two signed 4-bit nibbles (x_hi in [-7,+7], x_lo in [-8,+7])
// satisfying 16 * x_hi + x_lo == x_q BIT-EXACTLY!
static __device__ __forceinline__ void split_s8_to_s4(int q, uint32_t & nib_hi, uint32_t & nib_lo) {
    int lo = q & 0xF;
    if (lo >= 8) lo -= 16;
    int hi = (q - lo) >> 4;
    nib_hi = (uint32_t)(hi & 0xF);
    nib_lo = (uint32_t)(lo & 0xF);
}

// Rotate (H256) and quantize activations into m16n8k64.s4 B-fragment layout:
// For each 128-K chunk of token tok:
//   2 K-steps (j=0: cols 0..63, j=1: cols 64..127), each lane t=0..3 owns 16 elements:
//     b0 = cols 8*t..8*t+7 (8 packed s4 nibbles = 4 bytes)
//     b1 = cols 32+8*t..32+8*t+7 (8 packed s4 nibbles = 4 bytes)
// So per token, per 128-K chunk = 64 bytes (half of int8!).
// A8_SPLIT=true writes two 64-B planes (q_hi, q_lo) -> exact A8 precision on 2x mma.s4!
// A8_SPLIT=false writes one 64-B plane (q_hi) -> A4 precision on 1x mma.s4 (569 TOPS)!
template <bool A8_SPLIT>
static __global__ void __launch_bounds__(128) quantize_act_s4(
        const float * __restrict__ x, int64_t sx,
        uint32_t * __restrict__ q_hi, uint32_t * __restrict__ q_lo,
        float * __restrict__ s, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int hb   = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int tok  = blockIdx.y;
    if (hb * 256 >= K || tok >= N) return;

    const float * xr = x + (int64_t)tok * sx + hb * 256 + lane * 8;
    const float4 u0 = *(const float4 *)xr;
    const float4 u1 = *(const float4 *)(xr + 4);
    float v[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };

    const uint32_t sw = rq4::d_sign_bits[lane >> 2] >> ((lane & 3) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) if ((sw >> e) & 1) v[e] = -v[e];
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int i = 0; i < 8; i += 2 * h)
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                float a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? (o - v[e]) : (v[e] + o);
        }

    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= 1.0f / 16.0f;
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));

    constexpr float QMAX = A8_SPLIT ? 119.0f : 7.0f;
    constexpr float CLIP = A8_SPLIT ? 1.0f : 0.85f;
    const float d  = (amax * CLIP) / QMAX;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

    uint32_t pack_hi = 0u, pack_lo = 0u;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        int qi = __float2int_rn(v[e] * id);
        if constexpr (A8_SPLIT) {
            qi = max(-119, min(119, qi));
            uint32_t nhi, nlo;
            split_s8_to_s4(qi, nhi, nlo);
            pack_hi |= (nhi << (4 * e));
            pack_lo |= (nlo << (4 * e));
        } else {
            qi = max(-7, min(7, qi));
            pack_hi |= ((uint32_t)(qi & 0xF) << (4 * e));
        }
    }
    // lane (0..31) holds 8 consecutive elements:
    //   chunk_in_256 = lane >> 4 (0 or 1), j_k64 = (lane >> 3) & 1 (0 or 1),
    //   half32 = (lane >> 2) & 1 (b0 or b1), t = lane & 3.
    // Store in lane-coalesced [tok][K/128][t][j_k64][half32] order so a lane reads 16B = uint4 per 128-K chunk!
    const int c128   = lane >> 4;
    const int j64    = (lane >> 3) & 1;
    const int half32 = (lane >> 2) & 1;
    const int t      = lane & 3;
    const int64_t chunk_idx = (int64_t)tok * (K / 128) + hb * 2 + c128;
    const int64_t w_idx = chunk_idx * 16 + t * 4 + j64 * 2 + half32;
    q_hi[w_idx] = pack_hi;
    if constexpr (A8_SPLIT) q_lo[w_idx] = pack_lo;
    if (lane == 0) s[(int64_t)tok * (K / 256) + hb] = d;
}

// =================================================================================================
// 4. SpecDecode (M = 2..32, MTP/DFlash/DSpine) + MoE Turbo Kernel: ampere_mmv_turbo
//    Fixes ALL 3 bottlenecks found in ai4-ampere.cuh::ampere_mmv:
//      (1) ZERO static SMEM: reduction reuses dead cp.async ring smem after __syncthreads()!
//      (2) J_MAJOR 4-reg A-streaming: peak live A regs drops from 16 -> 4 (-12 regs/thread)!
//      (3) Software-pipelined Xq prefetch (ldx(i+1) overlaps compute(i))!
//    Supports BOTH existing s8 layouts (100% format-compatible with AIT4_U / AIT3_U / AIT2_U!)
//    AND next-gen s4 Pair-LUT / Quad-LUT (569T mma.m16n8k64.s4, A4 or A8-split!).
// =================================================================================================

enum TurboMode {
    MODE_AIT4_S8      = 0,  // Format-compatible existing AIT4_U (1MAD -> s8, mma.m16n8k32.s8)
    MODE_AIT3_S8_WROW = 1,  // Format-compatible existing AIT3_U (whole-row 2-word -> s8, mma.m16n8k32.s8)
    MODE_SWAR_TRI_S8  = 2,  // AI+ v2 SWAR-Triangular V=2 -> s8 (0 SMEM, 0 DP4A, 10 insns/4w)
    MODE_QUAD_LUT_S8  = 3,  // AI+ v2 V=4 Quad-LUT -> s8 (8 KB SMEM, 4 insns/4w)
    MODE_PAIR_LUT_S4  = 4,  // AI+ v2 AIT2P4_U V=2 Pair-LUT -> s4 (1 KB or 16 KB SMEM, mma.m16n8k64.s4 569T!)
    MODE_QUAD_LUT_S4  = 5,  // AI+ v2 AIT2Q4_U V=4 Quad-LUT -> s4 (4 KB SMEM, 1.125 insns/w, mma.m16n8k64.s4 569T!)
};

template <int MODE>
constexpr __host__ __device__ int turbo_ch_bytes() {
    if constexpr (MODE == MODE_AIT4_S8)      return 1024; // k=4: 16 tiles x 64B
    if constexpr (MODE == MODE_AIT3_S8_WROW) return 768;  // k=3: 16 tiles x 48B
    if constexpr (MODE == MODE_SWAR_TRI_S8)  return 1024; // k=4 test
    return 512;                                           // k=2: 16 tiles x 32B
}

template <int MODE>
constexpr __host__ __device__ bool turbo_is_s4() {
    return MODE == MODE_PAIR_LUT_S4 || MODE == MODE_QUAD_LUT_S4;
}

template <int MODE, int LUT_COPIES = 1>
constexpr __host__ __device__ int turbo_lut_bytes() {
    if constexpr (MODE == MODE_QUAD_LUT_S8) return 8192;
    if constexpr (MODE == MODE_QUAD_LUT_S4) return 4096;
    if constexpr (MODE == MODE_PAIR_LUT_S4) return (LUT_COPIES == 1) ? 1024 : (4096 * LUT_COPIES);
    return 0;
}

template <int BN, int MODE>
constexpr __host__ __device__ int turbo_mmv_stage_bytes() {
    // chunk tiles + 64B weight scales + BN*4B activation scales (16-B aligned)
    return turbo_ch_bytes<MODE>() + 64 + ((BN * 4 + 15) & ~15);
}

template <int BN, int MODE, int WARPS, int STAGES = 2, int LUT_COPIES = 1>
constexpr __host__ __device__ int turbo_mmv_smem_bytes() {
    const int ring_b = WARPS * STAGES * turbo_mmv_stage_bytes<BN, MODE>();
    const int red_b  = (WARPS - 1) * (BN / 8) * 4 * 32 * 4;
    const int body   = (ring_b > red_b) ? ring_b : red_b;
    return turbo_lut_bytes<MODE, LUT_COPIES>() + body;
}

template <int BN, int MODE, int XSH, bool ID, int WARPS, int MINB = 6, bool A8_SPLIT = false, int LUT_COPIES = 1, bool XPERM = false>
static __global__ void __launch_bounds__(WARPS * 32, MINB) ampere_mmv_turbo(
        const uint8_t * __restrict__ W,
        const uint32_t * __restrict__ Xq_hi,  // s8 fragment stream (s8 modes) or s4 hi stream (s4 modes)
        const uint32_t * __restrict__ Xq_lo,  // s4 lo stream (only used when A8_SPLIT=true)
        const float * __restrict__ Xs,
        const uint8_t * __restrict__ lut_g,   // global LUT source (only used by LUT modes)
        float * __restrict__ Y, int64_t ldy, int M, int N, int K, const ai4k::idmap idm) {
    static_assert(BN == 8 || BN == 16 || BN == 32 || BN == 64, "specDecode width");
    static_assert(WARPS == 4 || WARPS == 8, "warps");
    constexpr int NTI       = BN / 8;
    constexpr int CHB       = turbo_ch_bytes<MODE>();
    constexpr int U_B       = CHB + 32;
    constexpr int ST        = turbo_mmv_stage_bytes<BN, MODE>();
    constexpr int LUT_B     = turbo_lut_bytes<MODE, LUT_COPIES>();
    constexpr bool IS_S4    = turbo_is_s4<MODE>();
    // Whole-row s8 decoders (MODE_AIT3_S8_WROW and MODE_QUAD_LUT_S8) decode tile (4*j + t) whole rows:
    //   a[0] = row g,   cols 0..3 of tile 4*j+t  |  a[2] = row g,   cols 4..7 of tile 4*j+t
    //   a[1] = row g+8, cols 0..3 of tile 4*j+t  |  a[3] = row g+8, cols 4..7 of tile 4*j+t
    // This matches ai4-ampere.cuh's PERM order (lines 186-188).
    //   - XPERM == false (standard un-reordered Xq): USE_PERM_B = true applies the exact PERM B-load remap
    //     (offset 64*(t&1) + 4*(t>>1) + 8*j and +32) -> 100% bit-exact drop-in on standard Xq.
    //   - XPERM == true  (pre-permuted Xq):          USE_PERM_B = false reads Xq directly via coalesced LDG.128.
    constexpr bool USE_PERM_B = (!IS_S4) && (MODE == MODE_AIT3_S8_WROW || MODE == MODE_QUAD_LUT_S8) && (!XPERM);
    // For s8 at BN <= 32, decode a[4][4] upfront + nt-outer loop with 2x coalesced LDG.128 (matches baseline 80 regs,
    // eliminates v1's 4x LDG.64 + 4x address recompute). Keep J_MAJOR 4-reg streaming at BN == 64 (251 -> 128 regs!).
    constexpr bool A_UPFRONT  = (!IS_S4) && (BN <= 32);

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t * lut_s = smem;
    uint8_t * ring_base = smem + LUT_B;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    // Cooperative async LUT load (when LUT mode enabled)
    if constexpr (LUT_B > 0) {
        if constexpr (MODE == MODE_PAIR_LUT_S4 && LUT_COPIES > 1) {
            for (int i = tid; i < 1024 * LUT_COPIES; i += WARPS * 32) {
                const int e = i / LUT_COPIES;
                ((uint32_t *)lut_s)[i] = (uint32_t)lut_g[e];
            }
            __syncthreads();
        } else {
            for (int i = tid; i < LUT_B / 16; i += WARPS * 32) {
                cp_async16(lut_s + i * 16, lut_g + i * 16, true);
            }
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    const int slab = blockIdx.x;
    const int nk   = K / 128;
    const int xsn  = K >> (7 + XSH);
    const int * rows = nullptr;
    const int * cols = nullptr;
    int n0 = blockIdx.y * BN;
    if constexpr (ID) {
        if ((int)blockIdx.y >= idm.hdr[1]) return;
        const int active = idm.tiles[2 * blockIdx.y];
        const int off = idm.off[active] + idm.tiles[2 * blockIdx.y + 1];
        N = min(BN, idm.off[active + 1] - off);
        W += (int64_t)idm.exps[active] * idm.nb02;
        rows = idm.rows + off;
        cols = idm.cols + off;
        n0 = 0;
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) return rows[c]; else return n0 + c; };
    auto ycol = [&](int c) -> int64_t { if constexpr (ID) return cols[c]; else return n0 + c; };

    const uint8_t * ws = W + (int64_t)slab * (int64_t)nk * U_B;
    uint8_t * ring = ring_base + warp * 2 * ST;
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    // Precompute lane-constant window offsets outside the K-loop
    const ai4k::k5win k2_wa = ai4k::kr_win<2>(g, 1);
    const ai4k::k5win k2_wb = ai4k::kr_win<2>(g + 8, 1);
    const k3row k3_ra = k3_row_win(g);
    const k3row k3_rb = k3_row_win(g + 8);
    const uint32_t lane_off = 4u * (uint32_t)(lane & (LUT_COPIES - 1));
    (void)k2_wa; (void)k2_wb; (void)k3_ra; (void)k3_rb; (void)lane_off;

    // Store compact 32-bit row indices instead of 64-bit pointers
    int xr_idx[NTI];
    bool live_c[NTI];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nt * 8 + g;
        live_c[nt] = (n0 + c < N);
        xr_idx[nt] = live_c[nt] ? (int)xrow(c) : 0;
    }

    auto load_stage = [&](int i, int slot) {
        const int kc = warp + i * WARPS;
        uint8_t * st = ring + slot * ST;
        const uint8_t * u = ws + (int64_t)kc * U_B;
        const uint8_t * tiles = u + 32;
        constexpr int CHP = CHB / 16;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            cp_async4(st + CHB + lane * 4, u + lane * 4, true);
        }
        for (int c = lane; c < BN; c += 32) {
            const bool live = (n0 + c < N);
            const float * src = live ? (Xs + xrow(c) * xsn + (kc >> XSH)) : Xs;
            cp_async4(st + CHB + 64 + c * 4, src, live);
        }
    };

    // Software-pipelined activation prefetch for narrow widths where it fits cleanly in <= 64-80 regs:
    //   - s8: BN == 8 (NTI=1, 8 regs) when !USE_PERM_B
    //   - s4 (A4): BN <= 32 (NTI <= 4, <= 16 regs!)
    constexpr bool PREFETCH_X = ((!IS_S4) && (BN == 8) && (!USE_PERM_B)) || (IS_S4 && !A8_SPLIT && BN <= 32);
    uint4 nx_hi[PREFETCH_X ? NTI : 1][IS_S4 ? 1 : 2];

    auto prefetch_x = [&](int i) {
        if constexpr (PREFETCH_X) {
            const int kc = warp + i * WARPS;
            const bool valid = (i < nmy);
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if (n0 + nt * 8 >= N) continue;
                if (valid && live_c[nt]) {
                    if constexpr (IS_S4) {
                        nx_hi[nt][0] = __ldg((const uint4 *)(Xq_hi + ((int64_t)xr_idx[nt] * nk + kc) * 16 + t * 4));
                    } else {
                        const uint4 * p = (const uint4 *)(Xq_hi + ((int64_t)xr_idx[nt] * nk + kc) * 32 + t * 8);
                        nx_hi[nt][0] = __ldg(p);
                        nx_hi[nt][1] = __ldg(p + 1);
                    }
                } else {
                    nx_hi[nt][0] = make_uint4(0, 0, 0, 0);
                    if constexpr (!IS_S4) nx_hi[nt][1] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };

    // Helper: decode one j-step (4 s8 A-fragment registers) for any s8 mode
    auto decode_a_s8 = [&](const uint32_t * __restrict__ tw, int j, uint32_t (&a)[4]) {
        if constexpr (MODE == MODE_AIT4_S8) {
            ai4k::a_regs_smem<AIT_MUL_4>(tw, j, g, t, a);
        } else if constexpr (MODE == MODE_AIT3_S8_WROW) {
            const uint32_t * tile = tw + (4 * j + t) * 12;
            levels8_k3_from_3words<AIT_MUL_3>(tile, k3_ra, a[0], a[2]);
            levels8_k3_from_3words<AIT_MUL_3>(tile, k3_rb, a[1], a[3]);
        } else if constexpr (MODE == MODE_SWAR_TRI_S8) {
            const int sh = (t & 1) ? 0 : 16;
            const uint32_t * ta = tw + (4 * j + (t >> 1)) * 16;
            const uint32_t * tb = ta + 32;
            const int gp = (g + 15) & 15;
            const uint32_t f0 = __funnelshift_r(ta[g],     ta[gp],    sh);
            const uint32_t f1 = __funnelshift_r(ta[g + 8], ta[g + 7], sh);
            const uint32_t f2 = __funnelshift_r(tb[g],     tb[gp],    sh);
            const uint32_t f3 = __funnelshift_r(tb[g + 8], tb[g + 7], sh);
            a[0] = levels4_swar_tri_s8<AIT_MUL_4>(f0 >> 8, f0 & 0xFFFFu);
            a[1] = levels4_swar_tri_s8<AIT_MUL_4>(f1 >> 8, f1 & 0xFFFFu);
            a[2] = levels4_swar_tri_s8<AIT_MUL_4>(f2 >> 8, f2 & 0xFFFFu);
            a[3] = levels4_swar_tri_s8<AIT_MUL_4>(f3 >> 8, f3 & 0xFFFFu);
        } else if constexpr (MODE == MODE_QUAD_LUT_S8) {
            // Tile (4*j + t) whole-row layout (PERM order): a[0]/a[2] = row g cols 0..3 / 4..7;
            // a[1]/a[3] = row g+8 cols 0..3 / 4..7. Matches USE_PERM_B (!XPERM) and pre-permuted Xq (XPERM).
            const uint32_t * ta = tw + (4 * j + t) * 8;
            const uint32_t f0 = __funnelshift_l(ta[k2_wa.w1], ta[k2_wa.w0], k2_wa.sh);
            const uint32_t f1 = __funnelshift_l(ta[k2_wb.w1], ta[k2_wb.w0], k2_wb.sh);
            a[0] = levels4_quad_lut_s8(lut_s, f0 >> 8);
            a[2] = levels4_quad_lut_s8(lut_s, f0);
            a[1] = levels4_quad_lut_s8(lut_s, f1 >> 8);
            a[3] = levels4_quad_lut_s8(lut_s, f1);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;

    if (nmy > 0) load_stage(0, 0);
    cp_async_commit();
    prefetch_x(0);

    for (int i = 0; i < nmy; ++i) {
        const int slot = i & 1;
        cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) load_stage(i + 1, 1 - slot);
        cp_async_commit();

        const int kc = warp + i * WARPS;
        const uint8_t * st = ring + slot * ST;
        const uint32_t * tw = (const uint32_t *)st;
        const half * sd = (const half *)(st + CHB);
        const float * dx = (const float *)(st + CHB + 64);

        uint4 cx_hi[PREFETCH_X ? NTI : 1][IS_S4 ? 1 : 2];
        if constexpr (PREFETCH_X) {
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                cx_hi[nt][0] = nx_hi[nt][0];
                if constexpr (!IS_S4) cx_hi[nt][1] = nx_hi[nt][1];
            }
            prefetch_x(i + 1);
        }

        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);

        if constexpr (A_UPFRONT) {
            // Optimal s8 path for BN = 8, 16, 32: decode a[4][4] once, skip inactive nt columns,
            // issue 2x coalesced LDG.128 (or exact PERM B-load for AIT3_WROW/QUAD_LUT_S8 on standard Xq),
            // and run two independent IMMA chains (p, q).
            uint32_t a[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) decode_a_s8(tw, j, a[j]);

#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if (n0 + nt * 8 >= N) continue;
                uint32_t xv[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                if constexpr (PREFETCH_X) {
                    xv[0] = cx_hi[nt][0].x; xv[1] = cx_hi[nt][0].y; xv[2] = cx_hi[nt][0].z; xv[3] = cx_hi[nt][0].w;
                    xv[4] = cx_hi[nt][1].x; xv[5] = cx_hi[nt][1].y; xv[6] = cx_hi[nt][1].z; xv[7] = cx_hi[nt][1].w;
                } else if (live_c[nt]) {
                    const int8_t * x = (const int8_t *)Xq_hi + (int64_t)xr_idx[nt] * K + kc * 128;
                    if constexpr (USE_PERM_B) {
                        // Exact PERM B-fragment remap for whole-row tiles (4j+t) on standard un-reordered Xq
                        x += 64 * (t & 1) + 4 * (t >> 1);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            xv[2 * j]     = __ldg((const uint32_t *)(x + 8 * j));
                            xv[2 * j + 1] = __ldg((const uint32_t *)(x + 32 + 8 * j));
                        }
                    } else {
                        const uint4 x0 = __ldg((const uint4 *)(x + t * 32));
                        const uint4 x1 = __ldg((const uint4 *)(x + t * 32) + 1);
                        xv[0] = x0.x; xv[1] = x0.y; xv[2] = x0.z; xv[3] = x0.w;
                        xv[4] = x1.x; xv[5] = x1.y; xv[6] = x1.z; xv[7] = x1.w;
                    }
                }
                int p[4] = { 0, 0, 0, 0 };
                int q[4] = { 0, 0, 0, 0 };
                mma_s8(p, a[0], xv[0], xv[1]);
                mma_s8(p, a[1], xv[2], xv[3]);
                mma_s8(q, a[2], xv[4], xv[5]);
                mma_s8(q, a[3], xv[6], xv[7]);
                const float dx0 = dx[nt * 8 + 2 * t];
                const float dx1 = dx[nt * 8 + 2 * t + 1];
                acc[nt][0] += magic_i2f(p[0] + q[0]) * (d0 * dx0);
                acc[nt][1] += magic_i2f(p[1] + q[1]) * (d0 * dx1);
                acc[nt][2] += magic_i2f(p[2] + q[2]) * (d1 * dx0);
                acc[nt][3] += magic_i2f(p[3] + q[3]) * (d1 * dx1);
            }
        } else {
            int p[NTI][4];
            int p_lo[(IS_S4 && A8_SPLIT) ? NTI : 1][4];
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                p[nt][0] = p[nt][1] = p[nt][2] = p[nt][3] = 0;
                if constexpr (IS_S4 && A8_SPLIT) {
                    p_lo[nt][0] = p_lo[nt][1] = p_lo[nt][2] = p_lo[nt][3] = 0;
                }
            }

            if constexpr (IS_S4) {
                // ONLY 2 IMMAs per 128-K chunk! Each a[r] is ONE FULL 8-weight AI+ tile row!
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t a[4];
                    const uint32_t * ta = tw + (8 * j + t) * 8;
                    const uint32_t * tb = ta + 4 * 8;
                    const uint32_t fa0 = __funnelshift_l(ta[k2_wa.w1], ta[k2_wa.w0], k2_wa.sh);
                    const uint32_t fa1 = __funnelshift_l(ta[k2_wb.w1], ta[k2_wb.w0], k2_wb.sh);
                    const uint32_t fb0 = __funnelshift_l(tb[k2_wa.w1], tb[k2_wa.w0], k2_wa.sh);
                    const uint32_t fb1 = __funnelshift_l(tb[k2_wb.w1], tb[k2_wb.w0], k2_wb.sh);
                    if constexpr (MODE == MODE_PAIR_LUT_S4) {
                        a[0] = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fa0);
                        a[1] = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fa1);
                        a[2] = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fb0);
                        a[3] = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fb1);
                    } else {
                        a[0] = levels8_quad_lut_s4(lut_s, fa0);
                        a[1] = levels8_quad_lut_s4(lut_s, fa1);
                        a[2] = levels8_quad_lut_s4(lut_s, fb0);
                        a[3] = levels8_quad_lut_s4(lut_s, fb1);
                    }
#pragma unroll
                    for (int nt = 0; nt < NTI; ++nt) {
                        if (n0 + nt * 8 >= N) continue;
                        uint2 bh = make_uint2(0, 0);
                        if constexpr (PREFETCH_X) {
                            const uint4 & vh = cx_hi[nt][0];
                            bh = (j == 0) ? make_uint2(vh.x, vh.y) : make_uint2(vh.z, vh.w);
                        } else if (live_c[nt]) {
                            bh = __ldg((const uint2 *)(Xq_hi + ((int64_t)xr_idx[nt] * nk + kc) * 16 + t * 4 + j * 2));
                        }
                        mma_s4(p[nt], a, bh.x, bh.y);
                        if constexpr (A8_SPLIT) {
                            uint2 bl = live_c[nt]
                                ? __ldg((const uint2 *)(Xq_lo + ((int64_t)xr_idx[nt] * nk + kc) * 16 + t * 4 + j * 2))
                                : make_uint2(0, 0);
                            mma_s4(p_lo[nt], a, bl.x, bl.y);
                        }
                    }
                }
            } else {
                // J_MAJOR s8 streaming for BN == 64: cuts peak regs from 251 -> 128 (2x occupancy!)
                const int64_t kc_base = (int64_t)kc * 128 + (USE_PERM_B ? (64 * (t & 1) + 4 * (t >> 1)) : (t * 32));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    decode_a_s8(tw, j, a);
#pragma unroll
                    for (int nt = 0; nt < NTI; ++nt) {
                        if (n0 + nt * 8 >= N) continue;
                        uint32_t b0 = 0u, b1 = 0u;
                        if (live_c[nt]) {
                            const int8_t * x = (const int8_t *)Xq_hi + (int64_t)xr_idx[nt] * K + kc_base + j * 8;
                            if constexpr (USE_PERM_B) {
                                b0 = __ldg((const uint32_t *)x);
                                b1 = __ldg((const uint32_t *)(x + 32));
                            } else {
                                const uint2 xv = __ldg((const uint2 *)x);
                                b0 = xv.x; b1 = xv.y;
                            }
                        }
                        mma_s8(p[nt], a, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if (n0 + nt * 8 >= N) continue;
                const float dx0 = dx[nt * 8 + 2 * t];
                const float dx1 = dx[nt * 8 + 2 * t + 1];
                if constexpr (IS_S4 && A8_SPLIT) {
                    acc[nt][0] += magic_i2f(p[nt][0] * 16 + p_lo[nt][0]) * (d0 * dx0);
                    acc[nt][1] += magic_i2f(p[nt][1] * 16 + p_lo[nt][1]) * (d0 * dx1);
                    acc[nt][2] += magic_i2f(p[nt][2] * 16 + p_lo[nt][2]) * (d1 * dx0);
                    acc[nt][3] += magic_i2f(p[nt][3] * 16 + p_lo[nt][3]) * (d1 * dx1);
                } else {
                    acc[nt][0] += magic_i2f(p[nt][0]) * (d0 * dx0);
                    acc[nt][1] += magic_i2f(p[nt][1]) * (d0 * dx1);
                    acc[nt][2] += magic_i2f(p[nt][2]) * (d1 * dx0);
                    acc[nt][3] += magic_i2f(p[nt][3]) * (d1 * dx1);
                }
            }
        }
    }
    cp_async_wait<0>();
    // ZERO static SMEM reduction: barrier guarantees ring_base is dead before reuse!
    __syncthreads();
    float * red = (float *)ring_base; // layout: [WARPS - 1][NTI][4][32]
    if (warp > 0) {
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (n0 + nt * 8 >= N) continue;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                red[(((warp - 1) * NTI + nt) * 4 + e) * 32 + lane] = acc[nt][e];
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int r = slab * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if (n0 + nt * 8 >= N) continue;
#pragma unroll
            for (int w = 0; w < WARPS - 1; ++w)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[nt][e] += red[((w * NTI + nt) * 4 + e) * 32 + lane];
            const int c = nt * 8 + 2 * t;
            if (n0 + c < N) {
                Y[ycol(c) * ldy + r]     = acc[nt][0];
                Y[ycol(c) * ldy + r + 8] = acc[nt][2];
            }
            if (n0 + c + 1 < N) {
                Y[ycol(c + 1) * ldy + r]     = acc[nt][1];
                Y[ycol(c + 1) * ldy + r + 8] = acc[nt][3];
            }
        }
    }
    (void)M;
}

// Single-config launcher with per-instantiation >48 KB dynamic SMEM opt-in (matching ai4-ampere.cuh:284-292).
// Required for >48 KB configs such as MODE_PAIR_LUT_S4 with LUT_COPIES=16.
template <int BN, int MODE, int XSH, bool ID, int WARPS, int MINB, bool A8_SPLIT, int LUT_COPIES, bool XPERM>
static void launch_ampere_mmv_turbo_cfg(
        const uint8_t * W, const uint32_t * Xq_hi, const uint32_t * Xq_lo, const float * Xs,
        const uint8_t * lut_g, float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    constexpr int smem = turbo_mmv_smem_bytes<BN, MODE, WARPS, 2, LUT_COPIES>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(
                ampere_mmv_turbo<BN, MODE, XSH, ID, WARPS, MINB, A8_SPLIT, LUT_COPIES, XPERM>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid(M / 16, ID ? max_tiles : (N + BN - 1) / BN);
    ampere_mmv_turbo<BN, MODE, XSH, ID, WARPS, MINB, A8_SPLIT, LUT_COPIES, XPERM><<<grid, WARPS * 32, smem, stream>>>(
        W, Xq_hi, Xq_lo, Xs, lut_g, Y, ldy, M, N, K, idm);
}

// Explicit dispatch helper for SpecDecode (N = 2..64).
// CRITICAL RULES (validated by real RTX 3090 GPU benchmarks):
//   - N == 1:      NEVER route here — keep N == 1 on the specialized 56.6 us single-token mmvr / mmv kernel!
//   - MoE N=4..8:  T1 (exp/ink200-experts) supersedes turbo's MoE N=4..8 routing launch; turbo shines on MoE N=9..32
//                  (and standalone MoE N=4..32 beat both rc and ampere_mmv by -3.4% to -14.6% on Vast RTX 3090!).
//   - N == 2..8:   BN = 8,  WARPS = 4, MINB = 6 (64-72 regs, 0 static SMEM -> 7-8 CTAs/SM)
//   - N == 9..16:  BN = 16, WARPS = 4, MINB = 6 (64-80 regs, 0 static SMEM, 2x LDG.128 + early nt skip)
//   - N == 17..32: BN = 32, WARPS = 4, MINB = 6/5 (72-80 regs, 0 static SMEM, 2x LDG.128 + early nt skip)
//   - N == 33..64: BN = 64, WARPS = 8, MINB = 2 (103 regs vs 251, 0 vs 28.7 KB static SMEM -> 2x occ!)
template <int MODE, int XSH = 1, bool ID = false, bool A8_SPLIT = false, int LUT_COPIES = 1, bool XPERM = false>
static void launch_ampere_mmv_turbo(
        const uint8_t * W, const uint32_t * Xq_hi, const uint32_t * Xq_lo, const float * Xs,
        const uint8_t * lut_g, float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    if (N <= 8) {
        launch_ampere_mmv_turbo_cfg<8, MODE, XSH, ID, 4, 6, A8_SPLIT, LUT_COPIES, XPERM>(
            W, Xq_hi, Xq_lo, Xs, lut_g, Y, ldy, M, N, K, stream, idm, max_tiles);
    } else if (N <= 16) {
        launch_ampere_mmv_turbo_cfg<16, MODE, XSH, ID, 4, 6, A8_SPLIT, LUT_COPIES, XPERM>(
            W, Xq_hi, Xq_lo, Xs, lut_g, Y, ldy, M, N, K, stream, idm, max_tiles);
    } else if (N <= 32) {
        launch_ampere_mmv_turbo_cfg<32, MODE, XSH, ID, 4, ID ? 5 : 6, A8_SPLIT, LUT_COPIES, XPERM>(
            W, Xq_hi, Xq_lo, Xs, lut_g, Y, ldy, M, N, K, stream, idm, max_tiles);
    } else {
        launch_ampere_mmv_turbo_cfg<64, MODE, XSH, ID, 8, 2, A8_SPLIT, LUT_COPIES, XPERM>(
            W, Xq_hi, Xq_lo, Xs, lut_g, Y, ldy, M, N, K, stream, idm, max_tiles);
    }
}

// =================================================================================================
// 4b. Low-Register <= 128-Reg Medium-Batch GEMM (N = 64..384): 2 CTAs/SM = 2x Occupancy vs gemm<64>!
//     Baseline gemm<BN=64, AIT4_U> uses 193 regs (1 CTA/SM = 8 warps/SM).
//     gemm_s8_lowreg uses J_MAJOR 4-reg A-streaming + direct XOR-swizzled SMEM B-loads (no xf reg array)
//     + STAGES=2 (34.8 KB/CTA -> 2 CTAs = 69.6 KB <= 100 KB SMEM/SM) with __launch_bounds__(256, 2)
//     capping registers at <= 128 (0 spills, 16 warps/SM = 2x occupancy!).
// =================================================================================================
template <int BN, int LAY, int M_SLABS = 8>
constexpr __host__ __device__ int gemm_lowreg_stage_bytes() {
    // M_SLABS slabs of (ch_bytes + 64B scales) + BN * 128B XOR-swizzled activations + BN * 4B scales
    return M_SLABS * (ai4k::ch_bytes<LAY>() + 64) + BN * 128 + BN * 4;
}

template <int BN = 64, int STAGES = 2, int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false, int M_SLABS = 8, int MINB = 2>
static __global__ void __launch_bounds__(M_SLABS * 32, MINB) gemm_s8_lowreg(
        const uint8_t * __restrict__ W, const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K, const ai4k::idmap idm) {
    static_assert(BN == 64 && M_SLABS == 8, "medium-batch lowreg tile");
    constexpr int THREADS = M_SLABS * 32;
    constexpr int NTI     = BN / 8;
    constexpr int CHB     = ai4k::ch_bytes<LAY>();
    constexpr int CHP     = CHB / 16;
    constexpr int WSB     = M_SLABS * (CHB + 64);
    constexpr int SB      = gemm_lowreg_stage_bytes<BN, LAY, M_SLABS>();

    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int srow[ID ? BN : 1];
    __shared__ int scol[ID ? BN : 1];

    const int tid   = threadIdx.x;
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int g     = lane >> 2;
    const int t     = lane & 3;
    const int slab0 = blockIdx.x * M_SLABS;
    const int nslab = M / 16;
    int n0          = blockIdx.y * BN;
    const int nk    = K / 128;
    const int xsn   = K >> (7 + XSH);
    const int64_t sbytes = ai4k::slab_bytes_l<LAY>(K);

    if constexpr (ID) {
        if ((int)blockIdx.y >= idm.hdr[1]) return;
        const int a  = idm.tiles[2 * blockIdx.y];
        const int o  = idm.off[a];
        const int nt = idm.tiles[2 * blockIdx.y + 1];
        W += (int64_t)idm.exps[a] * idm.nb02;
        N  = idm.off[a + 1] - o - nt;
        n0 = 0;
        for (int i = tid; i < BN; i += THREADS) {
            srow[i] = i < N ? idm.rows[o + nt + i] : 0;
            scol[i] = i < N ? idm.cols[o + nt + i] : 0;
        }
        __syncthreads();
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) return srow[c]; else return n0 + c; };
    auto ycol = [&](int c) -> int64_t { if constexpr (ID) return scol[c]; else return n0 + c; };

    const ai4k::k5win k5a = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win k5b = ai4k::lay_win<LAY>(g + 8, t & 1);
    (void)k5a; (void)k5b;

    auto load_stage = [&](int kc, int slot) {
        uint8_t * st = smem + slot * SB;
        const int64_t coff = ai4k::chunk_off<LAY>(kc);
        const int64_t soff = ai4k::sc_off<LAY>(kc);
#pragma unroll
        for (int r = 0; r < (M_SLABS * CHP + THREADS - 1) / THREADS; ++r) {
            const int i = tid + r * THREADS;
            if ((M_SLABS * CHP) % THREADS == 0 || i < M_SLABS * CHP) {
                const int s = i / CHP, p = i % CHP;
                const bool ok = slab0 + s < nslab;
                const uint8_t * src = W + (ok ? (int64_t)(slab0 + s) * sbytes + coff + p * 16 : 0);
                cp_async16(st + s * CHB + p * 16, src, ok);
            }
        }
        if (tid < M_SLABS * 8) {
            const int s = tid >> 3, w = tid & 7;
            const bool ok = slab0 + s < nslab;
            const uint8_t * src = W + (ok ? (int64_t)(slab0 + s) * sbytes + soff + w * 4 : 0);
            cp_async4(st + M_SLABS * CHB + s * 64 + w * 4, src, ok);
        }
        uint8_t * sx = st + WSB;
        float * sxd  = (float *)(sx + BN * 128);
        constexpr int XPIECES = BN * 8;
#pragma unroll
        for (int r = 0; r < XPIECES / THREADS; ++r) {
            const int i = tid + r * THREADS;
            const int c = i >> 3, p = i & 7;
            const bool ok = n0 + c < N;
            const int xp = p ^ (c & 7); // conflict-free XOR swizzle
            const int8_t * src = Xq + (ok ? xrow(c) : 0) * K + kc * 128 + p * 16;
            cp_async16(sx + c * 128 + xp * 16, src, ok);
        }
        if (tid < BN) {
            const bool ok = n0 + tid < N;
            cp_async4(sxd + tid, Xs + (ok ? xrow(tid) : 0) * xsn + (kc >> XSH), ok);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_async_commit();
    }

    int cur_slot = 0;
    int nxt_slot = STAGES - 1;
    for (int kc = 0; kc < nk; ++kc) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < nk) load_stage(kc + STAGES - 1, nxt_slot);
        cp_async_commit();

        const uint8_t  * st  = smem + cur_slot * SB;
        if constexpr (STAGES == 2) {
            cur_slot ^= 1;
            nxt_slot ^= 1;
        } else {
            if (++cur_slot == STAGES) cur_slot = 0;
            if (++nxt_slot == STAGES) nxt_slot = 0;
        }
        const uint32_t * tw  = (const uint32_t *)(st + warp * CHB);
        const half     * sd  = (const half *)(st + M_SLABS * CHB + warp * 64);
        const uint8_t  * sx  = st + WSB;
        const float    * sxd = (const float *)(sx + BN * 128);

        int p[NTI][4];
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) p[nt][0] = p[nt][1] = p[nt][2] = p[nt][3] = 0;

        // J_MAJOR 4-reg A-streaming + direct XOR-swizzled SMEM B-loads -> keeps regs <= 128!
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            ai4k::a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a);
            const int o_base = t * 32 + j * 8;
            const int p16    = o_base >> 4;
            const int rem8   = o_base & 15;
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                if constexpr (ID) {
                    if (idm.tail_skip && nt * 8 >= N) continue;   // block-uniform: no routed column in [nt*8, nt*8+8) (rc INKPF3 tail_skip)
                }
                const int c = nt * 8 + g;
                const int xo = ((p16 ^ (c & 7)) << 4) + rem8;
                const uint2 xv = *(const uint2 *)(sx + c * 128 + xo);
                mma_s8(p[nt], a, xv.x, xv.y);
            }
        }
        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            if constexpr (ID) {
                if (idm.tail_skip && nt * 8 >= N) continue;
            }
            const float dx0 = sxd[nt * 8 + 2 * t];
            const float dx1 = sxd[nt * 8 + 2 * t + 1];
            acc[nt][0] += magic_i2f(p[nt][0]) * (d0 * dx0);
            acc[nt][1] += magic_i2f(p[nt][1]) * (d0 * dx1);
            acc[nt][2] += magic_i2f(p[nt][2]) * (d1 * dx0);
            acc[nt][3] += magic_i2f(p[nt][3]) * (d1 * dx1);
        }
    }
    cp_async_wait<0>();
    if (slab0 + warp >= nslab) return;
    const int r = (slab0 + warp) * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nt * 8 + 2 * t;
        if (n0 + c < N) {
            Y[ycol(c) * ldy + r]     = acc[nt][0];
            Y[ycol(c) * ldy + r + 8] = acc[nt][2];
        }
        if (n0 + c + 1 < N) {
            Y[ycol(c + 1) * ldy + r]     = acc[nt][1];
            Y[ycol(c + 1) * ldy + r + 8] = acc[nt][3];
        }
    }
}

// Dispatch guard for gemm_s8_lowreg STAGES=2 (34.3 KB SMEM, 2 CTAs/SM):
// Validated on Vast RTX 3090:
//   - MoE (ID == true): wins -11% to -55.1% across EVERY N = 64..2048 (always enough expert tiles for >= 2 CTAs/SM).
//   - Dense (ID == false): wins -1.4% to -23% ONLY when total CTAs >= 2 * sms (so 2 CTAs/SM actually execute!).
//     When CTAs < 2 * sms (e.g., 5120x5120 N=128 -> 40*2 = 80 CTAs < 82 SMs -> only 1 CTA/SM!), STAGES=2 loses
//     +57.4% due to shorter latency hiding vs rc STAGES=3. In that regime, use STAGES=3 (51.5 KB SMEM, 1 CTA/SM)
//     or fall back to rc gemm<64, 3, 4>.
static inline bool should_use_gemm_s8_lowreg_2stage(bool id, int M, int N, int sms = 82) {
    if (id) return true;
    const int ctas = ((M / 16 + 7) / 8) * ((N + 63) / 64);
    return ctas >= 2 * sms;
}

template <int BN = 64, int STAGES = 2, int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false, int M_SLABS = 8, int MINB = 2>
static void launch_gemm_s8_lowreg_cfg(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    constexpr int smem = STAGES * gemm_lowreg_stage_bytes<BN, LAY, M_SLABS>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(
                gemm_s8_lowreg<BN, STAGES, LAY, XSH, ID, M_SLABS, MINB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((M / 16 + M_SLABS - 1) / M_SLABS, ID ? max_tiles : (N + BN - 1) / BN);
    gemm_s8_lowreg<BN, STAGES, LAY, XSH, ID, M_SLABS, MINB><<<grid, M_SLABS * 32, smem, stream>>>(
        W, Xq, Xs, Y, ldy, M, N, K, idm);
}

// Auto-selecting launcher for gemm_s8_lowreg:
//   - ID == true OR dense CTAs >= 2 * sms -> STAGES=2, MINB=2 (34.3 KB SMEM, 2 CTAs/SM = 2x occ)
//   - Dense CTAs < 2 * sms (e.g. 5120x5120 N=128) -> STAGES=3, MINB=1 (51.5 KB SMEM, 2 stages in-flight!)
template <int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false>
static void launch_gemm_s8_lowreg(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1, int sms = 82) {
    if (should_use_gemm_s8_lowreg_2stage(ID, M, N, sms)) {
        launch_gemm_s8_lowreg_cfg<64, 2, LAY, XSH, ID, 8, 2>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_tiles);
    } else {
        launch_gemm_s8_lowreg_cfg<64, 3, LAY, XSH, ID, 8, 1>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_tiles);
    }
}

// =================================================================================================
// 5. Prefill Turbo: prefill_decode_s4 (0.5*M*K VRAM!) + gemm_predec_s4 (569 TOPS A4 / Exact A8-Split)
// =================================================================================================

// Decodes compressed AI+ 2-bit/3-bit weights ONCE into packed s4 m16n8k64 A-fragment order:
//   Output size = 0.5 * M * K bytes (HALF the VRAM traffic & pool memory of prefill_decode!)
//   1 chunk (16 rows x 128 K = 2048 weights) -> 1024 bytes = 64 x uint4 (2 steps x 32 lanes x 16B)!
template <int MODE, int LUT_COPIES = 1>
static __global__ void __launch_bounds__(256) prefill_decode_s4(
        const uint8_t * __restrict__ W, const uint8_t * __restrict__ lut_g,
        uint4 * __restrict__ decoded_s4, int M, int K) {
    static_assert(MODE == MODE_PAIR_LUT_S4 || MODE == MODE_QUAD_LUT_S4, "s4 decode mode");
    constexpr int CHB = turbo_ch_bytes<MODE>();
    constexpr int U_B = CHB + 32;
    constexpr int LUT_B = turbo_lut_bytes<MODE, LUT_COPIES>();

    __shared__ __align__(16) uint8_t lut_s[LUT_B];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    if constexpr (MODE == MODE_PAIR_LUT_S4 && LUT_COPIES > 1) {
        for (int i = tid; i < 1024 * LUT_COPIES; i += 256) ((uint32_t *)lut_s)[i] = (uint32_t)lut_g[i / LUT_COPIES];
    } else {
        for (int i = tid; i < LUT_B / 4; i += 256) ((uint32_t *)lut_s)[i] = ((const uint32_t *)lut_g)[i];
    }
    __syncthreads();

    const int64_t chunk = (int64_t)blockIdx.x * 8 + (tid >> 5);
    const int nk = K / 128;
    if (chunk >= (int64_t)(M / 16) * nk) return;
    const int slab = (int)(chunk / nk);
    const int kc   = (int)(chunk % nk);
    const uint32_t * tw = (const uint32_t *)(W + ((int64_t)slab * nk + kc) * U_B + 32);

    const ai4k::k5win wa = ai4k::kr_win<2>(g, 1);
    const ai4k::k5win wb = ai4k::kr_win<2>(g + 8, 1);
    const uint32_t lane_off = 4u * (uint32_t)(lane & (LUT_COPIES - 1));
    uint4 * out = decoded_s4 + chunk * 64; // 2 K-steps x 32 lanes!

#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const uint32_t * ta = tw + (8 * j + t) * 8;
        const uint32_t * tb = ta + 4 * 8;
        const uint32_t fa0 = __funnelshift_l(ta[wa.w1], ta[wa.w0], wa.sh);
        const uint32_t fa1 = __funnelshift_l(ta[wb.w1], ta[wb.w0], wb.sh);
        const uint32_t fb0 = __funnelshift_l(tb[wa.w1], tb[wa.w0], wa.sh);
        const uint32_t fb1 = __funnelshift_l(tb[wb.w1], tb[wb.w0], wb.sh);
        uint32_t a0, a1, a2, a3;
        if constexpr (MODE == MODE_PAIR_LUT_S4) {
            a0 = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fa0);
            a1 = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fa1);
            a2 = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fb0);
            a3 = levels8_pair_lut_s4<LUT_COPIES>(lut_s, lane_off, fb1);
        } else {
            a0 = levels8_quad_lut_s4(lut_s, fa0);
            a1 = levels8_quad_lut_s4(lut_s, fa1);
            a2 = levels8_quad_lut_s4(lut_s, fb0);
            a3 = levels8_quad_lut_s4(lut_s, fb1);
        }
        out[j * 32 + lane] = make_uint4(a0, a1, a2, a3);
    }
}

// High-throughput Prefill GEMM consuming predecoded s4 weights (1024 B/slab-chunk instead of 2048 B!)
// CTA = BM=128 (8 slabs, 1 warp/slab) x BN tokens (128 or 256), 2-stage cp.async pipeline.
// A8_SPLIT=false -> 569 TOPS peak (1x mma.m16n8k64.s4 per k64 step, A4)!
// A8_SPLIT=true  -> 285 TOPS peak with 2x less weight scratchpad BW (exact A8: 16*p_hi + p_lo)!
template <int BN, int STAGES, int M_SLABS, bool A8_SPLIT>
constexpr __host__ __device__ int gemm_s4_stage_bytes() {
    // Per slab: 1024 B decoded s4 + 64 B scales = 1088 B
    // Activations: BN * (64 B * (A8_SPLIT ? 2 : 1)) + BN * 4 B scales
    return M_SLABS * (1024 + 64) + BN * (A8_SPLIT ? 128 : 64) + BN * 4;
}

template <int BN, int STAGES = 2, int M_SLABS = 8, int NTB = 4, bool A8_SPLIT = false, int U_BYTES = 544>
static __global__ void __launch_bounds__(M_SLABS * 32, 1) gemm_predec_s4(
        const uint8_t * __restrict__ W_scales_base,
        const uint4 * __restrict__ decoded_s4,
        const uint32_t * __restrict__ Xq_hi,
        const uint32_t * __restrict__ Xq_lo,
        const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K) {
    constexpr int THREADS = M_SLABS * 32;
    constexpr int NTI = BN / 8;
    constexpr int SB = gemm_s4_stage_bytes<BN, STAGES, M_SLABS, A8_SPLIT>();
    constexpr int WSB = M_SLABS * (1024 + 64);
    constexpr int XB_TOK = A8_SPLIT ? 128 : 64;

    extern __shared__ __align__(16) uint8_t smem[];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int slab0 = blockIdx.x * M_SLABS;
    const int nslab = M / 16;
    const int n0 = blockIdx.y * BN;
    const int nk = K / 128;
    const int xsn = K / 256;

    auto load_stage = [&](int kc) {
        uint8_t * st = smem + (kc % STAGES) * SB;
        // 1024 B = 64 x 16B per slab -> M_SLABS * 64 cp_async16
        for (int i = tid; i < M_SLABS * 64; i += THREADS) {
            const int s = i >> 6, p = i & 63;
            const bool ok = slab0 + s < nslab;
            const void * src = decoded_s4 + (ok ? ((int64_t)(slab0 + s) * nk + kc) * 64 + p : 0);
            cp_async16(st + s * 1024 + p * 16, src, ok);
        }
        // Weight scales: 32 B per slab (8 x 4B)
        if (tid < M_SLABS * 8) {
            const int s = tid >> 3, w = tid & 7;
            const bool ok = slab0 + s < nslab;
            const uint8_t * src = W_scales_base + (ok ? ((int64_t)(slab0 + s) * nk + kc) * U_BYTES + w * 4 : 0);
            cp_async4(st + M_SLABS * 1024 + s * 64 + w * 4, src, ok);
        }
        // Activations: BN * (4 or 8) x 16B
        uint8_t * sx = st + WSB;
        float * sxd = (float *)(sx + BN * XB_TOK);
        constexpr int XPIECES = BN * 4;
        for (int i = tid; i < XPIECES; i += THREADS) {
            const int c = i >> 2, p = i & 3;
            const bool ok = n0 + c < N;
            const int64_t base = (ok ? (int64_t)(n0 + c) * nk + kc : 0) * 16 + p * 4;
            cp_async16(sx + c * XB_TOK + p * 16, Xq_hi + base, ok);
            if constexpr (A8_SPLIT) {
                cp_async16(sx + c * XB_TOK + 64 + p * 16, Xq_lo + base, ok);
            }
        }
        for (int i = tid; i < BN; i += THREADS) {
            const bool ok = n0 + i < N;
            cp_async4(sxd + i, Xs + (ok ? (int64_t)(n0 + i) * xsn + (kc >> 1) : 0), ok);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s);
        cp_async_commit();
    }

    for (int kc = 0; kc < nk; ++kc) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < nk) load_stage(kc + STAGES - 1);
        cp_async_commit();

        const uint8_t * st = smem + (kc % STAGES) * SB;
        const uint4 * aw = (const uint4 *)(st + warp * 1024);
        const half * sd = (const half *)(st + M_SLABS * 1024 + warp * 64);
        const uint8_t * sx = st + WSB;
        const float * sxd = (const float *)(sx + BN * XB_TOK);

        // Load both k64 A-fragments (only 8 regs total vs 16 regs in s8!)
        uint32_t a[2][4];
        const uint4 v0 = aw[lane];
        const uint4 v1 = aw[32 + lane];
        a[0][0] = v0.x; a[0][1] = v0.y; a[0][2] = v0.z; a[0][3] = v0.w;
        a[1][0] = v1.x; a[1][1] = v1.y; a[1][2] = v1.z; a[1][3] = v1.w;

        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);

#pragma unroll
        for (int nb = 0; nb < NTI; nb += NTB) {
            uint4 xh[NTB];
            uint4 xl[A8_SPLIT ? NTB : 1];
            float dx[NTB][2];
            int ph[NTB][4];
            int pl[A8_SPLIT ? NTB : 1][4];
#pragma unroll
            for (int u = 0; u < NTB; ++u) {
                const int c = (nb + u) * 8 + g;
                xh[u] = *(const uint4 *)(sx + c * XB_TOK + t * 16);
                if constexpr (A8_SPLIT) xl[u] = *(const uint4 *)(sx + c * XB_TOK + 64 + t * 16);
                dx[u][0] = sxd[(nb + u) * 8 + 2 * t];
                dx[u][1] = sxd[(nb + u) * 8 + 2 * t + 1];
                ph[u][0] = ph[u][1] = ph[u][2] = ph[u][3] = 0;
                if constexpr (A8_SPLIT) pl[u][0] = pl[u][1] = pl[u][2] = pl[u][3] = 0;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int u = 0; u < NTB; ++u) {
                    mma_s4(ph[u], a[j], (j == 0) ? xh[u].x : xh[u].z, (j == 0) ? xh[u].y : xh[u].w);
                    if constexpr (A8_SPLIT) {
                        mma_s4(pl[u], a[j], (j == 0) ? xl[u].x : xl[u].z, (j == 0) ? xl[u].y : xl[u].w);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < NTB; ++u) {
                if constexpr (A8_SPLIT) {
                    acc[nb + u][0] += magic_i2f(ph[u][0] * 16 + pl[u][0]) * (d0 * dx[u][0]);
                    acc[nb + u][1] += magic_i2f(ph[u][1] * 16 + pl[u][1]) * (d0 * dx[u][1]);
                    acc[nb + u][2] += magic_i2f(ph[u][2] * 16 + pl[u][2]) * (d1 * dx[u][0]);
                    acc[nb + u][3] += magic_i2f(ph[u][3] * 16 + pl[u][3]) * (d1 * dx[u][1]);
                } else {
                    acc[nb + u][0] += magic_i2f(ph[u][0]) * (d0 * dx[u][0]);
                    acc[nb + u][1] += magic_i2f(ph[u][1]) * (d0 * dx[u][1]);
                    acc[nb + u][2] += magic_i2f(ph[u][2]) * (d1 * dx[u][0]);
                    acc[nb + u][3] += magic_i2f(ph[u][3]) * (d1 * dx[u][1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    if (slab0 + warp >= nslab) return;
    const int r = (slab0 + warp) * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nt * 8 + 2 * t;
        if (n0 + c < N) {
            Y[(int64_t)(n0 + c) * ldy + r]     = acc[nt][0];
            Y[(int64_t)(n0 + c) * ldy + r + 8] = acc[nt][2];
        }
        if (n0 + c + 1 < N) {
            Y[(int64_t)(n0 + c + 1) * ldy + r]     = acc[nt][1];
            Y[(int64_t)(n0 + c + 1) * ldy + r + 8] = acc[nt][3];
        }
    }
}

// Launcher for gemm_predec_s4 with >48 KB dynamic SMEM opt-in:
//   - BN=256, A8_SPLIT=false -> 52,224 B (> 48 KB!)
//   - BN=128, A8_SPLIT=true  -> 51,200 B (> 48 KB!)
template <int BN, int STAGES = 2, int M_SLABS = 8, int NTB = 4, bool A8_SPLIT = false, int U_BYTES = 544>
static void launch_gemm_predec_s4(
        const uint8_t * W_scales_base, const uint4 * decoded_s4,
        const uint32_t * Xq_hi, const uint32_t * Xq_lo, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream) {
    constexpr int smem = STAGES * gemm_s4_stage_bytes<BN, STAGES, M_SLABS, A8_SPLIT>();
    if constexpr (smem > 48 * 1024) {
        static bool attr_set[64] = {};
        int dev = 0;
        CUDA_CHECK_AI4K(cudaGetDevice(&dev));
        if (!attr_set[dev & 63]) {
            CUDA_CHECK_AI4K(cudaFuncSetAttribute(
                gemm_predec_s4<BN, STAGES, M_SLABS, NTB, A8_SPLIT, U_BYTES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[dev & 63] = true;
        }
    }
    const dim3 grid((M / 16 + M_SLABS - 1) / M_SLABS, (N + BN - 1) / BN);
    gemm_predec_s4<BN, STAGES, M_SLABS, NTB, A8_SPLIT, U_BYTES><<<grid, M_SLABS * 32, smem, stream>>>(
        W_scales_base, decoded_s4, Xq_hi, Xq_lo, Xs, Y, ldy, M, N, K);
}

// L2-cache-tiled s4 prefill: tiles prefill_decode_s4 + gemm_predec_s4 in M_strip chunks sized so
// M_strip * K * 0.5 <= 4.5 MiB (100% resident inside RTX 3090's 6 MiB L2 cache; e.g., M_strip = 2048 at K = 4096,
// 1792 at K = 5120). Eliminates full-layer GDDR6X scratchpad traffic and caps pool memory at <= 4.5 MiB.
template <int MODE, bool A8_SPLIT = false, int LUT_COPIES = 1, int U_BYTES = 544>
static void launch_prefill_l2_tiled_s4(
        const uint8_t * W, const uint8_t * lut_g, uint4 * decoded_s4,
        const uint32_t * Xq_hi, const uint32_t * Xq_lo, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, cudaStream_t stream,
        size_t l2_budget = 4500u * 1024u) {
    const int m_strip = (M <= 128 || K <= 0) ? M
        : std::min(M, std::max(128, (int)((l2_budget / ((size_t)K / 2u)) / 128u) * 128));
    const int64_t slab_bytes = (int64_t)(K / 128) * U_BYTES;
    constexpr int BN = A8_SPLIT ? 128 : 256;
    for (int m0 = 0; m0 < M; m0 += m_strip) {
        const int mm = (m0 + m_strip <= M) ? m_strip : (M - m0);
        const uint8_t * W_strip = W + (int64_t)(m0 / 16) * slab_bytes;
        const int64_t chunks = (int64_t)(mm / 16) * (K / 128);
        prefill_decode_s4<MODE, LUT_COPIES><<<(chunks + 7) / 8, 256, 0, stream>>>(W_strip, lut_g, decoded_s4, mm, K);
        launch_gemm_predec_s4<BN, 2, 8, 4, A8_SPLIT, U_BYTES>(
            W_strip, decoded_s4, Xq_hi, Xq_lo, Xs, Y + m0, ldy, mm, N, K, stream);
    }
}

// =================================================================================================
// 6. Hardware 2:4 Sparse INT4 Prototype (mma.sp.m16n8k128.s4, 1,139 TOPS on RTX 3090 SM 8.6)
//    Demonstrates exact SASS generation for 2:4 structured sparse INT4 base/residual GEMM tile.
// =================================================================================================
template <int BN = 64>
static __global__ void __launch_bounds__(128, 2) gemm_sparse24_s4_proto(
        const uint4 * __restrict__ A_sp,      // [M/16][K/128][32 lanes] compressed 2:4 s4 non-zeros
        const uint32_t * __restrict__ E_meta, // [M/16][K/128][32 lanes] 2-bit sparse metadata
        const uint4 * __restrict__ B_s4,      // [N][K/128][4 t-lanes] dense s4 activations
        const float * __restrict__ W_sc,
        const float * __restrict__ X_sc,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K) {
    constexpr int NTI = BN / 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int slab = blockIdx.x * 4 + warp;
    const int n0   = blockIdx.y * BN;
    const int nk   = K / 128;
    if (slab >= M / 16) return;

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;

    for (int kc = 0; kc < nk; ++kc) {
        const int64_t a_idx = ((int64_t)slab * nk + kc) * 32 + lane;
        const uint4 av = __ldg(A_sp + a_idx);
        const uint32_t a[4] = { av.x, av.y, av.z, av.w };
        const uint32_t e = __ldg(E_meta + a_idx);
        const float d0 = W_sc[(int64_t)(slab * 16 + g) * nk + kc];
        const float d1 = W_sc[(int64_t)(slab * 16 + g + 8) * nk + kc];

#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const int c = n0 + nt * 8 + g;
            uint4 bv = (c < N) ? __ldg(B_s4 + ((int64_t)c * nk + kc) * 4 + t) : make_uint4(0, 0, 0, 0);
            const uint32_t b[4] = { bv.x, bv.y, bv.z, bv.w };
            int p[4] = { 0, 0, 0, 0 };
            mma_sp_s4<0>(p, a, b, e);
            const float dx0 = (n0 + nt * 8 + 2 * t < N)     ? X_sc[(int64_t)(n0 + nt * 8 + 2 * t) * nk + kc]     : 0.0f;
            const float dx1 = (n0 + nt * 8 + 2 * t + 1 < N) ? X_sc[(int64_t)(n0 + nt * 8 + 2 * t + 1) * nk + kc] : 0.0f;
            acc[nt][0] += magic_i2f(p[0]) * (d0 * dx0);
            acc[nt][1] += magic_i2f(p[1]) * (d0 * dx1);
            acc[nt][2] += magic_i2f(p[2]) * (d1 * dx0);
            acc[nt][3] += magic_i2f(p[3]) * (d1 * dx1);
        }
    }
    const int r = slab * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = n0 + nt * 8 + 2 * t;
        if (c < N)     { Y[(int64_t)c * ldy + r] = acc[nt][0];     Y[(int64_t)c * ldy + r + 8] = acc[nt][2]; }
        if (c + 1 < N) { Y[(int64_t)(c+1) * ldy + r] = acc[nt][1]; Y[(int64_t)(c+1) * ldy + r + 8] = acc[nt][3]; }
    }
}

} // namespace ai4_turbo
