// AI4 (Ampere INT4, formerly D2) fast path: trellis decode straight into s8 m16n8k32 A fragments, on track B's W4A8 skeleton (rq4.cuh: the
// activation kernel, IMMA, cp.async, magic_i2f epilogue). Format: ggml/src/ggml-ai4.h. No ggml dependencies, so the
// standalone bench/test (tools/ai4/ai4-kern-bench.cu) and the ggml backend (ai4.cu) build the same code.
//
// Decode of one A register (4 weights = 4 consecutive symbols of one row of one 16x8 tile):
//   f      = (t odd)  ? word[r]                         (columns 4..7: states at shifts 12, 8, 4, 0 of word[r])
//            (t even) ? (word[r-1]:word[r]) >> 16       (columns 0..3: shifts 28..16 of the 64-bit pair)
//   s_b    = (f >> (12 - 4b)) & 0xFFFF                   b = 0..3 (byte b of the register = column 4(t&1) + b)
//   r_b    = dp4a(s_b * 0x83DCD12D, 0x40404040, 128 - 32768): byte 1 of r_b = ((bytesum + 2) >> 2) - 128 as s8
//   A      = byte 1 of r_0..r_3 (3 PRMT)
// = 17 instructions per 4 weights (8 IMAD/DP4A, 9 ALU). The A fragment of IMMA j for lane (g, t) is rows g / g+8 of
// tiles 4j + (t>>1) (a[0], a[1]) and 4j + 2 + (t>>1) (a[2], a[3]) of the current 128-wide chunk; levels are the
// weights in units of the group scale, so out_mul = 1 and |level| <= 128 keeps magic_i2f exact (128*128*127 < 2^22).
//
// Kernels:
//   gemm  (N > 8): CTA = 128 rows (8 slabs) x BN tokens, 8 warps, one slab per warp (each A fragment decoded once per
//                  CTA), 3-stage cp.async pipeline of the slab chunk tiles + scale headers + activation tile.
//   mmv   (N <= 8): block = one slab, 4 warps split the K chunks; per-warp cp.async ring of tiles + activations +
//                  scales; 4 IMMAs per chunk; shared-memory reduction.
//
// Template knobs shared by both kernels:
//   LAY = LAY_SB (0): AI4, slab super blocks of 16 x 1024 ([d 16x8][128 tiles]); chunk kc at (kc>>3)*8448 + 256 + (kc&7)*1024,
//                     row r's scale of chunk kc = d[r*8 + kc%8]
//   LAY = LAY_U  (1): AI4_U, units of 16 x 128 ([d 16][16 tiles], 1056 B); chunk kc at kc*1056 + 32, scale = d[r]
//   (both: a slab is K/128*1056 bytes)
//   LAY = LAY_S  (3): AI4_S, spans of 16 x 1024 ([d 16][8 x 16 tiles], 8224 B); chunk kc at (kc>>3)*8224 + 32 + (kc&7)*1024,
//                     scale = d[r] of the span (one per 1024 K, so the scale load repeats across the span's 8 chunks)
//   LAY = LAY_U5 (2) / LAY_S5 (4): the same two layouts at rate 5 (20-word tiles: 1312 B units, 10272 B spans)
//   LAY = LAY_U2 (5) / LAY_S2 (7), LAY_U3 (6) / LAY_S3 (8): rates 2 and 3 (8- / 12-word tiles, generic window decode)
//   LAY = LAY_U6 (9) / LAY_S6 (10): the same two layouts at rate 6 (24-word tiles: 1568 B units, 12320 B spans) under the
//                     AI6 trellis constant 0xA1EEE321 (ggml_ai4_trellis_mul); a 4-symbol state window is 34 bits there, so
//                     each A register reads a 64-bit window (k6_win / levels4_k6)
//   XSH = 1: one activation scale per (token, 256 K) (rq4::quantize_act, H256), XSH = 0: per (token, 128 K) (quantize_act128, H128)
//   ID  = true: MUL_MAT_ID; the grid's y index selects an (active expert, token tile) from idmap (built by ai4k::ids_helper),
//         weights = W + expert*nb02, token c of the tile reads Xq row rows[c] and writes Y column cols[c]. Experts that no
//         token selects are never read.

#pragma once

#include "rq4.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// attribute-call check usable from both the ggml backend and the standalone bench
#define CUDA_CHECK_AI4K(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "ai4k: %s failed: %s\n", #x, cudaGetErrorString(e_)); abort(); } } while (0)

namespace ai4k {

constexpr int LAY_SB = 0;
constexpr int LAY_U  = 1;
constexpr int LAY_U5 = 2;   // AI5_U: rate-5 units of 16 x 128 ([d 16][16 tiles x 20 words], 1312 B)
constexpr int LAY_S  = 3;   // AI4_S: rate-4 spans of 16 x 1024 ([d 16][8 x 16 tiles], 8224 B), one scale per row and span
constexpr int LAY_S5 = 4;   // AI5_S: rate-5 spans of 16 x 1024 ([d 16][8 x 16 tiles x 20 words], 10272 B)
constexpr int LAY_U2 = 5;   // AI2_U: rate-2 units of 16 x 128 ([d 16][16 tiles x 8 words], 544 B)
constexpr int LAY_U3 = 6;   // AI3_U: rate-3 units of 16 x 128 ([d 16][16 tiles x 12 words], 800 B)
constexpr int LAY_S2 = 7;   // AI2_S: rate-2 spans of 16 x 1024 ([d 16][8 x 16 tiles x 8 words], 4128 B)
constexpr int LAY_S3 = 8;   // AI3_S: rate-3 spans of 16 x 1024 ([d 16][8 x 16 tiles x 12 words], 6176 B)
constexpr int LAY_U6 = 9;   // AI6_U: rate-6 units of 16 x 128 ([d 16][16 tiles x 24 words], 1568 B), constant 0xA1EEE321
constexpr int LAY_S6 = 10;   // AI6_S: rate-6 spans of 16 x 1024 ([d 16][8 x 16 tiles x 24 words], 12320 B), constant 0xA1EEE321

// mmv-ID columns per block: lanes g = 0..7 of the mmv kernel each own one token column
constexpr int MMV_ID_BN = 8;

struct idmap {
    const int * hdr;     // [0] = active experts, [1] = token tiles
    const int * exps;    // active expert ids
    const int * off;     // [active + 1] pair offsets per active expert
    const int * rows;    // pair -> Xq row
    const int * cols;    // pair -> Y column
    const int * tiles;   // per token tile: active index, first pair of the tile (relative to off)
    int64_t     nb02;    // bytes per expert
    const char * direct_ids = nullptr;
    int64_t direct_nb0 = 0, direct_nb1 = 0;
    int direct_n_used = 0, direct_ne11 = 0, direct_n_expert = 0;
    const float * direct_x = nullptr;
};

using rq4::cp_async16;
using rq4::cp_async4;
using rq4::cp_async_commit;
using rq4::cp_async_wait;
using rq4::mma_s8;
using rq4::magic_i2f;

static __device__ __forceinline__ void cp_async8(void * smem, const void * gmem, bool pred) {
    const uint32_t dst = (uint32_t) __cvta_generic_to_shared(smem);
    const int size = pred ? 8 : 0;
#ifdef RQ4_PRE_AMPERE
    __trap(); (void) dst; (void) size;
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(dst), "l"(gmem), "r"(size));
#endif
}

// Separate participation from zero filling without divergent C++ control flow.
static __device__ __forceinline__ void cp_async4_guard(void * smem, const void * gmem, bool live, bool participate) {
    const uint32_t dst = (uint32_t) __cvta_generic_to_shared(smem);
    const int size = live ? 4 : 0;
#ifdef RQ4_PRE_AMPERE
    __trap(); (void) dst; (void) size; (void) participate;
#else
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %3, 0; @p cp.async.ca.shared.global [%0], [%1], 4, %2; }\n"
                 :: "r"(dst), "l"(gmem), "r"(size), "r"((int) participate));
#endif
}

constexpr int SB_BYTES  = 8448;   // slab super block: 16 rows x 1024 K
constexpr int SC_BYTES  = 256;    // fp16 d[16][8]
constexpr int CH_BYTES  = 1024;   // tile words of one 128-wide chunk of one slab (16 tiles x 64 B)

constexpr int U_BYTES   = 1056;   // AI4_U unit: 16 rows x 128 K
constexpr int U_SC      = 32;     // fp16 d[16]

static __host__ __device__ __forceinline__ int64_t slab_bytes(int K) { return (int64_t) (K/128)*U_BYTES; }   // both layouts

// symbol rate of a layout (bits per weight in the tile stream) and the span layouts (one scale per row and 1024 K)
template <int LAY> constexpr __host__ __device__ int lay_rate() {
    return (LAY == LAY_U6 || LAY == LAY_S6) ? 6 : (LAY == LAY_U5 || LAY == LAY_S5) ? 5 : (LAY == LAY_U2 || LAY == LAY_S2) ? 2 : (LAY == LAY_U3 || LAY == LAY_S3) ? 3 : 4;
}
template <int LAY> constexpr __host__ __device__ bool lay_k5()  { return lay_rate<LAY>() == 5; }
template <int LAY> constexpr __host__ __device__ bool lay_span() { return LAY == LAY_S || LAY == LAY_S5 || LAY == LAY_S2 || LAY == LAY_S3 || LAY == LAY_S6; }

// per-layout chunk tile bytes (16 tiles x 16k bytes = 256k), unit bytes and slab bytes. The span layouts have no
// per-unit header, so their "unit bytes" is the 128-K share of a span (ch_bytes + 32/8) and slab_bytes_l stays
// (K/128)*u_bytes: 1028 B per 128 K at k = 4, 1284 at k = 5, 516 at k = 2, 772 at k = 3.
template <int LAY> constexpr __host__ __device__ int ch_bytes() { return 256*lay_rate<LAY>(); }
template <int LAY> constexpr __host__ __device__ int u_bytes()  {
    return lay_span<LAY>() ? ch_bytes<LAY>() + U_SC/8 : ch_bytes<LAY>() + U_SC;
}
template <int LAY> constexpr __host__ __device__ int span_bytes() { return U_SC + 8*ch_bytes<LAY>(); }
static_assert(ch_bytes<LAY_SB>() == CH_BYTES && u_bytes<LAY_U>() == U_BYTES && u_bytes<LAY_U5>() == 1312 &&
              span_bytes<LAY_S>() == 8224 && span_bytes<LAY_S5>() == 10272, "ai4k layout sizes");
template <int LAY> static __host__ __device__ __forceinline__ int64_t slab_bytes_l(int K) { return (int64_t) (K/128)*u_bytes<LAY>(); }
static_assert(span_bytes<LAY_S6>() == 12320 && u_bytes<LAY_U6>() == 1568 && u_bytes<LAY_S6>()*8 == 12320, "ai4k rate-6 sizes");

// byte offset of chunk kc's 16 tiles inside a slab
template <int LAY>
static __device__ __forceinline__ int64_t chunk_off(int kc) {
    if constexpr (lay_span<LAY>()) {
        return (int64_t) (kc >> 3)*span_bytes<LAY>() + U_SC + (kc & 7)*ch_bytes<LAY>();
    } else if constexpr (LAY != LAY_SB) {
        return (int64_t) kc*u_bytes<LAY>() + U_SC;
    } else {
        return (int64_t) (kc >> 3)*SB_BYTES + SC_BYTES + (kc & 7)*CH_BYTES;
    }
}

// byte offset of chunk kc's scale header inside a slab (LAY_SB: the super block's 256 B header, indexed at the call site)
template <int LAY>
static __device__ __forceinline__ int64_t sc_off(int kc) {
    if constexpr (lay_span<LAY>()) {
        return (int64_t) (kc >> 3)*span_bytes<LAY>();
    } else if constexpr (LAY != LAY_SB) {
        return (int64_t) kc*u_bytes<LAY>();
    } else {
        return (int64_t) (kc >> 3)*SB_BYTES;
    }
}

// four s8 levels from the states at shifts 12, 8, 4, 0 of f (byte 0 = shift 12)
static __device__ __forceinline__ uint32_t levels4(const uint32_t f) {
    const uint32_t g4 = f >> 4;
    const uint32_t s0 = __byte_perm(g4, 0u, 0x4421);
    const uint32_t s1 = __byte_perm(f,  0u, 0x4421);
    const uint32_t s2 = g4 & 0xFFFFu;
    const uint32_t s3 = f  & 0xFFFFu;
    // all three operands are uint32_t on purpose: the unsigned __dp4a overload (u8 x u8 bytes, mod 2^32 sum) is what
    // makes byte 1 of r = ((bytesum + 2) >> 2) - 128 (byte 0 of C is 0, so nothing borrows into byte 1)
    constexpr uint32_t C = 128u - 32768u;
    const uint32_t r0 = __dp4a(s0*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r1 = __dp4a(s1*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r2 = __dp4a(s2*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r3 = __dp4a(s3*0x83DCD12Du, 0x40404040u, C);
    const uint32_t lo = __byte_perm(r0, r1, 0x0051);
    const uint32_t hi = __byte_perm(r2, r3, 0x0051);
    return __byte_perm(lo, hi, 0x5410);
}

// A registers of IMMA j from the 16 tiles (16 words each) of one slab chunk in shared memory
static __device__ __forceinline__ void a_regs_smem(const uint32_t * __restrict__ tw, int j, int g, int t, uint32_t (&a)[4]) {
    const int sh = (t & 1) ? 0 : 16;
    const uint32_t * ta = tw + (4*j + (t >> 1))*16;
    const uint32_t * tb = ta + 2*16;
    const int gp = (g + 15) & 15;   // row g - 1 (tail-biting: row 15 for g = 0)
    a[0] = levels4(__funnelshift_r(ta[g],     ta[gp],    sh));
    a[1] = levels4(__funnelshift_r(ta[g + 8], ta[g + 7], sh));
    a[2] = levels4(__funnelshift_r(tb[g],     tb[gp],    sh));
    a[3] = levels4(__funnelshift_r(tb[g + 8], tb[g + 7], sh));
}

// AI5_U: four s8 levels from the states at shifts 15, 10, 5, 0 of the 32-bit stream window f (byte 0 = shift 15)
static __device__ __forceinline__ uint32_t levels4_k5(const uint32_t f) {
    const uint32_t s0 = (f >> 15) & 0xFFFFu;
    const uint32_t s1 = (f >> 10) & 0xFFFFu;
    const uint32_t s2 = (f >>  5) & 0xFFFFu;
    const uint32_t s3 =  f        & 0xFFFFu;
    constexpr uint32_t C = 128u - 32768u;
    const uint32_t r0 = __dp4a(s0*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r1 = __dp4a(s1*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r2 = __dp4a(s2*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r3 = __dp4a(s3*0x83DCD12Du, 0x40404040u, C);
    const uint32_t lo = __byte_perm(r0, r1, 0x0051);
    const uint32_t hi = __byte_perm(r2, r3, 0x0051);
    return __byte_perm(lo, hi, 0x5410);
}

// Window of the 4 symbols n0..n0+3 (n0 = 8R + 4h: row R, columns 4h..4h+3) of a rate-5 tile: the 32 stream bits
// [5*n0 - 12, 5*n0 + 20) (mod 640, tail-biting) = words w0 (high part) and w1 at bit offset sh. Per lane constant.
struct k5win { int w0, w1, sh; };
static __device__ __forceinline__ k5win k5_win(int R, int h) {
    int s = 40*R + 20*h - 12;
    s += s < 0 ? 640 : 0;
    k5win r;
    r.w0 = s >> 5; r.sh = s & 31; r.w1 = r.w0 + 1 == 20 ? 0 : r.w0 + 1;
    return r;
}
static __device__ __forceinline__ void a_regs_smem_k5(const uint32_t * __restrict__ tw, int j, int t, const k5win & wa, const k5win & wb,
                                                      uint32_t (&a)[4]) {
    const uint32_t * ta = tw + (4*j + (t >> 1))*20;
    const uint32_t * tb = ta + 2*20;
    a[0] = levels4_k5(__funnelshift_l(ta[wa.w1], ta[wa.w0], wa.sh));
    a[1] = levels4_k5(__funnelshift_l(ta[wb.w1], ta[wb.w0], wb.sh));
    a[2] = levels4_k5(__funnelshift_l(tb[wa.w1], tb[wa.w0], wa.sh));
    a[3] = levels4_k5(__funnelshift_l(tb[wb.w1], tb[wb.w0], wb.sh));
}

// Rate 6 (AI6_U / AI6_S). State of symbol n = the 16 stream bits ending at the end of symbol n (ggml-ai4.h aik_get_state),
// so the 4 states of symbols n0..n0+3 (n0 = 8R + 4h) span the 16 + 3*6 = 34 bits that end at 6*(n0 + 4): more than one
// 32-bit window. Take the 64 stream bits [6*(n0 + 4) - 64, 6*(n0 + 4)) (mod 768, tail-biting) = words w0, w1, w2 at bit
// offset sh (per lane constant); hi:lo is that window, state of n0 + b = bits 6*(3 - b) .. +15 of it.
struct k6win { int w0, w1, w2, sh; };
static __device__ __forceinline__ k6win k6_win(int R, int h) {
    int s = 48*R + 24*h - 40;   // 6*(8R + 4h + 4) - 64
    s += s < 0 ? 768 : 0;
    k6win r;
    r.w0 = s >> 5; r.sh = s & 31;
    r.w1 = r.w0 + 1 == 24 ? 0 : r.w0 + 1;
    r.w2 = r.w1 + 1 == 24 ? 0 : r.w1 + 1;
    return r;
}
// four s8 levels of the AI6 trellis (constant 0xA1EEE321) from the 64-bit window hi:lo (byte 0 = symbol n0)
static __device__ __forceinline__ uint32_t levels4_k6(const uint32_t hi, const uint32_t lo) {
    const uint32_t s0 = __funnelshift_r(lo, hi, 18) & 0xFFFFu;
    const uint32_t s1 = (lo >> 12) & 0xFFFFu;
    const uint32_t s2 = (lo >>  6) & 0xFFFFu;
    const uint32_t s3 =  lo        & 0xFFFFu;
    constexpr uint32_t C = 128u - 32768u;
    const uint32_t r0 = __dp4a(s0*0xA1EEE321u, 0x40404040u, C);
    const uint32_t r1 = __dp4a(s1*0xA1EEE321u, 0x40404040u, C);
    const uint32_t r2 = __dp4a(s2*0xA1EEE321u, 0x40404040u, C);
    const uint32_t r3 = __dp4a(s3*0xA1EEE321u, 0x40404040u, C);
    const uint32_t lo2 = __byte_perm(r0, r1, 0x0051);
    const uint32_t hi2 = __byte_perm(r2, r3, 0x0051);
    return __byte_perm(lo2, hi2, 0x5410);
}
static __device__ __forceinline__ uint32_t k6_reg(const uint32_t * __restrict__ tile, const k6win & w) {
    const uint32_t a = tile[w.w0], b = tile[w.w1], c = tile[w.w2];
    return levels4_k6(__funnelshift_l(b, a, w.sh), __funnelshift_l(c, b, w.sh));
}
static __device__ __forceinline__ void a_regs_smem_k6(const uint32_t * __restrict__ tw, int j, int t, const k6win & wa, const k6win & wb,
                                                      uint32_t (&a)[4]) {
    const uint32_t * ta = tw + (4*j + (t >> 1))*24;
    const uint32_t * tb = ta + 2*24;
    a[0] = k6_reg(ta, wa);
    a[1] = k6_reg(ta, wb);
    a[2] = k6_reg(tb, wa);
    a[3] = k6_reg(tb, wb);
}

// Any rate KR <= 5 (used for k = 2, 3): the window of the 4 symbols n0..n0+3 (n0 = 8R + 4h) is the 32 stream bits that end
// at the end of symbol n0 + 3, [KR*(n0 + 4) - 32, KR*(n0 + 4)) (mod 128 KR, tail-biting); state of symbol n0 + b =
// bits KR*(3 - b) .. +15 of it (16 + 3 KR <= 32 bits). For KR = 4 this is the a_regs_smem window, for KR = 5 k5_win.
template <int KR>
static __device__ __forceinline__ k5win kr_win(int R, int h) {
    static_assert(KR >= 1 && KR <= 5, "window rate");
    int s = KR*(8*R + 4*h + 4) - 32;
    s += s < 0 ? 128*KR : 0;
    k5win r;
    r.w0 = s >> 5; r.sh = s & 31; r.w1 = r.w0 + 1 == 4*KR ? 0 : r.w0 + 1;
    return r;
}
template <int KR>
static __device__ __forceinline__ uint32_t levels4_kr(const uint32_t f) {
    const uint32_t s0 = (f >> (3*KR)) & 0xFFFFu;
    const uint32_t s1 = (f >> (2*KR)) & 0xFFFFu;
    const uint32_t s2 = (f >>    KR ) & 0xFFFFu;
    const uint32_t s3 =  f            & 0xFFFFu;
    constexpr uint32_t C = 128u - 32768u;
    const uint32_t r0 = __dp4a(s0*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r1 = __dp4a(s1*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r2 = __dp4a(s2*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r3 = __dp4a(s3*0x83DCD12Du, 0x40404040u, C);
    const uint32_t lo = __byte_perm(r0, r1, 0x0051);
    const uint32_t hi = __byte_perm(r2, r3, 0x0051);
    return __byte_perm(lo, hi, 0x5410);
}
template <int KR>
static __device__ __forceinline__ void a_regs_smem_kr(const uint32_t * __restrict__ tw, int j, int t, const k5win & wa, const k5win & wb,
                                                      uint32_t (&a)[4]) {
    const uint32_t * ta = tw + (4*j + (t >> 1))*(4*KR);
    const uint32_t * tb = ta + 2*(4*KR);
    a[0] = levels4_kr<KR>(__funnelshift_l(ta[wa.w1], ta[wa.w0], wa.sh));
    a[1] = levels4_kr<KR>(__funnelshift_l(ta[wb.w1], ta[wb.w0], wb.sh));
    a[2] = levels4_kr<KR>(__funnelshift_l(tb[wa.w1], tb[wa.w0], wa.sh));
    a[3] = levels4_kr<KR>(__funnelshift_l(tb[wb.w1], tb[wb.w0], wb.sh));
}

// per-layout window (per lane constant) and A-register decode
template <int LAY>
static __device__ __forceinline__ k5win lay_win(int R, int h) {
    if constexpr (lay_rate<LAY>() == 5) { return k5_win(R, h); }
    else if constexpr ((lay_rate<LAY>() == 4 || lay_rate<LAY>() == 6)) { return k5win{0, 0, 0}; }
    else { return kr_win<lay_rate<LAY>()>(R, h); }
}
template <int LAY>
static __device__ __forceinline__ void a_regs_lay(const uint32_t * __restrict__ tw, int j, int g, int t, const k5win & wa, const k5win & wb,
                                                  uint32_t (&a)[4]) {
    if constexpr (lay_rate<LAY>() == 6) { a_regs_smem_k6(tw, j, t, k6_win(g, t & 1), k6_win(g + 8, t & 1), a); }
    else if constexpr (lay_rate<LAY>() == 5) { a_regs_smem_k5(tw, j, t, wa, wb, a); }
    else if constexpr (lay_rate<LAY>() == 4) { a_regs_smem(tw, j, g, t, a); }
    else { a_regs_smem_kr<lay_rate<LAY>()>(tw, j, t, wa, wb, a); }
}

// Whole-row rate-2 decoder is defined with the register-streaming kernels below.
static __device__ __forceinline__ void levels8_k2(uint32_t f, uint32_t & lo, uint32_t & hi);

// ---------------------------------------------------------------------------------------------
// GEMM (N > 8). Y[n*ldy + m] = sum_k W[m][k] x[n][k]; M % 16 == 0 (slabs past M are zero-filled and not stored),
// K % 1024 == 0, any N.

constexpr int GEMM_BM      = 128;
constexpr int GEMM_SLABS   = GEMM_BM/16;
constexpr int GEMM_THREADS = 256;
constexpr int X_STRIDE     = RQ4_CHUNK + 16;
constexpr int GSC_BYTES    = 64;   // per slab and stage: 16 rows x one fp16 pair

template <int BN, int LAY = LAY_SB, int M_SLABS = GEMM_SLABS, bool PREDECODED = false, bool X8 = false, bool XOR_X = false>
constexpr __host__ __device__ int gemm_stage_bytes() {
    return M_SLABS*((PREDECODED ? 2048 : ch_bytes<LAY>()) + GSC_BYTES) + BN*(XOR_X ? RQ4_CHUNK : X8 ? RQ4_CHUNK + 8 : PREDECODED ? RQ4_CHUNK : X_STRIDE) + BN*4;
}

template <int BN, int STAGES, int NTB, int LAY, int XSH, bool ID, int N_WARPS = 1, int M_SLABS = GEMM_SLABS, bool PREDECODED = false, int MINB = 1, bool STREAM_A = false, bool J_MAJOR = false, bool X8 = false, bool WHOLE = false, bool SPLIT = false, bool SWAP8 = false, bool FIXED_LOAD = false, bool XOR_X = false>
static __global__ void __launch_bounds__(M_SLABS*N_WARPS*32, MINB) gemm(
        const uint8_t * __restrict__ W, const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K, const idmap idm, const int8_t * __restrict__ decoded_weights, int split_k) {
    constexpr int GEMM_SLABS = M_SLABS;
    constexpr int GEMM_THREADS = M_SLABS*N_WARPS*32;
    constexpr int NTI = BN/(8*N_WARPS);
    static_assert(BN % (8*N_WARPS) == 0 && 4 % N_WARPS == 0, "warp tile");
    static_assert(NTI % NTB == 0, "NTB");
    static_assert(!PREDECODED || (!ID && N_WARPS == 1), "predecoded dense tile");
    static_assert(!STREAM_A || (!PREDECODED && N_WARPS > 1), "stream decoded fragments");
    static_assert(!J_MAJOR || (!PREDECODED && N_WARPS == 1), "fragment-first decode");
    static_assert(!X8 || J_MAJOR, "eight-byte activation fragments");
    static_assert(!XOR_X || (J_MAJOR && !X8 && !WHOLE && !PREDECODED && !ID), "compact XOR activation scratch");
    static_assert(!WHOLE || (X8 && lay_rate<LAY>() == 2), "whole-row dense decode");
    static_assert(!SPLIT || !ID, "dense split K");
    static_assert(!SWAP8 || (J_MAJOR && !X8 && !WHOLE && !ID), "swapped dense activation halves");
    constexpr int SB  = gemm_stage_bytes<BN, LAY, M_SLABS, PREDECODED, X8, XOR_X>();
    constexpr int CHB = PREDECODED ? 2048 : ch_bytes<LAY>();
    constexpr int XS = XOR_X ? RQ4_CHUNK : X8 ? RQ4_CHUNK + 8 : PREDECODED ? RQ4_CHUNK : X_STRIDE;
    constexpr int CHP = CHB/16;   // 16-B pieces per slab chunk
    constexpr int WSB = GEMM_SLABS*(CHB + GSC_BYTES);

    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int srow[ID ? BN : 1];
    __shared__ int scol[ID ? BN : 1];

    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int mw = warp/N_WARPS, nw = warp % N_WARPS;
    const int slab0 = blockIdx.x*GEMM_SLABS;
    const int nslab = M/16;
    int n0 = blockIdx.y*BN;
    const int nk = K/RQ4_CHUNK;
    const int kc_begin = SPLIT ? blockIdx.z*((nk + split_k - 1)/split_k) : 0;
    const int kc_end = SPLIT ? min(nk, kc_begin + (nk + split_k - 1)/split_k) : nk;
    if constexpr (SPLIT) Y += (int64_t) blockIdx.z*ldy*N;
    const int XSN = K >> (7 + XSH);
    const int64_t sbytes = slab_bytes_l<LAY>(K);

    if constexpr (ID) {
        if ((int) blockIdx.y >= idm.hdr[1]) {
            return;
        }
        const int a  = idm.tiles[2*blockIdx.y];
        const int o  = idm.off[a];
        const int nt = idm.tiles[2*blockIdx.y + 1];
        W += (int64_t) idm.exps[a]*idm.nb02;
        N  = idm.off[a + 1] - o - nt;   // tokens of this tile = min(BN, N)
        n0 = 0;
        for (int i = tid; i < BN; i += GEMM_THREADS) {
            srow[i] = i < N ? idm.rows[o + nt + i] : 0;
            scol[i] = i < N ? idm.cols[o + nt + i] : 0;
        }
        __syncthreads();
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) { return srow[c]; } else { return n0 + c; } };

    auto load_stage = [&](int kc) {
        uint8_t * st = smem + (kc % STAGES)*SB;
        const int64_t coff = chunk_off<LAY>(kc);
        // weights: 8 slabs x 64 x 16 B of chunk tiles
        auto load_weight = [&](int i) {
            const int  s  = i / CHP, p = i % CHP;
            const bool ok = slab0 + s < nslab;
            const uint8_t * src;
            if constexpr (PREDECODED) {
                src = (const uint8_t *) decoded_weights + (ok ? (int64_t) (slab0 + s)*K*16 + (int64_t) kc*2048 + p*16 : 0);
            } else {
                src = W + (ok ? (int64_t) (slab0 + s)*sbytes + coff + p*16 : 0);
            }
            cp_async16(st + s*CHB + p*16, src, ok);
        };
        if constexpr (FIXED_LOAD) {
#pragma unroll
            for (int r = 0; r < (GEMM_SLABS*CHP + GEMM_THREADS - 1)/GEMM_THREADS; ++r) {
                const int i = tid + r*GEMM_THREADS;
                if (GEMM_SLABS*CHP % GEMM_THREADS == 0 || i < GEMM_SLABS*CHP) load_weight(i);
            }
        } else {
            for (int i = tid; i < GEMM_SLABS*CHP; i += GEMM_THREADS) load_weight(i);
        }
        if constexpr (LAY != LAY_SB) {
            // scales: per slab the unit's d[16] (32 B)
            if (tid < GEMM_SLABS*8) {
                const int  s  = tid >> 3, w = tid & 7;
                const bool ok = slab0 + s < nslab;
                const uint8_t * src = W + (ok ? (int64_t) (slab0 + s)*sbytes + sc_off<LAY>(kc) + w*4 : 0);
                cp_async4(st + GEMM_SLABS*CHB + s*GSC_BYTES + w*4, src, ok);
            }
        } else {
            // scales: per slab the 16 rows' pair words d[r*8 + (cin & 6)], d[r*8 + (cin | 1)] (64 B instead of the 256 B header)
            if (tid < GEMM_SLABS*16) {
                const int  s  = tid >> 4, r = tid & 15;
                const bool ok = slab0 + s < nslab;
                const uint8_t * src = W + (ok ? (int64_t) (slab0 + s)*sbytes + (int64_t) (kc >> 3)*SB_BYTES + r*16 + (kc & 6)*2 : 0);
                cp_async4(st + GEMM_SLABS*CHB + s*GSC_BYTES + r*4, src, ok);
            }
        }
        uint8_t * sx  = st + WSB;
        float   * sxd = (float *) (sx + BN*XS);
        auto load_activation = [&](int i) {
            constexpr int PIECES = X8 ? 16 : 8;
            const int c = i/PIECES, p = i % PIECES;
            const bool ok = n0 + c < N;
            const int8_t * src = Xq + (ok ? xrow(c) : 0)*K + kc*RQ4_CHUNK + p*(X8 ? 8 : 16);
            if constexpr (X8) {
                cp_async8(sx + c*XS + p*8, src, ok);
            } else {
                const int xp = (PREDECODED || XOR_X) ? p ^ (c & 7) : p;
                cp_async16(sx + c*XS + xp*16, src, ok);
            }
        };
        constexpr int XPIECES = BN*(X8 ? 16 : 8);
        if constexpr (FIXED_LOAD) {
#pragma unroll
            for (int r = 0; r < (XPIECES + GEMM_THREADS - 1)/GEMM_THREADS; ++r) {
                const int i = tid + r*GEMM_THREADS;
                if (XPIECES % GEMM_THREADS == 0 || i < XPIECES) load_activation(i);
            }
#pragma unroll
            for (int r = 0; r < (BN + GEMM_THREADS - 1)/GEMM_THREADS; ++r) {
                const int i = tid + r*GEMM_THREADS;
                if (BN % GEMM_THREADS == 0 || i < BN) {
                    const bool ok = n0 + i < N;
                    cp_async4(sxd + i, Xs + (ok ? xrow(i) : 0)*XSN + (kc >> XSH), ok);
                }
            }
        } else {
            for (int i = tid; i < XPIECES; i += GEMM_THREADS) load_activation(i);
            for (int i = tid; i < BN; i += GEMM_THREADS) {
                const bool ok = n0 + i < N;
                cp_async4(sxd + i, Xs + (ok ? xrow(i) : 0)*XSN + (kc >> XSH), ok);
            }
        }
    };

    const k5win k5a = lay_win<LAY>(g, t & 1), k5b = lay_win<LAY>(g + 8, t & 1);
    (void) k5a; (void) k5b;

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (kc_begin + s < kc_end) {
            load_stage(kc_begin + s);
        }
        cp_async_commit();
    }

    for (int kc = kc_begin; kc < kc_end; ++kc) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kc + STAGES - 1 < kc_end) {
            load_stage(kc + STAGES - 1);
        }
        cp_async_commit();

        const uint8_t  * st  = smem + (kc % STAGES)*SB;
        const uint32_t * tw  = (const uint32_t *) (st + mw*CHB);
        const half     * sd  = (const half *) (st + GEMM_SLABS*CHB + mw*GSC_BYTES);
        const uint8_t  * sx  = st + WSB;
        const float    * sxd = (const float *) (sx + BN*XS);

        if constexpr (J_MAJOR) {
            int p[NTI][4] = {};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t a[4];
                if constexpr (WHOLE) {
                    const k5win pa = kr_win<2>(g, 1), pb = kr_win<2>(g + 8, 1);
                    const uint32_t * tile = tw + (4*j + t)*8;
                    levels8_k2(__funnelshift_l(tile[pa.w1], tile[pa.w0], pa.sh), a[0], a[2]);
                    levels8_k2(__funnelshift_l(tile[pb.w1], tile[pb.w0], pb.sh), a[1], a[3]);
                } else {
                    a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a);
                }
#pragma unroll
                for (int nt = 0; nt < NTI; ++nt) {
                    const int c = nt*8 + g;
                    if constexpr (WHOLE) {
                        const uint8_t * x = sx + c*XS + 64*(t & 1) + 4*(t >> 1) + j*8;
                        mma_s8(p[nt], a, *(const uint32_t *) x, *(const uint32_t *) (x + 32));
                    } else {
                        const int o = t*32 + (j ^ (SWAP8 ? ((c >> 1) & 1) : 0))*8;
                        const int xo = XOR_X ? ((o/16) ^ (c & 7))*16 + (o & 15) : o;
                        const uint2 x = *(const uint2 *) (sx + c*XS + xo);
                        mma_s8(p[nt], a, x.x, x.y);
                    }
                }
            }
            const float d0 = __half2float(LAY != LAY_SB ? sd[g] : sd[2*g + (kc & 1)]);
            const float d1 = __half2float(LAY != LAY_SB ? sd[g + 8] : sd[2*(g + 8) + (kc & 1)]);
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                const float dx0 = sxd[nt*8 + 2*t];
                const float dx1 = sxd[nt*8 + 2*t + 1];
                acc[nt][0] += magic_i2f(p[nt][0]) * (d0*dx0);
                acc[nt][1] += magic_i2f(p[nt][1]) * (d0*dx1);
                acc[nt][2] += magic_i2f(p[nt][2]) * (d1*dx0);
                acc[nt][3] += magic_i2f(p[nt][3]) * (d1*dx1);
            }
        } else {
            uint32_t a[STREAM_A ? 1 : 4][4];
            if constexpr (PREDECODED) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4 v = ((const uint4 *) tw)[j*32 + lane];
                    a[j][0] = v.x;
                    a[j][1] = v.y;
                    a[j][2] = v.z;
                    a[j][3] = v.w;
                }
            } else if constexpr (N_WARPS == 1) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a[j]);
                }
            } else {
                // Each N warp decodes part of A. All N warps reuse the same exact s8 fragments.
                uint32_t * decoded = (uint32_t *) (smem + STAGES*SB) + mw*4*4*32;
#pragma unroll
                for (int j = 0; j < 4/N_WARPS; ++j) {
                    const int aj = j*N_WARPS + nw;
                    uint32_t v[4];
                    a_regs_lay<LAY>(tw, aj, g, t, k5a, k5b, v);
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        decoded[(aj*4 + r)*32 + lane] = v[r];
                    }
                }
                __syncthreads();
                if constexpr (!STREAM_A) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            a[j][r] = decoded[(j*4 + r)*32 + lane];
                        }
                    }
                }
            }
            const float d0 = __half2float(LAY != LAY_SB ? sd[g]     : sd[2*g       + (kc & 1)]);
            const float d1 = __half2float(LAY != LAY_SB ? sd[g + 8] : sd[2*(g + 8) + (kc & 1)]);

#pragma unroll
            for (int nb = 0; nb < NTI; nb += NTB) {
                uint4 xf[NTB][2];
                float dx[NTB][2];
#pragma unroll
                for (int u = 0; u < NTB; ++u) {
                    const int c = (nw*NTI + nb + u)*8 + g;
                    const int p0 = PREDECODED ? (2*t) ^ (c & 7) : 2*t;
                    const int p1 = PREDECODED ? (2*t + 1) ^ (c & 7) : 2*t + 1;
                    xf[u][0] = *(const uint4 *) (sx + c*XS + p0*16);
                    xf[u][1] = *(const uint4 *) (sx + c*XS + p1*16);
                    dx[u][0] = sxd[(nw*NTI + nb + u)*8 + 2*t + 0];
                    dx[u][1] = sxd[(nw*NTI + nb + u)*8 + 2*t + 1];
                }
                int p[NTB][4];
#pragma unroll
                for (int u = 0; u < NTB; ++u)
#pragma unroll
                    for (int i = 0; i < 4; ++i) p[u][i] = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (STREAM_A) {
                        const uint32_t * decoded = (const uint32_t *) (smem + STAGES*SB) + mw*4*4*32;
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            a[0][r] = decoded[(j*4 + r)*32 + lane];
                        }
                    }
#pragma unroll
                    for (int u = 0; u < NTB; ++u) {
                        const uint4 & xv = xf[u][j >> 1];
                        mma_s8(p[u], a[STREAM_A ? 0 : j], (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
                    }
                }
#pragma unroll
                for (int u = 0; u < NTB; ++u) {
                    acc[nb + u][0] += magic_i2f(p[u][0]) * (d0*dx[u][0]);
                    acc[nb + u][1] += magic_i2f(p[u][1]) * (d0*dx[u][1]);
                    acc[nb + u][2] += magic_i2f(p[u][2]) * (d1*dx[u][0]);
                    acc[nb + u][3] += magic_i2f(p[u][3]) * (d1*dx[u][1]);
                }
            }
        }
    }
    cp_async_wait<0>();

    if (slab0 + mw >= nslab) {
        return;
    }
    const int r = (slab0 + mw)*16 + g;
    auto ycol = [&](int c) -> int64_t { if constexpr (ID) { return scol[c]; } else { return n0 + c; } };
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = (nw*NTI + nt)*8 + 2*t;
        if (n0 + c < N) {
            const int64_t y = ycol(c);
            Y[y*ldy + r]           = acc[nt][0];
            Y[y*ldy + r + 8]       = acc[nt][2];
        }
        if (n0 + c + 1 < N) {
            const int64_t y = ycol(c + 1);
            Y[y*ldy + r]     = acc[nt][1];
            Y[y*ldy + r + 8] = acc[nt][3];
        }
    }
}

// ID = true: grid.y = max_tiles (an upper bound of the idmap's token tiles; blocks past hdr[1] exit), N unused
template <int BN, int STAGES = 3, int NTB = 4, int LAY = LAY_SB, int XSH = 1, bool ID = false, int N_WARPS = 1, int M_SLABS = GEMM_SLABS, bool PREDECODED = false, int MINB = 1, bool STREAM_A = false, bool J_MAJOR = false, bool X8 = false, bool WHOLE = false, bool SPLIT = false, bool SWAP8 = false, bool FIXED_LOAD = false, bool XOR_X = false>
static void launch_gemm(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y, int64_t ldy,
                        int M, int N, int K, cudaStream_t stream, const idmap & idm = idmap{}, int max_tiles = 0, const int8_t * decoded_weights = nullptr, int split_k = 1) {
    constexpr int decoded = N_WARPS > 1 ? M_SLABS*4*4*32*4 : 0;
    constexpr int smem = STAGES*gemm_stage_bytes<BN, LAY, M_SLABS, PREDECODED, X8, XOR_X>() + decoded;
    static bool attr_set[64] = {};
    int dev = 0;
    CUDA_CHECK_AI4K(cudaGetDevice(&dev));
    if (!attr_set[dev & 63]) {
        CUDA_CHECK_AI4K(cudaFuncSetAttribute(gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, XOR_X>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev & 63] = true;
    }
    const dim3 grid((M/16 + M_SLABS - 1)/M_SLABS, ID ? max_tiles : (N + BN - 1)/BN, SPLIT ? split_k : 1);
    static const bool report = getenv("GGML_CUDA_AI23_RESOURCE") && atoi(getenv("GGML_CUDA_AI23_RESOURCE"));
    static bool reported[64] = {};
    if (report && !reported[dev & 63]) {
        cudaFuncAttributes attr;
        int blocks = 0, reserved = 0, sm_shared = 0;
        CUDA_CHECK_AI4K(cudaFuncGetAttributes(&attr, gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, XOR_X>));
        CUDA_CHECK_AI4K(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, XOR_X>, M_SLABS*N_WARPS*32, smem));
        CUDA_CHECK_AI4K(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
        CUDA_CHECK_AI4K(cudaDeviceGetAttribute(&sm_shared, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
        fprintf(stderr, "AI23 resource BN=%d ST=%d LAY=%d NW=%d MS=%d MINB=%d J=%d X8=%d WHOLE=%d SPLIT=%d SWAP8=%d FIXED_LOAD=%d XOR_X=%d REG=%d STATIC_SHARED=%zu DYNAMIC_SHARED=%d RESERVED_SHARED=%d SM_SHARED=%d LOCAL=%zu ACTIVE_BLOCKS=%d GRID=%u,%u,%u\n",
                BN, STAGES, LAY, N_WARPS, M_SLABS, MINB, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, XOR_X, attr.numRegs, attr.sharedSizeBytes, smem, reserved, sm_shared, attr.localSizeBytes, blocks, grid.x, grid.y, grid.z);
        reported[dev & 63] = true;
    }
    gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, XOR_X><<<grid, M_SLABS*N_WARPS*32, smem, stream>>>(W, Xq, Xs, Y, ldy, M, N, K, idm, decoded_weights, split_k);
}

// Split-K partials keep float accumulation; only the final sum order changes.
static __global__ void split_k_reduce(const float * partial, float * Y, int64_t size, int splits) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = partial[i];
        for (int s = 1; s < splits; ++s) sum += partial[(int64_t) s*size + i];
        Y[i] = sum;
    }
}

// ---------------------------------------------------------------------------------------------
// Decode / verify (N <= 8). Block = one slab (16 rows), MMV_WARPS warps take chunks warp, warp + MMV_WARPS, ...
// Per warp a cp.async ring of STAGES stages; a stage holds the chunk's 16 tiles (1 KB), the 16 rows' scale pair words
// (64 B), the 8 tokens' activation scales (32 B) and, with XREG = false, the activation tile in the lane's own fragment
// order (lane l: token l/4, bytes (l%4)*32..+31 of the chunk -> [h][lane][16 B], 1 KB). With XREG = true (default)
// each lane reads its own 32 activation bytes with ld.global.nc one chunk ahead into registers instead: the stage
// shrinks to 1120 B, so twice as many blocks fit per SM (fewer partial waves at M = 5120 / 6144).

constexpr int MMV_WARPS = 4;
// GGML_CUDA_AI4_MMV_WARPS (2, 4, 8) and GGML_CUDA_AI4_MMV_MINB (2, 4, 6, 8) select other
// instances of the stages-2 register kernel. Unset stays at 4 warps and 6 min blocks.
// 8 warps with min blocks 8 does not fit sm_86 launch_bounds (48 warps / 1536 threads per SM)
// and is not instantiated. Stages 3/4 and the shared-memory activation path stay on this default.

template <bool XREG, int LAY = LAY_SB>
constexpr __host__ __device__ int mmv_stage_bytes() { return ch_bytes<LAY>() + 64 + 32 + (XREG ? 0 : 1024); }

template <int STAGES, bool XREG, int LAY = LAY_SB, int WARPS = MMV_WARPS>
constexpr __host__ __device__ int mmv_smem_bytes() { return WARPS*STAGES*mmv_stage_bytes<XREG, LAY>(); }

template <int STAGES, int MINB, bool XREG, int LAY, int XSH, bool ID, int WARPS = MMV_WARPS>
static __global__ void __launch_bounds__(WARPS*32, MINB) mmv(
        const uint8_t * __restrict__ W, const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int K, const idmap idm) {
    constexpr int MMV_WARPS = WARPS;   // body below is written against this name
    static_assert(WARPS >= 2, "ai4 mmv needs at least 2 warps");
    static_assert(WARPS * 32 <= 1024, "ai4 mmv block");
    static_assert(WARPS * MINB <= 48, "ai4 mmv launch_bounds warps");
    static_assert(WARPS * 32 * MINB <= 1536, "ai4 mmv launch_bounds threads");
    constexpr int ST  = mmv_stage_bytes<XREG, LAY>();
    constexpr int CHB = ch_bytes<LAY>();
    constexpr int OSC = CHB + (XREG ? 0 : 1024);   // offset of the scale words in a stage
    extern __shared__ __align__(16) uint8_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int slab = blockIdx.x;
    const int nk = K/RQ4_CHUNK;
    const int XSN = K >> (7 + XSH);
    const int * prow = nullptr;
    const int * pcol = nullptr;
    if constexpr (ID) {
        // grid.y = token tile of MMV_ID_BN pairs (ids_helper with bn = MMV_ID_BN); the tile's pairs are the columns. With
        // distinct expert ids per token (top-k routing) every expert has <= n_tok <= 8 pairs, so tile = active expert.
        // Duplicate ids give an expert more than 8 pairs; its extra tiles are further blocks instead of unwritten columns.
        if ((int) blockIdx.y >= idm.hdr[1]) {
            return;
        }
        const int a  = idm.tiles[2*blockIdx.y];
        const int o  = idm.off[a] + idm.tiles[2*blockIdx.y + 1];
        N    = min(MMV_ID_BN, idm.off[a + 1] - o);
        W   += (int64_t) idm.exps[a]*idm.nb02;
        prow = idm.rows + o;
        pcol = idm.cols + o;
    }
    auto xrow = [&](int c) -> int64_t { if constexpr (ID) { return prow[c]; } else { return c; } };
    const uint8_t * ws = W + (int64_t) slab*slab_bytes_l<LAY>(K);
    const k5win k5a = lay_win<LAY>(g, t & 1), k5b = lay_win<LAY>(g + 8, t & 1);
    (void) k5a; (void) k5b;
    uint8_t * ring = smem + warp*STAGES*ST;

    const bool has_n = g < N;
    const int8_t * xr = Xq + (has_n ? xrow(g) : 0)*K + t*32;
    // activation scale pointer of the token this lane loads (lanes 16..23 load tokens 0..7)
    const float * xsr = Xs + ((lane >= 16 && lane < 24 && lane - 16 < N) ? xrow(lane - 16) : 0)*XSN;

    const int nmy = (nk - warp + MMV_WARPS - 1)/MMV_WARPS;   // chunks of this warp (unequal when nk % 4 != 0, e.g. AI4_U K = 640)

    auto load = [&](int i) {
        const int kc = warp + i*MMV_WARPS;
        uint8_t * st = ring + (i % STAGES)*ST;
        const uint8_t * tiles = ws + chunk_off<LAY>(kc);
        // CHB/16 pieces of 16 B: 32 (k = 2), 48 (k = 3), 64 (k = 4), 80 (k = 5)
        constexpr int CHP = CHB/16;
#pragma unroll
        for (int r = 0; r < (CHP + 31)/32; ++r) {
            if (CHP % 32 == 0 || lane + 32*r < CHP) {
                cp_async16(st + r*512 + lane*16, tiles + r*512 + lane*16, true);
            }
        }
        if constexpr (!XREG) {
            const int8_t * xc = xr + (int64_t) kc*RQ4_CHUNK;
            cp_async16(st + CHB + lane*16,       has_n ? (const void *) xc        : (const void *) Xq, has_n);
            cp_async16(st + CHB + 512 + lane*16, has_n ? (const void *) (xc + 16) : (const void *) Xq, has_n);
        }
        if (lane < 16) {
            if constexpr (LAY != LAY_SB) {
                // the unit's d[16]: 8 words
                if (lane < 8) {
                    cp_async4(st + OSC + lane*4, ws + sc_off<LAY>(kc) + lane*4, true);
                }
            } else {
                // row lane's scales of chunks (kc & ~1), (kc | 1): d[lane*8 + (kc & 6)]
                cp_async4(st + OSC + lane*4, ws + (int64_t) (kc >> 3)*SB_BYTES + lane*16 + (kc & 6)*2, true);
            }
        } else if (lane < 24) {
            const int n = lane - 16;
            const bool ok = n < N;
            cp_async4(st + OSC + 64 + n*4, ok ? (const void *) (xsr + (kc >> XSH)) : (const void *) Xs, ok);
        }
    };
    auto ldx = [&](int i, uint4 & v0, uint4 & v1) {
        if (has_n && i < nmy) {
            const uint4 * src = (const uint4 *) (xr + (int64_t) (warp + i*MMV_WARPS)*RQ4_CHUNK);
            v0 = __ldg(src);
            v1 = __ldg(src + 1);
        } else {
            v0 = make_uint4(0, 0, 0, 0);
            v1 = v0;
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nmy) {
            load(s);
        }
        cp_async_commit();
    }
    uint4 nx0, nx1;
    if constexpr (XREG) {
        ldx(0, nx0, nx1);
    }

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < nmy; ++i) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        if (i + STAGES - 1 < nmy) {
            load(i + STAGES - 1);
        }
        cp_async_commit();

        const int kc = warp + i*MMV_WARPS;
        const uint8_t * st = ring + (i % STAGES)*ST;
        const uint32_t * tw = (const uint32_t *) st;
        uint4 x0, x1;
        if constexpr (XREG) {
            x0 = nx0; x1 = nx1;
            ldx(i + 1, nx0, nx1);
        } else {
            x0 = *(const uint4 *) (st + CHB + lane*16);
            x1 = *(const uint4 *) (st + CHB + 512 + lane*16);
        }
        const half * sdw = (const half *) (st + OSC);
        const float * sdx = (const float *) (st + OSC + 64);
        const float d0  = __half2float(LAY != LAY_SB ? sdw[g]     : sdw[2*g       + (kc & 1)]);
        const float d1  = __half2float(LAY != LAY_SB ? sdw[g + 8] : sdw[2*(g + 8) + (kc & 1)]);
        const float dx0 = sdx[2*t], dx1 = sdx[2*t + 1];

        // two independent IMMA chains (j = 0, 1 and j = 2, 3), summed exactly in int32 (|sum| <= 128*128*128)
        int p[4] = { 0, 0, 0, 0 };
        int q[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            a_regs_lay<LAY>(tw, j, g, t, k5a, k5b, a);
            const uint4 & xv = j < 2 ? x0 : x1;
            if (j < 2) {
                mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
            } else {
                mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y);
            }
        }
        acc[0] += magic_i2f(p[0] + q[0]) * (d0*dx0);
        acc[1] += magic_i2f(p[1] + q[1]) * (d0*dx1);
        acc[2] += magic_i2f(p[2] + q[2]) * (d1*dx0);
        acc[3] += magic_i2f(p[3] + q[3]) * (d1*dx1);
    }
    cp_async_wait<0>();

    __shared__ float red[MMV_WARPS - 1][4][32];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[warp - 1][i][lane] = acc[i];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < MMV_WARPS - 1; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += red[w][i][lane];
            }
        }
        const int r = slab*16 + g;
        const int c = 2*t;
        auto ycol = [&](int cc) -> int64_t { if constexpr (ID) { return pcol[cc]; } else { return cc; } };
        if (c < N) {
            const int64_t y = ycol(c);
            Y[y*ldy + r]           = acc[0];
            Y[y*ldy + r + 8]       = acc[2];
        }
        if (c + 1 < N) {
            const int64_t y = ycol(c + 1);
            Y[y*ldy + r]     = acc[1];
            Y[y*ldy + r + 8] = acc[3];
        }
    }
    (void) M;
}

// ID = true: grid.y = max_experts = an upper bound of the token tiles (ids_max_tiles with bn = MMV_ID_BN; blocks past
// hdr[1] exit); idm must come from ids_helper with bn = MMV_ID_BN; N unused
template <int STAGES = 3, int MINB = 6, bool XREG = true, int LAY = LAY_SB, int XSH = 1, bool ID = false, int WARPS = MMV_WARPS>
static void launch_mmv(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y, int64_t ldy,
                       int M, int N, int K, cudaStream_t stream, const idmap & idm = idmap{}, int max_experts = 1) {
    constexpr int smem = mmv_smem_bytes<STAGES, XREG, LAY, WARPS>();
    static bool attr_set[64] = {};
    int dev = 0;
    CUDA_CHECK_AI4K(cudaGetDevice(&dev));
    if (!attr_set[dev & 63]) {
        CUDA_CHECK_AI4K(cudaFuncSetAttribute(mmv<STAGES, MINB, XREG, LAY, XSH, ID, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev & 63] = true;
    }
    const dim3 grid(M/16, ID ? max_experts : 1);
    mmv<STAGES, MINB, XREG, LAY, XSH, ID, WARPS><<<grid, WARPS*32, smem, stream>>>(W, Xq, Xs, Y, ldy, M, N, K, idm);
}

// ---------------------------------------------------------------------------------------------
// mmvr (N <= 8, rates 2 and 3, GGML_CUDA_AI23_MMVR): the mmv kernel with the per-chunk overhead taken out of the loop.
//   - the loop is unrolled by STAGES, so every shared-memory stage offset is an immediate, and the global tile / scale /
//     activation pointers advance by compile-time strides (no per-chunk 64-bit offset math, no register copies of the
//     activation double buffer, no zero fill of out-of-range activations: lanes past N read token 0 and are not stored);
//   - KT > 0 is the row length K at compile time (0 = runtime K);
//   - rate 2 (PERM): lane (g, t) decodes whole tile rows: IMMA j uses tile 4j + t, rows g and g + 8, columns 0..3 (a[0], a[1])
//     and 4..7 (a[2], a[3]). One 32-bit window per row holds all 8 states (shifts 14, 12, .., 0), so 8 weights cost
//     3 SHF + 8 LOP3/PRMT extracts + 8 IMAD + 8 DP4A + 6 PRMT + 1 SHF + 2 LDS instead of 2 x (4 + 3 + 4 + 4 + 3 + 1 + 2).
//     The activations follow: B register h of IMMA j for lane t' is word 2j + (t' >> 1) of the fragment block of lane
//     2(t' & 1) + h (8 four-byte loads per chunk instead of 2 sixteen-byte ones).
//   - rate 3 keeps the mmv fragment mapping (two 4-symbol windows per row).
// Span layouts need (8/WARPS) | STAGES (the span changes every 8/WARPS chunks of a warp).

static __device__ __forceinline__ void levels8_k2(const uint32_t f, uint32_t & lo, uint32_t & hi) {
    const uint32_t v = f >> 2, w = f >> 4, x = f >> 6;
    const uint32_t s0 = __byte_perm(x, 0u, 0x4421), s1 = __byte_perm(w, 0u, 0x4421);
    const uint32_t s2 = __byte_perm(v, 0u, 0x4421), s3 = __byte_perm(f, 0u, 0x4421);
    const uint32_t s4 = x & 0xFFFFu, s5 = w & 0xFFFFu, s6 = v & 0xFFFFu, s7 = f & 0xFFFFu;
    constexpr uint32_t C = 128u - 32768u;
    const uint32_t r0 = __dp4a(s0*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r1 = __dp4a(s1*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r2 = __dp4a(s2*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r3 = __dp4a(s3*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r4 = __dp4a(s4*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r5 = __dp4a(s5*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r6 = __dp4a(s6*0x83DCD12Du, 0x40404040u, C);
    const uint32_t r7 = __dp4a(s7*0x83DCD12Du, 0x40404040u, C);
    lo = __byte_perm(__byte_perm(r0, r1, 0x0051), __byte_perm(r2, r3, 0x0051), 0x5410);
    hi = __byte_perm(__byte_perm(r4, r5, 0x0051), __byte_perm(r6, r7, 0x0051), 0x5410);
}

template <int XSH, bool PERM>
static __device__ __forceinline__ void ai23_quant_frag(const float * x, int chunk_half, uint32_t (&xv)[8], float & d) {
    const int lane = threadIdx.x & 31;
    const int l = lane & (XSH ? 31 : 15);
    const float4 u0 = ((const float4 *) x)[2*l];
    const float4 u1 = ((const float4 *) x)[2*l + 1];
    float v[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    const uint32_t sw = rq4::d_sign_bits[l >> 2] >> ((l & 3)*8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1) v[e] = -v[e];
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
    for (int m = 1; m < (XSH ? 32 : 16); m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffff, v[e], m);
            v[e] = (l & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= XSH ? 1.0f/16.0f : 0.08838834764831845f;
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = XSH ? 16 : 8; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    }
    d = amax/127.0f;
    const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int qi = __float2int_rn(v[e]*id);
        w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3)*8);
    }
    const int t = lane & 3;
    const int half = XSH ? chunk_half*16 : 0;
    if constexpr (PERM) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xv[j] = __shfl_sync(0xffffffff, w[0], half + 4*j + t);
            xv[j + 4] = __shfl_sync(0xffffffff, w[1], half + 4*j + t);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int source = half + 2*i + (t >> 1);
            const uint32_t lo = __shfl_sync(0xffffffff, w[0], source);
            const uint32_t hi = __shfl_sync(0xffffffff, w[1], source);
            xv[i] = (t & 1) ? hi : lo;
        }
    }
}

template <int LAY, int WARPS>
constexpr __host__ __device__ int mmvr_period() { return lay_span<LAY>() ? 8/WARPS : 1; }

// byte offset of a warp's c-th chunk (c counted from an aligned base, c % period == 0 at the base) from its first chunk
template <int LAY, int WARPS>
constexpr __host__ __device__ int64_t mmvr_toff(int c) {
    if constexpr (lay_span<LAY>()) {
        constexpr int P = 8/WARPS;
        return (int64_t) (c/P)*span_bytes<LAY>() + (c % P)*WARPS*ch_bytes<LAY>();
    } else {
        return (int64_t) c*WARPS*u_bytes<LAY>();
    }
}
template <int LAY, int WARPS>
constexpr __host__ __device__ int64_t mmvr_soff(int c) {
    if constexpr (lay_span<LAY>()) {
        return (int64_t) (c/(8/WARPS))*span_bytes<LAY>();
    } else {
        return (int64_t) c*WARPS*u_bytes<LAY>();
    }
}

template <int LAY, int XSH, bool ID, int KT, int STAGES = 2, int WARPS = 4, int MINB = 6, bool DIRECT = false, bool FUSED_X = false, int SPLITS = 1, int LOOP_UNROLL = 1>
static __global__ void __launch_bounds__(WARPS*32, MINB) mmvr(
        const uint8_t * __restrict__ W, const int8_t * __restrict__ Xq, const float * __restrict__ Xs,
        float * __restrict__ Y, int64_t ldy, int M, int N, int Kr, const idmap idm) {
    constexpr int KR = lay_rate<LAY>();
    static_assert(SPLITS == 1 || (!ID && WARPS == 4 && (KT == 5120 || KT == 17408)), "small split scope");
    static_assert(!FUSED_X || DIRECT, "fused activation needs direct ids");
    static_assert(KR == 2 || KR == 3, "mmvr: rates 2 and 3");
    static_assert(STAGES % 2 == 0, "mmvr: even stages (activation double buffer)");
    static_assert(WARPS >= 2 && (WARPS & (WARPS - 1)) == 0 && WARPS <= 16, "mmvr warps");
    static_assert(!lay_span<LAY>() || WARPS <= 8, "mmvr: span warp limit");
    static_assert(STAGES % mmvr_period<LAY, WARPS>() == 0, "mmvr: span period must divide the unroll");
    constexpr bool PERM = KR == 2;
    constexpr int  CHB  = ch_bytes<LAY>();
    constexpr int  ST   = mmv_stage_bytes<true, LAY>();
    constexpr int  OSC  = CHB;
    constexpr int  TW   = 4*KR;                 // words per tile
    const int K = KT > 0 ? KT : Kr;
    extern __shared__ __align__(16) uint8_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int slab = blockIdx.x;
    const int nk = K/RQ4_CHUNK;
    const int XSN = K >> (7 + XSH);
    const int * prow = nullptr;
    const int * pcol = nullptr;
    int direct_row = 0;
    if constexpr (DIRECT) {
        static_assert(ID, "mmvr direct ids");
        const int u = blockIdx.y;
        const int tok = blockIdx.z;
        const int e = *(const int *) (idm.direct_ids + u*idm.direct_nb0 + tok*idm.direct_nb1);
        if (e < 0 || e >= idm.direct_n_expert) {
            return;
        }
        W += (int64_t) e*idm.nb02;
        direct_row = tok*idm.direct_ne11 + (idm.direct_ne11 == 1 ? 0 : u);
        N = 1;
    } else if constexpr (ID) {
        if ((int) blockIdx.y >= idm.hdr[1]) {
            return;
        }
        const int a  = idm.tiles[2*blockIdx.y];
        const int o  = idm.off[a] + idm.tiles[2*blockIdx.y + 1];
        N    = min(MMV_ID_BN, idm.off[a + 1] - o);
        W   += (int64_t) idm.exps[a]*idm.nb02;
        prow = idm.rows + o;
        pcol = idm.cols + o;
    }
    auto xrow = [&](int c) -> int64_t {
        if constexpr (DIRECT) { return direct_row; }
        else if constexpr (ID) { return prow[c]; }
        else { return c; }
    };
    const uint8_t * ws = W + (int64_t) slab*slab_bytes_l<LAY>(K);
    // first chunk (kc = warp) of this warp
    const uint8_t * gt = ws + (lay_span<LAY>() ? U_SC + warp*CHB : (int64_t) warp*u_bytes<LAY>() + U_SC);
    const uint8_t * gs = ws + (lay_span<LAY>() ? 0 : (int64_t) warp*u_bytes<LAY>());
    uint8_t * ring = smem + warp*STAGES*ST;

    const bool has_n = g < N;
    const int8_t * xr = Xq + (has_n ? xrow(g) : xrow(0))*K + warp*RQ4_CHUNK + (PERM ? 64*(t & 1) + 4*(t >> 1) : t*32);
    const float * xf = FUSED_X ? idm.direct_x + (int64_t) direct_row*K + (warp >> XSH)*(128 << XSH) : nullptr;
    const int  ns = lane - 16;
    const bool sok = lane >= 16 && lane < 24 && ns < N;
    const float * xsr = Xs + (sok ? xrow(ns) : 0)*XSN + (warp >> XSH);
    constexpr int XS_STEP = WARPS >> XSH;       // scale index stride per chunk of a warp (WARPS even)

    // Split boundaries cover whole eight-chunk scale spans, so relative span addressing stays valid.
    const int group_begin = SPLITS > 1 ? ((KT/(RQ4_CHUNK*WARPS*2))*(int) blockIdx.z/SPLITS)*2 : 0;
    const int group_end = SPLITS > 1 ? ((KT/(RQ4_CHUNK*WARPS*2))*((int) blockIdx.z + 1)/SPLITS)*2 : 0;
    const int nmy = SPLITS > 1 ? group_end - group_begin :
                   KT > 0 && (KT/RQ4_CHUNK) % WARPS == 0 ? KT/(RQ4_CHUNK*WARPS) : (nk - warp + WARPS - 1)/WARPS;
    if constexpr (SPLITS > 1) {
        gt += mmvr_toff<LAY, WARPS>(group_begin);
        gs += mmvr_soff<LAY, WARPS>(group_begin);
        xr += (int64_t) group_begin*WARPS*RQ4_CHUNK;
        xsr += group_begin*XS_STEP;
        Y += (int64_t) blockIdx.z*M*N;
    }

    // load the chunk c chunks past the current base into stage s
    auto load = [&](const uint8_t * bt, const uint8_t * bs, const float * bx, int c, int s) {
        uint8_t * st = ring + s*ST;
        const uint8_t * tiles = bt + mmvr_toff<LAY, WARPS>(c);
        constexpr int CHP = CHB/16;
#pragma unroll
        for (int r = 0; r < (CHP + 31)/32; ++r) {
            if (CHP % 32 == 0 || lane + 32*r < CHP) {
                cp_async16(st + r*512 + lane*16, tiles + r*512 + lane*16, true);
            }
        }
        const bool weight_scale = lane < 8;
        const bool activation_scale = !FUSED_X && lane >= 16 && lane < 24;
        const void * src = weight_scale ? (const void *) (bs + mmvr_soff<LAY, WARPS>(c) + lane*4)
                                       : sok ? (const void *) (bx + c*XS_STEP) : (const void *) Xs;
        cp_async4_guard(st + OSC + lane*4, src, weight_scale || sok, weight_scale || activation_scale);
    };
    // activations of chunk c past the base: PERM = 8 words (h0[j] = b0 of IMMA j, h1[j] = b1), else 2 x 16 B
    auto ldx = [&](const int8_t * bx, int c, uint32_t (&xv)[8], float & d) {
        if constexpr (FUSED_X) {
            ai23_quant_frag<XSH, PERM>(xf + c*WARPS*RQ4_CHUNK, warp & 1, xv, d);
            return;
        }
        d = 0.0f;
        const int8_t * p = bx + (int64_t) c*WARPS*RQ4_CHUNK;
        if constexpr (PERM) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xv[j]     = __ldg((const uint32_t *) (p + 8*j));
                xv[4 + j] = __ldg((const uint32_t *) (p + 32 + 8*j));
            }
        } else {
            const uint4 v0 = __ldg((const uint4 *) p);
            const uint4 v1 = __ldg((const uint4 *) p + 1);
            xv[0] = v0.x; xv[1] = v0.y; xv[2] = v0.z; xv[3] = v0.w;
            xv[4] = v1.x; xv[5] = v1.y; xv[6] = v1.z; xv[7] = v1.w;
        }
    };

    // lane-constant row windows (rate 2: tile 4j + t, rows g and g + 8; rate 3: the mmv windows)
    const k5win wa = PERM ? kr_win<KR>(g, 1) : kr_win<KR>(g, t & 1);
    const k5win wb = PERM ? kr_win<KR>(g + 8, 1) : kr_win<KR>(g + 8, t & 1);
    const uint32_t * tl = (const uint32_t *) ring + (PERM ? t*TW : (t >> 1)*TW);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    auto compute = [&](int s, const uint32_t (&xv)[8], float act_scale) {
        const uint8_t * st = ring + s*ST;
        const uint32_t * tw = tl + s*(ST/4);
        int p[4] = { 0, 0, 0, 0 };
        int q[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            uint32_t b0, b1;
            if constexpr (PERM) {
                const uint32_t * ta = tw + 4*j*TW;
                const uint32_t f0 = __funnelshift_l(ta[wa.w1], ta[wa.w0], wa.sh);
                const uint32_t f1 = __funnelshift_l(ta[wb.w1], ta[wb.w0], wb.sh);
                levels8_k2(f0, a[0], a[2]);
                levels8_k2(f1, a[1], a[3]);
                b0 = xv[j]; b1 = xv[4 + j];
            } else {
                const uint32_t * ta = tw + 4*j*TW;
                const uint32_t * tb = ta + 2*TW;
                const uint32_t f0 = __funnelshift_l(ta[wa.w1], ta[wa.w0], wa.sh);
                const uint32_t f1 = __funnelshift_l(ta[wb.w1], ta[wb.w0], wb.sh);
                const uint32_t f2 = __funnelshift_l(tb[wa.w1], tb[wa.w0], wa.sh);
                const uint32_t f3 = __funnelshift_l(tb[wb.w1], tb[wb.w0], wb.sh);
                a[0] = levels4_kr<KR>(f0);
                a[1] = levels4_kr<KR>(f1);
                a[2] = levels4_kr<KR>(f2);
                a[3] = levels4_kr<KR>(f3);
                b0 = xv[2*j]; b1 = xv[2*j + 1];
            }
            if (j < 2) {
                mma_s8(p, a, b0, b1);
            } else {
                mma_s8(q, a, b0, b1);
            }
        }
        const half  * sdw = (const half *) (st + OSC);
        const float * sdx = (const float *) (st + OSC + 64);
        const float d0  = __half2float(sdw[g]);
        const float d1  = __half2float(sdw[g + 8]);
        const float dx0 = FUSED_X ? act_scale : sdx[2*t];
        const float dx1 = FUSED_X ? act_scale : sdx[2*t + 1];
        acc[0] += magic_i2f(p[0] + q[0]) * (d0*dx0);
        acc[1] += magic_i2f(p[1] + q[1]) * (d0*dx1);
        acc[2] += magic_i2f(p[2] + q[2]) * (d1*dx0);
        acc[3] += magic_i2f(p[3] + q[3]) * (d1*dx1);
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nmy) {
            load(gt, gs, xsr, s, s);
        }
        cp_async_commit();
    }
    uint32_t xa[8], xb[8];
    float da = 0.0f, db = 0.0f;
    if (nmy > 0) {
        ldx(xr, 0, xa, da);
    }

    // The short five-chunk down projection has a known maximum trip count even when warps have unequal tails.
    auto step = [&](int i0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            if (i0 + s < nmy) {
                cp_async_wait<STAGES - 2>();
                __syncwarp();
                if (i0 + s + STAGES - 1 < nmy) {
                    load(gt, gs, xsr, s + STAGES - 1, (s + STAGES - 1) % STAGES);
                }
                cp_async_commit();
                if (i0 + s + 1 < nmy) {
                    ldx(xr, s + 1, (s & 1) ? xa : xb, (s & 1) ? da : db);
                }
                compute(s, (s & 1) ? xb : xa, (s & 1) ? db : da);
            }
        }
        gt  += mmvr_toff<LAY, WARPS>(STAGES);
        gs  += mmvr_soff<LAY, WARPS>(STAGES);
        xsr += STAGES*XS_STEP;
        xr  += (int64_t) STAGES*WARPS*RQ4_CHUNK;
        if constexpr (FUSED_X) xf += STAGES*WARPS*RQ4_CHUNK;
    };
    if constexpr (LOOP_UNROLL > 1) {
#pragma unroll LOOP_UNROLL
        for (int i0 = 0; i0 < (SPLITS > 1 ? nmy : KT > 0 ? (KT/RQ4_CHUNK + WARPS - 1)/WARPS : nmy); i0 += STAGES) {
            step(i0);
        }
    } else {
        for (int i0 = 0; i0 < (SPLITS > 1 ? nmy : KT > 0 ? (KT/RQ4_CHUNK + WARPS - 1)/WARPS : nmy); i0 += STAGES) {
            step(i0);
        }
    }
    cp_async_wait<0>();

    __shared__ float red[WARPS - 1][4][32];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[warp - 1][i][lane] = acc[i];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < WARPS - 1; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += red[w][i][lane];
            }
        }
        const int r = slab*16 + g;
        const int c = 2*t;
        auto ycol = [&](int cc) -> int64_t {
            if constexpr (DIRECT) { return blockIdx.z*idm.direct_n_used + blockIdx.y; }
            else if constexpr (ID) { return pcol[cc]; }
            else { return cc; }
        };
        if (c < N) {
            const int64_t y = ycol(c);
            Y[y*ldy + r]     = acc[0];
            Y[y*ldy + r + 8] = acc[2];
        }
        if (c + 1 < N) {
            const int64_t y = ycol(c + 1);
            Y[y*ldy + r]     = acc[1];
            Y[y*ldy + r + 8] = acc[3];
        }
    }
    (void) M;
}

template <int LAY, int XSH, bool ID, int KT, int STAGES = 2, int WARPS = 4, int MINB = 6, bool DIRECT = false, bool FUSED_X = false, int SPLITS = 1, int LOOP_UNROLL = 1>
static void launch_mmvr(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y, int64_t ldy,
                        int M, int N, int K, cudaStream_t stream, const idmap & idm = idmap{}, int max_experts = 1) {
    constexpr int smem = WARPS*STAGES*mmv_stage_bytes<true, LAY>();
    static bool attr_set[64] = {};
    int dev = 0;
    CUDA_CHECK_AI4K(cudaGetDevice(&dev));
    if (!attr_set[dev & 63]) {
        CUDA_CHECK_AI4K(cudaFuncSetAttribute(mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, SPLITS, LOOP_UNROLL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev & 63] = true;
    }
    // Direct routing has one slot and token per block; separate dimensions avoid device-side integer division.
    const dim3 grid(M/16, DIRECT ? idm.direct_n_used : ID ? max_experts : 1,
                    DIRECT ? max_experts/idm.direct_n_used : SPLITS);
    static const bool report = getenv("GGML_CUDA_AI23_RESOURCE") && atoi(getenv("GGML_CUDA_AI23_RESOURCE"));
    static bool reported[64] = {};
    if (report && !reported[dev & 63]) {
        cudaFuncAttributes attr;
        int blocks = 0, reserved = 0, sm_shared = 0;
        CUDA_CHECK_AI4K(cudaFuncGetAttributes(&attr, mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, SPLITS, LOOP_UNROLL>));
        CUDA_CHECK_AI4K(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, SPLITS, LOOP_UNROLL>, WARPS*32, smem));
        CUDA_CHECK_AI4K(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
        CUDA_CHECK_AI4K(cudaDeviceGetAttribute(&sm_shared, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
        fprintf(stderr, "AI23 MMVR resource LAY=%d XSH=%d ID=%d KT=%d ST=%d WARPS=%d MINB=%d DIRECT=%d FUSED_X=%d SPLITS=%d UNROLL=%d REG=%d STATIC_SHARED=%zu DYNAMIC_SHARED=%d RESERVED_SHARED=%d SM_SHARED=%d LOCAL=%zu ACTIVE_BLOCKS=%d GRID=%u,%u,%u\n",
                LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, SPLITS, LOOP_UNROLL,
                attr.numRegs, attr.sharedSizeBytes, smem, reserved, sm_shared, attr.localSizeBytes, blocks, grid.x, grid.y, grid.z);
        reported[dev & 63] = true;
    }
    mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, SPLITS, LOOP_UNROLL><<<grid, WARPS*32, smem, stream>>>(W, Xq, Xs, Y, ldy, M, N, K, idm);
}

// ---------------------------------------------------------------------------------------------
// Activations for H128 tensors (AI4_U with K % 256 != 0): rotate each 128 block by (1/sqrt(128)) H128 D (D = the first 128
// sign bits), int8-quantize with one fp32 scale per (token, 128 K) at s[tok*(K/128) + block], bytes in the same fragment
// order as rq4::quantize_act inside a 128 chunk. One half-warp per (token, 128 block); 4 warps = 8 blocks per CTA.
template <bool SWAP8 = false>
static __global__ void __launch_bounds__(128) quantize_act128(
        const float * __restrict__ x, int64_t sx, int8_t * __restrict__ q, float * __restrict__ s, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int l    = lane & 15;
    const int cb   = (blockIdx.x*4 + (threadIdx.x >> 5))*2 + (lane >> 4);
    const int tok  = blockIdx.y;
    const bool live = cb*128 < K && tok < N;   // the whole warp runs the shuffles; dead half-warps only skip I/O
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (live) {
        const float * xr = x + (int64_t) tok*sx + cb*128 + l*8;
        const float4 u0 = *(const float4 *) xr;
        const float4 u1 = *(const float4 *) (xr + 4);
        v[0] = u0.x; v[1] = u0.y; v[2] = u0.z; v[3] = u0.w; v[4] = u1.x; v[5] = u1.y; v[6] = u1.z; v[7] = u1.w;
    }
    const uint32_t sw = rq4::d_sign_bits[l >> 2] >> ((l & 3)*8);
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
    for (int m = 1; m < 16; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffff, v[e], m);
            v[e] = (l & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= 0.08838834764831845f;   // 1/sqrt(128)
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    }
    if (!live) {
        return;
    }
    const float d  = amax/127.0f;
    const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
    uint32_t w[2] = { 0, 0 };
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int qi = __float2int_rn(v[e]*id);
        w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3)*8);
    }
    const int j  = l >> 2;
    const int h  = (l & 3) >> 1;
    const int t0 = (l & 1)*2;
    int8_t * qr = q + (int64_t) tok*K + cb*128 + (j ^ (SWAP8 ? ((tok >> 1) & 1) : 0))*8 + h*4;
    *(uint32_t *) (qr + (t0 + 0)*32) = w[0];
    *(uint32_t *) (qr + (t0 + 1)*32) = w[1];
    if (l == 0) {
        s[(int64_t) tok*(K/128) + cb] = d;
    }
}

template <bool SWAP8 = false>
static void launch_quantize_act128(const float * x, int64_t sx, int8_t * q, float * s, int K, int N, cudaStream_t stream) {
    const dim3 grid((K/128 + 7)/8, N);
    quantize_act128<SWAP8><<<grid, 128, 0, stream>>>(x, sx, q, s, K, N);
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT_ID routing (one CTA of 1024 threads). ids: int32 [n_used, n_tok] with byte strides ids_nb0/ids_nb1; pair
// (slot u, token t) reads Xq row t*ne11 + u % ne11 and writes Y column t*n_used + u. Output (int buffer, layout of
// ids_helper_ints): hdr[2] = { active experts, token tiles }, exps[n_expert], off[n_expert + 1], rows[n_pairs],
// cols[n_pairs], tiles[2*max_tiles] (BN-wide tiles per active expert; bn = 0 skips them). Pair order inside an expert
// is not fixed (shared atomics), but every column is computed on its own, so results do not depend on it.
constexpr int IDS_THREADS = 1024;
constexpr int IDS_MAX_EXPERT = 4*IDS_THREADS;

static __host__ __device__ inline int64_t ids_helper_ints(int n_expert, int64_t n_pairs, int max_tiles) {
    return 2 + n_expert + (n_expert + 1) + 2*n_pairs + 2*(int64_t) max_tiles;
}

static __host__ inline int ids_max_tiles(int n_expert, int64_t n_pairs, int bn) {
    return bn > 0 ? (int) (std::min<int64_t>(n_expert, n_pairs) + n_pairs/bn) : 0;
}

static __global__ void __launch_bounds__(IDS_THREADS) ids_helper(
        const char * __restrict__ ids, int64_t ids_nb0, int64_t ids_nb1, int n_used, int n_tok, int ne11,
        int n_expert, int bn, int max_tiles, int * __restrict__ out) {
    __shared__ int cnt[IDS_MAX_EXPERT];
    __shared__ int wsum[2][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t n_pairs = (int64_t) n_used*n_tok;
    int * hdr   = out;
    int * exps  = hdr + 2;
    int * off   = exps + n_expert;
    int * rows  = off + n_expert + 1;
    int * cols  = rows + n_pairs;
    int * tiles = cols + n_pairs;

    for (int e = tid; e < n_expert; e += IDS_THREADS) {
        cnt[e] = 0;
    }
    __syncthreads();
    for (int64_t p = tid; p < n_pairs; p += IDS_THREADS) {
        const int u = (int) (p % n_used), t = (int) (p / n_used);
        const int e = *(const int *) (ids + u*ids_nb0 + t*ids_nb1);
        if (e >= 0 && e < n_expert) {
            atomicAdd(&cnt[e], 1);
        }
    }
    __syncthreads();
    // block scan over the experts (4 consecutive per thread): pair offsets, active index, tile offsets
    int c[4], a[4], tl[4];
    int sc = 0, sa = 0, st = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid*4 + i;
        c[i]  = e < n_expert ? cnt[e] : 0;
        a[i]  = c[i] > 0;
        tl[i] = bn > 0 ? (c[i] + bn - 1)/bn : 0;
        sc += c[i]; sa += a[i]; st += tl[i];
    }
    // pack (active, tiles) into one int: both < 2^15 here (active <= 4096, tiles <= max_tiles < 65536 is not guaranteed,
    // so scan them separately)
    int ic = sc, ia = sa, it = st;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const int oc = __shfl_up_sync(0xffffffff, ic, m);
        const int oa = __shfl_up_sync(0xffffffff, ia, m);
        const int ot = __shfl_up_sync(0xffffffff, it, m);
        if (lane >= m) { ic += oc; ia += oa; it += ot; }
    }
    __shared__ int wst[32];
    if (lane == 31) { wsum[0][warp] = ic; wsum[1][warp] = ia; wst[warp] = it; }
    __syncthreads();
    if (warp == 0) {
        int vc = wsum[0][lane], va = wsum[1][lane], vt = wst[lane];
#pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            const int oc = __shfl_up_sync(0xffffffff, vc, m);
            const int oa = __shfl_up_sync(0xffffffff, va, m);
            const int ot = __shfl_up_sync(0xffffffff, vt, m);
            if (lane >= m) { vc += oc; va += oa; vt += ot; }
        }
        wsum[0][lane] = vc; wsum[1][lane] = va; wst[lane] = vt;
    }
    __syncthreads();
    int bc = ic - sc + (warp > 0 ? wsum[0][warp - 1] : 0);   // exclusive prefix of this thread's first expert
    int ba = ia - sa + (warp > 0 ? wsum[1][warp - 1] : 0);
    int bt = it - st + (warp > 0 ? wst[warp - 1] : 0);
    const int n_active = wsum[1][31];
    const int n_tiles  = wst[31];
    __syncthreads();   // everyone has read wsum before cnt is reused as the scatter cursor
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid*4 + i;
        if (e < n_expert) {
            cnt[e] = bc;   // scatter cursor
        }
        if (a[i]) {
            exps[ba] = e;
            off[ba]  = bc;
            for (int j = 0; j < tl[i] && bt + j < max_tiles; ++j) {
                tiles[2*(bt + j)]     = ba;
                tiles[2*(bt + j) + 1] = j*bn;
            }
        }
        bc += c[i]; ba += a[i]; bt += tl[i];
    }
    if (tid == 0) {
        hdr[0] = n_active;
        hdr[1] = n_tiles < max_tiles ? n_tiles : max_tiles;
        off[n_active] = wsum[0][31];
    }
    __syncthreads();
    for (int64_t p = tid; p < n_pairs; p += IDS_THREADS) {
        const int u = (int) (p % n_used), t = (int) (p / n_used);
        const int e = *(const int *) (ids + u*ids_nb0 + t*ids_nb1);
        if (e >= 0 && e < n_expert) {
            const int k = atomicAdd(&cnt[e], 1);
            rows[k] = t*ne11 + u % ne11;
            cols[k] = t*n_used + u;
        }
    }
}

static idmap launch_ids_helper(const void * ids, int64_t ids_nb0, int64_t ids_nb1, int n_used, int n_tok, int ne11,
                               int n_expert, int bn, int max_tiles, int64_t nb02, int * out, cudaStream_t stream) {
    ids_helper<<<1, IDS_THREADS, 0, stream>>>((const char *) ids, ids_nb0, ids_nb1, n_used, n_tok, ne11, n_expert, bn, max_tiles, out);
    const int64_t n_pairs = (int64_t) n_used*n_tok;
    idmap m;
    m.hdr   = out;
    m.exps  = out + 2;
    m.off   = m.exps + n_expert;
    m.rows  = m.off + n_expert + 1;
    m.cols  = m.rows + n_pairs;
    m.tiles = m.cols + n_pairs;
    m.nb02  = nb02;
    return m;
}

} // namespace ai4k
