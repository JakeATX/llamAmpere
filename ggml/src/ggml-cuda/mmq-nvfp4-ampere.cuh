#pragma once

#include "common.cuh"
#include <cstdint>

// Ampere (SM 8.0/8.6/8.9) NVFP4 MMQ fast math primitives:
//   - Branchless bit-exact UE4M3 -> FP32 (0 ldexpf, 0 MUFU)
//   - No-MUFU Newton-Raphson reciprocal (0 MUFU, <6.5e-6 rel error on UE4M3 range)
//   - Exact magic int32 <-> fp32 conversion (0 I2FP, 0 F2I for |x| < 2^22)
//   - Register-only PRMT LUT decoders for BP-1B (scale-folded INT8) and BP-1A (bit-exact BF16)

// Exact branchless UE4M3 -> FP32 matching ggml_ue4m3_to_fp32 (includes the *0.5f pairing with kvalues_fp4).
// Verified bit-exact across all 128 valid UE4M3 codes (0x00..0x7F, mapping NaN 0x7F -> 0.0f).
static __device__ __forceinline__ float nvfp4_ue4m3_to_fp32_fast(const uint8_t u) {
    const uint32_t x     = static_cast<uint32_t>(u) & 0x7Fu;
    const uint32_t is_sn = x < 8u;
    const uint32_t base  = is_sn ? 0x3C000000u : 0x3B800000u;
    const float    sub   = is_sn ? 0.0078125f  : 0.0f; // 2^-7
    const float    val   = __fsub_rn(__uint_as_float(base + (x << 20)), sub);
    return (x == 0x7Fu) ? 0.0f : val;
}

// Reciprocal 1/a for positive normal floats a (all nonzero UE4M3 scales >= 2^-10) with ZERO MUFU instructions.
// IEEE-754 magic seed + 2 Newton-Raphson FMA iterations -> max rel error 6.5e-6 across all UE4M3 scales.
static __device__ __forceinline__ float nvfp4_fast_rcp_no_mufu(const float a) {
    float y = __uint_as_float(0x7EF311C2u - __float_as_uint(a));
    y = y * fmaf(-a, y, 2.0f);
    y = y * fmaf(-a, y, 2.0f);
    return (a > 0.0f) ? y : 0.0f;
}

// Exact int32 -> float32 for |x| < 2^22 using IADD3 + FADD (ZERO I2FP instructions).
// For BP-1B K=128 IMMA accumulation: max |acc| = 128 * 127 * 127 = 2,064,512 < 2^21 < 2^22.
static __device__ __forceinline__ float nvfp4_magic_i2f(const int x) {
    return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f);
}

// Exact float32 -> int32 (round-to-nearest-even) for |x| < 2^22 using FADD + IADD3 (ZERO F2I instructions).
static __device__ __forceinline__ int nvfp4_magic_f2i(const float x) {
    return __float_as_int(__fadd_rn(x, 12582912.0f)) - 0x4B400000;
}

// Register-only 16-entry INT8 nibble LUT via PRMT (__byte_perm).
// q4 holds 4 bytes = 8 nibbles; returns .x = 4 low-nibble int8s, .y = 4 high-nibble int8s.
static __device__ __forceinline__ int2 nvfp4_lut_i8_regs(
        const uint32_t q4, const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3) {
    const uint32_t idx = q4 & 0x77777777u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t lo0 = __byte_perm(t0, t1, idx);
    const uint32_t hi0 = __byte_perm(t2, t3, idx);
    const uint32_t lo1 = __byte_perm(t0, t1, idx >> 16);
    const uint32_t hi1 = __byte_perm(t2, t3, idx >> 16);
    const uint32_t m0  = __byte_perm(lo0, hi0, sel);
    const uint32_t m1  = __byte_perm(lo1, hi1, sel >> 16);
    return make_int2(
        static_cast<int>(__byte_perm(m0, m1, 0x6420u)),
        static_cast<int>(__byte_perm(m0, m1, 0x7531u)));
}

// Build 8-entry positive BF16 byte tables (lo0, lo1, hi0, hi1) for kvalues_fp4[0..7] * d.
// Because E2M1 has <= 2 mantissa bits and UE4M3 has <= 3 mantissa bits, every product has <= 5 mantissa bits,
// so FP32 upper-16-bit truncation is 100% bit-exact in BF16 (0 rounding error, 0 F2FP, 0 I2FP).
static __device__ __forceinline__ void nvfp4_build_bf16_lut(
        const float d, uint32_t & lo0, uint32_t & lo1, uint32_t & hi0, uint32_t & hi1) {
    // kvalues_fp4[0..7] = {0, 1, 2, 3, 4, 6, 8, 12}
    const uint32_t f1  = __float_as_uint(d);
    const uint32_t f2  = __float_as_uint(d * 2.0f);
    const uint32_t f3  = __float_as_uint(d * 3.0f);
    const uint32_t f4  = __float_as_uint(d * 4.0f);
    const uint32_t f6  = __float_as_uint(d * 6.0f);
    const uint32_t f8  = __float_as_uint(d * 8.0f);
    const uint32_t f12 = __float_as_uint(d * 12.0f);

    // Pack upper 16 bits (BF16) of pairs into 32-bit words:
    // p01 = [0x0000, bf16(1*d)], p23 = [bf16(2*d), bf16(3*d)]
    // p46 = [bf16(4*d), bf16(6*d)], p812 = [bf16(8*d), bf16(12*d)]
    const uint32_t p01  = __byte_perm(0u, f1, 0x7610u);
    const uint32_t p23  = __byte_perm(f2, f3, 0x7632u);
    const uint32_t p46  = __byte_perm(f4, f6, 0x7632u);
    const uint32_t p812 = __byte_perm(f8, f12, 0x7632u);

    // Extract low bytes and high bytes of each BF16 entry:
    // p01  bytes: [0_lo(0), 0_hi(1), 1_lo(2), 1_hi(3)]
    // p23  bytes: [2_lo(4), 2_hi(5), 3_lo(6), 3_hi(7)]
    lo0 = __byte_perm(p01,  p23,  0x6420u); // [b0_lo, b1_lo, b2_lo, b3_lo]
    hi0 = __byte_perm(p01,  p23,  0x7531u); // [b0_hi, b1_hi, b2_hi, b3_hi]
    lo1 = __byte_perm(p46,  p812, 0x6420u); // [b4_lo, b6_lo, b8_lo, b12_lo]
    hi1 = __byte_perm(p46,  p812, 0x7531u); // [b4_hi, b6_hi, b8_hi, b12_hi]
}

// Decode 4 bytes (8 nibbles) of NVFP4 codes directly into 4 packed nv_bfloat162 words (8 BF16 weights):
//   out.x = {bf16(b0_lo), bf16(b1_lo)}
//   out.y = {bf16(b2_lo), bf16(b3_lo)}
//   out.z = {bf16(b0_hi), bf16(b1_hi)}
//   out.w = {bf16(b2_hi), bf16(b3_hi)}
static __device__ __forceinline__ uint4 nvfp4_decode_q4_bf16x8(
        const uint32_t q4, const uint32_t lo0, const uint32_t lo1, const uint32_t hi0, const uint32_t hi1) {
    const uint32_t idx = q4 & 0x77777777u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);

    const uint32_t l0  = __byte_perm(lo0, lo1, idx);
    const uint32_t hp0 = __byte_perm(hi0, hi1, idx);
    const uint32_t h0  = __byte_perm(hp0, hp0 ^ 0x80808080u, sel);

    const uint32_t l1  = __byte_perm(lo0, lo1, idx >> 16);
    const uint32_t hp1 = __byte_perm(hi0, hi1, idx >> 16);
    const uint32_t h1  = __byte_perm(hp1, hp1 ^ 0x80808080u, sel >> 16);

    return make_uint4(
        __byte_perm(l0, h0, 0x6240u), // low nibbles of bytes 0, 1
        __byte_perm(l1, h1, 0x6240u), // low nibbles of bytes 2, 3
        __byte_perm(l0, h0, 0x7351u), // high nibbles of bytes 0, 1
        __byte_perm(l1, h1, 0x7351u)  // high nibbles of bytes 2, 3
    );
}
