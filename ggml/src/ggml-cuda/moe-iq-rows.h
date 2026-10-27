#pragma once
// Row kernels for the IQ up/gate weights of the host expert pool (GGML_MOE_CPU_SPLIT_ROWS, off by default).
//
// A call computes rows [r0, r1) of one expert against 1..8 tokens (Q8_K activations). The point is reuse: the
// weights of each 32-weight slice are unpacked once into a `slice` and then applied to every token, where ggml's
// vec_dot unpacks them again for each token.
//
// The arithmetic follows the AVX2 vec_dot of each type in ggml-cpu/arch/x86/quants.c: unsigned grid magnitudes
// times sign-adjusted int8 activations through maddubs, madd with the odd multiplier 2*scale+1, int32 sums per
// 256-weight block, then one fused multiply-add per block with (weight d * activation d) and ggml's lane sum. The
// outputs are therefore the same floats ggml produces (test-moe-iq-rows compares the bits, and compares each
// unpacked slice with ggml's own dequantize_row_* output).
//
// Sign bytes come from ggml-common.h's ksigns_iq2xs for the formats with 7-bit sign indices, and from ggml's
// shuffle-and-mask expansion for the formats that store one sign bit per weight.
//
// Needs the IQ grid and sign tables of ggml-common.h (GGML_COMMON_IMPL_CPP) included before this header.

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__) && defined(__BMI2__)
#define MOE_IQ_ROWS 1
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace moe_iq {

// ---- sign bytes ------------------------------------------------------------------------------------------------

// ksigns_iq2xs maps a 7-bit sign index to the 8 sign bits of a weight group (bit 7 is the parity bit). Widen each
// entry once to one byte per weight: 0x01 keeps the activation, 0xFF negates it (the second operand of
// _mm256_sign_epi8).
struct pm_lut {
    uint64_t v[128];
    constexpr pm_lut() : v() {
        for (int i = 0; i < 128; ++i) {
            uint64_t w = 0;
            for (int bit = 0; bit < 8; ++bit) {
                w |= ((ksigns_iq2xs[i] >> bit) & 1 ? 0xFFull : 0x01ull) << (8 * bit);
            }
            v[i] = w;
        }
    }
};
alignas(64) inline constexpr pm_lut k_pm{};
static_assert(k_pm.v[3] == 0x010101010101FFFFull && k_pm.v[127] == 0xFFFFFFFFFFFFFFFFull, "pm_lut");

// byte picks / bit selectors for the one-sign-bit-per-weight formats (same constants as ggml's IQ2_S / IQ3_S dot)
alignas(32) inline constexpr uint8_t k_pick_byte[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };
alignas(32) inline constexpr uint8_t k_pick_bit[32] = {
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

// 32 stored sign bits (bit j belongs to weight j) -> 0x01 / 0xFF bytes
inline __m256i signs_from_bits(uint32_t bits) {
    const __m256i pick = _mm256_load_si256((const __m256i *) k_pick_byte);
    const __m256i sel  = _mm256_load_si256((const __m256i *) k_pick_bit);
    const __m256i hit  = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) bits), pick), sel);
    return _mm256_or_si256(_mm256_cmpeq_epi8(hit, sel), _mm256_set1_epi8(1));
}

// four 7-bit sign indices in the low 28 bits of w -> 32 sign bytes
inline __m256i signs_from_indices(uint32_t w) {
    return _mm256_set_epi64x((long long) k_pm.v[(w >> 21) & 127], (long long) k_pm.v[(w >> 14) & 127],
                             (long long) k_pm.v[(w >> 7) & 127], (long long) k_pm.v[w & 127]);
}

// ---- small helpers ---------------------------------------------------------------------------------------------

inline uint32_t load32(const void * p) { uint32_t v; memcpy(&v, p, sizeof(v)); return v; }

// sum of the 8 float lanes in ggml's order (ggml-cpu hsum_float_8)
inline float lane_sum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// pull the cache lines `ahead` bytes in front of a weight block toward L1
template <class Blk>
inline void warm(const Blk * b, int ahead) {
    const char * p = (const char *) b + ahead;
    _mm_prefetch(p, _MM_HINT_T0);
    _mm_prefetch(p + sizeof(Blk) - 1, _MM_HINT_T0);
}

// ---- one unpacked 32-weight slice ------------------------------------------------------------------------------

struct slice {
    __m256i mag;    // grid magnitudes, unsigned bytes
    __m256i sgn;    // 0x01 / 0xFF per weight
    int mul_lo;     // 2*scale+1 for weights 0..15
    int mul_hi;     // 2*scale+1 for weights 16..31 (equals mul_lo when the format has one scale per slice)
};

// Per format: the block type, the final scale of the dot, and unpack(block, slice index 0..7).
template <ggml_type T, bool Joined = false> struct fmt;

template <> struct fmt<GGML_TYPE_IQ2_XXS> {
    using blk = block_iq2_xxs;
    static constexpr float post = 0.125f;
    // 8 bytes per slice: four grid-index bytes, then a word of four sign indices (7 bits each) and the scale on top
    static inline slice unpack(const blk & b, int c) {
        const uint8_t * p = (const uint8_t *) b.qs + 8 * c;
        const uint32_t tail = load32(p + 4);
        slice s;
        s.mag = _mm256_set_epi64x((long long) iq2xxs_grid[p[3]], (long long) iq2xxs_grid[p[2]],
                                  (long long) iq2xxs_grid[p[1]], (long long) iq2xxs_grid[p[0]]);
        s.sgn = signs_from_indices(tail);
        s.mul_lo = s.mul_hi = (int) (2 * (tail >> 28) + 1);
        return s;
    }
};

template <> struct fmt<GGML_TYPE_IQ2_XS> {
    using blk = block_iq2_xs;
    static constexpr float post = 0.125f;
    // one 16-bit entry per 8 weights: 9-bit grid index, 7-bit sign index; one scale byte (two nibbles) per slice
    static inline slice unpack(const blk & b, int c) {
        const uint16_t * q = b.qs + 4 * c;
        slice s;
        s.mag = _mm256_set_epi64x((long long) iq2xs_grid[q[3] & 511], (long long) iq2xs_grid[q[2] & 511],
                                  (long long) iq2xs_grid[q[1] & 511], (long long) iq2xs_grid[q[0] & 511]);
        s.sgn = _mm256_set_epi64x((long long) k_pm.v[q[3] >> 9], (long long) k_pm.v[q[2] >> 9],
                                  (long long) k_pm.v[q[1] >> 9], (long long) k_pm.v[q[0] >> 9]);
        s.mul_lo = 2 * (b.scales[c] & 15) + 1;
        s.mul_hi = 2 * (b.scales[c] >> 4) + 1;
        return s;
    }
};

template <> struct fmt<GGML_TYPE_IQ2_S> {
    using blk = block_iq2_s;
    static constexpr float post = 0.125f;
    // qs: 32 low index bytes, then 32 sign bytes; qh: two high index bits per 8 weights; scale byte as IQ2_XS
    static inline slice unpack(const blk & b, int c) {
        const uint8_t * q = b.qs + 4 * c;
        const uint32_t h = b.qh[c];
        slice s;
        s.mag = _mm256_set_epi64x((long long) iq2s_grid[q[3] | ((h << 2) & 0x300)], (long long) iq2s_grid[q[2] | ((h << 4) & 0x300)],
                                  (long long) iq2s_grid[q[1] | ((h << 6) & 0x300)], (long long) iq2s_grid[q[0] | ((h << 8) & 0x300)]);
        s.sgn = signs_from_bits(load32(b.qs + 32 + 4 * c));
        s.mul_lo = 2 * (b.scales[c] & 15) + 1;
        s.mul_hi = 2 * (b.scales[c] >> 4) + 1;
        return s;
    }
};

// 4-byte grid entries of IQ3 as eight lanes: one 32-bit insert each, or (Joined) paired up in scalar registers first
// so the vector side takes four 64-bit inserts
inline uint64_t entry_pair(const uint32_t * grid, uint32_t a, uint32_t b) { return (uint64_t) grid[a] | ((uint64_t) grid[b] << 32); }

template <bool Joined> inline __m256i grid8(const uint32_t * grid, const uint32_t (&ix)[8]) {
    if constexpr (Joined) {
        return _mm256_set_epi64x((long long) entry_pair(grid, ix[6], ix[7]), (long long) entry_pair(grid, ix[4], ix[5]),
                                 (long long) entry_pair(grid, ix[2], ix[3]), (long long) entry_pair(grid, ix[0], ix[1]));
    }
    return _mm256_set_epi32((int) grid[ix[7]], (int) grid[ix[6]], (int) grid[ix[5]], (int) grid[ix[4]],
                            (int) grid[ix[3]], (int) grid[ix[2]], (int) grid[ix[1]], (int) grid[ix[0]]);
}

template <bool Joined> struct fmt<GGML_TYPE_IQ3_XXS, Joined> {
    using blk = block_iq3_xxs;
    static constexpr float post = 0.25f;
    // qs[0..63]: one grid byte per 4 weights; after that, per slice, a word of four 7-bit sign indices + scale on top
    static inline slice unpack(const blk & b, int c) {
        const uint8_t * q = b.qs + 8 * c;
        const uint32_t tail = load32(b.qs + 64 + 4 * c);
        const uint32_t ix[8] = { q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7] };
        slice s;
        s.mag = grid8<Joined>(iq3xxs_grid, ix);
        s.sgn = signs_from_indices(tail);
        s.mul_lo = s.mul_hi = (int) (2 * (tail >> 28) + 1);
        return s;
    }
};

template <bool Joined> struct fmt<GGML_TYPE_IQ3_S, Joined> {
    using blk = block_iq3_s;
    static constexpr float post = 1.0f;
    // qs: 8 low index bytes per slice; qh: the ninth index bit of each, packed in one byte; signs: one bit per weight
    static inline slice unpack(const blk & b, int c) {
        const uint8_t * q = b.qs + 8 * c;
        const uint32_t h = b.qh[c];
        uint32_t ix[8];
        for (int k = 0; k < 8; ++k) { ix[k] = q[k] | ((h << (8 - k)) & 256); }
        slice s;
        s.mag = grid8<Joined>(iq3s_grid, ix);
        s.sgn = signs_from_bits(load32(b.signs + 4 * c));
        s.mul_lo = s.mul_hi = 2 * ((b.scales[c >> 1] >> (4 * (c & 1))) & 15) + 1;
        return s;
    }
};

// ---- the row loops ---------------------------------------------------------------------------------------------

typedef void (*rows_fn)(const char *, size_t, int64_t, int64_t, int, const block_q8_K * const *, float * const *, int);

// One 256-weight block of R rows against NT tokens: integer sums per slice, then one float step per block.
template <class Fm, int NT, int R>
inline void block_step(const typename Fm::blk * const (&wb)[R], const block_q8_K * const * y, int i,
        __m256 (&facc)[R][NT]) {
    __m256i isum[R][NT];
    for (int k = 0; k < R; ++k) { for (int t = 0; t < NT; ++t) { isum[k][t] = _mm256_setzero_si256(); } }
    for (int c = 0; c < 8; ++c) {
        __m256i act[NT];
        for (int t = 0; t < NT; ++t) { act[t] = _mm256_loadu_si256((const __m256i *) (y[t][i].qs + 32 * c)); }
        for (int k = 0; k < R; ++k) {
            const slice s = Fm::unpack(*wb[k], c);
            const __m256i mul = _mm256_setr_m128i(_mm_set1_epi16((short) s.mul_lo), _mm_set1_epi16((short) s.mul_hi));
            for (int t = 0; t < NT; ++t) {
                const __m256i pairs = _mm256_maddubs_epi16(s.mag, _mm256_sign_epi8(act[t], s.sgn));
                isum[k][t] = _mm256_add_epi32(isum[k][t], _mm256_madd_epi16(pairs, mul));
            }
        }
    }
    for (int k = 0; k < R; ++k) {
        const float dw = _cvtsh_ss(wb[k]->d);
        for (int t = 0; t < NT; ++t) {
            facc[k][t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(isum[k][t]), facc[k][t]);
        }
    }
}

// R consecutive rows starting at row r, NT tokens; writes out[t][r + k - r0]
template <class Fm, int NT, int R>
inline void tile(const char * w, size_t row_bytes, int64_t r, int64_t r0, int nb, const block_q8_K * const * y,
        float * const * out, int ahead) {
    using blk = typename Fm::blk;
    const blk * first[R];
    for (int k = 0; k < R; ++k) { first[k] = (const blk *) (w + (size_t) (r + k) * row_bytes); }
    __m256 facc[R][NT];
    for (int k = 0; k < R; ++k) { for (int t = 0; t < NT; ++t) { facc[k][t] = _mm256_setzero_ps(); } }
    for (int i = 0; i < nb; ++i) {
        const blk * cur[R];
        for (int k = 0; k < R; ++k) {
            cur[k] = first[k] + i;
            if (ahead > 0) { warm(cur[k], ahead); }
        }
        block_step<Fm, NT, R>(cur, y, i, facc);
    }
    for (int k = 0; k < R; ++k) {
        for (int t = 0; t < NT; ++t) { out[t][r + k - r0] = Fm::post * lane_sum(facc[k][t]); }
    }
}

// rows [r0, r1): R rows per tile, single rows for the remainder. `ahead` = prefetch distance in bytes (0 = off).
template <class Fm, int NT, int R>
void run_rows(const char * w, size_t row_bytes, int64_t r0, int64_t r1, int nb, const block_q8_K * const * y,
        float * const * out, int ahead) {
    int64_t r = r0;
    for (; r + R <= r1; r += R) { tile<Fm, NT, R>(w, row_bytes, r, r0, nb, y, out, ahead); }
    for (; r < r1; ++r) { tile<Fm, NT, 1>(w, row_bytes, r, r0, nb, y, out, ahead); }
}

// kernel for nt tokens (1..8); R1 rows per tile when nt == 1 (more rows only help the single-token case)
template <class Fm, int R1, int NT = 8>
rows_fn by_tokens(int nt) {
    if constexpr (NT == 0) {
        return nullptr;
    } else {
        return nt == NT ? &run_rows<Fm, NT, (NT == 1 ? R1 : 1)> : by_tokens<Fm, R1, NT - 1>(nt);
    }
}

template <class Fm> rows_fn choose(int nt, bool two_rows) {
    return two_rows ? by_tokens<Fm, 2>(nt) : by_tokens<Fm, 1>(nt);
}

// Kernel for `type` and nt tokens, or null when there is none (or `variant` leaves the case to ggml's vec_dot).
// variant bits: 1 = IQ3_XXS grid entries joined in scalar registers; 2 = two rows per tile for one token;
//               4 = the same joined entries for IQ3_S; 8 = one token of IQ2_S / IQ3_XXS stays on ggml's vec_dot
//               (there ggml's single-token loop measured faster on Zen 3 and Zen 5).
// variant_best = 1 | 8, the mix test-moe-iq-rows favored on the home 5800X3D and the Taichi 9900X.
inline constexpr int variant_best = 1 | 8;
inline rows_fn get(int type, int nt, int variant = variant_best) {
    const bool two = (variant & 2) != 0;
    const bool leave_to_ggml = nt == 1 && (variant & 8) != 0;
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return choose<fmt<GGML_TYPE_IQ2_XXS>>(nt, two);
        case GGML_TYPE_IQ2_XS:  return choose<fmt<GGML_TYPE_IQ2_XS>>(nt, two);
        case GGML_TYPE_IQ2_S:   return leave_to_ggml ? nullptr : choose<fmt<GGML_TYPE_IQ2_S>>(nt, two);
        case GGML_TYPE_IQ3_XXS:
            if (leave_to_ggml) { return nullptr; }
            return (variant & 1) ? choose<fmt<GGML_TYPE_IQ3_XXS, true>>(nt, two) : choose<fmt<GGML_TYPE_IQ3_XXS>>(nt, two);
        case GGML_TYPE_IQ3_S:
            return (variant & 4) ? choose<fmt<GGML_TYPE_IQ3_S, true>>(nt, two) : choose<fmt<GGML_TYPE_IQ3_S>>(nt, two);
        default: return nullptr;
    }
}

// Test hook: unpack slice c of one weight block of `type` into magnitudes (u8), signs (+1 / -1) and the two
// multipliers; `joined` picks the IQ3 grid-assembly arm. False for a type without a kernel.
template <class Fm> void unpack_to(const void * block, int c, uint8_t * mag, int8_t * sgn, int * mul) {
    const slice s = Fm::unpack(*(const typename Fm::blk *) block, c);
    _mm256_storeu_si256((__m256i *) mag, s.mag);
    _mm256_storeu_si256((__m256i *) sgn, s.sgn);   // 0x01 / 0xFF read as int8: +1 / -1
    mul[0] = s.mul_lo;
    mul[1] = s.mul_hi;
}
inline bool unpack_slice(int type, const void * block, int c, bool joined, uint8_t * mag, int8_t * sgn, int * mul) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: unpack_to<fmt<GGML_TYPE_IQ2_XXS>>(block, c, mag, sgn, mul); return true;
        case GGML_TYPE_IQ2_XS:  unpack_to<fmt<GGML_TYPE_IQ2_XS>>(block, c, mag, sgn, mul); return true;
        case GGML_TYPE_IQ2_S:   unpack_to<fmt<GGML_TYPE_IQ2_S>>(block, c, mag, sgn, mul); return true;
        case GGML_TYPE_IQ3_XXS:
            if (joined) { unpack_to<fmt<GGML_TYPE_IQ3_XXS, true>>(block, c, mag, sgn, mul); }
            else        { unpack_to<fmt<GGML_TYPE_IQ3_XXS>>(block, c, mag, sgn, mul); }
            return true;
        case GGML_TYPE_IQ3_S:
            if (joined) { unpack_to<fmt<GGML_TYPE_IQ3_S, true>>(block, c, mag, sgn, mul); }
            else        { unpack_to<fmt<GGML_TYPE_IQ3_S>>(block, c, mag, sgn, mul); }
            return true;
        default: return false;
    }
}

#if defined(__x86_64__) && defined(__GNUC__)
#define MOE_IQ_ROWS_512 1
#define MOE_IQ_T512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c")))

// AVX-512 VNNI variant: two slices side by side in one 512-bit register, dot products by vpdpbusd. One int32 lane
// covers four weights, so the multiplier is broadcast per four-weight group (16 weights share one multiplier).
inline __m256i mul_lanes(const slice & s) {
    return _mm256_setr_m128i(_mm_set1_epi32(s.mul_lo), _mm_set1_epi32(s.mul_hi));
}

template <class Fm, int NT>
MOE_IQ_T512 void run_rows512(const char * w, size_t row_bytes, int64_t r0, int64_t r1, int nb,
        const block_q8_K * const * y, float * const * out, int ahead) {
    using blk = typename Fm::blk;
    for (int64_t r = r0; r < r1; ++r) {
        const blk * row = (const blk *) (w + (size_t) r * row_bytes);
        __m512 facc[NT];
        for (int t = 0; t < NT; ++t) { facc[t] = _mm512_setzero_ps(); }
        for (int i = 0; i < nb; ++i) {
            if (ahead > 0) { warm(row + i, ahead); }
            __m512i isum[NT];
            for (int t = 0; t < NT; ++t) { isum[t] = _mm512_setzero_si512(); }
            for (int c = 0; c < 8; c += 2) {
                const slice a = Fm::unpack(row[i], c);
                const slice b = Fm::unpack(row[i], c + 1);
                const __m512i mag = _mm512_inserti64x4(_mm512_castsi256_si512(a.mag), b.mag, 1);
                const __m512i mul = _mm512_inserti64x4(_mm512_castsi256_si512(mul_lanes(a)), mul_lanes(b), 1);
                for (int t = 0; t < NT; ++t) {
                    const int8_t * q = y[t][i].qs + 32 * c;
                    const __m256i qa = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q), a.sgn);
                    const __m256i qb = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q + 32)), b.sgn);
                    const __m512i act = _mm512_inserti64x4(_mm512_castsi256_si512(qa), qb, 1);
                    const __m512i d4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, act);
                    isum[t] = _mm512_add_epi32(isum[t], _mm512_mullo_epi32(d4, mul));
                }
            }
            const float dw = _cvtsh_ss(row[i].d);
            for (int t = 0; t < NT; ++t) {
                facc[t] = _mm512_fmadd_ps(_mm512_set1_ps(dw * y[t][i].d), _mm512_cvtepi32_ps(isum[t]), facc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) { out[t][r - r0] = Fm::post * _mm512_reduce_add_ps(facc[t]); }
    }
}

template <class Fm, int NT = 8>
rows_fn by_tokens512(int nt) {
    if constexpr (NT == 0) {
        return nullptr;
    } else {
        return nt == NT ? &run_rows512<Fm, NT> : by_tokens512<Fm, NT - 1>(nt);
    }
}

inline rows_fn get512(int type, int nt) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return by_tokens512<fmt<GGML_TYPE_IQ2_XXS>>(nt);
        case GGML_TYPE_IQ2_XS:  return by_tokens512<fmt<GGML_TYPE_IQ2_XS>>(nt);
        case GGML_TYPE_IQ2_S:   return by_tokens512<fmt<GGML_TYPE_IQ2_S>>(nt);
        case GGML_TYPE_IQ3_XXS: return by_tokens512<fmt<GGML_TYPE_IQ3_XXS, true>>(nt);
        case GGML_TYPE_IQ3_S:   return by_tokens512<fmt<GGML_TYPE_IQ3_S>>(nt);
        default: return nullptr;
    }
}
#endif

} // namespace moe_iq
#endif
