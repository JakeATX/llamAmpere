#pragma once
// fork STRATAGAP: AVX2 row kernels for the IQ up/gate weights of the host expert pool (GGML_MOE_CPU_SPLIT_ROWS=1,
// default off). One call computes rows [r0, r1) of one expert for NT = 1..4 tokens. Each 32-weight chunk is decoded
// once (grid magnitudes, sign bytes, scale vector) and then multiplied with every token's activations; ggml's
// vec_dot decodes it again for every token.
//
// Same integer work as ggml-cpu's AVX2 vec_dot of each type: the same u8 x s8 pair products, every product lands in
// the same int32 lane, the per-block float step is the same fmadd(d_x * d_y, lanes) and the final horizontal sum is
// the same. So every output is bit-identical to ggml's (the parity test checks the bits).
//
// Decode details (per 32-weight chunk): IQ3_S indices are 8 low bytes interleaved with the high bits, which one
// pdep spreads one per byte; sign bytes come from tables or from one broadcast + shuffle of the 32 sign bits; scale
// vectors are table loads. The grid lookups stay scalar loads (on Zen 3 the AVX2 gather is slower).
//
// Needs ggml-common.h's tables (GGML_COMMON_IMPL_CPP) included before this header.

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__) && defined(__BMI2__)
#define MOE_IQ_ROWS 1
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace moe_iq {

// Sign bytes of an 8-weight group from its 7-bit sign index (the 8th sign makes the count of minus signs even):
// byte k = 0xFF (negative) or 0x01, the second operand of _mm256_sign_epi8. Same values as ggml's keven_signs_q2xs.
struct tables {
    uint64_t sign8[128];
    int16_t sc32[16][16];     // all lanes 2s+1 (one scale per 32 weights)
    int16_t sc16[256][16];    // lanes 0-7: 2*(b & 15)+1, lanes 8-15: 2*(b >> 4)+1 (one scale per 16 weights)
};

constexpr tables make_tables() {
    tables t{};
    for (int i = 0; i < 128; ++i) {
        int odd = 0;
        for (int k = 0; k < 7; ++k) { odd ^= (i >> k) & 1; }
        const int bits = i | (odd << 7);
        uint64_t v = 0;
        for (int k = 0; k < 8; ++k) { v |= (uint64_t) (((bits >> k) & 1) ? 0xFF : 0x01) << (8 * k); }
        t.sign8[i] = v;
    }
    for (int s = 0; s < 16; ++s) {
        for (int l = 0; l < 16; ++l) { t.sc32[s][l] = (int16_t) (2 * s + 1); }
    }
    for (int b = 0; b < 256; ++b) {
        for (int l = 0; l < 8; ++l) {
            t.sc16[b][l] = (int16_t) (2 * (b & 15) + 1);
            t.sc16[b][l + 8] = (int16_t) (2 * (b >> 4) + 1);
        }
    }
    return t;
}

alignas(64) inline constexpr tables T = make_tables();
static_assert(T.sign8[0] == 0x0101010101010101ull && T.sign8[1] == 0xFF010101010101FFull, "sign8");

inline __m256i load_sc(const int16_t * lanes) { return _mm256_load_si256((const __m256i *) lanes); }

// 32 sign bits (bit j = weight j) -> sign bytes 0xFF / 0x01
inline __m256i bits_to_signs(uint32_t b) {
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) b), spread), sel);
    return _mm256_or_si256(_mm256_cmpeq_epi8(v, sel), _mm256_set1_epi8(1));
}

inline uint32_t rd32(const void * p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint64_t rd64(const void * p) { uint64_t v; memcpy(&v, p, 8); return v; }

// ggml-cpu's hsum_float_8, same order
inline float hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Per type: dec(block, chunk c in 0..7) -> g (u8 magnitudes), s (sign bytes), sc (int16 scales per pair lane).
struct f_iq2_xxs {   // qs: per chunk 4 grid bytes, then 4 x 7-bit sign index + 4-bit scale
    using B = block_iq2_xxs;
    static constexpr float K = 0.125f;
    static inline void dec(const B * x, int c, __m256i & g, __m256i & s, __m256i & sc) {
        const uint8_t * q = (const uint8_t *) x->qs + 8 * c;
        const uint32_t a = rd32(q), w = rd32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[a >> 24], (long long) iq2xxs_grid[(a >> 16) & 255],
                              (long long) iq2xxs_grid[(a >> 8) & 255], (long long) iq2xxs_grid[a & 255]);
        s = _mm256_set_epi64x((long long) T.sign8[(w >> 21) & 127], (long long) T.sign8[(w >> 14) & 127],
                              (long long) T.sign8[(w >> 7) & 127], (long long) T.sign8[w & 127]);
        sc = load_sc(T.sc32[w >> 28]);
    }
};

struct f_iq2_xs {    // qs: u16 per 8 weights (9-bit grid index, 7-bit sign index); scales: one byte per chunk
    using B = block_iq2_xs;
    static constexpr float K = 0.125f;
    static inline void dec(const B * x, int c, __m256i & g, __m256i & s, __m256i & sc) {
        const uint64_t v = rd64(x->qs + 4 * c);
        const uint32_t v0 = (uint32_t) v & 0xffff, v1 = (uint32_t) (v >> 16) & 0xffff;
        const uint32_t v2 = (uint32_t) (v >> 32) & 0xffff, v3 = (uint32_t) (v >> 48);
        g = _mm256_set_epi64x((long long) iq2xs_grid[v3 & 511], (long long) iq2xs_grid[v2 & 511],
                              (long long) iq2xs_grid[v1 & 511], (long long) iq2xs_grid[v0 & 511]);
        s = _mm256_set_epi64x((long long) T.sign8[v3 >> 9], (long long) T.sign8[v2 >> 9],
                              (long long) T.sign8[v1 >> 9], (long long) T.sign8[v0 >> 9]);
        sc = load_sc(T.sc16[x->scales[c]]);
    }
};

struct f_iq2_s {     // qs[0..31] low index bytes, qs[32..63] sign bytes; qh: 2 high index bits per 8 weights
    using B = block_iq2_s;
    static constexpr float K = 0.125f;
    static inline void dec(const B * x, int c, __m256i & g, __m256i & s, __m256i & sc) {
        const uint8_t * q = x->qs + 4 * c;
        const uint32_t h = x->qh[c];
        g = _mm256_set_epi64x((long long) iq2s_grid[q[3] | ((h << 2) & 0x300)], (long long) iq2s_grid[q[2] | ((h << 4) & 0x300)],
                              (long long) iq2s_grid[q[1] | ((h << 6) & 0x300)], (long long) iq2s_grid[q[0] | ((h << 8) & 0x300)]);
        s = bits_to_signs(rd32(x->qs + 32 + 4 * c));
        sc = load_sc(T.sc16[x->scales[c]]);
    }
};

struct f_iq3_xxs {   // qs[0..63] grid bytes (4 weights each), then per chunk 4 x 7-bit sign index + 4-bit scale
    using B = block_iq3_xxs;
    static constexpr float K = 0.25f;
    static inline void dec(const B * x, int c, __m256i & g, __m256i & s, __m256i & sc) {
        const uint8_t * q = x->qs + 8 * c;
        const uint32_t w = rd32(x->qs + 64 + 4 * c);
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        s = _mm256_set_epi64x((long long) T.sign8[(w >> 21) & 127], (long long) T.sign8[(w >> 14) & 127],
                              (long long) T.sign8[(w >> 7) & 127], (long long) T.sign8[w & 127]);
        sc = load_sc(T.sc32[w >> 28]);
    }
};

struct f_iq3_s {     // qs: 8 low index bytes per chunk; qh: the 9th index bit (one byte per chunk); signs: 1 bit/weight
    using B = block_iq3_s;
    static constexpr float K = 1.0f;
    static inline void dec(const B * x, int c, __m256i & g, __m256i & s, __m256i & sc) {
        // indices as u16: low byte from qs, high byte = bit k of qh (pdep puts bit k at the bottom of byte k)
        alignas(16) uint16_t ix[8];
        const __m128i lo = _mm_cvtsi64_si128((long long) rd64(x->qs + 8 * c));
        const __m128i hi = _mm_cvtsi64_si128((long long) _pdep_u64(x->qh[c], 0x0101010101010101ull));
        _mm_store_si128((__m128i *) ix, _mm_unpacklo_epi8(lo, hi));
        g = _mm256_set_epi32((int) iq3s_grid[ix[7]], (int) iq3s_grid[ix[6]], (int) iq3s_grid[ix[5]], (int) iq3s_grid[ix[4]],
                             (int) iq3s_grid[ix[3]], (int) iq3s_grid[ix[2]], (int) iq3s_grid[ix[1]], (int) iq3s_grid[ix[0]]);
        s = bits_to_signs(rd32(x->signs + 4 * c));
        sc = load_sc(T.sc32[(x->scales[c >> 1] >> (4 * (c & 1))) & 15]);
    }
};

// rows [r0, r1) of one expert for NT tokens: out[t][r - r0]. y[t] = token t's Q8_K row. pf = prefetch distance in
// bytes (0 = off).
template <class F, int NT>
void rows(const char * w, size_t row_bytes, int64_t r0, int64_t r1, int nb, const block_q8_K * const * y,
        float * const * out, int pf) {
    for (int64_t r = r0; r < r1; ++r) {
        const typename F::B * x = (const typename F::B *) (w + (size_t) r * row_bytes);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) { accf[t] = _mm256_setzero_ps(); }
        for (int i = 0; i < nb; ++i) {
            if (pf > 0) {
                _mm_prefetch((const char *) (x + i) + pf, _MM_HINT_T0);
                _mm_prefetch((const char *) (x + i) + pf + 64, _MM_HINT_T0);
            }
            __m256i acc[NT];
            for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_si256(); }
            for (int c = 0; c < 8; ++c) {
                __m256i g, s, sc;
                F::dec(x + i, c, g, s, sc);
                for (int t = 0; t < NT; ++t) {
                    const __m256i q = _mm256_loadu_si256((const __m256i *) (y[t][i].qs + 32 * c));
                    acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(q, s)), sc));
                }
            }
            const float dx = _cvtsh_ss(x[i].d);
            for (int t = 0; t < NT; ++t) {
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acc[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) { out[t][r - r0] = F::K * hsum(accf[t]); }
    }
}

typedef void (*rows_fn)(const char *, size_t, int64_t, int64_t, int, const block_q8_K * const *, float * const *, int);

template <class F> rows_fn pick(int nt) {
    switch (nt) {
        case 1: return rows<F, 1>;
        case 2: return rows<F, 2>;
        case 3: return rows<F, 3>;
        case 4: return rows<F, 4>;
        default: return nullptr;
    }
}

// The kernel for `type` and nt tokens (1..4), or null when the type has none.
inline rows_fn get(int type, int nt) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return pick<f_iq2_xxs>(nt);
        case GGML_TYPE_IQ2_XS:  return pick<f_iq2_xs>(nt);
        case GGML_TYPE_IQ2_S:   return pick<f_iq2_s>(nt);
        case GGML_TYPE_IQ3_XXS: return pick<f_iq3_xxs>(nt);
        case GGML_TYPE_IQ3_S:   return pick<f_iq3_s>(nt);
        default: return nullptr;
    }
}

} // namespace moe_iq
#endif
