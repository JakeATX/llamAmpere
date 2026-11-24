#pragma once
// AVX2 pair-table decode for AIT2P_U unit slabs (ggml-ai4.h, trellis multiplier C == 0): the levels of a weight pair come
// from ait2p_tab[2 ait2p_entry(w) + {0, 1}] with w = the 20 stream bits ending at bit 4p + 4 of a k = 2 tile. Two tile rows
// (8 pairs = 16 weights) are decoded per step: one variable shift + mask cuts the eight 20-bit windows, one vpmulld forms
// the entry hash, and one 8-lane gather reads a 4 KB table of packed int16 pairs. Levels are bit-exact to aik_level_n.
// The dot product runs on activations already rotated into the weights' basis (sign + normalized FWHT per 128/256 block),
// so a slab row's result is sum over units of fp16 scale * sum(level * rotated x); the float summation order differs
// from the dequantize + f32 dot reference (valid-math change, not bit-exact).

#include "ggml-ai4.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GGML_AI4_PAIR_AVX2 1
#else
#define GGML_AI4_PAIR_AVX2 0
#endif

namespace ai4_pair_avx2 {

#if GGML_AI4_PAIR_AVX2

// 1024 entries x (int16 even level | int16 odd level << 16), from the int8 1024 x 2 ait2p_tab
struct pair_lut {
    alignas(64) int32_t packed[1024];
    pair_lut() {
        for (int e = 0; e < 1024; ++e) {
            const uint16_t w0 = (uint16_t) (int16_t) ait2p_tab[2*e];
            const uint16_t w1 = (uint16_t) (int16_t) ait2p_tab[2*e + 1];
            packed[e] = (int32_t) ((uint32_t) w0 | ((uint32_t) w1 << 16));
        }
    }
};

static inline const pair_lut & lut() {
    static const pair_lut t;
    return t;
}

// The four pair windows of tile row r (pairs 4r .. 4r + 3, b0 = 16r + 4j - 16 mod 256) all lie in the 32 stream bits
// [16r - 16, 16r + 16), at right shifts 12, 8, 4, 0: word r/2 for odd r, else the low half of word r/2 - 1 (circular)
// followed by the high half of word r/2.
static inline uint32_t row_win32(const uint32_t * wt, int r) {
    if (r & 1) {
        return wt[r >> 1];
    }
    const int m = r >> 1;
    return (wt[(m - 1) & 7] << 16) | (wt[m] >> 16);
}

// rows r2 and r2 + 1 -> 16 int16 levels: low 128-bit lane = row r2 (weights 0..7), high lane = row r2 + 1
static inline __m256i decode_row2(const uint32_t * wt, const pair_lut & T, int r2) {
    const __m256i win = _mm256_set_m128i(_mm_set1_epi32((int) row_win32(wt, r2 + 1)), _mm_set1_epi32((int) row_win32(wt, r2)));
    const __m256i w   = _mm256_and_si256(_mm256_srlv_epi32(win, _mm256_setr_epi32(12, 8, 4, 0, 12, 8, 4, 0)),
                                         _mm256_set1_epi32(0xFFFFF));
    // ait2p_entry(w) = ((w * (w + 1)) >> 10) & 1023, low 32 bits of the product as in the scalar decoder
    const __m256i e   = _mm256_and_si256(_mm256_srli_epi32(_mm256_mullo_epi32(w, _mm256_add_epi32(w, _mm256_set1_epi32(1))), 10),
                                         _mm256_set1_epi32(1023));
    return _mm256_i32gather_epi32(T.packed, e, 4);
}

static inline float hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

// one 16-row AIT2P_U slab (K/128 units of AI4U_UNIT_BYTES(2)) against NT rotated inputs: o[t][r] = row r . x_t
template<int NT>
static inline void slab_dot_nt(const uint8_t * slab, int64_t K, const float * const * x, float * const * o) {
    constexpr int ub = AI4U_UNIT_BYTES(2), tw = AIK_TILE_WORDS(2);
    const pair_lut & T = lut();
    float acc[NT][AI4_TR] = {};
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = slab + u*ub;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
        __m256 part[NT][AI4_TR];
        for (int t = 0; t < NT; ++t) {
            for (int r = 0; r < AI4_TR; ++r) { part[t][r] = _mm256_setzero_ps(); }
        }
        for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
            const uint32_t * wt = wd + ct*tw;
            __m256 xv[NT];
            for (int t = 0; t < NT; ++t) { xv[t] = _mm256_loadu_ps(x[t] + u*AI4U_UK + ct*AI4_TC); }
            for (int r = 0; r < AI4_TR; r += 2) {
                const __m256i p16 = decode_row2(wt, T, r);
                const __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16)));
                const __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16, 1)));
                for (int t = 0; t < NT; ++t) {
                    part[t][r]     = _mm256_fmadd_ps(l0, xv[t], part[t][r]);
                    part[t][r + 1] = _mm256_fmadd_ps(l1, xv[t], part[t][r + 1]);
                }
            }
        }
        for (int r = 0; r < AI4_TR; ++r) {
            const float d = ai4_fp16_to_fp32(dh[r]);
            for (int t = 0; t < NT; ++t) { acc[t][r] = std::fma(d, hsum8(part[t][r]), acc[t][r]); }
        }
    }
    for (int t = 0; t < NT; ++t) { memcpy(o[t], acc[t], sizeof(float)*AI4_TR); }
}

// nt in [1, 8]
static inline void slab_dot(const uint8_t * slab, int64_t K, int nt, const float * const * x, float * const * o) {
    switch (nt) {
        case 1: slab_dot_nt<1>(slab, K, x, o); break;
        case 2: slab_dot_nt<2>(slab, K, x, o); break;
        case 3: slab_dot_nt<3>(slab, K, x, o); break;
        case 4: slab_dot_nt<4>(slab, K, x, o); break;
        case 5: slab_dot_nt<5>(slab, K, x, o); break;
        case 6: slab_dot_nt<6>(slab, K, x, o); break;
        case 7: slab_dot_nt<7>(slab, K, x, o); break;
        case 8: slab_dot_nt<8>(slab, K, x, o); break;
        default: break;
    }
}

// rotated-basis levels of one AIT2P_U slab (dst[16][K], scale applied), bit-exact to aik_dequantize_slab_rotated_c(2, 0, ...)
static inline void dequantize_slab_rotated(const uint8_t * slab, float * dst, int64_t K) {
    constexpr int ub = AI4U_UNIT_BYTES(2), tw = AIK_TILE_WORDS(2);
    const pair_lut & T = lut();
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = slab + u*ub;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
        for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
            for (int r = 0; r < AI4_TR; r += 2) {
                const __m256i p16 = decode_row2(wd + ct*tw, T, r);
                const __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16)));
                const __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16, 1)));
                _mm256_storeu_ps(dst + (int64_t) r*K + u*AI4U_UK + ct*AI4_TC,
                                 _mm256_mul_ps(_mm256_set1_ps(ai4_fp16_to_fp32(dh[r])), l0));
                _mm256_storeu_ps(dst + (int64_t) (r + 1)*K + u*AI4U_UK + ct*AI4_TC,
                                 _mm256_mul_ps(_mm256_set1_ps(ai4_fp16_to_fp32(dh[r + 1])), l1));
            }
        }
    }
}

#endif // GGML_AI4_PAIR_AVX2

} // namespace ai4_pair_avx2
