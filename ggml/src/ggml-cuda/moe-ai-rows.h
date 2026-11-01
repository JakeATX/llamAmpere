#pragma once
// fork AIPFIX D2: host CPU dot products for AI+ unit experts (AITk_U, 16-row unit slabs, ggml-ai4.h) in the CPU
// expert split. The input row is rotated once into the weights' basis (sign + normalized FWHT per 128/256 block), so
// a slab row's dot product is sum over units of fp16 scale * sum(level * rotated x). Levels are decoded 8 K-columns
// at a time from a big-endian copy of each tile's bit stream. Same math as aik_dequantize_slab_c + f32 dot; the float
// summation order differs (valid-math change).

#include "../ggml-ai4.h"

#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace moe_ai {

// ggml_from_float_t: x (k floats) -> y (k floats) in the AI+ rotated basis
static void rotate(const float * x, void * vy, int64_t k) {
    float * y = (float *) vy;
    const int rb = ai4u_rot_size(k);
    const float s = ai4_rot_norm(rb);
    for (int64_t b = 0; b < k; b += rb) {
        for (int i = 0; i < rb; ++i) {
            y[b + i] = x[b + i] * (ai4_sign_neg(i) ? -s : s);
        }
        ai4_fwht_n(y + b, rb);
    }
}

// tile bit stream (4k words, MSB first in each word, circular) -> bytes in stream order with the last 4 bytes in front
// and the first 8 bytes behind, so bit b of the stream (b >= -32) is bit 7 - (b + 32) % 8 of byte (b + 32) / 8.
static inline void tile_bytes(const uint32_t * wd, int k, uint8_t * ext) {
    const int nw = AIK_TILE_WORDS(k);
    auto put = [&](int at, uint32_t w) { ext[at] = w >> 24; ext[at + 1] = w >> 16; ext[at + 2] = w >> 8; ext[at + 3] = w; };
    put(0, wd[nw - 1]);
    for (int i = 0; i < nw; ++i) { put(4 + 4*i, wd[i]); }
    put(4 + 4*nw, wd[0]);
    put(8 + 4*nw, wd[1 % nw]);
}

static inline uint64_t load_be64(const uint8_t * p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return __builtin_bswap64(v);
}

// levels of tile row r (weights n = 8r .. 8r + 7): state of n = the 16 stream bits ending at bit k*n + k
static inline void row_levels_scalar(const uint8_t * ext, int k, uint32_t C, int r, int * lev) {
    const int b = k*8*r + k - 16 + 32;  // first state's first bit, shifted by the 32-bit front pad
    const uint64_t w = load_be64(ext + (b >> 3));
    const int off = b & 7;
    for (int kk = 0; kk < 8; ++kk) {
        const uint32_t st = (uint32_t) (w >> (64 - 16 - off - k*kk)) & 0xFFFFu;
        lev[kk] = aik_level_c(st, C);
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// 8 levels in one 32-bit lane each: even levels from one 64-bit shift, odd ones from another moved up 32 bits,
// state * C in 32-bit lanes, byte sum by two pairwise adds, ((sum + 2) >> 2) - 128 as (sum - 510) >> 2 (512 = 4 * 128).
// Same integer levels as aik_level_c, so the dot products are unchanged.
static inline __m256 row_levels_avx2(const uint8_t * ext, int k, uint32_t C, int r) {
    const int b = k*8*r + k - 16 + 32;
    const int off = b & 7;
    const __m256i w = _mm256_set1_epi64x((long long) load_be64(ext + (b >> 3)));
    const int s0 = 48 - off;
    const __m256i sh_ev = _mm256_setr_epi64x(s0, s0 - 2*k, s0 - 4*k, s0 - 6*k);
    const __m256i sh_od = _mm256_setr_epi64x(s0 - k, s0 - 3*k, s0 - 5*k, s0 - 7*k);
    const __m256i ev = _mm256_srlv_epi64(w, sh_ev);
    const __m256i od = _mm256_slli_epi64(_mm256_srlv_epi64(w, sh_od), 32);
    __m256i v = _mm256_and_si256(_mm256_blend_epi32(ev, od, 0xAA), _mm256_set1_epi32(0xFFFF));
    v = _mm256_mullo_epi32(v, _mm256_set1_epi32((int) C));
    v = _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    v = _mm256_srai_epi32(_mm256_sub_epi32(v, _mm256_set1_epi32(510)), 2);
    return _mm256_cvtepi32_ps(v);
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
#endif

// rows [a, b) of one expert (a multiple of 16; b - a <= 16, rows past b are computed and dropped), nt tokens:
// o[t][row - a] = W[row] . x_t, with y[t] the rotated inputs from rotate()
static void slab_rows(const char * w, size_t row_bytes, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
                      const float * const * y, float * const * o) {
    const uint8_t * slab = (const uint8_t *) w + (size_t) a*row_bytes;
    const int ub = AI4U_UNIT_BYTES(k), tw = AIK_TILE_WORDS(k);
    float acc[8][AI4_TR] = {};
    uint8_t ext[12 + 16*5 + 4];  // k <= 5
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = slab + u*ub;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
#if defined(__AVX2__) && defined(__FMA__)
        if (C != 0) {
            __m256 part[8][AI4_TR];
            for (int t = 0; t < nt; ++t) { for (int r = 0; r < AI4_TR; ++r) { part[t][r] = _mm256_setzero_ps(); } }
            for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
                tile_bytes(wd + ct*tw, k, ext);
                __m256 yv[8];
                for (int t = 0; t < nt; ++t) { yv[t] = _mm256_loadu_ps(y[t] + u*AI4U_UK + ct*AI4_TC); }
                for (int r = 0; r < AI4_TR; ++r) {
                    const __m256 lev = row_levels_avx2(ext, k, C, r);
                    for (int t = 0; t < nt; ++t) { part[t][r] = _mm256_fmadd_ps(lev, yv[t], part[t][r]); }
                }
            }
            for (int r = 0; r < AI4_TR; ++r) {
                const float d = ai4_fp16_to_fp32(dh[r]);
                for (int t = 0; t < nt; ++t) { acc[t][r] += d*hsum8(part[t][r]); }
            }
            continue;
        }
#endif
        for (int r = 0; r < AI4_TR; ++r) {
            float part[8] = {};
            for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
                int lev[8];
                if (C != 0) {
                    tile_bytes(wd + ct*tw, k, ext);
                    row_levels_scalar(ext, k, C, r, lev);
                } else {
                    for (int kk = 0; kk < 8; ++kk) { lev[kk] = aik_level_n(wd + ct*tw, k, 0, r*AI4_TC + kk); }
                }
                for (int t = 0; t < nt; ++t) {
                    const float * yt = y[t] + u*AI4U_UK + ct*AI4_TC;
                    for (int kk = 0; kk < 8; ++kk) { part[t] += (float) lev[kk]*yt[kk]; }
                }
            }
            const float d = ai4_fp16_to_fp32(dh[r]);
            for (int t = 0; t < nt; ++t) { acc[t][r] += d*part[t]; }
        }
    }
    for (int t = 0; t < nt; ++t) {
        for (int64_t row = a; row < b; ++row) { o[t][row - a] = acc[t][row - a]; }
    }
}

} // namespace moe_ai
