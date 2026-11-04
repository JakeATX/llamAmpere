#pragma once
// fork AIPFIX D2: host CPU dot products for AI+ unit experts (AITk_U, 16-row unit slabs, ggml-ai4.h) in the CPU
// expert split. The input row is rotated once into the weights' basis (sign + normalized FWHT per 128/256 block), so
// a slab row's dot product is sum over units of fp16 scale * sum(level * rotated x). Levels are decoded 8 K-columns
// at a time from a big-endian copy of each tile's bit stream. Same math as aik_dequantize_slab_c + f32 dot; the float
// summation order differs (valid-math change).

#include "../ggml-ai4.h"

#include <cstdint>
#include <cstdlib>
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

// AIPFAST: the same circular stream as tile_bytes, kept as 32-bit words: E[0] = last word, E[1..nw] = the tile,
// E[nw+1..nw+2] = its first two words again. window64(E, B) equals load_be64(ext + B) for that tile's ext. Reading
// whole words avoids the byte stores + 8-byte reload of tile_bytes, which x86 cannot forward from the store buffer
// (the reload waited for every byte store to retire, ~600 us per 640x2560 AIT2_U projection on the 9900X).
static inline void tile_words(const uint32_t * wd, int k, uint32_t * E) {
    const int nw = AIK_TILE_WORDS(k);
    E[0] = wd[nw - 1];
    memcpy(E + 1, wd, (size_t) nw * sizeof(uint32_t));
    E[nw + 1] = wd[0];
    E[nw + 2] = wd[1 % nw];
}

static inline uint64_t window64(const uint32_t * E, int B) {
    const int j = B >> 2, q = 8*(B & 3);
    const uint64_t hi = ((uint64_t) E[j] << 32) | E[j + 1];
    return (hi << q) | (((uint64_t) E[j + 2] << q) >> 32);
}

// levels of tile row r (weights n = 8r .. 8r + 7): state of n = the 16 stream bits ending at bit k*n + k
static inline void row_levels_scalar(const uint32_t * E, int k, uint32_t C, int r, int * lev) {
    const int b = k*8*r + k - 16 + 32;  // first state's first bit, shifted by the 32-bit front pad
    const uint64_t w = window64(E, b >> 3);
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
static inline __m256 row_levels_avx2(const uint32_t * E, int k, uint32_t C, int r) {
    const int b = k*8*r + k - 16 + 32;
    const int off = b & 7;
    const __m256i w = _mm256_set1_epi64x((long long) window64(E, b >> 3));
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

// AIPFAST: AVX2 shuffle decode (Zen 3 / any AVX2 host; the default AVX2 path). Row r's states sit at fixed ext bytes
// (window byte m of row r = ext byte k*r + 2 + m; state kk starts k*(kk+1) bits into that window), so one vpshufb per
// row gathers, for each state, the 3 big-endian stream bytes holding it straight from the tile words (E layout), one
// vpsrlvd + mask cuts the 16-bit state, and the level is formed exactly as in row_levels_avx2. Replaces the scalar
// 64-bit window build + GPR broadcast + two variable shifts + blend per row. Same integer levels, so the dot products
// are bit-identical to row_levels_avx2 (the test checks every row). GGML_MOE_AI_HOST_SHUF=0 keeps the old decode.
struct shuf_tabs {
    uint8_t idx[6][AI4_TR][32];   // [k][row]: per 128-bit lane, E byte (relative to the lane base) of each state byte
    uint8_t base[6][AI4_TR][2];   // [k][row]: lane base, in E words (low lane = states 0-3, high lane = states 4-7)
    uint32_t sh[6][8];            // [k][state]: right shift that puts the state in the low 16 bits
    uint8_t dir[6][AI4_TR][2];    // [k][row][lane]: 1 = every byte the lane uses is a tile word (E[1..nw]), so the lane can
                                  // be loaded straight from the weights; 2 = also its whole 16-byte load stays in the tile
    shuf_tabs() {
        memset(this, 0, sizeof(*this));
        for (int k = 2; k <= 5; ++k) {
            for (int kk = 0; kk < 8; ++kk) { sh[k][kk] = 8 - (k*(kk + 1) & 7); }
            for (int r = 0; r < AI4_TR; ++r) {
                const int Bw = k*r + 2;                                  // ext byte of window byte 0
                for (int h = 0; h < 2; ++h) {
                    const int j0 = (Bw + ((k*(4*h + 1)) >> 3)) >> 2;     // word holding the lane's first state byte
                    base[k][r][h] = (uint8_t) j0;
                    for (int s = 0; s < 4; ++s) {
                        const int kk = 4*h + s, m0 = (k*(kk + 1)) >> 3;
                        uint8_t * d = idx[k][r] + 16*h + 4*s;
                        for (int i = 0; i < 3; ++i) {                    // lane byte i = window byte m0 + 2 - i
                            const int P = Bw + m0 + 2 - i;
                            d[i] = (uint8_t) (4*(P >> 2) + 3 - (P & 3) - 4*j0);
                        }
                        d[3] = 0x80;
                    }
                    int lo = 1 << 30, hi = -1;
                    for (int s = 0; s < 4; ++s) {
                        const int kk = 4*h + s, m0 = (k*(kk + 1)) >> 3;
                        for (int i = 0; i < 3; ++i) { const int P = Bw + m0 + 2 - i; lo = P/4 < lo ? P/4 : lo; hi = P/4 > hi ? P/4 : hi; }
                    }
                    const int nw = AIK_TILE_WORDS(k);
                    dir[k][r][h] = (uint8_t) (lo >= 1 && hi <= nw ? (j0 + 3 <= nw ? 2 : 1) : 0);
                }
            }
        }
    }
};

static const shuf_tabs & shuf_tables() {
    static const shuf_tabs T;
    return T;
}

static bool use_shuf() {
    static const bool on = [] { const char * e = getenv("GGML_MOE_AI_HOST_SHUF"); return !(e && e[0] == '0'); }();
    return on;
}

// E: tile_words output with at least 4 readable words past E[nw + 2] (lane loads read whole 16-byte groups)
static inline __m256 row_levels_shuf(const uint32_t * E, const shuf_tabs & T, int k, __m256i sh, __m256i Cv, int r) {
    const __m256i src = _mm256_loadu2_m128i((const __m128i *) (E + T.base[k][r][1]), (const __m128i *) (E + T.base[k][r][0]));
    __m256i v = _mm256_shuffle_epi8(src, _mm256_loadu_si256((const __m256i *) T.idx[k][r]));
    v = _mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(0xFFFF));
    v = _mm256_mullo_epi32(v, Cv);
    v = _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    v = _mm256_srai_epi32(_mm256_sub_epi32(v, _mm256_set1_epi32(510)), 2);
    return _mm256_cvtepi32_ps(v);
}

// AIPFAST: same decode, but lanes whose bytes all lie inside the tile load straight from the weights (wd = the tile's
// words, E[j] = wd[j - 1] there) instead of from E. E is rebuilt by small stores every tile, and a 16-byte load over
// several of those stores cannot be forwarded from the store buffer, so every row waited for E's stores to retire.
// Only the 2-3 edge lanes per tile (circular wrap) still read E. last = the slab's final tile of a unit: there a
// lane may only over-read inside the tile (dir == 2), so the load never runs past the unit. Same bytes, same levels.
static inline __m256 row_levels_shuf_d(const uint32_t * E, const uint32_t * wd, int lim, const shuf_tabs & T, int k,
                                       __m256i sh, __m256i Cv, int r) {
    const int b0 = T.base[k][r][0], b1 = T.base[k][r][1];
    const uint32_t * p0 = T.dir[k][r][0] >= lim ? wd + b0 - 1 : E + b0;
    const uint32_t * p1 = T.dir[k][r][1] >= lim ? wd + b1 - 1 : E + b1;
    const __m256i src = _mm256_loadu2_m128i((const __m128i *) p1, (const __m128i *) p0);
    __m256i v = _mm256_shuffle_epi8(src, _mm256_loadu_si256((const __m256i *) T.idx[k][r]));
    v = _mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(0xFFFF));
    v = _mm256_mullo_epi32(v, Cv);
    v = _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    v = _mm256_srai_epi32(_mm256_sub_epi32(v, _mm256_set1_epi32(510)), 2);
    return _mm256_cvtepi32_ps(v);
}

static bool use_direct() {
    static const bool on = [] { const char * e = getenv("GGML_MOE_AI_HOST_DIRECT"); return !(e && e[0] == '0'); }();
    return on;
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

// AVX-512 (Zen 4/5 hosts, runtime-dispatched; the build targets Zen 3): tile rows r and r + 1 in one 512-bit decode.
// Row r + 1 starts k bytes after row r at the same bit offset. Per row the lanes, FMA order and final sum are the
// AVX2 path's, so the results are bit-identical to it.
#define MOE_AI_T512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))

static bool has_avx512() {
    // GGML_MOE_AI_HOST_AVX512=0 forces the AVX2 path (default 1: AVX-512 whenever the CPU has it)
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_AI_HOST_AVX512");
        return !(e && e[0] == '0') && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
    }();
    return on;
}

MOE_AI_T512 static inline __m512 row2_levels_512(const uint32_t * E, int k, uint32_t C, int r) {
    const int b = k*8*r + k - 16 + 32;
    const int off = b & 7;
    const __m512i w = _mm512_inserti64x4(_mm512_set1_epi64((long long) window64(E, b >> 3)),
                                         _mm256_set1_epi64x((long long) window64(E, (b >> 3) + k)), 1);
    const long long s0 = 48 - off;
    const __m512i sh_ev = _mm512_setr_epi64(s0, s0 - 2*k, s0 - 4*k, s0 - 6*k, s0, s0 - 2*k, s0 - 4*k, s0 - 6*k);
    const __m512i sh_od = _mm512_setr_epi64(s0 - k, s0 - 3*k, s0 - 5*k, s0 - 7*k, s0 - k, s0 - 3*k, s0 - 5*k, s0 - 7*k);
    const __m512i ev = _mm512_srlv_epi64(w, sh_ev);
    const __m512i od = _mm512_slli_epi64(_mm512_srlv_epi64(w, sh_od), 32);
    __m512i v = _mm512_and_si512(_mm512_mask_blend_epi32(0xAAAA, ev, od), _mm512_set1_epi32(0xFFFF));
    v = _mm512_mullo_epi32(v, _mm512_set1_epi32((int) C));
    v = _mm512_madd_epi16(_mm512_maddubs_epi16(v, _mm512_set1_epi8(1)), _mm512_set1_epi16(1));
    v = _mm512_srai_epi32(_mm512_sub_epi32(v, _mm512_set1_epi32(510)), 2);
    return _mm512_cvtepi32_ps(v);
}

// one unit (16 rows x 128 K) of slab_rows: acc[t][r] += scale_r * (row r . y_t)
MOE_AI_T512 static void unit_rows_512(const uint8_t * blk, int k, uint32_t C, int nt, const float * const * y, int64_t koff,
                                      float (*acc)[AI4_TR]) {
    const uint16_t * dh = (const uint16_t *) blk;
    const uint32_t * wd = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
    const int tw = AIK_TILE_WORDS(k);
    uint32_t E[AIK_TILE_WORDS(5) + 3];
    __m512 part[8][AI4_TR/2];
    for (int t = 0; t < nt; ++t) { for (int r = 0; r < AI4_TR/2; ++r) { part[t][r] = _mm512_setzero_ps(); } }
    for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
        tile_words(wd + ct*tw, k, E);
        __m512 yv[8];
        for (int t = 0; t < nt; ++t) {
            const __m512 h = _mm512_castps256_ps512(_mm256_loadu_ps(y[t] + koff + ct*AI4_TC));
            yv[t] = _mm512_shuffle_f32x4(h, h, 0x44);
        }
        for (int r = 0; r < AI4_TR/2; ++r) {
            const __m512 lev = row2_levels_512(E, k, C, 2*r);
            for (int t = 0; t < nt; ++t) { part[t][r] = _mm512_fmadd_ps(lev, yv[t], part[t][r]); }
        }
    }
    for (int r = 0; r < AI4_TR/2; ++r) {
        const float d0 = ai4_fp16_to_fp32(dh[2*r]), d1 = ai4_fp16_to_fp32(dh[2*r + 1]);
        for (int t = 0; t < nt; ++t) {
            acc[t][2*r]     = __builtin_fmaf(d0, hsum8(_mm512_castps512_ps256(part[t][r])), acc[t][2*r]);
            acc[t][2*r + 1] = __builtin_fmaf(d1, hsum8(_mm512_extractf32x8_ps(part[t][r], 1)), acc[t][2*r + 1]);
        }
    }
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
// AIPFAST: AVX-512 VBMI tile decode (Zen 4/5), 4 tile rows per step instead of 2. Row r's 64-bit window starts at
// stream byte k*r + 2 with bit offset k (b = 8kr + k + 16), so every gather index and shift is fixed per rate k:
// vpermi2b builds the big-endian windows of 4 rows straight from the tile's little-endian words (2 qword lanes per row),
// vpmultishiftqb cuts the 32 16-bit states, and the level (byte sum of state * C, mod 2^32) is formed in 16-bit lanes:
// low half = lo16(st * Cl), high half = hi16(st * Cl) + lo16(st * Ch). Same integer levels and the same per-row-pair
// FMA order as unit_rows_512, so results are bit-identical to it (and to the AVX2 path).
#define MOE_AI_T512V __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,fma,f16c")))

static bool has_vbmi() {
    // GGML_MOE_AI_HOST_VBMI=0 keeps the plain AVX-512 decode
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_AI_HOST_VBMI");
        return !(e && e[0] == '0') && has_avx512() && __builtin_cpu_supports("avx512vbmi");
    }();
    return on;
}

struct vbmi_tabs {
    uint8_t idx[6][4][64];   // [k][row group]: tile byte for each window byte (vpermi2b over 128 tile bytes)
    uint8_t ctl[6][64];      // [k]: vpmultishiftqb bit offsets (2 per 16-bit state)
    vbmi_tabs() {
        memset(this, 0, sizeof(*this));
        for (int k = 2; k <= 5; ++k) {
            const int nw = AIK_TILE_WORDS(k);
            for (int g = 0; g < 4; ++g) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int r = 4*g + lane/2;
                    for (int m = 0; m < 8; ++m) {
                        const int P = k*r + 2 + 7 - m;            // ext byte (tile_bytes layout) of window byte m
                        const int j = P >> 2, q = P & 3;          // E word, big-endian byte in it
                        const int wi = j == 0 ? nw - 1 : (j - 1) % nw;
                        idx[k][g][8*lane + m] = (uint8_t) (4*wi + 3 - q);
                    }
                }
            }
            for (int lane = 0; lane < 8; ++lane) {
                for (int s = 0; s < 4; ++s) {
                    const int kk = 4*(lane & 1) + s;
                    const int p = 48 - k - k*kk;                  // state kk = (w >> p) & 0xFFFF
                    ctl[k][8*lane + 2*s]     = (uint8_t) p;
                    ctl[k][8*lane + 2*s + 1] = (uint8_t) (p + 8);
                }
            }
        }
    }
};

MOE_AI_T512V static void unit_rows_512_vbmi(const uint8_t * blk, int k, uint32_t C, int nt, const float * const * y,
                                            int64_t koff, float (*acc)[AI4_TR]) {
    static const vbmi_tabs T;
    const uint16_t * dh = (const uint16_t *) blk;
    const uint8_t  * wb = blk + AI4U_SCALE_BYTES;
    const int tb = 4*AIK_TILE_WORDS(k);                      // tile bytes, 16k
    const __mmask64 m0 = tb >= 64 ? ~0ULL : (1ULL << tb) - 1;
    const __mmask64 m1 = tb > 64 ? (1ULL << (tb - 64)) - 1 : 0;
    const __m512i ctl = _mm512_loadu_si512(T.ctl[k]);
    const __m512i cl = _mm512_set1_epi16((short) (C & 0xFFFF)), ch = _mm512_set1_epi16((short) (C >> 16));
    const __m512i one8 = _mm512_set1_epi8(1), k510 = _mm512_set1_epi16(510);
    __m512i idx[4];
    for (int g = 0; g < 4; ++g) { idx[g] = _mm512_loadu_si512(T.idx[k][g]); }
    __m512 part[8][AI4_TR/2];
    for (int t = 0; t < nt; ++t) { for (int r = 0; r < AI4_TR/2; ++r) { part[t][r] = _mm512_setzero_ps(); } }
    for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
        const uint8_t * tp = wb + ct*tb;
        const __m512i t0 = _mm512_maskz_loadu_epi8(m0, tp);
        const __m512i t1 = _mm512_maskz_loadu_epi8(m1, tp + 64);
        __m512 yv[8];
        for (int t = 0; t < nt; ++t) {
            const __m512 h = _mm512_castps256_ps512(_mm256_loadu_ps(y[t] + koff + ct*AI4_TC));
            yv[t] = _mm512_shuffle_f32x4(h, h, 0x44);
        }
        for (int g = 0; g < 4; ++g) {
            const __m512i win = _mm512_permutex2var_epi8(t0, idx[g], t1);
            const __m512i st  = _mm512_multishift_epi64_epi8(ctl, win);
            const __m512i lo  = _mm512_mullo_epi16(st, cl);
            const __m512i hi  = _mm512_add_epi16(_mm512_mulhi_epu16(st, cl), _mm512_mullo_epi16(st, ch));
            const __m512i sum = _mm512_add_epi16(_mm512_maddubs_epi16(lo, one8), _mm512_maddubs_epi16(hi, one8));
            const __m512i lev = _mm512_srai_epi16(_mm512_sub_epi16(sum, k510), 2);
            const __m512 l0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(lev)));
            const __m512 l1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(lev, 1)));
            for (int t = 0; t < nt; ++t) {
                part[t][2*g]     = _mm512_fmadd_ps(l0, yv[t], part[t][2*g]);
                part[t][2*g + 1] = _mm512_fmadd_ps(l1, yv[t], part[t][2*g + 1]);
            }
        }
    }
    for (int r = 0; r < AI4_TR/2; ++r) {
        const float d0 = ai4_fp16_to_fp32(dh[2*r]), d1 = ai4_fp16_to_fp32(dh[2*r + 1]);
        for (int t = 0; t < nt; ++t) {
            acc[t][2*r]     = __builtin_fmaf(d0, hsum8(_mm512_castps512_ps256(part[t][r])), acc[t][2*r]);
            acc[t][2*r + 1] = __builtin_fmaf(d1, hsum8(_mm512_extractf32x8_ps(part[t][r], 1)), acc[t][2*r + 1]);
        }
    }
}
#endif

// rows [a, b) of one expert (a multiple of 16; b - a <= 16, rows past b are computed and dropped), nt tokens:
// o[t][row - a] = W[row] . x_t, with y[t] the rotated inputs from rotate()
// path: -1 = best available (VBMI if present, else AVX2), 0 = AVX2 (or scalar), 1 = AVX-512 (VBMI decode when the host has it),
// 2 = AVX-512 without VBMI,
// 3 = AVX2 with the old window decode (row_levels_avx2; test reference), 4 = AVX2 shuffle reading every lane from E
// (the shuffle decode before direct lane loads; GGML_MOE_AI_HOST_DIRECT=0 makes it the default)
static void slab_rows(const char * w, size_t row_bytes, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
                      const float * const * y, float * const * o, int path = -1) {
    const uint8_t * slab = (const uint8_t *) w + (size_t) a*row_bytes;
    const int ub = AI4U_UNIT_BYTES(k), tw = AIK_TILE_WORDS(k);
    float acc[8][AI4_TR] = {};
    uint32_t E[AIK_TILE_WORDS(5) + 3 + 4] = {};  // k <= 5; +4 words of slack for row_levels_shuf's 16-byte lane loads
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = slab + u*ub;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
#if defined(__AVX2__) && defined(__FMA__)
        // default (-1): VBMI decode where the host has it; AVX-512 hosts WITHOUT VBMI (Skylake-X/Cascade Lake class) take the
        // AVX2 shuffle path below, which beat the word-read AVX-512 decode on the 9900X (249-280 vs 363 us, airows1)
        if (C != 0 && (path == 1 || path == 2 || (path == -1 && has_vbmi())) && has_avx512()) {
            if (path != 2 && has_vbmi()) { unit_rows_512_vbmi(blk, k, C, nt, y, u*AI4U_UK, acc); }
            else { unit_rows_512(blk, k, C, nt, y, u*AI4U_UK, acc); }
            continue;
        }
        if (C != 0) {
            __m256 part[8][AI4_TR];
            for (int t = 0; t < nt; ++t) { for (int r = 0; r < AI4_TR; ++r) { part[t][r] = _mm256_setzero_ps(); } }
            const bool shuf = path != 3 && use_shuf();
            const bool direct = shuf && path != 4 && use_direct();
            const shuf_tabs & T = shuf_tables();
            const __m256i shv = _mm256_loadu_si256((const __m256i *) T.sh[k]), Cv = _mm256_set1_epi32((int) C);
            for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
                tile_words(wd + ct*tw, k, E);
                __m256 yv[8];
                for (int t = 0; t < nt; ++t) { yv[t] = _mm256_loadu_ps(y[t] + u*AI4U_UK + ct*AI4_TC); }
                const uint32_t * wdt = wd + ct*tw;
                const int lim = !direct ? 3 : (ct == AI4U_UK/AI4_TC - 1 ? 2 : 1);  // 3 = never direct
                for (int r = 0; r < AI4_TR; ++r) {
                    const __m256 lev = !shuf ? row_levels_avx2(E, k, C, r)
                                     : row_levels_shuf_d(E, wdt, lim, T, k, shv, Cv, r);
                    for (int t = 0; t < nt; ++t) { part[t][r] = _mm256_fmadd_ps(lev, yv[t], part[t][r]); }
                }
            }
            for (int r = 0; r < AI4_TR; ++r) {
                const float d = ai4_fp16_to_fp32(dh[r]);
                for (int t = 0; t < nt; ++t) { acc[t][r] = __builtin_fmaf(d, hsum8(part[t][r]), acc[t][r]); }
            }
            continue;
        }
#endif
        for (int r = 0; r < AI4_TR; ++r) {
            float part[8] = {};
            for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
                int lev[8];
                if (C != 0) {
                    tile_words(wd + ct*tw, k, E);
                    row_levels_scalar(E, k, C, r, lev);
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
