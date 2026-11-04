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
#include <type_traits>
#include <utility>

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

// ---- AIPFIX: opt-in int8-activation host path (GGML_MOE_AI_HOST_Q8=1, default off) ----
// The rotated input row is quantized once per token the way the 3XS host rows quantize theirs: Q8_K blocks of 256 when
// K % 256 == 0 (gate/up: float d, q = min(127, nearest(-127/max * x)), as quantize_row_q8_K_ref) else Q8_0 blocks of 32
// (down, K = 640: d = fp16(amax/127), q = round(x * 127/amax), as quantize_row_q8_0_ref). Weight levels are exact
// integers in [-128, 127], so each block's level x q dot is an exact int32, and a row's result is
// sum over units of fp16 scale * (sum over the unit's blocks of d * int dot). This CHANGES output vs the float path
// (activation rounding, the same class of rounding 3XS has on its int8 activations).
// One token's buffer (q8_bytes): int16 xs[K/8][16] (tile c: x[8c..8c+3] twice, then x[8c+4..8c+7] twice, the lane order
// of a packed 2-row level pair), then float d[K/qb].
static inline int q8_block(int64_t K) { return K % 256 == 0 ? 256 : 32; }
static inline size_t q8_bytes(int64_t K) { return (size_t) K*4 + (size_t) (K/32)*sizeof(float); }

static bool use_host_q8() {
    static const bool on = [] { const char * e = getenv("GGML_MOE_AI_HOST_Q8"); return e && e[0] == '1'; }();
    return on;
}

static inline int q8_nearest_int(float fval) {   // ggml nearest_int (round half to even), |fval| <= 2^22
    const float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline uint16_t q8_fp32_to_fp16(float f) {   // ggml's portable fp32 -> fp16 (round to nearest even)
    auto to_bits = [](float x) { uint32_t u; memcpy(&u, &x, 4); return u; };
    auto from_bits = [](uint32_t u) { float x; memcpy(&x, &u, 4); return x; };
    float base = (__builtin_fabsf(f) * 0x1.0p+112f) * 0x1.0p-110f;
    const uint32_t w = to_bits(f), shl1_w = w + w, sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) { bias = 0x71000000u; }
    base = from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = to_bits(base);
    const uint32_t nonsign = ((bits >> 13) & 0x00007C00u) + (bits & 0x00000FFFu);
    return (uint16_t) ((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// y: K rotated floats (rotate() output) -> one token's q8 buffer
static void quant_q8(const float * y, void * vq, int64_t K) {
    int16_t * xs = (int16_t *) vq;
    float * d = (float *) ((uint8_t *) vq + (size_t) K*4);
    const int qb = q8_block(K);
    int8_t q[256];
    for (int64_t b = 0; b < K/qb; ++b) {
        const float * x = y + b*qb;
        if (qb == 256) {
            float amax = 0.0f, mx = 0.0f;
            for (int j = 0; j < qb; ++j) { const float ax = __builtin_fabsf(x[j]); if (ax > amax) { amax = ax; mx = x[j]; } }
            if (amax == 0.0f) {
                d[b] = 0.0f;
                memset(q, 0, sizeof(q));
            } else {
                const float iscale = -127.f/mx;
                for (int j = 0; j < qb; ++j) { const int v = q8_nearest_int(iscale*x[j]); q[j] = (int8_t) (v < 127 ? v : 127); }
                d[b] = 1/iscale;
            }
        } else {
            float amax = 0.0f;
            for (int j = 0; j < qb; ++j) { const float ax = __builtin_fabsf(x[j]); amax = ax > amax ? ax : amax; }
            const float dd = amax/127.f, id = dd != 0.0f ? 1.0f/dd : 0.0f;
            for (int j = 0; j < qb; ++j) { q[j] = (int8_t) __builtin_roundf(x[j]*id); }
            d[b] = ai4_fp16_to_fp32(q8_fp32_to_fp16(dd));
        }
        for (int j = 0; j < qb; j += AI4_TC) {
            int16_t * o = xs + (size_t) (b*qb + j)*2;
            for (int i = 0; i < 4; ++i) { o[i] = o[4 + i] = q[j + i]; o[8 + i] = o[12 + i] = q[j + 4 + i]; }
        }
    }
}

// ggml_from_float_t for the Q8 path: rotate into the weights' basis, then quantize (K <= 8192)
static void rotate_q8(const float * x, void * vq, int64_t k) {
    float y[8192];
    rotate(x, y, k);
    quant_q8(y, vq, k);
}

static inline const int16_t * q8_xs(const void * q) { return (const int16_t *) q; }
static inline const float * q8_d(const void * q, int64_t K) { return (const float *) ((const uint8_t *) q + (size_t) K*4); }

// scalar reference of the Q8 path (and the non-AVX2 fallback): same integer dots, same float order as the AVX2 kernel
static void unit_rows_q8_scalar(const uint8_t * blk, int k, uint32_t C, int nt, const void * const * xq, int64_t koff,
                                int64_t K, float (*acc)[AI4_TR]) {
    const uint16_t * dh = (const uint16_t *) blk;
    const uint32_t * wd = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
    const int tw = AIK_TILE_WORDS(k), qb = q8_block(K), st = (qb < AI4U_UK ? qb : AI4U_UK)/AI4_TC;
    uint32_t E[AIK_TILE_WORDS(5) + 3 + 4] = {};
    for (int r = 0; r < AI4_TR; ++r) {
        float rs[8] = {};
        for (int c0 = 0; c0 < AI4U_UK/AI4_TC; c0 += st) {
            int32_t is[8] = {};
            for (int ct = c0; ct < c0 + st; ++ct) {
                int lev[8];
                if (C != 0) {
                    tile_words(wd + ct*tw, k, E);
                    row_levels_scalar(E, k, C, r, lev);
                } else {
                    for (int kk = 0; kk < 8; ++kk) { lev[kk] = aik_level_n(wd + ct*tw, k, 0, r*AI4_TC + kk); }
                }
                for (int t = 0; t < nt; ++t) {
                    const int16_t * xt = q8_xs(xq[t]) + (koff/AI4_TC + ct)*16;
                    for (int kk = 0; kk < 8; ++kk) { is[t] += lev[kk]*xt[(kk < 4 ? 0 : 8) + (kk & 3)]; }
                }
            }
            // exact int32 row sum over the block, then one fma per block (rs starts at +0)
            for (int t = 0; t < nt; ++t) { rs[t] = __builtin_fmaf((float) is[t], q8_d(xq[t], K)[(koff + c0*AI4_TC)/qb], rs[t]); }
        }
        const float dw = ai4_fp16_to_fp32(dh[r]);
        for (int t = 0; t < nt; ++t) { acc[t][r] = __builtin_fmaf(dw, rs[t], acc[t][r]); }
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

// AIPFIX: AVX2 direct decode (default AVX2 path, e.g. the Zen 3 home host). row_levels_shuf's two 16-byte lane loads
// come from a per-tile tile_words copy: a memcpy of k-dependent size plus loads that span several of its 4-byte stores,
// which x86 cannot forward from the store buffer. Here all 24 stream bytes a tile row needs lie in 4 consecutive E words
// E[S..S+3] (S fixed per rate k and row r, both compile-time), so each row does ONE 16-byte load broadcast to both
// lanes, straight from the tile in the weight buffer (E word j = tile word j - 1), and vpshufb picks each state's
// 3 bytes from it. Only row 0 (S = 0, E[0..3] == E[nw..nw+3]) and row 15 (S = nw - 2) touch the circular wrap; their
// windows are cut once per tile from the tile's first and last 4 words with one alignr each. No read leaves the tile.
// Rows run in groups sized so the accumulators stay in registers; every (token, row) still sums tiles 0..15 in order
// with the same FMA and the same integer levels, so outputs are bit-identical to the shuffle path (the test checks).
// GGML_MOE_AI_HOST_ROWWIN=0 keeps the shuffle path.
static constexpr int direct_word(int k, int r, int kk, int i) {   // E word holding byte i (0 = lowest) of state kk's lane
    return (k*r + 2 + ((k*(kk + 1)) >> 3) + 2 - i) >> 2;
}

static constexpr int direct_lo(int k, int r) {
    int lo = 1 << 20;
    for (int kk = 0; kk < 8; ++kk) { for (int i = 0; i < 3; ++i) { lo = direct_word(k, r, kk, i) < lo ? direct_word(k, r, kk, i) : lo; } }
    return lo;
}

static constexpr int direct_hi(int k, int r) {
    int hi = 0;
    for (int kk = 0; kk < 8; ++kk) { for (int i = 0; i < 3; ++i) { hi = direct_word(k, r, kk, i) > hi ? direct_word(k, r, kk, i) : hi; } }
    return hi;
}

// first E word S of row r's 4-word window
static constexpr int direct_S(int k, int r) {
    const int nw = AIK_TILE_WORDS(k), lo = direct_lo(k, r), hi = direct_hi(k, r);
    if (lo >= 1 && hi <= nw) { return lo < nw - 3 ? lo : nw - 3; }   // inside the tile: load tile words S-1 .. S+2
    return lo == 0 ? 0 : hi - 3;                                       // wrap: S = 0 or nw-2 / nw-1
}

// window source: >= 0 = tile word offset of the 16-byte load; -m (m = 1..3) = alignr(first4, last4, 4m) = E[nw-3+m ..]
static constexpr int direct_src(int k, int r) {
    const int nw = AIK_TILE_WORDS(k), S = direct_S(k, r);
    if (S >= 1 && S <= nw - 3) { return S - 1; }
    return -(S == 0 ? 3 : S - nw + 3);
}

struct direct_tabs {
    alignas(32) uint8_t idx[6][AI4_TR][32];   // per lane: window byte of each state byte (0x80 = zero)
    bool ok = true;                           // layout self-check (every row fits its window; wrap rows as above)
    direct_tabs() {
        memset(idx, 0x80, sizeof(idx));
        for (int k = 2; k <= 5; ++k) {
            const int nw = AIK_TILE_WORDS(k);
            for (int r = 0; r < AI4_TR; ++r) {
                const int S = direct_S(k, r), src = direct_src(k, r), Sc = S == 0 ? 0 : S;
                ok = ok && direct_hi(k, r) - direct_lo(k, r) <= 3 && direct_lo(k, r) >= Sc && direct_hi(k, r) <= Sc + 3 &&
                     (src >= 0 ? src + 4 <= nw : (src >= -3 && (S == 0 || S == nw - 3 - src)));
                for (int kk = 0; kk < 8; ++kk) {
                    const int m0 = (k*(kk + 1)) >> 3;
                    uint8_t * d = idx[k][r] + 16*(kk >> 2) + 4*(kk & 3);
                    for (int i = 0; i < 3; ++i) {                    // lane byte i = window byte m0 + 2 - i
                        const int P = k*r + 2 + m0 + 2 - i;
                        d[i] = (uint8_t) (4*((P >> 2) - Sc) + 3 - (P & 3));
                    }
                }
            }
        }
    }
};

static const direct_tabs & direct_tables() {
    static const direct_tabs T;
    return T;
}

static bool use_rowwin() {
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_AI_HOST_ROWWIN");
        return !(e && e[0] == '0') && direct_tables().ok;
    }();
    return on;
}

// byte-pair sums of state * C for the 8 states of tile row R (two int16 per 32-bit lane, one lane per column)
template <int K, int R>
static inline __attribute__((always_inline)) __m256i row_bytepair_direct(const uint32_t * wt, __m128i f4, __m128i l4,
                                                                        const direct_tabs & T, __m256i sh, __m256i Cv) {
    constexpr int src = direct_src(K, R);
    __m256i v;
    if constexpr (src >= 0) {
        v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) (wt + src)));
    } else {
        v = _mm256_broadcastsi128_si256(_mm_alignr_epi8(f4, l4, -4*src));
    }
    v = _mm256_shuffle_epi8(v, _mm256_load_si256((const __m256i *) T.idx[K][R]));
    v = _mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(0xFFFF));
    v = _mm256_mullo_epi32(v, Cv);
    return _mm256_maddubs_epi16(v, _mm256_set1_epi8(1));
}

// byte sum S of state * C for the 8 states of tile row R (level = (S - 510) >> 2), one 32-bit lane per column
template <int K, int R>
static inline __attribute__((always_inline)) __m256i row_bytesum_direct(const uint32_t * wt, __m128i f4, __m128i l4,
                                                                        const direct_tabs & T, __m256i sh, __m256i Cv) {
    return _mm256_madd_epi16(row_bytepair_direct<K, R>(wt, f4, l4, T, sh, Cv), _mm256_set1_epi16(1));
}

template <int K, int R>
static inline __attribute__((always_inline)) __m256 row_levels_direct(const uint32_t * wt, __m128i f4, __m128i l4,
                                                                      const direct_tabs & T, __m256i sh, __m256i Cv) {
    const __m256i v = row_bytesum_direct<K, R>(wt, f4, l4, T, sh, Cv);
    return _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_sub_epi32(v, _mm256_set1_epi32(510)), 2));
}

template <int K, int NT, int G, int R0, size_t... I>
static inline __attribute__((always_inline)) void direct_rows(const uint32_t * wt, __m128i f4, __m128i l4, const direct_tabs & T,
                                                              __m256i sh, __m256i Cv, const __m256 (&yv)[NT],
                                                              __m256 (&part)[NT][G], std::index_sequence<I...>) {
    auto row = [&](auto gi) {
        constexpr int g = decltype(gi)::value;
        const __m256 lev = row_levels_direct<K, R0 + g>(wt, f4, l4, T, sh, Cv);
        for (int t = 0; t < NT; ++t) { part[t][g] = _mm256_fmadd_ps(lev, yv[t], part[t][g]); }
    };
    (row(std::integral_constant<int, (int) I>{}), ...);
}

template <int K, int NT, int G, int R0>
static inline __attribute__((always_inline)) void direct_group(const uint8_t * blk, const float * const * y, int64_t koff,
                                                               __m256i Cv, float (*acc)[AI4_TR]) {
    constexpr int tw = AIK_TILE_WORDS(K);
    const direct_tabs & T = direct_tables();
    const __m256i sh = _mm256_loadu_si256((const __m256i *) shuf_tables().sh[K]);
    const uint32_t * wd = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
    __m256 part[NT][G];
    for (int t = 0; t < NT; ++t) { for (int g = 0; g < G; ++g) { part[t][g] = _mm256_setzero_ps(); } }
    for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
        const uint32_t * wt = wd + ct*tw;
        const __m128i f4 = _mm_loadu_si128((const __m128i *) wt), l4 = _mm_loadu_si128((const __m128i *) (wt + tw - 4));
        __m256 yv[NT];
        for (int t = 0; t < NT; ++t) { yv[t] = _mm256_loadu_ps(y[t] + koff + ct*AI4_TC); }
        direct_rows<K, NT, G, R0>(wt, f4, l4, T, sh, Cv, yv, part, std::make_index_sequence<G>{});
    }
    const uint16_t * dh = (const uint16_t *) blk;
    for (int g = 0; g < G; ++g) {
        const float d = ai4_fp16_to_fp32(dh[R0 + g]);
        for (int t = 0; t < NT; ++t) { acc[t][R0 + g] = __builtin_fmaf(d, hsum8(part[t][g]), acc[t][R0 + g]); }
    }
}

// one unit (16 rows x 128 K): acc[t][r] += scale_r * (row r . y_t), rows in groups of G (NT * G accumulators <= 8)
template <int K, int NT, size_t... J>
static void unit_rows_direct_impl(const uint8_t * blk, const float * const * y, int64_t koff, __m256i Cv, float (*acc)[AI4_TR],
                                  std::index_sequence<J...>) {
    constexpr int G = AI4_TR/(int) sizeof...(J);
    (direct_group<K, NT, G, (int) J*G>(blk, y, koff, Cv, acc), ...);
}

template <int K, int NT>
static void unit_rows_direct_nt(const uint8_t * blk, const float * const * y, int64_t koff, __m256i Cv, float (*acc)[AI4_TR]) {
    constexpr int G = NT == 1 ? 8 : NT == 2 ? 4 : NT <= 4 ? 2 : 1;
    unit_rows_direct_impl<K, NT>(blk, y, koff, Cv, acc, std::make_index_sequence<AI4_TR/G>{});
}

template <int K>
static void unit_rows_direct_k(const uint8_t * blk, int nt, const float * const * y, int64_t koff, __m256i Cv, float (*acc)[AI4_TR]) {
    switch (nt) {
        case 1:  unit_rows_direct_nt<K, 1>(blk, y, koff, Cv, acc); break;
        case 2:  unit_rows_direct_nt<K, 2>(blk, y, koff, Cv, acc); break;
        case 3:  unit_rows_direct_nt<K, 3>(blk, y, koff, Cv, acc); break;
        case 4:  unit_rows_direct_nt<K, 4>(blk, y, koff, Cv, acc); break;
        case 5:  unit_rows_direct_nt<K, 5>(blk, y, koff, Cv, acc); break;
        case 6:  unit_rows_direct_nt<K, 6>(blk, y, koff, Cv, acc); break;
        case 7:  unit_rows_direct_nt<K, 7>(blk, y, koff, Cv, acc); break;
        default: unit_rows_direct_nt<K, 8>(blk, y, koff, Cv, acc); break;
    }
}

// k in 2..5; returns false (nothing done) for other rates
static bool unit_rows_direct(const uint8_t * blk, int k, uint32_t C, int nt, const float * const * y, int64_t koff,
                             float (*acc)[AI4_TR]) {
    const __m256i Cv = _mm256_set1_epi32((int) C);
    switch (k) {
        case 2: unit_rows_direct_k<2>(blk, nt, y, koff, Cv, acc); return true;
        case 3: unit_rows_direct_k<3>(blk, nt, y, koff, Cv, acc); return true;
        case 4: unit_rows_direct_k<4>(blk, nt, y, koff, Cv, acc); return true;
        case 5: unit_rows_direct_k<5>(blk, nt, y, koff, Cv, acc); return true;
        default: return false;
    }
}

// AIPFIX Q8 path, AVX2: rows decoded as in unit_rows_direct; two rows' byte-pair sums joined by vphaddw (= vpackssdw of
// their byte sums, S <= 1020), level (S - 510) >> 2 in int16, then one vpmaddwd per token against the tile's int16
// activations gives exact int32 pair sums. Per quant block (256 wide: one per unit; 32 wide: four per unit) the int32
// lanes are reduced exactly to row sums with vector adds (vphaddd + 128-bit half add), converted once, and
// rs = fma(row sum, d, rs) per block; then acc = fma(fp16 scale, rs, acc) per unit. unit_rows_q8_scalar does the same
// float steps in the same order, so the two are bit-identical. (r30: the reduce used to be scalar per token and row
// pair, and the 32-wide case kept a float lane per column pair live next to the int lanes, which spilled at 5-8 tokens.)
template <int K, int NT, int G, int R0, size_t... P>
static inline __attribute__((always_inline)) void q8_pairs(const uint32_t * wt, __m128i f4, __m128i l4, const direct_tabs & T,
                                                           __m256i sh, __m256i Cv, const int16_t * const (&xt)[NT],
                                                           __m256i (&ip)[NT][G/2], std::index_sequence<P...>) {
    auto pair = [&](auto pi) {
        constexpr int p = decltype(pi)::value;
#if !defined(MOE_AI_Q8_PACKS)
        const __m256i m0 = row_bytepair_direct<K, R0 + 2*p>(wt, f4, l4, T, sh, Cv);
        const __m256i m1 = row_bytepair_direct<K, R0 + 2*p + 1>(wt, f4, l4, T, sh, Cv);
        const __m256i s01 = _mm256_hadd_epi16(m0, m1);
#else   // bench reference: two vpmaddwd + vpackssdw
        const __m256i s0 = row_bytesum_direct<K, R0 + 2*p>(wt, f4, l4, T, sh, Cv);
        const __m256i s1 = row_bytesum_direct<K, R0 + 2*p + 1>(wt, f4, l4, T, sh, Cv);
        const __m256i s01 = _mm256_packs_epi32(s0, s1);
#endif
        const __m256i lev = _mm256_srai_epi16(_mm256_sub_epi16(s01, _mm256_set1_epi16(510)), 2);
        for (int t = 0; t < NT; ++t) {
            ip[t][p] = _mm256_add_epi32(ip[t][p], _mm256_madd_epi16(lev, _mm256_loadu_si256((const __m256i *) xt[t])));
        }
    };
    (pair(std::integral_constant<int, (int) P>{}), ...);
}

// pair lanes: [r0 c01, r0 c23, r1 c01, r1 c23 | r0 c45, r0 c67, r1 c45, r1 c67] (int32, exact)
static inline __attribute__((always_inline)) __m128i q8_half_add(__m256i v) {
    return _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
}

// dd[b][t][0] = dd[b][t][1] = token t's scale of the unit's quant block b (a token pair's [d0 d0 d1 d1] is one load)
template <int K, int NT, int G, int R0, int ST>
static inline __attribute__((always_inline)) void q8_group(const uint8_t * blk, const void * const * xq, int64_t koff,
                                                           const float (*dd)[8][2], __m256i Cv, float (*acc)[AI4_TR]) {
    constexpr int tw = AIK_TILE_WORDS(K);
    // float accumulators: G >= 8: one __m256 per token per 8 rows; G == 4: one __m128 per token; G == 2: one per token pair
    constexpr int N8 = G >= 8 ? NT*(G/8) : 1, N4 = G == 4 ? NT : G == 2 ? (NT + 1)/2 : 1;
    const direct_tabs & T = direct_tables();
    const __m256i sh = _mm256_loadu_si256((const __m256i *) shuf_tables().sh[K]);
    const uint32_t * wd = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
    __m256 F8[N8];
    __m128 F4[N4];
    for (int i = 0; i < N8; ++i) { F8[i] = _mm256_setzero_ps(); }
    for (int i = 0; i < N4; ++i) { F4[i] = _mm_setzero_ps(); }
    for (int c0 = 0; c0 < AI4U_UK/AI4_TC; c0 += ST) {
        __m256i ip[NT][G/2];
        for (int t = 0; t < NT; ++t) { for (int p = 0; p < G/2; ++p) { ip[t][p] = _mm256_setzero_si256(); } }
        for (int ct = c0; ct < c0 + ST; ++ct) {
            const uint32_t * wt = wd + ct*tw;
            const __m128i f4 = _mm_loadu_si128((const __m128i *) wt), l4 = _mm_loadu_si128((const __m128i *) (wt + tw - 4));
            const int16_t * xt[NT];
            for (int t = 0; t < NT; ++t) { xt[t] = q8_xs(xq[t]) + (koff/AI4_TC + ct)*16; }
            q8_pairs<K, NT, G, R0>(wt, f4, l4, T, sh, Cv, xt, ip, std::make_index_sequence<G/2>{});
        }
        const float (*db)[2] = dd[c0/ST];
        if constexpr (G >= 8) {
            for (int t = 0; t < NT; ++t) {
                for (int q = 0; q < G/8; ++q) {
                    const __m256i x01 = _mm256_hadd_epi32(ip[t][4*q], ip[t][4*q + 1]);   // [r0..r3 c0-3 | r0..r3 c4-7]
                    const __m256i x23 = _mm256_hadd_epi32(ip[t][4*q + 2], ip[t][4*q + 3]);
                    const __m256i z = _mm256_add_epi32(_mm256_permute2x128_si256(x01, x23, 0x20),
                                                       _mm256_permute2x128_si256(x01, x23, 0x31));   // rows 0..7
                    F8[t*(G/8) + q] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(z), _mm256_broadcast_ss(&db[t][0]), F8[t*(G/8) + q]);
                }
            }
        } else if constexpr (G == 4) {
            for (int t = 0; t < NT; ++t) {
                const __m128i z = q8_half_add(_mm256_hadd_epi32(ip[t][0], ip[t][1]));   // rows 0..3
                F4[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(z), _mm_broadcast_ss(&db[t][0]), F4[t]);
            }
        } else {
            static_assert(G == 2, "q8 rows per group");
            for (int tp = 0; tp < N4; ++tp) {
                const __m256i i1 = 2*tp + 1 < NT ? ip[2*tp + 1 < NT ? 2*tp + 1 : 0][0] : _mm256_setzero_si256();
                const __m128i z = q8_half_add(_mm256_hadd_epi32(ip[2*tp][0], i1));   // [r0 t0, r1 t0, r0 t1, r1 t1]
                F4[tp] = _mm_fmadd_ps(_mm_cvtepi32_ps(z), _mm_loadu_ps(&db[2*tp][0]), F4[tp]);
            }
        }
    }
    alignas(32) float rs[8][G];
    if constexpr (G >= 8) {
        for (int t = 0; t < NT; ++t) { for (int q = 0; q < G/8; ++q) { _mm256_storeu_ps(&rs[t][8*q], F8[t*(G/8) + q]); } }
    } else if constexpr (G == 4) {
        for (int t = 0; t < NT; ++t) { _mm_storeu_ps(&rs[t][0], F4[t]); }
    } else {
        for (int tp = 0; tp < N4; ++tp) { _mm_storeu_ps(&rs[2*tp][0], F4[tp]); }   // rs[t0][0..1], rs[t1][0..1]
    }
    const uint16_t * dh = (const uint16_t *) blk;
    for (int g = 0; g < G; ++g) {
        const float dw = ai4_fp16_to_fp32(dh[R0 + g]);
        for (int t = 0; t < NT; ++t) { acc[t][R0 + g] = __builtin_fmaf(dw, rs[t][g], acc[t][R0 + g]); }
    }
}

template <int K, int NT, int ST, size_t... J>
static void unit_rows_q8_impl(const uint8_t * blk, const void * const * xq, int64_t koff, const float (*dd)[8][2], __m256i Cv,
                              float (*acc)[AI4_TR], std::index_sequence<J...>) {
    constexpr int G = AI4_TR/(int) sizeof...(J);
    (q8_group<K, NT, G, (int) J*G, ST>(blk, xq, koff, dd, Cv, acc), ...);
}

// Rows per group by token count and block shape (home 5800X3D, 1 core): G = 4 at nt = 5 is 4-8% faster for down
// (ST = 4, four Q8_0 blocks per unit) but 6-15% slower for gate/up (ST = 16), so only down takes it.
#if !defined(MOE_AI_Q8_G)
#define MOE_AI_Q8_G(NT, ST) ((NT) == 1 ? 16 : (NT) == 2 ? 8 : (NT) <= 4 || ((NT) == 5 && (ST) == 4) ? 4 : 2)
#endif

template <int K, int NT, int ST>
static void unit_rows_q8_nt(const uint8_t * blk, const void * const * xq, int64_t koff, int64_t Kin, __m256i Cv, float (*acc)[AI4_TR]) {
    constexpr int G = MOE_AI_Q8_G(NT, ST);   // rows per group (G/2 int32 pair accumulators per token)
    constexpr int NB = AI4U_UK/AI4_TC/ST, QB = ST*AI4_TC < AI4U_UK ? 32 : 256;
    alignas(16) float dd[NB][8][2] = {};
    for (int b = 0; b < NB; ++b) {
        for (int t = 0; t < NT; ++t) { dd[b][t][0] = dd[b][t][1] = q8_d(xq[t], Kin)[(koff + b*ST*AI4_TC)/QB]; }
    }
    unit_rows_q8_impl<K, NT, ST>(blk, xq, koff, dd, Cv, acc, std::make_index_sequence<AI4_TR/G>{});
}

template <int K, int ST>
static void unit_rows_q8_k(const uint8_t * blk, int nt, const void * const * xq, int64_t koff, int64_t Kin, __m256i Cv,
                           float (*acc)[AI4_TR]) {
    switch (nt) {
        case 1:  unit_rows_q8_nt<K, 1, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 2:  unit_rows_q8_nt<K, 2, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 3:  unit_rows_q8_nt<K, 3, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 4:  unit_rows_q8_nt<K, 4, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 5:  unit_rows_q8_nt<K, 5, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 6:  unit_rows_q8_nt<K, 6, ST>(blk, xq, koff, Kin, Cv, acc); break;
        case 7:  unit_rows_q8_nt<K, 7, ST>(blk, xq, koff, Kin, Cv, acc); break;
        default: unit_rows_q8_nt<K, 8, ST>(blk, xq, koff, Kin, Cv, acc); break;
    }
}

template <int ST>
static bool unit_rows_q8_st(const uint8_t * blk, int k, uint32_t C, int nt, const void * const * xq, int64_t koff, int64_t Kin,
                            float (*acc)[AI4_TR]) {
    const __m256i Cv = _mm256_set1_epi32((int) C);
    switch (k) {
        case 2: unit_rows_q8_k<2, ST>(blk, nt, xq, koff, Kin, Cv, acc); return true;
        case 3: unit_rows_q8_k<3, ST>(blk, nt, xq, koff, Kin, Cv, acc); return true;
        case 4: unit_rows_q8_k<4, ST>(blk, nt, xq, koff, Kin, Cv, acc); return true;
        case 5: unit_rows_q8_k<5, ST>(blk, nt, xq, koff, Kin, Cv, acc); return true;
        default: return false;
    }
}

// k in 2..5 with valid row-window tables; returns false (nothing done) otherwise
static bool unit_rows_q8(const uint8_t * blk, int k, uint32_t C, int nt, const void * const * xq, int64_t koff, int64_t Kin,
                         float (*acc)[AI4_TR]) {
    if (!direct_tables().ok) { return false; }
    return q8_block(Kin) == 256 ? unit_rows_q8_st<16>(blk, k, C, nt, xq, koff, Kin, acc)
                                : unit_rows_q8_st<4>(blk, k, C, nt, xq, koff, Kin, acc);
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

// AIPFIX: opt-in AVX-512 VBMI + VNNI Q8 path (GGML_MOE_AI_HOST_Q8_VBMI=1 with GGML_MOE_AI_HOST_Q8=1; Zen 4/5 hosts).
// Levels come from the VBMI decode above (4 tile rows per zmm, 16-bit lanes), with the qword lanes ordered so each
// 128-bit lane pairs with the Q8 buffer's tile layout [x0..3 x0..3 | x4..7 x4..7]: qwords [rA lo, rB lo, rA hi, rB hi,
// rC lo, rD lo, rC hi, rD hi] (lo = states 0-3, hi = 4-7). The tile's 32 activation bytes are broadcast to both 256-bit
// halves by one load, and one vpdpwssd per token and row group gives the exact int32 pair sums (= the AVX2 vpmaddwd +
// vpaddd). Per quant block the pair lanes of each row are summed exactly, converted once, rs = fma(row sum, d, rs);
// then acc = fma(fp16 scale, rs, acc) per unit: the float steps and order of unit_rows_q8_scalar, so the result is
// bit-identical to it and to the AVX2 Q8 path (the test checks).
#define MOE_AI_T512Q __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512vnni,fma,f16c")))

static bool has_q8_vbmi() {
    static const bool on = has_vbmi() && __builtin_cpu_supports("avx512vnni");
    return on;
}

static bool use_q8_vbmi() {
    static const bool on = [] { const char * e = getenv("GGML_MOE_AI_HOST_Q8_VBMI"); return e && e[0] == '1' && has_q8_vbmi(); }();
    return on;
}

struct vbmi_q8_tabs {
    uint8_t idx[6][4][64];   // [k][row group]: tile byte for each window byte (vpermi2b over 128 tile bytes)
    uint8_t ctl[6][64];      // [k]: vpmultishiftqb bit offsets (2 per 16-bit state)
    uint8_t row[16];         // reduced lane p -> row of the unit (4 row groups)
    vbmi_q8_tabs() {
        memset(this, 0, sizeof(*this));
        for (int k = 2; k <= 5; ++k) {
            const int nw = AIK_TILE_WORDS(k);
            for (int g = 0; g < 4; ++g) {
                for (int L = 0; L < 8; ++L) {
                    const int r = 4*g + 2*(L >> 2) + (L & 1);
                    for (int m = 0; m < 8; ++m) {
                        const int P = k*r + 2 + 7 - m;            // ext byte (tile_bytes layout) of window byte m
                        const int j = P >> 2, q = P & 3;          // E word, big-endian byte in it
                        const int wi = j == 0 ? nw - 1 : (j - 1) % nw;
                        idx[k][g][8*L + m] = (uint8_t) (4*wi + 3 - q);
                    }
                }
            }
            for (int L = 0; L < 8; ++L) {
                for (int s = 0; s < 4; ++s) {
                    const int kk = 4*((L >> 1) & 1) + s;
                    const int p = 48 - k - k*kk;                  // state kk = (w >> p) & 0xFFFF
                    ctl[k][8*L + 2*s]     = (uint8_t) p;
                    ctl[k][8*L + 2*s + 1] = (uint8_t) (p + 8);
                }
            }
        }
        // q8v_reduce output lane p (128-bit lane p/4, element p%4): group (lane >= 2 ? 2 : 0) + (e >= 2), row in group
        // 2*(lane & 1) + (e & 1)
        for (int p = 0; p < 16; ++p) {
            const int lane = p/4, e = p%4;
            row[p] = (uint8_t) (4*((lane >= 2 ? 2 : 0) + (e >= 2 ? 1 : 0)) + 2*(lane & 1) + (e & 1));
        }
    }
};

static const vbmi_q8_tabs & vbmi_q8_tables() {
    static const vbmi_q8_tabs T;
    return T;
}

MOE_AI_T512Q static inline __attribute__((always_inline)) __m512i q8v_hadd(__m512i a, __m512i b) {   // [a0+a1 a2+a3 b0+b1 b2+b3]
    const __m512 fa = _mm512_castsi512_ps(a), fb = _mm512_castsi512_ps(b);
    return _mm512_add_epi32(_mm512_castps_si512(_mm512_shuffle_ps(fa, fb, _MM_SHUFFLE(2, 0, 2, 0))),
                            _mm512_castps_si512(_mm512_shuffle_ps(fa, fb, _MM_SHUFFLE(3, 1, 3, 1))));
}

// x_g: int32 pair sums of row group g (128-bit lanes [rA s01 s23, rB s01 s23 | rA s45 s67, rB .. | rC .. , rD .. | ..])
// -> 16 exact row sums, lane p holds row vbmi_q8_tabs::row[p]
MOE_AI_T512Q static inline __attribute__((always_inline)) __m512i q8v_reduce(__m512i x0, __m512i x1, __m512i x2, __m512i x3) {
    __m512i h01 = q8v_hadd(x0, x1), h23 = q8v_hadd(x2, x3);
    h01 = _mm512_add_epi32(h01, _mm512_shuffle_i32x4(h01, h01, _MM_SHUFFLE(2, 3, 0, 1)));   // lane 0 = lane 0 + lane 1, ..
    h23 = _mm512_add_epi32(h23, _mm512_shuffle_i32x4(h23, h23, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm512_shuffle_i32x4(h01, h23, _MM_SHUFFLE(2, 0, 2, 0));
}

// KR = rate k, GG = row groups (of 4 rows) per pass over the unit's tiles, ST = tiles per quant block
template <int KR, int NT, int GG, int ST>
MOE_AI_T512Q static void unit_rows_q8v_nt(const uint8_t * blk, uint32_t C, const void * const * xq, int64_t koff, int64_t Kin,
                                          float (*acc)[AI4_TR]) {
    static_assert(4 % GG == 0, "row groups per pass");
    const vbmi_q8_tabs & T = vbmi_q8_tables();
    constexpr int tb = 4*AIK_TILE_WORDS(KR);                 // tile bytes, 16k
    constexpr __mmask64 m0 = tb >= 64 ? ~0ULL : (1ULL << tb) - 1;
    constexpr __mmask64 m1 = tb > 64 ? (1ULL << (tb - 64)) - 1 : 0;
    constexpr int NB = AI4U_UK/AI4_TC/ST, QB = ST*AI4_TC < AI4U_UK ? 32 : 256;
    const uint8_t * wb = blk + AI4U_SCALE_BYTES;
    const __m512i ctl = _mm512_loadu_si512(T.ctl[KR]);
    const __m512i cl = _mm512_set1_epi16((short) (C & 0xFFFF)), ch = _mm512_set1_epi16((short) (C >> 16));
    const __m512i one8 = _mm512_set1_epi8(1), k510 = _mm512_set1_epi16(510);
    float d[NB][8];
    for (int b = 0; b < NB; ++b) { for (int t = 0; t < NT; ++t) { d[b][t] = q8_d(xq[t], Kin)[(koff + b*ST*AI4_TC)/QB]; } }
    const int16_t * xs[NT];
    for (int t = 0; t < NT; ++t) { xs[t] = q8_xs(xq[t]) + (koff/AI4_TC)*16; }
    alignas(64) float rs[8][16];
    for (int G0 = 0; G0 < 4; G0 += GG) {
        __m512i idx[GG];
        for (int g = 0; g < GG; ++g) { idx[g] = _mm512_loadu_si512(T.idx[KR][G0 + g]); }
        __m512 F[NT];
        for (int t = 0; t < NT; ++t) { F[t] = _mm512_setzero_ps(); }
        for (int b = 0; b < NB; ++b) {
            __m512i ip[NT][GG];
            for (int t = 0; t < NT; ++t) { for (int g = 0; g < GG; ++g) { ip[t][g] = _mm512_setzero_si512(); } }
            for (int ct = b*ST; ct < (b + 1)*ST; ++ct) {
                const uint8_t * tp = wb + ct*tb;
                const __m512i t0 = _mm512_maskz_loadu_epi8(m0, tp);
                const __m512i t1 = tb > 64 ? _mm512_maskz_loadu_epi8(m1, tp + 64) : _mm512_setzero_si512();
                for (int g = 0; g < GG; ++g) {
                    const __m512i win = _mm512_permutex2var_epi8(t0, idx[g], t1);
                    const __m512i st  = _mm512_multishift_epi64_epi8(ctl, win);
                    const __m512i lo  = _mm512_mullo_epi16(st, cl);
                    const __m512i hi  = _mm512_add_epi16(_mm512_mulhi_epu16(st, cl), _mm512_mullo_epi16(st, ch));
                    const __m512i sum = _mm512_add_epi16(_mm512_maddubs_epi16(lo, one8), _mm512_maddubs_epi16(hi, one8));
                    const __m512i lev = _mm512_srai_epi16(_mm512_sub_epi16(sum, k510), 2);
                    for (int t = 0; t < NT; ++t) {
                        const __m512i xb = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) (xs[t] + ct*16)));
                        ip[t][g] = _mm512_dpwssd_epi32(ip[t][g], lev, xb);
                    }
                }
            }
            const __m512i z = _mm512_setzero_si512();
            for (int t = 0; t < NT; ++t) {
                const __m512i rsum = q8v_reduce(ip[t][0], GG > 1 ? ip[t][GG > 1 ? 1 : 0] : z,
                                                GG > 2 ? ip[t][GG > 2 ? 2 : 0] : z, GG > 3 ? ip[t][GG > 3 ? 3 : 0] : z);
                F[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(rsum), _mm512_set1_ps(d[b][t]), F[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, F[t]);
            for (int p = 0; p < 16; ++p) {
                const int r = T.row[p];
                if (r/4 < GG) { rs[t][4*G0 + r] = tmp[p]; }   // lanes of groups past GG are the zero padding
            }
        }
    }
    const uint16_t * dh = (const uint16_t *) blk;
    for (int r = 0; r < AI4_TR; ++r) {
        const float dw = ai4_fp16_to_fp32(dh[r]);
        for (int t = 0; t < NT; ++t) { acc[t][r] = __builtin_fmaf(dw, rs[t][r], acc[t][r]); }
    }
}

// row groups per pass: all four while the int32 accumulators (NT x GG zmm) leave room for the decode temporaries
#if !defined(MOE_AI_Q8V_GG)
#define MOE_AI_Q8V_GG(NT) ((NT) <= 2 ? 4 : (NT) <= 5 ? 2 : 1)
#endif

template <int KR, int ST>
MOE_AI_T512Q static void unit_rows_q8v_k(const uint8_t * blk, uint32_t C, int nt, const void * const * xq, int64_t koff,
                                         int64_t Kin, float (*acc)[AI4_TR]) {
    switch (nt) {
        case 1:  unit_rows_q8v_nt<KR, 1, MOE_AI_Q8V_GG(1), ST>(blk, C, xq, koff, Kin, acc); break;
        case 2:  unit_rows_q8v_nt<KR, 2, MOE_AI_Q8V_GG(2), ST>(blk, C, xq, koff, Kin, acc); break;
        case 3:  unit_rows_q8v_nt<KR, 3, MOE_AI_Q8V_GG(3), ST>(blk, C, xq, koff, Kin, acc); break;
        case 4:  unit_rows_q8v_nt<KR, 4, MOE_AI_Q8V_GG(4), ST>(blk, C, xq, koff, Kin, acc); break;
        case 5:  unit_rows_q8v_nt<KR, 5, MOE_AI_Q8V_GG(5), ST>(blk, C, xq, koff, Kin, acc); break;
        case 6:  unit_rows_q8v_nt<KR, 6, MOE_AI_Q8V_GG(6), ST>(blk, C, xq, koff, Kin, acc); break;
        case 7:  unit_rows_q8v_nt<KR, 7, MOE_AI_Q8V_GG(7), ST>(blk, C, xq, koff, Kin, acc); break;
        default: unit_rows_q8v_nt<KR, 8, MOE_AI_Q8V_GG(8), ST>(blk, C, xq, koff, Kin, acc); break;
    }
}

template <int ST>
MOE_AI_T512Q static bool unit_rows_q8v_st(const uint8_t * blk, int k, uint32_t C, int nt, const void * const * xq, int64_t koff,
                                          int64_t Kin, float (*acc)[AI4_TR]) {
    switch (k) {
        case 2: unit_rows_q8v_k<2, ST>(blk, C, nt, xq, koff, Kin, acc); return true;
        case 3: unit_rows_q8v_k<3, ST>(blk, C, nt, xq, koff, Kin, acc); return true;
        case 4: unit_rows_q8v_k<4, ST>(blk, C, nt, xq, koff, Kin, acc); return true;
        case 5: unit_rows_q8v_k<5, ST>(blk, C, nt, xq, koff, Kin, acc); return true;
        default: return false;
    }
}

// k in 2..5 on a VBMI + VNNI host; returns false (nothing done) otherwise
static bool unit_rows_q8_vbmi(const uint8_t * blk, int k, uint32_t C, int nt, const void * const * xq, int64_t koff, int64_t Kin,
                              float (*acc)[AI4_TR]) {
    if (!has_q8_vbmi()) { return false; }
    return q8_block(Kin) == 256 ? unit_rows_q8v_st<16>(blk, k, C, nt, xq, koff, Kin, acc)
                                : unit_rows_q8v_st<4>(blk, k, C, nt, xq, koff, Kin, acc);
}
#endif

// rows [a, b) of one expert (a multiple of 16; b - a <= 16, rows past b are computed and dropped), nt tokens:
// o[t][row - a] = W[row] . x_t, with y[t] the rotated inputs from rotate()
// path: -1 = best available (VBMI if present, else AVX2), 0 = AVX2 (or scalar), 1 = AVX-512 (VBMI decode when the host has it),
// 2 = AVX-512 without VBMI,
// 3 = AVX2 with the old window decode (row_levels_avx2; test reference), 4 = AVX2 shuffle reading every lane from E
// (the shuffle decode before direct lane loads; GGML_MOE_AI_HOST_DIRECT=0 makes it the shuffle fallback),
// 5 = AVX2 row-window decode (unit_rows_direct: one 16-byte load per row, register-grouped rows; the default AVX2 path
// unless GGML_MOE_AI_HOST_ROWWIN=0, which falls back to path 0 = shuffle with direct interior lane loads)
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
        if (C != 0 && (path == 5 || (path == -1 && use_rowwin())) && unit_rows_direct(blk, k, C, nt, y, u*AI4U_UK, acc)) {
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

// Q8 path: rows [a, b) of one expert (as slab_rows), nt <= 8 tokens, xq[t] = quant_q8/rotate_q8 buffers.
// scalar = true forces the scalar reference (the test checks AVX2 == scalar bit for bit). vbmi: -1 = AVX-512 VBMI+VNNI
// kernel when GGML_MOE_AI_HOST_Q8_VBMI=1 and the host has it, 0 = never, 1 = whenever the host has it (test/bench).
static void slab_rows_q8(const char * w, size_t row_bytes, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
                         const void * const * xq, float * const * o, bool scalar = false, int vbmi = -1) {
    const uint8_t * slab = (const uint8_t *) w + (size_t) a*row_bytes;
    const int ub = AI4U_UNIT_BYTES(k);
    float acc[8][AI4_TR] = {};
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t * blk = slab + u*ub;
#if defined(__AVX2__) && defined(__FMA__)
        if (!scalar && C != 0 && (vbmi > 0 || (vbmi < 0 && use_q8_vbmi())) && unit_rows_q8_vbmi(blk, k, C, nt, xq, u*AI4U_UK, K, acc)) {
            continue;
        }
        if (!scalar && C != 0 && unit_rows_q8(blk, k, C, nt, xq, u*AI4U_UK, K, acc)) { continue; }
#endif
        unit_rows_q8_scalar(blk, k, C, nt, xq, u*AI4U_UK, K, acc);
    }
    for (int t = 0; t < nt; ++t) {
        for (int64_t row = a; row < b; ++row) { o[t][row - a] = acc[t][row - a]; }
    }
}

} // namespace moe_ai
