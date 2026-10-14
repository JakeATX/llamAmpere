#pragma once

// SJ-KVaRN-style sealed KV records: scalar reference codec shared by the CPU backend, the tests and
// the CUDA kernels' host side. Math follows github.com/huawei-csl/KVarN (variance balancing in the
// log domain, asymmetric per-row RTN, scale absorption) with one deliberate divergence: the balancing
// runs a fixed number of iterations and keeps the FINAL scales instead of the reference's best-so-far
// selection, which flips on float32 last bits once the imbalance reaches its fixed point and would make
// CPU and CUDA sealers disagree. The record layout is this project's own, see ggml_sj_kvarn_seal in ggml.h and
// the layout notes above code_bit() below.

#include "ggml.h"
#include "ggml-impl.h"

#ifdef __CUDACC__
#define SJKVARN_HD __host__ __device__
#else
#define SJKVARN_HD
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ggml_sj_kvarn {

struct layout {
    int    D, G, bits_k, bits_v;
    size_t k_row;      // bytes per token row of the K payload
    size_t v_row;      // bytes per token row of the V payload
    size_t k_payload;  // offsets
    size_t v_payload;
    size_t k_scale;    // fp16[D]
    size_t k_zero;     // fp16[D]
    size_t k_tok;      // fp16[G]
    size_t v_ch;       // fp16[D]
    size_t v_scale;    // fp16[G]
    size_t v_zero;     // fp16[G]
    size_t bytes;
};

inline layout make_layout(int D, int G, int bits_k, int bits_v) {
    layout l;
    l.D = D; l.G = G; l.bits_k = bits_k; l.bits_v = bits_v;
    l.k_row     = (size_t) D * bits_k / 8;
    l.v_row     = (size_t) D * bits_v / 8;
    l.k_payload = 0;
    l.v_payload = l.k_payload + l.k_row * G;
    l.k_scale   = l.v_payload + l.v_row * G;
    l.k_zero    = l.k_scale + 2 * (size_t) D;
    l.k_tok     = l.k_zero  + 2 * (size_t) D;
    l.v_ch      = l.k_tok   + 2 * (size_t) G;
    l.v_scale   = l.v_ch    + 2 * (size_t) D;
    l.v_zero    = l.v_scale + 2 * (size_t) G;
    l.bytes     = l.v_zero  + 2 * (size_t) G;
    return l;
}

// orthonormal Sylvester Hadamard, in place, n a power of two
inline void hadamard(float * x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

// Reduction contract shared with the CUDA sealer (sjkvarn-seal.cu) so both produce identical records:
// a vector of n values (n % SJKVARN_RED_LANES == 0) is split into SJKVARN_RED_LANES contiguous chunks, each
// chunk is summed sequentially in double, and the chunk partials are combined as a balanced xor
// butterfly ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)). The second moment uses an explicit double fma so
// neither compiler can change the contraction. Everything else is IEEE float/double arithmetic.
#define SJKVARN_RED_LANES 8

inline double reduce_lanes(const double * p) {
    double q[SJKVARN_RED_LANES];
    for (int i = 0; i < SJKVARN_RED_LANES; ++i) q[i] = p[i];
    for (int w = 1; w < SJKVARN_RED_LANES; w <<= 1) {
        for (int i = 0; i < SJKVARN_RED_LANES; i += 2*w) {
            q[i] = q[i] + q[i + w];
        }
    }
    return q[0];
}

// sample standard deviation (N-1) of a strided float sequence, see the reduction contract above
inline float sample_std(const float * x, int n, size_t stride) {
    const int chunk = n / SJKVARN_RED_LANES;
    double p[SJKVARN_RED_LANES];
    for (int l = 0; l < SJKVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (int i = 0; i < chunk; ++i) {
            acc += (double) x[(size_t) (l*chunk + i)*stride];
        }
        p[l] = acc;
    }
    const double mean = reduce_lanes(p) / n;
    for (int l = 0; l < SJKVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (int i = 0; i < chunk; ++i) {
            const double d = (double) x[(size_t) (l*chunk + i)*stride] - mean;
            acc = fma(d, d, acc);
        }
        p[l] = acc;
    }
    const double m2 = reduce_lanes(p);
    return (float) sqrt(m2 / (n > 1 ? (n - 1) : 1));
}

// log-domain variance balancing of a row-major [R][C] tile: exactly `iters` alternating column/row
// passes, final scales s_row[R], s_col[C] such that cur = M / s_col / s_row has balanced row and
// column spreads (no best-so-far selection, see the header comment). R and C must be multiples of
// SJKVARN_RED_LANES.
inline void balance(const float * M, int R, int C, int iters, float * s_row, float * s_col, float * cur /* [R*C] scratch */) {
    std::vector<double> log_sc(C, 0.0), log_sr(R, 0.0); // log domain in double (CUDA fast-math expf/logf differ, double exp/log do not)
    for (int c = 0; c < C; ++c) s_col[c] = 1.0f;
    for (int r = 0; r < R; ++r) s_row[r] = 1.0f;

    auto recompute = [&]() {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                cur[(size_t) r*C + c] = M[(size_t) r*C + c] / s_col[c] / s_row[r];
            }
        }
    };

    recompute();
    for (int it = 0; it < iters; ++it) {
        for (int c = 0; c < C; ++c) {
            const float sd = std::min(std::max(sample_std(cur + c, R, C), 1e-3f), 1e3f);
            log_sc[c] = std::min(std::max(log_sc[c] + log((double) sd), -0.3), 10.0);
            s_col[c] = (float) exp(log_sc[c]);
        }
        recompute();
        for (int r = 0; r < R; ++r) {
            const float sd = std::min(std::max(sample_std(cur + (size_t) r*C, C, 1), 1e-3f), 1e3f);
            log_sr[r] = std::min(std::max(log_sr[r] + log((double) sd), -0.3), 10.0);
            s_row[r] = (float) exp(log_sr[r]);
        }
        recompute();
    }
}

inline uint32_t rtn_code(float x, float lo, float step, uint32_t qmax) {
    const float y = (x - lo) / step;
    float r = nearbyintf(y); // ties to even under the default rounding mode
    if (r < 0.0f) r = 0.0f;
    if (r > (float) qmax) r = (float) qmax;
    return (uint32_t) r;
}

// ---------------------------------------------------------------------------------------------------------
// Trellis-coded 4-bit payload ("sjkvarn4t", body type GGML_TYPE_I16).
//
// The record layout and all metadata are those of the scalar codec; only the meaning of the 4-bit payload
// changes. The 8 nibbles of a lane's fragment word and that lane's words across the strips of the record form
// one little-endian bit stream of G/2 values (value n = nibble n%8 of strip n/8). Value n reconstructs as
// cb[window_n], window_n = bits[4n-(L-4), 4n+4) (its own nibble j on top of the L-4 bits before it, zero bits
// before the start of the stream): a trellis with 2^(L-4) states s (the shared bits) whose branch codebook from
// state s is {cb[s + 2^(L-4) j]}, next state = window >> 4, start state 0, no termination.
// The identity codebook cb[w] = w >> (L-4) reproduces scalar RTN (up to ties at exact half-integers); a trained
// codebook (ggml-sjkvarn-cb.h, generalized Lloyd on sealed tiles) is where the trellis coding gain comes from.
// Encoding is the Viterbi search below; the CPU reference and the CUDA sealer share its arithmetic contract so
// records are bit-identical: d = y - cb[w], e = d*d, cost = e + M[w >> 4] in IEEE float without contraction,
// per-state minima with strict < over ascending j, so ties pick the smallest nibble.
#ifndef SJKVARN_TRELLIS_L
#define SJKVARN_TRELLIS_L 10
#endif
#include "ggml-sjkvarn-cb.h"
#include "ggml-sjkvarn-cb-lowbits.h" // trained 3-bit and 2-bit codebooks (built in; env overrides for experiments)
static_assert(SJKVARN_TRELLIS_CB_L == SJKVARN_TRELLIS_L, "codebook header and trellis window length disagree");

namespace trellis {
constexpr int L      = SJKVARN_TRELLIS_L;
constexpr int NWIN   = 1 << L;    // windows = codebook entries
constexpr int NSTATE = NWIN >> 4; // states
constexpr int NBR    = 16;        // branches per state (the new nibble)
static_assert(L >= 8 && L <= 12, "trellis window length 8..12");
// host codebooks (fp16 bits, y units): the header defaults, or the file named by GGML_SJKVARN_TRELLIS_CB
// (uint16 K[NWIN] then V[NWIN], as written by gate_trellis2.py --save-codebook) loaded once per translation unit
inline const uint16_t * cb(bool is_V) {
    static uint16_t k[NWIN] = SJKVARN_CB_K_INIT;
    static uint16_t v[NWIN] = SJKVARN_CB_V_INIT;
    static const bool loaded = [] {
        const char * path = getenv("GGML_SJKVARN_TRELLIS_CB");
        if (path == nullptr) { return false; }
        FILE * f = fopen(path, "rb");
        bool ok = f != nullptr && fread(k, sizeof(uint16_t), NWIN, f) == (size_t) NWIN && fread(v, sizeof(uint16_t), NWIN, f) == (size_t) NWIN;
        if (f) { fclose(f); }
        fprintf(stderr, "sj_kvarn trellis: codebook %s from %s (L=%d)\n", ok ? "loaded" : "FAILED TO LOAD", path, L);
        if (!ok) { abort(); }
        return ok;
    }();
    (void) loaded;
    return is_V ? v : k;
}
}


// Low-bit codebook selection (trellis3 / trellis2): the built-in trained codebook is the default. The environment
// variable named by `env` may name a file (uint16 K[NWIN] then V[NWIN]) or say "identity" (cb[w] = w >> (L-BITS), i.e.
// the scalar codes: the matched-decoder control) or "trained" (the built-in). Reports the choice once on stderr.
inline bool trellis_lb_load(const char * env, const char * tag, int L, int BITS, int NWIN, uint16_t * k, uint16_t * v) {
    const char * s = getenv(env);
    if (s == nullptr || strcmp(s, "trained") == 0 || strcmp(s, "builtin") == 0) {
        fprintf(stderr, "sj_kvarn %s: built-in trained codebook (L=%d)\n", tag, L);
        return false;
    }
    if (strcmp(s, "identity") == 0) {
        for (int w = 0; w < NWIN; ++w) { k[w] = v[w] = GGML_FP32_TO_FP16((float) (w >> (L - BITS))); }
        fprintf(stderr, "sj_kvarn %s: identity codebook (scalar codes, L=%d)\n", tag, L);
        return true;
    }
    FILE * f = fopen(s, "rb");
    const bool ok = f != nullptr && fread(k, sizeof(uint16_t), NWIN, f) == (size_t) NWIN && fread(v, sizeof(uint16_t), NWIN, f) == (size_t) NWIN;
    if (f) { fclose(f); }
    fprintf(stderr, "sj_kvarn %s: codebook %s from %s (L=%d)\n", tag, ok ? "loaded" : "FAILED TO LOAD", s, L);
    if (!ok) { abort(); }
    return true;
}
// ---------------------------------------------------------------------------------------------------------
// 3-bit trellis body ("sjkvarn3t"): body type GGML_TYPE_I16 on a 3-bit side. Default (2026-10-05): the trellis
// runs along the tokens of each channel and the payload is the channel-major stream (see tokens() below).
// GGML_SJKVARN_TRELLIS_TOKENS=0 (channel axis): the payload keeps the plain token-major 3-bit stream of code_bit();
// the trellis runs along the channels of a token row and restarts (history = 0) every SEQ channels. The window of code (t, d) is the current code in its top 3 bits and the
// two codes before it below (older lower), i.e. exactly the L = 9 stream bits ending at the current code.
// The built-in trained codebook of the active axis (ggml-sjkvarn-cb-lowbits.h) is the default; the identity codebook cb[w] = w >> 6
// reproduces scalar RTN codes (GGML_SJKVARN_TRELLIS_CB3=identity, or a file of uint16 K[NWIN] then V[NWIN] fp16
// bits). GGML_SJKVARN_TRELLIS_REFIT = none | fit | norm refits the per-row scale/zero after coding (least squares /
// moment matching of the reconstruction); norm is the default.
namespace trellis3 {
constexpr int BITS   = 3;
constexpr int L      = 9;
constexpr int NWIN   = 1 << L;       // 512 codebook entries
constexpr int NSTATE = NWIN >> BITS; // 64 states
constexpr int NBR    = 1 << BITS;    // 8 branches
constexpr int SEQ    = 128;          // channels per sequence
static_assert(SEQ*BITS % 32 == 0, "a sequence must cover whole 32-bit words");
// GGML_SJKVARN_TRELLIS_TOKENS (default on since 2026-10-05; =0 selects the channel-axis payload of v0.4/the audit's
// production arm): the low-bit (3- and 2-bit) trellis runs along the TOKENS of each channel (one 128-token sequence per
// channel and record, history restarting at every record) instead of along the channels of a token row, and the payload
// becomes the channel-major stream (bit (d*G + t)*BITS). K and V are correlated along tokens (lag-1 0.42 / 0.28 in the
// balanced y domain, about 0 along channels), so the history-indexed codebook predicts. Each axis has its own built-in
// codebooks (SJKVARN_CB3_TOK_* / SJKVARN_CB2_TOK_* for tokens, SJKVARN_CB3_TRAINED_* / SJKVARN_CB2_TRAINED_* for channels).
// Codec audit KL (6 histories, generated region, vs the channel axis): 0.769x (3/3), 0.772x (3/2).
// The axis is process-wide and nothing sealed leaves the process (SJ-KVaRN state save/load is refused), so the reader of a
// record always shares its writer's axis.
inline bool tokens() {
    static const bool on = [] {
        const char * s = getenv("GGML_SJKVARN_TRELLIS_TOKENS");
        const bool v = s == nullptr || *s == '\0' || atoi(s) != 0;
        fprintf(stderr, v ? "sj_kvarn trellis: token-traversal payload (default; GGML_SJKVARN_TRELLIS_TOKENS=0 selects the channel axis)\n"
                          : "sj_kvarn trellis: channel-traversal payload (GGML_SJKVARN_TRELLIS_TOKENS=0)\n");
        return v;
    }();
    return on;
}
// the 4/4 token-axis trellis4 body (audit item B, k44t_tok) stays opt-in: only an explicit nonzero GGML_SJKVARN_TRELLIS_TOKENS
// (the audit's =1) turns a 4/4 trellis (I16, sjkvarn4t) body into trellis4. Unset = sjkvarn4t, unchanged.
inline bool tokens4() {
    static const bool on = [] {
        const char * s = getenv("GGML_SJKVARN_TRELLIS_TOKENS");
        return s != nullptr && *s != '\0' && atoi(s) != 0;
    }();
    return on;
}
inline const uint16_t * cb(bool is_V) {
    static uint16_t k[NWIN] = SJKVARN_CB3_TRAINED_K_INIT;
    static uint16_t v[NWIN] = SJKVARN_CB3_TRAINED_V_INIT;
    static const bool overridden = [] {
        if (tokens()) {
            static const uint16_t tk[NWIN] = SJKVARN_CB3_TOK_K_INIT;
            static const uint16_t tv[NWIN] = SJKVARN_CB3_TOK_V_INIT;
            memcpy(k, tk, sizeof(k)); memcpy(v, tv, sizeof(v));
        }
        return trellis_lb_load("GGML_SJKVARN_TRELLIS_CB3", tokens() ? "trellis3 (token axis)" : "trellis3 (channel axis)", L, BITS, NWIN, k, v);
    }();
    (void) overridden;
    return is_V ? v : k;
}
// per-row affine refit after coding: norm (moment matching) is the default (KL gate 2026-09-23: most of the 3-bit gain);
// GGML_SJKVARN_TRELLIS_REFIT = none | fit | norm overrides
inline int refit_mode() {
    static const int mode = [] {
        const char * s = getenv("GGML_SJKVARN_TRELLIS_REFIT");
        if (s == nullptr || strcmp(s, "norm") == 0) { return 2; }
        if (strcmp(s, "none") == 0 || strcmp(s, "0") == 0) { return 0; }
        if (strcmp(s, "fit") == 0) { return 1; }
        fprintf(stderr, "sj_kvarn trellis3: unknown GGML_SJKVARN_TRELLIS_REFIT=%s (none|fit|norm)\n", s);
        abort();
    }();
    return mode;
}
}
// GGML_SJKVARN_SCALAR_CLIP=cK,cV | row (audit 2026-10-03, default off): scalar (RTN) bodies shrink each row's min/max range
// symmetrically around its centre by cK (K rows = channels) / cV (V rows = tokens) before choosing the step; values
// outside clamp to the end codes (rtn_code). Record format and decode are unchanged. Off (unset) = factor 1, which is
// not applied at all, so the off path is the production arithmetic.
namespace scalar_clip {
inline const float * factors() {
    static float f[2] = { 1.0f, 1.0f };
    static const bool init = [] {
        const char * s = getenv("GGML_SJKVARN_SCALAR_CLIP");
        if (s != nullptr && strcmp(s, "row") == 0) {
            f[0] = f[1] = -1.0f; // per-row search (row_clip_search)
            fprintf(stderr, "sj_kvarn: scalar range clip per-row search (GGML_SJKVARN_SCALAR_CLIP=row)\n");
        } else if (s != nullptr && *s) {
            float a = 1.0f, b = 1.0f;
            const int n = sscanf(s, "%f,%f", &a, &b);
            if (n < 1 || !(a > 0.0f && a <= 1.0f) || (n == 2 && !(b > 0.0f && b <= 1.0f))) {
                fprintf(stderr, "sj_kvarn: bad GGML_SJKVARN_SCALAR_CLIP=%s (cK[,cV] in (0,1])\n", s);
                abort();
            }
            f[0] = a; f[1] = n == 2 ? b : a;
            fprintf(stderr, "sj_kvarn: scalar range clip K %.4f V %.4f (GGML_SJKVARN_SCALAR_CLIP)\n", f[0], f[1]);
        }
        return true;
    }();
    GGML_UNUSED(init);
    return f;
}
}
// identity codebook initializer for device copies (fp16 bits of 0..7, 64 entries each)
#define SJKVARN_CB3_R8(x)  x, x, x, x, x, x, x, x
#define SJKVARN_CB3_R64(x) SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x), SJKVARN_CB3_R8(x)
#define SJKVARN_CB3_IDENTITY_INIT { SJKVARN_CB3_R64(0x0000), SJKVARN_CB3_R64(0x3C00), SJKVARN_CB3_R64(0x4000), SJKVARN_CB3_R64(0x4200), \
                                  SJKVARN_CB3_R64(0x4400), SJKVARN_CB3_R64(0x4500), SJKVARN_CB3_R64(0x4600), SJKVARN_CB3_R64(0x4700) }

// 2-bit trellis body ("sjkvarn2t"): the 3-bit scheme at BITS = 2, L = 8 (current code on top of the three codes before it),
// 256 windows, the same 64 states, 4 branches; built-in trained codebook by default, GGML_SJKVARN_TRELLIS_CB2 = <file>
// (uint16 K[256] then V[256]) | identity (cb[w] = w >> 6) overrides. Refit mode is shared with trellis3.
namespace trellis2 {
constexpr int BITS   = 2;
constexpr int L      = 8;
constexpr int NWIN   = 1 << L;       // 256 codebook entries
constexpr int NSTATE = NWIN >> BITS; // 64 states
constexpr int NBR    = 1 << BITS;    // 4 branches
constexpr int SEQ    = 128;          // channels per sequence
static_assert(SEQ*BITS % 32 == 0, "a sequence must cover whole 32-bit words");
static_assert(NSTATE == trellis3::NSTATE && SEQ == trellis3::SEQ, "trellis2 and trellis3 share the Viterbi kernel shape");
inline const uint16_t * cb(bool is_V) {
    static uint16_t k[NWIN] = SJKVARN_CB2_TRAINED_K_INIT;
    static uint16_t v[NWIN] = SJKVARN_CB2_TRAINED_V_INIT;
    static const bool overridden = [] {
        if (trellis3::tokens()) {
            static const uint16_t tk[NWIN] = SJKVARN_CB2_TOK_K_INIT;
            static const uint16_t tv[NWIN] = SJKVARN_CB2_TOK_V_INIT;
            memcpy(k, tk, sizeof(k)); memcpy(v, tv, sizeof(v));
        }
        return trellis_lb_load("GGML_SJKVARN_TRELLIS_CB2", trellis3::tokens() ? "trellis2 (token axis)" : "trellis2 (channel axis)", L, BITS, NWIN, k, v);
    }();
    (void) overridden;
    return is_V ? v : k;
}
inline int refit_mode() { return trellis3::refit_mode(); }
}
#define SJKVARN_CB2_IDENTITY_INIT { SJKVARN_CB3_R64(0x0000), SJKVARN_CB3_R64(0x3C00), SJKVARN_CB3_R64(0x4000), SJKVARN_CB3_R64(0x4200) }

// 4-bit token-axis trellis (audit 2026-10-03, Item B "k44t_tok", opt-in: trellis3::tokens4(), i.e. an explicit
// GGML_SJKVARN_TRELLIS_TOKENS=1): a 4-bit side of a trellis (I16) body is coded by the low-bit scheme at BITS = 4, L = 10
// (current code over the 6 stream bits before it = 1.5 codes), 64 states, 16 branches, along the tokens of each channel
// (channel-major payload, G == 128), with the norm refit, instead of the fragment-order sjkvarn4t. Codebook: the built-in
// token-trained SJKVARN_CB4_TOK_* (the audit's cb4_tok.bin); GGML_SJKVARN_TRELLIS_CB4 = <file> (uint16 K[1024] then V[1024])
// | identity overrides. Measured +35%/round decode vs 4/4 scalar, hence opt-in. Unset TOKENS = sjkvarn4t, unchanged.
namespace trellis4 {
constexpr int BITS   = 4;
constexpr int L      = 10;
constexpr int NWIN   = 1 << L;       // 1024 codebook entries
constexpr int NSTATE = NWIN >> BITS; // 64 states
constexpr int NBR    = 1 << BITS;    // 16 branches
constexpr int SEQ    = 128;          // tokens per sequence
static_assert(NSTATE == trellis3::NSTATE && SEQ == trellis3::SEQ, "trellis4 shares the low-bit Viterbi kernel shape");
static_assert(SJKVARN_TRELLIS_L == L, "trellis4 fallback codebook is the sjkvarn4t table (L = 10)");
inline const uint16_t * cb(bool is_V) {
    static uint16_t k[NWIN] = SJKVARN_CB4_TOK_K_INIT;
    static uint16_t v[NWIN] = SJKVARN_CB4_TOK_V_INIT;
    static const bool overridden = trellis_lb_load("GGML_SJKVARN_TRELLIS_CB4", "trellis4 (token axis)", L, BITS, NWIN, k, v);
    (void) overridden;
    return is_V ? v : k;
}
}

// the two low-bit trellis codecs share every routine below; trellis_lb<BITS> names the constants and codebook of one
template<int BITS> struct trellis_lb;
template<> struct trellis_lb<3> {
    static constexpr int L = trellis3::L, NWIN = trellis3::NWIN, NSTATE = trellis3::NSTATE, NBR = trellis3::NBR, SEQ = trellis3::SEQ;
    static const uint16_t * cb(bool is_V) { return trellis3::cb(is_V); }
};
template<> struct trellis_lb<2> {
    static constexpr int L = trellis2::L, NWIN = trellis2::NWIN, NSTATE = trellis2::NSTATE, NBR = trellis2::NBR, SEQ = trellis2::SEQ;
    static const uint16_t * cb(bool is_V) { return trellis2::cb(is_V); }
};
template<> struct trellis_lb<4> {
    static constexpr int L = trellis4::L, NWIN = trellis4::NWIN, NSTATE = trellis4::NSTATE, NBR = trellis4::NBR, SEQ = trellis4::SEQ;
    static const uint16_t * cb(bool is_V) { return trellis4::cb(is_V); }
};

SJKVARN_HD inline float sub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    volatile float r = a - b; return r; // volatile: no fma contraction under -ffp-contract=fast
#endif
}
SJKVARN_HD inline float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float r = a * b; return r;
#endif
}
SJKVARN_HD inline float add_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    volatile float r = a + b; return r;
#endif
}
SJKVARN_HD inline float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    volatile float r = a / b; return r;
#endif
}
// symmetric range shrink of [lo, hi] by c (GGML_SJKVARN_SCALAR_CLIP); identical rounding on host and device
SJKVARN_HD inline void clip_range(float & lo, float & hi, float c) {
    const float mid  = mul_rn(add_rn(lo, hi), 0.5f);
    const float half = mul_rn(mul_rn(sub_rn(hi, lo), 0.5f), c);
    lo = sub_rn(mid, half);
    hi = add_rn(mid, half);
}
// GGML_SJKVARN_SCALAR_CLIP=row: per row, the shrink factor of SJKVARN_CLIP_NCAND candidates (1 first, ties keep the wider
// range) with the least float squared RTN error over the row. The error of one half of the row (values x0..x0+n-1) for
// candidate i; the caller adds the two halves as p0 + p1 (host) / a shuffle (device), identical rounding.
// candidates 1.00, 0.95, ..., 0.45 (offline E9: the 2-bit scalar V optimum sits near 0.6, 3/4-bit near 0.75-0.95)
#define SJKVARN_CLIP_NCAND 12
SJKVARN_HD inline float clip_cand(int i) {
    return i == 0 ? 1.0f : sub_rn(1.0f, mul_rn(0.05f, (float) i)); // _rn: no fma contraction, host == device
}
SJKVARN_HD inline void clip_cand_range(float lo, float hi, int i, uint32_t qmax, float & clo, float & cstep) {
    if (i > 0) { clip_range(lo, hi, clip_cand(i)); }
    clo = lo;
    cstep = fmaxf(div_rn(sub_rn(hi, lo), (float) qmax), 1e-10f);
}
SJKVARN_HD inline float clip_err_term(float x, float clo, float cstep, uint32_t qmax) {
    float y = div_rn(sub_rn(x, clo), cstep);
    float r = rintf(y);
    r = fminf(fmaxf(r, 0.0f), (float) qmax);
    const float e = sub_rn(x, add_rn(clo, mul_rn(cstep, r)));
    return mul_rn(e, e);
}
SJKVARN_HD inline double dadd_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    volatile double r = a + b; return r;
#endif
}
SJKVARN_HD inline double dsub_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    volatile double r = a - b; return r;
#endif
}
SJKVARN_HD inline double dmul_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    volatile double r = a * b; return r;
#endif
}
SJKVARN_HD inline double ddiv_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a, b);
#else
    volatile double r = a / b; return r;
#endif
}

// one 3-bit code of a token-major stream at bit offset `bit` (host and device)
template<int BITS>
SJKVARN_HD inline uint32_t code_lb_at(const uint8_t * payload, uint32_t bit) {
    const uint32_t byte = bit >> 3, shift = bit & 7;
    uint32_t v = payload[byte];
    if (shift + BITS > 8) {
        v |= (uint32_t) payload[byte + 1] << 8;
    }
    return (v >> shift) & ((1u << BITS) - 1);
}
SJKVARN_HD inline uint32_t code3_at(const uint8_t * payload, uint32_t bit) { return code_lb_at<3>(payload, bit); }
// low-bit trellis window (codebook index) of code (token t, channel d) of a token-major BITS-bit stream with D channels:
// the current code in the top BITS bits over the 6 stream bits before it (the 6/BITS previous codes, older lower);
// history restarts at every SEQ-channel boundary
template<int BITS>
SJKVARN_HD inline uint32_t trellis_lb_window(const uint8_t * payload, int t, int d, int D) {
    // HIST = codes needed to fill the 6 history bits = ceil(6/BITS): 2 (3-bit), 3 (2-bit), 2 (4-bit: 1.5 codes)
    constexpr int SEQ = 128, NWIN = 1 << (6 + BITS), HIST = (6 + BITS - 1)/BITS;
    const uint32_t bit = ((uint32_t) t*(uint32_t) D + (uint32_t) d)*(uint32_t) BITS;
    const int h = d & (SEQ - 1);
    if (h >= HIST) {
        const uint32_t bs = bit - 6u;
        const uint32_t byte = bs >> 3, shift = bs & 7;
        const uint32_t v = (uint32_t) payload[byte] | ((uint32_t) payload[byte + 1] << 8);
        return (v >> shift) & (uint32_t) (NWIN - 1);
    }
    uint32_t w = code_lb_at<BITS>(payload, bit) << 6;
    for (int k = 1; k <= h; ++k) {
        w |= code_lb_at<BITS>(payload, bit - (uint32_t) (k*BITS)) << (6 - k*BITS);
    }
    return w;
}
// Same window as trellis_lb_window, taken from 32-bit words already loaded in registers (the fast attention decode,
// the default; GGML_SJKVARN_TRELLIS_WORDS=0 selects the byte loader). w0..w2 are consecutive payload words starting at word `base`; rel = code bit - 32*base,
// h = channel % 128 (history restart). Requires rel >= min(h*BITS, 6) and the window inside the 96 loaded bits.
// The current code lands in the top BITS bits and missing history (h < HIST) reads as zero, exactly as above.
template<int BITS>
SJKVARN_HD inline uint32_t trellis_lb_window_w(const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t rel, const int h) {
    const uint32_t hb  = (uint32_t) h*BITS < 6u ? (uint32_t) h*BITS : 6u;
    const uint32_t pos = rel - hb;
    const uint32_t n   = BITS + hb;
    const uint64_t lo  = (uint64_t) w0 | ((uint64_t) w1 << 32);
    const uint64_t hi  = (uint64_t) w1 | ((uint64_t) w2 << 32);
    const uint64_t v   = pos + n <= 64u ? lo >> pos : hi >> (pos - 32u);
    return ((uint32_t) v & ((1u << n) - 1u)) << (6u - hb);
}
SJKVARN_HD inline uint32_t trellis3_window(const uint8_t * payload, int t, int d, int D) { return trellis_lb_window<3>(payload, t, d, D); }
SJKVARN_HD inline uint32_t trellis2_window(const uint8_t * payload, int t, int d, int D) { return trellis_lb_window<2>(payload, t, d, D); }
// window of code (t, d) under either payload order: tok = false the production token-major stream (history along the
// channels of row t), tok = true the channel-major stream of GGML_SJKVARN_TRELLIS_TOKENS (history along the tokens of
// channel d; requires G == 128, so the history restarts exactly at each record)
template<int BITS>
SJKVARN_HD inline uint32_t trellis_lb_window_any(const uint8_t * payload, int t, int d, int D, int G, bool tok) {
    return tok ? trellis_lb_window<BITS>(payload, d, t, G) : trellis_lb_window<BITS>(payload, t, d, D);
}
// refit of one row's affine from sequential double sums over its C values (x = balanced value, y = reconstruction):
// mode 1 least squares x ~ a*y + b, mode 2 moment matching (std and mean). Returns false when degenerate.
SJKVARN_HD inline bool trellis3_refit_solve(int mode, double n, double Sx, double Sy, double Sxx, double Syy, double Sxy, float & a, float & b) {
    const double vy = dsub_rn(dmul_rn(n, Syy), dmul_rn(Sy, Sy));
    if (!(vy > 0.0)) { return false; }
    double ad;
    if (mode == 1) {
        ad = ddiv_rn(dsub_rn(dmul_rn(n, Sxy), dmul_rn(Sx, Sy)), vy);
    } else {
        const double vx = dsub_rn(dmul_rn(n, Sxx), dmul_rn(Sx, Sx));
        if (!(vx > 0.0)) { return false; }
        ad = sqrt(ddiv_rn(vx, vy));
    }
    const double bd = ddiv_rn(dsub_rn(Sx, dmul_rn(ad, Sy)), n);
    if (!(ad > 0.0) || !(bd == bd) || ad > 1e30 || fabs(bd) > 1e30) { return false; }
    a = (float) ad; b = (float) bd;
    return true;
}

// (t, d) of nibble `nib` of fragment word `word` of a 4-bit fragment-order payload (inverse of code_bit)
SJKVARN_HD inline void frag_pos(int word, int nib, bool is_V, int D, int & t, int & d) {
    const int lane = word & 31;
    const int tile = (word >> 5) % (D/16);
    const int s    = (word >> 5) / (D/16);
    const int l = nib & 3, e = nib >> 2;
    const int row = (l & 1)*8 + (lane >> 2);
    const int col = 2*((l >> 1)*4 + (lane & 3)) + e;
    if (!is_V) { t = s*16 + row; d = tile*16 + col; }
    else       { d = tile*16 + row; t = s*16 + col; }
}

// window (codebook index) of nibble `nib` of word `word` given the previous word of the same lane sequence
// (one strip = 32*ntiles_c words back, 0 for the first strip)
SJKVARN_HD inline uint32_t trellis_window_of(uint32_t w, uint32_t wp, int nib) {
    const uint64_t bits = ((uint64_t) w << 32) | wp;
    return (uint32_t) (bits >> (32 + 4*nib + 4 - trellis::L)) & (uint32_t) (trellis::NWIN - 1);
}
SJKVARN_HD inline uint32_t trellis_window(const uint32_t * payload, uint32_t word, int nib, int ntiles_c) {
    const uint32_t s  = (word >> 5) / (uint32_t) ntiles_c;
    const uint32_t wp = s > 0 ? payload[word - 32u*(uint32_t) ntiles_c] : 0u;
    return trellis_window_of(payload[word], wp, nib);
}

// Viterbi encoder of one lane sequence: y[n] (balanced, row-normalised values), cbf[NWIN] the codebook as float,
// writes the nstrips = n/8 fragment words of the lane. Reference implementation (see the contract above).
inline void trellis_encode(const float * y, int n, const float * cbf, uint32_t * words) {
    using namespace trellis;
    std::vector<float>   M(NSTATE, 0.0f), Mn(NSTATE);
    std::vector<uint8_t> arg((size_t) n*NSTATE);
    for (int i = n - 1; i >= 0; --i) {
        for (int s = 0; s < NSTATE; ++s) {
            float best = INFINITY; int bj = 0;
            for (int j = 0; j < NBR; ++j) {
                const int w = s + NSTATE*j;
                const float d = sub_rn(y[i], cbf[w]);
                const float e = mul_rn(d, d);
                const float c = add_rn(e, M[w >> 4]);
                if (c < best) { best = c; bj = j; }
            }
            Mn[s] = best;
            arg[(size_t) i*NSTATE + s] = (uint8_t) bj;
        }
        M.swap(Mn);
    }
    for (int st = 0; st < n/8; ++st) words[st] = 0;
    int s = 0;
    for (int i = 0; i < n; ++i) {
        const int j = arg[(size_t) i*NSTATE + s];
        words[i >> 3] |= (uint32_t) j << (4*(i & 7));
        s = (s + NSTATE*j) >> 4;
    }
}

// Trellis-code the 4-bit payload of one tile: bal is the balanced [R][C] tile, lo/step the per-row affine,
// payload the fragment-ordered destination (all words of the tile).
inline void trellis_code_tile(const float * bal, int R, int C, const float * lo, const float * step, bool is_V,
                              int D, int G, uint32_t * payload) {
    using namespace trellis;
    const int ntiles_c = D/16, nstrips = G/16, n = G/2;
    std::vector<float> cbf(NWIN);
    const uint16_t * cbh = cb(is_V);
    for (int w = 0; w < NWIN; ++w) cbf[w] = GGML_FP16_TO_FP32(cbh[w]);
    std::vector<float> y(n);
    std::vector<uint32_t> words(nstrips);
    for (int c = 0; c < ntiles_c; ++c) {
        for (int lane = 0; lane < 32; ++lane) {
            for (int i = 0; i < n; ++i) {
                const int word = ((i >> 3)*ntiles_c + c)*32 + lane;
                int t, d;
                frag_pos(word, i & 7, is_V, D, t, d);
                const int r = is_V ? t : d, col = is_V ? d : t;
                y[i] = div_rn(sub_rn(bal[(size_t) r*C + col], lo[r]), step[r]);
            }
            trellis_encode(y.data(), n, cbf.data(), words.data());
            for (int st = 0; st < nstrips; ++st) payload[(st*ntiles_c + c)*32 + lane] = words[st];
        }
    }
    GGML_UNUSED(R);
}

// payload helpers defined further down (used by the trellis3 row coders)
SJKVARN_HD inline uint32_t code_bit(int t, int d, bool is_V, int D, int G, int bits);
SJKVARN_HD inline int k_ch_idx(int d, int D, int G, int bits);
inline void pack_code(uint8_t * payload, uint32_t bit, int bits, uint32_t v);
inline void put_half(uint8_t * rec, size_t off, float v);

// Viterbi encoder of one low-bit trellis sequence (contract of trellis_encode: IEEE ops without contraction, strict <
// over ascending branches, start state 0, no termination); codes[n] receives the BITS-bit codes.
template<int BITS>
inline void trellis_lb_encode(const float * y, int n, const float * cbf, uint8_t * codes) {
    constexpr int NSTATE = trellis_lb<BITS>::NSTATE, NBR = trellis_lb<BITS>::NBR;
    std::vector<float>   M(NSTATE, 0.0f), Mn(NSTATE);
    std::vector<uint8_t> arg((size_t) n*NSTATE);
    for (int i = n - 1; i >= 0; --i) {
        for (int s = 0; s < NSTATE; ++s) {
            float best = INFINITY; int bj = 0;
            for (int j = 0; j < NBR; ++j) {
                const int w = s + NSTATE*j;
                const float d = sub_rn(y[i], cbf[w]);
                const float e = mul_rn(d, d);
                const float c = add_rn(e, M[w >> BITS]);
                if (c < best) { best = c; bj = j; }
            }
            Mn[s] = best;
            arg[(size_t) i*NSTATE + s] = (uint8_t) bj;
        }
        M.swap(Mn);
    }
    int s = 0;
    for (int i = 0; i < n; ++i) {
        const int j = arg[(size_t) i*NSTATE + s];
        codes[i] = (uint8_t) j;
        s = (s + NSTATE*j) >> BITS;
    }
}

// Trellis-code the low-bit payload of one tile: bal is the balanced [R][C] tile (K: rows = channels, V: rows = tokens),
// lo/step the per-row affine; payload the token-major bit stream of the tile (zeroed by the caller).
template<int BITS>
inline void trellis_lb_code_rows(const float * bal, int R, int C, const float * lo, const float * step, bool is_V,
                                 int D, int G, uint8_t * payload) {
    constexpr int SEQ = trellis_lb<BITS>::SEQ, NWIN = trellis_lb<BITS>::NWIN;
    GGML_ASSERT(D % SEQ == 0);
    std::vector<float> cbf(NWIN);
    const uint16_t * cbh = trellis_lb<BITS>::cb(is_V);
    for (int w = 0; w < NWIN; ++w) cbf[w] = GGML_FP16_TO_FP32(cbh[w]);
    std::vector<float> y(SEQ);
    std::vector<uint8_t> codes(SEQ);
    if (BITS == 4 || trellis3::tokens()) { // trellis4 is token-axis only
        GGML_ASSERT(G == SEQ);
        for (int d = 0; d < D; ++d) {
            for (int t = 0; t < G; ++t) {
                const int r = is_V ? t : d, col = is_V ? d : t;
                y[t] = div_rn(sub_rn(bal[(size_t) r*C + col], lo[r]), step[r]);
            }
            trellis_lb_encode<BITS>(y.data(), SEQ, cbf.data(), codes.data());
            for (int t = 0; t < G; ++t) {
                pack_code(payload, ((uint32_t) d*(uint32_t) G + (uint32_t) t)*BITS, BITS, codes[t]);
            }
        }
        GGML_UNUSED(R);
        return;
    }
    for (int t = 0; t < G; ++t) {
        for (int h = 0; h < D/SEQ; ++h) {
            for (int i = 0; i < SEQ; ++i) {
                const int d = h*SEQ + i;
                const int r = is_V ? t : d, col = is_V ? d : t;
                y[i] = div_rn(sub_rn(bal[(size_t) r*C + col], lo[r]), step[r]);
            }
            trellis_lb_encode<BITS>(y.data(), SEQ, cbf.data(), codes.data());
            for (int i = 0; i < SEQ; ++i) {
                pack_code(payload, code_bit(t, h*SEQ + i, is_V, D, G, BITS), BITS, codes[i]);
            }
        }
    }
    GGML_UNUSED(R);
}

// Refit the per-row scale/zero of a low-bit trellis payload (mode: trellis3::refit_mode). Sequential double sums over
// the row's columns in column order, shared arithmetic with the CUDA sealer.
template<int BITS>
inline void trellis_lb_refit_rows(const float * bal, int R, int C, const float * s_row, bool is_V, int D, int G,
                                  const uint8_t * payload, uint8_t * rec, const layout & l, int mode) {
    if (mode == 0) { return; }
    const uint16_t * cbh = trellis_lb<BITS>::cb(is_V);
    for (int r = 0; r < R; ++r) {
        double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
        for (int c = 0; c < C; ++c) {
            const int t = is_V ? r : c, d = is_V ? c : r;
            const double x  = (double) bal[(size_t) r*C + c];
            const double yh = (double) GGML_FP16_TO_FP32(cbh[trellis_lb_window_any<BITS>(payload, t, d, D, G, BITS == 4 || trellis3::tokens())]);
            Sx = dadd_rn(Sx, x); Sy = dadd_rn(Sy, yh);
            Sxx = dadd_rn(Sxx, dmul_rn(x, x)); Syy = dadd_rn(Syy, dmul_rn(yh, yh)); Sxy = dadd_rn(Sxy, dmul_rn(x, yh));
        }
        float a, b;
        if (!trellis3_refit_solve(mode, (double) C, Sx, Sy, Sxx, Syy, Sxy, a, b)) { continue; }
        const int ri = is_V ? r : k_ch_idx(r, D, G, l.bits_k);
        put_half(rec, (is_V ? l.v_scale : l.k_scale) + 2*ri, mul_rn(s_row[r], a));
        put_half(rec, (is_V ? l.v_zero  : l.k_zero)  + 2*ri, mul_rn(s_row[r], b));
    }
}

// ---------------------------------------------------------------------------------------------------------
// Payload and metadata order.
//
// 4-bit payloads (the production width) are stored in "fragment order": the order in which one warp of the
// CUDA decode kernel consumes them as m16n8k16 tensor-core A operands, so a lane's 32-bit word is exactly its
// fragment and no shared-memory transpose is needed. Tokens are grouped in strips of 16 and channels in
// tiles of 16. For K (A = K rows, k = channel) word (strip s, channel tile ks, lane) holds the codes of
// tokens 16s + lane/4 (+8) at channels 16ks + 2(lane%4) + {0,1} (+8); for V (A = V^T, rows = channels,
// k = tokens) word (s, channel tile dt, lane) holds channels 16dt + lane/4 (+8) at tokens 16s + 2(lane%4)
// + {0,1} (+8). Inside a word, fragment register l (0..3) is the half2 pair (nibble l, nibble l+4), i.e.
// element e of pair l sits in nibble l + 4e (see fattn_sj_kvarn_decode_word). The per-channel fp16 vectors
// Kscale/Kzero/Vch are permuted the same way so that a lane finds its channels contiguous
// (k_ch_idx / v_ch_idx); Ktok/Vscale/Vzero stay in token order.
//
// Any other bit width keeps the plain token-major bit stream (token row t, value d at bit d*bits).
// ---------------------------------------------------------------------------------------------------------
SJKVARN_HD inline bool frag_order(int bits, int D, int G) {
    return bits == 4 && D % 16 == 0 && G % 16 == 0;
}

// bit offset (inside the K or V payload) of token t, channel d
SJKVARN_HD inline uint32_t code_bit(int t, int d, bool is_V, int D, int G, int bits) {
    if (!frag_order(bits, D, G)) {
        return (uint32_t) t * (uint32_t) (D * bits) + (uint32_t) d * bits;
    }
    const int s = t >> 4, tt = t & 15;
    const int c = d >> 4, dd = d & 15;
    int lane, l, e;
    if (!is_V) {
        lane = 4*(tt & 7) + ((dd >> 1) & 3);
        l    = (tt >> 3) + 2*(dd >> 3);
        e    = dd & 1;
    } else {
        lane = 4*(dd & 7) + ((tt >> 1) & 3);
        l    = (dd >> 3) + 2*(tt >> 3);
        e    = tt & 1;
    }
    const uint32_t word = (uint32_t) (s*(D/16) + c)*32u + (uint32_t) lane;
    return word*32u + (uint32_t) (l + 4*e)*4u;
}

// index of channel d inside Kscale/Kzero (K fragment order: class lane%4, then channel tile, then pair, then element)
SJKVARN_HD inline int k_ch_idx(int d, int D, int G, int bits) {
    if (!frag_order(bits, D, G)) {
        return d;
    }
    const int c = d >> 4, dd = d & 15, jj = dd >> 1, e = dd & 1;
    return (jj & 3)*(D/4) + c*4 + (jj >> 2)*2 + e;
}

// index of channel d inside Vch (V fragment order: class lane/4, then channel tile, then half)
SJKVARN_HD inline int v_ch_idx(int d, int D, int G, int bits) {
    if (!frag_order(bits, D, G)) {
        return d;
    }
    const int c = d >> 4, dd = d & 15;
    return (dd & 7)*(D/8) + c*2 + (dd >> 3);
}

// value v at bit offset `bit` of a payload (bits <= 8), little-endian across bytes
inline void pack_code(uint8_t * payload, uint32_t bit, int bits, uint32_t v) {
    const uint32_t byte  = bit >> 3;
    const uint32_t shift = bit & 7;
    const uint32_t w = (v & ((1u << bits) - 1)) << shift;
    payload[byte] |= (uint8_t) w;
    if (shift + bits > 8) {
        payload[byte + 1] |= (uint8_t) (w >> 8);
    }
}

inline uint32_t unpack_code(const uint8_t * payload, uint32_t bit, int bits) {
    const uint32_t byte  = bit >> 3;
    const uint32_t shift = bit & 7;
    uint32_t w = payload[byte];
    if (shift + bits > 8) {
        w |= (uint32_t) payload[byte + 1] << 8;
    }
    return (w >> shift) & ((1u << bits) - 1);
}

inline uint32_t k_code(const uint8_t * rec, const layout & l, int t, int d) {
    return unpack_code(rec + l.k_payload, code_bit(t, d, false, l.D, l.G, l.bits_k), l.bits_k);
}
inline uint32_t v_code(const uint8_t * rec, const layout & l, int t, int d) {
    return unpack_code(rec + l.v_payload, code_bit(t, d, true, l.D, l.G, l.bits_v), l.bits_v);
}

inline void put_half(uint8_t * rec, size_t off, float v) {
    const ggml_fp16_t h = GGML_FP32_TO_FP16(v);
    memcpy(rec + off, &h, 2);
}
inline float get_half(const uint8_t * rec, size_t off) {
    ggml_fp16_t h;
    memcpy(&h, rec + off, 2);
    return GGML_FP16_TO_FP32(h);
}

// host per-row clip search over a contiguous row of n values (n even): two halves summed sequentially, added p0 + p1
inline void row_clip_search(const float * row, int n, uint32_t qmax, float & lo, float & hi) {
    float best = 0.0f; int bi = 0;
    for (int i = 0; i < SJKVARN_CLIP_NCAND; ++i) {
        float clo, cstep;
        clip_cand_range(lo, hi, i, qmax, clo, cstep);
        float p[2] = { 0.0f, 0.0f };
        for (int h = 0; h < 2; ++h) {
            for (int c = h*(n/2); c < (h + 1)*(n/2); ++c) { p[h] = add_rn(p[h], clip_err_term(row[c], clo, cstep, qmax)); }
        }
        const float e = add_rn(p[0], p[1]);
        if (i == 0 || e < best) { best = e; bi = i; }
    }
    if (bi > 0) { clip_range(lo, hi, clip_cand(bi)); }
}

// Seal one (head, group): K and V are the rotated fp16 rows, row t at K + t*row_stride (elements),
// D contiguous values per row. Writes l.bytes bytes to rec.
inline void seal_group(const ggml_fp16_t * K, const ggml_fp16_t * V, size_t row_stride, const layout & l, int iters, uint8_t * rec, bool trellis_body = false) {
    const int D = l.D, G = l.G;
    std::vector<float> tile((size_t) D*G), bal((size_t) D*G), s_row(std::max(D, G)), s_col(std::max(D, G));
    memset(rec, 0, l.bytes);
    GGML_ASSERT(!trellis_body || ((l.bits_k >= 2 && l.bits_k <= 4) && (l.bits_v >= 2 && l.bits_v <= 4)));

    // K: tile [D][G] (rows = channels, cols = tokens)
    for (int t = 0; t < G; ++t) {
        for (int d = 0; d < D; ++d) {
            tile[(size_t) d*G + t] = GGML_FP16_TO_FP32(K[t*row_stride + d]);
        }
    }
    balance(tile.data(), D, G, iters, s_row.data(), s_col.data(), bal.data());
    {
        const uint32_t qmax = (1u << l.bits_k) - 1;
        std::vector<float> q_lo(trellis_body ? D : 0), q_step(trellis_body ? D : 0);
        for (int d = 0; d < D; ++d) {
            const float * row = bal.data() + (size_t) d*G;
            float lo = row[0], hi = row[0];
            for (int t = 1; t < G; ++t) { lo = std::min(lo, row[t]); hi = std::max(hi, row[t]); }
            if (!trellis_body && scalar_clip::factors()[0] != 1.0f) {
                if (scalar_clip::factors()[0] < 0.0f) { row_clip_search(row, G, qmax, lo, hi); } else { clip_range(lo, hi, scalar_clip::factors()[0]); }
            }
            const float step = std::max((hi - lo) / (float) qmax, 1e-10f);
            const int di = k_ch_idx(d, D, G, l.bits_k);
            put_half(rec, l.k_scale + 2*di, s_row[d] * step);
            put_half(rec, l.k_zero  + 2*di, s_row[d] * lo);
            if (trellis_body) {
                q_lo[d] = lo; q_step[d] = step;
                continue;
            }
            for (int t = 0; t < G; ++t) {
                pack_code(rec + l.k_payload, code_bit(t, d, false, D, G, l.bits_k), l.bits_k, rtn_code(row[t], lo, step, qmax));
            }
        }
        if (trellis_body && l.bits_k == 3) {
            trellis_lb_code_rows<3>(bal.data(), D, G, q_lo.data(), q_step.data(), false, D, G, rec + l.k_payload);
            trellis_lb_refit_rows<3>(bal.data(), D, G, s_row.data(), false, D, G, rec + l.k_payload, rec, l, trellis3::refit_mode());
        } else if (trellis_body && l.bits_k == 2) {
            trellis_lb_code_rows<2>(bal.data(), D, G, q_lo.data(), q_step.data(), false, D, G, rec + l.k_payload);
            trellis_lb_refit_rows<2>(bal.data(), D, G, s_row.data(), false, D, G, rec + l.k_payload, rec, l, trellis2::refit_mode());
        } else if (trellis_body && trellis3::tokens4()) { // 4-bit token-axis trellis (trellis4, opt-in)
            trellis_lb_code_rows<4>(bal.data(), D, G, q_lo.data(), q_step.data(), false, D, G, rec + l.k_payload);
            trellis_lb_refit_rows<4>(bal.data(), D, G, s_row.data(), false, D, G, rec + l.k_payload, rec, l, trellis3::refit_mode());
        } else if (trellis_body) {
            GGML_ASSERT(frag_order(l.bits_k, D, G) && G % 16 == 0);
            trellis_code_tile(bal.data(), D, G, q_lo.data(), q_step.data(), false, D, G, (uint32_t *) (rec + l.k_payload));
        }
        for (int t = 0; t < G; ++t) {
            put_half(rec, l.k_tok + 2*t, s_col[t]);
        }
    }

    // V: tile [G][D] (rows = tokens, cols = channels)
    for (int t = 0; t < G; ++t) {
        for (int d = 0; d < D; ++d) {
            tile[(size_t) t*D + d] = GGML_FP16_TO_FP32(V[t*row_stride + d]);
        }
    }
    balance(tile.data(), G, D, iters, s_row.data(), s_col.data(), bal.data());
    {
        const uint32_t qmax = (1u << l.bits_v) - 1;
        std::vector<float> q_lo(trellis_body ? G : 0), q_step(trellis_body ? G : 0);
        for (int t = 0; t < G; ++t) {
            const float * row = bal.data() + (size_t) t*D;
            float lo = row[0], hi = row[0];
            for (int d = 1; d < D; ++d) { lo = std::min(lo, row[d]); hi = std::max(hi, row[d]); }
            if (!trellis_body && scalar_clip::factors()[1] != 1.0f) {
                if (scalar_clip::factors()[1] < 0.0f) { row_clip_search(row, D, qmax, lo, hi); } else { clip_range(lo, hi, scalar_clip::factors()[1]); }
            }
            const float step = std::max((hi - lo) / (float) qmax, 1e-10f);
            put_half(rec, l.v_scale + 2*t, s_row[t] * step);
            put_half(rec, l.v_zero  + 2*t, s_row[t] * lo);
            if (trellis_body) {
                q_lo[t] = lo; q_step[t] = step;
                continue;
            }
            for (int d = 0; d < D; ++d) {
                pack_code(rec + l.v_payload, code_bit(t, d, true, D, G, l.bits_v), l.bits_v, rtn_code(row[d], lo, step, qmax));
            }
        }
        if (trellis_body && l.bits_v == 3) {
            trellis_lb_code_rows<3>(bal.data(), G, D, q_lo.data(), q_step.data(), true, D, G, rec + l.v_payload);
            trellis_lb_refit_rows<3>(bal.data(), G, D, s_row.data(), true, D, G, rec + l.v_payload, rec, l, trellis3::refit_mode());
        } else if (trellis_body && l.bits_v == 2) {
            trellis_lb_code_rows<2>(bal.data(), G, D, q_lo.data(), q_step.data(), true, D, G, rec + l.v_payload);
            trellis_lb_refit_rows<2>(bal.data(), G, D, s_row.data(), true, D, G, rec + l.v_payload, rec, l, trellis2::refit_mode());
        } else if (trellis_body && trellis3::tokens4()) { // 4-bit token-axis trellis (trellis4, opt-in)
            trellis_lb_code_rows<4>(bal.data(), G, D, q_lo.data(), q_step.data(), true, D, G, rec + l.v_payload);
            trellis_lb_refit_rows<4>(bal.data(), G, D, s_row.data(), true, D, G, rec + l.v_payload, rec, l, trellis3::refit_mode());
        } else if (trellis_body) {
            GGML_ASSERT(frag_order(l.bits_v, D, G) && G % 16 == 0);
            trellis_code_tile(bal.data(), G, D, q_lo.data(), q_step.data(), true, D, G, (uint32_t *) (rec + l.v_payload));
        }
        for (int d = 0; d < D; ++d) {
            put_half(rec, l.v_ch + 2*v_ch_idx(d, D, G, l.bits_v), s_col[d]);
        }
    }
}

// rotated-domain reconstruction of token t of a record
// value of the code at bit offset `bit` of a trellis-coded payload
inline float trellis_value(const uint8_t * payload, uint32_t bit, bool is_V, int D, int G) {
    const uint32_t idx = trellis_window((const uint32_t *) payload, bit >> 5, (bit & 31) >> 2, D/16);
    GGML_UNUSED(G);
    return GGML_FP16_TO_FP32(trellis::cb(is_V)[idx]);
}

inline void decode_k_row(const uint8_t * rec, const layout & l, int t, float * out, bool trellis_body = false) {
    const float tok = get_half(rec, l.k_tok + 2*t);
    for (int d = 0; d < l.D; ++d) {
        const float q  = !trellis_body    ? (float) k_code(rec, l, t, d)
                       : l.bits_k == 3    ? GGML_FP16_TO_FP32(trellis3::cb(false)[trellis_lb_window_any<3>(rec + l.k_payload, t, d, l.D, l.G, trellis3::tokens())])
                       : l.bits_k == 2    ? GGML_FP16_TO_FP32(trellis2::cb(false)[trellis_lb_window_any<2>(rec + l.k_payload, t, d, l.D, l.G, trellis3::tokens())])
                       : trellis3::tokens4() ? GGML_FP16_TO_FP32(trellis4::cb(false)[trellis_lb_window_any<4>(rec + l.k_payload, t, d, l.D, l.G, true)])
                                          : trellis_value(rec + l.k_payload, code_bit(t, d, false, l.D, l.G, l.bits_k), false, l.D, l.G);
        const int   di = k_ch_idx(d, l.D, l.G, l.bits_k);
        out[d] = (q * get_half(rec, l.k_scale + 2*di) + get_half(rec, l.k_zero + 2*di)) * tok;
    }
}
inline void decode_v_row(const uint8_t * rec, const layout & l, int t, float * out, bool trellis_body = false) {
    const float sc = get_half(rec, l.v_scale + 2*t);
    const float zp = get_half(rec, l.v_zero  + 2*t);
    for (int d = 0; d < l.D; ++d) {
        const float q = !trellis_body    ? (float) v_code(rec, l, t, d)
                      : l.bits_v == 3    ? GGML_FP16_TO_FP32(trellis3::cb(true)[trellis_lb_window_any<3>(rec + l.v_payload, t, d, l.D, l.G, trellis3::tokens())])
                      : l.bits_v == 2    ? GGML_FP16_TO_FP32(trellis2::cb(true)[trellis_lb_window_any<2>(rec + l.v_payload, t, d, l.D, l.G, trellis3::tokens())])
                      : trellis3::tokens4() ? GGML_FP16_TO_FP32(trellis4::cb(true)[trellis_lb_window_any<4>(rec + l.v_payload, t, d, l.D, l.G, true)])
                                         : trellis_value(rec + l.v_payload, code_bit(t, d, true, l.D, l.G, l.bits_v), true, l.D, l.G);
        out[d] = (q * sc + zp) * get_half(rec, l.v_ch + 2*v_ch_idx(d, l.D, l.G, l.bits_v));
    }
}

} // namespace ggml_sj_kvarn
