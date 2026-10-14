// AI4 (Ampere INT4, formerly D2): integer-output bitshift trellis (4 bits/weight) in RQ4's rotated basis, for the SM86 W4A8 IMMA path.
//
// Format (whole-tensor type like EXL3: rows come in slabs of AI4_TR = 16 output channels; K = ne0 % 1024 == 0,
// M = ne1 % 16 == 0):
//   - Every 256-wide K block is rotated offline with RQ4's rotation (w' = (1/16) H256 D w). The sign diagonal
//     and Hadamard order are copied from ggml-rq4.h (track B) and must stay identical to it, because the
//     activations come from B's rotate+quantize kernel.
//   - A trellis tile is 16 rows x 8 K-columns = 128 weights, streamed row-major (n = r*8 + c). 128 symbols of
//     4 bits = 64 bytes = 16 uint32 words, tail-biting. Symbol n sits in word n/8 at bit 4*(7 - n%8) (first
//     symbol in the top nibble), so word r of a tile holds row r's 8 symbols. The 16-bit state of weight n is
//         state_n = sym[n-3] << 12 | sym[n-2] << 8 | sym[n-1] << 4 | sym[n]      (indices mod 128)
//     i.e. ((word[w-1]:word[w]) >> 4*(7 - n%8)) & 0xFFFF with w = n/8 and word[-1] = word[15].
//     An m16n8k32 A fragment (rows g, g+8; k 4t..4t+3 and 16+4t..+3) reads 4 consecutive symbols per row.
//   - Level of a state (s8, |level| <= 128):
//         x = state * 0x83DCD12D (mod 2^32);  b = sum of the 4 bytes of x (0..1020);  level = ((b + 2) >> 2) - 128
//     This is EXL3's mul1 bytesum reduced to 256 levels; the GPU computes it with IMUL + DP4A + 2 ALU ops.
//   - w'[r][k] ~= d[r][k/128] * level: one fp16 scale per (row, 128-wide K group), the RQ4_G128 scale grid, so
//     B's activation kernel and scale epilogue apply unchanged.
//   - Slab super block (16 rows x 1024 K) = [fp16 d[16][8]] [128 tiles x 64 bytes, tile index = K-column step
//     0..127] = 8448 bytes = 16 x 528 (4.125 bpw). A slab row of K is K/1024 consecutive super blocks.
//   - Encoder: per 256 block, the 32 column steps are encoded in order (Viterbi per tile), and with an imatrix
//     each step's error is pushed onto the later columns through U = chol(Hr^-1) (block LDLQ, the GPTQ
//     update with 8-column blocks; Hr = rotated diag(imatrix), as in RQ4's GPTQ).

#pragma once


#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#define AI4_SB     1024
#define AI4_HB     256
#define AI4_TILE   128
#define AI4_NSTATE 65536
#define AI4_MUL    0x83DCD12Du
#define AI4_TR     16     // tile rows (one slab)
#define AI4_TC     8      // tile K-columns
#define AI4_GS     128    // scale group (K)
#define AI4_SCALE_BYTES   (2*AI4_TR*(AI4_SB/AI4_GS))                 // 256
#define AI4_SLAB_SB_BYTES (AI4_SCALE_BYTES + (AI4_SB/AI4_TC)*64)      // 8448 = 16 x 528

#ifdef __cplusplus
extern "C" {
#endif

// ---- rotation: identical to ggml-rq4.h (RQ4_SIGN_WORDS, rq4_rotate256 / rq4_unrotate256) ----
static const uint32_t ai4_sign_bits[8] = {
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu,
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au };

static inline int ai4_sign_neg(int i) {
    return (ai4_sign_bits[i >> 5] >> (i & 31)) & 1;
}

static inline void ai4_fwht256(float * x) {
    for (int h = 1; h < AI4_HB; h <<= 1) {
        for (int i = 0; i < AI4_HB; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

static inline void ai4_rotate256(float * x) {
    for (int i = 0; i < AI4_HB; ++i) {
        if (ai4_sign_neg(i)) {
            x[i] = -x[i];
        }
    }
    ai4_fwht256(x);
    for (int i = 0; i < AI4_HB; ++i) {
        x[i] *= 1.0f/16.0f;
    }
}

static inline void ai4_unrotate256(float * x) {
    ai4_fwht256(x);
    for (int i = 0; i < AI4_HB; ++i) {
        x[i] *= ai4_sign_neg(i) ? -1.0f/16.0f : 1.0f/16.0f;
    }
}

// ---- fp16 (IEEE binary16, round to nearest even), no ggml deps ----
static inline uint16_t ai4_fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7fffffu;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    if (((x >> 23) & 0xff) == 0xff) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) {
            h++;
        }
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) {
        h++;
    }
    return (uint16_t)(sign | h);
}

static inline float ai4_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &x, 4);
    return f;
}

static inline float ai4_fp16_round(float f) {
    return ai4_fp16_to_fp32(ai4_fp32_to_fp16(f));
}

// ---- codebook ----
// The level of a 16-bit state under the trellis constant C (C is part of the AI+ type: ggml_ai4_trellis_mul).
static inline int aik_level_c(uint32_t state, uint32_t C) {
    const uint32_t x = state * C;
    const int b = (int)(x & 255) + (int)((x >> 8) & 255) + (int)((x >> 16) & 255) + (int)(x >> 24);
    return ((b + 2) >> 2) - 128;
}

static inline int ai4_level(uint32_t state) {
    return aik_level_c(state, AI4_MUL);
}

// RMS of the level over all 65536 states, per constant (used to seed the scale search). Cached for up to AIK_RMS_CACHE
// constants; the cache is written at most once per constant and the value does not depend on the order of the writes,
// so a benign race between threads cannot change the result.
#define AIK_RMS_CACHE 8
static inline float aik_level_rms_c(uint32_t C) {
    static uint32_t cc[AIK_RMS_CACHE];
    static float    cr[AIK_RMS_CACHE];
    for (int i = 0; i < AIK_RMS_CACHE; ++i) {
        if (cr[i] != 0.0f && cc[i] == C) {
            return cr[i];
        }
    }
    double s = 0;
    for (uint32_t st = 0; st < AI4_NSTATE; ++st) {
        const double l = aik_level_c(st, C);
        s += l*l;
    }
    const float r = (float) sqrt(s/AI4_NSTATE);
    for (int i = 0; i < AIK_RMS_CACHE; ++i) {
        if (cr[i] == 0.0f) {
            cc[i] = C;
            cr[i] = r;
            break;
        }
    }
    return r;
}

static inline float ai4_level_rms(void) {
    return aik_level_rms_c(AI4_MUL);
}


// ---- tile bit access ----
static inline int ai4_get_sym(const uint32_t * wd, int n) {
    return (int)((wd[n >> 3] >> (4*(7 - (n & 7)))) & 0xF);
}

static inline uint32_t ai4_get_state(const uint32_t * wd, int n) {
    const int w = n >> 3;
    const uint64_t c = ((uint64_t) wd[(w + 15) & 15] << 32) | wd[w];
    return (uint32_t)((c >> (4*(7 - (n & 7)))) & 0xFFFFu);
}

// ---- encoder ----
// Workspace for one encoder thread (about 2.8 MB). arg holds 128 x 2^(16-k) decisions, so it is sized for the lowest
// rate (k = 2: 16384 predecessor classes); mscr is the per-step min-over-hi scratch (2^(16-k) entries), kept here and
// not on the stack because it is 64 KB at k = 2. lev is the level table of the constant `mul`.
#define AIK_MAXTL 16384   // 2^(16 - 2): predecessor classes at the lowest rate
typedef struct {
    float    cur[AI4_NSTATE];
    float    nxt[AI4_NSTATE];
    uint8_t  arg[AI4_TILE*AIK_MAXTL];
    float    lev[AI4_NSTATE];
    float    mscr[AIK_MAXTL];
    uint32_t mul;
} ai4_work;

// Point the workspace's level table at the constant C (no-op when it already holds it).
static inline void aik_work_set_mul(ai4_work * w, uint32_t C) {
    if (w == NULL || w->mul == C) {
        return;
    }
    for (uint32_t s = 0; s < AI4_NSTATE; ++s) {
        w->lev[s] = (float) aik_level_c(s, C);
    }
    w->mul = C;
}

static inline ai4_work * ai4_work_new(void) {
    ai4_work * w = (ai4_work *) malloc(sizeof(ai4_work));
    if (w) {
        w->mul = 0;
        aik_work_set_mul(w, AI4_MUL);
    }
    return w;
}

// Viterbi over y[0..127] (level units). t0 < 0: free start. t0 >= 0: the first state's top 12 bits and the
// last state's low 12 bits must both equal t0 (tail-biting). Writes the 128 states to st; returns the cost.
static inline float ai4_viterbi(ai4_work * ws, const float * y, int t0, uint16_t * st) {
    float * cur = ws->cur, * nxt = ws->nxt;
    const float * L = ws->lev;
    for (int s = 0; s < AI4_NSTATE; ++s) {
        const float e = L[s] - y[0];
        cur[s] = (t0 >= 0 && (s >> 4) != t0) ? FLT_MAX : e*e;
    }
    float m[4096];
    for (int i = 1; i < AI4_TILE; ++i) {
        uint8_t * a = ws->arg + (size_t) i*4096;
        for (int t = 0; t < 4096; ++t) {
            m[t] = cur[t];
            a[t] = 0;
        }
        for (int hi = 1; hi < 16; ++hi) {
            const float * c = cur + (hi << 12);
            for (int t = 0; t < 4096; ++t) {
                const int lt = c[t] < m[t];
                m[t] = lt ? c[t] : m[t];
                a[t] = lt ? (uint8_t) hi : a[t];
            }
        }
        const float yi = y[i];
        for (int s = 0; s < AI4_NSTATE; ++s) {
            const float e = L[s] - yi;
            nxt[s] = m[s >> 4] + e*e;
        }
        float * tmp = cur; cur = nxt; nxt = tmp;
    }
    int best = 0;
    float bc = FLT_MAX;
    for (int s = 0; s < AI4_NSTATE; ++s) {
        if (t0 >= 0 && (s & 0xFFF) != t0) {
            continue;
        }
        if (cur[s] < bc) {
            bc = cur[s];
            best = s;
        }
    }
    int s = best;
    for (int i = AI4_TILE - 1; i >= 0; --i) {
        st[i] = (uint16_t) s;
        if (i > 0) {
            const int t = s >> 4;
            s = t | (ws->arg[(size_t) i*4096 + t] << 12);
        }
    }
    return bc;
}

// Tail-biting encode of one tile (targets y in level units): pass 1 runs on the tile rotated by AI4_ROLL to find
// the wrap-around state, pass 2 runs with that state forced at both ends. AI4_ROLL = 31 puts original index 127 at
// rolled index 96, inside the GPU encoder's last 32-step decision window (tools/ai4/ai4-encode.cu).
#ifndef AI4_ROLL
#define AI4_ROLL 31
#endif
static inline void ai4_encode_tile(ai4_work * ws, const float * y, uint16_t * st) {
    float yr[AI4_TILE];
    uint16_t sr[AI4_TILE];
    for (int i = 0; i < AI4_TILE; ++i) {
        yr[i] = y[(i + AI4_ROLL) % AI4_TILE];
    }
    ai4_viterbi(ws, yr, -1, sr);
    const int t0 = sr[AI4_TILE - 1 - AI4_ROLL] & 0xFFF;   // state at original index 127
    ai4_viterbi(ws, y, t0, st);
}

// ---- block LDLQ factor: U = chol(Hr^-1)^T for one 256 block, Hr = R diag(lambda) R^T (same as RQ4's GPTQ) ----
#ifndef AI4_LDLQ_DAMP
#define AI4_LDLQ_DAMP 0.01
#endif

static inline int ai4_ldlq_prep(const float * lambda, float * U) {
    double mean = 0.0;
    for (int i = 0; i < AI4_HB; ++i) {
        mean += (double) lambda[i];
    }
    mean /= AI4_HB;
    if (!(mean > 0.0)) {
        return 0;
    }
    double f[AI4_HB];
    for (int i = 0; i < AI4_HB; ++i) {
        f[i] = 1.0/((double) lambda[i] + (double) AI4_LDLQ_DAMP*mean);
    }
    for (int len = 1; len < AI4_HB; len <<= 1) {
        for (int i = 0; i < AI4_HB; i += 2*len) {
            for (int j = i; j < i + len; ++j) {
                const double a = f[j], b = f[j + len];
                f[j] = a + b;
                f[j + len] = a - b;
            }
        }
    }
    const int n = AI4_HB;
    double * L = (double *) malloc(sizeof(double)*n*n);
    if (L == NULL) {
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = f[i ^ j]/256.0;
            for (int k = 0; k < j; ++k) {
                v -= L[i*n + k]*L[j*n + k];
            }
            if (i == j) {
                if (!(v > 0.0)) {
                    free(L);
                    return 0;
                }
                L[i*n + i] = sqrt(v);
            } else {
                L[i*n + j] = v/L[j*n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i*n + j] = j >= i ? (float) L[j*n + i] : 0.0f;
        }
    }
    free(L);
    return 1;
}

static inline void ai4_inv_upper8(const double * A, double * X) {
    for (int i = 0; i < AI4_TC*AI4_TC; ++i) {
        X[i] = 0.0;
    }
    for (int j = 0; j < AI4_TC; ++j) {
        X[j*AI4_TC + j] = 1.0/A[j*AI4_TC + j];
        for (int i = j - 1; i >= 0; --i) {
            double s = 0.0;
            for (int k = i + 1; k <= j; ++k) {
                s += A[i*AI4_TC + k]*X[k*AI4_TC + j];
            }
            X[i*AI4_TC + j] = -s/A[i*AI4_TC + i];
        }
    }
}

#ifndef AI4_SCALE_MUL
#define AI4_SCALE_MUL 1.10f   // scale seed = group rms / level rms * this (probe: 1.1 beat 1.0 on all 8 tensors)
#endif

// Encode one slab (16 rows) x one 256 block. y: rotated weights, row stride ys (modified: holds the LDLQ-
// updated targets afterwards). d: the two fp16-rounded scales per row (d[r*2 + g]). U: LDLQ factor or NULL
// (no feedback). tiles: 32 tiles x 16 words for this block's column steps.
static inline void ai4_encode_slab_block(ai4_work * ws, float * y, int ys, const float * d, const float * U, uint32_t * tiles) {
    for (int c0 = 0; c0 < AI4_HB; c0 += AI4_TC) {
        float t[AI4_TILE];
        uint16_t st[AI4_TILE];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int j = 0; j < AI4_TC; ++j) {
                const float dr = d[r*2 + (c0 + j)/AI4_GS];
                t[r*AI4_TC + j] = dr > 0.0f ? y[r*ys + c0 + j]/dr : 0.0f;
            }
        }
        ai4_encode_tile(ws, t, st);
        uint32_t * wd = tiles + (c0/AI4_TC)*16;
        memset(wd, 0, 64);
        for (int n = 0; n < AI4_TILE; ++n) {
            wd[n >> 3] |= (uint32_t)(st[n] & 0xF) << (4*(7 - (n & 7)));
        }
        if (U && c0 + AI4_TC < AI4_HB) {
            double Ub[AI4_TC*AI4_TC], Ui[AI4_TC*AI4_TC];
            for (int a = 0; a < AI4_TC; ++a) {
                for (int b = 0; b < AI4_TC; ++b) {
                    Ub[a*AI4_TC + b] = U[(c0 + a)*AI4_HB + c0 + b];
                }
            }
            ai4_inv_upper8(Ub, Ui);
            for (int r = 0; r < AI4_TR; ++r) {
                double e[AI4_TC], f[AI4_TC];
                for (int j = 0; j < AI4_TC; ++j) {
                    e[j] = y[r*ys + c0 + j] - (double) d[r*2 + (c0 + j)/AI4_GS]*ws->lev[st[r*AI4_TC + j]];
                }
                for (int j = 0; j < AI4_TC; ++j) {
                    double s = 0.0;
                    for (int a = 0; a <= j; ++a) {
                        s += e[a]*Ui[a*AI4_TC + j];
                    }
                    f[j] = s;
                }
                for (int k = c0 + AI4_TC; k < AI4_HB; ++k) {
                    double s = 0.0;
                    for (int j = 0; j < AI4_TC; ++j) {
                        s += f[j]*U[(c0 + j)*AI4_HB + k];
                    }
                    y[r*ys + k] -= (float) s;
                }
            }
        }
    }
}

// Quantize one slab: 16 rows of K floats (row stride K) into K/1024 slab super blocks at dst.
// imatrix: per-column (length K, original basis) or NULL. Uall/ok: optional precomputed LDLQ factors per 256
// block (ai4_ldlq_prep), shared by all slabs of a tensor; without them the factors are built on the fly.
static inline void ai4_quantize_slab(ai4_work * ws, const float * src, int64_t K, const float * imatrix,
                                    const float * Uall, const uint8_t * ok, uint8_t * dst) {
    float * y = (float *) malloc(sizeof(float)*AI4_TR*AI4_SB);
    float * tmpU = (imatrix && !Uall) ? (float *) malloc(sizeof(float)*AI4_HB*AI4_HB) : NULL;
    for (int64_t s = 0; s < K/AI4_SB; ++s) {
        for (int r = 0; r < AI4_TR; ++r) {
            memcpy(y + r*AI4_SB, src + r*K + s*AI4_SB, sizeof(float)*AI4_SB);
            for (int hb = 0; hb < AI4_SB/AI4_HB; ++hb) {
                ai4_rotate256(y + r*AI4_SB + hb*AI4_HB);
            }
        }
        uint8_t  * blk = dst + s*AI4_SLAB_SB_BYTES;
        uint16_t * dh  = (uint16_t *) blk;
        uint32_t * wd  = (uint32_t *)(blk + AI4_SCALE_BYTES);
        float dsc[AI4_TR][AI4_SB/AI4_GS];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int g = 0; g < AI4_SB/AI4_GS; ++g) {
                double ss = 0.0;
                for (int k = 0; k < AI4_GS; ++k) {
                    const double v = y[r*AI4_SB + g*AI4_GS + k];
                    ss += v*v;
                }
                float dv = ai4_fp16_round((float) sqrt(ss/AI4_GS)/ai4_level_rms()*AI4_SCALE_MUL);
                if (!isfinite(dv) || dv < 0.0f) {
                    dv = 0.0f;
                }
                dsc[r][g] = dv;
                dh[r*(AI4_SB/AI4_GS) + g] = ai4_fp32_to_fp16(dv);
            }
        }
        for (int hb = 0; hb < AI4_SB/AI4_HB; ++hb) {
            const int64_t bi = s*(AI4_SB/AI4_HB) + hb;
            const float * U = NULL;
            if (imatrix && Uall) {
                U = ok[bi] ? Uall + bi*AI4_HB*AI4_HB : NULL;
            } else if (imatrix && tmpU) {
                U = ai4_ldlq_prep(imatrix + bi*AI4_HB, tmpU) ? tmpU : NULL;
            }
            float dsc_hb[AI4_TR*2];
            for (int r = 0; r < AI4_TR; ++r) {
                dsc_hb[r*2 + 0] = dsc[r][hb*2 + 0];
                dsc_hb[r*2 + 1] = dsc[r][hb*2 + 1];
            }
            ai4_encode_slab_block(ws, y + hb*AI4_HB, AI4_SB, dsc_hb, U, wd + hb*(AI4_HB/AI4_TC)*16);
        }
    }
    free(y);
    free(tmpU);
}

// Rotated-domain dequantize of one slab (16 rows, row stride K).
static inline void ai4_dequantize_slab_rotated(const uint8_t * src, float * dst, int64_t K) {
    for (int64_t s = 0; s < K/AI4_SB; ++s) {
        const uint8_t  * blk = src + s*AI4_SLAB_SB_BYTES;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *)(blk + AI4_SCALE_BYTES);
        for (int ct = 0; ct < AI4_SB/AI4_TC; ++ct) {
            for (int n = 0; n < AI4_TILE; ++n) {
                const int r = n / AI4_TC, k = ct*AI4_TC + n % AI4_TC;
                const float d = ai4_fp16_to_fp32(dh[r*(AI4_SB/AI4_GS) + k/AI4_GS]);
                dst[r*K + s*AI4_SB + k] = d*(float) ai4_level(ai4_get_state(wd + ct*16, n));
            }
        }
    }
}

// Original-domain dequantize of one slab (unrotates).
static inline void ai4_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    ai4_dequantize_slab_rotated(src, dst, K);
    for (int r = 0; r < AI4_TR; ++r) {
        for (int64_t b = 0; b < K/AI4_HB; ++b) {
            ai4_unrotate256(dst + r*K + b*AI4_HB);
        }
    }
}


// =====================================================================================================================
// AI4_U (GGML_TYPE_AI4_U = 71): the same trellis, levels, scales and IMMA decode, in a unit-128 K layout so that any
// K % 128 == 0 tiles (MoE experts: K = 2560 = 20 units, K = 640 = 5 units). Part of the AI+ family (AI4/AI3/AI2 = rates).
//
//   - Unit = 16 rows x 128 K = [fp16 d[16] (one scale per row of this 128 group)] [16 tiles x AI4U_TILE_BYTES(k), tile index =
//     K-column step 0..15 of the unit]. A slab row of K is K/128 consecutive units; a tensor is M/16 slabs; an expert stack is
//     n_expert consecutive tensors. Tile bit layout and state/level definition are exactly AI4's (k = 4).
//   - Rate k (bits per trellis step) is a layout parameter: tile = 128 symbols x k bits = 16*k bytes, unit = 32 + 256*k bytes,
//     per row and 128 K = 2 + 16*k bytes. k = 4 (AI4_U, 1056 B/unit, 4.125 bpw) is implemented; ids 72 (k = 3), 73 (k = 2),
//     74 (k = 5) and 75 (k = 6, 6.125 bpw, a test candidate against Q6_K at 6.5625 bpw) are reserved for the other rates with the
//     same 16-bit state, level hash, rotation, scales and IMMA path. Q6_K / Q8_0 tensors can sit next to AI+ tensors in one file.
//   - Rotation: blocks of AI4U rotation size rb (ai4u_rot_size(K)): w' = (1/sqrt(rb)) H_rb D w with D = the first rb bits of
//     ai4_sign_bits. rb = 256 when K % 256 == 0 (values identical to AI4: the 1024 layout's encode, only reordered), else 128.
//     The activations are rotated with the same rb and int8-quantized with one fp32 scale per (token, 128 K).
//   - Encoder: block LDLQ inside each rotation block (rb columns), as in AI4.
// =====================================================================================================================

#define AI4U_UK              128
#define AI4U_TILE_BYTES(k)   (16*(k))
#define AI4U_UNIT_BYTES(k)   (2*AI4_TR + (AI4U_UK/AI4_TC)*AI4U_TILE_BYTES(k))
#define AI4U_ROW_BYTES(k)    (2 + AI4U_TILE_BYTES(k))      // per row and 128 K (ggml type_size, blck_size 128)
#define AI4U_UNIT_BYTES4     AI4U_UNIT_BYTES(4)            // 1056
#define AI4U_SCALE_BYTES     (2*AI4_TR)                    // 32

static inline int ai4u_rot_size(int64_t K) {
    return (K % 256 == 0) ? 256 : 128;
}

static inline void ai4_fwht_n(float * x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

// n = 256 is bit-identical to ai4_rotate256 / ai4_unrotate256
static inline float ai4_rot_norm(int n) {
    return n == 256 ? 1.0f/16.0f : 0.08838834764831845f;   // 1/sqrt(n)
}

static inline void ai4_rotate_n(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        if (ai4_sign_neg(i)) {
            x[i] = -x[i];
        }
    }
    ai4_fwht_n(x, n);
    const float s = ai4_rot_norm(n);
    for (int i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

static inline void ai4_unrotate_n(float * x, int n) {
    ai4_fwht_n(x, n);
    const float s = ai4_rot_norm(n);
    for (int i = 0; i < n; ++i) {
        x[i] *= ai4_sign_neg(i) ? -s : s;
    }
}

// LDLQ factor for one rotation block of n columns (n = 256 equals ai4_ldlq_prep).
static inline int ai4_ldlq_prep_n(const float * lambda, float * U, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
        mean += (double) lambda[i];
    }
    mean /= n;
    if (!(mean > 0.0)) {
        return 0;
    }
    double f[AI4_HB];
    for (int i = 0; i < n; ++i) {
        f[i] = 1.0/((double) lambda[i] + (double) AI4_LDLQ_DAMP*mean);
    }
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2*len) {
            for (int j = i; j < i + len; ++j) {
                const double a = f[j], b = f[j + len];
                f[j] = a + b;
                f[j + len] = a - b;
            }
        }
    }
    double * L = (double *) malloc(sizeof(double)*n*n);
    if (L == NULL) {
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = f[i ^ j]/(double) n;
            for (int k = 0; k < j; ++k) {
                v -= L[i*n + k]*L[j*n + k];
            }
            if (i == j) {
                if (!(v > 0.0)) {
                    free(L);
                    return 0;
                }
                L[i*n + i] = sqrt(v);
            } else {
                L[i*n + j] = v/L[j*n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i*n + j] = j >= i ? (float) L[j*n + i] : 0.0f;
        }
    }
    free(L);
    return 1;
}

// ai4_encode_slab_block for a rotation block of n columns (n = 128 or 256; n = 256 is the same arithmetic). d: scales per
// row and 128 group, d[r*(n/128) + g]. U: n x n LDLQ factor or NULL. tiles: n/8 tiles x 16 words, column-step order.
static inline void ai4_encode_slab_block_n(ai4_work * ws, float * y, int ys, const float * d, const float * U, int n, uint32_t * tiles) {
    const int ng = n/AI4_GS;
    for (int c0 = 0; c0 < n; c0 += AI4_TC) {
        float t[AI4_TILE];
        uint16_t st[AI4_TILE];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int j = 0; j < AI4_TC; ++j) {
                const float dr = d[r*ng + (c0 + j)/AI4_GS];
                t[r*AI4_TC + j] = dr > 0.0f ? y[r*ys + c0 + j]/dr : 0.0f;
            }
        }
        ai4_encode_tile(ws, t, st);
        uint32_t * wd = tiles + (c0/AI4_TC)*16;
        memset(wd, 0, 64);
        for (int i = 0; i < AI4_TILE; ++i) {
            wd[i >> 3] |= (uint32_t)(st[i] & 0xF) << (4*(7 - (i & 7)));
        }
        if (U && c0 + AI4_TC < n) {
            double Ub[AI4_TC*AI4_TC], Ui[AI4_TC*AI4_TC];
            for (int a = 0; a < AI4_TC; ++a) {
                for (int b = 0; b < AI4_TC; ++b) {
                    Ub[a*AI4_TC + b] = U[(c0 + a)*n + c0 + b];
                }
            }
            ai4_inv_upper8(Ub, Ui);
            for (int r = 0; r < AI4_TR; ++r) {
                double e[AI4_TC], f[AI4_TC];
                for (int j = 0; j < AI4_TC; ++j) {
                    e[j] = y[r*ys + c0 + j] - (double) d[r*ng + (c0 + j)/AI4_GS]*ws->lev[st[r*AI4_TC + j]];
                }
                for (int j = 0; j < AI4_TC; ++j) {
                    double s = 0.0;
                    for (int a = 0; a <= j; ++a) {
                        s += e[a]*Ui[a*AI4_TC + j];
                    }
                    f[j] = s;
                }
                for (int k = c0 + AI4_TC; k < n; ++k) {
                    double s = 0.0;
                    for (int j = 0; j < AI4_TC; ++j) {
                        s += f[j]*U[(c0 + j)*n + k];
                    }
                    y[r*ys + k] -= (float) s;
                }
            }
        }
    }
}

// Quantize one slab (16 rows of K floats, row stride K) to K/128 AI4_U units (k = 4) at dst. imatrix: per column (length K,
// original basis) or NULL. Uall/ok: optional LDLQ factors per rotation block (ai4_ldlq_prep_n with n = rb, rb*rb floats each).
// rb: rotation block (128 or 256, K % rb == 0); the format uses rb = ai4u_rot_size(K) (ai4u_quantize_slab).
static inline void ai4u_quantize_slab_rb(ai4_work * ws, const float * src, int64_t K, const float * imatrix,
                                        const float * Uall, const uint8_t * ok, uint8_t * dst, int rb) {
    float * y = (float *) malloc(sizeof(float)*AI4_TR*rb);
    float * tmpU = (imatrix && !Uall) ? (float *) malloc(sizeof(float)*rb*rb) : NULL;
    uint32_t tiles[(AI4_HB/AI4_TC)*16];
    for (int64_t b = 0; b < K/rb; ++b) {
        for (int r = 0; r < AI4_TR; ++r) {
            memcpy(y + r*rb, src + r*K + b*rb, sizeof(float)*rb);
            ai4_rotate_n(y + r*rb, rb);
        }
        const int ng = rb/AI4_GS;
        float dsc[AI4_TR*2];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int g = 0; g < ng; ++g) {
                double ss = 0.0;
                for (int k = 0; k < AI4_GS; ++k) {
                    const double v = y[r*rb + g*AI4_GS + k];
                    ss += v*v;
                }
                float dv = ai4_fp16_round((float) sqrt(ss/AI4_GS)/ai4_level_rms()*AI4_SCALE_MUL);
                if (!isfinite(dv) || dv < 0.0f) {
                    dv = 0.0f;
                }
                dsc[r*ng + g] = dv;
                uint16_t * dh = (uint16_t *) (dst + (b*ng + g)*AI4U_UNIT_BYTES4);
                dh[r] = ai4_fp32_to_fp16(dv);
            }
        }
        const float * U = NULL;
        if (imatrix && Uall) {
            U = ok[b] ? Uall + b*rb*rb : NULL;
        } else if (imatrix && tmpU) {
            U = ai4_ldlq_prep_n(imatrix + b*rb, tmpU, rb) ? tmpU : NULL;
        }
        ai4_encode_slab_block_n(ws, y, rb, dsc, U, rb, tiles);
        for (int g = 0; g < ng; ++g) {
            memcpy(dst + (b*ng + g)*AI4U_UNIT_BYTES4 + AI4U_SCALE_BYTES, tiles + g*16*16, 16*64);
        }
    }
    free(y);
    free(tmpU);
}

// Rotated-domain dequantize of one AI4_U slab (16 rows, row stride K).
static inline void ai4u_dequantize_slab_rotated(const uint8_t * src, float * dst, int64_t K) {
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = src + u*AI4U_UNIT_BYTES4;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
        for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
            for (int n = 0; n < AI4_TILE; ++n) {
                const int r = n / AI4_TC, k = ct*AI4_TC + n % AI4_TC;
                dst[r*K + u*AI4U_UK + k] = ai4_fp16_to_fp32(dh[r])*(float) ai4_level(ai4_get_state(wd + ct*16, n));
            }
        }
    }
}

static inline void ai4u_quantize_slab(ai4_work * ws, const float * src, int64_t K, const float * imatrix,
                                     const float * Uall, const uint8_t * ok, uint8_t * dst) {
    ai4u_quantize_slab_rb(ws, src, K, imatrix, Uall, ok, dst, ai4u_rot_size(K));
}

static inline void ai4u_dequantize_slab_rb(const uint8_t * src, float * dst, int64_t K, int rb) {
    ai4u_dequantize_slab_rotated(src, dst, K);
    for (int r = 0; r < AI4_TR; ++r) {
        for (int64_t b = 0; b < K/rb; ++b) {
            ai4_unrotate_n(dst + r*K + b*rb, rb);
        }
    }
}

static inline void ai4u_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    ai4u_dequantize_slab_rb(src, dst, K, ai4u_rot_size(K));
}

// Both layouts store a 16-row slab of K in K/128*1056 bytes (8448 per 1024 = 1056 per 128).
static inline int64_t ai4x_slab_bytes(int64_t K) {
    return (K/AI4U_UK)*AI4U_UNIT_BYTES4;
}

static inline void ai4x_dequantize_slab(int unit, const uint8_t * src, float * dst, int64_t K) {
    if (unit) {
        ai4u_dequantize_slab(src, dst, K);
    } else {
        ai4_dequantize_slab(src, dst, K);
    }
}


// =====================================================================================================================
// AI+ unit layout at rate k (AI5_U = 74: k = 5). Same 16-bit state, level hash, rotation (ai4u_rot_size), fp16 scale per
// (row, 128 K) and unit-128 layout as AI4_U; only the symbol width changes.
//   - A tile (16 rows x 8 K = 128 weights, row-major n = r*8 + c) is a 128*k-bit stream, MSB first: symbol n occupies stream
//     bits [k*n, k*n + k), word w holds stream bits 32w .. 32w + 31 (bit 32w = the word's MSB). Tile = 4k words = 16k bytes.
//   - state_n = the 16 stream bits ending at the end of symbol n (bits [k*n + k - 16, k*n + k), indices mod 128k: tail-biting),
//     so state_n = ((state_{n-1} << k) | sym_n) & 0xFFFF. For k = 4 this is exactly AI4's tile (aik_get_state == ai4_get_state).
//   - Unit = [fp16 d[16]] [16 tiles x 16k bytes] = 32 + 256k bytes (k = 5: 1312 B, 82 B per row and 128 K, 5.125 bpw).
//   - Encoder: the AI4 Viterbi with 2^k predecessors per state (predecessor low 16-k bits = s >> k), tail-biting with the
//     first state's top 16-k bits equal to the last state's low 16-k bits; block LDLQ per rotation block as in AI4_U.
// =====================================================================================================================

#define AIK_TILE_WORDS(k)   (4*(k))

static inline uint32_t aik_get_state(const uint32_t * wd, int k, int n) {
    const int nb = AI4_TILE*k, nw = AIK_TILE_WORDS(k);
    int b0 = k*n + k - 16;
    if (b0 < 0) {
        b0 += nb;
    }
    const int w = b0 >> 5, sh = b0 & 31;
    const uint64_t c = ((uint64_t) wd[w] << 32) | wd[(w + 1) % nw];
    return (uint32_t)((c >> (48 - sh)) & 0xFFFFu);
}

static inline void aik_put_sym(uint32_t * wd, int k, int n, uint32_t sym) {
    for (int i = 0; i < k; ++i) {
        const int b = k*n + i;
        if ((sym >> (k - 1 - i)) & 1u) {
            wd[b >> 5] |= 1u << (31 - (b & 31));
        }
    }
}

// Decision window of the GPU encoder at rate k (tools/ai4/ai4-encode.cu): 32 steps where the shared-memory budget allows
// it, fewer at the low rates (the min-over-hi buffer grows as 2^(16-k)). The tail-biting roll must put original index 127
// inside the last window, so it is WIN - 1; CPU and GPU encoders must agree on it. k = 4 keeps AI4_ROLL = 31.
static inline int aik_win(int k) {
    return k >= 4 ? 32 : (k == 3 ? 8 : 4);
}

static inline int aik_roll(int k) {
    return aik_win(k) - 1;
}

// Viterbi at rate k (k = 4 equals ai4_viterbi). t0 < 0: free start; t0 >= 0: the first state's top 16-k bits and the last
// state's low 16-k bits both equal t0. ws->arg holds 128 x 2^(16-k) decisions; levels come from ws->lev (aik_work_set_mul).
static inline float aik_viterbi_metric(ai4_work * ws, int k, const float * y, const float * weight,
                                     int t0, uint16_t * st) {
    float * cur = ws->cur, * nxt = ws->nxt;
    const float * L = ws->lev;
    const int tb = 16 - k, ntl = 1 << tb, nhi = 1 << k;
    for (int s = 0; s < AI4_NSTATE; ++s) {
        const float e = L[s] - y[0];
        cur[s] = (t0 >= 0 && (s >> k) != t0) ? FLT_MAX : e*e*(weight ? weight[0] : 1.0f);
    }
    float * m = ws->mscr;
    for (int i = 1; i < AI4_TILE; ++i) {
        uint8_t * a = ws->arg + (size_t) i*ntl;
        for (int t = 0; t < ntl; ++t) {
            m[t] = cur[t];
            a[t] = 0;
        }
        for (int hi = 1; hi < nhi; ++hi) {
            const float * c = cur + (hi << tb);
            for (int t = 0; t < ntl; ++t) {
                const int lt = c[t] < m[t];
                m[t] = lt ? c[t] : m[t];
                a[t] = lt ? (uint8_t) hi : a[t];
            }
        }
        const float yi = y[i];
        for (int s = 0; s < AI4_NSTATE; ++s) {
            const float e = L[s] - yi;
            nxt[s] = m[s >> k] + e*e*(weight ? weight[i] : 1.0f);
        }
        float * tmp = cur; cur = nxt; nxt = tmp;
    }
    int best = 0;
    float bc = FLT_MAX;
    for (int s = 0; s < AI4_NSTATE; ++s) {
        if (t0 >= 0 && (s & (ntl - 1)) != t0) {
            continue;
        }
        if (cur[s] < bc) {
            bc = cur[s];
            best = s;
        }
    }
    int s = best;
    for (int i = AI4_TILE - 1; i >= 0; --i) {
        st[i] = (uint16_t) s;
        if (i > 0) {
            const int t = s >> k;
            s = t | (ws->arg[(size_t) i*ntl + t] << tb);
        }
    }
    return bc;
}

static inline float aik_viterbi(ai4_work * ws, int k, const float * y, int t0, uint16_t * st) {
    return aik_viterbi_metric(ws, k, y, NULL, t0, st);
}

static inline void aik_encode_tile_metric(ai4_work * ws, int k, const float * y, const float * weight, uint16_t * st) {
    float yr[AI4_TILE], wr[AI4_TILE];
    uint16_t sr[AI4_TILE];
    const int roll = aik_roll(k);
    for (int i = 0; i < AI4_TILE; ++i) {
        yr[i] = y[(i + roll) % AI4_TILE];
        if (weight) wr[i] = weight[(i + roll) % AI4_TILE];
    }
    aik_viterbi_metric(ws, k, yr, weight ? wr : NULL, -1, sr);
    const int t0 = sr[AI4_TILE - 1 - roll] & ((1 << (16 - k)) - 1);
    aik_viterbi_metric(ws, k, y, weight, t0, st);
}

static inline void aik_encode_tile(ai4_work * ws, int k, const float * y, uint16_t * st) {
    aik_encode_tile_metric(ws, k, y, NULL, st);
}

// ai4_encode_slab_block_n at rate k. tiles: n/8 tiles x 4k words, column-step order.
static inline void aik_encode_slab_block_n_metric(ai4_work * ws, int k, float * y, int ys, const float * d,
                                                 const float * U, int n, uint32_t * tiles, int weight_metric) {
    const int ng = n/AI4_GS, tw = AIK_TILE_WORDS(k);
    for (int c0 = 0; c0 < n; c0 += AI4_TC) {
        float t[AI4_TILE], weight[AI4_TILE], maxd = 0.0f;
        uint16_t st[AI4_TILE];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int j = 0; j < AI4_TC; ++j) {
                const float dr = d[r*ng + (c0 + j)/AI4_GS];
                t[r*AI4_TC + j] = dr > 0.0f ? y[r*ys + c0 + j]/dr : 0.0f;
                weight[r*AI4_TC + j] = dr;
                if (dr > maxd) maxd = dr;
            }
        }
        if (weight_metric) {
            for (int i = 0; i < AI4_TILE; ++i) {
                const float v = maxd > 0.0f ? weight[i]/maxd : 0.0f;
                weight[i] = v*v;
            }
        }
        aik_encode_tile_metric(ws, k, t, weight_metric ? weight : NULL, st);
        uint32_t * wd = tiles + (c0/AI4_TC)*tw;
        memset(wd, 0, sizeof(uint32_t)*tw);
        for (int i = 0; i < AI4_TILE; ++i) {
            aik_put_sym(wd, k, i, st[i] & ((1u << k) - 1));
        }
        if (U && c0 + AI4_TC < n) {
            double Ub[AI4_TC*AI4_TC], Ui[AI4_TC*AI4_TC];
            for (int a = 0; a < AI4_TC; ++a) {
                for (int b = 0; b < AI4_TC; ++b) {
                    Ub[a*AI4_TC + b] = U[(c0 + a)*n + c0 + b];
                }
            }
            ai4_inv_upper8(Ub, Ui);
            for (int r = 0; r < AI4_TR; ++r) {
                double e[AI4_TC], f[AI4_TC];
                for (int j = 0; j < AI4_TC; ++j) {
                    e[j] = y[r*ys + c0 + j] - (double) d[r*ng + (c0 + j)/AI4_GS]*ws->lev[st[r*AI4_TC + j]];
                }
                for (int j = 0; j < AI4_TC; ++j) {
                    double s = 0.0;
                    for (int a = 0; a <= j; ++a) {
                        s += e[a]*Ui[a*AI4_TC + j];
                    }
                    f[j] = s;
                }
                for (int kk = c0 + AI4_TC; kk < n; ++kk) {
                    double s = 0.0;
                    for (int j = 0; j < AI4_TC; ++j) {
                        s += f[j]*U[(c0 + j)*n + kk];
                    }
                    y[r*ys + kk] -= (float) s;
                }
            }
        }
    }
}

static inline void aik_encode_slab_block_n(ai4_work * ws, int k, float * y, int ys, const float * d, const float * U, int n,
                                          uint32_t * tiles) {
    aik_encode_slab_block_n_metric(ws, k, y, ys, d, U, n, tiles, 0);
}

// Quantize one slab (16 rows of K, row stride K) to K/128 rate-k units at dst (ai4u_quantize_slab_rb at rate k).
// dfix (optional): per-(row, 128-group) RMS table [16][K/128] row-major used in place of each group's own rotated RMS,
// i.e. the CPU twin of the GPU encoder's sfix. A table that is constant inside every 1024 span (--scale-mode row / span)
// makes the output packable into the AIk_S span layout (ai4s_pack_slab). NULL = each group's own RMS.
static inline void aik_quantize_slab_rb_cfm(ai4_work * ws, int k, uint32_t C, const float * src, int64_t K, const float * imatrix,
                                       const float * Uall, const uint8_t * ok, uint8_t * dst, int rb, float smul,
                                       const float * dfix, int weight_metric) {
    const int ub = AI4U_UNIT_BYTES(k), tw = AIK_TILE_WORDS(k);
    aik_work_set_mul(ws, C);
    const float lrms = aik_level_rms_c(C);
    float * y = (float *) malloc(sizeof(float)*AI4_TR*rb);
    float * tmpU = (imatrix && !Uall) ? (float *) malloc(sizeof(float)*rb*rb) : NULL;
    uint32_t * tiles = (uint32_t *) malloc(sizeof(uint32_t)*(AI4_HB/AI4_TC)*AIK_TILE_WORDS(8));
    for (int64_t b = 0; b < K/rb; ++b) {
        for (int r = 0; r < AI4_TR; ++r) {
            memcpy(y + r*rb, src + r*K + b*rb, sizeof(float)*rb);
            ai4_rotate_n(y + r*rb, rb);
        }
        const int ng = rb/AI4_GS;
        float dsc[AI4_TR*2];
        for (int r = 0; r < AI4_TR; ++r) {
            for (int g = 0; g < ng; ++g) {
                double ss = 0.0;
                for (int kk = 0; kk < AI4_GS; ++kk) {
                    const double v = y[r*rb + g*AI4_GS + kk];
                    ss += v*v;
                }
                const float srms = dfix ? dfix[r*(K/AI4_GS) + b*ng + g] : (float) sqrt(ss/AI4_GS);
                float dv = ai4_fp16_round(srms/lrms*smul);
                if (!isfinite(dv) || dv < 0.0f) {
                    dv = 0.0f;
                }
                dsc[r*ng + g] = dv;
                uint16_t * dh = (uint16_t *) (dst + (b*ng + g)*ub);
                dh[r] = ai4_fp32_to_fp16(dv);
            }
        }
        const float * U = NULL;
        if (imatrix && Uall) {
            U = ok[b] ? Uall + b*rb*rb : NULL;
        } else if (imatrix && tmpU) {
            U = ai4_ldlq_prep_n(imatrix + b*rb, tmpU, rb) ? tmpU : NULL;
        }
        aik_encode_slab_block_n_metric(ws, k, y, rb, dsc, U, rb, tiles, weight_metric);
        for (int g = 0; g < ng; ++g) {
            memcpy(dst + (b*ng + g)*ub + AI4U_SCALE_BYTES, tiles + g*16*tw, (size_t) 16*tw*4);
        }
    }
    free(tiles);
    free(y);
    free(tmpU);
}

static inline void aik_quantize_slab_rb_cf(ai4_work * ws, int k, uint32_t C, const float * src, int64_t K, const float * imatrix,
                                         const float * Uall, const uint8_t * ok, uint8_t * dst, int rb, float smul,
                                         const float * dfix) {
    aik_quantize_slab_rb_cfm(ws, k, C, src, K, imatrix, Uall, ok, dst, rb, smul, dfix, 0);
}

static inline void aik_quantize_slab_rb_c(ai4_work * ws, int k, uint32_t C, const float * src, int64_t K, const float * imatrix,
                                       const float * Uall, const uint8_t * ok, uint8_t * dst, int rb, float smul) {
    aik_quantize_slab_rb_cf(ws, k, C, src, K, imatrix, Uall, ok, dst, rb, smul, NULL);
}

static inline void aik_dequantize_slab_rotated_c(int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    const int ub = AI4U_UNIT_BYTES(k), tw = AIK_TILE_WORDS(k);
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        const uint8_t  * blk = src + u*ub;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
        for (int ct = 0; ct < AI4U_UK/AI4_TC; ++ct) {
            for (int n = 0; n < AI4_TILE; ++n) {
                const int r = n / AI4_TC, kk = ct*AI4_TC + n % AI4_TC;
                dst[r*K + u*AI4U_UK + kk] = ai4_fp16_to_fp32(dh[r])*(float) aik_level_c(aik_get_state(wd + ct*tw, k, n), C);
            }
        }
    }
}

static inline void aik_dequantize_slab_c(int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    const int rb = ai4u_rot_size(K);
    aik_dequantize_slab_rotated_c(k, C, src, dst, K);
    for (int r = 0; r < AI4_TR; ++r) {
        for (int64_t b = 0; b < K/rb; ++b) {
            ai4_unrotate_n(dst + r*K + b*rb, rb);
        }
    }
}

// Any AI+ layout: unit = 0 is AI4 (1024 layout, k = 4), unit = 1 the unit-128 layout at rate k.
static inline int64_t ai4x_slab_bytes_k(int64_t K, int k) {
    return (K/AI4U_UK)*AI4U_UNIT_BYTES(k);
}

static inline void ai4x_dequantize_slab_kc(int unit, int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    if (unit && (k != 4 || C != AI4_MUL)) {
        aik_dequantize_slab_c(k, C, src, dst, K);
    } else {
        // the 1024 slab layout (unit == 0) exists only at k = 4 / the AI4 constant
        ai4x_dequantize_slab(unit, src, dst, K);
    }
}

static inline void ai4x_dequantize_slab_k(int unit, int k, const uint8_t * src, float * dst, int64_t K) {
    ai4x_dequantize_slab_kc(unit, k, AI4_MUL, src, dst, K);
}

// =====================================================================================================================
// AI+ span layout (AIk_S, GGML_TYPE_AI2_S = 96 .. GGML_TYPE_AI6_S = 100): AIk_U with the per-unit scale headers removed.
// Same trellis, levels, rotation, per-type constant and tile bit layout; only the scale granularity changes.
//
//   - Span = 16 rows x 1024 K = [fp16 d[16] (one scale per row for the whole span)] [8 units' worth of tiles back to back,
//     128 tiles x 16k bytes, tile index = K-column step 0..127 of the span] = 32 + 2048k bytes. A slab row of K is K/1024
//     consecutive spans; a tensor is M/16 slabs. K % 1024 == 0 is required (reject other K at quantize time).
//   - Per row and 1024 K = 2 + 128k bytes (the ggml type_size at blck_size 1024): k + 0.0156 bpw instead of k + 0.125.
//   - The bytes are exactly AIk_U's with the 7 redundant 32-byte headers of each span dropped, so an AIk_U tensor whose
//     scale is constant inside every 1024 span (llama-ai4-quantize --scale-mode row / span) repacks losslessly, and an
//     AIk_S tensor unpacks back to the identical AIk_U bytes. Dequantized floats are bit-identical across the repack.
//   - Rotation is unchanged (ai4u_rot_size(K) = 256 here, since K % 1024 == 0), and a 1024 span is a whole number of
//     rotation blocks, so a per-span scale is well defined in the rotated basis from the unrotated row (Hadamard is
//     orthogonal inside each block).
// =====================================================================================================================

#define AI4S_SPAN_K          1024
#define AI4S_TILES_PER_SPAN  (AI4S_SPAN_K/AI4_TC)                                  // 128
#define AI4S_TILE_BYTES(k)   (AI4S_TILES_PER_SPAN*AI4U_TILE_BYTES(k))              // 2048k
#define AI4S_SPAN_BYTES(k)   (AI4U_SCALE_BYTES + AI4S_TILE_BYTES(k))               // 32 + 2048k
#define AI4S_ROW_BYTES(k)    (2 + (AI4S_SPAN_K/AI4_TC)*(k))                        // ggml type_size at blck_size 1024
#define AI4S_UNITS_PER_SPAN  (AI4S_SPAN_K/AI4U_UK)                                 // 8

static inline int64_t ai4s_slab_bytes_k(int64_t K, int k) {
    return (K/AI4S_SPAN_K)*AI4S_SPAN_BYTES(k);
}

// Pack one 16-row slab of K from the unit-128 layout into the span layout. Returns 1 on success, 0 if any unit's 32-byte
// scale header differs from the first header of its span (that tensor is not span-scaled and must not be repacked).
static inline int ai4s_pack_slab(int k, const uint8_t * src, uint8_t * dst, int64_t K) {
    const int ub = AI4U_UNIT_BYTES(k), tb = AI4U_UNIT_BYTES(k) - AI4U_SCALE_BYTES;   // tb = 256k
    for (int64_t s = 0; s < K/AI4S_SPAN_K; ++s) {
        const uint8_t * su = src + s*AI4S_UNITS_PER_SPAN*(int64_t) ub;
        uint8_t * dp = dst + s*(int64_t) AI4S_SPAN_BYTES(k);
        memcpy(dp, su, AI4U_SCALE_BYTES);
        for (int j = 0; j < AI4S_UNITS_PER_SPAN; ++j) {
            if (j > 0 && memcmp(su + (int64_t) j*ub, su, AI4U_SCALE_BYTES) != 0) {
                return 0;
            }
            memcpy(dp + AI4U_SCALE_BYTES + (int64_t) j*tb, su + (int64_t) j*ub + AI4U_SCALE_BYTES, tb);
        }
    }
    return 1;
}

// Unpack one 16-row slab of K from the span layout back into the unit-128 layout (the span's scale header is copied into
// each of its 8 units), giving back the exact AIk_U bytes a span-scaled encode would have written.
static inline void ai4s_unpack_slab(int k, const uint8_t * src, uint8_t * dst, int64_t K) {
    const int ub = AI4U_UNIT_BYTES(k), tb = AI4U_UNIT_BYTES(k) - AI4U_SCALE_BYTES;
    for (int64_t s = 0; s < K/AI4S_SPAN_K; ++s) {
        const uint8_t * sp = src + s*(int64_t) AI4S_SPAN_BYTES(k);
        uint8_t * du = dst + s*AI4S_UNITS_PER_SPAN*(int64_t) ub;
        for (int j = 0; j < AI4S_UNITS_PER_SPAN; ++j) {
            memcpy(du + (int64_t) j*ub, sp, AI4U_SCALE_BYTES);
            memcpy(du + (int64_t) j*ub + AI4U_SCALE_BYTES, sp + AI4U_SCALE_BYTES + (int64_t) j*tb, tb);
        }
    }
}

// Rotated-domain dequantize of one AIk_S slab (16 rows, row stride K).
static inline void ai4s_dequantize_slab_rotated_c(int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    const int tw = AIK_TILE_WORDS(k);
    for (int64_t s = 0; s < K/AI4S_SPAN_K; ++s) {
        const uint8_t  * blk = src + s*(int64_t) AI4S_SPAN_BYTES(k);
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint32_t * wd  = (const uint32_t *) (blk + AI4U_SCALE_BYTES);
        for (int ct = 0; ct < AI4S_TILES_PER_SPAN; ++ct) {
            for (int n = 0; n < AI4_TILE; ++n) {
                const int r = n / AI4_TC, kk = ct*AI4_TC + n % AI4_TC;
                dst[r*K + s*AI4S_SPAN_K + kk] = ai4_fp16_to_fp32(dh[r])*(float) aik_level_c(aik_get_state(wd + ct*tw, k, n), C);
            }
        }
    }
}

static inline void ai4s_dequantize_slab_c(int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    const int rb = ai4u_rot_size(K);
    ai4s_dequantize_slab_rotated_c(k, C, src, dst, K);
    for (int r = 0; r < AI4_TR; ++r) {
        for (int64_t b = 0; b < K/rb; ++b) {
            ai4_unrotate_n(dst + r*K + b*rb, rb);
        }
    }
}

// Layout-aware entry points: lay = 0 the AI4 1024 slab, 1 the AIk_U unit-128 layout, 2 the AIk_S span layout
// (ggml_ai4_layout).
static inline int64_t ai4x_slab_bytes_lk(int lay, int64_t K, int k) {
    return lay == 2 ? ai4s_slab_bytes_k(K, k) : ai4x_slab_bytes_k(K, k);
}

static inline void ai4x_dequantize_slab_lkc(int lay, int k, uint32_t C, const uint8_t * src, float * dst, int64_t K) {
    if (lay == 2) {
        ai4s_dequantize_slab_c(k, C, src, dst, K);
    } else {
        ai4x_dequantize_slab_kc(lay == 1, k, C, src, dst, K);
    }
}

// Back-compat names at the AI4 constant.
static inline void aik_quantize_slab_rb(ai4_work * ws, int k, const float * src, int64_t K, const float * imatrix,
                                       const float * Uall, const uint8_t * ok, uint8_t * dst, int rb, float smul) {
    aik_quantize_slab_rb_c(ws, k, AI4_MUL, src, K, imatrix, Uall, ok, dst, rb, smul);
}

static inline void aik_dequantize_slab_rotated(int k, const uint8_t * src, float * dst, int64_t K) {
    aik_dequantize_slab_rotated_c(k, AI4_MUL, src, dst, K);
}

static inline void aik_dequantize_slab(int k, const uint8_t * src, float * dst, int64_t K) {
    aik_dequantize_slab_c(k, AI4_MUL, src, dst, K);
}

#ifdef __cplusplus
}
#endif
