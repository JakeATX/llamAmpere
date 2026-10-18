// RQ4: rotated 4-bit weights for the SM86 int8 tensor-core path (W4A8).
//
// Format (one ggml row = one output channel, K = ne0 must be a multiple of RQ4_SB = 1024):
//   - Every 256-wide K block of the row is rotated offline: w' = H256 * D * w, where D is a fixed
//     +-1 sign diagonal (rq4_sign) and H256 the Sylvester Walsh-Hadamard matrix scaled by 1/16
//     (orthonormal). Activations get the same rotation online, so W x = W' x'.
//   - Codes: signed 4-bit n in [-8, 7], w' ~= d_g * n, one fp16 d_g per group of RQ4_GS weights.
//   - Super block of 1024 weights = [fp16 d[1024/GS]] [512 bytes of nibbles]:
//       GS = 128: 16 + 512 = 528 bytes (4.125 bpw)   -> GGML_TYPE_RQ4_G128
//       GS =  64: 32 + 512 = 544 bytes (4.25 bpw)    -> GGML_TYPE_RQ4_G64
//     Both are multiples of 16 bytes, so the nibble area of every row is 16-byte aligned.
//   - Nibble order inside each 128-weight chunk (64 bytes) is the m16n8k32 s8 A-fragment order:
//       byte t*16 + j*4 + b  (t = 0..3, j = 0..3, b = 0..3)
//         high nibble = w[32*j      + 4*t + b]
//         low  nibble = w[32*j + 16 + 4*t + b]
//     Lane (groupID g, thread t) of a warp loads 16 contiguous bytes per row per chunk; masking the
//     high nibbles (w & 0xF0F0F0F0) gives 4 s8 values 16*n for A register 0/1 of IMMA j, and
//     (w << 4) & 0xF0F0F0F0 gives A register 2/3. No int->int8 conversion is needed: the factor 16
//     is folded into the output scale.
//
// RQ4NL (GGML_TYPE_RQ4NL_G128 / RQ4NL_G64): identical bytes and layout, but each nibble is an index into
// the 16-entry non-uniform int8 table rq4_nl_values (the IQ4_NL table), w' ~= d_g * rq4_nl_values[n]. The
// kernels turn a nibble word into s8 A registers with a byte-permute table lookup and use output factor 1.
// Scales may be negative (the encoder picks the sign that puts the extreme weight on the longer table end).
//
// Activations (runtime only, produced by the CUDA rotate+quantize kernel):
//   - int8 per value after the same rotation, one fp32 scale per (token, 256-wide K block).
//   - Byte order inside each 128 chunk: byte t*32 + j*8 + h*4 + b = x[32*j + 16*h + 4*t + b], so a lane
//     loads its B fragments for 4 IMMAs as 32 contiguous bytes.

#pragma once

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define RQ4_SB    1024   // super block (weights)
#define RQ4_HB    256    // Hadamard block
#define RQ4_CHUNK 128    // fragment chunk

#ifdef __cplusplus
extern "C" {
#endif

// Fixed sign diagonal for the 256-point rotation (bit i of word i/32 set -> negate element i).
// Arbitrary but frozen: changing it invalidates every RQ4 file.
#define RQ4_SIGN_WORDS \
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu, \
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au
static const uint32_t rq4_sign_bits[8] = { RQ4_SIGN_WORDS };

static inline int rq4_sign_neg(int i) {
    return (rq4_sign_bits[i >> 5] >> (i & 31)) & 1;
}

// In-place forward rotation of one 256 block: x <- (1/16) H256 D x. (1/16) H256 is symmetric and
// orthonormal, so the inverse is x <- D (1/16) H256 y (rq4_unrotate256).
static inline void rq4_rotate256(float * x) {
    for (int i = 0; i < RQ4_HB; ++i) {
        if (rq4_sign_neg(i)) {
            x[i] = -x[i];
        }
    }
    for (int h = 1; h < RQ4_HB; h <<= 1) {
        for (int i = 0; i < RQ4_HB; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < RQ4_HB; ++i) {
        x[i] *= 1.0f/16.0f;
    }
}

// Inverse rotation: x <- D (1/16) H256 x.
static inline void rq4_unrotate256(float * x) {
    for (int h = 1; h < RQ4_HB; h <<= 1) {
        for (int i = 0; i < RQ4_HB; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < RQ4_HB; ++i) {
        x[i] *= rq4_sign_neg(i) ? -1.0f/16.0f : 1.0f/16.0f;
    }
}

static inline int rq4_sb_bytes(int gs) {
    return 2*(RQ4_SB/gs) + RQ4_SB/2;
}

// Byte position and nibble (1 = high) of weight k (0..127) inside a 64-byte chunk.
static inline void rq4_nibble_pos(int k, int * byte, int * hi) {
    const int j = k >> 5;
    const int r = k & 31;
    const int h = r >> 4;       // 0 -> high nibble, 1 -> low nibble
    const int t = (r & 15) >> 2;
    const int b = r & 3;
    *byte = t*16 + j*4 + b;
    *hi   = h == 0;
}

// fp16 helpers (IEEE binary16, round to nearest even), kept local so this header has no ggml deps.
static inline uint16_t rq4_fp32_to_fp16(float f) {
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

static inline float rq4_fp16_to_fp32(uint16_t h) {
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

// Byte position of activation k (0..127) inside a 128-byte activation chunk.
static inline int rq4_act_byte(int k) {
    const int j = k >> 5;
    const int r = k & 31;
    const int h = r >> 4;
    const int t = (r & 15) >> 2;
    const int b = r & 3;
    return t*32 + j*8 + h*4 + b;
}

// CPU reference of the CUDA activation kernel for one token row x[K] (K % 256 == 0):
// q[K] int8 in fragment byte order, s[K/256] fp32 scales.
static inline void rq4_quantize_act_row(const float * x, int8_t * q, float * s, int64_t K) {
    float buf[RQ4_HB];
    for (int64_t hb = 0; hb < K/RQ4_HB; ++hb) {
        memcpy(buf, x + hb*RQ4_HB, sizeof(buf));
        rq4_rotate256(buf);
        float amax = 0.0f;
        for (int i = 0; i < RQ4_HB; ++i) {
            amax = fabsf(buf[i]) > amax ? fabsf(buf[i]) : amax;
        }
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        s[hb] = amax/127.0f;
        for (int i = 0; i < RQ4_HB; ++i) {
            const int k = (int) (hb*RQ4_HB) + i;
            q[(k/RQ4_CHUNK)*RQ4_CHUNK + rq4_act_byte(k & (RQ4_CHUNK - 1))] = (int8_t) lrintf(buf[i]*id);
        }
    }
}

// Candidate scales for one group of rotated weights: the extreme value mapped to -8 (sign-matched) or
// near +7, over a small grid (2*RQ4_NCAND values, fp16-rounded). Returns the count (0 for an all-zero group).
#define RQ4_NCAND 13
static inline int rq4_scale_candidates(const float * x, int n, float * cand) {
    float amax = 0.0f, vmax = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (fabsf(x[i]) > amax) {
            amax = fabsf(x[i]);
            vmax = x[i];
        }
    }
    if (amax == 0.0f) {
        return 0;
    }
    int nc = 0;
    for (int is = -(RQ4_NCAND/2); is <= RQ4_NCAND/2; ++is) {
        for (int side = 0; side < 2; ++side) {
            const float target = side == 0 ? -8.0f : 7.0f;
            const float d = rq4_fp16_to_fp32(rq4_fp32_to_fp16(vmax / (target + 0.1f*is*(side == 0 ? -1.0f : 1.0f))));
            if ((d > 0.0f || d < 0.0f) && isfinite(d)) {   // skip fp16 overflow (|vmax| > ~4.6e5)
                cand[nc++] = d;
            }
        }
    }
    return nc;
}

static inline int rq4_code(float x, float id) {
    int q = (int) lrintf(x*id);
    return q < -8 ? -8 : (q > 7 ? 7 : q);
}

// RQ4NL value table (= ggml kvalues_iq4nl). Frozen: changing it invalidates every RQ4NL file.
#define RQ4_NL_VALUES -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const int8_t rq4_nl_values[16] = { RQ4_NL_VALUES };

// Internal code of the encoder: the signed value in [-8, 7] for the uniform grid (nl = 0), the table index
// 0..15 for RQ4NL (nl = 1). The stored nibble is (code & 0xf) in both cases.
static inline float rq4_val(int nl, int q) {
    return nl ? (float) rq4_nl_values[q] : (float) q;
}

// Nearest code of x for inverse scale id.
static inline int rq4_qcode(int nl, float x, float id) {
    if (!nl) {
        return rq4_code(x, id);
    }
    // nearest table entry = number of midpoints below v (branchless; ties go to the lower entry)
    static const float mid[15] = {
        -115.5f, -93.5f, -74.0f, -57.0f, -42.0f, -28.5f, -16.0f, -4.5f, 7.0f, 19.0f, 31.5f, 45.5f, 61.0f, 79.0f, 101.0f,
    };
    const float v = x*id;
    int n = 0;
    for (int m = 0; m < 15; ++m) {
        n += v > mid[m];
    }
    return n;
}

// Code used for a group whose scale is 0 (value irrelevant).
static inline int rq4_zero_code(int nl) {
    return nl ? 8 : 0;
}

// RQ4NL candidate scales: the extreme value mapped onto either end of the table (so the scale sign is free),
// times (1 + RQ4_NL_STEP*i), i in [RQ4_NL_ILO, RQ4_NL_IHI], fp16-rounded. 28 candidates: on 6 Qwen3.8-27B
// tensors the weighted error is 0.7% above an 82-candidate 1% grid at 2.6x less encode time.
#ifndef RQ4_NL_STEP
#define RQ4_NL_STEP 0.03f
#endif
#ifndef RQ4_NL_ILO
#define RQ4_NL_ILO  (-10)
#endif
#ifndef RQ4_NL_IHI
#define RQ4_NL_IHI  3
#endif
#define RQ4_MAXCAND (2*(RQ4_NL_IHI - RQ4_NL_ILO + 1) > 2*RQ4_NCAND ? 2*(RQ4_NL_IHI - RQ4_NL_ILO + 1) : 2*RQ4_NCAND)

static inline int rq4_scale_candidates_g(int nl, const float * x, int n, float * cand) {
    if (!nl) {
        return rq4_scale_candidates(x, n, cand);
    }
    float amax = 0.0f, vmax = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (fabsf(x[i]) > amax) {
            amax = fabsf(x[i]);
            vmax = x[i];
        }
    }
    if (amax == 0.0f) {
        return 0;
    }
    int nc = 0;
    for (int is = RQ4_NL_ILO; is <= RQ4_NL_IHI; ++is) {
        for (int side = 0; side < 2; ++side) {
            const float t = side == 0 ? (float) rq4_nl_values[0] : (float) rq4_nl_values[15];
            const float d = rq4_fp16_to_fp32(rq4_fp32_to_fp16(vmax/t*(1.0f + RQ4_NL_STEP*(float) is)));
            if ((d > 0.0f || d < 0.0f) && isfinite(d)) {
                cand[nc++] = d;
            }
        }
    }
    return nc;
}

// Unweighted scale search for one group of rotated weights (squared error in the rotated basis, which
// equals the original basis because the rotation is orthonormal). Returns d (0 for an all-zero group).
static inline float rq4_choose_scale(int nl, const float * x, int n) {
    float cand[RQ4_MAXCAND];
    const int nc = rq4_scale_candidates_g(nl, x, n, cand);
    float best_d = 0.0f, best_err = INFINITY;
    for (int c = 0; c < nc; ++c) {
        const float d = cand[c], id = 1.0f/d;
        float err = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float e = x[i] - d*rq4_val(nl, rq4_qcode(nl, x[i], id));
            err += e*e;
        }
        if (err < best_err) {
            best_err = err;
            best_d = d;
        }
    }
    return best_d;
}

static inline int rq4_parity(int v) {   // v < 256
    v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return v & 1;
}

// Unnormalized in-place Sylvester Walsh-Hadamard transform of length n (power of two).
static inline void rq4_fwht(float * x, int n) {
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

// Scales of the 256/gs groups of one rotated 256 block y. Without an importance vector each group takes its
// unweighted best scale. With one (lambda, length 256, in the ORIGINAL basis) the objective is the weighted
// error of the dequantized weights in the original basis, sum_i lambda_i ((D H/16) e)_i^2 with e the rotated
// error; D drops out (D^2 = 1), so it is sum_i lambda_i p_i^2 with p = H e / 16. A rotated-basis diagonal
// weight cannot express this (the diagonal of H diag(lambda) H is the block mean). Two passes of coordinate
// descent over the groups, starting from the unweighted scales; a group's change enters p through
// H256 = H_{256/gs} (x) H_gs: p[a*gs + l] += (-1)^popcount(a & g) * (H_gs delta)[l] / 16.
static inline void rq4_block_scales(int nl, const float * y, const float * lambda, int gs, float * d) {
    const int ng = RQ4_HB/gs;
    float e[RQ4_HB];
    for (int g = 0; g < ng; ++g) {
        d[g] = rq4_fp16_to_fp32(rq4_fp32_to_fp16(rq4_choose_scale(nl, y + g*gs, gs)));
        const float id = d[g] != 0.0f ? 1.0f/d[g] : 0.0f;
        for (int i = 0; i < gs; ++i) {
            e[g*gs + i] = y[g*gs + i] - d[g]*rq4_val(nl, d[g] != 0.0f ? rq4_qcode(nl, y[g*gs + i], id) : rq4_zero_code(nl));
        }
    }
    if (lambda == NULL) {
        return;
    }
    float lsum = 0.0f;
    for (int i = 0; i < RQ4_HB; ++i) {
        lsum += lambda[i];
    }
    if (!(lsum > 0.0f)) {
        return;
    }
    float p[RQ4_HB];
    memcpy(p, e, sizeof(p));
    rq4_fwht(p, RQ4_HB);
    for (int i = 0; i < RQ4_HB; ++i) {
        p[i] *= 1.0f/16.0f;
    }
    float cand[RQ4_MAXCAND], en[RQ4_HB], w[RQ4_HB];
    for (int pass = 0; pass < 2; ++pass) {
        for (int g = 0; g < ng; ++g) {
            const float * yg = y + g*gs;
            float * eg = e + g*gs;
            const int nc = rq4_scale_candidates_g(nl, yg, gs, cand);
            float best_j = 0.0f;
            for (int i = 0; i < RQ4_HB; ++i) {
                best_j += lambda[i]*p[i]*p[i];
            }
            int best_c = -1;
            float best_w[RQ4_HB];
            for (int c = 0; c < nc; ++c) {
                const float dc = cand[c], id = 1.0f/dc;
                for (int i = 0; i < gs; ++i) {
                    en[i] = yg[i] - dc*rq4_val(nl, rq4_qcode(nl, yg[i], id));
                    w[i]  = en[i] - eg[i];
                }
                rq4_fwht(w, gs);
                float j = 0.0f;
                for (int a = 0; a < ng; ++a) {
                    const float sg = rq4_parity(a & g) ? -1.0f/16.0f : 1.0f/16.0f;
                    const float * pa = p + a*gs;
                    const float * la = lambda + a*gs;
                    for (int l = 0; l < gs; ++l) {
                        const float v = pa[l] + sg*w[l];
                        j += la[l]*v*v;
                    }
                }
                if (j < best_j) {
                    best_j = j;
                    best_c = c;
                    memcpy(best_w, w, gs*sizeof(float));
                }
            }
            if (best_c >= 0) {
                d[g] = cand[best_c];
                for (int a = 0; a < ng; ++a) {
                    const float sg = rq4_parity(a & g) ? -1.0f/16.0f : 1.0f/16.0f;
                    for (int l = 0; l < gs; ++l) {
                        p[a*gs + l] += sg*best_w[l];
                    }
                }
                const float id = 1.0f/d[g];
                for (int i = 0; i < gs; ++i) {
                    eg[i] = yg[i] - d[g]*rq4_val(nl, rq4_qcode(nl, yg[i], id));
                }
            }
        }
    }
}

// ---- GPTQ-style error-feedback rounding in the rotated basis (encoder only; format bytes unchanged) ----
// With an importance vector lambda (original basis, diagonal), the weighted error of a rotated block error e is
// e^T Hr e with Hr = (1/256) H256 diag(lambda) H256 (D drops out). Hr is dense, so rounding the 256 rotated
// coordinates one at a time and pushing each rounding error onto the not-yet-rounded ones (GPTQ, with
// U = chol(Hr^-1) upper) lowers the weighted error beyond per-element rounding. Hr^-1 has the same structure
// with 1/lambda: Hr^-1[a][b] = f[a ^ b] / 256, f = WHT(1/lambda'), lambda' = lambda + damp*mean(lambda).
#ifndef RQ4_GPTQ_DAMP
#define RQ4_GPTQ_DAMP 0.01
#endif

// U (row-major 256x256, upper triangle used) for one 256 block of lambda. Returns 0 if lambda is unusable.
static inline int rq4_gptq_prep(const float * lambda, float * U) {
    double mean = 0.0;
    for (int i = 0; i < RQ4_HB; ++i) {
        mean += (double) lambda[i];
    }
    mean /= RQ4_HB;
    if (!(mean > 0.0)) {
        return 0;
    }
    double f[RQ4_HB];
    for (int i = 0; i < RQ4_HB; ++i) {
        f[i] = 1.0/((double) lambda[i] + (double) RQ4_GPTQ_DAMP*mean);
    }
    for (int len = 1; len < RQ4_HB; len <<= 1) {
        for (int i = 0; i < RQ4_HB; i += 2*len) {
            for (int j = i; j < i + len; ++j) {
                const double a = f[j], b = f[j + len];
                f[j] = a + b;
                f[j + len] = a - b;
            }
        }
    }
    // Cholesky Hinv = L L^T (L lower, kept in the lower triangle of a local double matrix); U = L^T.
    const int n = RQ4_HB;
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

// Weighted objective sum_i lambda_i p_i^2, p = WHT(y - d.q)/16 (the rq4_block_scales objective).
static inline double rq4_block_obj(int nl, const float * y, const int8_t * q, const float * d, int gs, const float * lambda) {
    float p[RQ4_HB];
    for (int i = 0; i < RQ4_HB; ++i) {
        p[i] = y[i] - d[i/gs]*rq4_val(nl, q[i]);
    }
    rq4_fwht(p, RQ4_HB);
    double j = 0.0;
    for (int i = 0; i < RQ4_HB; ++i) {
        j += (double) lambda[i]*(double) (p[i]/16.0f)*(double) (p[i]/16.0f);
    }
    return j;
}

// Codes q of block y for scales d, by error feedback through U.
static inline void rq4_gptq_codes(int nl, const float * y, const float * d, int gs, const float * U, int8_t * q) {
    float w[RQ4_HB];
    memcpy(w, y, sizeof(w));
    for (int i = 0; i < RQ4_HB; ++i) {
        const float dg = d[i/gs];
        const int qi = dg != 0.0f ? rq4_qcode(nl, w[i], 1.0f/dg) : rq4_zero_code(nl);
        q[i] = (int8_t) qi;
        const float err = (w[i] - dg*rq4_val(nl, qi))/U[i*RQ4_HB + i];
        const float * Ui = U + i*RQ4_HB;
        for (int j = i + 1; j < RQ4_HB; ++j) {
            w[j] -= err*Ui[j];
        }
    }
}

// Least-squares group scales for fixed codes under the weighted objective (ng <= 4), fp16-rounded into dn.
// Uniform grid only: groups whose codes are all zero keep d0 (their scale does not enter the objective; RQ4NL
// codes are never 0, so every NL group is active) and are left out of the
// solve. Returns 0 if the system is singular or a scale is not finite / not representable.
static inline int rq4_refit_scales(int nl, const float * y, const int8_t * q, int gs, const float * lambda, const float * d0, float * dn) {
    const int ng = RQ4_HB/gs;
    float P[RQ4_HB/64][RQ4_HB], Y[RQ4_HB];
    int act[RQ4_HB/64], na = 0;
    for (int g = 0; g < ng; ++g) {
        dn[g] = d0[g];
        int nz = 0;
        for (int i = 0; i < gs; ++i) {
            nz |= rq4_val(nl, q[g*gs + i]) != 0.0f;
        }
        if (nz) {
            act[na++] = g;
        }
    }
    if (na == 0) {
        return 0;
    }
    for (int a = 0; a < na; ++a) {
        const int g = act[a];
        for (int i = 0; i < RQ4_HB; ++i) {
            P[a][i] = i/gs == g ? rq4_val(nl, q[i]) : 0.0f;
        }
        rq4_fwht(P[a], RQ4_HB);
    }
    memcpy(Y, y, sizeof(Y));
    // inactive groups contribute d0*q = 0, so the target is y itself
    rq4_fwht(Y, RQ4_HB);
    double A[4][5];
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < na; ++b) {
            double v = 0.0;
            for (int i = 0; i < RQ4_HB; ++i) {
                v += (double) lambda[i]*(double) P[a][i]*(double) P[b][i];
            }
            A[a][b] = v;
        }
        double v = 0.0;
        for (int i = 0; i < RQ4_HB; ++i) {
            v += (double) lambda[i]*(double) P[a][i]*(double) Y[i];
        }
        A[a][na] = v;
    }
    double amax = 0.0;
    for (int a = 0; a < na; ++a) {
        amax = fmax(amax, fabs(A[a][a]));
    }
    if (!(amax > 0.0)) {
        return 0;
    }
    for (int c = 0; c < na; ++c) {
        int piv = c;
        for (int r = c + 1; r < na; ++r) {
            if (fabs(A[r][c]) > fabs(A[piv][c])) {
                piv = r;
            }
        }
        if (!(fabs(A[piv][c]) > 1e-12*amax)) {
            return 0;
        }
        if (piv != c) {
            for (int k = 0; k <= na; ++k) {
                const double t = A[c][k]; A[c][k] = A[piv][k]; A[piv][k] = t;
            }
        }
        for (int r = 0; r < na; ++r) {
            if (r != c) {
                const double m = A[r][c]/A[c][c];
                for (int k = c; k <= na; ++k) {
                    A[r][k] -= m*A[c][k];
                }
            }
        }
    }
    for (int a = 0; a < na; ++a) {
        const float v = rq4_fp16_to_fp32(rq4_fp32_to_fp16((float) (A[a][na]/A[a][a])));
        if (!isfinite(v) || (v == 0.0f && !nl)) {   // NL: a zero-weight group may refit to ~0; kept only if the objective improves
            return 0;
        }
        dn[act[a]] = v;
    }
    return 1;
}

// Scales d and codes q of one rotated 256 block. Without lambda (or U == NULL): rq4_block_scales + plain
// rounding, as before. With both: start from that, then up to 2 rounds of {GPTQ codes for the current scales,
// least-squares scale refit}; every candidate (q, d) is scored on the weighted objective and the best is kept,
// so the result is never worse on that objective than plain rounding.
static inline void rq4_block_quant(int nl, const float * y, const float * lambda, const float * U, int gs, float * d, int8_t * q) {
    const int ng = RQ4_HB/gs;
    rq4_block_scales(nl, y, lambda, gs, d);
    for (int i = 0; i < RQ4_HB; ++i) {
        const float dg = d[i/gs];
        q[i] = (int8_t) (dg != 0.0f ? rq4_qcode(nl, y[i], 1.0f/dg) : rq4_zero_code(nl));
    }
    if (lambda == NULL || U == NULL) {
        return;
    }
    double best = rq4_block_obj(nl, y, q, d, gs, lambda);
    float dc[RQ4_HB/64], dn[RQ4_HB/64];
    int8_t qc[RQ4_HB];
    memcpy(dc, d, ng*sizeof(float));
    for (int it = 0; it < 2; ++it) {
        rq4_gptq_codes(nl, y, dc, gs, U, qc);
        double j = rq4_block_obj(nl, y, qc, dc, gs, lambda);
        if (j < best) {
            best = j; memcpy(d, dc, ng*sizeof(float)); memcpy(q, qc, sizeof(qc));
        }
        if (!rq4_refit_scales(nl, y, qc, gs, lambda, dc, dn)) {
            break;
        }
        j = rq4_block_obj(nl, y, qc, dn, gs, lambda);
        if (j < best) {
            best = j; memcpy(d, dn, ng*sizeof(float)); memcpy(q, qc, sizeof(qc));
        }
        memcpy(dc, dn, ng*sizeof(float));
    }
}

// GPTQ factors for a whole imatrix row of K columns: U for every 256 block (K/256 x 256 x 256 floats) and a
// per-block flag (1 = usable; 0 makes the block fall back to plain rounding). The imatrix is per column and shared by all rows
// of a tensor, so callers build this once per tensor (see ggml-rq4.c).
static inline void rq4_gptq_prep_row(const float * imatrix, int64_t K, float * Uall, uint8_t * ok) {
    for (int64_t b = 0; b < K/RQ4_HB; ++b) {
        ok[b] = (uint8_t) rq4_gptq_prep(imatrix + b*RQ4_HB, Uall + b*RQ4_HB*RQ4_HB);
    }
}

// Quantize one row of K floats (K % 1024 == 0) into RQ4 with group size gs (64 or 128).
// imatrix (optional, length K, original basis) weights the scale search and drives GPTQ rounding
// (rq4_block_quant). Uall/ok (optional, from rq4_gptq_prep_row on the same imatrix) avoid rebuilding the
// per-block GPTQ factor for every row; without them each block's factor is built on the fly.
static inline void rq4_quantize_row_c(const float * src, uint8_t * dst, int64_t K, int gs, int nl, const float * imatrix,
                                      const float * Uall, const uint8_t * ok) {
    float buf[RQ4_SB];
    float * tmpU = NULL;
    const int sbb = rq4_sb_bytes(gs);
    for (int64_t s = 0; s < K/RQ4_SB; ++s) {
        memcpy(buf, src + s*RQ4_SB, sizeof(buf));
        uint8_t  * blk = dst + s*sbb;
        uint16_t * dh  = (uint16_t *) blk;
        uint8_t  * nib = blk + 2*(RQ4_SB/gs);
        memset(nib, 0, RQ4_SB/2);
        for (int hb = 0; hb < RQ4_SB/RQ4_HB; ++hb) {
            float * y = buf + hb*RQ4_HB;
            rq4_rotate256(y);
            const float * lam = imatrix ? imatrix + s*RQ4_SB + hb*RQ4_HB : NULL;
            const float * U = NULL;
            const int64_t bi = s*(RQ4_SB/RQ4_HB) + hb;
            if (lam && Uall) {
                U = ok[bi] ? Uall + bi*RQ4_HB*RQ4_HB : NULL;
            } else if (lam) {
                if (tmpU == NULL) {
                    tmpU = (float *) malloc(sizeof(float)*RQ4_HB*RQ4_HB);
                }
                U = tmpU && rq4_gptq_prep(lam, tmpU) ? tmpU : NULL;
            }
            float d[RQ4_HB/64];
            int8_t qb[RQ4_HB];
            rq4_block_quant(nl, y, lam, U, gs, d, qb);
            for (int gl = 0; gl < RQ4_HB/gs; ++gl) {
                dh[hb*(RQ4_HB/gs) + gl] = rq4_fp32_to_fp16(d[gl]);
            }
            for (int i = 0; i < RQ4_HB; ++i) {
                const int k = hb*RQ4_HB + i;
                const int q = qb[i];
                int byte, hi;
                rq4_nibble_pos(k & (RQ4_CHUNK - 1), &byte, &hi);
                uint8_t * p = nib + (k/RQ4_CHUNK)*(RQ4_CHUNK/2) + byte;
                *p |= hi ? (uint8_t)((q & 0xf) << 4) : (uint8_t)(q & 0xf);
            }
        }
    }
    free(tmpU);
}

static inline void rq4_quantize_row(const float * src, uint8_t * dst, int64_t K, int gs, const float * imatrix) {
    rq4_quantize_row_c(src, dst, K, gs, 0, imatrix, NULL, NULL);
}

static inline void rq4nl_quantize_row(const float * src, uint8_t * dst, int64_t K, int gs, const float * imatrix) {
    rq4_quantize_row_c(src, dst, K, gs, 1, imatrix, NULL, NULL);
}

// Stored nibble (0..15) of weight k (0..1023) in a super block.
static inline int rq4_get_nibble(const uint8_t * nib, int k) {
    int byte, hi;
    rq4_nibble_pos(k & (RQ4_CHUNK - 1), &byte, &hi);
    const uint8_t v = nib[(k/RQ4_CHUNK)*(RQ4_CHUNK/2) + byte];
    return hi ? (v >> 4) : (v & 0xf);
}

// Code of weight k (0..1023) in a super block, as a signed value in [-8, 7] (uniform grid).
static inline int rq4_get_code(const uint8_t * nib, int k) {
    const int n = rq4_get_nibble(nib, k);
    return n >= 8 ? n - 16 : n;
}

// Grid value of weight k: the signed code (nl = 0) or rq4_nl_values[nibble] (nl = 1).
static inline int rq4_get_value(int nl, const uint8_t * nib, int k) {
    return nl ? rq4_nl_values[rq4_get_nibble(nib, k)] : rq4_get_code(nib, k);
}

// Rotated-domain dequantize (w' values), K % 1024 == 0.
static inline void rq4_dequantize_row_rotated_g(const uint8_t * src, float * dst, int64_t K, int gs, int nl) {
    const int sbb = rq4_sb_bytes(gs);
    for (int64_t s = 0; s < K/RQ4_SB; ++s) {
        const uint8_t  * blk = src + s*sbb;
        const uint16_t * dh  = (const uint16_t *) blk;
        const uint8_t  * nib = blk + 2*(RQ4_SB/gs);
        for (int k = 0; k < RQ4_SB; ++k) {
            dst[s*RQ4_SB + k] = rq4_fp16_to_fp32(dh[k/gs]) * (float) rq4_get_value(nl, nib, k);
        }
    }
}

static inline void rq4_dequantize_row_rotated(const uint8_t * src, float * dst, int64_t K, int gs) {
    rq4_dequantize_row_rotated_g(src, dst, K, gs, 0);
}

// Original-domain dequantize (unrotates), K % 1024 == 0.
static inline void rq4_dequantize_row_g(const uint8_t * src, float * dst, int64_t K, int gs, int nl) {
    rq4_dequantize_row_rotated_g(src, dst, K, gs, nl);
    for (int64_t b = 0; b < K/RQ4_HB; ++b) {
        rq4_unrotate256(dst + b*RQ4_HB);
    }
}

static inline void rq4_dequantize_row(const uint8_t * src, float * dst, int64_t K, int gs) {
    rq4_dequantize_row_g(src, dst, K, gs, 0);
}

#ifdef __cplusplus
}
#endif
