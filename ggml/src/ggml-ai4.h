// AI+ unit layout (AITk_U): inference-only reconstruction and inverse rotation.
// AIT2P_U (ggml type 121): AIT2_U bytes (k = 2, 16-row x 128-K units, 16 tiles of 16 rows x 8 cols, MSB-first tile
// streams), AIT2P pair-table levels. Pair p (0..63) of a tile = weights 2p, 2p + 1 (one row, two adjacent columns);
// window w_p = the 20 stream bits [4p - 16, 4p + 4) (circular, MSB-first, so w_p = ((w_{p-1} << 4) | 4 new bits) mod 2^20);
// level(2p + ce) = ait2p_tab[2 ait2p_entry(w_p) + ce]. Decoders take C = trellis multiplier; C == 0 selects this code.
#pragma once

#include <stdint.h>
#include <string.h>

#include "ggml-ait.h"   // ait2p_tab / ait2p_entry (AIT2P_U)


#define AI4_HB     256

#define AI4_TILE   128

#define AI4_TR     16     // tile rows (one 16-row unit slab)

#define AI4_TC     8      // tile K-columns

#define AI4U_UK              128

#define AI4U_TILE_BYTES(k)   (16*(k))

#define AI4U_UNIT_BYTES(k)   (2*AI4_TR + (AI4U_UK/AI4_TC)*AI4U_TILE_BYTES(k))

#define AI4U_SCALE_BYTES     (2*AI4_TR)                    // 32

#define AIK_TILE_WORDS(k)   (4*(k))


#ifdef __cplusplus
extern "C" {
#endif


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

static inline void ai4_unrotate256(float * x) {
    ai4_fwht256(x);
    for (int i = 0; i < AI4_HB; ++i) {
        x[i] *= ai4_sign_neg(i) ? -1.0f/16.0f : 1.0f/16.0f;
    }
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

static inline int aik_level_c(uint32_t state, uint32_t C) {
    const uint32_t x = state * C;
    const int b = (int)(x & 255) + (int)((x >> 8) & 255) + (int)((x >> 16) & 255) + (int)(x >> 24);
    return ((b + 2) >> 2) - 128;
}

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

static inline float ai4_rot_norm(int n) {
    return n == 256 ? 1.0f/16.0f : 0.08838834764831845f;   // 1/sqrt(n)
}

static inline void ai4_unrotate_n(float * x, int n) {
    ai4_fwht_n(x, n);
    const float s = ai4_rot_norm(n);
    for (int i = 0; i < n; ++i) {
        x[i] *= ai4_sign_neg(i) ? -s : s;
    }
}

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

// AIT2P_U pair window: the 20 stream bits of a k = 2 tile ending at bit 4p + 4 (circular)
static inline uint32_t aik_pair_window(const uint32_t * wd, int p) {
    const int nb = AI4_TILE*2, nw = AIK_TILE_WORDS(2);
    int b0 = 4*p + 4 - 20;
    if (b0 < 0) {
        b0 += nb;
    }
    const int w = b0 >> 5, sh = b0 & 31;
    const uint64_t c = ((uint64_t) wd[w] << 32) | wd[(w + 1) % nw];
    return (uint32_t)((c >> (44 - sh)) & 0xFFFFFu);
}

// level of weight n of a tile: 1MAD with multiplier C, or (C == 0, k == 2) the AIT2P pair table
static inline int aik_level_n(const uint32_t * wd, int k, uint32_t C, int n) {
    if (C == 0) {
        return ait2p_tab[2*ait2p_entry(aik_pair_window(wd, n >> 1)) + (n & 1)];
    }
    return aik_level_c(aik_get_state(wd, k, n), C);
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
                dst[r*K + u*AI4U_UK + kk] = ai4_fp16_to_fp32(dh[r])*(float) aik_level_n(wd + ct*tw, k, C, n);
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

static inline int64_t ai4x_slab_bytes_k(int64_t K, int k) {
    return (K/AI4U_UK)*AI4U_UNIT_BYTES(k);
}

#ifdef __cplusplus
}
#endif
