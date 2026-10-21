// AI+ inference-only slab layout, reconstruction, and inverse rotation.
#pragma once

#include <stdint.h>
#include <string.h>


#define AI4_SB     1024

#define AI4_HB     256

#define AI4_TILE   128

#define AI4_MUL    0x83DCD12Du

#define AI4_TR     16     // tile rows (one slab)

#define AI4_TC     8      // tile K-columns

#define AI4_GS     128    // scale group (K)

#define AI4_SCALE_BYTES   (2*AI4_TR*(AI4_SB/AI4_GS))                 // 256

#define AI4_SLAB_SB_BYTES (AI4_SCALE_BYTES + (AI4_SB/AI4_TC)*64)      // 8448 = 16 x 528

#define AI4U_UK              128

#define AI4U_TILE_BYTES(k)   (16*(k))

#define AI4U_UNIT_BYTES(k)   (2*AI4_TR + (AI4U_UK/AI4_TC)*AI4U_TILE_BYTES(k))

#define AI4U_UNIT_BYTES4     AI4U_UNIT_BYTES(4)            // 1056

#define AI4U_SCALE_BYTES     (2*AI4_TR)                    // 32

#define AIK_TILE_WORDS(k)   (4*(k))

#define AI4S_SPAN_K          1024

#define AI4S_TILES_PER_SPAN  (AI4S_SPAN_K/AI4_TC)                                  // 128

#define AI4S_TILE_BYTES(k)   (AI4S_TILES_PER_SPAN*AI4U_TILE_BYTES(k))              // 2048k

#define AI4S_SPAN_BYTES(k)   (AI4U_SCALE_BYTES + AI4S_TILE_BYTES(k))               // 32 + 2048k


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

static inline int ai4_level(uint32_t state) {
    return aik_level_c(state, AI4_MUL);
}

static inline uint32_t ai4_get_state(const uint32_t * wd, int n) {
    const int w = n >> 3;
    const uint64_t c = ((uint64_t) wd[(w + 15) & 15] << 32) | wd[w];
    return (uint32_t)((c >> (4*(7 - (n & 7)))) & 0xFFFFu);
}

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

static inline void ai4_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    ai4_dequantize_slab_rotated(src, dst, K);
    for (int r = 0; r < AI4_TR; ++r) {
        for (int64_t b = 0; b < K/AI4_HB; ++b) {
            ai4_unrotate256(dst + r*K + b*AI4_HB);
        }
    }
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

static inline void ai4x_dequantize_slab(int unit, const uint8_t * src, float * dst, int64_t K) {
    if (unit) {
        ai4u_dequantize_slab(src, dst, K);
    } else {
        ai4_dequantize_slab(src, dst, K);
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

static inline int64_t ai4s_slab_bytes_k(int64_t K, int k) {
    return (K/AI4S_SPAN_K)*AI4S_SPAN_BYTES(k);
}

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

#ifdef __cplusplus
}
#endif
