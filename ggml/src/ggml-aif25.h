// Fractional computed-pair storage shared by the encoder and the CPU reference.
#pragma once
#include "ggml-ai4.h"

#define AIF25_MUL 0x87D8D12Du
#define AIF25_UNIT_BYTES 672 // 16 fp16 scales + 16 circular 320-bit tile streams
#define AIF25_CONTRACT "pair5-l16-magic2-unit128-v1"

// Physical words are little-endian; each word's stream bits run most-significant first.
static inline int aif25_level(const uint8_t * tile, int index) {
    uint32_t state = 0;
    for (int j = 0; j < 16; ++j) {
        const int bit = (5*(index/2 + 1) - 16 + j + 320) % 320;
        const int byte = (bit/32)*4 + 3 - (bit%32)/8;
        state = (state << 1) | ((tile[byte] >> (7 - bit%8)) & 1);
    }
    const uint32_t mul = index & 1 ? (uint32_t)((uint64_t)AIF25_MUL*AIF25_MUL) : AIF25_MUL;
    return aik_level_c(state, mul);
}

// One slab is 16 complete output rows. Scales already contain the encoder's stretch.
static inline void aif25_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    for (int64_t u = 0; u < K/128; ++u) {
        const uint8_t * unit = src + u*AIF25_UNIT_BYTES;
        for (int r = 0; r < 16; ++r) {
            uint16_t h; memcpy(&h, unit + 2*r, 2);
            const float scale = ai4_fp16_to_fp32(h);
            for (int k = 0; k < 128; ++k) {
                dst[r*K + u*128 + k] = scale*aif25_level(unit + 32 + (k/8)*40, r*8 + k%8);
            }
        }
    }
    const int rb = ai4u_rot_size(K);
    for (int r = 0; r < 16; ++r) {
        for (int64_t k = 0; k < K; k += rb) ai4_unrotate_n(dst + r*K + k, rb);
    }
}
