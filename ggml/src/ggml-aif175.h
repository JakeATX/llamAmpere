#pragma once
#include "ggml-ai4.h"

#define AIF175_MUL 0x87D8D12Du
#define AIF175_TILE_BYTES 28
#define AIF175_BLOCK_BYTES 1824
#define AIF175_CONTRACT "pair34-l16-magic2-row512-v1"

static inline uint32_t aif175_state(const uint8_t * tile, int pair) {
    const int end = 7*(pair/2) + (pair%2 ? 7 : 3);
    const int begin = (end + 224 - 16)%224, word = begin/32, shift = begin%32;
    const uint8_t * a = tile + 4*word;
    const uint32_t hi = (uint32_t)a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);
    if (shift <= 16) { return (hi >> (16-shift)) & 0xffffu; }
    // The last word wraps to word zero; no read may cross the tile boundary.
    const uint8_t * b = tile + 4*((word+1)%7);
    const uint32_t lo = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return ((hi << (shift-16)) | (lo >> (48-shift))) & 0xffffu;
}

// Two states at an even pair index share twelve bits.
static inline uint32_t aif175_state_pair(const uint8_t * tile, int pair) {
    const int end = 7*(pair/2) + 7;
    const int begin = (end + 224 - 20)%224, word = begin/32, shift = begin%32;
    const uint8_t * a = tile + 4*word;
    const uint32_t hi = (uint32_t)a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);
    if (shift <= 12) { return (hi >> (12-shift)) & 0xfffffu; }
    const uint8_t * b = tile + 4*((word+1)%7);
    const uint32_t lo = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return ((hi << (shift-12)) | (lo >> (44-shift))) & 0xfffffu;
}

static inline int aif175_level(const uint8_t * tile, int index) {
    const uint32_t state = aif175_state(tile, index/2);
    const uint32_t mul = index & 1 ? (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL) : AIF175_MUL;
    return aik_level_c(state, mul);
}

static inline void aif175_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    for (int64_t b = 0; b < K/512; ++b) {
        const uint8_t * block = src + b*AIF175_BLOCK_BYTES;
        for (int r = 0; r < 16; ++r) {
            const uint16_t h = (uint16_t)block[2*r] | ((uint16_t)block[2*r+1] << 8);
            const float scale = ai4_fp16_to_fp32(h);
            float * row = dst + r*K + b*512;
            for (int u = 0; u < 4; ++u) {
                const uint8_t * tiles = block + 32 + u*448;
                for (int k = 0; k < 128; k += 4) {
                    const uint32_t states = aif175_state_pair(tiles + (k/8)*28, r*4 + (k%8)/2);
                    const uint32_t first = states >> 4, second = states & 0xffffu;
                    row[u*128 + k] = scale*aik_level_c(first, AIF175_MUL);
                    row[u*128 + k+1] = scale*aik_level_c(first, (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL));
                    row[u*128 + k+2] = scale*aik_level_c(second, AIF175_MUL);
                    row[u*128 + k+3] = scale*aik_level_c(second, (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL));
                }
            }
            ai4_unrotate_n(row, 256);
            ai4_unrotate_n(row + 256, 256);
        }
    }
}
