#pragma once
#include "ggml-ai4.h"

#define AIF175_MUL 0x87D8D12Du
#define AIF175_TILE_BYTES 28
#define AIF175_BLOCK_BYTES 1824
#define AIF175_CONTRACT "pair34-l16-magic2-row512-v1"

static inline int aif175_level_state(uint32_t state, uint32_t mul) {
    const uint32_t value = state*mul;
    const uint32_t pairs = (value & 0x00ff00ffu) + ((value >> 8) & 0x00ff00ffu);
    const uint32_t sum = (pairs & 0xffffu) + (pairs >> 16);
    return (int)((sum+2) >> 2)-128;
}

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

static inline uint32_t aif175_state_window(const uint32_t * words, int begin) {
    const int word = begin/32, shift = begin%32;
    const uint32_t hi = words[word];
    if (shift <= 5) { return (hi >> (5-shift)) & 0x7ffffffu; }
    const uint32_t lo = words[(word+1)%7];
    return ((hi << (shift-5)) | (lo >> (37-shift))) & 0x7ffffffu;
}

// Four states for one eight-weight row share a 27-bit window.
static inline uint32_t aif175_state_quad(const uint32_t * words, int row) {
    return aif175_state_window(words, (14*(row+1) + 224 - 27)%224);
}

static inline int aif175_level(const uint8_t * tile, int index) {
    const uint32_t state = aif175_state(tile, index/2);
    const uint32_t mul = index & 1 ? (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL) : AIF175_MUL;
    return aif175_level_state(state, mul);
}

static inline void aif175_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    for (int64_t b = 0; b < K/512; ++b) {
        const uint8_t * block = src + b*AIF175_BLOCK_BYTES;
        float scales[16];
        int rows[16], begins[16], n_rows = 0;
        for (int r = 0; r < 16; ++r) {
            const uint16_t h = (uint16_t)block[2*r] | ((uint16_t)block[2*r+1] << 8);
            scales[r] = ai4_fp16_to_fp32(h);
            if (scales[r] == 0.0f) {
                float * row = dst + r*K + b*512;
                for (int k = 0; k < 512; ++k) { row[k] = 0.0f; }
            } else {
                begins[n_rows] = (14*(r+1) + 224 - 27)%224;
                rows[n_rows++] = r;
            }
        }
        if (n_rows == 0) { continue; }
        for (int tile = 0; tile < 64; ++tile) {
            const uint8_t * packed = block + 32 + tile*AIF175_TILE_BYTES;
            uint32_t words[7];
            // Reuse the seven unaligned little-endian words across all sixteen rows.
            for (int w = 0; w < 7; ++w) {
                const uint8_t * a = packed + 4*w;
                words[w] = (uint32_t)a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);
            }
            for (int i = 0; i < n_rows; ++i) {
                const int r = rows[i];
                float * row = dst + r*K + b*512 + 8*tile;
                const uint32_t states = aif175_state_window(words, begins[i]);
                for (int p = 0; p < 4; ++p) {
                    const int shift = 11 - 7*(p/2) - 4*(p%2);
                    const uint32_t state = (states >> shift) & 0xffffu;
                    row[2*p] = scales[r]*aif175_level_state(state, AIF175_MUL);
                    row[2*p+1] = scales[r]*aif175_level_state(state, (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL));
                }
            }
        }
        for (int i = 0; i < n_rows; ++i) {
            const int r = rows[i];
            float * row = dst + r*K + b*512;
            ai4_unrotate_n(row, 256);
            ai4_unrotate_n(row + 256, 256);
        }
    }
}
