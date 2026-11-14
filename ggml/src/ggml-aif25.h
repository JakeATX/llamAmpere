// Fractional computed-pair storage shared by the encoder and the CPU reference.
#pragma once
#include "ggml-ai4.h"

#define AIF25_MUL 0x87D8D12Du
#define AIF25_UNIT_BYTES 672 // 16 fp16 scales + 16 circular 320-bit tile streams
#define AIF25_V2_BLOCK_BYTES 3232 // 16 fp16 scales + five 640-byte pair-stream chunks
#define AIF25_CONTRACT "pair5-l16-magic2-unit128-v1"

// Physical words are little-endian; each word's stream bits run most-significant first.
static inline uint32_t aif25_state(const uint8_t * tile, int pair) {
    const int begin = (5*(pair + 1) + 320 - 16)%320;
    const int word = begin/32, shift = begin%32;
    const uint8_t * a = tile + 4*word;
    const uint32_t hi = (uint32_t)a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);
    uint32_t state;
    if (shift <= 16) {
        state = (hi >> (16-shift)) & 0xffffu;
    } else {
        const uint8_t * b = tile + 4*((word+1)%10);
        const uint32_t lo = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
        state = ((hi << (shift-16)) | (lo >> (48-shift))) & 0xffffu;
    }
    return state;
}

static inline int aif25_level(const uint8_t * tile, int index) {
    const uint32_t mul = index & 1 ? (uint32_t)((uint64_t)AIF25_MUL*AIF25_MUL) : AIF25_MUL;
    return aik_level_c(aif25_state(tile, index/2), mul);
}

// Four pair states for one eight-weight row share a 31-bit window.
static inline uint32_t aif25_state_quad(const uint32_t * words, int row) {
    const int begin = (20*(row+1) + 320 - 31)%320;
    const int word = begin/32, shift = begin%32;
    const uint32_t hi = words[word];
    if (shift <= 1) { return (hi >> (1-shift)) & 0x7fffffffu; }
    const uint32_t lo = words[(word+1)%10];
    return ((hi << (shift-1)) | (lo >> (33-shift))) & 0x7fffffffu;
}

// One slab is 16 complete output rows. Scales already contain the encoder's stretch.
static inline void aif25_dequantize_slab_impl(const uint8_t * src, float * dst, int64_t K, int chunks_per_scale) {
    for (int64_t u = 0; u < K/128; ++u) {
        const uint8_t * block = src + (u/chunks_per_scale)*(32 + chunks_per_scale*640);
        const uint8_t * tiles = block + 32 + (u%chunks_per_scale)*640;
        float scales[16];
        for (int r = 0; r < 16; ++r) {
            const uint16_t h = (uint16_t)block[2*r] | ((uint16_t)block[2*r+1] << 8);
            scales[r] = ai4_fp16_to_fp32(h);
        }
        for (int tile = 0; tile < 16; ++tile) {
            const uint8_t * packed = tiles + tile*40;
            uint32_t words[10];
            for (int w = 0; w < 10; ++w) {
                const uint8_t * a = packed + 4*w;
                words[w] = (uint32_t)a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24);
            }
            for (int r = 0; r < 16; ++r) {
                float * row = dst + r*K + u*128 + 8*tile;
                const uint32_t states = aif25_state_quad(words, r);
                for (int p = 0; p < 4; ++p) {
                    const uint32_t state = (states >> (15-5*p)) & 0xffffu;
                    row[2*p] = scales[r]*aik_level_c(state, AIF25_MUL);
                    row[2*p+1] = scales[r]*aik_level_c(state, (uint32_t)((uint64_t)AIF25_MUL*AIF25_MUL));
                }
            }
        }
    }
    const int rb = ai4u_rot_size(K);
    for (int r = 0; r < 16; ++r) {
        for (int64_t k = 0; k < K; k += rb) ai4_unrotate_n(dst + r*K + k, rb);
    }
}

static inline void aif25_dequantize_slab(const uint8_t * src, float * dst, int64_t K) {
    aif25_dequantize_slab_impl(src, dst, K, 1);
}

static inline void aif25_dequantize_slab_v2(const uint8_t * src, float * dst, int64_t K) {
    aif25_dequantize_slab_impl(src, dst, K, 5);
}
