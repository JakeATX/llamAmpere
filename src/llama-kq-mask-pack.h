#pragma once

#include "ggml.h"

inline void llama_kq_mask_pack_rows(const ggml_fp16_t * src, uint32_t * dst, int64_t n_rows,
        int64_t n_kv, int64_t n_w, ggml_fp16_t keep, ggml_fp16_t drop) {
    const int64_t n_full = n_kv/32;
    const int64_t n_tail = n_kv%32;
    for (int64_t r = 0; r < n_rows; ++r) {
        const ggml_fp16_t * row = src + r*n_kv;
        uint32_t * w = dst + r*n_w;
        uint32_t invalid = 0;
        for (int64_t iw = 0; iw < n_full; ++iw) {
            uint32_t bits = 0;
            for (uint32_t b = 0; b < 32; ++b) {
                const uint16_t v = row[32*iw + b];
                invalid |= uint32_t((v != keep) & (v != drop));
                bits |= uint32_t(v == keep) << b;
            }
            w[iw] = bits;
        }
        if (n_tail) {
            uint32_t bits = 0;
            for (int64_t b = 0; b < n_tail; ++b) {
                const uint16_t v = row[32*n_full + b];
                invalid |= uint32_t((v != keep) & (v != drop));
                bits |= uint32_t(v == keep) << b;
            }
            w[n_full] = bits;
        }
        GGML_ASSERT(invalid == 0);
    }
}
