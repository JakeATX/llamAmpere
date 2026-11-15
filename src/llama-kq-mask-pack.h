#pragma once

#include "ggml.h"

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#else
#include <algorithm>
#endif

inline void llama_kq_mask_pack_rows(const ggml_fp16_t * src, uint32_t * dst, int64_t n_rows,
        int64_t n_kv, int64_t n_w, ggml_fp16_t keep, ggml_fp16_t drop) {
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    const int64_t n_full = n_kv/32;
    const int64_t n_tail = n_kv%32;
    const __m128i keep_v = _mm_set1_epi16((short) keep);
    const __m128i drop_v = _mm_set1_epi16((short) drop);
    const __m128i all_ones = _mm_set1_epi16(-1);
    for (int64_t r = 0; r < n_rows; ++r) {
        const ggml_fp16_t * row = src + r*n_kv;
        uint32_t * w = dst + r*n_w;
        __m128i invalid_lanes = _mm_setzero_si128();
        for (int64_t iw = 0; iw < n_full; ++iw) {
            const ggml_fp16_t * p = row + 32*iw;
            const __m128i v0 = _mm_loadu_si128((const __m128i *) (p +  0));
            const __m128i v1 = _mm_loadu_si128((const __m128i *) (p +  8));
            const __m128i v2 = _mm_loadu_si128((const __m128i *) (p + 16));
            const __m128i v3 = _mm_loadu_si128((const __m128i *) (p + 24));
            const __m128i k0 = _mm_cmpeq_epi16(v0, keep_v);
            const __m128i k1 = _mm_cmpeq_epi16(v1, keep_v);
            const __m128i k2 = _mm_cmpeq_epi16(v2, keep_v);
            const __m128i k3 = _mm_cmpeq_epi16(v3, keep_v);
            const __m128i valid0 = _mm_or_si128(k0, _mm_cmpeq_epi16(v0, drop_v));
            const __m128i valid1 = _mm_or_si128(k1, _mm_cmpeq_epi16(v1, drop_v));
            const __m128i valid2 = _mm_or_si128(k2, _mm_cmpeq_epi16(v2, drop_v));
            const __m128i valid3 = _mm_or_si128(k3, _mm_cmpeq_epi16(v3, drop_v));
            const __m128i valid = _mm_and_si128(_mm_and_si128(valid0, valid1),
                                               _mm_and_si128(valid2, valid3));
            invalid_lanes = _mm_or_si128(invalid_lanes, _mm_andnot_si128(valid, all_ones));
            // Packing -1/0 lanes preserves element order in the movemask bits.
            w[iw] = uint32_t(_mm_movemask_epi8(_mm_packs_epi16(k0, k1))) |
                    (uint32_t(_mm_movemask_epi8(_mm_packs_epi16(k2, k3))) << 16);
        }
        uint32_t invalid = uint32_t(_mm_movemask_epi8(invalid_lanes));
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
#else
    for (int64_t r = 0; r < n_rows; ++r) {
        const ggml_fp16_t * row = src + r*n_kv;
        uint32_t          * w   = dst + r*n_w;
        for (int64_t iw = 0; iw < n_w; ++iw) {
            const int64_t j0 = iw*32;
            const int64_t nj = std::min<int64_t>(32, n_kv - j0);
            uint32_t bits = 0;
            for (int64_t b = 0; b < nj; ++b) {
                const ggml_fp16_t v = row[j0 + b];
                // the f16 mask holds 0 or -INF here (ALiBi masks never take this path)
                GGML_ASSERT(v == keep || v == drop);
                bits |= (uint32_t) (v == keep) << b;
            }
            w[iw] = bits;
        }
    }
#endif
}
