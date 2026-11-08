// fork: Q2 host kernels, shared by the baseline and AVX2 math copies.
// Q2_0 (64 weights: an fp16 scale and 2-bit codes 0..3 meaning -1..2) against two Q8_0 blocks.
// BMI2 pdep spreads each 16-bit group of codes over 8 bytes, so a 32-weight half is four pdeps.
void dot_q2_0_q8_0(int n, float * s, const void * vx, const void * vy) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / QK2_0;
#if defined(__AVX2__) && defined(__BMI2__)
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = ggml_fp16_to_fp32(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 & yb = y[2 * i + k];
            uint64_t codes;
            memcpy(&codes, x[i].qs + 8 * k, 8);
            const uint64_t m = 0x0303030303030303ull;
            const __m256i w = _mm256_set_epi64x((long long) _pdep_u64(codes >> 48, m), (long long) _pdep_u64((codes >> 32) & 0xffff, m),
                    (long long) _pdep_u64((codes >> 16) & 0xffff, m), (long long) _pdep_u64(codes & 0xffff, m));
            const __m256i q = _mm256_loadu_si256((const __m256i *) yb.qs);
            const __m256i wq = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q), ones16);
            const __m256i sq = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16);
            const __m256 d = _mm256_set1_ps(d0 * ggml_fp16_to_fp32(yb.d));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_sub_epi32(wq, sq)), acc);
        }
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    *s = _mm_cvtss_f32(h);
#else
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d0 = ggml_fp16_to_fp32(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 & yb = y[2 * i + k];
            int isum = 0;
            for (int j = 0; j < 32; ++j) {
                const int code = (x[i].qs[8 * k + j / 4] >> (2 * (j % 4))) & 3;
                isum += (code - 1) * yb.qs[j];
            }
            sum += d0 * ggml_fp16_to_fp32(yb.d) * (float) isum;
        }
    }
    *s = sum;
#endif
}

// Q2_0 down rows, fast form (GGML_MOE_CPU_SPLIT_Q2_FAST=1). The hidden row is quantized once per route and its
// bytes are reordered inside each 32-weight block (position 8 s + b holds weight 4 b + s), so the 2-bit codes need
// no spreading: one 64-bit shift per lane (0, 2, 4, 6) and a mask line them up with the reordered bytes. The -1 code
// offset is folded out: sum (c - 1) q = sum c q - sum q, and sum q per block is computed once per hidden row (yc).
// Four weight rows share each hidden block load. Same integer products as dot_q2_0_q8_0; the float sums group
// differently (valid math).
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
inline float hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}
#endif

void q2_reorder_hidden(int n, block_q8_0 * y, float * yd, float * yc, bool shuffle_bytes) {
    const int nb = n / QK8_0;
#if defined(__AVX2__)
    const __m256i shuffle = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i lanes = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
#else
    (void) shuffle_bytes;
#endif
    for (int i = 0; i < nb; ++i) {
        int sum = 0;
#if defined(__AVX2__)
        if (shuffle_bytes) {
            const __m256i q = _mm256_loadu_si256((const __m256i *) y[i].qs);
            const __m256i reordered = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(q, shuffle), lanes);
            _mm256_storeu_si256((__m256i *) y[i].qs, reordered);
            const __m256i sums = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), q), _mm256_set1_epi16(1));
            __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            total = _mm_hadd_epi32(total, total);
            total = _mm_hadd_epi32(total, total);
            sum = _mm_cvtsi128_si32(total);
        } else
#endif
        {
            int8_t t[QK8_0];
            for (int s = 0; s < 4; ++s) {
                for (int b = 0; b < 8; ++b) {
                    const int8_t q = y[i].qs[4 * b + s];
                    t[8 * s + b] = q;
                    sum += q;
                }
            }
            memcpy(y[i].qs, t, QK8_0);
        }
        yd[i] = ggml_fp16_to_fp32(y[i].d);
        if (i & 1) { yc[i / 2] += yd[i] * (float) sum; } else { yc[i / 2] = yd[i] * (float) sum; }
    }
}

template<int NT>
void dot_q2_0_rows(int n, int nrows, const char * w, size_t row_bytes, const block_q8_0 * y, size_t y_stride,
        const float * yd, const float * yc, size_t aux_stride, float * out, size_t out_stride, int ahead) {
    const int nb = n / QK2_0;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    const __m256i shifts = _mm256_set_epi64x(6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    constexpr int rows = 4 / NT;
    for (int r = 0; r < nrows; r += rows) {
        const int nr = std::min(rows, nrows - r);
        __m256 acc[NT][rows];
        float corr[NT][rows] = {};
        for (int t = 0; t < NT; ++t) {
            for (int k = 0; k < rows; ++k) { acc[t][k] = _mm256_setzero_ps(); }
        }
        for (int i = 0; i < nb; ++i) {
            __m256i q0[NT], q1[NT];
            __m256 s0[NT], s1[NT];
            for (int t = 0; t < NT; ++t) {
                const auto * yt = (const block_q8_0 *) ((const char *) y + (size_t) t * y_stride);
                q0[t] = _mm256_loadu_si256((const __m256i *) yt[2 * i].qs);
                q1[t] = _mm256_loadu_si256((const __m256i *) yt[2 * i + 1].qs);
                s0[t] = _mm256_set1_ps(yd[(size_t) t * aux_stride + 2 * i]);
                s1[t] = _mm256_set1_ps(yd[(size_t) t * aux_stride + 2 * i + 1]);
            }
            for (int k = 0; k < nr; ++k) {
                const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) (r + k) * row_bytes) + i;
                if (ahead > 0 && i + ahead < nb && (i & 3) == 0) {
                    _mm_prefetch((const char *) (x + ahead), _MM_HINT_T0);
                    _mm_prefetch((const char *) (x + ahead + 1) - 1, _MM_HINT_T0);
                }
                const float d = _cvtsh_ss(x->d);
                long long c0, c1;
                memcpy(&c0, x->qs, 8);
                memcpy(&c1, x->qs + 8, 8);
                const __m256i w0 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c0), shifts), m3);
                const __m256i w1 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c1), shifts), m3);
                for (int t = 0; t < NT; ++t) {
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, q0[t]), ones16);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, q1[t]), ones16);
                    const __m256 f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), s0[t], _mm256_mul_ps(_mm256_cvtepi32_ps(p1), s1[t]));
                    acc[t][k] = _mm256_fmadd_ps(_mm256_set1_ps(d), f, acc[t][k]);
                    corr[t][k] += d * yc[(size_t) t * aux_stride + i];
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            for (int k = 0; k < nr; ++k) { out[(size_t) t * out_stride + r + k] = hsum8(acc[t][k]) - corr[t][k]; }
        }
    }
#else
    (void) ahead;
    for (int t = 0; t < NT; ++t) {
        const auto * yt = (const block_q8_0 *) ((const char *) y + (size_t) t * y_stride);
        for (int r = 0; r < nrows; ++r) {
            const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) r * row_bytes);
            float sum = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = ggml_fp16_to_fp32(x[i].d);
                for (int k = 0; k < 2; ++k) {
                    const block_q8_0 & yb = yt[2 * i + k];
                    int isum = 0;
                    for (int s = 0; s < 4; ++s) {
                        for (int b = 0; b < 8; ++b) { isum += ((x[i].qs[8 * k + b] >> (2 * s)) & 3) * yb.qs[8 * s + b]; }
                    }
                    sum += d * yd[(size_t) t * aux_stride + 2 * i + k] * (float) isum;
                }
                sum -= d * yc[(size_t) t * aux_stride + i];
            }
            out[(size_t) t * out_stride + r] = sum;
        }
    }
#endif
}
