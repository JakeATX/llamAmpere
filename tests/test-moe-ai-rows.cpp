// fork AIPFIX D2: check and time the host pool's AI+ unit slab kernel (moe-ai-rows.h) against the reference
// (aik_dequantize_slab_c to f32 in the original basis, then f32 dot products), on Qwen3.8 Flash-Next expert shapes:
// gate/up 2560 in x 640 out, down 640 in x 2560 out, for AIT2_U / AIT3_U / AIT4_U and 1, 2 and 8 tokens.
// Weights are random bit streams with random scales; inputs random. Pass: every output within 2e-5 relative to the
// largest reference output of its projection (float summation order differs). Exit code 1 on failure.
//
//   test-moe-ai-rows [reps for timing, default 20]
#include "../ggml/src/ggml-cuda/moe-ai-rows.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f32_to_f16_bits(float f) {
    return ggml_fp32_to_fp16(f);
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const ggml_type types[] = { GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U };
    const int64_t shapes[][2] = { { 2560, 640 }, { 640, 2560 } };   // K (in), M (out)
#if defined(__AVX2__) && defined(__FMA__)
    printf("path: AVX2%s%s\n", moe_ai::has_avx512() ? " + AVX-512 (default; AVX2 also timed)" : "", moe_ai::has_vbmi() ? " + VBMI decode (default)" : "");
#else
    printf("path: scalar\n");
#endif
    int fails = 0;
    for (ggml_type type : types) {
        const int k = ggml_ai4_rate(type);
        const uint32_t C = ggml_ai4_trellis_mul(type);
        for (const auto & sh : shapes) {
            const int64_t K = sh[0], M = sh[1];
            const size_t row_bytes = ggml_row_size(type, K);
            std::vector<uint8_t> w(row_bytes * M);
            for (auto & b : w) { b = (uint8_t) rng(); }
            const int ub = AI4U_UNIT_BYTES(k);
            for (int64_t sl = 0; sl < M / AI4_TR; ++sl) {
                for (int64_t u = 0; u < K / AI4U_UK; ++u) {
                    uint16_t * dh = (uint16_t *) (w.data() + sl * AI4_TR * row_bytes + u * ub);
                    for (int r = 0; r < AI4_TR; ++r) { dh[r] = f32_to_f16_bits(0.001f + 0.01f * (uni(rng) + 1.0f)); }
                }
            }
#if defined(__AVX2__) && defined(__FMA__)
            // the AVX2 level decode must give exactly the scalar levels (aik_level_c) for every tile row
            {
                int bad = 0;
                uint8_t ext[12 + 16*5 + 4];
                const int tw = AIK_TILE_WORDS(k);
                for (int64_t sl = 0; sl < M / AI4_TR; ++sl) {
                    for (int64_t u = 0; u < K / AI4U_UK; ++u) {
                        const uint32_t * wd = (const uint32_t *) (w.data() + sl * AI4_TR * row_bytes + u * ub + AI4U_SCALE_BYTES);
                        for (int ct = 0; ct < AI4U_UK / AI4_TC; ++ct) {
                            moe_ai::tile_bytes(wd + ct * tw, k, ext);
                            uint32_t E[AIK_TILE_WORDS(5) + 3];
                            moe_ai::tile_words(wd + ct * tw, k, E);
                            // AIPFAST: the word window must equal the old byte-stream read at every byte offset used
                            for (int B = 0; B + 8 <= 4 * tw + 12; ++B) { bad += moe_ai::window64(E, B) != moe_ai::load_be64(ext + B); }
                            for (int r = 0; r < AI4_TR; ++r) {
                                int lev[8];
                                float got[8];
                                moe_ai::row_levels_scalar(E, k, C, r, lev);
                                for (int kk = 0; kk < 8; ++kk) { bad += lev[kk] != aik_level_n(wd + ct * tw, k, C, r * AI4_TC + kk); }
                                _mm256_storeu_ps(got, moe_ai::row_levels_avx2(E, k, C, r));
                                for (int kk = 0; kk < 8; ++kk) { bad += got[kk] != (float) lev[kk]; }
                            }
                        }
                    }
                }
                printf("%-8s K=%5lld M=%5lld  AVX2 levels vs scalar: %d mismatches  %s\n",
                       ggml_type_name(type), (long long) K, (long long) M, bad, bad ? "FAIL" : "OK");
                fails += bad != 0;
            }
#endif
            // reference weights (original basis)
            std::vector<float> wf((size_t) M * K);
            for (int64_t sl = 0; sl < M / AI4_TR; ++sl) {
                aik_dequantize_slab_c(k, C, w.data() + sl * AI4_TR * row_bytes, wf.data() + sl * AI4_TR * K, K);
            }
            for (int nt : { 1, 2, 8 }) {
                std::vector<float> x((size_t) nt * K), y((size_t) nt * K), out((size_t) nt * M), ref((size_t) nt * M);
                for (auto & v : x) { v = uni(rng); }
                const float * yp[8];
                float * op[8];
                for (int t = 0; t < nt; ++t) {
                    moe_ai::rotate(x.data() + t * K, y.data() + t * K, K);
                    yp[t] = y.data() + t * K;
                    for (int64_t m = 0; m < M; ++m) {
                        double acc = 0.0;
                        for (int64_t c = 0; c < K; ++c) { acc += (double) wf[m * K + c] * x[t * K + c]; }
                        ref[t * M + m] = (float) acc;
                    }
                }
                auto run = [&]() {
                    for (int64_t a = 0; a < M; a += AI4_TR) {
                        for (int t = 0; t < nt; ++t) { op[t] = out.data() + t * M + a; }
                        moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, op);
                    }
                };
                run();
#if defined(__AVX2__) && defined(__FMA__)
                if (moe_ai::has_avx512()) {
                    // the AVX-512 path must equal the AVX2 path bit for bit
                    std::vector<float> o2((size_t) nt * M);
                    float * op2[8];
                    for (int64_t a = 0; a < M; a += AI4_TR) {
                        for (int t = 0; t < nt; ++t) { op2[t] = o2.data() + t * M + a; }
                        moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, op2, 0);
                    }
                    const bool same = memcmp(o2.data(), out.data(), o2.size() * sizeof(float)) == 0;
                    printf("%-8s K=%5lld M=%5lld nt=%d  AVX-512 vs AVX2: %s\n", ggml_type_name(type), (long long) K, (long long) M, nt,
                           same ? "bit-identical OK" : "DIFFERENT FAIL");
                    fails += !same;
                    const auto t2 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) {
                        for (int64_t a = 0; a < M; a += AI4_TR) {
                            for (int t = 0; t < nt; ++t) { op2[t] = o2.data() + t * M + a; }
                            moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, op2, 0);
                        }
                    }
                    printf("  AVX2 path %.1f us/projection (1 thread)\n",
                           std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t2).count() / reps);
                    // AIPFAST: plain AVX-512 decode (path 2) vs the default (VBMI where present): bit-identical, timed
                    for (int64_t a = 0; a < M; a += AI4_TR) {
                        for (int t = 0; t < nt; ++t) { op2[t] = o2.data() + t * M + a; }
                        moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, op2, 2);
                    }
                    const bool same2 = memcmp(o2.data(), out.data(), o2.size() * sizeof(float)) == 0;
                    printf("%-8s K=%5lld M=%5lld nt=%d  default%s vs plain AVX-512: %s\n", ggml_type_name(type), (long long) K,
                           (long long) M, nt, moe_ai::has_vbmi() ? " (VBMI)" : "", same2 ? "bit-identical OK" : "DIFFERENT FAIL");
                    fails += !same2;
                    const auto t3 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) {
                        for (int64_t a = 0; a < M; a += AI4_TR) {
                            for (int t = 0; t < nt; ++t) { op2[t] = o2.data() + t * M + a; }
                            moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, op2, 2);
                        }
                    }
                    printf("  plain AVX-512 path %.1f us/projection (1 thread)\n",
                           std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t3).count() / reps);
                }
#endif
                float scale = 0.0f, err = 0.0f;
                for (size_t i = 0; i < ref.size(); ++i) { scale = std::max(scale, std::fabs(ref[i])); }
                for (size_t i = 0; i < ref.size(); ++i) { err = std::max(err, std::fabs(out[i] - ref[i])); }
                const bool ok = err <= 2e-5f * scale && scale > 0.0f;
                fails += !ok;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) { run(); }
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                printf("%-8s K=%5lld M=%5lld nt=%d  max|err|/max|ref| = %.2e  %s  %.1f us/projection (1 thread)\n",
                       ggml_type_name(type), (long long) K, (long long) M, nt, scale > 0 ? err / scale : 0.0, ok ? "OK" : "FAIL", us);
            }
        }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
