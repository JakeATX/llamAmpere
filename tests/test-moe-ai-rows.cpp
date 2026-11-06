// fork AIPFIX D2: check and time the host pool's AI+ unit slab kernel (moe-ai-rows.h) against the reference
// (aik_dequantize_slab_c to f32 in the original basis, then f32 dot products), on Qwen3.8 Flash-Next expert shapes:
// gate/up 2560 in x 640 out, down 640 in x 2560 out, for AIT2_U / AIT3_U / AIT4_U / AIT5_U and 1, 2 and 8 tokens.
// Weights are random bit streams with random scales; inputs random. Pass: every output within 2e-5 relative to the
// largest reference output of its projection (float summation order differs). The opt-in q8 path (int8 activations)
// must match its scalar reference bit for bit and stay within 5% of max|ref| (its error is reported). Exit code 1 on failure.
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
    const ggml_type types[] = { GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U, GGML_TYPE_AIT5_U, GGML_TYPE_AIT2_U2, GGML_TYPE_AIT3_U2, GGML_TYPE_AIT4_U2 };
    const int64_t shapes[][2] = { { 2560, 640 }, { 640, 2560 } };   // K (in), M (out)
#if defined(__AVX2__) && defined(__FMA__)
    printf("path: AVX2%s%s\n", moe_ai::has_avx512() ? " + AVX-512 (default; AVX2 also timed)" : "", moe_ai::has_vbmi() ? " + VBMI decode (default)" : "");
#else
    printf("path: scalar\n");
#endif
    int fails = 0;
    {
        // AIPFAST: R8 RAM fit keeps the 5 GiB MemAvailable floor (asked size capped, auto = pct of the room, floor min 5)
        const double G = 1073741824.0, M = 1048576.0;
        struct { long long req; double avail, floor; int pct; double want; } c[] = {
            { 6144, 16*G, 5, 50, 6144*M }, { 6144, 9*G, 5, 50, 4*G }, { 6144, 4*G, 5, 50, 0 }, { -1, 15*G, 5, 50, 5*G },
            { -1, 15*G, 7, 100, 8*G }, { 6144, 16*G, 2, 50, 6144*M }, { 20000, 10*G, 1, 50, 5*G }, { -1, 5*G, 5, 50, 0 } };
        int bad = 0;
        for (const auto & t : c) { bad += moe_ai::r8_fit_bytes(t.req, t.avail, t.floor, t.pct) != (size_t) t.want; }
        printf("r8 RAM fit: %d/%zu cases wrong  %s\n", bad, sizeof(c) / sizeof(c[0]), bad ? "FAIL" : "OK");
        fails += bad != 0;
    }
#if defined(__AVX2__) && defined(__FMA__)
    {
        // AIPFAST: AVX2 rotate + quant_q8 == scalar bit for bit (random rows at several scales, rows with a tied largest
        // |x| of both signs, all-zero blocks, values on .5 steps), then timed per 2560/640-float row
        std::vector<float> x(2560), ya(2560), ys(2560);
        std::vector<uint8_t> qa(moe_ai::q8_bytes(2560)), qs(moe_ai::q8_bytes(2560));
        int bad_r = 0, bad_q = 0, cases = 0;
        for (int64_t K : { (int64_t) 2560, (int64_t) 640 }) {
            for (int it = 0; it < 4000; ++it) {
                const float sc = std::ldexp(1.0f, (it % 40) - 20);
                for (int64_t i = 0; i < K; ++i) { x[i] = uni(rng) * sc; }
                if (it % 7 == 1) { const int64_t a = rng() % K, b = rng() % K; x[a] = 2.0f * sc; x[b] = -2.0f * sc; }
                if (it % 11 == 2) { std::fill(x.begin(), x.begin() + (K % 256 ? 128 : 256), 0.0f); }   // one zero rotation / q8 block
                if (it % 13 == 3) { for (int64_t i = 0; i < K; ++i) { x[i] = 0.5f * (float) ((int) (rng() % 255) - 127); } }
                moe_ai::rotate_avx2(x.data(), ya.data(), K);
                moe_ai::rotate_scalar(x.data(), ys.data(), K);
                bad_r += memcmp(ya.data(), ys.data(), K * sizeof(float)) != 0;
                // quantize both the rotated rows and the raw rows (the raw .5 steps exercise the rounding ties)
                for (const float * src : { (const float *) ys.data(), (const float *) x.data() }) {
                    moe_ai::quant_q8_avx2(src, qa.data(), K);
                    moe_ai::quant_q8_scalar(src, qs.data(), K);
                    bad_q += memcmp(qa.data(), qs.data(), moe_ai::q8_bytes(K)) != 0;
                }
                ++cases;
            }
            const int nr = 20000;
            auto tm = [&](auto && f) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < nr; ++i) { f(); }
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / nr;
            };
            const double rs = tm([&] { moe_ai::rotate_scalar(x.data(), ys.data(), K); });
            const double ra = tm([&] { moe_ai::rotate_avx2(x.data(), ya.data(), K); });
            const double qsu = tm([&] { moe_ai::quant_q8_scalar(ys.data(), qs.data(), K); });
            const double qau = tm([&] { moe_ai::quant_q8_avx2(ys.data(), qa.data(), K); });
            printf("rotate/quant_q8 K=%5lld: rotate scalar %.2f us avx2 %.2f us, quant_q8 scalar %.2f us avx2 %.2f us (1 thread)\n",
                   (long long) K, rs, ra, qsu, qau);
        }
        printf("rotate/quant_q8 AVX2 vs scalar: %d cases, rotate mismatches %d, quant mismatches %d  %s\n",
               cases, bad_r, bad_q, bad_r || bad_q ? "FAIL" : "OK");
        fails += bad_r != 0 || bad_q != 0;
    }
#endif
    for (ggml_type type : types) {
        const int k = ggml_ai4_rate(type);
        if (k == 0) {   // type has no unit rate in this tree (e.g. AIT4_U2 before its kernels land): the host split refuses it
            printf("%-8s no unit rate in this tree, skipped (host split refuses the type)\n", ggml_type_name(type));
            continue;
        }
        const uint32_t C = ggml_ai4_trellis_mul(type);
        for (const auto & sh : shapes) {
            const int64_t K = sh[0], M = sh[1];
            const size_t row_bytes = ggml_row_size(type, K);
            std::vector<uint8_t> w(row_bytes * M);
            for (auto & b : w) { b = (uint8_t) rng(); }
            const int ub = AI4U_UNIT_BYTES(k);
            // AIPFAST: U2 row-scale types: 16 row scales per 640-K block, then the 5 units' tiles (ggml-ai4.h AI4V2_*)
            const bool v2 = ggml_ai4v2_rate(type) != 0;
            if (v2 != moe_ai::slab_is_v2(row_bytes, k, K)) { printf("%-8s K=%5lld layout from row size WRONG FAIL\n", ggml_type_name(type), (long long) K); ++fails; }
            const int tbu = (AI4U_UK / AI4_TC) * AI4U_TILE_BYTES(k);
            auto scale_at = [&](int64_t sl, int64_t u) {   // scales of unit u (U2: of its block)
                return w.data() + sl * AI4_TR * row_bytes + (v2 ? (u / AI4V2_UNITS) * AI4V2_BLOCK_BYTES(k) : u * ub);
            };
            auto tiles_at = [&](int64_t sl, int64_t u) {
                return v2 ? scale_at(sl, u) + AI4U_SCALE_BYTES + (u % AI4V2_UNITS) * tbu : scale_at(sl, u) + AI4U_SCALE_BYTES;
            };
            for (int64_t sl = 0; sl < M / AI4_TR; ++sl) {
                for (int64_t u = 0; u < K / AI4U_UK; u += v2 ? AI4V2_UNITS : 1) {
                    uint16_t * dh = (uint16_t *) scale_at(sl, u);
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
                        const uint32_t * wd = (const uint32_t *) tiles_at(sl, u);
                        for (int ct = 0; ct < AI4U_UK / AI4_TC; ++ct) {
                            moe_ai::tile_bytes(wd + ct * tw, k, ext);
                            uint32_t E[AIK_TILE_WORDS(5) + 3] = {};
                            moe_ai::tile_words(wd + ct * tw, k, E);
                            // AIPFAST: the word window must equal the old byte-stream read at every byte offset
                            // whose three words lie inside E (B <= 4*tw + 3)
                            for (int B = 0; B <= 4 * tw + 3; ++B) { bad += moe_ai::window64(E, B) != moe_ai::load_be64(ext + B); }
                            for (int r = 0; r < AI4_TR; ++r) {
                                int lev[8];
                                float got[8];
                                moe_ai::row_levels_scalar(E, k, C, r, lev);
                                for (int kk = 0; kk < 8; ++kk) { bad += lev[kk] != aik_level_n(wd + ct * tw, k, C, r * AI4_TC + kk); }
                                _mm256_storeu_ps(got, moe_ai::row_levels_avx2(E, k, C, r));
                                for (int kk = 0; kk < 8; ++kk) { bad += got[kk] != (float) lev[kk]; }
                                // AIPFAST: the AVX2 shuffle decode must give the same levels
                                uint32_t Es[AIK_TILE_WORDS(5) + 3 + 4] = {};
                                memcpy(Es, E, sizeof(uint32_t) * (tw + 3));
                                const moe_ai::shuf_tabs & T = moe_ai::shuf_tables();
                                _mm256_storeu_ps(got, moe_ai::row_levels_shuf(Es, T, k, _mm256_loadu_si256((const __m256i *) T.sh[k]),
                                                                              _mm256_set1_epi32((int) C), r));
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
                if (v2) { aik_dequantize_slab_v2_c(k, C, w.data() + sl * AI4_TR * row_bytes, wf.data() + sl * AI4_TR * K, K); }
                else    { aik_dequantize_slab_c(k, C, w.data() + sl * AI4_TR * row_bytes, wf.data() + sl * AI4_TR * K, K); }
            }
            for (int nt = 1; nt <= 8; ++nt) {  // AIPFIX: 3-7 cover the G=4 and G=2 q8 groups and their odd tails
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
                {
                    // AVX2 decodes, all bit-identical, all timed (order 5 0 4 3 3 4 0 5): row-window decode (path 5, AIPFIX),
                    // shuffle with direct lane loads (path 0, AIPFAST), shuffle reading E (path 4), old window decode (path 3)
                    constexpr int NP = 4;
                    const int ps[NP] = { 5, 0, 4, 3 };
                    std::vector<float> ov[NP];
                    auto run_p = [&](std::vector<float> & o, int p) {
                        float * opx[8];
                        for (int64_t a = 0; a < M; a += AI4_TR) {
                            for (int t = 0; t < nt; ++t) { opx[t] = o.data() + t * M + a; }
                            moe_ai::slab_rows((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, yp, opx, p);
                        }
                    };
                    bool same = true;
                    for (int j = 0; j < NP; ++j) {
                        ov[j].assign((size_t) nt * M, 0.0f);
                        run_p(ov[j], ps[j]);
                        same = same && memcmp(ov[j].data(), ov[0].data(), ov[j].size() * sizeof(float)) == 0;
                    }
                    fails += !same;
                    double us[NP] = {};
                    for (int rep = 0; rep < 2; ++rep) {
                        for (int j = 0; j < NP; ++j) {
                            const int q = rep ? NP - 1 - j : j;
                            const auto ta = std::chrono::steady_clock::now();
                            for (int i = 0; i < reps; ++i) { run_p(ov[q], ps[q]); }
                            us[q] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ta).count() / (2.0 * reps);
                        }
                    }
                    printf("%-8s K=%5lld M=%5lld nt=%d  AVX2 rowwin/direct/shuf/old: %s  rowwin %.1f us  direct %.1f us  shuf %.1f us  old %.1f us /projection (1 thread)\n",
                           ggml_type_name(type), (long long) K, (long long) M, nt, same ? "bit-identical OK" : "DIFFERENT FAIL",
                           us[0], us[1], us[2], us[3]);
                }
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
                {
                    // AIPFIX: opt-in int8-activation path (slab_rows_q8): AVX2 == scalar reference bit for bit, error vs the
                    // double reference from activation rounding only (reported; sanity bound 5% of max|ref|), timed vs the
                    // default float path
                    std::vector<uint8_t> qb((size_t) nt * moe_ai::q8_bytes(K));
                    const void * qp[8];
                    for (int t = 0; t < nt; ++t) {
                        qp[t] = qb.data() + t * moe_ai::q8_bytes(K);
                        moe_ai::quant_q8(yp[t], (void *) qp[t], K);
                    }
                    std::vector<float> oq((size_t) nt * M), os((size_t) nt * M), ov((size_t) nt * M);
                    auto run_q = [&](std::vector<float> & o, bool scalar, int vbmi = 0, int lv = 0) {
                        float * opq[8];
                        for (int64_t a = 0; a < M; a += AI4_TR) {
                            for (int t = 0; t < nt; ++t) { opq[t] = o.data() + t * M + a; }
                            moe_ai::slab_rows_q8((const char *) w.data(), row_bytes, k, C, a, a + AI4_TR, K, nt, qp, opq, scalar, vbmi, lv);
                        }
                    };
                    run_q(oq, false);
                    run_q(os, true);
                    // AIPFIX: opt-in gate/up level kernel (GGML_MOE_AI_HOST_Q8_LV) == scalar bit for bit (down ignores lv)
                    std::vector<float> ol((size_t) nt * M);
                    run_q(ol, false, 0, 1);
                    const bool samel = memcmp(ol.data(), os.data(), ol.size() * sizeof(float)) == 0;
                    if (!samel) { printf("%-8s K=%5lld M=%5lld nt=%d  q8 lv==scalar DIFFERENT FAIL\n", ggml_type_name(type), (long long) K, (long long) M, nt); }
                    const bool samea = samel && memcmp(oq.data(), os.data(), oq.size() * sizeof(float)) == 0;
                    // AIPFIX: AVX-512 VBMI + VNNI q8 kernel (opt-in GGML_MOE_AI_HOST_Q8_VBMI=1) == scalar bit for bit, where present
#if defined(__AVX2__) && defined(__FMA__)
                    const bool hv = moe_ai::has_q8_vbmi();
#else
                    const bool hv = false;
#endif
                    bool samev = true;
                    if (hv) { run_q(ov, false, 1); samev = memcmp(ov.data(), os.data(), ov.size() * sizeof(float)) == 0; }
                    const bool same = samea && samev;
#if defined(__AVX2__) && defined(__FMA__)
                    // AIPFAST R8 host cache: decode + keep == scalar, and the hit kernel on the kept slot == scalar, bit for bit
                    if (moe_ai::r8_ok(k, C, M, K)) {
                        std::vector<uint8_t> slot(moe_ai::r8_slot_bytes(M, K) + 64, 0xA5);
                        uint8_t * R = slot.data() + (64 - (uintptr_t) slot.data() % 64) % 64;
                        std::vector<float> ok8((size_t) nt * M), oh8((size_t) nt * M);
                        auto run_r = [&](std::vector<float> & o, bool keep) {
                            float * op[8];
                            for (int64_t a = 0; a < M; a += AI4_TR) {
                                for (int t = 0; t < nt; ++t) { op[t] = o.data() + t * M + a; }
                                if (keep) { moe_ai::slab_rows_q8_keep((const char *) w.data(), row_bytes, k, C, a, M, K, nt, qp, op, R); }
                                else      { moe_ai::slab_rows_r8(R, a, M, K, nt, qp, op); }
                            }
                        };
                        run_r(ok8, true);
                        run_r(oh8, false);
                        const bool sk = memcmp(ok8.data(), os.data(), os.size() * sizeof(float)) == 0;
                        const bool sh8 = memcmp(oh8.data(), os.data(), os.size() * sizeof(float)) == 0;
                        double uk = 0.0, uh = 0.0, u8 = 0.0;
                        for (int m3 = 0; m3 < 3; ++m3) {
                            const auto ta = std::chrono::steady_clock::now();
                            for (int i = 0; i < reps; ++i) { if (m3 == 0) { run_r(ok8, true); } else if (m3 == 1) { run_r(oh8, false); } else { run_q(oq, false); } }
                            (m3 == 0 ? uk : m3 == 1 ? uh : u8) = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ta).count() / reps;
                        }
                        printf("%-8s K=%5lld M=%5lld nt=%d  r8: keep==scalar %s  hit==scalar %s  keep %.1f us  hit %.1f us  q8 %.1f us /projection (1 thread, warm)\n",
                               ggml_type_name(type), (long long) K, (long long) M, nt, sk ? "OK" : "DIFFERENT", sh8 ? "OK" : "DIFFERENT", uk, uh, u8);
                        fails += !(sk && sh8);
                    }
#endif
                    float sc = 0.0f, eabs = 0.0f, erel = 0.0f;
                    for (size_t i = 0; i < ref.size(); ++i) { sc = std::max(sc, std::fabs(ref[i])); }
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const float e = std::fabs(oq[i] - ref[i]);
                        eabs = std::max(eabs, e);
                        if (std::fabs(ref[i]) > 0.1f * sc) { erel = std::max(erel, e / std::fabs(ref[i])); }
                    }
                    const bool ok = same && eabs <= 0.05f * sc;
                    fails += !ok;
                    double uq = 0.0, uf = 0.0, uv = 0.0;
                    if (hv) {
                        const auto ta = std::chrono::steady_clock::now();
                        for (int i = 0; i < reps; ++i) { run_q(ov, false, 1); }
                        uv = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ta).count() / reps;
                    }
                    for (int rep = 0; rep < 2; ++rep) {
                        for (int j = 0; j < 2; ++j) {
                            const bool q8 = (rep ^ j) == 0;
                            const auto ta = std::chrono::steady_clock::now();
                            for (int i = 0; i < reps; ++i) { if (q8) { run_q(oq, false); } else { run(); } }
                            const double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ta).count() / (2.0 * reps);
                            (q8 ? uq : uf) += d;
                        }
                    }
                    printf("%-8s K=%5lld M=%5lld nt=%d  q8: avx2==scalar %s  vbmi==scalar %s (%.1f us)  max|err| %.3e  max|err|/max|ref| %.2e  max rel err (|ref| > 10%% max) %.2e  %s  q8 %.1f us vs default %.1f us /projection (1 thread)\n",
                           ggml_type_name(type), (long long) K, (long long) M, nt, samea ? "OK" : "DIFFERENT",
                           hv ? (samev ? "OK" : "DIFFERENT") : "n/a", uv, eabs, eabs / sc, erel,
                           ok ? "OK" : "FAIL", uq, uf);
                }
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
