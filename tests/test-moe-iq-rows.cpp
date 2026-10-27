// fork MOEGAP: decode check, parity and speed of the host pool's IQ row kernels (moe-iq-rows.h,
// GGML_MOE_CPU_SPLIT_ROWS) against ggml's own code, on Qwen3.8 Flash-Next expert shapes (2560 in x 640 out).
//
//   test-moe-iq-rows <cpu,cpu,...> [copies, default 256] [check]
//
// Decode: for every IQ type the kernels handle (and both IQ3 grid arms), every 32-weight slice the kernels unpack
// (magnitudes x signs x odd multiplier x block d x format factor) must equal ggml's dequantize output for those
// weights. Exit code 1 on any difference.
//
// Parity: for every IQ up/gate type and nt = 1..8 tokens, every output must have the same bits as ggml's vec_dot
// (the kernels do the same integer work in the same lanes and the same float steps). Exit code 1 on any difference.
// Speed: one pinned thread per listed cpu, each with its own `copies` experts (cold: they rotate, far more than the
// last-level cache, as a decode's missed experts do; hot: one expert repeats). Prints us per expert projection and
// the summed GB/s of weights read, for ggml (vec_dot once per row per token) and the row kernels: variant bits
// (moe_iq::get) 0, 1 (IQ3_XXS grid pairs in general registers), 2 (two rows per tile for one token), 4 (grid pairs
// for IQ3_S), all without prefetch; variant 0 with prefetch 2048 B; and "best" (variant 9, GGML_MOE_CPU_SPLIT_ROWS=1),
// which leaves one token of IQ2_S and IQ3_XXS to ggml's vec_dot, as the pool does.
#include "../ggml/src/ggml-cuda/moe-cpu-split.h"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

using clk = std::chrono::steady_clock;

static void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static uint64_t g_rng = 88172645463325252ull;
static float frand() {   // roughly normal: sum of 4 uniforms
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
        s += (float) (g_rng >> 40) / (float) (1ull << 24) - 0.5f;
    }
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <cpu,cpu,...> [copies]\n", argv[0]); return 2; }
    std::vector<int> cpus;
    for (const char * c = argv[1]; *c;) {
        cpus.push_back(atoi(c));
        while (*c && *c != ',') { ++c; }
        if (*c == ',') { ++c; }
    }
    const int copies = argc > 2 ? std::max(1, atoi(argv[2])) : 256;
    ggml_cpu_init();
    const int64_t n_in = 2560, n_out = 640;
    const int NT = 8;
    const ggml_type types[] = { GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS };
    const int NM = 9;                                   // ggml, six AVX2 settings, two AVX-512 settings
    const int var_of[NM] = { -1, 0, 1, 2, 4, 0, 1 | 8, 0, 0 };
    const int pf_of[NM] = { 0, 0, 0, 0, 0, 2048, 0, 0, 2048 };
    const bool check_only = argc > 3 && strcmp(argv[3], "check") == 0;
    long long bad_total = 0;
    printf("threads %zu, %d expert copies per thread\n", cpus.size(), copies);
    printf("  %-8s nt | %-24s | %-24s | %-24s | %-24s | %-24s | %-24s | %-24s | %-24s | %-24s | parity failures\n", "type",
           "ggml cold/hot us GB/s", "v0", "v1 iq3_xxs pair", "v2 2-row", "v4 iq3_s pair", "v0 pf2048", "best (ROWS=1)", "AVX-512", "AVX-512 pf2048");
    for (ggml_type type : types) {
        const auto * tr = ggml_get_type_traits_cpu(type);
        const size_t row_bytes = ggml_row_size(type, n_in);
        const size_t bytes = row_bytes * (size_t) n_out;
        std::vector<float> w((size_t) n_in * n_out), imat((size_t) n_in, 1.0f);
        for (auto & v : w) { v = 0.05f * frand(); }
        std::vector<uint8_t> q(bytes);
        ggml_quantize_chunk(type, w.data(), q.data(), 0, n_out, n_in, imat.data());
        {   // decode check against ggml's dequantize
            const auto * tt = ggml_get_type_traits(type);
            const size_t tsz = ggml_type_size(type);
            const float post = type == GGML_TYPE_IQ3_S ? 1.0f : type == GGML_TYPE_IQ3_XXS ? 0.25f : 0.125f;
            std::vector<float> dq((size_t) n_in);
            long long dec_bad = 0, dec_n = 0, dec_arms = 0;
            for (int joined = 0; joined < 2; ++joined) {
                const bool iq3 = type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS;
                if (joined && !iq3) { continue; }
                ++dec_arms;
                for (int64_t r = 0; r < n_out; ++r) {
                    const uint8_t * row = q.data() + r * row_bytes;
                    tt->to_float(row, dq.data(), n_in);
                    for (int b = 0; b < n_in / QK_K; ++b) {
                        const uint8_t * blk = row + (size_t) b * tsz;
                        uint16_t dh;
                        memcpy(&dh, blk, sizeof(dh));
                        const float d = ggml_fp16_to_fp32(dh);
                        for (int c = 0; c < 8; ++c) {
                            uint8_t mag[32];
                            int8_t sgn[32];
                            int mul[2];
                            if (!moe_cs_iq_unpack((int) type, blk, c, joined != 0, mag, sgn, mul)) { dec_bad += 32; continue; }
                            for (int j = 0; j < 32; ++j) {
                                const float got = d * post * (float) mul[j >= 16] * (float) mag[j] * (float) sgn[j];
                                const float want = dq[(size_t) b * QK_K + (size_t) c * 32 + j];
                                ++dec_n;
                                dec_bad += !(fabsf(got - want) <= 1e-6f * fabsf(want) + 1e-12f);
                            }
                        }
                    }
                }
            }
            printf("  %-8s decode vs ggml dequantize: %lld weights, %lld arms, %lld differences\n", ggml_type_name(type), dec_n,
                   dec_arms, dec_bad);
            bad_total += dec_bad;
        }
        const size_t act_bytes = ggml_row_size(tr->vec_dot_type, n_in);
        std::vector<uint8_t> act(NT * act_bytes);
        for (int t = 0; t < NT; ++t) {
            std::vector<float> x(n_in);
            for (auto & v : x) { v = frand(); }
            ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), act.data() + t * act_bytes, n_in);
            if (t == NT - 1) {
                auto * blocks = (block_q8_K *) (act.data() + t * act_bytes);
                for (int b = 0; b < n_in / QK_K; ++b) { blocks[b].qs[0] = -128; blocks[b].qs[31] = 127; }
            }
        }
        const void * actp[NT];
        for (int t = 0; t < NT; ++t) { actp[t] = act.data() + t * act_bytes; }
        const size_t nth = cpus.size();
        std::vector<std::vector<uint8_t *>> ex(nth);
        for (size_t th = 0; th < nth; ++th) {
            for (int e = 0; e < copies; ++e) {
                uint8_t * d = (uint8_t *) aligned_alloc(4096, (bytes + 4095) & ~(size_t) 4095);
                memcpy(d, q.data(), bytes);
                ex[th].push_back(d);
            }
        }
        std::vector<float> ref((size_t) NT * n_out);
        for (int t = 0; t < NT; ++t) {
            for (int64_t r = 0; r < n_out; ++r) {
                tr->vec_dot((int) n_in, &ref[(size_t) t * n_out + r], 0, q.data() + r * row_bytes, 0, actp[t], 0, 1);
            }
        }
        // m = 0: ggml; m >= 1: row kernel variant var_of[m], prefetch pf_of[m] (a case the variant leaves to ggml
        // runs ggml's vec_dot, as the pool does)
        bool on_ggml[NM][NT + 1];
        for (int m = 0; m < NM; ++m) {
            for (int nt = 1; nt <= NT; ++nt) {   // zero rows: only asks whether the variant has a kernel for this case
                on_ggml[m][nt] = m == 0 || !moe_cs_iq_rows((int) type, nt, q.data(), row_bytes, 0, 0, n_in, actp, nullptr, 0, var_of[m], m >= 7);
            }
        }
        auto run = [&](int m, const uint8_t * wp, int nt, float * const * out) {
            if (on_ggml[m][nt]) {
                for (int t = 0; t < nt; ++t) {
                    for (int64_t r = 0; r < n_out; ++r) {
                        tr->vec_dot((int) n_in, &out[t][r], 0, wp + r * row_bytes, 0, actp[t], 0, 1);
                    }
                }
                return true;
            }
            return moe_cs_iq_rows((int) type, nt, wp, row_bytes, 0, n_out, n_in, actp, out, pf_of[m], var_of[m], m >= 7);
        };
        for (int nt = 1; nt <= NT; ++nt) {
            printf("  %-8s %2d |", ggml_type_name(type), nt);
            long long bad[NM] = {};
            for (int m = 0; m < NM; ++m) {
                if (m >= 7 && on_ggml[m][nt]) { printf(" %-24s |", "AVX-512 unavailable"); continue; }
                std::vector<float> o((size_t) NT * n_out, -1.0f);
                float * op[NT];
                for (int t = 0; t < NT; ++t) { op[t] = o.data() + (size_t) t * n_out; }
                if (!run(m, q.data(), nt, op)) { printf(" %-24s |", "no kernel"); bad[m] = -1; continue; }
                for (int t = 0; t < nt; ++t) {
                    for (int64_t r = 0; r < n_out; ++r) {
                        const float got = op[t][r], expected = ref[(size_t) t * n_out + r];
                        bad[m] += m >= 7 ? (!std::isfinite(got) || fabsf(got - expected) > 1e-5f + 2e-5f * fabsf(expected))
                                         : memcmp(&got, &expected, sizeof(float)) != 0;
                    }
                }
                if (check_only) { printf(" %6lld differences       |", bad[m]); continue; }
                double us[2];
                for (int hot = 0; hot < 2; ++hot) {
                    const int reps = hot ? 200 : copies;
                    std::atomic<int> ready{0};
                    std::atomic<bool> go{false};
                    std::vector<double> secs(nth);
                    std::vector<std::thread> ths;
                    for (size_t th = 0; th < nth; ++th) {
                        ths.emplace_back([&, th] {
                            pin_self(cpus[th]);
                            std::vector<float> ob((size_t) NT * n_out);
                            float * obp[NT];
                            for (int t = 0; t < NT; ++t) { obp[t] = ob.data() + (size_t) t * n_out; }
                            run(m, ex[th][0], nt, obp);
                            ready.fetch_add(1);
                            while (!go.load()) {}
                            const auto t0 = clk::now();
                            for (int i = 0; i < reps; ++i) { run(m, ex[th][hot ? 0 : i % copies], nt, obp); }
                            secs[th] = std::chrono::duration<double>(clk::now() - t0).count();
                        });
                    }
                    while (ready.load() < (int) nth) {}
                    go.store(true);
                    for (auto & x : ths) { x.join(); }
                    us[hot] = *std::max_element(secs.begin(), secs.end()) / reps * 1e6;
                }
                char buf[64];
                snprintf(buf, sizeof(buf), "%6.1f/%6.1f %5.1f", us[0], us[1], nth * bytes / us[0] / 1e3);
                printf(" %-24s |", buf);
            }
            for (int m = 1; m < NM; ++m) {
                printf(" %lld", bad[m]);
                bad_total += bad[m] < 0 ? 1 : bad[m];
            }
            printf("\n");
            fflush(stdout);
        }
        for (auto & v : ex) { for (auto * d : v) { free(d); } }
    }
    printf("  (us per expert projection of %lld x %lld; GB/s = cold weight bytes read by all threads)\n",
           (long long) n_out, (long long) n_in);
    printf("parity: %s (%lld outputs fail byte/AVX-512 numeric parity)\n", bad_total == 0 ? "PASS" : "FAIL", bad_total);
    return bad_total == 0 ? 0 : 1;
}
