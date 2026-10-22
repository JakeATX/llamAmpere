// fork STRATAGAP: parity and speed of the host pool's IQ row kernels (moe-iq-rows.h, GGML_MOE_CPU_SPLIT_ROWS)
// against ggml-cpu's vec_dot, on Qwen3.8 Flash-Next expert shapes (2560 in x 640 out).
//
//   test-moe-iq-rows <cpu,cpu,...> [copies, default 256]
//
// Parity: for every IQ up/gate type and nt = 1..4 tokens, every output must have the same bits as ggml's vec_dot
// (the kernels do the same integer work in the same lanes and the same float steps). Exit code 1 on any difference.
// Speed: one pinned thread per listed cpu, each with its own `copies` experts (cold: they rotate, far more than the
// last-level cache, as a decode's missed experts do; hot: one expert repeats). Prints us per expert projection and
// the summed GB/s of weights read, for ggml (vec_dot once per row per token) and the row kernels with prefetch off
// and at 2048 B.
#include "../ggml/src/ggml-cuda/moe-cpu-split.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    const int NT = 4;
    const ggml_type types[] = { GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS };
    const int pfs[] = { 0, 2048 };
    long long bad_total = 0;
    printf("threads %zu, %d expert copies per thread\n", cpus.size(), copies);
    printf("  %-8s nt | %-26s | %-26s | %-26s | bits differing (rows pf0 / pf2048)\n", "type", "ggml cold/hot us, GB/s",
           "rows pf 0", "rows pf 2048");
    for (ggml_type type : types) {
        const auto * tr = ggml_get_type_traits_cpu(type);
        const size_t row_bytes = ggml_row_size(type, n_in);
        const size_t bytes = row_bytes * (size_t) n_out;
        std::vector<float> w((size_t) n_in * n_out), imat((size_t) n_in, 1.0f);
        for (auto & v : w) { v = 0.05f * frand(); }
        std::vector<uint8_t> q(bytes);
        ggml_quantize_chunk(type, w.data(), q.data(), 0, n_out, n_in, imat.data());
        const size_t act_bytes = ggml_row_size(tr->vec_dot_type, n_in);
        std::vector<uint8_t> act(NT * act_bytes);
        for (int t = 0; t < NT; ++t) {
            std::vector<float> x(n_in);
            for (auto & v : x) { v = frand(); }
            ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), act.data() + t * act_bytes, n_in);
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
        // m = 0: ggml; m = 1, 2: row kernels with prefetch pfs[m - 1]
        auto run = [&](int m, const uint8_t * wp, int nt, float * const * out) {
            if (m == 0) {
                for (int t = 0; t < nt; ++t) {
                    for (int64_t r = 0; r < n_out; ++r) {
                        tr->vec_dot((int) n_in, &out[t][r], 0, wp + r * row_bytes, 0, actp[t], 0, 1);
                    }
                }
                return true;
            }
            return moe_cs_iq_rows((int) type, nt, wp, row_bytes, 0, n_out, n_in, actp, out, pfs[m - 1]);
        };
        for (int nt = 1; nt <= NT; ++nt) {
            printf("  %-8s %2d |", ggml_type_name(type), nt);
            long long bad[3] = { 0, 0, 0 };
            for (int m = 0; m < 3; ++m) {
                std::vector<float> o((size_t) NT * n_out, -1.0f);
                float * op[NT];
                for (int t = 0; t < NT; ++t) { op[t] = o.data() + (size_t) t * n_out; }
                if (!run(m, q.data(), nt, op)) { printf(" %-26s |", "no kernel"); bad[m] = -1; continue; }
                for (int t = 0; t < nt; ++t) {
                    for (int64_t r = 0; r < n_out; ++r) {
                        bad[m] += memcmp(&op[t][r], &ref[(size_t) t * n_out + r], sizeof(float)) != 0;
                    }
                }
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
                snprintf(buf, sizeof(buf), "%6.1f/%6.1f %5.1f GB/s", us[0], us[1], nth * bytes / us[0] / 1e3);
                printf(" %-26s |", buf);
            }
            printf(" %lld / %lld\n", bad[1], bad[2]);
            bad_total += std::max(0LL, bad[1]) + std::max(0LL, bad[2]) + (bad[1] < 0 || bad[2] < 0 ? 1 : 0);
            fflush(stdout);
        }
        for (auto & v : ex) { for (auto * d : v) { free(d); } }
    }
    printf("  (us per expert projection of %lld x %lld; GB/s = cold weight bytes read by all threads)\n",
           (long long) n_out, (long long) n_in);
    printf("parity: %s (%lld outputs differ from ggml's bits)\n", bad_total == 0 ? "PASS" : "FAIL", bad_total);
    return bad_total == 0 ? 0 : 1;
}
