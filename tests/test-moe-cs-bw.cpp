// fork STRATAGAP: how fast the host expert pool (GGML_MOE_CPU_SPLIT) runs against what the RAM can deliver.
//
//   test-moe-cs-bw <cpu,cpu,...> [RAM MiB] [experts] [check|time] [output file]
//
// Part 1, RAM read ceiling: threads pinned to the first 1..N cpus of the list each stream their own slice of a
// large buffer (AVX2 loads); prints GB/s for each thread count.
// Part 2, the pool itself: layers shaped like Qwen3.8 Flash-Next experts (2560 x 640, the model's up/gate and down
// types), 128 experts per layer, so a rotation reads from RAM (cold) as a decode's missed experts do; the same
// jobs on a fixed set of experts that stays in cache (hot) give the compute limit. One route per expert.
// The pool takes its thread count and pinning from the usual GGML_MOE_CPU_SPLIT_* / GGML_CPU_EXPERT_PIN variables.
#include "../ggml/src/ggml-cuda/moe-cpu-split.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

using clk = std::chrono::steady_clock;

static void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

__attribute__((target("avx2"))) static uint64_t read_slice(const uint8_t * p, size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        a0 = _mm256_add_epi64(a0, _mm256_load_si256((const __m256i *) (p + i)));
        a1 = _mm256_add_epi64(a1, _mm256_load_si256((const __m256i *) (p + i + 32)));
        a2 = _mm256_add_epi64(a2, _mm256_load_si256((const __m256i *) (p + i + 64)));
        a3 = _mm256_add_epi64(a3, _mm256_load_si256((const __m256i *) (p + i + 96)));
    }
    const __m256i s = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    alignas(32) uint64_t v[4];
    _mm256_store_si256((__m256i *) v, s);
    return v[0] + v[1] + v[2] + v[3];
}

static void ram_ceiling(const std::vector<int> & cpus, size_t bytes) {
    uint8_t * buf = (uint8_t *) aligned_alloc(4096, bytes);
    for (size_t i = 0; i < bytes; i += 4096) { buf[i] = (uint8_t) i; }
    memset(buf, 1, bytes);
    printf("RAM read ceiling (%.0f MiB buffer, one pinned thread per listed cpu):\n", bytes / 1048576.0);
    for (size_t t = 1; t <= cpus.size(); ++t) {
        double best = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<double> secs(t);
            std::vector<std::thread> th;
            std::atomic<uint64_t> sink{0};
            const size_t slice = (bytes / t) & ~(size_t) 4095;
            for (size_t i = 0; i < t; ++i) {
                th.emplace_back([&, i] {
                    pin_self(cpus[i]);
                    ready.fetch_add(1);
                    while (!go.load()) { _mm_pause(); }
                    const auto t0 = clk::now();
                    sink.fetch_add(read_slice(buf + i * slice, slice));
                    secs[i] = std::chrono::duration<double>(clk::now() - t0).count();
                });
            }
            while (ready.load() < (int) t) { _mm_pause(); }
            go.store(true);
            for (auto & x : th) { x.join(); }
            const double worst = *std::max_element(secs.begin(), secs.end());
            best = std::max(best, (double) slice * t / worst / 1e9);
        }
        printf("  %2zu threads (last cpu %2d): %6.1f GB/s\n", t, cpus[t - 1], best);
    }
    free(buf);
}

struct proj {
    ggml_type type;
    int64_t n_in, n_out;
    size_t row_bytes, bytes;
    std::vector<uint8_t *> expert;   // one copy per expert slot
    std::vector<const void *> src;
};

static uint64_t g_rng = 88172645463325252ull;
static float frand() {   // roughly normal: sum of 4 uniforms
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
        s += (float) (g_rng >> 40) / (float) (1ull << 24) - 0.5f;
    }
    return s;
}

static void make_proj(proj & p, ggml_type type, int64_t n_in, int64_t n_out, int n_expert) {
    p.type = type; p.n_in = n_in; p.n_out = n_out;
    p.row_bytes = ggml_row_size(type, n_in);
    p.bytes = p.row_bytes * (size_t) n_out;
    std::vector<float> w((size_t) n_in * n_out);
    for (auto & v : w) { v = 0.05f * frand(); }
    std::vector<float> imat((size_t) n_in, 1.0f);
    std::vector<uint8_t> q(p.bytes);
    ggml_quantize_chunk(type, w.data(), q.data(), 0, n_out, n_in, imat.data());
    for (int e = 0; e < n_expert; ++e) {
        uint8_t * d = (uint8_t *) aligned_alloc(4096, (p.bytes + 4095) & ~(size_t) 4095);
        memcpy(d, q.data(), p.bytes);
        p.expert.push_back(d);
        p.src.push_back(d);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <cpu,cpu,...> [ram MiB, default 2048] [experts, default 128] [check|time] [output file]\n", argv[0]);
        return 2;
    }
    std::vector<int> cpus;
    for (const char * c = argv[1]; *c;) {
        cpus.push_back(atoi(c));
        while (*c && *c != ',') { ++c; }
        if (*c == ',') { ++c; }
    }
    const size_t ram_mb = argc > 2 ? (size_t) atol(argv[2]) : 2048;
    FILE * output = argc > 5 ? fopen(argv[5], "wb") : nullptr;
    if (argc > 5 && !output) { perror("output file"); return 1; }
    if (ram_mb) { ram_ceiling(cpus, ram_mb << 20); }

    // Part 2: the pool. The calling thread plays the card's driver thread (pinned to cpu 0, which the pool skips).
    pin_self(0);
    ggml_cpu_init();
    const int64_t n_embd = 2560, n_ff = 640;
    const int n_expert = argc > 3 ? std::max(4, atoi(argv[3])) : 128;
    struct layer_spec { ggml_type up, down; };
    const layer_spec specs[] = {
        { GGML_TYPE_IQ3_S,   GGML_TYPE_Q2_0 },  { GGML_TYPE_IQ2_S,   GGML_TYPE_Q2_0 },
        { GGML_TYPE_IQ2_XS,  GGML_TYPE_Q2_0 },  { GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0 },
        { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL }, { GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_NL },
    };
    const int n_layers = (int) (sizeof(specs) / sizeof(specs[0]));
    moe_cs_job * job = (moe_cs_job *) aligned_alloc(128, (sizeof(moe_cs_job) + 127) & ~(size_t) 127);
    memset((void *) job, 0, sizeof(moe_cs_job));
    const size_t x_stride = (size_t) n_embd, out_stride = (size_t) n_embd;
    std::vector<float> x(MOE_CS_MAX_ROUTES * x_stride), out(MOE_CS_MAX_ROUTES * out_stride);
    for (size_t i = 0; i < x.size(); ++i) { x[i] = 0.02f * (float) ((int) ((i * 7919) % 101) - 50); }
    if (!moe_cs_start(job, x.data(), x_stride, out.data(), out_stride, n_ff, n_embd)) {
        fprintf(stderr, "pool did not start\n");
        return 1;
    }
    std::vector<proj> P((size_t) n_layers * 3);
    for (int l = 0; l < n_layers; ++l) {
        make_proj(P[3 * l + 0], specs[l].up, n_embd, n_ff, n_expert);
        make_proj(P[3 * l + 1], specs[l].up, n_embd, n_ff, n_expert);
        make_proj(P[3 * l + 2], specs[l].down, n_ff, n_embd, n_expert);
        moe_cs_role roles[3];
        for (int k = 0; k < 3; ++k) {
            const proj & p = P[3 * l + k];
            roles[k].type = p.type; roles[k].n_in = p.n_in; roles[k].n_out = p.n_out; roles[k].n_expert = n_expert;
            roles[k].row_bytes = p.row_bytes; roles[k].src = p.src.data();
        }
        if (!moe_cs_register_layer(l, roles)) {
            fprintf(stderr, "layer %d (%s/%s) not accepted by the pool\n", l, ggml_type_name(specs[l].up), ggml_type_name(specs[l].down));
            return 1;
        }
    }
    // Post card-shaped jobs without CUDA: repeated token IDs span different experts.
    double warm_ns, warm_ph[4];
    const int warm_expert = 0;
    moe_cs_bench(0, &warm_expert, 1, 1, 1, &warm_ns, warm_ph);
    const bool check = argc > 4 && strcmp(argv[4], "check") == 0;
    double max_error = 0.0;
    unsigned long long seq = 0, hash = 14695981039346656037ull;
    for (int l = 0; l < n_layers; ++l) {
        for (int pass = 0; pass < 16; ++pass) {
            const int nt = 1 + pass % 8;
            job->layer = l;
            job->n_exp = 4;
            job->n_route = 4 * nt;
            for (int e = 0; e < 4; ++e) {
                job->expert[e] = e;
                job->first[e] = e * nt;
                for (int t = 0; t < nt; ++t) {
                    const int q = e * nt + t;
                    job->input_row[q] = pass < 8 ? t : q;
                    for (int64_t r = 0; r < n_embd; ++r) {
                        x[(size_t) q * x_stride + r] = 0.02f * (float) ((int) (((job->input_row[q] + 1) * 131 + r * 7919) % 101) - 50);
                    }
                }
            }
            job->first[4] = 4 * nt;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            job->seq = ++seq;
            const auto deadline = clk::now() + std::chrono::seconds(30);
            while (job->done != seq) {
                if (clk::now() > deadline) { fprintf(stderr, "pool parity timeout\n"); std::_Exit(1); }
                _mm_pause();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (check) {
                const auto * ug = ggml_get_type_traits_cpu(specs[l].up);
                const auto * dn = ggml_get_type_traits_cpu(specs[l].down);
                std::vector<uint8_t> xq(ggml_row_size(ug->vec_dot_type, n_embd));
                std::vector<uint8_t> hq(ggml_row_size(dn->vec_dot_type, n_ff));
                std::vector<float> hidden(n_ff);
                for (int q = 0; q < 4 * nt; ++q) {
                    const int expert = q / nt;
                    ggml_get_type_traits_cpu(ug->vec_dot_type)->from_float(x.data() + (size_t) q * x_stride, xq.data(), n_embd);
                    for (int64_t row = 0; row < n_ff; ++row) {
                        float u, g;
                        ug->vec_dot((int) n_embd, &u, 0, P[3*l].expert[expert] + row * P[3*l].row_bytes, 0, xq.data(), 0, 1);
                        ug->vec_dot((int) n_embd, &g, 0, P[3*l+1].expert[expert] + row * P[3*l+1].row_bytes, 0, xq.data(), 0, 1);
                        hidden[row] = g / (1.0f + expf(-g)) * u;
                    }
                    ggml_get_type_traits_cpu(dn->vec_dot_type)->from_float(hidden.data(), hq.data(), n_ff);
                    double err2 = 0.0, ref2 = 0.0;
                    for (int64_t row = 0; row < n_embd; ++row) {
                        float ref;
                        dn->vec_dot((int) n_ff, &ref, 0, P[3*l+2].expert[expert] + row * P[3*l+2].row_bytes, 0, hq.data(), 0, 1);
                        const float got = out[(size_t) q * out_stride + row];
                        const double err = (double) got - ref;
                        err2 += err * err;
                        ref2 += (double) ref * ref;
                    }
                    const double relative = sqrt(err2 / std::max(ref2, 1e-30));
                    max_error = std::max(max_error, relative);
                    // A changed gate reduction can move hidden activation quantization across a boundary.
                    if (!std::isfinite(relative) || relative > 0.005) {
                        fprintf(stderr, "pool reference FAIL layer=%d width=%d route=%d relative=%g\n", l, nt, q, relative);
                        std::_Exit(1);
                    }
                }
            }
            for (int q = 0; q < 4 * nt; ++q) {
                for (int64_t r = 0; r < n_embd; ++r) {
                    const float v = out[(size_t) q * out_stride + r];
                    if (!std::isfinite(v)) { fprintf(stderr, "non-finite pool output\n"); std::_Exit(1); }
                    const unsigned char * b = (const unsigned char *) &v;
                    for (size_t i = 0; i < sizeof(v); ++i) { hash = (hash ^ b[i]) * 1099511628211ull; }
                }
            }
        }
    }
    printf("pool parity: 96 jobs, widths 1..8, four experts, shared and distinct inputs, output hash %016llx\n", hash);
    fflush(stdout);
    if (check) { printf("pool reference PASS: max relative L2 error %.9g\n", max_error); fflush(stdout); std::_Exit(0); }
    printf("\npool: %d threads; one route per expert; 128 experts per layer rotate (cold) or the same experts repeat (hot)\n",
           moe_cs_threads());
    printf("  %-8s %-7s n_exp | cold: us/job  up+gate us (GB/s)  down us (GB/s) | hot: us/job  up+gate us (GB/s)  down us (GB/s)\n",
           "up/gate", "down");
    std::vector<int> all(n_expert);
    for (int e = 0; e < n_expert; ++e) { all[e] = e; }
    for (int l = 0; l < n_layers; ++l) {
        const double ug_bytes = 2.0 * P[3 * l].bytes, dn_bytes = (double) P[3 * l + 2].bytes;
        double ns, ph[4];
        moe_cs_bench(l, all.data(), n_expert, 4, n_expert / 4, &ns, ph);   // first touch
        for (int ne : { 1, 2, 3, 4, 8 }) {
            double cns, cph[4], hns, hph[4];
            moe_cs_bench(l, all.data(), n_expert, ne, 400, &cns, cph);
            if (output && fwrite(out.data(), sizeof(float), (size_t) ne * out_stride, output) != (size_t) ne * out_stride) {
                perror("cold output"); return 1;
            }
            moe_cs_bench(l, all.data(), ne, ne, 400, &hns, hph);
            if (output && fwrite(out.data(), sizeof(float), (size_t) ne * out_stride, output) != (size_t) ne * out_stride) {
                perror("hot output"); return 1;
            }
            if (ne == 8) {   // output check: compare this line between runs with and without a switch
                double sum = 0.0, sabs = 0.0;
                for (int q = 0; q < ne; ++q) {
                    for (int64_t r = 0; r < n_embd; ++r) { const double v = out[(size_t) q * out_stride + r]; sum += v; sabs += v < 0 ? -v : v; }
                }
                printf("  check %s/%s: sum %.6e  sum|x| %.6e  out[0] %.6e out[777] %.6e\n", ggml_type_name(specs[l].up),
                       ggml_type_name(specs[l].down), sum, sabs, out[0], out[777]);
            }
            printf("  %-8s %-7s %5d | %10.1f %9.1f (%5.1f) %8.1f (%5.1f) | %10.1f %9.1f (%5.1f) %8.1f (%5.1f)\n",
                   ggml_type_name(specs[l].up), ggml_type_name(specs[l].down), ne,
                   cns / 1e3, cph[1] / 1e3, ne * ug_bytes / cph[1], cph[3] / 1e3, ne * dn_bytes / cph[3],
                   hns / 1e3, hph[1] / 1e3, ne * ug_bytes / hph[1], hph[3] / 1e3, ne * dn_bytes / hph[3]);
        }
    }
    if (output) {
        // Exercise repeated routes and the full job limits using the same publication as the card.
        for (int l = 0; l < n_layers; ++l) {
            for (int ne : { 1, 3, 8, 16 }) {
                for (int nr : { 1, 2, 3, 4 }) {
                    if (ne * nr > MOE_CS_MAX_ROUTES) { continue; }
                    for (int rep = 0; rep < 4; ++rep) {
                        job->layer = l;
                        job->n_exp = ne;
                        job->n_route = ne * nr;
                        for (int i = 0; i < ne; ++i) {
                            job->expert[i] = (rep * ne + i) % n_expert;
                            job->first[i] = i * nr;
                        }
                        job->first[ne] = ne * nr;
                        const auto want = job->seq + 1;
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        job->seq = want;
                        while (job->done != want) { moe_cs_help(); }
                        std::atomic_thread_fence(std::memory_order_acquire);
                        const size_t count = (size_t) ne * nr * out_stride;
                        if (fwrite(out.data(), sizeof(float), count, output) != count) {
                            perror("multi-route output"); return 1;
                        }
                    }
                }
            }
        }
        unsigned long long stats[9];
        moe_cs_stats(stats);
        if (stats[4] != 0) { fprintf(stderr, "pool reported %llu invalid jobs\n", stats[4]); return 1; }
        printf("multi-route jobs: %llu, invalid jobs: %llu\n", stats[0], stats[4]);
    }
    printf("  (projection bytes:");
    for (int l = 0; l < n_layers; ++l) {
        printf(" %s %zu KiB / %s %zu KiB;", ggml_type_name(specs[l].up), P[3 * l].bytes >> 10,
               ggml_type_name(specs[l].down), P[3 * l + 2].bytes >> 10);
    }
    printf(")\n");
    fflush(stdout);
    if (output && fclose(output) != 0) { perror("close output"); return 1; }
    std::_Exit(0);   // pool threads never exit
}
