#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#include "exl3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static bool check_codec() {
    std::mt19937 rng(123);
    uint32_t words[64];
    uint16_t states[256];
    float decoded[256], reference[256];
    ggml_context * ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    for (int bits = 2; bits <= 8; ++bits) {
        // Exhaust the codebook at a wrapping window and check every coordinate on random tiles.
        for (int code = 0; code < 65536; ++code) {
            for (int i = 0; i < 8 * bits; ++i) { words[i] = rng(); }
            const int start = (257 * bits - 16) % (256 * bits);
            for (int i = 0; i < 16; ++i) {
                const int bit = (start + i) % (256 * bits);
                const uint32_t mask = uint32_t(1) << (31 - bit % 32);
                words[bit / 32] = (words[bit / 32] & ~mask) | (((code >> (15 - i)) & 1) ? mask : 0);
            }
            if (!ggml_cpu_exl3_decode_tile(words, bits, states, decoded)) {
                fprintf(stderr, "SKIP: this CPU backend was built without AVX2/FMA/F16C\n");
                ggml_free(ctx);
                return false;
            }
            ggml_exl3_dequantize_row_group(words, 16, 16, bits, 0, reference);
            for (int t = 0; t < 256; ++t) {
                const int lane = t / 8, j = t % 8;
                const int r = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = 2 * (lane >> 3) + ((lane >> 2) & 1) + 8 * (j >> 2);
                uint16_t state = 0;
                for (int bit = 0; bit < 16; ++bit) {
                    const int pos = ((t + 257) * bits - 16 + bit) % (256 * bits);
                    state = (state << 1) | ((words[pos / 32] >> (31 - pos % 32)) & 1);
                }
                GGML_ASSERT(states[r * 16 + c] == state);
                GGML_ASSERT(memcmp(decoded + r * 16 + c, reference + c * 16 + r, sizeof(float)) == 0);
            }
            GGML_ASSERT(states[0] == code);
        }
        printf("codec bits=%d: all 65536 states, all tile coordinates and rounded weights exact\n", bits);
    }
    ggml_free(ctx);
    return true;
}

static double worst_l2 = 0, worst_max = 0;

static void compare(const std::vector<float> & got, const std::vector<float> & ref) {
    double error = 0, norm = 0, max_error = 0, max_ref = 0;
    GGML_ASSERT(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i) {
        GGML_ASSERT(std::isfinite(got[i]) && std::isfinite(ref[i]));
        const double d = double(got[i]) - ref[i];
        error += d * d;
        norm += double(ref[i]) * ref[i];
        max_error = std::max(max_error, std::abs(d));
        max_ref = std::max(max_ref, std::abs(double(ref[i])));
    }
    const double l2 = std::sqrt(error) / std::max(std::sqrt(norm), 1e-12);
    const double relative_max = max_error / std::max(1.0, max_ref);
    worst_l2 = std::max(worst_l2, l2);
    worst_max = std::max(worst_max, relative_max);
    if (l2 > 1e-4 || relative_max > 1e-3) {
        fprintf(stderr, "relative_l2=%.9g normalized_max=%.9g\n", l2, relative_max);
        GGML_ABORT("EXL3 CPU accuracy ceiling exceeded");
    }
}

struct fixture {
    ggml_context * ctx;
    ggml_backend_t cpu;
    ggml_backend_buffer_t buffer;
    ggml_threadpool_t pool;
    ggml_cgraph * graph;
    ggml_tensor * w;
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * y;
    ggml_tensor * suh = nullptr;
    ggml_tensor * svh = nullptr;
    int threads = 0;

    fixture(int bits, int64_t k, int64_t n, int64_t tokens, int routes, bool indexed, bool glue, int pattern = 1, bool side_half = false) {
        ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
        cpu = ggml_backend_cpu_init();
        const ggml_type type = (ggml_type) (GGML_TYPE_EXL3_2 + bits - 2);
        const int experts = indexed ? 5 : 1;
        w = ggml_new_tensor_3d(ctx, type, k, n, experts);
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, indexed ? routes : tokens, indexed ? tokens : 1);
        ids = indexed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, routes, tokens) : nullptr;
        y = indexed ? ggml_mul_mat_id(ctx, w, x, ids) : ggml_mul_mat(ctx, w, x);
        if (glue) {
            suh = ggml_new_tensor_2d(ctx, side_half ? GGML_TYPE_F16 : GGML_TYPE_F32, k, experts);
            svh = ggml_new_tensor_2d(ctx, side_half ? GGML_TYPE_F16 : GGML_TYPE_F32, n, experts);
            y->src[indexed ? 3 : 2] = suh;
            y->src[indexed ? 4 : 3] = svh;
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        GGML_ASSERT(buffer);
        std::mt19937 rng(456);
        auto * words = (uint32_t *) w->data;
        for (size_t i = 0; i < ggml_nbytes(w) / 4; ++i) { words[i] = rng(); }
        for (int64_t i = 0; i < ggml_nelements(x); ++i) {
            float v = float(int(rng() % 2001) - 1000) / 1000;
            if (pattern == 0) { v = 0; }
            if (pattern == 2 && i % 131) { v = 0; }
            if (pattern == 3) { v = i % 2 ? -1.0f : 1.0f; }
            if (pattern == 4) { v *= 1e8f; }
            ((float *) x->data)[i] = v;
        }
        if (indexed) {
            for (int64_t t = 0; t < tokens; ++t) {
                for (int r = 0; r < routes; ++r) { ((int32_t *) ids->data)[t * routes + r] = (r + t / 3) % experts; }
            }
        }
        for (ggml_tensor * scale : {suh, svh}) {
            if (!scale) { continue; }
            for (int64_t i = 0; i < ggml_nelements(scale); ++i) {
                const float value = float(int(rng() % 2001) - 1000) / 3000 + 0.5f + 0.1f * (i / scale->ne[0]);
                if (side_half) { ((ggml_fp16_t *) scale->data)[i] = ggml_fp32_to_fp16(value); }
                else { ((float *) scale->data)[i] = value; }
            }
        }
        pool = nullptr;
    }

    ~fixture() {
        ggml_backend_free(cpu);
        ggml_threadpool_free(pool);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    void compute(int nth, bool reference) {
        if (threads != nth) {
            ggml_backend_cpu_set_threadpool(cpu, nullptr);
            ggml_threadpool_free(pool);
            ggml_threadpool_params tp = ggml_threadpool_params_default(nth);
            pool = ggml_threadpool_new(&tp);
            ggml_backend_cpu_set_threadpool(cpu, pool);
            ggml_backend_cpu_set_n_threads(cpu, nth);
            threads = nth;
        }
        ggml_backend_cpu_set_use_ref(cpu, reference);
        GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    }

    std::vector<float> output() {
        const float * p = (const float *) y->data;
        return {p, p + ggml_nelements(y)};
    }
};

static void check_matmul() {
    env("GGML_CPU_EXL3", "1");
    for (int bits = 2; bits <= 8; ++bits) {
        for (int tokens : {1, 2, 3, 4, 8, 9, 16, 21, 64}) {
            const int routes = tokens == 1 ? 8 : tokens % 3 == 0 ? 4 : tokens % 2 == 0 ? 2 : 1;
            for (bool indexed : {false, true}) {
                fixture f(bits, 768, 256, tokens, routes, indexed, true, tokens % 5);
                f.compute(1, true);
                const auto ref = f.output();
                for (const char * piece : {"128", "256", "512"}) {
                    env("GGML_CPU_EXL3_PIECE", piece);
                    f.compute(1, false);
                    const auto one = f.output();
                    compare(one, ref);
                    for (int nth : {4, 8, 16}) {
                        f.compute(nth, false);
                        const auto many = f.output();
                        GGML_ASSERT(memcmp(one.data(), many.data(), one.size() * sizeof(float)) == 0);
                    }
                }
            }
        }
        fixture tail(bits, 768, 272, 3, 2, true, false);
        tail.compute(4, true);
        const auto ref = tail.output();
        tail.compute(8, false);
        compare(tail.output(), ref);
    }
    printf("matmul: all bits, dense/indexed, pieces 128/256/512, workers 1/4/8/16 exact across workers\n");

    fixture f(4, 768, 256, 4, 4, true, true);
    f.compute(1, true);
    const auto ref = f.output();
    std::vector<uint8_t> scratch(ggml_cpu_exl3_work_size(f.y, 1));
    ggml_compute_params params = {0, 1, scratch.size(), scratch.data(), f.pool, false};
    GGML_ASSERT(ggml_cpu_exl3_mul_mat(&params, f.y));
    compare(f.output(), ref);
    printf("direct dispatch: optimized kernel returned true\n");
    const auto batch_ref = f.output();
    for (const char * batch : {"1", "2", "3"}) {
        env("GGML_CPU_EXL3_BATCH", batch);
        f.compute(4, false);
        const auto got = f.output();
        GGML_ASSERT(memcmp(got.data(), batch_ref.data(), got.size() * sizeof(float)) == 0);
    }

    fixture dense(4, 768, 256, 1, 1, false, true);
    for (bool broadcast : {false, true}) {
        f.x->ne[1] = broadcast ? 1 : 4;
        f.compute(4, false);
        const auto indexed = f.output();
        for (int token = 0; token < 4; ++token) {
            for (int route = 0; route < 4; ++route) {
                const int expert = ((const int32_t *) f.ids->data)[token * 4 + route];
                memcpy(dense.w->data, (const char *) f.w->data + expert * f.w->nb[2], ggml_nbytes(dense.w));
                memcpy(dense.x->data, (const char *) f.x->data + token * f.x->nb[2] + (broadcast ? 0 : route * f.x->nb[1]), ggml_nbytes(dense.x));
                memcpy(dense.suh->data, (const char *) f.suh->data + expert * f.suh->nb[1], ggml_nbytes(dense.suh));
                memcpy(dense.svh->data, (const char *) f.svh->data + expert * f.svh->nb[1], ggml_nbytes(dense.svh));
                dense.compute(1, true);
                const float * begin = indexed.data() + (token * 4 + route) * 256;
                compare(std::vector<float>(begin, begin + 256), dense.output());
            }
        }
    }
    f.x->ne[1] = 4;
    printf("logical expert IDs and broadcast: checked against separate dense reference graphs\n");

    for (bool indexed : {false, true}) {
        fixture half(4, 512, 256, 9, 2, indexed, true, 1, true);
        fixture wide(4, 512, 256, 9, 2, indexed, true);
        for (const auto & pair : {std::make_pair(half.suh, wide.suh), std::make_pair(half.svh, wide.svh)}) {
            for (int64_t i = 0; i < ggml_nelements(pair.first); ++i) {
                ((float *) pair.second->data)[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) pair.first->data)[i]);
            }
        }
        wide.compute(4, true);
        half.compute(4, false);
        compare(half.output(), wide.output());
        half.compute(4, true);
        compare(half.output(), wide.output());
        for (int side = 0; side < 2; ++side) {
            ggml_tensor * saved_side = half.y->src[(indexed ? 3 : 2) + side];
            half.y->src[(indexed ? 3 : 2) + side] = side == 0 ? wide.suh : wide.svh;
            half.compute(4, true);
            compare(half.output(), wide.output());
            half.compute(4, false);
            compare(half.output(), wide.output());
            half.y->src[(indexed ? 3 : 2) + side] = saved_side;
        }
    }
    fixture concurrent_a(3, 1024, 512, 9, 4, true, true);
    fixture concurrent_b(7, 768, 256, 8, 2, true, false);
    concurrent_a.compute(4, true);
    concurrent_b.compute(4, true);
    const auto ref_a = concurrent_a.output(), ref_b = concurrent_b.output();
    std::thread a([&] { for (int i = 0; i < 8; ++i) { concurrent_a.compute(4, false); } });
    std::thread b([&] { for (int i = 0; i < 8; ++i) { concurrent_b.compute(4, false); } });
    a.join();
    b.join();
    compare(concurrent_a.output(), ref_a);
    compare(concurrent_b.output(), ref_b);
    const int64_t saved = f.svh->ne[0];
    f.svh->ne[0] = saved - 1;
    GGML_ASSERT(!ggml_cpu_exl3_supported(f.y));
    f.svh->ne[0] = saved;
    for (ggml_tensor * t : {f.x, f.y, f.suh, f.svh, f.w}) {
        const int dim = t == f.w ? 2 : 1;
        const size_t stride = t->nb[dim];
        t->nb[dim] = stride + 1;
        GGML_ASSERT(!ggml_cpu_exl3_supported(f.y));
        t->nb[dim] = SIZE_MAX - 7;
        GGML_ASSERT(!ggml_cpu_exl3_supported(f.y));
        t->nb[dim] = stride;
        t->view_offs = 1;
        GGML_ASSERT(!ggml_cpu_exl3_supported(f.y));
        t->view_offs = 0;
    }
    GGML_ASSERT(ggml_cpu_exl3_supported(f.y));
    env("GGML_CPU_EXL3", "0");
    GGML_ASSERT(!ggml_backend_dev_supports_op(ggml_backend_get_device(f.cpu), f.y));
    GGML_ASSERT(!ggml_cpu_exl3_mul_mat(&params, f.y));
    env("GGML_CPU_EXL3", "1");

    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    ggml_tensor * large_w = ggml_new_tensor_2d(ctx, GGML_TYPE_EXL3_4, 256, 256);
    ggml_tensor * large_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 65536);
    ggml_tensor * large_y = ggml_mul_mat(ctx, large_w, large_x);
    GGML_ASSERT(!ggml_cpu_exl3_mul_mat(&params, large_y));
    ggml_free(ctx);
    printf("F16 sides and simultaneous requests: OK\n");
    printf("worst relative_l2=%.9g normalized_max=%.9g\n", worst_l2, worst_max);
}

static void bench(int nth, bool down) {
    env("GGML_CPU_EXL3", "1");
    const int64_t k = down ? 17408 : 5120;
    const int64_t n = down ? 5120 : 17408;
    const char * batch = getenv("GGML_CPU_EXL3_BATCH");
    if (!batch) { batch = "3"; }
    printf("bits,K,N,T,threads,piece,batch,ref_us,fast_us,speedup,ref_min_us,ref_max_us,fast_min_us,fast_max_us\n");
    for (int bits : {3, 4}) {
        for (int tokens : {1, 3, 9}) {
            fixture f(bits, k, n, tokens, 1, false, true);
            for (const char * piece : {"128", "256", "512"}) {
                env("GGML_CPU_EXL3_PIECE", piece);
                std::vector<double> ref, fast;
                for (int rep = -1; rep < 5; ++rep) {
                    for (bool reference : {true, false}) {
                        const int64_t begin = ggml_time_us();
                        f.compute(nth, reference);
                        const double time = ggml_time_us() - begin;
                        if (rep >= 0) { (reference ? ref : fast).push_back(time); }
                    }
                }
                std::sort(ref.begin(), ref.end());
                std::sort(fast.begin(), fast.end());
                printf("%d,%lld,%lld,%d,%d,%s,%s,%.0f,%.0f,%.3f,%.0f,%.0f,%.0f,%.0f\n",
                    bits, (long long) k, (long long) n, tokens, nth, piece, batch,
                    ref[2], fast[2], ref[2] / fast[2], ref.front(), ref.back(), fast.front(), fast.back());
                fflush(stdout);
            }
        }
    }
}

static void bench_stream(int nth) {
    env("GGML_CPU_EXL3", "1");
    env("GGML_CPU_EXL3_PIECE", "256");
    printf("bits,matrices,compressed_MiB,threads,ref_us_per_matrix,fast_us_per_matrix,speedup,fast_GB_s,ref_min_us,ref_max_us,fast_min_us,fast_max_us\n");
    for (int bits : {3, 4}) {
        std::vector<std::unique_ptr<fixture>> bank;
        for (int i = 0; i < 16; ++i) { bank.emplace_back(new fixture(bits, 5120, 17408, 1, 1, false, true)); }
        const double bytes = 16.0 * ggml_nbytes(bank[0]->w);
        std::vector<double> ref, fast;
        for (int rep = -1; rep < 5; ++rep) {
            for (bool reference : {true, false}) {
                const int64_t begin = ggml_time_us();
                for (auto & f : bank) { f->compute(nth, reference); }
                const double time = double(ggml_time_us() - begin) / bank.size();
                if (rep >= 0) { (reference ? ref : fast).push_back(time); }
            }
        }
        std::sort(ref.begin(), ref.end());
        std::sort(fast.begin(), fast.end());
        printf("%d,16,%.1f,%d,%.0f,%.0f,%.3f,%.3f,%.0f,%.0f,%.0f,%.0f\n", bits, bytes / (1024*1024), nth,
            ref[2], fast[2], ref[2] / fast[2], bytes / (16 * fast[2] * 1000), ref.front(), ref.back(), fast.front(), fast.back());
        fflush(stdout);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench(argc > 2 ? std::max(1, atoi(argv[2])) : 8, argc > 3 && strcmp(argv[3], "down") == 0);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--bench-stream") == 0) {
        bench_stream(argc > 2 ? std::max(1, atoi(argv[2])) : 8);
        return 0;
    }
    if (!check_codec()) { return 77; }
    check_matmul();
    return 0;
}
