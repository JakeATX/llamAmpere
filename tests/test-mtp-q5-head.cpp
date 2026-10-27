#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

struct head_case {
    int64_t k;
    int64_t rows;
    int64_t cols;
    bool strided;
};

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
}

static float random_signed(std::mt19937 & rng) {
    return ((int32_t) (rng() & 0xFFFF) - 32768) / 32768.0f;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double kl_divergence(const std::vector<float> & a, const std::vector<float> & b, int64_t rows) {
    double total = 0.0;
    for (size_t col = 0; col < a.size() / rows; ++col) {
        const size_t off = col * rows;
        double ma = -std::numeric_limits<double>::infinity();
        double mb = ma;
        for (int64_t row = 0; row < rows; ++row) {
            ma = std::max(ma, (double) a[off + row]);
            mb = std::max(mb, (double) b[off + row]);
        }
        double za = 0.0;
        double zb = 0.0;
        for (int64_t row = 0; row < rows; ++row) {
            za += exp(a[off + row] - ma);
            zb += exp(b[off + row] - mb);
        }
        const double log_ratio = log(zb) + mb - log(za) - ma;
        for (int64_t row = 0; row < rows; ++row) {
            total += exp(a[off + row] - ma) / za * (a[off + row] - b[off + row] + log_ratio);
        }
    }
    return total / (a.size() / rows);
}

static void compute(ggml_backend_t backend, ggml_cgraph * graph, bool fast) {
    set_env("GGML_CUDA_MTP_Q5_HEAD", fast ? "1" : "0");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
}

static void fill_weight(ggml_tensor * storage, std::mt19937 & rng) {
    // Quantize a varied row bank once; repeat packed rows to bound host memory and setup work.
    const int64_t storage_k = storage->ne[0];
    constexpr int bank_rows = 512;
    std::vector<float> bank(storage_k * bank_rows);
    for (int row = 0; row < bank_rows; ++row) {
        const float scale = std::ldexp(0.03f, row % 9 - 4);
        for (int64_t i = 0; i < storage_k; ++i) {
            bank[row * storage_k + i] = row == 0 ? 0.0f : scale * random_signed(rng);
        }
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_K, storage_k);
    std::vector<uint8_t> bank_q(row_bytes * bank_rows);
    GGML_ASSERT(ggml_quantize_chunk(GGML_TYPE_Q5_K, bank.data(), bank_q.data(), 0, bank_rows, storage_k, nullptr) == bank_q.size());
    std::vector<uint8_t> packed(ggml_nbytes(storage));
    for (int64_t row = 0; row < storage->ne[1]; ++row) {
        const size_t bank_row = (row * 73 + row / bank_rows) % bank_rows;
        memcpy(packed.data() + row * row_bytes, bank_q.data() + bank_row * row_bytes, row_bytes);
    }
    ggml_backend_tensor_set(storage, packed.data(), 0, packed.size());
}

static void fill_activation(std::vector<float> & activation, std::mt19937 & rng, int pattern) {
    for (size_t i = 0; i < activation.size(); ++i) {
        const float value = random_signed(rng);
        switch (pattern) {
            case 0: activation[i] = 0.0f; break;
            case 1: activation[i] = 0.125f * value; break;
            case 2: activation[i] = (i & 1 ? -0.125f : 0.125f) * (1.0f + value * 0.0001f); break;
            case 3: activation[i] = std::ldexp(value, (int) ((i / 32) % 9) - 7); break;
            case 4: activation[i] = ((int) (i % 31) - 15) * 1e-5f; break;
        }
    }
}

static bool run_case(ggml_backend_t backend, const head_case & c) {
    const int64_t storage_k = c.k + (c.strided ? 256 : 0);
    ggml_init_params pw = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    GGML_ASSERT(ctx_w);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q5_K, storage_k, c.rows);
    ggml_tensor * weight = c.strided ? ggml_view_2d(ctx_w, storage, c.k, c.rows, storage->nb[1], 0) : storage;
    ggml_tensor * input = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.k, c.cols);
    ggml_set_input(input);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(buf_w);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(0x5A17B93U + (uint32_t) storage_k);
    fill_weight(storage, rng);

    ggml_init_params pg = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    GGML_ASSERT(ctx);
    ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_supports_op(backend, out));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));

    bool ok = true;
    std::vector<float> activation(c.k * c.cols);
    std::vector<float> original(c.rows * c.cols);
    std::vector<float> candidate(original.size());
    for (int pattern = 0; pattern < 5; ++pattern) {
        fill_activation(activation, rng, pattern);
        ggml_backend_tensor_set(input, activation.data(), 0, ggml_nbytes(input));
        compute(backend, graph, false);
        ggml_backend_tensor_get(out, original.data(), 0, ggml_nbytes(out));
        compute(backend, graph, true);
        ggml_backend_tensor_get(out, candidate.data(), 0, ggml_nbytes(out));

        size_t mismatches = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < original.size(); ++i) {
            if (!std::isfinite(original[i]) || !std::isfinite(candidate[i])) {
                max_abs = std::numeric_limits<double>::infinity();
            } else {
                max_abs = std::max(max_abs, fabs((double) original[i] - candidate[i]));
            }
            if (float_bits(original[i]) != float_bits(candidate[i])) {
                if (mismatches < 4) {
                    fprintf(stderr, "mismatch i=%zu original=%a [%08x] candidate=%a [%08x]\n",
                            i, original[i], float_bits(original[i]), candidate[i], float_bits(candidate[i]));
                }
                ++mismatches;
            }
        }
        const double kld = mismatches ? kl_divergence(original, candidate, c.rows) : 0.0;
        printf("k=%lld rows=%lld cols=%lld strided=%d pattern=%d mismatches=%zu/%zu max_abs=%.12g kld=%.12g\n",
                (long long) c.k, (long long) c.rows, (long long) c.cols, c.strided, pattern,
                mismatches, original.size(), max_abs, kld);
        ok = ok && mismatches == 0 && std::isfinite(max_abs);

        if (pattern == 1 && c.k == 2560 && c.cols == 1 && !c.strided && c.rows >= 65536) {
            for (int warmup = 0; warmup < 3; ++warmup) {
                compute(backend, graph, false);
                compute(backend, graph, true);
            }
            std::vector<double> elapsed[2];
            for (int rep = 0; rep < 10; ++rep) {
                for (int order = 0; order < 2; ++order) {
                    const int fast = (rep + order) & 1;
                    const auto start = std::chrono::steady_clock::now();
                    compute(backend, graph, fast != 0);
                    const auto stop = std::chrono::steady_clock::now();
                    elapsed[fast].push_back(std::chrono::duration<double, std::milli>(stop - start).count());
                }
            }
            for (auto & samples : elapsed) {
                std::sort(samples.begin(), samples.end());
            }
            printf("diagnostic_graph_ms rows=%lld original=%.6f candidate=%.6f samples=10 (includes quantization and host dispatch)\n",
                    (long long) c.rows, 0.5 * (elapsed[0][4] + elapsed[0][5]), 0.5 * (elapsed[1][4] + elapsed[1][5]));
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return ok;
}

struct head_graph {
    ggml_context * ctx;
    ggml_tensor * out;
    ggml_cgraph * graph;
    ggml_gallocr_t alloc;
};

static head_graph make_head_graph(ggml_backend_t backend, ggml_tensor * weight, ggml_tensor * input, const char * name) {
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    head_graph result{};
    result.ctx = ggml_init(params);
    GGML_ASSERT(result.ctx);
    result.out = ggml_mul_mat(result.ctx, weight, input);
    ggml_set_name(result.out, name);
    ggml_set_output(result.out);
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    GGML_ASSERT(ggml_backend_supports_op(backend, result.out));
    result.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(result.alloc, result.graph));
    return result;
}

static void free_head_graph(head_graph & graph) {
    ggml_gallocr_free(graph.alloc);
    ggml_free(graph.ctx);
}

static bool compare_batched_vectors(
        const std::vector<float> & reference, const std::vector<float> & candidate,
        int64_t rows, int64_t cols, int pattern, const char * label) {
    GGML_ASSERT(reference.size() == candidate.size());
    size_t mismatches = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            max_abs = std::numeric_limits<double>::infinity();
        } else {
            max_abs = std::max(max_abs, fabs((double) reference[i] - candidate[i]));
        }
        if (float_bits(reference[i]) != float_bits(candidate[i])) {
            if (mismatches < 4) {
                fprintf(stderr, "%s mismatch cols=%lld col=%zu row=%zu original=%a [%08x] candidate=%a [%08x]\n",
                        label, (long long) cols, i / rows, i % rows,
                        reference[i], float_bits(reference[i]), candidate[i], float_bits(candidate[i]));
            }
            ++mismatches;
        }
    }
    const double kld = mismatches ? kl_divergence(reference, candidate, rows) : 0.0;
    printf("%s rows=%lld k=2560 cols=%lld pattern=%d mismatches=%zu/%zu max_abs=%.12g kld=%.12g\n",
            label, (long long) rows, (long long) cols, pattern, mismatches, reference.size(), max_abs, kld);
    return mismatches == 0 && std::isfinite(max_abs);
}

static bool run_batched_case(ggml_backend_t backend, int64_t rows) {
    constexpr int64_t k = 2560;
    constexpr int64_t max_cols = 4;
    ggml_init_params pw = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    GGML_ASSERT(ctx_w);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q5_K, k, rows);
    ggml_tensor * input_one = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, 1);
    ggml_tensor * input_all = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, max_cols);
    ggml_tensor * input_two = ggml_view_2d(ctx_w, input_all, k, 2, input_all->nb[1], 0);
    ggml_tensor * input_three = ggml_view_2d(ctx_w, input_all, k, 3, input_all->nb[1], 0);
    ggml_set_input(input_one);
    ggml_set_input(input_all);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(buf_w);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(0x5A17B93U + (uint32_t) k);
    fill_weight(weight, rng);

    head_graph graph_one = make_head_graph(backend, weight, input_one, "mtp_chain_full_logits-47");
    head_graph graph_two = make_head_graph(backend, weight, input_two, "mtp_chain_full_logits-47");
    head_graph graph_three = make_head_graph(backend, weight, input_three, "mtp_chain_full_logits-47");
    head_graph graph_four = make_head_graph(backend, weight, input_all, "mtp_chain_full_logits-47");
    std::vector<float> activation(k * max_cols);
    std::vector<float> reference_all(rows * max_cols);
    const auto compute_reference = [&](int64_t cols, std::vector<float> * download) {
        for (int64_t col = 0; col < cols; ++col) {
            ggml_backend_tensor_set(input_one, activation.data() + col * k, 0, k * sizeof(float));
            compute(backend, graph_one.graph, false);
            if (download) {
                ggml_backend_tensor_get(graph_one.out, download->data() + col * rows, 0, rows * sizeof(float));
            }
        }
    };

    bool ok = true;
    for (int pattern = 0; pattern < 5; ++pattern) {
        fill_activation(activation, rng, pattern);
        ggml_backend_tensor_set(input_all, activation.data(), 0, ggml_nbytes(input_all));
        set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "0");
        compute_reference(max_cols, &reference_all);
        set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "1");
        std::vector<float> singleton_on(reference_all.size());
        compute_reference(max_cols, &singleton_on);
        ok = compare_batched_vectors(reference_all, singleton_on, rows, max_cols, pattern, "batched_singleton_fallback") && ok;
        for (int64_t cols : { 3, 4 }) {
            head_graph & graph = cols == 3 ? graph_three : graph_four;
            const char * name = pattern % 2 == 0 ? "mtp_chain_full_logits" : "mtp_chain_full_logits-47";
            ggml_set_name(graph.out, name);
            std::vector<float> candidate(rows * cols);
            std::vector<float> reference(reference_all.begin(), reference_all.begin() + rows * cols);
            set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "0");
            compute(backend, graph.graph, false);
            ggml_backend_tensor_get(graph.out, candidate.data(), 0, ggml_nbytes(graph.out));
            ok = compare_batched_vectors(reference, candidate, rows, cols, pattern, "batched_default_comparison") && ok;
            set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "1");
            compute(backend, graph.graph, false);
            ggml_backend_tensor_get(graph.out, candidate.data(), 0, ggml_nbytes(graph.out));
            ok = compare_batched_vectors(reference, candidate, rows, cols, pattern, "batched_comparison") && ok;

            if (pattern == 1) {
                for (int warmup = 0; warmup < 3; ++warmup) {
                    compute_reference(cols, nullptr);
                    compute(backend, graph.graph, false);
                }
                std::vector<double> elapsed[2];
                for (int rep = 0; rep < 10; ++rep) {
                    for (int order = 0; order < 2; ++order) {
                        const int batched = (rep + order) & 1;
                        const auto start = std::chrono::steady_clock::now();
                        if (batched) {
                            compute(backend, graph.graph, false);
                        } else {
                            compute_reference(cols, nullptr);
                        }
                        const auto stop = std::chrono::steady_clock::now();
                        elapsed[batched].push_back(std::chrono::duration<double, std::milli>(stop - start).count());
                    }
                }
                for (auto & samples : elapsed) {
                    std::sort(samples.begin(), samples.end());
                }
                printf("batched_diagnostic_graph_ms rows=%lld cols=%lld sequential=%.6f batched=%.6f samples=10 (includes quantization, host dispatch, and sequential input uploads)\n",
                        (long long) rows, (long long) cols,
                        0.5 * (elapsed[0][4] + elapsed[0][5]), 0.5 * (elapsed[1][4] + elapsed[1][5]));

                const char * fallback_names[] = { "", "result_output-47" };
                const char * fallback_labels[] = { "batched_default_name_fallback", "batched_othername_fallback" };
                for (int test = 0; test < 2; ++test) {
                    ggml_set_name(graph.out, fallback_names[test]);
                    set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "0");
                    compute(backend, graph.graph, false);
                    std::vector<float> fallback_off(candidate.size());
                    ggml_backend_tensor_get(graph.out, fallback_off.data(), 0, ggml_nbytes(graph.out));
                    set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "1");
                    compute(backend, graph.graph, false);
                    ggml_backend_tensor_get(graph.out, candidate.data(), 0, ggml_nbytes(graph.out));
                    ok = compare_batched_vectors(fallback_off, candidate, rows, cols, pattern, fallback_labels[test]) && ok;
                }
            }
        }

        if (pattern == 1) {
            std::vector<float> fallback_off(rows * 2);
            std::vector<float> fallback_on(fallback_off.size());
            set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "0");
            compute(backend, graph_two.graph, false);
            ggml_backend_tensor_get(graph_two.out, fallback_off.data(), 0, ggml_nbytes(graph_two.out));
            set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "1");
            compute(backend, graph_two.graph, false);
            ggml_backend_tensor_get(graph_two.out, fallback_on.data(), 0, ggml_nbytes(graph_two.out));
            ok = compare_batched_vectors(fallback_off, fallback_on, rows, 2, pattern, "batched_width2_fallback") && ok;
        }
    }

    free_head_graph(graph_four);
    free_head_graph(graph_three);
    free_head_graph(graph_two);
    free_head_graph(graph_one);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return ok;
}

int main(int argc, char ** argv) {
    const bool batched = argc == 2 && strcmp(argv[1], "--batched") == 0;
    if (argc > 1 && !batched) {
        fprintf(stderr, "usage: %s [--batched]\n", argv[0]);
        return 2;
    }
    set_env("GGML_CUDA_DISABLE_GRAPHS", "1");
    if (batched) {
        // These switches are cached on first use, so keep them fixed for this process.
        set_env("GGML_CUDA_SM86_EXACT_REUSE", "0");
        set_env("GGML_CUDA_QC4_NW1", "0");
        set_env("GGML_CUDA_MTP_Q5_HEAD", "0");
        set_env("GGML_CUDA_MTP_Q5_BATCH_HEAD", "1");
        set_env("GGML_CUDA_MMVQ_THIN", "0");
        set_env("GGML_MMVQ_NMAX", "q5_K=4");
        printf("batched_flags EXACT_REUSE=0 QC4_NW1=0 MTP_Q5_HEAD=0 MTP_Q5_BATCH_HEAD=1 MMVQ_THIN=0 MMVQ_NMAX=q5_K=4\n");
    }
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0) {
            device = dev;
            break;
        }
    }
    if (!device) {
        fprintf(stderr, "test-mtp-q5-head: CUDA backend required\n");
        return 2;
    }
    printf("device=%s description=%s; the candidate dispatch requires SM86\n",
            ggml_backend_dev_name(device), ggml_backend_dev_description(device));
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    GGML_ASSERT(backend);
    bool ok = true;
    if (batched) {
        for (int64_t rows : { 65536, 65539, 248320 }) {
            ok = run_batched_case(backend, rows) && ok;
        }
    } else {
        for (const head_case & c : {
                head_case{ 2560,  65536, 1, false },
                head_case{ 2560,  65539, 1, false },
                head_case{ 2560, 248320, 1, false },
                head_case{ 2560,  65535, 1, false },
                head_case{ 2304,  65536, 1, false },
                head_case{ 2560,  65536, 2, false },
                head_case{ 2560,  65536, 1, true  },
        }) {
            ok = run_case(backend, c) && ok;
        }
    }
    ggml_backend_free(backend);
    ggml_quantize_free();
    printf("test-mtp-q5-head%s: %s\n", batched ? " --batched" : "", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
