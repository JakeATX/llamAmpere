// Exact byte gathering and attention math for the helper-only compact QSA path.
// Default/--cpu-only uses CPU. --cuda-only explicitly loads and requires a GPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../src/models/models.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct graph_case {
    ggml_context * ctx = ggml_init({ 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_gallocr_t alloc = nullptr;
    ~graph_case() { if (alloc) { ggml_gallocr_free(alloc); } ggml_free(ctx); }
    void allocate(ggml_backend_t backend) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "backend supports compact test graph");
        }
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "allocate compact test graph");
    }
    void compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute compact test graph");
    }
};

static ggml_tensor * gather(ggml_context * ctx, ggml_tensor * cache, ggml_tensor * ids) {
    return llama_model_qwen4exp::graph::build_qsa_gather_bytes(ctx, cache, ids);
}

static void run_bytes(ggml_backend_t backend, ggml_type type) {
    graph_case test;
    constexpr int dim = 256, heads = 2, cells = 67, selected = 256;
    auto * cache = ggml_new_tensor_4d(test.ctx, type, dim, heads, cells, 1);
    // Native contiguous heads, with a gap between cells as in a strided KV view.
    cache->nb[2] += 16;
    cache->nb[3] = cache->nb[2]*cells;
    ggml_set_input(cache);
    ggml_set_output(cache); // verify source lifetime and preserve it for a post-compute byte check
    auto * ids = ggml_new_tensor_1d(test.ctx, GGML_TYPE_I32, selected);
    ggml_set_input(ids);
    auto * output = gather(test.ctx, cache, ids);
    ggml_set_output(output);
    ggml_build_forward_expand(test.graph, output);
    test.allocate(backend);
    const size_t cell_bytes = heads*ggml_row_size(type, dim);
    std::vector<uint8_t> source(ggml_nbytes(cache)), actual(ggml_nbytes(output)), unchanged(source.size());
    std::vector<int32_t> indices(selected);
    for (int trial = 0; trial < 3; ++trial) {
        for (size_t i = 0; i < source.size(); ++i) { source[i] = uint8_t((i*73 + trial*19) % 251); }
        // Unsorted/repeated live cells and safe dead padding exercise the actual byte alias helper.
        for (int i = 0; i < selected; ++i) { indices[i] = i < 137 ? (i*37 + trial) % cells : 0; }
        ggml_backend_tensor_set(cache, source.data(), 0, source.size());
        ggml_backend_tensor_set(ids, indices.data(), 0, ggml_nbytes(ids));
        test.compute(backend);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
        ggml_backend_tensor_get(cache, unchanged.data(), 0, unchanged.size());
        require(source == unchanged, "byte gather leaves the strided source unchanged");
        for (int i = 0; i < selected; ++i) {
            require(std::memcmp(actual.data() + i*cell_bytes, source.data() + indices[i]*cache->nb[2], cell_bytes) == 0,
                    "compact gather copies quantized cell bytes exactly");
        }
    }
    printf("PASS compact bytes %s type=%s row_bytes=%zu strided/repeated/padded/reused\n",
            ggml_backend_name(backend), ggml_type_name(type), ggml_row_size(type, dim));
}

static void quantize(ggml_tensor * tensor, std::mt19937 & rng) {
    std::normal_distribution<float> normal(0.0f, 0.7f);
    std::vector<float> values(ggml_nelements(tensor));
    for (float & value : values) { value = normal(rng); }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    require(ggml_quantize_chunk(tensor->type, values.data(), bytes.data(), 0,
                ggml_nrows(tensor), tensor->ne[0], nullptr) == bytes.size(), "quantize valid native KV cells");
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

static void run_attention(ggml_backend_t backend, int cells = 4096) {
    graph_case test;
    constexpr int dim = 256, heads = 24, heads_kv = 2, live = 2051, selected = 2304;
    auto * q = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, dim, 1, heads, 1);
    auto * k = ggml_new_tensor_4d(test.ctx, GGML_TYPE_TQ5_0, dim, heads_kv, cells, 1);
    auto * v = ggml_new_tensor_4d(test.ctx, GGML_TYPE_TURBO4_0, dim, heads_kv, cells, 1);
    auto * ids = ggml_new_tensor_1d(test.ctx, GGML_TYPE_I32, selected);
    // A single query still uses padded mask rows, as the model's FA graph does.
    auto * dense_mask = ggml_new_tensor_2d(test.ctx, GGML_TYPE_F16, cells, 64);
    auto * compact_mask = ggml_new_tensor_2d(test.ctx, GGML_TYPE_F16, selected, 64);
    for (auto * tensor : {q, k, v, ids, dense_mask, compact_mask}) { ggml_set_input(tensor); }
    auto * dense = ggml_flash_attn_ext(test.ctx, q, ggml_permute(test.ctx, k, 0, 2, 1, 3),
            ggml_permute(test.ctx, v, 0, 2, 1, 3), dense_mask, 1.0f/std::sqrt(float(dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(dense, live);
    auto * compact_k = gather(test.ctx, k, ids);
    auto * compact_v = gather(test.ctx, v, ids);
    auto * compact = ggml_flash_attn_ext(test.ctx, q, ggml_permute(test.ctx, compact_k, 0, 2, 1, 3),
            ggml_permute(test.ctx, compact_v, 0, 2, 1, 3), compact_mask, 1.0f/std::sqrt(float(dim)), 0.0f, 0.0f);
    for (auto * tensor : {dense, compact}) { ggml_set_output(tensor); ggml_build_forward_expand(test.graph, tensor); }
    test.allocate(backend);
    std::mt19937 rng(9127);
    quantize(k, rng);
    quantize(v, rng);
    std::normal_distribution<float> normal(0.0f, 0.7f);
    std::vector<float> query(ggml_nelements(q));
    for (float & value : query) { value = normal(rng); }
    std::vector<int32_t> order(cells), indices(selected, 0);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::copy(order.begin(), order.begin() + live, indices.begin());
    const auto drop = ggml_fp32_to_fp16(-INFINITY), keep = ggml_fp32_to_fp16(0.0f);
    std::vector<ggml_fp16_t> full(cells*64, drop), small(selected*64, drop);
    for (int i = 0; i < live; ++i) { full[indices[i]] = keep; small[i] = keep; }
    ggml_backend_tensor_set(q, query.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(ids, indices.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(dense_mask, full.data(), 0, ggml_nbytes(dense_mask));
    ggml_backend_tensor_set(compact_mask, small.data(), 0, ggml_nbytes(compact_mask));
    test.compute(backend);
    std::vector<float> expected(ggml_nelements(dense)), actual(expected.size());
    ggml_backend_tensor_get(dense, expected.data(), 0, ggml_nbytes(dense));
    ggml_backend_tensor_get(compact, actual.data(), 0, ggml_nbytes(compact));
    double error = 0.0, norm = 0.0, kl = 0.0;
    for (int head = 0; head < heads; ++head) {
        double reference_sum = 0.0, actual_sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            const size_t j = head*dim + i;
            require(std::isfinite(expected[j]) && std::isfinite(actual[j]), "compact attention output is finite");
            error += std::pow(double(expected[j]) - actual[j], 2);
            norm += double(expected[j])*expected[j];
            reference_sum += std::exp(double(expected[j]));
            actual_sum += std::exp(double(actual[j]));
        }
        for (int i = 0; i < dim; ++i) {
            const double p = std::exp(double(expected[head*dim + i]))/reference_sum;
            const double q_actual = std::exp(double(actual[head*dim + i]))/actual_sum;
            kl += p*std::log(p/q_actual);
        }
    }
    const double nmse = error/std::max(norm, 1e-30);
    kl /= heads;
    printf("COMPACT_KLD %s cells=%d q=1 heads=24 kv_heads=2 dim=256 live=2051 padded=2304 kl=%.9g nmse=%.9g\n",
            ggml_backend_name(backend), cells, kl, nmse);
    require(kl <= 1e-6 && nmse <= 1e-5, "compact quantized attention stays within the rounding budget");
}

int main(int argc, char ** argv) {
    const bool long_cuda = argc == 2 && std::strcmp(argv[1], "--cuda-long-only") == 0;
    const bool cuda = long_cuda || (argc == 2 && std::strcmp(argv[1], "--cuda-only") == 0);
    if (argc > 2 || (argc == 2 && !cuda && std::strcmp(argv[1], "--cpu-only") != 0)) { return 2; }
    if (cuda) { ggml_backend_load_all(); }
    ggml_backend_t backend = cuda ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "required backend is unavailable\n"); return 1; }
    int result = 0;
    try {
        if (long_cuda) {
            run_attention(backend, 100352);
        } else {
            run_bytes(backend, GGML_TYPE_TQ5_0);
            run_bytes(backend, GGML_TYPE_TURBO4_0);
            run_attention(backend);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL compact QSA: %s\n", e.what());
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
