// Exact byte gathering and attention math for the helper-only compact QSA path.
// Default/--cpu-only uses CPU. --cuda-only explicitly loads and requires a GPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../src/models/models.h"
#include "../ggml/src/ggml-sjkvarn.h"

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

// The real resident buffer is smaller than the selection's record span. Replays force
// eviction, hits, graph reuse, partial host writes and checkpoint restoration on one graph.
static void run_resident(ggml_backend_t backend, int queries, ggml_type seal_type = GGML_TYPE_I8) {
    constexpr int D = 256, G = 128, H = 2, QH = 24, records = 32, sink = 128, cap = 128;
    constexpr int live_cells = sink + records*G, cells = GGML_PAD(live_cells, 256), width = 16;
    const auto layout = ggml_sj_kvarn::make_layout(D, G, 4, 4);
    auto * dev = ggml_backend_get_device(backend);
    using type_fn = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t, size_t, uint32_t);
    auto get_type = reinterpret_cast<type_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_qsa_resident_buffer_type"));
    require(get_type != nullptr, "resident CUDA buffer entry point");
    auto * buft = get_type(dev, H*layout.bytes, 1); // 32 four-cell pages
    require(buft != nullptr, "resident CUDA buffer type");
    ggml_context * body_ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    auto * resident_body = ggml_new_tensor_1d(body_ctx, GGML_TYPE_I8, records*H*layout.bytes);
    auto * resident_buffer = ggml_backend_alloc_ctx_tensors_from_buft(body_ctx, buft);
    require(resident_buffer != nullptr, "resident CUDA body allocation");
    {
        graph_case test;
        auto * body = ggml_new_tensor_1d(test.ctx, GGML_TYPE_I8, records*H*layout.bytes);
        auto * desc = ggml_new_tensor_1d(test.ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
        auto * q = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, D, queries, QH, 1);
        auto * k = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F16, D, sink + cap, H, 1);
        auto * v = ggml_dup_tensor(test.ctx, k);
        auto * mask = ggml_new_tensor_2d(test.ctx, GGML_TYPE_F16, cells, queries);
        auto * ids = ggml_new_tensor_2d(test.ctx, GGML_TYPE_I32, width, queries);
        for (auto * t : {body, desc, q, k, v, mask, ids}) ggml_set_input(t);
        ggml_tensor * seal_k = nullptr;
        ggml_tensor * seal_v = nullptr;
        if (seal_type != GGML_TYPE_I8) {
            seal_k = ggml_new_tensor_2d(test.ctx, seal_type, D*H, sink + cap);
            seal_v = ggml_dup_tensor(test.ctx, seal_k);
            ggml_set_input(seal_k);
            ggml_set_input(seal_v);
        }
        // Keep the reference records alive across repeated graph evaluations.
        ggml_set_output(body);
        // The sealer aliases this buffer but replaces its source edge with K.
        ggml_build_forward_expand(test.graph, body);
        auto attention = [&](ggml_tensor * b) {
            if (seal_k) {
                b = ggml_sj_kvarn_seal_dyn(test.ctx, b, seal_k, seal_v, desc, D, G, 4, 4, 16, records);
            }
            auto * out = ggml_flash_attn_ext(test.ctx, q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_sj_kvarn(out, b, desc, 4, 4, cells);
            out->src[7] = ids;
            ggml_prec_set_acc(out, GGML_PREC_F32);
            ggml_set_output(out);
            ggml_build_forward_expand(test.graph, out);
            return out;
        };
        auto * expected = attention(body);
        auto * actual = attention(resident_body);
        test.allocate(backend);
        require(body->buffer != nullptr, "reference body is allocated before sealing");
        std::mt19937 rng(137 + queries);
        std::vector<uint8_t> bytes(ggml_nbytes(body));
        for (auto & b : bytes) b = uint8_t(rng());
        for (int r = 0; r < records*H; ++r) {
            auto * rec = bytes.data() + r*layout.bytes;
            for (int d = 0; d < D; ++d) {
                ggml_sj_kvarn::put_half(rec, layout.k_scale + 2*d, 0.03f + 0.001f*(d%5));
                ggml_sj_kvarn::put_half(rec, layout.k_zero + 2*d, -0.2f);
                ggml_sj_kvarn::put_half(rec, layout.v_ch + 2*d, 0.8f + 0.02f*(r%7));
            }
            for (int t = 0; t < G; ++t) {
                ggml_sj_kvarn::put_half(rec, layout.k_tok + 2*t, 0.8f + 0.01f*(t%9));
                ggml_sj_kvarn::put_half(rec, layout.v_scale + 2*t, 0.02f + 0.001f*(t%3));
                ggml_sj_kvarn::put_half(rec, layout.v_zero + 2*t, -0.1f);
            }
        }
        ggml_backend_tensor_set(body, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(resident_body, bytes.data(), 0, bytes.size());
        std::vector<int32_t> descriptor(GGML_SJKVARN_DESC_N_ENTRIES, 0);
        descriptor[GGML_SJKVARN_DESC_S] = sink;
        descriptor[GGML_SJKVARN_DESC_CAP] = cap;
        descriptor[GGML_SJKVARN_DESC_B] = live_cells;
        descriptor[GGML_SJKVARN_DESC_B_OLD] = sink;
        descriptor[GGML_SJKVARN_DESC_N] = live_cells;
        descriptor[GGML_SJKVARN_DESC_G] = G;
        descriptor[GGML_SJKVARN_DESC_D] = D;
        descriptor[GGML_SJKVARN_DESC_RECBYTES] = layout.bytes;
        descriptor[GGML_SJKVARN_DESC_HKV] = H;
        descriptor[GGML_SJKVARN_DESC_TYPE_K] = GGML_TYPE_F16;
        descriptor[GGML_SJKVARN_DESC_TYPE_V] = GGML_TYPE_F16;
        ggml_backend_tensor_set(desc, descriptor.data(), 0, ggml_nbytes(desc));
        std::vector<float> query(ggml_nelements(q));
        for (auto & x : query) x = (int(rng()%200) - 100)*0.01f;
        ggml_backend_tensor_set(q, query.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_memset(k, 0, 0, ggml_nbytes(k));
        ggml_backend_tensor_memset(v, 0, 0, ggml_nbytes(v));
        std::vector<int32_t> selected(width*queries);
        std::vector<ggml_fp16_t> masks(cells*queries);
        std::vector<float> ref(ggml_nelements(expected)), out(ref.size());
        const auto snapshot = bytes;
        for (int step = 0; step < 12; ++step) {
            if (seal_k) {
                quantize(seal_k, rng);
                quantize(seal_v, rng);
            }
            std::fill(masks.begin(), masks.end(), ggml_fp32_to_fp16(-INFINITY));
            for (int row = 0; row < queries; ++row) {
                for (int j = 0; j < width; ++j) {
                    const int page = ((step/2)*37 + row*4 + j/4)%(records*G/4);
                    const int p = sink + page*4 + j%4;
                    selected[row*width + j] = p;
                    masks[row*cells + p] = ggml_fp32_to_fp16(0.0f);
                }
            }
            if (!seal_k && step == 5) {
                // This record is still resident from the preceding call.
                const int r = ((step/2)*37)/32;
                const size_t start = size_t(r)*H*layout.bytes + layout.v_scale;
                ggml_sj_kvarn::put_half(bytes.data(), start, 0.125f);
                ggml_backend_tensor_set(body, bytes.data() + start, start, 2);
                ggml_backend_tensor_set(resident_body, bytes.data() + start, start, 2);
            }
            if (!seal_k && step == 9) {
                std::vector<uint8_t> checkpoint(bytes.size());
                ggml_backend_tensor_get(resident_body, checkpoint.data(), 0, checkpoint.size());
                require(checkpoint == bytes, "resident checkpoint contains authoritative bytes");
                bytes = snapshot;
                ggml_backend_tensor_set(body, bytes.data(), 0, bytes.size());
                ggml_backend_tensor_set(resident_body, bytes.data(), 0, bytes.size());
            }
            ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
            test.compute(backend);
            ggml_backend_tensor_get(expected, ref.data(), 0, ggml_nbytes(expected));
            ggml_backend_tensor_get(actual, out.data(), 0, ggml_nbytes(actual));
            std::vector<uint8_t> reference_bytes(bytes.size());
            ggml_backend_tensor_get(body, reference_bytes.data(), 0, reference_bytes.size());
            if (seal_k) {
                std::vector<uint8_t> host_bytes(bytes.size());
                ggml_backend_tensor_get(resident_body, host_bytes.data(), 0, host_bytes.size());
                require(reference_bytes == host_bytes, "GPU sealing produces identical device and mapped-host records");
            } else {
                require(reference_bytes == bytes, "reference body survives graph reuse");
            }
            double error = 0.0, norm = 0.0, maxabs = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                require(std::isfinite(out[i]) && std::isfinite(ref[i]), "resident attention is finite");
                error += double(out[i] - ref[i])*(out[i] - ref[i]);
                norm += double(ref[i])*ref[i];
                maxabs = std::max(maxabs, std::abs(double(out[i]) - ref[i]));
            }
            fprintf(stderr, "QSA CLOCK queries=%d seal=%s step=%d nmse=%.12g maxabs=%.12g ref0=%.9g out0=%.9g\n",
                    queries, ggml_type_name(seal_type), step, error/std::max(norm, 1e-30), maxabs, ref[0], out[0]);
            require(error/std::max(norm, 1e-30) <= 1e-8, "CLOCK attention matches fully resident body");
            require(std::memcmp(ref.data(), out.data(), ref.size()*sizeof(float)) == 0, "resident page repack is bit-exact");
        }
        printf("PASS QSA CLOCK queries=%d pages=32 records=32 hits/eviction/partial-write/checkpoint/restore\n", queries);
    }
    ggml_backend_buffer_free(resident_buffer);
    ggml_free(body_ctx);
}

int main(int argc, char ** argv) {
    const bool resident_only = argc == 2 && std::strcmp(argv[1], "--resident-only") == 0;
    const bool long_cuda = argc == 2 && std::strcmp(argv[1], "--cuda-long-only") == 0;
    const bool cuda = resident_only || long_cuda || (argc == 2 && std::strcmp(argv[1], "--cuda-only") == 0);
    if (argc > 2 || (argc == 2 && !cuda && std::strcmp(argv[1], "--cpu-only") != 0)) { return 2; }
    if (cuda) { ggml_backend_load_all(); }
    ggml_backend_t backend = cuda ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "required backend is unavailable\n"); return 1; }
    int result = 0;
    try {
        if (resident_only) {
            for (ggml_type type : {GGML_TYPE_I8, GGML_TYPE_F16, GGML_TYPE_TQ6_0}) {
                for (int queries : {1, 4, 5, 8}) run_resident(backend, queries, type);
            }
        } else if (long_cuda) {
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
