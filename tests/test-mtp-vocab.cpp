#include "../src/llama-mtp-vocab.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

static void invalid(const std::string & text, int max_size = 32768) {
    bool failed = false;
    try {
        std::istringstream in(text);
        llama_mtp_vocab_read(in, max_size);
    } catch (const std::runtime_error &) { failed = true; }
    GGML_ASSERT(failed);
}

static void test_head(ggml_backend_t backend, ggml_type type, int n_embd, int n_vocab) {
    const int n_sel = 65;
    const int top_k = 10;
    auto ctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true});
    GGML_ASSERT(ctx);
    auto head = ggml_new_tensor_2d(ctx, type, n_embd, n_vocab);
    auto hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    auto ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sel);
    auto indexed = ggml_reshape_1d(ctx, ggml_mul_mat_id(ctx,
            ggml_reshape_3d(ctx, head, n_embd, 1, n_vocab), hidden, ids), n_sel);
    auto full = ggml_mul_mat(ctx, head, hidden);
    auto selected = ggml_reshape_1d(ctx,
            ggml_get_rows(ctx, ggml_reshape_2d(ctx, full, 1, n_vocab), ids), n_sel);
    auto top_indexed = ggml_top_k(ctx, indexed, top_k);
    auto top_selected = ggml_top_k(ctx, selected, top_k);
    auto ids_rows = ggml_reshape_2d(ctx, ids, 1, n_sel);
    auto tokens_indexed = ggml_get_rows(ctx, ids_rows, top_indexed);
    auto tokens_selected = ggml_get_rows(ctx, ids_rows, top_selected);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tokens_indexed);
    ggml_build_forward_expand(graph, tokens_selected);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        GGML_ASSERT(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)));
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    uint32_t state = 0x6d2b79f5;
    auto random = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float) (state & 0xffff) / 65536.0f - 0.5f;
    };
    std::vector<float> h(n_embd);
    for (auto & value : h) { value = random(); }
    std::vector<float> row(n_embd);
    std::vector<uint8_t> packed(ggml_nbytes(head));
    for (int i = 0; i < n_vocab; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            row[j] = random() + h[j] * (float(i) / n_vocab);
        }
        const size_t size = ggml_quantize_chunk(type, row.data(), packed.data() + (size_t) i * head->nb[1], 0, 1, n_embd, nullptr);
        GGML_ASSERT(size == head->nb[1]);
    }
    std::vector<int32_t> selected_ids(n_sel);
    for (int i = 0; i < n_sel; ++i) {
        const int slot = (i * 37) % n_sel;
        selected_ids[i] = (int64_t) slot * (n_vocab - 1) / (n_sel - 1);
    }
    ggml_backend_tensor_set(head, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(hidden, h.data(), 0, h.size() * sizeof(float));
    ggml_backend_tensor_set(ids, selected_ids.data(), 0, selected_ids.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> a(n_sel), b(n_sel);
    int32_t tokens_a[top_k], tokens_b[top_k];
    ggml_backend_tensor_get(indexed, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(selected, b.data(), 0, b.size() * sizeof(float));
    ggml_backend_tensor_get(tokens_indexed, tokens_a, 0, sizeof(tokens_a));
    ggml_backend_tensor_get(tokens_selected, tokens_b, 0, sizeof(tokens_b));
    float max_error = 0.0f;
    for (int i = 0; i < n_sel; ++i) {
        GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(b[i]));
        const float error = std::fabs(a[i] - b[i]) / std::max(1.0f, std::fabs(b[i]));
        max_error = std::max(max_error, error);
        if (error > 1e-4f) {
            fprintf(stderr, "%s %s K=%d vocab=%d row=%d token=%d: indexed=%g full=%g relative_error=%g\n",
                    ggml_backend_name(backend), ggml_type_name(type), n_embd, n_vocab, i, selected_ids[i], a[i], b[i], error);
            GGML_ABORT("indexed head logits differ from the full head");
        }
    }
    for (int i = 0; i < top_k; ++i) { GGML_ASSERT(tokens_a[i] == tokens_b[i]); }
    printf("%s %s K=%d vocab=%d selected=%d: logits and mapped top-%d match (max relative error %g)\n",
            ggml_backend_name(backend), ggml_type_name(type), n_embd, n_vocab, n_sel, top_k, max_error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * backend_name = "CPU";
    if (argc == 3 && strcmp(argv[1], "--backend") == 0) {
        backend_name = argv[2];
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--backend CPU|CUDA0]\n", argv[0]);
        return 1;
    }
    std::istringstream text("llama-mtp-vocab-v1 8 3\n7 1 4\n");
    const auto map = llama_mtp_vocab_read(text);
    GGML_ASSERT(map.n_vocab == 8 && map.ids == std::vector<int32_t>({7, 1, 4}));
    invalid("bad 8 3 7 1 4");
    invalid("llama-mtp-vocab-v1 8 3 7 1 1");
    invalid("llama-mtp-vocab-v1 8 3 7 1 8");
    invalid("llama-mtp-vocab-v1 8 3 7 1 -1");
    invalid("llama-mtp-vocab-v1 8 3 7 1");
    invalid("llama-mtp-vocab-v1 8 3 7 1 4 extra");
    invalid("llama-mtp-vocab-v1 8 3 7 1 4", 2);
    invalid("llama-mtp-vocab-v1 8 0");
    invalid("llama-mtp-vocab-v1 16777217 1 0");

    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(backend);
    auto ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
    auto head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    auto hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    auto ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    auto indexed = ggml_reshape_1d(ctx, ggml_mul_mat_id(ctx, ggml_reshape_3d(ctx, head, 4, 1, 8), hidden, ids), 3);
    auto full = ggml_mul_mat(ctx, head, hidden);
    auto selected = ggml_get_rows(ctx, ggml_reshape_2d(ctx, full, 1, 8), ids);
    auto compact_argmax = ggml_argmax(ctx, indexed);
    auto token = ggml_get_rows(ctx, ggml_reshape_2d(ctx, ids, 1, 3), compact_argmax);
    auto probs = ggml_soft_max(ctx, indexed);
    auto prob = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probs, 1, 3), compact_argmax);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, selected);
    ggml_build_forward_expand(graph, token);
    ggml_build_forward_expand(graph, prob);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    float weights[32] = {};
    for (int i = 0; i < 8; ++i) { weights[4*i] = float(i); }
    const float h[4] = {1, 0, 0, 0};
    ggml_backend_tensor_set(head, weights, 0, sizeof(weights));
    ggml_backend_tensor_set(hidden, h, 0, sizeof(h));
    ggml_backend_tensor_set(ids, map.ids.data(), 0, 3 * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    float a[3], b[3], probability;
    int32_t result;
    ggml_backend_tensor_get(indexed, a, 0, sizeof(a));
    ggml_backend_tensor_get(selected, b, 0, sizeof(b));
    ggml_backend_tensor_get(token, &result, 0, sizeof(result));
    ggml_backend_tensor_get(prob, &probability, 0, sizeof(probability));
    GGML_ASSERT(result == 7);
    for (int i = 0; i < 3; ++i) { GGML_ASSERT(a[i] == b[i] && a[i] == map.ids[i]); }
    GGML_ASSERT(std::fabs(probability - 1.0f / (1.0f + std::exp(-6.0f) + std::exp(-3.0f))) < 1e-6f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    backend = ggml_backend_init_by_name(backend_name, nullptr);
    GGML_ASSERT(backend);
    test_head(backend, GGML_TYPE_Q5_K, 2560, 257);
    test_head(backend, GGML_TYPE_Q5_K, 2560, 1025);
    test_head(backend, GGML_TYPE_F32, 4, 65539);
    ggml_backend_free(backend);
    ggml_quantize_free();
}
