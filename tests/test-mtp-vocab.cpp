#include "../src/llama-mtp-vocab.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <sstream>

static void invalid(const std::string & text, int max_size = 32768) {
    bool failed = false;
    try {
        std::istringstream in(text);
        llama_mtp_vocab_read(in, max_size);
    } catch (const std::runtime_error &) { failed = true; }
    GGML_ASSERT(failed);
}

int main() {
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
}
