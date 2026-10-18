// Standalone regression: graph-held scratch survives a refused trim; after
// graph reset, scratch is returned and the identical graph can be run again.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        std::fprintf(stderr, "CUDA0 unavailable; this test was not run\n");
        return 2;
    }
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto reset = (bool (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reset");
    auto trim = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pool_trim");
    if (!backend || !reset || !trim) {
        std::fprintf(stderr, "required CUDA pool/graph interface unavailable\n");
        if (backend) { ggml_backend_free(backend); }
        return 2;
    }
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 1024);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 256);
    auto * output = ggml_mul_mat(ctx, weights, input);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buffer != nullptr;
    std::vector<float> ones(1024 * 1024, 1.0f);
    std::vector<ggml_fp16_t> half_ones(ones.size());
    ggml_fp32_to_fp16_row(ones.data(), half_ones.data(), ones.size());
    std::vector<float> values(1024 * 256);
    for (size_t i = 0; i < values.size(); ++i) { values[i] = float(i % 16) / 16; }
    if (ok) {
        ggml_backend_tensor_set(weights, half_ones.data(), 0, ggml_nbytes(weights));
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    }
    auto run = [&] {
        if (!ok) { return; }
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
    };
    // First call warms up; subsequent calls capture and replay the stable graph.
    for (int i = 0; i < 3; ++i) { run(); }
    std::vector<float> before(1024 * 256), after(before.size());
    if (ok) { ggml_backend_tensor_get(output, before.data(), 0, ggml_nbytes(output)); }
    for (float value : before) { ok = ok && std::isfinite(value) && value == 480.0f; }
    const size_t guarded = trim(backend);
    ok = ok && guarded == 0; // Graphs still reference the pool's original addresses.
    run();
    if (ok) { ggml_backend_tensor_get(output, after.data(), 0, ggml_nbytes(output)); }
    ok = ok && std::memcmp(before.data(), after.data(), ggml_nbytes(output)) == 0;
    ok = reset(backend) && ok;
    const size_t released = trim(backend);
    ok = ok && released > 0 && trim(backend) == 0; // Real reclaim, then idempotent.
    for (int i = 0; i < 3; ++i) { run(); }
    if (ok) { ggml_backend_tensor_get(output, after.data(), 0, ggml_nbytes(output)); }
    ok = ok && std::memcmp(before.data(), after.data(), ggml_nbytes(output)) == 0;
    std::printf("CUDA pool trim: %s; graph_guard_bytes=%zu released_bytes=%zu; replay bytes identical\n",
                ok ? "PASS" : "FAIL", guarded, released);
    reset(backend);
    trim(backend);
    if (buffer) { ggml_backend_buffer_free(buffer); }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
