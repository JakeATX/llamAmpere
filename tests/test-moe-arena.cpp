#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../src/llama-moe-arena.h"
#include <stdexcept>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>

static bool test_reclaimable_output(bool cuda = false) {
#ifndef _WIN32
    const char * temporary = std::getenv("TMPDIR");
    std::string name = std::string(temporary ? temporary : ".") + "/llama-moe-output-test-XXXXXX";
    std::vector<char> path(name.begin(), name.end());
    path.push_back(0);
    const char * directory = path.data();
    if (!mkdtemp(path.data())) { return false; }
    bool ok = true;
    {
        constexpr size_t bytes = 16ull << 20;
        llama_moe_output output(bytes, directory);
        auto * data = static_cast<uint32_t *>(output.data);
        for (size_t i = 0; i < bytes / sizeof(*data); ++i) { ok &= data[i] == 0; }
        // Cover arbitrary float bit patterns, including NaNs, without arithmetic.
        for (size_t i = 0; i < bytes / sizeof(*data); ++i) { data[i] = uint32_t(i * 2654435761u); }
        if (cuda) {
            ggml_backend_load_all();
            ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
            if (!backend) { return false; }
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes / sizeof(float));
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) { ggml_free(ctx); ggml_backend_free(backend); return false; }
            ggml_backend_tensor_set(tensor, data, 0, bytes);
            memset(data, 0, bytes);
            // Match the context's pageable D2H output copy and completion boundary.
            ggml_backend_tensor_get_async(backend, tensor, data, 0, bytes);
            ggml_backend_synchronize(backend);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
        }
        output.flush();
        ok &= !output.dirty;
        ok &= madvise(data, bytes, MADV_DONTNEED) == 0;
        for (size_t i = 0; i < bytes / sizeof(*data); ++i) { ok &= data[i] == uint32_t(i * 2654435761u); }
        data[0] = 0x7fc12345;
        output.dirty = true;
        output.flush();
        ok &= madvise(data, bytes, MADV_DONTNEED) == 0;
        ok &= data[0] == 0x7fc12345;
    }
    // The mapped file was unlinked immediately; destruction leaves no artifact.
    ok &= rmdir(directory) == 0;
    bool rejected = false;
    try { llama_moe_output invalid(4096, "/directory-that-does-not-exist/fnarena"); }
    catch (const std::runtime_error &) { rejected = true; }
    ok &= rejected;
    printf("reclaimable MTP original float bytes%s: %s\n", cuda ? " CUDA D2H" : "", ok ? "OK" : "FAIL");
    return ok;
#else
    return !cuda;
#endif
}

// Exercise the original mixed IQ blocks with the arena's gate/up/down stride.
int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--cuda-output") == 0) { return test_reclaimable_output(true) ? 0 : 1; }
    if (argc != 1) { return 2; }
    constexpr int64_t n_in = 256, n_out = 32, n_experts = 3;
    const ggml_type types[] = { GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL };
    size_t bytes[3], stride = 0;
    for (int role = 0; role < 3; ++role) {
        bytes[role] = ggml_row_size(types[role], n_in) * n_out;
        stride += bytes[role];
    }
    std::vector<float> values(n_in * n_out * n_experts), importance(n_in, 1.0f);
    ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    auto * packed_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), stride * n_experts);
    auto * arena_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), stride * n_experts);
    auto * packed = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_buf));
    auto * arena = static_cast<uint8_t *>(ggml_backend_buffer_get_base(arena_buf));
    ggml_tensor * weights[2][3], * outputs[2][3];
    size_t bank_offset = 0, role_offset = 0;
    bool ok = test_reclaimable_output();
    for (int role = 0; role < 3; ++role) {
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.2f * sinf(float(i + role * 17) * 0.013f); }
        const size_t written = ggml_quantize_chunk(types[role], values.data(), packed + bank_offset,
                0, n_out * n_experts, n_in, importance.data());
        ok &= written == bytes[role] * n_experts;
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            memcpy(arena + expert * stride + role_offset, packed + bank_offset + expert * bytes[role], bytes[role]);
        }
        for (int layout = 0; layout < 2; ++layout) {
            auto * t = ggml_new_tensor_3d(ctx, types[role], n_in, n_out, n_experts);
            weights[layout][role] = t;
            if (layout) {
                t->nb[2] = stride;
                t->nb[3] = stride * n_experts;
            }
            ok &= ggml_backend_tensor_alloc(layout ? arena_buf : packed_buf, t,
                    layout ? arena + role_offset : packed + bank_offset) == GGML_STATUS_SUCCESS;
        }
        bank_offset += bytes[role] * n_experts;
        role_offset += bytes[role];
    }
    auto * acts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, 1, 1);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    auto * graph = ggml_new_graph(ctx);
    for (int layout = 0; layout < 2; ++layout) {
        for (int role = 0; role < 3; ++role) {
            outputs[layout][role] = ggml_mul_mat_id(ctx, weights[layout][role], acts, ids);
            ggml_build_forward_expand(graph, outputs[layout][role]);
        }
    }
    auto * compute_buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> activation(n_in);
    for (int64_t i = 0; i < n_in; ++i) { activation[i] = cosf(float(i) * 0.017f); }
    const int32_t selected[] = { 2, 0 };
    ggml_backend_tensor_set(acts, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    ok &= ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    for (int role = 0; role < 3; ++role) {
        std::vector<float> reference(n_out * 2), actual(reference.size());
        ggml_backend_tensor_get(outputs[0][role], reference.data(), 0, reference.size() * sizeof(float));
        ggml_backend_tensor_get(outputs[1][role], actual.data(), 0, actual.size() * sizeof(float));
        ok &= reference == actual;
        for (float value : actual) { ok &= std::isfinite(value); }
    }
    ggml_backend_buffer_free(compute_buf);
    ggml_backend_buffer_free(arena_buf);
    ggml_backend_buffer_free(packed_buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_quantize_free();
    printf("mixed-IQ expert arena CPU outputs: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
