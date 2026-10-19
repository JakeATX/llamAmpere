#include "../src/llama-ple-cache.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

int main(int argc, char ** argv) {
#ifndef _WIN32
    const bool cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    if (argc != 1 && !cuda) { return 2; }
    if (cuda) { ggml_backend_load_all(); }
    ggml_backend_t backend = cuda ? ggml_backend_init_by_name("CUDA0", nullptr) : ggml_backend_cpu_init();
    if (!backend) { return 1; }
    const char * temporary = getenv("TMPDIR");
    std::string name = std::string(temporary ? temporary : "/tmp") + "/ple-test-XXXXXX";
    std::vector<char> path(name.begin(), name.end());
    path.push_back(0);
    const int fd = mkstemp(path.data());
    if (fd < 0) { return 1; }
    const std::string profile = std::string(path.data()) + ".rows";
    constexpr int64_t width = 160, n_rows = 1000;
    const auto type = GGML_TYPE_IQ4_NL;
    const size_t row_bytes = ggml_row_size(type, width), bytes = row_bytes * n_rows;
    std::vector<float> values(width * n_rows);
    for (size_t i = 0; i < values.size(); ++i) { values[i] = sinf(float(i) * 0.017f); }
    std::vector<char> quant(bytes);
    ggml_quantize_chunk(type, values.data(), quant.data(), 0, n_rows, width, nullptr);
    if (write(fd, quant.data(), bytes) != (ssize_t) bytes) { return 1; }
    void * source = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    unlink(path.data());
    if (source == MAP_FAILED) { return 1; }
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    auto * table = ggml_new_tensor_2d(ctx, type, width, n_rows);
    auto * source_buffer = ggml_backend_cpu_buffer_from_ptr(source, bytes);
    table->data = source;
    table->buffer = source_buffer;
    auto write_profile = [&](const std::string & tail, int64_t rows = 3, int64_t columns = 160) {
        std::ofstream output(profile);
        output << "LLAMA_PLE_ROWS_V1 " << int(type) << ' ' << columns << ' ' << n_rows << ' ' << row_bytes << ' ' << rows << '\n' << tail;
    };
    bool ok = true;
    for (int test = 0; test < 7; ++test) {
        if (test == 0) { write_profile("1\n1\n999\n"); }
        if (test == 1) { write_profile("1\n8\n1000\n"); }
        if (test == 2) { write_profile("1\n8\n"); }
        if (test == 3) { write_profile("1\n8\n999\njunk\n"); }
        if (test == 4) { write_profile("1\n8\n999\n", 3, width + 1); }
        if (test == 5) { write_profile("", -1); }
        if (test == 6) { write_profile("1\n8\n999\n", INT64_MAX); }
        bool rejected = false;
        try { auto invalid = llama_ple_cache::load(table, profile.c_str(), false); }
        catch (const std::exception &) { rejected = true; }
        ok &= rejected && table->data == source && table->ne[1] == n_rows;
    }
    write_profile("1\n8\n999\n");
    const std::string missed = profile + ".miss";
    setenv("LLAMA_PLE_MISS_PROFILE", missed.c_str(), 1);
    auto cache = llama_ple_cache::load(table, profile.c_str(), cuda);
    unsetenv("LLAMA_PLE_MISS_PROFILE");
    ok &= cache->size() == 3 * row_bytes && table->ne[1] == n_rows && table->data == source;
    auto * original_table = table;
    table = cache->tensor;
    ok &= table->ne[1] == 3;
    const int32_t original[] = { 999, 1, 8, 1 };
    int32_t indices[] = { 999, 1, 8, 1 };
    ok &= cache->remap(indices, 4) && indices[0] == 2 && indices[1] == 0 && indices[2] == 1 && indices[3] == 0;
    int32_t absent[] = { 1, -1, 1000, 7 };
    const int32_t unchanged[] = { 1, -1, 1000, 7 };
    ok &= !cache->remap(absent, 4) && memcmp(absent, unchanged, sizeof(absent)) == 0;
    ok &= cache->lookups == 8 && cache->misses == 3;
    {
        std::ifstream saved(missed);
        std::string magic;
        int64_t saved_type, saved_width, saved_rows, saved_bytes, saved_count;
        saved >> magic >> saved_type >> saved_width >> saved_rows >> saved_bytes >> saved_count;
        std::vector<int32_t> exported;
        int32_t row;
        while (saved >> row) { exported.push_back(row); }
        ok &= magic == "LLAMA_PLE_ROWS_V1" && saved_type == type && saved_width == width &&
              saved_rows == n_rows && saved_bytes == (int64_t) row_bytes && saved_count == 4 &&
              exported == std::vector<int32_t>({1, 7, 8, 999});
        auto expanded = llama_ple_cache::load(original_table, missed.c_str(), false);
        int32_t replay[] = {7, 999, 8, 1};
        ok &= expanded->remap(replay, 4) && replay[0] == 1 && expanded->misses == 0;
    }
    ok &= mprotect(source, bytes, PROT_NONE) == 0;
    // IQ4_NL rows of width 160 gather on CPU, then the consumer runs on CUDA.
    auto * cpu = cuda ? ggml_backend_cpu_init() : backend;
    ggml_backend_t backends[] = {backend, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, cuda ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(ids);
    auto * gather = ggml_get_rows(ctx, table, ids);
    auto * output = cuda ? ggml_dup(ctx, gather) : gather;
    if (cuda) { ggml_backend_sched_set_tensor_backend(sched, output, backend); }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ok &= ggml_backend_sched_alloc_graph(sched, graph);
    ok &= ggml_backend_sched_get_tensor_backend(sched, gather) == cpu;
    ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
    ok &= ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(width * 4), expected(width);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (int row = 0; row < 4; ++row) {
        ggml_get_type_traits(type)->to_float(quant.data() + original[row] * row_bytes, expected.data(), width);
        ok &= memcmp(actual.data() + row * width, expected.data(), width * sizeof(float)) == 0;
    }
    ggml_backend_sched_free(sched);
    if (cuda) { ggml_backend_free(cpu); }
    cache.reset();
    ggml_backend_buffer_free(source_buffer);
    munmap(source, bytes);
    unlink(profile.c_str());
    unlink(missed.c_str());
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("PLE startup bytes, protected source, gather, misses and malformed profiles %s: %s\n", cuda ? "CUDA" : "CPU", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    (void)argc;
    (void)argv;
    return 77;
#endif
}
