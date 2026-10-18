#include "llama-moe-arena.h"
#include "llama-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

llama_moe_output::llama_moe_output(size_t bytes, const char * directory) : size(bytes) {
#ifndef _WIN32
    std::string name = std::string(directory) + "/llama-moe-output-XXXXXX";
    std::vector<char> path(name.begin(), name.end());
    path.push_back(0);
    const int fd = mkstemp(path.data());
    if (fd < 0) { throw std::runtime_error("cannot create MoE temporary output file"); }
    unlink(path.data());
    bool reserved = bytes && bytes <= (size_t) INT64_MAX;
#ifdef __linux__
    reserved = reserved && posix_fallocate(fd, 0, bytes) == 0;
#else
    reserved = reserved && ftruncate(fd, bytes) == 0;
#endif
    if (!reserved) {
        close(fd);
        throw std::runtime_error("cannot reserve MoE temporary output file space");
    }
    data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { data = nullptr; throw std::runtime_error("cannot map MoE temporary output file"); }
    madvise(data, size, MADV_SEQUENTIAL);
#else
    (void)directory;
    throw std::runtime_error("MoE temporary output file requires mmap");
#endif
}

void llama_moe_output::flush() {
#ifndef _WIN32
    if (dirty && msync(data, size, MS_SYNC) != 0) {
        throw std::runtime_error("cannot flush MoE temporary output file");
    }
    dirty = false;
#endif
}

llama_moe_output::~llama_moe_output() {
#ifndef _WIN32
    if (data) { munmap(data, size); }
#endif
}

bool llama_moe_arena::is_expert(const char * name) {
    const char * suffixes[] = { ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" };
    const size_t length = strlen(name);
    for (const char * suffix : suffixes) {
        const size_t n = strlen(suffix);
        if (length >= n && strcmp(name + length - n, suffix) == 0) { return true; }
    }
    return false;
}

size_t llama_moe_arena::available_memory() {
#ifdef __linux__
    std::ifstream input("/proc/meminfo");
    std::string key, unit;
    size_t kib;
    while (input >> key >> kib >> unit) {
        if (key == "MemAvailable:" && kib <= SIZE_MAX / 1024) { return kib * 1024; }
    }
#endif
    return 0;
}

static ggml_backend_reg_t arena_cuda_reg() {
    return ggml_backend_reg_by_name("CUDA");
}

bool llama_moe_arena::pin_supported() {
#ifndef _WIN32
    auto * reg = arena_cuda_reg();
    return reg && ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
#else
    return false;
#endif
}

llama_moe_arena::llama_moe_arena(size_t bytes) : size(bytes) {
#ifndef _WIN32
    data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) { data = nullptr; throw std::bad_alloc(); }
    buffer.reset(ggml_backend_cpu_buffer_from_ptr(data, size));
    if (!buffer) { munmap(data, size); data = nullptr; throw std::bad_alloc(); }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#else
    throw std::runtime_error("expert arena requires anonymous mmap");
#endif
}

void llama_moe_arena::pin() {
    auto * reg = arena_cuda_reg();
    auto pin_fn = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
    unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
    if (!pin_fn || !unpin || !pin_fn(data, size)) {
        throw std::runtime_error("expert arena CUDA host registration failed");
    }
    pinned = true;
}

llama_moe_arena::~llama_moe_arena() {
    buffer.reset();
    if (pinned) { unpin(data); }
#ifndef _WIN32
    if (data) { munmap(data, size); }
#endif
}
