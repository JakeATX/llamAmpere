#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Registers the provider API with the backend registry (see ggml-backend-moe-cache.h).

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_moe_cache_register(void * reg);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}
// Bound only clean file pages that the kernel can reclaim from a service.
static inline uint64_t ggml_cuda_moe_cache_host_file_credit(uint64_t file, uint64_t inactive,
        uint64_t shmem, uint64_t dirty, uint64_t writeback, uint64_t registered) {
    file -= shmem < file ? shmem : file;
    file -= dirty < file ? dirty : file;
    file -= writeback < file ? writeback : file;
    const uint64_t clean = inactive < file ? inactive : file;
    return clean - (registered < clean ? registered : clean);
}

static inline bool ggml_cuda_moe_cache_host_merge(uintptr_t begin, uintptr_t end,
        uintptr_t next_begin, uintptr_t next_end, size_t cap) {
    return next_begin <= end && next_end >= begin && next_end - begin <= cap;
}

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_cuda_context;
// A nonzero token requires an active prepared session and opt-in capture.
uint64_t ggml_cuda_moe_cache_device_capture_generation(int device);
void ggml_cuda_moe_cache_device_graph_record(int device, bool captured, const ggml_cgraph * graph);
bool ggml_cuda_moe_cache_device_supports(const ggml_tensor * op, int device);
bool ggml_cuda_moe_cache_device_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op);
bool ggml_cuda_moe_cache_device_fused_compute(ggml_backend_cuda_context & ctx,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * glu);
#endif
