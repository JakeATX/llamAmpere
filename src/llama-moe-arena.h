#pragma once

#include "ggml-cpp.h"
#include "../include/llama.h"
#include <cstddef>
#include <cstdint>

// Pure copy boundary for tracing only owned output-file refaults, after synchronization.
extern "C" LLAMA_API void llama_copy_moe_output(void * dst, const void * src, size_t bytes);

// fork: reclaimable storage for a large, temporary MTP output, with unchanged floats.
struct llama_moe_output {
    llama_moe_output(size_t bytes, const char * directory);
    ~llama_moe_output();
    llama_moe_output(const llama_moe_output &) = delete;
    llama_moe_output & operator=(const llama_moe_output &) = delete;
    void flush();
    void release(size_t offset, size_t bytes);
    void * data = nullptr;
    size_t size = 0;
    bool dirty = true;
private:
    int file = -1;
};

// fork: one physical allocation, with strided views of unchanged expert blocks.
struct llama_moe_arena {
    static bool is_expert(const char * name);
    static bool is_trunk_expert(const char * name, uint32_t trunk_layers);
    static size_t available_memory();
    static bool pin_supported();

    explicit llama_moe_arena(size_t bytes);
    ~llama_moe_arena();
    void pin();

    void * data = nullptr;
    size_t size = 0;
    ggml_backend_buffer_ptr buffer;
private:
    bool pinned = false;
    void (*unpin)(void *) = nullptr;
};
