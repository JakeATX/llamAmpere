#pragma once

#include "ggml-cpp.h"
#include <cstddef>

// fork: one physical allocation, with strided views of unchanged expert blocks.
struct llama_moe_arena {
    static bool is_expert(const char * name);
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
