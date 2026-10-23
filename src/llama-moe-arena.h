#pragma once

#include "ggml-cpp.h"
#include "../include/llama.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Pure copy boundary for tracing only owned output-file refaults, after synchronization.
extern "C" LLAMA_API void llama_copy_moe_output(void * dst, const void * src, size_t bytes);
extern "C" LLAMA_API void llama_write_moe_output(void * dst, const void * src, size_t bytes);
extern "C" LLAMA_API int llama_compare_moe_output(const void * lhs, const void * rhs, size_t bytes);

// fork: reclaimable storage for a large, temporary MTP output, with unchanged floats.
struct llama_moe_output {
    llama_moe_output(size_t bytes, const char * directory);
    ~llama_moe_output();
    llama_moe_output(const llama_moe_output &) = delete;
    llama_moe_output & operator=(const llama_moe_output &) = delete;
    void flush();
    void release(size_t offset, size_t bytes);
    void readahead(size_t offset, size_t bytes); // [PREFILLFAST] async page-cache readahead hint
    void * data = nullptr;
    size_t size = 0;
    bool dirty = true;
private:
    int file = -1;
};

// fork: one expert tensor's copy into the arena: expert e of `expert_bytes` at file offset
// src_offset + e * expert_bytes lands at dst + e * dst_stride.
struct llama_moe_arena_copy {
    int fd = -1;               // buffered descriptor of the model file
    size_t src_offset = 0;
    size_t expert_bytes = 0;
    int64_t n_expert = 0;
    char * dst = nullptr;
    size_t dst_stride = 0;
};

struct llama_moe_arena_load_stats {
    double seconds = 0;
    size_t bytes_read = 0;
    int threads = 0;
    size_t chunk = 0;
    bool direct = false;
};

// fork (LOADFAST): parallel startup copy. Worker threads read aligned chunks straight from the
// file (O_DIRECT when the filesystem allows it, else buffered reads dropped from the page cache
// after use) and scatter each expert's bytes to its arena slot. The calling thread only reports
// progress and enforces the RAM floor. Bytes written are identical to the per-expert memcpy loop.
// progress(fraction of `bytes` done) returns false to cancel. Throws on any failure.
void llama_moe_arena_parallel_load(const std::vector<llama_moe_arena_copy> & copies,
        bool (*progress)(size_t done, void * user), void * user, size_t ram_floor,
        llama_moe_arena_load_stats & stats);

// fork (LOADFAST diagnostic): order-dependent 64-bit checksum of `bytes` at `data`, 8 threads.
uint64_t llama_moe_arena_checksum(const void * data, size_t bytes);

// fork (FNDEF): Flash-Next (arch qwen4exp) with the pinned expert arena on starts with the exact, measured-positive
// expert-cache, prompt-reading, kernel and helper switches already on. A switch the user set (any value, including
// 0) is never overwritten, so `NAME=0` still turns each one off. Called once, from the model loader, before any
// of the switches is read.
void llama_moe_arena_apply_flashnext_defaults();

// fork: one physical allocation, with strided views of unchanged expert blocks.
struct llama_moe_arena {
    static bool is_expert(const char * name);
    static bool is_trunk_expert(const char * name, uint32_t trunk_layers);
    static size_t available_memory();
    static bool pin_supported();

    explicit llama_moe_arena(size_t bytes);
    ~llama_moe_arena();
    void pin();
    size_t discard(const std::vector<std::pair<size_t, size_t>> & ranges);
    void pin_restored();

    void * data = nullptr;
    size_t size = 0;
    ggml_backend_buffer_ptr buffer;
private:
    std::vector<std::pair<size_t, size_t>> pin_ranges;
    std::vector<std::pair<size_t, size_t>> missing_ranges;
    bool (*pin_range)(void *, size_t) = nullptr;
    void (*unpin)(void *) = nullptr;
};
