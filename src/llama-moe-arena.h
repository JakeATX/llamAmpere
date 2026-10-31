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
    // Automatic expert storage owns half the 80% spare-RAM budget above 5 GiB.
    // Explicit arena settings retain the existing pinning admission checks.
    static size_t fit_memory_budget(uint64_t available) {
        constexpr uint64_t MiB = 1ull << 20;
        if (available <= (5ull << 30)) { return 0; }
        const uint64_t spare = (available - (5ull << 30)) / MiB;
        const uint64_t budget = (spare / 5) * 4 + (spare % 5) * 4 / 5;
        const uint64_t quota = budget / 2;
        return size_t(quota > SIZE_MAX / MiB ? SIZE_MAX / MiB : quota) * size_t(MiB);
    }
    static bool fits_ram_budget(size_t bytes, uint64_t available, bool fit, bool explicit_request) {
        return !fit || explicit_request || bytes <= fit_memory_budget(available);
    }
    static bool pin_supported();

    explicit llama_moe_arena(size_t bytes);
    ~llama_moe_arena();
    void pin();
    size_t discard(const std::vector<std::pair<size_t, size_t>> & ranges);
    void pin_restored();
    void report_thp() const;

    void * data = nullptr;
    size_t size = 0;
    ggml_backend_buffer_ptr buffer;
private:
    bool thp_requested = false;
    std::vector<std::pair<size_t, size_t>> pin_ranges;
    std::vector<std::pair<size_t, size_t>> missing_ranges;
    bool (*pin_range)(void *, size_t) = nullptr;
    void (*unpin)(void *) = nullptr;
    bool (*is_unpinned)(const void *, size_t) = nullptr;
};
