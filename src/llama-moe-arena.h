#pragma once

#include <string>
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
    // RAM we can really use, in bytes (0 when unknown; 1 when a group limit is reached). The single reader for
    // every RAM gate: min(MemAvailable, cgroup v2 headroom), where the headroom is the smallest of
    // limit - memory.current + inactive_file + active_file over our group and all its parents
    // (limit = min(memory.max, memory.high)). With no group limit it is MemAvailable, as before.
    // LLAMA_RAM_AVAIL_SOURCE=meminfo skips the group check.
    static size_t available_memory();
    struct ram_view {
        uint64_t meminfo   = 0;          // MemAvailable
        uint64_t cgroup    = UINT64_MAX; // tightest group headroom; UINT64_MAX = no limit found
        uint64_t available = 0;          // min of the two
        std::string group;               // the group directory that binds, when one does
        uint64_t memtotal  = 0;          // MemTotal (0 when not found)
        uint64_t limit     = UINT64_MAX; // tightest min(memory.max, memory.high) on the path; UINT64_MAX = none
        uint64_t total     = 0;          // min(memtotal, limit): host RAM this process may use; 0 when unknown
    };
    // Pure reader behind available_memory(): meminfo_path is a /proc/meminfo, cgroup_root a /sys/fs/cgroup and
    // cgroup_rel the "0::" path of /proc/self/cgroup ("" skips the group check). Tests pass fake files.
    static ram_view read_ram_view(const std::string & meminfo_path, const std::string & cgroup_root,
            const std::string & cgroup_rel);
    // RAMSAFE: host RAM from the same reader, capped by the tightest cgroup v2 limit; 0 when unknown.
    static size_t total_memory();
    // RAMSAFE (opt-in): RAM left free when page-locked expert storage is admitted (the arena, and the in-place
    // locked mappings behind it). One sixth of RAM, held to 8..16 GiB: 64 GB -> about 10.4 GiB, 96 GB and up ->
    // 16 GiB. The 5 GiB stop is separate and unchanged.
    static size_t ram_headroom_for(uint64_t total) {
        constexpr uint64_t GiB = 1ull << 30;
        if (total == 0) { return size_t(7 * GiB); } // unknown host size: the fixed 7 GiB reserve
        const uint64_t share = total / 6;
        return size_t(share < 8 * GiB ? 8 * GiB : share > 16 * GiB ? 16 * GiB : share);
    }
    // The startup / pinning RAM headroom. Default OFF: a fixed 7 GiB, as before. LLAMA_MOE_RAM_HEADROOM=auto (or
    // =1) uses ram_headroom_for(total_memory()). LLAMA_MOE_RAM_HEADROOM_MIB=<MiB> overrides either (plain MiB,
    // never below the 5 GiB floor; 7168 is the fixed default).
    static size_t ram_headroom();
    // True when either variable above changes the headroom from the fixed 7 GiB path.
    static bool ram_headroom_custom();
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
