#pragma once

#include "llama-moe-arena.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;

// Compact, startup-only copies of unchanged PLE rows. No request-time disk fallback.
struct llama_ple_cache {
    static std::unique_ptr<llama_ple_cache> load(const ggml_tensor * tensor, const char * profile, bool pin = true);
    bool remap(int32_t * indices, size_t count) const;
    ggml_tensor * tensor = nullptr;
    size_t size() const { return storage->size; }
    size_t count() const { return rows.size(); }
    mutable std::atomic<uint64_t> lookups{0};
    mutable std::atomic<uint64_t> misses{0};

private:
    void save_miss_profile(const int32_t * indices, size_t count) const;
    int64_t full_rows = 0;
    std::string miss_profile;
    mutable std::mutex miss_mutex;
    ggml_context_ptr context;
    std::vector<int32_t> rows;
    std::unique_ptr<llama_moe_arena> storage;
};
