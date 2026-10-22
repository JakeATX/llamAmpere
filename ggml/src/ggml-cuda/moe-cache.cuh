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

// fork: pure host planner for global profile bundles; no CUDA state or calls.
#include <algorithm>
#include <utility>
#include <vector>

struct ggml_cuda_moe_profile_item {
    size_t tensor;
    int32_t expert;
    size_t group;
    size_t pool;
    size_t bytes;
    uint64_t count;
};

struct ggml_cuda_moe_profile_plan {
    std::vector<size_t> slots;
    std::vector<size_t> ranks;
    std::vector<uint8_t> selected;
    size_t bytes = 0;
    size_t bundles = 0;
};

static inline bool ggml_cuda_moe_cache_profile_plan(const std::vector<ggml_cuda_moe_profile_item> & items,
        const std::vector<size_t> & caps, size_t budget, size_t minimum,
        ggml_cuda_moe_profile_plan & output, bool route_density = false) {
    if (items.empty() || caps.empty() || !minimum) {
        return false;
    }
    struct bundle {
        std::vector<size_t> members;
        std::vector<std::pair<size_t, size_t>> pools;
        size_t bytes = 0;
        long double utility = 0;
    };
    std::vector<size_t> order(items.size());
    std::vector<size_t> pool_bytes(caps.size(), 0);
    for (size_t i = 0; i < items.size(); ++i) {
        const auto & item = items[i];
        if (!item.bytes || item.expert < 0 || item.pool >= caps.size() ||
                (pool_bytes[item.pool] && pool_bytes[item.pool] != item.bytes)) {
            return false;
        }
        pool_bytes[item.pool] = item.bytes;
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (items[a].group != items[b].group) { return items[a].group < items[b].group; }
        if (items[a].expert != items[b].expert) { return items[a].expert < items[b].expert; }
        return items[a].tensor < items[b].tensor;
    });
    std::vector<bundle> bundles;
    size_t previous = SIZE_MAX;
    bool observed = false;
    for (size_t index : order) {
        const auto & item = items[index];
        if (previous == SIZE_MAX || item.group != items[previous].group || item.expert != items[previous].expert) {
            bundles.emplace_back();
        } else if (item.tensor == items[previous].tensor) {
            return false;
        }
        auto & b = bundles.back();
        if (item.bytes > SIZE_MAX - b.bytes) { return false; }
        b.bytes += item.bytes;
        b.utility += route_density ? (long double) item.count : (long double) item.count * item.bytes;
        b.members.push_back(index);
        auto found = std::find_if(b.pools.begin(), b.pools.end(), [&](const std::pair<size_t, size_t> & p) {
            return p.first == item.pool;
        });
        if (found == b.pools.end()) { b.pools.push_back({item.pool, 1}); }
        else { ++found->second; }
        observed = observed || item.count != 0;
        previous = index;
    }
    if (!observed) { return false; }
    std::vector<size_t> ranked(bundles.size());
    for (size_t i = 0; i < ranked.size(); ++i) { ranked[i] = i; }
    std::stable_sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) {
        if (route_density) {
            return bundles[a].utility / bundles[a].members.size() / bundles[a].bytes >
                bundles[b].utility / bundles[b].members.size() / bundles[b].bytes;
        }
        return bundles[a].utility / bundles[a].bytes > bundles[b].utility / bundles[b].bytes;
    });
    ggml_cuda_moe_profile_plan plan;
    plan.slots.assign(caps.size(), 0);
    plan.ranks.resize(items.size());
    plan.selected.assign(items.size(), 0);
    std::vector<uint8_t> chosen(bundles.size(), 0);
    auto admit = [&](size_t index) {
        const auto & b = bundles[index];
        if (b.bytes > budget - plan.bytes) { return false; }
        for (const auto & p : b.pools) {
            if (p.second > caps[p.first] - plan.slots[p.first]) { return false; }
        }
        plan.bytes += b.bytes;
        chosen[index] = 1;
        for (const auto & p : b.pools) { plan.slots[p.first] += p.second; }
        for (size_t member : b.members) { plan.selected[member] = 1; }
        ++plan.bundles;
        return true;
    };
    for (size_t index : ranked) {
        bool needed = false;
        for (const auto & p : bundles[index].pools) { needed = needed || plan.slots[p.first] < minimum; }
        if (needed) { admit(index); }
        bool complete = true;
        for (size_t count : plan.slots) { complete = complete && count >= minimum; }
        if (complete) { break; }
    }
    for (size_t count : plan.slots) { if (count < minimum) { return false; } }
    for (size_t index : ranked) { if (!chosen[index]) { admit(index); } }
    size_t rank = 0;
    for (int selected = 1; selected >= 0; --selected) {
        for (size_t index : ranked) {
            if (chosen[index] == selected) {
                for (size_t member : bundles[index].members) { plan.ranks[member] = rank++; }
            }
        }
    }
    output = std::move(plan);
    return true;
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
// fork PREFETCHFIX (GGML_MOE_PREFETCH_ASYNC): turns a "moe_prefetch-<src>-<target>" id list into
// an on-device prediction record; the copy engine is fed from it without stalling the stream.
bool ggml_cuda_moe_cache_prefetch_async_active(void);
void ggml_cuda_moe_cache_device_predict(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
// fork PARFIX: joins miss-join side copies a graph left pending; called before a graph evaluation ends.
void ggml_cuda_moe_cache_hf_graph_end(ggml_backend_cuda_context & ctx);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Construction/eager-eval only. Cached graph replay must not invoke these host hooks.
// Unsupported concurrent execution is rejected when an actual table dispatch occurs;
// non-MoE/draft graphs may enter/end without issuing a table branch.
void ggml_cuda_moe_cache_table_graph_begin(ggml_backend_cuda_context & ctx, bool allow_single_stream);
void ggml_cuda_moe_cache_table_graph_end(ggml_backend_cuda_context & ctx);
#endif
#endif
