#pragma once

// CUDA-side helpers for the fallback ledger (ggml-ledger.h). Host code only.
//
// What the counts mean: CUDA graph outcomes ("cuda.graph") are counted on every graph compute.
// Everything else (mul_mat routes, flash-attention kernels, fusions) is counted where the host
// launches kernels, i.e. on eager runs and while a CUDA graph is being captured, never on a replay.
// So route counts are "kernels launched per captured or eager graph", and a route that shows up
// once per capture runs on every replay of that graph.

#include "ggml.h"
#include "ggml-ledger.h"

#include <cstdint>
#include <cstdio>
#include <unordered_map>

// Count one event at a hot site. `id` must tell the site's keys apart; `make_key(buf, size)` writes
// the key text and runs once per distinct id per thread, so a hit costs one hash lookup.
template <typename F>
static inline void ggml_cuda_ledger_count(const char * site, uint64_t id, F && make_key, int64_t n = 1) {
    if (!ggml_ledger_enabled()) {
        return;
    }
    thread_local std::unordered_map<uint64_t, ggml_ledger_slot *> cache;
    auto it = cache.find(id);
    if (it == cache.end()) {
        char key[192];
        make_key(key, sizeof(key));
        it = cache.emplace(id, ggml_ledger_slot_get(site, key)).first;
    }
    ggml_ledger_slot_add(it->second, n);
}

// splitmix64 step over (h, v): builds cache ids from several fields
static inline uint64_t ggml_cuda_ledger_mix(uint64_t h, uint64_t v) {
    uint64_t z = h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// verify widths 1-8 kept exact (MTP/n-gram verify), the rest bucketed
static inline int ggml_cuda_ledger_width_bucket(int64_t n) {
    if (n <= 8)   return (int) (n < 1 ? 1 : n);
    if (n <= 16)  return 16;
    if (n <= 64)  return 64;
    if (n <= 512) return 512;
    return 513;
}

static inline void ggml_cuda_ledger_width_str(int64_t n, char * buf, size_t size) {
    const int b = ggml_cuda_ledger_width_bucket(n);
    if (b <= 8) {
        snprintf(buf, size, "%d", b);
    } else if (b == 16) {
        snprintf(buf, size, "9-16");
    } else if (b == 64) {
        snprintf(buf, size, "17-64");
    } else if (b == 512) {
        snprintf(buf, size, "65-512");
    } else {
        snprintf(buf, size, ">512");
    }
}

// mul_mat / mul_mat_id route, keyed by the weight type as the model stores it and the batch width
static inline void ggml_cuda_ledger_mm(const char * site, const char * route, ggml_type type, int64_t width) {
    if (!ggml_ledger_enabled()) {
        return;
    }
    // site and route are literals, so their addresses identify them within this translation unit
    const uint64_t id = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
        (uint64_t) (uintptr_t) site, (uint64_t) (uintptr_t) route), (uint64_t) type), (uint64_t) ggml_cuda_ledger_width_bucket(width));
    ggml_cuda_ledger_count(site, id, [&](char * buf, size_t size) {
        char w[16];
        ggml_cuda_ledger_width_str(width, w, sizeof(w));
        snprintf(buf, size, "route=%s type=%s width=%s", route, ggml_type_name(type), w);
    });
}
