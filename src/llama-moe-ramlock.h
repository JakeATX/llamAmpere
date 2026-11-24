#pragma once

// fork (#130, RAMLOCK): ranked RAM budget for host experts on low-RAM machines. Off by default.
//
// LLAMA_MOE_RAM_LOCK_GIB=N locks the hottest N GiB of the MoE cache's host experts in RAM (mlock), so the OS page
// cache cannot drop them when the model file is larger than RAM; the other experts stay file-backed and are read in
// place as before. Heat = per-(layer, expert) routing counts seen by the MoE cache lookups, halved at every re-rank.
//   LLAMA_MOE_RAM_LOCK_EVERY=K   re-rank after K cache lookups (default 4096; one lookup = one layer of one ubatch)
//   LLAMA_MOE_RAM_LOCK_PROFILE=F start from the counts in text file F ("layer expert count" per line) and lock at once
//   LLAMA_MOE_RAM_LOCK_SAVE=F    write the counts to F at exit (same format)
// An mlock failure (RLIMIT_MEMLOCK, ENOMEM) logs one warning and turns the lock off; nothing else changes.
// Numerics are untouched: only page residency changes.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

class llama_moe_ramlock {
public:
    // nullptr unless LLAMA_MOE_RAM_LOCK_GIB > 0
    static std::unique_ptr<llama_moe_ramlock> from_env(int32_t n_layer, int32_t n_expert);
    ~llama_moe_ramlock();

    // a physical (canonical) layer and its host expert tensors (gate/up/down)
    void add_layer(int32_t il, const std::vector<ggml_tensor *> & experts);
    // call once all layers are added: applies a loaded profile
    void start();
    // the experts one lookup of layer il selected
    void observe(int32_t il, const int32_t * ids, size_t n_ids);

private:
    llama_moe_ramlock() = default;
    void rerank();
    bool lock_one(int32_t key, bool lock);

    size_t budget = 0;
    int64_t every = 4096;
    int64_t n_obs = 0;
    int32_t n_layer = 0;
    int32_t n_expert = 0;
    bool failed = false;
    std::string save_path;
    std::string profile_path;
    std::vector<double> heat;                         // [n_layer*n_expert]
    std::vector<uint8_t> locked;                      // [n_layer*n_expert]
    std::vector<std::vector<ggml_tensor *>> tensors;  // [n_layer], empty = not tracked
    size_t locked_bytes = 0;
    size_t n_lock_calls = 0;
    size_t n_rerank = 0;
};
