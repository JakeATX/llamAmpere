#pragma once

// fork (R4, NEXTPF): next-layer expert upload during attention. Off by default.
//
// LLAMA_MOE_CACHE_NEXT_PREFETCH=K: at the MoE cache lookup of layer il in a decode ubatch (<= 8 rows), up to K experts that
// layer il+1 (same cache group) routed to in its last decode ubatch and that are no longer cached get slots and are
// uploaded at once, from a worker thread with its own stream, so the copy runs while layer il's experts and layer il+1's
// attention compute. The next lookup waits for the worker before it plans. Prediction = the next layer's last routing
// (temporal reuse; no router lookahead). Victims never take a slot of the current layer's selected experts, or of a
// predicted expert. Weights in a slot are the same bytes, so numerics are unchanged as long as no misses run on the
// CPU (with a CPU share, a prefetched expert runs on the device instead of the CPU: rounding differs).
// Off when round admission (LLAMA_MOE_CACHE_ROUND) or MCDET is on.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;

class llama_moe_prefetch {
public:
    struct job {
        ggml_tensor * dst;    // bank
        const void  * src;    // host expert bytes
        size_t        offset; // in dst
        size_t        size;
    };

    // LLAMA_MOE_CACHE_NEXT_PREFETCH clamped to [0, 64], 0 when unset
    static int32_t env_budget();
    // nullptr unless env_budget() > 0
    static std::unique_ptr<llama_moe_prefetch> from_env(int32_t n_layer);

    llama_moe_prefetch(int32_t n_layer, int32_t budget);
    ~llama_moe_prefetch(); // waits for the worker, then stops it

    int32_t budget() const { return n_budget; }

    // the distinct experts of a decode ubatch of layer il, sorted (replaces the previous record)
    void record(int32_t il, const int32_t * ids, size_t n_ids);
    const std::vector<int32_t> & predicted(int32_t il) const { return last[il]; }

    // the experts prefetched for layer il; account() counts how many of them the next lookup of il selects
    void mark(int32_t il, const std::vector<int32_t> & experts);
    void account(int32_t il, const int32_t * ids, size_t n_ids);

    // run the uploads on the worker (ggml_backend_tensor_set: the buffer's own synchronous path, off the backend stream)
    void submit(std::vector<job> && jobs);
    // until every submitted job is done
    void wait();

    size_t n_prefetched = 0; // experts uploaded ahead
    size_t n_used       = 0; // of them, selected by the next lookup of their layer
    size_t n_bytes      = 0;
    size_t n_rounds     = 0; // submit() calls

private:
    void loop();

    int32_t n_budget;
    std::vector<std::vector<int32_t>> last;    // [n_layer]
    std::vector<std::vector<int32_t>> marked;  // [n_layer]
    std::vector<int32_t> scratch;

    std::mutex              mu;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::vector<job>        queue;
    bool                    busy = false;
    bool                    stop = false;
    std::thread             worker;
};
