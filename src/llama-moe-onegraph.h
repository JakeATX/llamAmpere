#pragma once

// fork (ONEGRAPH, backlog #101; LLAMA_MOE_ONEGRAPH=1, off by default): GPU-anchored decode lookups for the MoE cache.
// The decode graph looks the selected experts up in a device copy of the slot maps (ggml_moe_og_slots) instead of a
// host slot map, so the scheduler starts no split at the lookup and the host does not synchronize the compute stream
// per MoE layer. A host worker thread reads each lookup's ids from a pinned ring (doorbell), runs the cache's LRU plan
// and, on a miss, answers with the copies (registered host experts -> bank slots) and slot map changes, which the GPU
// carries out. The worker makes no CUDA call (it could block on a driver lock while the GPU waits for it).
// Protocol: ggml/include/ggml-moe-onegraph.h.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-onegraph.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

struct llama_moe_og {
    // serve(rec, copies): runs on the worker under mu. It plans the lookup and appends the copies of the missing experts.
    // Returns 1 if it changed the slot maps, else 0.
    using serve_fn    = std::function<int(const ggml_moe_og_rec & rec, std::vector<ggml_moe_og_copy> & copies)>;
    // snapshot(table): the current slot maps, [n_rows][n_expert], under mu
    using snapshot_fn = std::function<void(int32_t * table)>;

    // LLAMA_MOE_ONEGRAPH=1 (read once); =2 is a diagnostic: the host experts are page-locked as for 1, but the
    // lookups stay on the regular path (separates the gain of the page-locked uploads from the GPU-side lookups)
    static bool requested();
    static int  mode();

    // nullptr-safe factory: returns nullptr (and logs why) if the device cannot run it
    // pins: the host expert ranges the GPU copies from (page-locked read-only and mapped here)
    static llama_moe_og * create(ggml_backend_t main, ggml_backend_buffer_type_t buft, int32_t n_expert, int32_t n_rows,
                                 const std::vector<std::pair<void *, size_t>> & pins, serve_fn serve, snapshot_fn snapshot);
    ~llama_moe_og();

    // the lookup node of layer visit `layer` (map row `row`) for ids [n_ids]
    ggml_tensor * slots(ggml_context * ctx, ggml_tensor * ids, int32_t row, int32_t layer, int32_t n_slots, int32_t n_used) const;

    // before every graph of the context: abort on a protocol error, upload the table if the maps changed, wake the worker
    void prepare();
    // the compute stream is idle: wait until the worker served every published lookup and its uploads finished
    void quiesce();
    // the maps were changed outside the worker (the regular lookup path, resize, resume)
    void mark_dirty() { dirty = true; }

    std::mutex mu; // the cache's LRU and slot maps (worker vs. the regular paths)

    // counters for the log
    std::atomic<uint64_t> n_records{0}, n_waits{0}, n_lag{0};

private:
    llama_moe_og() = default;
    void run();

    ggml_backend_t main_backend = nullptr;
    void (*unpin)(void *) = nullptr;
    std::vector<void *> pinned;
    ggml_context * ctx_dev = nullptr;
    ggml_backend_buffer_t buf_dev  = nullptr;
    ggml_backend_buffer_t buf_ring = nullptr;
    ggml_tensor * map  = nullptr; // I32 [n_expert, n_rows]
    ggml_tensor * ctrl = nullptr; // ggml-moe-onegraph.h ctrl words
    ggml_moe_og_ring * ring = nullptr;
    int32_t n_expert = 0, n_rows = 0;
    int32_t timeout_ms = 10000;
    serve_fn    serve;
    snapshot_fn snapshot;
    std::vector<int32_t> table;       // the device table as the GPU has it (after the answered changes)
    std::vector<int32_t> table_new;
    std::vector<ggml_moe_og_copy> copies;
    std::vector<std::pair<int32_t, int32_t>> upd;
    std::atomic<bool> dirty{true};
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> processed{0};
    std::atomic<int64_t>  active_until_us{0};
    std::mutex cv_mu;
    std::condition_variable cv;
    std::thread worker;
};
