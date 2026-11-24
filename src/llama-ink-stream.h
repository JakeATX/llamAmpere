#pragma once

// [INKSTREAM] prompt expert streamer for host-expert MoE models (Inkling 2 Mini on 8/10/12 GB cards).
//
// With the experts in host RAM and op offload off, a prompt ubatch larger than the MoE cache batch cap computes its
// experts on the CPU. Op offload on cannot boot: the scheduler keeps a device copy of every layer's experts alive for
// the whole graph (~10 GB). The streamer keeps the normal full-model graph and gives it two VRAM expert slots instead:
//  - layer visit i reads its experts from slot (i % 2); a copy stream fills the slot of visit i + 1 from host RAM
//    (through pinned staging, several memcpy workers) while visit i computes, so PCIe overlaps compute;
//  - each visit's MoE ids go through a small host identity map (a host weight). The scheduler starts a split there and
//    calls its copy callback, which is where the compute stream waits for the slot and the next upload is queued;
//  - the VRAM comes from the MoE expert cache, which the context suspends for the prompt and refills (same experts,
//    same slots) before the next decode, plus a larger prompt ubatch funded by the same loan.
// Same weights, same kernels as the cached path: only where the expert bytes sit in VRAM changes.
// Kill switch: LLAMA_INK_STREAM=0.

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;

struct llama_ink_stream_stats {
    uint64_t bytes_h2d   = 0; // expert bytes uploaded
    uint64_t uploads     = 0; // visits uploaded
    int64_t  t_wait_us   = 0; // graph thread waiting for an upload to be queued
};

class llama_ink_stream {
public:
    llama_ink_stream(const llama_model & model, ggml_backend_t backend_gpu);
    ~llama_ink_stream();

    // the model has streamable host experts and the device has events
    bool ok() const { return init_ok; }
    // whether the arch and the weights allow the streamer, without building it (used before the weights exist)
    static bool wanted(const llama_model & model);

    bool armed() const { return is_armed; }
    // copy stream, events, staging and workers (made once); false: the streamer cannot run (it stays off)
    bool prepare();
    // allocate the two slots (prepare() first); false (nothing allocated) when VRAM is short
    bool arm();
    // wait for the copies in flight, then free all of it
    void disarm();

    size_t slot_bytes() const { return slot_size; }
    ggml_backend_t backend() const { return backend_gpu; }

    // identity map of visit il: an I32 [1, n_expert] host weight; nullptr when not armed or il is not streamed
    ggml_tensor * get_map(int il) const;
    // the device view of host expert tensor w in the slot of visit il, nullptr if w is not one of its experts
    ggml_tensor * get_experts(const ggml_tensor * w, int il) const;

    // ggml_backend_sched copy callback: true if src is one of the maps (then dst holds the map on return)
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst);

    llama_ink_stream_stats take_stats();

private:
    struct visit {
        std::vector<const ggml_tensor *> host;   // host expert tensors
        std::vector<ggml_tensor *>       views;  // device views in the slot of this visit (same order)
        std::vector<size_t>              offs;   // offset of each view in the slot
        std::vector<size_t>              tails;  // bytes after each tensor that the kernels may read: uploaded as zeros
        int slot = -1;                           // 0 or 1
        int pos  = -1;                           // index in order
        ggml_tensor * map = nullptr;
        bool   direct      = false;              // fork (GS11): host experts page-locked, queued without staging
        size_t direct_span = 0;                  // fork (GS11): > 0: the whole visit is one copy of this many bytes
    };

    struct piece {
        const char * src;
        size_t dst_offs;
        size_t len;
    };

    bool init_ok  = false;
    bool is_armed = false;
    bool pipeline_ok = false;

    ggml_backend_t             backend_gpu = nullptr;
    ggml_backend_dev_t         dev         = nullptr;
    ggml_backend_buffer_type_t buft_dev    = nullptr;

    std::vector<visit> visits;   // [n_layer], slot < 0 when the visit is not streamed
    std::vector<int>   order;    // streamed visits in graph order
    std::unordered_map<const ggml_tensor *, int> map_visit;
    std::vector<std::unordered_map<const ggml_tensor *, ggml_tensor *>> view_of; // [n_layer] host -> view
    size_t slot_size = 0;

    ggml_context * ctx_meta = nullptr;     // views and raw slot tensors
    ggml_context * ctx_map  = nullptr;     // maps
    ggml_backend_buffer_t buf_map  = nullptr;
    ggml_backend_buffer_t buf_slot = nullptr;
    ggml_tensor * raw[2] = { nullptr, nullptr };

    // copy pipeline
    ggml_backend_t backend_copy = nullptr;
    ggml_backend_event_t ev_free[2]  = { nullptr, nullptr };
    ggml_backend_event_t ev_ready[2] = { nullptr, nullptr };
    int issued[2] = { -1, -1 }; // visit last queued into each slot

    int    n_workers = 4;
    size_t piece_max = 8u << 20;
    bool   layer_major_reuse = false; // see copy(): no wait for the other slot when a visit repeats (INKB1)
    struct worker_state {
        ggml_backend_buffer_t buf[2] = { nullptr, nullptr };
        char *                data[2] = { nullptr, nullptr };
        ggml_backend_event_t  ev[2] = { nullptr, nullptr };
        bool                  used[2] = { false, false };
        int                   next = 0;
    };
    std::vector<worker_state> wst;
    std::vector<std::thread>  threads;

    std::mutex mtx;            // request state
    std::condition_variable cv;
    std::mutex mtx_stream;     // copy stream calls
    std::vector<piece> pieces;
    int      req_slot = -1;
    uint64_t req_gen  = 0;     // incremented per request
    uint64_t done_gen = 0;     // last request whose ready event is recorded
    std::atomic<size_t> next_piece{0};
    size_t   pieces_done = 0;
    bool     stop = false;
    int      n_busy = 0;       // workers on a request

    llama_ink_stream_stats st;

    // fork (INKMISS, GS11; LLAMA_INK_STREAM_DIRECT=1): direct uploads from page-locked host experts
    bool direct_on = false;
    ggml_backend_buffer_t buf_zeros = nullptr; // pinned zeros for the tails
    size_t n_direct = 0, n_direct_span = 0;

    void issue(int il, ggml_backend_t backend);
    void wait_issued();
    void worker(int w);
    void free_slots();
    void free_pipeline();
};
