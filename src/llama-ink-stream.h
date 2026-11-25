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
class  llama_moe_cache;

struct llama_ink_stream_stats {
    uint64_t bytes_h2d   = 0; // expert bytes uploaded
    uint64_t bytes_d2d   = 0; // fork (PLOAN): expert bytes copied from the kept MoE cache banks
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
    // fork (CORERES): place up to LLAMA_INK_CORERES resident core layers in the VRAM that is still free once the prompt
    // call has sized its buffers (ubatch ceiling, B1 groups), leaving keep bytes free. The ubatch size and grouping never
    // depend on them, so the values are those of the two-slot stream. Returns the number placed. Call once per decode
    // call of an armed prompt, before its first graph; drop_residents() before the next call measures VRAM.
    int place_residents(size_t keep);
    // wait for the copies in flight and give the resident regions back (their visits stream again)
    void drop_residents();
    int residents() const { return n_res; }

    size_t slot_bytes() const { return slot_size; }
    ggml_backend_t backend() const { return backend_gpu; }

    // identity map of visit il: an I32 [1, n_expert] host weight; nullptr when not armed or il is not streamed
    ggml_tensor * get_map(int il) const;
    // the device view of host expert tensor w in the slot of visit il, nullptr if w is not one of its experts
    ggml_tensor * get_experts(const ggml_tensor * w, int il) const;

    // ggml_backend_sched copy callback: true if src is one of the maps (then dst holds the map on return)
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst);

    llama_ink_stream_stats take_stats();

    // fork (PLOAN, LLAMA_INK_PLOAN=1): while the MoE cache keeps a mapped prefix of its banks (suspend_keep), a visit
    // copies the experts it holds device to device and uploads only the others; the refill may write VMM banks
    void set_cache(const llama_moe_cache * cache);

    // fork (REFILLPIPE): one host -> device copy for upload_wait()
    struct upload {
        ggml_tensor * dst;
        size_t        dst_offs;
        const void  * src;
        size_t        len;
    };
    // copy the list through the pinned staging workers and the copy stream and wait until it has landed (the MoE cache
    // refill after a streamed prompt, whose plain path is one synchronous pageable copy per run of experts). Only while
    // not armed. false: the pipeline cannot run or a destination is not on this device (nothing copied).
    bool upload_wait(const std::vector<upload> & list);

private:
    struct visit {
        std::vector<const ggml_tensor *> host;   // host expert tensors
        std::vector<ggml_tensor *>       views;  // device views in the slot of this visit (same order)
        std::vector<size_t>              offs;   // offset of each view in the slot
        std::vector<size_t>              tails;  // bytes after each tensor that the kernels may read: uploaded as zeros
        int slot = -1;                           // 0 or 1 (streamed), 2 + grp (CORERES: resident for the prompt)
        int grp  = -1;                           // CORERES: index of its physical core layer (visited more than once)
        int pos  = -1;                           // index in order
        ggml_tensor * map = nullptr;
        bool   direct      = false;              // fork (GS11): host experts page-locked, queued without staging
        size_t direct_span = 0;                  // fork (GS11): > 0: the whole visit is one copy of this many bytes
    };

    struct piece {
        const char * src;
        size_t dst_offs;
        size_t len;
        ggml_tensor * dst = nullptr; // nullptr: the slot of the request (req_slot); else upload_wait()
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
    ggml_backend_buffer_t buf_res  = nullptr; // CORERES regions of this decode call
    std::vector<ggml_tensor *> raw;        // [2 + n_core]: the two streamed slots, then one region per resident core layer

    // copy pipeline
    ggml_backend_t backend_copy = nullptr;
    std::vector<ggml_backend_event_t> ev_free;  // [2 + n_core]
    std::vector<ggml_backend_event_t> ev_ready; // [2 + n_core]
    ggml_backend_event_t ev_list = nullptr; // upload_wait(): the last piece is on the card
    std::vector<int> issued;   // [2 + n_core] visit last queued into each slot (a resident region: any visit, once per call)
    // fork (CORERES): LLAMA_INK_CORERES=N keeps the experts of N core layers (the layers a looped model visits on every
    // pass) in VRAM for the whole armed prompt, uploaded once instead of on every pass; 0 (default) = two slots only
    int n_core   = 0;          // physical layers visited more than once
    int res_want = 0;
    int n_res    = 0;          // resident regions of this decode call (placed after its buffers, see place_residents)
    std::vector<int> res_region; // [n_core] region of each core layer for this call, -1 = streamed

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

    // fork (PLOAN)
    const llama_moe_cache * cache = nullptr;
    ggml_context * ctx_d2d = nullptr;               // views of the kept-expert copies of one visit
    ggml_backend_buffer_type_t buft_elastic = nullptr; // the VMM bank buffer type (upload_wait destinations)

    void issue(int il, ggml_backend_t backend);
    bool issue_kept(int il); // fork (PLOAN): false = no kept experts in this visit (nothing queued)
    void wait_issued();
    void worker(int w);
    void free_slots();
    void free_pipeline();
    // streamed visits alternate between the two slots in graph order, keeping the slot of an upload already queued for
    // the first one (the prefetch from the previous call); views follow their visit's slot
    void assign_slots();
    bool place_views(int il);
};
