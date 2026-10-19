#pragma once

// [B1] layer-major prefill (LLAMA_B1=1, opt-in)
//
// The normal prefill runs every layer for one ubatch before it moves to the next ubatch, so each expert that is not
// resident in VRAM crosses PCIe once per ubatch. B1 runs one layer over a group of ubatches before it moves to the
// next layer, so every expert crosses PCIe once per group. llama_b1 owns the VRAM that makes this possible:
//  - two expert slots: the copy thread fills the slot of layer L+1 while layer L computes from the other one;
//  - one carry tensor per ubatch of the group: the hidden state one layer hands to the next.
// That VRAM comes from free memory first, then from the resident expert layers it borrows ("lends"); a lent layer is
// streamed like any other, and its weights are copied back ("refill") before the next normal decode.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama-model.h"

struct llm_b1_ffn;

struct llama_b1_stats {
    double   t_copy_submit_ms = 0.0; // overlapped copy pipeline host wall time
    double   t_wait_ms    = 0.0; // compute thread waiting on the copy thread (host side)
    uint64_t bytes_h2d    = 0;   // expert bytes streamed to the GPU in this decode call
    uint64_t bytes_d2d    = 0;   // original permanent expert bytes copied within the GPU
    uint64_t bytes_lent   = 0;   // resident expert bytes borrowed
    uint64_t bytes_fresh  = 0;   // free VRAM allocated
    uint64_t bytes_carry  = 0;
    int64_t  t_stream_us  = 0;   // first layer_begin to end()
};

class llama_b1 {
public:
    llama_b1(const llama_model & model, ggml_backend_t backend_gpu);
    ~llama_b1();

    bool ok() const { return init_ok; }
    void set_ffn_enabled(bool enabled) { ffn_requested = enabled; }
    bool ffn_enabled() const { return !ffn_views.empty(); }
    llm_b1_ffn * ffn(size_t ub);
    llm_b1_ffn * ffn_group() { return ffn_all.get(); }
    bool ffn_gather();
    void ffn_release();
    const std::vector<uint32_t> & ubatch_sizes() const { return ub_n; }
    size_t scratch_bytes(const std::vector<uint32_t> & ub_tokens, uint32_t group_ub) const;
    void pass_begin(size_t ub);
    void pass_compute(size_t ub, int64_t host_us);
    void pass_end(size_t ub);

    // plan one decode call: ub_tokens are the ubatch sizes, group_ub the wanted ubatches per group (0 = all),
    // lend_cap the most resident expert bytes it may borrow. final_home: the last layer's experts that live in RAM
    // stay there (its FFN sees only the output rows, which the normal path computes on the CPU when they are few).
    // Returns false when even one ubatch per group does not fit.
    bool begin(const std::vector<uint32_t> & ub_tokens, uint32_t group_ub, size_t lend_cap, bool final_home = false, size_t * borrow_bytes = nullptr, bool defer_streaming = false);
    // A deferred plan owns real carry/slot storage but has no workers or transfers.
    // Bind one layer only for graph-size measurement, then restore it before starting.
    void reserve_layer_bind(int il);
    void reserve_layer_restore();
    void start_streaming();

    // would the normal path run an expert op with this many rows on the GPU when its weights are in RAM? The
    // scheduler copies such weights to the GPU only for batches the device takes (op offload: 32 rows on CUDA)
    bool host_rows_on_gpu(int64_t rows) const;
    // for one ubatch of the current layer il: point its streamed experts that live in RAM back home (true), so the
    // scheduler makes the same CPU choice as the normal path, or back at the slot (false)
    void ubatch_home(int il, bool home);

    size_t n_ub()     const { return ub_n.size(); }
    size_t n_groups() const { return grp_first.size(); }
    size_t group_ub() const { return g_ub; }
    bool   is_group_first(size_t ub) const { return ub % g_ub == 0; }
    bool   is_group_last (size_t ub) const { return ub % g_ub == g_ub - 1 || ub + 1 == ub_n.size(); }
    size_t group_end(size_t ub) const { return std::min(ub_n.size(), (ub / g_ub + 1) * g_ub); }

    // the carry tensor of ubatch ub (F32 [n_embd, hc, n_tokens of ub])
    ggml_tensor * carry(size_t ub);

    // the layer ranges of the plan: every streamed layer is its own range, a run of layers that stay resident is
    // one range; the last range [final_l0(), n_layer) is built with the model output by the normal decode body
    bool    layer_streamed(int il) const;
    int32_t final_l0() const;
    int32_t range_end(int32_t l0) const;

    // layer il of the current group is about to run: wait for its experts and point the model tensors at them
    bool layer_begin(int il);
    // the last graph of layer il in the current group was queued: restore the tensors, release the slot
    void layer_end(int il);

    // the decode call finished (or failed): stop the copy thread, free the fresh VRAM, keep lent layers marked dirty
    void end(bool failed);

    // copy the lent layers back; a no-op when nothing is dirty. Returns false on an I/O error.
    bool refill();
    bool dirty() const { return !dirty_tensors.empty(); }

    const llama_b1_stats & stats() const { return st; }

private:
    struct exps_tensor {
        ggml_tensor *         t       = nullptr;
        int                   role    = -1;   // index into roles
        size_t                nbytes  = 0;
        size_t                home_nb2 = 0;
        size_t                alloc   = 0;    // device alloc size (with the tail pad)
        ggml_backend_buffer_t home_buf  = nullptr;
        void *                home_data = nullptr;
        bool                  resident  = false; // home is VRAM
        llama_b1_file_src     src;
    };

    struct layer_info {
        std::vector<int> tensors; // indices into exps
        bool   resident = false;  // every exps tensor of the layer is in VRAM
        size_t bytes    = 0;
    };

    // a contiguous VRAM range the plan may place slots and carries in
    struct region {
        ggml_backend_buffer_t buf  = nullptr;
        char *                base = nullptr;
        size_t                size = 0;
        size_t                used = 0;
        ggml_tensor *         raw  = nullptr; // I8 view over the range, the target of the copies
    };

    struct placement {
        int    region = -1;
        size_t offs   = 0;
    };

    // one streamed layer of the decode call's global sequence (group-major, then layer)
    struct step {
        int il   = -1;
        int slot = -1;
    };

    const llama_model & model;
    ggml_backend_t      backend_gpu  = nullptr;
    ggml_backend_dev_t  dev          = nullptr;
    ggml_backend_t      backend_copy = nullptr;
    ggml_backend_buffer_type_t buft_dev = nullptr;
    bool init_ok = false;

    int n_layer = 0;
    int64_t n_embd = 0;
    int64_t n_hc   = 0;

    std::vector<std::string> roles;
    std::vector<size_t>      role_max; // max alloc size per role
    std::vector<exps_tensor> exps;
    std::vector<layer_info>  layers;

    // plan of the current call
    std::vector<uint32_t> ub_n;
    std::vector<size_t>   grp_first;
    size_t g_ub = 1;
    std::vector<region>   regions;
    ggml_backend_buffer_t fresh_buf = nullptr;
    std::vector<placement> slot_place; // [slot*n_roles + role]
    std::vector<placement> carry_place; // [ubatch index within the group]
    ggml_context * ctx_meta = nullptr;
    std::vector<ggml_tensor *> carries; // per ubatch of the call
    bool ffn_requested = false;
    std::vector<ggml_backend_buffer_t> ffn_buffers;
    std::unique_ptr<llm_b1_ffn> ffn_all;
    std::vector<std::unique_ptr<llm_b1_ffn>> ffn_views;
    std::vector<bool> lent_layer;
    bool final_home_ = false; // see begin()
    std::vector<bool> verified;        // exps index: device bytes checked against the file once
    std::vector<bool> dirty_flag;      // exps index: home VRAM holds borrowed data
    std::vector<int>  dirty_tensors;   // exps indices whose home VRAM holds borrowed data
    std::vector<step> steps;
    size_t step_next_begin = 0; // next step layer_begin expects
    int    cur_step  = -1;
    int    reserve_layer = -1;

    // copy pipeline
    struct staging {
        ggml_backend_buffer_t buf  = nullptr;
        char *                data = nullptr;
        ggml_backend_event_t  ev   = nullptr;
        bool                  used = false;
    };
    struct piece {
        const char * src_host = nullptr; // memcpy source, or null for pread
        int          fd       = -1;
        size_t       src_offs = 0;
        region *     dst      = nullptr;
        size_t       dst_offs = 0;
        size_t       len      = 0;
    };

    size_t piece_bytes = 16u << 20;
    int    n_workers   = 8;
    std::vector<std::vector<staging>> stg; // [worker][2]
    ggml_backend_buffer_t zero_buf = nullptr;
    char * zero_data = nullptr;
    size_t zero_size = 0;
    ggml_backend_event_t ev_ready[2] = { nullptr, nullptr };
    ggml_backend_event_t ev_free[2]  = { nullptr, nullptr };

    std::mutex mtx;            // guards the queues and counters below
    std::condition_variable cv;
    std::mutex mtx_stream;     // serializes the copy stream calls
    std::vector<piece> queue;
    size_t queue_head = 0;
    size_t pieces_left = 0;
    size_t n_ready = 0;  // steps whose ready event was recorded
    size_t n_freed = 0;  // steps whose free event was recorded
    bool   stop    = false;
    bool   io_error = false;
    std::thread coord;
    std::vector<std::thread> workers;

    llama_b1_stats st;
    int64_t t_first_begin_us = 0;
    std::vector<int64_t> pass_start_us, pass_wall_us, pass_compute_us;
    std::vector<double> pass_copy_ms, pass_wait_ms;
    std::vector<uint64_t> pass_bytes;

    void coordinator();
    void worker(int w);
    void release_plan();
    void stop_threads();
    bool tensor_streamed(int il, const exps_tensor & e) const;
    void push_tensor_pieces(std::vector<piece> & ps, const exps_tensor & e, region * dst, size_t dst_offs, bool from_file) const;
    bool run_pieces(std::vector<piece> && ps);
    void restore_step(int k);
};
