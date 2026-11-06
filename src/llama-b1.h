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
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    uint64_t bytes_carry_host = 0; // [PP12] pinned host RAM holding the carries (0 = carries in VRAM)
    uint64_t bytes_carry_h2d  = 0; // [PP12] carry bytes staged host -> VRAM
    uint64_t bytes_carry_d2h  = 0; // [PP12] carry bytes staged VRAM -> host
    int64_t  t_stream_us  = 0;   // first layer_begin to end()
};

class llama_b1 {
public:
    llama_b1(const llama_model & model, ggml_backend_t backend_gpu);
    ~llama_b1();

    // Only effective host-staging settings override the automatic RAM cap.
    static bool staging_explicit() {
        for (const char * name : {"LLAMA_B1_THREADS", "LLAMA_B1_PIECE_MIB"}) {
            const char * value = std::getenv(name);
            if (value && *value) { return true; }
        }
        const char * stage = std::getenv("LLAMA_B1_STAGE_MIB");
        return stage && *stage && std::atoll(stage) > 0;
    }

    static bool staging_floor_unmet(uint64_t available, int workers, size_t piece, size_t alignment, size_t zero) {
        constexpr uint64_t floor = 5ull << 30;
        if (available < floor || workers < 1 || workers > 64 || !piece || !alignment ||
                alignment > (SIZE_MAX - 4096) / 3) { return true; }
        const size_t padding = 3 * alignment + 4096;
        if (piece > SIZE_MAX - padding || zero > available - floor) { return true; }
        return uint64_t(piece + padding) > (available - floor - zero) / size_t(workers) / 3;
    }

    // fork: staging gets 10% of the 80% spare-RAM budget above 5 GiB.
    static size_t fit_staging_piece(uint64_t available, int workers, size_t piece, size_t alignment,
            size_t zero, bool fit, bool explicit_request) {
        if (!fit || explicit_request) { return piece; }
        constexpr uint64_t MiB = 1ull << 20;
        if (available <= (5ull << 30) || workers < 1 || workers > 64 || !alignment || !piece) { return 0; }
        const uint64_t spare = (available - (5ull << 30)) / MiB;
        const uint64_t budget = (spare / 5) * 4 + (spare % 5) * 4 / 5;
        const size_t quota = size_t(std::min<uint64_t>(budget / 10, SIZE_MAX / MiB)) * size_t(MiB);
        if (zero > quota || alignment > (SIZE_MAX - 4096) / 3) { return 0; }
        // Two pinned slots and one O_DIRECT bounce per worker; allow padding and page rounding for each.
        const size_t per_buffer = (quota - zero) / size_t(workers) / 3;
        const size_t padding = 3 * alignment + 4096;
        if (per_buffer <= padding) { return 0; }
        const size_t limit = (per_buffer - padding) / 4096 * 4096;
        return limit < std::min<size_t>(piece, MiB) ? 0 : std::min(piece, limit);
    }

    bool ok() const { return init_ok; }
    static uint32_t smaller_ubatch(uint32_t tokens) {
        return tokens > 4096 ? 4096 : tokens > 2048 ? 2048 : 0;
    }
    static uint32_t smaller_tail_ubatch(uint32_t tokens) {
        const uint32_t normal = smaller_ubatch(tokens);
        return normal ? normal : tokens > 1024 ? 1024 : tokens > 512 ? 512 : 0;
    }
    // fork: leave half the spare RAM for page faults and cache growth after admission.
    static uint64_t carry_ram_budget(uint64_t available) {
        constexpr uint64_t floor = 5ull << 30;
        return available > floor ? (available - floor) / 2 : 0;
    }
    static size_t cap_pass_group(size_t requested, size_t ub_tokens, bool small_card) {
        if (!ub_tokens) { return 0; }
        return small_card ? std::min(requested, size_t(32768) / ub_tokens) : requested;
    }
    static size_t fit_carry_group(uint64_t available, size_t carry_bytes, size_t requested,
            size_t ub_tokens = 1, bool small_card = false) {
        if (!carry_bytes || !ub_tokens) { return 0; }
        const size_t fit = size_t(std::min<uint64_t>(requested, carry_ram_budget(available) / carry_bytes));
        return cap_pass_group(fit, ub_tokens, small_card);
    }
    void set_small_card(bool value) { small_card = value; }
    bool small_card = false;
    void set_ffn_enabled(bool enabled) { ffn_requested = enabled; }
    // VRAM the plan must leave free beyond its keep, for scratch measured only after a plan exists (the legacy layer graph)
    void set_extra_keep(size_t bytes) { extra_keep = bytes; }
    size_t extra_keep = 0;
    // [REFIT] re-plan only: the plan's fresh VRAM must stay at or under this (SIZE_MAX = no cap). The legacy layer graph
    // is measured after the plan; when it lacks VRAM, extra keep alone changes nothing if the plan already had slack,
    // so the re-plan caps fresh below the failed plan's (smaller variant: sync carry, then one expert slot).
    void set_fresh_cap(size_t bytes) { fresh_cap = bytes; }
    size_t fresh_cap = SIZE_MAX;
    // Host carry plan sizes, largest first: two expert slots + async, two + sync, one + async, one + sync.
    // carry_step_below() = the next size under `fresh` (0 when none); carry_step_min() = the smallest one that can run.
    size_t carry_step_below(size_t fresh) const {
        const size_t s = step_slot_bytes, c = step_carry_bytes;
        for (size_t bytes : { 2*s + 2*c, 2*s + c, s + 2*c, s + c }) {
            if (bytes < fresh) {
                return bytes;
            }
        }
        return 0;
    }
    size_t carry_step_min() const { return step_slot_bytes + step_carry_bytes; }
    size_t step_slot_bytes = 0;
    size_t step_carry_bytes = 0;
    // [KEEPAUTO] VRAM keep reserve in MiB for this plan (-1 = LLAMA_B1_KEEP_FREE_MIB or the default). The legacy
    // layer check (llama-context.cpp) reads the same value so plan and check always agree.
    void set_keep_override(int64_t mib) { keep_override_mib = mib; }
    int64_t keep_override_mib = -1;
    size_t last_fresh() const { return fresh_last; }
    size_t fresh_last = 0;
    // [PINRE] keep the pinned host carry across a re-plan inside one call (begin() releases the plan first);
    // re-pinning 3.9 GiB costs ~2 s. drop_carry_spare() frees what the re-plan did not reuse.
    void hold_carry_pin(bool hold) { carry_pin_hold = hold; }
    void drop_carry_spare();
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

    // [PP12] LLAMA_B1_HOST_CARRY=1 (opt-in): when the carries of one whole-prompt group do not fit in VRAM, keep them
    // in pinned host RAM and stage each ubatch's carry through one VRAM slot, so the prompt still crosses the experts
    // once (one pass) instead of falling back to normal chunks. =2 forces it even when VRAM would fit (identity test).
    // Every carry() tensor then views the same VRAM slot: load before a range that reads it, store after one that
    // writes it. Both are async copies on the compute stream, so they are ordered with the graphs around them.
    bool carry_on_host() const { return carry_host; }
    void carry_load(size_t ub);
    void carry_store(size_t ub);
    // [PP12A] LLAMA_B1_HOST_CARRY_ASYNC=1 (opt-in, with host carry): two VRAM carry slots (ubatch u uses slot u % 2)
    // and a separate copy stream, so the store of ubatch u and the load of the next ubatch for that slot run while the
    // graph of u + 1 computes. Same bytes, same graphs: bit-identical to the one-slot path.
    // carry_write_begin: a range that writes the carry without reading it (layer 0) waits for the slot's last store.
    // carry_prefetch: load ub into its slot now (the caller has just stored the slot's previous ubatch).
    // carry_load_final: the final range (no store after it): frees the previous slot and prefetches the next ubatch.
    void carry_write_begin(size_t ub);
    void carry_prefetch(size_t ub);
    void carry_load_final(size_t ub);
    bool carry_async_on() const { return carry_host && carry_async; }

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

    // views over the parts of the lent VRAM regions that no slot or carry of the current plan uses, largest first,
    // each at least min_bytes; the layer graph may use them as compute scratch until refill() restores the lent tensors
    std::vector<ggml_backend_buffer_ptr> spare_views(size_t min_bytes, size_t max_views) const;

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
        std::vector<char>     kept; // pageable bytes; never registered with the GPU
    };

    struct layer_info {
        std::vector<int> tensors; // indices into exps
        bool   resident = false;  // every exps tensor of the layer is in VRAM
        size_t bytes    = 0;
        size_t res_bytes = 0;     // the VRAM-resident exps tensors only: what lending this layer frees
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
    std::vector<placement> carry_place; // [ubatch index within the group]; one entry when carry_host
    bool carry_host = false;
    bool carry_async = false;               // [PP12A] two slots + own copy stream
    size_t n_host_slots() const { return carry_async ? 2 : 1; }
    bool one_slot = false;                  // [PP12H] host carry with ONE expert layer slot: a layer's copy waits until
                                            // the previous layer's graphs ran (no copy/compute overlap, half the slot VRAM)
    size_t n_exp_slots() const { return one_slot ? 1 : 2; }
    ggml_backend_t backend_carry = nullptr; // [PP12A] the carry copy stream
    ggml_backend_event_t ev_cload[2]  = { nullptr, nullptr }; // slot loaded (carry stream)
    ggml_backend_event_t ev_cdone[2]  = { nullptr, nullptr }; // graph done with slot (compute stream)
    ggml_backend_event_t ev_cstore[2] = { nullptr, nullptr }; // slot stored to host (carry stream)
    size_t carry_pend[2] = { SIZE_MAX, SIZE_MAX };            // ubatch whose load into the slot is issued, not yet consumed
    void carry_fetch(size_t ub);
    void carry_async_free();
    ggml_backend_buffer_t carry_host_buf = nullptr;
    ggml_backend_buffer_t carry_pin_spare = nullptr; // [PINRE] pinned carry held over a re-plan
    bool carry_pin_hold = false;
    // [PINCH] LLAMA_B1_CARRY_PIN_CHUNKED=1: the carry RAM is plain memory that a helper thread page-locks in chunks,
    // front to back, so the ~2 s pin overlaps the first layer; a copy waits only for the chunks it touches
    struct carry_chunk_pin {
        void * ptr = nullptr;
        size_t size = 0;
        ggml_backend_buffer_t buf = nullptr;   // CPU view of ptr (does not own it)
        void (*unpin)(void *) = nullptr;
        std::vector<void *> pinned;            // chunk starts, written by the helper thread only
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<size_t> done{0};           // bytes [0, done) settled (pinned, or left pageable after a failure)
        std::atomic<bool> stop{false};
        int64_t pin_us = 0;                    // helper thread wall time
        int64_t wait_us = 0;                   // copies blocked on the helper
    };
    std::unique_ptr<carry_chunk_pin> carry_chunk, carry_chunk_spare;
    bool carry_chunk_alloc(size_t bytes);
    static void carry_chunk_free(std::unique_ptr<carry_chunk_pin> & c);
    void carry_pin_wait(size_t ub);
    std::vector<size_t> carry_host_offs; // per ubatch of the call
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
        const char * src_kept = nullptr; // pageable source, always copied through staging
        size_t alignment = 1;
        int          fd       = -1;
        size_t       src_offs = 0;
        region *     dst      = nullptr;
        size_t       dst_offs = 0;
        size_t       len      = 0;
    };

    size_t piece_bytes = 16u << 20;
    int    n_workers   = 8;
    // [PREFILLFAST] LLAMA_B1_DIRECT=1: the pinned expert arena is already page-locked, so arena pieces go straight
    // to the copy stream (no memcpy into a staging buffer, no staging event wait); direct_piece caps those pieces
    bool   direct       = false;
    size_t direct_piece = 64u << 20;
    // fork AIPNEXTK: LLAMA_B1_BANK_STAGE=1 gathers an arena piece that touches dropped bank-expert bytes into the
    // worker's pinned staging block (zeros for the dropped bytes) and uploads it with one copy; such pieces then span
    // experts up to piece_bytes instead of one copy (plus one memset per hole) per expert under the stream lock
    bool   bank_stage   = false;
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
