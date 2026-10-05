#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

// Auto-asymmetric turbo-K upgrade decision (see llama-kv-cache.cpp for the
// full rationale: high-GQA-ratio models amplify turbo K's quantization
// error, so symmetric turbo K+V gets K upgraded to q8_0). Exposed so
// llama-context.cpp's block KV streaming page-geometry pre-scan can size
// its bootstrap allocation off the same effective K type the llama_kv_cache
// constructor will actually use, instead of the raw requested type - the
// two must never diverge or the streaming runtime's bootstrap pool ends up
// sized for the wrong page geometry.
ggml_type llama_kv_cache_resolve_stream_type_k(
        const llama_model & model, const llama_hparams & hparams,
        ggml_type type_k, ggml_type type_v);

// Layer-adaptive per-layer KV precision override (TURBO_LAYER_ADAPTIVE env
// var - see llama-kv-cache.cpp for the mode legend). Exposed, like the
// resolver above, so llama-context.cpp's block KV streaming pre-scan can
// detect ahead of time whether a model will actually get non-uniform
// per-layer KV types: the streamed page pool has one page size and one
// buffer type for the whole arena, so mixed q8_0/turbo2/turbo4 layers would
// otherwise pack differently-shaped rows into pages sized for a layer that
// isn't theirs.
int llama_kv_cache_turbo_layer_adaptive_mode(ggml_type type_v, uint32_t n_layer);

ggml_type llama_kv_cache_turbo_layer_adaptive_type_k(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);
ggml_type llama_kv_cache_turbo_layer_adaptive_type_v(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0,
           llama_kvarn_config   kvarn = llama_kvarn_config());

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_target> get_kv_stream_targets() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;
    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    ggml_tensor * get_v_storage(int32_t il) const;
    bool get_v_transposed() const;

    bool kv_stream_adapt(uint32_t active_tokens, uint32_t query_tokens);
    bool kv_stream_resize_pool(
        size_t pool_bytes, uint32_t active_tokens, uint32_t ring_slots);

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in.
    // a cache that mirrors another one cell for cell (the qwen4exp indexer) cannot search for
    // its own cells here: a second independent search only happens to agree with the first.
    //   sinfos_out: if set, resized to n_stream and filled with the layout used; a stream that
    //               carried no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching for one. it must have one
    //               entry per stream and the entry must match the cell count in the blob,
    //               otherwise the read fails as it would on any other corrupt input
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
     const slot_info_vec_t *   sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const { return turbo_innerq_scale_inv; }

    //
    // KVarN region-aware cache (sink + ring rows exact fp16, body sealed into low-bit records)
    //
    bool is_kvarn() const { return kvarn.enabled(); }
    const llama_kvarn_config & get_kvarn() const { return kvarn; }
    // KVarN fused write rotation (#139): cpy_k/cpy_v rotate K/V inside the TQ6_0 cache write
    bool kvarn_fused_rot() const { return kvarn_fused_rot_on; }
    int  kvarn_rot_group() const { return kvarn.body_type == GGML_TYPE_TURBO4_0 ? 128 : 256; }
    ggml_tensor * get_kvarn_body(int32_t il) const;
    bool maintain_kvarn(llama_context * lctx);
    int32_t compress_kvarn_idle(llama_context * lctx, llama_seq_id seq_id, llama_pos accepted_end);
    uint32_t get_kvarn_sealed_end() const { return kvarn_B; }
    uint32_t get_kvarn_visible_end() const { return kvarn_N; }
    uint32_t get_kvarn_capacity() const { return kvarn_cap; }
    uint64_t get_kvarn_maintenance_count() const { return kvarn_maintenance_count; }
    uint64_t get_kvarn_maintenance_groups() const { return kvarn_maintenance_groups; }
    bool has_kvarn_maintenance() const { return kvarn_B_pending > kvarn_B; }


    // I32[GGML_KVARN_DESC_N_ENTRIES] graph input consumed by the seal op and the attention op
    ggml_tensor * build_input_kvarn_desc(ggml_context * ctx) const;
    void set_input_kvarn_desc(ggml_tensor * dst, const llama_ubatch * ubatch, int tier = 0) const;
    // tier (0 interior, 1 edge) and body bits of a model layer
    int get_kvarn_layer_tier(int32_t il, uint32_t & bits_k, uint32_t & bits_v) const;
    bool has_kvarn_edge_tier() const { return kvarn_n_layers_edge > 0; }

    // seal the groups that this ubatch pushed out of the tail: k_store/v_store are the set_rows outputs of cpy_k/cpy_v
    ggml_tensor * build_kvarn_seal(ggml_context * ctx, ggml_tensor * k_store, ggml_tensor * v_store, ggml_tensor * desc, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

    // attention sinks for a sliding-window cache: positions [0, n) are never SWA-masked or evicted
    void set_swa_sink(uint32_t n) { n_swa_sink = n; }

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // KVarN: sealed record pool, I8 [rec_bytes*n_head_kv*kvarn_n_groups]
        ggml_tensor * body = nullptr;
        // KVarN tier of this layer: 0 = interior (kvarn.bits_k/v, body_type), 1 = edge (kvarn.edge_*); record geometry follows
        int       kvarn_tier      = 0;
        uint32_t  kvarn_bits_k    = 0;
        uint32_t  kvarn_bits_v    = 0;
        ggml_type kvarn_body_type = GGML_TYPE_F32;
        size_t    kvarn_rec_bytes = 0;
    };

    bool v_trans = true;  // the value tensor is transposed

    // KVarN state (see is_kvarn()). Positions: [0, sink) exact rows 0..sink-1; [sink, B) sealed records
    // (sink..B in groups of `group`); [B, N) exact ring rows sink + (p - sink) % cap. Cells stay one per
    // position (cell index == position) so the mask and the sequence bookkeeping are unchanged.
    llama_kvarn_config kvarn;
    uint32_t kvarn_cap           = 0; // ring rows
    uint32_t kvarn_n_groups      = 0; // records per head in the pool
    uint32_t kvarn_n_groups_seal = 0; // static per-ubatch seal launch size
    bool     kvarn_fused_rot_on  = false; // #139, see kvarn_fused_rot()
    size_t   kvarn_rec_bytes[2]  = {0, 0}; // per tier (0 interior, 1 edge)
    uint32_t kvarn_n_layers_edge = 0;      // cache layers on tier 1 (first + last edge_layers)
    uint32_t kvarn_B      = 0;        // sealed end
    uint32_t kvarn_B_pending = 0;     // proposed end, published after all layers complete
    bool     kvarn_draining  = false; // an adaptive-tail flush is being sealed in flush_chunk steps
    uint64_t kvarn_maintenance_count = 0;
    uint64_t kvarn_maintenance_groups = 0;
    uint32_t kvarn_B_prev = 0;        // sealed end before the current ubatch
    uint32_t kvarn_N      = 0;        // positions present

    uint32_t kvarn_ring_row(uint32_t pos) const { return pos < kvarn.sink ? pos : kvarn.sink + (pos - kvarn.sink) % kvarn_cap; }

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // positions [0, n_swa_sink) are never SWA-masked or evicted (attention sinks)
    // used by the optional MTP drafter attention window (--spec-draft-window); 0 = off
    uint32_t n_swa_sink = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // the K rotation is the functional Hadamard transform of a DSA lightning-indexer cache (not tuning): the indexer
    // graphs of deepseek32/dots3note multiply by it as one full-width matrix, so it must span the whole head
    bool attn_rot_k_full = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    struct kvarn_maintenance_graph {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_backend_t backend = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * desc[2] = {nullptr, nullptr}; // per tier
    };
    std::vector<kvarn_maintenance_graph> kvarn_maintenance_graphs;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    struct kv_stream_runtime_owner {
        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *);
        using span_feedback_fn_t = bool (*)(void *, double);
        using reconfigure_fn_t = bool (*)(void *, uint32_t, uint32_t);
        using repartition_fn_t = bool (*)(void *, uint32_t);
        using decode_layout_fn_t = bool (*)(void *, uint32_t);
        using mark_dirty_rows_fn_t = bool (*)(void *, const int64_t *, size_t);
        using resize_pool_fn_t = bool (*)(void *, size_t, uint32_t, uint32_t);

        void * runtime = nullptr;
        void (*free_fn)(void *) = nullptr;
        feedback_fn_t feedback_fn = nullptr;
        span_feedback_fn_t span_feedback_fn = nullptr;
        reconfigure_fn_t reconfigure_fn = nullptr;
        repartition_fn_t repartition_fn = nullptr;
        decode_layout_fn_t decode_layout_fn = nullptr;
        mark_dirty_rows_fn_t mark_dirty_rows_fn = nullptr;
        resize_pool_fn_t resize_pool_fn = nullptr;
        uint32_t layer_count = 0;
        uint32_t minimum_ring_slots = 0;
        uint32_t decode_layout_pages = 0;
        uint32_t starved_evaluations = 0;
        uint32_t overprovisioned_evaluations = 0;
        uint32_t evaluations_since_repartition = UINT32_MAX;
        uint64_t previous_deadline_samples = 0;
        uint64_t previous_deadline_misses = 0;
        int64_t previous_adapt_us = 0;
        uint32_t previous_query_tokens = UINT32_MAX;

        ~kv_stream_runtime_owner() {
            if (runtime != nullptr) {
                free_fn(runtime);
            }
        }
    };

    // Declared before ctxs_bufs so the custom buffers release their runtime
    // references before this owner releases the initial reference.
    kv_stream_runtime_owner kv_stream_runtime;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization (128 floats)
    ggml_tensor * turbo_innerq_scale_inv = nullptr;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_active_target> get_kv_stream_active_targets() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    // KVarN (see llama_kv_cache)
    bool is_kvarn() const;
    ggml_tensor * get_kvarn_body(int32_t il) const;
    const llama_kvarn_config & get_kvarn() const;
    bool kvarn_fused_rot() const; // #139: the cache write rotates K/V (graph skips its K/V WHT)
    ggml_tensor * build_input_kvarn_desc(ggml_context * ctx) const;
    void set_input_kvarn_desc(ggml_tensor * dst, const llama_ubatch * ubatch, int tier = 0) const;
    int get_kvarn_layer_tier(int32_t il, uint32_t & bits_k, uint32_t & bits_v) const;
    bool has_kvarn_edge_tier() const;
    ggml_tensor * build_kvarn_seal(ggml_context * ctx, ggml_tensor * k_store, ggml_tensor * v_store, ggml_tensor * desc, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
