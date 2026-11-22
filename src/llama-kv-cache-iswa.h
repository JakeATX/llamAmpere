#pragma once

#include "llama-kv-cache.h"
#include "llama-batch.h"

#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

// INKKV multi-sequence far/near split switch (LLAMA_INKKV_MULTI_SPLIT, defined in llama-graph.cpp)
bool llama_inkkv_multi_split_on();

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0,
                   llama_sj_kvarn_config sj_kvarn = llama_sj_kvarn_config(), // fork (INKKV): see below
                          uint32_t base_window = 0,  // fork (DSPARKWIN): see below
                          uint32_t base_sink   = 0);

    llama_kv_cache_iswa(
            const llama_model & model,
                         bool   no_alloc,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0,
        // fork (INKKV): SJ-KVaRN on the non-SWA (full-attention) half only; the SWA half keeps a plain cache
                   llama_sj_kvarn_config sj_kvarn = llama_sj_kvarn_config(),
        // fork (DSPARKWIN): base_window > 0 makes the non-SWA half a sliding-window ring of the last base_window
        // positions plus base_sink sink positions (the DSpark drafter's --spec-dspark-window); needs a plain cache
                          uint32_t base_window = 0,
                          uint32_t base_sink   = 0);

    ~llama_kv_cache_iswa() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_full_single_seq() override;

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

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_target> get_kv_stream_targets() const override;

    //
    // llama_kv_cache_iswa specific API
    //

    llama_kv_cache * get_base()    const;
    llama_kv_cache * get_swa ()    const;
    llama_kv_cache * get_swa_mid() const;

    // INKKV v2: the SWA cache holds the full-attention window copy (the far/near split can run)
    bool inkkv_near_copy() const { return near_copy; }
    bool inkkv_swa_shared() const { return swa_shared; }

    // INKKV v2: most tokens per sequence in a ubatch shared by several sequences (init_batch)
    static uint32_t inkkv_mix_max();
    llama_batch_allocr::seq_compat_fn inkkv_mix_compat(const llama_batch_allocr & balloc, std::vector<uint32_t> & seq_n) const;

private:

    bool near_copy = false;
    bool swa_shared = false; // INKKV shared SWA staging (one unified SWA pool, per-stream base)

    const bool unified;

    // fork (INKKV): head-packed hparams of the SJ-KVaRN base half (see llama_kv_cache_iswa ctor), owned here
    std::unique_ptr<llama_hparams> hparams_base_sj;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;
    std::unique_ptr<llama_kv_cache> kv_swa_mid;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv, bool single_seq = false);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches,
            slot_info_vec_t sinfos_swa_mid = {});

    virtual ~llama_kv_cache_iswa_context();

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
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base()    const;
    const llama_kv_cache_context * get_swa()     const;
    const llama_kv_cache_context * get_swa_mid() const;

private:
    //llama_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;
    const llama_memory_context_ptr ctx_swa_mid;

    const llama_memory_status status;
};
