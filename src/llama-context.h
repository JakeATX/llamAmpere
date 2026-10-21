#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-mtp-vocab.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;
class llama_b1;
class llama_ple_prefetch;
struct llama_moe_output;
class llama_kv_cache;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
class  llama_memory_recurrent;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();
    void detach_compute_borrower();

    void synchronize();
    ggml_backend_t sj_kvarn_backend(ggml_backend_buffer_t buffer) const;
    bool maintain_sj_kvarn();
    bool b1_seal_sj_kvarn(int32_t il0, int32_t il1); // [B1] seal the current ubatch's groups in layers [il0, il1) only
    int32_t compress_sj_kvarn_idle(llama_seq_id seq_id, llama_pos accepted_end, llama_pos keep_from = -1);


    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;
    int b1_status() const {
        if (!b1_tried) { return -1; }
        if (!b1 || moe_cache) { return 0; }
        for (bool extract : cparams.embeddings_layer_inp) {
            if (extract) { return 0; }
        }
        return 1;
    }

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;
    const llama_sj_kvarn_config * get_sj_kvarn() const;

    // return the *effective* K/V cache tensor types currently in use by the memory object.
    // this can differ from what was requested via llama_context_params.type_k/type_v because
    // some memory implementations silently rewrite the requested type (e.g. TurboQuant
    // auto-asymmetric upgrades K to q8_0 for high-GQA-ratio models - see llama_kv_cache ctor).
    // returns GGML_TYPE_COUNT if the memory object has no single meaningful K/V type
    // (e.g. pure recurrent memory, or a composite cache like DSV4 whose sub-caches can
    // legitimately hold different types).
    enum ggml_type get_kv_type_k() const;
    enum ggml_type get_kv_type_v() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);
    bool copy_embeddings_nextn(float * dst, int32_t first, int32_t count);
    bool release_embeddings_nextn(int32_t first, int32_t count);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);
    void set_decode_phase(enum llama_decode_phase phase);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_mtp_chain(bool value);
    void set_mtp_chain_sampling(int32_t top_k, float temp, float top_p, float min_p, const float * u, int32_t n_u); // [#69]
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    bool mtp_prefill_init();
    void mtp_prefill_reset(llama_seq_id seq);
    int mtp_prefill_decode(llama_context & target, llama_batch batch, const float * const * carry, uint32_t n_seq);
    void mtp_prefill_capture(ggml_tensor * hidden, const llama_ubatch & ubatch);
    void mtp_prefill_input(llm_graph_input_embd_h & input, const llama_ubatch & ubatch);
    void mtp_prefill_place(llm_graph_result & res);
    // [PREFILLFAST] LLAMA_MTP_B1_HANDOFF=1: per-ubatch device handoff of the target's hidden rows to the helper
    bool mtp_handoff_init();
    void mtp_handoff_capture(ggml_tensor * hidden, const llama_ubatch & ubatch);
    bool mtp_handoff_ready(int32_t first, int32_t count) const;
    void mtp_handoff_release();
    void set_ubatch_rows_callback(void (*cb)(void * user, int32_t first, int32_t count), void * user);
    int mtp_handoff_decode(llama_context & target, llama_batch batch, int32_t first, const float * carry);
    void (*ubatch_rows_cb)(void * user, int32_t first, int32_t count) = nullptr;
    void * ubatch_rows_cb_user = nullptr;

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out,
 llama_state_seq_flags   flags = 0);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count,
 llama_state_seq_flags   flags = 0);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs, int32_t n_tokens = 0);

    // re-run the reservation for the rows reserved so far, so that a configuration change
    // (e.g. enabling nextn embeddings) grows the pinned buffers at setup time, not inside a request
    void output_reserve_grow();

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // parse a `llama-mtp-vocab-v1` map and upload it to the head's device (throws on error)
    // the last `n_hot` entries become adaptive slots, see llama_draft_vocab_observe/refresh
    void init_draft_vocab(const char * path, int32_t n_hot);

    // adaptive draft-vocabulary tail (request granularity, must not race with a graph)
    void    draft_vocab_observe(const llama_token * toks, size_t n);
    int32_t draft_vocab_refresh();
    int32_t draft_vocab_n_hot()        const { return draft_vocab.n_hot; }
    size_t  draft_vocab_n_candidates() const { return draft_vocab.hot.cands.size(); }

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);
    static bool sched_copy_mtp_inputs(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    struct kv_stream_phase_arena_owner {
        struct layout {
            size_t kv_bytes = 0;
            size_t compute_offset = 0;
            size_t compute_bytes = 0;
            uint32_t ring_slots = 0;
            uint32_t resident_pages_per_layer = 0;
            std::vector<size_t> backend_sizes;
        };

        void * arena = nullptr;
        void (*free_fn)(void *) = nullptr;
        bool (*set_compute_fn)(void *, size_t, size_t) = nullptr;
        ggml_backend_buffer_type_t (*buffer_type_fn)(void *) = nullptr;
        bool (*graph_reset_fn)(ggml_backend_t) = nullptr;
        ggml_backend_dev_t device = nullptr;
        ggml_backend_buffer_type_t buffer_type = nullptr;
        size_t arena_bytes = 0;
        size_t page_bytes = 0;
        size_t conversion_bytes = 0;
        uint32_t layer_count = 0;
        uint32_t minimum_ring_slots = 8;
        size_t backend_index = SIZE_MAX;
        size_t max_nodes = 0;
        size_t current_kv_bytes = 0;
        size_t current_compute_offset = 0;
        size_t current_compute_bytes = 0;
        uint32_t current_ring_slots = 0;
        bool decode = false;
        bool configured = false;
        layout prefill;
        layout token_generation;

        ~kv_stream_phase_arena_owner() {
            if (arena != nullptr) {
                free_fn(arena);
            }
        }
    };

    bool kv_stream_switch_phase(bool decode, uint32_t active_tokens);

    // Declared before memory and scheduler so their arena leases are released first.
    kv_stream_phase_arena_owner kv_stream_phase_arena;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // draft-only vocabulary shortlist (MTP draft contexts): persistent I32 map, compact index -> token id.
    // Owned here so graphs reference a fixed device tensor instead of uploading an input per eval.
    struct draft_vocab_info {
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor           * ids = nullptr; // [n_sel]
        // [#81] compact resident draft head [n_embd, n_sel] (row i scores token ids[i]) and the head it was built
        // from; set for EXL3 heads (or LLAMA_DRAFT_VOCAB_COMPACT=1), see llama-draft-vocab-compact.h
        ggml_tensor           * compact     = nullptr;
        const ggml_tensor     * compact_src = nullptr;
        std::vector<int32_t>    host;          // same ids, host copy
        int32_t                 n_hot = 0;     // trailing adaptive slots (0 = fully static map)
        llama_mtp_hot_vocab     hot;           // ranking policy for those slots
        mutable std::atomic<uint32_t> warned{0}; // full-head fallback reasons already logged (llm_graph_params)
    };

    draft_vocab_info draft_vocab;

    // [#69] sampled MTP chain inputs: the packed sampling params, then one uniform per chain step
    std::vector<float> mtp_chain_samp;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    // Opt-in MTP catch-up translation storage, at most128MiB hidden payload.
    // Independent from ubatch/output memory; decode's allocr copies its contents.
    std::unique_ptr<llama_batch_ext> mtp_compat_reuse;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // another context of the same model whose compute buffers this context shares (target <-> MTP);
    // the two never compute at the same time, each synchronizes the other before computing
    llama_context * compute_peer = nullptr;

    // fork: Private MTP storage permits target/draft stream overlap without sharing scratch allocations.
    bool mtp_prefill_async = false;
    bool mtp_reserve_tight = false;
    bool mtp_scratch_enabled = false;
    bool mtp_scratch_active = false;
    llama_context * mtp_scratch_target = nullptr;
    std::vector<llama_context *> mtp_scratch_clients;
    llama_kv_cache * mtp_scratch_cache = nullptr;
    ggml_backend_t mtp_scratch_backend = nullptr;
    std::vector<ggml_backend_buffer_ptr> mtp_scratch_views;
    void mtp_scratch_prepare(bool kv_only, const llama_memory_context_i * mctx = nullptr, const llama_ubatch * ubatch = nullptr);
    void mtp_scratch_end();
    bool compute_is_shared() const;
    struct {
        bool enabled = false;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor * rows = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_backend_event_ptr ready;
        ggml_backend_event_ptr consumed;
        bool consuming = false;
        std::vector<bool> captured;
        std::vector<llama_pos> carry_pos;
        bool ring = false;        // [PREFILLFAST] handoff staging holds one ubatch (rows [0, n_ubatch) + one carry row per seq)
        int64_t ring_first = -1;  // batch index of staged row 0, or -1
        int32_t ring_rows = 0;
    } mtp_hidden;
    llama_context * mtp_hidden_source = nullptr; // scoped to mtp_prefill_decode
    std::vector<int32_t> mtp_hidden_rows;
    std::vector<ggml_backend_buffer_ptr> mtp_input_uploads;
    size_t mtp_input_upload_next = 0;
    ggml_backend_event_ptr mtp_input_upload_done;

    bool sched_need_reserve = true;
    // fork: limit only the reserved graph, not the caller's physical batch size.
    uint32_t sched_reserve_tokens = 0;
    // The scheduler owns expert residency independently of its compute allocator.
    // Reuse it for graph-shape changes while its fixed capacity/configuration fits.
    size_t sched_node_capacity = 0;
    int sched_cache_mode = -1;
    size_t sched_cache_budget_mib = 0;
    bool sched_pipeline_parallel = false;
    bool sched_op_offload = false;
    bool prompt_buffers_resize(uint32_t n_tokens, const llama_batch_ext * batch);

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;
    uint64_t           gf_res_prev_epoch  = 0; // ggml_backend_sched_get_alloc_epoch when gf_res_prev_active was allocated
    uint64_t           backend_buf_exp_epoch = 0; // ggml_backend_sched_get_alloc_epoch when backend_buf_exp_size was taken

    // [B1] layer-major prefill (LLAMA_B1=1, opt-in; see llama-b1.h). Declared after the backends so it is
    // destroyed first; the destructor also resets it explicitly.
    std::unique_ptr<llama_b1> b1;
    // [PLEPF] PLE page prefetch (LLAMA_PLE_PREFETCH=<threads>, opt-in; see llama-ple-prefetch.h)
    std::unique_ptr<llama_ple_prefetch> ple_pf;
    bool ple_pf_tried = false;
    bool b1_env   = false; // LLAMA_B1 set
    bool b1_tried = false; // b1 construction attempted
    bool b1_ffn_scratch_active = false;
    bool b1_layer_scratch_active = false;
    bool b1_kv_scratch_enabled = false;
    uint32_t b1_kv_prompt_cells = 0;
    uint32_t b1_kv_outputs = 1;
    ggml_backend_t b1_kv_backend = nullptr;
    std::vector<size_t> b1_layer_compute_bytes;
    std::vector<ggml_backend_buffer_ptr> b1_kv_views;
    ggml_backend_buffer_ptr b1_kv_owned;
    bool b1_kv_scratch_release();
    bool b1_kv_scratch_bind(int32_t l0, int32_t l1);

    bool b1_off_logged = false; // the once-per-context note on why B1 is not used was printed
    bool b1_mc_suspended = false; // this call freed the MoE expert cache's pool (made again in b1_end)
    struct {
        bool          active = false;   // the graph being built is a B1 graph (memory already applied)
        int32_t       il0    = 0;
        int32_t       il1    = -1;
        ggml_tensor * carry  = nullptr;
        const llm_b1_ffn * ffn = nullptr;
        int ffn_phase = 0;
    } b1_cur;

    // returns 1 when this decode call runs layer-major (every ubatch applied and saved, the plan made),
    // 0 for the normal path, < 0 for a decode error code
    int  b1_prepare(llama_memory_context_i * mctx, const llama_batch_ext & batch_inp, uint32_t n_tokens_all, int64_t n_outputs_all);
    bool b1_reserve_legacy_layers(const std::vector<uint32_t> & sizes, int64_t n_outputs_all);
    // runs layers [0, n_layer - 1) of every ubatch of the group that starts at ub0, then seeks back to ub0 with
    // the last layer's experts ready; false on failure (status set)
    bool b1_run_prefix(llama_memory_context_i * mctx, size_t ub0, ggml_status & status);
    // ends the B1 call (b1->end) and gives the MoE expert cache back if the call suspended it
    void b1_end(bool failed);
    bool b1_cache_graph_reset();
    void b1_cache_resume();
    int64_t b1_final_rows(int64_t n_out, int64_t n_tok) const;

    // fork: opt-in prompt routing counts use layer IDs, never temporary B1 weight addresses.
    bool b1_cache_warm_enabled = false;
    bool b1_cache_collect = false;
    bool b1_cache_needs_refresh = false;
    bool b1_cache_profile_attempted = false;
    bool b1_cache_device_pending = false;
    void b1_cache_begin();
    ggml_tensor * b1_cache_user_node = nullptr;
    std::vector<std::vector<uint64_t>> b1_cache_counts;
    uint64_t b1_cache_readbacks = 0;
    int64_t b1_compute_submit_us = 0;
    static bool b1_cache_eval(ggml_tensor * tensor, bool ask, void * user_data);
    void b1_cache_profile_load();
    void b1_cache_warm();
    void b1_cache_device_prepare();

public:
    // in-call checkpoints (llama_state_seq_capture_arm)
    struct ckpt_write {
        ggml_tensor * t;
        size_t        offset;   // in the tensor
        size_t        size;
        size_t        blob_off; // in the blob
        int32_t       il;       // layer of the tensor
    };
    struct ckpt_capture {
        llama_pos n_tokens = 0; // copied after the ubatch whose last position is n_tokens - 1
        llama_pos pos_min  = -1;
        llama_pos pos_max  = -1;
        std::vector<uint8_t>    blob;   // the bytes state_seq_get_data(PARTIAL_ONLY) would write at that spot
        std::unique_ptr<llama_moe_output> blob_file;
        uint8_t * blob_data();
        const uint8_t * blob_data() const;
        size_t blob_size() const;
        std::vector<ckpt_write> writes; // where the blob's tensor rows come from
        std::vector<uint8_t>    layer_done;
        bool laid_out = false;
        bool failed   = false;
    };

    int32_t capture_arm(llama_seq_id seq_id, const llama_pos * n_tokens, int32_t n);
    int32_t capture_min_tail() const;
    bool    capture_select(int32_t k, llama_pos * pos_min, llama_pos * pos_max);

private:
    llama_seq_id              ckpt_seq   = -1;
    bool                      ckpt_armed = false; // set by capture_arm, consumed by the next decode
    int32_t                   ckpt_sel   = -1;
    std::vector<ckpt_capture> ckpt_caps;

    // the recurrent memory whose PARTIAL_ONLY state the captures copy, nullptr when unsupported
    const llama_memory_recurrent * capture_mem() const;
    // after a ubatch graph over layers [il0, il1) (il1 < 0: to the last layer) ran: copy the rows of a capture
    // whose spot this ubatch ends
    void capture_ubatch(const llama_ubatch & ubatch, int32_t il0, int32_t il1);

    // host buffer for the model output (logits and embeddings)
    std::unique_ptr<llama_moe_output> output_file;
    ggml_backend_buffer_ptr buf_output;

    // host buffer for the backend sampling outputs (sampled ids, and the optional logits/probs/candidates)
    // kept separate from buf_output so that installing/removing a backend sampler never reallocates the
    // (much larger) logits/embeddings block
    ggml_backend_buffer_ptr buf_sampling;

    // largest number of output rows reserved so far; output_reserve() never sizes below it, so the pinned
    // buffers only ever grow (each growth is a cudaFreeHost/cudaMallocHost pair)
    uint32_t n_outputs_reserved = 0;
    bool output_buffer_reuse = true;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    enum llama_decode_phase decode_phase =
        LLAMA_DECODE_PHASE_AUTOMATIC;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
