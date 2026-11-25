#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-inklsa.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <atomic>
#include <map>
#include <unordered_map>
#include <tuple>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

// Maps a folded model weight to the activation-side transform applied
// immediately before the matmul: optional sign flip, then the normalized
// blockwise Hadamard rotation.
struct llama_hadamard_transform {
    ggml_tensor * rot;
    ggml_tensor * signs; // nullptr for identity sign mode
    // when perm_rep > 1 the activation arrives with its feature axis in tiled
    // head order [hd, nk, rep] and must be permuted to the grouped order
    // [hd, rep, nk] the fold was computed in, before signs and rotation
    int64_t perm_hd  = 0;
    int64_t perm_nk  = 0;
    int64_t perm_rep = 0;
};
using llama_hadamard_rotations = std::unordered_map<const ggml_tensor *, llama_hadamard_transform>;

struct llama_cparams;
struct llama_model;
struct llama_layer;
struct llama_model;
struct llama_prec_policy;
struct llama_hadamard;
struct llama_lsa_idx_slot;

class llama_moe_cache;
class llama_ink_stream;
class llama_peer_experts;
class llama_ple_prefetch;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_dsa_iswa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;

class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    LLM_FUSED_OP_GDN_AR,
    LLM_FUSED_OP_GDN_CH,
    LLM_FUSED_OP_LIGHTNING_INDEXER,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
    LLM_FUSED_OP_TOP_K_MASK,
};


enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_GELU_ERF,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_GEGLU_ERF,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,           // kimi-k3
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens       = nullptr; // I32 [n_batch]
    ggml_tensor * embd         = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * mixed_tokens = nullptr; // I32 [n_tok_rows], mixed path: ids of the token rows
    ggml_tensor * mixed_slots  = nullptr; // I64 [n_tok_rows], mixed path: batch index of the token rows
    ggml_tensor * mixed_embd   = nullptr; // F32 [n_embd, n_batch], mixed path: embd rows, token rows are overwritten
    ggml_tensor * scale_rows   = nullptr; // F32 [1, n_batch], per-row scale: scale_tok for token rows, 1 for embd rows
    ggml_tensor * embd_rows    = nullptr; // I32 [n_batch], embd row path: rows of ctx_other's layer input device buffer

    float scale_tok = 1.0f;

    const int64_t n_embd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input, fed from ubatch.embd_state
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
    llm_graph_input_embd_h(int64_t n_embd, int64_t n_embd_state) : n_embd(n_embd), n_embd_state(n_embd_state) {}
    virtual ~llm_graph_input_embd_h() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * h      = nullptr; // F32 [n_embd_state, n_batch], the host input or the rows gathered with h_rows
    ggml_tensor * h_rows = nullptr; // I32 [n_batch], rows of the h_nextn device buffer of ctx_state_src, nullptr on the host path
    bool h_device = false; // fork: the context supplies hidden rows by backend copy
    // [RTPORT] LLAMA_MTP_PIPELINE: the token embedding row as a device input, written on the stream by the
    // context (llama_context::mtp_pipe_input) instead of a host get_rows; F32 [n_embd_inp, 1], or nullptr
    ggml_tensor * tok_row = nullptr;

    const int64_t n_embd       = 0;
    const int64_t n_embd_state = 0;
};

// original batch index of each ubatch token, used as set_rows destination rows
class llm_graph_input_batch_idxs : public llm_graph_input_i {
public:
    llm_graph_input_batch_idxs() = default;
    virtual ~llm_graph_input_batch_idxs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * idxs = nullptr; // I64 [n_batch]
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd, llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ)
        : n_pos_per_embd(n_pos_per_embd), rope_section_order(rope_section_order) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
    const llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset, bool attn_temp_pow = false)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset), attn_temp_pow(attn_temp_pow) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
    const bool     attn_temp_pow;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main; // I32 [n_seqs]
    ggml_tensor * s_copy_tail; // I32 [n_rs - 1]

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;

    // DRC phase 2: pending-replay length, checkpoint span and s-staleness baked into this
    // graph's topology (the shape of the ggml_gated_delta_net(K=1) reconstruction subtree and
    // of the ring/checkpoint writes depends on all three) -- must match for valid reuse, same
    // as head/rs_z above. span_new is what the ubatch leaves behind the checkpoint once this
    // graph has run; set_input() hands it back to the memory.
    uint32_t replay_len = 0;
    uint32_t ckpt_span  = 0;
    bool     s_stale    = false;
    uint32_t span_new   = 0;
    uint32_t snap_shift = 0; // [TAG_RECURRENT_ROLLBACK_SHIFT] see llama_memory_recurrent_context::get_snap_shift

    // row indices for gathering all snap_shift older snapshot groups of the ubatch's cells in one
    // ggml_get_rows: [j * n_seqs + s] = j * mem_size + s_copy_main[s] for the groups lane s really
    // has (j < K - max(n, r_s)), and lane s's own destination row (n + j) * mem_size + head + s --
    // a copy onto itself -- for the rest; nullptr when snap_shift == 0
    ggml_tensor * s_copy_shift = nullptr; // I32 [snap_shift * n_seqs]

    // fill s_copy_shift from the already-filled s_copy (call after the s_copy fill; s_copy() has a
    // rs_idx-reset side effect, so the values are never re-read from the memory context)
    void set_input_shift(const llama_memory_recurrent_context * m);

    // [TAG_RECURRENT_ROLLBACK_RING] ring mode (mctx->get_rs_ring()); all nullptr/0 otherwise, and
    // n_written == 0 / rs_wr == nullptr for a ubatch that fills the ring (n >= K: the builders use
    // the static strided copies, see llama_memory_recurrent_context::get_n_written).
    // Shapes are graph topology -> checked in can_reuse.
    uint32_t n_written = 0;
    uint32_t n_older   = 0;
    ggml_tensor * rs_wr      = nullptr; // I32 [n_written * n_seqs]: cache row of group g of seq s at [g * n_seqs + s]
    // I32 [n_seqs] per written slot (oldest group first): one input tensor per slot -- a per-layer view
    // of a single input would be scheduled as a CPU node and cost a H2D copy + sync per layer
    std::vector<ggml_tensor *> rs_wr_conv;
    std::vector<int32_t *>     rs_wr_conv_host; // fill_s_copy scratch: host data of rs_wr_conv[j]
    ggml_tensor * rs_old_src = nullptr; // I32 [n_older]: older-group rows to carry for seqs whose data moves cells
    ggml_tensor * rs_old_dst = nullptr; // I32 [n_older]

    // --gdn-replay, ubatches of n >= n_rs_seq tokens (every speculative verify): the pending rollback
    // is data, not topology, so one graph serves every round. s_copy_conv = s_copy + replay_len *
    // mem_size picks the conv history group; replay_w = K slot weights (1 for the accepted prefix
    // [0, ckpt_span - replay_len) when the logical state needs the replay, else all 0). s_copy_pair
    // gathers the replay base from the state/checkpoint pair tensor: the checkpoint row (plane 1)
    // when the logical state needs the replay, else the state row. nullptr otherwise.
    ggml_tensor * s_copy_conv      = nullptr; // I32 [n_rs]
    ggml_tensor * s_copy_conv_main = nullptr; // I32 [n_seqs] view
    ggml_tensor * s_copy_pair      = nullptr; // I32 [n_rs]
    ggml_tensor * s_copy_pair_main = nullptr; // I32 [n_seqs] view
    ggml_tensor * replay_w         = nullptr; // F32 [n_rs_seq], or [n_rs_seq, n_seqs] per lane ([TAG_GDN_REPLAY_LANES])

    // [CONVGATHER] one-graph conv history gather (LLAMA_INKLING2_CONV_GATHER=1): s_copy once per layer plane of the
    // memory's r_all, and the gathered rows {n_embd_r, n_rs, n_layer} built by the first layer that asks (build_rs_all).
    // Only when no state is zeroed, moved or copied as an extra (n_rs == n_seqs, rs_z < 0, n_older == 0): those keep
    // the per-layer build_rs. nullptr otherwise.
    ggml_tensor * s_copy_all = nullptr; // I32 [n_rs, n_layer]
    ggml_tensor * r_rows_all = nullptr;

    // fill every rs input (s_copy and the shift or ring tensors) from the memory context; exactly
    // once per ubatch -- both fills consume the pending rollback
    void fill_s_copy(const llama_memory_recurrent_context * m);
    void fill_s_copy_all(const int32_t * data); // [CONVGATHER] s_copy once per layer plane

    // [TAG_GDN_REPLAY_LANES] 2-D replay_w: w, s_copy_conv and s_copy_pair per lane, from the filled s_copy data
    void fill_replay_lanes(const llama_memory_recurrent_context * m, const int32_t * data, int64_t n_rs);

    // can_reuse's replay check: the replay fields must match unless replay_w makes them data (then
    // they and span_new are refreshed from the new context)
    bool replay_reusable(const llama_memory_recurrent_context * m, const llama_ubatch & ubatch);
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

// KQ mask and the optional per-sequence cell rows of a unified cache (see ggml_flash_attn_ext_rows)
// the mask has two shapes depending on whether `rows` is used:
//   rows != nullptr: [n_kv_rows, n_batch/n_slices, 1, n_slices]
//   rows == nullptr: [n_kv,      n_batch/n_stream, 1, n_stream]
struct llm_attn_mask {
    ggml_tensor * raw  = nullptr;
    ggml_tensor * rows = nullptr; // I32 [n_kv_rows, n_slices]
    uint32_t      n_kv = 0;

    // fill every token of a sequence with the mask row of its first token (dspark draft blocks)
    bool block_uniform = false;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // n_tokens == n_batch, n_kv == n_tokens padded to 256
    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;

    // note: assumes v_rot^2 == I
    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    // SJ-KVaRN: I32 region descriptor (sink/ring/body bounds for this ubatch), nullptr unless the cache is SJ-KVaRN
    ggml_tensor * self_sj_kvarn_desc = nullptr;
    ggml_tensor * self_sj_kvarn_desc_edge = nullptr; // tiered body: descriptor of the edge tier (nullptr unless edge layers exist)
    // multi-stream SJ-KVaRN: attended (padded) position count per stream of the ubatch; one FA node per stream
    std::vector<uint32_t> sj_kvarn_n_kv_streams;
    bool skip_unused = false; // fork: KV-only device-input graphs do not consume attention geometry

    // LSA probe (llama-lsa.h): F32 [n_kv, 1] position of each KV cell (1e9 for empty cells) and F32 [1, n_tokens]
    // position of each query row; nullptr unless LLAMA_LSA_MODE is set
    ggml_tensor * lsa_cell_pos = nullptr;
    ggml_tensor * lsa_q_pos    = nullptr;
    // LLAMA_LSA_PACKED=1 with the packed (I32 bit) KQ mask on: F16 dense mask [n_kv, n_tokens/n_stream, 1, n_stream]
    // that only the LSA layers start from; every other layer keeps the packed mask. nullptr otherwise
    ggml_tensor * lsa_kq_mask  = nullptr;
    // SJ-KVaRN cache (index cache only): I64 [n_tokens] index-cache row of each query row (cell == position there,
    // while the K store goes to ring rows); nullptr on plain caches
    ggml_tensor * lsa_q_cell   = nullptr;
    // LLAMA_LSA_J3: built once per graph and shared by every stale-query prediction (each fresh reshape of a host
    // input is its own split input, so per-target views uploaded the mask and cell positions 26 times per token)
    ggml_tensor * lsa_j3_mk2   = nullptr; // [n_kv, T] view of the KQ mask
    ggml_tensor * lsa_j3_force = nullptr; // [n_kv/F, T or 1] units forced by sinks and the window
    // what the two were built from: reused only while every key matches (a layer with another mask, position
    // input, window, sinks or unit size builds its own)
    const ggml_tensor * lsa_j3_mk2_src  = nullptr; // kq_mask
    const ggml_tensor * lsa_j3_force_cp = nullptr; // lsa_cell_pos
    const ggml_tensor * lsa_j3_force_qp = nullptr; // lsa_q_pos
    int64_t lsa_j3_force_key[3] = {0, 0, 0};       // F, window, sinks
    // LSA index cache (LLAMA_LSA_IDX_CACHE=1): layers this graph rebuilds from the K cache; committed as valid
    // in set_input (so reserve graphs, which never run set_input, cannot mark a cache valid)
    std::vector<std::pair<const ggml_tensor *, uint64_t>> lsa_idx_commit;
    std::vector<std::pair<const ggml_tensor *, uint64_t>> lsa_mean_commit; // LLAMA_LSA_COMP=2 block-mean caches rebuilt
    // LLAMA_LSA_PICK_EVERY: per layer pick/reuse decisions of this graph, committed in set_input
    // expire = a ubatch with more than one sequence touched this layer without the pick cache: the next graph re-picks
    struct lsa_pick_rec { const ggml_tensor * k; uint64_t gen; bool reuse; int64_t pos0, edge, nbs; int32_t seq; bool expire = false; };
    mutable std::vector<lsa_pick_rec> lsa_pick_commit; // filled while the graph is built (const inp)
    // LLAMA_LSA_BLOCK_SUM: I32 [n_tokens] summary block of each row's index row (filled after the row tensor), the
    // summary block count it is clamped to, and the current layer's updated summaries (set by build_attn_lsa_idx for
    // layer lsa_sum_il, read by build_attn_lsa_mask of the same layer)
    ggml_tensor * lsa_q_blk    = nullptr;
    int64_t       lsa_n_blk    = 0;
    // LLAMA_LSA_COMP=2 block-mean refresh (llama_lsa_set_mean_inputs): distinct blocks of this ubatch, their staging-ring
    // blocks and each row's staging-ring row (the last two on SJ-KVaRN only, ring of lsa_ring_n rows)
    ggml_tensor * lsa_u_blk    = nullptr;
    ggml_tensor * lsa_u_rblk   = nullptr;
    ggml_tensor * lsa_q_ring   = nullptr;
    int64_t       lsa_ring_n   = 0;
    uint64_t      lsa_graph_gen = 0;  // llama_lsa_graph_gen() and first position at graph build (LLAMA_LSA_GRAPH_REUSE)
    int64_t       lsa_pos0      = 0;
    bool          lsa_ring_rows = false; // carry ring (not LLAMA_LSA_RING_LEGACY): u_rblk/q_ring as in llama_lsa_set_mean_inputs
    ggml_tensor * lsa_smax_cur = nullptr;
    ggml_tensor * lsa_smin_cur = nullptr;
    int           lsa_sum_il   = -1;
    // per-KV-head summaries (LLAMA_LSA_PER_HEAD=1 on SJ-KVaRN): one updated F16 [d_idx, n_blk] plane per head
    std::vector<ggml_tensor *> lsa_smax_h;
    std::vector<ggml_tensor *> lsa_smin_h;
    // LLAMA_LSA_BLOCK_SUM: per-ubatch base mask F32 [n_kv, T] (kq_mask at sinks / window / rows before min_pos,
    // kq_mask - 1e30 elsewhere), built by the first LSA layer and shared by the rest; keyed by the kq_mask it came from
    mutable ggml_tensor *       lsa_blk_base     = nullptr;
    mutable const ggml_tensor * lsa_blk_base_src = nullptr;
    // LLAMA_LSA_COMP=2: causally visible cells per B-cell block F32 [nbs, T] (sum of exp(kq_mask), kq_mask is 0 / -inf),
    // built by the first compensated layer and shared by the rest; keyed by the kq_mask it came from
    mutable ggml_tensor *       lsa_comp_vis     = nullptr;
    mutable const ggml_tensor * lsa_comp_vis_src = nullptr;
    // block pick -> COMP=2: visible droppable cells per block (once per graph) and the layer's dropped counts, valid
    // only for the mask tensor they were built with (lsa_comp_nd_mask)
    mutable ggml_tensor *       lsa_blk_dcnt     = nullptr;
    mutable const ggml_tensor * lsa_blk_dcnt_src = nullptr;
    mutable ggml_tensor *       lsa_comp_nd      = nullptr;
    mutable const ggml_tensor * lsa_comp_nd_mask = nullptr;
    // LLAMA_LSA_VK: lsa_blk_dcnt + 1 (once per graph), keyed by the count it came from
    mutable ggml_tensor *       lsa_vk_dcnt1     = nullptr;
    mutable const ggml_tensor * lsa_vk_dcnt1_src = nullptr;
    // LLAMA_LSA_BLKLIST: the per-graph F16 mask with GGML_LSA_MASK_SEL at visible droppable cells (keyed by the base A)
    mutable ggml_tensor *       lsa_blk_m3       = nullptr;
    mutable const ggml_tensor * lsa_blk_m3_src   = nullptr;
    // LLAMA_LSA_REFINE + FUSED_SEL: visible droppable cells per F-cell unit (once per graph, keyed by the base mask)
    mutable ggml_tensor *       lsa_unit_dcnt     = nullptr;
    mutable const ggml_tensor * lsa_unit_dcnt_src = nullptr;
    mutable ggml_tensor *       lsa_cpos_u        = nullptr; // LLAMA_LSA_UNIT_OP: first-cell position of each F-cell unit
    mutable int64_t             lsa_cpos_u_f      = 0;

    // LLAMA_LSA_MULTI=1 (opt-in): the stream (= sequence; SJ-KVaRN keeps one stream per sequence) whose LSA side caches
    // this input set addresses; a ubatch with several streams gets one LSA input set per stream in lsa_streams (only
    // their lsa_* fields are used, sized for that stream's rows and attended length) and none of its own
    int32_t lsa_stream = 0;
    std::vector<std::shared_ptr<llm_graph_input_attn_kv>> lsa_streams; // shared: the input is copied by some model builders
    int64_t lsa_sub_s  = 0; // this set's stream chunk of the parent ubatch (rows [s*T/ns, (s+1)*T/ns))
    int64_t lsa_n_kv_s = 0; // attended (padded) positions of that stream
    std::vector<llama_pos> lsa_sub_pos; // build-time sub-ubatch of that chunk (llm_graph_context::lsa_ubatch)
    llama_ubatch           lsa_sub_ub = {};
    mutable ggml_tensor *       lsa_sub_mask     = nullptr; // the stream's [n_kv_s, T/ns] slice of the KQ mask (once per graph)
    mutable const ggml_tensor * lsa_sub_mask_src = nullptr;
    // the LSA part of set_input (also used by llm_graph_input_mem_hybrid): positions, rows, ring slots, commits
    void set_input_lsa(const llama_kv_cache_context * kv, const llama_ubatch * ubatch);

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask_mla;
    llm_attn_mask self_attn_mask_lid;

    std::vector<ggml_tensor *> self_k_rot_lid;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};

// DSA input (full-attention layers + indexer) with K-only input for the SWA layers
class llm_graph_input_attn_k_dsa_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa_iswa(
            std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa,
            std::unique_ptr<llm_graph_input_attn_k>     inp_swa,
            const llama_kv_cache_dsa_iswa_context *     mctx) :
        inp_dsa(std::move(inp_dsa)),
        inp_swa(std::move(inp_swa)),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_attn_k_dsa * get_dsa() const { return inp_dsa.get(); }
    llm_graph_input_attn_k     * get_swa() const { return inp_swa.get(); }

    std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa;
    std::unique_ptr<llm_graph_input_attn_k>     inp_swa;

    const llama_kv_cache_dsa_iswa_context * mctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_msa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_msa_context * mctx);
    ~llm_graph_input_attn_kv_msa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_idx() const { return self_k_idxs_idx; }

    ggml_tensor * self_k_idxs_idx = nullptr; // I64 [n_batch]

    const llama_kv_cache_msa_context * mctx_msa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()         const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()         const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa()     const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa()     const { return self_v_idxs_swa; }
    ggml_tensor * get_k_idxs_swa_mid() const { return self_k_idxs_swa_mid; }
    ggml_tensor * get_v_idxs_swa_mid() const { return self_v_idxs_swa_mid; }

    ggml_tensor * self_k_idxs         = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs         = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa_mid = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa_mid = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;
    llm_attn_mask self_attn_mask_swa_mid;

    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    std::vector<ggml_tensor *> self_k_rot_swa;
    std::vector<ggml_tensor *> self_v_rot_swa;

    std::vector<ggml_tensor *> self_k_rot_swa_mid;
    std::vector<ggml_tensor *> self_v_rot_swa_mid;

    // INKKV: SJ-KVaRN base (the Inkling DSpark drafter, build_attn_inp_kv_iswa): region descriptor and per-stream
    // attended lengths; nullptr / empty otherwise (the Inkling trunk keeps its own in llm_graph_input_mem_hybrid_iswa_rel)
    ggml_tensor * self_sj_kvarn_desc = nullptr;
    std::vector<uint32_t> sj_kvarn_n_kv_streams;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_k_rot_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

// DSV4 raw graph inputs are SWA-only, but their mask may be stream-shaped
// so raw K can be concatenated with DSV4 compressed K in one attention op.
class llm_graph_input_dsv4_raw {
public:
    llm_graph_input_dsv4_raw(
            const llama_cparams & cparams,
            const llama_kv_cache_dsv4_raw_context * mctx) :
        cparams(cparams),
        mctx(mctx) {
    }

    void set_input(const llama_ubatch * ubatch);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;

    std::vector<ggml_tensor *> self_k_rot;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_raw_context * mctx;
};

class llm_graph_input_dsv4 : public llm_graph_input_i {
public:
    struct comp_input {
        ggml_tensor * state_pos        = nullptr; // I32 [n_state]
        ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_restore_src_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_restore_dst_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_snapshot_src_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_snapshot_dst_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_read_idxs  = nullptr; // I32 [ratio*n_state_write]
        ggml_tensor * state_write_idxs = nullptr; // I64 [n_state_write]
        ggml_tensor * state_write_pos  = nullptr; // I32 [n_state_write]

        ggml_tensor * kq_mask    = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]

        std::vector<ggml_tensor *> k_rot;
    };

    llm_graph_input_dsv4(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw,
            const llama_kv_cache_dsv4_context * mctx) :
        inp_raw(std::move(inp_raw)),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_dsv4() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_dsv4_raw * get_raw() const { return inp_raw.get(); }
    const comp_input & get_csa() const { return inp_csa; }
    const comp_input & get_hca() const { return inp_hca; }
    const comp_input & get_lid() const { return inp_lid; }

    std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw;

    comp_input inp_csa;
    comp_input inp_hca;
    comp_input inp_lid;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;


    llm_attn_mask cross_attn_mask;

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

// hybrid iswa memory with a per-head attention bias that depends on the distance from the query to the cell (Inkling)
// the KQ masks are always dense (rows == nullptr) and are filled from the same cells as the distances, so that the bias
// gathered by distance lines up with the mask
class llm_graph_input_mem_hybrid_iswa_rel : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa_rel(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx,
            uint32_t extent,
            uint32_t extent_swa,
            uint32_t extent_swa_mid = 0) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        extent(extent),
        extent_swa(extent_swa),
        extent_swa_mid(extent_swa_mid > 0 ? extent_swa_mid : extent_swa),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa_rel() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    // INKLSA (llama-inklsa.h, LLAMA_INKLSA=1 or per-layer LSA): sparse full-attention inputs, the visits' index caches
    // (with expected validity at build time) and the epochs a rebuild graph commits when it runs
    bool inklsa_on      = false;
    bool inklsa_rebuild = false;
    int64_t inklsa_sj_d = 0;
    llama_inklsa_inputs inklsa;
    std::vector<std::pair<const ggml_tensor *, bool>> inklsa_kcaches;
    mutable std::vector<std::pair<const ggml_tensor *, uint64_t>> inklsa_commit;

    // INKREL (llama-inklsa.h, default on): split full-attention inputs for wide ubatches
    llama_inkrel_inputs inkrel;

    // INKKV: SJ-KVaRN on the full-attention (base) cache: region descriptor and per-stream attended lengths
    ggml_tensor * self_sj_kvarn_desc = nullptr; // I32, see llama_kv_cache::build_input_sj_kvarn_desc
    std::vector<uint32_t> sj_kvarn_n_kv_streams;
    // INKKV v2 far/near split: [n_kv, T] F16 mask of the cells at distance >= extent (null = the dense v1 path)
    ggml_tensor * self_sj_far_mask = nullptr;
    ggml_tensor * self_sj_far_desc = nullptr; // INKKV: the SJ desc of the far nodes, with the far visible end (GGML_SJKVARN_DESC_FAR_END)
    int32_t       sj_il_full       = -1; // a full-attention layer (the SWA cache window copy check)

    // distance from the token to the cell, extent for the empty, future and more distant cells
    ggml_tensor * self_rel_pos         = nullptr; // I32 [n_kv,         n_batch]
    ggml_tensor * self_rel_pos_swa     = nullptr; // I32 [n_kv_swa,     n_batch]
    ggml_tensor * self_rel_pos_swa_mid = nullptr; // I32 [n_kv_swa_mid, n_batch]

    // [INKB1] layer-major prompt: the large per-ubatch inputs a one-visit graph reads from a device copy made by an
    // earlier graph of the same ubatch (bit set = bound to that copy, set_input leaves it alone); the far mask's
    // visible end comes with it (set_input stores the one it computes in ink_b1_far_end)
    enum {
        INK_B1_MASK         = 1 << 0,
        INK_B1_REL_POS      = 1 << 1,
        INK_B1_MASK_SWA     = 1 << 2,
        INK_B1_REL_POS_SWA  = 1 << 3,
        INK_B1_FAR_MASK     = 1 << 4,
        INK_B1_N_ROLES      = 5,
    };
    uint32_t ink_b1_bound   = 0;
    int64_t  ink_b1_far_end = 0;
    ggml_tensor * ink_b1_role(int r) const {
        switch (r) {
            case 0:  return inp_attn->self_attn_mask.raw;
            case 1:  return self_rel_pos;
            case 2:  return inp_attn->self_attn_mask_swa.raw;
            case 3:  return self_rel_pos_swa;
            case 4:  return self_sj_far_mask;
            default: return nullptr;
        }
    }

    const uint32_t extent;
    const uint32_t extent_swa;
    const uint32_t extent_swa_mid;

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
    std::vector<llama_seq_id> output_seqs;

    // row-block sampling: per sequence, the I32 input tensor holding the logit rows of that sequence
    struct row_input {
        ggml_tensor * t_rows;
        std::vector<int32_t> rows;
    };
    std::map<llama_seq_id, row_input> row_inputs;
};

// [#69] sampled MTP chain: the packed sampling params and the per-step uniforms, owned by the context
class llm_graph_input_mtp_chain_samp : public llm_graph_input_i {
public:
    llm_graph_input_mtp_chain_samp(const std::vector<float> * src) : src(src) {}
    virtual ~llm_graph_input_mtp_chain_samp() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * samp = nullptr; // F32 [LLAMA_MTP_CHAIN_SAMP_N]
    ggml_tensor * u    = nullptr; // F32 [n_chain]

    const std::vector<float> * src;
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};


// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

// fork: prompt routing is packed across all ubatches before the expert GEMMs.
struct llm_b1_ffn_bucket {
    struct span { size_t offset; size_t count; };
    std::vector<int32_t> rows;
    std::vector<int32_t> experts;
    std::vector<float> weights;
    // Each span belongs to one expert and therefore has unique token rows.
    std::vector<span> spans;
};

struct llm_b1_ffn {
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * inject = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
    std::vector<std::vector<int32_t>> expert_rows;
    std::vector<std::vector<float>> expert_weights;
    uint32_t max_rows = 8192;
    size_t piece_first = 0;
    size_t piece_count = 64; // Bound graph metadata independently of routing skew and the row dial.
    bool grouped_mmid = false;
    uint32_t routed_rows = 32768;
    size_t bucket_index = 0;
    std::vector<llm_b1_ffn_bucket> buckets;
};

// [POSTREV4] GGML_SJKVARN_MU_SPLIT=1 (default off): SJ-KVaRN FA nodes of a unified-KV multi-user ubatch carry the
// rows where a new sequence starts (ggml_flash_attn_ext_set_row_groups), so a 9..32-row verify can run one decode
// launch per user that reads only that user's records instead of the tile kernel over the whole unified cache
static inline bool llm_graph_mu_split_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_SJKVARN_MU_SPLIT");
        return e != nullptr && *e != '\0' && atoi(e) != 0;
    }();
    return on;
}

// bit r = token r starts a new sequence (bit 0 always); 0 when the ubatch is not a 9..32-token multi-sequence one
static inline uint32_t llm_graph_ubatch_row_groups(const llama_ubatch & ub) {
    if (!llm_graph_mu_split_enabled() || ub.n_tokens <= 8 || ub.n_tokens > 32 || ub.n_seqs_unq < 2 ||
            ub.seq_id == nullptr || ub.n_seq_id == nullptr) {
        return 0;
    }
    uint32_t g = 1u;
    for (uint32_t r = 1; r < ub.n_tokens; ++r) {
        if (ub.n_seq_id[r] < 1 || ub.n_seq_id[r - 1] < 1 || ub.seq_id[r][0] != ub.seq_id[r - 1][0]) {
            g |= 1u << r;
        }
    }
    return g;
}

struct llm_graph_params {
    const llama_model * model = nullptr;

    llama_hparams hparams;
    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_hadamard_rotations * hadamard_rotations;
    const llama_hadamard_rotations * hadamard_inverses;
    const llama_moe_cache        * moe_cache;

    const llama_prec_policy * prec_policy = nullptr;

    const llama_hadamard * hdmd = nullptr;

    ggml_tensor * layer_inp_dev = nullptr; // own layer input device buffer (write)
    ggml_tensor * layer_inp_src = nullptr; // ctx_other's layer input device buffer (read)
    ggml_tensor * h_nextn_dev   = nullptr; // own h_nextn device buffer (write)
    ggml_tensor * h_nextn_src   = nullptr; // ctx_state_src's h_nextn device buffer (read)

    std::map<llama_seq_id, llama_sampler *> samplers;

    // draft-only vocabulary shortlist: [n_sel] I32 token ids owned by the context, or nullptr
    ggml_tensor * draft_vocab_ids = nullptr;

    // [#81] compact resident draft head [n_embd, n_sel] owned by the context, and the head tensor it replaces
    // (the compact logits are used only for that head), or nullptr
    ggml_tensor       * draft_vocab_compact     = nullptr;
    const ggml_tensor * draft_vocab_compact_src = nullptr;

    // bit per full-head fallback reason already warned about, owned by the context (warn once per context)
    std::atomic<uint32_t> * draft_vocab_warned = nullptr;

    // [#69] sampled MTP chain inputs, owned by the context (llama_context::mtp_chain_samp)
    const std::vector<float> * mtp_chain_samp = nullptr;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // fork: the owning model (EXL3 side tensors for build_lora_mm) is the upstream `model` member at the top

    // [B1] layer-major prefill (LLAMA_B1, qwen4exp): build only the layers [b1_il0, b1_il1) (b1_il1 < 0: to the end).
    // A range that starts above layer 0 reads the hidden state from b1_carry, a range that stops before the last layer
    // writes it back there and builds no output. Set after construction.
    int32_t       b1_il0   = 0;
    int32_t       b1_il1   = -1;
    ggml_tensor * b1_carry = nullptr;

    // [PLEPF] the context's PLE page prefetch (null when off); set after construction
    llama_ple_prefetch * ple_pf = nullptr;
    const llama_ink_stream * moe_stream = nullptr; // fork (INKSTREAM): armed prompt expert streamer, or nullptr
    const llama_peer_experts * moe_peer = nullptr; // [PEERX] second-card expert tier (GGML_PEER_EXPERTS), or nullptr

    // [RTPORT] 1-row MTP step whose token embedding row is a device input (LLAMA_MTP_PIPELINE); set after construction
    bool mtp_tok_row = false;
    // fork AIPFIX ONEPATH: ggml_moe_bank_onepath_live() when the params were taken; a graph built either way is only
    // reused while it holds (the one path drops the bank graph)
    bool moe_onepath_live = false;

    // true for a graph built only to reserve compute buffers (llama_context::graph_reserve): its output rows are
    // placeholders and may sit on sequences that have no backend sampler (none is attached yet at context creation)
    // set after construction, so it stays last
    bool is_reserve = false;
    const llm_b1_ffn * b1_ffn = nullptr;
    int b1_ffn_phase = 0; // 1: attention/router, 2: packed experts, 3: residual combine


    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            ubatch.is_mixed()   == other.ubatch.is_mixed() &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            ) &&
            (!ubatch.embd_state == !other.ubatch.embd_state) &&
            (!ubatch.embd_row   == !other.ubatch.embd_row);

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        // [POSTREV4] the FA row-group hint is baked into the graph; a different sequence layout rebuilds it
        if (can_reuse_ubatch && llm_graph_mu_split_enabled() && !ubatch.equal_seqs()) {
            can_reuse_ubatch = ubatch.data && llm_graph_ubatch_row_groups(ubatch) == llm_graph_ubatch_row_groups(other.ubatch);
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        // TODO: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448035248
        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset || cparams.dspark_walk != other.cparams.dspark_walk ||
            cparams.dspark_ids_out != other.cparams.dspark_ids_out || cparams.dspark_sample_k != other.cparams.dspark_sample_k ||
            cparams.dspark_sample_t != other.cparams.dspark_sample_t) {
            return false;
        }

        if (b1_il0 != other.b1_il0 || b1_il1 != other.b1_il1 || b1_carry != other.b1_carry || b1_ffn != other.b1_ffn || b1_ffn_phase != other.b1_ffn_phase) {
            return false;
        }

        if (mtp_tok_row != other.mtp_tok_row) {
            return false;
        }

        if (moe_onepath_live != other.moe_onepath_live) {
            return false;
        }

        if (moe_stream != other.moe_stream) {
            return false;
        }

        if (moe_peer != other.moe_peer) {
            return false;
        }

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.mtp_chain               == other.cparams.mtp_chain               &&
            cparams.mtp_chain_top_k         == other.cparams.mtp_chain_top_k         &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            layer_inp_dev == other.layer_inp_dev &&
            layer_inp_src == other.layer_inp_src &&
            h_nextn_dev   == other.h_nextn_dev   &&
            h_nextn_src   == other.h_nextn_src   &&
            model == other.model &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross;
    }
};


class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;
    const llm_b1_ffn * get_b1_ffn() const { return params.b1_ffn; }
    int get_b1_ffn_phase() const { return params.b1_ffn_phase; }
    bool get_moe_onepath_live() const { return params.moe_onepath_live; }

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);

    // Hadamard-transformed activations, keyed by (input, rotation): folded weights that read the same activation share one transform
    ggml_tensor * get_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot) const;
    void          set_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot, ggml_tensor * res);

    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    void set_params(const llm_graph_params & params);



    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm
    ggml_tensor * t_logits_ids  = nullptr; // [n_sel] when t_logits rows are a vocabulary shortlist: row -> token id

    std::vector<ggml_tensor *> t_layer_inp;

    std::map<llama_seq_id, ggml_tensor *> t_sampled_logits;
    std::map<llama_seq_id, ggml_tensor *> t_candidates;
    std::map<llama_seq_id, ggml_tensor *> t_sampled;
    ggml_tensor * t_greedy_rows = nullptr;
    std::vector<uint32_t> greedy_rows;

    // row-block sampling (speculative verification): per sequence, the sampled ids of all its output rows
    // ([n_rows] I32) and the logit row of each entry
    std::map<llama_seq_id, ggml_tensor *> t_sampled_rows;
    std::map<llama_seq_id, std::vector<uint32_t>> sampled_rows;
    // top-k row export (a row-block chain that only truncates): per sequence the K kept ids and logits of each row
    // ([K, n_rows] I32 and F32), sent to the host instead of the full logit rows
    std::map<llama_seq_id, std::pair<ggml_tensor *, ggml_tensor *>> t_topk_rows;
    std::vector<llm_graph_fused_node> fused_nodes;
    std::map<llama_seq_id, ggml_tensor *> t_sampled_probs;

    std::vector<ggml_tensor *> t_sampling_all;

    std::vector<llm_graph_input_ptr> inputs;
    bool mtp_device_input = false; // fork: device-input and host-input allocations cannot share a graph

    std::map<std::pair<const ggml_tensor *, const ggml_tensor *>, ggml_tensor *> hdmd_inputs;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::unique_ptr<uint8_t[]> buf_compute_meta;
    size_t buf_compute_meta_size = 0;

    ggml_cgraph * gf;

    int64_t max_nodes;

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_amax_src1 {
    ggml_tensor * res;     // cur rescaled
    ggml_tensor * scale_s; // scale to undo on the matmul result, or nullptr
};

struct llm_graph_moe_cache_slots {
    ggml_tensor * slots;   // [n_expert_used, n_tokens] slots of the selected experts in the MoE cache, or nullptr
    ggml_tensor * cpu_ids; // [n_expert_used, n_tokens] misses computed from the host experts, -1 for the other experts, or nullptr
};

struct llm_graph_context {
    const llama_model & model;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;
    // LLAMA_LSA_MULTI=1: the LSA builders read lsa_ubatch(), the stream's own rows while a per-stream LSA layer is built
    mutable const llama_ubatch * lsa_ub = nullptr;
    const llama_ubatch & lsa_ubatch() const { return lsa_ub ? *lsa_ub : ubatch; }

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_layer_nextn;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_hadamard_rotations * hadamard_rotations;
    const llama_hadamard_rotations * hadamard_inverses;
    const llama_moe_cache        * moe_cache;
    const llama_ink_stream       * moe_stream = nullptr;
    mutable int                    moe_stream_il = -1; // visit whose experts build_lora_mm_id reads from the streamer
    const llama_peer_experts     * moe_peer = nullptr; // [PEERX] second-card expert tier, or nullptr
    mutable bool                   moe_peer_on = false; // build_lora_mm_id reads the peer copies of the experts

    const llama_prec_policy * prec_policy;

    const llama_hadamard * hdmd;

    ggml_tensor * layer_inp_dev;
    ggml_tensor * layer_inp_src;
    ggml_tensor * h_nextn_dev;
    ggml_tensor * h_nextn_src;

    std::map<llama_seq_id, llama_sampler *> samplers;

    ggml_tensor * draft_vocab_ids; // see llm_graph_params
    ggml_tensor       * draft_vocab_compact;     // see llm_graph_params
    const ggml_tensor * draft_vocab_compact_src; // see llm_graph_params
    std::atomic<uint32_t> * draft_vocab_warned; // see llm_graph_params
    const bool is_reserve; // see llm_graph_params
    const std::vector<float> * mtp_chain_samp; // see llm_graph_params
    const int32_t       b1_il0;   // see llm_graph_params
    const int32_t       b1_il1;   // see llm_graph_params (-1 = n_layer)
    ggml_tensor * const b1_carry; // see llm_graph_params
    llama_ple_prefetch * const ple_pf; // see llm_graph_params
    const bool mtp_tok_row; // see llm_graph_params

    // [#81] logits of the compact draft head for `cur` [n_embd, n], [n_sel, n]: plain mul_mat on the activation the
    // head consumes (EXL3: x_rot = H128(suh * cur), the fused kernel's input glue built from generic ops)
    ggml_tensor * build_draft_vocab_compact(ggml_tensor * head_w, ggml_tensor * head_s, ggml_tensor * cur) const;

    // full-head fallback of build_draft_vocab_logits: warns once per context
    ggml_tensor * draft_vocab_fallback(int reason, const char * detail) const;

    // Shared transforms are valid for one graph build only.
    mutable std::map<std::tuple<const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, int64_t, int64_t, int64_t>, ggml_tensor *> hadamard_memo;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    const llama_model * model_ref; // EXL3 side tensors (llama_model::exl3_side_of); null if none

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    // SJ-KVaRN: handed from build_attn (which owns the cache) to build_attn_mha (which owns the FA node)
    mutable struct {
        ggml_tensor * body = nullptr; // seal output (view of the layer's record pool)
        ggml_tensor * desc = nullptr;
        ggml_tensor * selected = nullptr; // QSA candidate positions; invalid slots lie outside the mask
        int32_t bits_k = 0;
        int32_t bits_v = 0;
        bool fused_rot = false; // Q/output rotation inside the FA node (ggml_sj_kvarn_fused_rot_enabled)
        const std::vector<uint32_t> * n_kv_streams = nullptr; // multi-stream: per-stream n_kv (see llm_graph_input_attn_kv)
        ggml_tensor * wush_ao = nullptr; // EXPERIMENTAL WUSH-KV: output transform replacing the inverse Hadamard
        bool inksj = false; // INKKV: Inkling pair-packed heads (ggml-inksj.h marker on the node)
    } sj_kvarn_pending;
    // QSA prompt (LLAMA_QSA_DIRECT_LISTS): final selected-cell rows [n_sel, n_tokens] for the next flash attention node's
    // src[8], so the CUDA sparse path builds its per-tile cell lists from them instead of scanning the n_kv-wide mask
    mutable ggml_tensor * fa_sel_rows = nullptr;
    // LSA compensation (LLAMA_LSA_COMP=2): when set, the next plain flash attention node also writes its row
    // log-sum-exp to lsa_lse (F32 [n_head, n_tokens, n_seq]); the caller takes it and clears both
    mutable bool          lsa_lse_req = false;
    // INKKV row-chunked rel-bias path: one view per (graph input, first row, rows), shared by every layer, so the
    // scheduler uploads each input chunk once per split instead of once per layer (ctx x ub x layers otherwise)
    mutable std::map<std::tuple<const ggml_tensor *, int64_t, int64_t>, ggml_tensor *> inkkv_row_views;
    // Logging only: an earlier full-attention layer's residual projected through this layer's exact Q weights.
    mutable ggml_tensor * lsa_spec_q = nullptr;
    mutable ggml_tensor * lsa_spec_q2 = nullptr; // same target, residual from two attention layers earlier
    mutable ggml_tensor * lsa_spec_q3 = nullptr; // same target, residual from three attention layers earlier
    mutable ggml_tensor * lsa_spec_sel = nullptr;
    // LLAMA_LSA_PICK_EVERY: K cache tensor of the layer whose LSA mask is being built (key of its pick cache); set by
    // build_attn_lsa_mask, taken and cleared by build_attn_lsa_mask_sb
    mutable const ggml_tensor * lsa_pick_kbase = nullptr;
    // LLAMA_LSA_SUM_OP: query index rows + summary plane views handed from build_attn_lsa_mask_blk to the fused select
    mutable ggml_tensor * lsa_sum_q = nullptr;
    mutable ggml_tensor * lsa_sum_x = nullptr;
    mutable ggml_tensor * lsa_sum_n = nullptr;
    // LLAMA_LSA_BLKLIST: block pick [nbs] and block size for the layer's SJ-KVaRN flash attention node; set by
    // build_attn_lsa_mask_sb, taken and cleared by build_attn_mha (cleared by build_attn if no node took it)
    mutable ggml_tensor * lsa_fa_sel = nullptr;
    mutable int32_t       lsa_fa_blk = 0;
    mutable const ggml_tensor * lsa_fa_m3 = nullptr; // the mask the pick belongs to (attached only to an FA node on it)
    mutable ggml_tensor * lsa_lse     = nullptr;
    // LLAMA_LSA_VK: the layer's rotated block means ([D*H_kv, lsa_vk_nbs] views, set by build_attn before the mask
    // build when virtual keys can run); the block-list branch then hands the FA node the virtual-key plane instead of
    // the pick (lsa_fa_vk), and build_attn_mha attaches both (lsa_vk_done: the node took them, no merge follows)
    mutable ggml_tensor * lsa_vk_km  = nullptr;
    mutable ggml_tensor * lsa_vk_vm  = nullptr;
    mutable int64_t       lsa_vk_nbs = 0;
    mutable bool          lsa_fa_vk   = false;
    mutable int32_t       lsa_fa_vk_n = 0;     // > 0: the plane holds lsa_fa_vk_n mean-block counts, then the unit pick
    mutable bool          lsa_vk_done = false;
    // multi-stream SJ-KVaRN: per-stream mask views cut to the stream's own n_kv are not contiguous when the ubatch
    // has more than one token row, and FA needs a contiguous mask; the contiguous copies are made once per graph
    // (the mask is shared by every layer)
    // hot-expert bank of the next build_moe_ffn call (set by the model, consumed by the call):
    // the selected experts are split between an always-resident hot bank and the regular (cold) bank
    struct moe_hot_bank {
        ggml_tensor * up       = nullptr;
        ggml_tensor * gate     = nullptr;
        ggml_tensor * down     = nullptr;
        ggml_tensor * hot_map  = nullptr; // F32 [n_expert]: hot slot or -1
    };
    mutable moe_hot_bank  moe_hot;
    mutable ggml_tensor * moe_weights_override = nullptr;
    mutable bool          moe_defer_reduce     = false;
    mutable bool          moe_b1_bypass        = false; // [B1] hot bank call inside the routing phase: compute it now
    mutable ggml_tensor * moe_route_skip       = nullptr; // per-route value >= 0: the cold MUL_MAT_IDs may skip the route
    // fork AIPFIX ONECHAIN: prompt batches of a banked layer run one expert chain; after each regular expert product
    // the bank product of the same projection fills the bank routes (moe_onechain_ids >= 0), see build_moe_ffn
    mutable const moe_hot_bank * moe_onechain_bank = nullptr;
    mutable ggml_tensor * moe_onechain_ids  = nullptr; // I32 [n_expert_used, n_tokens]: bank slot or -1
    mutable ggml_tensor * moe_onechain_cold = nullptr; // F32 [1, n_expert_used, n_tokens]: 1 = regular route, 0 = bank route
    mutable ggml_tensor * sj_kvarn_mask_src = nullptr;
    mutable std::vector<ggml_tensor *> sj_kvarn_mask_s;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;
    ggml_tensor * build_moe_onechain_fill(ggml_tensor * regular, ggml_tensor * bank_w, ggml_tensor * x) const;

    // true when the last layer must be narrowed to the output rows before the nextn hidden state is captured
    bool crop_before_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    }

    // true when the nextn hidden state must be narrowed to the output rows after it is captured
    bool crop_after_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;
    }

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // apply the activation-side transform of a Hadamard-folded weight, if any
    ggml_tensor * build_hadamard_input(
              ggml_tensor * w,
              ggml_tensor * cur) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    // if slots is set, the experts are selected by slots instead of ids, from the MoE cache if cached (see build_moe_cache_slots)
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s    = nullptr,
              ggml_tensor * slots  = nullptr,
                     bool   cached = true) const;

    // rescale `cur` (the src1 of a matmul) by a power of two, computed from its maximum absolute
    // keep both individual values and Q8 activation block sums within F16 range
    llm_graph_amax_src1 build_amax_src1_f16(
              ggml_tensor * cur) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    // the slots of the selected experts in the MoE cache, nullptr if the experts of the layer are not read from the cache
    // with cpu_part, also the misses computed from the host experts, if the cache leaves any to the CPU
    // fork (INKSTREAM): slots of the streamed experts of visit il (an identity lookup that hooks the copy), or nullptr
    ggml_tensor * build_moe_stream_slots(ggml_tensor * selected_experts, int il) const;
    // [PEERX] pin the expert chain that ends in out onto the peer card (stops at the given inputs and at leaves)
    void build_moe_peer_pin(ggml_tensor * out, const std::vector<const ggml_tensor *> & stops, int il) const;

    llm_graph_moe_cache_slots build_moe_cache_slots(
             ggml_tensor * selected_experts,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * gate_up_exps,
                    bool   cpu_part,
                     int   il) const;

    //
    // inputs
    //

    // tok_scale: applied to token rows only
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd, float tok_scale = 1.0f) const;
    ggml_tensor * build_hadamard_inverse(ggml_tensor * table, ggml_tensor * cur) const;

    // the state input of an MTP graph: rows of the h_nextn device buffer of ctx_state_src (see llama_set_ctx_state_src()),
    // or uploaded from ubatch.embd_state
    ggml_tensor * build_inp_embd_state(llm_graph_input_embd_h * inp, int64_t n_embd_state) const;

    ggml_tensor * build_inp_pos() const;

    // applies the RoPE of layer il, cur is returned as is if the layer has none
    ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) const;

    // same, for the lightning indexer
    ggml_tensor * build_rope_indexer(ggml_tensor * cur, ggml_tensor * pos, int il) const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * kv_rows, // [n_kv, n_stream], k and v are shared by the streams
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;

    // LSA probe (llama-lsa.h): replacement KQ mask that keeps sinks + window + the best-scoring cells per query
    // row, or nullptr when the layer stays dense. *n_kv_max gets the kept count when LLAMA_LSA_FA_SPARSE=1.
    // LLAMA_LSA_FAST=1: *k and *v are replaced by the gathered kept cells and the returned mask covers only them.
    ggml_tensor * build_attn_lsa_mask(
            const llm_graph_input_attn_kv * inp,
            ggml_tensor * q,       // [n_embd_head, n_head, n_tokens] post-RoPE
            ggml_tensor ** k,      // KV cache view [n_embd_head, n_head_kv, n_kv, n_stream]
            ggml_tensor ** v,      // same layout as k (fast path only)
            ggml_tensor * kq_mask, // [n_kv, n_tokens, 1, n_stream]
                    int   il,
                int64_t * n_kv_max,
            ggml_tensor * idx = nullptr,        // F16 [d_idx, n_kv] cached index keys, or nullptr = recompute
                  float   kq_scale_lsa = 1.0f,  // mode=mass: softmax scale
            std::vector<ggml_tensor *> * masks_h = nullptr) const; // per-head picks (LLAMA_LSA_PER_HEAD=1) land here

    // LSA index cache (LLAMA_LSA_IDX_CACHE=1): stores the new tokens' index keys (or rebuilds the layer from the K
    // cache after an invalidation) and returns the F16 [d_idx, n_kv] view the scoring reads, or nullptr
    ggml_tensor * build_attn_lsa_idx(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * k_cur,   // [n_embd_head, n_head_kv, n_tokens] as stored
            ggml_tensor * k,       // KV cache view after the store
                    int   il) const;

    // LLAMA_LSA_BLOCK_SUM: refresh the block summaries after the index store w (full = every whole block below n_kv,
    // else only the blocks of this ubatch's rows) and publish them for this layer's mask
    void build_attn_lsa_sum(
            llm_graph_input_attn_kv * inp,
            const llama_lsa_idx_slot & slot,
            ggml_tensor * w,       // index cache after this graph's store
                int64_t   n_kv,
                   bool   full,
                    int   il) const;

    // LLAMA_LSA_BLOCK_SUM mask: blocks ranked by the summary upper bound max.q+ + min.q-, top keep/block blocks
    // kept whole, plus sinks + window per cell; nullptr = fall back to the per-cell path
    ggml_tensor * build_attn_lsa_mask_blk(
            const llm_graph_input_attn_kv * inp,
            ggml_tensor * qi,      // F32 [d_idx, T] index queries
            ggml_tensor * kq_mask,
                int64_t   n_kv,
                int64_t   keep_lr,
                    int   il,
            ggml_tensor * smax = nullptr,  // summaries to rank (default: inp->lsa_smax_cur / lsa_smin_cur)
            ggml_tensor * smin = nullptr,
                   bool   keep_f32 = false) const; // return the F32 mask (caller casts)

    // block scores -> mask: top keep_lr/B blocks per row (blocks inside the window or the sinks excluded), whole blocks
    // kept, the causal/sink/window part from the shared per-graph base (decode/verify widths); hands COMP=2 its dropped
    // counts (inp->lsa_comp_nd). Used by the block-sum pick and by the record pick's LLAMA_LSA_REC_BLK=1.
    ggml_tensor * build_attn_lsa_mask_sb(
            const llm_graph_input_attn_kv * inp,
            ggml_tensor * Sb,      // F32 [nbs, T] block scores, nbs = ceil(n_kv/B) or fewer; nullptr: S_cells
            ggml_tensor * kq_mask,
                int64_t   n_kv,
                int64_t   keep_lr,
                    int   il,
                   bool   keep_f32 = false,
            ggml_tensor * S_cells  = nullptr, // F32 [n_kv, T] cell scores (LLAMA_LSA_FUSED_SEL: block logsumexp in the op)
                  float   tau      = 1.0f) const;

    // per-graph base mask of the block/unit picks (inp->lsa_blk_base at decode/verify widths)
    ggml_tensor * build_attn_lsa_blk_base(
            const llm_graph_input_attn_kv * inp, ggml_tensor * kq_mask, int64_t n_kv, int64_t T, int il) const;

    // LLAMA_LSA_REFINE + FUSED_SEL: mask (+ COMP=2 dropped counts) from a 0/1 selection of U-cell units
    // (shared_pick: every row of selu holds the same pick, so LLAMA_LSA_BLKLIST may hand it to the SJ-KVaRN node)
    ggml_tensor * build_attn_lsa_mask_units(
            const llm_graph_input_attn_kv * inp, ggml_tensor * selu, int64_t U, ggml_tensor * kq_mask, int64_t n_kv,
            int il, bool shared_pick = false) const;

    // LLAMA_LSA_BLKLIST helpers: node/shape conditions for a block list of B-cell blocks; per-graph sentinel mask M3
    bool lsa_blist_ok(int64_t T, int64_t B, const ggml_tensor * kq_mask) const;
    ggml_tensor * build_attn_lsa_blk_m3(
            const llm_graph_input_attn_kv * inp, ggml_tensor * A, ggml_tensor * kq_mask, int64_t n_kv, int64_t T, int il) const;

    // SJ-KVaRN variant of build_attn_lsa_idx (k_cur before the SJ-KVaRN rotation; rows = positions)
    ggml_tensor * build_attn_lsa_idx_sj(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * k_cur,
            ggml_tensor * k,
                    int   il) const;

    // LLAMA_LSA_J3: side-stream slot prefetch for LSA layer il, built at an earlier (source) layer so it runs there in
    // graph order. sq: [d_head, n_head, T] query of layer il computed from the source layer's residual (or nullptr: the
    // previous-pick source only); pct: pick budget in percent; flags: GGML_LSA_PREFETCH_PREV/_PROTECT_PREV; src: scratch.
    void build_attn_lsa_j3(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * sq,
                    int   il,
                    int   pct,
                    int   flags,
                    int   src,
            llm_graph_input_attn_kv * shadow = nullptr, // LLAMA_LSA_J3_LEAD + LLAMA_LSA_MULTI: stream s input set of inp
                int64_t   s = 0) const;
    // the [n_kv/F, T] 0/1 unit prediction of layer il's record pick from sq (nullptr: layer il would not pick)
    ggml_tensor * build_attn_lsa_j3_units(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * sq,
                    int   il,
                    int   pct) const;

    // LSA fast path (LLAMA_LSA_FAST=1): S = [n_kv, T] index scores; gathers the kept cells into *k / *v and returns
    // their mask, or nullptr when the layer stays dense
    ggml_tensor * build_attn_lsa_fast(
            const llm_graph_input_attn_kv * inp,
            ggml_tensor * S,
            ggml_tensor ** k,
            ggml_tensor ** v,
            ggml_tensor * kq_mask,
                int64_t   keep_lr,
                    int   il) const;

    // LSA centroid compensation (LLAMA_LSA_COMP=1): o_kept = the masked FA output [Dv*H, T]; adds the dropped blocks'
    // stand-in keys/values to the softmax and returns the corrected output, or nullptr if the shapes are not supported
    // LLAMA_LSA_COMP=2: the same correction from the masked FA output's log-sum-exp and the cached block means (no dense
    // scores); nullptr if the shapes/cache are not supported
    // Refresh (every ubatch, any width): plain caches recompute this ubatch's blocks from the K/V cache (all whole
    // blocks when the cache is invalid); SJ-KVaRN (k_new/v_new = this ubatch's un-rotated K/V) writes them to the
    // staging ring and recomputes this ubatch's blocks from it, valid only once built from position 0. Returns false
    // when unsupported or not valid for use; *kmv/*vmv = [D*H_kv, *nbs] views ordered after the refresh.
    // rot: k_new/v_new are in the rotated SJ-KVaRN basis (LLAMA_LSA_VK), so are the means (separate cache tag).
    bool build_attn_lsa_mean_refresh(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * k,
            ggml_tensor * v,
            ggml_tensor * k_new,
            ggml_tensor * v_new,
                int64_t   n_kv,
            ggml_tensor ** kmv,
            ggml_tensor ** vmv,
                int64_t * nbs,
                    int   il,
                   bool   rot = false) const;

    ggml_tensor * build_attn_lsa_comp_fast(
            const llm_graph_input_attn_kv * inp,
            ggml_tensor * o_kept,
            ggml_tensor * lse,
            ggml_tensor * q,
            ggml_tensor * kmv,
            ggml_tensor * vmv,
                int64_t   nbs,
                int64_t   H_kv,
            ggml_tensor * lsa_mask,
            ggml_tensor * kq_mask,
                  float   kq_scale,
                    int   il) const;

    ggml_tensor * build_attn_lsa_comp(
            ggml_tensor * o_kept,
            ggml_tensor * q,
            ggml_tensor * k,
            ggml_tensor * v,
            ggml_tensor * lsa_mask,
            ggml_tensor * kq_mask,
                  float   kq_scale,
                    int   il) const;

    // TurboQuant/TQ K caches hold WHT-rotated K (the set_rows quantizer rotates on store): pad Q per head
    // to the cached K head size and rotate it to match. No-op for other K types and for SJ-KVaRN caches
    // (their tq6_0 staging K is stored in the SJ-KVaRN Hadamard basis without the turbo WHT). Any graph that calls
    // build_attn_mha on llama_kv_cache views must apply this and build_attn_turbo_v_out as build_attn does.
    ggml_tensor * build_attn_turbo_q(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
      const ggml_tensor * k,       // K cache view (its type selects the rotation)
      const llama_kv_cache_context * mctx_cur) const;

    // TurboQuant: build_attn_mha undoes the V rotation on its output; when the cached V head was padded
    // to 128, cut the output back to n_embd_head_v(il). No-op for other V types.
    ggml_tensor * build_attn_turbo_v_out(
            ggml_tensor * cur,     // [n_embd_head_v_cache * n_head_q, n_tokens] build_attn_mha output
      const ggml_tensor * v,       // V cache view
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    // [#91] the K/V store half of the build_attn above, for graphs whose attention output nothing reads
    void build_attn_store(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
                    int   il) const;

    // SJ-KVaRN steps of build_attn above, for model graphs that store K/V and call build_attn_mha themselves
    // (qwen4exp QSA): rotate q/k/v into the cache basis before the stores (returns true when the FA node rotates
    // q and un-rotates its output itself), then, after the stores, hand the layer's sealed records and descriptor
    // to the next build_attn_mha. No-ops unless the cache is SJ-KVaRN.
    bool build_attn_sj_kvarn_rot(
            llm_graph_input_attn_kv * inp,
            ggml_tensor *& q_cur,
            ggml_tensor *& k_cur,
            ggml_tensor *& v_cur,
                    int   il) const;

    void build_attn_sj_kvarn_pend(
            llm_graph_input_attn_kv * inp,
                   bool   fused_rot,
                    int   il) const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    llm_graph_input_attn_k_dsa_iswa * build_attn_inp_k_dsa_iswa() const;

    llm_graph_input_attn_kv_msa * build_attn_inp_kv_msa(bool msa_enabled) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;

    // ggml_top_k_mask with top_k viewed in the row layout of kq_mask, see ggml.h
    ggml_tensor * build_fused_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;

    // kq_mask with only the cells named by top_k left visible, for sparse attention
    ggml_tensor * build_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    llm_graph_input_dsv4 * build_inp_dsv4() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;

    // note: if k_cur is not provided, it will not be stored in the memory
    // note: the K cache is used as V (MLA-style attention)
    ggml_tensor * build_attn(
            llm_graph_input_attn_k_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,

            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy,
            ggml_tensor * state_copy_main,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = nullptr) const;

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = nullptr) const;

    // [CONVGATHER] the conv history of the layer whose state is s (a plane of the memory's r_all), from one gather of
    // every layer's rows per graph; nullptr when the graph cannot take it (then build_rs)
    ggml_tensor * build_rs_all(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs) const;

    // the conv history gather: build_rs, except that under --gdn-replay's static verify graph the rows
    // come from the pending rollback's group (inp->s_copy_conv)
    ggml_tensor * build_rs_conv(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs) const;
    // store the conv history of the ubatch cells, the last n_cols columns of conv_x {n_cols + n_seq_tokens, channels, n_seqs},
    // at byte offset offs of each row of s, and with n_rs_seq > 0 once per rollback slot of this ubatch
    void build_conv_state_store(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
            ggml_tensor * conv_x,
                int64_t   n_cols,
                 size_t   offs,
                   bool   slot_cpy = false) const; // slot_cpy: one CPY per rollback slot instead of the SET_ROWS

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    // extent, extent_swa: the distance that the bias tables of the full and the SWA layers stop at
    llm_graph_input_mem_hybrid_iswa_rel * build_inp_mem_hybrid_iswa_rel(
            uint32_t extent,
            uint32_t extent_swa,
            uint32_t extent_swa_mid = 0) const;

    // Inkling 2 attention: kq_b is gathered here from the per-query bias table [n_head_q, extent + 1, n_tokens]
    // (row extent = the zero pad row minus the sink, used for every cell at distance >= extent); the full-attention
    // visits take the INKLSA sparse path when it is on (llama-inklsa.h)
    ggml_tensor * build_attn_rel(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * table, // [n_head_q, extent + 1, n_tokens]
            ggml_tensor * sinks, // [n_head_q]
                  float   kq_scale,
                    int   il,
            ggml_tensor * gate = nullptr) const; // [n_embd_head_v*n_head_q, n_tokens]: returns out * sigmoid(gate)

    // the body of build_attn_rel; sets gated when the gate was applied inside the attention (INKATTN)
    // INKKV: the dense / near per-head bias path in row chunks of tc rows per stream
    ggml_tensor * build_attn_rel_chunked(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * table,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il,
                  int64_t tc) const;

    ggml_tensor * build_attn_rel_base(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * table,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il,
            ggml_tensor * gate,
                   bool & gated) const;

    // INKKV: full-attention visit over an SJ-KVaRN base cache. The 128-wide K/V heads are stored in pairs as 256-wide
    // heads and Q is packed into the matching half; kq_mask is the dense per-head mask [n_kv, n_tokens, n_head_q] F16
    ggml_tensor * build_attn_rel_sj(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il,
            ggml_tensor ** lse_out = nullptr) const; // INKKV v2: also the row log-sum-exp [n_head, T]

    // INKKV: the pair-packed SJ-KVaRN visit on any iSWA input (desc = the SJ region descriptor of this node), and the
    // K/V store alone (the Inkling DSpark drafter writes its context K/V without attending)
    ggml_tensor * build_attn_sj_pair(
            llm_graph_input_attn_kv_iswa * inp_attn,
            ggml_tensor * desc,
            const std::vector<uint32_t> * n_kv_streams,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il,
            ggml_tensor ** lse_out = nullptr) const;
    void build_sj_pair_store(
            llm_graph_input_attn_kv_iswa * inp_attn,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
                    int   il) const;

    // INKREL (llama-inklsa.h): full-attention visit as far + near flash attention merged by log-sum-exp; k and v are
    // the cache views with this ubatch already stored, the sink logit is zero (folded into table)
    ggml_tensor * build_attn_inkrel(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * q_cur,
            ggml_tensor * k,
            ggml_tensor * v,
            ggml_tensor * table,
                  float   kq_scale,
                    int   il) const;

    // kq_b is added to the KQ mask, after the KQ scale
    ggml_tensor * build_attn(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,  // [n_kv, n_tokens/n_stream, n_head_q, n_stream]
            ggml_tensor * sinks, // [n_head_q]
                  float   kq_scale,
                    int   il) const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // logits
    //

    void build_logits() const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    // draft-only vocabulary shortlist: logits over the shortlisted rows of head_w only, [n_sel, n_outputs].
    // Returns nullptr when the shortlist does not apply to this graph (caller uses the full head).
    ggml_tensor * build_draft_vocab_logits(
            ggml_tensor * head_w,
            ggml_tensor * head_s,
            ggml_tensor * cur) const;

    // [#69/BL7c] the shortlist for one MTP chain step, [n_sel, 1] over the rows of draft_vocab_ids. The chain
    // emits token ids itself, so no backend sampler has to consume the compact row. nullptr: use the full head.
    ggml_tensor * build_draft_vocab_logits_chain(
            ggml_tensor * head_w,
            ggml_tensor * head_s,
            ggml_tensor * cur) const;
    // write enabled layer inputs into layer_inp_dev at their batch rows
    void build_layer_inp_dev() const;

    // write the unmasked h_nextn rows into h_nextn_dev at their batch rows
    void build_h_nextn_dev() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
