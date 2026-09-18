#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
                         bool   gdn_replay_req,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // RB1b: how many rollback snapshots behind the current head actually hold data written for
    // this sequence. rs_idx is only *clamped* to n_rs_seq, which says nothing about whether the
    // slot it selects was ever produced: a fresh sequence, a cleared slot, or a session restored
    // from a state file all have rs_idx == 0 but n_rs_seq slots of stale or foreign data behind
    // them. Without this bound a seq_rm rollback of r steps silently restores a state that never
    // existed -- the read-side twin of the writer bug fixed in 3772c377e.
    // Grows by the number of tokens a ubatch contributes for the seq (capped at n_rs_seq),
    // shrinks by the rollback distance when one is accepted, and is zeroed by every path that
    // invalidates the snapshots: clear(), rm_all, and state_read.
    std::vector<uint32_t> rs_valid;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // DRC phase 2 (opt-in, env-gated by LLAMA_GDN_REPLAY): when true, `s_l` holds only the
    // authoritative state (no (1+n_rs_seq) widening) and `ingr_l` holds a per-seq ring of
    // n_rs_seq ggml_gated_delta_net emit_mode==1 ingredient slots instead. Rollback marks
    // `replay_len[seq]` (steps to replay from the ingredient ring) rather than calling
    // set_rs_idx. Only meaningful for GDN/KDA-style recurrent layers.
    bool gdn_replay = false;

    // per-seq pending replay length (0 = no replay pending); parallel to rs_idx, cleared on
    // full seq_rm/clear so a released slot can't leave a dangling replay for its next occupant.
    std::vector<uint32_t> replay_len;

    // gdn_replay: how many ingredient slots behind s_ckpt_l the ring holds for this seq, i.e.
    // the checkpoint's real span (0..n_rs_seq). The logical state of the sequence is
    //   s_ckpt + ring[0, ckpt_span - replay_len)
    // and s_l equals s_ckpt + ring[0, ckpt_span) unless s_stale says otherwise. A rollback may
    // reach at most ckpt_span - replay_len steps back (and no further than the conv snapshot
    // groups tracked by rs_valid). Maintained by the graph builder through
    // llama_memory_recurrent_context::consume_replay(), copied by seq_cp, zeroed by every path
    // that empties the ring.
    std::vector<uint32_t> ckpt_span;

    // gdn_replay: s_l does not hold the logical state and must be rebuilt from s_ckpt_l and the
    // ring on the next decode. Set by state_read (the blob carries the checkpoint and the ring,
    // not a materialized rolled-back state), cleared by consume_replay().
    std::vector<uint8_t> s_stale;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

    // per layer, only allocated when gdn_replay is true: [n_embd_s_ingredient() * n_rs_seq, mem_size]
    // -- one row per cell holding that cell's n_rs_seq ingredient slots back to back, slot 0 the
    // oldest (chronological). A row per cell (rather than a slot-major plane) lets build_rs gather
    // and relocate a sequence's whole ring with the same s_copy machinery as its state, and lets
    // state_write serialise it as one contiguous row.
    std::vector<ggml_tensor *> ingr_l;

    // per layer, only allocated when gdn_replay is true: [n_embd_s(), mem_size] -- the state as of
    // ckpt_span[seq] tokens before the end of the last decode (the oldest edge of the retained
    // window). Replay must start from THIS, not from s_l: by the time a rollback is discovered, s_l
    // already holds the (now known-wrong) optimistic "everything got accepted" final state, and the
    // delta-net rank-1 update is not stably invertible (Sherman-Morrison exists, but its
    // denominator 1 - beta*|k|^2 sits near zero for normalized k and beta near 1). Replaying
    // ingredients only ever moves state forward, so the checkpoint to replay from must predate the
    // whole uncertain trailing window, not follow it.
    std::vector<ggml_tensor *> s_ckpt_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    // cell_ranges_r selects the r (conv) rows -- a rollback snapshot group when one is pending;
    // cell_ranges_s selects the s rows (no widening under gdn_replay, so its own list).
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_r, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const;

    // gdn_replay only: the checkpoint row, the ingredient ring row and the checkpoint's span per
    // cell. Written after the s rows; a blob without it cannot be restored into a replay context.
    void state_write_replay(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s, const std::vector<uint32_t> & cell_spans) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
    bool state_read_replay(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_ingr_l(int32_t il) const;    // nullptr unless mem->gdn_replay
    ggml_tensor * get_s_ckpt_l(int32_t il) const;  // nullptr unless mem->gdn_replay
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // DRC phase 2: pending replay length for the (single, in the n_seq_max==1 case) sequence
    // in the current ubatch, or 0 if none. Used by can_reuse() and the graph builder.
    uint32_t get_replay_len() const;

    // DRC phase 2: the checkpoint's span for the sequence in the current ubatch (see
    // llama_memory_recurrent::ckpt_span), and whether s_l is stale for it. With several lanes
    // the maximum span / any-stale is taken, same as get_replay_len(); lanes that disagree are
    // not supported (logged once) -- the replay subtree has one shape per graph.
    uint32_t get_ckpt_span() const;
    bool     get_s_stale()   const;
    uint32_t get_n_rs_seq()  const;

    // [TAG_RECURRENT_ROLLBACK_SHIFT] number of older snapshot groups (conv in both modes, the
    // recurrent state in the non-replay mode) the builder must move back by n_seq_tokens for the
    // current ubatch: K - max(n_seq_tokens, pending rollback) when n_seq_tokens < K = n_rs_seq + 1,
    // else 0. The op only rewrites the newest min(n, K) groups, so without the move group g
    // holds the state g tokens behind the PREVIOUS head after a short ubatch, and a rollback of
    // exactly one short batch (the multi-seq test's shape) restores a state that never existed.
    // Never nonzero on the speculative verify path (n = n_draft + 1 = K).
    uint32_t get_snap_shift() const;

    // DRC phase 2: mark the pending replay as consumed and record the span the graph just
    // built leaves behind the checkpoint. Called exactly once per decode, after the graph has
    // been built (every GDN layer reads get_replay_len()/get_ckpt_span() during build). Mirrors
    // s_copy_idx()'s consume-and-clear of rs_idx: without it the first partial rejection latches
    // a rollback that is re-applied on every later decode, so the recurrent state permanently
    // trails the token stream.
    void consume_replay(uint32_t new_span) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
