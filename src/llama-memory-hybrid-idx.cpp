#include "llama-memory-hybrid-idx.h"

#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

void llama_pfdiag_kv_begin(uint32_t n);
void llama_pfdiag_kv_mark(int which);
void llama_pfdiag_kv_report(const char * tag, size_t n_ub);

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   gdn_replay_req,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
      llama_sj_kvarn_config   sj_kvarn) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, gdn_replay_req, offload, unified,
        filter_attn, filter_recr,
        0, nullptr, 0, sj_kvarn),
    compact_pool([&] {
        const char * value = std::getenv("LLAMA_QSA_POOL_COMPACT");
        const bool rotating = type_k == GGML_TYPE_TURBO2_0 || type_k == GGML_TYPE_TURBO3_0 ||
                type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TQ5_0 || type_k == GGML_TYPE_TQ6_0;
        return value && std::atoi(value) != 0 && filter_idx && n_seq_max == 1 &&
                model.arch == LLM_ARCH_QWEN4EXP && !filter_idx(model.hparams.n_layer()) &&
                model.hparams.indexer_kpool > 1 && model.hparams.indexer_kpool_row == 2 &&
                (sj_kvarn.enabled() || rotating || type_k == GGML_TYPE_F16);
    }()),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 && !compact_pool ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        // the indexer gathers its raw keys with GET_ROWS. The WHT-rotating KV types (turbo*, tq5_0, tq6_0, and the
        // SJ-KVaRN staging type) rotate on write and have no GET_ROWS kernel on any backend, so their indexer keys stay
        // f16. With SJ-KVaRN the attention cache keeps one stream per sequence, and the indexer must use the same
        // streams, since cell j is the same token in both.
        const bool rotating_k = type_k == GGML_TYPE_TURBO2_0 || type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 ||
                                type_k == GGML_TYPE_TQ5_0    || type_k == GGML_TYPE_TQ6_0;
        const ggml_type type_k_idx = sj_kvarn.enabled() || rotating_k ? GGML_TYPE_F16 : type_k;
        const ggml_type type_v_idx = sj_kvarn.enabled() || rotating_k ? GGML_TYPE_F16 : type_v; // no V rows are cached
        const bool      unified_idx = unified && !sj_kvarn.enabled();

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, type = %s\n", __func__, kv_size, ggml_type_name(type_k_idx));

        return new llama_kv_cache(
            model, hparams_idx, type_k_idx, type_v_idx, v_trans, offload, unified_idx,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    if (compact_pool) {
        pool_storage.resize(model.hparams.n_layer_all, nullptr);
        for (uint32_t il : mem_idx->get_layer_ids()) {
            auto * raw = mem_idx->get_k_storage(il);
            GGML_ASSERT(raw->type == GGML_TYPE_F16 && raw->ne[2] == 1);
            ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
            if (!ctx) {
                throw std::runtime_error("failed to create compact indexer pool context");
            }
            auto * pooled = ggml_new_tensor_2d(ctx.get(), raw->type, raw->ne[0], raw->ne[1]/get_kpool() + 1);
            ggml_format_name(pooled, "idx_pool_compact_%u", il);
            const auto buft = ggml_backend_buffer_get_type(raw->buffer);
            ggml_backend_buffer_ptr buffer(model.hparams.no_alloc ? ggml_backend_buft_alloc_buffer(buft, 0) :
                    ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buffer) {
                throw std::runtime_error("failed to allocate compact indexer pool buffer");
            }
            if (model.hparams.no_alloc) {
                pooled->buffer = buffer.get();
            } else {
                ggml_backend_buffer_clear(buffer.get(), 0);
            }
            LLAMA_LOG_INFO("%s: layer %u compact indexer pool: %lld rows, %.3f MiB (raw rows retained)\n",
                    __func__, il, (long long) pooled->ne[1], ggml_backend_buffer_get_size(buffer.get())/1048576.0);
            pool_storage[il] = pooled;
            pool_buffers.emplace_back(std::move(ctx), std::move(buffer));
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    llama_pfdiag_kv_begin(balloc.get_n_tokens());
    do {
        balloc.split_reset();
        balloc.set_split_pos_grid(get_mem_attn()->get_sj_kvarn_split_grid()); // [#197] full grid mode only

        // [TAG_GDN_REPLAY_SPLIT]
        auto rsplit = get_mem_recr()->make_replay_split(balloc);

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0, rsplit.fn());
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            rsplit.advance(ubatch);

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        llama_pfdiag_kv_mark(0);
        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        llama_pfdiag_kv_mark(1);
        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        const size_t pf_n_ub = ubatches.size();
        auto pf_res = std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
        llama_pfdiag_kv_mark(5);
        llama_pfdiag_kv_report("hybrid_idx", pf_n_ub);
        return pf_res;
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full_single_seq() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, true);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        if (data) {
            for (auto & entry : pool_buffers) {
                ggml_backend_buffer_clear(entry.second.get(), 0);
            }
        }
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    // the attention cache can refuse too (SJ-KVaRN: a range inside the sealed records), so the indexer follows it
    if (!get_mem_attn()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        // fork (bookkeep): a no-op removal leaves all pooled keys valid.
        static const bool bookkeep_kv = [] {
            const char * e = getenv("LLAMA_BOOKKEEP_KV");
            return !(e && strcmp(e, "0") == 0);  // default on (owner 2026-10-22); 0 turns it off
        }();
        const auto * positions = bookkeep_kv && seq_id >= 0 ? &mem_idx->get_cells(seq_id).seq_pos_get(seq_id) : nullptr;
        const size_t size_before = positions ? positions->size() : 0;
        mem_idx->seq_rm(seq_id, p0, p1);
        if (!positions || positions->size() != size_before) {
            mem_idx_stale_set(seq_id, stale);
        }
    }

    return true;
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & entry : pool_buffers) {
        const auto buft = ggml_backend_buffer_get_type(entry.second.get());
        mb[buft] += hparams_idx.no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(entry.first.get(), buft) :
                ggml_backend_buffer_get_size(entry.second.get());
    }
    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);
        state_read_sj_exact(io, seq_id, flags); // [#240] PARTIAL_ONLY only, so never together with the indexer section

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                mem_idx->state_mirror_cell_ext(*get_mem_attn(), sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

ggml_tensor * llama_memory_hybrid_idx::get_pool_storage(int32_t il) const {
    return compact_pool ? pool_storage.at(il) : nullptr;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // fork: cache pool inputs with the layout, including B1 snapshots.
        std::vector<int32_t> pool_cells;
        std::vector<int32_t> pool_idxs;
        std::vector<llama_pos> pool_end;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
    std::vector<uint32_t> rep_count;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;
    std::vector<uint32_t> new_pools;
    bool sparse_marks = false; // fork: incremental states need only the changed pool indices, including B1 snapshots

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// fork: keep the original path available for exact token comparisons.
bool kpool_incremental() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MTP_POOL_INCREMENTAL");
        return value == nullptr || std::atoi(value) != 0;  // default on (owner 2026-10-22); 0 turns it off
    }();
    return enabled;
}

bool kpool_validate() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MTP_POOL_VALIDATE");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

void kpool_scan(llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t kpool, bool by_order) {
    size_t j = sq.j_next;
    if (by_order) {
        for (; j + kpool <= sq.cells.size(); j += kpool) {
            sq.pools.push_back((uint32_t) j);
        }
    } else {
        while (j + kpool <= sq.cells.size()) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
            } else {
                ++j;
            }
        }
    }
    sq.j_next = j;
}

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;
    const bool incremental = kpool_incremental();

    if (incremental && lay.rep_count.empty()) {
        lay.rep_count.resize((size_t) mem_idx->get_size()*n_stream_kv, 0);
    }

    auto truncate = [&](kpool_layout::seq & sq, size_t count) {
        if (incremental) {
            for (size_t i = count; i < sq.pool_cells.size(); ++i) {
                auto & refs = lay.rep_count[sq.pool_cells[i]];
                GGML_ASSERT(refs > 0);
                --refs;
            }
            sq.pool_cells.resize(count);
            sq.pool_idxs.resize(count*kpool);
            sq.pool_end.resize(count);
        }
        sq.pools.resize(count);
    };

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            truncate(sq, 0);
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        const uint32_t strm = unified ? 0 : mem_idx->get_stream(s);
        const bool same_origin = !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first && sq.strm == strm;

        bool suffix_rebuilt = false;
        if (incremental && same_origin && mem_idx_stale[s] != POS_CLEAN && mem_idx_stale[s] > sq.pos_min) {
            const llama_pos stale = mem_idx_stale[s];
            const size_t prefix = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(stale, 0u)) - sq.cells.begin();
            const size_t keep = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale,
                    [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; }) - sq.pools.begin();
            truncate(sq, keep);

            // Only starts overlapping the edited suffix can change their pool membership.
            const size_t restart = prefix > kpool - 1 ? prefix - (kpool - 1) : 0;
            sq.j_next = std::max(keep == 0 ? size_t(0) : (size_t) sq.pools.back() + kpool,
                    std::min(sq.j_next, restart));
            sq.cells.resize(prefix);
            for (auto it = sp.lower_bound(std::make_pair(stale, 0u)); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
            suffix_rebuilt = true;
        }

        sq.strm = strm;

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || (!suffix_rebuilt && mem_idx_stale[s] != POS_CLEAN) ||
                (incremental && !same_origin)) {
            sq.cells.assign(sp.begin(), sp.end());
            truncate(sq, 0);
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        kpool_scan(sq, kpool, hparams_idx.indexer_kpool_by_order);

        if (incremental) {
            for (size_t i = sq.pool_cells.size(); i < sq.pools.size(); ++i) {
                const uint32_t start = sq.pools[i];
                const int32_t rep = (int32_t) ((size_t) sq.strm*mem_idx->get_size() + sq.cells[start + kpool - 1].second);
                sq.pool_cells.push_back(rep);
                ++lay.rep_count[rep];
                sq.pool_end.push_back(sq.cells[start + kpool - 1].first);
                for (uint32_t k = 0; k < kpool; ++k) {
                    sq.pool_idxs.push_back((int32_t) sq.cells[start + k].second);
                }
            }
        }

        lay.n_pool_real += (uint32_t) sq.pools.size();

        if (incremental && kpool_validate()) {
            kpool_layout::seq reference;
            reference.pos_min = sp.empty() ? 0 : sp.begin()->first;
            reference.cells.assign(sp.begin(), sp.end());
            kpool_scan(reference, kpool, hparams_idx.indexer_kpool_by_order);
            if (reference.cells != sq.cells || reference.pools != sq.pools ||
                    reference.pos_min != sq.pos_min || reference.j_next != sq.j_next) {
                GGML_ABORT("incremental pool layout mismatch: sequence %d stale position %d", s, mem_idx_stale[s]);
            }
        }
    }

    if (incremental && kpool_validate()) {
        std::vector<uint32_t> reference(lay.rep_count.size(), 0);
        for (const auto & sq : lay.seqs) {
            for (uint32_t j : sq.pools) {
                ++reference[(size_t) sq.strm*mem_idx->get_size() + sq.cells[j + kpool - 1].second];
            }
        }
        GGML_ASSERT(reference == lay.rep_count && "incremental pool representative counts differ");
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem, bool single_seq) :
    llama_memory_hybrid_context(mem, single_seq),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ single_seq ? 1u : mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), single_seq)) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::b1_supported() const {
    return llama_memory_hybrid_context::b1_supported() && (!ctx_idx || ctx_idx->b1_supported());
}

void llama_memory_hybrid_idx_context::b1_save() {
    llama_memory_hybrid_context::b1_save();
    if (ctx_idx) {
        ctx_idx->b1_save();
    }
    if (b1_st.size() <= i_cur) {
        b1_st.resize(i_cur + 1);
        b1_lay.resize(i_cur + 1);
    }
    if (kpool_track() && kpool_st && i_kpool == i_cur) {
        b1_st[i_cur]  = std::make_shared<kpool_state>(*kpool_st);
        b1_lay[i_cur] = std::make_shared<llama_memory_hybrid_idx::kpool_layout>(mem->kpool_layout_get());
    }
}

bool llama_memory_hybrid_idx_context::b1_seek(size_t i) {
    if (!llama_memory_hybrid_context::b1_seek(i) || (ctx_idx && !ctx_idx->b1_seek(i))) {
        return false;
    }
    i_cur = i;
    if (kpool_track()) {
        if (i >= b1_st.size() || !b1_st[i] || !b1_lay[i]) {
            return false;
        }
        b1_st_cur  = b1_st[i].get();
        b1_lay_cur = b1_lay[i].get();
        i_kpool    = i;
    }
    return true;
}

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx_context::kpool_lay_cur() const {
    return b1_lay_cur ? *b1_lay_cur : mem->kpool_layout_get();
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd, ggml_tensor * compact) : ctx(ctx) {
    if (compact) {
        GGML_ASSERT(k->ne[0] == n_embd && k->ne[2] == 1);
        key_gate = ggml_view_2d(ctx, k, n_embd, k->ne[1], k->nb[1], 0);
        pooled = compact;
        return;
    }
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd, mem->get_pool_storage(il));
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;
    st.sparse_marks = kpool_incremental();

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    st.new_pools.clear();
    st.sparse_marks = kpool_incremental();
    if (!st.sparse_marks) {
        if (++st.generation == 0) {
            std::fill(st.is_new.begin(),  st.is_new.end(),  0);
            std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
            st.generation = 1;
        }
        st.is_new.resize(lay.n_pool_real, 0);
        st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);
    }

    // fork: ordinary decode changes zero or one pool. Keep these representatives on the stack, with a hash set
    // only for larger batches so prefill never searches a growing vector once per token.
    std::array<size_t, 16> small_reps;
    size_t n_small_reps = 0;
    std::unordered_set<size_t> large_reps;
    auto insert_rep = [&](size_t rep) {
        if (!large_reps.empty()) {
            return large_reps.insert(rep).second;
        }
        if (std::find(small_reps.begin(), small_reps.begin() + n_small_reps, rep) != small_reps.begin() + n_small_reps) {
            return false;
        }
        if (n_small_reps < small_reps.size()) {
            small_reps[n_small_reps++] = rep;
            return true;
        }
        large_reps.reserve(2*small_reps.size());
        large_reps.insert(small_reps.begin(), small_reps.end());
        return large_reps.insert(rep).second;
    };

    // The generic tensor reference also consumes new_pools for sparse states, so independently check the marking
    // decisions against the original dense representative/pool maps when validation is requested.
    const bool validate_marks = st.sparse_marks && kpool_validate();
    std::vector<uint32_t> reference_reps;
    std::vector<uint32_t> reference_new;
    if (validate_marks) {
        reference_reps.resize((size_t) kv_size*idx->get_n_stream(), 0);
        reference_new.resize(lay.n_pool_real, 0);
    }

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        bool fresh;
        if (st.sparse_marks) {
            fresh = insert_rep(rep);
            if (validate_marks) {
                const bool expected = reference_reps[rep] == 0;
                if (expected) {
                    reference_reps[rep] = 1;
                    reference_new[pool_start[s] + k] = 1;
                }
                if (fresh != expected) {
                    GGML_ABORT("incremental pool mark mismatch: sequence %d pool %zu representative %zu", s, k, rep);
                }
            }
            if (fresh) {
                st.new_pools.push_back(pool_start[s] + k);
            }
        } else {
            fresh = st.rep_gen[rep] != st.generation;
            if (fresh) {
                st.rep_gen[rep] = st.generation;
                st.is_new[pool_start[s] + k] = st.generation;
            }
        }
        if (fresh) {
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.n_pool_real && (st.sparse_marks || ip == st.is_new.size()));

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
    if (mem->has_compact_pool()) {
        // Keep one dead row beyond the dense pools; never scatter padding over a live pool.
        st.n_new_g = std::min(st.n_new_g, st.n_new + n_pool_max - st.n_pool_real + 1);
    }
    std::sort(st.new_pools.begin(), st.new_pools.end());
    if (validate_marks) {
        std::vector<uint32_t> expected;
        for (uint32_t i = 0; i < reference_new.size(); ++i) {
            if (reference_new[i] != 0) {
                expected.push_back(i);
            }
        }
        if (st.new_pools != expected || st.n_new != expected.size()) {
            GGML_ABORT("incremental pool marks differ from dense reference: %u pools, %zu expected", st.n_new, expected.size());
        }
    }
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return b1_st_cur ? *b1_st_cur : *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

bool llama_memory_hybrid_idx_context::use_kpool_store_only() const {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MTP_POOL_STORE_ONLY");
        return value == nullptr || std::atoi(value) != 0;  // default on (owner 2026-10-22); 0 turns it off
    }();
    return enabled && kpool_incremental();
}

// fork: compact slots follow sequence pool order, independent of physical raw-cell placement.
void llama_memory_hybrid_idx_context::compact_pool_inputs(ggml_tensor * pool_cells, ggml_tensor * new_pool_rep) const {
    if (!mem->has_compact_pool()) {
        return;
    }
    const auto & st = kpool_cur();
    const uint32_t capacity = mem->get_mem_idx()->get_size()/mem->get_kpool() + 1;
    GGML_ASSERT(st.n_pool_real < capacity && st.n_new_g - st.n_new <= capacity - st.n_pool_real);
    if (pool_cells) {
        auto * cells = (int32_t *) pool_cells->data;
        for (uint32_t i = 0; i < pool_cells->ne[0]; ++i) {
            cells[i] = std::min(i, st.n_pool_real);
        }
    }
    auto * reps = (int64_t *) new_pool_rep->data;
    uint32_t out = 0;
    if (st.sparse_marks) {
        for (uint32_t pi : st.new_pools) {
            reps[out++] = pi;
        }
    } else {
        for (uint32_t pi = 0; pi < st.n_pool_real; ++pi) {
            if (st.is_new[pi] == st.generation) {
                reps[out++] = pi;
            }
        }
    }
    GGML_ASSERT(out == st.n_new);
    for (; out < st.n_new_g; ++out) {
        reps[out] = st.n_pool_real + out - st.n_new;
    }
}

// fork: a KV-only helper graph reads no historic pool maps, visibility masks or tails. Fill only the new pool
// writes from the cached layout, with exactly the same global members, representatives, M-RoPE rows and padding.
void llama_memory_hybrid_idx_context::set_input_kpool_store(ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(use_kpool_store_only());
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    const auto & st = kpool_cur();
    const auto & lay = kpool_lay_cur();
    const uint32_t kpool = mem->get_kpool();
    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const int64_t n_cells = (int64_t) kv_size*mem->get_mem_idx()->get_n_stream();
    const uint32_t n_new_g = st.n_new_g;
    GGML_ASSERT(st.n_new <= n_new_g);
    GGML_ASSERT(new_pool_idxs && ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->type == GGML_TYPE_I32 && ggml_is_contiguous(new_pool_idxs));
    GGML_ASSERT(new_pool_idxs->ne[0] == kpool && new_pool_idxs->ne[1] == n_new_g);
    GGML_ASSERT(new_pool_rep && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->type == GGML_TYPE_I64 && ggml_is_contiguous(new_pool_rep));
    GGML_ASSERT(new_pool_rep->ne[0] == n_new_g);
    GGML_ASSERT(new_pool_pos && ggml_backend_buffer_is_host(new_pool_pos->buffer));
    GGML_ASSERT(new_pool_pos->type == GGML_TYPE_I32 && ggml_is_contiguous(new_pool_pos));
    GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    GGML_ASSERT(st.new_pools.size() == st.n_new && lay.rep_count.size() == (size_t) n_cells);

    const bool validate = kpool_validate();
    ggml_context_ptr reference_ctx;
    ggml_backend_buffer_ptr reference_buffer;
    std::array<ggml_tensor *, 3> expected = {};
    const std::array<ggml_tensor *, 3> actual = { new_pool_idxs, new_pool_rep, new_pool_pos };
    if (validate) {
        // Validation deliberately allocates the original full inputs and calls the unmodified reference fill.
        // Its buffers belong only to this CPU comparison and can never alias the graph's live store inputs.
        reference_ctx.reset(ggml_init({ 7*ggml_tensor_overhead(), nullptr, true }));
        GGML_ASSERT(reference_ctx != nullptr);
        const uint32_t n_pool = get_n_kpool();
        auto * cells = ggml_new_tensor_1d(reference_ctx.get(), GGML_TYPE_I32, n_pool);
        auto * idxs = ggml_new_tensor_2d(reference_ctx.get(), GGML_TYPE_I32, kpool, n_pool);
        auto * mask = ggml_new_tensor_2d(reference_ctx.get(), GGML_TYPE_I32, 2, ubatch->n_tokens);
        auto * tail = ggml_new_tensor_2d(reference_ctx.get(), GGML_TYPE_I32, kpool - 1, ubatch->n_tokens);
        auto * ref_idx = ggml_dup_tensor(reference_ctx.get(), new_pool_idxs);
        auto * ref_rep = ggml_dup_tensor(reference_ctx.get(), new_pool_rep);
        auto * ref_pos = ggml_dup_tensor(reference_ctx.get(), new_pool_pos);
        reference_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(reference_ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(reference_buffer != nullptr);
        set_input_kpool_impl(cells, idxs, mask, tail, nullptr, ref_idx, ref_rep, ubatch, ref_pos, false);
        expected = { ref_idx, ref_rep, ref_pos };
        // Every destination byte starts different from the independent reference, so omitted writes cannot pass.
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto * want = (const uint8_t *) expected[i]->data;
            auto * got = (uint8_t *) actual[i]->data;
            const size_t size = ggml_nbytes(actual[i]);
            for (size_t j = 0; j < size; ++j) {
                got[j] = (uint8_t) ~want[j];
            }
        }
    }

    auto * nidx = (int32_t *) new_pool_idxs->data;
    auto * nrep = (int64_t *) new_pool_rep->data;
    auto * npos = (int32_t *) new_pool_pos->data;
    // All four position planes include zeroed stable-graph padding.
    std::fill(npos, npos + 4*n_new_g, 0);
    uint32_t start = 0;
    uint32_t i_new = 0;
    // This scans the fixed sequence table and only the pools changed by this ubatch, never the historic pools.
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        GGML_ASSERT(sq.pool_cells.size() == sq.pools.size() && sq.pool_idxs.size() == sq.pools.size()*kpool);
        const uint32_t end = start + (uint32_t) sq.pools.size();
        while (i_new < st.new_pools.size() && st.new_pools[i_new] < end) {
            GGML_ASSERT(st.new_pools[i_new] >= start);
            const uint32_t pi = st.new_pools[i_new] - start;
            const uint32_t j = sq.pools[pi];
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i_new*kpool + k] = (int32_t) ((int64_t) sq.strm*kv_size + sq.pool_idxs[(size_t) pi*kpool + k]);
            }
            nrep[i_new] = sq.pool_cells[pi];
            const uint32_t c = sq.cells[j].second;
            const auto & ext = mem->get_mem_idx()->get_cells(s).ext_get(c);
            npos[0*n_new_g + i_new] = sq.cells[j].first;
            npos[1*n_new_g + i_new] = ext.y;
            npos[2*n_new_g + i_new] = ext.x;
            npos[3*n_new_g + i_new] = sq.cells[j].first;
            ++i_new;
        }
        start = end;
    }
    GGML_ASSERT(start == st.n_pool_real && i_new == st.n_new);

    // Keep the generic fill's smallest unused destinations, in ascending order. Representatives shared by several
    // sequences remain protected by rep_count; every padded entry has a distinct scatter destination.
    int64_t pad_cell = 0;
    for (; i_new < n_new_g; ++i_new, ++pad_cell) {
        while (pad_cell < n_cells && lay.rep_count[pad_cell] != 0) {
            ++pad_cell;
        }
        GGML_ASSERT(pad_cell < n_cells);
        for (uint32_t k = 0; k < kpool; ++k) {
            nidx[(size_t) i_new*kpool + k] = (int32_t) pad_cell;
        }
        nrep[i_new] = pad_cell;
    }

    compact_pool_inputs(nullptr, new_pool_rep);

    if (validate) {
        const char * names[] = { "new_pool_idxs", "new_pool_rep", "new_pool_pos" };
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto * want = (const uint8_t *) expected[i]->data;
            const auto * got = (const uint8_t *) actual[i]->data;
            const size_t size = ggml_nbytes(actual[i]);
            if (std::memcmp(want, got, size) != 0) {
                size_t offset = 0;
                while (offset < size && want[offset] == got[offset]) {
                    ++offset;
                }
                GGML_ABORT("store-only pool input mismatch: %s byte %zu sequence %d position %d (expected %u, got %u)",
                        names[i], offset, ubatch->seq_id[0][0], ubatch->pos[0], (unsigned) want[offset], (unsigned) got[offset]);
            }
        }
        // A regression counts this marker per real fill, so a graph silently falling back cannot pass.
        LLAMA_LOG_DEBUG("%s: MTP pool store-only inputs validated (%u real, %u slots)\n", __func__, st.n_new, n_new_g);
    }
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    const bool incremental = kpool_incremental();
    if (incremental && kpool_validate()) {
        std::array<ggml_tensor *, 8> tensors = { pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, new_pool_pos };
        const char * names[] = { "pool_cells", "pool_idxs", "pool_mask", "tail_idxs", "sel_mask", "new_pool_idxs", "new_pool_rep", "new_pool_pos" };
        std::array<std::vector<uint8_t>, 8> reference;
        set_input_kpool_impl(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch, new_pool_pos, false);
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i] != nullptr) {
                const size_t size = ggml_nbytes(tensors[i]);
                const auto * data = (const uint8_t *) tensors[i]->data;
                if (size > 0) {
                    reference[i].assign(data, data + size);
                }
            }
        }
        // Poison from the captured reference only after all tensors have been saved, before the optimized fill.
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!reference[i].empty()) {
                auto * data = (uint8_t *) tensors[i]->data;
                for (size_t j = 0; j < reference[i].size(); ++j) {
                    data[j] = (uint8_t) ~reference[i][j];
                }
            }
        }
        set_input_kpool_impl(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch, new_pool_pos, true);
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!reference[i].empty() && std::memcmp(reference[i].data(), tensors[i]->data, reference[i].size()) != 0) {
                const auto * data = (const uint8_t *) tensors[i]->data;
                size_t offset = 0;
                while (reference[i][offset] == data[offset]) {
                    ++offset;
                }
                const size_t token = i >= 2 && i <= 4 ? offset/(reference[i].size()/ubatch->n_tokens) : 0;
                GGML_ABORT("incremental pool input mismatch: %s byte %zu sequence %d position %d (expected %u, got %u)",
                        names[i], offset, ubatch->seq_id[token][0], ubatch->pos[token], (unsigned) reference[i][offset], (unsigned) data[offset]);
            }
        }
        return;
    }
    set_input_kpool_impl(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch, new_pool_pos, incremental);
}

void llama_memory_hybrid_idx_context::set_input_kpool_impl(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos, bool incremental) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = kpool_lay_cur();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.sparse_marks ? st.new_pools.size() == st.n_new : st.is_new.size() == st.n_pool_real);
    // an I32 pool_mask is a range mask: [lo, hi) of the visible pools per token
    GGML_ASSERT(pool_mask->ne[0] == (pool_mask->type == GGML_TYPE_I32 ? 2 : (int64_t) n_pool) && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    if (!incremental) {
        pool_end.reserve(n_pool);
    }

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        // All four position planes include zeroed stable-graph padding.
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    auto emit_new = [&](llama_seq_id s, size_t pi) {
        const auto & sq = lay.seqs[s];
        const uint32_t j = sq.pools[pi];
        GGML_ASSERT(i_new < n_new);
        for (uint32_t k = 0; k < kpool; ++k) {
            nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
        }
        nrep[i_new] = gcell(sq, sq.cells[j + kpool - 1].second);
        if (npos != nullptr) {
            // a pooled key is rotated to the M-RoPE position of its first member
            const uint32_t c = sq.cells[j].second;
            const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
            npos[0*n_new_g + i_new] = sq.cells[j].first;
            npos[1*n_new_g + i_new] = e.y;
            npos[2*n_new_g + i_new] = e.x;
            npos[3*n_new_g + i_new] = sq.cells[j].first;
        }
        ++i_new;
    };

    uint32_t n_pool_real = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        const uint32_t start = n_pool_real;
        seq_pool_start[s] = start;
        n_pool_real += (uint32_t) sq.pools.size();
        GGML_ASSERT(n_pool_real < n_pool);

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];
        if (incremental) {
            GGML_ASSERT(sq.pool_cells.size() == sq.pools.size() && sq.pool_idxs.size() == sq.pools.size()*kpool);
            if (!sq.pools.empty()) {
                std::memcpy(pcell + start, sq.pool_cells.data(), sq.pool_cells.size()*sizeof(int32_t));
                if (inert) {
                    std::fill(pidx + (size_t) start*kpool, pidx + (size_t) n_pool_real*kpool, sentinel);
                } else {
                    std::memcpy(pidx + (size_t) start*kpool, sq.pool_idxs.data(), sq.pool_idxs.size()*sizeof(int32_t));
                }
            }
            while (i_new < st.new_pools.size() && st.new_pools[i_new] < n_pool_real) {
                GGML_ASSERT(st.new_pools[i_new] >= start);
                emit_new(s, st.new_pools[i_new] - start);
            }
        } else {
            for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
                const uint32_t j  = sq.pools[pi];
                const uint32_t ip = start + (uint32_t) pi;
                pcell[ip] = (int32_t) gcell(sq, sq.cells[j + kpool - 1].second);
                for (uint32_t k = 0; k < kpool; ++k) {
                    pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
                }
                if (st.sparse_marks ? std::binary_search(st.new_pools.begin(), st.new_pools.end(), ip) :
                        st.is_new[ip] == st.generation) {
                    emit_new(s, pi);
                }
                pool_end.push_back(sq.cells[j + kpool - 1].first);
            }
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps;
        if (!incremental) {
            reps.assign(pcell, pcell + n_pool_real);
            std::sort(reps.begin(), reps.end());
        }

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (pad_cell < (int64_t) kv_size*n_stream_kv && (incremental ?
                    lay.rep_count[pad_cell] != 0 : std::binary_search(reps.begin(), reps.end(), pad_cell))) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const auto & ends = incremental ? lay.seqs[s].pool_end : pool_end;
            const auto first = ends.begin() + (incremental ? 0 : p0);
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(first, first + (p1 - p0), p) - first);

            if constexpr (std::is_same_v<T, int32_t>) {
                // the range mask: the visible pools are [p0, p0 + nv)
                data[(size_t) 2*i + 0] = (int32_t) p0;
                data[(size_t) 2*i + 1] = (int32_t) (p0 + nv);
            } else {
                const T keep = llama_cast<T>(0.0f);
                const T drop = llama_cast<T>(-INFINITY);

                T * row = data + (size_t) i*n_pool;
                std::fill(row, row + n_pool, drop);
                std::fill(row + p0, row + p0 + nv, keep);
            }

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_I32) {
        fill_mask((int32_t *) pool_mask->data);
    } else if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
    compact_pool_inputs(pool_cells, new_pool_rep);
}
