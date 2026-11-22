#include "llama-graph.h"
#include "ggml-inkrel.h"
#include "llama-lsa.h"
#include "../ggml/src/ggml-lsa-recall.h"
#include "llama-lsa-gather.h"
#include "llama-sampler.h"
#include "llama-ext.h"
#include "llama-mtp-chain-sample.h"
#include "llama-draft-vocab-compact.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <map>
#include <mutex>
#include <typeinfo>
#include <vector>

// MoE down-proj activation path — CUDA is first-class, not an exception.
//
// Default (CUDA / HIP / Vulkan / CPU): pass SwiGLU acts straight into
// MUL_MAT_ID. On CUDA the act path is F32→Q8_1; an L2 rescale chain is pure
// overhead (~6 ops × every MoE layer) with no correctness benefit.
//
// Metal-only exception: Metal MUL_MAT_ID casts acts to f16 and can NaN on
// large SwiGLU outliers. Apply per-column L2 rescale only when a Metal
// backend is actually scheduled.
//
// A/B override: LLAMA_MOE_F16_ACT_GUARD=0|1
static bool llm_sched_needs_moe_f16_act_guard(ggml_backend_sched_t sched) {
    if (const char * env = std::getenv("LLAMA_MOE_F16_ACT_GUARD")) {
        if (env[0] != '\0') {
            return std::atoi(env) != 0;
        }
    }
    // Direct path unless Metal is on the schedule. Null sched → direct (CUDA-first).
    if (!sched) {
        return false;
    }
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (!b) {
            continue;
        }
        const char * name = ggml_backend_name(b);
        if (name && std::strstr(name, "Metal")) {
            return true;
        }
    }
    return false;
}

// llamAmpere: K/V cache types whose FA paths do not implement kv_rows (ggml_flash_attn_ext_rows)
static bool llm_graph_kv_type_is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 ||
           t == GGML_TYPE_TQ5_0    || t == GGML_TYPE_TQ6_0;
}

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        bool allow_packed = false) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    if (allow_packed && cparams.kq_mask_packed && cparams.flash_attn && !cparams.training) {
        // [#175] one-bit packed mask: 32 KV positions per I32 word (see ggml_flash_attn_ext); op_params[0] keeps n_kv
        // for the fill and for the users that need the KV length
        ggml_tensor * res = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, (n_kv + 31)/32, n_tokens/n_stream, 1, n_stream);
        res->op_params[0] = (int32_t) n_kv;
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_kq_mask");
        return res;
    }

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

// INKKV v2: the far/near relative-bias split on an SJ-KVaRN base (llm_graph_context::build_attn_rel): one causal
// sequence on an offloaded KQV, the iSWA SWA cache holds the full-attention window copy (il_full), the relative extent
// equals the SWA window, every row of the ubatch has far cells (pos >= extent, and the base still holds a cell at or
// below pos - extent, so no far row is all -inf), and the SWA copy covers the whole near window of the lowest row (no
// gap after a seq_rm). A reserve graph decides from shapes only (base n_kv > extent) so the split graph is the one
// sized at load; the dense path then only runs below extent, where its mask is small.
static bool inkkv_split_want(const llama_kv_cache_iswa_context * attn_ctx, const llama_ubatch & ubatch, const llama_cparams & cparams,
        uint32_t extent, uint32_t extent_swa, int32_t il_full, bool reserve) {
    if (il_full < 0 || !attn_ctx->get_base()->is_sj_kvarn() || !attn_ctx->get_swa()->has_layer(il_full) ||
            extent == 0 || extent != extent_swa || cparams.n_seq_max != 1 || ubatch.n_seqs_unq > 1 || ubatch.n_tokens == 0 ||
            !cparams.causal_attn || cparams.training || !cparams.offload_kqv) {
        return false;
    }
    if (reserve) {
        return attn_ctx->get_base()->get_n_kv() > extent;
    }
    if (!ubatch.pos || !ubatch.seq_id) {
        return false;
    }
    llama_pos p_min = ubatch.pos[0];
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        p_min = std::min(p_min, ubatch.pos[i]);
    }
    if (p_min < (llama_pos) extent) {
        return false;
    }
    const llama_seq_id seq = ubatch.seq_id[0][0];
    const llama_pos base_min = attn_ctx->get_base()->get_seq_pos_min(seq);
    const llama_pos swa_min  = attn_ctx->get_swa()->get_seq_pos_min(seq);
    return base_min >= 0 && base_min <= p_min - (llama_pos) extent && swa_min >= 0 && swa_min <= p_min - (llama_pos) extent + 1;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    if (kq_mask->type == GGML_TYPE_I32) {
        // [#175] packed mask (the choice is fixed per context)
        res &= (kq_mask->op_params[0] == (int32_t) n_kv);
        res &= (kq_mask->ne[0] == (int64_t) (n_kv + 31)/32);
    } else {
        res &= (kq_mask->ne[0] == n_kv);
    }
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// llamAmpere: the kv_rows length of a mask, 0 when the graph must not use kv_rows. cparams.fused_kv_rows is opt-in
// (LLAMA_KV_ROWS=1, see llama_context); kv_rows is never combined with SJ-KVaRN, kv-stream, a TurboQuant/TQ K/V cache
// or the packed (I32) mask, whose FA paths do not implement it (QSA archs are refused in llama_context)
static uint32_t llm_graph_n_kv_rows(
        const llama_kv_cache_context * mctx,
        const llama_cparams & cparams,
        bool packed_mask) {
    if (!cparams.fused_kv_rows || packed_mask) {
        return 0;
    }
    if (mctx->is_sj_kvarn() || cparams.kv_stream_arena_mib > 0 ||
            llm_graph_kv_type_is_turbo(mctx->type_k()) || llm_graph_kv_type_is_turbo(mctx->type_v())) {
        return 0;
    }
    return mctx->get_n_kv_rows();
}

static bool can_reuse_attn_mask(
        const llm_attn_mask & attn_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const bool     packed    = attn_mask.raw && attn_mask.raw->type == GGML_TYPE_I32; // [#175] fixed per context
    const uint32_t n_kv_rows = llm_graph_n_kv_rows(mctx, cparams, packed);
    const uint32_t n_slices  = llama_kv_cache::get_n_kv_slices(ubatch);
    const auto     n_tokens  = ubatch.n_tokens;

    // the current ubatch uses rows only when rows are enabled and it can split into streams
    const bool use_rows = n_kv_rows > 0 && n_slices > 0;

    // the previous mask and the current ubatch must agree on the layout
    if (use_rows != (attn_mask.rows != nullptr)) {
        return false;
    }

    if (!use_rows) {
        return can_reuse_kq_mask(attn_mask.raw, mctx, ubatch, cparams);
    }

    bool res = true;
    res &= attn_mask.n_kv == mctx->get_n_kv();
    res &= attn_mask.rows->ne[0] == n_kv_rows;
    res &= attn_mask.rows->ne[1] == n_slices;
    res &= attn_mask.raw->ne[0] == n_kv_rows;
    res &= attn_mask.raw->ne[1] == n_tokens/n_slices;
    res &= attn_mask.raw->ne[3] == n_slices;

    return res;
}

// KQ mask over the cell lists of kv_rows, or over the whole cache if kv_rows is not used
static llm_attn_mask build_attn_inp_attn_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        bool allow_packed = false) {
    llm_attn_mask attn_mask;

    attn_mask.n_kv = mctx->get_n_kv();

    const bool     packed    = allow_packed && cparams.kq_mask_packed && cparams.flash_attn && !cparams.training;
    const uint32_t n_kv_rows = llm_graph_n_kv_rows(mctx, cparams, packed);

    if (n_kv_rows == 0) {
        attn_mask.rows = nullptr;
        attn_mask.raw = build_attn_inp_kq_mask(ctx, mctx, ubatch, cparams, allow_packed);
    } else {
        const uint32_t n_slices = llama_kv_cache::get_n_kv_slices(ubatch);

        attn_mask.rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv_rows, n_slices);
        ggml_set_input(attn_mask.rows);
        ggml_set_name(attn_mask.rows, "attn_inp_kv_rows");

        attn_mask.raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_rows, ubatch.n_tokens/n_slices, 1, n_slices);
        ggml_set_input(attn_mask.raw);
        ggml_set_name(attn_mask.raw, "attn_inp_kq_mask");
    }

    return attn_mask;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd_row && embd_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd_rows, ubatch->embd_row, 0, n_tokens*ggml_element_size(embd_rows));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;
    res &= (!params.ubatch.embd_row) || (embd_rows && embd_rows->ne[0] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    ASSERT_EMBD_OR_TOKEN(*ubatch);

    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        // [RTPORT] a pipelined MTP step reads its token row from tok_row; its token ids are not a graph input
        if (!tok_row) {
            ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
        }
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (h_device) {
        // fork: the context supplies the hidden rows by backend copy
        return;
    }

    if (h_rows) {
        GGML_ASSERT(ubatch->data && ubatch->data->embd_state_row.size() == (size_t) n_tokens);
        ggml_backend_tensor_set(h_rows, ubatch->data->embd_state_row.data(), 0, n_tokens*ggml_element_size(h_rows));
        return;
    }

    // fork: our MTP paths still hand the hidden state over as ubatch.embd (ref #23643), upstream uses ubatch.embd_state
    if (!ubatch->embd_state && ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
        return;
    }

    GGML_ASSERT(ubatch->embd_state && "this graph requires a state embedding, see llama_batch_ext_set_embd_state()");
    GGML_ASSERT(n_embd_state == h->ne[0]);

    ggml_backend_tensor_set(h, ubatch->embd_state, 0, n_tokens*n_embd_state*ggml_element_size(h));
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token)      || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)       || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd_state) || (h      && h->ne[1]      == params.ubatch.n_tokens);
    // fork: legacy hidden state in ubatch.embd
    res &= (!params.ubatch.embd)       || (h      && h->ne[1]      == params.ubatch.n_tokens);
    res &= (tok_row != nullptr) == params.mtp_tok_row;

    const bool has_rows = params.ubatch.data && !params.ubatch.data->embd_state_row.empty();
    res &= has_rows == (h_rows != nullptr);
    res &= !has_rows || h_rows->ne[0] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_batch_idxs::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ubatch->data && ubatch->data->batch_idxs.size() == (size_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(idxs->buffer));

    int64_t * data = (int64_t *) idxs->data;
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = ubatch->data->batch_idxs[i];
    }
}

bool llm_graph_input_batch_idxs::can_reuse(const llm_graph_params & params) {
    return idxs->ne[0] == params.ubatch.n_tokens;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    // a layer-range graph (LLAMA_B1) without a positional op leaves pos unallocated
    if (ubatch->pos && pos && pos->buffer) {
        const int64_t n_tokens = ubatch->n_tokens;

        const bool has_embd = ubatch->is_mixed() || ubatch->token == nullptr;
        if (rope_section_order == LLAMA_ROPE_SECTION_ORDER_TYXZ || !has_embd) {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
            return;
        }

        // input is always [t, y, x, z]
        // token entries are expanded by the batch to [p, p, p, 0]
        GGML_ASSERT(n_pos_per_embd == 4);

        // slot index per section, slots are t = 0, y = 1, x = 2, z = 3
        std::array<int64_t, 4> slot_of_section = { 0, 1, 2, 3 };
        switch (rope_section_order) {
            case LLAMA_ROPE_SECTION_ORDER_TYXZ: break;
            case LLAMA_ROPE_SECTION_ORDER_ZYXT: slot_of_section = { 3, 1, 2, 0 }; break;
            default: GGML_ABORT("unsupported rope section order");
        }

        std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = ubatch->is_mixed() ? ubatch->type[i] != 0 : true;
            for (int64_t s = 0; s < 4; ++s) {
                const int64_t slot = is_embd ? slot_of_section[s] : s;
                pos_data[s*n_tokens + i] = ubatch->pos[slot*n_tokens + i];
            }
        }
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        if (attn_temp_pow) {
            // (1 + (pos + offset) / length) ^ power
            for (int i = 0; i < n_tokens; ++i) {
                const double pos = ubatch->pos[i];
                attn_scale_data[i] = (float) std::exp((double) f_attn_temp_scale *
                    std::log1p((pos + f_attn_temp_offset) / (double) n_attn_temp_floor_scale));
            }
        } else {
            for (int i = 0; i < n_tokens; ++i) {
                const float pos = ubatch->pos[i];
                attn_scale_data[i] = std::log(
                    std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
                ) * f_attn_temp_scale + 1.0;
            }
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

bool llm_graph_input_attn_temp::can_reuse(const llm_graph_params & params) {
    // only the power form opts in; the log form keeps no reuse
    return attn_temp_pow && attn_scale != nullptr && attn_scale->ne[2] == params.ubatch.n_tokens;
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    fill_s_copy(mctx);

    mctx->consume_replay(span_new);
}

void llm_graph_input_rs::fill_s_copy(const llama_memory_recurrent_context * m) {
    // a layer-range graph (LLAMA_B1) without a recurrent layer leaves s_copy unallocated. It still consumes the pending
    // rollback groups, as any graph of its ubatch does in the normal order (B1 restores them before each layer range)
    if (!s_copy || !s_copy->buffer) {
        if (s_copy && m->b1_has_views() && !m->get_rs_ring()) {
            for (uint32_t i = 0; i < m->get_n_rs(); ++i) {
                (void) m->s_copy((int) i);
            }
        }
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
    int32_t * data = (int32_t *) s_copy->data;

    if (m->get_rs_ring()) {
        auto host_data = [](ggml_tensor * t) -> int32_t * {
            if (t == nullptr) {
                return nullptr;
            }
            GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));
            return (int32_t *) t->data;
        };
        GGML_ASSERT((rs_wr != nullptr) == (n_written > 0) && rs_wr_conv.size() == n_written);
        // host pointers of the per-slot conv row inputs (sized once per graph build)
        rs_wr_conv_host.resize(n_written);
        for (uint32_t j = 0; j < n_written; ++j) {
            rs_wr_conv_host[j] = host_data(rs_wr_conv[j]);
        }
        m->fill_rs_ring(data, host_data(rs_wr), n_written > 0 ? rs_wr_conv_host.data() : nullptr,
                host_data(rs_old_src), host_data(rs_old_dst));
        return;
    }

    const int64_t n_rs = m->get_n_rs();
    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (int64_t i = 0; i < n_rs; ++i) {
        data[i] = m->s_copy((int) i);
    }

    // [TAG_GDN_REPLAY_LANES] per-lane replay: lane i reads its own (r, S, stale) through lane_seq(i)
    if (replay_w != nullptr && replay_w->buffer && replay_w->ne[1] > 1) {
        fill_replay_lanes(m, data, n_rs);
        set_input_shift(m);
        return;
    }

    if (s_copy_conv != nullptr && s_copy_conv->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy_conv->buffer));
        int32_t *     conv = (int32_t *) s_copy_conv->data;
        const int64_t off  = (int64_t) m->get_replay_len() * m->get_size();
        for (int64_t i = 0; i < n_rs; ++i) {
            conv[i] = (int32_t) (data[i] + off);
        }
    }
    if (replay_w != nullptr && replay_w->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(replay_w->buffer));
        float *        w = (float *) replay_w->data;
        const uint32_t K = m->get_n_rs_seq();
        const uint32_t r = m->get_replay_len();
        const uint32_t S = m->get_ckpt_span();
        GGML_ASSERT(r <= S && S <= K && (int64_t) K == replay_w->ne[0]);
        // need_base: replay the accepted prefix from the checkpoint; else the replay base is the
        // state row itself and every slot is masked (the replay then returns it bit-exact)
        const bool need_base = r > 0 || m->get_s_stale();
        for (uint32_t j = 0; j < K; ++j) {
            w[j] = need_base && j < S - r ? 1.0f : 0.0f;
        }
        if (s_copy_pair != nullptr && s_copy_pair->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(s_copy_pair->buffer));
            int32_t *     pair   = (int32_t *) s_copy_pair->data;
            const int64_t n_main = s_copy_pair_main->ne[0];
            for (int64_t i = 0; i < n_rs; ++i) {
                pair[i] = (int32_t) (data[i] + (need_base && i < n_main ? (int64_t) m->get_size() : 0));
            }
        }
    }

    set_input_shift(m);
}

void llm_graph_input_rs::fill_replay_lanes(const llama_memory_recurrent_context * m, const int32_t * data, int64_t n_rs) {
    GGML_ASSERT(ggml_backend_buffer_is_host(replay_w->buffer));
    // conv / pair are optional, as in the 1-D form
    auto host_data = [](ggml_tensor * t) -> int32_t * {
        if (t == nullptr || t->buffer == nullptr) {
            return nullptr;
        }
        GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));
        return (int32_t *) t->data;
    };
    float *        w    = (float *) replay_w->data;
    int32_t *      conv = host_data(s_copy_conv);
    int32_t *      pair = host_data(s_copy_pair);
    const uint32_t K    = m->get_n_rs_seq();
    const int64_t  n_seqs   = replay_w->ne[1];
    const int64_t  mem_size = m->get_size();
    GGML_ASSERT((int64_t) K == replay_w->ne[0] && n_rs >= n_seqs);

    for (int64_t s = 0; s < n_seqs; ++s) {
        const llama_seq_id seq = m->lane_seq((int) s);
        const uint32_t r = m->replay_len_seq(seq);
        const uint32_t S = m->ckpt_span_seq(seq);
        const bool     z = m->s_stale_seq(seq);
        GGML_ASSERT(r <= S && S <= K);
        const bool need_base = r > 0 || z;
        for (uint32_t j = 0; j < K; ++j) {
            w[s * K + j] = need_base && j < S - r ? 1.0f : 0.0f;
        }
        if (conv) {
            conv[s] = (int32_t) (data[s] + (int64_t) r * mem_size);
        }
        if (pair) {
            pair[s] = (int32_t) (data[s] + (need_base ? mem_size : 0));
        }
    }
    // rows past n_seqs are bystander cells (seqs not in this ubatch, e.g. an idle slot between two that verify):
    // plain move of plane 0 / the state row. Their own pending replay stays in replay_len for their next decode.
    for (int64_t i = n_seqs; i < n_rs; ++i) {
        if (conv) {
            conv[i] = data[i];
        }
        if (pair) {
            pair[i] = data[i];
        }
    }
}

void llm_graph_input_rs::set_input_shift(const llama_memory_recurrent_context * m) {
    // read by the shift-layout snapshot writers (delta-net, Mamba2, and the kimi-k3 / bailingmoe3 / qwen4exp / lfm2
    // conv); a graph with no shift-layout writer never reads it and leaves it without a buffer
    if (!s_copy_shift || !s_copy_shift->buffer) {
        return;
    }
    GGML_ASSERT(s_copy && snap_shift > 0 && ggml_backend_buffer_is_host(s_copy_shift->buffer));
    // s_copy_main[s] (the first n_seqs entries of s_copy) already carries the rollback group offset
    // rs_idx * mem_size + src0, so group j of the shift reads plane plane0 + j + rs_idx -- exactly
    // what the per-group form did with a view at plane0 + j indexed by s_copy_main.
    const int32_t * main = (const int32_t *) s_copy->data;
    int32_t *       data = (int32_t *) s_copy_shift->data;

    const uint32_t mem_size = m->get_size();
    const uint32_t K        = m->get_n_rs_seq() + 1;
    const uint32_t n        = m->get_ubatch().n_seq_tokens;
    GGML_ASSERT(n + snap_shift <= K);

    // [TAG_RECURRENT_ROLLBACK_SHIFT] per lane: snap_shift is sized for the lane with the smallest
    // pending rollback. Lane s, with pending rollback r_s, has only K - max(n, r_s) older groups; its
    // remaining groups are gathered from its own destination rows (n + j) * mem_size + head + s, so the
    // write puts back what is there. Those planes stay stale and uncredited: rs_valid after the ubatch is
    // at most rs_valid - r_s + n <= n + K - 1 - r_s. In replay mode s_copy carries no rollback group
    // (rs_idx stays 0, plane0 = replay_len) and the lanes agree, so every lane moves snap_shift groups,
    // except for joined short rounds (replay_short_join), which take the per-lane count.
    // replay mode reads the pending rollback's plane through the indices (the gather's view starts at plane 0)
    // [TAG_GDN_REPLAY_LANES] per-lane replay: each lane reads its own pending replay's plane
    const bool    lanes  = replay_w != nullptr && replay_w->ne[1] > 1;
    const int64_t n_seqs = s_copy_shift->ne[0] / snap_shift;
    for (int64_t s = 0; s < n_seqs; ++s) {
        const int64_t plane0 = !m->get_gdn_replay() ? 0 :
            (int64_t) (lanes ? m->replay_len_seq(m->lane_seq((int) s)) : m->get_replay_len()) * mem_size;
        uint32_t n_move = snap_shift;
        if (lanes && m->get_gdn_replay() && llama_memory_recurrent::replay_short_join()) {
            // a joined short static round: lanes differ in r, so lane s has K - max(n, r_s) older groups
            // (plane r_s + j <= K - 1); the surplus groups are self-copies, as in the non-replay form below
            const uint32_t r_s = m->replay_len_seq(m->lane_seq((int) s));
            n_move = K - std::max(n, r_s);
            GGML_ASSERT(r_s < K && n_move >= 1 && n_move <= snap_shift);
        } else if (replay_len == 0) {
            const uint32_t r_s = (uint32_t) main[s] / mem_size;
            n_move = K - std::max(n, r_s);
            GGML_ASSERT(n_move <= snap_shift);
        }
        for (uint32_t j = 0; j < snap_shift; ++j) {
            data[j * n_seqs + s] = j < n_move
                ? (int32_t) (plane0 + j * (int64_t) mem_size + main[s])
                : (int32_t) ((int64_t) (n + j) * mem_size + head + s);
        }
    }
}

bool llm_graph_input_rs::replay_reusable(const llama_memory_recurrent_context * m, const llama_ubatch & ubatch) {
    const uint32_t r = m->get_replay_len();
    const uint32_t S = m->get_ckpt_span();
    const bool     z = m->get_s_stale();
    if (replay_w == nullptr) {
        return replay_len == r && ckpt_span == S && s_stale == z;
    }
    // same graph for every pending rollback; set_input() hands the new span back to the memory
    const uint32_t n = ubatch.n_seqs > 0 ? ubatch.n_tokens / ubatch.n_seqs : 0;
    replay_len = r;
    ckpt_span  = S;
    s_stale    = z;
    span_new   = llama_memory_recurrent::replay_span_new(S, r, n, m->get_n_rs_seq());
    return llama_memory_recurrent::replay_static_width(n, m->get_n_rs_seq());
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    // DRC phase 2: a nonzero (or changed) replay length needs a differently-shaped extra
    // reconstruction node in the graph, so it can't be satisfied by reusing existing topology --
    // except in the static verify graph (replay_w), where it is data.
    res &= replay_reusable(mctx, params.ubatch);
    res &= snap_shift == mctx->get_snap_shift();
    res &= n_written  == mctx->get_n_written();
    res &= n_older    == mctx->get_n_older();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = self_attn_mask.raw->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type, bool causal) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (causal && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    // see llama_non_causal_type
    const bool causal     = cparams.causal_attn;
    const bool causal_swa = cparams.causal_attn || hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY;

    GGML_ASSERT(self_attn_mask.raw);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_attn_mask.raw->buffer));
    if (self_attn_mask.raw->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_attn_mask.raw->data, ggml_nelements(self_attn_mask.raw), 0, LLAMA_SWA_TYPE_NONE, causal);
    } else {
        fill_mask((float       *) self_attn_mask.raw->data, ggml_nelements(self_attn_mask.raw), 0, LLAMA_SWA_TYPE_NONE, causal);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_attn_mask_swa.raw);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_attn_mask_swa.raw->buffer));
        if (self_attn_mask_swa.raw->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_attn_mask_swa.raw->data, ggml_nelements(self_attn_mask_swa.raw), hparams.n_swa, hparams.swa_type, causal_swa);
        } else {
            fill_mask((float       *) self_attn_mask_swa.raw->data, ggml_nelements(self_attn_mask_swa.raw), hparams.n_swa, hparams.swa_type, causal_swa);
        }
    }
}

// LLAMA_LSA_MULTI=1: stream chunk s of ns of a multi-stream ubatch (rows are stream-major, T/ns per stream) as a
// one-sequence ubatch for the LSA inputs and builders; pos holds the chunk's first position section
static llama_ubatch lsa_sub_ubatch(const llama_ubatch & ub, int64_t s, int64_t ns, std::vector<llama_pos> & pos) {
    GGML_ASSERT(ns > 0 && s >= 0 && s < ns && ub.n_tokens % ns == 0);
    const uint32_t T = ub.n_tokens/(uint32_t) ns;
    const uint32_t o = (uint32_t) s*T;
    llama_ubatch r = ub;
    pos.assign(T, 0);
    if (ub.pos) {
        std::copy(ub.pos + o, ub.pos + o + T, pos.begin());
    }
    r.n_tokens       = T;
    r.n_seq_tokens   = T;
    r.n_seqs         = 1;
    r.n_seqs_unq     = 1;
    r.n_pos          = 1;
    r.token          = ub.token ? ub.token + o : nullptr;
    r.embd           = nullptr;
    r.embd_state     = nullptr;
    r.embd_row       = nullptr;
    r.pos            = ub.pos ? pos.data() : nullptr;
    r.n_seq_id       = ub.n_seq_id ? ub.n_seq_id + o : nullptr;
    r.seq_id         = ub.seq_id ? ub.seq_id + o : nullptr;
    r.seq_id_unq     = r.seq_id && r.seq_id[0] ? r.seq_id[0] : ub.seq_id_unq ? ub.seq_id_unq + s : nullptr;
    r.output         = ub.output ? ub.output + o : nullptr;
    r.type           = nullptr;
    r.decision_order = nullptr;
    return r;
}

// LLAMA_LSA_MULTI=1: stream s's [n_kv_s, T/ns] slice of the KQ mask of inp_all, made once per graph (every layer)
static ggml_tensor * lsa_stream_mask(ggml_context * ctx0, const llm_graph_input_attn_kv * inp_all, llm_graph_input_attn_kv * x,
        int64_t s, int64_t ns) {
    ggml_tensor * m_all = inp_all->self_attn_mask.raw;
    GGML_ASSERT(m_all && m_all->type != GGML_TYPE_I32 && m_all->ne[3] == ns);
    GGML_ASSERT(x->lsa_n_kv_s > 0 && x->lsa_n_kv_s % 64 == 0 && x->lsa_n_kv_s <= m_all->ne[0]);
    if (x->lsa_sub_mask == nullptr || x->lsa_sub_mask_src != m_all) {
        x->lsa_sub_mask = ggml_cont(ctx0, ggml_view_4d(ctx0, m_all, x->lsa_n_kv_s, m_all->ne[1], 1, 1,
                m_all->nb[1], m_all->nb[2], m_all->nb[3], s*m_all->nb[3]));
        x->lsa_sub_mask_src = m_all;
    }
    return x->lsa_sub_mask;
}

void llm_graph_input_attn_kv::set_input_lsa(const llama_kv_cache_context * kv, const llama_ubatch * ubatch) {
    if (!lsa_streams.empty()) {
        // LLAMA_LSA_MULTI=1: every stream's input set from its own rows, its caches current
        const int64_t ns = (int64_t) lsa_streams.size();
        std::vector<llama_pos> pos;
        for (auto & x : lsa_streams) {
            const llama_ubatch sub = lsa_sub_ubatch(*ubatch, x->lsa_sub_s, ns, pos);
            x->set_input_lsa(kv, &sub);
        }
        return;
    }
    llama_lsa_stream_scope lsa_scope(lsa_stream); // no-op unless LLAMA_LSA_MULTI=1
    if (lsa_cell_pos && lsa_cell_pos->buffer && lsa_q_pos->buffer) {
        kv->set_input_lsa_pos(lsa_cell_pos, lsa_q_pos, ubatch);
    }
    if (lsa_kq_mask && lsa_kq_mask->buffer) {
        kv->set_input_kq_mask(lsa_kq_mask, ubatch, cparams.causal_attn);
    }
    llama_lsa_set_q_cell(lsa_q_cell, ubatch);
    if (lsa_n_blk > 0) {
        llama_lsa_set_q_blk(lsa_q_blk, lsa_q_cell ? lsa_q_cell : self_k_idxs, lsa_n_blk);
    }
    if (lsa_n_blk > 0 && lsa_u_blk) {
        llama_lsa_set_mean_inputs(lsa_u_blk, lsa_u_rblk, lsa_q_ring, lsa_q_cell ? lsa_q_cell : self_k_idxs, lsa_n_blk, lsa_ring_n,
                lsa_ring_rows, llama_lsa_ubatch_seq(ubatch));
    }
    for (const auto & c : lsa_idx_commit) {
        llama_lsa_idx_commit(c.first, c.second);
    }
    for (const auto & c : lsa_mean_commit) {
        llama_lsa_mean_commit(c.first, c.second);
    }
    for (const auto & c : lsa_pick_commit) {
        llama_lsa_pick_commit(c.k, c.gen, c.reuse, c.pos0, c.edge, c.nbs, c.seq, c.expire);
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    // the idxs are left unallocated when the KV cache is bypassed during training
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        mctx->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);
    }

    if (self_sj_kvarn_desc && self_sj_kvarn_desc->buffer) {
        mctx->set_input_sj_kvarn_desc(self_sj_kvarn_desc, ubatch, 0);
    }

    set_input_lsa(mctx, ubatch);
    if (self_sj_kvarn_desc_edge && self_sj_kvarn_desc_edge->buffer) {
        mctx->set_input_sj_kvarn_desc(self_sj_kvarn_desc_edge, ubatch, 1);
    }
}

// LSA probe: the selection shape (dense or sparse, kept count) is decided at graph build from n_tokens, n_kv and whether
// the layer's index/mean caches are built. Default: rebuild every ubatch. LLAMA_LSA_GRAPH_REUSE=1: reuse when n_kv is the
// same (the caller's mask check), this graph built no cache (nothing to commit, so it is the settled form), no cache
// changed since (a restore or K shift bumps the generation), and neither ubatch starts a sequence (position 0 rebuilds).
// Positions, rows and ring slots are graph inputs, refreshed by set_input on every decode.
static bool lsa_graph_reusable(const llm_graph_input_attn_kv & inp, const llama_ubatch & ubatch, uint32_t n_kv) {
    if (!inp.lsa_streams.empty() || (inp.lsa_cell_pos && llama_lsa_cfg().multi)) {
        return false; // LLAMA_LSA_MULTI=1: per-stream inputs and caches follow the ubatch's sequences, rebuild
    }
    if (inp.lsa_cell_pos == nullptr) {
        return true;
    }
    return llama_lsa_graph_reuse_on() && ubatch.pos != nullptr && ubatch.pos[0] != 0 && inp.lsa_pos0 != 0 &&
            inp.lsa_cell_pos->ne[0] == (int64_t) n_kv && inp.lsa_graph_gen == llama_lsa_graph_gen() &&
            inp.lsa_idx_commit.empty() && inp.lsa_mean_commit.empty() && inp.lsa_pick_commit.empty();
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    if (!skip_unused || (self_attn_mask.raw && self_attn_mask.raw->buffer)) {
        res &= can_reuse_attn_mask(self_attn_mask, mctx, params.ubatch, params.cparams);
    }

    if (self_sj_kvarn_desc && (!skip_unused || self_sj_kvarn_desc->buffer)) {
        res &= sj_kvarn_n_kv_streams == mctx->get_sj_kvarn_n_kv_streams();
    }

    res &= lsa_graph_reusable(*this, params.ubatch, mctx->get_n_kv());

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_attn_mask(self_attn_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_attn_mask(self_attn_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    // leave unallocated when the top-k layers use the indexer mask
    if (self_attn_mask_mla.raw->buffer) {
        mctx->get_mla()->set_input_attn_mask(self_attn_mask_mla, ubatch, cparams.causal_attn);
    }

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_attn_mask(self_attn_mask_lid, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_attn_mask(self_attn_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_attn_mask(self_attn_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    mctx->get_base()->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    mctx->get_swa()->set_input_attn_mask(self_attn_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_attn_mask_swa.raw && self_attn_mask_swa.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}


void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    mctx->get_base()->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    mctx->get_swa()->set_input_attn_mask(self_attn_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_attn_mask_swa.raw && self_attn_mask_swa.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        mctx->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_attn_mask.raw && inp_raw->self_attn_mask.raw->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_attn_mask.raw, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}


void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_attn_mask.raw);

    const int64_t n_enc    = cross_attn_mask.raw->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_attn_mask.raw->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_attn_mask.raw->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_attn_mask.raw->data);
    } else {
        fill_mask((float *) cross_attn_mask.raw->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // a graph can leave these unallocated: qwen4exp's direct QSA selection never reads the causal mask, and a
    // layer-range graph (LLAMA_B1) without an attention layer writes no K/V
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    }
    if (inp_attn->self_v_idxs && inp_attn->self_v_idxs->buffer) {
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    // kv_rows and the mask stay unallocated if all attention layers have their own (e.g. qwen4exp QSA)
    if (inp_attn->self_attn_mask.raw && inp_attn->self_attn_mask.raw->buffer) {
        mctx->get_attn()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_sj_kvarn_desc && inp_attn->self_sj_kvarn_desc->buffer) {
        mctx->get_attn()->set_input_sj_kvarn_desc(inp_attn->self_sj_kvarn_desc, ubatch, 0);
    }
    if (inp_attn->self_sj_kvarn_desc_edge && inp_attn->self_sj_kvarn_desc_edge->buffer) {
        mctx->get_attn()->set_input_sj_kvarn_desc(inp_attn->self_sj_kvarn_desc_edge, ubatch, 1);
    }
    inp_attn->set_input_lsa(mctx->get_attn(), ubatch);

    inp_rs->fill_s_copy(mctx->get_recr());

    mctx->get_recr()->consume_replay(inp_rs->span_new);
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask, mctx->get_attn(), params.ubatch, params.cparams);

    if (inp_attn->self_sj_kvarn_desc) {
        res &= inp_attn->sj_kvarn_n_kv_streams == mctx->get_attn()->get_sj_kvarn_n_kv_streams();
    }

    res &= lsa_graph_reusable(*inp_attn, params.ubatch, mctx->get_attn()->get_n_kv());

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // DRC phase 2: same guard as llm_graph_input_rs::can_reuse -- a changed replay length means a
    // differently-shaped reconstruction subtree, which reused topology cannot express.
    res &= inp_rs->replay_reusable(mctx->get_recr(), params.ubatch);
    res &= inp_rs->snap_shift == mctx->get_recr()->get_snap_shift();
    res &= inp_rs->n_written  == mctx->get_recr()->get_n_written();
    res &= inp_rs->n_older    == mctx->get_recr()->get_n_older();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    // leave the mask unallocated if all attention layers have their own (e.g. glm5-next DSA)
    if (inp_attn->self_attn_mask.raw->buffer) {
        mctx->get_attn()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);
    }

    inp_rs->fill_s_copy(mctx->get_recr());

    mctx->get_recr()->consume_replay(inp_rs->span_new);
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // DRC phase 2: same guard as llm_graph_input_rs::can_reuse -- a changed replay length means a
    // differently-shaped reconstruction subtree, which reused topology cannot express.
    res &= inp_rs->replay_reusable(mctx->get_recr(), params.ubatch);
    res &= inp_rs->snap_shift == mctx->get_recr()->get_snap_shift();
    res &= inp_rs->n_written  == mctx->get_recr()->get_n_written();
    res &= inp_rs->n_older    == mctx->get_recr()->get_n_older();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_attn_mask.raw && inp_attn->self_attn_mask.raw->buffer) {
        attn_ctx->get_base()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_attn_mask_swa.raw && inp_attn->self_attn_mask_swa.raw->buffer) {
        attn_ctx->get_swa()->set_input_attn_mask(inp_attn->self_attn_mask_swa, ubatch, cparams.causal_attn);
    }

    inp_rs->fill_s_copy(mctx->get_recr());

    mctx->get_recr()->consume_replay(inp_rs->span_new);
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // DRC phase 2: same guard as llm_graph_input_rs::can_reuse -- a changed replay length means a
    // differently-shaped reconstruction subtree, which reused topology cannot express.
    res &= inp_rs->replay_reusable(mctx->get_recr(), params.ubatch);
    res &= inp_rs->snap_shift == mctx->get_recr()->get_snap_shift();
    res &= inp_rs->n_written  == mctx->get_recr()->get_n_written();
    res &= inp_rs->n_older    == mctx->get_recr()->get_n_older();

    return res;
}

void llm_graph_input_mem_hybrid_iswa_rel::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_attn_mask.raw && inp_attn->self_attn_mask.raw->buffer) {
        attn_ctx->get_base()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);
    }

    if (self_rel_pos->buffer) {
        attn_ctx->get_base()->set_input_relative_pos(self_rel_pos, ubatch, extent);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_attn_mask_swa.raw && inp_attn->self_attn_mask_swa.raw->buffer) {
        attn_ctx->get_swa()->set_input_attn_mask(inp_attn->self_attn_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_rel_pos_swa->buffer) {
        attn_ctx->get_swa()->set_input_relative_pos(self_rel_pos_swa, ubatch, extent_swa);
    }

    if (self_sj_kvarn_desc && self_sj_kvarn_desc->buffer) {
        attn_ctx->get_base()->set_input_sj_kvarn_desc(self_sj_kvarn_desc, ubatch, 0);
    }

    if (self_sj_far_mask && self_sj_far_mask->buffer) {
        attn_ctx->get_base()->set_input_sj_far_mask(self_sj_far_mask, ubatch, extent);
    }

    if (inkrel.on && inkrel.far_mask->buffer) {
        attn_ctx->get_base()->set_input_inkrel(inkrel, ubatch, extent);
    }

    if (inklsa_on) {
        attn_ctx->get_base()->set_input_inklsa(inklsa, ubatch, extent);
        for (const auto & c : inklsa_commit) {
            llama_inklsa_idx_commit(c.first, c.second);
        }
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa_rel::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // the masks are always dense, see build_inp_mem_hybrid_iswa_rel()

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask.raw, attn_ctx->get_base(), params.ubatch, params.cparams);

    res &= self_rel_pos->ne[0] == attn_ctx->get_base()->get_n_kv();
    res &= self_rel_pos->ne[1] == params.ubatch.n_tokens;

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask_swa.raw, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= self_rel_pos_swa->ne[0] == attn_ctx->get_swa()->get_n_kv();
    res &= self_rel_pos_swa->ne[1] == params.ubatch.n_tokens;

    {
        const int64_t n_kv = attn_ctx->get_base()->get_n_kv();
        const bool want = params.cparams.flash_attn && !params.cparams.training && params.cparams.n_seq_max == 1 &&
                params.ubatch.n_seqs_unq <= 1 && !attn_ctx->get_base()->is_sj_kvarn() &&
                llama_inkrel_want(n_kv, params.ubatch.n_tokens, extent);
        res &= want == inkrel.on;
        if (inkrel.on) {
            res &= inkrel.far_mask->ne[0] == n_kv && inkrel.far_mask->ne[1] == params.ubatch.n_tokens;
        }
    }

    if (self_sj_kvarn_desc) {
        res &= sj_kvarn_n_kv_streams == attn_ctx->get_base()->get_sj_kvarn_n_kv_streams();
        const bool split = inkkv_split_want(attn_ctx, params.ubatch, params.cparams, extent, extent_swa, sj_il_full, false);
        res &= split == (self_sj_far_mask != nullptr);
        if (self_sj_far_mask) {
            res &= self_sj_far_mask->ne[0] == attn_ctx->get_base()->get_n_kv() && self_sj_far_mask->ne[1] == params.ubatch.n_tokens;
        }
    }

    if (inklsa_on) {
        // a rebuild graph is never reused; a graph built on a valid index only while every index is still valid
        res &= !inklsa_rebuild;
        res &= params.ubatch.n_seqs_unq == 1 || !inklsa.sparse;
        for (const ggml_tensor * kc : inklsa_kcaches) {
            llama_inklsa_idx idx;
            res &= llama_inklsa_idx_get(kc, kc->ne[0], llama_inklsa_idx_rows(kc->ne[1], inklsa.B), ggml_backend_buffer_get_type(kc->buffer), &idx) && idx.valid;
        }
    }

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }

    for (auto & [seq_id, inp] : row_inputs) {
        GGML_ASSERT(inp.t_rows && ggml_nelements(inp.t_rows) == (int64_t) inp.rows.size());
        ggml_backend_tensor_set(inp.t_rows, inp.rows.data(), 0, ggml_nbytes(inp.t_rows));
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    size_t output_row = 0;
    for (uint32_t i = 0; i < params.ubatch.n_tokens; ++i) {
        if (params.ubatch.output[i]) {
            if (output_row >= output_seqs.size() || output_seqs[output_row++] != params.ubatch.seq_id[i][0]) {
                return false;
            }
        }
    }
    if (output_row != output_seqs.size()) {
        return false;
    }

    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

void llm_graph_input_mtp_chain_samp::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const size_t n_u = (size_t) ggml_nelements(u);
    GGML_ASSERT(src != nullptr && src->size() >= LLAMA_MTP_CHAIN_SAMP_N + n_u && "llama_set_mtp_chain_sampling: too few uniforms for the chain");

    ggml_backend_tensor_set(samp, src->data(), 0, LLAMA_MTP_CHAIN_SAMP_N * sizeof(float));
    ggml_backend_tensor_set(u, src->data() + LLAMA_MTP_CHAIN_SAMP_N, 0, n_u * sizeof(float));
}

bool llm_graph_input_mtp_chain_samp::can_reuse(const llm_graph_params & params) {
    // the chain length follows from the ubatch shape, which allow_reuse already matched
    return params.mtp_chain_samp == src && params.cparams.mtp_chain_top_k > 0;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;
    t_logits_ids  = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_greedy_rows = nullptr;
    greedy_rows.clear();
    t_sampled_rows.clear();
    sampled_rows.clear();
    t_topk_rows.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();
    t_sampling_all.clear();

    params = {};

    inputs.clear();
    mtp_device_input = false;
    fused_nodes.clear();
    hdmd_inputs.clear();

    const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    if (meta_size != buf_compute_meta_size) {
        // not make_unique: it would zero the whole reserve and make its pages resident
        buf_compute_meta.reset(new uint8_t[meta_size]);
        buf_compute_meta_size = meta_size;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta_size,
        /*.mem_buffer =*/ buf_compute_meta.get(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

// fork INKPF3: LLAMA_INKPF3_TIMING=1 logs the host time of every graph input setter over 2 ms (diagnostic only)
static bool inkpf3_timing_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_INKPF3_TIMING"); return e && atoi(e) != 0; }();
    return on;
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    if (inkpf3_timing_on()) {
        const int64_t t0 = ggml_time_us();
        for (auto & input : inputs) {
            const int64_t t = ggml_time_us();
            input->set_input(ubatch);
            const int64_t dt = ggml_time_us() - t;
            if (dt > 2000) {
                LLAMA_LOG_INFO("INKPF3 set_input %s: %.1f ms (n_tokens %u)\n", typeid(*input).name(), dt/1000.0, ubatch->n_tokens);
            }
        }
        LLAMA_LOG_INFO("INKPF3 set_inputs total: %.1f ms (n_tokens %u)\n", (ggml_time_us() - t0)/1000.0, ubatch->n_tokens);
        return;
    }
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size() && !params.cparams.embeddings_layer_inp_dev; ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    // the chains are read on the host after compute, so keep every one alive
    for (auto * tensor : t_sampling_all) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        return false;
    }

    bool res = true;

    for (auto & input : inputs) {
        res = res && input->can_reuse(params);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

ggml_tensor * llm_graph_result::get_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot) const {
    const auto it = hdmd_inputs.find({ cur, rot });
    return it == hdmd_inputs.end() ? nullptr : it->second;
}

void llm_graph_result::set_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot, ggml_tensor * res) {
    hdmd_inputs[{ cur, rot }] = res;
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    model            (*params.model),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    hadamard_rotations(params.hadamard_rotations),
    hadamard_inverses (params.hadamard_inverses),
    moe_cache        (params.moe_cache),
    prec_policy      (params.prec_policy),
    hdmd             (params.hdmd),
    layer_inp_dev    (params.layer_inp_dev),
    layer_inp_src    (params.layer_inp_src),
    h_nextn_dev      (params.h_nextn_dev),
    h_nextn_src      (params.h_nextn_src),
    samplers         (params.samplers),
    draft_vocab_ids  (params.draft_vocab_ids),
    draft_vocab_compact    (params.draft_vocab_compact),
    draft_vocab_compact_src(params.draft_vocab_compact_src),
    draft_vocab_warned(params.draft_vocab_warned),
    is_reserve       (params.is_reserve),
    mtp_chain_samp   (params.mtp_chain_samp),
    b1_il0           (params.b1_il0),
    b1_il1           (params.b1_il1),
    b1_carry         (params.b1_carry),
    ple_pf           (params.ple_pf),
    mtp_tok_row      (params.mtp_tok_row),
    cb_func          (params.cb),
    res              (params.res),
    model_ref        (params.model),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

// EXL3: apply the 128-block Hadamard (shared f32 [128,128] H128 = Sylvester / sqrt(128), symmetric) along ne0 of t.
// t is [D, ...] with D % 128 == 0: reshape to [128, D/128, rest], mul_mat(H128, .), reshape back.
static ggml_tensor * build_exl3_had128(ggml_context * ctx0, ggml_tensor * had128, ggml_tensor * t) {
    const int64_t D = t->ne[0];
    GGML_ASSERT(D % 128 == 0);
    if (!ggml_is_contiguous(t)) {
        t = ggml_cont(ctx0, t);
    }
    const int64_t rest = ggml_nelements(t) / D;
    ggml_tensor * t3 = ggml_reshape_3d(ctx0, t, 128, D / 128, rest);
    t3 = ggml_mul_mat(ctx0, had128, t3); // [128, D/128, rest]
    return ggml_reshape_4d(ctx0, t3, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ggml_tensor * llm_graph_context::build_hadamard_input(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    if (!hdmd) {
        return cur;
    }
    const auto it = hdmd->rot.find(w);
    if (it == hdmd->rot.end()) {
        return cur;
    }
    const auto & t = it->second;
    if (ggml_tensor * x = res->get_hdmd_input(cur, t.rot)) {
        return x;
    }
    ggml_tensor * x = cur;
    if (t.perm_rep > 1) {
        // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
        x = ggml_is_contiguous(x) ? x : ggml_cont(ctx0, x);
        const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
        x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
    }
    if (t.signs) {
        x = ggml_mul(ctx0, x, t.signs);
    }
    x = llama_mul_mat_hadamard(ctx0, x, t.rot);
    res->set_hdmd_input(cur, t.rot, x);
    return x;
}

static ggml_tensor * build_rope_impl(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * pos, const llama_hparams & hparams, const llama_rope_params & rp) {
    if (rp.n_rot == 0) {
        return cur;
    }

    if (rp.type & GGML_ROPE_TYPE_MROPE) {
        int sections[GGML_MROPE_SECTIONS];
        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.end(), sections);

        return ggml_rope_multi(ctx, cur, pos, rp.factors,
                rp.n_rot, sections, rp.type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
                rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    }

    return ggml_rope_ext(ctx, cur, pos, rp.factors,
            rp.n_rot, rp.type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
}

ggml_tensor * llm_graph_context::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) const {
    return build_rope_impl(ctx0, cur, pos, hparams, model.get_rope_params(cparams, il));
}

ggml_tensor * llm_graph_context::build_rope_indexer(ggml_tensor * cur, ggml_tensor * pos, int il) const {
    const llama_rope_params rp = model.get_rope_params_indexer(cparams, il);

    ggml_tensor * res = build_rope_impl(ctx0, cur, pos, hparams, rp);

    return rp.n_rot > 0 && rp.n_offs > 0 ? ggml_rope_set_offset(res, rp.n_offs) : res;
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * cur_mm = cur;
    const ggml_tensor * transform_weight = w;
    if (res->get_b1_ffn_phase() == 2) {
        while (transform_weight->view_src) {
            transform_weight = transform_weight->view_src;
        }
    }
    if (hadamard_rotations) {
        const auto it = hadamard_rotations->find(transform_weight);
        if (it != hadamard_rotations->end()) {
            const auto & t = it->second;
            // another folded weight on this same activation already built the transform
            const auto memo_key = std::make_tuple((const ggml_tensor *) cur, (const ggml_tensor *) t.rot, (const ggml_tensor *) t.signs, t.perm_hd, t.perm_nk, t.perm_rep);
            const auto memo_it  = hadamard_memo.find(memo_key);
            if (memo_it != hadamard_memo.end()) {
                cur_mm = memo_it->second;
            } else {
                if (t.perm_rep > 1) {
                    // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
                    ggml_tensor * x = ggml_is_contiguous(cur_mm) ? cur_mm : ggml_cont(ctx0, cur_mm);
                    const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
                    x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
                    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
                    cur_mm = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
                }
                if (t.signs) {
                    cur_mm = ggml_mul(ctx0, cur_mm, t.signs);
                }
                cur_mm = llama_mul_mat_hadamard(ctx0, cur_mm, t.rot);
                hadamard_memo[memo_key] = cur_mm;
            }
        }
    }
    if (cur_mm == cur) {
        // upstream prism.hadamard folded weights (llama_hadamard); the fork table above wins when it has the weight
        cur_mm = build_hadamard_input(w, cur);
    }


    ggml_tensor * res = nullptr;

    const llama_model::exl3_side * exl3 = model_ref ? model_ref->exl3_side_of(w) : nullptr;
    if (exl3) {
        // y = svh * H128( W_trellis . H128( suh * x ) )   (exllamav3 exl3.py forward, Hadamard block 128)
        // fused form: the mul_mat node carries suh/svh as src[2]/src[3] and the backend applies the scales and
        // the 128-block Hadamards itself (ggml-cuda exl3.cu, ggml-cpu mul_mat_exl3). LLAMA_EXL3_GLUE_GRAPH=1
        // builds the same math from generic ops (ggml_mul + mul_mat with exl3_had128.weight) for A/B checks.
        static const bool graph_glue = getenv("LLAMA_EXL3_GLUE_GRAPH") != nullptr;
        if (graph_glue) {
            GGML_ASSERT(model_ref->exl3_had128 != nullptr);
            ggml_tensor * x = ggml_mul(ctx0, cur_mm, exl3->suh);
            x   = build_exl3_had128(ctx0, model_ref->exl3_had128, x);
            res = ggml_mul_mat(ctx0, w, x);
            res = build_exl3_had128(ctx0, model_ref->exl3_had128, res);
            res = ggml_mul(ctx0, res, exl3->svh);
        } else {
            GGML_ASSERT(w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0);
            if (!ggml_is_contiguous(cur_mm)) {
                cur_mm = ggml_cont(ctx0, cur_mm);
            }
            res = ggml_mul_mat(ctx0, w, cur_mm);
            res->src[2] = exl3->suh;
            res->src[3] = exl3->svh;
        }
    } else {
        res = ggml_mul_mat(ctx0, w, cur_mm);
    }

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

llm_graph_amax_src1 llm_graph_context::build_amax_src1_f16(
          ggml_tensor * cur) const {
    llm_graph_amax_src1 result = {
        /*.res     =*/ cur,
        /*.scale_s =*/ nullptr,
    };

    const int64_t n_embd = cur->ne[0];

    // amax = max|cur| along the rows for each [expert, token] -> [1, n_expert, n_tokens]
    ggml_tensor * ab = ggml_abs(ctx0, cur);
    ggml_tensor * amax = ggml_pool_1d(ctx0, ab, GGML_OP_POOL_MAX, n_embd, n_embd, 0);

    // ref: https://github.com/ggml-org/llama.cpp/pull/26223
    constexpr float k_log2_e = 1.4426950408889634f; // 1 / ln(2)
    constexpr float k_ln_2   = 0.6931471805599453f;

    ggml_tensor * e;
    // note: this can technically be:
    // e = ggml_scale(ctx0, amax, 1.0f/32768.0f);
    // but we leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks
    e = ggml_scale(ctx0, amax, 1.0f/1024.0f);
    e = ggml_log  (ctx0, e);
    e = ggml_scale(ctx0, e, k_log2_e);
    e = ggml_ceil (ctx0, e);
    e = ggml_clamp(ctx0, e, 0.0f, 127.0f);
    e = ggml_scale(ctx0, e, k_ln_2);
    e = ggml_exp  (ctx0, e);

    result = {
        /*.res     =*/ ggml_div(ctx0, cur, e),
        /*.scale_s =*/ e,
    };

    return result;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s,
          ggml_tensor * slots,
                 bool   cached) const {
    ggml_tensor * cur_mm = cur;
    // A packed B1 bucket may view only the contiguous expert interval it
    // uses. The folded input transform still belongs to the original bank.
    const ggml_tensor * transform_weight = w;
    if (res->get_b1_ffn_phase() == 2) {
        while (transform_weight->view_src) {
            transform_weight = transform_weight->view_src;
        }
    }
    if (hadamard_rotations) {
        const auto it = hadamard_rotations->find(transform_weight);
        if (it != hadamard_rotations->end()) {
            const auto & t = it->second;
            // another folded weight on this same activation already built the transform
            const auto memo_key = std::make_tuple((const ggml_tensor *) cur, (const ggml_tensor *) t.rot, (const ggml_tensor *) t.signs, t.perm_hd, t.perm_nk, t.perm_rep);
            const auto memo_it  = hadamard_memo.find(memo_key);
            if (memo_it != hadamard_memo.end()) {
                cur_mm = memo_it->second;
            } else {
                if (t.perm_rep > 1) {
                    // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
                    ggml_tensor * x = ggml_is_contiguous(cur_mm) ? cur_mm : ggml_cont(ctx0, cur_mm);
                    const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
                    x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
                    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
                    cur_mm = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
                }
                if (t.signs) {
                    cur_mm = ggml_mul(ctx0, cur_mm, t.signs);
                }
                cur_mm = llama_mul_mat_hadamard(ctx0, cur_mm, t.rot);
                hadamard_memo[memo_key] = cur_mm;
            }
        }
    }
    if (cur_mm == cur) {
        // as in build_lora_mm: the folded weight reads the transformed input, the LoRA delta reads `cur`
        cur_mm = build_hadamard_input(w, cur);
    }
    // the experts in the MoE cache are selected by their slots
    ggml_tensor * res = slots == nullptr ?
        ggml_mul_mat_id(ctx0, w, cur_mm, ids) :
        ggml_mul_mat_id(ctx0, cached ? moe_cache->get_experts(w) : w, cur_mm, slots);
    if (moe_route_skip && !ggml_exl3_bits(w->type) && ggml_nelements(moe_route_skip) == res->ne[1]*res->ne[2]) {
        res->src[3] = moe_route_skip; // a backend that ignores it computes those routes (their weight is zero)
    }

    // fork: native CPU experts use logical IDs for both compressed weights and side vectors.
    const char * cpu_exl3 = getenv("GGML_CPU_EXL3");
    if (cpu_exl3 && strcmp(cpu_exl3, "1") == 0 && ggml_exl3_bits(w->type)) {
        const llama_model::exl3_side * side = model_ref ? model_ref->exl3_side_of(w) : nullptr;
        GGML_ASSERT(slots == nullptr && side != nullptr);
        res->src[3] = side->suh;
        res->src[4] = side->svh;
    }

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        // the CUDA broadcast-mul kernel has no path for an F32 activation times an F16 operand
        // (only the reverse, F16 activation times F32/F16 operand); most checkpoints keep norm
        // weights in F32 so this is normally a no-op, but some conversions store them narrower.
        // This re-inserts a cast node into the graph on every build (every token), but norm
        // weight tensors are n_embd-sized (a few KB), so the added cost is not worth caching
        // across builds versus upcasting these specific tensors once at load time.
        if (mw->type != cur->type && cur->type == GGML_TYPE_F32) {
            mw = ggml_cast(ctx0, mw, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        // see mw cast note above
        if (mb->type != cur->type && cur->type == GGML_TYPE_F32) {
            mb = ggml_cast(ctx0, mb, GGML_TYPE_F32);
        }
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
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
                 int   il) const {
    const llm_arch arch = model.arch;

    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    auto is_scaled_low_precision = [](ggml_tensor * w) {
        return w && (w->type == GGML_TYPE_NVFP4 || w->type == GGML_TYPE_F8_E4M3);
    };

    GGML_ASSERT(!up_s   || !up_b   || !is_scaled_low_precision(up));
    GGML_ASSERT(!gate_s || !gate_b || !is_scaled_low_precision(gate));
    GGML_ASSERT(!down_s || !down_b || !is_scaled_low_precision(down));
    GGML_ASSERT(!up_s   || !is_scaled_low_precision(up)   || !has_lora(up));
    GGML_ASSERT(!gate_s || !is_scaled_low_precision(gate) || !has_lora(gate));
    GGML_ASSERT(!down_s || !is_scaled_low_precision(down) || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    // default zero-filled — only archs loading clamp metadata
                    // (Step35, DSv4) get non-zero.
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_GELU_ERF:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_GEGLU_ERF:
            {
                cur = ggml_geglu_erf(ctx0, cur);
                cb(cur, "ffn_geglu_erf", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 have numerical issues with F16 accumulators
            ggml_prec_set_acc(cur, GGML_PREC_BF16);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

// fork: CPU fallback is exact; lossy weight policies require device residency tables.
static void moe_skipmass_cpu(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor *, int ith, int, void *) {
    GGML_ASSERT(!getenv("GGML_MOE_SKIP_POLICY") || strcmp(getenv("GGML_MOE_SKIP_POLICY"), "rank") == 0);
    if (ith == 0) { memcpy(dst->data, a->data, ggml_nbytes(a)); }
}

// the CUDA fusion matches this name (same literal in ggml-cuda.cu)
#define GGML_MOE_ONECHAIN_BANK_NAME "moe_onechain_bank"

// fork AIPFIX ONECHAIN: regular route values stay (cold mask 1), bank routes (cold mask 0, skipped by the regular
// MUL_MAT_ID) take the bank product. The three nodes stay adjacent: the CUDA backend writes the bank product straight
// into the regular output (ggml-cuda.cu, ggml_cuda_try_moe_onechain_fill); the bank kernel never writes skipped routes.
ggml_tensor * llm_graph_context::build_moe_onechain_fill(ggml_tensor * regular, ggml_tensor * bank_w, ggml_tensor * x) const {
    GGML_ASSERT(moe_onechain_ids && moe_onechain_cold);
    ggml_build_forward_expand(gf, regular);
    ggml_tensor * kept = ggml_mul_inplace(ctx0, regular, moe_onechain_cold);
    ggml_build_forward_expand(gf, kept);
    ggml_tensor * bank = ggml_mul_mat_id(ctx0, bank_w, x, moe_onechain_ids);
    ggml_set_name(bank, GGML_MOE_ONECHAIN_BANK_NAME);
    ggml_build_forward_expand(gf, bank);
    ggml_tensor * out = ggml_add_inplace(ctx0, kept, bank);
    ggml_build_forward_expand(gf, out);
    return out;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
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
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in,
        post_exp_norm,
        post_exp_norm_eps
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
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
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    const llm_arch arch = model.arch;
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        // Compact router weights must not round the activations to BF16 during prefill.
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS || hparams.expert_router_f32 ||
                (arch == LLM_ARCH_QWEN4EXP && gate_inp->type == GGML_TYPE_BF16)) {
            // the reference computes the router in F32: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L565
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (arch == LLM_ARCH_DEEPSEEK41) {
            // model.py Gate.forward: weights / (sum + 1e-20); the sum can fall below the F16 clamp below
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0f, 1e-20f);
        } else if (hparams.expert_weights_norm_eps > 0.0f) {
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0f, hparams.expert_weights_norm_eps);
        } else {
            // Avoid division by zero, clamp to smallest number representable by F16
            weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        }
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    if (moe_weights_override != nullptr) {
        // inner call of the hot/cold split below: the routing weights were already masked
        weights = moe_weights_override;
        moe_weights_override = nullptr;
    } else if ((getenv("GGML_MOE_SKIP_POLICY") || getenv("GGML_MOE_SKIP_TRACE")) &&
            norm_w && gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX && n_expert == 512 &&
            il < 48 && ubatch.n_seqs == 1 && n_expert_used == 10 && n_tokens > 0 && n_tokens <= 8) {
        weights = ggml_map_custom2(ctx0, weights, selected_experts, moe_skipmass_cpu, 1, nullptr);
        // fork: explicit graph output keeps adjusted weights alive until the final expert sum.
        ggml_format_name(weights, "moe_skipmass-%d", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    // warmup routes every expert (n_expert_used == n_expert): too many for the bank's filler slots, so it
    // runs on the cold bank alone, which holds every expert
    if (moe_hot.hot_map != nullptr && moe_hot.down->ne[2] < 2*n_expert_used) {
        moe_hot = {};
    }
    // fork AIPFIX ONEPATH (LLAMA_MOE_BANK_ONEPATH=1): decode-size batches run a banked layer as a plain layer. The expert
    // cache already holds the bank experts as zero-weight entries; it computes their routes from the bank rows itself
    // (moe-cache.cu, AI+ table GEMV), so no routing split, mask or merge nodes. Only batches the cache takes (at most
    // 64 routed rows, moe_cache_node_rows_max) qualify: past it the stock call would read the dropped bank rows.
    // Larger batches keep the bank graph. Prompt ubatches (not every token an output) and every B1 phase keep it too:
    // the scheduler runs those on the CPU path, which reads the dropped host rows of the bank experts.
    // So does every batch while the cache's device tables lack the bank rows (not built yet, or refused: f3l's 8K MTP
    // arm had partial bundles, no tables, and the stock call read the dropped rows).
    static const bool onepath = [] {
        const char * value = getenv("LLAMA_MOE_BANK_ONEPATH");
        return value && atoi(value) != 0;
    }();
    auto ai_unit = [](const ggml_tensor * t) {
        return t && (t->type == GGML_TYPE_AIT2_U || t->type == GGML_TYPE_AIT2P_U || t->type == GGML_TYPE_AIT3_U ||
                t->type == GGML_TYPE_AIT4_U);
    };
    if (onepath && moe_hot.hot_map != nullptr && (int64_t) n_tokens*n_expert_used <= 64 &&
            n_outputs == n_tokens && res->get_b1_ffn_phase() == 0 && res->get_moe_onepath_live() &&
            !(getenv("GGML_MOE_HOT_DBG") && atoi(getenv("GGML_MOE_HOT_DBG")) != 0) &&
            up_exps && gate_exps && down_exps && !gate_up_exps &&
            ai_unit(moe_hot.up) && ai_unit(moe_hot.gate) && ai_unit(moe_hot.down)) {
        moe_hot = {};
    }
    if (moe_hot.hot_map != nullptr) {
        // split the selected experts between the hot bank and the cold bank; each bank runs on all
        // n_expert_used columns with the weights of the other bank's experts set to zero
        const moe_hot_bank hot = moe_hot;
        moe_hot = {};
        // [B1] layer-major prompt: the routing phase runs the hot call here (the hot bank is on the card) and records
        // the cold weights, hot slots zeroed, for the whole-prompt cold GEMMs; their zero-weight rows are skipped.
        const bool b1_routing = res->get_b1_ffn_phase() == 1 && res->get_b1_ffn() != nullptr;
        GGML_ASSERT(!weight_before_ffn);

        // the maps hold one row per expert; look them up with the flattened expert ids
        ggml_tensor * sel_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, selected_experts), n_expert_used*n_tokens, 1);
        ggml_tensor * slot = ggml_reshape_3d(ctx0,
                ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, hot.hot_map, 1, n_expert), sel_flat), 1, n_expert_used, n_tokens);
        cb(slot, "ffn_moe_hot_slot", il);
        static const bool mixed_enabled = [] {
            const char * value = getenv("LLAMA_MOE_BANK_MIXED");
            return !value || atoi(value) != 0;
        }();
        if (mixed_enabled && !b1_routing && n_tokens <= 8 &&
                !(getenv("GGML_MOE_HOT_DBG") && atoi(getenv("GGML_MOE_HOT_DBG")) != 0) &&
                type_op == LLM_FFN_SILU && il >= 0 && hparams.swiglu_clamp_exp[il] <= 1e-6f &&
                loras->empty() && (!hadamard_rotations || hadamard_rotations->empty()) &&
                up_exps && gate_exps && down_exps && !gate_up_exps &&
                !up_exps_b && !gate_exps_b && !down_exps_b &&
                !up_exps_s && !gate_exps_s && !down_exps_s &&
                (hot.up->type == GGML_TYPE_AIT2_U || hot.up->type == GGML_TYPE_AIT2P_U || hot.up->type == GGML_TYPE_AIT3_U) &&
                (hot.gate->type == GGML_TYPE_AIT2_U || hot.gate->type == GGML_TYPE_AIT2P_U || hot.gate->type == GGML_TYPE_AIT3_U) &&
                (hot.down->type == GGML_TYPE_AIT2_U || hot.down->type == GGML_TYPE_AIT2P_U || hot.down->type == GGML_TYPE_AIT3_U ||
                 (hot.down->type == GGML_TYPE_AIT4_U && hot.down->ne[0] == 640) ||
                 (hot.down->type == GGML_TYPE_IQ4_NL && down_exps->type == GGML_TYPE_IQ4_NL))) {
            ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, nullptr, /*cpu_part=*/false, il).slots;
            // fork: slot-cache IDs have a different domain from the bank map; retain the old path there.
            if (!slots) {
                ggml_tensor * x = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
                ggml_build_forward_expand(gf, x);
                ggml_tensor * hot_ids = ggml_cast(ctx0, ggml_reshape_2d(ctx0, slot, n_expert_used, n_tokens), GGML_TYPE_I32);
                ggml_tensor * cold_mask = ggml_reshape_3d(ctx0,
                        ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, hot_ids, GGML_TYPE_F32), -1.0f, -0.5f)),
                        1, n_expert_used, n_tokens);
                ggml_build_forward_expand(gf, cold_mask);
                auto mixed_mm = [&](ggml_tensor * hw, ggml_tensor * cw, ggml_tensor * x) {
                    ggml_tensor * h = build_lora_mm_id(hw, x, hot_ids);
                    ggml_tensor * c = build_lora_mm_id(cw, x, selected_experts);
                    return ggml_add(ctx0, h, ggml_mul(ctx0, c, cold_mask));
                };
                ggml_tensor * up = mixed_mm(hot.up, up_exps, x);
                ggml_tensor * gate = mixed_mm(hot.gate, gate_exps, x);
                x = ggml_swiglu_split(ctx0, gate, up);
                ggml_tensor * experts = mixed_mm(hot.down, down_exps, x);
                ggml_tensor * weighted = ggml_mul(ctx0, experts, weights);
                ggml_tensor * sum = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, weighted, 1, 0, 2, 3)));
                ggml_tensor * out = ggml_reshape_2d(ctx0, sum, n_embd, n_tokens);
                cb(out, "ffn_moe_out", il);
                return out;
            }
        }
        ggml_tensor * mask  = ggml_step(ctx0, ggml_scale_bias(ctx0, slot, 1.0f, 0.5f)); // 1 = hot, 0 = cold
        ggml_tensor * w_hot  = ggml_mul(ctx0, weights, mask);
        ggml_tensor * w_cold = ggml_sub(ctx0, weights, w_hot);
        cb(w_hot,  "ffn_moe_weights_hot",  il);
        cb(w_cold, "ffn_moe_weights_cold", il);

        // mul_mat_id needs every expert at most once per token, so the masked (cold) slots of the hot call
        // get distinct filler slots that the token does not use: one-hot the token's hot slots over the
        // bank, then take the n_expert_used least-used bank slots (argsort puts the used ones last)
        const int64_t n_hot = hot.down->ne[2];
        // AI+ banks: the masked slots get id -1, which the AI+ MUL_MAT_ID skips (zero output, CUDA and CPU), so the hot
        // call computes only the token's hot pairs instead of n_expert_used pairs with fillers. LLAMA_MOE_HOT_FILLER=1
        // keeps the filler slots (an A/B control; the outputs are identical because the filler weights are zero).
        static const bool hot_filler = getenv("LLAMA_MOE_HOT_FILLER") != nullptr && atoi(getenv("LLAMA_MOE_HOT_FILLER")) != 0;
        const bool hot_skip = !hot_filler && ggml_ai4_rate(hot.up->type) != 0 && ggml_ai4_rate(hot.gate->type) != 0 &&
                              ggml_ai4_rate(hot.down->type) != 0 &&
                              !(getenv("GGML_MOE_HOT_DBG") && atoi(getenv("GGML_MOE_HOT_DBG")) == 1); // dbg 1 runs the cold bank
        ggml_tensor * hot_ids = nullptr;
        if (hot_skip) {
            // (slot + 1) * mask - 1: the bank slot where hot, -1 where cold
            ggml_tensor * hot_f = ggml_scale_bias(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, slot, 1.0f, 1.0f), mask), 1.0f, -1.0f);
            hot_ids = ggml_cast(ctx0, ggml_reshape_2d(ctx0, hot_f, n_expert_used, n_tokens), GGML_TYPE_I32);
        } else {
            ggml_tensor * bank_pos = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_hot, 1.0f), n_hot, 1, 1),
                    n_hot, n_expert_used, n_tokens, 1);
            ggml_tensor * onehot = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, bank_pos, slot)), -1.0f, 0.5f)); // [n_hot, n_expert_used, n_tokens]
            ggml_tensor * used   = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, onehot, 1, 0, 2, 3)));               // [1, n_hot, n_tokens]
            ggml_tensor * filler = ggml_argsort_top_k(ctx0, ggml_scale(ctx0, ggml_reshape_2d(ctx0, used, n_hot, n_tokens), -1.0f), n_expert_used); // [n_expert_used, n_tokens]
            ggml_tensor * fill_f = ggml_reshape_3d(ctx0, ggml_cast(ctx0, ggml_cont(ctx0, filler), GGML_TYPE_F32), 1, n_expert_used, n_tokens);
            ggml_tensor * hot_f  = ggml_add(ctx0, ggml_mul(ctx0, slot, mask), ggml_sub(ctx0, fill_f, ggml_mul(ctx0, fill_f, mask)));
            hot_ids = ggml_cast(ctx0, ggml_reshape_2d(ctx0, hot_f, n_expert_used, n_tokens), GGML_TYPE_I32);
        }
        cb(hot_ids, "ffn_moe_hot_ids", il);
        // the cold bank keeps all n_expert experts, so the routed ids stay unique there; the hot experts
        // run with zero weight in the cold call
        GGML_ASSERT(down_exps == nullptr || down_exps->ne[2] == n_expert);
        ggml_tensor * cold_ids = selected_experts;

        static const int hot_dbg = getenv("GGML_MOE_HOT_DBG") ? atoi(getenv("GGML_MOE_HOT_DBG")) : 0;
        static const bool joint_enabled = [] {
            const char * value = getenv("LLAMA_MOE_BANK_JOINT_REDUCE");
            return value && atoi(value) != 0;
        }();
        const bool joint_reduce = joint_enabled && !b1_routing && hot_skip && hot_dbg == 0 &&
                n_tokens <= 8 && n_expert_used == hparams.n_expert_used(il);
        // fork AIPFIX ONECHAIN (default on; LLAMA_MOE_BANK_ONECHAIN=0 opts out): prompt batches skip the separate bank chain (its
        // full-width zero fill, SwiGLU, weighting and reduction); the regular chain below takes the bank products
        // in place with the full routing weights. Valid math: one sum over all routes instead of bank + regular sums.
        static const bool onechain_enabled = [] {
            const char * value = getenv("LLAMA_MOE_BANK_ONECHAIN");
            return !value || atoi(value) != 0;
        }();
        static const bool onechain_cold_skip = [] {
            const char * value = getenv("LLAMA_MOE_COLD_SKIP");
            return !value || atoi(value) != 0;
        }();
        if (onechain_enabled && onechain_cold_skip && hot_dbg == 0 && !b1_routing && n_tokens > 8 && hot_skip && !joint_reduce && type_op == LLM_FFN_SILU &&
                il >= 0 && hparams.swiglu_clamp_exp[il] <= 1e-6f && loras->empty() &&
                (!hadamard_rotations || hadamard_rotations->empty()) &&
                up_exps && gate_exps && down_exps && !gate_up_exps && !up_exps_b && !gate_exps_b && !down_exps_b &&
                !up_exps_s && !gate_exps_s && !down_exps_s && !llm_sched_needs_moe_f16_act_guard(sched)) {
            moe_route_skip = slot;
            moe_weights_override = weights;
            moe_onechain_bank = &hot;
            moe_onechain_ids  = hot_ids;
            moe_onechain_cold = ggml_scale_bias(ctx0, mask, -1.0f, 1.0f);
            ggml_build_forward_expand(gf, hot_ids);
            ggml_build_forward_expand(gf, moe_onechain_cold);
            ggml_tensor * out = build_moe_ffn(cur, gate_inp, nullptr,
                    up_exps, nullptr, gate_exps, nullptr, down_exps, nullptr, exp_probs_b,
                    n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il,
                    logits, nullptr, nullptr, nullptr, nullptr, nullptr, cold_ids);
            moe_onechain_bank = nullptr;
            moe_onechain_ids  = nullptr;
            moe_onechain_cold = nullptr;
            moe_route_skip = nullptr;
            GGML_ASSERT(moe_weights_override == nullptr);
            return out;
        }
        GGML_ASSERT(!moe_defer_reduce);
        moe_defer_reduce = joint_reduce;
        moe_weights_override = w_hot;
        moe_b1_bypass = b1_routing;
        ggml_tensor * out_hot = hot_dbg == 1 ?
            build_moe_ffn(cur, gate_inp, nullptr,
                up_exps, up_exps_b, gate_exps, gate_exps_b, down_exps, down_exps_b, exp_probs_b,
                n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il,
                logits, gate_up_exps, gate_up_exps_b, up_exps_s, gate_exps_s, down_exps_s, selected_experts) :
            build_moe_ffn(cur, gate_inp, nullptr,
                hot.up, nullptr, hot.gate, nullptr, hot.down, nullptr, exp_probs_b,
                n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il,
                logits, nullptr, nullptr, nullptr, nullptr, nullptr, hot_ids);
        moe_b1_bypass = false;
        if (hot_dbg == 2) {
            out_hot = ggml_scale(ctx0, out_hot, 0.0f);
            w_cold  = weights;
        }
        moe_weights_override = w_cold;
        // prompt batches: the cold GEMMs leave out the hot routes (zero weight here, computed by the hot call)
        // instead of running them against the zero-filled bank holes. LLAMA_MOE_COLD_SKIP=0 opts out.
        static const bool cold_skip = [] {
            const char * value = getenv("LLAMA_MOE_COLD_SKIP");
            return !value || atoi(value) != 0;
        }();
        if (cold_skip && hot_dbg == 0 && !b1_routing && n_tokens > 8) {
            moe_route_skip = slot;
        }
        ggml_tensor * out_cold = build_moe_ffn(cur, gate_inp, nullptr,
                up_exps, up_exps_b, gate_exps, gate_exps_b, down_exps, down_exps_b, exp_probs_b,
                n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il,
                logits, gate_up_exps, gate_up_exps_b, up_exps_s, gate_exps_s, down_exps_s, cold_ids);
        moe_route_skip = nullptr;
        GGML_ASSERT(moe_weights_override == nullptr);
        moe_defer_reduce = false;
        if (joint_reduce) {
            // fork: keep both expert slabs live until one weighted reduction consumes them.
            ggml_build_forward_expand(gf, out_hot);
            ggml_build_forward_expand(gf, out_cold);
            ggml_tensor * hot_weighted = ggml_mul(ctx0, out_hot, w_hot);
            ggml_tensor * cold_weighted = ggml_mul(ctx0, out_cold, w_cold);
            ggml_tensor * joined = ggml_add(ctx0, hot_weighted, cold_weighted);
            ggml_tensor * summed = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, joined, 1, 0, 2, 3)));
            ggml_tensor * moe_out = ggml_reshape_2d(ctx0, summed, n_embd, n_tokens);
            cb(moe_out, "ffn_moe_out", il);
            return moe_out;
        }
        if (b1_routing) {
            GGML_ASSERT(out_cold == nullptr);
            return out_hot; // added to the shared expert output that seeds the whole-prompt accumulator
        }
        ggml_tensor * moe_out = ggml_add(ctx0, out_hot, out_cold);
        cb(moe_out, "ffn_moe_out", il);
        return moe_out;
    }

    const auto * b1_ffn = res->get_b1_ffn();
    if (res->get_b1_ffn_phase() == 1 && b1_ffn && !moe_b1_bypass) {
        GGML_ASSERT(selected_experts->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, selected_experts, b1_ffn->ids));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens), b1_ffn->weights));
        return nullptr;
    }

    // compute the expert input before the slot lookup, so that the scheduler prefetches it for the CPU part (the router can read a different tensor)
    ggml_build_forward_expand(gf, cur);

    // the experts of host-resident layers may be read from the MoE cache, and part of the misses computed from the host experts
    // the CPU part does not support weights before the FFN, LoRAs or rotated experts
    const std::array<ggml_tensor *, 4> exps = { up_exps, gate_exps, down_exps, gate_up_exps };
    const bool can_cpu_share = !weight_before_ffn && loras->empty() &&
        std::none_of(exps.begin(), exps.end(), [&](ggml_tensor * w) { return w != nullptr && hdmd != nullptr && hdmd->rot.count(w) > 0; }) &&
        // fork: the hot/cold bank split, the onechain bank fill and the fork Hadamard table keep the cached path only
        (!hadamard_rotations || hadamard_rotations->empty()) && moe_onechain_bank == nullptr && moe_route_skip == nullptr;
    const llm_graph_moe_cache_slots cache = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, can_cpu_share, il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    ggml_tensor * const moe_in = cur;

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    // the experts selected by mm_ids, up to the down projection, from the MoE cache if cached
    // ids are the selected experts, for the biases and scales
    auto build_experts = [&](ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached) {
        ggml_tensor * up = nullptr;

        if (gate_up_exps) {
            // merged gate_up path: one mul_mat_id, then split into gate and up views
            ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, ids, up_exps_s, mm_ids, cached); // [n_ff*2, n_expert_used, n_tokens]
            cb(gate_up, "ffn_moe_gate_up", il);

            if (up_exps_s) {
                cb(gate_up, "ffn_moe_gate_up_scaled", il);
            }

            if (gate_up_exps_b) {
                gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, ids);
                cb(gate_up, "ffn_moe_gate_up_biased", il);
            }

            const int64_t n_ff = gate_up->ne[0] / 2;
            cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            cb(cur, "ffn_moe_gate", il);
            up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
            cb(up, "ffn_moe_up", il);
        } else {
            // separate gate and up path
            up = build_lora_mm_id(up_exps, cur, ids, up_exps_s, mm_ids, cached); // [n_ff, n_expert_used, n_tokens]
            cb(up, "ffn_moe_up", il);

            if (up_exps_s) {
                cb(up, "ffn_moe_up_scaled", il);
            }

            if (up_exps_b) {
                up = ggml_add_id(ctx0, up, up_exps_b, ids);
                cb(up, "ffn_moe_up_biased", il);
            }

            if (gate_exps) {
                cur = build_lora_mm_id(gate_exps, cur, ids, gate_exps_s, mm_ids, cached); // [n_ff, n_expert_used, n_tokens]
                cb(cur, "ffn_moe_gate", il);
            } else {
                cur = up;
            }

            if (gate_exps_s) {
                cb(cur, "ffn_moe_gate_scaled", il);
            }

            if (gate_exps_b) {
                cur = ggml_add_id(ctx0, cur, gate_exps_b, ids);
                cb(cur, "ffn_moe_gate_biased", il);
            }
            if (moe_onechain_bank && cached) {
                // fork AIPFIX ONECHAIN: keep the two regular products adjacent (shared prompt MMQ prep), then fill their bank routes
                ggml_build_forward_expand(gf, cur);
                ggml_build_forward_expand(gf, up);
                up  = build_moe_onechain_fill(up,  moe_onechain_bank->up,   moe_in);
                cur = build_moe_onechain_fill(cur, moe_onechain_bank->gate, moe_in);
            }
        }

        const bool has_gate = gate_exps || gate_up_exps;

        switch (type_op) {
            case LLM_FFN_SILU:
                if (gate_exps) {
                    if (il >= 0) {
                        const float limit = hparams.swiglu_clamp_exp[il];
                        constexpr float eps = 1e-6f;
                        if (limit > eps) {
                            if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                                cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                            } else {
                                up = ggml_clamp(ctx0, up, -limit, limit);
                                cb(up, "ffn_moe_up_clamped", il);
                                ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                                cb(gate_act, "ffn_moe_silu", il);
                                gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                                cb(gate_act, "ffn_moe_silu_clamped", il);
                                cur = ggml_mul(ctx0, gate_act, up);
                            }
                            cb(cur, "ffn_moe_swiglu_limited", il);
                            break;
                        }
                    }
                }

                if (has_gate) {
                    cur = ggml_swiglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_SITU:
                {
                    // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                    GGML_ASSERT(has_gate);
                    const float beta = hparams.situ_beta;
                    const float lb   = hparams.situ_linear_beta;

                    ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                    act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                    if (lb > 0.0f) {
                        up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                    }
                    cur = ggml_mul(ctx0, act, up);
                    cb(cur, "ffn_moe_situ", il);
                } break;
            case LLM_FFN_GELU:
                if (has_gate) {
                    cur = ggml_geglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_moe_gelu", il);
                } break;
            case LLM_FFN_SWIGLU_OAI_MOE:
                {
                    // TODO: move to hparams?
                    constexpr float alpha = 1.702f;
                    constexpr float limit = 7.0f;
                    cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                    cb(cur, "ffn_moe_swiglu_oai", il);
                } break;
            case LLM_FFN_RELU:
                if (has_gate) {
                    cur = ggml_reglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_reglu", il);
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_moe_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                if (has_gate) {
                    // TODO: add support for gated squared relu
                    GGML_ABORT("fatal error: gated squared relu not implemented");
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_moe_relu_sqr", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        // TODO: this graph-level rescaling should only be applied when the corresponding backend does not support
        //       ggml_prec_set_src(experts, GGML_PREC_BF16, 1);
        //       the amax op cannot be split across devices, so these architectures are denied -sm tensor [TAG_MOE_AMAX_SCALE]
        //       ref: https://github.com/ggml-org/llama.cpp/pull/30656#issuecomment-6269597922
        llm_graph_amax_src1 amax_src1 = { nullptr, nullptr };

        if (arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_MELLUM || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_LAGUNA) {
            amax_src1 = build_amax_src1_f16(cur);
            cur = amax_src1.res;
        }

        // fork: direct path by default (CUDA-first). Metal-only L2 rescale when needed
        // (see llm_sched_needs_moe_f16_act_guard). Mathematically identity on Metal;
        // keeps f16 MUL_MAT_ID acts under 65504. Not combined with the upstream amax rescale.
        ggml_tensor * experts = nullptr;
        if (amax_src1.scale_s == nullptr && llm_sched_needs_moe_f16_act_guard(sched)) {
            const float f16_safe = 32768.0f; // stay well under the f16 max of 65504
            ggml_tensor * col_l2 = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur))); // [1, n_expert_used, n_tokens]
            col_l2 = ggml_clamp(ctx0, col_l2, 1e-8f, 1e30f); // guard empty columns against div-by-zero
            ggml_tensor * cur_s = ggml_div(ctx0, ggml_scale(ctx0, cur, f16_safe), col_l2);
            experts = build_lora_mm_id(down_exps, cur_s, ids, down_exps_s, mm_ids, cached); // [n_embd, n_expert_used, n_tokens]
            experts = ggml_scale(ctx0, ggml_mul(ctx0, experts, col_l2), 1.0f/f16_safe);
        } else {
            experts = build_lora_mm_id(down_exps, cur, ids, down_exps_s, mm_ids, cached); // [n_embd, n_expert_used, n_tokens]
        }
        //ggml_prec_set_src(experts, GGML_PREC_BF16, 1); // (see TODO above)
        cb(experts, "ffn_moe_down", il);
        if (moe_onechain_bank && cached) {
            // fork AIPFIX ONECHAIN: fill the bank routes once (the cached call; the CPU part is off with onechain)
            experts = build_moe_onechain_fill(experts, moe_onechain_bank->down, cur);
        }

        if (amax_src1.scale_s) {
            experts = ggml_mul(ctx0, experts, amax_src1.scale_s);
        }

        return experts;
    };

    ggml_tensor * experts = build_experts(cur, selected_experts, cache.slots, true);
    if (cache.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(gf, experts);
        experts = ggml_add(ctx0, experts, build_experts(cur, selected_experts, cache.cpu_ids, false));
        cb(experts, "ffn_moe_down_merged", il);
    }

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // an expert output can be scaled by a norm weight, then normalized, before the router weights apply
    if (post_exp_norm) {
        experts = ggml_mul(ctx0, experts, post_exp_norm);
        experts = ggml_rms_norm(ctx0, experts, post_exp_norm_eps);
        cb(experts, "ffn_moe_post_exp_norm", il);
    }

    if (moe_defer_reduce) {
        GGML_ASSERT(!weight_before_ffn);
        return experts;
    }

    if (!weight_before_ffn) {
        // LLAMA_MOE_WEIGHT_INPLACE=1 (opt-in): prompt batches scale the expert outputs in place, so the weighted copy
        // (n_embd x n_expert_used x n_tokens f32, 400 MiB at 4,096 tokens) does not sit next to the unweighted one in the
        // layer graph. Same values; only valid while nothing else reads ffn_moe_down after this node.
        static const bool w_inplace = [] { const char * e = getenv("LLAMA_MOE_WEIGHT_INPLACE"); if (e) return atoi(e) != 0; const char * l = getenv("LLAMA_LEAN_PROMPT"); return l && atoi(l) != 0; }(); // LLAMA_LEAN_PROMPT (unset = off; llama_context sets it on small cards)
        experts = w_inplace && n_tokens > 8 ? ggml_mul_inplace(ctx0, experts, weights) : ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    assert(n_expert_used > 0);

    // experts layout: [n_embd, n_expert_used, n_tokens]
    // Decode (n_tokens==1): permute+sum_rows beats 10 views + 9 adds.
    // Prefill: the cont/permute of a large expert slab is slower than the
    // classic view/add tree — keep that path for multi-token.
    // Use the per-layer n_expert_used (not the ubatch one) so the graph stays
    // bounded during warmup. ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    static const bool small_reduce = [] {
        const char * value = getenv("LLAMA_MOE_SMALL_REDUCE");
        return !value || atoi(value) != 0;
    }();
    // Small verify batches avoid one add launch per expert. Keep warmup's wider routing unchanged.
    const bool reduce_verify = small_reduce && n_tokens <= 8 && n_expert_used_il == n_expert_used;
    ggml_tensor * moe_out;
    if (n_expert_used_il == 1) {
        moe_out = ggml_cont(ctx0, ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], 0));
    } else if (n_tokens == 1 || reduce_verify) {
        ggml_tensor * experts_pe = ggml_cont(ctx0, ggml_permute(ctx0, experts, 1, 0, 2, 3));
        ggml_tensor * summed     = ggml_sum_rows(ctx0, experts_pe); // [1, n_embd, 1]
        moe_out = ggml_reshape_2d(ctx0, summed, n_embd, n_tokens);
    } else {
        ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
        for (uint32_t i = 0; i < n_expert_used_il; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        moe_out = cur_experts[0];
        for (uint32_t i = 1; i < n_expert_used_il; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

ggml_tensor * llm_graph_context::build_hadamard_inverse(ggml_tensor * table, ggml_tensor * cur) const {
    if (hadamard_inverses) {
        const auto it = hadamard_inverses->find(table);
        if (it != hadamard_inverses->end()) {
            cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
            if (it->second.signs) {
                cur = ggml_mul(ctx0, cur, it->second.signs);
            }
        }
    }
    return cur;
}

llm_graph_moe_cache_slots llm_graph_context::build_moe_cache_slots(
         ggml_tensor * selected_experts,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * gate_up_exps,
                bool   cpu_part,
                 int   il) const {
    if (moe_cache == nullptr) {
        return { nullptr, nullptr };
    }

    ggml_tensor * cpu_map  = cpu_part ? moe_cache->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = moe_cache->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return { nullptr, nullptr };
    }
    for (ggml_tensor * w : { up_exps, gate_exps, down_exps, gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = moe_cache->get_experts(w);
        if (experts == nullptr) {
            return { nullptr, nullptr };
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(moe_cache->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return { nullptr, nullptr };
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(moe_cache->backend(il), slots)) {
        return { nullptr, nullptr };
    }
    ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il));
    cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(gf, slots);

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    // llamAmpere: build the mixed branch only for a mixed ubatch. Its three inputs live in host memory, and the
    // scheduler uploads every split input with a synchronous copy whether or not its select branch runs, so the
    // always-built branch cost three extra blocking H2D copies per decode on every text-only round (~30 us/round
    // on a 5800X3D, more on slower hosts). The graph params compare is_mixed(), so a mixed ubatch still gets a
    // fresh graph (and, the first time, a larger compute buffer).
    const bool has_mixed = llm_arch_supports_mixed_batch(model.arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        ubatch.is_mixed();

    const int64_t n_tok_rows = has_mixed ? llm_graph_n_tok_rows(ubatch) : 0;

    // we have 3 standard paths to produce the input embeddings for the first layer:
    // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
    // - embd1: pass raw embeddings, skipping the `tok_embd`
    // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
    ggml_tensor * embd0 = nullptr;
    ggml_tensor * embd1 = nullptr;
    ggml_tensor * embd2 = nullptr;

    // construct the input tensors
    {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        if (has_mixed) {
            inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
            cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
            ggml_set_input(inp->mixed_tokens);
        }
    }

    // the embeddings placeholders for the 3 paths
    // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
    // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
    {
        embd0 = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        ggml_build_forward_order(gf, embd0);

        embd1 = inp->embd;

        if (has_mixed) {
            embd2 = ggml_get_rows(ctx0, tok_embd, inp->mixed_tokens);
            ggml_build_forward_order(gf, embd2);
        }
    }

    // helper for extracting token embeddings with lora and padding
    // TODO: when lora is active, this is likely going to cause issues similar to https://github.com/ggml-org/llama.cpp/pull/30160
    //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
    auto build_tok = [&](ggml_tensor * cur, ggml_tensor * ids) {
        // a Hadamard-latent table stores rotated rows: restore the primal basis, h = s * (H z)
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd); it != hdmd->inv.end()) {
                cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
                if (it->second.signs) {
                    cur = ggml_mul(ctx0, cur, it->second.signs);
                }
            }
        }

        // a Hadamard-latent embedding table stores rotated rows; restore the
        // primal basis right after the lookup: h = s * (H z)
        cur = build_hadamard_inverse(tok_embd, cur);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(embd0, inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = embd1;

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    if (has_mixed) {
        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        ggml_tensor * embd_mixed = build_tok(embd2, inp->mixed_tokens);
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), embd_mixed, inp->mixed_slots);

        assert(ggml_are_same_shape (inps[0], inps[2]));
        assert(ggml_are_same_stride(inps[0], inps[2]));
    }

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    const bool scale_tok_only = hparams.f_embedding_scale != 0.0f && hparams.n_deepstack_layers > 0;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f && !scale_tok_only) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale*(scale_tok_only ? hparams.f_embedding_scale : 1.0f);
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_embd_state(llm_graph_input_embd_h * inp, int64_t n_embd_state) const {
    if (ubatch.data && !ubatch.data->embd_state_row.empty()) {
        GGML_ASSERT(h_nextn_src != nullptr && h_nextn_src->ne[0] == n_embd_state);

        inp->h_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->h_rows);
        ggml_set_name(inp->h_rows, "mtp_h_rows");

        return ggml_get_rows(ctx0, h_nextn_src, inp->h_rows);
    }

    ggml_tensor * h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_state, n_tokens);
    ggml_set_input(h);
    ggml_set_name(h, "mtp_h_input");

    return h;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd(), hparams.rope_section_order);

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset, hparams.attn_temp_pow);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    cb(cur, "inp_attn_scale", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, model.arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);
    ggml_set_name(cur, "cross_embd");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_enc");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_dec");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_turbo_q(
              ggml_tensor * q,
        const ggml_tensor * k,
        const llama_kv_cache_context * mctx_cur) const {
    // TurboQuant pre-rotate-queries: O(d log d) WHT rotation via custom op
    // Q shape: (n_embd_head, n_head, n_tokens)
    // For zero-padded models (head_dim not 128-aligned), pad Q to match padded K dim first.
    // SJ-KVaRN: the staging K view is tq6_0 but holds K in the SJ-KVaRN Hadamard basis (build_attn rotates Q/K/V
    // itself and stores K with ggml_set_rows_tq6_rotated, no WHT and no InnerQ), so no turbo rotation here.
    const bool is_sj_kvarn = mctx_cur->is_sj_kvarn();
    if (!is_sj_kvarn && (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0 || k->type == GGML_TYPE_TQ6_0 || k->type == GGML_TYPE_TQ5_0)) {
        // Pad Q per-head to next multiple of 128 if needed
        if (q->ne[0] % 128 != 0) {
            const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
            q = ggml_pad(ctx0, q, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx0, q); }
        ggml_tensor * innerq_scale = mctx_cur->get_turbo_innerq_scale_inv();
        q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);  // 0 = forward, 0 = auto group size from q->ne[0]
    }

    return q;
}

ggml_tensor * llm_graph_context::build_attn_turbo_v_out(
              ggml_tensor * cur,
        const ggml_tensor * v,
                      int   il) const {
    // TurboQuant: if V was padded, the output has padded dimensions.
    // Extract original V head_dim after inverse WHT (applied inside build_attn_mha).
    // NOTE: gate on v->type (not k->type) for asymmetric configs where K=q8_0 but V=turbo
    if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0 || v->type == GGML_TYPE_TQ6_0 || v->type == GGML_TYPE_TQ5_0) {
        const int64_t orig_v_head = hparams.n_embd_head_v(il);
        // cur is 2D: (n_embd_head * n_head, n_tokens) after build_attn_mha
        const int64_t padded_v_head = v->ne[0];
        if (padded_v_head != orig_v_head) {
            // Reshape to 4D, extract original head_dim, reshape back to 2D
            // Fix #78 (bingh0): cur shape post-MHA is (n_embd_head * n_head, n_tokens),
            // not (n_embd_head * n_head_kv, n_tokens). The output carries one
            // row per Q head, so derive the head count from the tensor itself:
            // hparams.n_head(il) is correct but is a second source of truth
            // that has to agree with cur — ne[0]/padded_v_head cannot disagree,
            // and the assert turns a silent nelements mismatch (observed live:
            // gpt-oss 64->128 V pad, 64q/8kv, abort in ggml_reshape_3d during
            // graph reserve) into a loud failure at the exact site.
            GGML_ASSERT(cur->ne[0] % padded_v_head == 0);
            const int64_t n_head_v = cur->ne[0] / padded_v_head;
            const int64_t n_tokens_cur = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_tokens_cur);
            // ggml_view_3d to extract first orig_v_head elements per head
            cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_tokens_cur,
                               cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * n_head_v, n_tokens_cur);
        }
    }

    return cur;
}

static ggml_tensor * llm_graph_sj_wush_apply(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * A); // EXPERIMENTAL WUSH-KV

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * kv_rows,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = kv_rows ? kv_rows->ne[1] : k->ne[3];

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // TurboQuant note: graph-side Q rotation (pre-rotate-queries) is implemented below
    // in the flash-attn path. The VEC kernel bug (wrong Q/K stride in
    // vec_dot_fattn_vec_KQ_turbo3_0) was fixed in fattn-common.cuh to match f16 pattern.

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    GGML_ASSERT(use_flash_attn || kv_rows == nullptr);
    // LLAMA_LSA_BLKLIST mask: only the single-stream flash attention node below resolves its sentinel cells
    GGML_ASSERT(lsa_fa_sel == nullptr || kq_mask != lsa_fa_m3 || (use_flash_attn && k->ne[3] == 1));
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        // llamAmpere: kv_rows (upstream ggml_flash_attn_ext_rows, FA src[9] in this tree) is never combined with
        // SJ-KVaRN (src[5..7]), QSA selected rows (src[8]), the packed mask or a TurboQuant/TQ K/V cache; the mask
        // builders do not create kv_rows for those, so reaching here with both is a bug
        if (kv_rows) {
            GGML_ASSERT(sj_kvarn_pending.body == nullptr && "kv_rows cannot be combined with an SJ-KVaRN body on one FA node");
            GGML_ASSERT(fa_sel_rows == nullptr && "kv_rows cannot be combined with QSA selected rows (FA src[8])");
            GGML_ASSERT(kq_mask && kq_mask->type != GGML_TYPE_I32 && "kv_rows cannot be combined with the packed KQ mask");
            GGML_ASSERT(!llm_graph_kv_type_is_turbo(k->type) && !llm_graph_kv_type_is_turbo(v->type) &&
                    "kv_rows cannot be combined with a TurboQuant/TQ K/V cache");
        }

        if (sj_kvarn_pending.body != nullptr && k->ne[3] > 1) {
            // SJ-KVaRN with several streams (one per sequence) in the ubatch: one FA node per stream, each with its
            // own descriptor row (sealed end, ring state, group table) and its own attended length, exactly the
            // node a single-sequence cache builds for that stream; the outputs are concatenated stream-major
            const int64_t ns = k->ne[3];
            const bool packed_mask = kq_mask->type == GGML_TYPE_I32;
            const int64_t mask_width = kq_mask->ne[0]*(packed_mask ? 32 : 1);
            GGML_ASSERT(sj_kvarn_pending.n_kv_streams && (int64_t) sj_kvarn_pending.n_kv_streams->size() == ns);
            ggml_tensor * desc = sj_kvarn_pending.desc;
            GGML_ASSERT(desc->ne[1] == ns && q->ne[3] == ns && v->ne[3] == ns && kq_mask->ne[3] == ns);
            const int group = sj_kvarn_pending.body->op_params[7] == GGML_TYPE_TURBO4_0 ? 128 : 256;
            const bool fused_rot = sj_kvarn_pending.fused_rot;
            ggml_tensor * wush_ao = sj_kvarn_pending.wush_ao; // EXPERIMENTAL WUSH-KV layer (fused_rot is off then)
            ggml_tensor * out = nullptr;
            for (int64_t s = 0; s < ns; ++s) {
                const int32_t n_kv_s = (int32_t) (*sj_kvarn_pending.n_kv_streams)[s];
                GGML_ASSERT(n_kv_s > 0 && n_kv_s % 64 == 0 && n_kv_s <= mask_width);
                ggml_tensor * q_s = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], s*q->nb[3]);
                ggml_tensor * k_s = ggml_view_4d(ctx0, k, k->ne[0], k->ne[1], k->ne[2], 1, k->nb[1], k->nb[2], k->nb[3], s*k->nb[3]);
                ggml_tensor * v_s = ggml_view_4d(ctx0, v, v->ne[0], v->ne[1], v->ne[2], 1, v->nb[1], v->nb[2], v->nb[3], s*v->nb[3]);
                if (sj_kvarn_mask_src != kq_mask) {
                    sj_kvarn_mask_src = kq_mask;
                    sj_kvarn_mask_s.assign(ns, nullptr);
                }
                GGML_ASSERT((int64_t) sj_kvarn_mask_s.size() == ns);
                ggml_tensor * m_s = sj_kvarn_mask_s[s];
                const int64_t mask_cols = packed_mask ? (n_kv_s + 31)/32 : n_kv_s;
                if (m_s == nullptr || m_s->ne[0] != mask_cols) {
                    m_s = ggml_view_4d(ctx0, kq_mask, mask_cols, kq_mask->ne[1], kq_mask->ne[2], 1,
                            kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], s*kq_mask->nb[3]);
                    if (!ggml_is_contiguous(m_s)) {
                        m_s = ggml_cont(ctx0, m_s);
                    }
                    sj_kvarn_mask_s[s] = m_s;
                }
                ggml_tensor * d_s = ggml_view_1d(ctx0, desc, desc->ne[0], s*desc->nb[1]);
                ggml_tensor * c = ggml_flash_attn_ext(ctx0, q_s, k_s, v_s, m_s, kq_scale, hparams.f_max_alibi_bias,
                                          hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
                ggml_flash_attn_ext_add_sinks(c, sinks);
                GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
                ggml_flash_attn_ext_set_n_kv_max(c, static_cast<int32_t>(n_kv_max));
                ggml_prec_set_acc(c, GGML_PREC_F32);
                ggml_flash_attn_ext_set_sj_kvarn(c, sj_kvarn_pending.body, d_s,
                        sj_kvarn_pending.bits_k, sj_kvarn_pending.bits_v, n_kv_s);
                if (fused_rot) {
                    ggml_flash_attn_ext_set_sj_kvarn_rot(c, 256);
                }
                out = out ? ggml_concat(ctx0, out, c, 3) : c;
            }
            sj_kvarn_pending = {};
            cur = out;
            if (wush_ao) {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], cur->ne[1], cur->ne[2]*cur->ne[3]);
                cur = llm_graph_sj_wush_apply(ctx0, cur, wush_ao);
            } else if (!fused_rot) {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                cur = ggml_turbo_wht(ctx0, cur, 1, group, nullptr);
            }
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            ggml_build_forward_expand(gf, cur);
            return cur;
        }

        if (kq_mask && kq_mask->type == GGML_TYPE_I32 && sj_kvarn_pending.body == nullptr) {
            // packed mask on a plain cache: one bit per K row (SJ-KVaRN is checked in ggml_flash_attn_ext_set_sj_kvarn)
            GGML_ASSERT(kq_mask->ne[0] >= (k->ne[1] + 31)/32);
        }

        if (kv_rows) {
            // kv_rows: src[9] here (upstream uses src[5], taken by SJ-KVaRN)
            cur = ggml_flash_attn_ext_rows(ctx0, q, k, v, kq_mask, kv_rows, kq_scale, hparams.f_max_alibi_bias,
                                           hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        } else {
            cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                      hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        }
        res->add_fused_node({kv_rows ? LLM_FUSED_OP_FLASH_ATTN_KV_ROWS : LLM_FUSED_OP_FLASH_ATTN, cur, il});
        ggml_tensor * fa_node = cur; // the pick below attaches after the SJ-KVaRN record pool (src[6]/src[7])

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (fa_sel_rows != nullptr) {
            GGML_ASSERT(lsa_fa_sel == nullptr || kq_mask != lsa_fa_m3);
            cur->src[8] = fa_sel_rows;
            fa_sel_rows = nullptr;
        }
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        if (lsa_lse_req) { // LLAMA_LSA_COMP=2 (opt-in): row log-sum-exp output (plain and SJ-KVaRN FA)
            lsa_lse = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, q->ne[2], q->ne[1], q->ne[3]);
            ggml_format_name(lsa_lse, "lsa_lse-%d", il);
            ggml_flash_attn_ext_set_lse(cur, lsa_lse);
        }
        lsa_lse_req = false;

        // SJ-KVaRN: attach the sealed record pool + region descriptor; Q/K/V are Hadamard-rotated, so un-rotate the output
        const bool is_sj_kvarn_attn = sj_kvarn_pending.body != nullptr;
        if (is_sj_kvarn_attn) {
            // the attended position count: the mask width, or n_kv for the packed mask (op_params[0] of the input;
            // a row view's op_params hold its offset, so read the source tensor's)
            const ggml_tensor * kq_mask_src = kq_mask && kq_mask->view_src ? kq_mask->view_src : kq_mask;
            // (a packed mask built in the graph, LLAMA_QSA_ROWS_MASK, has no n_kv there: every bit of a row counts)
            const int64_t n_kv_packed = kq_mask && kq_mask->type == GGML_TYPE_I32 && kq_mask_src->op == GGML_OP_NONE ?
                                        kq_mask_src->op_params[0] : 0;
            const int64_t n_kv_pad = kq_mask && kq_mask->type == GGML_TYPE_I32 ?
                                     (n_kv_packed > 0 ? n_kv_packed : kq_mask->ne[0]*32) :
                                     kq_mask ? kq_mask->ne[0] : 0;
            GGML_ASSERT(kq_mask && n_kv_pad > 0 && n_kv_pad % 64 == 0);
            // src[9] may hold the LSA LSE output set above (F32, LLAMA_LSA_COMP=2): only kv_rows (I32) is refused
            GGML_ASSERT(kv_rows == nullptr && ggml_flash_attn_ext_get_kv_rows(cur) == nullptr &&
                    "kv_rows (FA src[9]) cannot be combined with an SJ-KVaRN body (src[5]) on one FA node");
            cur->src[7] = sj_kvarn_pending.selected;
            ggml_flash_attn_ext_set_sj_kvarn(cur, sj_kvarn_pending.body, sj_kvarn_pending.desc,
                    sj_kvarn_pending.bits_k, sj_kvarn_pending.bits_v, (int32_t) n_kv_pad);
            const int group = sj_kvarn_pending.body->op_params[7] == GGML_TYPE_TURBO4_0 ? 128 : 256;
            const bool fused_rot = sj_kvarn_pending.fused_rot;
            ggml_tensor * wush_ao = sj_kvarn_pending.wush_ao;
            sj_kvarn_pending = {};
            if (wush_ao) {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                cur = llm_graph_sj_wush_apply(ctx0, cur, wush_ao);
            } else if (fused_rot) {
                ggml_flash_attn_ext_set_sj_kvarn_rot(cur, 256); // the node un-rotates its output itself
            } else {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                cur = ggml_turbo_wht(ctx0, cur, 1, group, nullptr);
            }
        }

        if (lsa_fa_sel != nullptr && kq_mask == lsa_fa_m3) {
            // after ggml_flash_attn_ext_set_sj_kvarn: VK needs src[6] set and src[7] (QSA list) free
            // LLAMA_LSA_BLKLIST: the mask holds GGML_LSA_MASK_SEL at droppable cells, resolved by the block pick
            GGML_ASSERT(is_sj_kvarn_attn); // sj_kvarn_pending is already cleared here
            if (lsa_fa_vk) {
                // LLAMA_LSA_VK: the pick travels in the virtual-key plane; the dropped blocks' means join the softmax
                GGML_ASSERT(!lsa_lse_req && lsa_vk_km && lsa_vk_vm);
                if (lsa_fa_vk_n > 0) { // unit pick after the mean-block counts
                    ggml_flash_attn_ext_set_lsa_vk_units(fa_node, lsa_fa_sel, lsa_fa_blk, lsa_fa_vk_n, lsa_vk_km, lsa_vk_vm,
                            (float) llama_lsa_cfg().comp_bias);
                } else {
                    ggml_flash_attn_ext_set_lsa_vk(fa_node, lsa_fa_sel, lsa_fa_blk, lsa_vk_km, lsa_vk_vm,
                            (float) llama_lsa_cfg().comp_bias);
                }
                lsa_vk_done = true;
            } else {
                ggml_flash_attn_ext_set_lsa_blocks(fa_node, lsa_fa_sel, lsa_fa_blk);
            }
            lsa_fa_sel = nullptr;
            lsa_fa_m3  = nullptr;
            lsa_fa_vk  = false;
            lsa_fa_vk_n = 0;
        }

        if (lsa_spec_sel) {
            GGML_ASSERT(is_sj_kvarn_attn && kq_mask->type == GGML_TYPE_F16 && q->ne[3] == 1);
            ggml_flash_attn_ext_set_lsa_spec_recall(fa_node, lsa_spec_sel);
            lsa_spec_sel = nullptr;
        }

        // TurboQuant: inverse WHT on FA output when V values are WHT-rotated.
        // For MLA, V is a view of K with different ne[0] (e.g. V=512, K=576).
        // Group size must come from K (which determines the WHT rotation), not V.
        if (!is_sj_kvarn_attn && (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0 || v->type == GGML_TYPE_TQ6_0 || v->type == GGML_TYPE_TQ5_0)) {
            const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0 || k->type == GGML_TYPE_TQ6_0 || k->type == GGML_TYPE_TQ5_0);
            const ggml_tensor * group_src = k_is_turbo ? k : v;
            const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
            if (cur->ne[0] % turbo_group == 0) {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                ggml_tensor * innerq_scale = mctx ? mctx->get_turbo_innerq_scale_inv() : nullptr;
                cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, innerq_scale);  // 1 = inverse
            }
        }

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        ggml_prec_set_acc(kq, GGML_PREC_F32); // llamAmpere: keep v0.6.3 F32 accumulation (upstream BF16)

        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // TurboQuant: inverse WHT on attention output (non-FA path)
        if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0 || v->type == GGML_TYPE_TQ5_0 || v->type == GGML_TYPE_TQ6_0) {
            const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0 || k->type == GGML_TYPE_TQ5_0 || k->type == GGML_TYPE_TQ6_0);
            const ggml_tensor * group_src = k_is_turbo ? k : v;
            const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
            if (kqv->ne[0] % turbo_group == 0) {
                if (!ggml_is_contiguous(kqv)) { kqv = ggml_cont(ctx0, kqv); }
                ggml_tensor * innerq_scale = mctx ? mctx->get_turbo_innerq_scale_inv() : nullptr;
                kqv = ggml_turbo_wht(ctx0, kqv, 1, turbo_group, innerq_scale);
            }
        }

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    // TurboQuant: graph-side inverse WHT on attention output (undoes V rotation)

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the KV values are the tokens of the batch, padded like the KV cache (see llama_kv_cache::get_n_kv)
    const int64_t n_kv = GGML_PAD(n_tokens, 256);

    inp->self_attn_mask.raw = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
    ggml_set_input(inp->self_attn_mask.raw);
    cb(inp->self_attn_mask.raw, "self_kq_mask", -1);

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_attn_mask_swa.raw = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
        ggml_set_input(inp->self_attn_mask_swa.raw);
    } else {
        inp->self_attn_mask_swa.raw = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->self_attn_mask_swa.raw : inp->self_attn_mask.raw;
    const auto & kv_rows = is_swa ? inp->self_attn_mask_swa.rows : inp->self_attn_mask.rows;

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    // the KV values and the KQ bias are padded with zeros up to the KV dimension of the mask
    const int64_t n_kv_pad = kq_mask->ne[0] - k->ne[2];

    k = ggml_pad(ctx0, k, 0, 0, n_kv_pad, 0);
    v = ggml_pad(ctx0, v, 0, 0, n_kv_pad, 0);

    if (kq_b) {
        kq_b = ggml_pad(ctx0, kq_b, n_kv_pad, 0, 0, 0);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur,
    bool is_reserve) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask = build_attn_inp_attn_mask(ctx0, mctx_cur, ubatch, cparams, /*allow_packed=*/true);
    }

    inp->self_k_rot = mctx_cur->get_k_rot();
    inp->self_v_rot = mctx_cur->get_v_rot();

    if (mctx_cur->is_sj_kvarn()) {
        inp->self_sj_kvarn_desc = mctx_cur->build_input_sj_kvarn_desc(ctx0, ubatch);
        if (mctx_cur->has_sj_kvarn_edge_tier()) {
            inp->self_sj_kvarn_desc_edge = mctx_cur->build_input_sj_kvarn_desc(ctx0, ubatch);
        }
        inp->sj_kvarn_n_kv_streams = mctx_cur->get_sj_kvarn_n_kv_streams();
    }

    // LSA probe (llama-lsa.h): one stream only; rotated-K caches stay dense. SJ-KVaRN: key/pos mode with the index
    // cache only (its K is stored rotated and sealed, so the index keys must come from k_cur as it is stored)
    const auto & lsa_cfg = llama_lsa_cfg();
    const bool lsa_sj_ok = !mctx_cur->is_sj_kvarn() || (lsa_cfg.idx_cache && !lsa_cfg.fast &&
        (lsa_cfg.mode == llama_lsa_config::KEY || lsa_cfg.mode == llama_lsa_config::POS));
    // LLAMA_LSA_MULTI=1 (opt-in): SJ-KVaRN keeps one stream per sequence; each stream gets its own LSA side caches
    // (keyed by its sequence id) and, when the ubatch spans several streams, its own LSA input set
    const bool lsa_multi = lsa_cfg.mode != llama_lsa_config::OFF && lsa_cfg.multi && !cparams.training && lsa_sj_ok &&
            mctx_cur->is_sj_kvarn() && !cparams.kv_unified;
    const int64_t lsa_ns = std::max<int64_t>(1, (int64_t) inp->sj_kvarn_n_kv_streams.size());
    // the sequence of stream chunk s, or -1 when the chunk is not one sequence (reserve graphs: the chunk index)
    auto lsa_chunk_seq = [&](int64_t s) -> int32_t {
        if (ubatch.n_tokens == 0 || ubatch.n_tokens % lsa_ns != 0) {
            return -1;
        }
        if (is_reserve || !ubatch.seq_id || !ubatch.n_seq_id) {
            return (int32_t) s;
        }
        const uint32_t T = ubatch.n_tokens/(uint32_t) lsa_ns;
        const uint32_t o = (uint32_t) s*T;
        const llama_seq_id seq = ubatch.seq_id[o][0];
        for (uint32_t i = o; i < o + T; ++i) {
            if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != seq) {
                return -1;
            }
        }
        return llama_lsa_stream_of(seq);
    };
    bool lsa_multi_ok = lsa_multi;
    for (int64_t s = 0; lsa_multi_ok && s < lsa_ns; ++s) {
        lsa_multi_ok = lsa_chunk_seq(s) >= 0;
    }
    if (lsa_multi && !lsa_multi_ok && !is_reserve) {
        static uint32_t warned_multi = 0;
        if (llama_lsa_log_once(warned_multi)) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_MULTI: ubatch of %u rows does not split into %lld one-sequence stream chunks; "
                    "this graph runs dense\n", __func__, ubatch.n_tokens, (long long) lsa_ns);
        }
    }
    if (lsa_multi && !lsa_multi_ok && !is_reserve && ubatch.seq_id && ubatch.n_seq_id) {
        // the rows of this ubatch never reach the side caches: every stream it touches goes stale (guard as mixed)
        std::set<llama_seq_id> seqs;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                seqs.insert(ubatch.seq_id[i][j]);
            }
        }
        for (const llama_seq_id seq : seqs) {
            if (seq >= 0 && seq < LLAMA_LSA_MAX_STREAMS) {
                llama_lsa_stream_scope lsa_scope(seq);
                llama_lsa_seq_guard(seq, false, true, false);
            }
        }
    }
    if (lsa_multi) {
        for (int64_t s = 0; lsa_multi_ok && s < lsa_ns; ++s) {
            // one-owner guard per stream: fires only when a stream's ubatch is not its own sequence
            const int32_t seq = lsa_chunk_seq(s);
            llama_lsa_stream_scope lsa_scope(seq);
            const uint32_t T = ubatch.n_tokens/(uint32_t) lsa_ns;
            llama_lsa_seq_guard(seq, ubatch.pos && ubatch.pos[s*T] == 0, false, is_reserve);
        }
    } else if (lsa_cfg.mode != llama_lsa_config::OFF && !cparams.training && lsa_sj_ok && mctx_cur->is_sj_kvarn()) {
        const int32_t lsa_seq = llama_lsa_ubatch_seq(&ubatch);
        llama_lsa_seq_guard(lsa_seq, ubatch.pos && ubatch.pos[0] == 0, ubatch.n_seqs_unq > 1 && ubatch.seq_id != nullptr, is_reserve);
    }
    // the LSA inputs of one input set: n_kv attended cells, n_tok query rows of ub
    auto lsa_make_inputs = [&](llm_graph_input_attn_kv * x, int64_t n_kv, uint32_t n_tok, const llama_ubatch & ub) {
    x->lsa_graph_gen = llama_lsa_graph_gen();
    x->lsa_pos0      = ub.pos ? ub.pos[0] : 0;
    x->lsa_cell_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, 1);
    ggml_set_input(x->lsa_cell_pos);
    ggml_set_name(x->lsa_cell_pos, "attn_inp_lsa_cell_pos");
    x->lsa_q_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tok);
    ggml_set_input(x->lsa_q_pos);
    ggml_set_name(x->lsa_q_pos, "attn_inp_lsa_q_pos");
    if (x->self_attn_mask.raw && x->self_attn_mask.raw->type == GGML_TYPE_I32) {
        // LLAMA_LSA_PACKED=1 (llama-context keeps the packed mask only then): f16 dense mask for the LSA layers
        x->lsa_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ub, cparams, /*allow_packed=*/false);
        ggml_set_name(x->lsa_kq_mask, "attn_inp_lsa_kq_mask");
    }
    if (mctx_cur->is_sj_kvarn()) {
        x->lsa_q_cell = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok);
        ggml_set_input(x->lsa_q_cell);
        ggml_set_name(x->lsa_q_cell, "attn_inp_lsa_q_cell");
    }
    if (lsa_cfg.block_sum || lsa_cfg.comp_fast) {
        x->lsa_q_blk = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok);
        ggml_set_input(x->lsa_q_blk);
        ggml_set_name(x->lsa_q_blk, "attn_inp_lsa_q_blk");
    }
    if (lsa_cfg.comp_fast) {
        const int64_t B   = lsa_cfg.block > 1 ? lsa_cfg.block : 128;
        const int64_t n_u = (n_tok + B - 1)/B + 1; // most distinct blocks a ubatch of consecutive rows touches
        x->lsa_u_blk = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_u);
        ggml_set_input(x->lsa_u_blk);
        ggml_set_name(x->lsa_u_blk, "attn_inp_lsa_u_blk");
        if (mctx_cur->is_sj_kvarn()) {
            // carry ring (default): a few blocks per layer persist; the ubatch's rows are staged in graph scratch
            // (shared by every layer). Legacy: every row of the ubatch goes through a per-layer ring of
            // n_ubatch + 2 blocks (277 MiB over 16 layers at n_ubatch 4096, 1024-dim K and V, f16)
            x->lsa_ring_rows = !lsa_cfg.ring_legacy;
            x->lsa_ring_n = x->lsa_ring_rows ? lsa_cfg.ring_rows : ((int64_t) (cparams.n_ubatch + B - 1)/B + 1)*B;
            x->lsa_u_rblk = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, x->lsa_ring_rows ? n_u*B : n_u);
            ggml_set_input(x->lsa_u_rblk);
            ggml_set_name(x->lsa_u_rblk, "attn_inp_lsa_u_rblk");
            x->lsa_q_ring = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64,
                    x->lsa_ring_rows ? std::min<int64_t>(n_tok, x->lsa_ring_n) : n_tok);
            ggml_set_input(x->lsa_q_ring);
            ggml_set_name(x->lsa_q_ring, "attn_inp_lsa_q_ring");
        }
    }
    };
    if (lsa_multi_ok && lsa_ns > 1) {
        const uint32_t T = ubatch.n_tokens/(uint32_t) lsa_ns;
        for (int64_t s = 0; s < lsa_ns; ++s) {
            auto x = std::make_shared<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);
            x->self_k_rot = inp->self_k_rot;
            x->self_v_rot = inp->self_v_rot;
            x->lsa_stream = lsa_chunk_seq(s);
            x->lsa_sub_s  = s;
            x->lsa_n_kv_s = inp->sj_kvarn_n_kv_streams[s];
            x->lsa_sub_ub = lsa_sub_ubatch(ubatch, s, lsa_ns, x->lsa_sub_pos);
            lsa_make_inputs(x.get(), x->lsa_n_kv_s, T, x->lsa_sub_ub);
            inp->lsa_streams.push_back(std::move(x));
        }
    } else if (lsa_cfg.mode != llama_lsa_config::OFF && !cparams.training && lsa_sj_ok &&
            (lsa_multi ? lsa_multi_ok : (cparams.kv_unified || ubatch.n_seqs_unq == 1))) {
        // LLAMA_LSA_MULTI: LSA inputs only when every chunk is one in-range sequence (never stream 0's caches for another)
        inp->lsa_stream = lsa_multi_ok ? lsa_chunk_seq(0) : 0;
        lsa_make_inputs(inp.get(), mctx_cur->get_n_kv(), ubatch.n_tokens, ubatch);
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur, is_reserve);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

// SJ-KVaRN fused write rotation (#139, GGML_SJKVARN_FUSED_ROT): when the cache write rotates K/V itself, hand
// cpy_k/cpy_v the unrotated input of the K/V GGML_OP_TURBO_WHT node; the WHT node (and its ggml_cont when the
// raw input already has packed heads) is never expanded into the graph.
// EXPERIMENTAL WUSH-KV (arXiv 2609.38121): y_h = A_{h / (H/Hkv)} x_h for x [D, H, T] and A [D, D, Hkv] (row-major
// per head). The heads go to dim 2 so ggml_mul_mat broadcasts the KV-head matrices over their query-head groups.
static ggml_tensor * llm_graph_sj_wush_apply(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * A) {
    GGML_ASSERT(x->ne[0] == A->ne[0] && x->ne[3] == 1 && x->ne[1] % A->ne[2] == 0);
    ggml_tensor * xp = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // [D, T, H]
    ggml_tensor * y  = ggml_mul_mat(ctx0, A, xp);                         // [D, T, H]
    return ggml_cont(ctx0, ggml_permute(ctx0, y, 0, 2, 1, 3));            // [D, H, T]
}

static ggml_tensor * llm_graph_sj_kvarn_unrotate(ggml_tensor * cur) {
    GGML_ASSERT(cur->op == GGML_OP_TURBO_WHT);
    ggml_tensor * src = cur->src[0];
    if (src->op == GGML_OP_CONT && src->src[0]->nb[0] == sizeof(float) &&
            src->src[0]->nb[1] == ggml_row_size(GGML_TYPE_F32, src->src[0]->ne[0])) {
        src = src->src[0];
    }
    return src;
}

void llm_graph_context::build_attn_store(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
            int       il) const {
    // same rotations, expand order and stores as build_attn, so the cache bytes match it
    if (inp->self_k_rot[il]) {
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot[il]);
    }
    if (inp->self_v_rot[il]) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot[il]);
    }

    const auto * mctx_cur = inp->mctx;

    // FP8 KV: store K/s and V/s as build_attn does (build_attn_scale_kv_inputs; no Q here)
    if (ggml_tensor * scale = mctx_cur->get_k_scale(il)) {
        k_cur = ggml_div(ctx0, k_cur, scale);
    }
    if (ggml_tensor * scale = mctx_cur->get_v_scale(il)) {
        v_cur = ggml_div(ctx0, v_cur, scale);
    }

    if (inp->self_sj_kvarn_desc != nullptr) {
        GGML_ASSERT(k_cur->ne[0] == 256 && v_cur->ne[0] == 256);
        if (!ggml_is_contiguous(k_cur)) { k_cur = ggml_cont(ctx0, k_cur); }
        if (!ggml_is_contiguous(v_cur)) { v_cur = ggml_cont(ctx0, v_cur); }
        const int group = mctx_cur->get_sj_kvarn().body_type == GGML_TYPE_TURBO4_0 ? 128 : 256;
        if (const auto * w = mctx_cur->get_sj_wush(il)) {
            k_cur = llm_graph_sj_wush_apply(ctx0, k_cur, w->ak);
            v_cur = llm_graph_sj_wush_apply(ctx0, v_cur, w->av);
        } else {
            k_cur = ggml_turbo_wht(ctx0, k_cur, 0, group, nullptr);
            v_cur = ggml_turbo_wht(ctx0, v_cur, 0, group, nullptr);
        }
        cb(k_cur, "k_sj_kvarn_rot", il);
        cb(v_cur, "v_sj_kvarn_rot", il);
    }

    if (inp->self_sj_kvarn_desc != nullptr && mctx_cur->sj_kvarn_fused_rot()) {
        k_cur = llm_graph_sj_kvarn_unrotate(k_cur);  // #139: rotated inside the cache write
        v_cur = llm_graph_sj_kvarn_unrotate(v_cur);
    }

    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));
}

// the same rotation build_attn applies below, step for step, so the cache bytes and the FA node match it
bool llm_graph_context::build_attn_sj_kvarn_rot(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *& q_cur,
        ggml_tensor *& k_cur,
        ggml_tensor *& v_cur,
                int   il) const {
    if (inp->self_sj_kvarn_desc == nullptr) {
        return false;
    }
    const auto * mctx_cur = inp->mctx;
    const auto * sj_wush = mctx_cur->get_sj_wush(il); // EXPERIMENTAL WUSH-KV layer
    const bool fused_rot = sj_wush == nullptr && mctx_cur->get_sj_kvarn().body_type != GGML_TYPE_TURBO4_0 &&
        ggml_sj_kvarn_fused_rot_enabled();
    GGML_ASSERT(q_cur->ne[0] == 256 && k_cur->ne[0] == 256 && v_cur->ne[0] == 256);
    if (!fused_rot && !ggml_is_contiguous(q_cur)) { q_cur = ggml_cont(ctx0, q_cur); }
    if (!ggml_is_contiguous(k_cur)) { k_cur = ggml_cont(ctx0, k_cur); }
    if (!ggml_is_contiguous(v_cur)) { v_cur = ggml_cont(ctx0, v_cur); }
    const int group = mctx_cur->get_sj_kvarn().body_type == GGML_TYPE_TURBO4_0 ? 128 : 256;
    if (sj_wush) {
        q_cur = llm_graph_sj_wush_apply(ctx0, q_cur, sj_wush->aq);
        k_cur = llm_graph_sj_wush_apply(ctx0, k_cur, sj_wush->ak);
        v_cur = llm_graph_sj_wush_apply(ctx0, v_cur, sj_wush->av);
        cb(q_cur, "q_sj_kvarn_rot", il);
    } else {
        if (!fused_rot) {
            q_cur = ggml_turbo_wht(ctx0, q_cur, 0, group, nullptr);
            cb(q_cur, "q_sj_kvarn_rot", il);
        }
        k_cur = ggml_turbo_wht(ctx0, k_cur, 0, group, nullptr);
        v_cur = ggml_turbo_wht(ctx0, v_cur, 0, group, nullptr);
    }
    cb(k_cur, "k_sj_kvarn_rot", il);
    cb(v_cur, "v_sj_kvarn_rot", il);

    // fused write rotation (#139): the K/V WHT nodes above are dropped, the cache write rotates
    if (mctx_cur->sj_kvarn_fused_rot()) {
        k_cur = llm_graph_sj_kvarn_unrotate(k_cur);
        v_cur = llm_graph_sj_kvarn_unrotate(v_cur);
    }
    return fused_rot;
}

void llm_graph_context::build_attn_sj_kvarn_pend(
        llm_graph_input_attn_kv * inp,
               bool   fused_rot,
                int   il) const {
    if (inp->self_sj_kvarn_desc == nullptr) {
        return;
    }
    const auto * mctx_cur = inp->mctx;
    const auto * sj_wush = mctx_cur->get_sj_wush(il);
    uint32_t bits_k = 0, bits_v = 0;
    const int tier = mctx_cur->get_sj_kvarn_layer_tier(il, bits_k, bits_v);
    GGML_ASSERT(tier == 0 || inp->self_sj_kvarn_desc_edge != nullptr);
    sj_kvarn_pending.body   = mctx_cur->get_sj_kvarn_body(il);
    sj_kvarn_pending.desc   = tier == 1 ? inp->self_sj_kvarn_desc_edge : inp->self_sj_kvarn_desc;
    sj_kvarn_pending.bits_k = (int32_t) bits_k;
    sj_kvarn_pending.bits_v = (int32_t) bits_v;
    sj_kvarn_pending.fused_rot = fused_rot;
    sj_kvarn_pending.n_kv_streams = &inp->sj_kvarn_n_kv_streams;
    sj_kvarn_pending.wush_ao = sj_wush ? sj_wush->ao : nullptr;
}

static void build_attn_scale_kv_inputs(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        ggml_tensor * & q,
        ggml_tensor * & k,
        ggml_tensor * & v,
        int32_t il) {
    if (ggml_tensor * scale = mctx->get_k_scale(il)) {
        // Store K/s and move its dequant scale to Q: (Q*s) dot (K/s) = Q dot K.
        q = ggml_mul(ctx, q, scale);
        if (k) {
            k = ggml_div(ctx, k, scale);
        }
    }

    if (v) {
        // Store V/s here and restore its dequant scale after the weighted sum.
        ggml_tensor * scale = mctx->get_v_scale(il);
        if (scale) {
            v = ggml_div(ctx, v, scale);
        }
    }
}

static ggml_tensor * build_attn_scale_kv_output(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        ggml_tensor * cur,
        int32_t il) {
    if (ggml_tensor * scale = mctx->get_v_scale(il)) {
        // P*(V/s)*s = P*V.
        cur = ggml_mul(ctx, cur, scale);
    }

    return cur;
}

// plain K types only: the TurboQuant/TQ types hold WHT-rotated K, where the map would see mixed dims
// LLAMA_LSA_STATS: a = [3, T] per-row (kept, valid, sink+window) cell counts of one layer's final mask. CPU custom
// op, so the scheduler copies the tiny tensor over each step (probe only). Logs the running kept share at WARN level
// (llama-server filters libllama INFO) every 256 calls of a layer.
// LLAMA_LSA_DUMP: write one tensor of one layer as raw f32 (once per layer and tensor), plus a line in <prefix>.index.txt
static const char * const lsa_dump_names[] = {"q", "k", "v", "score", "mask", "cell_pos", "q_pos"};
static std::atomic<bool> lsa_dump_done[256][7];

// v0.7 keeps the K/V rotations per layer; LSA keeps its per-cache rule: any rotated layer keeps the whole cache dense
static bool llama_lsa_any_rot(const std::vector<ggml_tensor *> & rot) {
    for (const ggml_tensor * t : rot) {
        if (t) {
            return true;
        }
    }
    return false;
}

// LLAMA_LSA_RESIDENT_STICKY (opt-in): hand a select node the layer's page body (src 7) and SJ-KVaRN desc (src 8),
// the bonus (param 10) and the cells per scored element (param 11); the CUDA select adds the bonus to units whose
// 128-cell record is in a card slot. Off (0 or no residency): the node is unchanged.
static ggml_tensor * lsa_sticky(ggml_tensor * sel, ggml_tensor * body, ggml_tensor * desc, int cells_per_elem) {
    const auto & cfg = llama_lsa_cfg();
    if (sel == nullptr || cfg.resident_sticky == 0.0 || cfg.resident <= 0.0 || body == nullptr || desc == nullptr) {
        return sel;
    }
    const float b = (float) cfg.resident_sticky;
    const int32_t c = cells_per_elem;
    memcpy((char *) sel->op_params + 10*sizeof(int32_t), &b, sizeof(b));
    memcpy((char *) sel->op_params + 11*sizeof(int32_t), &c, sizeof(c));
    sel->src[7] = body;
    sel->src[8] = desc;
    return sel;
}

// LLAMA_LSA_SLICE (opt-in): slice-local quota pick on a select node (op param 13 = units per slice; both refine stages)
static ggml_tensor * lsa_slice(ggml_tensor * sel, int64_t cells_per_unit) {
    const auto & cfg = llama_lsa_cfg();
    if (sel == nullptr || cfg.slice <= 0) {
        return sel;
    }
    const int32_t u = (int32_t) std::max<int64_t>(1, cfg.slice/std::max<int64_t>(1, cells_per_unit));
    memcpy((char *) sel->op_params + 13*sizeof(int32_t), &u, sizeof(u));
    return sel;
}

static void lsa_dump_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
    const int il = (int) ((intptr_t) userdata >> 4), w = (int) ((intptr_t) userdata & 15);
    if (il < 0 || il >= 256 || w >= 7 || a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) ||
            lsa_dump_done[il][w].exchange(true)) {
        return;
    }
    const std::string & pre = llama_lsa_cfg().dump;
    const std::string path = pre + ".L" + std::to_string(il) + "." + lsa_dump_names[w] + ".f32";
    FILE * f = fopen(path.c_str(), "wb");
    const bool ok = f && fwrite(a->data, 1, ggml_nbytes(a), f) == ggml_nbytes(a);
    if (f) {
        fclose(f);
    }
    if (FILE * fi = fopen((pre + ".index.txt").c_str(), "a")) {
        fprintf(fi, "%d %s %lld %lld %lld %lld %s\n", il, lsa_dump_names[w], (long long) a->ne[0], (long long) a->ne[1],
                (long long) a->ne[2], (long long) a->ne[3], ok ? "ok" : "FAIL");
        fclose(fi);
    }
    LLAMA_LOG_WARN("lsa-dump: layer %d %s [%lld, %lld] -> %s%s\n", il, lsa_dump_names[w], (long long) a->ne[0],
            (long long) a->ne[1], path.c_str(), ok ? "" : " (write FAILED)");
}

// LLAMA_LSA_SCORE_LOG: a = [2 + 2*nu] F32 = (row 0 position, rows T, the max over rows of each unit's logsumexp score,
// then 0/1 per unit: holds a middle cell of row 0 = visible, past the sinks, outside the window). Appends one record per
// call to <prefix>.L<il>.f16: int32 {pos0, nu, F, T}, nu f16 scores (non-finite clamp to +-65504), then the middle flags
// as ceil(nu/8) bytes (bit i%8 of byte i/8). Diagnostic only; the graph output is a copy of a.
static void lsa_score_log_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
    const int il = (int) ((intptr_t) userdata & 255), F = (int) ((intptr_t) userdata >> 8);
    if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || ggml_nelements(a) < 3) {
        return;
    }
    const float * r  = (const float *) a->data;
    const int32_t nu = (int32_t) ((ggml_nelements(a) - 2)/2);
    std::vector<ggml_fp16_t> h(nu);
    std::vector<uint8_t>     m((nu + 7)/8, 0);
    for (int32_t i = 0; i < nu; ++i) {
        const float v = r[2 + i];
        h[i] = ggml_fp32_to_fp16(std::isfinite(v) ? std::max(-65504.0f, std::min(65504.0f, v)) : (v > 0 ? 65504.0f : -65504.0f));
        if (r[2 + nu + i] > 0.5f) {
            m[i/8] |= (uint8_t) (1u << (i%8));
        }
    }
    const int32_t hdr[4] = { (int32_t) r[0], nu, F, (int32_t) r[1] };
    const std::string path = llama_lsa_cfg().score_log + ".L" + std::to_string(il) + ".f16";
    FILE * f = fopen(path.c_str(), "ab");
    const bool ok = f && fwrite(hdr, sizeof(hdr), 1, f) == 1 && fwrite(h.data(), sizeof(ggml_fp16_t), nu, f) == (size_t) nu &&
        fwrite(m.data(), 1, m.size(), f) == m.size();
    if (f) {
        fclose(f);
    }
    static std::atomic<int> logged{0};
    if (!ok || logged.fetch_add(1) == 0) {
        LLAMA_LOG_WARN("lsa-score-log: layer %d pos %d units %d of %d rows %d -> %s%s\n", il, hdr[0], nu, F, hdr[3],
                path.c_str(), ok ? "" : " (write FAILED)");
    }
}

// LLAMA_LSA_PICK_CHECK: a = [3] over row 0's middle units (outside the window, past the sinks, visible) =
// (fresh AND reused, fresh, reused). Logs per layer every 64 reuse graphs: share of the fresh pick the reused one
// holds (recall), and reused/fresh size. Low recall = the stored pick is not the pick it should be.
static void lsa_pick_check_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    // userdata = layer | age << 8 (age = graphs since the pick graph, capped at 15): recall by age separates a wrong
    // stored pick (low at every age, age 1 included) from staleness (high at age 1, falling with age).
    // a = [3*NL]: (fresh AND reused, fresh, reused) at the pick's own unit, then the same with both picks widened to
    // 16-, 64- and 128-cell blocks (a block counts as kept if any of its units is): does the same region stay kept
    // even when the exact units move inside it
    constexpr int NL = 4;
    static const char * lvl_name[NL] = { "unit", "b16", "b64", "b128" };
    static double acc[256][16][NL][3];
    static int64_t calls[256];
    const int il  = (int) ((intptr_t) userdata & 255);
    const int age = (int) std::min<intptr_t>(15, (intptr_t) userdata >> 8);
    if (a->type != GGML_TYPE_F32 || ggml_nelements(a) != 3*NL) {
        return;
    }
    const float * r = (const float *) a->data;
    for (int l = 0; l < NL; ++l) {
        for (int j = 0; j < 3; ++j) {
            acc[il][0][l][j]   += r[3*l + j];
            acc[il][age][l][j] += r[3*l + j];
        }
    }
    if (++calls[il] % 64 == 0) {
        std::string by_age;
        char buf[48];
        for (int l = 0; l < NL; ++l) {
            by_age += l == 0 ? "" : " |";
            by_age += " ";
            by_age += lvl_name[l];
            for (int k = 1; k < 16; ++k) {
                if (acc[il][k][l][1] > 0) {
                    snprintf(buf, sizeof(buf), " a%d %.1f%%", k, 100.0*acc[il][k][l][0]/acc[il][k][l][1]);
                    by_age += buf;
                }
            }
        }
        LLAMA_LOG_WARN("lsa-pick-check: layer %d reuse graphs %lld recall %.2f%% reused/fresh %.3f (fresh %.1f units/graph) by age:%s\n",
                il, (long long) calls[il], 100.0*acc[il][0][0][0]/std::max(acc[il][0][0][1], 1.0),
                acc[il][0][0][2]/std::max(acc[il][0][0][1], 1.0), acc[il][0][0][1]/(double) calls[il], by_age.c_str());
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

static void lsa_stats_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    static double acc[256][3];
    static int64_t calls[256];
    const int il = (int) (intptr_t) userdata;
    if (il < 0 || il >= 256 || a->type != GGML_TYPE_F32 || a->ne[0] != 3) {
        return;
    }
    for (int64_t t = 0; t < a->ne[1]; ++t) {
        const float * r = (const float *) ((const char *) a->data + t*a->nb[1]);
        for (int j = 0; j < 3; ++j) {
            acc[il][j] += r[j];
        }
    }
    if (++calls[il] % 256 == 0) {
        const double mid_kept = acc[il][0] - acc[il][2], mid_all = acc[il][1] - acc[il][2];
        LLAMA_LOG_WARN("lsa-stats: layer %d calls %lld kept %.2f%% of cells, %.2f%% of middle\n", il, (long long) calls[il],
                100.0*acc[il][0]/std::max(acc[il][1], 1.0), 100.0*mid_kept/std::max(mid_all, 1.0));
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

// LLAMA_LSA_STATS: middle 16-cell strips / 128-cell blocks (= SJ-KVaRN body strips / records) that some row of the
// ubatch keeps, out of those holding any visible middle cell: what the SJ decode kernel's strip / record skip can drop
static void lsa_stats_blk_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    static double acc[256][4];
    static int64_t calls[256];
    const int il = (int) (intptr_t) userdata;
    if (il < 0 || il >= 256 || a->type != GGML_TYPE_F32 || ggml_nelements(a) != 4) {
        return;
    }
    const float * r = (const float *) a->data;
    for (int j = 0; j < 4; ++j) {
        acc[il][j] += r[j];
    }
    if (++calls[il] % 256 == 0) {
        LLAMA_LOG_WARN("lsa-stats: layer %d ubatches %lld live middle strips %.2f%%, live middle records %.2f%%\n", il,
                (long long) calls[il], 100.0*acc[il][0]/std::max(acc[il][1], 1.0), 100.0*acc[il][2]/std::max(acc[il][3], 1.0));
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

// LLAMA_LSA_STATS, block pick: a = [3] = (picked blocks summed over rows, blocks picked by any row, rows). Logs the
// running mean per row, the union and their ratio per layer every 256 ubatches.
static void lsa_stats_union_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    static double acc[256][3];
    static int64_t calls[256];
    const int il = (int) (intptr_t) userdata;
    if (il < 0 || il >= 256 || a->type != GGML_TYPE_F32 || ggml_nelements(a) != 3) {
        return;
    }
    const float * r = (const float *) a->data;
    for (int j = 0; j < 3; ++j) {
        acc[il][j] += r[j];
    }
    if (++calls[il] % 256 == 0) {
        const double per_row = acc[il][0]/std::max(acc[il][2], 1.0), uni = acc[il][1]/(double) calls[il];
        LLAMA_LOG_WARN("lsa-stats: layer %d ubatches %lld rows/ubatch %.2f picked blocks/row %.1f union %.1f (x%.2f)\n", il,
                (long long) calls[il], acc[il][2]/(double) calls[il], per_row, uni, uni/std::max(per_row, 1e-9));
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

// LLAMA_LSA_RECALL: a = [3] = (sum over rows x q heads of the true attention probability on the kept cells,
// rows x heads whose missed probability is > 10%, rows x heads). Logs the running kept mass per layer every 256 calls.
static void lsa_recall_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    static double acc[256][3];
    static int64_t calls[256];
    const int il = (int) (intptr_t) userdata;
    if (il < 0 || il >= 256 || a->type != GGML_TYPE_F32 || ggml_nelements(a) != 3) {
        return;
    }
    const float * r = (const float *) a->data;
    for (int j = 0; j < 3; ++j) {
        acc[il][j] += r[j];
    }
    if (++calls[il] % 256 == 0) {
        const double n = std::max(acc[il][2], 1.0);
        LLAMA_LOG_WARN("lsa-recall: layer %d calls %lld kept attention mass %.4f%% missed %.4f%% rows>10%%missed %.2f%% n %.0f\n",
                il, (long long) calls[il], 100.0*acc[il][0]/n, 100.0*(1.0 - acc[il][0]/n), 100.0*acc[il][1]/n, acc[il][2]);
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

static bool lsa_k_plain(ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_F32 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_0 ||
           t == GGML_TYPE_Q5_1 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_IQ4_NL;
}

// LSA probe (llama-lsa.h). Scores every KV cell for each query row in the 64-dim index space, adds a large bonus to
// the first `sinks` positions and the last `window` positions, adds the causal mask, keeps the top
// sinks + window + keep cells, and builds the replacement mask the way the DSA path does (fill -inf, set the kept
// cells to 0, add the original mask). The layer stays dense when it has no map, when the cache is not plain
// single-stream f16/bf16/q8_0-style K, when the ubatch is wider than max_rows or starts before min_pos, or when
// n_kv is not larger than the kept count.
// A shared pick (LLAMA_LSA_SHARED, LLAMA_LSA_FAST) ranks blocks by all rows of the ubatch together. Rows of different
// sequences (several server slots in one ubatch) see different cells, so one sequence's rows would spend the keep
// budget on cells the other sequence's rows cannot see. Such ubatches pick per row; one-sequence ubatches are unchanged.
static bool lsa_one_seq(const llama_ubatch & ub, int il = -1) {
    if (ub.n_seqs_unq <= 1) {
        return true;
    }
    static bool warned = false;
    if (!warned && il >= 0) {
        warned = true;
        LLAMA_LOG_WARN("%s: LSA ubatch holds %u sequences: shared pick off for it, every row picks its own blocks\n",
            __func__, (unsigned) ub.n_seqs_unq);
    }
    return false;
}

ggml_tensor * llm_graph_context::build_attn_lsa_mask(
        const llm_graph_input_attn_kv * inp,
        ggml_tensor * q,
        ggml_tensor ** k_io,
        ggml_tensor ** v_io,
        ggml_tensor * kq_mask,
                int   il,
            int64_t * n_kv_max,
        ggml_tensor * idx,
              float   kq_scale_lsa,
        std::vector<ggml_tensor *> * masks_h) const {
    const auto & cfg = llama_lsa_cfg();
    if (cfg.mode == llama_lsa_config::OFF || !cfg.layer_enabled(il)) {
        return nullptr;
    }
    const bool cfg_shared = cfg.shared && lsa_one_seq(lsa_ubatch(), il); // rows of other sequences never share a pick
    const bool cfg_union  = cfg.shared_union && lsa_one_seq(lsa_ubatch(), il); // SHARED=3: per-row picks, OR-ed (block/unit pick)
    ggml_tensor * k = *k_io;

    // SJ-KVaRN (inp->lsa_q_cell set): k is the sink + ring view, cells == positions; n_kv comes from the mask and
    // scoring must use the index cache
    const bool sj = inp->lsa_q_cell != nullptr;
    if (sj && (idx == nullptr || cfg.fast || masks_h != nullptr ||
            (cfg.mode != llama_lsa_config::KEY && cfg.mode != llama_lsa_config::POS))) {
        return nullptr;
    }
    const int64_t n_kv = sj ? kq_mask->ne[0] : k->ne[2];
    const int64_t T    = q->ne[2];
    if (sj && (inp->lsa_cell_pos->ne[0] != n_kv || idx->ne[1] != n_kv)) {
        return nullptr;
    }

    if (k->ne[3] != 1 || q->ne[3] != 1 || kq_mask->ne[3] != 1) {
        return nullptr;
    }
    if (kq_mask->type != GGML_TYPE_F16 && kq_mask->type != GGML_TYPE_F32) {
        return nullptr; // packed mask: llama-context turns it off when LSA is on
    }
    if (kq_mask->ne[0] != n_kv || kq_mask->ne[1] != T || T > cfg.max_rows) {
        return nullptr;
    }
    if (lsa_ubatch().pos && cfg.min_pos > 0) {
        // rows before min_pos keep the full mask (handled per row below); skip the work if no row is past it
        bool any = false;
        for (int64_t i = 0; i < T; ++i) {
            any = any || lsa_ubatch().pos[i] >= cfg.min_pos;
        }
        if (!any) {
            return nullptr;
        }
    }

    // LLAMA_LSA_KEEP_SCHED (opt-in): the fraction follows the context length; below the first point the layer stays dense
    const double  sched   = cfg.keep_frac_sched(n_kv);
    if (sched == 0.0) {
        return nullptr;
    }
    const double  frac    = sched > 0.0 ? cfg.keep_frac_for(il, sched) : cfg.keep_frac_for(il);
    if (frac >= 1.0) {
        return nullptr; // per-layer override: this layer stays dense
    }
    int64_t keep_lr = frac > 0.0 ? (int64_t) (frac * (double) n_kv) : cfg.keep;
    // LLAMA_LSA_RESIDENT_LOWER_KEEP (opt-in): the resident sizing capped keep so one row's reads fit the card slots
    if (const double cf = llama_lsa_keep_cap_frac(); cf > 0.0 && frac > 0.0 && frac > cf) {
        keep_lr = (int64_t) (cf * (double) n_kv);
    } else if (const int64_t cc = llama_lsa_keep_cap_cells(); cc > 0 && frac <= 0.0) {
        keep_lr = std::min(keep_lr, cc);
    }
    const int64_t k_total = cfg.sinks + cfg.window + std::max<int64_t>(keep_lr, 0);
    if (n_kv <= k_total) {
        return nullptr;
    }

    // K must be a plain cache view: heads of one cell packed back to back
    const int64_t n_head    = q->ne[1];
    const int64_t n_head_kv = k->ne[1];
    const int64_t d_head    = k->ne[0];
    const int64_t d_kv      = d_head*n_head_kv;
    // plain K types only: the TurboQuant/TQ types hold WHT-rotated K, where the map would see mixed dims
    const bool k_plain = lsa_k_plain(k->type);
    if (!sj && (!k_plain || k->nb[2] != k->nb[1]*n_head_kv)) {
        return nullptr;
    }
    if (q->ne[0] != d_head || n_head % n_head_kv != 0) {
        return nullptr;
    }

    const ggml_tensor * kbase = k->view_src ? k->view_src : k;
    lsa_pick_kbase = nullptr;
    if (!kbase->buffer) {
        return nullptr;
    }
    lsa_pick_kbase = kbase; // LLAMA_LSA_PICK_EVERY key; build_attn_lsa_mask_sb takes it
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    if (!cfg.mapless_probe()) {
        if (!llama_lsa_get_map(il, ggml_backend_buffer_get_type(kbase->buffer), &wq, &wk)) {
            return nullptr;
        }
        if (wk->ne[0] != d_kv) {
            static uint32_t warned_ep = 0;
            if (llama_lsa_log_once(warned_ep)) {
                LLAMA_LOG_WARN("%s: LSA map width %lld != n_head_kv*head_dim %lld, LSA stays off\n", __func__,
                        (long long) wk->ne[0], (long long) d_kv);
            }
            return nullptr;
        }
    }
    const int64_t d_idx = wk ? wk->ne[1] : 0;

    // index query: sum the q heads of each GQA group (q head h reads KV head h / g) -> [d_kv, T]
    const int64_t g = n_head/n_head_kv;
    ggml_tensor * qc = ggml_is_contiguous(q) ? q : ggml_cont(ctx0, q);
    ggml_tensor * q4 = ggml_reshape_4d(ctx0, qc, d_head, g, n_head_kv, T);
    q4 = ggml_cont(ctx0, ggml_permute(ctx0, q4, 1, 0, 2, 3)); // [g, d_head, n_head_kv, T]
    ggml_tensor * qs = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, q4), d_kv, T);
    if (qs->type != GGML_TYPE_F32) {
        qs = ggml_cast(ctx0, qs, GGML_TYPE_F32);
    }

    // all heads of a cell as one row: [d_kv, n_kv]
    ggml_tensor * kv2 = sj ? nullptr : ggml_view_2d(ctx0, k, d_kv, n_kv, k->nb[2], 0); // SJ-KVaRN: index cache only

    // SJ-KVaRN per-KV-head pick (LLAMA_LSA_PER_HEAD=1, key mode, per-head index cache): one mask per KV head,
    // repeated over its q heads (mask ne[2] == n_head). Only the 4/4 stream decode kernel reads a per-head mask, so
    // wider ubatches (and other SJ widths) rank by the sum of the head scores, one shared mask.
    static const int64_t sj_stream_max = std::min<int64_t>(8, getenv("GGML_SJKVARN_DIRECT_STREAM_MAX") ?
            atoi(getenv("GGML_SJKVARN_DIRECT_STREAM_MAX")) : 8);
    const bool sj_ph   = sj && cfg.per_head && cfg.mode == llama_lsa_config::KEY && idx->ne[0] == d_idx &&
        idx->ne[2] == n_head_kv;
    const bool sj_ph_m = sj_ph && T <= sj_stream_max && kq_mask->type == GGML_TYPE_F16 &&
        sj_kvarn_pending.bits_k == 4 && sj_kvarn_pending.bits_v == 4 && !cfg.sj_compact;
    if (sj && !sj_ph && (idx->ne[0] != d_idx || idx->ne[2] != 1)) {
        return nullptr;
    }

    if (!sj_ph && cfg.block_sum && cfg.mode == llama_lsa_config::KEY && !cfg.fast && !(cfg.per_head && masks_h) &&
            inp->lsa_smax_cur && inp->lsa_sum_il == il) {
        if (ggml_tensor * mb = build_attn_lsa_mask_blk(inp, ggml_mul_mat(ctx0, wq, qs), kq_mask, n_kv, keep_lr, il)) {
            return mb;
        }
    }

    ggml_tensor * S = nullptr; // [n_kv, T]
    std::vector<ggml_tensor *> S_heads; // per-KV-head scores (LLAMA_LSA_PER_HEAD=1, oracle/mass only)
    auto dump_dense = [&]() {
        if (cfg.dump.empty()) {
            return false;
        }
        // Dump the first qualifying decode graph per layer. This run's attention stays dense.
        ggml_tensor * v = *v_io;
        const bool v_ok = v && v->ne[0] == d_head && v->ne[1] == n_head_kv && v->ne[2] == n_kv &&
            v->nb[2] == v->nb[1]*n_head_kv;
        bool need = false;
        for (int w = 0; w < 7 && il >= 0 && il < 256; ++w) {
            need = need || !lsa_dump_done[il][w].load();
        }
        if (need && kv2 && v_ok && T <= 8 && n_kv >= cfg.dump_min_kv) {
            ggml_tensor * mk32 = ggml_cont(ctx0, ggml_reshape_2d(ctx0, kq_mask->type == GGML_TYPE_F32 ? kq_mask :
                    ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T));
            ggml_tensor * vv = ggml_view_2d(ctx0, v, d_kv, n_kv, v->nb[2], 0);
            ggml_tensor * parts[7] = {
                ggml_cont(ctx0, q->type == GGML_TYPE_F32 ? q : ggml_cast(ctx0, q, GGML_TYPE_F32)),
                ggml_cpy(ctx0, kv2, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_kv, n_kv)),
                ggml_cpy(ctx0, vv, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_kv, n_kv)),
                ggml_cont(ctx0, S), mk32, ggml_cont(ctx0, inp->lsa_cell_pos), ggml_cont(ctx0, inp->lsa_q_pos),
            };
            for (int w = 0; w < 7; ++w) {
                ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, parts[w], lsa_dump_op, 1,
                        (void *) (intptr_t) ((il << 4) | w)));
            }
        }
        return true;
    };
    const bool per_head = cfg.per_head && masks_h && !cfg.fast && !sj &&
        (cfg.mode == llama_lsa_config::ORACLE || cfg.mode == llama_lsa_config::MASS || cfg.mode == llama_lsa_config::KEY ||
         cfg.mode == llama_lsa_config::QUEST);
    if (sj_ph) {
        // query_j = rms_norm(Wq[j]^T qs_j), score_j = key_j . query_j, batched over the KV heads
        ggml_tensor * wq3 = ggml_cont(ctx0, ggml_view_3d(ctx0, wq, d_head, d_idx, n_head_kv, wq->nb[1], d_head*wq->nb[0], 0));
        ggml_tensor * qs3 = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, qs, d_head, n_head_kv, T), 0, 2, 1, 3));
        ggml_tensor * qi3 = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, wq3, qs3), 1e-12f);                    // [d_idx, T, H]
        // LLAMA_LSA_BLOCK_SUM + per-head masks: each KV head ranks the blocks by its own summary plane (no per-cell
        // scores); wider ubatches keep the per-cell summed-head path below
        if (sj_ph_m && cfg.block_sum && inp->lsa_sum_il == il && (int64_t) inp->lsa_smax_h.size() == n_head_kv) {
            ggml_tensor * m4 = nullptr;
            for (int64_t j = 0; j < n_head_kv && (j == 0 || m4); ++j) {
                ggml_tensor * qj = ggml_cont(ctx0, ggml_view_2d(ctx0, qi3, d_idx, T, qi3->nb[1], j*qi3->nb[2]));
                ggml_tensor * mh = build_attn_lsa_mask_blk(inp, qj, kq_mask, n_kv, keep_lr, il,
                        inp->lsa_smax_h[j], inp->lsa_smin_h[j], true);
                m4 = mh ? (m4 ? ggml_concat(ctx0, m4, mh, 3) : mh) : nullptr;
            }
            if (m4) {
                m4 = ggml_repeat_4d(ctx0, m4, n_kv, T, g, n_head_kv);
                ggml_tensor * m = ggml_cast(ctx0, ggml_reshape_4d(ctx0, m4, n_kv, T, n_head, 1), kq_mask->type);
                cb(m, "lsa_kq_mask_blk_heads", il);
                static uint32_t logged_ep = 0;
                if (llama_lsa_log_once(logged_ep)) {
                    LLAMA_LOG_WARN("%s: LSA per-KV-head block-sum pick active (layer %d, T %lld, %lld heads)\n", __func__,
                            il, (long long) T, (long long) n_head_kv);
                }
                return m;
            }
        }
        // head-major index cache view [d_idx, n_kv, H]: each head's keys are read in place (no copy of the cache)
        ggml_tensor * S3  = ggml_mul_mat(ctx0, idx, qi3);                                                 // [n_kv, T, H]
        for (int64_t j = 0; j < n_head_kv; ++j) {
            S_heads.push_back(ggml_view_2d(ctx0, S3, n_kv, T, S3->nb[1], j*S3->nb[2]));
        }
        if (!sj_ph_m) {
            for (ggml_tensor * Sh : S_heads) {
                S = S ? ggml_add(ctx0, S, Sh) : ggml_cont(ctx0, Sh);
            }
            S_heads.clear();
        } else {
            S = S_heads.front();
        }
    } else if (cfg.mode == llama_lsa_config::QUEST) {
        // training-free real-key block bound (probe, plain K): per KV head j, blocks of B cells keep the channel max
        // and min of their keys; bound_j(b, t) = sum_c Q+_jc(t) max_jbc + Q-_jc(t) min_jbc, where Q+/Q- sum the
        // positive / negative parts of the g q heads of group j (= the sum of the heads' own bounds, so a block whose
        // keys might score high for any head ranks high). Only whole blocks get a bound; the tail block (and anything
        // past n_kv) is in the window. Cell score = its block's bound (cells of a block tie).
        const int64_t B  = std::max<int64_t>(1, cfg.bucket);
        const int64_t nb = n_kv/B;
        if (nb < 2) {
            return nullptr;
        }
        ggml_tensor * qf = q->type == GGML_TYPE_F32 ? q : ggml_cast(ctx0, q, GGML_TYPE_F32);
        qf = ggml_is_contiguous(qf) ? qf : ggml_cont(ctx0, qf);
        ggml_tensor * qpos = ggml_relu(ctx0, qf);
        ggml_tensor * qneg = ggml_sub(ctx0, qf, qpos);
        auto group_sum = [&](ggml_tensor * x) -> ggml_tensor * { // [d_head, n_head, T] -> [d_head, T, n_head_kv]
            ggml_tensor * x4 = ggml_reshape_4d(ctx0, x, d_head, g, n_head_kv, T);
            x4 = ggml_cont(ctx0, ggml_permute(ctx0, x4, 1, 0, 3, 2)); // [g, d_head, T, n_head_kv]
            return ggml_reshape_3d(ctx0, ggml_sum_rows(ctx0, x4), d_head, T, n_head_kv);
        };
        ggml_tensor * QP = group_sum(qpos);
        ggml_tensor * QN = group_sum(qneg);
        for (int64_t j = 0; j < n_head_kv; ++j) {
            // head j's keys of the whole blocks, dequantized row by row: [d_head, B*nb]
            ggml_tensor * kj = ggml_view_2d(ctx0, k, d_head, nb*B, k->nb[2], j*k->nb[1]);
            kj = ggml_cpy(ctx0, kj, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head, nb*B));
            ggml_tensor * x  = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, kj, d_head, B, nb), 1, 0, 2, 3)); // [B, d, nb]
            ggml_tensor * mx = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, (int) B, (int) B, 0), d_head, nb);
            ggml_tensor * mn = ggml_neg(ctx0, ggml_reshape_2d(ctx0,
                    ggml_pool_1d(ctx0, ggml_neg(ctx0, x), GGML_OP_POOL_MAX, (int) B, (int) B, 0), d_head, nb));
            ggml_tensor * qpj = ggml_view_2d(ctx0, QP, d_head, T, QP->nb[1], j*QP->nb[2]);
            ggml_tensor * qnj = ggml_view_2d(ctx0, QN, d_head, T, QN->nb[1], j*QN->nb[2]);
            ggml_tensor * Sb  = ggml_add(ctx0, ggml_mul_mat(ctx0, mx, qpj), ggml_mul_mat(ctx0, mn, qnj));    // [nb, T]
            Sb = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, Sb, 1, nb, T), B, nb, T, 1), B*nb, T);
            if (B*nb < n_kv) {
                Sb = ggml_pad(ctx0, Sb, (int) (n_kv - B*nb), 0, 0, 0);
            }
            if (per_head) {
                S_heads.push_back(Sb);
            } else {
                S = S ? ggml_add(ctx0, S, Sb) : Sb;
            }
        }
        if (per_head) {
            S = S_heads.front();
        }
        static uint32_t logged_ep = 0;
        if (llama_lsa_log_once(logged_ep)) {
            LLAMA_LOG_WARN("%s: LSA quest real-key block bound active (layer %d, n_kv %lld, block %lld, per_head %d)\n",
                    __func__, il, (long long) n_kv, (long long) B, per_head ? 1 : 0);
        }
    } else if (cfg.mode == llama_lsa_config::MASS) {
        // diagnostic oracle: score = attention probability summed over the heads that share the pick (all heads,
        // or the g q heads of one KV head). Head j's q is zero-padded into its slot of the [d_kv] cell row, so K
        // is read as the contiguous all-heads view.
        ggml_tensor * mask2 = ggml_reshape_2d(ctx0,
                kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T);
        ggml_tensor * qf = q->type == GGML_TYPE_F32 ? q : ggml_cast(ctx0, q, GGML_TYPE_F32);
        for (int64_t j = 0; j < n_head_kv; ++j) {
            ggml_tensor * qg = ggml_view_3d(ctx0, qf, d_head, g, T, qf->nb[1], qf->nb[2], j*g*qf->nb[1]);
            qg = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, qg, 0, 2, 1, 3)), d_head, T*g); // [d, T*g]
            qg = ggml_pad_ext(ctx0, qg, (int) (j*d_head), (int) ((n_head_kv - 1 - j)*d_head), 0, 0, 0, 0, 0, 0);
            ggml_tensor * P = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, kv2, qg), n_kv, T, g);
            P = ggml_soft_max_ext(ctx0, P, mask2, kq_scale_lsa, 0.0f);
            ggml_tensor * Sj = nullptr;
            for (int64_t r = 0; r < g; ++r) {
                ggml_tensor * pr = ggml_view_2d(ctx0, P, n_kv, T, P->nb[1], r*P->nb[2]);
                Sj = Sj ? ggml_add(ctx0, Sj, pr) : ggml_cont(ctx0, pr);
            }
            if (per_head) {
                S_heads.push_back(Sj);
            } else {
                S = S ? ggml_add(ctx0, S, Sj) : Sj;
            }
        }
        if (per_head) {
            S = S_heads.front();
        }
    } else if (cfg.mode == llama_lsa_config::ORACLE && per_head) {
        // exact head-summed q.k, but each KV head ranks its own cells
        for (int64_t j = 0; j < n_head_kv; ++j) {
            ggml_tensor * qj = ggml_cont(ctx0, ggml_view_2d(ctx0, qs, d_head, T, qs->nb[1], j*d_head*qs->nb[0]));
            qj = ggml_pad_ext(ctx0, qj, (int) (j*d_head), (int) ((n_head_kv - 1 - j)*d_head), 0, 0, 0, 0, 0, 0);
            S_heads.push_back(ggml_mul_mat(ctx0, kv2, qj));
        }
        S = S_heads.front();
    } else if (cfg.mode == llama_lsa_config::ORACLE) {
        S = ggml_mul_mat(ctx0, kv2, qs);
    } else if (cfg.mode == llama_lsa_config::KEY && per_head) {
        // training-free per-KV-head index from the shared map: head j uses only its own slice of the map rows
        // (block-diagonal approximation of Wk / Wq). key_j = norm(k_j @ Wk[j]), query_j = norm(qs_j @ Wq[j]),
        // score_j = key_j . query_j. Reads K once per head (probe only; the index cache stays shared).
        ggml_tensor * wkf = wk->type == GGML_TYPE_F32 ? wk : ggml_cast(ctx0, wk, GGML_TYPE_F32);
        ggml_tensor * wqf = wq->type == GGML_TYPE_F32 ? wq : ggml_cast(ctx0, wq, GGML_TYPE_F32);
        for (int64_t j = 0; j < n_head_kv; ++j) {
            const int lo = (int) (j*d_head);
            const int hi = (int) ((n_head_kv - 1 - j)*d_head);
            ggml_tensor * wkj = ggml_cont(ctx0, ggml_view_2d(ctx0, wkf, d_head, d_idx, wkf->nb[1], j*d_head*wkf->nb[0]));
            ggml_tensor * wqj = ggml_cont(ctx0, ggml_view_2d(ctx0, wqf, d_head, d_idx, wqf->nb[1], j*d_head*wqf->nb[0]));
            wkj = ggml_pad_ext(ctx0, wkj, lo, hi, 0, 0, 0, 0, 0, 0);                         // [d_kv, d_idx], other heads 0
            ggml_tensor * qj = ggml_cont(ctx0, ggml_view_2d(ctx0, qs, d_head, T, qs->nb[1], j*d_head*qs->nb[0]));
            ggml_tensor * Pj = ggml_mul_mat(ctx0, kv2, wkj);                                  // [n_kv, d_idx]
            ggml_tensor * PTj = ggml_rms_norm(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, Pj)), 1e-12f);
            ggml_tensor * qij = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, wqj, qj), 1e-12f);     // [d_idx, T]
            S_heads.push_back(ggml_mul_mat(ctx0, PTj, qij));
        }
        S = S_heads.front();
    } else {
        ggml_tensor * PT = nullptr; // [d_idx, n_kv] normalized index keys
        if (idx && idx->ne[0] == wk->ne[1] && idx->ne[1] == n_kv && idx->ne[2] == 1) {
            // M1a: cached index keys (F16), 128 B per cell instead of reading all of K
            PT = cfg.mode == llama_lsa_config::POS ? ggml_cast(ctx0, idx, GGML_TYPE_F32) : idx;
        } else {
            ggml_tensor * P = ggml_mul_mat(ctx0, kv2, wk);                                 // [n_kv, d_idx]
            PT = ggml_rms_norm(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, P)), 1e-12f);
        }
        ggml_tensor * qi = ggml_mul_mat(ctx0, wq, qs);                                     // [d_idx, T]
        S = ggml_mul_mat(ctx0, PT, qi);
        if (dump_dense()) {
            return nullptr;
        }
        if (cfg.mode == llama_lsa_config::POS) {
            // fixed blocks of B consecutive cells: centroid = normalized sum of the block's index keys
            const int64_t B  = cfg.bucket;
            const int64_t nb = (n_kv + B - 1)/B;
            ggml_tensor * PTp = nb*B > n_kv ? ggml_pad(ctx0, PT, 0, (int) (nb*B - n_kv), 0, 0) : PT;
            ggml_tensor * c3  = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, PTp, d_idx, B, nb), 1, 0, 2, 3));
            ggml_tensor * C   = ggml_rms_norm(ctx0, ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, c3), d_idx, nb), 1e-12f);
            ggml_tensor * Sb  = ggml_mul_mat(ctx0, C, qi);                                 // [nb, T]
            Sb = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, Sb, 1, nb, T), B, nb, T, 1);
            Sb = ggml_reshape_2d(ctx0, Sb, B*nb, T);
            Sb = ggml_cont(ctx0, ggml_view_2d(ctx0, Sb, n_kv, T, Sb->nb[1], 0));
            // the cell score only breaks ties inside a block (|S| <= d_idx after rms_norm on both sides is bounded)
            S = ggml_add(ctx0, Sb, ggml_scale(ctx0, S, 1e-4f));
        } else if (cfg.block > 1) {
            // whole blocks of B cells (B = 128 = an SJ-KVaRN body record, so a dropped block is a skipped record):
            // block score = log of the block's share of softmax(block_tau * S) (causal mask applied); the cell score
            // only breaks ties inside a block, so top-k takes whole blocks except at most one partial block
            const int64_t B  = cfg.block;
            const int64_t nb = (n_kv + B - 1)/B;
            if (cfg.rec_blk && cfg.fused_sel && !cfg.fast && cfg.topp <= 0.0) {
                // LLAMA_LSA_FUSED_SEL=1: block logsumexp(tau*S + mask) + eligibility + top-k in one op; same ranking
                // as the softmax share below (the row normalizer is common to all blocks), no 200K-wide softmax row
                if (ggml_tensor * mb = build_attn_lsa_mask_sb(inp, nullptr, kq_mask, n_kv, keep_lr, il, false, S,
                            (float) cfg.block_tau)) {
                    static uint32_t logged_ep = 0;
                    if (llama_lsa_log_once(logged_ep)) {
                        LLAMA_LOG_WARN("%s: LSA record pick by fused block select active (layer %d, %lld blocks of %lld)\n",
                                __func__, il, (long long) nb, (long long) B);
                    }
                    return mb;
                }
            }
            ggml_tensor * mk = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
            ggml_tensor * Mb = nullptr;
            ggml_tensor * x  = nullptr; // [nb*B, T] tau*S + mask, clamped and padded (lse path)
            if ((cfg.rec_blk || cfg.blk_lse || cfg.refine > 0) && !cfg.fast && cfg.topp <= 0.0) {
                // floor-free block score: logsumexp_block(tau*S + mask) = bmax + log sum exp(x - bmax). Same ranking as
                // log(block share) wherever that share is above the 1e-30 floor; the query latent is not normalized, so
                // at tau 2 softmax(tau*S) is near one-hot and most blocks hit the floor (-69): the block top-k then
                // picked among ties (rbx: REC_BLK KLD 0.053 vs per-cell 0.0016). Hidden cells clamp to -1e30, so a
                // fully hidden block scores ~-1e30 (never NaN) and is ineligible anyway.
                x = ggml_add(ctx0, ggml_scale(ctx0, S, (float) cfg.block_tau), ggml_reshape_2d(ctx0, mk, n_kv, T));
                x = ggml_clamp(ctx0, x, -1e30f, 1e30f);
                if (nb*B > n_kv) {
                    // tail cells without a cache cell score -1e30 (exp underflows to 0 below any real cell)
                    x = ggml_concat(ctx0, x, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nb*B - n_kv, T), -1e30f), 0);
                }
                ggml_tensor * bm = ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, (int) B, (int) B, 0);       // [nb, T]
                ggml_tensor * e  = ggml_exp(ctx0, ggml_sub(ctx0, ggml_reshape_3d(ctx0, x, B, nb, T),
                                                             ggml_reshape_3d(ctx0, bm, 1, nb, T)));
                Mb = ggml_add(ctx0, ggml_log(ctx0, ggml_sum_rows(ctx0, e)), ggml_reshape_3d(ctx0, bm, 1, nb, T)); // [1, nb, T]
            } else {
                ggml_tensor * Pc = ggml_soft_max_ext(ctx0, S, ggml_reshape_2d(ctx0, mk, n_kv, T), (float) cfg.block_tau, 0.0f);
                if (nb*B > n_kv) {
                    Pc = ggml_pad(ctx0, Pc, (int) (nb*B - n_kv), 0, 0, 0);
                }
                Mb = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, Pc, B, nb, T));                   // [1, nb, T]
                Mb = ggml_log(ctx0, ggml_scale_bias(ctx0, Mb, 1.0f, 1e-30f));
            }
            if (cfg.rec_blk && !cfg.fast && cfg.topp <= 0.0) {
                // LLAMA_LSA_REC_BLK=1: rank the nb blocks by the same block score instead of a per-cell top-k over
                // n_kv (at 200K x 25% keep that top-k sorts ~50K of 200K per row per layer); whole blocks only, so
                // the per-cell pick's single partial block (< B cells) is not taken
                if (ggml_tensor * mb = build_attn_lsa_mask_sb(inp, ggml_reshape_2d(ctx0, Mb, nb, T), kq_mask, n_kv,
                            keep_lr, il)) {
                    static uint32_t logged_ep = 0;
                    if (llama_lsa_log_once(logged_ep)) {
                        LLAMA_LOG_WARN("%s: LSA record pick by block top-k active (layer %d, %lld blocks of %lld)\n",
                                __func__, il, (long long) nb, (long long) B);
                    }
                    return mb;
                }
            }
            if (cfg.refine > 0 && x && !cfg.rec_blk) {
                // LLAMA_LSA_REFINE=F: two-stage pick. Stage 1 keeps the kc best eligible blocks (a block is eligible if it
                // holds a visible middle cell of the row); stage 2 ranks the cells inside them by the logsumexp score of
                // their F-cell unit (F = 1: the cell's own tau*S), so the per-cell top-k in make_mask takes the best keep
                // cells of the candidates. Non-candidates get -1e15: below every candidate, above hidden cells, and far
                // under the 1e20 sink/window bonus, so sinks and the window stay kept.
                const int64_t F  = cfg.refine;
                const int64_t kc = std::min<int64_t>(nb, (int64_t) std::ceil(cfg.refine_mult*(double) keep_lr/(double) B));
                // LLAMA_LSA_REFINE_SH=1: with SHARED at T > 1 both stages run in the op's shared mode (one pick per
                // ubatch, row-identical gate), else SHARED keeps the graph chain below
                const bool rsh = cfg.refine_sh && cfg_shared && T > 1;
                const int  rsm = rsh ? (cfg.shared_max ? 2 : 1) : 0;
                const bool runi = cfg_union && T > 1; // LLAMA_LSA_SHARED=3: rows pick their own units, then OR
                if (cfg.fused_sel && (!(cfg_shared && T > 1) || rsh) && n_kv % B == 0 && S->type == GGML_TYPE_F32 &&
                        inp->lsa_cell_pos->ne[0] == n_kv) {
                    // LLAMA_LSA_FUSED_SEL=1: stage 1 = block select of the kc best blocks (block logsumexp in the op),
                    // stage 2 = the same op over F-cell units gated by stage 1, keep_lr/F units; whole units only and
                    // the window/sink rule per unit (first cell), otherwise the same ranking as the chain below
                    ggml_tensor * Sc  = ggml_is_contiguous(S) ? S : ggml_cont(ctx0, S);
                    ggml_tensor * mk2 = ggml_reshape_2d(ctx0, ggml_is_contiguous(kq_mask) ? kq_mask : ggml_cont(ctx0, kq_mask), n_kv, T);
                    const int64_t nu  = n_kv/F;
                    const int64_t ku  = std::max<int64_t>(1, std::min<int64_t>(nu, keep_lr/F));
                    // LLAMA_LSA_PICK_EVERY=N with the unit pick (shared pick or one row): reuse this layer's last unit
                    // pick for up to N - 1 graphs (both select ops and the index scoring drop out of those graphs).
                    // The stored plane is row 0's pick OR row 0's window units, so a unit that leaves the window
                    // during reuse stays kept (no re-pick on every window-edge move: edge is not checked here);
                    // units newer than the pick read 0 from the zero padding and are inside the window anyway.
                    const ggml_tensor * pick_k = lsa_pick_kbase;
                    lsa_pick_kbase = nullptr;
                    ggml_tensor * selu       = nullptr;
                    ggml_tensor * selu_chk   = nullptr; // LLAMA_LSA_PICK_CHECK: the reused pick, while selu is the fresh one
                    ggml_tensor * pick_store = nullptr;
                    int64_t       pick_age   = 0;
                    if (pick_k && cfg.pick_every > 1 && (rsh || runi || T == 1) && lsa_ubatch().pos && (int64_t) lsa_ubatch().n_tokens == T &&
                            lsa_ubatch().n_seqs_unq == 1 && lsa_ubatch().seq_id_unq && pick_k->buffer && inp->lsa_q_pos->ne[0] == 1) {
                        const int32_t seq   = lsa_ubatch().seq_id_unq[0];
                        const int64_t n_cap = inp->mctx->get_kv_size()/F + 1;
                        const int64_t pos0  = lsa_ubatch().pos[0];
                        llama_lsa_pick_slot ps;
                        if (llama_lsa_pick_get(pick_k, n_cap, ggml_backend_buffer_get_type(pick_k->buffer), pos0, 0, nu, seq, &ps)) {
                            inp->lsa_pick_commit.push_back({ pick_k, ps.gen, ps.reuse, pos0, 0, nu, seq });
                            if (ps.reuse) {
                                pick_age = ps.age;
                                ggml_tensor * v = ggml_view_2d(ctx0, ps.sel, nu, 1, ps.sel->nb[0]*nu, 0);
                                selu = T == 1 ? ggml_cont(ctx0, v) : ggml_repeat(ctx0, v, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nu, T));
                                cb(selu, "lsa_unit_sel_reuse", il);
                                if (cfg.pick_check) {
                                    selu_chk = selu; // attention keeps the reused pick; the fresh one is built below for the log
                                    selu     = nullptr;
                                }
                                static uint32_t logged_ur = 0;
                                if (llama_lsa_log_once(logged_ur)) {
                                    LLAMA_LOG_WARN("%s: LSA unit pick reused (LLAMA_LSA_PICK_EVERY=%lld, layer %d, %lld units of %lld)\n",
                                            __func__, (long long) cfg.pick_every, il, (long long) nu, (long long) F);
                                }
                            } else {
                                pick_store = ps.sel;
                            }
                        }
                    } else if (pick_k && cfg.pick_every > 1 && lsa_ubatch().n_seqs_unq > 1) {
                        // this graph's rows are of several sequences and skip the cache, but they add cells, so the
                        // stored pick must not be reused across it (n_use would not count this graph)
                        inp->lsa_pick_commit.push_back({ pick_k, 0, false, 0, 0, 0, -1, true });
                    }
                    // LLAMA_LSA_UNIT_OP=1: one op scores the index keys and takes the F-cell unit logsumexp (the score
                    // tensor S is never written); stage 1 ranks blocks of B/F units of it, stage 2 its units. Positions
                    // are the units' first cells and sinks' = sinks - B + B/F (stage 1) / sinks - F + 1 (stage 2), so
                    // eligibility is the cell rule of the chain; the block score is the logsumexp of the unit scores
                    // (same value as over the cells, other rounding)
                    const bool uop = cfg.unit_op && (F & (F - 1)) == 0 && F <= 128 && B % F == 0 &&
                        (PT->type == GGML_TYPE_F16 || PT->type == GGML_TYPE_F32) && PT->nb[0] == ggml_type_size(PT->type) &&
                        PT->ne[1] == n_kv && PT->ne[2] == 1 && PT->ne[3] == 1 && qi->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(qi) && qi->ne[0] == PT->ne[0] && qi->ne[1] == T && qi->ne[2] == 1;
                    const bool fresh = selu == nullptr; // this graph makes the pick (union below applies)
                    if (!selu && uop) {
                        ggml_tensor * Ul = ggml_lsa_unit_lse(ctx0, PT, qi, mk2, (int) F, (float) cfg.block_tau);
                        cb(Ul, "lsa_unit_lse", il);
                        if (!cfg.score_log.empty() && n_kv >= cfg.score_log_min_kv && Ul->type == GGML_TYPE_F32) {
                            // LLAMA_LSA_SCORE_LOG: row 0 position, T, then the row max of each unit score, to the host log
                            ggml_tensor * um = Ul;
                            if (T > 1) {
                                ggml_tensor * ut = ggml_cont(ctx0, ggml_transpose(ctx0, Ul));                   // [T, nu]
                                um = ggml_pool_1d(ctx0, ut, GGML_OP_POOL_MAX, (int) T, (int) T, 0);
                            }
                            um = ggml_reshape_1d(ctx0, ggml_cont(ctx0, um), nu);
                            ggml_tensor * q0 = ggml_reshape_1d(ctx0, ggml_cont(ctx0,
                                    ggml_view_2d(ctx0, inp->lsa_q_pos, 1, 1, inp->lsa_q_pos->nb[1], 0)), 1);
                            ggml_tensor * hd = ggml_concat(ctx0, q0, ggml_scale_bias(ctx0, q0, 0.0f, (float) T), 0);
                            // middle flag per unit, row 0 (same rule as LLAMA_LSA_PICK_CHECK): visible, outside the
                            // window, past the sinks, by cell position (cells need not be in position order)
                            ggml_tensor * qp0 = ggml_view_2d(ctx0, inp->lsa_q_pos, 1, 1, inp->lsa_q_pos->nb[1], 0);
                            ggml_tensor * cp  = ggml_reshape_2d(ctx0, inp->lsa_cell_pos, n_kv, 1);
                            ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, cp, qp0), 1.0f, (float) cfg.window - 0.5f));
                            ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, cp, -1.0f, (float) cfg.sinks - 0.5f));
                            ggml_tensor * mk0 = ggml_view_2d(ctx0, mk2, n_kv, 1, mk2->nb[1], 0);
                            ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0,
                                    mk0->type == GGML_TYPE_F32 ? mk0 : ggml_cast(ctx0, mk0, GGML_TYPE_F32), 1.0f, 1e29f));
                            ggml_tensor * mid = ggml_mul(ctx0, vis, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, win, snk), -1.0f, 0.5f)));
                            ggml_tensor * mu  = ggml_reshape_1d(ctx0, ggml_cont(ctx0,
                                    ggml_pool_1d(ctx0, mid, GGML_OP_POOL_MAX, (int) F, (int) F, 0)), nu);
                            ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, ggml_concat(ctx0, ggml_concat(ctx0, hd, um, 0), mu, 0),
                                    lsa_score_log_op, 1,
                                    (void *) (intptr_t) ((il & 255) | (F << 8))));
                        }
                        ggml_tensor * cpu = inp->lsa_cell_pos;
                        if (F > 1) {
                            if (inp->lsa_cpos_u == nullptr || inp->lsa_cpos_u_f != F || inp->lsa_cpos_u->ne[0] != nu) {
                                inp->lsa_cpos_u   = ggml_reshape_1d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, inp->lsa_cell_pos,
                                        1, nu, F*inp->lsa_cell_pos->nb[0], 0)), nu);
                                inp->lsa_cpos_u_f = F;
                            }
                            cpu = inp->lsa_cpos_u;
                        }
                        ggml_tensor * sel1 = lsa_slice(lsa_sticky(ggml_lsa_block_select_cells(ctx0, Ul, nullptr, cpu, inp->lsa_q_pos, (int) (B/F), 1.0f,
                                (int) cfg.window, (int) (cfg.sinks - B + B/F), (int) std::max<int64_t>(1, kc), rsm),
                                sj_kvarn_pending.body, sj_kvarn_pending.desc, (int) F), B);
                        selu = lsa_slice(lsa_sticky(ggml_lsa_block_select_cells_gated_ext(ctx0, Ul, nullptr, cpu, inp->lsa_q_pos, 1, 1.0f,
                                (int) cfg.window, (int) (cfg.sinks - F + 1), (int) ku, sel1, (int) (B/F), rsm),
                                sj_kvarn_pending.body, sj_kvarn_pending.desc, (int) F), F);
                        static uint32_t logged_uo = 0;
                        if (llama_lsa_log_once(logged_uo)) {
                            LLAMA_LOG_WARN("%s: LSA unit-lse pick active (LLAMA_LSA_UNIT_OP=1, layer %d, %lld units of %lld, T %lld)\n",
                                    __func__, il, (long long) nu, (long long) F, (long long) T);
                        }
                    }
                    if (!selu) {
                        ggml_tensor * sel1 = lsa_slice(lsa_sticky(ggml_lsa_block_select_cells(ctx0, Sc, mk2, inp->lsa_cell_pos, inp->lsa_q_pos,
                                (int) B, (float) cfg.block_tau, (int) cfg.window, (int) cfg.sinks, (int) std::max<int64_t>(1, kc), rsm),
                                sj_kvarn_pending.body, sj_kvarn_pending.desc, 1), B);
                        selu = lsa_slice(lsa_sticky(ggml_lsa_block_select_cells_gated_ext(ctx0, Sc, mk2, inp->lsa_cell_pos, inp->lsa_q_pos,
                                (int) F, (float) cfg.block_tau, (int) cfg.window, (int) cfg.sinks, (int) ku, sel1, (int) (B/F), rsm),
                                sj_kvarn_pending.body, sj_kvarn_pending.desc, 1), F);
                    }
                    if (fresh) {
                        if (runi) {
                            ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, selu));                             // [T, nu]
                            ggml_tensor * u  = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, st, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nu, 1);
                            selu = ggml_repeat(ctx0, u, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nu, T));
                            static uint32_t logged_uu = 0;
                            if (llama_lsa_log_once(logged_uu)) {
                                LLAMA_LOG_WARN("%s: LSA union of per-row unit picks (LLAMA_LSA_SHARED=3, layer %d, %lld units of %lld per row, T %lld)\n",
                                        __func__, il, (long long) ku, (long long) F, (long long) T);
                            }
                        }
                    }
                    if (selu_chk) {
                        // row 0 middle units: visible, outside row 0's window, past the sinks (cell positions)
                        ggml_tensor * q0  = ggml_view_2d(ctx0, inp->lsa_q_pos, 1, 1, inp->lsa_q_pos->nb[1], 0);
                        ggml_tensor * cp  = ggml_reshape_2d(ctx0, inp->lsa_cell_pos, n_kv, 1);
                        ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, cp, q0), 1.0f, (float) cfg.window - 0.5f));
                        ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, cp, -1.0f, (float) cfg.sinks - 0.5f));
                        ggml_tensor * mk0 = ggml_view_2d(ctx0, mk2, n_kv, 1, mk2->nb[1], 0);
                        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0,
                                mk0->type == GGML_TYPE_F32 ? mk0 : ggml_cast(ctx0, mk0, GGML_TYPE_F32), 1.0f, 1e29f));
                        ggml_tensor * mid = ggml_mul(ctx0, vis, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, win, snk), -1.0f, 0.5f)));
                        ggml_tensor * mu  = ggml_reshape_1d(ctx0, ggml_pool_1d(ctx0, mid, GGML_OP_POOL_MAX, (int) F, (int) F, 0), nu);
                        ggml_tensor * f0  = ggml_mul(ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, selu, nu, 0)), mu);
                        ggml_tensor * r0  = ggml_mul(ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, selu_chk, nu, 0)), mu);
                        // recall at the unit, then with both picks widened to 16/64/128-cell blocks (g units per
                        // block; a level finer than the unit repeats the unit level; a ragged tail block is dropped)
                        auto lvl = [&](int64_t G) {
                            const int g = (int) std::max<int64_t>(1, G/F);
                            ggml_tensor * fb = f0, * rb = r0;
                            if (g > 1 && nu >= g) {
                                fb = ggml_reshape_1d(ctx0, ggml_pool_1d(ctx0, ggml_reshape_2d(ctx0, f0, nu, 1), GGML_OP_POOL_MAX, g, g, 0), nu/g);
                                rb = ggml_reshape_1d(ctx0, ggml_pool_1d(ctx0, ggml_reshape_2d(ctx0, r0, nu, 1), GGML_OP_POOL_MAX, g, g, 0), nu/g);
                            }
                            return ggml_concat(ctx0, ggml_concat(ctx0, ggml_sum(ctx0, ggml_mul(ctx0, fb, rb)), ggml_sum(ctx0, fb), 0),
                                    ggml_sum(ctx0, rb), 0);
                        };
                        ggml_tensor * st = lvl(F);
                        for (int64_t G : { 16, 64, 128 }) {
                            st = ggml_concat(ctx0, st, lvl(G), 0);
                        }
                        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st, lsa_pick_check_op, 1,
                                (void *) (intptr_t) ((il & 255) | (std::min<int64_t>(pick_age, 15) << 8))));
                        selu = selu_chk;
                    }
                    if (pick_store) {
                        // u = sel0 OR win0 (both 0/1): sel0 = row 0's unit pick, win0 = unit holds a cell inside row 0's
                        // window (cell - q0 > -window), zero-padded to the cache length
                        ggml_tensor * s0  = ggml_view_1d(ctx0, selu, nu, 0);
                        ggml_tensor * q0  = ggml_view_2d(ctx0, inp->lsa_q_pos, 1, 1, inp->lsa_q_pos->nb[1], 0);
                        ggml_tensor * cp  = ggml_reshape_2d(ctx0, inp->lsa_cell_pos, n_kv, 1);
                        ggml_tensor * w   = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, cp, q0), 1.0f, (float) cfg.window - 0.5f));
                        ggml_tensor * wu  = ggml_reshape_1d(ctx0, ggml_pool_1d(ctx0, w, GGML_OP_POOL_MAX, (int) F, (int) F, 0), nu);
                        ggml_tensor * u   = ggml_sub(ctx0, ggml_add(ctx0, s0, wu), ggml_mul(ctx0, s0, wu));
                        if (pick_store->ne[0] > nu) {
                            u = ggml_pad(ctx0, u, (int) (pick_store->ne[0] - nu), 0, 0, 0);
                        }
                        ggml_build_forward_expand(gf, ggml_cpy(ctx0, u, pick_store));
                    }
                    cb(selu, "lsa_unit_sel", il);
                    if (cfg.spec_recall && sj && T <= 8) {
                        auto predict = [&](ggml_tensor * stale_q, int budget_pct) -> ggml_tensor * {
                            // Missing history in the first three mapped layers remains in the denominator.
                            if (!stale_q) {
                                return ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nu, T), 0.0f);
                            }
                            // Logging-only stale-query predictor. Score after this layer's index append so there is
                            // no uninitialized read. Newly appended entries are in the dense tail, outside the ranked
                            // middle. The logger excludes newly sealed pages from prediction (but counts their misses).
                            GGML_ASSERT((F == 16 || F == 128) && B == 128 && PT == idx && !cfg_shared && cfg.slice == 0);
                            ggml_tensor * sq4 = ggml_reshape_4d(ctx0, ggml_cont(ctx0, stale_q), d_head, g, n_head_kv, T);
                            sq4 = ggml_cont(ctx0, ggml_permute(ctx0, sq4, 1, 0, 2, 3));
                            ggml_tensor * sqs = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sq4), d_kv, T);
                            if (sqs->type != GGML_TYPE_F32) {
                                sqs = ggml_cast(ctx0, sqs, GGML_TYPE_F32);
                            }
                            ggml_tensor * ss = ggml_mul_mat(ctx0, PT, ggml_mul_mat(ctx0, wq, sqs));
                            ggml_tensor * sp = llama_lsa_build_recall_pick(ctx0, ss, mk2, inp->lsa_cell_pos, inp->lsa_q_pos,
                                    (int) B, (int) F, (float) cfg.block_tau, (int) cfg.window, (int) cfg.sinks,
                                    (int) std::max<int64_t>(1, kc), (int) ku, budget_pct);
                            if (runi) {
                                ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, sp));
                                ggml_tensor * su = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, st, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nu, 1);
                                sp = ggml_repeat(ctx0, su, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nu, T));
                            }
                            // Predict compulsory sink/tail reads too. Visibility remains row-specific, as in the real
                            // mask, even when the selected middle is shared by union. Any live cell makes a page live.
                            ggml_tensor * cp = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, inp->lsa_cell_pos, n_kv, 1), ss);
                            ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, cp, inp->lsa_q_pos),
                                    1.0f, (float) cfg.window - 0.5f));
                            ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, cp, -1.0f, (float) cfg.sinks - 0.5f));
                            ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, mk2, GGML_TYPE_F32), 1.0f, 1e29f));
                            ggml_tensor * force = ggml_mul(ctx0, ggml_add(ctx0, win, snk), vis);
                            force = ggml_pool_1d(ctx0, force, GGML_OP_POOL_MAX, (int) F, (int) F, 0);
                            return ggml_cont(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, sp, force), 1.0f, -0.5f)));
                        };
                        ggml_tensor * stale_q[] = { lsa_spec_q, lsa_spec_q2, lsa_spec_q3 };
                        ggml_tensor * budgets = nullptr;
                        for (int b = 0; b < GGML_LSA_RECALL_BUDGETS; ++b) {
                            const int pct = ggml_lsa_recall_budget_pct(b);
                            ggml_tensor * picks = predict(stale_q[0], pct);
                            for (int h = 1; h < GGML_LSA_RECALL_HORIZONS; ++h) {
                                picks = ggml_concat(ctx0, picks, predict(stale_q[h], pct), 2);
                            }
                            budgets = budgets ? ggml_concat(ctx0, budgets, picks, 3) : picks;
                        }
                        lsa_spec_sel = ggml_cont(ctx0, budgets);
                        cb(lsa_spec_sel, "lsa_spec_pick", il);
                    }
                    if (ggml_tensor * mu = build_attn_lsa_mask_units(inp, selu, F, kq_mask, n_kv, il, rsm > 0 || runi)) {
                        static uint32_t logged_ep = 0;
                        if (llama_lsa_log_once(logged_ep)) {
                            LLAMA_LOG_WARN("%s: LSA fused refine pick active (layer %d, %lld of %lld blocks of %lld, %lld units of %lld)\n",
                                    __func__, il, (long long) kc, (long long) nb, (long long) B, (long long) ku, (long long) F);
                        }
                        return mu;
                    }
                }
                ggml_tensor * dq  = ggml_sub(ctx0, ggml_repeat(ctx0, inp->lsa_cell_pos, S), inp->lsa_q_pos);
                ggml_tensor * bon = ggml_add(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, dq, 1.0f, (float) cfg.window - 0.5f)),
                        ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_cell_pos, -1.0f, (float) cfg.sinks - 0.5f)));
                ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_reshape_2d(ctx0, mk, n_kv, T), 1.0f, 1e29f));
                ggml_tensor * mid = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, bon, -1.0f, 0.5f)), vis); // [n_kv, T]
                if (nb*B > n_kv) {
                    mid = ggml_pad(ctx0, mid, (int) (nb*B - n_kv), 0, 0, 0);
                }
                ggml_tensor * elig = ggml_pool_1d(ctx0, mid, GGML_OP_POOL_MAX, (int) B, (int) B, 0);           // [nb, T]
                ggml_tensor * rank = ggml_add(ctx0, ggml_reshape_2d(ctx0, Mb, nb, T), ggml_scale_bias(ctx0, elig, 1e30f, -1e30f));
                ggml_tensor * top  = ggml_top_k(ctx0, rank, (int) kc);                                          // [kc, T]
                ggml_tensor * cm   = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, nb, T, 1), -1e15f);
                cm = ggml_set_rows(ctx0, cm, ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, kc, T, 1), 0.0f), top);
                cm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, cm, B, nb, T, 1), B*nb, T);
                cm = ggml_cont(ctx0, ggml_view_2d(ctx0, cm, n_kv, T, cm->nb[1], 0));
                ggml_tensor * Sf = nullptr;
                if (F == 1) {
                    Sf = ggml_scale(ctx0, S, (float) cfg.block_tau);
                } else {
                    const int64_t nf = nb*B/F;
                    ggml_tensor * fm = ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, (int) F, (int) F, 0);           // [nf, T]
                    ggml_tensor * fe = ggml_exp(ctx0, ggml_sub(ctx0, ggml_reshape_3d(ctx0, x, F, nf, T),
                                                                 ggml_reshape_3d(ctx0, fm, 1, nf, T)));
                    ggml_tensor * Mf = ggml_add(ctx0, ggml_log(ctx0, ggml_sum_rows(ctx0, fe)), ggml_reshape_3d(ctx0, fm, 1, nf, T));
                    Mf = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, Mf, F, nf, T, 1), F*nf, T);
                    Sf = ggml_add(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, Mf, n_kv, T, Mf->nb[1], 0)), ggml_scale(ctx0, S, 1e-4f));
                }
                S = ggml_add(ctx0, cm, Sf);
                static uint32_t logged_ep = 0;
                if (llama_lsa_log_once(logged_ep)) {
                    LLAMA_LOG_WARN("%s: LSA refine pick active (layer %d, %lld of %lld blocks of %lld, unit %lld, keep %lld)\n",
                            __func__, il, (long long) kc, (long long) nb, (long long) B, (long long) F, (long long) keep_lr);
                }
            } else {
            Mb = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, Mb, B, nb, T, 1), B*nb, T);
            Mb = ggml_cont(ctx0, ggml_view_2d(ctx0, Mb, n_kv, T, Mb->nb[1], 0));
            S = ggml_add(ctx0, Mb, ggml_scale(ctx0, S, 1e-4f));
            }
        }
    }
    if ((cfg.mode == llama_lsa_config::ORACLE || cfg.mode == llama_lsa_config::MASS) && dump_dense()) {
        return nullptr;
    }
    if (cfg.fast && lsa_one_seq(lsa_ubatch())) {
        return build_attn_lsa_fast(inp, S, k_io, v_io, kq_mask, keep_lr, il);
    }
    if (cfg_shared && T > 1 && S_heads.empty()) {
        // mb-MTP verify option A: every row of the ubatch gets the same long-range pick, ranked by the sum of the
        // rows' index scores (= the score of the summed index query). The window covers the draft positions.
        ggml_tensor * St = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, S)));        // [1, n_kv]
        S = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, St, n_kv, 1), S);
    } else if (cfg_shared && T > 1 && sj_ph_m) {
        // same, per KV head
        for (ggml_tensor *& Sh : S_heads) {
            ggml_tensor * St = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, Sh)));
            Sh = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, St, n_kv, 1), Sh);
        }
    }
    // adaptive keep (LLAMA_LSA_TOPP): the top-k cap stays k_total; floor = the k_floor best (sinks/window included)
    const int64_t k_floor = std::min<int64_t>(k_total, cfg.sinks + cfg.window + (int64_t) (cfg.topp_min * (double) n_kv));
    auto make_mask = [&](ggml_tensor * S, bool keep_f32 = false) -> ggml_tensor * {
    cb(S, "lsa_score", il);
    ggml_tensor * S_raw = S;

    // keep rule: first `sinks` positions and the last `window` positions (query included) get a large bonus
    ggml_tensor * d   = ggml_sub(ctx0, ggml_repeat(ctx0, inp->lsa_cell_pos, S), inp->lsa_q_pos);    // cell - query
    ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, d, 1.0f, (float) cfg.window - 0.5f));
    ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_cell_pos, -1.0f, (float) cfg.sinks - 0.5f));
    ggml_tensor * bonus = ggml_add(ctx0, win, snk);
    S = ggml_add(ctx0, S, ggml_scale(ctx0, bonus, 1e20f));
    ggml_tensor * mask_f32 = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    mask_f32 = ggml_reshape_2d(ctx0, mask_f32, n_kv, T);
    S = ggml_add(ctx0, S, mask_f32);

    ggml_tensor * top_k = ggml_top_k(ctx0, S, (int) k_total); // I32 [k_total, T]
    cb(top_k, "lsa_top_k", il);

    // DSA-style mask, in f32: a large negative fill (rows before min_pos get 0, i.e. stay dense), 0 at the kept
    // cells, then the causal mask added back and the result cast to the mask type (f16 overflow gives -inf)
    ggml_tensor * m = nullptr;
    if (cfg.min_pos > 0) {
        ggml_tensor * sparse_row = ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_q_pos, 1.0f, 0.5f - (float) cfg.min_pos));
        m = ggml_repeat(ctx0, ggml_scale(ctx0, sparse_row, -1e30f), S);
    } else {
        m = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, T), -INFINITY);
    }
    m = ggml_view_4d(ctx0, m, 1, n_kv, T, 1, m->nb[0], m->nb[1], m->nb[1]*T, 0);                 // [1, n_kv, T, 1]
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_total, T, 1), 0.0f);
    m = ggml_set_rows(ctx0, m, zeros, top_k);
    m = ggml_view_2d(ctx0, m, n_kv, T, m->nb[2], 0);                                             // [n_kv, T]
    m = ggml_add(ctx0, m, mask_f32);
    ggml_tensor * mid = ggml_step(ctx0, ggml_scale_bias(ctx0, bonus, -1.0f, 0.5f)); // 1 = middle cell (not sink/window)
    if (cfg.topp > 0.0 && k_floor < k_total) {
        // smallest set of middle cells holding share P of softmax(tau * z) over the middle, z = row z-score of the
        // index score: per-row threshold t on z by bisection (12 steps over z in [-4, 12]), mass(t) = sum p[z >= t]
        ggml_tensor * z    = ggml_norm(ctx0, S_raw, 1e-6f);
        ggml_tensor * excl = ggml_add(ctx0, mask_f32, ggml_scale_bias(ctx0, mid, 1e30f, -1e30f));
        ggml_tensor * p    = ggml_soft_max_ext(ctx0, z, excl, (float) cfg.topp_tau, 0.0f);
        ggml_tensor * t    = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, T), 4.0f);
        float delta = 4.0f;
        for (int it = 0; it < 12; ++it) {
            ggml_tensor * above = ggml_step(ctx0, ggml_sub(ctx0, z, t));
            ggml_tensor * mass  = ggml_sum_rows(ctx0, ggml_mul(ctx0, p, above));                    // [1, T]
            ggml_tensor * up    = ggml_step(ctx0, ggml_scale_bias(ctx0, mass, 1.0f, (float) -cfg.topp)); // mass >= P
            t = ggml_add(ctx0, t, ggml_scale_bias(ctx0, up, 2.0f*delta, -delta));
            delta *= 0.5f;
        }
        // drop = middle cell below t and outside the floor; -1e30 on top of the capped mask (f16 cast gives -inf)
        ggml_tensor * below = ggml_step(ctx0, ggml_neg(ctx0, ggml_sub(ctx0, z, t)));
        ggml_tensor * fl = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_kv, T, 1), 1.0f);
        fl = ggml_set_rows(ctx0, fl, ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_floor, T, 1), 0.0f),
                ggml_top_k(ctx0, S, (int) k_floor));
        fl = ggml_view_2d(ctx0, fl, n_kv, T, fl->nb[2], 0);
        ggml_tensor * drop = ggml_mul(ctx0, ggml_mul(ctx0, below, mid), fl);
        if (cfg.min_pos > 0) {
            // rows before min_pos stay dense
            drop = ggml_mul(ctx0, drop, ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_q_pos, 1.0f, 0.5f - (float) cfg.min_pos)));
        }
        m = ggml_add(ctx0, m, ggml_scale(ctx0, drop, -1e30f));
        cb(m, "lsa_topp_mask", il);
    }
    if (cfg.stats && il >= 0 && il < 256) {
        ggml_tensor * valid = ggml_step(ctx0, ggml_scale_bias(ctx0, mask_f32, 1.0f, 1e29f));
        ggml_tensor * kept  = ggml_sum_rows(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, m, 1.0f, 1e29f)));
        ggml_tensor * bon   = ggml_sum_rows(ctx0, ggml_mul(ctx0, valid, ggml_scale_bias(ctx0, mid, -1.0f, 1.0f)));
        ggml_tensor * st    = ggml_concat(ctx0, ggml_concat(ctx0, kept, ggml_sum_rows(ctx0, valid), 0), bon, 0); // [3, T]
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st, lsa_stats_op, 1, (void *) (intptr_t) il));
        // union over the ubatch rows (the decode kernel walks a strip / record if any row keeps a cell of it)
        auto uni = [&](ggml_tensor * x) { // [n_kv, T] 0/1 -> [n_kv] 0/1
            ggml_tensor * u = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, x)));          // [1, n_kv]
            return ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_reshape_1d(ctx0, u, n_kv), 1.0f, -0.5f));
        };
        auto cnt = [&](ggml_tensor * u, int64_t B) { // live blocks of B cells
            const int64_t nb = (n_kv + B - 1)/B;
            ggml_tensor * up = nb*B > n_kv ? ggml_pad(ctx0, u, (int) (nb*B - n_kv), 0, 0, 0) : u;
            ggml_tensor * bs = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, up, B, nb));                   // [1, nb]
            return ggml_sum(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, bs, 1.0f, -0.5f)));
        };
        ggml_tensor * keptm = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, m, 1.0f, 1e29f)), mid);
        ggml_tensor * uk = uni(keptm);
        ggml_tensor * uv = uni(ggml_mul(ctx0, valid, mid));
        ggml_tensor * st2 = ggml_concat(ctx0, ggml_concat(ctx0, cnt(uk, 16), cnt(uv, 16), 0),
                ggml_concat(ctx0, cnt(uk, 128), cnt(uv, 128), 0), 0);                                  // [4]
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st2, lsa_stats_blk_op, 1, (void *) (intptr_t) il));
    }
    if (!keep_f32 && kq_mask->type != GGML_TYPE_F32) {
        m = ggml_cast(ctx0, m, kq_mask->type);
    }
    m = ggml_reshape_4d(ctx0, m, n_kv, T, 1, 1);
    cb(m, "lsa_kq_mask", il);
    return m;
    };

    ggml_tensor * m = nullptr;
    if (S_heads.empty()) {
        m = make_mask(S);
    } else if (sj_ph_m) {
        // [n_kv, T, 1, H] -> each KV head's mask repeated over its g q heads (head h reads KV head h / g)
        ggml_tensor * m4 = nullptr;
        for (ggml_tensor * Sh : S_heads) {
            ggml_tensor * mh = make_mask(Sh, true);
            m4 = m4 ? ggml_concat(ctx0, m4, mh, 3) : mh;
        }
        m4 = ggml_repeat_4d(ctx0, m4, n_kv, T, g, n_head_kv);
        m  = ggml_cast(ctx0, ggml_reshape_4d(ctx0, m4, n_kv, T, n_head, 1), kq_mask->type);
        cb(m, "lsa_kq_mask_heads", il);
    } else {
        // per-KV-head picks: one mask per KV head, the caller runs one FA per GQA group
        for (ggml_tensor * Sh : S_heads) {
            masks_h->push_back(make_mask(Sh));
        }
        m = masks_h->front();
    }

    if (cfg.recall && kv2 && T <= 64 && il >= 0 && il < 256) {
        // true attention of each q head (same scale and causal mask as the model), summed over the cells the pick keeps
        ggml_tensor * mask2 = ggml_reshape_2d(ctx0,
                kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T);
        ggml_tensor * qf = q->type == GGML_TYPE_F32 ? q : ggml_cast(ctx0, q, GGML_TYPE_F32);
        ggml_tensor * acc = nullptr;
        for (int64_t j = 0; j < n_head_kv; ++j) {
            ggml_tensor * mj = (!S_heads.empty() && masks_h && (int64_t) masks_h->size() == n_head_kv) ? (*masks_h)[j] : m;
            mj = ggml_reshape_2d(ctx0, mj->type == GGML_TYPE_F32 ? mj : ggml_cast(ctx0, mj, GGML_TYPE_F32), n_kv, T);
            ggml_tensor * keep = ggml_step(ctx0, ggml_scale_bias(ctx0, mj, 1.0f, 1e29f));                  // [n_kv, T] 0/1
            ggml_tensor * qg = ggml_view_3d(ctx0, qf, d_head, g, T, qf->nb[1], qf->nb[2], j*g*qf->nb[1]);
            qg = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, qg, 0, 2, 1, 3)), d_head, T*g); // [d, T*g]
            qg = ggml_pad_ext(ctx0, qg, (int) (j*d_head), (int) ((n_head_kv - 1 - j)*d_head), 0, 0, 0, 0, 0, 0);
            ggml_tensor * P = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, kv2, qg), n_kv, T, g);
            P = ggml_soft_max_ext(ctx0, P, mask2, kq_scale_lsa, 0.0f);
            ggml_tensor * r = ggml_sum_rows(ctx0, ggml_mul(ctx0, P, keep));                                  // [1, T, g]
            ggml_tensor * bad = ggml_sum(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, r, -1.0f, 0.9f)));     // missed > 10%
            ggml_tensor * aj = ggml_concat(ctx0, ggml_sum(ctx0, r), bad, 0);                                 // [2]
            acc = acc ? ggml_add(ctx0, acc, aj) : aj;
        }
        ggml_tensor * nrows = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), (float) (T*n_head));
        ggml_tensor * rc = ggml_concat(ctx0, acc, nrows, 0);                                                 // [3]
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, rc, lsa_recall_op, 1, (void *) (intptr_t) il));
        static uint32_t logged_ep = 0;
        if (llama_lsa_log_once(logged_ep)) {
            LLAMA_LOG_WARN("%s: LSA recall probe active (layer %d, n_kv %lld, T %lld)\n", __func__, il, (long long) n_kv,
                    (long long) T);
        }
    }

    if (cfg.fa_sparse && !sj) {
        *n_kv_max = k_total;
    } else if (sj && cfg.sj_compact && S_heads.empty()) {
        // SJ-KVaRN compact path: its capacity is n_kv_max * T and must hold the union of the rows' kept cells (it traps
        // on overflow). Per row: k_total. Shared pick: rows differ only by the window shift and the long-range cells
        // that shift moves in or out, union <= k_total + 2(T-1); the bound is set from that (+256 slack).
        *n_kv_max = cfg_shared && T > 1 ? (k_total + 2*T + 256 + T - 1)/T : k_total;
    }

    static int n_logged = 0;
    if (n_logged < 4) {
        ++n_logged;
        LLAMA_LOG_INFO("%s: LSA layer %d: n_kv %lld, T %lld, kept %lld (%.1f%%)\n", __func__, il, (long long) n_kv,
                (long long) T, (long long) k_total, 100.0*(double) k_total/(double) n_kv);
    }

    return m;
}

// LSA centroid compensation (LLAMA_LSA_COMP=1, opt-in, lossy probe; llama-lsa.h). The masked FA drops the long-range
// cells outside the pick entirely, so their softmax mass is lost. Here every block of B cells with at least one dropped
// cell re-enters as one stand-in: key = block mean K, value = block mean V, logit = q.k_mean*scale + log(n_dropped)
// (+ LLAMA_LSA_COMP_BIAS), i.e. n_dropped copies of the mean key (a Jensen under-estimate of their summed exp).
// With P = softmax over [kept cells | stand-ins], the kept part's share alpha = 1 - sum(P_standin), so
//   out = alpha * o_kept + sum_b P_b * v_mean_b
// with o_kept the unchanged masked FA output; the FA kernel and the pick are untouched. Probe cost: dense K scores
// plus f32 cell-major copies of K and V per layer (no cache); plain K/V types, one stream, T <= comp_max_t only.
ggml_tensor * llm_graph_context::build_attn_lsa_comp(
        ggml_tensor * o_kept,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * lsa_mask,
        ggml_tensor * kq_mask,
              float   kq_scale,
                int   il) const {
    const auto & cfg = llama_lsa_cfg();
    const int64_t D = k->ne[0], H_kv = k->ne[1], n_kv = k->ne[2];
    const int64_t Dv = v->ne[0], H = q->ne[1], T = q->ne[2];
    if (k->ne[3] != 1 || v->ne[3] != 1 || q->ne[3] != 1 || T > cfg.comp_max_t || q->ne[0] != D ||
            !lsa_k_plain(k->type) || !lsa_k_plain(v->type) || v->ne[1] != H_kv || v->ne[2] != n_kv ||
            v->nb[1] > v->nb[2] || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || H % H_kv != 0 ||
            lsa_mask->ne[0] != n_kv || lsa_mask->ne[1] != T || ggml_nelements(lsa_mask) != n_kv*T ||
            ggml_nelements(kq_mask) != n_kv*T || o_kept->ne[0] != Dv*H || o_kept->ne[1] != T) {
        static uint32_t warned_ep = 0;
        if (llama_lsa_log_once(warned_ep)) {
            LLAMA_LOG_WARN("%s: layer %d: LSA compensation skipped (unsupported shape/type)\n", __func__, il);
        }
        return nullptr;
    }
    const int64_t B   = cfg.block > 1 ? cfg.block : 128;
    const int64_t nb  = (n_kv + B - 1)/B;
    const int64_t pad = nb*B - n_kv;

    // block means [nb, Dx, H_kv] from a cell-major f32 copy [n_kv, Dx*H_kv]; the padded tail block is never dropped
    // (it lies in the window), so its mean is not used
    auto blk_mean = [&](ggml_tensor * x, int64_t Dx) {
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_cast(ctx0, x, GGML_TYPE_F32), Dx*H_kv, n_kv)));
        if (pad) {
            xt = ggml_pad(ctx0, xt, (int) pad, 0, 0, 0);
        }
        ggml_tensor * sm = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, xt, B, nb, Dx*H_kv));              // [1, nb, Dx*H_kv]
        return ggml_reshape_3d(ctx0, ggml_scale(ctx0, sm, 1.0f/(float) B), nb, Dx, H_kv);
    };
    ggml_tensor * Kb = ggml_cont(ctx0, ggml_permute(ctx0, blk_mean(k, D), 1, 0, 2, 3));                // [D, nb, H_kv]
    ggml_tensor * Vb = blk_mean(v, Dv);                                                                   // [nb, Dv, H_kv]

    ggml_tensor * qp = ggml_permute(ctx0, q, 0, 2, 1, 3);                                                 // [D, T, H]
    ggml_tensor * kq = ggml_mul_mat(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), qp);                         // [n_kv, T, H]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_tensor * ks = ggml_mul_mat(ctx0, Kb, qp);                                                        // [nb, T, H]
    ggml_prec_set_acc(ks, GGML_PREC_F32);

    ggml_tensor * mf = ggml_reshape_2d(ctx0, lsa_mask->type == GGML_TYPE_F32 ? lsa_mask : ggml_cast(ctx0, lsa_mask, GGML_TYPE_F32), n_kv, T);
    ggml_tensor * cf = ggml_reshape_2d(ctx0, kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T);
    // dropped = causally visible (causal mask 0) but masked by the pick (rows before min_pos have none)
    ggml_tensor * drop = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, cf, 1.0f, 1e29f)),
                                        ggml_step(ctx0, ggml_scale_bias(ctx0, mf, -1.0f, -1e29f)));
    if (pad) {
        drop = ggml_pad(ctx0, drop, (int) pad, 0, 0, 0);
    }
    ggml_tensor * nd   = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, drop, B, nb, T)), nb, T);
    ggml_tensor * off  = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, nd, 1.0f, -0.5f)), 1e30f, -1e30f); // 0 or -1e30
    ggml_tensor * bias = ggml_add(ctx0, ggml_scale_bias(ctx0, ggml_log(ctx0, ggml_scale_bias(ctx0, nd, 1.0f, 1e-30f)), 1.0f,
                                  (float) cfg.comp_bias), off);                                           // [nb, T]

    ggml_tensor * S = ggml_concat(ctx0, ggml_add(ctx0, ggml_scale(ctx0, kq, kq_scale), mf),
                                        ggml_add(ctx0, ggml_scale(ctx0, ks, kq_scale), bias), 0);        // [n_kv+nb, T, H]
    ggml_tensor * P  = ggml_soft_max(ctx0, S);
    ggml_tensor * Ps = ggml_cont(ctx0, ggml_view_3d(ctx0, P, nb, T, H, P->nb[1], P->nb[2], n_kv*ggml_element_size(P)));
    cb(Ps, "lsa_comp_p", il);
    ggml_tensor * alpha = ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, Ps), -1.0f, 1.0f);                   // [1, T, H]
    ggml_tensor * os = ggml_mul_mat(ctx0, Vb, Ps);                                                        // [Dv, T, H]
    os    = ggml_cont(ctx0, ggml_permute(ctx0, os, 0, 2, 1, 3));                                          // [Dv, H, T]
    alpha = ggml_cont(ctx0, ggml_permute(ctx0, alpha, 0, 2, 1, 3));                                       // [1, H, T]
    ggml_tensor * ok = ggml_reshape_3d(ctx0, o_kept, Dv, H, T);
    ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, ok, alpha), os);
    cb(out, "lsa_comp_out", il);
    return ggml_reshape_2d(ctx0, out, Dv*H, T);
}

// LSA compensation without dense scores (LLAMA_LSA_COMP=2, opt-in, lossy; needs the index cache for the epoch rule).
// Same math as build_attn_lsa_comp: the kept part enters the softmax as one logit, its log-sum-exp L_k from the masked
// FA node (src[9]); the stand-ins are the dropped blocks' cached mean K/V with log(n dropped) on their logits:
//   P = softmax([q.Kmean_b*scale + log n_b (+bias) | L_k]),  out = P_last * o_kept + sum_b P_b * Vmean_b.
// The F16 block means [D*H_kv, kv_size/B] live beside the K cache: blocks touched by this ubatch are recomputed from
// the cache rows after the store (B cells, get_rows), every whole block when the cache is invalid (first graph,
// state restore, K shift). Per step: ~n_kv/B dot products per row plus the touched blocks' re-read. One stream,
// plain K/V types, T <= comp_max_t. Node order: o_kept is consumed before the merge reads the LSE (src order).
bool llm_graph_context::build_attn_lsa_mean_refresh(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * k_new,
        ggml_tensor * v_new,
            int64_t   n_kv,
        ggml_tensor ** kmv,
        ggml_tensor ** vmv,
            int64_t * nbs_out,
                int   il,
               bool   rot) const {
    const auto & cfg = llama_lsa_cfg();
    const int64_t B  = cfg.block > 1 ? cfg.block : 128;
    const bool    sj = inp->lsa_q_cell != nullptr;
    ggml_tensor * kb = k->view_src ? k->view_src : k;
    ggml_tensor * vb = v->view_src ? v->view_src : v;
    if (!inp->lsa_u_blk || !kb->buffer || (sj && (!k_new || !v_new || !inp->lsa_q_ring || !inp->lsa_u_rblk || !lsa_ubatch().pos))) {
        return false;
    }
    int64_t D, Dv, H_kv, n_cells;
    if (sj) {
        // cells == positions; K/V of this ubatch in the un-rotated basis [D, H_kv, T]
        D = k_new->ne[0]; H_kv = k_new->ne[1]; Dv = v_new->ne[0];
        n_cells = inp->mctx->get_kv_size();
        if (v_new->ne[1] != H_kv || k_new->ne[2] != v_new->ne[2] || k_new->ne[3] != 1 || v_new->ne[3] != 1 ||
                inp->lsa_ring_n <= 0 || inp->lsa_ring_n % B != 0 || inp->lsa_q_ring->ne[0] !=
                (inp->lsa_ring_rows ? std::min<int64_t>(k_new->ne[2], inp->lsa_ring_n) : k_new->ne[2])) {
            return false;
        }
    } else {
        D = k->ne[0]; H_kv = k->ne[1]; Dv = v->ne[0];
        n_cells = kb->ne[1];
        if (k->ne[3] != 1 || v->ne[3] != 1 || !lsa_k_plain(k->type) || !lsa_k_plain(v->type) || v->ne[1] != H_kv ||
                v->ne[2] != k->ne[2] || kb->ne[0] != D*H_kv || vb->ne[0] != Dv*H_kv || kb->ne[2] != 1 || vb->ne[2] != 1 ||
                vb->ne[1] != kb->ne[1] || k->view_offs != 0 || v->view_offs != 0 || k->nb[2] != kb->nb[1] ||
                v->nb[2] != vb->nb[1]) {
            return false;
        }
    }
    const int64_t nbm = n_cells/B;
    const int64_t nbs = std::min<int64_t>(n_kv/B, nbm);
    if (nbm <= 0) {
        return false;
    }
    ggml_tensor * km = nullptr, * vm = nullptr, * kr = nullptr, * vr = nullptr;
    bool     valid = false;
    uint64_t epoch = 0;
    if (!llama_lsa_mean_get(kb, D*H_kv, Dv*H_kv, nbm, ggml_backend_buffer_get_type(kb->buffer), &km, &vm, &valid, &epoch,
                sj ? inp->lsa_ring_n : 0, &kr, &vr, rot && sj)) {
        return false;
    }
    inp->lsa_n_blk = nbm;

    // F32 [dx*B, n] rows of B cells -> [dx, n] block means
    auto mean_rows = [&](ggml_tensor * x, int64_t dx) {
        const int64_t n = x->ne[1];
        ggml_tensor * xt = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, x, dx, B, n), 1, 0, 2, 3)); // [B, dx, n]
        return ggml_reshape_2d(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, xt), 1.0f/(float) B), dx, n);
    };
    const bool fresh = sj && lsa_ubatch().pos[0] == 0;
    auto refresh = [&](ggml_tensor * base, ggml_tensor * ring, ggml_tensor * x_new, ggml_tensor * mc, int64_t dx) {
        ggml_tensor * w = nullptr;
        if (sj && inp->lsa_ring_rows) {
            // carry ring: gather each block row by row from [carry ring ; this ubatch's rows] (graph scratch). The f16
            // round trip gives the values the legacy ring held, so the means are the same bits.
            const int64_t T  = x_new->ne[2];
            ggml_tensor * xn = ggml_is_contiguous(x_new) ? x_new : ggml_cont(ctx0, x_new);
            xn = ggml_cast(ctx0, ggml_cast(ctx0, ggml_reshape_2d(ctx0, xn, dx, T), GGML_TYPE_F16), GGML_TYPE_F32);
            ggml_tensor * xa = ggml_concat(ctx0, ggml_cast(ctx0, ring, GGML_TYPE_F32), xn, 1); // [dx, n_ring + T]
            ggml_tensor * g  = ggml_get_rows(ctx0, xa, inp->lsa_u_rblk);                      // [dx, n_u*B]
            w = ggml_set_rows(ctx0, mc, mean_rows(ggml_reshape_2d(ctx0, g, dx*B, g->ne[1]/B), dx), inp->lsa_u_blk);
            // carry the ubatch's last m rows; reading them from xa orders this write after the gather's ring read
            const int64_t m = inp->lsa_q_ring->ne[0];
            ggml_tensor * tail = ggml_view_2d(ctx0, xa, dx, m, xa->nb[1], (size_t) (ring->ne[1] + T - m)*xa->nb[1]);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ring, tail, inp->lsa_q_ring));
        } else if (sj) {
            // staging ring: rows = position mod n_ring; a block of this ubatch is whole in the ring (n_ring >= n_ubatch + B),
            // its unwritten tail is stale but such a block is inside the always-kept window until a later ubatch completes
            // it and recomputes it (rejected draft rows are rewritten the same way)
            ggml_tensor * xn = ggml_is_contiguous(x_new) ? x_new : ggml_cont(ctx0, x_new);
            ggml_tensor * wr = ggml_set_rows(ctx0, ring, ggml_reshape_2d(ctx0, xn, dx, x_new->ne[2]), inp->lsa_q_ring);
            ggml_build_forward_expand(gf, wr);
            ggml_tensor * g = ggml_get_rows(ctx0, ggml_view_2d(ctx0, wr, dx*B, ring->ne[1]/B, wr->nb[1]*B, 0), inp->lsa_u_rblk);
            w = ggml_set_rows(ctx0, mc, mean_rows(g, dx), inp->lsa_u_blk); // get_rows gives F32
        } else if (!valid) {
            if (nbs <= 0) {
                return (ggml_tensor *) nullptr;
            }
            // LLAMA_LSA_MEAN_CHUNKS=1 (opt-in): full rebuild in at most 8 chunks of CB >= 16 blocks: only one chunk's F32
            // copy and its permuted copy are live at a time (2*CB*B*dx*4 B instead of 2*nbs*B*dx*4 B, 130 MiB for a 16K MTP
            // draft window); the cap keeps the extra graph nodes bounded on a large non-SJ main cache. Each block's mean is
            // the same ops on the same rows and concat only moves the F32 results, so the means are the bits of one pass.
            // Off (default): one chunk of all nbs blocks, the same graph as before.
            const int64_t CB = llama_lsa_cfg().mean_chunks ? std::max<int64_t>(16, (nbs + 7)/8) : nbs;
            ggml_tensor * m = nullptr;
            for (int64_t c0 = 0; c0 < nbs; c0 += CB) {
                const int64_t nc = std::min<int64_t>(CB, nbs - c0);
                ggml_tensor * x  = ggml_cast(ctx0, ggml_view_2d(ctx0, base, dx*B, nc, base->nb[1]*B, (size_t) c0*B*base->nb[1]),
                                             GGML_TYPE_F32);
                ggml_tensor * mx = mean_rows(x, dx);
                m = m ? ggml_concat(ctx0, m, mx, 1) : mx;
            }
            w = ggml_cpy(ctx0, m, ggml_view_2d(ctx0, mc, dx, nbs, mc->nb[1], 0));
        } else {
            ggml_tensor * g = ggml_get_rows(ctx0, ggml_view_2d(ctx0, base, dx*B, nbm, base->nb[1]*B, 0), inp->lsa_u_blk);
            w = ggml_set_rows(ctx0, mc, mean_rows(g, dx), inp->lsa_u_blk);
        }
        ggml_build_forward_expand(gf, w);
        return ggml_view_2d(ctx0, w, dx, std::max<int64_t>(nbs, 1), mc->nb[1], 0); // ordered after this graph's refresh
    };
    ggml_tensor * kw = refresh(kb, kr, k_new, km, D*H_kv);
    ggml_tensor * vw = refresh(vb, vr, v_new, vm, Dv*H_kv);
    if (!kw || !vw) {
        return false;
    }
    cb(kw, "lsa_kmean", il);
    cb(vw, "lsa_vmean", il);
    if (!valid && (!sj || fresh)) {
        inp->lsa_mean_commit.emplace_back(kb, epoch);
    }
    const int32_t seq0 = llama_lsa_ubatch_seq(&lsa_ubatch());
    if (sj && !valid && !fresh) {
        static uint32_t warned_ep = 0;
        if (llama_lsa_log_once(warned_ep)) {
            LLAMA_LOG_WARN("%s: LSA block means on SJ-KVaRN are stale (restore or K shift); compensation off until a "
                    "sequence restarts at position 0\n", __func__);
        }
        llama_lsa_comp_note(false, seq0, 2, "block means are stale (restore without means, K shift, or a rewind past the staging ring)", false);
        return false;
    }
    if (sj && (nbs > 0 || fresh)) {
        llama_lsa_comp_note(true, seq0, 2, nullptr, fresh);
    }
    *kmv = kw;
    *vmv = vw;
    *nbs_out = nbs;
    return nbs > 0;
}

ggml_tensor * llm_graph_context::build_attn_lsa_comp_fast(
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
                int   il) const {
    const auto & cfg = llama_lsa_cfg();
    const int64_t B  = cfg.block > 1 ? cfg.block : 128;
    const int64_t D  = q->ne[0], H = q->ne[1], T = q->ne[2];
    const int64_t n_kv = lsa_mask->ne[0];
    const int64_t Dv = H_kv > 0 ? vmv->ne[0]/H_kv : 0;
    if (q->ne[3] != 1 || T > cfg.comp_max_t || H_kv <= 0 || H % H_kv != 0 || kmv->ne[0] != D*H_kv || vmv->ne[0] != Dv*H_kv ||
            nbs <= 0 || kmv->ne[1] < nbs || nbs*B > n_kv || lsa_mask->ne[1] != T || ggml_nelements(lsa_mask) != n_kv*T ||
            ggml_nelements(kq_mask) != n_kv*T || o_kept->ne[0] != Dv*H || o_kept->ne[1] != T || lse->ne[0] != H ||
            lse->ne[1] != T) {
        static uint32_t warned_ep = 0;
        if (llama_lsa_log_once(warned_ep)) {
            LLAMA_LOG_WARN("%s: layer %d: LSA fast compensation skipped (unsupported shape/type)\n", __func__, il);
        }
        return nullptr;
    }
    kmv = ggml_view_2d(ctx0, kmv, kmv->ne[0], nbs, kmv->nb[1], 0);
    vmv = ggml_view_2d(ctx0, vmv, vmv->ne[0], nbs, vmv->nb[1], 0);
    ggml_tensor * Kb = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, kmv, D, H_kv, nbs), 0, 2, 1, 3));   // [D, nbs, H_kv]
    ggml_tensor * Vb = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, vmv, Dv, H_kv, nbs), 1, 2, 0, 3));  // [nbs, Dv, H_kv]

    ggml_tensor * qp = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));                                          // [D, T, H]
    ggml_tensor * ks = ggml_mul_mat(ctx0, Kb, qp);                                                                  // [nbs, T, H]
    ggml_prec_set_acc(ks, GGML_PREC_F32);

    ggml_tensor * nd = nullptr;
    if (inp && inp->lsa_comp_nd && inp->lsa_comp_nd_mask == lsa_mask && !cfg.comp_cnt_legacy && !cfg.comp_cnt_exp &&
            inp->lsa_comp_nd->ne[0] == nbs && inp->lsa_comp_nd->ne[1] == T) {
        nd = inp->lsa_comp_nd; // from the block pick (build_attn_lsa_mask_sb): no full-width op here
    } else if (cfg.comp_cnt_legacy) {
        ggml_tensor * mf = ggml_reshape_2d(ctx0, lsa_mask->type == GGML_TYPE_F32 ? lsa_mask : ggml_cast(ctx0, lsa_mask, GGML_TYPE_F32), n_kv, T);
        ggml_tensor * cf = ggml_reshape_2d(ctx0, kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T);
        ggml_tensor * drop = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, cf, 1.0f, 1e29f)),
                                            ggml_step(ctx0, ggml_scale_bias(ctx0, mf, -1.0f, -1e29f)));              // [n_kv, T]
        drop = ggml_cont(ctx0, ggml_view_2d(ctx0, drop, nbs*B, T, drop->nb[1], 0));    // the partial tail is in the window
        nd   = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, drop, B, nbs, T)), nbs, T);
    } else {
        // Both masks hold only 0 / -inf here (no ALiBi, no soft cap; -1e30 fills become -inf in f16 and exp to 0 in
        // f32), and the pick mask is the causal mask plus -inf at dropped cells, so exp() is the 0/1 indicator and
        // dropped = visible - kept, per block. The visible count does not depend on the layer: one per graph. Same
        // integer counts as the legacy build (exact f32 sums of <= B ones): 3 full-width ops per layer instead of 9.
        auto blk_count = [&](ggml_tensor * mask) {
            ggml_tensor * e = ggml_exp(ctx0, ggml_reshape_2d(ctx0, mask, n_kv, T));
            if (e->type != GGML_TYPE_F32) {
                e = ggml_cast(ctx0, e, GGML_TYPE_F32);
            }
            e = nbs*B == n_kv ? ggml_reshape_3d(ctx0, e, B, nbs, T)    // the partial tail is in the window
                              : ggml_cont(ctx0, ggml_view_3d(ctx0, e, B, nbs, T, e->nb[0]*B, e->nb[1], 0));
            return ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, e), nbs, T);
        };
        ggml_tensor * vis = inp ? inp->lsa_comp_vis : nullptr;
        if (vis == nullptr || inp->lsa_comp_vis_src != kq_mask || vis->ne[0] != nbs || vis->ne[1] != T) {
            vis = blk_count(kq_mask);
            cb(vis, "lsa_comp_vis", il);
            if (inp) {
                inp->lsa_comp_vis     = vis;
                inp->lsa_comp_vis_src = kq_mask;
            }
        }
        nd = ggml_sub(ctx0, vis, blk_count(lsa_mask));
    }
    if (cfg.fused_comp && D % 2 == 0 && D <= 512 && H/H_kv <= 8 && q->nb[0] == sizeof(float) && o_kept->nb[0] == sizeof(float) &&
            nd->nb[0] == sizeof(float) && ggml_nrows(nd) == T && ggml_is_contiguous(lse) && kmv->type == GGML_TYPE_F16 &&
            vmv->type == GGML_TYPE_F16) {
        // LLAMA_LSA_FUSED_COMP=1: the chain below as one op (valid-math change: f32 scores instead of the f16-input
        // mul_mat, one exp per stand-in). nd < 0.5 leaves a block out, as the -1e30 offset does below.
        ggml_tensor * out = ggml_lsa_comp_merge(ctx0, o_kept, lse, q, kmv, vmv, nd, (int) H_kv, kq_scale, (float) cfg.comp_bias);
        cb(out, "lsa_comp_out", il);
        static uint32_t logged_ep = 0;
        if (llama_lsa_log_once(logged_ep)) {
            LLAMA_LOG_WARN("%s: LSA fused compensation merge active (layer %d, %lld mean blocks, T %lld)\n", __func__, il,
                    (long long) kmv->ne[1], (long long) T);
        }
        return out;
    }
    ggml_tensor * off  = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, nd, 1.0f, -0.5f)), 1e30f, -1e30f);
    ggml_tensor * bias = ggml_add(ctx0, ggml_scale_bias(ctx0, ggml_log(ctx0, ggml_scale_bias(ctx0, nd, 1.0f, 1e-30f)), 1.0f,
                                  (float) cfg.comp_bias), off);                                                     // [nbs, T]
    ggml_tensor * Lk = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, lse, H, T, 1), 2, 1, 0, 3));        // [1, T, H]
    ggml_tensor * S  = ggml_concat(ctx0, ggml_add(ctx0, ggml_scale(ctx0, ks, kq_scale), bias), Lk, 0);              // [nbs+1, T, H]
    ggml_tensor * P  = ggml_soft_max(ctx0, S);
    ggml_tensor * Ps = ggml_cont(ctx0, ggml_view_3d(ctx0, P, nbs, T, H, P->nb[1], P->nb[2], 0));
    ggml_tensor * alpha = ggml_cont(ctx0, ggml_view_3d(ctx0, P, 1, T, H, P->nb[1], P->nb[2], nbs*ggml_element_size(P)));
    cb(Ps, "lsa_comp_p", il);
    ggml_tensor * os = ggml_mul_mat(ctx0, Vb, Ps);                                                                  // [Dv, T, H]
    os    = ggml_cont(ctx0, ggml_permute(ctx0, os, 0, 2, 1, 3));                                                    // [Dv, H, T]
    alpha = ggml_cont(ctx0, ggml_permute(ctx0, alpha, 0, 2, 1, 3));                                                 // [1, H, T]
    ggml_tensor * ok  = ggml_reshape_3d(ctx0, o_kept, Dv, H, T);
    ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, ok, alpha), os);
    cb(out, "lsa_comp_out", il);
    return ggml_reshape_2d(ctx0, out, Dv*H, T);
}

// LSA index cache (llama-lsa.h, M1a). Valid cache: Wk^T k_cur of the new tokens, rms-normalized, stored to their
// cells with set_rows (the cells K was just stored to). Invalid cache (first graph, or after a prompt-cache
// restore / K shift): the whole [0, n_kv) range is recomputed from the K cache, which already holds the new
// tokens, and the layer is committed as valid when this graph's inputs are set.
ggml_tensor * llm_graph_context::build_attn_lsa_idx(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur,
        ggml_tensor * k,
                int   il) const {
    const auto & cfg = llama_lsa_cfg();
    inp->lsa_smax_cur = nullptr;
    inp->lsa_smin_cur = nullptr;
    inp->lsa_sum_il   = -1;
    if (!cfg.idx_cache || (cfg.mode != llama_lsa_config::KEY && cfg.mode != llama_lsa_config::POS)) {
        return nullptr;
    }
    // LLAMA_LSA_IDX_SKIP_DENSE: a KEEP_FRAC_LAYERS L:1 layer never picks (build_attn_lsa_mask returns before scoring),
    // so its index is not allocated or written. Only without a keep schedule, which can change the fraction per context.
    if (cfg.idx_skip_dense && cfg.keep_sched.empty() && cfg.keep_frac_for(il) >= 1.0) {
        return nullptr;
    }
    if (inp->lsa_q_cell) {
        return build_attn_lsa_idx_sj(inp, k_cur, k, il);
    }
    if (k->ne[3] != 1 || k_cur->ne[3] != 1 || !lsa_k_plain(k->type) || k->nb[2] != k->nb[1]*k->ne[1]) {
        return nullptr;
    }
    const ggml_tensor * kbase = k->view_src ? k->view_src : k;
    if (!kbase->buffer || kbase->ne[2] != 1) {
        return nullptr;
    }
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    const auto buft = ggml_backend_buffer_get_type(kbase->buffer);
    const int64_t d_kv = k->ne[0]*k->ne[1];
    if (!llama_lsa_get_map(il, buft, &wq, &wk) || wk->ne[0] != d_kv || kbase->ne[0] != d_kv) {
        return nullptr;
    }
    const int64_t d_idx = wk->ne[1];
    const int64_t n_kv  = k->ne[2];
    llama_lsa_idx_slot slot;
    if (!llama_lsa_idx_get(kbase, d_idx, buft, &slot)) {
        return nullptr;
    }

    ggml_tensor * w = nullptr;
    if (!slot.valid) {
        ggml_tensor * kv2 = ggml_view_2d(ctx0, k, d_kv, n_kv, k->nb[2], 0);
        ggml_tensor * PT  = ggml_rms_norm(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul_mat(ctx0, kv2, wk))), 1e-12f);
        w = ggml_cpy(ctx0, PT, ggml_view_2d(ctx0, slot.t, d_idx, n_kv, slot.t->nb[1], 0));
        inp->lsa_idx_commit.emplace_back(kbase, slot.epoch);
        cb(w, "lsa_idx_rebuild", il);
    } else {
        ggml_tensor * kc = ggml_is_contiguous(k_cur) ? k_cur : ggml_cont(ctx0, k_cur);
        kc = ggml_reshape_2d(ctx0, kc, d_kv, k_cur->ne[2]);
        if (kc->type != GGML_TYPE_F32) {
            kc = ggml_cast(ctx0, kc, GGML_TYPE_F32);
        }
        ggml_tensor * Pn = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, wk, kc), 1e-12f); // [d_idx, T]
        w = ggml_set_rows(ctx0, slot.t, Pn, inp->get_k_idxs());
        cb(w, "lsa_idx_store", il);
    }
    ggml_build_forward_expand(gf, w);
    build_attn_lsa_sum(inp, slot, w, n_kv, !slot.valid, il);
    return ggml_view_2d(ctx0, w, d_idx, n_kv, w->nb[1], 0);
}

// LSA index cache on SJ-KVaRN. The K cache holds only the sink + ring rows, rotated, and sealed records for the rest,
// so an invalid layer cannot be rebuilt from it: the index cache is (re)validated only by a ubatch whose first row is
// position 0 (a fresh sequence), which writes every row from k_cur like the valid path. Otherwise the layer stays
// dense (returns nullptr) until the next fresh sequence. Rows are cells == positions (lsa_q_cell), not the ring rows
// K is stored to.
ggml_tensor * llm_graph_context::build_attn_lsa_idx_sj(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur,
        ggml_tensor * k,
                int   il) const {
    const auto * mctx_cur = inp->mctx;
    if (k->ne[3] != 1 || k_cur->ne[3] != 1 || !lsa_ubatch().pos) {
        return nullptr;
    }
    const ggml_tensor * kbase = k->view_src ? k->view_src : k;
    if (!kbase->buffer) {
        return nullptr;
    }
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    const auto buft = ggml_backend_buffer_get_type(kbase->buffer);
    const int64_t d_kv = k_cur->ne[0]*k_cur->ne[1];
    if (!llama_lsa_get_map(il, buft, &wq, &wk) || wk->ne[0] != d_kv) {
        return nullptr;
    }
    const int64_t d_idx   = wk->ne[1];
    const int64_t kv_size = mctx_cur->get_kv_size();
    const int64_t n_kv    = inp->lsa_cell_pos->ne[0];
    // LLAMA_LSA_PER_HEAD=1 (key mode): one index key per KV head from that head's slice of the map, stored head-major
    // [d_idx, kv_size, n_head_kv] (4x the shared cache); state save/restore carries the head count
    const auto &  cfg   = llama_lsa_cfg();
    const int64_t n_hkv = k_cur->ne[1];
    const int64_t d_hd  = k_cur->ne[0];
    const bool    ph    = cfg.per_head && cfg.mode == llama_lsa_config::KEY;
    const int64_t n_hc  = ph ? n_hkv : 1;
    llama_lsa_idx_slot slot;
    if (n_kv > kv_size || !llama_lsa_idx_get(kbase, d_idx, buft, &slot, kv_size, n_hc)) {
        return nullptr;
    }
    const bool fresh = lsa_ubatch().pos[0] == 0;
    if (!slot.valid && !fresh) {
        static uint32_t warned_ep = 0;
        static int      n_trace   = 0;
        if (llama_lsa_idx_trace() && n_trace < 256) {
            ++n_trace;
            LLAMA_LOG_WARN("%s: [lsa-trace] stale: layer %d %s at pos %d (epoch %llu)\n", __func__, il, kbase->name,
                    (int) lsa_ubatch().pos[0], (unsigned long long) slot.epoch);
        }
        if (llama_lsa_log_once(warned_ep)) {
            LLAMA_LOG_WARN("%s: LSA index cache on SJ-KVaRN is stale (K shift, or a restored state without the LSA "
                    "suffix); layers stay dense until a sequence restarts at position 0\n", __func__);
        }
        return nullptr;
    }
    ggml_tensor * kc = ggml_is_contiguous(k_cur) ? k_cur : ggml_cont(ctx0, k_cur);
    kc = ggml_reshape_2d(ctx0, kc, d_kv, k_cur->ne[2]);
    if (kc->type != GGML_TYPE_F32) {
        kc = ggml_cast(ctx0, kc, GGML_TYPE_F32);
    }
    ggml_tensor * Pn = nullptr;
    if (ph) {
        // key_j = rms_norm(Wk[j]^T k_j), Wk[j] = rows j*d_head.. of the map (block-diagonal part): [d_idx, T, H]
        ggml_tensor * wk3 = ggml_cont(ctx0, ggml_view_3d(ctx0, wk, d_hd, d_idx, n_hkv, wk->nb[1], d_hd*wk->nb[0], 0));
        ggml_tensor * kc3 = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, kc, d_hd, n_hkv, kc->ne[1]), 0, 2, 1, 3));
        Pn = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, wk3, kc3), 1e-12f); // [d_idx, T, H]: head-major rows as stored
    } else {
        Pn = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, wk, kc), 1e-12f); // [d_idx, T]
    }
    ggml_tensor * w  = ggml_set_rows(ctx0, slot.t, Pn, inp->lsa_q_cell);
    cb(w, "lsa_idx_store_sj", il);
    ggml_build_forward_expand(gf, w);
    build_attn_lsa_sum(inp, slot, w, n_kv, false, il); // per-head: one summary plane per KV head
    if (!slot.valid) {
        inp->lsa_idx_commit.emplace_back(kbase, slot.epoch);
    }
    return ggml_view_3d(ctx0, w, d_idx, n_kv, n_hc, w->nb[1], w->nb[2], 0); // [d_idx, n_kv, n_hc]
}

// LLAMA_LSA_J3: the same stale-query pick as the LLAMA_LSA_SPEC_RECALL logger, for a later layer il, built from what
// exists at the source layer: il's index cache as of the previous ubatch (this ubatch's rows are appended by il itself;
// they sit in the always-read window and are forced below), il's map, the shared mask and positions. The keep budget
// follows build_attn_lsa_mask for il; every guard there that turns il's pick off returns nullptr here.
ggml_tensor * llm_graph_context::build_attn_lsa_j3_units(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * sq,
                int   il,
                int   pct) const {
    const auto & cfg = llama_lsa_cfg();
    const auto * mctx_cur = inp->mctx;
    ggml_tensor * kq_mask = inp->lsa_kq_mask ? inp->lsa_kq_mask : inp->lsa_sub_mask ? inp->lsa_sub_mask : inp->self_attn_mask.raw;
    const int64_t T = sq->ne[2];
    if (!kq_mask || (kq_mask->type != GGML_TYPE_F16 && kq_mask->type != GGML_TYPE_F32) || kq_mask->ne[1] != T ||
            kq_mask->ne[2] != 1 || kq_mask->ne[3] != 1 || T > cfg.max_rows) {
        return nullptr;
    }
    const int64_t n_kv = kq_mask->ne[0];
    if (inp->lsa_cell_pos->ne[0] != n_kv) {
        return nullptr;
    }
    if (cfg.min_pos > 0) {
        bool any = false;
        for (int64_t i = 0; i < T; ++i) {
            any = any || lsa_ubatch().pos[i] >= cfg.min_pos;
        }
        if (!any) {
            return nullptr;
        }
    }
    const double sched = cfg.keep_frac_sched(n_kv);
    if (sched == 0.0) {
        return nullptr;
    }
    const double frac = sched > 0.0 ? cfg.keep_frac_for(il, sched) : cfg.keep_frac_for(il);
    if (frac >= 1.0) {
        return nullptr;
    }
    int64_t keep_lr = frac > 0.0 ? (int64_t) (frac * (double) n_kv) : cfg.keep;
    if (const double cf = llama_lsa_keep_cap_frac(); cf > 0.0 && frac > 0.0 && frac > cf) {
        keep_lr = (int64_t) (cf * (double) n_kv);
    } else if (const int64_t cc = llama_lsa_keep_cap_cells(); cc > 0 && frac <= 0.0) {
        keep_lr = std::min(keep_lr, cc);
    }
    if (n_kv <= cfg.sinks + cfg.window + std::max<int64_t>(keep_lr, 0)) {
        return nullptr;
    }
    const int64_t B = cfg.block;
    const int64_t F = cfg.refine;
    if (B != 128 || (F != 16 && F != 128) || n_kv % B != 0) {
        return nullptr;
    }
    const int64_t nb = n_kv/B;
    const int64_t kc = std::min<int64_t>(nb, (int64_t) std::ceil(cfg.refine_mult*(double) keep_lr/(double) B));
    const int64_t nu = n_kv/F;
    const int64_t ku = std::max<int64_t>(1, std::min<int64_t>(nu, keep_lr/F));

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    const ggml_tensor * kbase = k->view_src ? k->view_src : k;
    if (!kbase->buffer) {
        return nullptr;
    }
    const auto buft = ggml_backend_buffer_get_type(kbase->buffer);
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    const int64_t d_head    = sq->ne[0];
    const int64_t n_head    = sq->ne[1];
    const int64_t n_head_kv = hparams.n_head_kv(il);
    const int64_t d_kv      = d_head*n_head_kv;
    if (n_head_kv <= 0 || n_head % n_head_kv != 0 || !llama_lsa_get_map(il, buft, &wq, &wk) || wk->ne[0] != d_kv) {
        return nullptr;
    }
    const int64_t d_idx   = wk->ne[1];
    const int64_t kv_size = mctx_cur->get_kv_size();
    llama_lsa_idx_slot slot;
    if (n_kv > kv_size || !llama_lsa_idx_get(kbase, d_idx, buft, &slot, kv_size, 1) || !slot.valid) {
        return nullptr; // il rebuilds (or has no) index cache this ubatch: nothing to rank yet
    }
    ggml_tensor * PT  = ggml_view_2d(ctx0, slot.t, d_idx, n_kv, slot.t->nb[1], 0);
    if (!inp->lsa_j3_mk2 || inp->lsa_j3_mk2_src != kq_mask || inp->lsa_j3_mk2->ne[0] != n_kv || inp->lsa_j3_mk2->ne[1] != T) {
        inp->lsa_j3_mk2     = ggml_reshape_2d(ctx0, ggml_is_contiguous(kq_mask) ? kq_mask : ggml_cont(ctx0, kq_mask), n_kv, T);
        inp->lsa_j3_mk2_src = kq_mask;
        inp->lsa_j3_force   = nullptr;
    }
    ggml_tensor * mk2 = inp->lsa_j3_mk2;
    const int64_t g = n_head/n_head_kv;
    ggml_tensor * sq4 = ggml_reshape_4d(ctx0, ggml_cont(ctx0, sq), d_head, g, n_head_kv, T);
    sq4 = ggml_cont(ctx0, ggml_permute(ctx0, sq4, 1, 0, 2, 3));
    ggml_tensor * sqs = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sq4), d_kv, T);
    if (sqs->type != GGML_TYPE_F32) {
        sqs = ggml_cast(ctx0, sqs, GGML_TYPE_F32);
    }
    ggml_tensor * ss = ggml_mul_mat(ctx0, PT, ggml_mul_mat(ctx0, wq, sqs)); // [n_kv, T]
    ggml_tensor * sp = llama_lsa_build_recall_pick(ctx0, ss, mk2, inp->lsa_cell_pos, inp->lsa_q_pos,
            (int) B, (int) F, (float) cfg.block_tau, (int) cfg.window, (int) cfg.sinks,
            (int) std::max<int64_t>(1, kc), (int) ku, pct);
    if (T > 1) {
        // one prefetch per ubatch: the union of the rows' picks
        ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, sp));
        sp = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, st, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nu, 1);
    }
    // sinks and the window are read by every row that sees them; a unit with one such cell counts. The same for
    // every target of this graph (cell positions, query positions, mask, window, sinks, F): built once
    ggml_tensor * force = inp->lsa_j3_force;
    const int64_t force_key[3] = {F, (int64_t) cfg.window, (int64_t) cfg.sinks};
    if (!force || force->ne[0] != nu || inp->lsa_j3_force_cp != inp->lsa_cell_pos || inp->lsa_j3_force_qp != inp->lsa_q_pos ||
            !std::equal(force_key, force_key + 3, inp->lsa_j3_force_key)) {
        ggml_tensor * cp  = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, inp->lsa_cell_pos, n_kv, 1), ss);
        ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, cp, inp->lsa_q_pos), 1.0f, (float) cfg.window - 0.5f));
        ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, cp, -1.0f, (float) cfg.sinks - 0.5f));
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, mk2, GGML_TYPE_F32), 1.0f, 1e29f));
        force = ggml_pool_1d(ctx0, ggml_mul(ctx0, ggml_add(ctx0, win, snk), vis), GGML_OP_POOL_MAX, (int) F, (int) F, 0);
        if (T > 1) {
            ggml_tensor * ft = ggml_cont(ctx0, ggml_transpose(ctx0, force));
            force = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, ft, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nu, 1);
        }
        inp->lsa_j3_force    = force;
        inp->lsa_j3_force_cp = inp->lsa_cell_pos;
        inp->lsa_j3_force_qp = inp->lsa_q_pos;
        std::copy(force_key, force_key + 3, inp->lsa_j3_force_key);
    }
    ggml_tensor * units = ggml_cont(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, sp, force), 1.0f, -0.5f)));
    cb(units, "lsa_j3_units", il);
    return units;
}

void llm_graph_context::build_attn_lsa_j3(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * sq,
                int   il,
                int   pct,
                int   flags,
                int   src,
        llm_graph_input_attn_kv * shadow,
            int64_t   s) const {
    const auto & cfg = llama_lsa_cfg();
    if (!shadow && inp && !inp->lsa_streams.empty()) {
        // LLAMA_LSA_J3_LEAD under LLAMA_LSA_MULTI: one pass per stream (its rows of sq, mask slice, descriptor row,
        // LSA caches and scratch src + 2*stream), exactly the one-stream pass of that stream
        const int64_t ns = (int64_t) inp->lsa_streams.size();
        if (!cfg.j3_lead || (sq && sq->ne[2] % ns != 0)) {
            return;
        }
        for (int64_t u = 0; u < ns; ++u) {
            llm_graph_input_attn_kv * x = inp->lsa_streams[u].get();
            lsa_stream_mask(ctx0, inp, x, u, ns);
            const llama_lsa_stream_scope scope(x->lsa_stream);
            lsa_ub = &x->lsa_sub_ub;
            const int64_t Ts = sq ? sq->ne[2]/ns : 0;
            ggml_tensor * sq_u = sq ? ggml_view_3d(ctx0, sq, sq->ne[0], sq->ne[1], Ts, sq->nb[1], sq->nb[2], u*Ts*sq->nb[2]) : nullptr;
            build_attn_lsa_j3(inp, sq_u, il, pct, flags, src + 2*(int) (u % (GGML_LSA_PREFETCH_SOURCES/2)), x, u);
        }
        lsa_ub = nullptr;
        return;
    }
    llm_graph_input_attn_kv * x = shadow ? shadow : inp;
    if (!cfg.j3 || cfg.mode == llama_lsa_config::OFF || !inp || !inp->self_sj_kvarn_desc || !x->lsa_cell_pos ||
            !x->lsa_q_pos || !x->lsa_q_cell || !lsa_ubatch().pos) {
        return;
    }
    // ns == 1 under LLAMA_LSA_MULTI=1: the one input set addresses its sequence's caches (no-op otherwise)
    const llama_lsa_stream_scope scope(shadow ? llama_lsa_stream() : inp->lsa_stream);
    const auto * mctx_cur = x->mctx;
    ggml_tensor * body = mctx_cur->get_sj_kvarn_body(il);
    uint32_t bits_k = 0, bits_v = 0;
    const int tier = mctx_cur->get_sj_kvarn_layer_tier(il, bits_k, bits_v);
    ggml_tensor * desc = tier == 1 ? inp->self_sj_kvarn_desc_edge : inp->self_sj_kvarn_desc;
    if (!body || !desc) {
        return;
    }
    if (shadow) {
        GGML_ASSERT(desc->ne[1] > s);
        desc = ggml_view_1d(ctx0, desc, desc->ne[0], s*desc->nb[1]); // the stream's descriptor row
    }
    ggml_tensor * units = sq && pct > 0 ? build_attn_lsa_j3_units(x, sq, il, pct) : nullptr;
    flags &= GGML_LSA_PREFETCH_PREV | GGML_LSA_PREFETCH_PROTECT_PREV;
    if (units) {
        flags |= GGML_LSA_PREFETCH_UNITS;
    }
    if (!(flags & (GGML_LSA_PREFETCH_UNITS | GGML_LSA_PREFETCH_PREV))) {
        return;
    }
    ggml_tensor * r = ggml_lsa_prefetch(ctx0, desc, body, units, (int) cfg.refine, flags, src);
    // Pin the op to the GPU backend that owns the body. Left to the scheduler, this consumer-less op lands in a CPU
    // split: the whole body (72 MiB per layer at 100K) is copied to the host every step and the CPU no-op runs, so
    // the slot prefetch never happens (r387diag: 28/54 splits per decode graph vs 2, no pf_hits in r384).
    const ggml_tensor * bb = body->view_src ? body->view_src : body;
    ggml_backend_t owner = nullptr;
    for (int b = 0; bb->buffer && b < ggml_backend_sched_get_n_backends(sched) && !owner; ++b) {
        ggml_backend_t be = ggml_backend_sched_get_backend(sched, b);
        if (be != backend_cpu && ggml_backend_supports_buft(be, ggml_backend_buffer_get_type(bb->buffer)) &&
                ggml_backend_supports_op(be, r)) {
            owner = be;
        }
    }
    if (!owner) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: LLAMA_LSA_J3: no GPU backend owns the body of layer %d; prefetch op not built\n", __func__, il);
        }
        return;
    }
    ggml_backend_sched_set_tensor_backend(sched, r, owner);
    cb(r, "lsa_j3_prefetch", il);
    ggml_build_forward_expand(gf, r);
}

// LLAMA_LSA_BLOCK_SUM (llama-lsa.h): per-block channel max / min of the index keys, F16 [d_idx, kv_size/B] next to the
// index cache. Valid cache: only the blocks of this ubatch's rows are recomputed from the index cache after the store
// (duplicate block ids write identical values). Rebuilt index cache (plain caches): every whole block below n_kv.
// The partial tail block has no summary; it lies inside the always-kept window.
void llm_graph_context::build_attn_lsa_sum(
        llm_graph_input_attn_kv * inp,
        const llama_lsa_idx_slot & slot,
        ggml_tensor * w,
            int64_t   n_kv,
               bool   full,
                int   il) const {
    const auto & cfg = llama_lsa_cfg();
    if (!cfg.block_sum || !slot.smax || !slot.smin || !inp->lsa_q_blk) {
        return;
    }
    const int64_t B   = cfg.block;
    const int64_t d   = slot.t->ne[0];
    const int64_t nbm = slot.smax->ne[1];
    // x F32 [B, d, n] (one row per channel and block) -> the two summary planes [d, n]: channel max / min, or with
    // LLAMA_LSA_BLOCK_SUM=2 channel mean / variance (avg of squares - mean^2, clamped at 0)
    auto planes = [&](ggml_tensor * x, int64_t n, ggml_tensor *& pa, ggml_tensor *& pb) {
        if (cfg.block_sum_mode == 2) {
            ggml_tensor * mu = ggml_pool_1d(ctx0, x, GGML_OP_POOL_AVG, (int) B, (int) B, 0);
            ggml_tensor * m2 = ggml_pool_1d(ctx0, ggml_sqr(ctx0, x), GGML_OP_POOL_AVG, (int) B, (int) B, 0);
            pa = ggml_reshape_2d(ctx0, mu, d, n);
            pb = ggml_reshape_2d(ctx0, ggml_relu(ctx0, ggml_sub(ctx0, m2, ggml_sqr(ctx0, mu))), d, n);
            return;
        }
        pa = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, (int) B, (int) B, 0), d, n);
        pb = ggml_neg(ctx0, ggml_reshape_2d(ctx0,
                ggml_pool_1d(ctx0, ggml_neg(ctx0, x), GGML_OP_POOL_MAX, (int) B, (int) B, 0), d, n));
    };
    inp->lsa_n_blk = nbm;
    inp->lsa_smax_h.clear();
    inp->lsa_smin_h.clear();
    if (slot.t->ne[2] > 1) {
        // per-KV-head cache: valid-cache update only (SJ-KVaRN has no full rebuild), one plane per head; the planes
        // are disjoint views, so each head's mask depends only on its own two writes
        if (full || slot.smax->ne[2] != slot.t->ne[2] || w->ne[2] != slot.t->ne[2]) {
            return;
        }
        for (int64_t h = 0; h < slot.t->ne[2]; ++h) {
            ggml_tensor * vb = ggml_view_2d(ctx0, w, d*B, nbm, w->nb[1]*B, h*w->nb[2]);
            ggml_tensor * g  = ggml_get_rows(ctx0, vb, inp->lsa_q_blk);                      // F32 [d*B, T]
            ggml_tensor * xh = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, g, d, B, g->ne[1]), 1, 0, 2, 3));
            ggml_tensor * mx = nullptr;
            ggml_tensor * mn = nullptr;
            planes(xh, xh->ne[2], mx, mn);
            ggml_tensor * pmax = ggml_view_2d(ctx0, slot.smax, d, nbm, slot.smax->nb[1], h*slot.smax->nb[2]);
            ggml_tensor * pmin = ggml_view_2d(ctx0, slot.smin, d, nbm, slot.smin->nb[1], h*slot.smin->nb[2]);
            ggml_tensor * wmax = ggml_set_rows(ctx0, pmax, mx, inp->lsa_q_blk);
            ggml_tensor * wmin = ggml_set_rows(ctx0, pmin, mn, inp->lsa_q_blk);
            cb(wmax, "lsa_smax_h", il);
            cb(wmin, "lsa_smin_h", il);
            ggml_build_forward_expand(gf, wmax);
            ggml_build_forward_expand(gf, wmin);
            inp->lsa_smax_h.push_back(wmax);
            inp->lsa_smin_h.push_back(wmin);
        }
        inp->lsa_smax_cur = inp->lsa_smax_h.front();
        inp->lsa_smin_cur = inp->lsa_smin_h.front();
        inp->lsa_sum_il   = il;
        return;
    }
    ggml_tensor * x = nullptr; // F32 [B, d, n] one row per (channel, block)
    int64_t nfull = 0;
    if (full) {
        nfull = std::min<int64_t>(n_kv/B, nbm);
        if (nfull <= 0) {
            return;
        }
        ggml_tensor * v = ggml_view_3d(ctx0, w, d, B, nfull, w->nb[1], w->nb[1]*B, 0);
        x = ggml_cpy(ctx0, ggml_permute(ctx0, v, 1, 0, 2, 3), ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, B, d, nfull));
    } else {
        ggml_tensor * vb = ggml_view_2d(ctx0, w, d*B, nbm, w->nb[1]*B, 0);                 // one row per block
        ggml_tensor * g  = ggml_get_rows(ctx0, vb, inp->lsa_q_blk);                          // F32 [d*B, T]
        x = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, g, d, B, g->ne[1]), 1, 0, 2, 3));
    }
    ggml_tensor * mx = nullptr;
    ggml_tensor * mn = nullptr;
    planes(x, x->ne[2], mx, mn);
    ggml_tensor * wmax = nullptr;
    ggml_tensor * wmin = nullptr;
    if (full) {
        wmax = ggml_cpy(ctx0, mx, ggml_view_2d(ctx0, slot.smax, d, nfull, slot.smax->nb[1], 0));
        wmin = ggml_cpy(ctx0, mn, ggml_view_2d(ctx0, slot.smin, d, nfull, slot.smin->nb[1], 0));
        wmax = ggml_view_2d(ctx0, wmax, d, nbm, slot.smax->nb[1], 0); // whole summary, ordered after the rebuild
        wmin = ggml_view_2d(ctx0, wmin, d, nbm, slot.smin->nb[1], 0);
    } else {
        wmax = ggml_set_rows(ctx0, slot.smax, mx, inp->lsa_q_blk);
        wmin = ggml_set_rows(ctx0, slot.smin, mn, inp->lsa_q_blk);
    }
    cb(wmax, "lsa_smax", il);
    cb(wmin, "lsa_smin", il);
    ggml_build_forward_expand(gf, wmax);
    ggml_build_forward_expand(gf, wmin);
    inp->lsa_smax_cur = wmax;
    inp->lsa_smin_cur = wmin;
    inp->lsa_sum_il   = il;
}

ggml_tensor * llm_graph_context::build_attn_lsa_mask_blk(
        const llm_graph_input_attn_kv * inp,
        ggml_tensor * qi,
        ggml_tensor * kq_mask,
            int64_t   n_kv,
            int64_t   keep_lr,
                int   il,
        ggml_tensor * smax,
        ggml_tensor * smin,
               bool   keep_f32) const {
    const auto & cfg = llama_lsa_cfg();
    smax = smax ? smax : inp->lsa_smax_cur;
    smin = smin ? smin : inp->lsa_smin_cur;
    const int64_t B   = cfg.block;
    const int64_t T   = qi->ne[1];
    const int64_t nbs = std::min<int64_t>((n_kv + B - 1)/B, smax->ne[1]);
    const int64_t d   = smax->ne[0];
    if (nbs < 2 || qi->ne[0] != d || inp->lsa_cell_pos->ne[0] != n_kv) {
        return nullptr;
    }
    // block score = upper bound of its cells' index scores: sum_c max(q_c*max_c, q_c*min_c)
    ggml_tensor * vmax = ggml_view_2d(ctx0, smax, d, nbs, smax->nb[1], 0);
    ggml_tensor * vmin = ggml_view_2d(ctx0, smin, d, nbs, smin->nb[1], 0);
    if (cfg.sum_op && cfg.fused_sel && qi->type == GGML_TYPE_F32 && d % 2 == 0 && smax->type == GGML_TYPE_F16 &&
            smin->type == GGML_TYPE_F16 && smax->nb[1] == smin->nb[1]) {
        // LLAMA_LSA_SUM_OP: the fused select scores the blocks itself; the score chain below then has no consumer
        lsa_sum_q = ggml_is_contiguous(qi) ? qi : ggml_cont(ctx0, qi);
        lsa_sum_x = vmax;
        lsa_sum_n = vmin;
    }
    if (cfg.block_sum_mode == 2) {
        // LLAMA_LSA_BLOCK_SUM=2: Gaussian estimate of log sum_{c in block} exp(tau*s_c) - log B with per-channel mean
        // mu and variance v (channels treated as independent): tau*(q.mu) + tau^2/2*(q^2.v)
        const float tau = (float) cfg.block_tau;
        ggml_tensor * Sb = ggml_add(ctx0, ggml_scale(ctx0, ggml_mul_mat(ctx0, vmax, qi), tau),
                                          ggml_scale(ctx0, ggml_mul_mat(ctx0, vmin, ggml_sqr(ctx0, qi)), 0.5f*tau*tau)); // [nbs, T]
        ggml_tensor * mb = build_attn_lsa_mask_sb(inp, Sb, kq_mask, n_kv, keep_lr, il, keep_f32);
        lsa_sum_q = lsa_sum_x = lsa_sum_n = nullptr; // never left for another caller (early returns in mask_sb)
        return mb;
    }
    ggml_tensor * qp = ggml_relu(ctx0, qi);
    ggml_tensor * qn = ggml_sub(ctx0, qi, qp);
    ggml_tensor * Sb = ggml_add(ctx0, ggml_mul_mat(ctx0, vmax, qp), ggml_mul_mat(ctx0, vmin, qn));   // [nbs, T]
    ggml_tensor * mb = build_attn_lsa_mask_sb(inp, Sb, kq_mask, n_kv, keep_lr, il, keep_f32);
    lsa_sum_q = lsa_sum_x = lsa_sum_n = nullptr;
    return mb;
}

// per-graph base of the block/unit masks: kq_mask where always kept (sinks, window, rows before min_pos), kq_mask - 1e30
// elsewhere. Shared by every layer for decode/verify widths (T <= 64), rebuilt per layer at prefill widths.
ggml_tensor * llm_graph_context::build_attn_lsa_blk_base(
        const llm_graph_input_attn_kv * inp, ggml_tensor * kq_mask, int64_t n_kv, int64_t T, int il) const {
    const auto & cfg = llama_lsa_cfg();
    const bool share = T <= 64;
    ggml_tensor * A = share ? inp->lsa_blk_base : nullptr;
    if (A == nullptr || inp->lsa_blk_base_src != kq_mask || A->ne[0] != n_kv || A->ne[1] != T) {
        ggml_tensor * mk = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        mk = ggml_reshape_2d(ctx0, mk, n_kv, T);
        ggml_tensor * dc  = ggml_sub(ctx0, ggml_repeat(ctx0, inp->lsa_cell_pos, mk), inp->lsa_q_pos);
        ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, dc, 1.0f, (float) cfg.window - 0.5f));
        ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_cell_pos, -1.0f, (float) cfg.sinks - 0.5f));
        ggml_tensor * drop = ggml_scale_bias(ctx0, ggml_clamp(ctx0, ggml_add(ctx0, win, snk), 0.0f, 1.0f), -1.0f, 1.0f);
        if (cfg.min_pos > 0) {
            drop = ggml_mul(ctx0, drop, ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_q_pos, 1.0f, 0.5f - (float) cfg.min_pos)));
        }
        A = ggml_cont(ctx0, ggml_add(ctx0, ggml_scale(ctx0, drop, -1e30f), mk));
        cb(A, "lsa_blk_base", il);
        if (share) {
            inp->lsa_blk_base     = A;
            inp->lsa_blk_base_src = kq_mask;
        }
    }
    return A;
}

// LLAMA_LSA_BLKLIST: shape/cache conditions under which the layer's SJ-KVaRN 4/4 stream node can take a pick of B-cell
// blocks (T decode/verify rows, F16 KQ mask, COMP=2 counts available when COMP is on); the caller adds pick sharing
bool llm_graph_context::lsa_blist_ok(int64_t T, int64_t B, const ggml_tensor * kq_mask) const {
    const auto & cfg = llama_lsa_cfg();
    static const int64_t sj_stream_max = std::min<int64_t>(8, getenv("GGML_SJKVARN_DIRECT_STREAM_MAX") ?
            atoi(getenv("GGML_SJKVARN_DIRECT_STREAM_MAX")) : 8);
    const bool share   = T <= 64;
    const bool comp_nd = share && cfg.comp_fast && !cfg.comp_cnt_legacy && !cfg.comp_cnt_exp && T <= cfg.comp_max_t;
    const int  body_t  = sj_kvarn_pending.body ? sj_kvarn_pending.body->op_params[7] : -1;
    // LLAMA_LSA_RESIDENT record pages (CUDA_LSA_Page_) take the mask-gated loader: no CUDA block-list reader serves them, so
    // a block-list node there would fail supports_op and run on the CPU (rr1 r44_ref: 15.8 s/token at 98K, GPU 0%).
    // Strip slots (CUDA_LSA_Strip_) keep the block list: the stream kernel reads them (GGML_CUDA_LSA_STRIP_STREAM=0 = old route).
    static const bool strip_stream = !(getenv("GGML_CUDA_LSA_STRIP_STREAM") && atoi(getenv("GGML_CUDA_LSA_STRIP_STREAM")) == 0);
    const ggml_backend_buffer_t body_buf = sj_kvarn_pending.body ? sj_kvarn_pending.body->buffer : nullptr;
    const char * body_bn = body_buf ? ggml_backend_buft_name(ggml_backend_buffer_get_type(body_buf)) : "";
    if (strncmp(body_bn, "CUDA_LSA_", 9) == 0 && !(strip_stream && strncmp(body_bn, "CUDA_LSA_Strip_", 15) == 0)) {
        static bool warned = false;
        if (!warned && cfg.blk_list) {
            warned = true;
            LLAMA_LOG_WARN("%s: LLAMA_LSA_RESIDENT layers use the F16 selection mask, not LLAMA_LSA_BLKLIST\n", __func__);
        }
        return false;
    }
    return share && sj_kvarn_pending.body && T <= sj_stream_max && sj_kvarn_pending.bits_k == 4 &&
        sj_kvarn_pending.bits_v == 4 && body_t != GGML_TYPE_I16 && body_t != GGML_TYPE_TURBO4_0 && !cfg.sj_compact &&
        kq_mask->type == GGML_TYPE_F16 && B >= cfg.blist_min_unit && (B & (B - 1)) == 0 && (!cfg.comp || comp_nd) && fa_sel_rows == nullptr;
}

// LLAMA_LSA_BLKLIST: per-graph F16 mask M3 (kq_mask where always kept, -inf where hidden, GGML_LSA_MASK_SEL at visible
// droppable cells) from the per-graph base A; the same M3 serves block and unit picks of any size
ggml_tensor * llm_graph_context::build_attn_lsa_blk_m3(
        const llm_graph_input_attn_kv * inp, ggml_tensor * A, ggml_tensor * kq_mask, int64_t n_kv, int64_t T, int il) const {
    ggml_tensor * m3 = inp->lsa_blk_m3;
    if (m3 == nullptr || inp->lsa_blk_m3_src != A || m3->ne[0] != n_kv || m3->ne[1] != T) {
        ggml_tensor * mk = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        ggml_tensor * vd = ggml_mul(ctx0, ggml_exp(ctx0, ggml_reshape_2d(ctx0, mk, n_kv, T)),
                                          ggml_step(ctx0, ggml_scale_bias(ctx0, A, -1.0f, -1.0f))); // 1 = visible droppable
        // A x (1 - vd) + SEL x vd: vd = 1 only where A = -1e30 (finite), so no inf x 0
        m3 = ggml_add(ctx0, ggml_mul(ctx0, A, ggml_scale_bias(ctx0, vd, -1.0f, 1.0f)), ggml_scale(ctx0, vd, GGML_LSA_MASK_SEL));
        m3 = ggml_reshape_4d(ctx0, ggml_cast(ctx0, m3, GGML_TYPE_F16), n_kv, T, 1, 1);
        cb(m3, "lsa_kq_mask_m3", il);
        inp->lsa_blk_m3     = m3;
        inp->lsa_blk_m3_src = A;
    }
    return m3;
}

// LLAMA_LSA_REFINE=F + LLAMA_LSA_FUSED_SEL=1: mask from a 0/1 selection of U-cell units (selu F32 [n_kv/U, T]), kept
// = selected unit, sink or window, causal mask added back; COMP=2 gets the dropped cells per B-cell block,
// nd_b = sum over the block's units of (1 - sel_u) x D_u (D_u = visible droppable cells of the unit, once per graph).
// nullptr if the shapes do not fit (the caller falls back to the graph refine chain).
ggml_tensor * llm_graph_context::build_attn_lsa_mask_units(
        const llm_graph_input_attn_kv * inp, ggml_tensor * selu, int64_t U, ggml_tensor * kq_mask, int64_t n_kv,
        int il, bool shared_pick) const {
    const auto & cfg = llama_lsa_cfg();
    const int64_t B  = cfg.block;
    const int64_t nu = selu->ne[0];
    const int64_t T  = selu->ne[1];
    lsa_fa_sel = nullptr;
    lsa_fa_m3  = nullptr;
    lsa_fa_vk  = false;
    lsa_fa_vk_n = 0;
    if (cfg.blk_mask_legacy || hparams.f_max_alibi_bias != 0.0f || U < 1 || U*nu != n_kv || B % U != 0 || n_kv % B != 0) {
        return nullptr;
    }
    const bool share = T <= 64;
    ggml_tensor * A = build_attn_lsa_blk_base(inp, kq_mask, n_kv, T, il);
    const bool blist_on = cfg.blk_list && (shared_pick || T == 1) && lsa_blist_ok(T, U, kq_mask);
    if (cfg.stats && il >= 0 && il < 256) {
        // LLAMA_LSA_STATS on the unit pick (it returns before make_mask's stats): kept share of the middle, and the
        // middle 16-cell strips / 128-cell records some row keeps (what the SJ decode kernel's skip can drop), from
        // the pick the attention applies (row 0 for every row on the block list)
        ggml_tensor * mk  = ggml_reshape_2d(ctx0, kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32), n_kv, T);
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, mk, 1.0f, 1e29f));                  // visible
        ggml_tensor * dv  = ggml_mul(ctx0, vis, ggml_step(ctx0, ggml_scale_bias(ctx0, A, -1.0f, -1.0f))); // visible droppable
        ggml_tensor * sp  = blist_on && T > 1 ? ggml_repeat(ctx0, ggml_view_2d(ctx0, selu, nu, 1, selu->nb[1], 0), selu) : selu;
        ggml_tensor * sc  = U == 1 ? sp : ggml_reshape_2d(ctx0,
                ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, sp), 1, nu, T), U, nu, T, 1), n_kv, T);
        ggml_tensor * kd  = ggml_mul(ctx0, dv, sc);                                                    // kept droppable
        ggml_tensor * s_v = ggml_sum_rows(ctx0, vis);
        ggml_tensor * bon = ggml_sub(ctx0, s_v, ggml_sum_rows(ctx0, dv));                              // always-kept visible
        ggml_tensor * st  = ggml_concat(ctx0, ggml_concat(ctx0, ggml_add(ctx0, bon, ggml_sum_rows(ctx0, kd)), s_v, 0), bon, 0);
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st, lsa_stats_op, 1, (void *) (intptr_t) il));
        auto uni = [&](ggml_tensor * x) { // [n_kv, T] 0/1 -> [n_kv] 0/1, union over rows
            ggml_tensor * u = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, x)));
            return ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_reshape_1d(ctx0, u, n_kv), 1.0f, -0.5f));
        };
        auto cnt = [&](ggml_tensor * u, int64_t Bc) { // live blocks of Bc cells
            const int64_t nbc = (n_kv + Bc - 1)/Bc;
            ggml_tensor * up = nbc*Bc > n_kv ? ggml_pad(ctx0, u, (int) (nbc*Bc - n_kv), 0, 0, 0) : u;
            return ggml_sum(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0,
                    ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, up, Bc, nbc)), 1.0f, -0.5f)));
        };
        ggml_tensor * uk = uni(kd);
        ggml_tensor * uv = uni(dv);
        ggml_tensor * st2 = ggml_concat(ctx0, ggml_concat(ctx0, cnt(uk, 16), cnt(uv, 16), 0),
                ggml_concat(ctx0, cnt(uk, 128), cnt(uv, 128), 0), 0);                                  // [4]
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st2, lsa_stats_blk_op, 1, (void *) (intptr_t) il));
    }
    // LLAMA_LSA_BLKLIST: a unit pick shared by all rows (REFINE_SH or one row) goes to the SJ-KVaRN stream node as a block
    // list with block = U (power of 2, >= 16; >= 1 with LLAMA_LSA_BLKLIST_FINE=1), on the per-graph M3, so no full-width op runs per layer. With
    // LLAMA_LSA_VK the node takes a split plane (B-block counts for the extra keys, then the unit pick), else COMP=2 merges.
    if (blist_on) {
        ggml_tensor * m3 = build_attn_lsa_blk_m3(inp, A, kq_mask, n_kv, T, il);
        lsa_fa_sel = ggml_view_1d(ctx0, selu, nu, 0); // row 0: every row holds the same pick
        lsa_fa_blk = (int32_t) U;
        lsa_fa_m3  = m3;
        if (cfg.comp) {
            ggml_tensor * D = inp->lsa_unit_dcnt;
            if (D == nullptr || inp->lsa_unit_dcnt_src != A || D->ne[0] != nu || D->ne[1] != T) {
                ggml_tensor * mk = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
                ggml_tensor * vd = ggml_mul(ctx0, ggml_exp(ctx0, ggml_reshape_2d(ctx0, mk, n_kv, T)),
                                                  ggml_step(ctx0, ggml_scale_bias(ctx0, A, -1.0f, -1.0f)));
                D = U == 1 ? vd : ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, vd, U, nu, T)), nu, T);
                cb(D, "lsa_unit_dcnt", il);
                inp->lsa_unit_dcnt     = D;
                inp->lsa_unit_dcnt_src = A;
            }
            // counts from row 0's pick, the one the node applies to every row (rows can differ at the window edge);
            // row 0 is broadcast over the rows by the mul (same values as repeating it, without the [nu, T] copies)
            ggml_tensor * s0  = T == 1 || cfg.blist_lean ? ggml_view_2d(ctx0, selu, nu, 1, selu->nb[1], 0) :
                ggml_repeat(ctx0, ggml_view_2d(ctx0, selu, nu, 1, selu->nb[1], 0), selu);
            ggml_tensor * ndu = ggml_mul(ctx0, D, ggml_scale_bias(ctx0, s0, -1.0f, 1.0f));             // [nu, T]
            ggml_tensor * ndb = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, ndu, B/U, n_kv/B, T)),
                                                n_kv/B, T);                                              // [nbs, T]
            if (cfg.vk && lsa_vk_km && lsa_vk_vm && lsa_vk_nbs == n_kv/B) {
                // LLAMA_LSA_VK with units: plane [nbs + nu, T] = (1 + dropped cells per mean block, unit pick); the
                // node resolves the mask by unit (row 0) and adds one extra key per block with dropped cells
                lsa_fa_sel  = ggml_concat(ctx0, ggml_scale_bias(ctx0, ndb, 1.0f, 1.0f), selu, 0);
                cb(lsa_fa_sel, "lsa_vk_plane_u", il);
                lsa_fa_vk   = true;
                lsa_fa_vk_n = (int32_t) (n_kv/B);
                static uint32_t logged_vku = 0;
                if (llama_lsa_log_once(logged_vku)) {
                    LLAMA_LOG_WARN("%s: LSA compensation as virtual keys with the unit pick active (LLAMA_LSA_VK=1, "
                            "layer %d, %lld blocks, %lld units of %lld, T %lld)\n", __func__, il, (long long) (n_kv/B),
                            (long long) nu, (long long) U, (long long) T);
                }
            } else {
                inp->lsa_comp_nd      = ndb;
                inp->lsa_comp_nd_mask = m3;
            }
        }
        static uint32_t logged_ub = 0;
        if (llama_lsa_log_once(logged_ub)) {
            LLAMA_LOG_WARN("%s: LSA unit pick as block list to the SJ-KVaRN kernel active (LLAMA_LSA_BLKLIST=1, layer %d, "
                    "%lld units of %lld, T %lld)\n", __func__, il, (long long) nu, (long long) U, (long long) T);
        }
        return m3;
    }
    ggml_tensor * m = ggml_add(ctx0, ggml_reshape_3d(ctx0, A, U, nu, T),
                                     ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, selu, 1e30f, 0.0f), 1, nu, T));
    m = ggml_clamp(ctx0, ggml_reshape_2d(ctx0, m, n_kv, T), -INFINITY, 0.0f);
    if (kq_mask->type != GGML_TYPE_F32) {
        m = ggml_cast(ctx0, m, kq_mask->type);
    }
    m = ggml_reshape_4d(ctx0, m, n_kv, T, 1, 1);
    cb(m, "lsa_kq_mask_unit", il);
    if (share && cfg.comp_fast && !cfg.comp_cnt_legacy && !cfg.comp_cnt_exp && T <= cfg.comp_max_t) {
        ggml_tensor * D = inp->lsa_unit_dcnt;
        if (D == nullptr || inp->lsa_unit_dcnt_src != A || D->ne[0] != nu || D->ne[1] != T) {
            ggml_tensor * mk = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
            ggml_tensor * vd = ggml_mul(ctx0, ggml_exp(ctx0, ggml_reshape_2d(ctx0, mk, n_kv, T)),
                                              ggml_step(ctx0, ggml_scale_bias(ctx0, A, -1.0f, -1.0f)));
            D = U == 1 ? vd : ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, vd, U, nu, T)), nu, T);
            cb(D, "lsa_unit_dcnt", il);
            inp->lsa_unit_dcnt     = D;
            inp->lsa_unit_dcnt_src = A;
        }
        ggml_tensor * ndu = ggml_mul(ctx0, D, ggml_scale_bias(ctx0, selu, -1.0f, 1.0f));               // [nu, T]
        inp->lsa_comp_nd      = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, ndu, B/U, n_kv/B, T)),
                                                n_kv/B, T);                                              // [nbs, T]
        inp->lsa_comp_nd_mask = m;
    }
    return m;
}

ggml_tensor * llm_graph_context::build_attn_lsa_mask_sb(
        const llm_graph_input_attn_kv * inp,
        ggml_tensor * Sb,
        ggml_tensor * kq_mask,
            int64_t   n_kv,
            int64_t   keep_lr,
                int   il,
               bool   keep_f32,
        ggml_tensor * S_cells,
              float   tau) const {
    const auto & cfg = llama_lsa_cfg();
    const bool cfg_shared = cfg.shared && lsa_one_seq(lsa_ubatch(), il); // rows of other sequences never share a pick
    const bool cfg_union  = cfg.shared_union && lsa_one_seq(lsa_ubatch(), il); // SHARED=3: per-row picks, OR-ed below
    const int64_t B   = cfg.block;
    const bool cells  = Sb == nullptr; // fused op only: the op scores the blocks from the cell scores
    lsa_fa_sel = nullptr; // a pick left by an earlier layer never reaches this layer's FA node
    lsa_fa_m3  = nullptr;
    lsa_fa_vk  = false;
    lsa_fa_vk_n = 0;
    const int64_t T0  = cells ? (S_cells ? S_cells->ne[1] : 1) : Sb->ne[1];
    const bool smax   = cfg_shared && cfg.shared_max && T0 > 1; // SHARED=2: fused op mode 2, or the chain below
    const bool smax_chain = smax && cfg.smax_chain;              // LLAMA_LSA_SMAX_CHAIN=1: old graph chain (A/B)
    const int  shmode = cfg_shared && T0 > 1 ? (cfg.shared_max ? 2 : 1) : 0;
    if (cells && (S_cells == nullptr || !cfg.fused_sel || smax_chain || S_cells->ne[0] != n_kv || S_cells->type != GGML_TYPE_F32)) {
        return nullptr;
    }
    const int64_t nbs = cells ? (n_kv + B - 1)/B : Sb->ne[0];
    const int64_t T   = cells ? S_cells->ne[1] : Sb->ne[1];
    if (B <= 1 || nbs < 2 || (!cells && (Sb->ne[2] != 1 || Sb->type != GGML_TYPE_F32)) || inp->lsa_cell_pos->ne[0] != n_kv) {
        return nullptr;
    }
    // LLAMA_LSA_FUSED_SEL=1: eligibility, top-k and the 0/1 selection in one op (same selected blocks: an ineligible
    // block the top-k path could take changes neither the mask nor the dropped counts). sel01 = 1 where selected.
    ggml_tensor * sel01 = nullptr;
    const int64_t kb = std::max<int64_t>(1, std::min<int64_t>(nbs, keep_lr/B));
    // LLAMA_LSA_PICK_EVERY=N: reuse this layer's last pick (one 0/1 plane for all rows, the union of the pick's rows)
    // for up to N - 1 graphs. The scoring above then has no consumer and drops out of the graph.
    const ggml_tensor * pick_k = lsa_pick_kbase;
    lsa_pick_kbase = nullptr;
    ggml_tensor * sum_q = lsa_sum_q, * sum_x = lsa_sum_x, * sum_n = lsa_sum_n; // LLAMA_LSA_SUM_OP
    lsa_sum_q = lsa_sum_x = lsa_sum_n = nullptr;
    ggml_tensor * pick_store = nullptr; // set on a pick graph: write the union here
    bool pick_k_reused = false;         // sel01 is a reused pick: one plane for all rows
    // one sequence only: rows of different sequences must not share a pick, and a pick is reused only by its own sequence
    if (pick_k && cfg.pick_every > 1 && lsa_ubatch().pos && (int64_t) lsa_ubatch().n_tokens == T && B*nbs <= n_kv && pick_k->buffer &&
            lsa_ubatch().n_seqs_unq == 1 && lsa_ubatch().seq_id_unq) {
        const int32_t seq   = lsa_ubatch().seq_id_unq[0];
        const int64_t n_cap = inp->mctx->get_kv_size()/B + 1;
        const int64_t pos0  = lsa_ubatch().pos[0];
        const int64_t edge  = (int64_t) std::floor(((double) lsa_ubatch().pos[T - 1] - (double) cfg.window + 1.0)/(double) B);
        llama_lsa_pick_slot ps;
        if (llama_lsa_pick_get(pick_k, n_cap, ggml_backend_buffer_get_type(pick_k->buffer), pos0, edge, nbs, seq, &ps)) {
            inp->lsa_pick_commit.push_back({ pick_k, ps.gen, ps.reuse, pos0, edge, nbs, seq });
            if (ps.reuse) {
                pick_k_reused = true;
                ggml_tensor * v = ggml_view_2d(ctx0, ps.sel, nbs, 1, ps.sel->nb[0]*nbs, 0);
                sel01 = ggml_repeat(ctx0, v, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nbs, T));
                cb(sel01, "lsa_block_sel_reuse", il);
                static uint32_t logged_pr = 0;
                if (llama_lsa_log_once(logged_pr)) {
                    LLAMA_LOG_WARN("%s: LSA block pick reused (LLAMA_LSA_PICK_EVERY=%lld, layer %d, %lld blocks)\n", __func__,
                            (long long) cfg.pick_every, il, (long long) nbs);
                }
            } else {
                pick_store = ps.sel;
            }
        }
    } else if (pick_k && cfg.pick_every > 1 && lsa_ubatch().n_seqs_unq > 1) {
        inp->lsa_pick_commit.push_back({ pick_k, 0, false, 0, 0, 0, -1, true }); // multi-sequence graph: expire the pick
    }
    // visible droppable cells per block and row, D [nbs, T], once per graph (layer independent): kq_mask 0 and not
    // sink / window / before min_pos. Used by COMP=2 counts and by LLAMA_LSA_ELIG eligibility.
    auto blk_dcnt = [&](ggml_tensor * A) -> ggml_tensor * {
        ggml_tensor * D = inp->lsa_blk_dcnt;
        if (D == nullptr || inp->lsa_blk_dcnt_src != A || D->ne[0] != nbs || D->ne[1] != T) {
            // A + 1e30 is 0 at visible droppable cells, +1e30 at always-kept visible ones, -inf where hidden;
            // exp(-(A + 1e30)^2) would be fragile, so: visible = exp(kq_mask), droppable = step(-A - 1)
            ggml_tensor * mk  = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
            ggml_tensor * vd  = ggml_mul(ctx0, ggml_exp(ctx0, ggml_reshape_2d(ctx0, mk, n_kv, T)),
                                               ggml_step(ctx0, ggml_scale_bias(ctx0, A, -1.0f, -1.0f)));
            D = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, vd, B, nbs, T)), nbs, T);
            cb(D, "lsa_blk_dcnt", il);
            inp->lsa_blk_dcnt     = D;
            inp->lsa_blk_dcnt_src = A;
        }
        return D;
    };
    // LLAMA_LSA_ELIG=1: eligibility from D (fix for the plain cache, where a block's first cell need not be its
    // oldest position: holes after seq_rm, several sequences). The fused ops test only cell_pos[b*B], so they get a
    // stand-in position per block: sinks - B + 1 (outside the sinks and, for every row with droppable cells, outside
    // its window: such a row has q_pos >= sinks + window) where any row has a droppable cell, else 1e9 (ineligible as
    // future). Written as e*c + (1 - e)*1e9 so both values are exact in f32.
    ggml_tensor * elig_D    = nullptr;
    ggml_tensor * cpos_sel  = inp->lsa_cell_pos;
    if (!sel01 && cfg.elig_drop && !cfg.blk_mask_legacy && hparams.f_max_alibi_bias == 0.0f && B*nbs == n_kv && n_kv % B == 0) {
        elig_D = blk_dcnt(build_attn_lsa_blk_base(inp, kq_mask, n_kv, T, il));
        ggml_tensor * Dt = ggml_cont(ctx0, ggml_transpose(ctx0, elig_D));                                // [T, nbs]
        ggml_tensor * e  = ggml_step(ctx0, ggml_scale_bias(ctx0,
                ggml_pool_1d(ctx0, Dt, GGML_OP_POOL_MAX, (int) T, (int) T, 0), 1.0f, -0.5f));             // [1, nbs]
        ggml_tensor * p  = ggml_add(ctx0, ggml_scale(ctx0, e, (float) (cfg.sinks - B + 1)),
                                          ggml_scale(ctx0, ggml_scale_bias(ctx0, e, -1.0f, 1.0f), 1e9f));
        ggml_tensor * z  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, B - 1, nbs), 0.0f);
        cpos_sel = ggml_reshape_1d(ctx0, ggml_concat(ctx0, p, z, 0), n_kv);                              // p at b*B
        cb(cpos_sel, "lsa_elig_pos", il);
        static uint32_t logged_el = 0;
        if (llama_lsa_log_once(logged_el)) {
            LLAMA_LOG_WARN("%s: LSA block eligibility from droppable cells (LLAMA_LSA_ELIG=1, layer %d)\n", __func__, il);
        }
    }
    if (sel01) {
        // reused pick (LLAMA_LSA_PICK_EVERY)
    } else if (cells) {
        ggml_tensor * Sc = ggml_is_contiguous(S_cells) ? S_cells : ggml_cont(ctx0, S_cells);
        ggml_tensor * mk = ggml_reshape_2d(ctx0, ggml_is_contiguous(kq_mask) ? kq_mask : ggml_cont(ctx0, kq_mask), n_kv, T);
        sel01 = lsa_slice(lsa_sticky(ggml_lsa_block_select_cells(ctx0, Sc, mk, cpos_sel, inp->lsa_q_pos, (int) B, tau,
                (int) cfg.window, (int) cfg.sinks, (int) kb, shmode), sj_kvarn_pending.body, sj_kvarn_pending.desc, 1), B);
    } else if (cfg.fused_sel && !smax_chain && Sb->ne[1] == inp->lsa_q_pos->ne[1] && inp->lsa_cell_pos->ne[0] >= B*nbs) {
        if (sum_q && sum_q->ne[1] == T && sum_x->ne[1] >= nbs) {
            sel01 = lsa_slice(lsa_sticky(ggml_lsa_block_select_sum(ctx0, sum_q, sum_x, sum_n, cpos_sel, inp->lsa_q_pos, (int) nbs, (int) B,
                    (float) cfg.block_tau, cfg.block_sum_mode, (int) cfg.window, (int) cfg.sinks, (int) kb, shmode),
                    sj_kvarn_pending.body, sj_kvarn_pending.desc, 1), B);
            static uint32_t logged_so = 0;
            if (llama_lsa_log_once(logged_so)) {
                LLAMA_LOG_WARN("%s: LSA summary scores inside the block select (LLAMA_LSA_SUM_OP=1, mode %d, layer %d)\n",
                        __func__, cfg.block_sum_mode, il);
            }
        } else {
            ggml_tensor * Sc = ggml_is_contiguous(Sb) ? Sb : ggml_cont(ctx0, Sb);
            cb(Sc, "lsa_block_score", il);
            sel01 = lsa_slice(lsa_sticky(ggml_lsa_block_select(ctx0, Sc, cpos_sel, inp->lsa_q_pos, (int) B, (int) cfg.window,
                    (int) cfg.sinks, (int) kb, shmode), sj_kvarn_pending.body, sj_kvarn_pending.desc, 1), B);
        }
    }
    if (sel01) {
        cb(sel01, "lsa_block_sel", il);
        static uint32_t logged_ep = 0;
        if (llama_lsa_log_once(logged_ep)) {
            LLAMA_LOG_WARN("%s: LSA fused block select active (layer %d, %lld of %lld blocks, T %lld, shared %d)\n", __func__, il,
                    (long long) kb, (long long) nbs, (long long) T, shmode);
        }
    }
    if (!sel01 && smax) {
        // SHARED=2: one pick for the ubatch (the record skip reads one kb-block set, not the union of T picks),
        // block rank = max over rows of the row-normalized log share, Sb - logsumexp_row(Sb): a block one verify
        // row needs strongly stays in even if the other rows ignore it (the summed log scores of SHARED=1 act like
        // a product of shares and drop such blocks). Masked/empty blocks sit at ~-1e30 and stay ineligible below.
        ggml_tensor * Sc = ggml_is_contiguous(Sb) ? Sb : ggml_cont(ctx0, Sb);
        ggml_tensor * bm = ggml_pool_1d(ctx0, Sc, GGML_OP_POOL_MAX, (int) nbs, (int) nbs, 0);        // [1, T]
        ggml_tensor * ls = ggml_add(ctx0, ggml_log(ctx0, ggml_sum_rows(ctx0,
                                ggml_exp(ctx0, ggml_sub(ctx0, Sc, ggml_repeat(ctx0, bm, Sc))))), bm); // [1, T]
        ggml_tensor * Sn = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_sub(ctx0, Sc, ggml_repeat(ctx0, ls, Sc)))); // [T, nbs]
        ggml_tensor * St = ggml_pool_1d(ctx0, Sn, GGML_OP_POOL_MAX, (int) T, (int) T, 0);           // [1, nbs]
        Sb = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, St, nbs, 1), Sc);
    } else if (!sel01 && cfg_shared && T > 1) {
        ggml_tensor * St = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, Sb)));          // [1, nbs]
        Sb = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, St, nbs, 1), Sb);
    }
    ggml_tensor * topb = nullptr;
    if (!sel01) {
        cb(Sb, "lsa_block_score", il);
        // a block wholly inside the window (or in the future, or empty: cell_pos 1e9) or inside the sinks is kept or
        // masked anyway, so it may not take a pick; its first cell's position decides
        if (elig_D && elig_D->ne[0] == Sb->ne[0] && elig_D->ne[1] == Sb->ne[1]) {
            // LLAMA_LSA_ELIG=1: per row, a block with no visible droppable cell may not take a pick
            Sb = ggml_add(ctx0, Sb, ggml_scale(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, elig_D, -1.0f, 0.5f)), -1e30f));
        } else {
        ggml_tensor * bpos = ggml_cont(ctx0, ggml_view_2d(ctx0, inp->lsa_cell_pos, 1, nbs, B*inp->lsa_cell_pos->nb[0], 0));
        bpos = ggml_reshape_2d(ctx0, bpos, nbs, 1);
        ggml_tensor * dq    = ggml_sub(ctx0, ggml_repeat(ctx0, bpos, Sb), inp->lsa_q_pos);                // [nbs, T]
        ggml_tensor * inwin = ggml_step(ctx0, ggml_scale_bias(ctx0, dq, 1.0f, (float) cfg.window - 0.5f));
        ggml_tensor * insnk = ggml_step(ctx0, ggml_scale_bias(ctx0, bpos, -1.0f, (float) (cfg.sinks - B) + 0.5f));
        Sb = ggml_add(ctx0, Sb, ggml_scale(ctx0, ggml_add(ctx0, inwin, insnk), -1e30f));
        }
        topb = ggml_top_k(ctx0, Sb, (int) kb);                                                           // I32 [kb, T]
        cb(topb, "lsa_top_blocks", il);
    }
    // block selection as a [1, nbs, T] plane: value `on` where selected, `off` elsewhere
    auto sel_plane = [&](float on, float off) -> ggml_tensor * {
        if (sel01) {
            return ggml_reshape_4d(ctx0, ggml_scale_bias(ctx0, sel01, on - off, off), 1, nbs, T, 1);
        }
        ggml_tensor * p = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, nbs, T, 1), off);
        return ggml_set_rows(ctx0, p, ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, kb, T, 1), on), topb);
    };
    if (pick_store) {
        // pick graph: union over rows -> [nbs], zero-padded to the cache length so later, wider graphs read 0 for blocks
        // that did not exist at pick time (they are newer than the window edge, so kept anyway)
        ggml_tensor * s01 = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, sel_plane(1.0f, 0.0f), nbs, T))); // [T, nbs]
        ggml_tensor * u   = ggml_reshape_1d(ctx0, ggml_pool_1d(ctx0, s01, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nbs);
        if (pick_store->ne[0] > nbs) {
            u = ggml_pad(ctx0, u, (int) (pick_store->ne[0] - nbs), 0, 0, 0);
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, u, pick_store));
    }
    if (cfg.stats && il >= 0 && il < 256) {
        // LLAMA_LSA_STATS on the block pick: blocks picked per row vs blocks picked by any row of the ubatch (the SJ
        // record skip walks the union, so union/per-row is what a verify ubatch pays over a single-row pick)
        ggml_tensor * s01 = ggml_reshape_2d(ctx0, sel_plane(1.0f, 0.0f), nbs, T);
        ggml_tensor * u   = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, s01)));            // [1, nbs]
        ggml_tensor * uc  = ggml_sum(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, u, 1.0f, -0.5f)));
        ggml_tensor * st  = ggml_concat(ctx0, ggml_concat(ctx0, ggml_sum(ctx0, s01), uc, 0),
                ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), (float) T), 0);             // [3]
        ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, st, lsa_stats_union_op, 1, (void *) (intptr_t) il));
    }
    // LLAMA_LSA_SHARED=3: OR the rows' own picks into one plane for all rows (a reused pick already is one);
    // after the stats op, which counts the per-row picks against their union
    bool unioned = false;
    if (cfg_union && T > 1 && !pick_k_reused) {
        ggml_tensor * s01 = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, sel_plane(1.0f, 0.0f), nbs, T))); // [T, nbs]
        ggml_tensor * u   = ggml_reshape_2d(ctx0, ggml_pool_1d(ctx0, s01, GGML_OP_POOL_MAX, (int) T, (int) T, 0), nbs, 1);
        sel01   = ggml_repeat(ctx0, u, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nbs, T));
        cb(sel01, "lsa_block_sel_union", il);
        unioned = true;
        static uint32_t logged_un = 0;
        if (llama_lsa_log_once(logged_un)) {
            LLAMA_LOG_WARN("%s: LSA union of per-row block picks (LLAMA_LSA_SHARED=3, layer %d, %lld per row of %lld blocks, T %lld)\n",
                    __func__, il, (long long) kb, (long long) nbs, (long long) T);
        }
    }
    // Shared base mask: sinks, window, min_pos and the causal mask do not depend on the layer, so they are built once
    // per graph (A = kq_mask where always kept, kq_mask - 1e30 elsewhere). kq_mask holds only 0 / -inf here (no ALiBi),
    // so a layer's mask is clamp(A + 1e30 x selected block, -inf, 0), broadcast over the block's cells: 2 full-width
    // ops per layer instead of ~13. Same f16 mask bits as the per-layer build below (+0 kept, -inf dropped/hidden).
    if (!cfg.blk_mask_legacy && hparams.f_max_alibi_bias == 0.0f && B*nbs == n_kv) {
        // shared only for decode / verify ubatches: at prefill widths A would stay live across the whole graph
        // (n_kv x T x 4 B, ~400 MB at 200K x 512), so there it is rebuilt per layer like the old temporaries
        const bool share = T <= 64;
        ggml_tensor * A = build_attn_lsa_blk_base(inp, kq_mask, n_kv, T, il);
        // LLAMA_LSA_BLKLIST=1: one pick for all rows (shared, reused or one row) on the SJ-KVaRN 4/4 stream kernel:
        // the FA node gets the pick itself plus one per-graph F16 mask M3 (kq_mask where always kept, -inf where
        // hidden, GGML_LSA_MASK_SEL at visible droppable cells), so no full-width op runs per layer. Same visible set
        // as the per-layer mask below: a droppable visible cell is seen iff its block is picked.
        const bool blist = cfg.blk_list && sel01 && share && !keep_f32 && (shmode > 0 || pick_k_reused || unioned || T == 1) &&
            lsa_blist_ok(T, B, kq_mask);
        if (blist) {
            ggml_tensor * m3 = build_attn_lsa_blk_m3(inp, A, kq_mask, n_kv, T, il);
            lsa_fa_sel = ggml_view_1d(ctx0, sel01, nbs, 0); // row 0: every row holds the same pick
            lsa_fa_blk = (int32_t) B;
            lsa_fa_m3  = m3;
            if (cfg.comp) {
                ggml_tensor * D = blk_dcnt(A);
                if (cfg.vk && lsa_vk_km && lsa_vk_vm && lsa_vk_nbs == nbs) {
                    // LLAMA_LSA_VK: plane (D + 1) x (1 - sel): 0 = picked, else 1 + dropped cells (same integer counts
                    // as the merge's nd); one op per layer against the per-graph D + 1, no merge after the FA node
                    ggml_tensor * D1 = inp->lsa_vk_dcnt1;
                    if (D1 == nullptr || inp->lsa_vk_dcnt1_src != D) {
                        D1 = ggml_scale_bias(ctx0, D, 1.0f, 1.0f);
                        cb(D1, "lsa_vk_dcnt1", il);
                        inp->lsa_vk_dcnt1     = D1;
                        inp->lsa_vk_dcnt1_src = D;
                    }
                    lsa_fa_sel = ggml_mul(ctx0, D1, ggml_reshape_2d(ctx0, sel_plane(0.0f, 1.0f), nbs, T)); // [nbs, T]
                    cb(lsa_fa_sel, "lsa_vk_plane", il);
                    lsa_fa_vk  = true;
                    static uint32_t logged_vk = 0;
                    if (llama_lsa_log_once(logged_vk)) {
                        LLAMA_LOG_WARN("%s: LSA compensation as virtual keys in the SJ-KVaRN kernel active (LLAMA_LSA_VK=1, "
                                "layer %d, %lld blocks, T %lld)\n", __func__, il, (long long) nbs, (long long) T);
                    }
                } else {
                    inp->lsa_comp_nd      = ggml_mul(ctx0, D, ggml_reshape_2d(ctx0, sel_plane(0.0f, 1.0f), nbs, T));
                    inp->lsa_comp_nd_mask = m3;
                }
            }
            static uint32_t logged_bl = 0;
            if (llama_lsa_log_once(logged_bl)) {
                LLAMA_LOG_WARN("%s: LSA block list to the SJ-KVaRN kernel active (LLAMA_LSA_BLKLIST=1, layer %d, %lld blocks, T %lld)\n",
                        __func__, il, (long long) nbs, (long long) T);
            }
            return m3;
        }
        ggml_tensor * selb = sel_plane(1e30f, 0.0f);
        ggml_tensor * m = ggml_add(ctx0, ggml_reshape_3d(ctx0, A, B, nbs, T), ggml_reshape_3d(ctx0, selb, 1, nbs, T));
        m = ggml_clamp(ctx0, ggml_reshape_2d(ctx0, m, n_kv, T), -INFINITY, 0.0f);
        if (kq_mask->type != GGML_TYPE_F32 && !keep_f32) {
            m = ggml_cast(ctx0, m, kq_mask->type);
        }
        m = ggml_reshape_4d(ctx0, m, n_kv, T, 1, 1);
        cb(m, "lsa_kq_mask_blk", il);
        if (share && cfg.comp_fast && !cfg.comp_cnt_legacy && !cfg.comp_cnt_exp && T <= cfg.comp_max_t) {
            // COMP=2 dropped cells per block without reading the mask again: a cell is dropped iff it is visible
            // (kq_mask 0), droppable (A = -1e30: not sink/window, row past min_pos) and its block is not selected, so
            // dropped_b = (1 - sel_b) x D_b with D_b = visible droppable cells of block b, which is layer independent
            // (once per graph). Same integers as the exp(mask) count (exact f32 sums of <= B ones, x 0/1).
            ggml_tensor * D = blk_dcnt(A);
            ggml_tensor * sel1 = sel_plane(0.0f, 1.0f);
            inp->lsa_comp_nd      = ggml_mul(ctx0, D, ggml_reshape_2d(ctx0, sel1, nbs, T));            // [nbs, T]
            inp->lsa_comp_nd_mask = m;
        }
        return m;
    }
    ggml_tensor * sel = sel_plane(1.0f, 0.0f);
    ggml_tensor * selc = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, sel, 1, nbs, T), B, nbs, T, 1),
            B*nbs, T);
    if (B*nbs < n_kv) {
        selc = ggml_pad(ctx0, selc, (int) (n_kv - B*nbs), 0, 0, 0); // tail without a summary: window covers it
    } else if (B*nbs > n_kv) {
        selc = ggml_cont(ctx0, ggml_view_2d(ctx0, selc, n_kv, T, selc->nb[1], 0));
    }
    // cells: kept = selected block, sink or window (query included); then the causal mask is added back
    ggml_tensor * dc   = ggml_sub(ctx0, ggml_repeat(ctx0, inp->lsa_cell_pos, selc), inp->lsa_q_pos);
    ggml_tensor * win  = ggml_step(ctx0, ggml_scale_bias(ctx0, dc, 1.0f, (float) cfg.window - 0.5f));
    ggml_tensor * snk  = ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_cell_pos, -1.0f, (float) cfg.sinks - 0.5f));
    ggml_tensor * keep = ggml_clamp(ctx0, ggml_add(ctx0, ggml_add(ctx0, selc, win), snk), 0.0f, 1.0f);
    ggml_tensor * drop = ggml_scale_bias(ctx0, keep, -1.0f, 1.0f);
    if (cfg.min_pos > 0) {
        drop = ggml_mul(ctx0, drop, ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_q_pos, 1.0f, 0.5f - (float) cfg.min_pos)));
    }
    ggml_tensor * mask_f32 = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    ggml_tensor * m = ggml_add(ctx0, ggml_scale(ctx0, drop, -1e30f), ggml_reshape_2d(ctx0, mask_f32, n_kv, T));
    if (kq_mask->type != GGML_TYPE_F32 && !keep_f32) {
        m = ggml_cast(ctx0, m, kq_mask->type);
    }
    m = ggml_reshape_4d(ctx0, m, n_kv, T, 1, 1);
    cb(m, "lsa_kq_mask_blk", il);
    static int n_logged = 0;
    if (n_logged < 2) {
        ++n_logged;
        LLAMA_LOG_INFO("%s: LSA layer %d: n_kv %lld, T %lld, %lld of %lld blocks of %lld + sinks + window\n", __func__, il,
                (long long) n_kv, (long long) T, (long long) kb, (long long) nbs, (long long) B);
    }
    return m;
}

// LSA fast path (LLAMA_LSA_FAST=1, llama-lsa-gather.h): one pick for the whole ubatch, ranked by the summed index
// scores of its rows (the shared pick). Always kept: the first `sinks` positions and every cell from `window` before
// the first row's position on (the window of each row, widened to cover all rows). The last row's mask drops cells no
// row can see. The kept cells are copied out of K and V and attention runs over them with the original mask's
// columns, so a row never sees a cell its causal mask hides. Returns the gathered mask, or nullptr (layer stays dense).
ggml_tensor * llm_graph_context::build_attn_lsa_fast(
        const llm_graph_input_attn_kv * inp,
        ggml_tensor * S,
        ggml_tensor ** k_io,
        ggml_tensor ** v_io,
        ggml_tensor * kq_mask,
            int64_t   keep_lr,
                int   il) const {
    const auto & cfg = llama_lsa_cfg();
    ggml_tensor * k = *k_io;
    ggml_tensor * v = *v_io;
    const int64_t n_kv = S->ne[0];
    const int64_t T    = S->ne[1];

    if (v == nullptr || v->ne[1] != k->ne[1] || v->ne[2] != n_kv || v->ne[3] != 1) {
        return nullptr;
    }
    if (lsa_ubatch().pos && cfg.min_pos > 0) {
        for (int64_t i = 0; i < T; ++i) {
            if (lsa_ubatch().pos[i] < cfg.min_pos) {
                return nullptr; // the fast path has no per-row dense fallback: the whole ubatch stays dense
            }
        }
    }
    // rounded up to 256 cells (FATTN_KQ_STRIDE): the GQA-packed attention kernels need it; the extra cells are the next
    // best-ranked ones (or masked, if fewer cells are visible)
    const int64_t n_sel = GGML_PAD(cfg.sinks + cfg.window + (T - 1) + std::max<int64_t>(keep_lr, 0), 256);
    if (n_kv <= n_sel) {
        return nullptr;
    }

    ggml_tensor * s = T > 1 ? ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, S))), n_kv, 1) : S;

    ggml_tensor * mask_f32 = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    mask_f32 = ggml_reshape_2d(ctx0, mask_f32, n_kv, T);

    ggml_tensor * q0  = ggml_view_2d(ctx0, inp->lsa_q_pos, 1, 1, inp->lsa_q_pos->nb[1], 0);           // first row
    ggml_tensor * d   = ggml_sub(ctx0, inp->lsa_cell_pos, q0);                                         // [n_kv, 1]
    ggml_tensor * win = ggml_step(ctx0, ggml_scale_bias(ctx0, d, 1.0f, (float) cfg.window - 0.5f));
    ggml_tensor * snk = ggml_step(ctx0, ggml_scale_bias(ctx0, inp->lsa_cell_pos, -1.0f, (float) cfg.sinks - 0.5f));
    s = ggml_add(ctx0, s, ggml_scale(ctx0, ggml_add(ctx0, win, snk), 1e20f));
    s = ggml_add(ctx0, s, ggml_view_2d(ctx0, mask_f32, n_kv, 1, mask_f32->nb[1], (T - 1)*mask_f32->nb[1]));
    cb(s, "lsa_score_shared", il);

    ggml_tensor * ids = ggml_top_k(ctx0, s, (int) n_sel); // I32 [n_sel, 1]
    cb(ids, "lsa_ids", il);

    ggml_tensor * kg = llama_lsa_gather_cells(ctx0, k, ids);
    ggml_tensor * vg = kg ? llama_lsa_gather_cells(ctx0, v, ids) : nullptr;
    if (kg == nullptr || vg == nullptr) {
        static uint32_t warned_ep = 0;
        if (llama_lsa_log_once(warned_ep)) {
            LLAMA_LOG_WARN("%s: LSA fast path: K/V cache layout not gatherable, layer stays dense\n", __func__);
        }
        return nullptr;
    }
    cb(kg, "lsa_k_sel", il);
    cb(vg, "lsa_v_sel", il);

    ggml_tensor * m = llama_lsa_gather_mask(ctx0, mask_f32, ids); // F32 [n_sel, T]
    if (kq_mask->type != GGML_TYPE_F32) {
        m = ggml_cast(ctx0, m, kq_mask->type);
    }
    m = ggml_reshape_4d(ctx0, m, n_sel, T, 1, 1);
    cb(m, "lsa_kq_mask_sel", il);

    *k_io = kg;
    *v_io = vg;

    static int n_logged = 0;
    if (n_logged < 4) {
        ++n_logged;
        LLAMA_LOG_INFO("%s: LSA fast layer %d: n_kv %lld, T %lld, gathered %lld (%.1f%%)\n", __func__, il,
                (long long) n_kv, (long long) T, (long long) n_sel, 100.0*(double) n_sel/(double) n_kv);
    }
    return m;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    const llm_arch arch = model.arch;

    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot[il]) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot[il]);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot[il]);
    }

    if (inp->self_v_rot[il]) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot[il]);
    }

    const auto * mctx_cur = inp->mctx;

    const bool is_sj_kvarn = inp->self_sj_kvarn_desc != nullptr;
    ggml_tensor * q_lsa = q_cur; // LSA on SJ-KVaRN: the index map reads q/k before the SJ-KVaRN rotation
    ggml_tensor * k_lsa = k_cur;
    ggml_tensor * v_lsa = v_cur; // LLAMA_LSA_COMP=2 on SJ-KVaRN: block means in the un-rotated basis

    // SJ-KVaRN: the whole cache lives in the 256-pt Hadamard-rotated basis (ring rows and sealed records alike),
    // so rotate Q/K/V here; the FA output is un-rotated in build_attn_mha. The plain Sylvester Hadamard is an
    // involution, so the same op serves both directions.
    // With the fused rotation (plain-256 bodies only) the FA node rotates Q and un-rotates its output itself.
    const auto * sj_wush = is_sj_kvarn ? mctx_cur->get_sj_wush(il) : nullptr; // EXPERIMENTAL WUSH-KV layer
    const bool sj_kvarn_fused_rot = is_sj_kvarn && sj_wush == nullptr && mctx_cur->get_sj_kvarn().body_type != GGML_TYPE_TURBO4_0 &&
        ggml_sj_kvarn_fused_rot_enabled();
    if (is_sj_kvarn) {
        GGML_ASSERT(q_cur->ne[0] == 256 && k_cur->ne[0] == 256 && v_cur->ne[0] == 256);
        if (!sj_kvarn_fused_rot && !ggml_is_contiguous(q_cur)) { q_cur = ggml_cont(ctx0, q_cur); }
        if (!ggml_is_contiguous(k_cur)) { k_cur = ggml_cont(ctx0, k_cur); }
        if (!ggml_is_contiguous(v_cur)) { v_cur = ggml_cont(ctx0, v_cur); }
        const int group = mctx_cur->get_sj_kvarn().body_type == GGML_TYPE_TURBO4_0 ? 128 : 256;
        if (sj_wush) {
            q_cur = llm_graph_sj_wush_apply(ctx0, q_cur, sj_wush->aq);
            k_cur = llm_graph_sj_wush_apply(ctx0, k_cur, sj_wush->ak);
            v_cur = llm_graph_sj_wush_apply(ctx0, v_cur, sj_wush->av);
            cb(q_cur, "q_sj_kvarn_rot", il);
        } else {
        if (!sj_kvarn_fused_rot) {
            q_cur = ggml_turbo_wht(ctx0, q_cur, 0, group, nullptr);
            cb(q_cur, "q_sj_kvarn_rot", il);
        }
        k_cur = ggml_turbo_wht(ctx0, k_cur, 0, group, nullptr);
        v_cur = ggml_turbo_wht(ctx0, v_cur, 0, group, nullptr);
        }
        cb(k_cur, "k_sj_kvarn_rot", il);
        cb(v_cur, "v_sj_kvarn_rot", il);
    }

    // LLAMA_LSA_VK: this ubatch's K/V rows in the cache basis (plain 256-pt rotation) for the rotated block means
    ggml_tensor * k_lsa_rot = nullptr;
    ggml_tensor * v_lsa_rot = nullptr;
    if (is_sj_kvarn && sj_wush == nullptr && mctx_cur->get_sj_kvarn().body_type != GGML_TYPE_TURBO4_0 &&
            llama_lsa_cfg().vk && llama_lsa_cfg().blk_list) {
        k_lsa_rot = k_cur;
        v_lsa_rot = v_cur;
    }

    // SJ-KVaRN fused write rotation (#139): the K/V WHT nodes above are dropped, the cache write rotates
    if (is_sj_kvarn && mctx_cur->sj_kvarn_fused_rot()) {
        k_cur = llm_graph_sj_kvarn_unrotate(k_cur);
        v_cur = llm_graph_sj_kvarn_unrotate(v_cur);
    }

    // FP8 KV: no-op unless the cache has per-layer K/V scales
    build_attn_scale_kv_inputs(ctx0, mctx_cur, q_cur, k_cur, v_cur, il);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_tensor * q = q_cur;
    ggml_tensor * k;
    ggml_tensor * v;

    if (cparams.training) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);
        GGML_ASSERT(!is_sj_kvarn);

        k = k_cur;
        v = v_cur;
    } else {
        // store to KV cache
        {
            const auto & k_idxs = inp->get_k_idxs();
            const auto & v_idxs = inp->get_v_idxs();

            ggml_tensor * k_store = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
            ggml_tensor * v_store = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);

            ggml_build_forward_expand(gf, k_store);
            ggml_build_forward_expand(gf, v_store);

            if (is_sj_kvarn) {
                // Historical groups are sealed before this graph runs; stores above remain graph dependencies.
                ggml_tensor * body = mctx_cur->get_sj_kvarn_body(il);

                uint32_t bits_k = 0, bits_v = 0;
                const int tier = mctx_cur->get_sj_kvarn_layer_tier(il, bits_k, bits_v);
                GGML_ASSERT(tier == 0 || inp->self_sj_kvarn_desc_edge != nullptr);
                sj_kvarn_pending.body   = body;
                sj_kvarn_pending.desc   = tier == 1 ? inp->self_sj_kvarn_desc_edge : inp->self_sj_kvarn_desc;
                sj_kvarn_pending.bits_k = (int32_t) bits_k;
                sj_kvarn_pending.bits_v = (int32_t) bits_v;
                sj_kvarn_pending.fused_rot = sj_kvarn_fused_rot;
                sj_kvarn_pending.n_kv_streams = &inp->sj_kvarn_n_kv_streams;
                sj_kvarn_pending.wush_ao = sj_wush ? sj_wush->ao : nullptr;
            }
        }

        k = mctx_cur->get_k(ctx0, il);
        v = mctx_cur->get_v(ctx0, il);
    }

    // LLAMA_LSA_MULTI=1 (opt-in): a ubatch over several SJ-KVaRN streams with per-stream LSA input sets runs the
    // attention below once per stream (its rows, its cache stream, its descriptor row and attended length, its LSA
    // caches), exactly the single-stream node of that stream; the outputs are concatenated stream-major. Otherwise one
    // pass over the whole ubatch (unchanged).
    const int64_t lsa_n_units = (!inp->lsa_streams.empty() && is_sj_kvarn && !cparams.training &&
            k->ne[3] == (int64_t) inp->lsa_streams.size() && q->ne[2] % k->ne[3] == 0) ? k->ne[3] : 1;
    GGML_ASSERT(lsa_n_units > 1 || inp->lsa_streams.empty() || !is_sj_kvarn || cparams.training);
    llm_graph_input_attn_kv * const inp_all = inp;
    ggml_tensor * const q_all = q, * const k_all = k, * const v_all = v;
    ggml_tensor * const q_lsa_all = q_lsa, * const k_lsa_all = k_lsa, * const v_lsa_all = v_lsa;
    ggml_tensor * const k_lsa_rot_all = k_lsa_rot, * const v_lsa_rot_all = v_lsa_rot;
    const auto sj_pending_all = sj_kvarn_pending;
    ggml_tensor * cur_all = nullptr;
    // ns == 1 under LLAMA_LSA_MULTI=1: the one input set addresses its sequence's caches (no-op otherwise)
    llama_lsa_stream_scope lsa_scope_one(inp->lsa_streams.empty() ? inp->lsa_stream : llama_lsa_stream());
    for (int64_t lsa_u = 0; lsa_u < lsa_n_units; ++lsa_u) {
        ggml_tensor * lsa_mask_u = nullptr;
        std::unique_ptr<llama_lsa_stream_scope> lsa_scope_u;
        if (lsa_n_units > 1) {
            inp = inp_all->lsa_streams[lsa_u].get();
            lsa_scope_u = std::make_unique<llama_lsa_stream_scope>(inp->lsa_stream);
            lsa_ub = &inp->lsa_sub_ub;
            const int64_t Ts = q_all->ne[2]/lsa_n_units;
            auto rows3 = [&](ggml_tensor * t) -> ggml_tensor * { // this stream's token rows of a [D, H, T] tensor
                return t ? ggml_view_3d(ctx0, t, t->ne[0], t->ne[1], Ts, t->nb[1], t->nb[2], lsa_u*Ts*t->nb[2]) : nullptr;
            };
            q = rows3(q_all);
            q_lsa = rows3(q_lsa_all);
            k_lsa = rows3(k_lsa_all);
            v_lsa = rows3(v_lsa_all);
            k_lsa_rot = rows3(k_lsa_rot_all);
            v_lsa_rot = rows3(v_lsa_rot_all);
            k = ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], k_all->ne[2], 1, k_all->nb[1], k_all->nb[2], k_all->nb[3],
                    lsa_u*k_all->nb[3]);
            v = ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], v_all->ne[2], 1, v_all->nb[1], v_all->nb[2], v_all->nb[3],
                    lsa_u*v_all->nb[3]);
            // the stream's [n_kv_s, T/ns] slice of the KQ mask, once per graph (shared by every layer)
            lsa_mask_u = lsa_stream_mask(ctx0, inp_all, inp, lsa_u, lsa_n_units);
            // the stream's descriptor row; the single-stream FA node reads its attended length from the mask width
            sj_kvarn_pending = sj_pending_all;
            if (sj_kvarn_pending.body) {
                ggml_tensor * d = sj_kvarn_pending.desc;
                sj_kvarn_pending.desc = ggml_view_1d(ctx0, d, d->ne[0], lsa_u*d->nb[1]);
                sj_kvarn_pending.n_kv_streams = nullptr;
            }
        }
        ggml_tensor * kq_mask = lsa_mask_u ? lsa_mask_u : inp->self_attn_mask.raw;
        const auto & kv_rows = inp->self_attn_mask.rows;
        // LLAMA_LSA_PACKED=1: the LSA builders start from the f16 dense mask input; a layer LSA leaves dense goes back to
        // the packed mask below
        ggml_tensor * const kq_mask_in = kq_mask;
        if (inp->lsa_kq_mask && inp->lsa_cell_pos) {
            kq_mask = inp->lsa_kq_mask;
        }
        ggml_tensor * const kq_mask_dense = kq_mask;

        int64_t lsa_n_kv_max = 0;
        std::vector<ggml_tensor *> lsa_masks_h; // LLAMA_LSA_PER_HEAD=1: one mask per KV head
        // LLAMA_LSA_COMP=2: block means refreshed for every ubatch while LSA runs (prefill included)
        ggml_tensor * lsa_kmv = nullptr;
        ggml_tensor * lsa_vmv = nullptr;
        int64_t       lsa_nbs = 0;
        bool          lsa_means_ok = false;
        bool          lsa_means_rot = false; // LLAMA_LSA_VK: means in the rotated cache basis
        lsa_vk_km = nullptr; lsa_vk_vm = nullptr; lsa_vk_nbs = 0; lsa_vk_done = false;
        if (inp->lsa_cell_pos && is_sj_kvarn && inp->lsa_q_cell && !cparams.training) {
            // SJ-KVaRN: masked attention only (dense compute over the sealed records), index keys from the side cache
            if (ggml_tensor * lsa_idx = build_attn_lsa_idx(inp, k_lsa, k, il)) {
                if (llama_lsa_cfg().comp_fast && sj_wush == nullptr && !llama_lsa_any_rot(inp->self_k_rot) && !llama_lsa_any_rot(inp->self_v_rot)) {
                    lsa_means_rot = k_lsa_rot != nullptr;
                    lsa_means_ok = build_attn_lsa_mean_refresh(inp, k, v, lsa_means_rot ? k_lsa_rot : k_lsa,
                            lsa_means_rot ? v_lsa_rot : v_lsa, kq_mask->ne[0], &lsa_kmv, &lsa_vmv, &lsa_nbs, il, lsa_means_rot);
                    if (lsa_means_ok && lsa_means_rot && q->ne[2] <= llama_lsa_cfg().comp_max_t &&
                            lsa_kmv->view_src && lsa_kmv->view_src->ne[1] >= 16) {
                        lsa_vk_km  = lsa_kmv; // the block-list branch of the mask build takes them (LLAMA_LSA_VK)
                        lsa_vk_vm  = lsa_vmv;
                        lsa_vk_nbs = lsa_nbs;
                    }
                }
                if (ggml_tensor * lsa_mask = build_attn_lsa_mask(inp, q_lsa, &k, &v, kq_mask, il, &lsa_n_kv_max, lsa_idx,
                            kq_scale, nullptr)) {
                    kq_mask = lsa_mask;
                }
            }
        } else if (inp->lsa_cell_pos && !is_sj_kvarn && !llama_lsa_any_rot(inp->self_k_rot) && !cparams.training) {
            ggml_tensor * lsa_idx = build_attn_lsa_idx(inp, k_cur, k, il);
            if (llama_lsa_cfg().comp_fast) {
                lsa_means_ok = build_attn_lsa_mean_refresh(inp, k, v, nullptr, nullptr, k->ne[2], &lsa_kmv, &lsa_vmv, &lsa_nbs, il);
            }
            const bool can_split = kq_b == nullptr && sinks == nullptr && v_mla == nullptr && k->ne[3] == 1 &&
                !(v->nb[1] > v->nb[2]);
            if (ggml_tensor * lsa_mask = build_attn_lsa_mask(inp, q, &k, &v, kq_mask, il, &lsa_n_kv_max, lsa_idx,
                        kq_scale, can_split ? &lsa_masks_h : nullptr)) {
                kq_mask = lsa_mask;
            }
        }

        const bool lsa_applied = kq_mask != kq_mask_dense;
        // LSA rebuilds the mask over its own cell order: upstream kv_rows (opt-in LLAMA_KV_ROWS) never runs with it
        GGML_ASSERT((!lsa_applied && lsa_masks_h.empty()) || kv_rows == nullptr);
        if (!lsa_applied && lsa_masks_h.empty()) {
            kq_mask = kq_mask_in;
        }

        q = build_attn_turbo_q(q, k, mctx_cur);

        ggml_tensor * cur = nullptr;
        if (!lsa_masks_h.empty() && (int64_t) lsa_masks_h.size() == k->ne[1] && q->ne[1] % k->ne[1] == 0) {
            // per-KV-head LSA probe: FA per GQA group with that KV head's mask, outputs concatenated head-major
            const int64_t g = q->ne[1]/k->ne[1];
            for (int64_t j = 0; j < k->ne[1]; ++j) {
                ggml_tensor * qj = ggml_view_3d(ctx0, q, q->ne[0], g, q->ne[2], q->nb[1], q->nb[2], j*g*q->nb[1]);
                ggml_tensor * kj = ggml_view_4d(ctx0, k, k->ne[0], 1, k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], j*k->nb[1]);
                ggml_tensor * vj = ggml_view_4d(ctx0, v, v->ne[0], 1, v->ne[2], v->ne[3], v->nb[1], v->nb[2], v->nb[3], j*v->nb[1]);
                ggml_tensor * c  = build_attn_mha(qj, kj, vj, nullptr, lsa_masks_h[j], nullptr, nullptr, nullptr, lsa_n_kv_max, kq_scale, il);
                cur = cur ? ggml_concat(ctx0, cur, c, 0) : c;
            }
        } else {
            const auto & lcfg = llama_lsa_cfg();
            // SJ-KVaRN: fast compensation only (LLAMA_LSA_COMP=2), merged after the FA output is un-rotated, with the
            // un-rotated q (the rotation is orthonormal, so the scores and the log-sum-exp are the same)
            // the low-bit / trellis SJ-KVaRN kernels (2/2, sjkvarn4t, and 3/3t, 3/2t unless LLAMA_LSA_COMP_LB=1) write no LSE
            // output, so the merge would read an unwritten tensor: those layers run LSA uncompensated
            const int  sj_body_t = is_sj_kvarn && sj_kvarn_pending.body ? sj_kvarn_pending.body->op_params[7] : -1;
            // A5 (LLAMA_LSA_COMP_LB=1, opt-in): the 3/2 and 3/3 trellis bodies decode only through the low-bit tile kernel,
            // whose uniform stream-k fixup writes the LSE; the scalar low-bit width kernel, 2/2 and sjkvarn4t stay refused
            const bool sj_lb_lse = is_sj_kvarn && sj_kvarn_pending.body && lcfg.comp_lb && sj_body_t == GGML_TYPE_I16 &&
                    sj_kvarn_pending.bits_k == 3 && (sj_kvarn_pending.bits_v == 2 || sj_kvarn_pending.bits_v == 3);
            const int  sj_lb_bv  = sj_kvarn_pending.bits_v; // build_attn_mha consumes sj_kvarn_pending
            const bool sj_no_lse = is_sj_kvarn && sj_kvarn_pending.body && (sj_kvarn_pending.bits_k != 4 ||
                    sj_kvarn_pending.bits_v != 4 || sj_body_t == GGML_TYPE_I16) && !sj_lb_lse;
            if (lsa_applied && lcfg.comp && sj_no_lse) {
                static uint32_t warned_lb = 0;
                if (llama_lsa_log_once(warned_lb)) {
                    LLAMA_LOG_WARN("%s: layer %d: LSA compensation off (SJ-KVaRN %d/%d %s body has no LSE output)\n", __func__,
                            il, sj_kvarn_pending.bits_k, sj_kvarn_pending.bits_v, sj_body_t == GGML_TYPE_I16 ? "trellis" : "scalar");
                }
            }
            const bool lsa_comp = lsa_applied && lcfg.comp && lsa_n_kv_max == 0 && kq_b == nullptr &&
                    sinks == nullptr && v_mla == nullptr && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap &&
                    (!is_sj_kvarn || (lcfg.comp_fast && lsa_means_ok && !sj_no_lse));
            const bool vk_pending = lsa_fa_vk && lsa_fa_sel != nullptr; // LLAMA_LSA_VK: compensation inside the FA node
            // no LSE at >= 128 rows on any SJ-KVaRN body (the expand-to-f16 prefill path would take it and writes none)
            lsa_lse_req = lsa_comp && !vk_pending && lcfg.comp_fast && lsa_means_ok && q->ne[2] <= lcfg.comp_max_t &&
                    !(is_sj_kvarn && q->ne[2] >= 128);
            cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, lsa_n_kv_max, kq_scale, il);
            ggml_tensor * lse = lsa_lse;
            lsa_lse     = nullptr;
            lsa_lse_req = false;
            const bool vk_done = lsa_vk_done;
            lsa_vk_done = false;
            lsa_vk_km   = nullptr;
            lsa_vk_vm   = nullptr;
            lsa_vk_nbs  = 0;
            if (vk_pending && !vk_done) {
                static uint32_t warned_vk = 0;
                if (llama_lsa_log_once(warned_vk)) {
                    LLAMA_LOG_WARN("%s: layer %d: LSA virtual keys not attached to the FA node, layer uncompensated\n", __func__, il);
                }
            }
            if (lsa_comp && lsa_means_rot && lse && !vk_done) {
                // rotated means (LLAMA_LSA_VK) on a graph the kernel path does not serve (e.g. 9..16 rows): merge in the
                // cache basis, rotated q and o_kept in, output rotated back (the rotation is orthonormal and an involution)
                ggml_tensor * qr = ggml_turbo_wht(ctx0, ggml_is_contiguous(q_lsa) ? q_lsa : ggml_cont(ctx0, q_lsa), 0, 256, nullptr);
                ggml_tensor * orr = ggml_turbo_wht(ctx0, ggml_is_contiguous(cur) ? cur : ggml_cont(ctx0, cur), 0, 256, nullptr);
                ggml_tensor * c = build_attn_lsa_comp_fast(inp, orr, lse, qr, lsa_kmv, lsa_vmv, lsa_nbs, k_lsa->ne[1], kq_mask,
                        kq_mask_dense, kq_scale, il);
                if (c) {
                    cur = ggml_turbo_wht(ctx0, c, 0, 256, nullptr);
                }
            } else if (lsa_comp && !vk_done) {
                ggml_tensor * c = !lcfg.comp_fast ? build_attn_lsa_comp(cur, q, k, v, kq_mask, kq_mask_dense, kq_scale, il) :
                                  lse ? build_attn_lsa_comp_fast(inp, cur, lse, is_sj_kvarn ? q_lsa : q, lsa_kmv, lsa_vmv, lsa_nbs,
                                                                 is_sj_kvarn ? k_lsa->ne[1] : k->ne[1], kq_mask, kq_mask_dense,
                                                                 kq_scale, il) : nullptr;
                if (c) {
                    cur = c;
                    if (sj_lb_lse) {
                        // A5 route proof for gates: one line per layer the first time its merge is built on a trellis body
                        static std::atomic<uint64_t> lb_seen[8];
                        const uint64_t bit = uint64_t(1) << (il & 63);
                        if (il >= 0 && il < 512 && !(lb_seen[il >> 6].fetch_or(bit) & bit)) {
                            LLAMA_LOG_WARN("%s: layer %d: LSA compensation on the SJ-KVaRN %d/%d trellis body (LLAMA_LSA_COMP_LB=1)\n",
                                    __func__, il, 3, sj_lb_bv);
                        }
                    }
                }
            }
        }
        cur_all = cur_all ? ggml_concat(ctx0, cur_all, cur, 1) : cur;
    }
    inp    = inp_all;
    lsa_ub = nullptr;
    if (lsa_n_units > 1) {
        v = v_all; // build_attn_turbo_v_out below: the cache view (one pass keeps whatever the LSA path left in v)
    }
    ggml_tensor * cur = cur_all;
    cur = build_attn_scale_kv_output(ctx0, mctx_cur, cur, il);
    cb(cur, "kqv_out", il);

    GGML_ASSERT(sj_kvarn_pending.body == nullptr && "SJ-KVaRN cache requires the flash-attention path");

    cur = build_attn_turbo_v_out(cur, v, il);

    if (inp->self_v_rot[il]) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot[il]);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 have numerical issues with F16 accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_BF16);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams, /*allow_packed=*/true);
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const llm_arch arch = model.arch;

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->self_attn_mask.raw;
    const auto & kv_rows = inp->self_attn_mask.rows;

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    // TurboQuant: pre-rotate Q for K-only (MLA) attention
    q = build_attn_turbo_q(q, k, mctx_cur);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    // TurboQuant: if V was padded (MLA: V is view of K, may have padded dim),
    // extract original V head_dim after inverse WHT.
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0 || k->type == GGML_TYPE_TQ5_0 || k->type == GGML_TYPE_TQ6_0) {
        const int64_t orig_v_head = v_cur->ne[0];  // original V head_dim from model
        const int64_t padded_v_head = v->ne[0];     // padded V head_dim in cache
        if (padded_v_head != orig_v_head) {
            // cur is 2D: (padded_v_head * n_head, n_tokens) after build_attn_mha
            // Fix #78 (bingh0): cur shape post-MHA is (n_embd_head * n_head, n_tokens),
            // not (n_embd_head * n_head_kv, n_tokens). The output carries one
            // row per Q head, so derive the head count from the tensor itself:
            // hparams.n_head(il) is correct but is a second source of truth
            // that has to agree with cur — ne[0]/padded_v_head cannot disagree,
            // and the assert turns a silent nelements mismatch (observed live:
            // gpt-oss 64->128 V pad, 64q/8kv, abort in ggml_reshape_3d during
            // graph reserve) into a loud failure at the exact site.
            GGML_ASSERT(cur->ne[0] % padded_v_head == 0);
            const int64_t n_head_v = cur->ne[0] / padded_v_head;
            const int64_t n_tokens_cur = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_tokens_cur);
            cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_tokens_cur,
                               cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * n_head_v, n_tokens_cur);
        }
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE have numerical issues with F16 accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_BF16);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_fused_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
                int   il) const {
    top_k = ggml_reshape_4d(ctx0, top_k, top_k->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);

    ggml_tensor * cur = ggml_top_k_mask(ctx0, kq_mask, top_k, nullptr, nullptr, nullptr);
    res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, cur, il});

    return cur;
}

ggml_tensor * llm_graph_context::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
                int   il) const {
    if (cparams.fused_top_k_mask) {
        return build_fused_top_k_mask(kq_mask, top_k, il);
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    return ggml_add(ctx0, kq_mask_top_k, kq_mask);
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    // note: k_cur is null for layers that reuse the KV cache of another layer
    if (k_cur) {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    // top_k indexes the cells, and the indexer mask of the same type is built the same way
    // let's use it, so when no other layer reads the MLA mask is not allocated
    ggml_tensor * kq_mask = inp->self_attn_mask_mla.raw;
    if (inp->self_attn_mask_lid.raw->type == kq_mask->type) {
        kq_mask = inp->self_attn_mask_lid.raw;
    }

    ggml_tensor * kq_mask_top_k = build_top_k_mask(kq_mask, top_k, il);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    // MLA: V is a view of the first n_embd_v elements of K
    const int64_t n_embd_v = v_cur ? v_cur->ne[0] : v_mla->ne[0];
    ggml_tensor * v = ggml_view_4d(ctx0, k, n_embd_v, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, nullptr, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// the iswa build_attn with the KQ mask and kv_rows given by the caller, which has to keep them in sync
static ggml_tensor * build_attn_iswa_impl(
        const llm_graph_context & gctx,
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * kq_mask,
        ggml_tensor * kv_rows,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) {
    ggml_context * ctx0 = gctx.ctx0;
    ggml_cgraph  * gf   = gctx.gf;

    const auto & hparams = gctx.hparams;
    const auto & cparams = gctx.cparams;

    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa[il] : inp->self_k_rot[il];
    auto * v_rot = is_swa ? inp->self_v_rot_swa[il] : inp->self_v_rot[il];

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    build_attn_scale_kv_inputs(ctx0, mctx_cur, q_cur, k_cur, v_cur, il);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    // whole seq fits into batch in training mode
    const bool use_kv_cur = cparams.training && k_cur && v_cur;
    if (use_kv_cur) {
        GGML_ASSERT(mctx_cur->get_n_kv() == gctx.n_tokens);
    }

    const bool store_kv = !use_kv_cur || hparams.n_layer_kv_from_start >= 0;

    // optionally store to KV cache
    if (store_kv && k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (store_kv && v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    // TurboQuant: pre-rotate Q for ISWA attention (pad to 128-aligned if needed)
    q = gctx.build_attn_turbo_q(q, k, mctx_cur);

    ggml_tensor * cur = gctx.build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cur = build_attn_scale_kv_output(ctx0, mctx_cur, cur, il);
    gctx.cb(cur, "kqv_out", il);

    cur = gctx.build_attn_turbo_v_out(cur, v, il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = gctx.build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->self_attn_mask_swa.raw  : inp->self_attn_mask.raw;
    const auto & kv_rows = is_swa ? inp->self_attn_mask_swa.rows : inp->self_attn_mask.rows;

    return build_attn_iswa_impl(*this, inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, kq_mask, kv_rows, sinks, v_mla, kq_scale, il);
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa[il] : inp->self_k_rot[il];

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->self_attn_mask_swa.raw : inp->self_attn_mask.raw;
    const auto & kv_rows = is_swa ? inp->self_attn_mask_swa.rows : inp->self_attn_mask.rows;

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_attn_mask.raw = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_attn_mask.raw);

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->cross_attn_mask.raw;
    const auto & kv_rows = inp->cross_attn_mask.rows;

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_attn_mask_mla.raw = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_attn_mask_lid.raw = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);

        inp->self_k_rot_lid = mctx_cur->get_lid()->get_k_rot();
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
    }

    inp->self_k_rot = mctx_base->get_k_rot();
    inp->self_v_rot = mctx_base->get_v_rot();

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask = build_attn_inp_attn_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask_swa = build_attn_inp_attn_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    }

    inp->self_k_rot = mctx_cur->get_base()->get_k_rot();
    inp->self_v_rot = mctx_cur->get_base()->get_v_rot();

    inp->self_k_rot_swa = mctx_cur->get_swa()->get_k_rot();
    inp->self_v_rot_swa = mctx_cur->get_swa()->get_v_rot();

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_attn_mask_swa.raw = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    }

    inp->self_k_rot = mctx_cur->get_base()->get_k_rot();

    inp->self_k_rot_swa = mctx_cur->get_swa()->get_k_rot();

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_attn_mask.raw = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);

    inp_raw->self_k_rot = raw_ctx->get_k_rot();
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->get_k_rot();
    inp->inp_hca.k_rot = mctx_cur->get_hca()->get_k_rot();
    inp->inp_lid.k_rot = mctx_cur->get_lid()->get_k_rot();

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    if (s->type == GGML_TYPE_F32) {
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
    } else {
        // bf16/f16/q8_0 recurrent state (--cache-type-s): scale is F32-only on the backends
        ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));
    }

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

    ggml_tensor * output_states = get_state_rows ?
        get_state_rows(ctx0, states, state_copy_main) :
        ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    inp->s_copy_main = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();
    inp->replay_len = mctx_cur->get_replay_len();
    inp->ckpt_span  = mctx_cur->get_ckpt_span();
    inp->s_stale    = mctx_cur->get_s_stale();
    inp->snap_shift = mctx_cur->get_snap_shift();

    // DRC phase 2: the span the ring holds behind the checkpoint after this ubatch -- the
    // accepted prefix (ckpt_span - replay_len) plus this ubatch's tokens, capped at the ring
    // capacity (beyond it the builder re-bases the checkpoint onto the last n_rs_seq tokens).
    // Must agree with llm_build_delta_net_base::build_recurrent_attn's bookkeeping.
    {
        const uint32_t n_seq_tokens = n_seqs > 0 ? (uint32_t) (ubatch.n_tokens / n_seqs) : 0;
        inp->span_new = llama_memory_recurrent::replay_span_new(inp->ckpt_span, inp->replay_len, n_seq_tokens, mctx_cur->get_n_rs_seq());
    }

    if (inp->snap_shift > 0) {
        inp->s_copy_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) inp->snap_shift * n_seqs);
        ggml_set_input(inp->s_copy_shift);
    }

    // --gdn-replay: a ubatch that fills the ring (or, opt-in, a short verify round) takes the
    // rollback-independent graph (see replay_w)
    if (mctx_cur->get_gdn_replay() && n_seqs > 0 &&
            llama_memory_recurrent::replay_static_width(ubatch.n_tokens / n_seqs, mctx_cur->get_n_rs_seq())) {
        inp->s_copy_conv = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
        ggml_set_input(inp->s_copy_conv);
        ggml_set_name(inp->s_copy_conv, "rs_s_copy_conv");
        inp->s_copy_conv_main = ggml_view_1d(ctx0, inp->s_copy_conv, n_seqs, 0);
        inp->s_copy_pair = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
        ggml_set_input(inp->s_copy_pair);
        ggml_set_name(inp->s_copy_pair, "rs_s_copy_pair");
        inp->s_copy_pair_main = ggml_view_1d(ctx0, inp->s_copy_pair, n_seqs, 0);
        // [TAG_GDN_REPLAY_LANES] one K-slot mask per lane; 1-D (one mask for all lanes) otherwise
        inp->replay_w = mctx_cur->get_replay_lanes() && n_seqs > 1
            ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t) mctx_cur->get_n_rs_seq(), n_seqs)
            : ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t) mctx_cur->get_n_rs_seq());
        ggml_set_input(inp->replay_w);
        ggml_set_name(inp->replay_w, "rs_replay_w");
    }

    // [TAG_RECURRENT_ROLLBACK_RING]
    inp->n_written = mctx_cur->get_n_written();
    inp->n_older   = mctx_cur->get_n_older();
    if (inp->n_written > 0) {
        inp->rs_wr = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) inp->n_written * n_seqs);
        ggml_set_input(inp->rs_wr);
        inp->rs_wr_conv.resize(inp->n_written);
        for (uint32_t j = 0; j < inp->n_written; ++j) {
            inp->rs_wr_conv[j] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
            ggml_set_input(inp->rs_wr_conv[j]);
        }
    }
    if (inp->n_older > 0) {
        inp->rs_old_src = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_older);
        inp->rs_old_dst = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_older);
        ggml_set_input(inp->rs_old_src);
        ggml_set_input(inp->rs_old_dst);
    }

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    // [TAG_RECURRENT_ROLLBACK_RING] a seq whose data moves cells this ubatch takes its older
    // rollback groups along, plane for plane. Gather before build_rs's extra-state copy (an
    // extra may be the seq that was swapped out of our destination cell, and it reads from
    // there), scatter after it.
    ggml_tensor * states = nullptr;
    ggml_tensor * older  = nullptr;
    if (inp->rs_old_src != nullptr) {
        states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);
        older  = ggml_get_rows(ctx0, states, inp->rs_old_src);
        ggml_build_forward_expand(gf, older);
    }

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;
    ggml_tensor * copy_main  = inp->s_copy_main;

    // --gdn-replay verify graph: the GDN state (plane 0 of the state/checkpoint pair) is gathered
    // through s_copy_pair, so the ubatch's rows are the replay base (see llm_graph_input_rs)
    if (inp->s_copy_pair != nullptr && older == nullptr && s->view_src != nullptr && s->view_offs == 0 &&
            s->view_src->ne[1] == 2 * s->ne[1] && strncmp(s->view_src->name, "cache_s2_l", 10) == 0) {
        s          = s->view_src;
        state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy_pair;
        copy_main  = inp->s_copy_pair_main;
    }

    ggml_tensor * out = build_rs(s, state_copy, copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);

    if (older != nullptr) {
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, states, older, inp->rs_old_dst));
    }

    return out;
}

ggml_tensor * llm_graph_context::build_rs_conv(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs) const {
    if (inp->s_copy_conv == nullptr) {
        return build_rs(inp, s, state_size, n_seqs);
    }
    const auto * kv_state = inp->mctx;
    return build_rs(s, inp->s_copy_conv, inp->s_copy_conv_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z());
}

void llm_graph_context::build_conv_state_store(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
        ggml_tensor * conv_x,
            int64_t   n_cols,
             size_t   offs,
               bool   slot_cpy) const {
    const auto * mctx_cur = inp->mctx;

    const int64_t n        = conv_x->ne[0] - n_cols; // n_seq_tokens
    const int64_t channels = conv_x->ne[1];
    const int64_t n_seqs   = conv_x->ne[2];

    const size_t row_size  = s->nb[1];
    const size_t cell_offs = mctx_cur->get_head()*row_size + offs;

    if (cparams.n_rs_seq == 0) {
        ggml_tensor * last = ggml_view_3d(ctx0, conv_x,
                n_cols, channels, n_seqs,
                conv_x->nb[1], conv_x->nb[2],
                ggml_row_size(conv_x->type, n));

        ggml_tensor * cell = ggml_view_2d(ctx0, s,
                n_cols*channels, n_seqs,
                row_size, cell_offs);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, cell));
        return;
    }

    // [TAG_RECURRENT_ROLLBACK_SPLITS]
    // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
    //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument
    // a rollback of k tokens reads slot k, and the memory allows k < n only (see rs_max), so slots from n on are left unwritten
    const int64_t K  = (int64_t) cparams.n_rs_seq + 1;
    const int64_t Kw = std::min(n, K);

    if (slot_cpy) {
        // [INKGLUE SLOTCPY] the same stores as the SET_ROWS below, as one CPY per slot (window w -> slot Kw - 1 - w): the
        // CUDA conv ring takes CPY windows, so the concat, the stores and the slot-order arange/scale/cast fold into its
        // ssm_conv launch
        for (int64_t w = 0; w < Kw; ++w) {
            ggml_tensor * win = ggml_view_3d(ctx0, conv_x,
                    n_cols, channels, n_seqs,
                    conv_x->nb[1], conv_x->nb[2],
                    ggml_row_size(conv_x->type, n - (Kw - 1) + w));
            ggml_tensor * cell = ggml_view_2d(ctx0, s,
                    n_cols*channels, n_seqs,
                    row_size, cell_offs + (size_t) (Kw - 1 - w)*mctx_cur->get_size()*row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, win, cell));
        }
        return;
    }

    // each conv layer builds its own slot order {K-1, ..., 0}, on the device of the layer, and takes its tail
    // {Kw-1, ..., 0}; the size K never depends on the ubatch, so the reserved graph covers them all
    ggml_tensor * s_slot_rev = ggml_arange(ctx0, 0.0f, (float) K, 1.0f);
    s_slot_rev = ggml_scale_bias(ctx0, s_slot_rev, -1.0f, (float) (K - 1));
    s_slot_rev = ggml_cast(ctx0, s_slot_rev, GGML_TYPE_I32);
    cb(s_slot_rev, "s_slot_rev", -1);

    ggml_tensor * slot_order = ggml_view_1d(ctx0, s_slot_rev, Kw, ggml_row_size(GGML_TYPE_I32, K - Kw));

    // window w starts at column n - (Kw - 1) + w (ascending time) and is rollback slot Kw - 1 - w,
    // consecutive windows start one element apart, so they overlap
    ggml_tensor * windows = ggml_view_4d(ctx0, conv_x,
            n_cols, Kw, channels, n_seqs,
            ggml_element_size(conv_x), conv_x->nb[1], conv_x->nb[2],
            ggml_row_size(conv_x->type, n - (Kw - 1)));

    // [n_cols, slot, channel, seq] view of the cells of this ubatch
    ggml_tensor * slots = ggml_view_4d(ctx0, s,
            n_cols, K, channels, n_seqs,
            (size_t) mctx_cur->get_size()*row_size, ggml_row_size(s->type, n_cols), row_size,
            cell_offs);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, slots, windows, slot_order));
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn(), is_reserve);

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask = build_attn_inp_attn_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask_swa = build_attn_inp_attn_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
    }

    inp_attn->self_k_rot = attn_ctx->get_base()->get_k_rot();
    inp_attn->self_v_rot = attn_ctx->get_base()->get_v_rot();

    inp_attn->self_k_rot_swa = attn_ctx->get_swa()->get_k_rot();
    inp_attn->self_v_rot_swa = attn_ctx->get_swa()->get_v_rot();

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa_rel * llm_graph_context::build_inp_mem_hybrid_iswa_rel(
        uint32_t extent,
        uint32_t extent_swa) const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    // dense masks: the bias is gathered by the distance to each cell of the K/V view
    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask_swa.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
    }

    inp_attn->self_k_rot = attn_ctx->get_base()->get_k_rot();
    inp_attn->self_v_rot = attn_ctx->get_base()->get_v_rot();

    inp_attn->self_k_rot_swa = attn_ctx->get_swa()->get_k_rot();
    inp_attn->self_v_rot_swa = attn_ctx->get_swa()->get_v_rot();

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa_rel>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur, extent, extent_swa);

    inp->self_rel_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_base()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos);
    ggml_set_name(inp->self_rel_pos, "attn_inp_rel_pos");

    inp->self_rel_pos_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_swa()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos_swa);
    ggml_set_name(inp->self_rel_pos_swa, "attn_inp_rel_pos_swa");

    // INKKV: SJ-KVaRN full-attention cache (the INKLSA / INKREL routes below need plain F16 K/V and stay off)
    const bool base_sj = attn_ctx->get_base()->is_sj_kvarn();
    if (base_sj) {
        GGML_ASSERT(cparams.flash_attn && "INKKV: SJ-KVaRN needs flash attention");
        GGML_ASSERT(!attn_ctx->get_base()->has_sj_kvarn_edge_tier() && "INKKV: SJ-KVaRN edge tiers are not supported");
        inp->self_sj_kvarn_desc = attn_ctx->get_base()->build_input_sj_kvarn_desc(ctx0, ubatch);
        inp->sj_kvarn_n_kv_streams = attn_ctx->get_base()->get_sj_kvarn_n_kv_streams();
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.is_swa(il) && hparams.has_kv(il) && !hparams.is_recr(il)) {
                inp->sj_il_full = (int32_t) il;
                break;
            }
        }
        if (inkkv_split_want(attn_ctx, ubatch, cparams, extent, extent_swa, inp->sj_il_full, is_reserve)) {
            inp->self_sj_far_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, attn_ctx->get_base()->get_n_kv(), ubatch.n_tokens, 1, 1);
            ggml_set_input(inp->self_sj_far_mask);
            ggml_set_name(inp->self_sj_far_mask, "attn_inp_sj_far_mask");
        } else if (!is_reserve && attn_ctx->get_base()->get_n_kv() > extent + ubatch.n_tokens) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: INKKV: dense SJ-KVaRN rel-bias path at n_kv = %u (no far/near split: a gap in the SWA "
                        "window copy or several sequences); its per-head bias is [n_kv, T, n_head] and may need far more memory\n",
                        __func__, attn_ctx->get_base()->get_n_kv());
            }
        }
    }

    // INKLSA (llama-inklsa.h): per-graph inputs shared by the full-attention visits
    const auto & icfg = llama_inklsa_cfg();
    if (icfg.on && !base_sj && cparams.flash_attn && !cparams.training && cparams.n_seq_max == 1) {
        int64_t il_full = -1;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.is_swa(il)) {
                il_full = il;
                break;
            }
        }
        const int64_t n_kv = attn_ctx->get_base()->get_n_kv();
        const int64_t T    = ubatch.n_tokens;
        auto & in = inp->inklsa;
        if (il_full >= 0 && n_kv % icfg.block == 0) {
            inp->inklsa_on = true;
            in.B     = icfg.block;
            in.n_blk = n_kv/icfg.block;
            in.n_fix = llama_inklsa_n_fix();
            in.n_u   = std::min<int64_t>(in.n_blk, (T + in.B - 1)/in.B + 1);
            in.sparse = T <= icfg.max_t && n_kv >= icfg.min_kv && ubatch.n_seqs_unq == 1 &&
                        n_kv >= in.n_fix + llama_inklsa_n_keep(n_kv) + icfg.sinks;
            in.n_keep = llama_inklsa_n_keep(n_kv);
            in.u_blk   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, in.n_u);
            in.u_blk64 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, in.n_u);
            ggml_set_input(in.u_blk);
            ggml_set_input(in.u_blk64);
            ggml_set_name(in.u_blk, "inklsa_u_blk");
            ggml_set_name(in.u_blk64, "inklsa_u_blk64");
            if (in.sparse) {
                in.onehot   = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, hparams.n_head_kv(il_full), hparams.n_head(il_full), 1);
                in.fix_ids  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, in.n_fix);
                in.fix_rel  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, in.n_fix, T);
                in.fix_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in.n_fix, T);
                in.elig     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in.n_blk, 1);
                for (ggml_tensor * t : {in.onehot, in.fix_ids, in.fix_rel, in.fix_mask, in.elig}) {
                    ggml_set_input(t);
                }
                ggml_set_name(in.onehot, "inklsa_onehot");
                ggml_set_name(in.fix_ids, "inklsa_fix_ids");
                ggml_set_name(in.fix_rel, "inklsa_fix_rel");
                ggml_set_name(in.fix_mask, "inklsa_fix_mask");
                ggml_set_name(in.elig, "inklsa_elig");
            }
        }
    }

    // INKREL (llama-inklsa.h): far + near inputs for wide ubatches (decided from the shapes only, so the reserve graph
    // sizes the same split)
    {
        const int64_t n_kv = attn_ctx->get_base()->get_n_kv();
        const int64_t T    = ubatch.n_tokens;
        auto & in = inp->inkrel;
        if (cparams.flash_attn && !cparams.training && cparams.n_seq_max == 1 && ubatch.n_seqs_unq <= 1 && !base_sj &&
                llama_inkrel_want(n_kv, T, extent)) {
            in.on        = true;
            in.n_near    = (extent - 1 + T + 255)/256*256;
            in.far_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, T);
            in.near_ids  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, in.n_near);
            in.near_rel  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, in.n_near, T);
            in.near_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in.n_near, T);
            for (ggml_tensor * t : {in.far_mask, in.near_ids, in.near_rel, in.near_mask}) {
                ggml_set_input(t);
            }
            ggml_set_name(in.far_mask,  "inkrel_far_mask");
            ggml_set_name(in.near_ids,  "inkrel_near_ids");
            ggml_set_name(in.near_rel,  "inkrel_near_rel");
            ggml_set_name(in.near_mask, "inkrel_near_mask");
        }
    }

    return (llm_graph_input_mem_hybrid_iswa_rel *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * inp_attn = inp->get_attn();

    const auto & attn_mask = is_swa ? inp_attn->self_attn_mask_swa : inp_attn->self_attn_mask;

    // TODO: support kv_rows - fill the distances over the kv_rows cell lists and pass the rows to build_attn_mha,
    //       needs a kv_rows flash attention with a per-head mask (CUDA has none)
    GGML_ASSERT(attn_mask.rows == nullptr);

    // a mask per head, in the dense layout of the input mask, which flash attention takes in F16
    ggml_tensor * kq_mask = attn_mask.raw;
    if (kq_mask->type != GGML_TYPE_F32) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    }
    kq_mask = ggml_add(ctx0, kq_b, kq_mask);
    if (cparams.flash_attn) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }
    cb(kq_mask, "kq_mask_rel_pos", il);

    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);
}

// INKLSA: rows of a cache view [d, h, n_kv] (cells nb[2] apart, each cell's heads packed) grouped `cells` at a time,
// reinterpreted as I32 words so GET_ROWS copies raw bytes ([cell_bytes*cells/4, n_kv/cells])
static ggml_tensor * inklsa_words(ggml_context * ctx, ggml_tensor * t, int64_t cells) {
    const size_t cell_bytes = ggml_row_size(t->type, t->ne[0])*t->ne[1];
    GGML_ASSERT(t->nb[0] == ggml_type_size(t->type) && t->nb[1] == ggml_row_size(t->type, t->ne[0]) && t->nb[2] == cell_bytes);
    GGML_ASSERT(t->ne[2] % cells == 0 && (cell_bytes*cells) % sizeof(int32_t) == 0);
    ggml_tensor * w = ggml_view_2d(ctx, t, t->ne[0], t->ne[2]/cells, cell_bytes*cells, 0);
    w->type  = GGML_TYPE_I32;
    w->ne[0] = (int64_t) (cell_bytes*cells/sizeof(int32_t));
    w->ne[1] = t->ne[2]/cells;
    w->nb[0] = sizeof(int32_t);
    w->nb[1] = cell_bytes*cells;
    w->nb[2] = w->nb[1]*w->ne[1];
    w->nb[3] = w->nb[2];
    return w;
}

// gather rows of inklsa_words (ids I32 [n]) back as [d, h, n*cells] of t's type (contiguous)
static ggml_tensor * inklsa_gather(ggml_context * ctx, ggml_tensor * t, int64_t cells, ggml_tensor * ids) {
    ggml_tensor * g = ggml_get_rows(ctx, inklsa_words(ctx, t, cells), ids);
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    ggml_tensor * r = ggml_view_2d(ctx, g, g->ne[0], g->ne[1], g->nb[1], 0);
    r->type  = t->type;
    r->ne[0] = t->ne[0];
    r->ne[1] = t->ne[1];
    r->ne[2] = ids->ne[0]*cells;
    r->ne[3] = 1;
    r->nb[0] = ggml_type_size(t->type);
    r->nb[1] = row_bytes;
    r->nb[2] = row_bytes*t->ne[1];
    r->nb[3] = r->nb[2]*r->ne[2];
    return r;
}

// mean over the B cells of each block: x F16 [D, B, n] (any strides) -> F32 [D, n]
static ggml_tensor * inklsa_block_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t D = x->ne[0], B = x->ne[1], n = x->ne[2];
    ggml_tensor * f = ggml_cast(ctx, x, GGML_TYPE_F32);                       // [D, B, n] contiguous
    f = ggml_cont(ctx, ggml_permute(ctx, f, 1, 0, 2, 3));                     // [B, D, n]
    f = ggml_sum_rows(ctx, f);                                                // [1, D, n]
    return ggml_reshape_2d(ctx, ggml_scale(ctx, f, 1.0f/B), D, n);
}

// two attention outputs [dv*n_head, T] with their LSE [n_head, T] and the zero-logit sink: softmax over
// (l_a, l_b, 0) per head and row weighs o_a and o_b
static ggml_tensor * inkrel_merge(ggml_context * ctx0, ggml_tensor * o_a, ggml_tensor * l_a, ggml_tensor * o_b,
        ggml_tensor * l_b, int64_t n_head, int64_t T) {
    ggml_tensor * l0 = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head, T), 0.0f);
    ggml_tensor * l3 = ggml_concat(ctx0, ggml_concat(ctx0, ggml_reshape_3d(ctx0, l_a, 1, n_head, T),
            ggml_reshape_3d(ctx0, l_b, 1, n_head, T), 0), ggml_reshape_3d(ctx0, l0, 1, n_head, T), 0);  // [3, n_head, T]
    ggml_tensor * w = ggml_soft_max(ctx0, l3);
    ggml_tensor * w_a = ggml_view_3d(ctx0, w, 1, n_head, T, w->nb[1], w->nb[2], 0);
    ggml_tensor * w_b = ggml_view_3d(ctx0, w, 1, n_head, T, w->nb[1], w->nb[2], w->nb[0]);
    const int64_t dv = o_a->ne[0]/n_head;
    ggml_tensor * cur = ggml_add(ctx0,
            ggml_mul(ctx0, ggml_reshape_3d(ctx0, o_a, dv, n_head, T), ggml_cont(ctx0, w_a)),
            ggml_mul(ctx0, ggml_reshape_3d(ctx0, o_b, dv, n_head, T), ggml_cont(ctx0, w_b)));
    return ggml_reshape_2d(ctx0, cur, dv*n_head, T);
}

ggml_tensor * llm_graph_context::build_attn_inkrel(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * q_cur,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * table,
              float   kq_scale,
                int   il) const {
    const auto & in = inp->inkrel;
    const int64_t n_head = table->ne[0];
    const int64_t E      = table->ne[1] - 1;
    const int64_t T      = n_tokens;
    GGML_ASSERT(in.on && k->ne[3] == 1 && in.far_mask->ne[0] == k->ne[2] && E == (int64_t) inp->extent);
    // ordering: the LSE outputs are leaves written through FA src[9] (no graph edge); build_attn_mha expands each FA
    // node before the merge below is built, so the default in-order schedule runs both FA calls first

    // (a) far: every cell at distance >= E, broadcast mask; its bias is the per-(head,row) constant table[:, E, :]
    lsa_lse_req = true;
    ggml_tensor * o_f = build_attn_mha(q_cur, k, v, nullptr, in.far_mask, nullptr, nullptr, nullptr, 0, kq_scale, il);
    ggml_tensor * l_f = lsa_lse;                                                              // [n_head, T]
    lsa_lse = nullptr;
    GGML_ASSERT(l_f != nullptr);
    ggml_tensor * c_far = ggml_view_2d(ctx0, table, n_head, T, table->nb[2], E*table->nb[1]);
    l_f = ggml_add(ctx0, ggml_reshape_2d(ctx0, l_f, n_head, T), c_far);

    // (b) near: the cells within E of some row, gathered, with the per-head bias
    ggml_tensor * kn = inklsa_gather(ctx0, k, 1, in.near_ids);
    ggml_tensor * vn = inklsa_gather(ctx0, v, 1, in.near_ids);
    ggml_tensor * mn = ggml_get_rows(ctx0, table, in.near_rel);                               // [n_head, n_near, T]
    mn = ggml_cont(ctx0, ggml_permute(ctx0, mn, 2, 0, 1, 3));                                 // [n_near, T, n_head]
    mn = ggml_cast(ctx0, ggml_add(ctx0, mn, in.near_mask), GGML_TYPE_F16);
    cb(mn, "inkrel_near_mask", il);
    lsa_lse_req = true;
    ggml_tensor * o_n = build_attn_mha(q_cur, kn, vn, nullptr, mn, nullptr, nullptr, nullptr, 0, kq_scale, il);
    ggml_tensor * l_n = lsa_lse;
    lsa_lse = nullptr;
    GGML_ASSERT(l_n != nullptr);
    l_n = ggml_reshape_2d(ctx0, l_n, n_head, T);

    ggml_tensor * cur = inkrel_merge(ctx0, o_f, l_f, o_n, l_n, n_head, T);
    cb(cur, "kqv_out", il);
    return cur;
}

ggml_tensor * llm_graph_context::build_attn_rel_sj(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
              float   kq_scale,
                int   il,
        ggml_tensor ** lse_out) const {
    auto * inp_attn = inp->get_attn();
    const auto * mctx_cur = inp_attn->mctx->get_base();
    GGML_ASSERT(mctx_cur->is_sj_kvarn() && !cparams.training);
    GGML_ASSERT(!inp_attn->self_k_rot[il] && !inp_attn->self_v_rot[il] && !mctx_cur->get_k_scale(il) && !mctx_cur->get_v_scale(il));
    GGML_ASSERT(mctx_cur->get_sj_wush(il) == nullptr && "INKKV: WUSH-KV transforms are not supported");
    // the per-q-head mask needs the sinks: CUDA serves SJ-KVaRN per-head masks without sinks on the stream decode
    // kernel, which reads one mask row per KV group (LSA per-KV-head picks)
    GGML_ASSERT((kq_mask->ne[2] == 1 || sinks != nullptr) && "INKKV: the per-q-head mask route needs the sinks tensor");

    const int64_t dh  = q_cur->ne[0];
    const int64_t H   = q_cur->ne[1];
    const int64_t Hkv = k_cur->ne[1];
    const int64_t T   = q_cur->ne[2];
    GGML_ASSERT(dh == 128 && k_cur->ne[0] == dh && v_cur->ne[0] == dh && v_cur->ne[1] == Hkv);
    GGML_ASSERT(Hkv % 2 == 0 && H % Hkv == 0 && q_cur->ne[3] == 1);
    const int64_t g = H/Hkv; // q heads per KV head
    const int64_t P = Hkv/2; // 256-wide cache heads: KV heads 2i (lanes [0, 128)) and 2i+1 (lanes [128, 256))

    // Q: packed head 2g*i + c (= q head 2g*i + c) holds [q, 0] for c < g (KV head 2i) and [0, q] for c >= g (KV head
    // 2i+1); the zero half meets the other KV head, so every logit is the 128-wide one, and the mask, sinks and
    // output head order are unchanged (GQA ratio 2g over P heads)
    ggml_tensor * q;
    {
        const size_t hs = q_cur->nb[1];
        ggml_tensor * qe = ggml_view_4d(ctx0, q_cur, dh, g, P, T, hs, 2*g*hs, q_cur->nb[2], 0);
        ggml_tensor * qo = ggml_view_4d(ctx0, q_cur, dh, g, P, T, hs, 2*g*hs, q_cur->nb[2], g*hs);
        ggml_tensor * z  = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, q_cur->type, dh, g, P, T), 0.0f);
        q = ggml_concat(ctx0, ggml_concat(ctx0, qe, z, 0), ggml_concat(ctx0, z, qo, 0), 1); // [256, 2g, P, T]
        q = ggml_reshape_3d(ctx0, q, 2*dh, H, T);
    }
    ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_is_contiguous(k_cur) ? k_cur : ggml_cont(ctx0, k_cur), 2*dh, P, T);
    ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_is_contiguous(v_cur) ? v_cur : ggml_cont(ctx0, v_cur), 2*dh, P, T);
    cb(q, "q_inkkv_pack", il);

    // the SJ-KVaRN rotation, step for step as build_attn (llm_graph_input_attn_kv) does it
    const bool fused_rot = mctx_cur->get_sj_kvarn().body_type != GGML_TYPE_TURBO4_0 && ggml_sj_kvarn_fused_rot_enabled();
    const int group = mctx_cur->get_sj_kvarn().body_type == GGML_TYPE_TURBO4_0 ? 128 : 256;
    if (!fused_rot) {
        q = ggml_turbo_wht(ctx0, q, 0, group, nullptr);
        cb(q, "q_sj_kvarn_rot", il);
    }
    k = ggml_turbo_wht(ctx0, k, 0, group, nullptr);
    v = ggml_turbo_wht(ctx0, v, 0, group, nullptr);
    cb(k, "k_sj_kvarn_rot", il);
    cb(v, "v_sj_kvarn_rot", il);
    if (mctx_cur->sj_kvarn_fused_rot()) {
        k = llm_graph_sj_kvarn_unrotate(k); // #139: rotated inside the cache write
        v = llm_graph_sj_kvarn_unrotate(v);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k, inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v, inp_attn->get_v_idxs(), il));

    uint32_t bits_k = 0, bits_v = 0;
    const int tier = mctx_cur->get_sj_kvarn_layer_tier(il, bits_k, bits_v);
    GGML_ASSERT(tier == 0);
    sj_kvarn_pending.body         = mctx_cur->get_sj_kvarn_body(il);
    sj_kvarn_pending.desc         = inp->self_sj_kvarn_desc;
    sj_kvarn_pending.bits_k       = (int32_t) bits_k;
    sj_kvarn_pending.bits_v       = (int32_t) bits_v;
    sj_kvarn_pending.fused_rot    = fused_rot;
    sj_kvarn_pending.n_kv_streams = &inp->sj_kvarn_n_kv_streams;
    sj_kvarn_pending.wush_ao      = nullptr;

    lsa_lse_req = lse_out != nullptr;
    ggml_tensor * cur = build_attn_mha(q, mctx_cur->get_k(ctx0, il), mctx_cur->get_v(ctx0, il), nullptr, kq_mask, nullptr, sinks,
            nullptr, 0, kq_scale, il); // [256*H, T], un-rotated
    GGML_ASSERT(sj_kvarn_pending.body == nullptr);
    if (lse_out != nullptr) {
        // packed q head index == q head index, so the LSE rows are the 128-wide heads' (the rotation keeps the logits)
        GGML_ASSERT(lsa_lse != nullptr && lsa_lse->ne[0] == H);
        *lse_out = ggml_reshape_2d(ctx0, lsa_lse, H, T);
        lsa_lse = nullptr;
    }

    // unpack: packed head 2g*i + c keeps lanes [0, 128) for c < g and [128, 256) for c >= g
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    {
        const size_t es = ggml_element_size(cur);
        ggml_tensor * oe = ggml_view_4d(ctx0, cur, dh, g, P, T, 2*dh*es, 2*g*2*dh*es, H*2*dh*es, 0);
        ggml_tensor * oo = ggml_view_4d(ctx0, cur, dh, g, P, T, 2*dh*es, 2*g*2*dh*es, H*2*dh*es, g*2*dh*es + dh*es);
        cur = ggml_reshape_2d(ctx0, ggml_concat(ctx0, oe, oo, 1), dh*H, T);
    }
    cb(cur, "kqv_out", il);
    return cur;
}

ggml_tensor * llm_graph_context::build_attn_rel(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * table,
        ggml_tensor * sinks,
              float   kq_scale,
                int   il,
        ggml_tensor * gate) const {
    bool gated = false;
    ggml_tensor * cur = build_attn_rel_base(inp, q_cur, k_cur, v_cur, table, sinks, kq_scale, il, gate, gated);
    if (gate && !gated) {
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    }
    return cur;
}

// INKATTN (ggml-inkrel.h): the relative bias table, rel_pos and the output gate taken inside one FA node at decode-sized
// ubatches. Off with LLAMA_INKATTN_REL=0 or GGML_CUDA_INKATTN=0 (the dense per-head mask path then).
static bool llama_inkattn_rel_on() {
    static const bool on = [] {
        const char * a = getenv("LLAMA_INKATTN_REL");
        const char * b = getenv("GGML_CUDA_INKATTN");
        return !(a && *a && atoi(a) == 0) && !(b && *b && atoi(b) == 0);
    }();
    return on;
}

// INKATTN minimum cache depth. Default 0 = every depth: with INKGLUE RELTAB (the rel table in one launch) INKATTN is
// at least as fast as the dense mask path at 3K and 16K (g3, 3090 Ti, AI4 fold, chat temp 1: 3K 111.67 vs 111.58,
// 16K 104.09 vs 103.56 tok/s). GGML_CUDA_INKATTN_MIN_KV=N = dense below N cells (e.g. 16384 for builds without RELTAB).
// The CUDA side (inkattn.cu) reads the same variable for plain FA nodes.
static int64_t llama_inkattn_min_kv() {
    static const int64_t v = [] {
        const char * e = getenv("GGML_CUDA_INKATTN_MIN_KV");
        return e && *e ? (int64_t) atoll(e) : (int64_t) 0;
    }();
    return v;
}

ggml_tensor * llm_graph_context::build_attn_rel_base(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * table,
        ggml_tensor * sinks,
              float   kq_scale,
                int   il,
        ggml_tensor * gate,
               bool & gated) const {
    gated = false;
    const bool is_swa = hparams.is_swa(il);
    auto * inp_attn = inp->get_attn();
    const int64_t n_head = table->ne[0];

    const auto * mctx_cur = inp_attn->mctx->get_base();
    ggml_tensor * kc = nullptr;
    if (!is_swa && inp->inklsa_on) {
        ggml_tensor * kv = mctx_cur->get_k(ctx0, il);
        kc = kv->view_src ? kv->view_src : kv;
        if (!kc->buffer || kv->type != GGML_TYPE_F16 || kv->ne[3] != 1 || inp_attn->self_k_rot[il] || inp_attn->self_v_rot[il] ||
                mctx_cur->get_k_scale(il) || mctx_cur->get_v_scale(il)) {
            kc = nullptr; // memory-fit dry run, or a cache INKLSA does not cover: dense
        }
    }

    // INKLSA on but this graph does not keep the index (n_kv % B != 0, no flash attention, several sequences, ...):
    // the K it writes is not folded into the index, so mark the index stale (rebuilt by the next INKLSA graph)
    if (!is_swa && !inp->inklsa_on && llama_inklsa_cfg().on) {
        ggml_tensor * kv = mctx_cur->get_k(ctx0, il);
        const ggml_tensor * kst = kv->view_src ? kv->view_src : kv;
        if (kst->buffer) {
            llama_inklsa_invalidate(&kst, 1);
        }
    }

    // INKATTN: decode-sized ubatches (q heads per KV head x tokens <= 16) on a plain F16 cache: one FA node with the
    // bias table + rel_pos (+ the output gate) inside, no [n_kv, T, n_head] mask; same f16 bias values as the dense path
    // (not under -sm tensor: the meta backend splits heads and does not split the table / gate)
    if (kc == nullptr && llama_inkattn_rel_on() && cparams.flash_attn && !cparams.training && cparams.kv_stream_arena_mib == 0 &&
            model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
        const auto * mctx_l = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
        const auto & am     = is_swa ? inp_attn->self_attn_mask_swa : inp_attn->self_attn_mask;
        ggml_tensor * raw     = am.raw;
        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;
        ggml_tensor * kv = mctx_l->get_k(ctx0, il);
        ggml_tensor * vv = mctx_l->get_v(ctx0, il);
        const int64_t n_head_kv = kv->ne[1];
        const int64_t n_tok     = q_cur->ne[2];
        const bool rot = is_swa ? (inp_attn->self_k_rot_swa[il] || inp_attn->self_v_rot_swa[il]) :
                                  (inp_attn->self_k_rot[il] || inp_attn->self_v_rot[il]);
        if (am.rows == nullptr && raw->type == GGML_TYPE_F16 && raw->ne[2] == 1 && raw->ne[3] == 1 &&
                kv->type == GGML_TYPE_F16 && vv->type == GGML_TYPE_F16 && kv->ne[3] == 1 && vv->nb[1] < vv->nb[2] &&
                kv->ne[0] == 128 && vv->ne[0] == 128 && q_cur->ne[0] == 128 && n_head % n_head_kv == 0 &&
                (n_head/n_head_kv)*n_tok <= 16 && rel_pos->ne[0] == kv->ne[2] && rel_pos->ne[1] == n_tok &&
                kv->ne[2] >= llama_inkattn_min_kv() &&
                !rot && !mctx_l->get_k_scale(il) && !mctx_l->get_v_scale(il) &&
                hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap &&
                sinks && sinks->type == GGML_TYPE_F32 && (!gate || gate->type == GGML_TYPE_F32)) {
            ggml_build_forward_expand(gf, q_cur);
            ggml_build_forward_expand(gf, k_cur);
            ggml_build_forward_expand(gf, v_cur);
            ggml_build_forward_expand(gf, mctx_l->cpy_k(ctx0, k_cur, is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_l->cpy_v(ctx0, v_cur, is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs(), il));
            ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx0, mctx_l->get_k(ctx0, il), 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx0, mctx_l->get_v(ctx0, il), 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx0, q, k, v, raw, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, sinks);
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            ggml_tensor * g2 = gate && ggml_is_contiguous(gate) ? ggml_reshape_2d(ctx0, gate, q_cur->ne[0]*n_head, n_tok) : nullptr;
            ggml_flash_attn_ext_set_inkrel(fa, table, rel_pos, g2);
            gated = g2 != nullptr;
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, fa, il}); // the auto-FA probe checks this node's device
            ggml_tensor * cur = ggml_reshape_2d(ctx0, fa, fa->ne[0]*fa->ne[1], fa->ne[2]*fa->ne[3]);
            cb(cur, "kqv_out", il);
            return cur;
        }
    }

    // INKREL: a cache the split covers (plain K/V types, no rotation or scales, one stream)
    if (!is_swa && kc == nullptr && inp->inkrel.on) {
        ggml_tensor * kv = mctx_cur->get_k(ctx0, il);
        ggml_tensor * vv = mctx_cur->get_v(ctx0, il);
        // F16 only: other types lose the bounded-prefill FA plan once an LSE output is attached (larger reserve);
        // no kv-stream (host-paged K/V: the near gather and the LSE need device-resident K/V)
        if (kv->type == GGML_TYPE_F16 && vv->type == GGML_TYPE_F16 && cparams.kv_stream_arena_mib == 0 && kv->ne[3] == 1 && vv->nb[1] < vv->nb[2] && !inp_attn->self_k_rot[il] &&
                !inp_attn->self_v_rot[il] && !mctx_cur->get_k_scale(il) && !mctx_cur->get_v_scale(il)) {
            ggml_build_forward_expand(gf, q_cur);
            ggml_build_forward_expand(gf, k_cur);
            ggml_build_forward_expand(gf, v_cur);
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs(), il));
            GGML_UNUSED(sinks); // zero sink logit (folded into table)
            return build_attn_inkrel(inp, q_cur, mctx_cur->get_k(ctx0, il), mctx_cur->get_v(ctx0, il), table, kq_scale, il);
        }
    }

    if (kc == nullptr) {
        // dense: gather the bias of every cell by its distance
        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;
        const int64_t n_stream = (is_swa ? inp_attn->self_attn_mask_swa.raw : inp_attn->self_attn_mask.raw)->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
        kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
        cb(kq_b, "kq_b", il);

        if (!is_swa && inp->self_sj_far_mask != nullptr) {
            // INKKV v2: far/near split. Inkling's bias is table[h, d, t] for a cell at distance d < E (E = relative extent
            // = SWA window) and the per-(head,row) constant table[h, E, t] beyond, so (as INKREL does for F16 caches):
            //   far : the SJ-KVaRN 4/4 cache, cells at distance >= E (broadcast [n_kv, T] mask: the fast GQA routes, no
            //         [n_kv, T, n_head] mask), log-sum-exp + table[:, E, :]
            //   near: the F16 window copy of this layer in the SWA cache (distance < E), the gathered per-head bias
            //   merge: softmax over (far, near, zero sink) log-sum-exps (inkrel_merge)
            const auto * mswa = inp_attn->mctx->get_swa();
            ggml_tensor * mask_swa = inp_attn->self_attn_mask_swa.raw;
            GGML_ASSERT(inp_attn->self_attn_mask_swa.rows == nullptr && mask_swa->type != GGML_TYPE_I32 && mask_swa->ne[3] == 1);
            const int64_t E = table->ne[1] - 1;
            GGML_ASSERT(E == (int64_t) inp->extent && E == (int64_t) inp->extent_swa && mswa->has_layer(il));
            ggml_build_forward_expand(gf, q_cur);
            ggml_build_forward_expand(gf, k_cur);
            ggml_build_forward_expand(gf, v_cur);
            ggml_build_forward_expand(gf, mswa->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs_swa(), il));
            ggml_build_forward_expand(gf, mswa->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs_swa(), il));

            const int64_t n_kv_n = mask_swa->ne[0];
            const int64_t T      = n_tokens;
            ggml_tensor * mn = ggml_get_rows(ctx0, table, inp->self_rel_pos_swa);                   // [n_head, n_kv_n*T]
            mn = ggml_reshape_4d(ctx0, mn, n_head, n_kv_n, T, 1);
            mn = ggml_cont(ctx0, ggml_permute(ctx0, mn, 2, 0, 1, 3));                                // [n_kv_n, T, n_head]
            ggml_tensor * ms = mask_swa->type == GGML_TYPE_F32 ? mask_swa : ggml_cast(ctx0, mask_swa, GGML_TYPE_F32);
            mn = ggml_cast(ctx0, ggml_add(ctx0, mn, ms), GGML_TYPE_F16);
            cb(mn, "inkkv_near_mask", il);
            lsa_lse_req = true;
            ggml_tensor * o_n = build_attn_mha(q_cur, mswa->get_k(ctx0, il), mswa->get_v(ctx0, il), nullptr, mn, nullptr, nullptr,
                    nullptr, 0, kq_scale, il);
            ggml_tensor * l_n = lsa_lse;
            lsa_lse = nullptr;
            GGML_ASSERT(l_n != nullptr);
            l_n = ggml_reshape_2d(ctx0, l_n, n_head, T);

            ggml_tensor * l_f = nullptr;
            ggml_tensor * o_f = build_attn_rel_sj(inp, q_cur, k_cur, v_cur, inp->self_sj_far_mask, nullptr, kq_scale, il, &l_f);
            ggml_tensor * c_far = ggml_view_2d(ctx0, table, n_head, T, table->nb[2], E*table->nb[1]);
            l_f = ggml_add(ctx0, l_f, c_far);

            ggml_tensor * cur = inkrel_merge(ctx0, o_f, l_f, o_n, l_n, n_head, T);
            cb(cur, "kqv_out", il);
            return cur;
        }
        if (!is_swa && inp->self_sj_kvarn_desc != nullptr) {
            // INKKV: SJ-KVaRN full-attention cache, the same per-head mask as build_attn below
            ggml_tensor * kq_mask = inp_attn->self_attn_mask.raw;
            GGML_ASSERT(inp_attn->self_attn_mask.rows == nullptr && kq_mask->type != GGML_TYPE_I32);
            if (kq_mask->type != GGML_TYPE_F32) {
                kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
            }
            kq_mask = ggml_cast(ctx0, ggml_add(ctx0, kq_b, kq_mask), GGML_TYPE_F16);
            cb(kq_mask, "kq_mask_rel_pos", il);
            const auto * mswa = inp_attn->mctx->get_swa();
            if (mswa->has_layer(il)) {
                // INKKV v2: keep the F16 window copy current on the dense ubatches too (the split reads it later)
                ggml_build_forward_expand(gf, k_cur);
                ggml_build_forward_expand(gf, v_cur);
                ggml_build_forward_expand(gf, mswa->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs_swa(), il));
                ggml_build_forward_expand(gf, mswa->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs_swa(), il));
            }
            return build_attn_rel_sj(inp, q_cur, k_cur, v_cur, kq_mask, sinks, kq_scale, il);
        }

        return build_attn(inp, nullptr, nullptr, nullptr, q_cur, k_cur, v_cur, kq_b, sinks, kq_scale, il);
    }

    auto & in = inp->inklsa;
    const int64_t B = in.B;

    // store K and V first: the index refresh below re-reads the blocks this ubatch writes
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs(), il));

    ggml_tensor * k = mctx_cur->get_k(ctx0, il); // [d, h_kv, n_kv, 1]
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(v->type == GGML_TYPE_F16 && v->nb[1] < v->nb[2] && "INKLSA: needs a non-transposed F16 V cache");
    const int64_t D = k->ne[0]*k->ne[1];

    llama_inklsa_idx idx;
    GGML_ASSERT(llama_inklsa_idx_get(kc, D, llama_inklsa_idx_rows(kc->ne[1], B), ggml_backend_buffer_get_type(kc->buffer), &idx));
    const bool    comp  = llama_inklsa_cfg().comp;
    const int64_t vrow0 = comp ? idx.t->ne[1]/2 : 0; // first mean-V row
    inp->inklsa_kcaches.push_back(kc);
    if (!idx.valid) {
        // rebuild every block [0, n_blk) from K, in chunks that bound the F32 scratch
        const int64_t CH = 1024;
        for (int64_t b0 = 0; b0 < in.n_blk; b0 += CH) {
            const int64_t nb = std::min(CH, in.n_blk - b0);
            ggml_tensor * x = ggml_view_3d(ctx0, k, D, B, nb, k->nb[2], k->nb[2]*B, k->nb[2]*B*b0);
            ggml_tensor * dst = ggml_view_2d(ctx0, idx.t, D, nb, idx.t->nb[1], idx.t->nb[1]*b0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, inklsa_block_mean(ctx0, x), dst));
            if (comp) {
                ggml_tensor * xv = ggml_view_3d(ctx0, v, D, B, nb, v->nb[2], v->nb[2]*B, v->nb[2]*B*b0);
                ggml_tensor * dv = ggml_view_2d(ctx0, idx.t, D, nb, idx.t->nb[1], idx.t->nb[1]*(vrow0 + b0));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, inklsa_block_mean(ctx0, xv), dv));
            }
        }
        inp->inklsa_commit.emplace_back(kc, idx.epoch);
        inp->inklsa_rebuild = true;
    }
    // refresh the blocks this ubatch wrote (rows past the newest position are stale until written; such a block is
    // never eligible before it is whole)
    ggml_tensor * ub = inklsa_gather(ctx0, k, B, in.u_blk);                                   // [128, 8, n_u*B]
    ub = ggml_reshape_3d(ctx0, ub, D, B, in.n_u);
    ggml_tensor * index = ggml_set_rows(ctx0, idx.t, inklsa_block_mean(ctx0, ub), in.u_blk64); // view of idx.t
    ggml_build_forward_expand(gf, index);
    ggml_tensor * vindex = nullptr; // COMP: mean-V rows, refreshed the same way
    if (comp) {
        ggml_tensor * uv = ggml_reshape_3d(ctx0, inklsa_gather(ctx0, v, B, in.u_blk), D, B, in.n_u);
        ggml_tensor * vrows = ggml_view_2d(ctx0, idx.t, D, vrow0, idx.t->nb[1], idx.t->nb[1]*vrow0);
        vindex = ggml_set_rows(ctx0, vrows, inklsa_block_mean(ctx0, uv), in.u_blk64);
        ggml_build_forward_expand(gf, vindex);
    }

    if (!in.sparse && inp->inkrel.on) {
        return build_attn_inkrel(inp, q_cur, k, v, table, kq_scale, il);
    }

    if (!in.sparse) {
        // dense attention over the full cache (K/V already stored: pass no k_cur / v_cur)
        ggml_tensor * rel_pos = inp->self_rel_pos;
        const int64_t n_kv = rel_pos->ne[0];
        ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
        kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens, 1);
        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
        ggml_tensor * kq_mask = inp_attn->self_attn_mask.raw;
        if (kq_mask->type != GGML_TYPE_F32) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        }
        kq_mask = ggml_cast(ctx0, ggml_add(ctx0, kq_b, kq_mask), GGML_TYPE_F16);
        ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, kq_mask, nullptr, sinks, nullptr, 0, kq_scale, il);
        cb(cur, "kqv_out", il);
        return cur;
    }

    {
        static std::atomic<int> n_logged{0};
        if (n_logged.fetch_add(1) == 0) {
            LLAMA_LOG_WARN("%s: INKLSA sparse attention active (first sparse graph: n_kv %lld, T %lld, keep %lld, fixed %lld)\n",
                    __func__, (long long) (in.n_blk*B), (long long) n_tokens, (long long) in.n_keep, (long long) in.n_fix);
        }
    }

    // pick: score every block by its mean key, softmax over blocks per (q head, row), sum the shares
    const int64_t T   = n_tokens;
    const int64_t Hkv = k->ne[1];
    const int64_t dh  = k->ne[0];
    ggml_tensor * qbd = ggml_reshape_4d(ctx0, q_cur, dh, 1, n_head, T);
    qbd = ggml_mul(ctx0, ggml_repeat_4d(ctx0, qbd, dh, Hkv, n_head, T), in.onehot);           // block-diagonal per KV group
    qbd = ggml_reshape_2d(ctx0, qbd, D, n_head*T);
    ggml_tensor * iview = ggml_view_2d(ctx0, index, D, in.n_blk, idx.t->nb[1], 0);
    ggml_tensor * sc = ggml_mul_mat(ctx0, iview, qbd);                                       // [n_blk, n_head*T]
    sc = ggml_soft_max(ctx0, ggml_add(ctx0, ggml_scale(ctx0, sc, kq_scale), in.elig));
    if (llama_inklsa_cfg().pick == 1) {
        // oracle (diagnostic): the exact attention mass of each eligible block from every key, softmax over the
        // eligible cells per (q head, row), summed like the index votes
        const int64_t H = n_head;
        ggml_tensor * kq = ggml_mul_mat(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), ggml_permute(ctx0, q_cur, 0, 2, 1, 3)); // [n_kv, T, H]
        kq = ggml_reshape_3d(ctx0, ggml_scale(ctx0, kq, kq_scale), B, in.n_blk, T*H);
        kq = ggml_add(ctx0, kq, ggml_reshape_3d(ctx0, in.elig, 1, in.n_blk, 1));
        kq = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, kq, B*in.n_blk, T*H));
        kq = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, kq, B, in.n_blk, T*H));                // [1, n_blk, T*H]
        sc = ggml_reshape_2d(ctx0, kq, in.n_blk, T*H);
    }
    ggml_tensor * votes = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, sc)));    // [1, n_blk]
    // ineligible blocks vote 0 after the softmax, like eligible blocks whose scores underflow: keep them last
    votes = ggml_add(ctx0, ggml_reshape_2d(ctx0, votes, in.n_blk, 1), in.elig);
    const int64_t nk = in.n_keep/B;
    ggml_tensor * pick = ggml_top_k(ctx0, votes, (int) nk);                                   // I32 [nk, 1]
    pick = ggml_reshape_1d(ctx0, pick, nk);
    cb(pick, "inklsa_pick", il);

    // compact K/V: fixed cells, then the picked blocks
    ggml_tensor * ks = ggml_concat(ctx0, inklsa_gather(ctx0, k, 1, in.fix_ids), inklsa_gather(ctx0, k, B, pick), 2);
    ggml_tensor * vs = ggml_concat(ctx0, inklsa_gather(ctx0, v, 1, in.fix_ids), inklsa_gather(ctx0, v, B, pick), 2);

    // mask [n_sel, T, n_head]: fixed cells get their bias row (+ causal/padding mask), picked blocks the far row
    // (index extent) of every head and row, -inf for an ineligible pick (fewer eligible blocks than nk)
    const int64_t E = table->ne[1] - 1;
    ggml_tensor * mf = ggml_get_rows(ctx0, table, in.fix_rel);                               // [n_head, n_fix, T]
    mf = ggml_cont(ctx0, ggml_permute(ctx0, mf, 2, 0, 1, 3));                                 // [n_fix, T, n_head]
    mf = ggml_add(ctx0, mf, in.fix_mask);
    ggml_tensor * far = ggml_view_3d(ctx0, table, n_head, 1, T, table->nb[1], table->nb[2], E*table->nb[1]);
    far = ggml_cont(ctx0, ggml_permute(ctx0, far, 2, 0, 1, 3));                              // [1, T, n_head]
    ggml_tensor * pe = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, in.elig, 1, in.n_blk), pick); // [1, nk]
    pe = ggml_reshape_4d(ctx0, ggml_repeat_4d(ctx0, pe, B, nk, 1, 1), B*nk, 1, 1, 1);
    ggml_tensor * mb = ggml_add(ctx0, ggml_repeat_4d(ctx0, far, B*nk, T, n_head, 1), pe);    // [nk*B, T, n_head]
    ggml_tensor * mask = ggml_cast(ctx0, ggml_concat(ctx0, mf, mb, 0), GGML_TYPE_F16);
    cb(mask, "inklsa_mask", il);

    if (!comp) {
        ggml_tensor * cur = build_attn_mha(q_cur, ks, vs, nullptr, mask, nullptr, sinks, nullptr, 0, kq_scale, il);
        cb(cur, "kqv_out", il);
        return cur;
    }

    // COMP (assumes the zero sink of inkling2.cpp, which folds the sink logit into table): the compact set without
    // the sink, then every eligible unpicked block as one virtual cell
    // (mean K, mean V, logit + log B + the far bias), merged with the zero sink by log-sum-exp
    lsa_lse_req = true;
    ggml_tensor * o_s = build_attn_mha(q_cur, ks, vs, nullptr, mask, nullptr, nullptr, nullptr, 0, kq_scale, il);
    ggml_tensor * l_s = lsa_lse;
    lsa_lse = nullptr;
    const int64_t n_pad = GGML_PAD(in.n_blk, 256);
    GGML_ASSERT(l_s && n_pad <= vrow0);
    ggml_tensor * kc3 = ggml_view_3d(ctx0, index,  dh, Hkv, n_pad, dh*ggml_element_size(index), index->nb[1], 0);
    ggml_tensor * vc3 = ggml_view_3d(ctx0, vindex, dh, Hkv, n_pad, dh*ggml_element_size(vindex), vindex->nb[1], 0);
    // per-block mask [n_pad]: log B for an eligible unpicked block, -inf for picked / ineligible / padding
    ggml_tensor * bm = ggml_scale_bias(ctx0, in.elig, 1.0f, logf((float) B));                 // [n_blk, 1]
    bm = ggml_set_rows(ctx0, ggml_reshape_2d(ctx0, bm, 1, in.n_blk),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, nk), -INFINITY), pick);
    bm = ggml_reshape_2d(ctx0, bm, in.n_blk, 1);
    if (n_pad > in.n_blk) {
        bm = ggml_concat(ctx0, bm, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pad - in.n_blk, 1), -INFINITY), 0);
    }
    ggml_tensor * mc = ggml_add(ctx0, ggml_repeat_4d(ctx0, far, n_pad, T, n_head, 1), bm);  // [n_pad, T, n_head]
    // a row with no unpicked eligible block would be all -inf (0/0 in FA): -30000 keeps it finite at LSE ~ -30000,
    // which the merge weighs as zero
    mc = ggml_cast(ctx0, ggml_clamp(ctx0, mc, -30000.0f, INFINITY), GGML_TYPE_F16);
    lsa_lse_req = true;
    ggml_tensor * o_c = build_attn_mha(q_cur, kc3, vc3, nullptr, mc, nullptr, nullptr, nullptr, 0, kq_scale, il);
    ggml_tensor * l_c = lsa_lse;
    lsa_lse = nullptr;
    GGML_ASSERT(l_c);
    ggml_tensor * cur = inkrel_merge(ctx0, o_s, l_s, o_c, l_c, n_head, T);
    cb(cur, "kqv_out", il);
    return cur;
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    const llm_arch arch = model.arch;

    if (!cparams.embeddings) {
        // a graph without logits still computes t_embd, so that it is never empty
        if (res->t_logits == nullptr && res->t_embd != nullptr) {
            ggml_build_forward_expand(gf, res->t_embd);
        }
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_MEAN) {
                    // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_unary(ctx0, cur, hparams.act_cls);
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_logits() const {
    // a pooled embeddings context outputs one vector per sequence, so its graph skips the LM head
    if (cparams.embeddings && pooling_type != LLAMA_POOLING_TYPE_NONE) {
        res->t_logits = nullptr;
    }

    // the models declare t_logits without expanding it, so the LM head is computed only when it is expanded here
    if (res->t_logits != nullptr) {
        ggml_build_forward_expand(gf, res->t_logits);
    }
}

void llm_graph_context::build_layer_inp_dev() const {
    if (!cparams.embeddings_layer_inp_dev || layer_inp_dev == nullptr) {
        return;
    }

    auto inp = std::make_unique<llm_graph_input_batch_idxs>();

    inp->idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
    ggml_set_input(inp->idxs);

    // row layout: [layer_0 | layer_1 | ...] in ascending layer id order
    int64_t col = 0;
    for (size_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }

        ggml_tensor * src = res->t_layer_inp[il];
        GGML_ASSERT(src != nullptr && "layer input tensor is null");
        GGML_ASSERT(src->ne[0] == n_embd);

        ggml_tensor * dst = ggml_view_2d(ctx0, layer_inp_dev, n_embd, layer_inp_dev->ne[1], layer_inp_dev->nb[1], col*n_embd*ggml_element_size(layer_inp_dev));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, dst, ggml_reshape_2d(ctx0, src, n_embd, n_tokens), inp->idxs));
        col++;
    }
    GGML_ASSERT(col*n_embd == layer_inp_dev->ne[0]);

    res->add_input(std::move(inp));
}

void llm_graph_context::build_h_nextn_dev() const {
    ggml_tensor * src = res->t_h_nextn;
    if (h_nextn_dev == nullptr || src == nullptr) {
        return;
    }

    // the device buffer is only enabled for the unmasked extraction, which has a row per batch token
    GGML_ASSERT(src->ne[0] == h_nextn_dev->ne[0] && ggml_nrows(src) == n_tokens);

    auto inp = std::make_unique<llm_graph_input_batch_idxs>();

    inp->idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
    ggml_set_input(inp->idxs);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, h_nextn_dev, ggml_reshape_2d(ctx0, src, src->ne[0], n_tokens), inp->idxs));

    res->add_input(std::move(inp));
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    auto * sampling_input = inp_sampling.get();
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    std::map<llama_seq_id, std::vector<int32_t>> seq_rows; // all logit rows of each sequence, in output order
    int32_t logit_row_idx = 0;

    // shortlisted logits: sampler picks are compact row indices and must be mapped back to token ids
    ggml_tensor * ids_map = res->t_logits_ids;

    // Which chains can share the single argmax over all output rows below.
    //
    // A bare greedy chain always can. A biased greedy chain ([logit-bias, greedy], e.g. a request with
    // ignore_eos or a model with suppress tokens) cannot: its bias has to be added to its own rows before
    // the argmax, so it runs a private row block further down. The exception is a draft graph with a
    // vocabulary shortlist, whose logit columns are compact shortlist positions rather than token ids: a
    // token-indexed bias would be misapplied there, and it is not needed either, because every draft token
    // is re-checked against the biased full-vocabulary argmax of the verification rows.
    auto shares_argmax = [&](const llama_sampler * sampler) {
        return llama_sampler_is_greedy_chain(sampler) ||
               (ids_map != nullptr && llama_sampler_is_argmax_chain(sampler));
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            sampling_input->output_seqs.push_back(seq_id);
            const auto it = samplers.find(seq_id);
            if (it != samplers.end() && shares_argmax(it->second)) {
                res->greedy_rows.push_back(logit_row_idx);
            }
            seq_to_logit_row[seq_id] = logit_row_idx;
            seq_rows[seq_id].push_back(logit_row_idx);
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    if (!res->greedy_rows.empty()) {
        res->t_greedy_rows = ggml_argmax(ctx0, res->t_logits);
        if (ids_map) {
            ggml_tensor * rows = ggml_reshape_2d(ctx0, ids_map, 1, ids_map->ne[0]);
            res->t_greedy_rows = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, rows, res->t_greedy_rows), res->t_greedy_rows->ne[0]);
        }
        ggml_set_name(res->t_greedy_rows, "greedy_verify_rows");
        ggml_build_forward_expand(gf, res->t_greedy_rows);
    }

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    // The single-row samplers below read row 0 at most when no row is an output, so the pad is only
    // needed when there are no logit rows; otherwise they view res->t_logits directly (same row
    // stride, same bytes) and the copy is not run. Graph reuse already compares n_outputs.
    ggml_tensor * logits_t = res->t_logits->ne[1] > 0 && ggml_is_contiguous(res->t_logits)
            ? res->t_logits : ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    // upstream: give stateful backend sampler chains a chance to reset before the graph is rebuilt
    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    for (const auto & [seq_id, sampler] : samplers) {
        if (shares_argmax(sampler)) {
            continue;
        }

        // a biased greedy chain exports token ids only, so it must never reach the single-row path below
        // (which would also export its full logit row); with no output rows in this ubatch it has nothing
        // to sample and is left out of the graph entirely
        const bool argmax_only = llama_sampler_is_argmax_chain(sampler);
        if (argmax_only && seq_rows.find(seq_id) == seq_rows.end()) {
            continue;
        }

        // row-block sampling: a sequence with several output rows (speculative verification) runs its
        // chain once over the [n_vocab, n_rows] block of its rows and exports only the sampled ids
        if (const auto it_rows = seq_rows.find(seq_id); it_rows != seq_rows.end() &&
                (it_rows->second.size() > 1 || argmax_only)) {
            const auto & rows = it_rows->second;
            const int64_t n_rows = (int64_t) rows.size();

            GGML_ASSERT(llama_sampler_backend_supports_rows(sampler) && "sampler cannot sample several rows of one sequence");

            ggml_tensor * t_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
            ggml_format_name(t_rows, "sampling_rows_%d", seq_id);
            ggml_set_input(t_rows);
            sampling_input->row_inputs[seq_id] = { t_rows, rows };

            ggml_tensor * logits_rows = ggml_get_rows(ctx0, res->t_logits, t_rows); // [n_vocab, n_rows]
            ggml_format_name(logits_rows, "logits_rows_%d", seq_id);

            struct llama_sampler_data data = {
                /*.logits      =*/ logits_rows,
                /*.probs       =*/ nullptr,
                /*.sampled     =*/ nullptr,
                /*.candidates  =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            // every chain is read on the host after compute: mark them all as outputs to prevent aliasing (upstream 40f4f4ac9)
            if (data.sampled     != nullptr) { res->t_sampling_all.push_back(data.sampled);     }
            if (data.probs       != nullptr) { res->t_sampling_all.push_back(data.probs);       }
            if (data.logits      != nullptr) { res->t_sampling_all.push_back(data.logits);      }
            if (data.candidates  != nullptr) { res->t_sampling_all.push_back(data.candidates);  }

            if (data.sampled == nullptr && !ids_map && data.candidates != nullptr && data.logits != nullptr && data.candidates->ne[1] == n_rows) {
                // truncation-only chain: export the kept ids and logits of every row, the host finishes the sampling
                ggml_format_name(data.candidates, "topk_ids_rows_%d", seq_id);
                ggml_format_name(data.logits, "topk_logits_rows_%d", seq_id);
                ggml_set_output(data.candidates);
                ggml_set_output(data.logits);
                ggml_build_forward_expand(gf, data.candidates);
                ggml_build_forward_expand(gf, data.logits);
                res->t_topk_rows[seq_id] = { data.candidates, data.logits };
                res->sampled_rows[seq_id].assign(rows.begin(), rows.end());
                continue;
            }

            GGML_ASSERT(data.sampled != nullptr && ggml_nelements(data.sampled) == n_rows && "row-block sampler chain produced no sampled ids");

            ggml_format_name(data.sampled, "sampled_rows_%d", seq_id);
            ggml_build_forward_expand(gf, data.sampled);

            res->t_sampled_rows[seq_id] = data.sampled;
            res->sampled_rows[seq_id].assign(rows.begin(), rows.end());

            continue;
        }

        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        // with a shortlist the candidate list starts as the id map, so the backend
        // samplers (top_k, temp, dist) emit real token ids
        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ ids_map,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        // every chain is read on the host after compute: mark them all as outputs to prevent aliasing (upstream 40f4f4ac9)
        if (data.sampled     != nullptr) { res->t_sampling_all.push_back(data.sampled);     }
        if (data.probs       != nullptr) { res->t_sampling_all.push_back(data.probs);       }
        if (data.logits      != nullptr) { res->t_sampling_all.push_back(data.logits);      }
        if (data.candidates  != nullptr) { res->t_sampling_all.push_back(data.candidates);  }

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}


// Only these head layouts reach the CUDA MMVQ one-row-expert path (no expert sorting, no dequantization).
// Returns nullptr when the head qualifies, else why it does not.
static const char * draft_vocab_direct(const ggml_tensor * head) {
    if (!head->buffer) {
        return "head has no buffer";
    }
    if (!ggml_is_contiguous(head)) {
        return "head is not contiguous";
    }
    auto * buft = ggml_backend_buffer_get_type(head->buffer);
    auto * dev  = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return "head buffer has no device";
    }
    // a Meta device (--split-mode tensor) has no backend registry: the head is sharded across devices there,
    // so the one-row-expert path does not apply and the full head is used instead
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
        return "head is not on a CUDA device";
    }
    if (buft != ggml_backend_dev_buffer_type(dev)) {
        return "head is in a split or host buffer";
    }
    switch (head->type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
            return nullptr;
        case GGML_TYPE_AIF25_U: case GGML_TYPE_AIF25_U2: case GGML_TYPE_AIF175_U2: case GGML_TYPE_AIF25_U2_512:
            {
                // [KAIFUSE R2] the CUDA row-group kernel (ggml_cuda_mul_mat_id_aif_rows), opt-in (LLAMA_RM24H_AIHEAD=1;
                // default off). No other backend runs this op, so ask the head's device about the exact node shape.
                if (llama_draft_vocab_aihead_mode() != 1) {
                    return "head type has no one-row MMVQ path";
                }
                ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
                ggml_context * pctx = ggml_init(ip);
                ggml_tensor * rows = ggml_new_tensor_3d(pctx, head->type, head->ne[0], 1, head->ne[1]);
                ggml_tensor * x    = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, head->ne[0], 1, 8);
                ggml_tensor * ids  = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, 1024, 8);
                const bool ok = ggml_backend_dev_supports_op(dev, ggml_mul_mat_id(pctx, rows, x, ids));
                ggml_free(pctx);
                return ok ? nullptr : "fractional AI+ head shape or device not supported by the row-group kernel";
            }
        default:
            return "head type has no one-row MMVQ path";
    }
}

enum draft_vocab_fallback_reason {
    DRAFT_VOCAB_FALLBACK_HADAMARD,
    DRAFT_VOCAB_FALLBACK_LORA,
    DRAFT_VOCAB_FALLBACK_HEAD_SCALE,
    DRAFT_VOCAB_FALLBACK_NOT_SMALLER,
    DRAFT_VOCAB_FALLBACK_HEAD_LAYOUT,
    DRAFT_VOCAB_FALLBACK_NO_BACKEND_SAMPLER,
};

static const char * draft_vocab_fallback_name(int reason) {
    switch (reason) {
        case DRAFT_VOCAB_FALLBACK_HADAMARD:           return "hadamard_head";
        case DRAFT_VOCAB_FALLBACK_LORA:               return "lora";
        case DRAFT_VOCAB_FALLBACK_HEAD_SCALE:         return "per_row_head_scale";
        case DRAFT_VOCAB_FALLBACK_NOT_SMALLER:        return "map_not_smaller_than_head";
        case DRAFT_VOCAB_FALLBACK_HEAD_LAYOUT:        return "head_layout";
        case DRAFT_VOCAB_FALLBACK_NO_BACKEND_SAMPLER: return "no_backend_sampler";
        default:                                      return "unknown";
    }
}

ggml_tensor * llm_graph_context::draft_vocab_fallback(int reason, const char * detail) const {
    // a reserve graph places its output rows on placeholder sequences, and the draft's backend sampler is attached
    // only after the context (and its first reserve) exists: a missing sampler there says nothing about decode,
    // so it is not warned about. A real decode graph without a sampler still warns.
    if (is_reserve && reason == DRAFT_VOCAB_FALLBACK_NO_BACKEND_SAMPLER) {
        return nullptr;
    }
    const uint32_t bit = 1u << reason;
    if (draft_vocab_warned == nullptr || (draft_vocab_warned->fetch_or(bit) & bit) == 0) {
        LLAMA_LOG_WARN("%s: draft vocabulary shortlist not applied (%s%s%s), the draft scores the full head; "
                       "logged once per context\n", __func__, draft_vocab_fallback_name(reason),
                       detail ? ": " : "", detail ? detail : "");
    }
    return nullptr;
}

ggml_tensor * llm_graph_context::build_draft_vocab_compact(
        ggml_tensor * head_w,
        ggml_tensor * head_s,
        ggml_tensor * cur) const {
    const int64_t n_sel = draft_vocab_compact->ne[1];
    GGML_ASSERT(n_sel == draft_vocab_ids->ne[0] && draft_vocab_compact->ne[0] == head_w->ne[0]);

    ggml_tensor * x = cur;
    if (ggml_exl3_bits(head_w->type) != 0) {
        // the compact rows live in the rotated input domain of the EXL3 head (llama-draft-vocab-compact.h): feed them
        // x_rot = H128(suh * x), the same input glue build_lora_mm builds with LLAMA_EXL3_GLUE_GRAPH=1
        const llama_model::exl3_side * exl3 = model_ref ? model_ref->exl3_side_of(head_w) : nullptr;
        GGML_ASSERT(exl3 != nullptr && model_ref->exl3_had128 != nullptr);
        x = ggml_mul(ctx0, x, exl3->suh);
        x = build_exl3_had128(ctx0, model_ref->exl3_had128, x);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx0, draft_vocab_compact, x); // [n_sel, n]
    if (head_s) {
        logits = ggml_mul(ctx0, logits, head_s);
    }
    return logits;
}

ggml_tensor * llm_graph_context::build_draft_vocab_logits(
        ggml_tensor * head_w,
        ggml_tensor * head_s,
        ggml_tensor * cur) const {
    if (draft_vocab_ids == nullptr || n_outputs == 0 || head_w == nullptr) {
        return nullptr;
    }
    // [#81] a compact head built from this head replaces the hadamard_head / head_layout outcomes
    const bool compact = draft_vocab_compact != nullptr && draft_vocab_compact_src == head_w;
    const bool full_head = !compact && llama_draft_vocab_full_head(head_w);
    if (!compact && hadamard_rotations && hadamard_rotations->count(head_w)) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HADAMARD, "the output head is Hadamard-rotated");
    }
    if (loras && !loras->empty()) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_LORA, "a LoRA adapter is loaded");
    }
    if (head_s && ggml_nelements(head_s) != 1) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HEAD_SCALE, "the output head has a per-row scale");
    }
    if (draft_vocab_ids->ne[0] >= head_w->ne[1]) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_NOT_SMALLER, "the map has as many rows as the head");
    }
    if (!compact && !full_head) {
        if (const char * why = draft_vocab_direct(head_w)) {
            return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HEAD_LAYOUT, why);
        }
    }
    // fork: Reserve graphs precede sampler setup; real compact output rows need a backend sampler.
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!ubatch.output[i]) {
            continue;
        }
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            if (!is_reserve && samplers.find(ubatch.seq_id[i][j]) == samplers.end()) {
                return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_NO_BACKEND_SAMPLER, "an output row has no backend sampler");
            }
        }
    }
    GGML_ASSERT(cur->ne[1] == n_outputs);

    if (full_head) {
        ggml_tensor * logits = llama_draft_vocab_gather_logits(
                ctx0, build_lora_mm(head_w, cur, head_s), draft_vocab_ids);
        cb(logits, "draft_vocab_logits", -1);
        res->t_logits_ids = draft_vocab_ids;
        return logits;
    }

    if (compact) {
        ggml_tensor * logits = build_draft_vocab_compact(head_w, head_s, cur); // [n_sel, n_outputs]
        cb(logits, "draft_vocab_logits", -1);
        res->t_logits_ids = draft_vocab_ids;
        return logits;
    }

    const int64_t n_sel = draft_vocab_ids->ne[0];

    // each vocabulary row is a one-row expert of the existing head: no second weight matrix
    ggml_tensor * rows = ggml_reshape_3d(ctx0, head_w, head_w->ne[0], 1, head_w->ne[1]);

    ggml_tensor * logits = nullptr;
    // [PARW] several drafting slots: one MUL_MAT_ID over all outputs with the id list broadcast (row stride 0), so
    // the CUDA indexed-rows kernel reads each shortlist row once instead of once per slot. LLAMA_DRAFT_VOCAB_BATCHED=0
    // keeps the per-output loop.
    static const bool batched = [] {
        const char * env = getenv("LLAMA_DRAFT_VOCAB_BATCHED");
        return env == nullptr || atoi(env) != 0;
    }();
    if (batched && n_outputs > 1 && n_outputs <= 8 && ggml_is_contiguous(cur)) {
        ggml_tensor * ids_b = ggml_view_2d(ctx0, draft_vocab_ids, n_sel, 1, draft_vocab_ids->nb[1], 0);
        ids_b->ne[1] = n_outputs; // broadcast view: every output reads the same id list
        ids_b->nb[1] = 0;
        ggml_tensor * x3 = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_outputs);
        logits = ggml_reshape_2d(ctx0, ggml_mul_mat_id(ctx0, rows, x3, ids_b), n_sel, n_outputs);
    }
    const int64_t n_loop = logits == nullptr ? n_outputs : 0;
    for (int64_t i = 0; i < n_loop; ++i) {
        ggml_tensor * h_i = n_outputs == 1 ? cur : ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], i*cur->nb[1]);
        ggml_tensor * l_i = ggml_mul_mat_id(ctx0, rows, h_i, draft_vocab_ids); // [1, n_sel, 1]
        l_i = ggml_reshape_2d(ctx0, l_i, n_sel, 1);
        logits = logits ? ggml_concat(ctx0, logits, l_i, 1) : l_i;
    }
    if (head_s) {
        logits = ggml_mul(ctx0, logits, head_s);
    }
    cb(logits, "draft_vocab_logits", -1);

    res->t_logits_ids = draft_vocab_ids;

    return logits;
}

ggml_tensor * llm_graph_context::build_draft_vocab_logits_chain(
        ggml_tensor * head_w,
        ggml_tensor * head_s,
        ggml_tensor * cur) const {
    if (draft_vocab_ids == nullptr || head_w == nullptr) {
        return nullptr;
    }
    const bool compact = draft_vocab_compact != nullptr && draft_vocab_compact_src == head_w;
    const bool full_head = !compact && llama_draft_vocab_full_head(head_w);
    if (!compact && hadamard_rotations && hadamard_rotations->count(head_w)) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HADAMARD, "the output head is Hadamard-rotated");
    }
    if (loras && !loras->empty()) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_LORA, "a LoRA adapter is loaded");
    }
    if (head_s && ggml_nelements(head_s) != 1) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HEAD_SCALE, "the output head has a per-row scale");
    }
    if (draft_vocab_ids->ne[0] >= head_w->ne[1]) {
        return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_NOT_SMALLER, "the map has as many rows as the head");
    }
    if (!compact && !full_head) {
        if (const char * why = draft_vocab_direct(head_w)) {
            return draft_vocab_fallback(DRAFT_VOCAB_FALLBACK_HEAD_LAYOUT, why);
        }
    }
    GGML_ASSERT(cur->ne[1] == 1);
    if (full_head) {
        return llama_draft_vocab_gather_logits(ctx0, build_lora_mm(head_w, cur, head_s), draft_vocab_ids);
    }
    if (compact) {
        return build_draft_vocab_compact(head_w, head_s, cur);
    }

    const int64_t n_sel = draft_vocab_ids->ne[0];

    ggml_tensor * rows   = ggml_reshape_3d(ctx0, head_w, head_w->ne[0], 1, head_w->ne[1]);
    ggml_tensor * logits = ggml_reshape_2d(ctx0, ggml_mul_mat_id(ctx0, rows, cur, draft_vocab_ids), n_sel, 1);
    if (head_s) {
        logits = ggml_mul(ctx0, logits, head_s);
    }
    return logits;
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
