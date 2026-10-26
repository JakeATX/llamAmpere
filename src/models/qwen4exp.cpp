#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-mtp-chain-sample.h"
#include "llama-ple-prefetch.h"
#include "llama-ple-cache.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <cmath>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
    hparams.indexer_kpool = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (hparams.indexer_kpool != 0 && r != hparams.indexer_kpool) {
            throw std::runtime_error(format("QSA layers must share one compress ratio, got %u and %u", hparams.indexer_kpool, r));
        }
        hparams.indexer_kpool = r;
    }
    if (hparams.indexer_kpool == 1 || (hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool != 0)) {
        throw std::runtime_error(format("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                        hparams.indexer_kpool, hparams.indexer_top_k));
    }
    // the reference groups the visible tokens in cache order and always keeps the tail
    hparams.indexer_kpool_row         = 2; // raw key | pooled key
    hparams.indexer_kpool_by_order    = true;
    hparams.indexer_kpool_select_tail = true;

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const auto nf = nextn_flags(ml, LLM_TENSOR_HC_ATTN_NORM);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    auto load_block = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block(il, trunk_flags);
    }

    // the MTP block: one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream
    for (int il = n_layer; il < n_layer_all; ++il) {
        load_block(il, mtp_flags);

        auto & nextn = layers[il].nextn;
        nextn.eh_proj      = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,      "weight", il), { 2*n_embd, n_embd }, mtp_flags);
        nextn.enorm        = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,        "weight", il), { n_embd }, mtp_flags);
        // RMS per hc stream of the trunk residual, so the gammas load as [n_embd, hc] like the mixer norms
        nextn.hnorm        = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,        "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, mtp_flags);
        nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo_mm = build_lora_mm(w_down, xn);
    ggml_tensor * lo = ggml_silu(ctx0, ggml_scale(ctx0, lo_mm, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    // [HCFUSED] LLAMA_HC_FUSED (default on 2026-10-22: verify -3.5..-5.4%/round on 3 trios, KLD inside ub4 noise; 0 = off): at 1..8 tokens run the whole mix as two ops, ggml_dsv4_hc_down (norm folded
    // into one split-K pass over W_down and W_inject, then silu and the 2*sigmoid scatter weights) and ggml_dsv4_hc_up
    // (W_up, sigmoid gate and the mean over the streams). Only plain BF16 matmuls on xn / lo qualify (no LoRA, Hadamard
    // fold, scale tensor or precision policy). The activations stay f32 where the unfused small-batch mul_mat_f rounds
    // them to BF16, so the output moves by rounding only. The unfused nodes above are left unreferenced.
    {
        static const bool hc_fused = [] { const char * e = getenv("LLAMA_HC_FUSED"); return !e || atoi(e) != 0; }();
        const auto plain_bf16 = [](const ggml_tensor * t, const ggml_tensor * w, const ggml_tensor * src) {
            if (t->op != GGML_OP_MUL_MAT || t->src[0] != w || t->src[1] != src || w->type != GGML_TYPE_BF16 ||
                    t->src[2] != nullptr || t->src[3] != nullptr) {
                return false;
            }
            for (int i = 0; i < 4; ++i) {
                if (t->op_params[i] != 0) {
                    return false;
                }
            }
            return true;
        };
        const int64_t R = w_down->ne[1];
        // [B1] the FFN phases copy the injection weights to b1_ffn->inject and combine them in another graph,
        // which expects the raw product
        bool ok = hc_fused && !hc_fused_blocked && nt >= 1 && nt <= 8 && hc == 4 && n_embd % 512 == 0 && R % 64 == 0 &&
            x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == n_embd && x->ne[1] == hc &&
            w_norm->type == GGML_TYPE_F32 && ggml_is_contiguous(w_norm) && ggml_nelements(w_norm) == hc_dim &&
            plain_bf16(lo_mm, w_down, xn) && plain_bf16(gate, w_up, lo);
        ggml_tensor * inj_mm = nullptr;
        if (ok && inject) {
            inj_mm = build_lora_mm(w_inject, xn);
            ok = plain_bf16(inj_mm, w_inject, xn);
        }
        if (ok) {
            ggml_tensor * x3 = ggml_reshape_3d(ctx0, x, n_embd, hc, nt);
            ggml_tensor * wn = ggml_reshape_2d(ctx0, w_norm, n_embd, hc);
            ggml_tensor * dn = ggml_dsv4_hc_down(ctx0, x3, wn, w_down, inject ? w_inject : nullptr,
                    hparams.f_norm_rms_eps, 1.0f / (float) hc, 1.0f / (float) hc);
            cb(dn, "hc_fused_down", il);
            if (inject) {
                // already 2*sigmoid(inject/hc): build_hc_combine sees the HC_DOWN view and skips its activation
                *inject = ggml_view_2d(ctx0, dn, hc, nt, hc*sizeof(float), R*nt*sizeof(float));
                cb(*inject, "hc_inject", il);
            }
            ggml_tensor * mixed = ggml_dsv4_hc_up(ctx0, x3, wn, dn, w_up, 1.0f / (float) hc);
            cb(mixed, "hc_mixed", il);
            return mixed;
        }
    }

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);

        // LLAMA_HC_BF16_ONCE=1 (opt-in): the down and inject weights are BF16, so the CUDA cuBLAS path rounds xn to
        // BF16 once per matmul; cast it once and feed both, the same rounding (round to nearest even) and the same GEMM.
        // Only plain default-precision matmuls on xn qualify (no LoRA, Hadamard fold or precision policy), and only at
        // 32+ tokens: small batches take the vector / mmf kernels, which read xn in f32 and must keep doing so.
        static const bool bf16_once = [] { const char * e = getenv("LLAMA_HC_BF16_ONCE"); return e && atoi(e) != 0; }();
        const auto plain_bf16_mm = [&](const ggml_tensor * t, const ggml_tensor * w) {
            if (t->op != GGML_OP_MUL_MAT || t->src[0] != w || t->src[1] != xn || w->type != GGML_TYPE_BF16) {
                return false;
            }
            for (int i = 0; i < 4; ++i) {
                if (t->op_params[i] != 0) {
                    return false;
                }
            }
            return true;
        };
        if (bf16_once && nt >= 32 && plain_bf16_mm(lo_mm, w_down) && plain_bf16_mm(*inject, w_inject)) {
            ggml_tensor * xb = ggml_cast(ctx0, xn, GGML_TYPE_BF16);
            cb(xb, "hc_norm_bf16", il);
            lo_mm->src[1]     = xb;
            (*inject)->src[1] = xb;
        }
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    // ([HCFUSED] the ggml_dsv4_hc_down injection segment already holds the activated weights; it may reach here
    // through the last layer's crop to the output rows, so look through row gathers and views, not only the view)
    const auto activated = [](const ggml_tensor * t) {
        while (t) {
            if (t->op == GGML_OP_DSV4_HC_DOWN) {
                return true;
            }
            if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_GET_ROWS && t->op != GGML_OP_CONT) {
                return false;
            }
            t = t->src[0];
        }
        return false;
    };
    ggml_tensor * w = inject;
    if (!activated(inject)) {
        w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx0, w, 2.0f);
    }

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

class llm_graph_input_b1_rows : public llm_graph_input_i {
public:
    ggml_tensor * rows;
    ggml_tensor * weights;
    const std::vector<int32_t> * row_data;
    const std::vector<float> * weight_data;
    size_t offset;
    size_t count;

    void set_input(const llama_ubatch *) override {
        ggml_backend_tensor_set(rows, row_data->data() + offset, 0, count * sizeof(int32_t));
        ggml_backend_tensor_set(weights, weight_data->data() + offset, 0, count * sizeof(float));
    }
};

class llm_graph_input_b1_bucket : public llm_graph_input_i {
public:
    ggml_tensor * rows;
    ggml_tensor * experts;
    ggml_tensor * weights;
    const llm_b1_ffn_bucket * bucket;
    std::vector<int32_t> local_experts;

    void set_input(const llama_ubatch *) override {
        const size_t count = bucket->rows.size();
        ggml_backend_tensor_set(rows, bucket->rows.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set(experts, local_experts.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set(weights, bucket->weights.data(), 0, count * sizeof(float));
    }
};

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const auto * b1_ffn = params.b1_ffn;
    const int b1_ffn_phase = params.b1_ffn_phase;
    hc_fused_blocked = b1_ffn != nullptr && b1_ffn_phase != 0;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    if (b1_ffn_phase == 2 && b1_ffn) {
        GGML_ASSERT(b1_il1 == b1_il0 + 1 && loras->empty());
        const auto & layer = model.layers[b1_il0];
        ggml_tensor * accumulator = b1_ffn->output;
        if (b1_ffn->grouped_mmid) {
            const auto & bucket = b1_ffn->buckets.at(b1_ffn->bucket_index);
            const int64_t count = (int64_t) bucket.rows.size();
            GGML_ASSERT(count > 0 && bucket.experts.size() == (size_t) count);
            // Buckets are expert-major. Restrict the bank to the experts
            // actually present so indexed MMQ sizes its launch from their
            // real row count rather than averaging across the entire model.
            static const bool interval_views = !std::getenv("LLAMA_B1_FFN_INTERVAL") ||
                    std::atoi(std::getenv("LLAMA_B1_FFN_INTERVAL")) != 0;
            const int64_t first_expert = interval_views ? bucket.experts.front() : 0;
            const int64_t expert_count = interval_views ? bucket.experts.back() - first_expert + 1 : hparams.n_expert;
            GGML_ASSERT(first_expert >= 0 && expert_count > 0 && first_expert + expert_count <= hparams.n_expert);
            const auto bank_view = [&](ggml_tensor * w) {
                GGML_ASSERT(w && first_expert + expert_count <= w->ne[2]);
                auto * v = ggml_view_3d(ctx0, w, w->ne[0], w->ne[1], expert_count,
                        w->nb[1], w->nb[2], first_expert * w->nb[2]);
                ggml_set_name(v, ggml_get_name(w));
                return v;
            };
            const auto scale_view = [&](ggml_tensor * scale) {
                return scale ? ggml_view_1d(ctx0, scale, expert_count, first_expert * scale->nb[0]) : nullptr;
            };
            auto input = std::make_unique<llm_graph_input_b1_bucket>();
            input->bucket = &bucket;
            input->local_experts.reserve(count);
            for (int32_t e : bucket.experts) {
                GGML_ASSERT(e >= first_expert && e < first_expert + expert_count);
                input->local_experts.push_back((int32_t) (e - first_expert));
            }
            input->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, count);
            input->experts = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, count);
            input->weights = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, count);
            ggml_set_input(input->rows);
            ggml_set_input(input->experts);
            ggml_set_input(input->weights);
            ggml_tensor * row_ids = input->rows;
            ggml_tensor * expert_ids = input->experts;
            ggml_tensor * row_weights = input->weights;
            res->add_input(std::move(input));
            ggml_tensor * x = ggml_get_rows(ctx0, b1_ffn->input, row_ids);
            x = ggml_reshape_3d(ctx0, x, n_embd, 1, count);
            ggml_tensor * gate;
            ggml_tensor * up;
            if (layer.ffn_gate_up_exps) {
                ggml_tensor * both = build_lora_mm_id(bank_view(layer.ffn_gate_up_exps), x, expert_ids, scale_view(layer.ffn_up_exps_s));
                const int64_t hidden = both->ne[0] / 2;
                gate = ggml_view_3d(ctx0, both, hidden, 1, count, both->nb[1], both->nb[2], 0);
                up = ggml_view_3d(ctx0, both, hidden, 1, count, both->nb[1], both->nb[2], hidden * both->nb[0]);
            } else {
                up = build_lora_mm_id(bank_view(layer.ffn_up_exps), x, expert_ids, scale_view(layer.ffn_up_exps_s));
                gate = build_lora_mm_id(bank_view(layer.ffn_gate_exps), x, expert_ids, scale_view(layer.ffn_gate_exps_s));
            }
            const float limit = hparams.swiglu_clamp_exp[b1_il0];
            ggml_tensor * activated;
            if (layer.ffn_gate_exps && limit > 1e-6f) {
                up = ggml_clamp(ctx0, up, -limit, limit);
                gate = ggml_clamp(ctx0, ggml_silu(ctx0, gate), -INFINITY, limit);
                activated = ggml_mul(ctx0, gate, up);
            } else {
                activated = ggml_swiglu_split(ctx0, gate, up);
            }
            ggml_tensor * down = build_lora_mm_id(bank_view(layer.ffn_down_exps), activated, expert_ids, scale_view(layer.ffn_down_exps_s));
            down = ggml_reshape_2d(ctx0, down, n_embd, count);
            down = ggml_mul(ctx0, down, row_weights);
            // All expert GEMMs share the bucket, while each scatter span keeps
            // unique rows. Chaining the accumulator orders overlapping spans.
            for (const auto & span : bucket.spans) {
                auto * ids = ggml_view_1d(ctx0, row_ids, span.count, span.offset * sizeof(int32_t));
                auto * values = ggml_view_2d(ctx0, down, n_embd, span.count, down->nb[1], span.offset * down->nb[1]);
                auto * previous = ggml_get_rows(ctx0, accumulator, ids);
                accumulator = ggml_set_rows(ctx0, accumulator, ggml_add(ctx0, previous, values), ids);
                ggml_build_forward_expand(gf, accumulator);
            }
            return;
        }
        const auto expert_view = [&](ggml_tensor * w, size_t e) {
            auto * v = ggml_view_2d(ctx0, w, w->ne[0], w->ne[1], w->nb[1], e * w->nb[2]);
            ggml_set_name(v, ggml_get_name(w));
            return v;
        };
        const auto expert_scale = [&](ggml_tensor * scale, size_t e) {
            return scale ? ggml_view_1d(ctx0, scale, 1, e * scale->nb[0]) : nullptr;
        };
        size_t piece = 0;
        for (size_t e = 0; e < b1_ffn->expert_rows.size(); ++e) {
            const auto & rows = b1_ffn->expert_rows[e];
            const auto & weights = b1_ffn->expert_weights[e];
            for (size_t offset = 0; offset < rows.size(); offset += b1_ffn->max_rows, ++piece) {
                if (piece < b1_ffn->piece_first || piece - b1_ffn->piece_first >= b1_ffn->piece_count) {
                    continue;
                }
                const size_t count = std::min<size_t>(b1_ffn->max_rows, rows.size() - offset);
                auto input = std::make_unique<llm_graph_input_b1_rows>();
                input->row_data = &rows;
                input->weight_data = &weights;
                input->offset = offset;
                input->count = count;
                input->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, count);
                input->weights = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, count);
                ggml_set_input(input->rows);
                ggml_set_input(input->weights);
                ggml_tensor * row_ids = input->rows;
                ggml_tensor * row_weights = input->weights;
                res->add_input(std::move(input));
                ggml_tensor * x = ggml_get_rows(ctx0, b1_ffn->input, row_ids);
                ggml_tensor * gate;
                ggml_tensor * up;
                if (layer.ffn_gate_up_exps) {
                    ggml_tensor * gate_up = build_lora_mm(expert_view(layer.ffn_gate_up_exps, e), x, expert_scale(layer.ffn_up_exps_s, e));
                    const int64_t hidden = gate_up->ne[0] / 2;
                    gate = ggml_view_2d(ctx0, gate_up, hidden, count, gate_up->nb[1], 0);
                    up = ggml_view_2d(ctx0, gate_up, hidden, count, gate_up->nb[1], hidden * gate_up->nb[0]);
                } else {
                    up = build_lora_mm(expert_view(layer.ffn_up_exps, e), x, expert_scale(layer.ffn_up_exps_s, e));
                    gate = build_lora_mm(expert_view(layer.ffn_gate_exps, e), x, expert_scale(layer.ffn_gate_exps_s, e));
                }
                const float limit = hparams.swiglu_clamp_exp[b1_il0];
                ggml_tensor * activated;
                if (layer.ffn_gate_exps && limit > 1e-6f) {
                    up = ggml_clamp(ctx0, up, -limit, limit);
                    gate = ggml_clamp(ctx0, ggml_silu(ctx0, gate), -INFINITY, limit);
                    activated = ggml_mul(ctx0, gate, up);
                } else {
                    activated = ggml_swiglu_split(ctx0, gate, up);
                }
                ggml_tensor * down = build_lora_mm(expert_view(layer.ffn_down_exps, e), activated, expert_scale(layer.ffn_down_exps_s, e));
                down = ggml_mul(ctx0, down, row_weights);
                // Expert-major order changes only the order of the exact weighted additions.
                ggml_tensor * previous = ggml_get_rows(ctx0, accumulator, row_ids);
                accumulator = ggml_set_rows(ctx0, accumulator, ggml_add(ctx0, previous, down), row_ids);
                ggml_build_forward_expand(gf, accumulator);
            }
        }
        return;
    }
    if (b1_ffn_phase == 4 && b1_ffn) {
        res->t_embd = build_layer_ffn(b1_ffn->input, b1_il0);
        ggml_build_forward_expand(gf, res->t_embd);
        return;
    }
    if (b1_ffn_phase == 3 && b1_ffn) {
        GGML_ASSERT(b1_il1 == b1_il0 + 1);
        ggml_tensor * combined = build_hc_combine(b1_carry, b1_ffn->output, b1_ffn->inject, b1_il0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, combined, b1_carry));
        return;
    }

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // [B1] layer range of this graph: the whole model unless the layer-major prefill builds one slice of it
    const int  b1_l0    = b1_il0;
    const int  b1_l1    = b1_il1 < 0 ? (int) n_layer : b1_il1;
    const bool b1_full  = b1_l0 == 0 && b1_l1 == (int) n_layer;
    const bool b1_first = b1_l0 == 0;
    const bool b1_last  = b1_l1 == (int) n_layer;
    GGML_ASSERT(0 <= b1_l0 && b1_l0 < b1_l1 && b1_l1 <= (int) n_layer);
    GGML_ASSERT((b1_first && b1_last) || b1_carry != nullptr);

    bool b1_has_qsa = b1_full;
    bool b1_has_ple = b1_full;
    for (int il = b1_l0; il < b1_l1; ++il) {
        b1_has_qsa = b1_has_qsa || !hparams.is_recr(il);
        b1_has_ple = b1_has_ple || hparams.is_ple(il);
    }

    ggml_tensor * inpL = nullptr;
    if (b1_first) {
        inpL = build_inp_embd(model.tok_embd);
        cb(inpL, "model.input_embed", -1);
        ggml_build_forward_expand(gf, inpL);
    }

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    // the QSA layers share one set of k-pool inputs
    // the CUDA lightning indexer takes 32 or 64 heads, QSA has a few, so it scores with plain ops
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_idx && hparams.indexer_kpool > 0 && b1_has_qsa) {
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = b1_last ? build_inp_out_ids() : nullptr;

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0 && b1_has_ple) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    ggml_tensor * res_hc = nullptr;
    if (b1_first) {
        // the wide residual starts as hc identical copies of the embedding
        res_hc = ggml_repeat_4d(ctx0,
                ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                n_embd, hc, n_tokens, 1);
        cb(res_hc, "hc_init", -1);
        // make sure hc_init is in the same graph split as the first layer (-sm tensor)
        ggml_build_forward_expand(gf, res_hc);
    } else {
        // [B1] the hidden state the previous layer range left for this ubatch
        GGML_ASSERT(b1_carry->type == GGML_TYPE_F32 && b1_carry->ne[0] == n_embd && b1_carry->ne[1] == hc &&
                    b1_carry->ne[2] == n_tokens);
        res_hc = b1_carry;
    }

    for (int il = b1_l0; il < b1_l1; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        if (cparams.moe_prefetch != 0 && b1_full && n_tokens > 0 && n_tokens <= 8 &&
                cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
            cb(cur, "moe_prefetch_layer_start", il);
        }
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        // Predictions use this layer's post-attention FFN input and never feed model math.
        // fork PREFETCHFIX: LLAMA_MOE_PREFETCH_AFTER_FFN=1 schedules them after this layer's experts,
        // so their copies never compete with this layer's own miss reads.
        static const bool prefetch_after_ffn = [] {
            const char * v = getenv("LLAMA_MOE_PREFETCH_AFTER_FFN");
            return v && strcmp(v, "1") == 0;
        }();
        std::vector<ggml_tensor *> prefetch_pending;
        if (cparams.moe_prefetch != 0 && b1_full && n_tokens > 0 && n_tokens <= 8 &&
                cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
                strncmp(ggml_backend_dev_name(model.dev_layer(il)), "CUDA", 4) == 0) {
            // fork PREFETCHFIX: the asynchronous path predicts one distance only (--moe-prefetch N, -1 = 1)
            // with LLAMA_MOE_PREFETCH_K candidates (default 32, at most 64).
            static const bool prefetch_async = [] {
                const char * v = getenv("GGML_MOE_PREFETCH_ASYNC");
                return v && (strcmp(v, "1") == 0 || strcmp(v, "2") == 0);
            }();
            static const int64_t prefetch_k = [] {
                const char * v = getenv("LLAMA_MOE_PREFETCH_K");
                const int k = v ? atoi(v) : 32;
                return (int64_t) std::max(1, std::min(k, 64));
            }();
            const int lead = cparams.moe_prefetch < 0 ? (prefetch_async ? 1 : 3) : cparams.moe_prefetch;
            for (int d = prefetch_async ? lead : 1; d <= lead && il + d < n_layer; ++d) {
                const auto & future = model.layers[il + d];
                if (!future.ffn_gate_inp) {
                    continue;
                }
                bool supported = true;
                auto gpu = [&](ggml_tensor * t) {
                    supported &= ggml_backend_dev_supports_op(model.dev_layer(il), t);
                    cb(t, "moe_prefetch_gpu", il);
                    return t;
                };
                // Qwen4Exp routes with softmax and has no routing bias.
                ggml_tensor * scores = gpu(ggml_soft_max(ctx0, gpu(build_lora_mm(future.ffn_gate_inp, cur))));
                if (scores->ne[1] > 1) {
                    const int64_t tokens = scores->ne[1];
                    scores = gpu(ggml_cont(ctx0, ggml_transpose(ctx0, scores)));
                    scores = gpu(ggml_pool_2d(ctx0, scores, GGML_OP_POOL_MAX, tokens, 1, tokens, 1, 0, 0));
                    scores = ggml_reshape_1d(ctx0, scores, n_expert);
                }
                ggml_tensor * ids = gpu(ggml_cont(ctx0, gpu(ggml_argsort_top_k(ctx0, scores, std::min<int64_t>(prefetch_async ? prefetch_k : 32, n_expert)))));
                // An unsupported predictor must not fall back to CPU execution.
                if (!supported) {
                    continue;
                }
                ggml_format_name(ids, "moe_prefetch-%d-%d", il, il + d);
                ggml_set_output(ids);
                if (prefetch_after_ffn) {
                    prefetch_pending.push_back(ids);
                } else {
                    ggml_build_forward_expand(gf, ids);
                }
            }
        }

        if (b1_ffn_phase == 1 && b1_ffn) {
            GGML_ASSERT(b1_l1 == b1_l0 + 1 && il == b1_l0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, res_hc, b1_carry));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, b1_ffn->input));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, inject, b1_ffn->inject));
            // The shared expert initializes the accumulator while the routed experts wait for all prompt rows.
            cur = build_layer_ffn(cur, il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, b1_ffn->output));
            return;
        }

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);
        if (!prefetch_pending.empty()) {
            ggml_build_forward_expand(gf, cur);
            for (ggml_tensor * ids : prefetch_pending) {
                ggml_build_forward_expand(gf, ids);
            }
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    if (!b1_last) {
        // [B1] hand the hidden state to the next layer range; this graph has no outputs
        GGML_ASSERT(res_hc->type == b1_carry->type && b1_carry->ne[0] == res_hc->ne[0] && b1_carry->ne[1] == res_hc->ne[1] &&
                    b1_carry->ne[2] == res_hc->ne[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, res_hc, b1_carry));
        return;
    }

    // the MTP head reads the hc-wide residual, before the final mixer
    res->t_h_nextn = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
    cb(res->t_h_nextn, "h_nextn", -1);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    if (crop_after_nextn(inp_out_ids)) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order, see llama_memory_hybrid_idx
class llama_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (source) {
            const auto * src_idx = (const int32_t *) source->new_pool_idxs->data;
            const auto * src_rep = (const int64_t *) source->new_pool_rep->data;
            const auto * src_pos = (const int32_t *) source->new_pool_pos->data;
            auto * dst_idx = (int32_t *) new_pool_idxs->data;
            auto * dst_rep = (int64_t *) new_pool_rep->data;
            auto * dst_pos = (int32_t *) new_pool_pos->data;
            const int64_t dummy = ((const int64_t *) source->k_idxs->data)[row0];
            for (uint32_t i = 0; i < n_new; ++i) {
                const bool ready = source->new_ready[i] < row0 + width;
                for (uint32_t k = 0; k < kpool; ++k) {
                    dst_idx[(size_t) i*kpool + k] = ready ? src_idx[(size_t) i*kpool + k] : (int32_t) dummy;
                }
                dst_rep[i] = ready ? src_rep[i] : source->new_dummy[i];
                for (uint32_t d = 0; d < 4; ++d) {
                    dst_pos[(size_t) d*n_new + i] = ready ? src_pos[(size_t) d*n_new + i] : 0;
                }
            }
            return;
        }
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        if (store_only) {
            GGML_ASSERT(!pool_cells && !pool_idxs && !pool_mask && !tail_idxs && storage_cells == 0);
            mctx->set_input_kpool_store(new_pool_idxs, new_pool_rep, ubatch, new_pool_pos);
            return;
        }
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, new_pool_idxs, new_pool_rep,
                              ubatch, new_pool_pos);
        if (storage_cells > 0) {
            // A future chain row has no raw key yet. Pool it only after its last member is written.
            const auto * cells = (const int32_t *) pool_cells->data;
            const auto * reps = (const int64_t *) new_pool_rep->data;
            rep_pool.assign(storage_cells, -1);
            for (int64_t ip = 0; ip < pool_cells->ne[0]; ++ip) {
                GGML_ASSERT(cells[ip] >= 0 && cells[ip] < storage_cells);
                const auto * members = (const int32_t *) pool_idxs->data + ip*kpool;
                if (members[0] != (int32_t) n_kv) {
                    rep_pool[cells[ip]] = (int32_t) ip;
                } else if (rep_pool[cells[ip]] == -1) {
                    rep_pool[cells[ip]] = -2;
                }
            }
            new_ready.assign(n_new, ubatch->n_tokens);
            new_dummy.resize(n_new);
            int64_t dummy = 0;
            for (uint32_t i = 0; i < n_new; ++i) {
                GGML_ASSERT(reps[i] >= 0 && reps[i] < storage_cells);
                const int32_t ip = rep_pool[reps[i]];
                for (uint32_t row = 0; ip >= 0 && row < ubatch->n_tokens; ++row) {
                    bool visible;
                    if (pool_mask->type == GGML_TYPE_I32) {
                        const auto * range = (const int32_t *) pool_mask->data + 2*row;
                        visible = ip >= range[0] && ip < range[1];
                    } else {
                        GGML_ASSERT(pool_mask->type == GGML_TYPE_F16);
                        const auto * mask = (const ggml_fp16_t *) ((const char *) pool_mask->data + row*pool_mask->nb[1]);
                        visible = std::isfinite(ggml_fp16_to_fp32(mask[ip]));
                    }
                    if (visible) {
                        new_ready[i] = row;
                        break;
                    }
                }
                // Dummy destinations are distinct and never hold a real pooled key.
                while (dummy < storage_cells && rep_pool[dummy] != -1) {
                    ++dummy;
                }
                GGML_ASSERT(dummy < storage_cells);
                new_dummy[i] = dummy++;
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        if (source) {
            return row0 + width <= params.ubatch.n_tokens && n_new == mctx->get_n_kpool_new();
        }
        if (store_only) {
            // fork: the KV-only graph addresses fixed cache storage, not history-sized pool maps or a n_kv sentinel.
            return !params.cparams.mtp_chain && params.n_outputs == 0 &&
                k_idxs->ne[0] == params.ubatch.n_tokens && n_new == mctx->get_n_kpool_new();
        }
        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // the scatter mask shape follows n_kv
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == mctx->get_n_kpool_new();

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]         cell caching each block's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for the padded blocks
    ggml_tensor * pool_mask     = nullptr; // F16 [n_pool, n_tokens], or I32 [2, n_tokens] = [lo, hi) of the visible blocks
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]          cell to write each new pooled key into
    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0; // padded to a stable bound, never below 1
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
    bool store_only = false; // fork: genuine KV-only graph; chain roots still need the full pool metadata
    llm_graph_input_kpool * source = nullptr;
    uint32_t row0 = 0;
    uint32_t width = 0;
    int64_t storage_cells = 0;
    std::vector<int32_t> rep_pool;
    std::vector<uint32_t> new_ready;
    std::vector<int64_t> new_dummy;
};

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP has a single block");
    GGML_ASSERT(ubatch.token && "qwen4exp MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "qwen4exp MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    const char * full_head_env = std::getenv("LLAMA_MTP_CHAIN_FULL_HEAD");
    const bool full_head_mode = full_head_env && std::strcmp(full_head_env, "1") == 0;

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.hc_head_norm &&
            "MTP block missing, load the model with MTP enabled");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = nullptr;
    if (mtp_tok_row) {
        // [RTPORT] LLAMA_MTP_PIPELINE: the context writes this step's token row on the stream (same f32 values the
        // host get_rows produces), so the step can be queued before its token is known
        GGML_ASSERT(n_tokens == 1);
        inp->tok_row = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.tok_embd->ne[0], 1);
        ggml_set_input(inp->tok_row);
        ggml_set_name(inp->tok_row, "mtp_tok_row");
        tok_embd = inp->tok_row;
    } else {
        tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
        cb(tok_embd, "mtp_tok_embd", il);
    }

    ggml_tensor * h = inp->h;

    res->add_input(std::move(inp));

    auto * inp_hyb = build_inp_mem_hybrid();
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);

    // the draft memory has no recurrent layer, but its input still has to be allocated
    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    // A catch-up call supplies the TARGET's previous hidden row for every
    // token. With one MTP block, only that block's attention/indexer K/V
    // survive into the next call. Its queries, attention output and FFN are
    // not inputs to later prompt positions. An unmasked hidden-state reader
    // still needs those outputs and therefore keeps the complete graph.
    static const bool update_kv_only = [] {
        const char * value = std::getenv("LLAMA_MTP_UPDATE_KV_ONLY");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    const bool kv_only = update_kv_only && hparams.n_layer_nextn == 1 &&
        n_outputs == 0 && !cparams.embeddings && !cparams.mtp_chain && !cparams.training &&
        (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    if (kv_only) {
        static std::atomic_flag noted = ATOMIC_FLAG_INIT;
        if (!noted.test_and_set()) {
            LLAMA_LOG_INFO("%s: MTP catch-up stores exact K/V only (LLAMA_MTP_UPDATE_KV_ONLY=0 restores full graph)\n", __func__);
        }
    }

    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0) {
        GGML_ASSERT(mctx_hyb->get_idx()->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
        inp_kpool = build_inp_kpool(mctx_hyb, kv_only || cparams.mtp_chain);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = kv_only ? nullptr : build_inp_out_ids();

    if (cparams.mtp_chain) {
        const char * enabled = std::getenv("LLAMA_MTP_QWEN4EXP_CHAIN");
        GGML_ASSERT(enabled && std::strcmp(enabled, "1") == 0);
        GGML_ASSERT(ubatch.n_seqs_unq == 1 && mctx_hyb->get_n_stream() == 1 && !cparams.training && !cparams.embeddings);
        GGML_ASSERT(!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
        int64_t n_catchup = 0;
        while (n_catchup < n_tokens && ubatch.output && !ubatch.output[n_catchup]) {
            ++n_catchup;
        }
        const int64_t n_chain = n_tokens - n_catchup;
        GGML_ASSERT(n_chain >= 1 && n_chain <= LLAMA_MTP_QWEN4EXP_CHAIN_MAX && n_chain == n_outputs);
        for (int64_t row = n_catchup; row < n_tokens; ++row) {
            GGML_ASSERT(!ubatch.output || ubatch.output[row]);
        }

        auto build_proj = [&](ggml_tensor * token, ggml_tensor * hidden, int64_t width) {
            ggml_tensor * hn = build_norm(ggml_reshape_3d(ctx0, hidden, n_embd, hc, width),
                    layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * en = build_norm(token, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
            en = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, en, n_embd, 1, width), n_embd, hc, width, 1);
            return build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, en, hn, 0));
        };
        auto build_pos = [&](int64_t row0, int64_t width) {
            ggml_tensor * pos = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_pos, width, 4,
                    n_tokens*inp_pos->nb[0], row0*inp_pos->nb[0]));
            return ggml_reshape_1d(ctx0, pos, 4*width);
        };
        auto build_block = [&](ggml_tensor * projected, int64_t row0, int64_t width, bool front_only) -> ggml_tensor * {
            llm_graph_input_attn_kv attn = *inp_hyb->get_attn();
            attn.self_k_idxs = ggml_view_1d(ctx0, attn.self_k_idxs, width, row0*attn.self_k_idxs->nb[0]);
            const int64_t v_stride = attn.self_v_idxs->ne[0]/n_tokens;
            attn.self_v_idxs = ggml_view_1d(ctx0, attn.self_v_idxs, width*v_stride, row0*v_stride*attn.self_v_idxs->nb[0]);
            ggml_tensor * mask = attn.get_kq_mask();
            GGML_ASSERT(mask->ne[2] == 1 && mask->ne[3] == 1);
            attn.self_kq_mask_cnv = ggml_view_4d(ctx0, mask, mask->ne[0], width, 1, 1,
                    mask->nb[1], mask->nb[2], mask->nb[3], row0*mask->nb[1]);
            ggml_tensor * pos = build_pos(row0, width);
            llm_graph_input_kpool * pool = inp_kpool ? build_inp_kpool_slice(inp_kpool, row0, width) : nullptr;
            ggml_tensor * inject = nullptr;
            ggml_tensor * cur = build_hc_mix(projected, layer.hc_attn_norm, layer.hc_attn_down,
                    layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
            if (front_only) {
                if (pool && hparams.dsv4_compress_ratios[il] > 0) {
                    build_qsa_store(mctx_hyb, pool, cur, sections, il);
                }
                ggml_tensor * k = build_lora_mm(layer.wk, cur, layer.wk_s);
                ggml_tensor * v = build_lora_mm(layer.wv, cur, layer.wv_s);
                k = build_norm(ggml_reshape_3d(ctx0, k, hparams.n_embd_head_k(), n_head_kv, width),
                        layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
                k = ggml_rope_multi(ctx0, k, pos, nullptr,
                        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                v = ggml_reshape_3d(ctx0, v, hparams.n_embd_head_v(), n_head_kv, width);
                build_attn_store(&attn, k, v, il);
                return nullptr;
            }
            cur = build_layer_attn(&attn, mctx_hyb, pool, cur, pos, sections, il);
            ggml_tensor * residual = build_hc_combine(projected, cur, inject, il);
            cur = build_hc_mix(residual, layer.hc_ffn_norm, layer.hc_ffn_down,
                    layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
            cur = build_layer_ffn(cur, il);
            return build_hc_combine(residual, cur, inject, il);
        };

        // Keep the host input leaves whole across scheduler splits, as in the Qwen3.5 chain.
        ggml_tensor * projected = build_proj(tok_embd, h, n_tokens);
        if (n_catchup > 0) {
            ggml_tensor * front = ggml_view_3d(ctx0, projected, n_embd, hc, n_catchup,
                    projected->nb[1], projected->nb[2], 0);
            build_block(front, 0, n_catchup, true);
        }
        ggml_tensor * projected_cur = ggml_view_3d(ctx0, projected, n_embd, hc, 1,
                projected->nb[1], projected->nb[2], n_catchup*projected->nb[2]);
        const int32_t chain_k = cparams.mtp_chain_top_k;
        // FULL_HEAD always emits the full-vocabulary q, even when a mapped rollout cannot be built.
        // Only the bounded sampled rollout uses the map, solely to guess its next conditioning token.
        bool batch_full_head = full_head_mode && chain_k > 0 && chain_k <= LLAMA_MTP_CHAIN_TOP_K_MAX &&
                hparams.n_layer_nextn == 1 && n_chain <= LLAMA_MTP_QWEN4EXP_CHAIN_MAX &&
                model.output && model.output->ne[0] == n_embd && model.output->ne[2] == 1 &&
                model.output->ne[3] == 1 && ggml_is_contiguous(model.output) &&
                draft_vocab_ids && draft_vocab_ids->type == GGML_TYPE_I32 &&
                draft_vocab_ids->ne[0] >= chain_k && draft_vocab_ids->ne[0] < model.output->ne[1] &&
                draft_vocab_ids->ne[1] == 1 && draft_vocab_ids->ne[2] == 1 && draft_vocab_ids->ne[3] == 1 &&
                ggml_is_contiguous(draft_vocab_ids) &&
                (!model.output_s || ggml_nelements(model.output_s) == 1) && (!loras || loras->empty()) &&
                (!hadamard_rotations || !hadamard_rotations->count(model.output));
        std::vector<int32_t> test_ids;
        if (batch_full_head && !is_reserve) {
            // Test-only fault injection. Fresh contexts are required when changing this environment value:
            // it is consumed while building the graph and is not part of the graph-reuse key.
            const char * next = std::getenv("LLAMA_MTP_CHAIN_TEST_IDS");
            if (next && *next) {
                for (;;) {
                    char * end = nullptr;
                    errno = 0;
                    const long long token = std::strtoll(next, &end, 10);
                    GGML_ASSERT(errno != ERANGE && end != next && token >= 0 && token < model.output->ne[1]);
                    GGML_ASSERT(*end == '\0' || *end == ',');
                    test_ids.push_back((int32_t) token);
                    GGML_ASSERT(test_ids.size() <= (size_t) n_chain);
                    if (*end == '\0') {
                        break;
                    }
                    next = end + 1;
                }
                GGML_ASSERT(test_ids.size() == (size_t) (n_chain - 1) || test_ids.size() == (size_t) n_chain);
            }
        }
        ggml_tensor * chain_samp = nullptr;
        ggml_tensor * chain_u = nullptr;
        if (chain_k > 0) {
            auto sample = std::make_unique<llm_graph_input_mtp_chain_samp>(mtp_chain_samp);
            sample->samp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, LLAMA_MTP_CHAIN_SAMP_N);
            sample->u = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_chain);
            ggml_set_input(sample->samp);
            ggml_set_input(sample->u);
            chain_samp = sample->samp;
            chain_u = sample->u;
            res->add_input(std::move(sample));
        }
        ggml_tensor * logits_all = nullptr;
        ggml_tensor * hidden_all = nullptr;
        ggml_tensor * head_all = nullptr;
        std::vector<ggml_tensor *> conditioned_ids;
        if (batch_full_head) {
            conditioned_ids.reserve(n_chain - 1);
        }
        for (int64_t j = 0; j < n_chain; ++j) {
            ggml_tensor * residual = build_block(projected_cur, n_catchup + j, 1, false);
            ggml_tensor * hidden = ggml_reshape_2d(ctx0, residual, n_embd*hc, 1);
            hidden_all = hidden_all ? ggml_concat(ctx0, hidden_all, hidden, 1) : hidden;
            ggml_tensor * head = build_hc_mix(residual, layer.nextn.hc_head_norm, layer.nextn.hc_head_down,
                    layer.nextn.hc_head_up, nullptr, nullptr, il);
            ggml_tensor * id = nullptr;
            ggml_tensor * provisional_logits = nullptr;
            if (batch_full_head && j + 1 < n_chain) {
                provisional_logits = build_draft_vocab_logits_chain(model.output, model.output_s, head);
                if (!provisional_logits) {
                    // Layout/backend eligibility is checked by the existing map helper. It is invariant across
                    // these columns, so only the first step may fall back, before any provisional conditioning.
                    GGML_ASSERT(j == 0);
                    batch_full_head = false;
                }
            }
            if (batch_full_head) {
                // Save the already computed post-HC head input without batching any of the autoregressive block.
                head_all = head_all ? ggml_concat(ctx0, head_all, head, 1) : head;
                if (j + 1 < n_chain) {
                    GGML_ASSERT(provisional_logits->ne[0] == draft_vocab_ids->ne[0]);
                    ggml_tensor * u = ggml_view_1d(ctx0, chain_u, 1, j*chain_u->nb[0]);
                    (void) llama_mtp_chain_sample_graph(ctx0, provisional_logits, chain_k, chain_samp, u, draft_vocab_ids, &id, true);
                    if (!test_ids.empty()) {
                        // Keep the real mapped projection in the test graph while replacing the exact ID used
                        // by both the next embedding lookup and the emitted conditioning-history field.
                        id = ggml_cast(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, id, GGML_TYPE_F32),
                                0.0f, (float) test_ids[j]), GGML_TYPE_I32);
                    }
                    conditioned_ids.push_back(id);
                }
            } else {
                ggml_tensor * logits = full_head_mode ? nullptr :
                        build_draft_vocab_logits_chain(model.output, model.output_s, head);
                ggml_tensor * id_map = logits ? draft_vocab_ids : nullptr;
                if (!logits) {
                    logits = build_lora_mm(model.output, head, model.output_s);
                }
                ggml_tensor * output = nullptr;
                if (chain_k > 0) {
                    ggml_tensor * u = ggml_view_1d(ctx0, chain_u, 1, j*chain_u->nb[0]);
                    output = llama_mtp_chain_sample_graph(ctx0, logits, chain_k, chain_samp, u, id_map, &id, true);
                } else {
                    ggml_tensor * pick = ggml_argmax(ctx0, logits);
                    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
                    ggml_tensor * prob = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, probs, 1, probs->ne[0]), pick);
                    id = id_map ? ggml_reshape_1d(ctx0, ggml_get_rows(ctx0,
                            ggml_reshape_2d(ctx0, id_map, 1, id_map->ne[0]), pick), 1) : pick;
                    output = ggml_concat(ctx0, ggml_reshape_2d(ctx0, ggml_cast(ctx0, id, GGML_TYPE_F32), 1, 1),
                            ggml_reshape_2d(ctx0, prob, 1, 1), 0);
                }
                logits_all = logits_all ? ggml_concat(ctx0, logits_all, output, 1) : output;
            }
            if (j + 1 < n_chain) {
                GGML_ASSERT(id);
                projected_cur = build_proj(ggml_get_rows(ctx0, model.tok_embd, id), hidden, 1);
            }
        }
        if (batch_full_head) {
            // One full-vocabulary projection covers all saved head columns. Do not use
            // build_draft_vocab_logits[_chain]: the loaded map is provisional-only in this mode.
            ggml_tensor * full_logits = build_lora_mm(model.output, head_all, model.output_s);
            GGML_ASSERT(full_logits->ne[0] == model.output->ne[1] && full_logits->ne[1] == n_chain);
            cb(full_logits, "mtp_chain_full_logits", il);
            for (int64_t j = 0; j < n_chain; ++j) {
                ggml_tensor * logits = ggml_view_2d(ctx0, full_logits, full_logits->ne[0], 1,
                        full_logits->nb[1], j*full_logits->nb[1]);
                ggml_tensor * u = ggml_view_1d(ctx0, chain_u, 1, j*chain_u->nb[0]);
                ggml_tensor * output = llama_mtp_chain_sample_graph(ctx0, logits, chain_k, chain_samp, u, nullptr, nullptr, true);
                if (j + 1 < n_chain || test_ids.size() == (size_t) n_chain) {
                    // row[0] records the token that actually conditioned the next residual. The full candidate
                    // IDs/logits and row[1] remain untouched; the host compares token IDs before using later rows.
                    ggml_tensor * provisional_id;
                    if (j + 1 < n_chain) {
                        provisional_id = ggml_reshape_2d(ctx0,
                                ggml_cast(ctx0, conditioned_ids[j], GGML_TYPE_F32), 1, 1);
                    } else {
                        // The optional Nth test ID changes only the final comparison field; there is no next row.
                        provisional_id = ggml_scale_bias(ctx0,
                                ggml_view_2d(ctx0, output, 1, 1, output->nb[1], 0), 0.0f, (float) test_ids[j]);
                    }
                    ggml_tensor * rest = ggml_view_2d(ctx0, output, LLAMA_MTP_CHAIN_ROW(chain_k) - 1, 1,
                            output->nb[1], sizeof(float));
                    output = ggml_concat(ctx0, provisional_id, rest, 0);
                }
                logits_all = logits_all ? ggml_concat(ctx0, logits_all, output, 1) : output;
            }
        }
        res->t_h_nextn = hidden_all;
        res->t_logits = logits_all;
        cb(hidden_all, "h_nextn", il);
        cb(logits_all, "result_output", -1);
        ggml_build_forward_expand(gf, hidden_all);
        if (inp_out_ids) {
            ggml_build_forward_expand(gf, inp_out_ids);
        }
        ggml_build_forward_expand(gf, logits_all);
        return;
    }

    ggml_tensor * h_norm = build_norm(ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens), layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * eh_input = ggml_concat(ctx0, e_norm, h_norm, 0);
    // The original [2*n_embd, hc, n_tokens] shape dispatches a separate
    // hc-column projection per token. Flattening the contiguous token/HC
    // columns lets prompt catch-up reuse the same weights in a large GEMM.
    // Keep sampling and the default path unchanged until this shape is gated.
    static const bool batch_eh = [] {
        const char * value = std::getenv("LLAMA_MTP_EH_BATCH");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    const bool eh_batched = batch_eh && kv_only && n_tokens > 1;
    if (eh_batched) {
        GGML_ASSERT(ggml_is_contiguous(eh_input));
        eh_input = ggml_reshape_2d(ctx0, eh_input, 2*n_embd, hc*n_tokens);
        static std::atomic_flag noted = ATOMIC_FLAG_INIT;
        if (!noted.test_and_set()) {
            LLAMA_LOG_INFO("%s: MTP catch-up EH projection batches HC/token columns (LLAMA_MTP_EH_BATCH=1)\n", __func__);
        }
    }
    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, eh_input);
    if (eh_batched) {
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, n_tokens);
    }
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
    if (kv_only) {
        if (inp_kpool && hparams.dsv4_compress_ratios[il] > 0) {
            // The next output query needs every raw indexer key and pooled key,
            // but no catch-up row consumes the indexer scores or selection.
            build_qsa_store(mctx_hyb, inp_kpool, cur, sections, il);
        }
        // Match build_layer_attn's consumed K/V operations and their order.
        // Keep the normal HC mix above, including its BF16 cast decision.
        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        Kcur = ggml_reshape_3d(ctx0, Kcur, hparams.n_embd_head_k(), n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);
        Vcur = ggml_reshape_3d(ctx0, Vcur, hparams.n_embd_head_v(), n_head_kv, n_tokens);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        build_attn_store(inp_hyb->get_attn(), Kcur, Vcur, il);
        return;
    }
    cur    = build_layer_attn(inp_hyb->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    cur    = build_hc_mix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
    cur    = build_layer_ffn(cur, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    // the next draft step reads this residual as its h
    ggml_tensor * flat     = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
    res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    cb(res->t_h_nextn, "h_nextn", il);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    // The full catch-up fallback has no output head. Unmasked nextn readers
    // still use its complete residual above; sampling/embedding requests keep the head.
    if (n_outputs == 0 && !cparams.embeddings && !cparams.mtp_chain) {
        return;
    }

    cur = build_hc_mix(ggml_reshape_3d(ctx0, flat_out, n_embd, hc, flat_out->ne[1]),
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, il);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // draft-only vocabulary shortlist (when the context carries a map, e.g. the built-in flash-next-65536):
    // score only the shortlisted head rows; the backend sampler maps its picks back to token ids.
    // model.output_s is null except for an NVFP4 head; both calls accept a null scale.
    // (ported from bl/grok-242 cbafc8c3f; before this the map was loaded but the draft scored all 248,320 rows)
    // Repair and unsupported-chain fallbacks must retain full q while the provisional map remains loaded.
    ggml_tensor * logits = full_head_mode ? nullptr : build_draft_vocab_logits(model.output, model.output_s, cur);
    cur = logits ? logits : build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb, bool store_only) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    // fork: chain roots also pass store_only, but their row-readiness inputs still need every pool.
    // Without the incremental layout, keep both allocation and fill on the original full path.
    inp->store_only = store_only && !cparams.mtp_chain && mctx_hyb->use_kpool_store_only();
    inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    if (!inp->store_only) {
        inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
        inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
        // LLAMA_QSA_POOL_RANGE=1 (opt-in): each token's visible blocks are one run [lo, hi) of the block order, so the mask
        // can be two numbers per token instead of a row of n_pool 0/-inf values (the CPU fill and the copy of that row cost
        // seconds per 100K prompt). The indexer adds the same 0 or -inf, so the scores are the same bit for bit.
        static const bool pool_range = [] { const char * e = getenv("LLAMA_QSA_POOL_RANGE"); return e && atoi(e) != 0; }();
        inp->pool_mask  = pool_range ? ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 2, n_tokens) :
                                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n_tokens);
        inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
        ggml_set_input(inp->pool_cells);
        ggml_set_input(inp->pool_idxs);
        ggml_set_input(inp->pool_mask);
        ggml_set_input(inp->tail_idxs);

        // set_input fills them all. Store-only has no later readers, so pin
        // their lifetime against aliasing other inputs. A VIEW node also keeps
        // growing leaf sizes visible to the allocator's node/source validation.
        const auto expand_input = [&](ggml_tensor * tensor) {
            if (store_only) {
                ggml_set_output(tensor);
                ggml_tensor * input = tensor;
                tensor = ggml_view_4d(ctx0, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                        tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
                GGML_ASSERT(tensor->op == GGML_OP_VIEW && tensor->src[0] == input);
                ggml_set_output(tensor);
            }
            ggml_build_forward_expand(gf, tensor);
        };
        expand_input(inp->pool_cells);
        expand_input(inp->pool_idxs);
        expand_input(inp->pool_mask);
        expand_input(inp->tail_idxs);

        // the top blocks plus the tail
        inp->n_sel = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;
    }
    inp->n_kv  = mctx_idx->get_n_kv();
    inp->n_new = mctx_hyb->get_n_kpool_new();

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    // one scatter row per new pool, each a distinct rep row (see kpool_build_state)
    inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
    ggml_set_input(inp->new_pool_rep);
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_pos);

    if (cparams.mtp_chain) {
        inp->storage_cells = mctx_hyb->get_kpool_access(ctx0, hparams.n_layer(), hparams.indexer_head_size).get_n_cells();
        for (ggml_tensor * tensor : {inp->new_pool_idxs, inp->new_pool_rep, inp->new_pool_pos}) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(gf, tensor);
        }
    }

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool_slice(
        llm_graph_input_kpool * source, int64_t row0, int64_t width) {
    auto inp = std::make_unique<llm_graph_input_kpool>(source->mctx, source->kpool);
    inp->source = source;
    inp->row0 = row0;
    inp->width = width;
    inp->n_kv = source->n_kv;
    inp->n_sel = source->n_sel;
    inp->n_new = source->n_new;
    inp->pool_cells = source->pool_cells;
    inp->pool_idxs = source->pool_idxs;
    inp->k_idxs = ggml_view_1d(ctx0, source->k_idxs, width, row0*source->k_idxs->nb[0]);
    inp->pool_mask = ggml_view_2d(ctx0, source->pool_mask, source->pool_mask->ne[0], width,
            source->pool_mask->nb[1], row0*source->pool_mask->nb[1]);
    inp->tail_idxs = ggml_view_2d(ctx0, source->tail_idxs, source->tail_idxs->ne[0], width,
            source->tail_idxs->nb[1], row0*source->tail_idxs->nb[1]);
    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, inp->kpool, inp->n_new);
    inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    ggml_set_input(inp->new_pool_rep);
    ggml_set_input(inp->new_pool_pos);
    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

// Persist the indexer state shared by ordinary attention and no-output MTP catch-up.
// Queries/scores/selection below do not contribute to these cache writes.
void llama_model_qwen4exp::graph::build_qsa_store(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        int *                                   sections,
        int                                     il) {
    const int64_t n_tokens = cur->ne[1];
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_new   = inp_kpool->n_new;

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == kpool);

    // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    cb(k_raw, "indexer_k_raw", il);

    ggml_tensor * packed = k_raw;
    if (!mctx_hyb->has_compact_pool()) {
        ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
        packed = ggml_concat(ctx0, k_raw, pzero, 0);
    }
    packed = ggml_reshape_3d(ctx0, packed, packed->ne[0], 1, n_tokens);
    auto * raw_write = mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il);
    if (mctx_hyb->has_host_raw()) {
        // Only fresh keys cross to the host; never migrate the full original-key history.
        ggml_backend_sched_set_tensor_backend(sched, raw_write, backend_cpu);
    }
    ggml_build_forward_expand(gf, raw_write);

    // the raw keys and the persistent pooled slots, see llama_memory_hybrid_idx::mem_idx_stale
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    ggml_backend_t pool_backend = nullptr;
    if (mctx_hyb->has_host_raw()) {
        // Gather before transferring: only the members of new/stale pools visit the device.
        ggml_backend_sched_set_tensor_backend(sched, rows, backend_cpu);
        const auto pool_buft = kpool_cache.get_pool_buffer_type();
        for (int b = 0; b < ggml_backend_sched_get_n_backends(sched); ++b) {
            auto * backend = ggml_backend_sched_get_backend(sched, b);
            // CPU buffer types need not expose a device handle.
            const bool same_device = ggml_backend_buft_is_host(pool_buft) ? backend == backend_cpu :
                    ggml_backend_get_device(backend) == ggml_backend_buft_get_device(pool_buft);
            if (same_device && ggml_backend_supports_buft(backend, pool_buft)) {
                pool_backend = backend;
                break;
            }
        }
        GGML_ASSERT(pool_backend);
    }
    rows = ggml_reshape_3d(ctx0, rows, idx_dim, kpool, n_new);

    // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled_new = nullptr;
    for (int64_t i = 0; i < kpool; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx0, rows, idx_dim, n_new, rows->nb[2], i*rows->nb[1]);
        pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
        if (pool_backend) {
            ggml_backend_sched_set_tensor_backend(sched, pooled_new, pool_backend);
        }
    }
    pooled_new = ggml_scale(ctx0, pooled_new, 1.0f/(float) kpool);
    if (pool_backend) {
        ggml_backend_sched_set_tensor_backend(sched, pooled_new, pool_backend);
    }
    pooled_new = build_norm(pooled_new, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    pooled_new = ggml_reshape_3d(ctx0, pooled_new, idx_dim, 1, n_new);
    pooled_new = ggml_rope_multi(ctx0, pooled_new, inp_kpool->new_pool_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, idx_dim, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    // Preserve fresh pooled keys for the next query, including after a cache reload.
    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
}

// Gather complete quantized cells without decoding or requantizing them. GET_ROWS copies I32 words;
// the surrounding VIEW nodes only change byte interpretation and retain their source dependencies.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_gather_bytes(
        ggml_context * ctx, ggml_tensor * cache, ggml_tensor * ids) {
    GGML_ASSERT(cache->type == GGML_TYPE_TQ5_0 || cache->type == GGML_TYPE_TURBO4_0);
    GGML_ASSERT(cache->ne[0] == 256 && cache->ne[1] > 0 && cache->ne[2] > 0 && cache->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(ids->ne[0] > 0 && ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1);

    const size_t row_bytes = ggml_row_size(cache->type, cache->ne[0]);
    GGML_ASSERT(cache->nb[0] == ggml_type_size(cache->type) && cache->nb[1] == row_bytes);
    GGML_ASSERT(row_bytes % sizeof(int32_t) == 0);
    GGML_ASSERT((uint64_t) cache->ne[1] <= SIZE_MAX / row_bytes);
    const size_t cell_bytes = row_bytes * cache->ne[1];
    GGML_ASSERT(cache->nb[2] >= cell_bytes && cache->nb[2] % sizeof(int32_t) == 0);
    GGML_ASSERT(cache->nb[3] % sizeof(int32_t) == 0);
    GGML_ASSERT(cache->data == nullptr || (uintptr_t) cache->data % alignof(int32_t) == 0);
    GGML_ASSERT(cell_bytes / sizeof(int32_t) <= INT64_MAX);
    GGML_ASSERT((uint64_t) ids->ne[0] <= SIZE_MAX / cell_bytes);

    ggml_tensor * words = ggml_view_4d(ctx, cache, cache->ne[0], cache->ne[1], cache->ne[2], 1,
            cache->nb[1], cache->nb[2], cache->nb[3], 0);
    words->type  = GGML_TYPE_I32;
    words->ne[0] = cell_bytes / sizeof(int32_t);
    words->ne[1] = cache->ne[2];
    words->ne[2] = 1;
    words->nb[0] = sizeof(int32_t);
    words->nb[1] = cache->nb[2];
    words->nb[2] = cache->nb[3];
    GGML_ASSERT(ggml_nbytes(words) == ggml_nbytes(cache));

    ggml_tensor * gathered = ggml_get_rows(ctx, words, ids);
    GGML_ASSERT(gathered->type == GGML_TYPE_I32 && ggml_is_contiguous(gathered));
    ggml_tensor * result = ggml_view_4d(ctx, gathered, gathered->ne[0], gathered->ne[1], 1, 1,
            gathered->nb[1], gathered->nb[2], gathered->nb[3], 0);
    result->type  = cache->type;
    result->ne[0] = cache->ne[0];
    result->ne[1] = cache->ne[1];
    result->ne[2] = ids->ne[0];
    result->nb[0] = cache->nb[0];
    result->nb[1] = row_bytes;
    result->nb[2] = cell_bytes;
    result->nb[3] = cell_bytes * ids->ne[0];
    GGML_ASSERT(ggml_nbytes(result) == ggml_nbytes(gathered) && ggml_is_contiguous(result));
    return result;
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_sel(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        ggml_tensor **                          sel_ids,
        int *                                   sections,
        int                                     il) {
    const int64_t n_tokens = cur->ne[1];
    build_qsa_store(mctx_hyb, inp_kpool, cur, sections, il);
    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);
    // Fresh pooled keys were written above; older pools retain earlier ubatch values.
    ggml_tensor * pool_cells = inp_kpool->pool_cells;
    if (cparams.mtp_chain) {
        GGML_ASSERT(n_tokens == 1);
        ggml_tensor * cells = ggml_cast(ctx0, pool_cells, GGML_TYPE_F32);
        ggml_tensor * live;
        if (inp_kpool->pool_mask->type == GGML_TYPE_I32) {
            ggml_tensor * range = ggml_cast(ctx0, inp_kpool->pool_mask, GGML_TYPE_F32);
            ggml_tensor * lo = ggml_view_1d(ctx0, range, 1, 0);
            ggml_tensor * hi = ggml_view_1d(ctx0, range, 1, sizeof(float));
            ggml_tensor * rank = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, cells, 1.0f)), 1.0f, -1.0f);
            live = ggml_mul(ctx0,
                    ggml_clamp(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, rank, lo), 1.0f, 1.0f), 0.0f, 1.0f),
                    ggml_clamp(ctx0, ggml_neg(ctx0, ggml_sub(ctx0, rank, hi)), 0.0f, 1.0f));
        } else {
            live = ggml_reshape_1d(ctx0, ggml_cast(ctx0, inp_kpool->pool_mask, GGML_TYPE_F32), n_pool);
            live = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live, 1.0f, 1.0f), 0.0f, 1.0f);
        }
        // The padded last cell is the first batch row: its raw write also zeros its pooled half.
        // Masked future pools must not feed uninitialized values to a dot product before the mask is added.
        ggml_tensor * dummy = ggml_view_1d(ctx0, cells, 1, (n_pool - 1)*sizeof(float));
        pool_cells = ggml_cast(ctx0, ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, cells, dummy), live), dummy), GGML_TYPE_I32);
    }
    ggml_tensor * pooled = kpool_cache.gather_pooled(pool_cells);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim),
    // which is the lightning indexer with every head weight set to that scale
    ggml_tensor * weights = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_idx_h, n_tokens), 1.0f/sqrtf((float) idx_dim));
    ggml_tensor * score = ggml_lightning_indexer(ctx0, q, pooled, weights, inp_kpool->pool_mask); // [n_pool, n_tokens]
    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], unordered
    cb(top_k, "indexer_top_k", il);

    // the top blocks, then the incomplete tail with n_kv for missing cells
    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens)); // [kpool, n_top_pool*n_tokens]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);
    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == inp_kpool->n_sel);

    ggml_build_forward_expand(gf, sel_idx);

    // TODO: figure out to reduce the large copmute buffer that this creates
    // LLAMA_QSA_SEL_INPLACE=1 (opt-in) lowers the peak by ~3 GiB at ub 8192 and 128K context, same values:
    //  - the final slot indices (the last reads of the score) are queued before the n_kv-wide buffers exist;
    //  - the causal mask is added in place, so the selection stays a row-strided view of mask_all instead of a
    //    second n_kv x n_tokens buffer
    // LLAMA_QSA_SEL_DIRECT=1 (opt-in, implies the in-place ordering and padded rows) passes the selection to attention
    // without adding the causal mask, same values: every live selected cell is already visible to its token (a pool is
    // live only when it belongs to the token's sequence and ends at or before it, a tail cell is the token's own or an
    // earlier position of its sequence, and every dead slot goes to a dump row outside the n_kv-wide view), so the add
    // only ever adds 0 to 0 or something to -inf. Nothing then reads the n_kv x n_tokens causal mask, so it is never
    // filled on the host, uploaded or kept in the compute buffer (3 GiB at ub 12288 and 128K context)
    static const bool sel_direct  = [] { const char * e = getenv("LLAMA_QSA_SEL_DIRECT"); return e && atoi(e) != 0; }();
    static const bool sel_inplace = sel_direct || [] { const char * e = getenv("LLAMA_QSA_SEL_INPLACE"); return e && atoi(e) != 0; }();

    // scatter zeros for the selected cells into an all -inf row, each dead slot into its own dump row n_kv + slot
    // seeding from sel_idx ties the scatter storage lifetime to this layer
    const int64_t n_kv = inp_kpool->n_kv;

    // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and
    // top_k fills a short selection with invisible pools that can overlap the tail, so the scatter would write
    // some cells from several threads: map every dead slot to its own dump row, idx = dump + live*(idx - dump)
    // a picked pool is live when visible: a visible score is a rectified sum >= 0, an invisible one is -inf
    ggml_tensor * top_score = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
    ggml_tensor * live_pool = ggml_clamp(ctx0, ggml_scale_bias(ctx0, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, live_pool, kpool, n_top_pool, n_tokens, 1), kpool*n_top_pool, n_tokens);
    // a tail cell is live unless it is the n_kv sentinel
    ggml_tensor * live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);
    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]

    if (sel_ids) {
        GGML_ASSERT(n_tokens == 1 && n_kv > 0 && n_kv <= (1 << 24));
        GGML_ASSERT(n_sel > 0 && n_sel <= INT32_MAX - 255);
        GGML_ASSERT(kq_mask->ne[0] == n_kv && ggml_nelements(kq_mask) == n_kv);
        const int64_t n_compact = GGML_PAD(n_sel, 256);
        const int pad = (int) (n_compact - n_sel);
        ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);

        // Preserve original cell order among live selections; invisible slots sort after every real cell.
        ggml_tensor * sort_key = ggml_scale_bias(ctx0,
                ggml_mul(ctx0, ggml_scale_bias(ctx0, idx_f, 1.0f, -(float) n_kv), live), 1.0f, (float) n_kv);
        ggml_tensor * order = ggml_argsort(ctx0, sort_key, GGML_SORT_ORDER_ASC);
        ggml_tensor * ordered_idx = ggml_reshape_1d(ctx0,
                ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, idx_f, 1, n_sel), order), n_sel);
        ggml_tensor * ordered_live = ggml_reshape_1d(ctx0,
                ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, live, 1, n_sel), order), n_sel);

        // There is always a visible current cell, in either a complete pool or the incomplete tail.
        // Reuse a live cell for dead slots and FA padding: even masked slots must not decode uninitialized KV bytes.
        ggml_tensor * safe_idx = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, idx_f, 1, n_sel), ggml_argmax(ctx0, live));
        ggml_tensor * delta = ggml_mul(ctx0, ggml_sub(ctx0, ordered_idx, safe_idx), ordered_live);
        ggml_tensor * live_pad = ggml_pad(ctx0, ordered_live, pad, 0, 0, 0);
        *sel_ids = ggml_cast(ctx0, ggml_add(ctx0, ggml_pad(ctx0, delta, pad, 0, 0, 0), safe_idx), GGML_TYPE_I32);
        cb(*sel_ids, "indexer_compact_ids", il);

        // log(1) = 0 and log(0) = -inf, avoiding 0 * -inf for dead slots. DIRECT deliberately omits the
        // host causal input; otherwise retain every original mask value by gathering the same selected cells.
        ggml_tensor * mask = ggml_log(ctx0, live_pad);
        if (!sel_direct) {
            ggml_tensor * causal = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, kq_mask, 1, n_kv), *sel_ids);
            mask = ggml_add(ctx0, mask, ggml_reshape_1d(ctx0, causal, n_compact));
        }
        mask = ggml_reshape_4d(ctx0, ggml_cast(ctx0, mask, kq_mask->type), n_compact, 1, 1, 1);
        cb(mask, "indexer_sel_compact", il);
        return mask;
    }

    // In place, the selection is a row-strided view of mask_all: keep the stride a multiple of 16 bytes for FA.
    // Compact attention above never constructs these history-wide mask buffers.
    const int64_t n_row = sel_inplace ? GGML_PAD(n_kv + n_sel, 64) : n_kv + n_sel;
    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_row, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_row, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_row, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source
    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);
    if (sel_inplace) {
        ggml_build_forward_expand(gf, sel_idx);
    }

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
            row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
    if (!sel_direct) {
        sel = sel_inplace ? ggml_add_inplace(ctx0, sel, kq_mask) : ggml_add(ctx0, sel, kq_mask);
    }
    cb(sel, "indexer_sel", il);

    return sel;
}

// Dense GQA self-attention over the cells that the QSA mask keeps.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             sel,
        ggml_tensor *             sel_ids,
        int64_t                   n_sel,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_sel, so the selection is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // SJ-KVaRN: the same Q/K/V rotation build_attn applies; the FA node gets the sealed records attached below
    const bool sj       = inp->self_sj_kvarn_desc != nullptr;
    const bool sj_fused = sj && build_attn_sj_kvarn_rot(inp, q_cur, k_cur, v_cur, il);

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
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    if (sj) {
        build_attn_sj_kvarn_pend(inp, sj_fused, il);
    }

    // the selection mask already carries the causal mask
    ggml_tensor * kq_mask = inp->get_kq_mask();
    // an in-place selection is a row-strided view that already has the mask shape
    GGML_ASSERT(sel_ids || ggml_is_contiguous(sel) || ggml_are_same_shape(sel, kq_mask));
    ggml_tensor * mask = sel_ids ? sel : ggml_is_contiguous(sel) ?
            ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]) : sel;
    cb(mask, "kq_mask_qsa", il);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    if (sel_ids) {
        GGML_ASSERT(!sj && q_cur->ne[2] == 1);
        k = build_qsa_gather_bytes(ctx0, k, sel_ids);
        v = build_qsa_gather_bytes(ctx0, v, sel_ids);
        cb(k, "K_cache_compact", il);
        cb(v, "V_cache_compact", il);
    }

    // TurboQuant/TQ K caches hold WHT-rotated K: rotate Q to match, as the dense path does
    q = build_attn_turbo_q(q, k, mctx_cur);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
    cb(cur, "kqv_out", il);

    GGML_ASSERT(sj_kvarn_pending.body == nullptr && "SJ-KVaRN cache requires the flash-attention path");

    cur = build_attn_turbo_v_out(cur, v, il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *   inp_kpool,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    static const bool compact = [] { const char * e = getenv("LLAMA_MTP_QSA_COMPACT"); return e && atoi(e) != 0; }();
    bool use_compact = compact && qsa && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && n_tokens == 1 &&
            cparams.flash_attn && n_embd_head == 256 && inp->self_sj_kvarn_desc == nullptr &&
            inp_kpool->n_kv > 0 && inp_kpool->n_kv <= (1 << 24);
    if (use_compact) {
        // Only native, non-transposed compressed KV cells are eligible. Other layouts retain the dense mask path.
        auto eligible = [](const ggml_tensor * t) {
            return (t->type == GGML_TYPE_TQ5_0 || t->type == GGML_TYPE_TURBO4_0) &&
                    t->ne[0] == 256 && t->ne[1] > 0 && t->ne[2] > 0 && t->ne[3] == 1 &&
                    t->nb[0] == ggml_type_size(t->type) && t->nb[1] == ggml_row_size(t->type, 256) &&
                    t->nb[2] >= t->nb[1]*(size_t) t->ne[1] && t->nb[2] % sizeof(int32_t) == 0 &&
                    t->nb[3] % sizeof(int32_t) == 0;
        };
        use_compact = eligible(inp->mctx->get_k(ctx0, il)) && eligible(inp->mctx->get_v(ctx0, il));
    }
    ggml_tensor * sel_ids = nullptr;
    ggml_tensor * sel = qsa ? build_qsa_sel(mctx_hyb, inp_kpool, cur, inp_pos, inp->get_kq_mask(),
            use_compact ? &sel_ids : nullptr, sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (sel) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, sel, sel_ids, inp_kpool->n_sel, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    // [GDNGETTER] LLAMA_GDN_STATE_GETTER=1: gather the ubatch states with their own GET_ROWS(cache, s_copy_main)
    // instead of a view of the one combined gather, so that GET_ROWS has the gdn as its only reader and the CUDA
    // #87 plan (gdn reads the cache rows in place) can take it. Same rows, same bytes; default off.
    static const bool gdn_getter = [] { const char * e = getenv("LLAMA_GDN_STATE_GETTER"); return e != nullptr && atoi(e) != 0; }();
    ggml_tensor * state = gdn_getter ?
        build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs,
                 [](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) { return ggml_get_rows(ctx, states, ids); }) :
        build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    if (moe_out) {
        cb(moe_out, "ffn_moe_out", il);
    }

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = moe_out ? ggml_add(ctx0, moe_out, ffn_shexp) : ffn_shexp;
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out ? moe_out : ggml_scale(ctx0, cur, 0.0f);
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
    llm_graph_input_qwen4exp_ple(const llama_model & model,
                        const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}
    virtual ~llm_graph_input_qwen4exp_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) model.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // [PLEPF] the context's page prefetch, null when off
    llama_ple_prefetch * pf = nullptr;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_qwen4exp_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hparams = model.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hparams.ple_image_token_id != 0
        ? (llama_token) hparams.ple_image_token_id
        : (llama_token) hparams.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[k]);
        return is_embd ? img_tok : ubatch->token[k];
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t n_heads  = hparams.ple_n_heads;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
            }
        }
    }

    if (model.ple_cache && !model.ple_cache->remap(idx.data(), idx.size())) {
        GGML_ABORT("PLE startup profile is incomplete; stopping before the gather");
    }

    {
        ggml_tensor * ple = model.per_layer_tok_embd;

        // [RTPORT] LLAMA_PLE_PREFETCH_MIN_ROWS=N (default 0 = always): skip the page prefetch for calls gathering fewer
        // than N rows. A check or guess batch gathers 16 rows per token, one madvise(MADV_WILLNEED) system call each
        // (~250 us per 5-token check on resident pages, card idle meanwhile). The prefetch is only advice to the
        // kernel: the gather reads the same bytes either way.
        static const size_t min_rows = [] {
            const char * v = getenv("LLAMA_PLE_PREFETCH_MIN_ROWS");
            return v ? (size_t) std::max(0, atoi(v)) : (size_t) 0;
        }();
        const bool prefetch = model.can_prefetch.count(ple) && idx.size() >= min_rows;
        // [PLEPF] the context already reads this call's pages ahead: wait for this ubatch's instead
        const bool covered = pf && pf->consume((uint32_t) n_tokens, idx.data(), idx.size());
        if (prefetch && !covered) {
            llama_prefetch_rows(ple, idx.data(), idx.size());
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    const bool first_use = it == rs_rows.end();
    if (first_use) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots   = (int64_t) cparams.n_rs_seq + 1;
    const int64_t n_written = std::min<int64_t>(ubatch.n_seq_tokens, n_slots);

    // only this ubatch's windows are written (slot < min(n, K)); the older slots move back by n
    // ([TAG_RECURRENT_ROLLBACK_SHIFT], no-op for n >= K). Writing every slot with the window clamped to
    // the pre-ubatch one filled the slots past n with it while rs_valid counted them as older history.
    // The shift runs once per state tensor and graph, on the call that also builds its rows (rs_rows):
    // a second shift of the same tensor would move the history back twice. plane0 follows
    // build_conv_state (the pending replay in replay mode, s_copy_shift carries rs_idx otherwise).
    std::vector<ggml_tensor *> older;
    if (first_use) {
        snapshot_shift_gather(ctx0, gf, inp, conv_states_all, row_total, mctx_cur->get_replay_len(), mem_size, older);
    }

    for (int64_t slot = 0; slot < n_written; ++slot) {
        const int64_t s_idx = conv_input->ne[0] - state_cols - slot; // = n_seq_tokens - slot >= 1

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // [CONVNOCONT] LLAMA_CONV_NOCONT=1 copies the strided window straight into the cache row (the copy
        // takes any source strides, so the bytes are the same) instead of making it contiguous first. That is
        // one launch less per slot, and it is the CPY(VIEW(concat)) form the CUDA conv ring fusion
        // (ggml_cuda_conv_ring_match_at) recognizes, so the concat and every window copy fold into the
        // ssm_conv launch. Default on (byte-identical tokens, 100K verify -1.2/-1.4 ms at w3/w4); LLAMA_CONV_NOCONT=0 turns it off.
        static const bool no_cont = [] {
            const char * e = getenv("LLAMA_CONV_NOCONT");
            return e == nullptr || atoi(e) != 0;
        }();
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, no_cont ? tail : ggml_cont(ctx0, tail), dst));
    }

    snapshot_shift_write(ctx0, gf, inp, conv_states_all, row_total, ubatch.n_seq_tokens, kv_head, mem_size, older);

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_qwen4exp_ple>(model, mctx_hyb->get_attn());
    ple_inp->pf = ple_pf;

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * table = model.ple_cache ? model.ple_cache->tensor : model.per_layer_tok_embd;
    ggml_tensor * emb = ggml_get_rows(ctx0, table, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
