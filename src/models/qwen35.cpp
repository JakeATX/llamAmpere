#include "models.h"
#include "llama-impl.h"
#include "llama-lsa.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-mtp-chain-sample.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <vector>

// Output-row gather indices, or nullptr when every row is an output (decode, speculative verify,
// MTP drafts): the gather would then be an identity copy. The graph topology still depends only on
// n_outputs vs n_tokens, which graph reuse already compares. Pipeline parallelism keeps the gather
// for a constant topology (see build_inp_out_ids()).
static ggml_tensor * qwen35_build_inp_out_ids(const llm_graph_context & g) {
    if (g.n_outputs == g.n_tokens && !g.cparams.pipeline_parallel) {
        return nullptr;
    }
    return g.build_inp_out_ids();
}

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // Mixed-shape pruned model (ngprune): per-layer GDN value heads, sublayer kinds, rank/span widths. All seven keys are
    // present together; a plain Qwen3.5 file has none of them and takes none of the paths gated on hparams.mix_enabled.
    {
        const uint32_t nl = hparams.n_layer();
        std::array<uint32_t, LLAMA_MAX_LAYERS> v_heads{}, mixer_kind{}, mlp_kind{}, mixer_width{}, mlp_width{}, mixer_span{}, mlp_span{};
        if (ml.get_key_or_arr(LLM_KV_MIX_MIXER_KIND, mixer_kind, nl, false)) {
            if (hparams.ssm_dt_rank == 0 || hparams.ssm_d_inner % hparams.ssm_dt_rank != 0) {
                throw std::runtime_error("invalid mixed-shape GDN head count");
            }
            ml.get_key_or_arr(LLM_KV_MIX_SSM_V_HEADS,  v_heads,     nl, true);
            ml.get_key_or_arr(LLM_KV_MIX_MLP_KIND,     mlp_kind,    nl, true);
            ml.get_key_or_arr(LLM_KV_MIX_MIXER_WIDTH,  mixer_width, nl, true);
            ml.get_key_or_arr(LLM_KV_MIX_MLP_WIDTH,    mlp_width,   nl, true);
            ml.get_key_or_arr(LLM_KV_MIX_MIXER_SPAN,   mixer_span,  nl, true); // span id + 1, 0 = none
            ml.get_key_or_arr(LLM_KV_MIX_MLP_SPAN,     mlp_span,    nl, true);

            hparams.mix_ssm_k_heads.fill(hparams.ssm_n_group);
            hparams.mix_ssm_k_dim.fill(hparams.ssm_d_state);
            hparams.mix_ssm_v_dim.fill(hparams.ssm_d_inner / hparams.ssm_dt_rank);
            hparams.mix_attn_k_dim.fill(hparams.n_embd_head_k_full);
            hparams.mix_attn_v_dim.fill(hparams.n_embd_head_v_full);
            ml.get_key_or_arr(LLM_KV_MIX_SSM_K_HEADS, hparams.mix_ssm_k_heads, nl, false);
            ml.get_key_or_arr(LLM_KV_MIX_SSM_K_DIM, hparams.mix_ssm_k_dim, nl, false);
            ml.get_key_or_arr(LLM_KV_MIX_SSM_V_DIM, hparams.mix_ssm_v_dim, nl, false);
            ml.get_key_or_arr(LLM_KV_MIX_ATTN_K_DIM, hparams.mix_attn_k_dim, nl, false);
            ml.get_key_or_arr(LLM_KV_MIX_ATTN_V_DIM, hparams.mix_attn_v_dim, nl, false);

            hparams.mix_enabled = true;
            for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
                const bool trunk = il < nl;
                hparams.mix_ssm_v_heads[il] = trunk ? v_heads[il] : hparams.ssm_dt_rank;
                hparams.mix_mixer_kind [il] = trunk ? mixer_kind [il] : (uint32_t) llama_hparams::MIX_FULL;
                hparams.mix_mlp_kind   [il] = trunk ? mlp_kind   [il] : (uint32_t) llama_hparams::MIX_FULL;
                hparams.mix_mixer_width[il] = trunk ? mixer_width[il] : hparams.n_embd;
                hparams.mix_mlp_width  [il] = trunk ? mlp_width  [il] : hparams.n_embd;
                hparams.mix_mixer_span [il] = trunk ? (int32_t) mixer_span[il] - 1 : -1;
                hparams.mix_mlp_span   [il] = trunk ? (int32_t) mlp_span  [il] - 1 : -1;

                if (hparams.mix_mixer_kind[il] != llama_hparams::MIX_REMOVED && (hparams.mix_ssm_k_heads[il] == 0 || hparams.mix_ssm_k_heads[il] > hparams.ssm_n_group ||
                    hparams.mix_ssm_k_dim[il] == 0 || hparams.mix_ssm_k_dim[il] > hparams.ssm_d_state ||
                    hparams.mix_ssm_v_dim[il] == 0 || hparams.mix_ssm_v_dim[il] > hparams.ssm_d_inner / hparams.ssm_dt_rank ||
                    hparams.mix_attn_k_dim[il] < hparams.n_rot_full || hparams.mix_attn_k_dim[il] > hparams.n_embd_head_k_full ||
                    hparams.mix_attn_v_dim[il] == 0 || hparams.mix_attn_v_dim[il] > hparams.n_embd_head_v_full ||
                    hparams.mix_ssm_v_heads[il] > hparams.ssm_dt_rank)) {
                    throw std::runtime_error(format("invalid mixed-shape head dimensions in layer %u", il));
                }
                if (hparams.mix_mixer_kind[il] > llama_hparams::MIX_SPAN || hparams.mix_mlp_kind[il] > llama_hparams::MIX_SPAN) {
                    throw std::runtime_error(format("invalid mixed-shape sublayer kind in layer %u", il));
                }
                if (trunk) {
                    for (bool mixer : {true, false}) {
                        const uint32_t kind = mixer ? mixer_kind[il] : mlp_kind[il];
                        const uint32_t width = mixer ? mixer_width[il] : mlp_width[il];
                        const uint32_t span = mixer ? mixer_span[il] : mlp_span[il];
                        if ((kind != llama_hparams::MIX_REMOVED && (width == 0 || width > hparams.n_embd)) ||
                            (kind == llama_hparams::MIX_FULL && width != hparams.n_embd) || span > LLAMA_MAX_LAYERS) {
                            throw std::runtime_error(format("invalid mixed-shape width or span in layer %u", il));
                        }
                    }
                }
                if (trunk && hparams.is_recr(il) && hparams.mix_mixer_kind[il] != llama_hparams::MIX_REMOVED && hparams.mix_ssm_v_heads[il] == 0) {
                    throw std::runtime_error(format("mixed-shape layer %u has a GDN mixer with no value heads", il));
                }
                if (trunk && (hparams.mix_mixer_kind[il] == llama_hparams::MIX_SPAN) != (hparams.mix_mixer_span[il] >= 0)) {
                    throw std::runtime_error(format("mixed-shape layer %u: mixer span id does not match its kind", il));
                }
                if (trunk && (hparams.mix_mlp_kind[il] == llama_hparams::MIX_SPAN) != (hparams.mix_mlp_span[il] >= 0)) {
                    throw std::runtime_error(format("mixed-shape layer %u: mlp span id does not match its kind", il));
                }
            }
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;



    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            // three layouts: upstream merged attn_qkv_gate (wqkv_z), our EXL3 fused attn_qkv = [in_proj_qkv | in_proj_z]
            // (ne1 = conv_dim + value_dim, attn_gate absent; the graph splits it with views in build_qkvz), or separate
            const int separate_flags = flags & TENSOR_SKIP ? flags | TENSOR_NOT_REQUIRED : flags;
            layer.wqkv_z = create_tensor(tn(LLM_TENSOR_ATTN_QKV_GATE, "weight", il), { n_embd, key_dim * 2 + value_dim * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wqkv_z) {
                const ggml_tensor * qkv_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_QKV, "weight", il).str().c_str());
                const bool fused_qkvz = qkv_meta != nullptr && qkv_meta->ne[1] == conv_dim + value_dim;
                layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, fused_qkvz ? conv_dim + value_dim : conv_dim }, TENSOR_NOT_REQUIRED);
                layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
                GGML_ASSERT(!fused_qkvz || layer.wqkv_gate == nullptr);
            }
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta_alpha_cat = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA_CAT, "weight", il), { n_embd, n_v_heads * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.ssm_beta_alpha_cat) {
                layer.ssm_beta  = create_tensor(tn(LLM_TENSOR_SSM_BETA,  "weight", il), { n_embd, n_v_heads }, separate_flags);
                layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, separate_flags);
            }
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // EXL3 fused layout: ffn_up = [ffn_gate | ffn_up] (ne1 = 2*n_ff) and ffn_gate absent (the FFN uses swiglu).
        const ggml_tensor * up_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_UP, "weight", il).str().c_str());
        const bool fused_gate_up = up_meta != nullptr && up_meta->ne[1] == 2 * n_ff;
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, fused_gate_up ? (flags | TENSOR_NOT_REQUIRED) : flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, fused_gate_up ? 2 * n_ff : n_ff}, flags);
        GGML_ASSERT(!fused_gate_up || layer.ffn_gate == nullptr);

        // Require every tensor when the parallel branch is enabled.
        if (hparams.n_ff_par > 0) {
            const int64_t n_ff_par = hparams.n_ff_par;
            layer.ffn_gate_par = create_tensor(tn(LLM_TENSOR_FFN_GATE_PAR, "weight", il), {n_embd,   n_ff_par}, flags);
            layer.ffn_down_par = create_tensor(tn(LLM_TENSOR_FFN_DOWN_PAR, "weight", il), {n_ff_par, n_embd}, flags);
            layer.ffn_up_par   = create_tensor(tn(LLM_TENSOR_FFN_UP_PAR,   "weight", il), {n_embd,   n_ff_par}, flags);
        }
    };

    // Mixed-shape pruned layer (ngprune, hparams.mix_enabled): per-layer head / neuron counts, absent sublayers, rank and
    // span read/write ramps. A full sublayer of this path has the same tensors as load_block_trunk() creates.
    mix_span_read.clear();
    mix_span_write.clear();

    auto load_block_trunk_mix = [&](int il, int flags) {
        auto & layer = layers[il];

        const uint32_t mixer_kind = hparams.mixer_kind(il);
        const uint32_t mlp_kind   = hparams.mlp_kind(il);

        // the read/write ramps of a sublayer: a rank sublayer owns them, a span sublayer shares them with the layers of its
        // span (created once, at the first layer that uses the span, so they sit on that layer's device)
        auto load_ramps = [&](uint32_t kind, int32_t span, int64_t width, bool is_mixer) {
            if (kind == llama_hparams::MIX_RANK) {
                ggml_tensor * rd = create_tensor(tn(is_mixer ? LLM_TENSOR_MIX_MIXER_READ  : LLM_TENSOR_MIX_MLP_READ,  "weight", il), { n_embd, width }, flags);
                ggml_tensor * wr = create_tensor(tn(is_mixer ? LLM_TENSOR_MIX_MIXER_WRITE : LLM_TENSOR_MIX_MLP_WRITE, "weight", il), { width, n_embd }, flags);
                (is_mixer ? layer.mixer_read  : layer.mlp_read)  = rd;
                (is_mixer ? layer.mixer_write : layer.mlp_write) = wr;
            } else if (kind == llama_hparams::MIX_SPAN) {
                GGML_ASSERT(span >= 0);
                if ((size_t) span >= mix_span_read.size()) {
                    mix_span_read .resize(span + 1, nullptr);
                    mix_span_write.resize(span + 1, nullptr);
                }
                if (mix_span_read[span] == nullptr) {
                    mix_span_read [span] = create_tensor(tn(LLM_TENSOR_MIX_SPAN_READ,  "weight", il, span), { n_embd, width }, flags);
                    mix_span_write[span] = create_tensor(tn(LLM_TENSOR_MIX_SPAN_WRITE, "weight", il, span), { width, n_embd }, flags);
                }
                if (mix_span_read[span] != nullptr && mix_span_read[span]->ne[1] != width) {
                    throw std::runtime_error(format("mixed-shape span %d has mismatched widths", span));
                }
            }
        };

        // ---- mixer (GDN or attention)
        if (mixer_kind != llama_hparams::MIX_REMOVED) {
            const int64_t mw = hparams.mix_mixer_width[il]; // width the sublayer weights read and write
            GGML_ASSERT(mixer_kind != llama_hparams::MIX_FULL || mw == n_embd);

            // a span sublayer has no norm: its weight is folded into the input projections
            if (mixer_kind != llama_hparams::MIX_SPAN) {
                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, flags);
            }
            load_ramps(mixer_kind, hparams.mix_mixer_span[il], mw, true);

            if (!hparams.is_recr(il)) {
                const int64_t n_q  = hparams.n_head(il);
                const int64_t n_kv = hparams.n_head_kv(il);
                const int64_t dk = hparams.mix_attn_k_dim[il];
                const int64_t dv = hparams.mix_attn_v_dim[il];
                create_tensor_qkv(layer, il, mw, (dk + dv) * n_q, dk * n_kv, dv * n_kv, flags);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { dv * n_q, mw }, flags);

                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { dk }, flags);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { dk }, flags);

                // uneven GQA groups: the converter pads every KV group to the largest group (see build_layer_attn)
                const ggml_tensor * pad_meta = ml.get_tensor_meta(tn(LLM_TENSOR_MIX_ATTN_Q_PAD_IDX, "weight", il).str().c_str());
                const ggml_tensor * inv_meta = ml.get_tensor_meta(tn(LLM_TENSOR_MIX_ATTN_Q_INV_IDX, "weight", il).str().c_str());
                if (pad_meta != nullptr && inv_meta != nullptr) {
                    if (pad_meta->type != GGML_TYPE_I32 || inv_meta->type != GGML_TYPE_I32 ||
                        n_kv == 0 || pad_meta->ne[0] <= 0 || pad_meta->ne[0] % n_kv != 0 || pad_meta->ne[0] > n_q * n_kv) {
                        throw std::runtime_error(format("invalid mixed-shape attention head map in layer %d", il));
                    }
                    layer.attn_q_pad_idx = create_tensor(tn(LLM_TENSOR_MIX_ATTN_Q_PAD_IDX, "weight", il), { pad_meta->ne[0] }, flags);
                    layer.attn_q_inv_idx = create_tensor(tn(LLM_TENSOR_MIX_ATTN_Q_INV_IDX, "weight", il), { n_q }, flags);
                } else {
                    if (pad_meta != nullptr || inv_meta != nullptr || n_kv == 0 || n_q == 0 || n_q % n_kv != 0) {
                        throw std::runtime_error(format("missing mixed-shape attention head map in layer %d", il));
                    }
                }
            } else {
                const int64_t head_k_dim = hparams.ssm_k_dim_l(il);
                const int64_t head_v_dim = hparams.ssm_v_dim_l(il);
                const int64_t n_k_heads  = hparams.ssm_k_heads_l(il);
                const int64_t n_v_heads  = hparams.mix_ssm_v_heads[il];
                const int64_t key_dim    = head_k_dim * n_k_heads;
                const int64_t value_dim  = head_v_dim * n_v_heads;
                const int64_t conv_dim   = key_dim * 2 + value_dim;

                layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { mw, conv_dim }, flags);
                layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { mw, value_dim }, flags);
                layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
                layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { n_v_heads }, flags);
                layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { n_v_heads }, flags);
                layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { mw, n_v_heads }, flags);
                layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { mw, n_v_heads }, flags);
                layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
                layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, mw }, flags);

                // value head -> key head map; absent when the layer keeps every value head (the usual tiled mapping applies)
                const ggml_tensor * kidx_meta = ml.get_tensor_meta(tn(LLM_TENSOR_MIX_SSM_KEY_IDX, "weight", il).str().c_str());
                if (kidx_meta != nullptr) {
                    if (kidx_meta->type != GGML_TYPE_I32) {
                        throw std::runtime_error(format("invalid mixed-shape GDN head map type in layer %d", il));
                    }
                    layer.ssm_key_idx = create_tensor(tn(LLM_TENSOR_MIX_SSM_KEY_IDX, "weight", il), { n_v_heads }, flags);
                }
            }
        }

        // ---- MLP
        if (mlp_kind != llama_hparams::MIX_REMOVED) {
            const int64_t fw   = hparams.mix_mlp_width[il];
            const int64_t n_ff_l = hparams.n_ff(il);
            GGML_ASSERT(n_ff_l > 0);
            GGML_ASSERT(mlp_kind != llama_hparams::MIX_FULL || fw == n_embd);

            if (mlp_kind != llama_hparams::MIX_SPAN) {
                layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);
            }
            load_ramps(mlp_kind, hparams.mix_mlp_span[il], fw, false);

            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { fw, n_ff_l }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff_l, fw }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { fw, n_ff_l }, flags);
        } else {
            GGML_ASSERT(hparams.n_ff(il) == 0);
        }
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];
        const int64_t n_head = hparams.n_head(il);
        const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
        const int64_t n_embd_head_v = hparams.n_embd_head_v(il);
        const int64_t n_embd_k_gqa = n_embd_head_k * hparams.n_head_kv(il);
        const int64_t n_embd_v_gqa = n_embd_head_v * hparams.n_head_kv(il);
        const int64_t n_ff = hparams.n_ff(il);

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // EXL3 fused layout: ffn_up = [ffn_gate | ffn_up] (ne1 = 2*n_ff) and ffn_gate absent (the FFN uses swiglu).
        const ggml_tensor * up_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_UP, "weight", il).str().c_str());
        const bool fused_gate_up = up_meta != nullptr && up_meta->ne[1] == 2 * n_ff;
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, fused_gate_up ? (mtp_flags | TENSOR_NOT_REQUIRED) : mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, fused_gate_up ? 2 * n_ff : n_ff}, mtp_flags);
        GGML_ASSERT(!fused_gate_up || layer.ffn_gate == nullptr);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        if (hparams.mix_enabled) {
            load_block_trunk_mix(i, trunk_flags);
        } else {
            load_block_trunk(i, trunk_flags);
        }
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_qwen35::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur = nullptr;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = qwen35_build_inp_out_ids(*this);

    // Target layer -> predictions captured one to three attention layers earlier (graph-local only).
    const auto & lsa = llama_lsa_cfg();
    // Mixed-shape files (hparams.mix_enabled) take the recall probe and J3 only when every attention layer keeps a full
    // mixer with full head counts and dims (stale_query reads that layer's own norm and wq straight off the residual and
    // views it with the model-wide n_head / n_embd_head); otherwise both
    // are off for this graph, with one warning. Either way the attention itself stays exact (J3 only prefetches).
    bool lsa_side_ok = true;
    if (hparams.mix_enabled && (lsa.spec_recall || lsa.j3) && lsa.mode != llama_lsa_config::OFF) {
        for (int jl = 0; jl < n_layer; ++jl) {
            if (!hparams.is_recr(jl) && (hparams.mixer_kind(jl) != llama_hparams::MIX_FULL ||
                    hparams.mix_attn_k_dim[jl] != hparams.n_embd_head_k_full || hparams.mix_attn_v_dim[jl] != hparams.n_embd_head_v_full ||
                    hparams.n_head(jl) != (uint32_t) n_head || hparams.n_head_kv(jl) != (uint32_t) n_head_kv ||
                    model.layers[jl].attn_q_pad_idx != nullptr)) {
                lsa_side_ok = false;
            }
        }
        if (!lsa_side_ok) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: mixed-shape model with a cut attention layer: LLAMA_LSA_SPEC_RECALL and LLAMA_LSA_J3 off\n", __func__);
            }
        }
    }
    const bool spec_probe = lsa.spec_recall && lsa.mode != llama_lsa_config::OFF && lsa_side_ok;
    std::vector<ggml_tensor *> spec_q1(spec_probe ? n_layer : 0, nullptr), spec_q2(spec_probe ? n_layer : 0, nullptr);
    std::vector<ggml_tensor *> spec_q3(spec_probe ? n_layer : 0, nullptr);
    if (spec_probe) {
        // Match the rs3 record pick or its 16-cell refinement, per row/union. Reject other arms, never silently
        // measure a different predictor. No prefetch, slot-map write or change to the real attention is made.
        GGML_ASSERT(lsa.mode == llama_lsa_config::KEY && lsa.idx_cache && !lsa.per_head &&
                lsa.block == 128 && (lsa.refine == 128 || lsa.refine == 16) && lsa.fused_sel && !lsa.unit_op && !lsa.rec_blk &&
                !lsa.block_sum && !lsa.fast && lsa.topp <= 0.0 && lsa.pick_every == 1 && !lsa.pick_check &&
                !lsa.blk_list && !lsa.vk && !lsa.sj_compact && lsa.resident_sticky == 0.0 &&
                lsa.slice == 0 && lsa.window >= 8192 && lsa.sinks >= 128 && (!lsa.shared || lsa.shared_union));
    }

    // Query of LSA layer `target` from an earlier layer's residual: its norm, projection, q-norm and rope.
    auto stale_query = [&](ggml_tensor * resid, int target) -> ggml_tensor * {
        const auto & layer = model.layers[target];
        GGML_ASSERT(layer.wq && !layer.wqkv); // Swift's separate joint query/gate projection
        ggml_tensor * sq = build_norm(resid, layer.attn_norm, nullptr, LLM_NORM_RMS, target);
        sq = build_lora_mm(layer.wq, sq, layer.wq_s);
        if (layer.wq_b) {
            sq = ggml_add(ctx0, sq, layer.wq_b);
        }
        sq = ggml_view_3d(ctx0, sq, n_embd_head, n_head, n_tokens,
                ggml_element_size(sq)*n_embd_head*2, ggml_element_size(sq)*n_embd_head*2*n_head, 0);
        sq = build_norm(sq, layer.attn_q_norm, nullptr, LLM_NORM_RMS, target);
        return build_rope(sq, inp_pos, target); // the same rope the target layer applies to its own Q
    };

    // LLAMA_LSA_J3 (opt-in): at mapped LSA layer i, prefetch into the slots of mapped layer i+1 (top-up) and i+3
    // (look-ahead, with the target's previous-step pick) on the side stream; mapped layers before j3_first keep the
    // whole body on the card. Decode-sized single-sequence ubatches with the record pick the stale query predicts.
    std::vector<int> j3_mapped;
    // LLAMA_LSA_J3_LEAD: also several streams under LLAMA_LSA_MULTI (at most 8 rows each), built per stream
    const bool j3_rows = ubatch.n_seqs_unq == 1 ? n_tokens <= 8 : lsa.j3_lead && lsa.multi &&
            n_tokens % ubatch.n_seqs_unq == 0 && n_tokens/ubatch.n_seqs_unq <= 8;
    if (lsa.j3 && lsa.mode != llama_lsa_config::OFF && lsa_side_ok && j3_rows) {
        const bool j3_cfg = lsa.mode == llama_lsa_config::KEY && lsa.idx_cache && !lsa.per_head &&
                lsa.block == 128 && (lsa.refine == 128 || lsa.refine == 16) && lsa.fused_sel && !lsa.unit_op && !lsa.rec_blk &&
                !lsa.block_sum && !lsa.fast && lsa.topp <= 0.0 && lsa.pick_every == 1 && !lsa.pick_check &&
                !lsa.blk_list && !lsa.vk && !lsa.sj_compact && lsa.resident_sticky == 0.0 && lsa.slice == 0 &&
                (!lsa.shared || lsa.shared_union);
        if (j3_cfg) {
            for (int jl = 0; jl < n_layer; ++jl) {
                if (!hparams.is_recr(jl) && lsa.layer_enabled(jl) && llama_lsa_map_has_layer(jl)) {
                    j3_mapped.push_back(jl);
                }
            }
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: LLAMA_LSA_J3 needs the key-mode fused record pick (BLOCK 128, REFINE 16/128, FUSED_SEL, "
                        "IDX_CACHE, SHARED 0/3, no PER_HEAD/UNIT_OP/BLOCK_SUM/PICK_EVERY/VK/SJ_COMPACT/STICKY/SLICE): J3 off\n", __func__);
            }
        }
    }
    auto * j3_inp = j3_mapped.empty() ? nullptr : inp->get_attn();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        if (!hparams.mix_enabled) {
            cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_build_forward_expand(gf, cur);
        }

        lsa_spec_q = spec_probe ? spec_q1[il] : nullptr;
        lsa_spec_q2 = spec_probe ? spec_q2[il] : nullptr;
        lsa_spec_q3 = spec_probe ? spec_q3[il] : nullptr;
        lsa_spec_sel = nullptr;
        if (spec_probe && !hparams.is_recr(il) &&
                lsa.layer_enabled(il) && llama_lsa_map_has_layer(il) && n_tokens <= 8 && ubatch.n_seqs_unq == 1) {
            int next = il;
            for (int ahead = 1; ahead <= 3; ++ahead) {
                do { ++next; } while (next < n_layer && (hparams.is_recr(next) ||
                        !lsa.layer_enabled(next) || !llama_lsa_map_has_layer(next)));
                if (next == n_layer) { break; }
                ggml_tensor * sq = stale_query(inpL, next);
                (ahead == 1 ? spec_q1 : ahead == 2 ? spec_q2 : spec_q3)[next] = sq;
                cb(sq, ahead == 1 ? "lsa_spec_q" : ahead == 2 ? "lsa_spec_q2" : "lsa_spec_q3", next);
                ggml_build_forward_expand(gf, sq); // capture before this source layer runs
            }
        }

        if (j3_inp) {
            const auto it = std::find(j3_mapped.begin(), j3_mapped.end(), il);
            if (it != j3_mapped.end()) {
                const int i = (int) (it - j3_mapped.begin()), first = lsa.j3_first, n_map = (int) j3_mapped.size();
                const int protect = lsa.j3_prev ? GGML_LSA_PREFETCH_PROTECT_PREV : 0;
                const int ahead = lsa.j3_lead ? lsa.j3_ahead : 3; // LLAMA_LSA_J3_LEAD: RESIDENT_FIRST layers ahead
                // nearest target first: the side stream runs the passes in issue order
                // LLAMA_LSA_J3_LEAD with J3_A1 0 (its default): no top-up at all (it would queue behind the far passes)
                if (ahead > 1 && (!lsa.j3_lead || lsa.j3_a1 > 0) && i + 1 < n_map && i + 1 >= first) {
                    const int t = j3_mapped[i + 1];
                    const bool prev_here = lsa.j3_prev && i + 1 < ahead; // no look-ahead source for this target
                    build_attn_lsa_j3(j3_inp, lsa.j3_a1 > 0 ? stale_query(inpL, t) : nullptr, t, lsa.j3_a1,
                            protect | (prev_here ? GGML_LSA_PREFETCH_PREV : 0), 1);
                }
                if (i + ahead < n_map && i + ahead >= first) {
                    const int t = j3_mapped[i + ahead];
                    build_attn_lsa_j3(j3_inp, lsa.j3_a3 > 0 ? stale_query(inpL, t) : nullptr, t, lsa.j3_a3,
                            protect | (lsa.j3_prev ? GGML_LSA_PREFETCH_PREV : 0), 0);
                }
            }
        }

        if (hparams.mix_enabled) {
            // the recall-probe / J3 hooks above ran on this layer's input residual exactly as for a plain layer
            inpL = build_layer_mix(inp, inpL, inp_pos, inp_out_ids, il);
            continue;
        }

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// Heads of x [d, n_heads, ...] picked by idx [n_out] (I32), giving [d, n_out, ...]. ggml_get_rows picks along dim 1 of a
// 2D view, so the head dim is moved to the outside, picked, and moved back.
static ggml_tensor * qwen35_mix_gather_heads(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * idx) {
    GGML_ASSERT(x->ne[3] >= 1 && idx->type == GGML_TYPE_I32);
    const int64_t d = x->ne[0], t = x->ne[2], sq = x->ne[3], n_out = idx->ne[0];

    ggml_tensor * xp = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 3, 1, 2));          // [d, t, sq, n_heads]
    xp = ggml_reshape_2d(ctx0, xp, d * t * sq, x->ne[1]);
    xp = ggml_get_rows(ctx0, xp, idx);                                               // [d*t*sq, n_out]
    xp = ggml_reshape_4d(ctx0, xp, d, t, sq, n_out);
    return ggml_cont(ctx0, ggml_permute(ctx0, xp, 0, 2, 3, 1));                      // [d, n_out, t, sq]
}

ggml_tensor * llama_model_qwen35::graph::build_mix_read(
        ggml_tensor * h,
        ggml_tensor * norm,
        ggml_tensor * rank_read,
           uint32_t   kind,
            int32_t   span,
                int   il) {
    if (kind == llama_hparams::MIX_SPAN) {
        // y = (h D_r) * rs(h): the span's norm weight is folded into the sublayer's input projections, so the
        // read is a plain (weightless) rms scale of h followed by D_r^T
        GGML_ASSERT(norm == nullptr && (size_t) span < model.mix_span_read.size() && model.mix_span_read[span] != nullptr);
        ggml_tensor * y = build_norm(h, nullptr, nullptr, LLM_NORM_RMS, il);
        cb(y, "mix_span_rms", il);
        y = build_lora_mm(model.mix_span_read[span], y);
        cb(y, "mix_span_read", il);
        return y;
    }

    ggml_tensor * y = build_norm(h, norm, nullptr, LLM_NORM_RMS, il);
    if (kind == llama_hparams::MIX_RANK) {
        GGML_ASSERT(rank_read != nullptr);
        y = build_lora_mm(rank_read, y);
        cb(y, "mix_rank_read", il);
    }
    return y;
}

ggml_tensor * llama_model_qwen35::graph::build_mix_write(
        ggml_tensor * out,
        ggml_tensor * rank_write,
           uint32_t   kind,
            int32_t   span,
                int   il) {
    if (kind == llama_hparams::MIX_SPAN) {
        GGML_ASSERT((size_t) span < model.mix_span_write.size() && model.mix_span_write[span] != nullptr);
        out = build_lora_mm(model.mix_span_write[span], out);
        cb(out, "mix_span_write", il);
    } else if (kind == llama_hparams::MIX_RANK) {
        GGML_ASSERT(rank_write != nullptr);
        out = build_lora_mm(rank_write, out);
        cb(out, "mix_rank_write", il);
    }
    return out;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_mix(
        llm_graph_input_mem_hybrid * inp,
                      ggml_tensor * inpL,
                      ggml_tensor * inp_pos,
                      ggml_tensor * inp_out_ids,
                              int   il) {
    const auto & layer = model.layers[il];

    const uint32_t mixer_kind = hparams.mixer_kind(il);
    const uint32_t mlp_kind   = hparams.mlp_kind(il);

    ggml_tensor * h   = inpL;
    ggml_tensor * cur = nullptr;

    const bool crop = il == n_layer - 1 && crop_before_nextn(inp_out_ids);

    // ---- mixer (GDN or attention) with its residual; a removed mixer leaves the residual stream untouched
    if (mixer_kind != llama_hparams::MIX_REMOVED) {
        cur = build_mix_read(h, layer.attn_norm, layer.mixer_read, mixer_kind, hparams.mix_mixer_span[il], il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        cur = build_mix_write(cur, layer.mixer_write, mixer_kind, hparams.mix_mixer_span[il], il);

        if (crop) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            h   = ggml_get_rows(ctx0, h,   inp_out_ids);
        }

        h = ggml_add(ctx0, cur, h);
        cb(h, "attn_residual", il);
    } else if (crop) {
        h = ggml_get_rows(ctx0, h, inp_out_ids);
    }

    // ---- MLP with its residual
    if (mlp_kind != llama_hparams::MIX_REMOVED) {
        ggml_tensor * y = build_mix_read(h, layer.attn_post_norm, layer.mlp_read, mlp_kind, hparams.mix_mlp_span[il], il);
        cb(y, "attn_post_norm", il);

        cur = build_layer_ffn(y, il);
        cb(cur, "ffn_out", il);

        cur = build_mix_write(cur, layer.mlp_write, mlp_kind, hparams.mix_mlp_span[il], il);

        h = ggml_add(ctx0, cur, h);
        cb(h, "post_ffn", il);
    }

    h = build_cvec(h, il);
    cb(h, "l_out", il);

    return h;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    if (model.layers[il].wqkv_gate == nullptr) {
        // EXL3 fused [in_proj_qkv | in_proj_z]: one matmul, split by views
        const int64_t n_tok     = n_seq_tokens * n_seqs;
        const int64_t value_dim = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t conv_dim  = model.layers[il].wqkv->ne[1] - value_dim;
        GGML_ASSERT(model.layers[il].wqkv_s == nullptr && model.layers[il].wqkv_gate_s == nullptr);

        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv, input, nullptr);
        cb(qkvz, "linear_attn_qkvz", il);
        const size_t es = ggml_element_size(qkvz);

        ggml_tensor * qkv_mixed = ggml_view_2d(ctx0, qkvz, conv_dim, n_tok, qkvz->nb[1], 0);
        qkv_mixed = ggml_cont_3d(ctx0, qkv_mixed, conv_dim, n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, value_dim, n_tok, qkvz->nb[1], conv_dim * es);
        z = ggml_cont(ctx0, z);
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

// Dense SwiGLU FFN of a layer: separate gate/up (silu(gate)*up), or (EXL3 fused layout) one ffn_up = [gate | up]
// matmul followed by ggml_swiglu (silu of the first half times the second half).
static ggml_tensor * qwen35_build_dense_ffn(llm_graph_context & g, const llama_layer & layer, ggml_tensor * cur, int il) {
    if (layer.ffn_gate == nullptr) {
        GGML_ASSERT(layer.ffn_up_s == nullptr && layer.ffn_gate_s == nullptr);
        return g.build_ffn(cur,
            layer.ffn_up,   nullptr, nullptr,
            nullptr,        nullptr, nullptr,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
    }
    return g.build_ffn(cur,
        layer.ffn_up,   nullptr, layer.ffn_up_s,
        layer.ffn_gate, nullptr, layer.ffn_gate_s,
        layer.ffn_down, nullptr, layer.ffn_down_s,
        nullptr,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
}

// Q(+gate)/K/V projections for a row range that is not the whole ubatch (the batched MTP draft block), where
// llm_graph_context::build_qkv cannot be used because it views n_tokens rows. Separate wq/wk/wv, or (EXL3 fused
// layout) one wqkv matmul [attn_q(with gate) | attn_k | attn_v] split by views. Qfull is [(n_embd_head*2)*n_head, n_tok]
// (row stride in nb[1]), K/V are [n_embd_head, n_head_kv, n_tok] views.
static void qwen35_build_qkv_rows(
        llm_graph_context & g,
        const llama_layer & layer,
        ggml_tensor *       cur,
        int64_t             n_tok,
        ggml_tensor *&      Qfull,
        ggml_tensor *&      K,
        ggml_tensor *&      V,
        int                 il) {
    ggml_context * ctx0 = g.ctx0;
    const int64_t n_head      = g.n_head;
    const int64_t n_head_kv   = g.n_head_kv;
    const int64_t n_embd_head = g.hparams.n_embd_head_v();

    if (layer.wqkv != nullptr) {
        GGML_ASSERT(layer.wqkv_s == nullptr && layer.wqkv_b == nullptr);
        const int64_t n_q = n_embd_head * 2 * n_head;
        const int64_t n_k = n_embd_head * n_head_kv;
        const int64_t n_v = n_embd_head * n_head_kv;
        GGML_ASSERT(layer.wqkv->ne[1] == n_q + n_k + n_v);

        ggml_tensor * qkv = g.build_lora_mm(layer.wqkv, cur, nullptr);
        g.cb(qkv, "qkv_fused", il);
        const size_t es = ggml_element_size(qkv);

        Qfull = ggml_view_2d(ctx0, qkv, n_q, n_tok, qkv->nb[1], 0);
        K     = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tok, es * n_embd_head, qkv->nb[1], es * n_q);
        V     = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tok, es * n_embd_head, qkv->nb[1], es * (n_q + n_k));
        return;
    }

    Qfull = g.build_lora_mm(layer.wq, cur, layer.wq_s);
    K = g.build_lora_mm(layer.wk, cur, layer.wk_s);
    K = ggml_reshape_3d(ctx0, K, n_embd_head, n_head_kv, n_tok);
    V = g.build_lora_mm(layer.wv, cur, layer.wv_s);
    V = ggml_reshape_3d(ctx0, V, n_embd_head, n_head_kv, n_tok);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t dk = hparams.mix_enabled ? hparams.mix_attn_k_dim[il] : hparams.n_embd_head_k(il);
    const int64_t dv = hparams.mix_enabled ? hparams.mix_attn_v_dim[il] : hparams.n_embd_head_v(il);
    const int64_t cache_dim = hparams.n_embd_head_k(il);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    // mixed-shape pruned models cut query heads per layer (KV heads are never cut)
    const int64_t n_head_l    = hparams.mix_enabled ? hparams.n_head(il)    : n_head;
    const int64_t n_head_kv_l = hparams.mix_enabled ? hparams.n_head_kv(il) : n_head_kv;

    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            dk + dv, n_head_l,
            dk,             n_head_kv_l,
            dv,             n_head_kv_l,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, dk, n_head_l, n_tokens,
        ggml_element_size(Qcur_full) * (dk + dv),
        ggml_element_size(Qcur_full) * (dk + dv) * n_head_l, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, dk, n_head_kv_l, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, dv, n_head_l, n_tokens,
        ggml_element_size(Qcur_full) * (dk + dv),
        ggml_element_size(Qcur_full) * (dk + dv) * n_head_l,
        ggml_element_size(Qcur_full) * dk);
    gate = ggml_cont_2d(ctx0, gate, dv * n_head_l, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, dv, n_head_kv_l, n_tokens);

    // Apply MRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Uneven GQA groups (a mixed-shape model whose kept query heads are not the same count for every KV group): attention
    // broadcasts KV head g over query heads [g*m, (g+1)*m), so every group is padded to the largest group size m. The
    // padded slots repeat a real head of their group and their outputs are dropped again after attention.
    const bool padded_q = model.layers[il].attn_q_pad_idx != nullptr;
    GGML_ASSERT(padded_q || n_head_l % n_head_kv_l == 0);
    if (padded_q) {
        GGML_ASSERT(model.layers[il].attn_q_inv_idx != nullptr);
        Qcur = qwen35_mix_gather_heads(ctx0, ggml_reshape_4d(ctx0, Qcur, dk, n_head_l, n_tokens, 1), model.layers[il].attn_q_pad_idx);
        Qcur = ggml_reshape_3d(ctx0, Qcur, dk, Qcur->ne[1], n_tokens);
        cb(Qcur, "Qcur_padded", il);
    }

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(dk)) : hparams.f_attention_scale;

    if (dk != cache_dim) {
        Qcur = ggml_pad(ctx0, Qcur, cache_dim - dk, 0, 0, 0);
        Kcur = ggml_pad(ctx0, Kcur, cache_dim - dk, 0, 0, 0);
    }
    if (dv != cache_dim) {
        Vcur = ggml_pad(ctx0, Vcur, cache_dim - dv, 0, 0, 0);
    }

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    if (dv != cache_dim) {
        cur = ggml_reshape_3d(ctx0, cur, cache_dim, cur->ne[0] / cache_dim, n_tokens);
        cur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, dv, cur->ne[1], n_tokens, cur->nb[1], cur->nb[2], 0));
        cur = ggml_reshape_2d(ctx0, cur, dv * cur->ne[1], n_tokens);
    }

    if (padded_q) {
        cur = ggml_reshape_4d(ctx0, cur, dv, cur->ne[0] / dv, n_tokens, 1);
        cur = qwen35_mix_gather_heads(ctx0, cur, model.layers[il].attn_q_inv_idx);
        cur = ggml_reshape_2d(ctx0, cur, dv * n_head_l, n_tokens);
        cb(cur, "attn_unpadded", il);
    }

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    // a mixed-shape pruned model cuts value heads per layer (the key heads are never cut)
    const int64_t d_inner      = hparams.ssm_d_inner_l(il);
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_k_dim_l(il);
    const int64_t num_k_heads  = hparams.ssm_k_heads_l(il);
    const int64_t num_v_heads  = hparams.mix_enabled ? hparams.mix_ssm_v_heads[il] : hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t state_dim    = hparams.ssm_state_dim_l(il);
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha_cat) {
        ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);

        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    if (!model.layers[il].ssm_beta_alpha_cat) {   // separate tensors: same node order as before the v1.0 merge
        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
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
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    // [GDNGETTER] LLAMA_GDN_STATE_GETTER=1: own GET_ROWS(cache, s_copy_main) for the ubatch states, so the CUDA #87
    // plan can let the gdn read the cache rows in place (see qwen4exp.cpp). Same rows, same bytes; default off.
    static const bool gdn_getter = [] { const char * e = getenv("LLAMA_GDN_STATE_GETTER"); return e != nullptr && atoi(e) != 0; }();
    ggml_tensor * state = gdn_getter ?
        build_rs(inp, ssm_states_all, hparams.n_embd_s(il), n_seqs,
                 [](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) { return ggml_get_rows(ctx, states, ids); }) :
        build_rs(inp, ssm_states_all, hparams.n_embd_s(il), n_seqs);
    state = ggml_reshape_4d(ctx0, state, state_dim, state_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

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

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // a layer that lost value heads keeps an explicit value head -> key head map (the surviving heads no longer follow the
    // tiled pattern): pick each kept head's key head here, after the per-key-head L2 norm, so the delta rule sees one
    // key head per value head
    bool keys_per_value = false;
    if (model.layers[il].ssm_key_idx != nullptr) {
        q_conv = qwen35_mix_gather_heads(ctx0, q_conv, model.layers[il].ssm_key_idx);
        k_conv = qwen35_mix_gather_heads(ctx0, k_conv, model.layers[il].ssm_key_idx);
        keys_per_value = true;
    }

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (!keys_per_value && num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    if (head_k_dim != state_dim) {
        q_conv = ggml_pad(ctx0, q_conv, state_dim - head_k_dim, 0, 0, 0);
        k_conv = ggml_pad(ctx0, k_conv, state_dim - head_k_dim, 0, 0, 0);
        q_conv = ggml_scale(ctx0, q_conv, sqrtf(float(state_dim) / head_k_dim));
    }
    if (head_v_dim != state_dim) {
        v_conv = ggml_pad(ctx0, v_conv, state_dim - head_v_dim, 0, 0, 0);
    }

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    if (head_v_dim != state_dim) {
        output = ggml_cont(ctx0, ggml_view_4d(ctx0, output, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            output->nb[1], output->nb[2], output->nb[3], 0));
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    // z is a strided view of the merged qkv_gate output with wqkv_z (upstream layout), contiguous otherwise (graph as before)
    ggml_tensor * z_2d = model.layers[il].wqkv_z
        ? ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0)
        : ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    ggml_tensor * inp = cur;

    cur = qwen35_build_dense_ffn(*this, model.layers[il], inp, il);

    // Run the parallel FFN on the same input as the main FFN.
    if (model.layers[il].ffn_up_par != nullptr) {
        ggml_tensor * par = build_ffn(inp,
            model.layers[il].ffn_up_par, NULL, NULL,
            model.layers[il].ffn_gate_par, NULL, NULL,
            model.layers[il].ffn_down_par, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(par, "ffn_par_out", il);
        cur = ggml_add(ctx0, cur, par);
    }

    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    const int64_t n_embd_head = hparams.n_embd_head_v(il);
    const int64_t n_head = hparams.n_head(il);
    const int64_t n_head_kv = hparams.n_head_kv(il);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(il));

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
        tok_embd = build_hadamard_inverse(tok_embd_w, tok_embd);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    // [#91] a decode without output rows (the MTP catch-up update) only has to store the MTP layer's K/V;
    // its query, attention, FFN and head are never read. On by default; LLAMA_MTP_UPDATE_KV_ONLY=0 builds the full graph.
    // An unmasked nextn-embedding read copies every row even without outputs, so it keeps the full graph.
    static const bool update_kv_only = [] {
        const char * value = getenv("LLAMA_MTP_UPDATE_KV_ONLY");
        return value == nullptr || strcmp(value, "0") != 0;
    }();
    const bool kv_only = update_kv_only && n_outputs == 0 && !cparams.mtp_chain && !cparams.embeddings &&
        (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    if (kv_only) {
        static std::atomic_flag noted = ATOMIC_FLAG_INIT;
        if (!noted.test_and_set()) {
            LLAMA_LOG_INFO("%s: MTP update builds the KV-only graph (LLAMA_MTP_UPDATE_KV_ONLY=0 disables)\n", __func__);
        }
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = kv_only ? nullptr : qwen35_build_inp_out_ids(*this);

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // chained drafting: rows 1..n-1 take their inputs from the previous row's
    // in-graph argmax and hidden state, so one decode drafts n_tokens tokens
    if (params.cparams.mtp_chain && ubatch.n_seqs_unq == 1 && ubatch.token) {
        const auto * mctx_kv = static_cast<const llama_kv_cache_context *>(mctx);

        // per-step causal masks: row j of the stock kq mask (the mask builder pads
        // masks to exactly n_tokens rows, so plain row views work)
        GGML_ASSERT(inp_attn->self_attn_mask.rows == nullptr && "the MTP chain needs the plain [n_kv, n_tokens] mask");
        ggml_tensor * kq_mask = inp_attn->self_attn_mask.raw;
        const int64_t n_kv = kq_mask->ne[0];

        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        ggml_tensor * k_idxs = inp_attn->get_k_idxs();
        ggml_tensor * v_idxs = inp_attn->get_v_idxs();

        // normalize and project the token/hidden inputs into the block input
        // (norm -> eh_proj); rows are independent through this projection
        auto build_proj = [&](ggml_tensor * tok_in, ggml_tensor * h_in) -> ggml_tensor * {
            ggml_tensor * h_norm_b = build_norm(h_in, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * e_norm_b = build_norm(tok_in, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);

            return build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm_b, h_norm_b, 0), layer.nextn.eh_proj_s);
        };

        // one MTP block over `width` projected rows starting at batch row `row0`;
        // returns the post-ffn hidden state. The K/V stores are expanded, so a
        // caller that only needs the rows' cache side effects can ignore the result.
        auto build_block = [&](ggml_tensor * cur_in, int64_t row0, int64_t width) -> ggml_tensor * {
            ggml_tensor * cur_b = cur_in;

            ggml_tensor * inpSA_b = cur_b;

            cur_b = build_norm(cur_b, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * Qfull_b = nullptr;
            ggml_tensor * K_b     = nullptr;
            ggml_tensor * V_b     = nullptr;
            qwen35_build_qkv_rows(*this, layer, cur_b, width, Qfull_b, K_b, V_b, il);

            ggml_tensor * Q_b = ggml_view_3d(ctx0, Qfull_b,
                    n_embd_head, n_head, width,
                    ggml_element_size(Qfull_b) * n_embd_head * 2,
                    Qfull_b->nb[1],
                    0);
            Q_b = build_norm(Q_b, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * gate_b = ggml_view_3d(ctx0, Qfull_b,
                    n_embd_head, n_head, width,
                    ggml_element_size(Qfull_b) * n_embd_head * 2,
                    Qfull_b->nb[1],
                    ggml_element_size(Qfull_b) * n_embd_head);
            gate_b = ggml_cont_2d(ctx0, gate_b, n_embd_head * n_head, width);

            K_b = build_norm(K_b, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

            // M-RoPE positions are section-major: [dim0 x n_tokens, dim1 x n_tokens, ...]
            ggml_tensor * pos_b = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_pos, width, 4,
                    (size_t) n_tokens*inp_pos->nb[0], (size_t) row0*inp_pos->nb[0]));
            pos_b = ggml_reshape_1d(ctx0, pos_b, 4*width);

            // the same rope as the single-row path (build_rope: rope_multi with the M-RoPE sections)
            Q_b = build_rope(Q_b, pos_b, il);
            K_b = build_rope(K_b, pos_b, il);

            // attention below reads the cache through build_attn_mha directly, so it applies the
            // same rotations build_attn(llm_graph_input_attn_kv *) does, in the same order
            if (inp_attn->self_k_rot[il]) {
                Q_b = llama_mul_mat_hadamard(ctx0, Q_b, inp_attn->self_k_rot[il]);
                K_b = llama_mul_mat_hadamard(ctx0, K_b, inp_attn->self_k_rot[il]);
            }

            if (inp_attn->self_v_rot[il]) {
                V_b = llama_mul_mat_hadamard(ctx0, V_b, inp_attn->self_v_rot[il]);
            }

            ggml_build_forward_expand(gf, Q_b);
            ggml_build_forward_expand(gf, V_b);
            ggml_build_forward_expand(gf, K_b);

            // store these rows' K/V so later rows attend to them through the cache
            ggml_tensor * k_idx_b = ggml_view_1d(ctx0, k_idxs, width, row0*k_idxs->nb[0]);
            // the V cache is transposed without flash attention, making v_idxs
            // per-head (n_embd_gqa rows per token, contiguous); k_idxs is always
            // per-token. scale the view to the actual per-token stride.
            const int64_t v_stride = v_idxs->ne[0] / n_tokens;
            ggml_tensor * v_idx_b = ggml_view_1d(ctx0, v_idxs, width*v_stride, row0*v_stride*v_idxs->nb[0]);
            ggml_build_forward_expand(gf, mctx_kv->cpy_k(ctx0, K_b, k_idx_b, il));
            ggml_build_forward_expand(gf, mctx_kv->cpy_v(ctx0, V_b, v_idx_b, il));

            ggml_tensor * k_view = mctx_kv->get_k(ctx0, il);
            ggml_tensor * v_view = mctx_kv->get_v(ctx0, il);

            ggml_tensor * mask_b = ggml_view_2d(ctx0, kq_mask, n_kv, width, kq_mask->nb[1], (size_t) row0*kq_mask->nb[1]);

            // TurboQuant/TQ K caches hold WHT-rotated K: Q must be rotated to match
            Q_b = build_attn_turbo_q(Q_b, k_view, mctx_kv);

            cur_b = build_attn_mha(Q_b, k_view, v_view, nullptr, mask_b, nullptr /* kv_rows */, nullptr, nullptr, 0, kq_scale, il);

            cur_b = build_attn_turbo_v_out(cur_b, v_view, il);

            if (inp_attn->self_v_rot[il]) {
                cur_b = llama_mul_mat_hadamard(ctx0, cur_b, inp_attn->self_v_rot[il]);
            }

            cur_b = ggml_mul(ctx0, cur_b, ggml_sigmoid(ctx0, gate_b));
            cur_b = build_lora_mm(layer.wo, cur_b, layer.wo_s);

            cur_b = ggml_add(ctx0, cur_b, inpSA_b);

            ggml_tensor * ffn_res_b = cur_b;
            cur_b = build_norm(cur_b, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);

            cur_b = qwen35_build_dense_ffn(*this, layer, cur_b, il);

            cur_b = ggml_add(ctx0, cur_b, ffn_res_b);

            return cur_b;
        };

        // leading rows without outputs are deferred catch-up rows carrying real batch
        // inputs; they only contribute their K/V. The chain starts at the first output row.
        int64_t n_catchup = 0;
        while (n_catchup < n_tokens && ubatch.output && ubatch.output[n_catchup] == 0) {
            n_catchup++;
        }
        const int64_t n_chain = n_tokens - n_catchup;
        GGML_ASSERT(n_chain >= 1);

        // project the whole ubatch in one pass, exactly like the stock path: the
        // host-buffer inputs must reach split ops as full leaf tensors, not views.
        // Slices of the projected result are sched-allocated and safe to split.
        ggml_tensor * proj_all = build_proj(tok_embd, h_embd);

        if (n_catchup > 0) {
            ggml_tensor * proj_c = ggml_view_2d(ctx0, proj_all, proj_all->ne[0], n_catchup, proj_all->nb[1], 0);
            build_block(proj_c, 0, n_catchup);
        }

        ggml_tensor * proj_cur = ggml_view_2d(ctx0, proj_all, proj_all->ne[0], 1, proj_all->nb[1], (size_t) n_catchup*proj_all->nb[1]);

        ggml_tensor * logits_all = nullptr;
        ggml_tensor * h_all      = nullptr;

        // [#69] sampled chain: each step draws in-graph over its top chain_k candidates against a
        // host-drawn uniform (llama-mtp-chain-sample.h); 0 keeps the argmax chain
        const int32_t chain_k = params.cparams.mtp_chain_top_k;
        ggml_tensor * chain_samp = nullptr;
        ggml_tensor * chain_u    = nullptr;
        if (chain_k > 0) {
            auto inp_s = std::make_unique<llm_graph_input_mtp_chain_samp>(mtp_chain_samp);
            inp_s->samp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, LLAMA_MTP_CHAIN_SAMP_N);
            ggml_set_input(inp_s->samp);
            ggml_set_name(inp_s->samp, "mtp_chain_samp");
            inp_s->u = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_chain);
            ggml_set_input(inp_s->u);
            ggml_set_name(inp_s->u, "mtp_chain_u");
            chain_samp = inp_s->samp;
            chain_u    = inp_s->u;
            res->add_input(std::move(inp_s));
        }

        for (int64_t j = 0; j < n_chain; ++j) {
            ggml_tensor * cur_j = build_block(proj_cur, n_catchup + j, 1);

            ggml_tensor * head_norm_w2 = layer.nextn.shared_head_norm
                    ? layer.nextn.shared_head_norm
                    : model.output_norm;
            ggml_tensor * h_next_j = build_norm(cur_j, head_norm_w2, nullptr, LLM_NORM_RMS, -1);

            ggml_tensor * head_w2 = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
            ggml_tensor * head_s2 = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;

            static const int64_t n_sub_env = [] {
                const char * env = getenv("LLAMA_SPEC_CHAIN_SUB");
                return env != nullptr ? atoll(env) : 32768;
            }();

            // [BL7c] a loaded draft vocabulary shortlist replaces the leading-rows cut: logit positions are
            // then map rows, translated back to token ids through draft_vocab_ids
            ggml_tensor * id_map   = nullptr;
            ggml_tensor * logits_j = build_draft_vocab_logits_chain(head_w2, head_s2, h_next_j);
            // an EXL3 head is K-major tiled and needs the suh/svh glue (build_lora_mm): a leading-rows view of it
            // is not its first rows, so it scores the full head as with LLAMA_SPEC_CHAIN_SUB=0
            const bool sub_exl3 = ggml_exl3_bits(head_w2->type) != 0;
            if (logits_j != nullptr) {
                id_map = draft_vocab_ids;
            } else if (n_sub_env > 0 && n_sub_env < head_w2->ne[1] &&
                    !(hadamard_rotations && hadamard_rotations->count(head_w2)) && !sub_exl3) {
                ggml_tensor * head_sub = ggml_view_2d(ctx0, head_w2,
                        head_w2->ne[0], n_sub_env, head_w2->nb[1], 0);
                logits_j = ggml_mul_mat(ctx0, head_sub, h_next_j);
                if (head_s2 != nullptr) {
                    logits_j = ggml_mul(ctx0, logits_j, head_s2);
                }
            } else {
                static bool logged_sub_exl3 = false;
                if (sub_exl3 && n_sub_env > 0 && n_sub_env < head_w2->ne[1] && !logged_sub_exl3) {
                    logged_sub_exl3 = true;
                    LLAMA_LOG_INFO("%s: MTP chain: the output head is EXL3, the %lld-row sub-head cut does not apply; "
                                   "the chain scores the full head (logged once)\n", __func__, (long long) n_sub_env);
                }
                logits_j = build_lora_mm(head_w2, h_next_j, head_s2);
            }

            ggml_tensor * id_j;
            ggml_tensor * out_j;
            if (chain_k > 0) {
                ggml_tensor * u_j = ggml_view_1d(ctx0, chain_u, 1, (size_t) j*chain_u->nb[0]);
                out_j = llama_mtp_chain_sample_graph(ctx0, logits_j, chain_k, chain_samp, u_j, id_map, &id_j);
            } else {
                ggml_tensor * pos_j = ggml_argmax(ctx0, logits_j);
                ggml_tensor * probs_j = ggml_soft_max(ctx0, logits_j);
                ggml_tensor * p_j = ggml_get_rows(ctx0,
                        ggml_reshape_2d(ctx0, probs_j, 1, probs_j->ne[0]), pos_j);
                p_j = ggml_reshape_2d(ctx0, p_j, 1, 1);
                id_j = id_map == nullptr ? pos_j
                    : ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, id_map, 1, id_map->ne[0]), pos_j), 1);
                ggml_tensor * id_f = ggml_cast(ctx0, ggml_reshape_2d(ctx0, id_j, 1, 1), GGML_TYPE_F32);
                out_j = ggml_concat(ctx0, id_f, p_j, 0);
            }

            logits_all = logits_all == nullptr ? out_j : ggml_concat(ctx0, logits_all, out_j, 1);
            h_all      = h_all      == nullptr ? h_next_j : ggml_concat(ctx0, h_all, h_next_j, 1);

            if (j + 1 < n_chain) {
                ggml_tensor * tok_j = ggml_get_rows(ctx0, tok_embd_w, id_j);
                tok_j = build_hadamard_inverse(tok_embd_w, tok_j);
                proj_cur = build_proj(tok_j, h_next_j);
            }
        }

        cb(h_all, "h_nextn", -1);
        res->t_h_nextn = h_all;
        ggml_build_forward_expand(gf, h_all);

        if (inp_out_ids) {
            ggml_build_forward_expand(gf, ggml_view_1d(ctx0, inp_out_ids, inp_out_ids->ne[0], 0));
        }

        cb(logits_all, "result_output", -1);
        res->t_logits = logits_all;
        ggml_build_forward_expand(gf, logits_all);

        return;
    }

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    if (kv_only) {
        // the Q/gate nodes built above are not expanded, so they drop out of the graph
        build_attn_store(inp_attn, Kcur, Vcur, il);
        return;
    }

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = qwen35_build_dense_ffn(*this, layer, cur, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    // fork: Masked MTP carries must select the draft rows, not preceding catch-up rows.
    if (cparams.embeddings_nextn_masked) {
        res->t_h_nextn = cur;
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    // draft-only vocabulary shortlist (when the context carries a map): score only the
    // shortlisted head rows; the backend sampler maps its picks back to token ids
    ggml_tensor * logits = build_draft_vocab_logits(head_w, head_s, cur);
    cur = logits ? logits : build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
