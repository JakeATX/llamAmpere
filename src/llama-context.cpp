#include "llama-context.h"
#include "llama-moe-arena.h"
#include "llama-sampler.h"

// minimum number of output rows reserved at context creation (covers MTP verification widths)
#define LLAMA_OUTPUT_RESERVE_MIN 8

#include "../ggml/src/ggml-backend-moe-cache.h"
#include "ggml.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache.h"
#include "llama-kv-stream-config.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-mtp-vocab.h"
#include "llama-mtp-vocab-builtin.h"
#include "llama-draft-vocab-compact.h"
#include "llama-mtp-chain-sample.h"
#include "llama-moe-cache.h"
#include "llama-b1.h"
#include "llama-ple-prefetch.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

//
// llama_context
//

static llama_kv_cache * llama_sj_kvarn_cache(llama_memory_i * memory);

// Diagnostics only: these process-wide buckets aggregate default target and MTP contexts.
// Host times are inclusive and nested in server phases (and release includes synchronize).
// Do not add these diagnostics together or add them to native round "other" time.
static const bool llama_bookkeep_timing_enabled = [] {
    const char * e = std::getenv("LLAMA_BOOKKEEP_TIMING");
    return e && std::atoi(e) != 0;
}();

struct llama_bookkeep_context_counters {
    std::atomic<uint64_t> sync_calls{0};
    std::atomic<uint64_t> sync_host_us{0};
    std::atomic<uint64_t> copy_calls{0};
    std::atomic<uint64_t> copy_bytes{0};
    std::atomic<uint64_t> copy_host_us{0};
    std::atomic<uint64_t> release_calls{0};
    std::atomic<uint64_t> release_host_us{0};
    std::atomic<uint64_t> memory_init_layout_host_us{0};
    std::atomic<uint64_t> graph_prepare_host_us{0};
    std::atomic<uint64_t> set_inputs_host_us{0};
    std::atomic<uint64_t> peer_sync_host_us{0};
    std::atomic<uint64_t> graph_submit_host_us{0};
    std::atomic<uint64_t> graph_rebuild_calls{0};
    std::atomic<uint64_t> graph_reuse_calls{0};
    std::array<std::atomic<uint64_t>, 10> backend_submit{};
};

static_assert(std::atomic<uint64_t>::is_always_lock_free, "bookkeeping counters must be lock-free");
static llama_bookkeep_context_counters llama_bookkeep_context_totals[2];

static llama_bookkeep_context_counters * llama_bookkeep_context_bucket(llama_context_type type) {
    if (!llama_bookkeep_timing_enabled) {
        return nullptr;
    }
    return &llama_bookkeep_context_totals[type == LLAMA_CONTEXT_TYPE_MTP ? 1 : 0];
}

struct llama_bookkeep_context_timer {
    std::atomic<uint64_t> * calls;
    std::atomic<uint64_t> * host_us;
    const int64_t started;

    llama_bookkeep_context_timer(std::atomic<uint64_t> * calls, std::atomic<uint64_t> * host_us) :
        calls(calls), host_us(host_us), started(calls ? ggml_time_us() : 0) {}

    ~llama_bookkeep_context_timer() {
        if (calls) {
            host_us->fetch_add(uint64_t(ggml_time_us() - started), std::memory_order_relaxed);
            calls->fetch_add(1, std::memory_order_relaxed);
        }
    }
};

// One active stage at a time; changing stages closes the old interval. Keeping this timer
// outside build_graph lets retries remain in graph preparation without overlapping stages.
struct llama_bookkeep_stage_timer {
    std::atomic<uint64_t> * host_us;
    int64_t started;

    explicit llama_bookkeep_stage_timer(std::atomic<uint64_t> * host_us) :
        host_us(host_us), started(host_us ? ggml_time_us() : 0) {}

    void next(std::atomic<uint64_t> * next_host_us) {
        if (host_us == next_host_us) {
            return;
        }
        const int64_t now = ggml_time_us();
        if (host_us) {
            host_us->fetch_add(uint64_t(now - started), std::memory_order_relaxed);
        }
        host_us = next_host_us;
        started = now;
    }

    ~llama_bookkeep_stage_timer() {
        next(nullptr);
    }
};

// Verify that every Hadamard-folded weight consumed by the graph receives its
// activation-side transform, and every latent lookup table gets the inverse.
// An architecture whose matmul path bypasses the transform helpers would
// otherwise load cleanly and silently compute wrong results.
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses) {
    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(unwrap(node->src[0]))) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                const auto & transform = inverses.at(unwrap(lk->first->src[0]));
                lk->second = node->src[0] == transform.rot;
            }
            continue;
        }

        const auto it = rotations.find(unwrap(node->src[0]));
        if (it == rotations.end()) {
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t     n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/"Flash Attention",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/LLM_FUSED_OP_GDN_AR,
    /*.name             =*/"fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/LLM_FUSED_OP_GDN_CH,
    /*.name             =*/"fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/"Lightning Indexer",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/"fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/"fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/"fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/1,
};

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT:
            return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP:
            return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

// fork: free allocator pages between B1 layers; live buffers remain allocated.
static void b1_memory_trim(const char * phase) {
#if defined(__GLIBC__)
    const int saved_errno = errno;
    const char * enabled = std::getenv("LLAMA_B1_MALLOC_TRIM");
    if (!enabled || strcmp(enabled, "1") != 0) { errno = saved_errno; return; }
#if __GLIBC_PREREQ(2, 33)
    const auto before = mallinfo2();
#endif
    const int64_t started = ggml_time_us();
    const int released = malloc_trim(0);
    const int64_t elapsed = ggml_time_us() - started;
#if __GLIBC_PREREQ(2, 33)
    const auto after = mallinfo2();
    LLAMA_LOG_INFO("%s: phase=%s released=%d time_us=%" PRId64 " arena_before=%zu arena_after=%zu used_before=%zu used_after=%zu free_before=%zu free_after=%zu mmap_before=%zu mmap_after=%zu\n",
            __func__, phase, released, elapsed, before.arena, after.arena, before.uordblks, after.uordblks,
            before.fordblks, after.fordblks, before.hblkhd, after.hblkhd);
#else
    LLAMA_LOG_INFO("%s: phase=%s released=%d time_us=%" PRId64 "\n", __func__, phase, released, elapsed);
#endif
    errno = saved_errno;
#else
    (void) phase;
#endif
}

llama_context::llama_context(const llama_model & model, llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    // Research control: restore the original output allocation policy for paired measurements.
    if (const char * value = std::getenv("LLAMA_OUTPUT_BUFFER_REUSE")) {
        output_buffer_reuse = std::strcmp(value, "0") != 0;
    }
    LLAMA_LOG_INFO("%s: output buffer reuse = %d\n", __func__, (int) output_buffer_reuse);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max     = std::max(1u, params.n_seq_max);
    cparams.n_outputs_max = params.n_outputs_max;
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }
    cparams.gdn_replay = params.gdn_replay;

    cparams.n_threads            = params.n_threads;
    cparams.n_threads_batch      = params.n_threads_batch;
    if (params.moe_prefetch < -1 || params.moe_prefetch > 3) {
        throw std::invalid_argument("moe_prefetch must be -1..3");
    }
    cparams.moe_prefetch = params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT ? params.moe_prefetch : 0;
    cparams.moe_cache_mode       = params.moe_cache_mode;
    cparams.moe_cache_budget_mib = params.moe_cache_budget_mib;
    cparams.yarn_ext_factor      = params.yarn_ext_factor >= 0.0f ? params.yarn_ext_factor : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor     = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast       = params.yarn_beta_fast >= 0.0f ? params.yarn_beta_fast : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow       = params.yarn_beta_slow >= 0.0f ? params.yarn_beta_slow : hparams.yarn_beta_slow;
    cparams.embeddings           = params.embeddings;
    cparams.embeddings_nextn     = false;
    cparams.embeddings_nextn_masked = false;
    cparams.mtp_chain               = false;
    cparams.mtp_chain_top_k         = 0;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.kv_stream_arena_mib     = params.kv_stream_arena_mib;
    cparams.draft_attn_window       = params.draft_attn_window > 0 ? (uint32_t) params.draft_attn_window : 0;
    cparams.draft_attn_sink         = params.draft_attn_sink   > 0 ? (uint32_t) params.draft_attn_sink   : 0;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx           = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
    cparams.rope_freq_base  = params.rope_freq_base == 0.0f ? hparams.rope_freq_base_train : params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn = params.yarn_orig_ctx != 0    ? params.yarn_orig_ctx :
                              hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                             hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error(
                "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(
                    model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    // draft-only vocabulary shortlist: parsed once, kept as a persistent I32 map on the head's device
    {
        const char * path = params.draft_vocab_map;
        // the environment override also replaces common's default ("auto"), which a caller cannot tell from an explicit auto
        if ((path == nullptr || path[0] == '\0' || std::strcmp(path, "auto") == 0) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
            if (const char * env = getenv("LLAMA_SPEC_DRAFT_VOCAB"); env != nullptr && env[0] != '\0') {
                path = env;
            }
        }
        if (path != nullptr && std::strcmp(path, "none") == 0) {
            path = nullptr;
        }
        if (path != nullptr && path[0] != '\0') {
            if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
                // the head is sharded across the Meta device and backend sampling is unavailable there:
                // the draft scores the full head, so the map tensor is not created (see draft_vocab_direct)
                LLAMA_LOG_WARN("%s: --spec-draft-vocab-map is not used with SPLIT_MODE_TENSOR, the draft scores the full head\n", __func__);
            } else {
                init_draft_vocab(path, params.draft_vocab_hot);
            }
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
            float       mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                           __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.fused_lid  = true;
    cparams.auto_flid  = true;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    if (cparams.n_rs_seq > 0) {
        const uint32_t n_batch_min = cparams.n_rs_seq + 2;
        if (cparams.n_ctx < n_batch_min) {
            throw std::runtime_error(
                format("n_ctx (%u) must be at least n_rs_seq + 2 (%u)", cparams.n_ctx, n_batch_min));
        }
        if (cparams.n_batch < n_batch_min) {
            LLAMA_LOG_WARN("%s: n_batch (%u) is too small for n_rs_seq=%u; increasing to %u\n", __func__,
                           cparams.n_batch, cparams.n_rs_seq, n_batch_min);
            cparams.n_batch = n_batch_min;
        }
        if (cparams.n_ubatch < n_batch_min) {
            LLAMA_LOG_WARN("%s: n_ubatch (%u) is too small for n_rs_seq=%u; increasing to %u\n", __func__,
                           cparams.n_ubatch, cparams.n_rs_seq, n_batch_min);
            cparams.n_ubatch = n_batch_min;
        }
    }

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;

    // SJ-KVaRN keeps one sink/ring stream per sequence in every mode; --kv-unified shares the sealed-record
    // pool between them instead (every sequence may address the whole context), so the rest of the context
    // runs the per-sequence-stream path: unified off, n_ctx_seq = n_ctx
    const bool sj_kvarn_shared_pool = params.sj_kvarn_bits_k > 0 && params.sj_kvarn_bits_v > 0 &&
        params.kv_unified && params.n_seq_max > 1;
    if (sj_kvarn_shared_pool) {
        cparams.kv_unified = false;
        LLAMA_LOG_INFO("%s: SJ-KVaRN with kv_unified: per-sequence streams over one shared record pool\n", __func__);
    }

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified || sj_kvarn_shared_pool) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx = cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                       __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", __func__,
                       cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn =
                    (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            // reserve room for speculative verification batches up front: growing the pinned output
            // buffer later costs a cudaFreeHost/cudaMallocHost pair (~60 ms) in the middle of decoding
            const int32_t n_outputs_init = std::min<int32_t>(std::max<int32_t>(params.n_seq_max, LLAMA_OUTPUT_RESERVE_MIN), cparams.n_outputs_max);
            if (output_reserve(n_outputs_init) < (uint32_t) n_outputs_init) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                           ggml_backend_buffer_name(buf_output.get()),
                           ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        const uint64_t kv_stream_arena_bytes         = uint64_t(cparams.kv_stream_arena_mib) * 1024ULL * 1024ULL;
        uint64_t       kv_stream_stage_bytes         = kv_stream_arena_bytes;
        uint64_t       kv_stream_minimum_stage_bytes = 0;

        // TurboQuant: this fork's MLA/indexer caches (DSA/DSV4/MSA) and the
        // recurrent memory implementation compress or index the KV store in
        // ways block KV streaming's uniform-page-size pool does not model -
        // same exclusion set as this fork's now-removed PR #326 attempt.
        // DFLASH only needs the exclusion when its DSpark stages hold a
        // single MLA-style K per position (dsv4_hc_mult > 0); the plain
        // multi-head-K variant behaves like an ordinary unified cache.
        // TurboQuant: SWA (iSWA) architectures are no longer blanket-excluded
        // here - llama_kv_cache_iswa/llama_memory_hybrid_iswa now forward the
        // streaming params to their base cache (see llama-kv-cache-iswa.cpp),
        // and llama_context::kv_stream_switch_phase/process_ubatch resolve
        // targets via llama_memory_i::get_kv_stream_targets() rather than a
        // fixed set of dynamic_casts, so an un-wired memory type (e.g.
        // llama_memory_hybrid_idx) fails closed with a clean error instead
        // of the hard crash this exclusion used to prevent.
        //
        // LLM_ARCH_DEEPSEEK4 (llama_kv_cache_dsv4) stays excluded, along with
        // DEEPSEEK32/GLM_DSA (llama_kv_cache_dsa) and MINIMAX_M3
        // (llama_kv_cache_msa). kv_csa has streaming plumbing wired up (see
        // llama-kv-cache-dsv4.cpp), but real-model KL-divergence testing
        // against a real DeepSeek-V4 model found a genuine, large, and
        // arena-size-independent correctness regression (mean KLD 0.07, max
        // 4.3, vs ~0.000000 for every other architecture validated this
        // way) when streaming is enabled. The initial hypothesis - that the
        // shared page-geometry function in fattn.cu mis-sizes the V region
        // for kv_csa's K-only shape, since DSV4's compressed-cache attention
        // (models/deepseek4.cpp, build_attn_mha with k_all passed as both K
        // and V) always needs a real V-shaped region even though the
        // persistent cache doesn't store V - was tested and RULED OUT: the
        // KLD was bit-identical whether or not that geometry path reserved a
        // V region at all. The actual root cause is still unidentified;
        // likely candidates are DSV4's online-compression bookkeeping
        // (state_pos/state_persist_*/state_snapshot_* row addressing in
        // llama_kv_cache_dsv4_context::comp_plan) not surviving the
        // resident-cache/ring-buffer model correctly, but this needs
        // dedicated investigation into that machinery before re-enabling.
        // The same K-only cache shape is used by DSA's kv_mla and by plain
        // MLA, so - although neither has been empirically tested against a
        // real model - both should be treated as similarly unverified until
        // this is root-caused, not assumed safe by extension of Phase 4's
        // (inspection-only, never-tested) conclusion that MLA "just works"
        // via the generalized single-cache path.
        const bool kv_stream_unified_kv_cache = !llm_arch_is_recurrent(model.arch) &&
                                                model.arch != LLM_ARCH_MINIMAX_M3 && model.arch != LLM_ARCH_GLM_DSA &&
                                                model.arch != LLM_ARCH_DEEPSEEK32 && model.arch != LLM_ARCH_DEEPSEEK4 &&
                                                (model.arch != LLM_ARCH_DFLASH || hparams.dsv4_hc_mult == 0) &&
                                                // without SWA, qwen4exp builds llama_memory_hybrid_idx (a per-token
                                                // indexer cache beside the attention one), which does not carry the
                                                // streaming parameters. with SWA it takes the hybrid-iswa path like
                                                // any other hybrid model, which does stream.
                                                (model.arch != LLM_ARCH_QWEN4EXP ||
                                                 hparams.swa_type != LLAMA_SWA_TYPE_NONE);

        const llama_kv_stream_config stream_config = {
            /*.arena_bytes         =*/kv_stream_arena_bytes,
            /*.minimum_arena_bytes =*/kv_stream_arena_bytes == 0 ? uint64_t(0) : uint64_t(1),
            /*.unified_kv_cache    =*/kv_stream_unified_kv_cache,
            /*.context_default     =*/cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT,
            /*.single_sequence     =*/cparams.n_seq_max == 1,
            /*.flash_attention     =*/cparams.flash_attn,
            /*.kv_offload          =*/cparams.offload_kqv,
            /*.swa_full_conflict   =*/params.swa_full && hparams.is_swa_any(),
        };
        const auto stream_validation = llama_kv_stream_config_validate(stream_config);
        if (!stream_validation.valid) {
            throw std::runtime_error(stream_validation.error);
        }
        // TurboQuant: a no_alloc context (e.g. common_get_device_memory_data's
        // no_alloc=true measurement pass, used to size the parent model for
        // MTP) never actually allocates backend buffers, so llama_kv_cache's
        // own streaming-attach loop skips it (see the matching !hparams.no_alloc
        // guard in llama-kv-cache.cpp). Building the phase arena here anyway
        // would leave it with no cache ever attached, and kv_stream_switch_phase
        // would then fail closed with a confusing "requires a memory type with
        // a streamable KV cache" error on every no_alloc measurement pass.
        if (stream_validation.enabled && !hparams.no_alloc) {
            using page_bytes_fn_t = bool (*)(ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
            using arena_new_fn_t  = void * (*) (ggml_backend_dev_t, size_t);
            using arena_free_fn_t = void (*)(void *);
            using arena_set_compute_fn_t = bool (*)(void *, size_t, size_t);
            using arena_buffer_type_fn_t = ggml_backend_buffer_type_t (*)(void *);
            using graph_reset_fn_t       = bool (*)(ggml_backend_t);

            // llama_kv_cache's constructor may itself upgrade K away from the
            // requested type (turbo auto-asymmetric on high-GQA-ratio models -
            // see llama-kv-cache.cpp). Size this bootstrap pre-scan off the
            // same effective type it will actually construct with, or the
            // streaming runtime's internal page-count arithmetic can end up
            // inconsistent with what gets allocated.
            const ggml_type stream_type_k =
                llama_kv_cache_resolve_stream_type_k(model, hparams, params.type_k, params.type_v);

            // TURBO_LAYER_ADAPTIVE (see llama-kv-cache.cpp) can give individual
            // layers a different K/V precision than the rest of the cache -
            // including auto-enabling itself with no env var at all whenever
            // type_v is turbo2 on an >=8-layer model. The streamed page pool
            // has one page size and one buffer type for the whole arena, so a
            // model that would actually get non-uniform per-layer types must
            // be refused here, before ever touching the CUDA runtime, rather
            // than silently packing wrongly-shaped rows into shared pages.
            const int kv_stream_adaptive_mode =
                llama_kv_cache_turbo_layer_adaptive_mode(params.type_v, hparams.n_layer());
            if (kv_stream_adaptive_mode != 0) {
                for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                    if (!hparams.has_kv(il) || hparams.is_recr(il) || (hparams.is_swa(il) && !params.swa_full)) {
                        continue;
                    }
                    const ggml_type layer_type_k = llama_kv_cache_turbo_layer_adaptive_type_k(
                        kv_stream_adaptive_mode, stream_type_k, params.type_v, il, hparams.n_layer());
                    const ggml_type layer_type_v = llama_kv_cache_turbo_layer_adaptive_type_v(
                        kv_stream_adaptive_mode, stream_type_k, params.type_v, il, hparams.n_layer());
                    if (layer_type_k != stream_type_k || layer_type_v != params.type_v) {
                        throw std::runtime_error(
                            "block KV streaming does not support per-layer mixed KV precision "
                            "(TURBO_LAYER_ADAPTIVE=" +
                            std::to_string(kv_stream_adaptive_mode) +
                            "); the streamed page pool assumes uniform per-layer geometry");
                    }
                }
            }

            ggml_backend_dev_t stream_device       = nullptr;
            size_t             page_bytes          = 0;
            size_t             conversion_bytes    = 0;
            uint32_t           layer_count         = 0;
            uint32_t           stream_head_count_q = 0;
            uint32_t           stream_head_dim_v   = 0;
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                // For iSWA-shaped models (llama_kv_cache_iswa), SWA layers
                // land in kv_swa, not kv_base - and kv_swa only gets a
                // streaming runtime attached when swa_full makes it
                // full-context-length too (see llama-kv-cache-iswa.cpp).
                // Some models (e.g. Gemma-family) have genuinely different
                // per-layer geometry between their SWA and full-attention
                // layers (separate key_length/key_length_swa in the GGUF),
                // so including SWA layers in this scan's uniform-geometry
                // check would wrongly reject models whose actually-streamed
                // layer set (kv_base's non-SWA layers) IS uniform.
                if (!hparams.has_kv(il) || hparams.is_recr(il) || (hparams.is_swa(il) && !params.swa_full)) {
                    continue;
                }
                auto * dev = model.dev_layer(il);
                if (stream_device != nullptr && stream_device != dev) {
                    throw std::runtime_error("block KV streaming requires every attention layer on one CUDA device");
                }
                stream_device = dev;
                ++layer_count;

                auto reg = ggml_backend_dev_backend_reg(dev);
                auto page_bytes_fn =
                    (page_bytes_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_page_bytes");
                auto workspace_bytes_fn = (page_bytes_fn_t) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_cuda_kv_stream_workspace_bytes");
                if (page_bytes_fn == nullptr || workspace_bytes_fn == nullptr) {
                    throw std::runtime_error("block KV streaming requires the CUDA backend");
                }
                size_t layer_page_bytes       = 0;
                size_t layer_conversion_bytes = 0;
                if (!page_bytes_fn(stream_type_k, params.type_v, hparams.n_embd_head_k(il), hparams.n_embd_head_v(il),
                                   hparams.n_head_kv(il), 256, &layer_page_bytes) ||
                    !workspace_bytes_fn(stream_type_k, params.type_v, hparams.n_embd_head_k(il),
                                        hparams.n_embd_head_v(il), hparams.n_head_kv(il), 256,
                                        &layer_conversion_bytes)) {
                    throw std::runtime_error("invalid block KV streaming page geometry");
                }
                if ((page_bytes != 0 && page_bytes != layer_page_bytes) ||
                    (conversion_bytes != 0 && conversion_bytes != layer_conversion_bytes)) {
                    throw std::runtime_error("block KV streaming requires uniform attention-layer geometry");
                }
                page_bytes          = layer_page_bytes;
                conversion_bytes    = layer_conversion_bytes;
                stream_head_count_q = hparams.n_head(il);
                stream_head_dim_v   = hparams.n_embd_head_v(il);
            }
            if (stream_device == nullptr || page_bytes == 0 || layer_count == 0) {
                throw std::runtime_error("block KV streaming found no attention layers");
            }

            const uint64_t bootstrap_pages = uint64_t(layer_count) + kv_stream_phase_arena.minimum_ring_slots;
            if (bootstrap_pages > std::numeric_limits<uint64_t>::max() / page_bytes ||
                conversion_bytes > std::numeric_limits<uint64_t>::max() - bootstrap_pages * page_bytes) {
                throw std::runtime_error("block KV streaming bootstrap size overflow");
            }
            const uint64_t bootstrap_raw  = conversion_bytes + bootstrap_pages * page_bytes;
            kv_stream_minimum_stage_bytes = (bootstrap_raw + 127ULL) & ~127ULL;
            if (kv_stream_minimum_stage_bytes >= kv_stream_arena_bytes) {
                throw std::runtime_error("block KV streaming arena is too small for bootstrap KV and compute slices");
            }

            // The streamed attention kernel's chunked partial-attention
            // reduction (fattn.cu's ggml_cuda_flash_attn_ext_streamed) draws
            // its scratch buffers from the device's general CUDA memory
            // pool, not from this arena - an arena that fits by itself can
            // still leave too little free VRAM for that pool to grow into
            // during actual inference, which aborts the whole process
            // instead of failing cleanly (see docs/kv-stream.md's "Sizing
            // the arena"). Confirm enough headroom survives reserving the
            // arena for that pool's worst case before ever allocating it.
            // This uses free VRAM as measured right now; other allocations
            // later in context construction (e.g. the compute buffer) are
            // not yet accounted for, so this catches the clearly-too-tight
            // case, not every possible one - the empirical probing in
            // benchmarks/benchmark_kv_stream.py remains the reliable way to
            // size an arena for production.
            {
                using transient_workspace_fn_t = bool (*)(uint32_t, uint32_t, uint32_t, size_t *);
                auto transient_workspace_fn    = (transient_workspace_fn_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(stream_device),
                    "ggml_backend_cuda_kv_stream_transient_workspace_bytes");
                if (transient_workspace_fn == nullptr) {
                    throw std::runtime_error("block KV streaming requires the CUDA backend");
                }
                size_t transient_workspace_bytes = 0;
                if (!transient_workspace_fn(stream_head_count_q, stream_head_dim_v, cparams.n_ubatch,
                                            &transient_workspace_bytes)) {
                    throw std::runtime_error("invalid block KV streaming transient workspace geometry");
                }
                size_t dev_free_bytes  = 0;
                size_t dev_total_bytes = 0;
                ggml_backend_dev_memory(stream_device, &dev_free_bytes, &dev_total_bytes);
                if (kv_stream_arena_bytes > dev_free_bytes ||
                    transient_workspace_bytes > dev_free_bytes - kv_stream_arena_bytes) {
                    throw std::runtime_error(
                        "block KV streaming arena leaves insufficient free VRAM for the streamed "
                        "attention kernel's transient workspace (needs roughly " +
                        std::to_string((transient_workspace_bytes + 1024 * 1024 - 1) / (1024 * 1024)) +
                        " MiB of headroom beyond the arena) - reduce --kv-stream-arena-mib or "
                        "--ubatch-size, or free more device memory");
                }
            }

            auto reg = ggml_backend_dev_backend_reg(stream_device);
            auto arena_new_fn =
                (arena_new_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_phase_arena_new_for_device");
            auto arena_free_fn =
                (arena_free_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_phase_arena_free");
            auto arena_set_compute_fn = (arena_set_compute_fn_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_phase_arena_set_compute");
            auto arena_buffer_type_fn = (arena_buffer_type_fn_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_phase_arena_buffer_type");
            auto graph_reset_fn =
                (graph_reset_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reset");
            if (arena_new_fn == nullptr || arena_free_fn == nullptr || arena_set_compute_fn == nullptr ||
                arena_buffer_type_fn == nullptr || graph_reset_fn == nullptr) {
                throw std::runtime_error("block KV streaming phase arena is unavailable");
            }

            kv_stream_phase_arena.arena                  = arena_new_fn(stream_device, kv_stream_arena_bytes);
            kv_stream_phase_arena.free_fn                = arena_free_fn;
            kv_stream_phase_arena.set_compute_fn         = arena_set_compute_fn;
            kv_stream_phase_arena.buffer_type_fn         = arena_buffer_type_fn;
            kv_stream_phase_arena.graph_reset_fn         = graph_reset_fn;
            kv_stream_phase_arena.device                 = stream_device;
            kv_stream_phase_arena.arena_bytes            = kv_stream_arena_bytes;
            kv_stream_phase_arena.page_bytes             = page_bytes;
            kv_stream_phase_arena.conversion_bytes       = conversion_bytes;
            kv_stream_phase_arena.layer_count            = layer_count;
            kv_stream_phase_arena.current_kv_bytes       = kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_compute_offset = kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_compute_bytes  = kv_stream_arena_bytes - kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_ring_slots     = kv_stream_phase_arena.minimum_ring_slots;
            if (kv_stream_phase_arena.arena == nullptr ||
                !arena_set_compute_fn(kv_stream_phase_arena.arena, kv_stream_minimum_stage_bytes,
                                      kv_stream_arena_bytes - kv_stream_minimum_stage_bytes)) {
                throw std::runtime_error("failed to create CUDA block KV streaming phase arena");
            }
            kv_stream_phase_arena.buffer_type = arena_buffer_type_fn(kv_stream_phase_arena.arena);
            if (kv_stream_phase_arena.buffer_type == nullptr) {
                throw std::runtime_error("failed to obtain phase arena CUDA buffer type");
            }
            kv_stream_stage_bytes = kv_stream_minimum_stage_bytes;
            LLAMA_LOG_INFO("%s: experimental block KV streaming enabled, arena = %.2f MiB, bootstrap KV = %.2f MiB\n",
                           __func__, kv_stream_arena_bytes / 1024.0 / 1024.0, kv_stream_stage_bytes / 1024.0 / 1024.0);
        }

        llama_memory_params params_mem = {
            /*.type_k                =*/params.type_k,
            /*.type_v                =*/params.type_v,
            /*.kv_stream_stage_bytes =*/kv_stream_stage_bytes,
            /*.kv_stream_phase_arena =*/kv_stream_phase_arena.arena,
            /*.kv_stream_maximum_pool_bytes =*/kv_stream_arena_bytes,
            /*.type_s                =*/params.type_s,
            /*.swa_full              =*/params.swa_full,
            /*.ctx_type              =*/cparams.ctx_type,
            /*.mem_other             =*/llama_get_memory(cparams.ctx_other),
            /*.sj_kvarn     =*/ {
                /*.bits_k   =*/ params.sj_kvarn_bits_k,
                /*.bits_v   =*/ params.sj_kvarn_bits_v,
                /*.sink_type =*/ params.sj_kvarn_sink_type,
                /*.sink     =*/ params.sj_kvarn_sink,
                /*.group    =*/ 128,
                /*.tail     =*/ params.sj_kvarn_tail,
                /*.n_ubatch =*/ cparams.n_ubatch,
                /*.body_type =*/ params.sj_kvarn_body_type,
                /*.tail_max =*/ params.sj_kvarn_tail_max,
                /*.edge_layers =*/ params.sj_kvarn_edge_layers,
                /*.edge_bits_k =*/ params.sj_kvarn_edge_bits_k,
                /*.edge_bits_v =*/ params.sj_kvarn_edge_bits_v,
                /*.edge_body_type =*/ params.sj_kvarn_edge_body_type,
                /*.flush_chunk =*/ params.sj_kvarn_flush_chunk,
                /*.resident_cells =*/ params.kv_resident,
                /*.shared_pool =*/ sj_kvarn_shared_pool,
                /*.transform   =*/ params.sj_kvarn_transform ? params.sj_kvarn_transform : "",
            },
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft         = ggml_backend_get_default_buffer_type(backend.get());
            auto   backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev       = model.devices[0];
                auto *       host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel = model.n_devices() > 1 && model.n_gpu_layers() > model.hparams.n_layer_all &&
                                 model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.offload_kqv &&
                                 !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto *                 dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // fork: Reserve only catch-up and small generation graphs for the independent MTP scheduler.
        const char * mtp_async = getenv("GGML_MTP_PREFILL_ASYNC");
        const char * kv_only = getenv("LLAMA_MTP_UPDATE_KV_ONLY");
        const char * mtp_tight = getenv("LLAMA_MTP_RESERVE_TIGHT");
        const char * mtp_scratch = getenv("LLAMA_MTP_PREFILL_SCRATCH");
        ggml_backend_dev_props mtp_props = {};
        ggml_backend_dev_get_props(model.dev_output(), &mtp_props);
        mtp_scratch_enabled = mtp_scratch && strcmp(mtp_scratch, "1") == 0 &&
            (!kv_only || strcmp(kv_only, "0") != 0) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
            (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN4EXP) && model.hparams.n_layer_nextn == 1 && params.ctx_other &&
            model.dev_output() == params.ctx_other->model.dev_output() && model.n_devices() <= 1 &&
            params.ctx_other->model.n_devices() <= 1 && params.ctx_other->model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
            model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
            (mtp_props.type == GGML_BACKEND_DEVICE_TYPE_CPU || mtp_props.caps.events);
        if (mtp_scratch_enabled) {
            mtp_scratch_target = params.ctx_other;
            LLAMA_LOG_INFO("%s: LLAMA_MTP_PREFILL_SCRATCH=1: borrow unused target KV for MTP catch-up\n", __func__);
        }
        mtp_reserve_tight = mtp_tight && strcmp(mtp_tight, "1") == 0 &&
            (!kv_only || strcmp(kv_only, "0") != 0) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
            (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN4EXP) && model.hparams.n_layer_nextn == 1;
        mtp_reserve_tight = mtp_reserve_tight || mtp_scratch_enabled;
        if (mtp_reserve_tight) {
            LLAMA_LOG_INFO("%s: LLAMA_MTP_RESERVE_TIGHT=1: reserve KV-only prefill and deferred generation rows\n", __func__);
        }
        mtp_prefill_async = !mtp_scratch_enabled && mtp_async && strcmp(mtp_async, "1") == 0 &&
            (!kv_only || strcmp(kv_only, "0") != 0) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
            model.arch == LLM_ARCH_QWEN35 &&
            model.hparams.n_layer_nextn == 1 && params.ctx_other && model.n_devices() <= 1 &&
            model.split_mode() != LLAMA_SPLIT_MODE_TENSOR;
        if (mtp_prefill_async) {
            LLAMA_LOG_INFO("%s: GGML_MTP_PREFILL_ASYNC=1: private MTP compute and scratch buffers\n", __func__);
        }

        // share the target's compute buffers with an MTP context on the same device (default on,
        // LLAMA_SHARED_COMPUTE=0 disables): the two graphs never run at the same time
        {
            const char * env = getenv("LLAMA_SHARED_COMPUTE");
            const bool enabled = env == nullptr || env[0] != '0';
            // not with a tensor split: the meta backend keeps per-device tensor mappings in its compute buffer that
            // the donor's graph allocation resets while the MTP context still reuses its graph (each context gets
            // its own compute buffers instead)
            if (enabled && !mtp_prefill_async && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && params.ctx_other != nullptr &&
                    model.n_devices() <= 1 && params.ctx_other->model.n_devices() <= 1 &&
                    model.dev_output() == params.ctx_other->model.dev_output() &&
                    params.ctx_other->model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
                    model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
                    backend_buft == params.ctx_other->backend_buft &&
                    params.ctx_other->compute_peer == nullptr) {
                compute_peer = params.ctx_other;
                LLAMA_LOG_INFO("%s: MTP context will share the target context's compute buffers\n", __func__);
            }
        }
        if (cparams.moe_cache_size > 0) {
            if (cparams.pipeline_parallel || model.n_devices() > 1) {
                throw std::runtime_error("MoE cache does not support multiple devices");
            }
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[i]));
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs[i], backend_buft[i], cparams.moe_cache_size);
                    break;
                }
            }
            if (!moe_cache) {
                throw std::runtime_error("MoE cache requires a GPU backend");
            }
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
    if (mtp_scratch_target) {
        mtp_scratch_target->mtp_scratch_clients.push_back(this);
    }
    // Publish the backlink only after all throwing initialization has completed.
    if (compute_peer) {
        compute_peer->compute_peer = this;
    }
}

llama_context::~llama_context() {
    if (auto * kv = llama_sj_kvarn_cache(memory.get())) {
        kv->scratch_cancel();
    }
    mtp_scratch_end();
    for (auto * client : mtp_scratch_clients) {
        client->mtp_scratch_end();
        client->mtp_scratch_target = nullptr;
        client->mtp_scratch_enabled = false;
    }
    if (mtp_scratch_target) {
        auto & clients = mtp_scratch_target->mtp_scratch_clients;
        clients.erase(std::remove(clients.begin(), clients.end(), this), clients.end());
    }
    if (compute_peer != nullptr) {
        compute_peer->synchronize();
        if (compute_peer->sched) { ggml_backend_sched_set_donor(compute_peer->sched.get(), nullptr); }
        if (compute_peer->compute_peer == this) {
            compute_peer->compute_peer = nullptr;
        }
        compute_peer = nullptr;
    }
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    if (mtp_hidden.consuming && mtp_hidden.consumed) {
        ggml_backend_event_synchronize(mtp_hidden.consumed.get());
    }

    if (!b1_kv_scratch_release()) { GGML_ABORT("cannot return B1 KV scratch"); }

    // [B1] give lent expert layers back to the model before the backends go away
    b1.reset();
    ple_pf.reset();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp != size_act && backend_buf_exp_epoch != ggml_backend_sched_get_alloc_epoch(sched.get())) {
                // a peer context sharing this compute buffer (the MTP drafter) grew it after this context reserved
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, grown from %8.4f MiB by a peer sharing it\n",
                                __func__, ggml_backend_buft_name(buft), size_act / (1024.0 * 1024.0),
                                size_exp / (1024.0 * 1024.0));
            } else if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                                __func__, ggml_backend_buft_name(buft), size_act / (1024.0 * 1024.0),
                                size_exp / (1024.0 * 1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                               __func__, ggml_backend_buft_name(buft), size_act / (1024.0 * 1024.0),
                               size_exp / (1024.0 * 1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

// tokenizer identity for the built-in shortlists: every token string and every BPE merge, in order
// (definition in llama-mtp-vocab-builtin.h, scripts/gen-mtp-vocab-builtin.py computes the same value)
uint64_t llama_model_tokenizer_fingerprint(const llama_model * model) {
    const llama_vocab & vocab = model->vocab;
    llama_fnv1a64 h;
    const uint32_t n = vocab.n_tokens();
    for (uint32_t i = 0; i < n; ++i) {
        const char * text = vocab.token_get_text((llama_token) i);
        h.add(text, std::strlen(text));
        const uint8_t z = 0;
        h.add(&z, 1);
    }
    const uint8_t sep = 1;
    h.add(&sep, 1);
    for (const auto & m : vocab.get_bpe_merges()) {
        h.add_str0(m);
    }
    return h.h;
}

void llama_context::init_draft_vocab(const char * path, int32_t n_hot) {
    const int64_t n_vocab = model.vocab.n_tokens();

    llama_mtp_vocab_map map;
    std::string source = path;

    int64_t auto_size = 0;
    if (llama_mtp_vocab_parse_auto(path, auto_size)) {
        // pick the list compiled into the binary for this tokenizer + architecture, if there is one
        const uint64_t fp   = llama_model_tokenizer_fingerprint(&model);
        const char *   arch = llm_arch_name(model.arch);
        const llama_mtp_vocab_builtin * b = llama_mtp_vocab_builtin_find(fp, n_vocab, arch, auto_size);
        if (b == nullptr) {
            LLAMA_LOG_INFO("%s: draft vocabulary '%s': no built-in shortlist for this tokenizer (fingerprint %016" PRIx64
                    ", %lld tokens, arch %s), the draft scores the full head\n",
                    __func__, path, fp, (long long) n_vocab, arch);
            return;
        }
        map.n_vocab = b->n_vocab;
        map.ids     = llama_mtp_vocab_builtin_ids(*b);
        source      = format("built-in %s (%s family, tokenizer %016" PRIx64 ")", b->name, b->family, fp);
    } else {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error(format("cannot open draft vocabulary map '%s'", path));
        }
        map = llama_mtp_vocab_read(file, n_vocab);
    }
    if (map.n_vocab != n_vocab) {
        throw std::runtime_error(format("draft vocabulary map '%s' is for a %lld-token vocabulary, model has %lld",
                    path, (long long) map.n_vocab, (long long) n_vocab));
    }

    // the map lives next to the head it indexes
    const ggml_tensor * head = model.output;
    if (model.hparams.n_layer_nextn > 0 && model.hparams.n_layer() < model.layers.size()) {
        const auto & nextn = model.layers[model.hparams.n_layer()].nextn;
        if (nextn.shared_head_head) {
            head = nextn.shared_head_head;
        }
    }
    if (head == nullptr || head->buffer == nullptr) {
        throw std::runtime_error("draft vocabulary map: model has no allocated output head");
    }

    const int64_t n_sel = (int64_t) map.ids.size();

    if (n_hot < 0 || (int64_t) n_hot >= n_sel) {
        throw std::runtime_error(format("draft vocabulary hot tail of %d slots is not in [0, %lld)",
                    n_hot, (long long) n_sel));
    }

    // [#81] compact resident draft head: auto for EXL3 and AI+ heads (no one-row path), LLAMA_DRAFT_VOCAB_COMPACT=1
    // forces it on any head (A/B), =0 keeps the full-head fallback; prism-rotated heads keep their fallback.
    // Row type: iq4_xs from an EXL3 head, q6_K from an AI+ head (the AI+ output head is AI6_S, about 6 bits, and the
    // IQ4_XS model's head is Q6_K: same draft-head precision class); LLAMA_DRAFT_VOCAB_COMPACT_TYPE overrides.
    ggml_type compact_type = GGML_TYPE_COUNT;
    {
        const char * env = getenv("LLAMA_DRAFT_VOCAB_COMPACT");
        const int force = env != nullptr && env[0] != '\0' ? atoi(env) : -1;
        const bool is_exl3 = ggml_exl3_bits(head->type) != 0;
        const bool is_ai   = ggml_ai4_rate(head->type) != 0;
        const bool want = !model.hparams.no_alloc && !model.hadamard_rotations.count(head) &&
                n_sel < head->ne[1] && (force == 1 || (force != 0 && (is_exl3 || is_ai)));
        if (want) {
            compact_type = is_ai ? GGML_TYPE_Q6_K : GGML_TYPE_IQ4_XS;
            if (const char * te = getenv("LLAMA_DRAFT_VOCAB_COMPACT_TYPE"); te != nullptr && te[0] != '\0') {
                compact_type = GGML_TYPE_COUNT;
                for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                    const char * name = ggml_type_name((ggml_type) t);
                    auto ieq = [](const char * a, const char * b) {
                        for (; *a && *b; ++a, ++b) {
                            if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
                                return false;
                            }
                        }
                        return *a == *b;
                    };
                    if (name != nullptr && ieq(name, te)) {
                        compact_type = (ggml_type) t;
                        break;
                    }
                }
                if (compact_type == GGML_TYPE_COUNT || !llama_draft_compact_type_ok(compact_type, head->ne[0])) {
                    throw std::runtime_error(format("LLAMA_DRAFT_VOCAB_COMPACT_TYPE=%s is not one of f32, f16, q8_0, q6_K, "
                                "q5_K, q4_K, iq4_xs, iq4_nl for rows of %lld", te, (long long) head->ne[0]));
                }
            }
            if (is_exl3 && (model.exl3_side_of(head) == nullptr || model.exl3_had128 == nullptr ||
                        head->ne[0] % 128 != 0 || head->ne[1] % 128 != 0)) {
                LLAMA_LOG_WARN("%s: EXL3 head %s has no .suh/.svh or no exl3_had128, no compact draft head\n",
                        __func__, ggml_get_name(head));
                compact_type = GGML_TYPE_COUNT;
            } else if (is_ai && (head->ne[1] % 16 != 0 || ggml_ai4_slab_nbytes(head->type, head->ne[0]) == 0 ||
                        !ggml_is_contiguous(head))) {
                LLAMA_LOG_WARN("%s: AI+ head %s (%s) is not whole 16-row slabs, no compact draft head\n",
                        __func__, ggml_get_name(head), ggml_type_name(head->type));
                compact_type = GGML_TYPE_COUNT;
            } else if (!is_exl3 && !is_ai && (ggml_get_type_traits(head->type)->to_float == nullptr || !ggml_is_contiguous(head))) {
                LLAMA_LOG_WARN("%s: head %s (%s) has no row decoder, no compact draft head\n",
                        __func__, ggml_get_name(head), ggml_type_name(head->type));
                compact_type = GGML_TYPE_COUNT;
            } else {
                // a split (multi-device row) buffer cannot hold the compact head as one tensor
                ggml_backend_buffer_type_t hb = ggml_backend_buffer_get_type(head->buffer);
                ggml_backend_dev_t hd = ggml_backend_buft_get_device(hb);
                if (!ggml_backend_buft_is_host(hb) && (hd == nullptr || ggml_backend_dev_buffer_type(hd) != hb)) {
                    LLAMA_LOG_WARN("%s: head %s is in a split buffer (%s), no compact draft head\n",
                            __func__, ggml_get_name(head), ggml_backend_buft_name(hb));
                    compact_type = GGML_TYPE_COUNT;
                }
            }
        }
    }
    const bool use_compact = compact_type != GGML_TYPE_COUNT;
    if (use_compact && n_hot > 0) {
        // the hot tail repoints ids at request boundaries; the compact rows are built once at load
        LLAMA_LOG_WARN("%s: the %d adaptive hot slots are disabled with the compact draft head (static map)\n", __func__, n_hot);
        n_hot = 0;
    }

    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    draft_vocab.ctx.reset(ggml_init(ip));
    draft_vocab.ids = ggml_new_tensor_1d(draft_vocab.ctx.get(), GGML_TYPE_I32, n_sel);
    ggml_set_name(draft_vocab.ids, "draft_vocab_ids");
    if (use_compact) {
        draft_vocab.compact = ggml_new_tensor_2d(draft_vocab.ctx.get(), compact_type, head->ne[0], n_sel);
        ggml_set_name(draft_vocab.compact, "draft_vocab_compact");
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(head->buffer);
    draft_vocab.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(draft_vocab.ctx.get(), buft));
    if (!draft_vocab.buf) {
        throw std::runtime_error("draft vocabulary map: failed to allocate the id map");
    }
    if (use_compact) {
        ggml_backend_buffer_set_usage(draft_vocab.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_backend_tensor_set(draft_vocab.ids, map.ids.data(), 0, (size_t) n_sel * sizeof(int32_t));

    if (use_compact) {
        const int64_t t0 = ggml_time_us();
        // host copies of the head and its EXL3 side data; freed when the rows are built
        std::vector<uint8_t> head_bytes(ggml_nbytes(head));
        ggml_backend_tensor_get(head, head_bytes.data(), 0, head_bytes.size());
        std::vector<float> svh, had;
        llama_draft_compact_src src;
        src.data = head_bytes.data();
        src.type = head->type;
        src.K    = head->ne[0];
        src.N    = head->ne[1];
        if (ggml_exl3_bits(head->type) != 0) {
            const auto * side = model.exl3_side_of(head);
            svh.resize(head->ne[1]);
            had.resize(128 * 128);
            ggml_backend_tensor_get(side->svh, svh.data(), 0, svh.size() * sizeof(float));
            ggml_backend_tensor_get(model.exl3_had128, had.data(), 0, had.size() * sizeof(float));
            src.svh = svh.data();
            src.had = had.data();
        }
        std::vector<uint8_t> rows(ggml_nbytes(draft_vocab.compact));
        const int n_thr = std::max(1, (int) std::min<uint32_t>(cparams.n_threads_batch, 16));
        if (!llama_draft_compact_build(src, map.ids.data(), n_sel, compact_type, rows.data(), n_thr)) {
            throw std::runtime_error("draft vocabulary map: building the compact draft head failed");
        }
        ggml_backend_tensor_set(draft_vocab.compact, rows.data(), 0, rows.size());
        draft_vocab.compact_src = head;
        // WARN so it shows at the server's default verbosity: resident memory next to the model weights
        LLAMA_LOG_WARN("%s: compact draft head: type %s, n_sel %lld, %.1f MiB on %s, from %s (%s) in %.2f s (%d threads)\n",
                __func__, ggml_type_name(compact_type), (long long) n_sel,
                ggml_nbytes(draft_vocab.compact) / 1024.0 / 1024.0, ggml_backend_buft_name(buft),
                ggml_get_name(head), ggml_type_name(head->type), (ggml_time_us() - t0) / 1e6, n_thr);
    }
    draft_vocab.host = std::move(map.ids);

    draft_vocab.n_hot = n_hot;
    if (n_hot > 0) {
        draft_vocab.hot.init(draft_vocab.host, n_vocab, n_hot);
    }

    LLAMA_LOG_INFO("%s: draft vocabulary shortlist: %lld of %lld tokens from %s, map on %s (%.1f KiB), head %s (%s)\n",
            __func__, (long long) n_sel, (long long) n_vocab, source.c_str(), ggml_backend_buft_name(buft),
            (double) n_sel * sizeof(int32_t) / 1024.0, ggml_get_name(head), ggml_type_name(head->type));
    if (n_hot > 0) {
        LLAMA_LOG_INFO("%s: draft vocabulary tail: %d of %lld entries are adaptive hot slots\n",
                __func__, n_hot, (long long) n_sel);
    }
}

void llama_context::draft_vocab_observe(const llama_token * toks, size_t n) {
    if (draft_vocab.n_hot <= 0 || toks == nullptr || n == 0) {
        return;
    }
    draft_vocab.hot.observe(toks, n);
}

int32_t llama_context::draft_vocab_refresh() {
    if (draft_vocab.n_hot <= 0 || draft_vocab.ids == nullptr) {
        return 0;
    }
    const auto reps = draft_vocab.hot.select();
    if (reps.empty()) {
        return 0;
    }

    const int64_t n_sel = draft_vocab.ids->ne[0];
    const int64_t base  = n_sel - draft_vocab.n_hot;
    for (const auto & rep : reps) {
        draft_vocab.host[base + rep.slot] = rep.id;
    }

    // one upload of the whole hot region: cheap at request granularity and keeps the tensor fixed
    ggml_backend_tensor_set(draft_vocab.ids, draft_vocab.host.data() + base,
            (size_t) base * sizeof(int32_t), (size_t) draft_vocab.n_hot * sizeof(int32_t));

    return (int32_t) reps.size();
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func    = __func__;
    auto         resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t     backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused  = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN(
                    "%s: layer %d is assigned to device %s but %s "
                            "is assigned to device %s (usually due to missing support)\n",
                    func, node.il, device_layer ? ggml_backend_dev_name(device_layer) : "none", probe.name,
                    device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe, cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static bool llama_model_has_cacheable_moe_weights(const llama_model &                 model,
                                                  llama_moe_cache_mode                mode,
                                                  size_t                              budget_mib,
                                                  const std::vector<ggml_backend_t> & backends) {
    // Probe the provider that owns the model's backends; fall back to the
    // thread's active provider (first registered).
    ggml_moe_cache_api api = ggml_moe_cache_active();
    for (ggml_backend_t backend : backends) {
        if (!backend) {
            continue;
        }
        const ggml_moe_cache_api owned =
            ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));
        if (owned.owner) {
            api = owned;
            break;
        }
    }
    if (mode == LLAMA_MOE_CACHE_MODE_OFF) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (mode=off)\n", __func__);
        return false;
    }
    if (!api.query_config || !api.query_device || !api.query_shape) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (no provider registered)\n", __func__);
        return false;
    }

    ggml_moe_cache_config config    = {};
    const int             automatic = mode == LLAMA_MOE_CACHE_MODE_UNSPECIFIED ? -1 : mode == LLAMA_MOE_CACHE_MODE_AUTO;
    if (!api.query_config(automatic, budget_mib, &config)) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (provider query_config returned false)\n", __func__);
        return false;
    }

    std::vector<int32_t> physical_devices;
    size_t               min_expert_bytes = 0;
    for (ggml_backend_t backend : backends) {
        if (!backend) {
            continue;
        }
        ggml_moe_cache_device_caps caps = {};
        if (!api.query_device(ggml_backend_get_device(backend), &config, &caps) ||
            std::find(physical_devices.begin(), physical_devices.end(), caps.physical_device) !=
                physical_devices.end()) {
            continue;
        }
        physical_devices.push_back(caps.physical_device);
        min_expert_bytes = std::max(min_expert_bytes, caps.min_expert_bytes);
    }
    if ((int) physical_devices.size() < config.min_devices) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (eligible devices=%d, need=%d, min_cc=%d)\n", __func__,
                       (int) physical_devices.size(), config.min_devices, config.min_compute_capability);
        return false;
    }

    size_t largest_expert_bytes = 0;
    for (const auto & entry : model.tensors_by_name) {
        const std::string & name   = entry.first;
        const ggml_tensor * tensor = entry.second;
        if (!tensor || (name.find("_exps") == std::string::npos &&
                        name.find("_chexps") == std::string::npos) ||
            ggml_n_dims(tensor) != 3 || tensor->ne[0] <= 0 ||
            tensor->ne[1] <= 0 || tensor->ne[2] <= 0) {
            continue;
        }
        largest_expert_bytes = std::max(largest_expert_bytes, tensor->nb[2]);
        if (tensor->nb[2] < min_expert_bytes) {
            continue;
        }

        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !ggml_backend_buffer_is_host(buffer) ||
            ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            continue;
        }

        ggml_moe_cache_shape_caps shape = {};
        if (api.query_shape(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[2], &shape)) {
            const size_t slab_bytes = std::max(shape.pool_bytes, config.minimum_slab_bytes);
            if (config.budget_bytes > 0 &&
                (shape.scratch_bytes > config.budget_bytes || slab_bytes > config.budget_bytes - shape.scratch_bytes)) {
                continue;
            }
            return true;
        }
    }
    LLAMA_LOG_INFO("%s: MoE cache disabled (no cacheable expert tensors found; largest expert slab=%zu KiB, minimum=%zu KiB)\n",
            __func__, largest_expert_bytes >> 10, min_expert_bytes >> 10);
    return false;
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

bool llama_context::kv_stream_switch_phase(bool decode, uint32_t active_tokens) {
    auto & arena = kv_stream_phase_arena;
    if (!arena.configured) {
        return true;
    }
    if (sched && arena.decode == decode) {
        return true;
    }
    if (arena.backend_index >= backend_ptrs.size()) {
        LLAMA_LOG_ERROR("%s: invalid phase-arena backend index\n", __func__);
        return false;
    }

    // TurboQuant: memory->get_kv_stream_targets() reports which sub-cache(s)
    // of this memory object (a plain llama_kv_cache, llama_memory_hybrid's
    // attention sub-cache, or - once wired up - one of iSWA/DSA/MSA/DSV4's
    // sub-caches) have a streaming runtime attached. Only one target is
    // ever produced today; a second concurrent target needs the CUDA phase
    // arena's single-lease design generalized first (not yet done - see the
    // multi-cache streaming plan), so more than one here is a hard error
    // rather than silently only switching the first.
    const auto stream_targets = memory->get_kv_stream_targets();
    if (stream_targets.empty()) {
        LLAMA_LOG_ERROR("%s: phase arena requires a memory type with a streamable KV cache\n", __func__);
        return false;
    }
    if (stream_targets.size() > 1) {
        LLAMA_LOG_ERROR("%s: phase arena does not yet support more than one concurrent streaming target\n", __func__);
        return false;
    }
    llama_kv_cache * kv = stream_targets[0].cache;
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: phase arena requires a memory type with a streamable KV cache\n", __func__);
        return false;
    }
    const auto & target = decode ? arena.token_generation : arena.prefill;

    synchronize();
    if (sched && !arena.graph_reset_fn(backend_ptrs[arena.backend_index])) {
        LLAMA_LOG_ERROR("%s: failed to invalidate CUDA graphs before arena repartition\n", __func__);
        return false;
    }

    // Releasing the scheduler returns the only borrowed compute lease.
    sched.reset();
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    const size_t   old_kv_bytes       = arena.current_kv_bytes;
    const size_t   old_compute_offset = arena.current_compute_offset;
    const size_t   old_compute_bytes  = arena.current_compute_bytes;
    const uint32_t old_ring_slots     = arena.current_ring_slots;

    auto resize_kv = [&](size_t bytes, uint32_t ring_slots) {
        return kv->kv_stream_resize_pool(bytes, active_tokens, ring_slots);
    };
    auto restore_compute = [&] {
        return arena.set_compute_fn(arena.arena, old_compute_offset, old_compute_bytes);
    };

    if (target.kv_bytes > old_kv_bytes) {
        // Moving compute upward frees the bytes before KV claims them.
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice upward\n", __func__);
            return false;
        }
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            (void) restore_compute();
            LLAMA_LOG_ERROR("%s: failed to grow arena KV slice\n", __func__);
            return false;
        }
    } else if (target.kv_bytes < old_kv_bytes) {
        // KV releases the bytes before compute expands downward.
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            LLAMA_LOG_ERROR("%s: failed to shrink arena KV slice\n", __func__);
            return false;
        }
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            (void) resize_kv(old_kv_bytes, old_ring_slots);
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice downward\n", __func__);
            return false;
        }
    } else {
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to update arena compute slice\n", __func__);
            return false;
        }
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            (void) restore_compute();
            LLAMA_LOG_ERROR("%s: failed to update arena KV layout\n", __func__);
            return false;
        }
    }

    arena.current_kv_bytes       = target.kv_bytes;
    arena.current_compute_offset = target.compute_offset;
    arena.current_compute_bytes  = target.compute_bytes;
    arena.current_ring_slots     = target.ring_slots;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), arena.max_nodes,
                                       cparams.pipeline_parallel, cparams.op_offload));
    if (!sched) {
        LLAMA_LOG_ERROR("%s: failed to recreate scheduler\n", __func__);
        return false;
    }

    auto reserve_mctx = memory->init_full();
    if (!reserve_mctx) {
        LLAMA_LOG_ERROR("%s: failed to initialize full memory for phase reserve\n", __func__);
        return false;
    }

    const uint32_t n_seqs    = cparams.n_seq_max;
    const uint32_t n_tokens  = decode ? n_seqs : std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_outputs = decode ? n_seqs : std::min(n_tokens, cparams.n_outputs_max);
    auto *         gf        = graph_reserve(n_tokens, n_seqs, n_outputs, reserve_mctx.get());
    if (gf == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to reserve %s compute graph in phase arena\n", __func__,
                        decode ? "decode" : "prefill");
        return false;
    }

    backend_buf_exp_size = target.backend_sizes;
    arena.decode         = decode;
    LLAMA_LOG_INFO("%s: activated %s phase: KV %.2f MiB, compute %.2f MiB, active tokens %u\n", __func__,
                   decode ? "decode" : "prefill", target.kv_bytes / 1024.0 / 1024.0,
                   target.compute_bytes / 1024.0 / 1024.0, active_tokens);
    return true;
}

void llama_context::detach_compute_borrower() {
    for (auto * client : mtp_scratch_clients) { client->mtp_scratch_end(); }
    // fork: Replacing the target allocator must not leave the helper's donor pointer stale.
    if (compute_peer && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && compute_peer->sched) {
        compute_peer->synchronize();
        ggml_backend_sched_set_donor(compute_peer->sched.get(), nullptr);
        for (auto & backend : compute_peer->backends) {
            auto * dev = ggml_backend_get_device(backend.get());
            using reset_fn = bool (*)(ggml_backend_t);
            auto reset = (reset_fn) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_reset_scratch");
            if (reset && !reset(backend.get())) {
                throw std::runtime_error("failed to reset helper CUDA scratch before target reserve");
            }
        }
        for (auto & res : compute_peer->gf_res_prev) { res.reset(); }
        compute_peer->gf_res_prev_active = nullptr;
        if (!ggml_backend_sched_reset_compute(compute_peer->sched.get())) {
            throw std::runtime_error("failed to release helper compute before target reserve");
        }
        compute_peer->sched_need_reserve = true;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    mtp_scratch_end();
    detach_compute_borrower();
    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();
    if (kv_stream_phase_arena.configured && sched && kv_stream_phase_arena.backend_index < backend_ptrs.size()) {
        if (!kv_stream_phase_arena.graph_reset_fn(backend_ptrs[kv_stream_phase_arena.backend_index])) {
            throw std::runtime_error("failed to reset CUDA graphs before measuring phase arena");
        }
        sched.reset();
    }

    const int64_t t_start_us = ggml_time_us();

    const char * rotate = getenv("LLAMA_MOE_CACHE_RELEASE_PROMPT");
    if (!sched && sched_reserve_tokens == 0 && rotate && rotate[0] == '1' &&
            cparams.causal_attn && !cparams.training && !cparams.embeddings && !compute_peer && !kv_stream_phase_arena.configured) {
        sched_reserve_tokens = std::min(cparams.n_ubatch, 512u);
    }
    const uint32_t n_seqs   = cparams.n_seq_max;
    const uint32_t reserve_tokens = sched_reserve_tokens ? std::min(cparams.n_ubatch, sched_reserve_tokens) : cparams.n_ubatch;
    const uint32_t n_tokens = std::min(cparams.n_ctx, reserve_tokens);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    const bool moe_cache_eligible = llama_model_has_cacheable_moe_weights(
        model, (llama_moe_cache_mode) cparams.moe_cache_mode, cparams.moe_cache_budget_mib, backend_ptrs);
    const ggml_moe_cache_mode moe_cache_mode =
        moe_cache_eligible ? (ggml_moe_cache_mode) cparams.moe_cache_mode : GGML_MOE_CACHE_MODE_OFF;
    const char * moe_cache_requested = "provider";
    switch (cparams.moe_cache_mode) {
        case LLAMA_MOE_CACHE_MODE_OFF:
            moe_cache_requested = "off";
            break;
        case LLAMA_MOE_CACHE_MODE_AUTO:
            moe_cache_requested = "auto";
            break;
        case LLAMA_MOE_CACHE_MODE_ON:
            moe_cache_requested = "on";
            break;
        case LLAMA_MOE_CACHE_MODE_UNSPECIFIED:
            break;
    }
    LLAMA_LOG_INFO("%s: MoE cache requested=%s resolved=%s\n", __func__, moe_cache_requested,
                   moe_cache_eligible ? moe_cache_requested : "off");

    const size_t node_capacity = graph_max_nodes(cparams.n_ubatch);
    bool preserve_cache = sched && !cparams.training && !kv_stream_phase_arena.configured &&
        node_capacity <= sched_node_capacity && sched_cache_mode == (int) moe_cache_mode &&
        sched_cache_budget_mib == cparams.moe_cache_budget_mib &&
        sched_pipeline_parallel == cparams.pipeline_parallel && sched_op_offload == cparams.op_offload;

    // Nextn output selection and slot samplers change graph shape, not the
    // immutable expert weights. Destroying their scheduler would silently drop
    // a startup-warmed cache on the first request. Captured graphs must go away
    // before either their tensor metadata or their compute storage is released.
    if (preserve_cache) {
        for (ggml_backend_t backend : backend_ptrs) {
            auto dev = ggml_backend_get_device(backend);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            using reset_fn = bool (*)(ggml_backend_t);
            auto reset = (reset_fn) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
            if (!reset) {
                // Other GPU providers retain their original scheduler lifecycle.
                preserve_cache = false;
                break;
            }
            if (!reset(backend)) {
                throw std::runtime_error("failed to reset CUDA graphs before graph reserve");
            }
        }
    }
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    if (preserve_cache) {
        size_t compute_bytes_before = 0;
        for (ggml_backend_t backend : backend_ptrs) {
            compute_bytes_before += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (!ggml_backend_sched_reset_compute(sched.get())) {
            throw std::runtime_error("failed to release graph compute buffers");
        }
        LLAMA_LOG_INFO("%s: retained expert cache session across graph reserve (rows=%u, node_capacity=%zu, compute_bytes_before=%zu)\n",
                       __func__, n_tokens, sched_node_capacity, compute_bytes_before);
    } else {
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                          node_capacity, cparams.pipeline_parallel, cparams.op_offload));
        sched_node_capacity = node_capacity;
        sched_cache_mode = (int) moe_cache_mode;
        sched_cache_budget_mib = cparams.moe_cache_budget_mib;
        sched_pipeline_parallel = cparams.pipeline_parallel;
        sched_op_offload = cparams.op_offload;
    }
    if (getenv("LLAMA_SHARED_COMPUTE_DEBUG") != nullptr) {
        fprintf(stderr, "shared-compute: sched_reserve peer=%p peer_sched=%p\n", (void *) compute_peer, compute_peer ? (void *) compute_peer->sched.get() : nullptr);
    }
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && compute_peer != nullptr && compute_peer->sched) {
        if (!ggml_backend_sched_set_donor(sched.get(), compute_peer->sched.get())) {
            LLAMA_LOG_WARN("%s: could not share compute buffers with the peer context\n", __func__);
        } else if (const char * env = getenv("LLAMA_SHARED_POOL"); env == nullptr || env[0] != '0') {
            // [FIT12 overhead] also share the backends' scratch pools (op temporaries outside the compute buffer):
            // the two contexts already synchronize each other before every compute, so one pool serves both and its
            // high-water mark is the larger of the two instead of the sum. Default on; LLAMA_SHARED_POOL=0 opts out.
            using share_pool_fn_t = bool (*)(ggml_backend_t, ggml_backend_t);
            for (auto & b : backends) {
                ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
                if (dev == nullptr) {
                    continue;
                }
                auto fn = (share_pool_fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_share_pool");
                if (fn == nullptr) {
                    continue;
                }
                for (auto & pb : compute_peer->backends) {
                    if (ggml_backend_get_device(pb.get()) == dev) {
                        if (!fn(b.get(), pb.get())) {
                            LLAMA_LOG_WARN("%s: could not share the scratch pool of %s with the peer context\n", __func__, ggml_backend_dev_name(dev));
                        }
                        break;
                    }
                }
            }
        }
    }
    if (!preserve_cache) {
        ggml_backend_sched_set_moe_cache(sched.get(), moe_cache_mode, cparams.moe_cache_budget_mib);
        ggml_backend_sched_moe_cache_prefetch_configure(sched.get(), cparams.moe_prefetch);
    }
    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    if (mtp_prefill_async || mtp_scratch_enabled) {
        ggml_backend_sched_set_input_copy_callback(sched.get(), sched_copy_mtp_inputs, this);
    }

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs,
                    n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", __func__);
        auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
            if (!enabled) {
                return;
            }

            const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;

            auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
            }

            bool device_mismatch = false;
            for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
                if (node.op != probe.op) {
                    continue;
                }

                GGML_ASSERT(node.il >= 0);

                ggml_backend_t     backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
                ggml_backend_dev_t device_fused  = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

                ggml_backend_dev_t device_layer = model.dev_layer(node.il);

                if (device_fused != device_layer) {
                    LLAMA_LOG_WARN(
                        "%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, node.il, device_layer ? ggml_backend_dev_name(device_layer) : "none", probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                    device_mismatch = true;
                    break;
                }
            }

            if (device_mismatch) {
                enabled = false;
                LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", __func__, probe.name);
            } else {
                enabled = true;
                LLAMA_LOG_INFO("%s: %s enabled\n", __func__, probe.name);
            }
        };

        resolve(llm_fused_op_dsv4_hc_pre_probe, cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (kv_stream_phase_arena.arena != nullptr) {
        size_t backend_index = SIZE_MAX;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            if (ggml_backend_get_device(backend_ptrs[i]) == kv_stream_phase_arena.device) {
                backend_index = i;
                break;
            }
        }
        if (backend_index == SIZE_MAX) {
            throw std::runtime_error("failed to match phase arena to its CUDA scheduler backend");
        }

        const uint32_t      n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);
        std::vector<size_t> sizes_pp(backend_ptrs.size(), 0);
        std::vector<size_t> sizes_tg(backend_ptrs.size(), 0);

        auto * gf_pp = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, sizes_pp.data());
        if (gf_pp == nullptr) {
            throw std::runtime_error("failed to measure compute pp buffers");
        }
        const int n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        const int n_nodes_pp  = ggml_graph_n_nodes(gf_pp);

        auto * gf_tg = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), true, sizes_tg.data());
        if (gf_tg == nullptr) {
            throw std::runtime_error("failed to measure compute tg buffers");
        }
        const int n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        const int n_nodes_tg  = ggml_graph_n_nodes(gf_tg);

        const uint64_t compute_alignment = ggml_backend_buft_get_alignment(backend_buft[backend_index]);
        const auto     plan_pp           = llama_kv_stream_phase_plan_make({
            /*.arena_bytes        =*/kv_stream_phase_arena.arena_bytes,
            /*.compute_bytes      =*/sizes_pp[backend_index],
            /*.compute_alignment  =*/compute_alignment,
            /*.page_bytes         =*/kv_stream_phase_arena.page_bytes,
            /*.conversion_bytes   =*/kv_stream_phase_arena.conversion_bytes,
            /*.layer_count        =*/kv_stream_phase_arena.layer_count,
            /*.minimum_ring_pages =*/kv_stream_phase_arena.minimum_ring_slots,
        });
        const auto     plan_tg           = llama_kv_stream_phase_plan_make({
            /*.arena_bytes        =*/kv_stream_phase_arena.arena_bytes,
            /*.compute_bytes      =*/sizes_tg[backend_index],
            /*.compute_alignment  =*/compute_alignment,
            /*.page_bytes         =*/kv_stream_phase_arena.page_bytes,
            /*.conversion_bytes   =*/kv_stream_phase_arena.conversion_bytes,
            /*.layer_count        =*/kv_stream_phase_arena.layer_count,
            /*.minimum_ring_pages =*/kv_stream_phase_arena.minimum_ring_slots,
        });
        if (!plan_pp.valid) {
            throw std::runtime_error(std::string("prefill phase does not fit shared CUDA arena: ") + plan_pp.error);
        }
        if (!plan_tg.valid) {
            throw std::runtime_error(std::string("token-generation phase does not fit shared CUDA arena: ") +
                                     plan_tg.error);
        }

        auto make_layout = [](const llama_kv_stream_phase_plan & plan, std::vector<size_t> sizes) {
            kv_stream_phase_arena_owner::layout result;
            result.kv_bytes                 = plan.kv_bytes;
            result.compute_offset           = plan.compute_offset;
            result.compute_bytes            = plan.compute_bytes;
            result.ring_slots               = plan.ring_pages;
            result.resident_pages_per_layer = plan.resident_pages_per_layer;
            result.backend_sizes            = std::move(sizes);
            return result;
        };

        kv_stream_phase_arena.prefill          = make_layout(plan_pp, std::move(sizes_pp));
        kv_stream_phase_arena.token_generation = make_layout(plan_tg, std::move(sizes_tg));
        kv_stream_phase_arena.backend_index    = backend_index;
        kv_stream_phase_arena.max_nodes        = max_nodes;
        kv_stream_phase_arena.configured       = true;

        sched.reset();
        backend_buft[backend_index] = kv_stream_phase_arena.buffer_type;
        if (!kv_stream_switch_phase(false, 0)) {
            throw std::runtime_error("failed to activate shared CUDA arena prefill phase");
        }

        LLAMA_LOG_INFO(
            "%s: phase arena prefill: KV %.2f MiB, compute %.2f MiB, resident %u pages/layer, ring %u pages\n",
            __func__, kv_stream_phase_arena.prefill.kv_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.prefill.compute_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.prefill.resident_pages_per_layer, kv_stream_phase_arena.prefill.ring_slots);
        LLAMA_LOG_INFO(
            "%s: phase arena decode:  KV %.2f MiB, compute %.2f MiB, resident %u pages/layer, ring %u pages\n",
            __func__, kv_stream_phase_arena.token_generation.kv_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.token_generation.compute_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.token_generation.resident_pages_per_layer,
            kv_stream_phase_arena.token_generation.ring_slots);

        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens,
                           n_nodes_tg);
        }
        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens,
                           n_splits_tg);
        }

        const int64_t t_end_us = ggml_time_us();
        LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (t_end_us - t_start_us) / 1000.0,
                       ggml_backend_sched_get_n_copies(sched.get()));
        return;
    }

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const bool reserve_mtp_tight = (mtp_prefill_async || mtp_reserve_tight) && !cparams.embeddings &&
        (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    const uint32_t n_outputs_pp = reserve_mtp_tight ? 0 : std::min(n_tokens, cparams.n_outputs_max);

    if (preserve_cache) {
        std::vector<size_t> required(backend_ptrs.size());
        if (!graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, required.data())) {
            throw std::runtime_error("failed to measure turn compute buffers");
        }
        size_t trim = 0;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            auto dev = ggml_backend_get_device(backend_ptrs[i]);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            const size_t card_limit = (size_t) 23552 << 20;
            const size_t headroom = std::max<size_t>(256u << 20, total > card_limit ? total - card_limit + (64u << 20) : 0);
            const size_t need = required[i] + headroom;
            trim += need > free ? need - free : 0;
        }
        if (trim > 0) {
            b1_cache_needs_refresh = true;
            const size_t released = ggml_backend_sched_moe_cache_trim(sched.get(), trim);
            LLAMA_LOG_INFO("%s: turn scratch borrowed %.2f MiB (requested %.2f MiB)\n", __func__, released / 1048576.0, trim / 1048576.0);
        }
    }

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc,
                                  model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n",
                               __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                   node_capacity, false, cparams.op_offload));
                sched_node_capacity = node_capacity;
                sched_pipeline_parallel = false;
                ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
                if (mtp_prefill_async || mtp_scratch_enabled) {
                    ggml_backend_sched_set_input_copy_callback(sched.get(), sched_copy_mtp_inputs, this);
                }
                ggml_backend_sched_set_moe_cache(sched.get(), moe_cache_mode, cparams.moe_cache_budget_mib);
                ggml_backend_sched_moe_cache_prefetch_configure(sched.get(), cparams.moe_prefetch);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        // fork: common/speculative.cpp defers at most 64 catch-up rows before a generation batch.
        const uint64_t n_mtp_rows = 64ull + std::max(n_seqs, cparams.n_outputs_max);
        const uint32_t n_tg = reserve_mtp_tight ? std::max(n_seqs, uint32_t(std::min<uint64_t>(n_tokens, n_mtp_rows))) : n_seqs;
        const uint32_t n_outputs_tg = reserve_mtp_tight ? std::min(n_tg, cparams.n_outputs_max) : n_tg;
        std::vector<size_t> tg_sizes(reserve_mtp_tight && model.hparams.no_alloc ? backend_ptrs.size() : 0);
        auto * gf = graph_reserve(n_tg, n_seqs, n_outputs_tg, mctx.get(), model.hparams.no_alloc,
                tg_sizes.empty() ? nullptr : tg_sizes.data());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }
        for (size_t i = 0; i < tg_sizes.size(); ++i) {
            backend_buf_exp_size[i] = std::max(backend_buf_exp_size[i], tg_sizes[i]);
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            backend_buf_exp_epoch   = ggml_backend_sched_get_alloc_epoch(sched.get());
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft),
                           backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (t_end_us - t_start_us) / 1000.0,
                   ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    auto * perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    const llama_bookkeep_context_timer timer(perf ? &perf->sync_calls : nullptr, perf ? &perf->sync_host_us : nullptr);
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us          = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens    = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

// Resolve the effective K/V cache type of a memory object.
//
// llama_memory_i has many concrete implementations (plain KV cache, iSWA,
// DSA, DSV4, MSA, recurrent, hybrid, hybrid-iSWA, ...) and none of them
// carry a common "type_k()/type_v()" accessor on the base interface, so we
// have to downcast to figure out which one we're holding. This mirrors how
// llama-model.cpp::create_memory() picks the concrete type in the first
// place - see the switch over `arch` there.
//
// `want_k` selects K (true) or V (false). Returns GGML_TYPE_COUNT when the
// memory object has no single representative K/V type (pure recurrent
// memory, or a composite cache like DSV4 whose sub-caches - raw token
// attention, CSA/HCA compressed blocks, lightning-indexer keys - are
// configured independently and need not share a single type. Note that
// auto-asymmetric is not the reason here: it skips MLA architectures, DSV4
// included, so it never rewrites these.
static enum ggml_type llama_memory_get_kv_type(const llama_memory_i * mem, bool want_k) {
    if (!mem) {
        return GGML_TYPE_COUNT;
    }

    // plain unified/non-SWA KV cache
    if (const auto * kv = dynamic_cast<const llama_kv_cache *>(mem)) {
        return want_k ? kv->type_k() : kv->type_v();
    }

    // interleaved-SWA cache: base and swa sub-caches are built with the same
    // requested type_k/type_v and the same hparams-derived GQA ratio, so any
    // auto-asymmetric rewrite is consistent between them - report the base.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_iswa *>(mem)) {
        const llama_kv_cache * base = kv->get_base();
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // DeepSeek sparse-attention (DSA): the MLA cache is the real K/V cache
    // (MLA models are always symmetric and skip auto-asymmetric); the
    // lightning-indexer cache is a separate, unrelated key-only cache.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_dsa *>(mem)) {
        const llama_kv_cache * mla = kv->get_mla();
        return mla ? (want_k ? mla->type_k() : mla->type_v()) : GGML_TYPE_COUNT;
    }

    // MiniMax-style sparse attention (MSA): report the base attention cache,
    // not the separate indexer key cache.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_msa *>(mem)) {
        const llama_kv_cache * base = kv->get_base();
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // DSV4: raw token attention (iSWA), plus CSA/HCA compressed block caches
    // and a lightning-indexer key cache. These are independent llama_kv_cache
    // instances configured independently, so there is no single type that
    // represents "the" cache - be honest about it rather than picking one.
    if (dynamic_cast<const llama_kv_cache_dsv4 *>(mem)) {
        return GGML_TYPE_COUNT;
    }

    // hybrid (recurrent + attention): report the attention sub-cache; the
    // recurrent sub-cache has no K/V concept (type_r/type_s instead).
    if (const auto * hy = dynamic_cast<const llama_memory_hybrid *>(mem)) {
        const llama_kv_cache * attn = hy->get_mem_attn();
        return attn ? (want_k ? attn->type_k() : attn->type_v()) : GGML_TYPE_COUNT;
    }

    if (const auto * hy = dynamic_cast<const llama_memory_hybrid_iswa *>(mem)) {
        const llama_kv_cache_iswa * attn = hy->get_mem_attn();
        const llama_kv_cache *      base = attn ? attn->get_base() : nullptr;
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // pure recurrent memory (Mamba/RWKV-style): no K/V cache at all.
    if (dynamic_cast<const llama_memory_recurrent *>(mem)) {
        return GGML_TYPE_COUNT;
    }

    // unknown memory implementation - don't guess.
    return GGML_TYPE_COUNT;
}

static llama_kv_cache * llama_sj_kvarn_cache(llama_memory_i * memory) {
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) { return kv; }
    if (auto * hy = dynamic_cast<llama_memory_hybrid *>(memory)) { return hy->get_mem_attn(); }
    return nullptr;
}

const llama_sj_kvarn_config * llama_context::get_sj_kvarn() const {
    const auto * kv = llama_sj_kvarn_cache(memory.get());
    return kv && kv->is_sj_kvarn() ? &kv->get_sj_kvarn() : nullptr;
}

ggml_backend_t llama_context::sj_kvarn_backend(ggml_backend_buffer_t buffer) const {
    const auto buft = ggml_backend_buffer_get_type(buffer);
    const auto device = ggml_backend_buft_get_device(buft);
    for (const auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == device && ggml_backend_supports_buft(backend.get(), buft)) {
            return backend.get();
        }
    }
    // host-resident body (CPU-only builds, -ngl 0): the CPU backend can compute on any host buffer
    if (backend_cpu && ggml_backend_buffer_is_host(buffer)) {
        return backend_cpu;
    }
    LLAMA_LOG_ERROR("%s: no backend for buffer type %s (device %s)\n", __func__, ggml_backend_buft_name(buft), device ? ggml_backend_dev_name(device) : "null");
    return nullptr;
}

bool llama_context::maintain_sj_kvarn() {
    auto * kv = llama_sj_kvarn_cache(memory.get());
    if (!kv || !kv->is_sj_kvarn() || !kv->has_sj_kvarn_maintenance()) { return true; }
    if (compute_is_shared() && compute_peer->sched) { ggml_backend_sched_synchronize(compute_peer->sched.get()); }
    return kv->maintain_sj_kvarn(this);
}

bool llama_context::b1_seal_sj_kvarn(int32_t il0, int32_t il1) {
    auto * kv = llama_sj_kvarn_cache(memory.get());
    if (!kv || !kv->is_sj_kvarn()) { return true; }
    return kv->b1_sj_seal(this, il0, il1);
}

int32_t llama_context::compress_sj_kvarn_idle(llama_seq_id seq_id, llama_pos accepted_end, llama_pos keep_from) {
    auto * kv = llama_sj_kvarn_cache(memory.get());
    if (!kv || !kv->is_sj_kvarn()) { return -1; }
    if (compute_is_shared() && compute_peer->sched) { ggml_backend_sched_synchronize(compute_peer->sched.get()); }
    return kv->compress_sj_kvarn_idle(this, seq_id, accepted_end, keep_from);
}

int32_t llama_sj_kvarn_sealed_end(llama_context * ctx) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    return kv && kv->is_sj_kvarn() ? (int32_t) kv->get_sj_kvarn_sealed_end() : -1;
}

int32_t llama_sj_kvarn_seq_unsealed_max(llama_context * ctx) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    return kv && kv->is_sj_kvarn() ? (int32_t) (kv->get_sj_kvarn_sink() + kv->get_sj_kvarn_capacity()) : -1;
}

int32_t llama_sj_kvarn_seq_sealed_end(llama_context * ctx, llama_seq_id seq_id) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    if (!kv || !kv->is_sj_kvarn() || seq_id < 0 || (uint32_t) seq_id >= kv->get_n_stream()) {
        return -1;
    }
    return (int32_t) kv->get_sj_kvarn_sealed_end(seq_id);
}

int32_t llama_sj_kvarn_compress_idle(llama_context * ctx, llama_seq_id seq_id, llama_pos accepted_end) {
    return ctx ? ctx->compress_sj_kvarn_idle(seq_id, accepted_end) : -1;
}

int32_t llama_sj_kvarn_compress_idle_keep(llama_context * ctx, llama_seq_id seq_id, llama_pos accepted_end, llama_pos keep_from) {
    return ctx ? ctx->compress_sj_kvarn_idle(seq_id, accepted_end, keep_from) : -1;
}

llama_pos llama_sj_kvarn_rm_floor(llama_context * ctx, llama_seq_id seq_id, llama_pos pos) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    return kv && kv->is_sj_kvarn() ? kv->sj_kvarn_rm_floor(seq_id, pos) : pos;
}

llama_pos llama_sj_kvarn_group_floor(llama_context * ctx, llama_pos pos) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    if (!kv || !kv->is_sj_kvarn() || pos < 0) { return pos; }
    const auto & c = kv->get_sj_kvarn();
    if ((uint32_t) pos <= c.sink) { return pos; }
    return (llama_pos) (c.sink + c.group*(((uint32_t) pos - c.sink)/c.group));
}

bool llama_sj_kvarn_prefill_cut(llama_context * ctx, llama_seq_id seq_id, llama_pos pos) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    if (!kv || !kv->is_sj_kvarn() || pos <= 0) { return false; }
    return kv->sj_kvarn_prefill_cut(seq_id, pos);
}

llama_pos llama_sj_kvarn_ckpt_floor(llama_context * ctx, llama_pos pos) {
    auto * kv = ctx ? llama_sj_kvarn_cache(ctx->get_memory()) : nullptr;
    if (!kv || !kv->is_sj_kvarn() || pos < 0) { return pos; }
    const uint32_t grid = kv->get_sj_kvarn_pos_grid();
    if (grid > 0) {
        pos -= pos % (llama_pos) grid;
    }
    return llama_sj_kvarn_group_floor(ctx, pos);
}

enum ggml_type llama_context::get_kv_type_k() const {
    return llama_memory_get_kv_type(memory.get(), /* want_k */ true);
}

enum ggml_type llama_context::get_kv_type_v() const {
    return llama_memory_get_kv_type(memory.get(), /* want_k */ false);
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                }
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs   = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const bool reserve_kv_only = (mtp_reserve_tight || mtp_prefill_async) && !cparams.embeddings &&
                (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
        const uint32_t n_outputs_max = reserve_kv_only ? 0 : std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens() const {
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t  j          = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j * n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();
    if (output_file) { output_file->flush(); }

    return embd_nextn.data;
}

bool llama_context::copy_embeddings_nextn(float * dst, int32_t first, int32_t count) {
    auto * perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    const llama_bookkeep_context_timer timer(perf ? &perf->copy_calls : nullptr, perf ? &perf->copy_host_us : nullptr);
    const size_t width = model.hparams.n_embd_out();
    if (cparams.embeddings_nextn_masked || first < 0 || count < 0 || !width || !embd_nextn.data ||
            (count && !dst) || (size_t) first > embd_nextn.size / width ||
            (size_t) count > embd_nextn.size / width - first) { return false; }
    const float * data = get_embeddings_nextn();
    if (count) {
        const auto * src = data + (size_t) first * width;
        const size_t bytes = (size_t) count * width * sizeof(float);
        if (output_file) {
            const uintptr_t base = (uintptr_t) output_file->data;
            const uintptr_t begin = (uintptr_t) src;
            if (begin < base || begin - base > output_file->size || bytes > output_file->size - (begin - base)) {
                return false;
            }
            static const bool readahead = [] {
                const char * v = getenv("LLAMA_MOE_OUTPUT_READAHEAD");
                return v && atoi(v) != 0;
            }();
            if (readahead) {
                const size_t offset = begin - base;
                // current range first (no-op when cached), then the next chunk of the same size
                output_file->readahead(offset, bytes);
                output_file->readahead(offset + bytes, bytes);
            }
            llama_copy_moe_output(dst, src, bytes);
        } else {
            memcpy(dst, src, bytes);
        }
        if (perf) {
            perf->copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    return true;
}

bool llama_context::release_embeddings_nextn(int32_t first, int32_t count) {
    const char * reclaim = getenv("LLAMA_MOE_OUTPUT_RECLAIM");
    if (!output_file || !reclaim || strcmp(reclaim, "1") != 0) { return true; }
    const size_t width = model.hparams.n_embd_out();
    if (cparams.embeddings_nextn_masked || first < 0 || count < 0 || !width ||
            (size_t) first > embd_nextn.size / width || (size_t) count > embd_nextn.size / width - first) {
        LLAMA_LOG_ERROR("%s: invalid unmasked output range\n", __func__);
        return false;
    }
    // Count only the validated file-backed reclaim path, not disabled/invalid early returns.
    auto * perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    const llama_bookkeep_context_timer timer(perf ? &perf->release_calls : nullptr, perf ? &perf->release_host_us : nullptr);
    synchronize();
    output_reorder();
    try {
        const size_t begin = (const char *) (embd_nextn.data + (size_t) first * width) - (const char *) output_file->data;
        output_file->release(begin, (size_t) count * width * sizeof(float));
        LLAMA_LOG_DEBUG("%s: released consumed MTP output rows=%d bytes=%zu\n", __func__, count, (size_t) count * width * sizeof(float));
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();
    if (output_file) { output_file->flush(); }

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t) (i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j * n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() && (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row * model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

llama_sampler_output llama_context::get_sampler_output_ith(int32_t idx) {
    llama_sampler_output result;
    result.token = get_sampled_token_ith(idx);
    result.probs = get_sampled_probs_ith(idx);
    result.logits = get_sampled_logits_ith(idx);
    result.candidates = get_sampled_candidates_ith(idx);
    if (result.probs) {
        result.probs_count = static_cast<uint32_t>(get_sampled_probs_count(idx));
    }
    if (result.logits) {
        result.logits_count = static_cast<uint32_t>(get_sampled_logits_count(idx));
    }
    if (result.probs || result.logits) {
        result.candidates_count = static_cast<uint32_t>(get_sampled_candidates_count(idx));
    }
    if (result.token == LLAMA_TOKEN_NULL && !result.probs && !result.logits) {
        result.raw_logits = get_logits_ith(idx);
        result.raw_logits_count = result.raw_logits ? model.vocab.n_tokens() : 0;
    }
    return result;
}

void llama_context::attach_threadpool(ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_decode_phase(enum llama_decode_phase phase) {
    if (phase == LLAMA_DECODE_PHASE_GENERATION) {
        const char * rotate = getenv("LLAMA_MOE_CACHE_RELEASE_PROMPT");
        if (rotate && rotate[0] == '1') {
            synchronize();
            if (b1 && b1->dirty() && !b1->refill()) {
                throw std::runtime_error("failed to restore prompt expert lending");
            }
            if (!prompt_buffers_resize(0, nullptr)) {
                throw std::runtime_error("failed to return prompt scratch");
            }
            for (ggml_backend_t backend : backend_ptrs) {
                auto dev = ggml_backend_get_device(backend);
                if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                    continue;
                }
                using trim_fn = size_t (*)(ggml_backend_t);
                auto trim = (trim_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_pool_trim");
                if (trim) {
                    using reset_fn = bool (*)(ggml_backend_t);
                    auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
                    if (reset && reset(backend)) {
                        const size_t released = trim(backend);
                        LLAMA_LOG_INFO("%s: returned %.2f MiB of unused CUDA temporary storage\n", __func__, released / 1048576.0);
                        b1_cache_needs_refresh = b1_cache_needs_refresh || released > 0;
                    }
                }
            }
            if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
                b1_cache_warm();
            } else if (b1_cache_needs_refresh) {
                // Restore resets scheduler metadata, including rewritten source copies.
                // A prior graph must be rebuilt before that arena is reused.
                for (auto & res : gf_res_prev) {
                    if (res) {
                        res->reset();
                    }
                }
                gf_res_prev_active = nullptr;
                ggml_backend_sched_moe_cache_restore(sched.get());
                b1_cache_needs_refresh = false;
            }
        }
    }
    if (phase == LLAMA_DECODE_PHASE_GENERATION) {
        b1_cache_profile_load();
    }
    switch (phase) {
        case LLAMA_DECODE_PHASE_AUTOMATIC:
        case LLAMA_DECODE_PHASE_PROMPT:
        case LLAMA_DECODE_PHASE_GENERATION:
            decode_phase = phase;
            return;
    }
    decode_phase = LLAMA_DECODE_PHASE_AUTOMATIC;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // Reserve the initial nextn output. Decode grows it to the input row count.
    // note: like any reservation this invalidates the outputs of the previous batch
    output_reserve_grow();
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;

    output_reserve_grow();
}

void llama_context::output_reserve_grow() {
    if (!output_buffer_reuse || !buf_output) {
        return; // constructor has not reserved yet
    }

    if (output_reserve(n_outputs_reserved) < n_outputs_reserved) {
        throw std::runtime_error("failed to grow the output buffer");
    }
}

// fork: Copy contiguous row spans without creating graph nodes or retaining graph-owned tensors.
static ggml_tensor mtp_row_view(const ggml_tensor * tensor, int64_t first, int64_t count) {
    GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    GGML_ASSERT(first >= 0 && count > 0 && first <= tensor->ne[1] - count);
    ggml_tensor view = *tensor;
    view.ne[1] = count;
    view.nb[2] = view.nb[1] * count;
    view.nb[3] = view.nb[2];
    view.data = (char *) tensor->data + first * tensor->nb[1];
    view.view_src = nullptr;
    return view;
}

bool llama_context::mtp_prefill_init() {
    const char * device = getenv("GGML_MTP_PREFILL_DEVICE");
    const char * async = getenv("GGML_MTP_PREFILL_ASYNC");
    const char * scratch = getenv("LLAMA_MTP_PREFILL_SCRATCH");
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(model.dev_output(), &props);
    mtp_hidden.enabled = !(scratch && strcmp(scratch, "1") == 0) &&
        (props.type == GGML_BACKEND_DEVICE_TYPE_CPU || props.caps.events) &&
        ((device && strcmp(device, "1") == 0) || (async && strcmp(async, "1") == 0)) &&
        model.arch == LLM_ARCH_QWEN35 &&
        model.hparams.n_layer_nextn == 1 && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        model.n_devices() <= 1 && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
        cparams.n_batch <= (uint32_t) INT32_MAX - cparams.n_seq_max;
    return mtp_hidden.enabled;
}

bool llama_context::compute_is_shared() const {
    return compute_peer && !mtp_scratch_active && !compute_peer->mtp_scratch_active;
}

static bool mtp_reset_backend_scratch(ggml_backend_t backend) {
    auto * dev = ggml_backend_get_device(backend);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }
    using reset_fn_t = bool (*)(ggml_backend_t);
    auto reset = (reset_fn_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_reset_scratch");
    return reset && reset(backend);
}

void llama_context::mtp_scratch_end() {
    if (!mtp_scratch_cache) {
        return;
    }
    ggml_backend_sched_synchronize(sched.get());
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;
    gf_res_reserve->reset();
    ggml_backend_sched_reset(sched.get());
    GGML_ASSERT(mtp_reset_backend_scratch(mtp_scratch_backend));
    GGML_ASSERT(ggml_backend_sched_set_buffer(sched.get(), mtp_scratch_backend, nullptr));
    // Attention can read masked padding before it applies the mask. Remove scratch NaNs from unused KV.
    for (auto & view : mtp_scratch_views) {
        ggml_backend_buffer_clear(view.get(), 0);
    }
    const char * shared = getenv("LLAMA_SHARED_POOL");
    if (compute_peer && (!shared || shared[0] != '0')) {
        auto * dev = ggml_backend_get_device(mtp_scratch_backend);
        using share_fn_t = bool (*)(ggml_backend_t, ggml_backend_t);
        auto share = (share_fn_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_share_pool");
        for (auto & peer : compute_peer->backends) {
            if (share && ggml_backend_get_device(peer.get()) == dev) {
                GGML_ASSERT(share(mtp_scratch_backend, peer.get()));
                break;
            }
        }
    }
    mtp_scratch_cache->scratch_release();
    mtp_scratch_views.clear();
    mtp_scratch_cache = nullptr;
    mtp_scratch_backend = nullptr;
    mtp_scratch_active = false;
    LLAMA_LOG_INFO("%s: released MTP KV scratch; shared serialized compute restored\n", __func__);
}

void llama_context::mtp_scratch_prepare(bool kv_only, const llama_memory_context_i * mctx, const llama_ubatch * ubatch) {
    kv_only = kv_only && !cparams.embeddings && !cparams.mtp_chain && cparams.embeddings_nextn_masked;
    if (mtp_scratch_active && (!kv_only || sched_need_reserve)) {
        mtp_scratch_end();
    }
    if (mtp_scratch_enabled && kv_only) {
        ggml_backend_sched_synchronize(sched.get());
        mtp_input_upload_next = 0;
    }
    if (!mtp_scratch_enabled || mtp_scratch_active || !kv_only || cparams.embeddings || cparams.mtp_chain ||
            !cparams.embeddings_nextn_masked || model.hparams.no_alloc || !mtp_scratch_target ||
            (compute_peer && compute_peer != mtp_scratch_target) || !mctx || !ubatch) {
        return;
    }
    auto * cache = llama_sj_kvarn_cache(mtp_scratch_target->memory.get());
    if (!cache) {
        return;
    }
    size_t index = SIZE_MAX;
    for (size_t i = 0; i < backends.size(); ++i) {
        if (ggml_backend_get_device(backends[i].get()) == model.dev_output()) {
            index = i;
            break;
        }
    }
    if (index == SIZE_MAX) {
        return;
    }
    auto * backend = backends[index].get();
    auto * buft = backend_buft[index];
    auto * dev = ggml_backend_get_device(backend);
    const auto type = ggml_backend_dev_type(dev);
    using requirements_fn_t = bool (*)(ggml_backend_t, const ggml_cgraph *, size_t *, size_t *);
    using set_pool_fn_t = bool (*)(ggml_backend_t, ggml_backend_buffer_t *, int);
    using preflight_fn_t = bool (*)(ggml_backend_t, const ggml_cgraph *);
    requirements_fn_t requirements = nullptr;
    set_pool_fn_t set_pool = nullptr;
    preflight_fn_t preflight = nullptr;
    if (type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        auto * reg = ggml_backend_dev_backend_reg(dev);
        requirements = (requirements_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_scratch_requirements");
        set_pool = (set_pool_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_scratch_buffers");
        preflight = (preflight_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_scratch_preflight");
        if (!requirements || !set_pool || !preflight) {
            return;
        }
    }
    // The full-cache reserve repools every indexer block. Size loans after apply(), for this actual batch.
    const uint32_t rows = ubatch->n_tokens;
    const uint32_t n_seqs = ubatch->n_seqs;
    std::vector<size_t> sizes(backends.size());
    auto * gf = graph_reserve(rows, n_seqs, 0, mctx, true, sizes.data());
    if (!gf || !sizes[index]) {
        return;
    }
    size_t pool_bytes = 0, pool_largest = 0;
    if (requirements && !requirements(backend, gf, &pool_bytes, &pool_largest)) {
        return;
    }
    size_t chunk_bytes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (!node->view_src && !node->data && ggml_backend_sched_get_tensor_backend(sched.get(), node) == backend) {
            chunk_bytes = std::max(chunk_bytes, ggml_backend_buft_get_alloc_size(buft, node));
        }
    }
    if (!chunk_bytes) {
        return;
    }
    if (pool_bytes) {
        chunk_bytes = std::max<size_t>(chunk_bytes, pool_largest + 4ull * 1024 * 1024);
    }
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    chunk_bytes = GGML_PAD(chunk_bytes, alignment);
    const size_t compute_chunks = (sizes[index] + chunk_bytes - 1) / chunk_bytes + 2;
    const size_t pool_chunks = pool_bytes ? (pool_bytes + chunk_bytes - 1) / chunk_bytes + 1 : 0;
    const size_t chunks = compute_chunks + pool_chunks;
    if (chunks > 16) {
        return;
    }
    const size_t floor = type == GGML_BACKEND_DEVICE_TYPE_CPU ? 0 : 500ull * 1024 * 1024;
    ggml_backend_sched_synchronize(mtp_scratch_target->sched.get());
    auto views = cache->scratch_borrow(buft, chunk_bytes, chunks, floor,
            2 * mtp_scratch_target->cparams.n_ubatch,
            [](void * user) { static_cast<llama_context *>(user)->mtp_scratch_end(); }, this);
    // A resident QSA body is host-backed. Its indexer has an ordinary device KV suffix instead.
    if (views.size() < chunks) {
        cache->scratch_release();
        views.clear();
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid_idx *>(mtp_scratch_target->memory.get())) {
            cache = hybrid->get_mem_idx();
            if (cache) {
                views = cache->scratch_borrow(buft, chunk_bytes, chunks, floor,
                        2 * mtp_scratch_target->cparams.n_ubatch,
                        [](void * user) { static_cast<llama_context *>(user)->mtp_scratch_end(); }, this);
            }
        }
    }
    if (views.empty()) {
        return;
    }
    if (views.size() < chunks) {
        cache->scratch_release();
        return;
    }
    mtp_scratch_cache = cache;
    mtp_scratch_backend = backend;
    mtp_scratch_views = std::move(views);
    std::vector<ggml_backend_buffer_t> buffers;
    for (const auto & view : mtp_scratch_views) {
        buffers.push_back(view.get());
    }
    ggml_backend_sched_reset(sched.get());
    if (!mtp_reset_backend_scratch(backend) ||
            (set_pool && !set_pool(backend, buffers.data() + compute_chunks, pool_chunks)) ||
            !ggml_backend_sched_set_buffers(sched.get(), backend, buffers.data(), compute_chunks) ||
            !graph_reserve(rows, n_seqs, 0, mctx)) {
        mtp_scratch_end();
        return;
    }
    if (preflight && !preflight(backend, gf_res_reserve->get_gf())) {
        mtp_scratch_end();
        return;
    }
    mtp_scratch_active = true;
    if (mtp_input_upload_done) {
        ggml_backend_event_synchronize(mtp_input_upload_done.get());
    }
    mtp_input_upload_next = 0;
    LLAMA_LOG_INFO("%s: MTP prefill compute = %.3f MiB, backend scratch = %.3f MiB in borrowed KV, device allocation = 0 bytes\n", __func__,
            ggml_backend_sched_get_buffer_size(sched.get(), backend) / 1048576.0, pool_chunks * chunk_bytes / 1048576.0);
}

void llama_context::mtp_prefill_reset(llama_seq_id seq) {
    if (seq >= 0 && (size_t) seq < mtp_hidden.carry_pos.size()) {
        mtp_hidden.carry_pos[seq] = -2;
    }
}

void llama_context::mtp_prefill_capture(ggml_tensor * hidden, const llama_ubatch & ubatch) {
    auto & stage = mtp_hidden;
    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), hidden);
    GGML_ASSERT(backend && ubatch.data && hidden->ne[1] == ubatch.n_tokens);
    if (!stage.rows) {
        stage.ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!stage.ctx) {
            throw std::runtime_error("failed to create MTP hidden staging context");
        }
        stage.rows = ggml_new_tensor_2d(stage.ctx.get(), GGML_TYPE_F32, hidden->ne[0],
                (int64_t) cparams.n_batch + cparams.n_seq_max);
        stage.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(stage.ctx.get(), ggml_backend_get_default_buffer_type(backend)));
        if (!stage.buf) {
            throw std::runtime_error("failed to allocate MTP hidden staging buffer");
        }
        stage.backend = backend;
        auto * dev = ggml_backend_get_device(backend);
        stage.ready.reset(ggml_backend_event_new(dev));
        stage.consumed.reset(ggml_backend_event_new(dev));
        stage.carry_pos.assign(cparams.n_seq_max, -2);
        LLAMA_LOG_INFO("%s: MTP device hidden staging = %.3f MiB\n", __func__,
                ggml_backend_buffer_get_size(stage.buf.get()) / (1024.0 * 1024.0));
    }
    GGML_ASSERT(stage.backend == backend);
    if (stage.consuming) {
        if (stage.consumed) {
            ggml_backend_event_wait(backend, stage.consumed.get());
        }
        stage.consuming = false;
    }
    const auto & idxs = ubatch.data->batch_idxs;
    for (uint32_t i = 0; i < ubatch.n_tokens;) {
        const uint32_t first = i;
        while (++i < ubatch.n_tokens && idxs[i] == idxs[first] + (int32_t) (i - first)) {}
        auto src = mtp_row_view(hidden, first, i - first);
        auto dst = mtp_row_view(stage.rows, idxs[first], i - first);
        ggml_backend_tensor_copy_async(backend, backend, &src, &dst);
        for (uint32_t j = first; j < i; ++j) {
            GGML_ASSERT((size_t) idxs[j] < stage.captured.size());
            stage.captured[idxs[j]] = true;
        }
    }
}

void llama_context::mtp_prefill_input(llm_graph_input_embd_h & input, const llama_ubatch & ubatch) {
    auto & source = mtp_hidden_source->mtp_hidden;
    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), input.h);
    GGML_ASSERT(backend && ubatch.data);

    GGML_ASSERT(ggml_backend_get_device(backend) == ggml_backend_get_device(source.backend));
    if (source.ready) {
        ggml_backend_event_wait(backend, source.ready.get());
    }
    const auto & idxs = ubatch.data->batch_idxs;
    for (uint32_t i = 0; i < ubatch.n_tokens;) {
        const uint32_t first = i;
        const int32_t row = mtp_hidden_rows.at(idxs[first]);
        while (++i < ubatch.n_tokens && mtp_hidden_rows.at(idxs[i]) == row + (int32_t) (i - first)) {}
        auto src = mtp_row_view(source.rows, row, i - first);
        auto dst = mtp_row_view(input.h, first, i - first);
        // Both allocations are on this device; copy on the consumer stream after its previous graph.
        ggml_backend_tensor_copy_async(backend, backend, &src, &dst);
    }
}

void llama_context::mtp_prefill_place(llm_graph_result & res) {
    for (auto & input : res.inputs) {
        if (auto * attn = dynamic_cast<llm_graph_input_attn_kv *>(input.get())) {
            attn->skip_unused = true;
        }
        if (auto * h = dynamic_cast<llm_graph_input_embd_h *>(input.get())) {
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == model.dev_output()) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), h->h, backend.get());
                    res.mtp_device_input = true;
                    break;
                }
            }
        }
    }
}

int llama_context::mtp_prefill_decode(llama_context & target, llama_batch batch, const float * const * carry, uint32_t n_seq) {
    auto & source = target.mtp_hidden;
    if (!source.enabled || !source.rows || &target.model != &model || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
            cparams.embeddings || cparams.mtp_chain || !cparams.embeddings_nextn_masked ||
            batch.n_tokens <= 0 || !batch.token || !batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !carry || n_seq == 0 ||
            source.captured.size() != (size_t) batch.n_tokens || n_seq > source.carry_pos.size() ||
            !std::all_of(source.captured.begin(), source.captured.end(), [](bool value) { return value; })) {
        return -4;
    }
    ggml_backend_t consumer = nullptr;
    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == model.dev_output()) {
            consumer = backend.get();
        }
    }
    if (!consumer || ggml_backend_get_device(consumer) != ggml_backend_get_device(source.backend)) {
        return -4;
    }
    if (compute_peer) {
        // The serialized device-only arm must finish the donor graph before changing its allocations.
        target.synchronize();
    }
    if (mtp_input_upload_done) {
        ggml_backend_event_synchronize(mtp_input_upload_done.get());
    }
    mtp_input_upload_next = 0;
    std::vector<int32_t> last(n_seq, -1);
    mtp_hidden_rows.resize(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr) {
            return -4;
        }
        const llama_seq_id seq = batch.seq_id[i][0];
        if (seq < 0 || (uint32_t) seq >= n_seq || !carry[seq]) {
            return -4;
        }
        mtp_hidden_rows[i] = last[seq] < 0 ? target.cparams.n_batch + seq : last[seq];
        last[seq] = i;
    }
    std::vector<bool> seen(n_seq, false);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id seq = batch.seq_id[i][0];
        if (!seen[seq] && (batch.pos[i] == 0 || source.carry_pos[seq] != batch.pos[i] - 1)) {
            // A new prompt or restored sidecar seeds only its boundary row; prompt rows stay on device.
            ggml_backend_synchronize(source.backend);
            ggml_backend_tensor_set(source.rows, carry[seq],
                    (target.cparams.n_batch + seq) * source.rows->nb[1], source.rows->nb[1]);
        }
        seen[seq] = true;
    }
    if (source.ready) {
        ggml_backend_event_record(source.ready.get(), source.backend);
    }
    auto finish = [&]() {
        if (source.consumed) {
            ggml_backend_event_record(source.consumed.get(), consumer);
            source.consuming = true;
        }
        if (!mtp_input_upload_done) {
            mtp_input_upload_done.reset(ggml_backend_event_new(ggml_backend_get_device(consumer)));
        }
        if (mtp_input_upload_done) {
            ggml_backend_event_record(mtp_input_upload_done.get(), consumer);
        }
    };
    mtp_hidden_source = &target;
    int result;
    try {
        result = decode(batch);
    } catch (...) {
        mtp_hidden_source = nullptr;
        std::fill(source.carry_pos.begin(), source.carry_pos.end(), -2);
        finish();
        throw;
    }
    mtp_hidden_source = nullptr;
    if (result == 0) {
        for (uint32_t seq = 0; seq < n_seq; ++seq) {
            if (last[seq] < 0) {
                continue;
            }
            auto src = mtp_row_view(source.rows, last[seq], 1);
            auto dst = mtp_row_view(source.rows, target.cparams.n_batch + seq, 1);
            ggml_backend_tensor_copy_async(consumer, consumer, &src, &dst);
            source.carry_pos[seq] = batch.pos[last[seq]];
        }
    } else {
        std::fill(source.carry_pos.begin(), source.carry_pos.end(), -2);
    }
    finish();
    return result;
}

// [PREFILLFAST] LLAMA_MTP_B1_HANDOFF=1: the target stages one ubatch of hidden rows on its device (plus one carry row) and
// calls back after the ubatch, so the helper's catch-up reads them there instead of from the host output file.
bool llama_context::mtp_handoff_init() {
    // default on (owner ruling 2026-10-21, PREFILLFAST/RESULT.md); LLAMA_MTP_B1_HANDOFF=0 turns it off
    const char * env = getenv("LLAMA_MTP_B1_HANDOFF");
    if (env && strcmp(env, "0") == 0) {
        return false;
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(model.dev_output(), &props);
    mtp_hidden.ring = !mtp_hidden.enabled && props.type != GGML_BACKEND_DEVICE_TYPE_CPU && props.caps.events &&
        (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN4EXP) &&
        // the helper head may live in its own GGUF (n_layer_nextn 0 here); the helper checks the row width per call
        model.hparams.n_layer_nextn <= 1 && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        model.n_devices() <= 1 && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR;
    LLAMA_LOG_WARN("%s: LLAMA_MTP_B1_HANDOFF=1: per-ubatch device handoff %s (staged %d, cpu %d, events %d, arch ok %d, nextn %u, type %d, devices %d)\n",
            __func__, mtp_hidden.ring ? "on" : "off", (int) mtp_hidden.enabled, (int) (props.type == GGML_BACKEND_DEVICE_TYPE_CPU),
            (int) props.caps.events, (int) (model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN4EXP),
            model.hparams.n_layer_nextn, (int) cparams.ctx_type, (int) model.n_devices());
    return mtp_hidden.ring;
}

void llama_context::set_ubatch_rows_callback(void (*cb)(void * user, int32_t first, int32_t count), void * user) {
    ubatch_rows_cb = mtp_hidden.ring ? cb : nullptr;
    ubatch_rows_cb_user = mtp_hidden.ring ? user : nullptr;
    if (!cb) {
        const bool used = mtp_hidden.ring && mtp_hidden.buf != nullptr;
        mtp_handoff_release();
        if (used) {
            // the helper reserved its graphs against this decode's allocator; make it reserve again
            detach_compute_borrower();
        }
    }
}

void llama_context::mtp_handoff_release() {
    auto & stage = mtp_hidden;
    if (!stage.ring || !stage.buf) {
        stage.ring_first = -1;
        stage.ring_rows = 0;
        return;
    }
    if (stage.consuming && stage.consumed) {
        ggml_backend_event_synchronize(stage.consumed.get());
    }
    stage.consuming = false;
    if (stage.backend) {
        ggml_backend_synchronize(stage.backend);
    }
    stage.buf.reset();
    stage.ctx.reset();
    stage.rows = nullptr;
    stage.ring_first = -1;
    stage.ring_rows = 0;
}

void llama_context::mtp_handoff_capture(ggml_tensor * hidden, const llama_ubatch & ubatch) {
    auto & stage = mtp_hidden;
    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), hidden);
    GGML_ASSERT(backend && ubatch.data && hidden->ne[1] == ubatch.n_tokens && ubatch.n_tokens <= cparams.n_ubatch);
    if (!stage.rows) {
        stage.ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!stage.ctx) {
            throw std::runtime_error("failed to create MTP handoff staging context");
        }
        stage.rows = ggml_new_tensor_2d(stage.ctx.get(), GGML_TYPE_F32, hidden->ne[0], (int64_t) cparams.n_ubatch + 1);
        stage.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(stage.ctx.get(), ggml_backend_get_default_buffer_type(backend)));
        if (!stage.buf) {
            throw std::runtime_error("failed to allocate MTP handoff staging buffer");
        }
        if (stage.backend != backend || !stage.ready || !stage.consumed) {
            auto * dev = ggml_backend_get_device(backend);
            stage.ready.reset(ggml_backend_event_new(dev));
            stage.consumed.reset(ggml_backend_event_new(dev));
        }
        stage.backend = backend;
        LLAMA_LOG_INFO("%s: MTP handoff staging = %.3f MiB\n", __func__,
                ggml_backend_buffer_get_size(stage.buf.get()) / (1024.0 * 1024.0));
    }
    GGML_ASSERT(stage.backend == backend && stage.rows->ne[0] == hidden->ne[0]);
    if (stage.consuming) {
        if (stage.consumed) {
            ggml_backend_event_wait(backend, stage.consumed.get());
        }
        stage.consuming = false;
    }
    auto src = mtp_row_view(hidden, 0, ubatch.n_tokens);
    auto dst = mtp_row_view(stage.rows, 0, ubatch.n_tokens);
    ggml_backend_tensor_copy_async(backend, backend, &src, &dst);
    if (stage.ready) {
        ggml_backend_event_record(stage.ready.get(), backend);
    }
    stage.ring_first = ubatch.data->batch_idxs[0];
    stage.ring_rows = (int32_t) ubatch.n_tokens;
}

bool llama_context::mtp_handoff_ready(int32_t first, int32_t count) const {
    const auto & stage = mtp_hidden;
    return stage.ring && stage.rows && ubatch_rows_cb && stage.ring_first >= 0 && count > 0 &&
        first >= stage.ring_first && (int64_t) first + count <= stage.ring_first + stage.ring_rows;
}

bool llama_context::mtp_handoff_row(float * dst, int32_t row) const {
    // mid-decode the unmasked rows sit in batch order (the staged ubatch is one contiguous run from n_tokens_prev),
    // so the row is read in place; output_reorder() would consume the decode's pending row map
    const auto & stage = mtp_hidden;
    const size_t width = model.hparams.n_embd_out();
    if (!dst || !stage.ring || stage.ring_first < 0 || row < stage.ring_first || row >= stage.ring_first + stage.ring_rows ||
            cparams.embeddings_nextn_masked || !embd_nextn.data || (size_t) (row + 1) * width > embd_nextn.size) {
        return false;
    }
    std::memcpy(dst, embd_nextn.data + (size_t) row * width, width * sizeof(float));
    return true;
}

int llama_context::mtp_handoff_decode(llama_context & target, llama_batch batch, int32_t first, const float * carry) {
    auto & source = target.mtp_hidden;
    // [PREFILLFAST] LLAMA_MTP_HANDOFF_NO_EMBD=1: the hidden rows come from the target's device ring (mtp_prefill_input),
    // so the batch's host embd rows are never read; dropping them skips three copies of n_tokens x n_embd_out floats
    // per helper call (batch conversion, batch allocator, ubatch), about 84 MB each at 2,048 rows.
    // default on (owner ruling 2026-10-21); LLAMA_MTP_HANDOFF_NO_EMBD=0 turns it off
    static const bool no_embd = [] { const char * e = getenv("LLAMA_MTP_HANDOFF_NO_EMBD"); return !(e && strcmp(e, "0") == 0); }();
    if (!source.ring || !source.rows || source.ring_first < 0 || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
            cparams.embeddings || cparams.mtp_chain || !cparams.embeddings_nextn_masked ||
            source.rows->ne[0] != (int64_t) model.hparams.n_embd_out() ||
            batch.n_tokens <= 0 || batch.n_tokens > (int32_t) cparams.n_ubatch || !batch.token || !batch.embd || !batch.pos ||
            !batch.n_seq_id || !batch.seq_id) {
        return -4;
    }
    ggml_backend_t consumer = nullptr;
    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == model.dev_output()) {
            consumer = backend.get();
        }
    }
    if (!consumer || ggml_backend_get_device(consumer) != ggml_backend_get_device(source.backend)) {
        return -4;
    }
    const int32_t carry_row = (int32_t) source.rows->ne[1] - 1;
    mtp_hidden_rows.resize(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != batch.seq_id[0][0]) {
            return -4;
        }
        if (i == 0 && carry) {
            mtp_hidden_rows[i] = carry_row;
            continue;
        }
        const int64_t row = (int64_t) first + i - 1 - source.ring_first;
        if (row < 0 || row >= source.ring_rows) {
            return -4;
        }
        mtp_hidden_rows[i] = (int32_t) row;
    }
    // the target's staging copies were queued on its stream; its decode loop synchronized before the callback
    ggml_backend_sched_synchronize(target.sched.get());
    if (mtp_input_upload_done) {
        ggml_backend_event_synchronize(mtp_input_upload_done.get());
    }
    mtp_input_upload_next = 0;
    if (carry) {
        // the previous helper graph may still read the carry row
        ggml_backend_synchronize(consumer);
        ggml_backend_tensor_set(source.rows, carry, (size_t) carry_row * source.rows->nb[1], source.rows->nb[1]);
    }
    auto finish = [&]() {
        if (source.consumed) {
            ggml_backend_event_record(source.consumed.get(), consumer);
            source.consuming = true;
        }
        if (!mtp_input_upload_done) {
            mtp_input_upload_done.reset(ggml_backend_event_new(ggml_backend_get_device(consumer)));
        }
        if (mtp_input_upload_done) {
            ggml_backend_event_record(mtp_input_upload_done.get(), consumer);
        }
    };
    mtp_hidden_source = &target;
    if (no_embd) {
        batch.embd = nullptr;
    }
    int result;
    try {
        result = decode(batch);
    } catch (...) {
        mtp_hidden_source = nullptr;
        finish();
        throw;
    }
    mtp_hidden_source = nullptr;
    finish();
    return result;
}

bool llama_mtp_handoff_init(llama_context * target) {
    return target && target->mtp_handoff_init();
}

void llama_set_ubatch_rows_callback(llama_context * target, void (*cb)(void * user, int32_t first, int32_t count), void * user) {
    if (target) {
        target->set_ubatch_rows_callback(cb, user);
    }
}

bool llama_mtp_handoff_ready(llama_context * target, int32_t first, int32_t count) {
    return target && target->mtp_handoff_ready(first, count);
}

bool llama_mtp_handoff_row(llama_context * target, float * dst, int32_t row) {
    return target && target->mtp_handoff_row(dst, row);
}

int llama_mtp_handoff_decode(llama_context * draft, llama_context * target, llama_batch batch, int32_t first, const float * carry) {
    return draft && target ? draft->mtp_handoff_decode(*target, batch, first, carry) : -4;
}

bool llama_mtp_prefill_init(llama_context * target) {
    return target && target->mtp_prefill_init();
}

void llama_mtp_prefill_reset(llama_context * target, llama_seq_id seq) {
    if (target) {
        target->mtp_prefill_reset(seq);
    }
}

int llama_mtp_prefill_decode(llama_context * draft, llama_context * target, llama_batch batch,
        const float * const * carry, uint32_t n_seq) {
    return draft && target ? draft->mtp_prefill_decode(*target, batch, carry, n_seq) : -4;
}

bool llama_context::mtp_device_feedback_init() {
    const char * value = getenv("LLAMA_MTP_DEVICE_FEEDBACK");
    if (!value || strcmp(value, "1") != 0 || model.arch != LLM_ARCH_QWEN4EXP ||
            model.hparams.no_alloc || model.hparams.n_layer_nextn != 1 || model.hparams.n_embd_out() == 0 ||
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || cparams.n_seq_max != 1 ||
            cparams.training || cparams.embeddings || !cparams.embeddings_nextn ||
            !cparams.embeddings_nextn_masked || cparams.mtp_chain ||
            cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || model.n_devices() > 1 ||
            model.split_mode() == LLAMA_SPLIT_MODE_TENSOR ||
            model.dev_layer(model.hparams.n_layer()) != model.dev_output()) {
        return false;
    }
    auto & stage = mtp_feedback;
    if (stage.rows) {
        return true;
    }
    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == model.dev_output()) {
            stage.backend = backend.get();
            break;
        }
    }
    if (!stage.backend) {
        return false;
    }
    stage.ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!stage.ctx) {
        throw std::runtime_error("failed to create MTP feedback context");
    }
    stage.rows = ggml_new_tensor_2d(stage.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_out(), cparams.n_seq_max);
    stage.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(stage.ctx.get(), ggml_backend_get_default_buffer_type(stage.backend)));
    if (!stage.buf) {
        stage.rows = nullptr;
        throw std::runtime_error("failed to allocate MTP feedback buffer");
    }
    LLAMA_LOG_INFO("%s: sequential MTP hidden feedback = %.3f MiB\n", __func__,
            ggml_backend_buffer_get_size(stage.buf.get()) / 1048576.0);
    return true;
}

void llama_context::mtp_device_feedback_capture(ggml_tensor * hidden, const llama_ubatch & ubatch) {
    auto & stage = mtp_feedback;
    auto * backend = hidden ? ggml_backend_sched_get_tensor_backend(sched.get(), hidden) : nullptr;
    GGML_ASSERT(stage.active && stage.rows && hidden && backend == stage.backend);
    GGML_ASSERT(ubatch.n_tokens == 1 && hidden->type == GGML_TYPE_F32 && ggml_is_contiguous(hidden));
    GGML_ASSERT(hidden->ne[0] == stage.rows->ne[0] && hidden->ne[1] == 1);
    auto dst = mtp_row_view(stage.rows, 0, 1);
    // The persistent row survives the next scheduler reset; both copies use the helper stream.
    ggml_backend_tensor_copy_async(backend, backend, hidden, &dst);
    stage.valid = true;
}

void llama_context::mtp_device_feedback_input(llm_graph_input_embd_h & input) {
    auto & stage = mtp_feedback;
    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), input.h);
    GGML_ASSERT(stage.active && stage.use_previous && backend == stage.backend);
    GGML_ASSERT(input.h->type == GGML_TYPE_F32 && input.h->ne[0] == stage.rows->ne[0] && input.h->ne[1] == 1);
    auto src = mtp_row_view(stage.rows, 0, 1);
    ggml_backend_tensor_copy_async(backend, backend, &src, input.h);
}

int llama_context::mtp_device_feedback_decode(llama_batch batch, bool use_previous) {
    auto & stage = mtp_feedback;
    if (!stage.rows || stage.active || mtp_hidden_source || cparams.mtp_chain || cparams.embeddings ||
            !cparams.embeddings_nextn || !cparams.embeddings_nextn_masked ||
            batch.n_tokens != 1 || !batch.token || !batch.embd || !batch.pos || !batch.logits || !batch.logits[0] ||
            !batch.n_seq_id || batch.n_seq_id[0] != 1 || !batch.seq_id || !batch.seq_id[0] || batch.seq_id[0][0] != 0 ||
            (use_previous && (!stage.valid || batch.pos[0] <= 0 || stage.pos != batch.pos[0] - 1))) {
        return -4;
    }
    stage.active = true;
    stage.use_previous = use_previous;
    stage.valid = false;
    int result;
    try {
        result = decode(batch);
    } catch (...) {
        stage.active = false;
        stage.use_previous = false;
        stage.valid = false;
        throw;
    }
    stage.active = false;
    stage.use_previous = false;
    stage.valid = stage.valid && result == 0;
    stage.pos = batch.pos[0];
    return result;
}

bool llama_mtp_device_feedback_init(llama_context * draft) {
    return draft && draft->mtp_device_feedback_init();
}

int llama_mtp_device_feedback_decode(llama_context * draft, llama_batch batch, bool use_previous) {
    return draft ? draft->mtp_device_feedback_decode(batch, use_previous) : -4;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_mtp_chain(bool value) {
    cparams.mtp_chain = value;
    if (value && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && model.arch == LLM_ARCH_QWEN4EXP) {
        mtp_chain_qwen4exp_reserve = true;
        const uint32_t reserve_tokens = sched_reserve_tokens ? std::min(cparams.n_ubatch, sched_reserve_tokens) : cparams.n_ubatch;
        const size_t metadata_capacity = graph_max_nodes(std::min(cparams.n_ctx, reserve_tokens));
        if (!gf_res_reserve || (size_t) gf_res_reserve->get_max_nodes() < metadata_capacity ||
                sched_node_capacity < graph_max_nodes(cparams.n_ubatch)) {
            sched_need_reserve = true;
        }
    }
}

void llama_context::set_mtp_chain_sampling(int32_t top_k, float temp, float top_p, float min_p, const float * u, int32_t n_u) {
    if (top_k <= 0) {
        cparams.mtp_chain_top_k = 0;
        return;
    }
    GGML_ASSERT(top_k <= LLAMA_MTP_CHAIN_TOP_K_MAX && temp > 0.0f && n_u >= 0 && (u != nullptr || n_u == 0));
    cparams.mtp_chain_top_k = top_k;
    mtp_chain_samp.resize(LLAMA_MTP_CHAIN_SAMP_N + (size_t) n_u);
    llama_mtp_chain_samp_pack(temp, top_p, min_p, mtp_chain_samp.data());
    std::copy(u, u + n_u, mtp_chain_samp.begin() + LLAMA_MTP_CHAIN_SAMP_N);
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler && sampler->iface->backend_init && sampler->iface->backend_apply && llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__,
                       llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] != 0.0f) {
            loras->insert({ adapters[i], scales[i] });
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// [PREFILLFAST diag] LLAMA_DECODE_HOST_US=1: host time split of each helper (MTP) decode of >= 256 rows, one log line
// per call. Diagnostic only (one clock read per mark); never set it on timed runs.
static bool llama_decode_host_us_enabled() {
    static const bool v = [] { const char * e = getenv("LLAMA_DECODE_HOST_US"); return e && e[0] == '1'; }();
    return v;
}
struct llama_decode_host_us {
    bool on = false;
    int n = 0;
    int64_t t_last = 0;
    const char * name[32];
    int64_t us[32];
    void start(bool enable) { on = enable; n = 0; if (on) { t_last = ggml_time_us(); } }
    void mark(const char * nm) {
        if (!on || n >= 32) { return; }
        const int64_t t = ggml_time_us();
        name[n] = nm; us[n++] = t - t_last; t_last = t;
    }
    void print(uint32_t rows) {
        if (!on) { return; }
        on = false;
        std::string line; char buf[64]; int64_t total = 0;
        for (int i = 0; i < n; ++i) { snprintf(buf, sizeof(buf), " %s %.2f", name[i], us[i]/1000.0); line += buf; total += us[i]; }
        LLAMA_LOG_WARN("decode host ms (helper, %u rows): total %.2f |%s\n", rows, total/1000.0, line.c_str());
    }
};
static thread_local llama_decode_host_us g_decode_host_us;
// [VFLOOR] LLAMA_VF_PHASES=1: host wall split of small (2..8 token) decode calls, i.e. speculative verifies.
// Off by default; it only reads the clock. Printed every 128 calls per context.
namespace {
struct vf_phase_stats {
    enum { ENTRY, BUFS, RESERVE, MEMUPD, INITB, OUTRES, B1, PRE, APPLY, GRAPH, INPUTS, COMPUTE, EXTRACT, POST, N };
    uint64_t us[N] = {};
    uint64_t calls = 0, rebuilds = 0;
};
bool vf_phases_enabled() {
    static const bool v = [] { const char * e = getenv("LLAMA_VF_PHASES"); return e && atoi(e) != 0; }();
    return v;
}
thread_local int64_t vf_t_mark = 0;
thread_local vf_phase_stats * vf_cur = nullptr;
inline void vf_mark(int phase) {
    if (vf_cur) {
        const int64_t t = ggml_time_us();
        vf_cur->us[phase] += (uint64_t) (t - vf_t_mark);
        vf_t_mark = t;
    }
}
struct vf_scope {
    vf_phase_stats * prev;
    vf_phase_stats * st;
    vf_scope(vf_phase_stats * s) : prev(vf_cur), st(s) {
        vf_cur = s;
        if (s) {
            vf_t_mark = ggml_time_us();
        }
    }
    ~vf_scope() {
        if (st) {
            vf_mark(vf_phase_stats::POST);
            st->calls++;
            if (st->calls % 128 == 0) {
                const double n = (double) st->calls;
                static const char * names[vf_phase_stats::N] = { "entry", "bufs", "reserve", "memupd", "initb", "outres", "b1",
                                                 "pre", "apply", "graph", "inputs", "compute-submit", "extract", "post" };
                std::string line;
                char buf[64];
                for (int i = 0; i < vf_phase_stats::N; ++i) {
                    snprintf(buf, sizeof(buf), " %s %.3f", names[i], st->us[i] / n / 1e3);
                    line += buf;
                }
                LLAMA_LOG_INFO("vfloor decode phases (LLAMA_VF_PHASES) ctx %p, %" PRIu64 " calls, %" PRIu64 " graph rebuilds, ms/call:%s\n",
                        (void *) st, st->calls, st->rebuilds, line.c_str());
            }
        }
        vf_cur = prev;
    }
};
std::mutex vf_stats_mtx;
std::unordered_map<const void *, vf_phase_stats> vf_stats_by_ctx;
vf_phase_stats * vf_stats_for(const void * ctx, uint32_t n_tokens) {
    if (!vf_phases_enabled() || n_tokens < 2 || n_tokens > 8) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(vf_stats_mtx);
    return &vf_stats_by_ctx[ctx];
}
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch &     ubatch,
                                                 llm_graph_type           gtype,
                                                 llama_memory_context_i * mctx,
                                                 ggml_status &            ret) {
    auto * bookkeep_perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    llama_bookkeep_stage_timer bookkeep_stage(bookkeep_perf ? &bookkeep_perf->memory_init_layout_host_us : nullptr);

    // [B1] the layer-major prefill applied every ubatch up front and seeks between their saved views
    if (mctx && !b1_cur.active && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // [B1] a layer slice seals its own layers for its ubatch (the bookkeeping was advanced when it was applied)
    if (b1_cur.active ? (b1_cur.ffn_phase <= 1 && !b1_seal_sj_kvarn(b1_cur.il0, b1_cur.il1)) : !maintain_sj_kvarn()) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // Consume feedback from the previous completed decode before building the
    // next graph. Repartition itself synchronizes only when hysteresis selects
    // a new boundary, so steady-state token generation stays asynchronous.
    //
    // TurboQuant: this used to be gated on dynamic_cast<llama_memory_hybrid*>
    // with no fallback, so for a PLAIN (non-hybrid) llama_kv_cache - what
    // every model validated on this fork so far actually uses - the phase
    // never switched past its initial construction-time prefill layout, and
    // kv_stream_adapt() never ran at all. mctx->get_kv_stream_active_targets()
    // covers both cases (and, once wired up, iSWA/DSA/MSA/DSV4's sub-caches)
    // uniformly.
    if (mctx != nullptr && mctx->has_kv_stream_targets()) {
        for (const auto & active_target : mctx->get_kv_stream_active_targets()) {
            if (active_target.cache == nullptr) {
                continue;
            }
            llama_kv_stream_phase phase = LLAMA_KV_STREAM_PHASE_AUTOMATIC;
            if (decode_phase == LLAMA_DECODE_PHASE_PROMPT) {
                phase = LLAMA_KV_STREAM_PHASE_PROMPT;
            } else if (decode_phase == LLAMA_DECODE_PHASE_GENERATION) {
                phase = LLAMA_KV_STREAM_PHASE_GENERATION;
            }
            const bool generation = llama_kv_stream_phase_is_generation(phase, ubatch.n_tokens);
            if (kv_stream_phase_arena.configured && generation && ubatch.n_tokens != cparams.n_seq_max) {
                LLAMA_LOG_ERROR("%s: phase arena currently supports TG1 without speculative batches\n", __func__);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (!kv_stream_switch_phase(generation, active_target.n_kv)) {
                LLAMA_LOG_ERROR("%s: failed to switch shared CUDA arena phase\n", __func__);
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }
            (void) active_target.cache->kv_stream_adapt(active_target.n_kv, ubatch.n_tokens);
        }
    }

    g_decode_host_us.mark("apply");
    mtp_scratch_prepare(n_outputs == 0, mctx, &ubatch);
    g_decode_host_us.mark("scratch2");
    vf_mark(vf_phase_stats::APPLY);

build_graph:
    bookkeep_stage.next(bookkeep_perf ? &bookkeep_perf->graph_prepare_host_us : nullptr);
    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // a peer context that grew the shared compute buffer (ggml_gallocr donor) invalidates this context's allocated graph
    if (!graph_reuse_disable && !b1_cur.active && gf_res_prev_active == res &&
            gf_res_prev_epoch == ggml_backend_sched_get_alloc_epoch(sched.get()) &&
            res->mtp_device_input == (mtp_hidden_source != nullptr || mtp_feedback.active) && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        if (bookkeep_perf) {
            bookkeep_perf->graph_reuse_calls.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (bookkeep_perf) {
            bookkeep_perf->graph_rebuild_calls.fetch_add(1, std::memory_order_relaxed);
        }
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        // fork PREFETCHFIX: the asynchronous prefetch reads its predictions on the device, so it
        // needs no per-node host callback (which would split graphs and disable CUDA graphs).
        static const bool prefetch_async = [] {
            const char * v = getenv("GGML_MOE_PREFETCH_ASYNC");
            return v && (strcmp(v, "1") == 0 || strcmp(v, "2") == 0);
        }();
        const bool cache_eval = b1_cache_collect || (cparams.moe_prefetch != 0 && !prefetch_async);
        ggml_backend_sched_set_eval_callback(sched.get(), cache_eval ? b1_cache_eval : cparams.cb_eval,
                                             cache_eval ? this : cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);
        g_decode_host_us.mark("build");

        if (gf && (!model.hadamard_rotations.empty() || !model.hadamard_inverses.empty())) {
            llama_verify_hadamard_graph(gf, model.hadamard_rotations, model.hadamard_inverses);
        }

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (mtp_hidden_source || mtp_feedback.active) {
            mtp_prefill_place(*res);
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            if (mtp_scratch_active) {
                LLAMA_LOG_INFO("%s: MTP KV scratch no longer fits; retrying with serialized compute\n", __func__);
                mtp_scratch_end();
                goto build_graph;
            }
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        g_decode_host_us.mark("alloc");
        // [B1] a layer-range graph is never reused: its carry, KV views and expert slots change every call
        gf_res_prev_active = b1_cur.active ? nullptr : res;
        gf_res_prev_epoch  = ggml_backend_sched_get_alloc_epoch(sched.get());
        if (vf_cur) {
            vf_cur->rebuilds++;
        }
    }
    vf_mark(vf_phase_stats::GRAPH);

    if (mtp_scratch_active && ggml_backend_dev_type(ggml_backend_get_device(mtp_scratch_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        using preflight_fn_t = bool (*)(ggml_backend_t, const ggml_cgraph *);
        auto preflight = (preflight_fn_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(mtp_scratch_backend)), "ggml_backend_cuda_scratch_preflight");
        if (!preflight || !preflight(mtp_scratch_backend, gf)) {
            LLAMA_LOG_INFO("%s: MTP KV scratch preflight refused; retrying with serialized compute\n", __func__);
            mtp_scratch_end();
            goto build_graph;
        }
    }

    bookkeep_stage.next(bookkeep_perf ? &bookkeep_perf->set_inputs_host_us : nullptr);
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        for (auto & input : res->inputs) {
            if (auto * h = dynamic_cast<llm_graph_input_embd_h *>(input.get())) {
                h->h_device = mtp_hidden_source != nullptr || mtp_feedback.use_previous;
            }
        }
        res->set_inputs(&ubatch);
        if (mtp_hidden_source) {
            for (auto & input : res->inputs) {
                if (auto * h = dynamic_cast<llm_graph_input_embd_h *>(input.get())) {
                    mtp_prefill_input(*h, ubatch);
                }
            }
        }
        if (mtp_feedback.use_previous) {
            for (auto & input : res->inputs) {
                if (auto * h = dynamic_cast<llm_graph_input_embd_h *>(input.get())) {
                    mtp_device_feedback_input(*h);
                }
            }
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    g_decode_host_us.mark("inputs");
    bookkeep_stage.next(bookkeep_perf ? &bookkeep_perf->peer_sync_host_us : nullptr);
    if (compute_is_shared() && compute_peer->sched) {
        // the peer's last graph may still be running on its own stream over the shared buffers
        ggml_backend_sched_synchronize(compute_peer->sched.get());
    }
    g_decode_host_us.mark("peersync");

    bookkeep_stage.next(bookkeep_perf ? &bookkeep_perf->graph_submit_host_us : nullptr);
    // One bounded diagnostic snapshot, never enabled in timing claims. Scheduling
    // assignment is inspected after allocation; no callback or extra device fence.
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && n_outputs == 0 && ubatch.n_tokens <= 8 &&
            getenv("LLAMA_BOOKKEEP_SPLIT_DUMP") && std::atoi(getenv("LLAMA_BOOKKEEP_SPLIT_DUMP")) != 0) {
        static std::atomic_flag dumped = ATOMIC_FLAG_INIT;
        if (!dumped.test_and_set(std::memory_order_relaxed)) {
            fprintf(stderr, "BOOKKEEP_GRAPH nodes=%d tokens=%u outputs=%d\n", ggml_graph_n_nodes(gf), ubatch.n_tokens, n_outputs);
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto * node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                        node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) { continue; }
                auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                fprintf(stderr, "BOOKKEEP_GRAPH node=%d op=%s name=%s backend=%s shape=%lld,%lld,%lld,%lld\n",
                        i, ggml_op_desc(node), node->name, backend ? ggml_backend_name(backend) : "none",
                        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
            }
        }
    }
    vf_mark(vf_phase_stats::INPUTS);
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    vf_mark(vf_phase_stats::COMPUTE);
    g_decode_host_us.mark("compute");
    bookkeep_stage.next(nullptr);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens, n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status  status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:
                return 2;
            case GGML_STATUS_ALLOC_FAILED:
                return -2;
            case GGML_STATUS_FAILED:
                return -3;
            case GGML_STATUS_SUCCESS:
                GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        // chain mode: logits are [2, n_chain] (token_id, prob) pairs, not [n_vocab, n_tokens]
        const size_t logits_bytes =
            cparams.mtp_chain ? ggml_nbytes(t_logits) : (size_t) n_tokens * n_vocab * sizeof(float);
        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, logits_bytes);
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens * n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0,
                                                  n_tokens * n_embd_out * sizeof(float));
                }
                break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                      (n_embd_out * seq_idx) * sizeof(float),
                                                      n_embd_out * sizeof(float));
                    }
                }
                break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                      (n_cls_out * seq_idx) * sizeof(float), n_cls_out * sizeof(float));
                    }
                }
                break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens * n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens * n_embd * sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd * cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t                         local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id]        = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                   const buffer_view<llama_token> &              sampled,
                                   const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                   ggml_backend_sched_t                          sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                     const buffer_view<float> &                    dst,
                                     size_t                                        stride,
                                     std::vector<uint32_t> &                       counts,
                                     const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                     ggml_backend_sched_t                          sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float *        row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                         const buffer_view<llama_token> &              dst,
                                         size_t                                        stride,
                                         std::vector<uint32_t> &                       counts,
                                         const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                         ggml_backend_sched_t                          sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token *  row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}
static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false;  // all sequences use backend sampling
}

bool llama_context::prompt_buffers_resize(uint32_t n_tokens, const llama_batch_ext * batch) {
    const char * rotate = getenv("LLAMA_MOE_CACHE_RELEASE_PROMPT");
    if (!rotate || rotate[0] != '1' || cparams.warmup || !cparams.causal_attn || cparams.training ||
            cparams.embeddings || kv_stream_phase_arena.configured || cparams.n_ubatch <= 512) {
        return true;
    }
    bool alone = batch != nullptr;
    llama_seq_id seq = -1;
    const std::vector<llama_batch_ext::token> empty_tokens;
    const auto & tokens = batch ? batch->tokens : empty_tokens;
    for (const auto & token : tokens) {
        if (token.seq_ids.size() != 1) {
            alone = false;
            break;
        }
        const llama_seq_id current = *token.seq_ids.begin();
        if (seq >= 0 && current != seq) {
            alone = false;
            break;
        }
        seq = current;
    }
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq_max() && alone; ++s) {
        alone = s == seq || memory->seq_pos_max(s) < 0;
    }
    // fork: n_tokens contains only the incoming rows after prompt-cache reuse.
    uint32_t rows = 512;
    if (alone && n_tokens > 512 && cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT &&
            decode_phase == LLAMA_DECODE_PHASE_PROMPT) {
        // fork: a large parent prompt submits bounded draft handoff pieces.
        rows = std::min(cparams.n_ubatch, std::max(n_tokens, sched_reserve_tokens));
    } else if (alone && n_tokens > 3072) {
        rows = n_tokens < 16384 ? std::min(2048u, n_tokens) : std::min(cparams.n_ubatch, n_tokens);
    }
    rows = std::min(rows, cparams.n_ubatch);
    // fork: packed prefill grows single-layer graphs, not a full-model MoE graph.
    const char * packed = getenv("LLAMA_B1_FFN");
    if (alone && n_tokens >= 16384 && packed && packed[0] == '1' &&
            cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && model.arch == LLM_ARCH_QWEN4EXP &&
            loras->empty() && model.exl3_sides.empty() && model.hadamard_inverses.empty()) {
        rows = std::min(512u, cparams.n_ubatch);
        b1_ffn_scratch_active = true;
    }
    const char * layer_reserve = getenv("LLAMA_B1_LAYER_RESERVE");
    const char * layer_major = getenv("LLAMA_B1");
    if (!b1_ffn_scratch_active && alone && n_tokens >= 16384 && layer_reserve && layer_reserve[0] == '1' &&
            layer_major && layer_major[0] == '1' && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            model.arch == LLM_ARCH_QWEN4EXP && loras->empty() && model.exl3_sides.empty() &&
            model.hadamard_inverses.empty()) {
        rows = std::min(512u, cparams.n_ubatch);
        b1_layer_scratch_active = true;
    }
    const bool trim_packed = n_tokens == 0 && (b1_ffn_scratch_active || b1_layer_scratch_active);
    if (rows == sched_reserve_tokens && !trim_packed) {
        return true;
    }
    synchronize();
    for (ggml_backend_t backend : backend_ptrs) {
        auto dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        using reset_fn = bool (*)(ggml_backend_t);
        auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
        if (!reset || !reset(backend)) {
            LLAMA_LOG_ERROR("%s: cannot reset graphs on %s before scratch resize\n", __func__, ggml_backend_name(backend));
            return false;
        }
    }
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_prev_active = nullptr;
    const uint32_t previous = sched_reserve_tokens;
    sched_reserve_tokens = rows;
    sched_need_reserve = true;
    sched_reserve();
    if (rows < previous || trim_packed) {
        b1_cache_needs_refresh = true;
    }
    if (trim_packed) {
        b1_ffn_scratch_active = false;
        b1_layer_scratch_active = false;
    }
    LLAMA_LOG_INFO("%s: turn scratch %u -> %u rows, %u uncached tokens, %s, %s\n", __func__, previous, rows,
            n_tokens, cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT ? "target" : "draft", alone ? "alone" : "shared slots");
    return true;
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    vf_scope vf_scope_(vf_stats_for(this, (uint32_t) batch_inp.tokens.size()));
    if (!mtp_feedback.active) {
        mtp_feedback.valid = false;
    }
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    // in-call checkpoints: a request is for this call only; an older one's copies are dropped here
    if (!ckpt_armed) {
        ckpt_caps.clear();
    }
    ckpt_armed = false;
    ckpt_sel   = -1;

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    g_decode_host_us.start(llama_decode_host_us_enabled() && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.tokens.size() >= 256);
    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                const auto sampler = sampling.samplers.find(seq_id);
                if (seq_output_count[seq_id] > 1 && sampler != sampling.samplers.end()) {
                    // multi-row (speculative verification) sampling needs every sampler of the chain on the backend:
                    // a partially offloaded chain would leave rows unsampled and abort in the graph build
                    if (!llama_sampler_backend_rows_ready(sampler->second)) {
                        LLAMA_LOG_ERROR("%s: backend sampler '%s' requires at most one output token per sequence (seq_id %d had %d)\n",
                                __func__, llama_sampler_name(sampler->second), seq_id, seq_output_count[seq_id]);
                        return -1;
                    }
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    g_decode_host_us.mark("balloc");
    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    vf_mark(vf_phase_stats::ENTRY);

    // [PLEPF] start reading the PLE pages of a prompt call; stopped when the call returns
    if (!ple_pf_tried) {
        ple_pf_tried = true;
        ple_pf = llama_ple_prefetch::create(model, cparams.n_ubatch);
    }
    struct ple_pf_guard {
        llama_ple_prefetch * p = nullptr;
        ~ple_pf_guard() {
            if (p) {
                p->stop();
            }
        }
    } ple_pf_g;
    if (ple_pf && ple_pf->start(batch_inp, balloc->get_batch().pos[0])) {
        ple_pf_g.p = ple_pf.get();
    }

    if (auto * kv = llama_sj_kvarn_cache(memory.get()); kv && !kv->scratch_safe(n_tokens_all)) {
        kv->scratch_cancel();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_idx *>(memory.get())) {
        if (auto * kv = hybrid->get_mem_idx(); kv && !kv->scratch_safe(n_tokens_all)) {
            kv->scratch_cancel();
        }
    }

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR(
                "%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) &&
                "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    g_decode_host_us.mark("pre");
    if (!prompt_buffers_resize(n_tokens_all, &batch_inp)) {
        return -2;
    }
    g_decode_host_us.mark("bufresize");
    vf_mark(vf_phase_stats::BUFS);
    b1_cache_collect = false;
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens_all > 3072 &&
            (sched_reserve_tokens > 512 || b1_ffn_scratch_active || b1_layer_scratch_active)) {
        b1_cache_begin();
        b1_cache_collect = b1_cache_warm_enabled && b1_cache_collect_wanted();
    }
    struct cache_capture_guard {
        bool & active;
        ~cache_capture_guard() { active = false; }
    } cache_capture_guard_ { b1_cache_collect };
    // Packed B1 reserves a small full-model graph and grows phase graphs on demand.
    // Its causal attention/shared-expert chunks still use the requested turn size.
    const uint32_t turn_ubatch = (b1_ffn_scratch_active || b1_layer_scratch_active) ? cparams.n_ubatch :
            (sched_reserve_tokens ? std::min(cparams.n_ubatch, sched_reserve_tokens) : cparams.n_ubatch);

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();
    if (mtp_hidden.enabled) {
        mtp_hidden.captured.assign(n_tokens_all, false);
    }

    if (mtp_scratch_active && sched_need_reserve) {
        mtp_scratch_end();
    }
    sched_reserve();
    g_decode_host_us.mark("reserve");
    mtp_scratch_prepare(n_outputs_all == 0);
    g_decode_host_us.mark("scratch");
    vf_mark(vf_phase_stats::RESERVE);

    auto * bookkeep_perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    llama_bookkeep_stage_timer bookkeep_memory(bookkeep_perf ? &bookkeep_perf->memory_init_layout_host_us : nullptr);
    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);
    g_decode_host_us.mark("memupd");
    vf_mark(vf_phase_stats::MEMUPD);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, turn_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                }
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__,
                                            balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__,
                                   balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__,
                                    balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }
    bookkeep_memory.next(nullptr);
    vf_mark(vf_phase_stats::INITB);

    g_decode_host_us.mark("initbatch");
    // reserve output buffer
    if (output_reserve(n_outputs_all, n_tokens_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };
    vf_mark(vf_phase_stats::OUTRES);

    // [B1] layer-major prefill: plan the call and apply every ubatch up front, or give lent experts back first
    g_decode_host_us.mark("outres");
    const int b1_mode = b1_prepare(mctx.get(), batch_inp, n_tokens_all, n_outputs_all);
    g_decode_host_us.mark("b1prep");
    if (b1_mode < 0) {
        b1_cache_counts.clear();
        b1_cache_readbacks = 0;
        return b1_mode;
    }
    if (b1_mode == 0 && (b1_ffn_scratch_active || b1_layer_scratch_active) && turn_ubatch > sched_reserve_tokens) {
        // Eligibility and lending can fail after the memory context was prepared.
        // No views were applied on that path; rebuild them at the safe normal size.
        {
            llama_bookkeep_stage_timer bookkeep_fallback(bookkeep_perf ? &bookkeep_perf->memory_init_layout_host_us : nullptr);
            mctx = memory->init_batch(*balloc, sched_reserve_tokens, output_all);
        }
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: cannot prepare normal chunks after packed B1 fallback\n", __func__);
            return -2;
        }
        LLAMA_LOG_INFO("%s: layer-sized B1 fallback uses %u-token normal chunks\n", __func__, sched_reserve_tokens);
    }
    if (b1_mode == 0 && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens_all <= 8) {
        b1_cache_warm();
        b1_cache_device_prepare();
    }
    vf_mark(vf_phase_stats::B1);
    llama_seq_id b1_seq      = -1;
    llama_pos    b1_pos0     = 0;
    int32_t      b1_final_l0 = 0;
    size_t       b1_i        = 0;
    if (b1_mode == 1) {
        b1_seq = *batch_inp.tokens[0].seq_ids.begin();
        b1_pos0 = balloc->seq_pos_min(b1_seq);
        b1_final_l0 = b1->final_l0();
    }
    // on a failure every ubatch of the call is incomplete: drop the whole batch from the memory
    auto b1_fail = [&](ggml_status st) -> int {
        b1_cur = {};
        if (!b1_kv_scratch_release()) { throw std::runtime_error("cannot return B1 KV scratch"); }
        b1_kv_scratch_enabled = false;
        b1_end(true);
        b1_cache_counts.clear();
        b1_cache_readbacks = 0;
        // SJ-KVaRN: layers stopped at different ubatches, so ring rows below b1_pos0 may already be overwritten in
        // some layers and sealed groups disagree between layers; only a full clear of the sequence is exact
        const bool sj_on = llama_sj_kvarn_cache(memory.get()) && llama_sj_kvarn_cache(memory.get())->is_sj_kvarn();
        const llama_pos rm0 = sj_on ? -1 : b1_pos0;
        LLAMA_LOG_WARN("%s: [B1] removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, b1_seq, std::max<llama_pos>(0, rm0));
        memory->seq_rm(b1_seq, rm0, -1);
        switch (st) {
            case GGML_STATUS_ABORTED:      return 2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            default:                       return -3;
        }
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    vf_mark(vf_phase_stats::PRE);

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        if (b1_mode == 1) {
            if (b1->is_group_first(b1_i)) {
                b1->pass_begin(b1_i);
                ggml_status st_pre = GGML_STATUS_SUCCESS;
                if (!b1_run_prefix(mctx.get(), b1_i, st_pre)) {
                    return b1_fail(st_pre);
                }
            }
            b1_cur = { true, b1_final_l0, -1, b1_final_l0 > 0 ? b1->carry(b1_i) : nullptr };
        }

        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        // [B1] the last range holds a streamed layer only when it is the last layer; match the normal path's CPU
        // choice for an ubatch whose last-layer FFN rows are too few to offload
        const bool b1_home = b1_mode == 1 && b1->layer_streamed(b1_final_l0) &&
            !b1->host_rows_on_gpu(b1_final_rows(n_outputs, ubatch.n_tokens));
        if (b1_home) {
            b1->ubatch_home(b1_final_l0, true);
        }

        const int64_t t_compute0 = b1_mode == 1 ? ggml_time_us() : 0;
        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);
        if (b1_mode == 1) {
            const int64_t compute_us = ggml_time_us() - t_compute0;
            b1_compute_submit_us += compute_us;
            b1->pass_compute(b1_i, compute_us);
        }

        if (b1_home) {
            b1->ubatch_home(b1_final_l0, false);
        }

        if (b1_mode == 1) {
            b1_cur = {};
            if (!res) {
                return b1_fail(status);
            }
            if (b1->is_group_last(b1_i)) {
                b1->layer_end(b1_final_l0);
            }
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s,
                               pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }

        // in-call checkpoints: the normal path ran every layer, B1's last range the layers from b1_final_l0 on
        capture_ubatch(ubatch, b1_mode == 1 ? b1_final_l0 : 0, -1);

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings ? res->get_embd() : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

        if (mtp_hidden.enabled && t_h_nextn && !cparams.embeddings_nextn_masked) {
            mtp_prefill_capture(t_h_nextn, ubatch);
        }
        // [PREFILLFAST] LLAMA_MTP_B1_HANDOFF: stage this ubatch's rows for the helper when it is one contiguous run
        bool handoff_ub = false;
        if (mtp_hidden.ring && ubatch_rows_cb && t_h_nextn && !cparams.embeddings_nextn_masked && embd_nextn.data &&
                cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && ubatch.data && ubatch.n_tokens > 0 &&
                ubatch.n_tokens <= cparams.n_ubatch && ubatch.data->batch_idxs.size() == ubatch.n_tokens &&
                ubatch.data->batch_idxs[0] == n_tokens_prev) {
            const auto & idxs = ubatch.data->batch_idxs;
            handoff_ub = true;
            for (uint32_t j = 1; j < ubatch.n_tokens && handoff_ub; ++j) {
                handoff_ub = idxs[j] == idxs[0] + (int32_t) j;
            }
            if (handoff_ub) {
                mtp_handoff_capture(t_h_nextn, ubatch);
            }
        }
        // the helper reads the staged rows; the host keeps only each ubatch's last row (the next carry), and the
        // call's final ubatch in full (later draft rounds read rows from it)
        const bool handoff_last_only = handoff_ub && n_tokens_prev + (int64_t) ubatch.n_tokens < (int64_t) n_tokens_all;
        if (mtp_feedback.active) {
            mtp_device_feedback_capture(t_h_nextn, ubatch);
        }

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev * n_vocab;

            if (n_outputs) {
                // chain mode: logits are [2, n_chain] pairs, not [n_vocab, n_outputs]
                const size_t extract_bytes =
                    cparams.mtp_chain ? ggml_nbytes(t_logits) : (size_t) n_outputs * n_vocab * sizeof(float);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, extract_bytes);
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float *        embd_out   = embd.data + n_outputs_prev * n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs) * n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0,
                                                          n_outputs * n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_embd_out * seq_idx) * sizeof(float),
                                                          n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_cls_out * seq_idx) * sizeof(float),
                                                          n_cls_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool    masked = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && !mtp_feedback.active) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd         = hparams.n_embd_out();
                float *        embd_nextn_out = embd_nextn.data + offset * n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                if (handoff_last_only && !masked) {
                    ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out + (n_rows - 1)*n_embd,
                            (size_t) (n_rows - 1)*n_embd*sizeof(float), n_embd*sizeof(float));
                } else {
                    ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                }
                const char * reclaim = getenv("LLAMA_MOE_OUTPUT_RECLAIM");
                if (output_file && reclaim && strcmp(reclaim, "1") == 0) {
                    ggml_backend_sched_synchronize(sched.get());
                    const size_t begin = (const char *) embd_nextn_out - (const char *) output_file->data;
                    output_file->release(begin, (size_t) n_rows * n_embd * sizeof(float));
                    LLAMA_LOG_DEBUG("%s: released completed MTP output offset=%zu bytes=%zu\n", __func__,
                            begin, (size_t) n_rows * n_embd * sizeof(float));
                }
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (res->t_greedy_rows) {
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_greedy_rows);
            // Coalesce adjacent rows, including all rows of a single-sequence MTP verification.
            for (size_t i = 0; i < res->greedy_rows.size();) {
                const uint32_t first = res->greedy_rows[i];
                uint32_t count = 1;
                while (++i < res->greedy_rows.size() && res->greedy_rows[i] == first + count) {
                    ++count;
                }
                GGML_ASSERT(n_outputs_prev + first + count <= (int64_t) sampling.sampled.size);
                ggml_backend_tensor_get_async(backend, res->t_greedy_rows,
                        sampling.sampled.data + n_outputs_prev + first,
                        first * sizeof(llama_token), count * sizeof(llama_token));
            }
        }

        // row-block sampling: copy the sampled ids of each sequence's rows (coalescing adjacent rows)
        for (const auto & [seq_id, t_sampled_rows] : res->t_sampled_rows) {
            const auto & rows = res->sampled_rows.at(seq_id);
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t_sampled_rows);
            GGML_ASSERT(sampling.sampled.has_data());
            for (size_t i = 0; i < rows.size();) {
                const uint32_t first = rows[i];
                const size_t   i0    = i;
                uint32_t count = 1;
                while (++i < rows.size() && rows[i] == first + count) {
                    ++count;
                }
                GGML_ASSERT(n_outputs_prev + first + count <= (int64_t) sampling.sampled.size);
                ggml_backend_tensor_get_async(backend, t_sampled_rows,
                        sampling.sampled.data + n_outputs_prev + first,
                        i0 * sizeof(llama_token), count * sizeof(llama_token));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers &&
            (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride            = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats(res->t_sampled_logits, sampling.logits, stride, sampling.logits_count,
                                     seq_to_output_row, sched.get());
            copy_tensor_async_floats(res->t_sampled_probs, sampling.probs, stride, sampling.probs_count,
                                     seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates, sampling.candidates, stride, sampling.candidates_count,
                                         seq_to_output_row, sched.get());
        }

        vf_mark(vf_phase_stats::EXTRACT);
        n_outputs_prev += n_outputs;
        n_tokens_prev += ubatch.n_tokens;
        if (b1_mode == 1 && b1->is_group_last(b1_i)) {
            b1->pass_end(b1_i);
        }
        if (handoff_ub) {
            // every copy of this ubatch (staging, last host row, logits) must be complete before the helper runs
            ggml_backend_sched_synchronize(sched.get());
            ubatch_rows_cb(ubatch_rows_cb_user, (int32_t) (n_tokens_prev - ubatch.n_tokens), (int32_t) ubatch.n_tokens);
        }
    } while (b1_mode == 1 ? (++b1_i < b1->n_ub() && mctx->b1_seek(b1_i)) : mctx->next());

    if (b1_mode == 1) {
        GGML_ASSERT(b1_i == b1->n_ub());
        if (!b1_kv_scratch_release()) { return b1_fail(GGML_STATUS_FAILED); }
        b1_kv_scratch_enabled = false;
        b1_end(false);
        LLAMA_LOG_INFO("%s: [B1] graph compute calls host wall %.3f ms (includes routing readbacks; overlaps expert copies)\n",
                __func__, b1_compute_submit_us / 1000.0);
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id     = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    g_decode_host_us.mark("tail");
    g_decode_host_us.print(n_tokens_all);
    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs, int32_t n_tokens) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    // Keep the logits row reservation: small sampling changes must not churn pinned memory.
    const int64_t n_outputs_max = std::max<int64_t>({(int64_t) n_outputs, (int64_t) n_seq_max(),
                                                  output_buffer_reuse ? (int64_t) n_outputs_reserved : 0});

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count        = 0;
    size_t backend_token_count        = 0;
    size_t embd_layer_inp_float_count = 0;

    // A real target batch invalidates its previous outputs. The MTP helper has already
    // copied pending/deferred hidden rows before the next target decode starts. Setter
    // calls use n_tokens == 0 and must retain those still-live outputs and their layout.
    // Keep this scoped to unmasked target hidden output; draft, embedding, training and
    // layer-extraction contexts retain their existing allocation policy.
    const char * shrink_env = std::getenv("LLAMA_MTP_OUTPUT_SHRINK");
    const bool shrink_nextn = (!shrink_env || std::strcmp(shrink_env, "0") != 0) &&
            output_buffer_reuse && n_tokens > 0 && has_embd_nextn &&
            !cparams.embeddings_nextn_masked && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            cparams.causal_attn && !cparams.training && !has_embd &&
            std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                    [](bool enabled) { return enabled; });
    const size_t nextn_rows = std::max<size_t>(std::max(n_tokens, 1),
            output_buffer_reuse && !shrink_nextn ? embd_nextn.size / n_embd_out : 0);

    logits.size     = has_logits ? n_vocab * n_outputs_max : 0;
    embd.size       = has_embd ? n_embd_out * n_outputs_max : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out * n_outputs_max : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // Unmasked nextn needs every input row, including rows without logits.
        embd_nextn.size = (size_t) n_embd_out * nextn_rows;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    // argmax chains (bare greedy, or greedy behind a logit-bias) export only token ids
    const bool compact_sampling = has_sampling && std::all_of(sampling.samplers.begin(), sampling.samplers.end(),
            [](const auto & entry) { return llama_sampler_is_argmax_chain(entry.second); });
    if (compact_sampling) {
        backend_token_count = n_outputs_max;
    } else if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t sampling_size = backend_float_count * sizeof(float) + backend_token_count * sizeof(llama_token);
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count) * sizeof(float) +
        (output_buffer_reuse ? 0 : sampling_size);

    const size_t prev_size_sampling = buf_sampling ? ggml_backend_buffer_get_size(buf_sampling.get()) : 0;
    const size_t new_size_sampling  = sampling_size;

    auto * buft = ggml_backend_cpu_buffer_type();
    // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
    auto * output_dev = model.dev_output();
    auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
    if (output_dev_host_buft) {
        buft = output_dev_host_buft;
    }

    // Return a prompt-sized hidden arena only if it saves at least 256 MiB and
    // halves the allocation, avoiding pinned reallocations
    // for ordinary changes in speculative batch size. Reuse the synchronized free
    // and complete view rebind used by growth; previous outputs expire at this call.
    constexpr size_t shrink_min_bytes = (size_t) 256 * 1024 * 1024;
    const bool shrink_output = shrink_nextn && new_size <= prev_size / 2 &&
            prev_size - new_size >= shrink_min_bytes;
    if (!buf_output || prev_size < new_size || shrink_output) {
        if (buf_output) {
            // Pinned reallocations can be costly, so log both growth and substantial shrink.
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB "
                    "(n_outputs = %d, n_outputs_max = %" PRId64 ", logits = %zu, embd = %zu, embd_nextn = %zu, layer_inp = %zu)\n",
                    __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0,
                    n_outputs, n_outputs_max, logits.size, embd.size, embd_nextn.size, embd_layer_inp_float_count);
            synchronize();

            // TODO: not needed?
            buf_output      = nullptr;
            logits.data     = nullptr;
            embd.data       = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = { nullptr, 0 };
            }
        }

        output_file.reset();
        const char * output_directory = std::getenv("LLAMA_MOE_ARENA_OUTPUT_DIR");
        if (model.moe_arena_size() && has_embd_nextn && !cparams.embeddings_nextn_masked &&
                new_size >= (256ull << 20) && output_directory && output_directory[0]) {
            try {
                output_file = std::make_unique<llama_moe_output>(new_size, output_directory);
                buf_output.reset(ggml_backend_cpu_buffer_from_ptr(output_file->data, new_size));
                LLAMA_LOG_INFO("%s: MoE temporary output %.2f MiB, file-backed original floats\n", __func__, new_size / 1048576.0);
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
                return 0;
            }
        } else {
            buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        }
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__,
                            new_size / (1024.0 * 1024.0));
            return 0;
        }
        if (!output_file) { ggml_backend_buffer_clear(buf_output.get(), 0); }
    }
    if (output_file && n_tokens > 0) { output_file->dirty = true; }

    // the backend sampling outputs live in their own (small) pinned buffer: samplers are installed and removed
    // per request, and their layout (compact greedy vs full chain) must not reallocate the logits block above
    if (output_buffer_reuse && has_sampling && (!buf_sampling || prev_size_sampling < new_size_sampling)) {
        if (buf_sampling) {
            LLAMA_LOG_INFO("%s: reallocating sampling buffer from size %.02f MiB to %.02f MiB "
                    "(n_outputs = %d, n_outputs_max = %" PRId64 ", compact = %d, sampling floats = %zu, sampling tokens = %zu)\n",
                    __func__, prev_size_sampling / 1024.0 / 1024.0, new_size_sampling / 1024.0 / 1024.0,
                    n_outputs, n_outputs_max, (int) compact_sampling, backend_float_count, backend_token_count);
        }
        synchronize();

        buf_sampling = nullptr;

        buf_sampling.reset(ggml_backend_buft_alloc_buffer(buft, new_size_sampling));
        if (buf_sampling == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate sampling buffer of size %.2f MiB\n", __func__, new_size_sampling / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_sampling.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t    offset = 0;
    uint8_t * base   = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{ output_base, logits.size } : buffer_view<float>{ nullptr, 0 };
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{ (float *) (base + offset), embd.size } : buffer_view<float>{ nullptr, 0 };
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{ (float *) (base + offset), embd_nextn.size } :
                                  buffer_view<float>{ nullptr, 0 };
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{ (float *) (base + offset), (size_t) n_embd * n_batch };
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{ nullptr, 0 };
        }
    }

    if (has_sampling) {
        // the sampling views are carved from buf_sampling, not from buf_output
        if (output_buffer_reuse) {
            offset = 0;
            base   = (uint8_t *) ggml_backend_buffer_get_base(buf_sampling.get());
        }

        sampling.logits = compact_sampling ? buffer_view<float>{nullptr, 0} : buffer_view<float>{(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = compact_sampling ? buffer_view<float>{nullptr, 0} : buffer_view<float>{(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = { (llama_token *) (base + offset), (size_t) n_outputs_max };
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = compact_sampling ? buffer_view<llama_token>{nullptr, 0} : buffer_view<llama_token>{(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(), sampling.logits_count.end(), 0);
        std::fill(sampling.probs_count.begin(), sampling.probs_count.end(), 0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = { nullptr, 0 };
        sampling.probs      = { nullptr, 0 };
        sampling.sampled    = { nullptr, 0 };
        sampling.candidates = { nullptr, 0 };

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;
    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);
    n_outputs_reserved = (uint32_t) n_outputs_max;
    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes  = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab    = model.vocab.n_tokens();
    const uint64_t n_embd     = model.hparams.n_embd;
    const uint64_t n_embd_out = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0 * n_vocab + k], logits.data[i1 * n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0 * n_embd_out + k], embd.data[i1 * n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0 * n_embd_out + k], embd_nextn.data[i1 * n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; sampling.logits.has_data() && k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; sampling.probs.has_data() && k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; sampling.candidates.has_data() && k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0], sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0], sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0], sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DFLASH ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    if (mtp_chain_qwen4exp_reserve) {
        // fork: Each unrolled step builds one helper body; catch-up builds one additional front.
        res *= LLAMA_MTP_QWEN4EXP_CHAIN_MAX + 1;
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    const bool qwen_chain = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && model.arch == LLM_ARCH_QWEN4EXP && cparams.mtp_chain;
    if (qwen_chain) {
        GGML_ASSERT(n_seqs == 1);
        n_outputs = std::min(n_outputs, uint32_t(LLAMA_MTP_QWEN4EXP_CHAIN_MAX));
    }
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1 || ((mtp_prefill_async || mtp_reserve_tight) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP));

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs;  // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n",
                        __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch       ubatch = balloc.ubatch_reserve(n_tokens / n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);
    if (qwen_chain) {
        // fork: Chain outputs follow the catch-up prefix, unlike generic reserve outputs.
        for (uint32_t i = 0; i < n_tokens; ++i) {
            ubatch.output[i] = i >= n_tokens - n_outputs;
        }
    }

    auto * res = gf_res_reserve.get();

    auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));
    gparams.is_reserve = true;
    if (qwen_chain && n_outputs == 0) {
        gparams.cparams.mtp_chain = false;
    }

    res->reset();

    auto * gf = model.build_graph(gparams);

    if (mtp_prefill_async && n_outputs == 0) {
        mtp_prefill_place(*res);
    }

    // verify transform coverage on the pristine graph: after scheduling,
    // cross-backend copies break the producer chain the check follows
    if (gf && (!model.hadamard_rotations.empty() || !model.hadamard_inverses.empty())) {
        llama_verify_hadamard_graph(gf, model.hadamard_rotations, model.hadamard_inverses);
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

//
// [B1] layer-major prefill
//

static int64_t b1_ctx_env(const char * name, int64_t def) {
    const char * e = getenv(name);
    return e && *e ? atoll(e) : def;
}

bool llama_context::b1_kv_scratch_release() {
    if (!b1_kv_backend) {
        b1_kv_views.clear();
        b1_kv_owned.reset();
        return true;
    }
    synchronize();
    auto * dev = ggml_backend_get_device(b1_kv_backend);
    using reset_fn = bool (*)(ggml_backend_t);
    auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
    if (!reset || !reset(b1_kv_backend)) { return false; }
    for (auto & res : gf_res_prev) { if (res) { res->reset(); } }
    gf_res_prev_active = nullptr;
    gf_res_reserve->reset();
    ggml_backend_sched_reset(sched.get());
    if (!ggml_backend_sched_set_buffers(sched.get(), b1_kv_backend, nullptr, 0)) { return false; }
    // Masked attention padding may be read before masking: scratch can leave NaN bit patterns.
    // Every view still covers empty storage at this synchronized layer boundary.
    for (auto & view : b1_kv_views) { ggml_backend_buffer_clear(view.get(), 0); }
    b1_kv_views.clear();
    b1_kv_owned.reset();
    b1_kv_backend = nullptr;
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
    }
    backend_buf_exp_epoch = ggml_backend_sched_get_alloc_epoch(sched.get());
    return true;
}

bool llama_context::b1_kv_scratch_bind(int32_t l0, int32_t l1) {
    if (!b1_kv_scratch_enabled) { return true; }
    if (!b1_kv_scratch_release()) { return false; }
    size_t gpu = backend_ptrs.size();
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        auto * dev = ggml_backend_get_device(backend_ptrs[i]);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpu = i; }
    }
    if (gpu == backend_ptrs.size() || (size_t) l0 >= b1_layer_compute_bytes.size()) { return false; }
    auto * kv = llama_sj_kvarn_cache(memory.get());
    if (!kv) { return true; }
    auto views = kv->b1_scratch_views(backend_buft[gpu], l1, b1_kv_prompt_cells);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_idx *>(memory.get())) {
        if (auto * idx = hybrid->get_mem_idx()) {
            auto more = idx->b1_scratch_views(backend_buft[gpu], l1, b1_kv_prompt_cells);
            for (auto & view : more) { views.push_back(std::move(view)); }
        }
    }
    std::sort(views.begin(), views.end(), [](const ggml_backend_buffer_ptr & a, const ggml_backend_buffer_ptr & b) {
        return ggml_backend_buffer_get_size(a.get()) > ggml_backend_buffer_get_size(b.get());
    });
    if (views.size() > 14) { views.resize(14); }
    if (views.empty()) { return true; }
    struct bind_guard {
        llama_context * ctx;
        bool keep = false;
        ~bind_guard() {
            if (keep) { return; }
            ctx->b1_cur = {};
            if (!ctx->b1_kv_scratch_release()) { GGML_ABORT("cannot return B1 KV scratch after failed bind"); }
        }
    } guard {this};
    size_t borrowed = 0;
    for (auto & view : views) { borrowed += ggml_backend_buffer_get_size(view.get()); }
    b1_kv_backend = backend_ptrs[gpu];
    if (!b1_kv_scratch_release()) { return false; }
    const size_t need = b1_layer_compute_bytes[l0];
    const size_t saving = std::min(need / 4, borrowed / 2);
    const size_t alignment = ggml_backend_buft_get_alignment(backend_buft[gpu]);
    const uint32_t tokens = *std::max_element(b1->ubatch_sizes().begin(), b1->ubatch_sizes().end());
    const size_t max_ub = std::max_element(b1->ubatch_sizes().begin(), b1->ubatch_sizes().end()) - b1->ubatch_sizes().begin();
    auto reserve_mctx = memory->init_full();
    if (!reserve_mctx) { return false; }
    // fork: first try a smaller owned buffer; retry with the measured full size if chunks fragment.
    for (size_t reduction : {saving, (size_t) 0}) {
        if (!b1_kv_scratch_release()) { return false; }
        const size_t bytes = GGML_PAD(need - reduction + alignment, alignment);
        b1_kv_backend = backend_ptrs[gpu];
        b1_kv_owned.reset(ggml_backend_buft_alloc_buffer(backend_buft[gpu], bytes));
        if (!b1_kv_owned) { return false; }
        std::vector<ggml_backend_buffer_t> chunks;
        for (auto & view : views) { chunks.push_back(view.get()); }
        chunks.push_back(b1_kv_owned.get());
        synchronize();
        for (auto & res : gf_res_prev) { if (res) { res->reset(); } }
        gf_res_prev_active = nullptr;
        ggml_backend_sched_reset(sched.get());
        // Binding discards the previous owned compute buffer only after all its work is complete.
        if (!ggml_backend_sched_set_buffers(sched.get(), backend_ptrs[gpu], chunks.data(), chunks.size())) { return false; }
        b1_kv_backend = backend_ptrs[gpu];
        // layer_begin has acquired the real layer slot; worker-owned metadata stays untouched.
        b1_cur = {true, l0, l1, b1->carry(max_ub), nullptr, 0};
        auto * graph = graph_reserve(tokens, 1, std::min(tokens, b1_kv_outputs), reserve_mctx.get());
        b1_cur = {};
        gf_res_reserve->reset();
        ggml_backend_sched_reset(sched.get());
        if (graph) {
            b1_kv_views = std::move(views);
            guard.keep = true;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
            }
            backend_buf_exp_epoch = ggml_backend_sched_get_alloc_epoch(sched.get());
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(b1_kv_backend), &free, &total);
            LLAMA_LOG_INFO("%s: [B1] empty KV range [%d,%d): borrowed_capacity_bytes=%zu owned_bytes=%zu measured_bytes=%zu owned_reduction_bytes=%zu chunks=%zu prompt_limit=%u free_bytes=%zu total_bytes=%zu\n",
                    __func__, l0, l1, borrowed, bytes, need, need > bytes ? need - bytes : 0, b1_kv_views.size(), b1_kv_prompt_cells, free, total);
            return true;
        }
    }
    return b1_kv_scratch_release();
}

bool llama_context::b1_reserve_legacy_layers(const std::vector<uint32_t> & sizes, int64_t n_outputs_all) {
    // The deferred plan owns real carries and expert slots. No worker, input
    // upload, graph compute, checkpoint or memory-context apply runs here.
    bool success = false;
    struct probe_guard {
        llama_context * ctx;
        bool & success;
        ~probe_guard() {
            ctx->b1->reserve_layer_restore();
            ctx->b1_cur = {};
            ctx->gf_res_reserve->reset();
            for (auto & res : ctx->gf_res_prev) { if (res) { res->reset(); } }
            ctx->gf_res_prev_active = nullptr;
            ggml_backend_sched_reset(ctx->sched.get());
            if (!success) { ggml_backend_sched_reset_compute(ctx->sched.get()); }
        }
    } guard { this, success };
    try {
        synchronize();
        size_t gpu_index = backend_ptrs.size();
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            const auto dev = ggml_backend_get_device(backend_ptrs[i]);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            if (gpu_index != backend_ptrs.size()) { return false; }
            gpu_index = i;
            using reset_fn = bool (*)(ggml_backend_t);
            auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
            if (!reset || !reset(backend_ptrs[i])) { return false; }
        }
        if (gpu_index == backend_ptrs.size() || sizes.empty()) { return false; }
        const size_t max_ub = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
        const uint32_t tokens = sizes[max_ub];
        const uint32_t outputs = (uint32_t) std::min<int64_t>(tokens, std::max<int64_t>(1, n_outputs_all));
        auto reserve_mctx = memory->init_full();
        if (!reserve_mctx) { return false; }
        const int32_t final_l0 = b1->final_l0();
        b1_layer_compute_bytes.assign(model.hparams.n_layer(), 0);
        int32_t worst_l0 = -1, worst_l1 = -1;
        std::vector<size_t> maximum(backend_ptrs.size(), 0);
        for (int32_t l0 = 0; l0 < (int32_t) model.hparams.n_layer(); ) {
            const int32_t l1 = l0 < final_l0 ? b1->range_end(l0) : model.hparams.n_layer();
            b1->reserve_layer_bind(l0);
            b1_cur = { true, l0, l1, b1->carry(max_ub), nullptr, 0 };
            std::vector<size_t> measured(backend_ptrs.size(), 0);
            if (!graph_reserve(tokens, 1, outputs, reserve_mctx.get(), true, measured.data())) { return false; }
            b1->reserve_layer_restore();
            b1_cur = {};
            b1_layer_compute_bytes[l0] = measured[gpu_index];
            if (worst_l0 < 0 || measured[gpu_index] > maximum[gpu_index]) {
                worst_l0 = l0;
                worst_l1 = l1;
            }
            for (size_t i = 0; i < maximum.size(); ++i) { maximum[i] = std::max(maximum[i], measured[i]); }
            LLAMA_LOG_INFO("%s: [B1] legacy range [%d,%d) %u rows: GPU %.2f MiB\n", __func__, l0, l1, tokens,
                    measured[gpu_index] / 1048576.0);
            l0 = l1;
        }
        auto dev = ggml_backend_get_device(backend_ptrs[gpu_index]);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        const size_t card_limit = (size_t) 23552 << 20;
        const size_t keep = std::max<size_t>(
                (size_t) std::max<int64_t>(1536, b1_ctx_env("LLAMA_B1_KEEP_FREE_MIB", 1536)) << 20,
                total > card_limit ? total - card_limit + (64u << 20) : 0);
        const size_t need = maximum[gpu_index] + keep;
        // The layer graph needs a second loan after carries/expert slots are
        // allocated. Re-probe physical free after each trim; a logical slot shrink
        // may need another bounded trim to cross the physical allocation boundary.
        constexpr int max_layer_loan_attempts = 4;
        size_t retry_loan_floor = 0;
        for (int attempt = 1; attempt <= max_layer_loan_attempts && need > free; ++attempt) {
            const size_t free_before = free;
            const size_t requested = std::max(need - free, retry_loan_floor);
            b1_cache_needs_refresh = true;
            const size_t released = ggml_backend_sched_moe_cache_trim(sched.get(), requested);
            LLAMA_LOG_INFO("%s: [B1] legacy layer scratch borrowed %.2f MiB (requested %.2f MiB)\n", __func__,
                    released / 1048576.0, requested / 1048576.0);
            ggml_backend_dev_memory(dev, &free, &total);
            LLAMA_LOG_INFO("%s: [B1] legacy loan probe attempt=%d requested_bytes=%zu released_bytes=%zu free_before=%zu free_after=%zu actual_free_gain_bytes=%zu actual_free_loss_bytes=%zu remaining_bytes=%zu fit=%d\n",
                    __func__, attempt, requested, released, free_before, free, free > free_before ? free - free_before : 0, free_before > free ? free_before - free : 0, need > free ? need - free : 0, need <= free);
            // fork: a logical shrink may stay inside one physical allocation.
            retry_loan_floor = released > 0 && free <= free_before ? (size_t) 2 << 20 : 0;
            if (need > free && released == 0 && free <= free_before) {
                LLAMA_LOG_WARN("%s: [B1] legacy cache loan made no logical or physical progress\n", __func__);
                break;
            }
        }
        LLAMA_LOG_INFO("%s: [B1] legacy layer fit: range [%d,%d), compute %.2f MiB, free %.2f MiB, keep %.2f MiB\n",
                __func__, worst_l0, worst_l1, maximum[gpu_index] / 1048576.0, free / 1048576.0, keep / 1048576.0);
        if (need > free) { return false; }
        b1->reserve_layer_bind(worst_l0);
        b1_cur = { true, worst_l0, worst_l1, b1->carry(max_ub), nullptr, 0 };
        if (!graph_reserve(tokens, 1, outputs, reserve_mctx.get())) { return false; }
        b1->reserve_layer_restore();
        b1_cur = {};
        for (size_t i = 0; i < maximum.size(); ++i) {
            const size_t actual = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
            backend_buf_exp_size[i] = actual;
            LLAMA_LOG_INFO("%s: [B1] legacy reserved %s: %.2f MiB (range maximum %.2f MiB)\n", __func__,
                    ggml_backend_name(backend_ptrs[i]), actual / 1048576.0, maximum[i] / 1048576.0);
        }
        backend_buf_exp_epoch = ggml_backend_sched_get_alloc_epoch(sched.get());
        ggml_backend_dev_memory(dev, &free, &total);
        // Allocator padding and physical pool growth are only known after the
        // real reserve. Keep its live compute buffers, and lend only the measured
        // remaining deficit before any worker, input upload or state apply. Logical
        // slot progress permits a bounded retry across allocation granularity.
        LLAMA_LOG_INFO("%s: [B1] legacy post-allocation fit: free_bytes=%zu keep_bytes=%zu remaining_bytes=%zu fit=%d\n",
                __func__, free, keep, keep > free ? keep - free : 0, keep <= free);
        constexpr int max_post_alloc_loan_attempts = 4;
        size_t post_alloc_retry_loan_floor = 0;
        for (int attempt = 1; attempt <= max_post_alloc_loan_attempts && keep > free; ++attempt) {
            const size_t free_before = free;
            const size_t requested = std::max(keep - free, post_alloc_retry_loan_floor);
            b1_cache_needs_refresh = true;
            const size_t released = ggml_backend_sched_moe_cache_trim(sched.get(), requested);
            ggml_backend_dev_memory(dev, &free, &total);
            LLAMA_LOG_INFO("%s: [B1] legacy post-allocation loan probe attempt=%d requested_bytes=%zu released_bytes=%zu free_before=%zu free_after=%zu actual_free_gain_bytes=%zu actual_free_loss_bytes=%zu remaining_bytes=%zu fit=%d\n",
                    __func__, attempt, requested, released, free_before, free, free > free_before ? free - free_before : 0, free_before > free ? free_before - free : 0, keep > free ? keep - free : 0, keep <= free);
            // fork: a logical shrink may stay inside one physical allocation.
            post_alloc_retry_loan_floor = released > 0 && free <= free_before ? (size_t) 2 << 20 : 0;
            if (keep > free && released == 0 && free <= free_before) {
                LLAMA_LOG_WARN("%s: [B1] legacy post-allocation cache loan made no logical or physical progress\n", __func__);
                break;
            }
        }
        if (free < keep) { return false; }
        success = true;
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: [B1] legacy layer probe failed: %s\n", __func__, e.what());
        return false;
    }
}

int llama_context::b1_prepare(llama_memory_context_i * mctx, const llama_batch_ext & batch_inp, uint32_t n_tokens_all, int64_t n_outputs_all) {
    if (!b1_tried) {
        b1_tried = true;
        b1_env   = b1_ctx_env("LLAMA_B1", 0) != 0;
        if (b1_env) {
            const char * why = nullptr;
            ggml_backend_t gpu = nullptr;
            int n_gpu = 0;
            for (auto & be : backends) {
                ggml_backend_dev_t d = ggml_backend_get_device(be.get());
                if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    ++n_gpu;
                    gpu = be.get();
                }
            }
            if (model.arch != LLM_ARCH_QWEN4EXP) {
                why = "only qwen4exp builds layer-range graphs";
            } else if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
                why = "this is a draft (MTP) context; only the main context reads prompts layer-major";
            } else if (n_gpu != 1) {
                why = "it needs exactly one GPU";
            } else if (cparams.embeddings) {
                why = "the context computes embeddings";
            } else if (!cparams.op_offload) {
                why = "op offload is off, so the normal path computes the experts in RAM on the CPU";
            }
            if (why) {
                LLAMA_LOG_WARN("%s: [B1] LLAMA_B1 is set but layer-major prefill is off: %s\n", __func__, why);
            } else {
                b1 = std::make_unique<llama_b1>(model, gpu);
                if (!b1->ok()) {
                    b1.reset();
                }
            }
        }
    }
    if (!b1) {
        return 0;
    }

    auto normal = [&]() -> int {
        b1_cache_resume();
        if (b1->dirty() && !b1->refill()) {
            LLAMA_LOG_ERROR("%s: [B1] cannot restore the lent expert layers\n", __func__);
            return -3;
        }
        return 0;
    };

    // per call: the conditions another component may have changed since construction
    if (moe_cache || std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                [](bool b) { return b; })) {
        if (!b1_off_logged) {
            b1_off_logged = true;
            LLAMA_LOG_WARN("%s: [B1] prompts take the normal path: %s\n", __func__,
                moe_cache ? "the MoE cache is on (--moe-cache off enables B1)" :
                "layer inputs are being extracted");
        }
        return normal();
    }

    // one sequence; batches that mix slots take the normal path
    llama_seq_id seq = -1;
    for (const auto & tok : batch_inp.tokens) {
        if (tok.seq_ids.size() != 1) {
            return normal();
        }
        const llama_seq_id s = *tok.seq_ids.begin();
        if (seq < 0) {
            seq = s;
        } else if (s != seq) {
            return normal();
        }
    }

    const std::vector<uint32_t> sizes = mctx->b1_ubatch_sizes();
    // auto threshold: shorter calls take the normal path. A one-ubatch call also runs layer-major: it moves the same
    // expert bytes as the normal path but overlaps the copies with compute (the server sends the last n_ubatch + 4
    // prompt tokens as their own calls for its context checkpoints)
    const int64_t min_tok = b1_ctx_env("LLAMA_B1_MIN_TOKENS", 16384);
    if (sizes.empty() || (int64_t) n_tokens_all < min_tok) {
        return normal();
    }
    if (!mctx->b1_supported()) {
        if (!b1_off_logged) {
            b1_off_logged = true;
            LLAMA_LOG_WARN("%s: [B1] prompts take the normal path: the memory cannot replay its ubatches layer by layer "
                "(rollback ring, --gdn-replay, KV block streaming or a full cache)\n", __func__);
        }
        return normal();
    }

    // fork: retained state in another slot conservatively forbids borrowing.
    bool others = false;
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq_max() && !others; ++s) {
        others = s != seq && memory->seq_pos_max(s) >= 0;
    }
    if (others) {
        return normal();
    }
    size_t lend_cap = SIZE_MAX;
    const int64_t lend_mib = b1_ctx_env("LLAMA_B1_LEND_MIB", -1);
    if (lend_mib >= 0) {
        lend_cap = std::min(lend_cap, (size_t) lend_mib << 20);
    }

    // tokens per group (S): 0 = the whole call
    const int64_t s_tok = b1_ctx_env("LLAMA_B1_GROUP", 0);
    uint32_t group_ub = 0;
    if (s_tok > 0) {
        group_ub = (uint32_t) std::max<int64_t>(1, (s_tok + cparams.n_ubatch/2)/cparams.n_ubatch);
    }

    // the last layer is cropped to the output rows before its FFN; when even all outputs together are too few for the
    // scheduler to copy RAM experts to the GPU, the normal path computes that layer's RAM experts on the CPU in every
    // ubatch, so B1 leaves them home (same result, one layer less to stream). Otherwise ubatch_home() matches it per ubatch.
    const bool final_home = !b1->host_rows_on_gpu(b1_final_rows(n_outputs_all, n_tokens_all));
    // the MoE expert cache (--moe-cache) keeps the free VRAM above its reserve once it has run, which leaves B1 no
    // room when the experts are in RAM. Opt-in (LLAMA_B1_MOE_CACHE_SUSPEND=1): free its pool for this call when no
    // other sequence decodes from it; b1_end makes it again and it refills from decode. Off by default because the
    // refilled pool serves different experts than a pool that stayed warm, so later steps can differ in rounding.
    b1_mc_suspended = !others && b1_ctx_env("LLAMA_B1_MOE_CACHE_SUSPEND", 0) != 0 &&
                      b1_cache_graph_reset() && ggml_backend_sched_moe_cache_suspend(sched.get());
    if (b1_mc_suspended) {
        b1_cache_needs_refresh = true;
        LLAMA_LOG_WARN("%s: [B1] MoE expert cache suspended for this call\n", __func__);
    }
    const char * rotate = getenv("LLAMA_MOE_CACHE_RELEASE_PROMPT");
    size_t borrow = 0;
    const bool rotating = rotate && rotate[0] == '1';
    const bool packed_ffn = rotating && b1_ctx_env("LLAMA_B1_FFN", 0) != 0 && loras->empty() &&
        model.exl3_sides.empty() && model.hadamard_inverses.empty() && n_tokens_all <= (1u << 20) &&
        model.hparams.n_embd <= 65536 && model.hparams.n_expert <= 4096 && model.hparams.n_expert_used() <= 128;
    b1->set_ffn_enabled(packed_ffn);
    const bool lean_legacy = b1_layer_scratch_active && !packed_ffn;
    b1_kv_scratch_enabled = false;
    const bool kv_cold = memory->seq_pos_max(seq) < 0;

    detach_compute_borrower();
    bool planned = b1->begin(sizes, group_ub, lend_cap, final_home, rotating ? &borrow : nullptr, lean_legacy);
    if (!planned && borrow > 0) {
        synchronize();
        ggml_backend_dev_t loan_device = nullptr;
        for (ggml_backend_t backend : backend_ptrs) {
            auto dev = ggml_backend_get_device(backend);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            loan_device = dev;
            using reset_fn = bool (*)(ggml_backend_t);
            auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
            if (!reset || !reset(backend)) {
                return normal();
            }
        }
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        ggml_backend_sched_reset(sched.get());
        if (!loan_device) { return normal(); }
        b1_cache_needs_refresh = true;
        // Loaned slot bytes and physical free VRAM can differ at allocation
        // granularity. Allow bounded logical progress across that boundary, and
        // re-probe the full group, never a smaller post-loan plan.
        constexpr int max_loan_attempts = 4;
        size_t retry_loan_floor = 0;
        for (int attempt = 1; attempt <= max_loan_attempts && !planned && borrow > 0; ++attempt) {
            size_t free_before = 0, free_after = 0, total = 0;
            ggml_backend_dev_memory(loan_device, &free_before, &total);
            const size_t requested = std::max(borrow, retry_loan_floor);
            const size_t released = ggml_backend_sched_moe_cache_trim(sched.get(), requested);
            ggml_backend_dev_memory(loan_device, &free_after, &total);
            LLAMA_LOG_INFO("%s: B1 scratch borrowed %.2f MiB from runtime experts (requested %.2f MiB)\n", __func__, released / 1048576.0, requested / 1048576.0);
            planned = b1->begin(sizes, group_ub, lend_cap, final_home, &borrow, lean_legacy);
            LLAMA_LOG_INFO("%s: [B1] loan probe attempt=%d requested_bytes=%zu released_bytes=%zu free_before=%zu free_after=%zu actual_free_gain_bytes=%zu actual_free_loss_bytes=%zu remaining_bytes=%zu fit=%d\n",
                    __func__, attempt, requested, released, free_before, free_after, free_after > free_before ? free_after - free_before : 0, free_before > free_after ? free_before - free_after : 0, borrow, planned);
            // fork: a logical shrink may stay inside one physical allocation.
            retry_loan_floor = released > 0 && free_after <= free_before ? (size_t) 2 << 20 : 0;
            if (!planned && borrow > 0 && released == 0 && free_after <= free_before) {
                LLAMA_LOG_WARN("%s: [B1] cache loan made no logical or physical progress; using normal prefill\n", __func__);
                break;
            }
        }
    }
    if (!planned) {
        return normal();
    }
    if (lean_legacy) {
        if (b1->n_groups() != 1 || !b1_reserve_legacy_layers(sizes, n_outputs_all)) {
            LLAMA_LOG_WARN("%s: [B1] legacy one-pass layer reservation failed before applying prompt state\n", __func__);
            b1_end(true);
            return normal();
        }
        // detach_compute_borrower() has synchronized the serialized helper, removed
        // its donor and invalidated its allocator. It cannot retain these views;
        // its next reserve runs only after this decode returns all B1 scratch.
        // Device-prefill hidden rows have independent storage, outside KV views.
        const bool peer_detached = !compute_peer ||
            (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
             compute_peer->cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
             !compute_peer->mtp_prefill_async && !compute_peer->mtp_scratch_active &&
             compute_peer->sched_need_reserve);
        b1_kv_scratch_enabled = kv_cold && peer_detached && mtp_scratch_clients.empty() && n_tokens_all <= UINT32_MAX - 512 && b1->n_groups() == 1 && b1_ctx_env("LLAMA_B1_KV_SCRATCH", 0) != 0;
        b1_kv_prompt_cells = n_tokens_all + 512;
        b1_kv_outputs = (uint32_t) std::max<int64_t>(1, std::min<int64_t>(n_tokens_all, n_outputs_all));
        b1->start_streaming();
    }

    // apply every ubatch now, saving each memory view
    auto * sj = llama_sj_kvarn_cache(memory.get());
    if (sj && !sj->is_sj_kvarn()) {
        sj = nullptr;
    }
    if (sj) {
        sj->b1_sj_begin(seq);
    }
    size_t n_applied = 0;
    do {
        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: [B1] failed to apply memory context of ubatch %zu\n", __func__, n_applied);
            b1_end(true);
            if (sj) {
                sj->b1_sj_rewind();
            }
            memory->seq_rm(seq, balloc->seq_pos_min(seq), -1);
            return -3;
        }
        mctx->b1_save();
        ++n_applied;
    } while (mctx->next());

    if (n_applied != sizes.size() || !mctx->b1_seek(0)) {
        LLAMA_LOG_ERROR("%s: [B1] the memory context did not keep its ubatch views\n", __func__);
        b1_end(true);
        if (sj) {
            sj->b1_sj_rewind();
        }
        memory->seq_rm(seq, balloc->seq_pos_min(seq), -1);
        return -3;
    }

    b1_compute_submit_us = 0;
    b1_cache_begin();
    b1_cache_collect = b1_cache_warm_enabled && b1_cache_collect_wanted();

    LLAMA_LOG_WARN("%s: [B1] layer-major prefill of %u tokens: %zu ubatches, %zu groups of %zu, %s\n", __func__,
            n_tokens_all, sizes.size(), b1->n_groups(), b1->group_ub(), others ? "other sequences present" : "alone");
    b1_memory_trim("B1 prepared");
    return 1;
}

// [PREFILLFAST] LLAMA_B1_CACHE_COLLECT=0: do not read back the prompt's expert routing during the
// layer-major prefill. Collecting installs a per-node host callback for the whole prompt pass: every
// ffn_moe_topk node splits the layer graph, waits for a read-back, and turns off CUDA graphs (~2.2 s
// at 100K). With GGML_MOE_CACHE_PROFILE_GLOBAL=1 the post-prompt warm ranks experts by the saved
// profile and ignores these counts, so the warm is the same. The split also stops the CUDA backend
// fusing the routing ops across it, so collect-off runs the fused routing math that every other
// decode uses; tokens then differ by rounding only (identical with GGML_CUDA_DISABLE_FUSION=1 in
// both arms). Default (owner ruling 2026-10-21): no read-back when GGML_MOE_CACHE_PROFILE_GLOBAL=1, because the warm
// then ignores the counts; without the global profile the warm still needs them, so collection stays on.
// LLAMA_B1_CACHE_COLLECT=1 / =0 forces either way.
bool llama_context::b1_cache_collect_wanted() {
    static const bool wanted = [] {
        const char * e = getenv("LLAMA_B1_CACHE_COLLECT");
        if (e && e[0]) {
            return strcmp(e, "0") != 0;
        }
        const char * g = getenv("GGML_MOE_CACHE_PROFILE_GLOBAL");
        return !(g && strcmp(g, "1") == 0);
    }();
    return wanted;
}

void llama_context::b1_cache_begin() {
    b1_cache_warm_enabled = b1_ctx_env("LLAMA_B1_CACHE_WARM", 0) != 0 ||
        b1_ctx_env("LLAMA_MOE_CACHE_RELEASE_PROMPT", 0) != 0;
    const size_t n_layers = model.hparams.n_layer();
    const size_t n_experts = model.hparams.n_expert;
    if (b1_cache_warm_enabled && b1_cache_counts.empty()) {
        b1_cache_readbacks = 0;
        if (n_experts == 0 || n_experts > 4096 || n_layers > (1u << 20) / n_experts) {
            b1_cache_warm_enabled = false;
        } else {
            try {
                b1_cache_counts.assign(n_layers, std::vector<uint64_t>(n_experts, 0));
            } catch (...) {
                b1_cache_warm_enabled = false;
            }
        }
    }
}

bool llama_context::b1_cache_eval(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & ctx = *static_cast<llama_context *>(user_data);
    int layer = -1;
    int parsed = 0;
    int source_layer = -1;
    int target_layer = -1;
    const bool predicted = ctx.cparams.moe_prefetch != 0 &&
        sscanf(tensor->name, "moe_prefetch-%d-%d%n", &source_layer, &target_layer, &parsed) == 2 &&
        parsed > 0 && tensor->name[parsed] == '\0' && source_layer >= 0 && target_layer > source_layer &&
        target_layer - source_layer <= 3 && (size_t) target_layer < ctx.model.layers.size() &&
        tensor->type == GGML_TYPE_I32 && tensor->ne[0] > 0 && tensor->ne[0] <= 32 &&
        tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
    parsed = 0;
    int observed_layer = -1;
    const bool entry = ctx.cparams.moe_prefetch != 0 &&
        sscanf(tensor->name, "moe_prefetch_layer_start-%d%n", &observed_layer, &parsed) == 1 &&
        parsed > 0 && tensor->name[parsed] == '\0' && tensor->ne[1] > 0 && tensor->ne[1] <= 8;
    parsed = 0;
    const size_t max_bytes = 64u << 20;
    const bool valid_shape = tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
        (uint64_t) tensor->ne[0] <= max_bytes / sizeof(int32_t) &&
        tensor->nb[1] >= (size_t) tensor->ne[0] * sizeof(int32_t) && tensor->nb[1] <= max_bytes &&
        (uint64_t) tensor->ne[1] <= max_bytes / tensor->nb[1];
    const bool routed = valid_shape && sscanf(tensor->name, "ffn_moe_topk-%d%n", &layer, &parsed) == 1 &&
        parsed > 0 && tensor->name[parsed] == '\0' && layer >= 0 && (size_t) layer < ctx.model.layers.size() &&
        tensor->type == GGML_TYPE_I32 && tensor->nb[0] == sizeof(int32_t) && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
        ggml_nbytes(tensor) <= (64u << 20);
    if (ask) {
        const bool user_need = ctx.cparams.cb_eval && ctx.cparams.cb_eval(tensor, true, ctx.cparams.cb_eval_user_data);
        if (user_need) {
            ctx.b1_cache_user_node = tensor;
        }
        return (routed && (ctx.b1_cache_collect || (ctx.cparams.moe_prefetch != 0 && tensor->ne[1] <= 8))) || predicted || entry || user_need;
    }
    if (ctx.cparams.moe_prefetch != 0) {
        const auto api = ggml_moe_cache_active();
        if (entry && api.prefetch_observe) {
            api.prefetch_observe(nullptr, observed_layer, nullptr, 0);
        }
        if ((predicted || (routed && tensor->ne[1] <= 8)) && tensor->ne[0] <= 64) {
            int32_t ids[512];
            const size_t row_bytes = tensor->ne[0] * sizeof(int32_t);
            ggml_backend_tensor_get_2d(tensor, ids, 0, row_bytes, tensor->ne[1], tensor->nb[1], row_bytes);
            if (predicted && api.prefetch && tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
                const auto & future = ctx.model.layers[target_layer];
                ggml_moe_cache_tensor_desc descs[3] = {};
                ggml_tensor * weights[] = {future.ffn_up_exps, future.ffn_gate_exps, future.ffn_down_exps};
                int count = 0;
                for (auto * w : weights) {
                    if (!w || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
                        continue;
                    }
                    descs[count++] = {w->name, w->data, w->nb[2], w->ne[0], w->ne[1], w->ne[2], (int32_t) w->type};
                }
                api.prefetch(nullptr, descs, count, ids, (int) tensor->ne[0], source_layer, target_layer, ctx.cparams.moe_prefetch < 0);
            } else if (routed && api.prefetch_observe) {
                api.prefetch_observe(nullptr, layer, ids, (int) (tensor->ne[0] * tensor->ne[1]));
            }
        }
    }
    if (routed && ctx.b1_cache_collect && (size_t) layer < ctx.b1_cache_counts.size()) {
        try {
            // argsort_top_k is normally a ten-row view of the full expert
            // sort. Its token stride still covers every expert. Read only
            // the selected rows, rather than the unused gaps between tokens.
            // valid_shape bounds the source span; keep the compact allocation
            // and its multiplication independently bounded as well.
            const size_t row_bytes = (size_t) tensor->ne[0] * sizeof(int32_t);
            const size_t n_tokens = (size_t) tensor->ne[1];
            GGML_ASSERT(row_bytes > 0 && row_bytes <= max_bytes);
            GGML_ASSERT(n_tokens <= max_bytes / row_bytes);
            std::vector<int32_t> ids((row_bytes / sizeof(int32_t)) * n_tokens);
            ggml_backend_tensor_get_2d(tensor, ids.data(), 0, row_bytes,
                    n_tokens, tensor->nb[1], row_bytes);
            auto & counts = ctx.b1_cache_counts[layer];
            for (int64_t token = 0; token < tensor->ne[1]; ++token) {
                for (int64_t rank = 0; rank < tensor->ne[0]; ++rank) {
                    const int32_t expert = ids[(size_t) token * (size_t) tensor->ne[0] + (size_t) rank];
                    if (expert >= 0 && (size_t) expert < counts.size() && counts[expert] < UINT64_MAX) {
                        ++counts[expert];
                    }
                }
            }
            ++ctx.b1_cache_readbacks;
        } catch (...) {
            ctx.b1_cache_warm_enabled = false;
            ctx.b1_cache_counts.clear();
            ctx.b1_cache_readbacks = 0;
            LLAMA_LOG_WARN("%s: prompt cache warm disabled after routing capture allocation failure\n", __func__);
        }
    }
    if (ctx.b1_cache_user_node == tensor) {
        ctx.b1_cache_user_node = nullptr;
        return ctx.cparams.cb_eval(tensor, false, ctx.cparams.cb_eval_user_data);
    }
    return true;
}

// Explicit GENERATION is the application handoff after all peer contexts have been allocated.
// A target constructor/reserve must never spend the draft model's still-free card memory.
void llama_context::b1_cache_profile_load() {
    const char * profile = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
    if (b1_cache_profile_attempted || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
            model.hparams.no_alloc || !profile || !profile[0]) {
        return;
    }
    b1_cache_profile_attempted = true;
    synchronize();
    try {
        std::vector<ggml_moe_cache_tensor_desc> tensors;
        for (const auto & entry : model.tensors_by_name) {
            int layer = -1;
            int parsed = 0;
            ggml_tensor * tensor = entry.second;
            if (sscanf(entry.first.c_str(), "blk.%d.%n", &layer, &parsed) != 1 || parsed <= 0 ||
                    entry.first.find("_exps") == std::string::npos || layer < 0 || (uint32_t) layer >= model.hparams.n_layer() ||
                    !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || !tensor->data ||
                    tensor->ne[2] != model.hparams.n_expert || tensor->ne[3] != 1 || (!ggml_is_contiguous_2(tensor) || tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0])) ||
                    !ggml_moe_cache_wtype_supported(tensor->type)) {
                continue;
            }
            tensors.push_back({tensor->name, tensor->data, tensor->nb[2], tensor->ne[0], tensor->ne[1], tensor->ne[2], (int32_t) tensor->type});
        }
        for (ggml_backend_t backend : backend_ptrs) {
            auto * device = ggml_backend_get_device(backend);
            if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                model.moe_resident_init(device, tensors.data(), (int) tensors.size());
                break;
            }
        }
        ggml_backend_sched_moe_cache_warm(sched.get(), tensors.data(), nullptr, (int) tensors.size());
        const char * device = getenv("GGML_MOE_CACHE_DEVICE");
        b1_cache_device_pending = device && strcmp(device, "1") == 0;
    } catch (const std::bad_alloc &) {
        LLAMA_LOG_WARN("%s: profile warm skipped after host allocation failure\n", __func__);
    }
}

void llama_context::b1_cache_warm() {
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || (!b1_cache_needs_refresh && b1_cache_counts.empty())) {
        return;
    }
    try {
        b1_cache_begin();
        if (b1_cache_counts.empty()) {
            return;
        }
        // Restore resets the scheduler even when it returns no bytes. Its
        // rewritten cross-backend source tensors live in that reset arena,
        // so reuse requires a fresh model graph on both legacy and device paths.
        synchronize();
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        const size_t returned = ggml_backend_sched_moe_cache_restore(sched.get());
        LLAMA_LOG_INFO("%s: returned %.2f MiB to runtime expert cache\n", __func__, returned / 1048576.0);
        std::vector<ggml_moe_cache_tensor_desc> tensors;
        std::vector<const uint64_t *> counts;
        for (const auto & entry : model.tensors_by_name) {
            int layer = -1;
            int parsed = 0;
            ggml_tensor * tensor = entry.second;
            if (sscanf(entry.first.c_str(), "blk.%d.%n", &layer, &parsed) != 1 || parsed <= 0 ||
                entry.first.find("_exps") == std::string::npos || layer < 0 || (size_t) layer >= b1_cache_counts.size() ||
                !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || !tensor->data ||
                tensor->ne[2] != (int64_t) b1_cache_counts[layer].size() || tensor->ne[3] != 1 ||
                (!ggml_is_contiguous_2(tensor) || tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0])) || !ggml_moe_cache_wtype_supported(tensor->type)) {
                continue;
            }
            tensors.push_back({tensor->name, tensor->data, tensor->nb[2], tensor->ne[0], tensor->ne[1], tensor->ne[2], (int32_t) tensor->type});
            counts.push_back(b1_cache_counts[layer].data());
        }
        const int queued = ggml_backend_sched_moe_cache_warm(sched.get(), tensors.data(), counts.data(), (int) tensors.size());
        // Defer table publication until the prompt outputs have been consumed
        // and the next generation decode has resized its output buffer.
        const char * device = getenv("GGML_MOE_CACHE_DEVICE");
        b1_cache_device_pending = device && strcmp(device, "1") == 0;
        LLAMA_LOG_INFO("%s: [B1] cache warm: %zu tensors, %llu synchronized routing readbacks, %d fills queued\n", __func__,
                       tensors.size(), (unsigned long long) b1_cache_readbacks, queued);
    } catch (...) {
        LLAMA_LOG_WARN("%s: prompt cache warm skipped after allocation failure\n", __func__);
    }
    b1_cache_counts.clear();
    b1_cache_readbacks = 0;
    b1_cache_needs_refresh = false;
}

// This generation-sized decode has already consumed the prompt outputs and
// resized its output arena. Pin exact cold weights only now, before its graph.
void llama_context::b1_cache_device_prepare() {
    if (!b1_cache_device_pending || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return;
    }
    b1_cache_device_pending = false;
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE");
    if (!enabled || strcmp(enabled, "1") != 0) {
        return;
    }
    try {
        // Both captured CUDA graphs and cached model graphs can retain old
        // cache addresses. Refuse activation unless all can be invalidated.
        if (!b1_cache_graph_reset()) {
            LLAMA_LOG_WARN("%s: device prepare skipped: graph reset unavailable\n", __func__);
            return;
        }
        ggml_backend_sched_reset(sched.get());
        std::vector<ggml_moe_cache_tensor_desc> tensors;
        for (const auto & entry : model.tensors_by_name) {
            int layer = -1;
            int parsed = 0;
            ggml_tensor * tensor = entry.second;
            if (sscanf(entry.first.c_str(), "blk.%d.%n", &layer, &parsed) != 1 || parsed <= 0 ||
                    entry.first.find("_exps") == std::string::npos || layer < 0 || (uint32_t)layer >= model.hparams.n_layer() ||
                    !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || !tensor->data ||
                    tensor->ne[2] != model.hparams.n_expert || tensor->ne[3] != 1 || (!ggml_is_contiguous_2(tensor) || tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0])) ||
                    !ggml_moe_cache_wtype_supported(tensor->type)) {
                continue;
            }
            tensors.push_back({tensor->name, tensor->data, tensor->nb[2], tensor->ne[0], tensor->ne[1], tensor->ne[2], (int32_t)tensor->type});
        }
        const int64_t started = ggml_time_us();
        const int ready = ggml_backend_sched_moe_cache_device_prepare(sched.get(), tensors.data(), (int)tensors.size());
        LLAMA_LOG_INFO("%s: deferred device expert prepare: tensors=%zu ready=%d elapsed_ms=%.3f phase=post-output-reserve-pregraph\n",
                __func__, tensors.size(), ready, (ggml_time_us() - started) / 1000.0);
        if (ready > 0) {
            LLAMA_LOG_INFO("%s: device expert dispatch activated; response graphs invalidated\n", __func__);
        }
    } catch (const std::bad_alloc &) {
        LLAMA_LOG_WARN("%s: deferred device prepare skipped after host allocation failure\n", __func__);
    }
}

int64_t llama_context::b1_final_rows(int64_t n_out, int64_t n_tok) const {
    // the graph crops the last layer to the output rows before its FFN unless the nextn hidden state needs every row
    return !cparams.embeddings_nextn || cparams.embeddings_nextn_masked ? n_out : n_tok;
}

void llama_context::b1_end(bool failed) {
    b1->end(failed);
    b1_cache_resume();
}

bool llama_context::b1_cache_graph_reset() {
    synchronize();
    // Cached CUDA graphs can retain pool addresses even after model graph metadata is discarded.
    for (ggml_backend_t backend : backend_ptrs) {
        auto dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        using reset_fn = bool (*)(ggml_backend_t);
        auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
        if (!reset || !reset(backend)) {
            return false;
        }
    }
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;
    return true;
}

void llama_context::b1_cache_resume() {
    if (!b1_mc_suspended) {
        return;
    }
    // Suspension required this hook, so it must remain available until the session is restored.
    GGML_ASSERT(b1_cache_graph_reset());
    ggml_backend_sched_moe_cache_resume(sched.get());
    b1_mc_suspended = false;
}

bool llama_context::b1_run_prefix(llama_memory_context_i * mctx, size_t ub0, ggml_status & status) {
    const int32_t l_final = b1->final_l0();
    const size_t  ub1     = b1->group_end(ub0);
    const llm_graph_type gtype = ctx_type_to_graph_type(cparams.ctx_type);

    status = GGML_STATUS_FAILED;
    for (int32_t l0 = 0; l0 < l_final; ) {
        const int32_t l1 = b1->range_end(l0);
        if (!b1->layer_begin(l0) || !b1_kv_scratch_bind(l0, l1)) {
            return false;
        }
        for (size_t u = ub0; u < ub1; ++u) {
            if (!mctx->b1_seek(u)) {
                LLAMA_LOG_ERROR("%s: [B1] cannot seek to ubatch %zu\n", __func__, u);
                return false;
            }
            n_outputs = 0;
            b1_cur = { true, l0, l1, b1->carry(u), b1->ffn(u), b1->ffn_enabled() ? 1 : 0 };
            // a short ubatch: the normal path computes the RAM experts of this layer on the CPU, so B1 does too
            const bool home = b1->layer_streamed(l0) && !b1->host_rows_on_gpu(mctx->get_ubatch().n_tokens);
            if (home) {
                b1->ubatch_home(l0, true);
            }
            const int64_t t_compute0 = ggml_time_us();
            const auto * res = process_ubatch(mctx->get_ubatch(), gtype, mctx, status);
            const int64_t compute_us = ggml_time_us() - t_compute0;
            b1_compute_submit_us += compute_us;
            b1->pass_compute(u, compute_us);
            if (home) {
                b1->ubatch_home(l0, false);
            }
            b1_cur = {};
            if (!res) {
                return false;
            }
            capture_ubatch(mctx->get_ubatch(), l0, l1);
        }
        if (b1->ffn_enabled()) {
            if (!b1->ffn_gather() || !mctx->b1_seek(ub0)) {
                return false;
            }
            size_t busiest = 0, pieces = 0, used_experts = 0;
            for (const auto & rows : b1->ffn_group()->expert_rows) {
                busiest = std::max(busiest, rows.size());
                pieces += (rows.size() + b1->ffn_group()->max_rows - 1) / b1->ffn_group()->max_rows;
                used_experts += !rows.empty();
            }
            LLAMA_LOG_INFO("%s: [B1] whole-prompt FFN layer %d: %zu experts, busiest %zu rows, %zu GEMM pieces\n", __func__, l0, used_experts, busiest, pieces);
            int64_t t_compute = 0;
            int64_t compute_us = 0;
            auto * ffn_group = b1->ffn_group();
            const size_t graphs = ffn_group->grouped_mmid ? ffn_group->buckets.size() :
                    (pieces + ffn_group->piece_count - 1) / ffn_group->piece_count;
            if (ffn_group->grouped_mmid) {
                LLAMA_LOG_INFO("%s: [B1] grouped routed-row GEMMs: %zu buckets, max %u rows\n", __func__, graphs, ffn_group->routed_rows);
            }
            for (size_t graph = 0; graph < graphs; ++graph) {
                ffn_group->piece_first = graph * ffn_group->piece_count;
                ffn_group->bucket_index = graph;
                b1_cur = { true, l0, l1, b1->carry(ub0), b1->ffn_group(), 2 };
                t_compute = ggml_time_us();
                auto * packed = process_ubatch(mctx->get_ubatch(), gtype, mctx, status);
                compute_us = ggml_time_us() - t_compute;
                b1_compute_submit_us += compute_us;
                b1->pass_compute(ub0, compute_us);
                b1_cur = {};
                if (!packed) {
                    return false;
                }
            }
            if (b1_ctx_env("LLAMA_B1_FFN_CHECK", 0) != 0) {
                for (size_t u = ub0; u < ub1; ++u) {
                    if (!mctx->b1_seek(u)) {
                        return false;
                    }
                    const auto * ffn = b1->ffn(u);
                    b1_cur = { true, l0, l1, b1->carry(u), ffn, 4 };
                    auto * reference = process_ubatch(mctx->get_ubatch(), gtype, mctx, status);
                    b1_cur = {};
                    if (!reference || !reference->t_embd) {
                        return false;
                    }
                    ggml_backend_sched_synchronize(sched.get());
                    const size_t count = ggml_nelements(ffn->output);
                    std::vector<float> expected(count), actual(count);
                    ggml_backend_tensor_get(reference->t_embd, expected.data(), 0, count * sizeof(float));
                    ggml_backend_tensor_get(ffn->output, actual.data(), 0, count * sizeof(float));
                    double error = 0.0, norm = 0.0, max_error = 0.0;
                    for (size_t i = 0; i < count; ++i) {
                        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
                            return false;
                        }
                        const double d = (double) actual[i] - expected[i];
                        error += d*d;
                        norm += (double) expected[i]*expected[i];
                        max_error = std::max(max_error, std::abs(d));
                    }
                    const double nmse = error / std::max(norm, 1e-30);
                    LLAMA_LOG_INFO("%s: [B1] packed FFN check layer %d ubatch %zu: NMSE %.9g max_abs %.9g (%zu values)\n", __func__, l0, u, nmse, max_error, count);
                    if (nmse > 1e-4) {
                        status = GGML_STATUS_FAILED;
                        return false;
                    }
                }
            }
            for (size_t u = ub0; u < ub1; ++u) {
                if (!mctx->b1_seek(u)) {
                    return false;
                }
                b1_cur = { true, l0, l1, b1->carry(u), b1->ffn(u), 3 };
                t_compute = ggml_time_us();
                auto * tail = process_ubatch(mctx->get_ubatch(), gtype, mctx, status);
                compute_us = ggml_time_us() - t_compute;
                b1_compute_submit_us += compute_us;
                b1->pass_compute(u, compute_us);
                b1_cur = {};
                if (!tail) {
                    return false;
                }
            }
        }
        b1->layer_end(l0);
        b1_memory_trim("B1 layer complete");
        l0 = l1;
    }

    // The final output range can reuse the card space used by the packed expert inputs and accumulator.
    b1->ffn_release();
    if (!mctx->b1_seek(ub0) || !b1->layer_begin(l_final) || !b1_kv_scratch_bind(l_final, model.hparams.n_layer())) {
        status = GGML_STATUS_FAILED;
        return false;
    }
    status = GGML_STATUS_SUCCESS;
    return true;
}

llm_graph_params llama_context::graph_params(llm_graph_result *             res,
                                             const llama_ubatch &           ubatch,
                                             const llama_memory_context_i * mctx,
                                             llm_graph_type                 gtype) const {
    llm_graph_params p = {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.hadamard_rotations =*/ &model.hadamard_rotations,
        /*.hadamard_inverses  =*/ &model.hadamard_inverses,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.samplers    =*/ sampling.samplers,
        /*.draft_vocab =*/ draft_vocab.ids,
        /*.draft_vocab_compact =*/ draft_vocab.compact,
        /*.draft_vocab_compact_src =*/ draft_vocab.compact_src,
        /*.draft_vocab_warned =*/ &draft_vocab.warned,
        /*.mtp_chain_samp =*/ &mtp_chain_samp,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.model       =*/ &model,
    };
    p.ple_pf = ple_pf.get();
    if (b1_cur.active) {
        p.b1_il0   = b1_cur.il0;
        p.b1_il1   = b1_cur.il1;
        p.b1_carry = b1_cur.carry;
        p.b1_ffn = b1_cur.ffn;
        p.b1_ffn_phase = b1_cur.ffn_phase;
    }
    return p;
}

ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    int               n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp        = batched ? threadpool_batch : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg               = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto * perf = llama_bookkeep_context_bucket(cparams.ctx_type);
    const auto snapshot = [&]() -> std::array<uint64_t, 10> {
        const auto p = ggml_backend_sched_get_bookkeep_perf(sched.get());
        return { p.session_host_us, p.transition_host_us, p.input_copy_host_us,
                 p.user_input_sync_host_us, p.cpu_compute_host_us, p.gpu_compute_host_us,
                 p.other_compute_host_us, p.cpu_split_calls, p.gpu_split_calls, p.other_split_calls };
    };
    const auto before = perf ? snapshot() : std::array<uint64_t, 10>{};
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (perf) {
        const auto after = snapshot();
        for (size_t i = 0; i < before.size(); ++i) {
            perf->backend_submit[i].fetch_add(after[i] - before[i], std::memory_order_relaxed);
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_mtp_inputs(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // fork: Retain each MTP input snapshot until the logical catch-up batch completes.
    if (((lctx->mtp_prefill_async && lctx->mtp_hidden_source) || lctx->mtp_scratch_active) &&
            ggml_backend_buffer_is_host(src->buffer) && ggml_is_contiguous(src) &&
            ggml_backend_get_device(backend) == (lctx->mtp_scratch_active ? lctx->model.dev_output() :
                ggml_backend_get_device(lctx->mtp_hidden_source->mtp_hidden.backend))) {
        auto & uploads = lctx->mtp_input_uploads;
        const size_t index = lctx->mtp_input_upload_next++;
        if (index == uploads.size()) {
            uploads.emplace_back();
        }
        const size_t size = ggml_nbytes(src);
        if (!uploads[index] || ggml_backend_buffer_get_size(uploads[index].get()) < size) {
            auto * buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            uploads[index].reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), size));
            if (!uploads[index]) {
                throw std::runtime_error("failed to allocate MTP input snapshot");
            }
        }
        void * data = ggml_backend_buffer_get_base(uploads[index].get());
        std::memcpy(data, src->data, size);
        ggml_backend_tensor_set_async(backend, dst, data, 0, size);
        return true;
    }
    return false;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // fork: pack arena expert planes without copying the intervening roles.
    const size_t expert_bytes = src->nb[1] * src->ne[1];
    if (src->ne[3] == 1 && src->nb[2] != dst->nb[2] && dst->nb[2] == expert_bytes &&
            strstr(src->name, "_exps.weight") && ggml_is_contiguous_2(src) &&
            src->nb[1] == ggml_row_size(src->type, src->ne[0])) {
        std::vector<bool> used((size_t)src->ne[2], true);
        const auto * node = ggml_graph_n_nodes(graph) ? ggml_graph_node(graph, 0) : nullptr;
        if (node && node->op == GGML_OP_MUL_MAT_ID && node->src[0] == dst) {
            const auto * ids = node->src[2];
            std::vector<int32_t> data(ggml_nbytes(ids) / sizeof(int32_t));
            ggml_backend_tensor_get_async(backend, ids, data.data(), 0, ggml_nbytes(ids));
            ggml_backend_synchronize(backend);
            std::fill(used.begin(), used.end(), false);
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                    const int32_t id = data[(token * ids->nb[1] + rank * ids->nb[0]) / sizeof(int32_t)];
                    GGML_ASSERT(id >= 0 && id < src->ne[2]);
                    used[id] = true;
                }
            }
        }
        for (int64_t e = 0; e < src->ne[2]; ++e) {
            if (used[e]) {
                ggml_backend_tensor_set_async(backend, dst, (const char *)src->data + e * src->nb[2],
                        e * dst->nb[2], expert_bytes);
                if (e + 1 < src->ne[2]) {
                    static const uint8_t padding[512] = {};
                    ggml_backend_tensor_set_async(backend, dst, padding, (e + 1) * dst->nb[2],
                            std::min(expert_bytes, sizeof(padding)));
                }
            }
        }
        return true;
    }

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (strcmp(name, "moe_prefetch_gpu") == 0 && il >= 0) {
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == model.dev_layer(il) &&
                        ggml_backend_supports_op(backend.get(), cur)) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                    break;
                }
            }
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
  public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override { size_written += size; }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override { return size_written; }

  private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
  public:
    llama_io_write_host(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({ tensor, ptr, size, offset });

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override { return size_written; }

  private:
    uint8_t * ptr;
    size_t    buf_size     = 0;
    size_t    size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t *     ptr;
        size_t        size;
        size_t        offset;
    };

    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
  public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({ tensor, ptr, size, offset });

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

  private:
    const uint8_t * ptr;
    size_t          buf_size  = 0;
    size_t          size_read = 0;

    struct read_info {
        ggml_tensor *   tensor;
        const uint8_t * ptr;
        size_t          size;
        size_t          offset;
    };

    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
  public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override { return size_written; }

  private:
    llama_file *         file;
    size_t               size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
  public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override { return size_read; }

  private:
    llama_file *         file;
    size_t               size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

// Packed caches can contain F16 sink bytes inside a quantized staging tensor.
static ggml_tensor * llama_io_view_bytes(ggml_context * ctx, ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    auto * view = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, size);
    view->view_src = tensor;
    view->view_offs = offset;
    return view;
}

class llama_io_write_device : public llama_io_write_i {
  public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) :
        ptr(p),
        buf_size(len),
        mbufs(mbufs) {}

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/2 * mbuf.n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(llama_io_view_bytes(mbuf.ctx.get(), winfo.tensor, winfo.offset, winfo.size));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), GGML_TYPE_I8, winfo.size));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    // fork: a split QSA checkpoint needs authoritative RAM bytes, not a second card window.
                    auto * copy_buft = strncmp(ggml_backend_buft_name(buft), "CUDA_QSA_Resident_", 18) == 0 ? ggml_backend_cpu_buffer_type() : buft;
                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), copy_buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft),
                                   mbuf.total_size / 1024.0 / 1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({ tensor, ptr, size, offset });
    }

    size_t n_bytes() override { return size_written; }

  private:
    uint8_t * ptr;
    size_t    buf_size     = 0;
    size_t    size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t *     ptr;
        size_t        size;
        size_t        offset;
    };

    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
  public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) :
        ptr(p),
        buf_size(len),
        mbufs(mbufs) {}

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/mbuf.n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(llama_io_view_bytes(mbuf.ctx.get(), rinfo.tensor, rinfo.offset, rinfo.size));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                auto * src_v = llama_io_view_bytes(ctx_scratch, src_t, src_off, n_copy);
                ggml_backend_view_init(src_v);
                auto * dst_v = llama_io_view_bytes(ctx_scratch, dst_t, dst_off, n_copy);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({ tensor, ptr, size, offset });
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

  private:
    const uint8_t * ptr;
    size_t          buf_size  = 0;
    size_t          size_read = 0;

    struct read_info {
        ggml_tensor *   tensor;
        const uint8_t * ptr;
        size_t          size;
        size_t          offset;
    };

    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

uint8_t * llama_context::ckpt_capture::blob_data() {
    return blob_file ? static_cast<uint8_t *>(blob_file->data) : blob.data();
}

const uint8_t * llama_context::ckpt_capture::blob_data() const {
    return blob_file ? static_cast<const uint8_t *>(blob_file->data) : blob.data();
}

size_t llama_context::ckpt_capture::blob_size() const {
    return blob_file ? blob_file->size : blob.size();
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (ckpt_sel >= 0 && seq_id == ckpt_seq && flags == LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return ckpt_caps[ckpt_sel].blob_size();
    }

    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (ckpt_sel >= 0 && seq_id == ckpt_seq && flags == LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        const auto & capture = ckpt_caps[ckpt_sel];
        if (size < capture.blob_size()) {
            LLAMA_LOG_ERROR("%s: error saving state: unexpectedly reached end of buffer\n", __func__);
            return 0;
        }
        if (capture.blob_file) {
            capture.blob_file->flush();
            llama_copy_moe_output(dst, capture.blob_data(), capture.blob_size());
        } else {
            memcpy(dst, capture.blob_data(), capture.blob_size());
        }
        return capture.blob_size();
    }

    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

// in-call checkpoints: lays out the PARTIAL_ONLY state like llama_io_write_host would write it, without reading the
// tensors (fill), or checks that a later layout is the same one (check)
class llama_io_write_layout : public llama_io_write_i {
  public:
    llama_io_write_layout(uint8_t * blob, size_t capacity, std::vector<llama_context::ckpt_write> & writes, bool check, bool file_backed) :
        blob(blob), capacity(capacity), writes(writes), check(check), file_backed(file_backed) {}

    void write(const void * src, size_t size) override {
        if (off > capacity || size > capacity - off) { throw std::runtime_error("checkpoint layout exceeds size pass"); }
        if (check) {
            same = same && (file_backed ? llama_compare_moe_output(blob + off, src, size) : memcmp(blob + off, src, size)) == 0;
        } else if (file_backed) {
            llama_write_moe_output(blob + off, src, size);
        } else {
            memcpy(blob + off, src, size);
        }
        off += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (off > capacity || size > capacity - off) { throw std::runtime_error("checkpoint tensor exceeds size pass"); }
        if (check) {
            same = same && iw < writes.size() && writes[iw].t == tensor && writes[iw].offset == offset &&
                writes[iw].size == size && writes[iw].blob_off == off;
            ++iw;
        } else {
            writes.push_back({ tensor, offset, size, off, -1 });
        }
        off += size;
    }

    size_t n_bytes() override { return off; }

    // check mode: the layout matched the earlier one exactly
    bool matched() const { return same && iw == writes.size() && off == capacity; }

  private:
    uint8_t * const                         blob;
    const size_t                            capacity;
    std::vector<llama_context::ckpt_write> & writes;
    const bool check;
    const bool file_backed;

    size_t off  = 0;
    size_t iw   = 0;
    bool   same = true;
};

const llama_memory_recurrent * llama_context::capture_mem() const {
    // PARTIAL_ONLY is exactly the recurrent state for these memories (the hybrid ones skip their attention cache and
    // indexer); a replay ring travels in an extra section the copy does not keep
    const llama_memory_recurrent * mem_recr = nullptr;
    if (const auto * hyb = dynamic_cast<const llama_memory_hybrid *>(memory.get())) {
        mem_recr = hyb->get_mem_recr();
    } else {
        mem_recr = dynamic_cast<const llama_memory_recurrent *>(memory.get());
    }
    if (mem_recr == nullptr || mem_recr->gdn_replay) {
        return nullptr;
    }
    return mem_recr;
}

int32_t llama_context::capture_arm(llama_seq_id seq_id, const llama_pos * n_tokens, int32_t n) {
    ckpt_caps.clear();
    ckpt_sel   = -1;
    ckpt_armed = false;
    ckpt_seq   = -1;
    if (n <= 0) {
        return 0;
    }
    if (capture_mem() == nullptr || seq_id < 0) {
        return -1;
    }
    ckpt_seq = seq_id;
    ckpt_caps.resize(n);
    for (int32_t i = 0; i < n; ++i) {
        ckpt_caps[i].n_tokens = n_tokens[i];
    }
    ckpt_armed = true;
    return 0;
}

int32_t llama_context::capture_min_tail() const {
    const auto * mem_recr = capture_mem();
    if (mem_recr == nullptr) {
        return -1;
    }
    // split_equal keeps the last n_rs_seq + 1 tokens of a sequence in one ubatch
    return mem_recr->n_rs_seq > 0 ? (int32_t) mem_recr->n_rs_seq + 1 : 1;
}

bool llama_context::capture_select(int32_t k, llama_pos * pos_min, llama_pos * pos_max) {
    ckpt_sel = -1;
    if (k < 0 || k >= (int32_t) ckpt_caps.size()) {
        return false;
    }
    const auto & c = ckpt_caps[k];
    if (c.failed || !c.laid_out) {
        return false;
    }
    for (size_t il = 0; il < c.layer_done.size(); ++il) {
        if (!c.layer_done[il]) {
            return false;
        }
    }
    ckpt_sel = k;
    if (pos_min) {
        *pos_min = c.pos_min;
    }
    if (pos_max) {
        *pos_max = c.pos_max;
    }
    return true;
}

void llama_context::capture_ubatch(const llama_ubatch & ubatch, int32_t il0, int32_t il1) {
    if (ckpt_caps.empty() || ubatch.n_tokens == 0) {
        return;
    }

    // the token count this ubatch ends at; a ubatch of another sequence fails the request
    llama_pos p_end = -1;
    bool      other = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != ckpt_seq) {
            other = true;
        } else {
            p_end = std::max<llama_pos>(p_end, ubatch.pos[i] + 1);
        }
    }

    const auto * mem_recr = capture_mem();
    const int32_t n_layer = (int32_t) mem_recr->r_l.size();
    if (il1 < 0 || il1 > n_layer) {
        il1 = n_layer;
    }

    for (auto & c : ckpt_caps) {
        if (c.failed) {
            continue;
        }
        if (other) {
            LLAMA_LOG_WARN("%s: in-call checkpoint at %d dropped: a ubatch holds another sequence\n", __func__, c.n_tokens);
            c.failed = true;
            continue;
        }
        if (c.n_tokens != p_end) {
            continue;
        }

        // the rows are written by the graph that just ran
        ggml_backend_sched_synchronize(sched.get());

        // the memory's bookkeeping is now that of the ubatch just computed, so it lays out the state the way a call
        // ending here would write it; every later layer range must find the same layout
        {
            try {
                if (!c.laid_out) {
                    // Size first: vector growth otherwise doubles large recurrent
                    // states and touches rows that later layer ranges will fill.
                    const size_t bytes = state_seq_get_size(ckpt_seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    const char * directory = std::getenv("LLAMA_MOE_ARENA_OUTPUT_DIR");
                    if (model.moe_arena_size() && directory && bytes >= (64ull << 20)) {
                        c.blob_file = std::make_unique<llama_moe_output>(bytes, directory);
                        LLAMA_LOG_INFO("%s: MoE checkpoint at %d: %.2f MiB file-backed, original state bytes\n",
                                __func__, c.n_tokens, bytes/1048576.0);
                    } else {
                        c.blob.resize(bytes);
                    }
                }
                llama_io_write_layout io(c.blob_data(), c.blob_size(), c.writes, c.laid_out, c.blob_file != nullptr);
                io.write(&io_magic, sizeof(io_magic));
                io.write(&ckpt_seq, sizeof(ckpt_seq));
                state_seq_write_data(io, ckpt_seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (io.n_bytes() != c.blob_size()) {
                    throw std::runtime_error("checkpoint layout disagrees with size pass");
                }
                if (c.laid_out && !io.matched()) {
                    throw std::runtime_error("the state layout changed between layer ranges");
                }
            } catch (const std::exception & err) {
                LLAMA_LOG_WARN("%s: in-call checkpoint at %d dropped: %s\n", __func__, c.n_tokens, err.what());
                c.failed = true;
                continue;
            }
        }
        if (!c.laid_out) {
            c.laid_out = true;
            c.pos_min  = memory->seq_pos_min(ckpt_seq);
            c.pos_max  = memory->seq_pos_max(ckpt_seq);
            c.layer_done.assign(n_layer, 1);
            for (int32_t il = 0; il < n_layer; ++il) {
                if (mem_recr->r_l[il] || mem_recr->s_l[il] || mem_recr->p_l[il]) {
                    c.layer_done[il] = 0;
                }
            }
            for (auto & w : c.writes) {
                for (int32_t il = 0; il < n_layer && w.il < 0; ++il) {
                    if (w.t == mem_recr->r_l[il] || w.t == mem_recr->s_l[il] || w.t == mem_recr->p_l[il]) {
                        w.il = il;
                    }
                }
                if (w.il < 0) {
                    LLAMA_LOG_WARN("%s: in-call checkpoint at %d dropped: a state row is not a layer's\n", __func__, c.n_tokens);
                    c.failed = true;
                    break;
                }
            }
            if (c.failed) {
                continue;
            }
        }

        // copy the rows of the layers this graph computed
        for (const auto & w : c.writes) {
            if (w.il >= il0 && w.il < il1) {
                ggml_backend_tensor_get(w.t, c.blob_data() + w.blob_off, w.offset, w.size);
                if (c.blob_file) { c.blob_file->dirty = true; }
            }
        }
        for (int32_t il = il0; il < il1; ++il) {
            c.layer_done[il] = 1;
        }
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id          seq_id,
                                         const uint8_t *       src,
                                         size_t                size,
                                         llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char *  filepath,
                                    llama_token * tokens_out,
                                    size_t        n_token_capacity,
                                    size_t *      n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count,
                            n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io(&file);
        const size_t       n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__,
                            n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id  seq_id,
                                          const char *  filepath,
                                          llama_token * tokens_out,
                                          size_t        n_token_capacity,
                                          size_t *      n_token_count_out,
                                   llama_state_seq_flags flags) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic,
                            version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__,
                            n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    try {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, flags);
        if (nread == 0 || nread != state_size) {
            throw std::runtime_error("invalid sequence state length");
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to restore sequence state: %s\n", __func__, err.what());
        if (memory) {
            memory->seq_rm(seq_id, -1, -1);
        }
        return 0;
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id        seq_id,
                                          const char *        filepath,
                                          const llama_token * tokens,
                                          size_t              n_token_count,
                                          llama_state_seq_flags flags) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, flags);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    mtp_feedback.valid = false;
    std::fill(mtp_hidden.carry_pos.begin(), mtp_hidden.carry_pos.end(), -2);
    mtp_hidden.captured.clear();
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(
                format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mtp_feedback.valid = false;
    if (seq_id >= 0 && (size_t) seq_id < mtp_hidden.carry_pos.size()) {
        mtp_hidden.carry_pos[seq_id] = -2;
    }
    mtp_hidden.captured.clear();
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us = ggml_time_us();
    t_eval_us = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused               = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (mtp_hidden.buf) {
        ret[ggml_backend_buffer_get_type(mtp_hidden.buf.get())].context += ggml_backend_buffer_get_size(mtp_hidden.buf.get());
    }
    if (mtp_feedback.buf) {
        ret[ggml_backend_buffer_get_type(mtp_feedback.buf.get())].context += ggml_backend_buffer_get_size(mtp_feedback.buf.get());
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return;  // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return;  // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(), model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch == 0);
    GGML_ASSERT(n_batch % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx                    = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter    = lopt_params.param_filter;
    void *                 param_filter_ud = lopt_params.param_filter_ud;

    //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd, param_filter, param_filter_ud);
    llama_set_param(model->pos_embd, param_filter, param_filter_ud);
    llama_set_param(model->tok_norm, param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b, param_filter, param_filter_ud);
    llama_set_param(model->output_norm, param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b, param_filter, param_filter_ud);
    llama_set_param(model->output, param_filter, param_filter_ud);
    llama_set_param(model->output_b, param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls, param_filter, param_filter_ud);
    llama_set_param(model->cls_b, param_filter, param_filter_ud);
    llama_set_param(model->cls_out, param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b, param_filter, param_filter_ud);
    llama_set_param(model->cls_norm, param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer) / sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(ggml_opt_dataset_t               dataset,
                                   ggml_opt_result_t                result,
                                   const std::vector<llama_token> & tokens,
                                   const std::vector<llama_token> & labels_sparse,
                                   llama_batch &                    batch,
                                   ggml_opt_epoch_callback          callback,
                                   bool                             train,
                                   int64_t                          idata_in_loop,
                                   int64_t                          ndata_in_loop,
                                   int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(), n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token[pos_batch]     = tokens[pos_ctx + pos_batch];
            batch.pos[pos_batch]       = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]  = 1;
            batch.seq_id[pos_batch][0] = 0;
            batch.logits[pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all, n_tokens_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            if (gf && (!model.hadamard_rotations.empty() || !model.hadamard_inverses.empty())) {
                llama_verify_hadamard_graph(gf, model.hadamard_rotations, model.hadamard_inverses);
            }

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta =
                    4 * size_gf * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/size_meta,
                    /*.mem_buffer =*/nullptr,
                    /*.no_alloc   =*/true,
                };
                ctx_compute_opt = ggml_init(params);
            }

            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef,
                                            (pos_ubatch * labels->ne[0] + labels_sparse[ilabel]) * sizeof(float),
                                            sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch) / n_ubatch + 1,
                         ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(ggml_opt_dataset_t      dataset,
                              ggml_opt_result_t       result_train,
                              ggml_opt_result_t       result_eval,
                              int64_t                 idata_split,
                              ggml_opt_epoch_callback callback_train,
                              ggml_opt_epoch_callback callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch, n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const int64_t  ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch       batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token> tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start  = ggml_time_us();
    int64_t ndata_in_loop = idata_split * ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train         = true;
        const int64_t  idata_in_loop = idata * ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx * sizeof(llama_token), labels_sparse.data(),
                                        idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch, callback_train, train, idata_in_loop,
                       ndata_in_loop, t_loop_start);
    }

    t_loop_start  = ggml_time_us();
    ndata_in_loop = (ndata - idata_split) * ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train         = false;
        const int64_t  idata_in_loop = (idata - idata_split) * ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx * sizeof(llama_token), labels_sparse.data(),
                                        idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch, callback_eval, train, idata_in_loop,
                       ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.gdn_replay                  =*/ false,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.type_s                      =*/ GGML_TYPE_F32,
        /*.moe_cache_mode              =*/ LLAMA_MOE_CACHE_MODE_UNSPECIFIED,
        /*.moe_prefetch                =*/ 0,
        /*.moe_cache_budget_mib        =*/ 0,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.sj_kvarn_bits_k                =*/ 0,
        /*.sj_kvarn_bits_v                =*/ 0,
        /*.sj_kvarn_tail                  =*/ 4096,
        /*.sj_kvarn_sink                  =*/ 128,
        /*.sj_kvarn_body_type             =*/ GGML_TYPE_F32,
        /*.sj_kvarn_sink_type             =*/ GGML_TYPE_F16,
        /*.sj_kvarn_staging_type          =*/ GGML_TYPE_TQ6_0,
        /*.sj_kvarn_tail_max              =*/ 8192,
        /*.sj_kvarn_edge_layers           =*/ 0,
        /*.sj_kvarn_edge_bits_k           =*/ 4,
        /*.sj_kvarn_edge_bits_v           =*/ 4,
        /*.sj_kvarn_edge_body_type        =*/ GGML_TYPE_COUNT,
        /*.sj_kvarn_flush_chunk           =*/ 0,
        /*.kv_resident                    =*/ 0,
        /*.sj_kvarn_transform             =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
        /*.draft_vocab_map             =*/ nullptr,
        /*.draft_vocab_hot             =*/ 0,
        /*.kv_stream_arena_mib         =*/ 0,
        /*.draft_attn_window           =*/ 0,
        /*.draft_attn_sink             =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if (params.type_s != GGML_TYPE_F32 && params.type_s != GGML_TYPE_BF16 && params.type_s != GGML_TYPE_F16 &&
        params.type_s != GGML_TYPE_Q8_0) {
        LLAMA_LOG_ERROR("%s: recurrent state cache type %s is not supported (f32, bf16, f16, q8_0)\n", __func__, ggml_type_name(params.type_s));
        return nullptr;
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__,
                        ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (params.type_k == GGML_TYPE_TURBO2_0) {
        LLAMA_LOG_ERROR("%s: turbo2 is a V-only cache type; pick a different K cache type (q8_0, turbo4, turbo5, turbo6)\n", __func__);
        return nullptr;
    }

    // the adaptive tail is a SJ-KVaRN-only setting; the 8192 default must not block non-SJ-KVaRN caches
    if (params.sj_kvarn_tail_max > 0 && params.sj_kvarn_bits_k == 0 && params.sj_kvarn_bits_v == 0) {
        params.sj_kvarn_tail_max = 0;
    }

    // SJ-KVaRN region-aware cache: staged sink + tail, sealed low-bit body
    // tiered body: the edge tier (first/last sj_kvarn_edge_layers cache layers) resolves its codec by the same auto rule
    // (trellis only for a trellis-capable pair and only when the body codec is auto or trellis)
    if (params.sj_kvarn_edge_body_type == GGML_TYPE_COUNT) {
        const int pair = (params.sj_kvarn_edge_bits_k << 8) | params.sj_kvarn_edge_bits_v;
        const bool trellis_pair = pair == ((3 << 8) | 3) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
        const bool trellis_ok = params.sj_kvarn_body_type == GGML_TYPE_COUNT || params.sj_kvarn_body_type == GGML_TYPE_I16;
        params.sj_kvarn_edge_body_type = (params.sj_kvarn_edge_layers > 0 && trellis_pair && trellis_ok) ? GGML_TYPE_I16 : GGML_TYPE_F32;
    }
    if (params.sj_kvarn_body_type == GGML_TYPE_COUNT) {
        // auto body codec: the trellis body (built-in trained codebooks) for the 3/3, 3/2 and 2/2 pairs, scalar records otherwise
        const int pair = (params.sj_kvarn_bits_k << 8) | params.sj_kvarn_bits_v;
        const bool trellis_pair = pair == ((3 << 8) | 3) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
        params.sj_kvarn_body_type = trellis_pair ? GGML_TYPE_I16 : GGML_TYPE_F32;
    }
    if (params.sj_kvarn_bits_k > 0 || params.sj_kvarn_bits_v > 0) {
        {
            const int pair = (params.sj_kvarn_bits_k << 8) | params.sj_kvarn_bits_v;
            const bool lower_pair = pair == ((4 << 8) | 3) || pair == ((3 << 8) | 3) || pair == ((4 << 8) | 2) || pair == ((2 << 8) | 4) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
            const bool trellis    = params.sj_kvarn_body_type == GGML_TYPE_I16;
            // scalar bodies: 4/4 plus the lower-bit pairs; trellis bodies (sjkvarn4t / auto): 4/4, 3/3, 3/2 and 2/2 (a 4-bit side of a
            // low-bit pair would be read as scalar by the low-bit tile loader)
            const bool trellis_pair = pair == ((3 << 8) | 3) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
            const bool ok = pair == ((4 << 8) | 4) || (lower_pair && (!trellis || trellis_pair));
            if (!ok) {
                LLAMA_LOG_ERROR("%s: SJ-KVaRN cache: supported K/V pairs are 4/4, 4/3, 3/3, 4/2, 2/4, 3/2 and 2/2 (trellis body: 4/4, 3/3, 3/2 and 2/2), got %d/%d\n",
                        __func__, params.sj_kvarn_bits_k, params.sj_kvarn_bits_v);
                return nullptr;
            }
        }
        if (params.sj_kvarn_edge_layers > 0) {
            const int pair = (params.sj_kvarn_edge_bits_k << 8) | params.sj_kvarn_edge_bits_v;
            const bool ok = pair == ((4 << 8) | 4) || pair == ((4 << 8) | 3) || pair == ((3 << 8) | 3) || pair == ((4 << 8) | 2) || pair == ((2 << 8) | 4) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
            // audit 2026-10-03: an explicit I16 (trellis) edge tier is valid only where the main tier allows a trellis body (4/4, 3/3,
            // 3/2, 2/2); for 4/3, 4/2, 2/4 the 4-bit side would be sealed as sjkvarn4t and read as scalar nibbles by the low-bit loader
            const bool edge_trellis_ok = pair == ((4 << 8) | 4) || pair == ((3 << 8) | 3) || pair == ((3 << 8) | 2) || pair == ((2 << 8) | 2);
            if (params.sj_kvarn_edge_body_type == GGML_TYPE_I16 && !edge_trellis_ok) {
                LLAMA_LOG_ERROR("%s: SJ-KVaRN tiered body: a trellis edge tier needs a 4/4, 3/3, 3/2 or 2/2 edge pair (got %u/%u)\n",
                        __func__, params.sj_kvarn_edge_bits_k, params.sj_kvarn_edge_bits_v);
                return nullptr;
            }
            if (!ok || params.sj_kvarn_body_type == GGML_TYPE_TURBO4_0 || (params.sj_kvarn_edge_body_type != GGML_TYPE_F32 && params.sj_kvarn_edge_body_type != GGML_TYPE_I16)) {
                LLAMA_LOG_ERROR("%s: SJ-KVaRN tiered body: edge pair must be one of 4/4, 4/3, 3/3, 4/2, 2/4, 3/2, 2/2 (got %u/%u) and the body codec cannot be turbo4\n",
                        __func__, params.sj_kvarn_edge_bits_k, params.sj_kvarn_edge_bits_v);
                return nullptr;
            }
        }
        if (model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN cache does not support MLA models\n", __func__);
            return nullptr;
        }
        if (params.sj_kvarn_staging_type != GGML_TYPE_F16 && params.sj_kvarn_staging_type != GGML_TYPE_Q8_0 && params.sj_kvarn_staging_type != GGML_TYPE_TQ6_0) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN staging type must be f16, q8_0 or tq6_0\n", __func__);
            return nullptr;
        }
        if (params.sj_kvarn_body_type != GGML_TYPE_F32 && params.sj_kvarn_body_type != GGML_TYPE_TURBO4_0 && params.sj_kvarn_body_type != GGML_TYPE_I16) {
            LLAMA_LOG_ERROR("invalid sealed body codec\n");
            return nullptr;
        }
        if (params.sj_kvarn_sink_type != GGML_TYPE_COUNT && params.sj_kvarn_sink_type != GGML_TYPE_F16) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN sink type must inherit staging or be f16\n", __func__);
            return nullptr;
        }
        if (params.sj_kvarn_sink_type == GGML_TYPE_F16 && params.sj_kvarn_staging_type == GGML_TYPE_Q8_0) {
            LLAMA_LOG_ERROR("%s: independent F16 SJ-KVaRN sink requires TQ6 staging\n", __func__);
            return nullptr;
        }
        if (params.sj_kvarn_staging_type == GGML_TYPE_F16) {
            params.sj_kvarn_sink_type = GGML_TYPE_COUNT;
        }
        params.type_k = params.sj_kvarn_staging_type;
        params.type_v = params.sj_kvarn_staging_type;
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN cache requires flash attention\n", __func__);
            return nullptr;
        }
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (params.sj_kvarn_tail_max != 0 && (params.sj_kvarn_tail_max < params.sj_kvarn_tail || params.sj_kvarn_tail_max % 128 != 0)) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN tail-max must be aligned to 128 and >= tail\n", __func__);
            return nullptr;
        }
        if (params.sj_kvarn_sink == 0 || params.sj_kvarn_sink % 64 != 0 || params.sj_kvarn_tail % 128 != 0) {
            LLAMA_LOG_ERROR("%s: SJ-KVaRN cache: sink must be a positive multiple of 64 and tail a multiple of 128 (got %u / %u)\n",
                    __func__, params.sj_kvarn_sink, params.sj_kvarn_tail);
            return nullptr;
        }
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (!model->hparams.has_kv(il) || model->hparams.is_recr(il)) {
                continue;
            }
            if (model->hparams.n_embd_head_k(il) != 256 || model->hparams.n_embd_head_v(il) != 256) {
                LLAMA_LOG_ERROR("%s: SJ-KVaRN cache requires 256-channel K/V heads (layer %u has %u/%u)\n", __func__,
                        il, model->hparams.n_embd_head_k(il), model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    // TurboQuant cache types require flash attention — auto-enable if disabled
    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED &&
        (params.type_k == GGML_TYPE_TURBO2_0 || params.type_k == GGML_TYPE_TURBO3_0 || params.type_k == GGML_TYPE_TURBO4_0 || params.type_k == GGML_TYPE_TQ6_0 || params.type_k == GGML_TYPE_TQ5_0 ||
         params.type_v == GGML_TYPE_TURBO2_0 || params.type_v == GGML_TYPE_TURBO3_0 || params.type_v == GGML_TYPE_TURBO4_0 || params.type_v == GGML_TYPE_TQ6_0 || params.type_v == GGML_TYPE_TQ5_0)) {
        LLAMA_LOG_WARN("%s: turbo cache types require flash_attn — enabling automatically\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        const bool k_is_turbo = (params.type_k == GGML_TYPE_TURBO2_0 ||
                                 params.type_k == GGML_TYPE_TURBO3_0 ||
                                 params.type_k == GGML_TYPE_TURBO4_0 ||
                                 params.type_k == GGML_TYPE_TQ6_0 || params.type_k == GGML_TYPE_TQ5_0);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            uint32_t head_k = model->hparams.n_embd_head_k(il);
            // Turbo types zero-pad heads to next multiple of 128 in llama-kv-cache.cpp
            if (k_is_turbo && head_k % 128 != 0) {
                head_k = ((head_k + 127) / 128) * 128;
            }
            if (head_k % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n", __func__,
                                ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        const bool v_is_turbo = (params.type_v == GGML_TYPE_TURBO2_0 ||
                                 params.type_v == GGML_TYPE_TURBO3_0 ||
                                 params.type_v == GGML_TYPE_TURBO4_0 ||
                                 params.type_v == GGML_TYPE_TQ6_0 || params.type_v == GGML_TYPE_TQ5_0);
        const bool is_mla = model->hparams.is_mla();
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            uint32_t head_v = model->hparams.n_embd_head_v(il);
            // Turbo types zero-pad; MLA has no separate V cache (V = view of K)
            if (v_is_turbo && !is_mla && head_v % 128 != 0) {
                head_v = ((head_v + 127) / 128) * 128;
            }
            if (head_v % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n", __func__,
                                ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(llama_model * model, llama_context_params params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(llama_context * ctx, ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

void llama_set_decode_phase(llama_context * ctx, enum llama_decode_phase phase) {
    ctx->set_decode_phase(phase);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

bool llama_model_supports_mtp_chain(const llama_model * model) {
    const char * enabled = std::getenv("LLAMA_MTP_QWEN4EXP_CHAIN");
    return model != nullptr && (model->arch == LLM_ARCH_QWEN35 ||
            (model->arch == LLM_ARCH_QWEN4EXP && enabled && std::strcmp(enabled, "1") == 0));
}

bool llama_model_uses_shared_position_draft(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_GEMMA4_ASSISTANT;
}

void llama_set_mtp_chain(llama_context * ctx, bool value) {
    ctx->set_mtp_chain(value);
}

void llama_set_mtp_chain_sampling(llama_context * ctx, int32_t top_k, float temp, float top_p, float min_p, const float * u, int32_t n_u) {
    ctx->set_mtp_chain_sampling(top_k, temp, top_p, min_p, u, n_u);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

enum ggml_type llama_get_kv_cache_type_k(const struct llama_context * ctx) {
    if (!ctx) {
        return GGML_TYPE_COUNT;
    }

    return ctx->get_kv_type_k();
}

enum ggml_type llama_get_kv_cache_type_v(const struct llama_context * ctx) {
    if (!ctx) {
        return GGML_TYPE_COUNT;
    }

    return ctx->get_kv_type_v();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

bool llama_release_embeddings_nextn(llama_context * ctx, int32_t first, int32_t count) {
    return ctx && ctx->release_embeddings_nextn(first, count);
}

bool llama_copy_embeddings_nextn(llama_context * ctx, float * dst, int32_t first, int32_t count) {
    if (!ctx) { return false; }
    ctx->synchronize();
    return ctx->copy_embeddings_nextn(dst, first, count);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

void llama_draft_vocab_observe(llama_context * ctx, const llama_token * toks, size_t n) {
    ctx->draft_vocab_observe(toks, n);
}

int32_t llama_draft_vocab_refresh(llama_context * ctx) {
    return ctx->draft_vocab_refresh();
}

int32_t llama_draft_vocab_n_hot(const llama_context * ctx) {
    return ctx->draft_vocab_n_hot();
}

size_t llama_draft_vocab_n_candidates(const llama_context * ctx) {
    return ctx->draft_vocab_n_candidates();
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

llama_sampler_output llama_get_sampler_output_ith(llama_context * ctx, int32_t idx) {
    ctx->synchronize();

    return ctx->get_sampler_output_ith(idx);
}

llama_bookkeep_context_perf llama_perf_bookkeep_context(const llama_context * ctx) {
    llama_bookkeep_context_perf result;
    auto * perf = ctx ? llama_bookkeep_context_bucket(ctx->get_cparams().ctx_type) : nullptr;
    if (perf) {
        result.sync_calls      = perf->sync_calls.load(std::memory_order_relaxed);
        result.sync_host_us    = perf->sync_host_us.load(std::memory_order_relaxed);
        result.copy_calls      = perf->copy_calls.load(std::memory_order_relaxed);
        result.copy_bytes      = perf->copy_bytes.load(std::memory_order_relaxed);
        result.copy_host_us    = perf->copy_host_us.load(std::memory_order_relaxed);
        result.release_calls   = perf->release_calls.load(std::memory_order_relaxed);
        result.release_host_us = perf->release_host_us.load(std::memory_order_relaxed);
        result.memory_init_layout_host_us = perf->memory_init_layout_host_us.load(std::memory_order_relaxed);
        result.graph_prepare_host_us      = perf->graph_prepare_host_us.load(std::memory_order_relaxed);
        result.set_inputs_host_us         = perf->set_inputs_host_us.load(std::memory_order_relaxed);
        result.peer_sync_host_us          = perf->peer_sync_host_us.load(std::memory_order_relaxed);
        result.graph_submit_host_us       = perf->graph_submit_host_us.load(std::memory_order_relaxed);
        result.graph_rebuild_calls        = perf->graph_rebuild_calls.load(std::memory_order_relaxed);
        result.graph_reuse_calls          = perf->graph_reuse_calls.load(std::memory_order_relaxed);
        for (size_t i = 0; i < perf->backend_submit.size(); ++i) {
            result.backend_submit[i] = perf->backend_submit[i].load(std::memory_order_relaxed);
        }
    }
    return result;
}

struct ggml_cgraph * llama_graph_reserve(struct llama_context * ctx,
                                         uint32_t               n_tokens,
                                         uint32_t               n_seqs,
                                         uint32_t               n_outputs) {
    auto                     memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(llama_context *       ctx,
                                llama_adapter_lora ** adapters,
                                size_t                n_adapters,
                                float *               scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(llama_context * ctx,
                               const float *   data,
                               size_t          len,
                               int32_t         n_embd,
                               int32_t         il_start,
                               int32_t         il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(llama_memory_t mem,
                         llama_seq_id   seq_id_src,
                         llama_seq_id   seq_id_dst,
                         llama_pos      p0,
                         llama_pos      p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx,
                             const char *    path_session,
                             llama_token *   tokens_out,
                             size_t          n_token_capacity,
                             size_t *        n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context *     ctx,
                             const char *        path_session,
                             const llama_token * tokens,
                             size_t              n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx,
                           const char *    path_session,
                           llama_token *   tokens_out,
                           size_t          n_token_capacity,
                           size_t *        n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context *     ctx,
                           const char *        path_session,
                           const llama_token * tokens,
                           size_t              n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context *       ctx,
                                    uint8_t *             dst,
                                    size_t                size,
                                    llama_seq_id          seq_id,
                                    llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

int32_t llama_state_seq_capture_arm(llama_context * ctx, llama_seq_id seq_id, const llama_pos * n_tokens, int32_t n) {
    return ctx->capture_arm(seq_id, n_tokens, n);
}

int32_t llama_state_seq_capture_min_tail(llama_context * ctx) {
    return ctx->capture_min_tail();
}

bool llama_state_seq_capture_select(llama_context * ctx, int32_t k, llama_pos * pos_min, llama_pos * pos_max) {
    return ctx->capture_select(k, pos_min, pos_max);
}

size_t llama_state_seq_set_data_ext(llama_context *       ctx,
                                    const uint8_t *       src,
                                    size_t                size,
                                    llama_seq_id          seq_id,
                                    llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context *     ctx,
                                 const char *        filepath,
                                 llama_seq_id        seq_id,
                                 const llama_token * tokens,
                                 size_t              n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx,
                                 const char *    filepath,
                                 llama_seq_id    dest_seq_id,
                                 llama_token *   tokens_out,
                                 size_t          n_token_capacity,
                                 size_t *        n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_save_file_ext(llama_context *       ctx,
                                     const char *          filepath,
                                     llama_seq_id          seq_id,
                                     const llama_token *   tokens,
                                     size_t                n_token_count,
                                     llama_state_seq_flags flags) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file_ext(llama_context *       ctx,
                                     const char *          filepath,
                                     llama_seq_id          dest_seq_id,
                                     llama_token *         tokens_out,
                                     size_t                n_token_capacity,
                                     size_t *              n_token_count_out,
                                     llama_state_seq_flags flags) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    static const bool reuse_enabled = [] {
        const char * value = std::getenv("LLAMA_MTP_COMPAT_REUSE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (reuse_enabled && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
            !cparams.training && !cparams.embeddings && !cparams.mtp_chain && !ple_pf &&
            llama_batch_compat::can_reuse_mtp(batch_inp, cparams.n_batch, model.hparams.n_embd_out())) {
        if (!mtp_compat_reuse) {
            mtp_compat_reuse = std::make_unique<llama_batch_ext>(this);
            LLAMA_LOG_INFO("%s: bounded MTP catch-up compatibility reuse enabled\n", __func__);
        }
        // The compatibility vectors are read only on the host by decode/allocr.
        // balloc owns a deep copy before any graph input upload, and graph/memory
        // snapshots own their separate llama_ubatch::data. No GPU fence is needed
        // to clear this independent translation buffer after the prior call.
        llama_batch_compat::reset(*mtp_compat_reuse, batch_inp);
        return decode(*mtp_compat_reuse);
    }
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval,
                   1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval,
                   1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms),
                   (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(struct llama_context *  ctx,
                     ggml_opt_dataset_t      dataset,
                     ggml_opt_result_t       result_train,
                     ggml_opt_result_t       result_eval,
                     int64_t                 idata_split,
                     ggml_opt_epoch_callback callback_train,
                     ggml_opt_epoch_callback callback_eval) {
    ctx->opt_epoch(dataset, result_train, result_eval, idata_split, callback_train, callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
