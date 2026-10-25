#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "speculative-adaptive.h"
#include "speculative-cdepth.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "../src/llama-mtp-chain-sample.h" // [#69] sampled chain: host re-derivation

#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <random>
#include <mutex>
#include <thread>
#include <cinttypes>

// [RTPORT] LLAMA_PLE_HINT=1: hand the target each token the next check will hold (with its predecessors), so the
// pages of its per-layer n-gram rows are read while the helper keeps drawing (see llama_ple_hint)
static void spec_ple_hint(llama_context * ctx_tgt, const common_speculative_draft_params & dp, const llama_tokens * drafted) {
    static const bool on = [] {
        const char * v = getenv("LLAMA_PLE_HINT");
        return v && atoi(v) != 0;
    }();
    if (!on || !ctx_tgt) {
        return;
    }
    constexpr int n_win = 4; // the token and three predecessors (qwen4exp uses ngram_size - 1 = 2)
    llama_token w[n_win];
    int n = 0;
    // newest first, then reversed
    if (drafted) {
        for (size_t k = drafted->size(); k-- > 0 && n < n_win;) {
            w[n++] = (*drafted)[k];
        }
    }
    if (n < n_win) {
        w[n++] = dp.id_last;
    }
    if (dp.prompt) {
        for (size_t k = dp.prompt->size(); k-- > 0 && n < n_win;) {
            w[n++] = (*dp.prompt)[k];
        }
    }
    std::reverse(w, w + n);
    llama_ple_hint(ctx_tgt, w, n);
}

// [#47] host phase split of the MTP draft call (LLAMA_HOST_PHASES=1), see speculative.h
bool common_speculative_host_phases_enabled() {
    static const bool v = [] { const char * e = getenv("LLAMA_HOST_PHASES"); return e && atoi(e) != 0; }();
    return v;
}

static std::mutex                     g_spec_host_phases_mtx;
static common_speculative_host_phases g_spec_host_phases;

common_speculative_host_phases common_speculative_host_phases_take() {
    std::lock_guard<std::mutex> lock(g_spec_host_phases_mtx);
    common_speculative_host_phases r = g_spec_host_phases;
    g_spec_host_phases = {};
    return r;
}

// one draft call: accumulates locally, adds to the process totals on destruction (rest = wall - the named parts)
struct common_spec_host_phase_call {
    const bool on = common_speculative_host_phases_enabled();
    const int64_t t0 = on ? ggml_time_us() : 0;
    common_speculative_host_phases acc;

    ~common_spec_host_phase_call() {
        if (!on) {
            return;
        }
        const uint64_t wall  = (uint64_t) (ggml_time_us() - t0);
        const uint64_t named = acc.decode_us + acc.sync_us + acc.sample_us;
        std::lock_guard<std::mutex> lock(g_spec_host_phases_mtx);
        g_spec_host_phases.n_calls   += 1;
        g_spec_host_phases.n_steps   += acc.n_steps;
        g_spec_host_phases.decode_us += acc.decode_us;
        g_spec_host_phases.sync_us   += acc.sync_us;
        g_spec_host_phases.sample_us += acc.sample_us;
        g_spec_host_phases.rest_us   += wall > named ? wall - named : 0;
    }
};

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// fork: keep this predicate in sync with the chain graph in qwen35.cpp.
static bool spec_conflog_enabled() {
    static const bool on = []() {
        const char * p = std::getenv("LLAMA_SPEC_CONFLOG");
        return p != nullptr && p[0] != '\0';
    }();
    return on;
}

static void spec_conf_clear(common_speculative_draft_params & dp) {
    if (dp.conf_p1) {
        dp.conf_p1->clear();
    }
    if (dp.conf_p2) {
        dp.conf_p2->clear();
    }
}

static void spec_conf_push(common_speculative_draft_params & dp, float p1, float p2) {
    if (!dp.conf_p1) {
        return;
    }
    dp.conf_p1->push_back(p1);
    if (dp.conf_p2) {
        dp.conf_p2->push_back(p2);
    }
}

// Used only when the sampler has logits but no stored probabilities.
static void spec_conf_top2_from_row(const float * row, int32_t n, float & p1, float & p2) {
    p1 = NAN;
    p2 = NAN;
    if (row == nullptr || n <= 0) {
        return;
    }
    float m1 = -INFINITY;
    float m2 = -INFINITY;
    for (int32_t i = 0; i < n; ++i) {
        const float z = row[i];
        if (z > m1) {
            m2 = m1;
            m1 = z;
        } else if (z > m2) {
            m2 = z;
        }
    }
    if (!std::isfinite(m1)) {
        return;
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += std::exp((double) row[i] - (double) m1);
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return;
    }
    p1 = (float) (1.0 / sum);
    p2 = 0.0f;
    if (std::isfinite(m2)) {
        p2 = (float) (std::exp((double) m2 - (double) m1) / sum);
    }
}

static void spec_conf_top2_logits(llama_context * ctx, int32_t idx, float & p1, float & p2) {
    p1 = NAN;
    p2 = NAN;
    if (ctx == nullptr || idx < 0) {
        return;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    // Sampled logits can be a short row; compact greedy can omit the raw row entirely.
    const float * sampled = llama_get_sampled_logits_ith(ctx, idx);
    if ((sampled && llama_get_sampled_logits_count_ith(ctx, idx) != (uint32_t) n_vocab) ||
            (!sampled && llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL)) {
        return;
    }
    spec_conf_top2_from_row(llama_get_logits_ith(ctx, idx), n_vocab, p1, p2);
}

// Reuse sampling probabilities. Only the logits-only path needs a softmax.
static void spec_conf_push_candidates(common_speculative_draft_params & dp, const llama_token_data_array * cur,
        llama_context * ctx, int32_t idx) {
    if (!dp.conf_p1) {
        return;
    }
    float p1 = 0.0f;
    float p2 = 0.0f;
    bool any = false;
    if (cur != nullptr && cur->data != nullptr && cur->size > 0) {
        for (size_t i = 0; i < cur->size; ++i) {
            const float p = cur->data[i].p;
            if (!(p > 0.0f)) {
                continue;
            }
            any = true;
            if (p > p1) {
                p2 = p1;
                p1 = p;
            } else if (p > p2) {
                p2 = p;
            }
        }
    }
    if (!any) {
        spec_conf_top2_logits(ctx, idx, p1, p2);
    }
    spec_conf_push(dp, p1, p2);
}

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-mtp-adaptive", COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

// llama_get_kv_cache_type_k/v return GGML_TYPE_COUNT for memory without a single K/V type
static const char * common_speculative_kv_type_name(ggml_type type) {
    return type < GGML_TYPE_COUNT ? ggml_type_name(type) : "n/a";
}

static bool common_speculative_mtp_chain_enabled(const common_params_speculative_draft & params) {
    const char * env = getenv("LLAMA_SPEC_CHAIN");
    return params.chain || (env != nullptr && std::strcmp(env, "0") != 0);
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.
    int32_t n_accepted_ctrl = -1; // [MBFIX] cap on the accepted count the adaptive controller sees (-1 = none), set per accept call

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;
    virtual bool process_range(const llama_batch & batch, int32_t output_offset) { return output_offset == 0 && process(batch); }

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) target top-1 probabilities of the tokens one verify round committed (CDEPTH history)
    virtual void note_target_conf(llama_seq_id /*seq_id*/, const float * /*p*/, size_t /*n*/) {}

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // Slot resume is separate from partial checkpoints: adding carryover must not
    // change whether the server can capture an in-call prompt checkpoint.
    virtual bool get_resume_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const { return get_state(seq_id, data); }
    virtual bool set_resume_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) { set_state(seq_id, data); return !data.empty(); }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                common_speculative_kv_type_name(llama_get_kv_cache_type_k(ctx_dft)),
                common_speculative_kv_type_name(llama_get_kv_cache_type_v(ctx_dft)),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers.assign(1, COMMON_SAMPLER_TYPE_TOP_K);

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);
                if (dp.conf_p1) {
                    spec_conf_push_candidates(dp, cur_p, ctx_dft, i_batch - 1);
                }

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
                if (dp.conf_p1) {
                    spec_conf_clear(dp);
                }
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);
                if (dp.conf_p1) {
                    spec_conf_push_candidates(dp, cur_p, ctx_dft, i_batch - 1);
                }

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
                if (dp.conf_p1) {
                    spec_conf_clear(dp);
                }
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;
    int32_t         n_layer_tgt        = 0;       // extract id == n_layer_tgt -> pre-final-norm state (nextn)

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // Enable extraction of target model layer outputs for DFlash encoder; an id
        // equal to the target's layer count means the pre-final-norm hidden state,
        // which is captured through the unmasked nextn path instead
        n_layer_tgt = llama_model_n_layer(model_tgt);
        // (upstream's backend-sampling offload for the DFlash drafter is not in this tree; CPU sampler path)

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            }
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);

        // generic DFlash drafts with non-causal block attention; Laguna drafters are
        // trained with a causal noise block and older GGUFs carry no dflash.attention.causal
        {
            bool causal = causal_attn;
            char buf[32] = {};
            // DFlash2 (lattice selector) drafters are always non-causal: the
            // decoder_arch metadata only applies to the plain DFlash decoder.
            if (!causal && !is_dflash2 && llama_model_meta_val_str(model_dft, "dflash.decoder_arch", buf, sizeof(buf)) >= 0) {
                causal = strcmp(buf, "laguna") == 0;
            }
            llama_set_causal_attn(ctx_dft, causal);
        }
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings (image chunks).
        // Image chunks are not mirrored into the draft: their target-layer features are
        // vision states the draft was never trained on, and their M-RoPE positions collapse
        // onto one temporal position, so injecting them poisons every draft after the
        // image (acceptance fell from ~0.5 to ~0.03 for the rest of the conversation).
        // Skipping them leaves a gap in the draft's cache between the text before and after
        // the image; the draft keeps the target's positions, and the text tokens after the
        // image carry the image's influence in their injected features. The gap is fine for
        // the default sliding-window draft cache: every batch it does see has consecutive
        // positions, which is all find_slot requires (the crash was the image batch itself,
        // whose tokens all carry one temporal position). No server-side change is involved.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (!has_tokens || has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const bool pos_pinned = batch_in.pos[i_batch_beg[seq_id]] == batch_in.pos[i_batch_end[seq_id]];
            if (has_embeddings && n_rows > 1 && pos_pinned) {
                continue;
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                batch_inject.n_tokens = n_chunk;
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = target_layer_ids[k] == n_layer_tgt
                        ? llama_get_embeddings_nextn(ctx_tgt)
                        : llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                const int32_t rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            // dp.pos0 is the slot's token count. For an M-RoPE target that is no longer the
            // next position once an image has been in the prompt (image tokens advance the
            // position by the grid size, not by their count), and a noise block placed at the
            // token count sits hundreds of positions past the draft's own cache. Take the next
            // position from the target's memory instead; without images the two are equal.
            const llama_pos pos_max_tgt = llama_memory_seq_pos_max(llama_get_memory(params.ctx_tgt), seq_id);
            const int32_t n = pos_max_tgt >= 0 ? (int32_t) pos_max_tgt + 1 : (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + selector_top_k));
                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                    if (dp.conf_p1) {
                        // selector scores are already on the host; softmax is over that short row only
                        float p1 = 0.0f;
                        float p2 = 0.0f;
                        spec_conf_top2_from_row(scores, selector_top_k, p1, p2);
                        spec_conf_push(dp, p1, p2);
                    }
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.conf_p1) {
                        spec_conf_clear(dp);
                    }
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                    if (dp.conf_p1) {
                        spec_conf_push_candidates(dp, cur_p, ctx_dft, idx);
                    }
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                    if (dp.conf_p1) {
                        spec_conf_push_candidates(dp, cur_p, ctx_dft, beg + i);
                    }
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
                if (dp.conf_p1) {
                    spec_conf_clear(dp);
                }
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // exact p/q drafting (LLAMA_SPEC_PQ=1, sequential path): per-seq sampler over the draft logits built
    // from the target's request params (temp / top-k / top-p / min-p, an independent seed); each draft
    // token is sampled from it and the candidate distribution is recorded for the verifier
    bool pq_enabled = false;
    bool pq_backend_warned = false;
    std::vector<common_sampler_ptr>          q_smpls;
    std::vector<common_params_sampling>      q_params; // the target params each q_smpls entry was built from

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;
    int32_t batch_capacity = 0;
    int32_t ubatch_capacity = 0;
    bool prefill_overlap = false;
    bool prefill_device = false;
    bool handoff = false; // [PREFILLFAST] LLAMA_MTP_B1_HANDOFF=1: per-ubatch device rows from the target decode
    bool device_feedback = false;
    bool mtp_pipe = false; // [RTPORT] LLAMA_MTP_PIPELINE=1: next guess step queued on the card before its token is drawn

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    same_position_draft = false; // gemma4-assistant only: every draft row in a round shares n_past
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    // catch-up rows deferred from process() and prepended to the first draft
    // decode, which merges the two evals. Only for the single-head own-memory
    // path; flushed standalone when drafting does not follow.
    static constexpr int32_t defer_max = 64;
    bool defer_enabled = false;
    bool chain_graph   = false;
    bool chain_qwen4exp = false;

    // [#69] sampled chain (LLAMA_SPEC_CHAIN_SAMPLED=1 with the chain graph and exact p/q): the chain draws each
    // step in-graph against uniforms drawn here, the host re-derives every step and cuts at the first
    // disagreement.
    bool chain_sampled = false;
    std::vector<std::mt19937_64> chain_rng;
    std::vector<std::mt19937>    chain_rng_qwen;
    std::vector<double>          chain_u_host;
    std::vector<uint8_t>         chain_rng_seeded;
    std::vector<float>           chain_u;
    struct {
        uint64_t rounds    = 0; // sampled chain decodes
        uint64_t steps     = 0; // draft tokens taken from them
        uint64_t cut       = 0; // rounds cut where the GPU draw disagreed with the host
        uint64_t bad       = 0; // rounds stopped on a malformed row
        uint64_t repaired  = 0;
        uint64_t repair_steps = 0;
        uint64_t repair_failed = 0;
    } chain_st;
    struct {
        std::vector<llama_token>  tok;
        std::vector<llama_pos>    pos;
        std::vector<llama_seq_id> seq;
        std::vector<float>        embd;
    } defer;

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<int32_t> verify_h_beg;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // Adaptive draft depth (draft-mtp-adaptive), see common_speculative_adaptive
    bool adaptive = false;
    std::vector<int> n_cap;   // [n_seq] effective draft cap for the current draft() call
    std::vector<int> n_last;  // [n_seq] drafts attempted in the most recent draft() call
    // [PARW] the seq's own draft cap in the most recent draft() call, before the multi-seq equal-depth lift
    // (n_cap can be raised to the round's deepest cap); the adaptive controller is fed at this depth
    std::vector<int> n_own;
    std::vector<common_speculative_adaptive> adaptive_ctrl; // [n_seq] per-seq adaptive depth controller

    // CDEPTH (--spec-depth-policy cdepth): per-token confidence stop rule between n_min_adaptive and n_max
    bool cdepth = false;
    common_speculative_cdepth_model cd_model;
    std::vector<common_speculative_cdepth_state> cd_state; // [n_seq]
    bool cd_chain_warned = false;

    void note_target_conf(llama_seq_id seq_id, const float * p, size_t n) override {
        if (cdepth && seq_id >= 0 && seq_id < (llama_seq_id) cd_state.size()) {
            cd_state[seq_id].note_target(p, n, cd_model.window);
        }
    }

    static bool equal_depth_enabled() {
        static const bool on = [] {
            const char * e = getenv("LLAMA_SPEC_EQUAL_DEPTH");
            return e == nullptr || e[0] != '0';
        }();
        return on;
    }

    int32_t defer_capacity() const {
        const int32_t draft_rows = std::max(this->params.n_max, (int32_t) n_seq);
        const int32_t decode_capacity = std::min(batch_capacity, ubatch_capacity);
        return std::min(defer_max, std::max(0, decode_capacity - draft_rows));
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq, bool adaptive = false)
        : common_speculative_impl(adaptive ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE : COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        this->adaptive = adaptive;
        // n_cap/n_last are written by the shared draft loop in both modes
        n_cap.assign(n_seq, 0);
        n_own.assign(n_seq, 0);
        n_last.assign(n_seq, 0);

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                common_speculative_kv_type_name(llama_get_kv_cache_type_k(ctx_dft)),
                common_speculative_kv_type_name(llama_get_kv_cache_type_v(ctx_dft)),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        ubatch_capacity = (int32_t) llama_n_ubatch(ctx_dft);
        batch_capacity = std::min<int32_t>(llama_n_batch(ctx_dft), std::max<int32_t>(ubatch_capacity, n_seq * (this->params.n_max + 1)));
        const char * overlap = std::getenv("GGML_MTP_PREFILL_OVERLAP");
        prefill_overlap = overlap && std::strcmp(overlap, "1") == 0;
        const char * scratch = getenv("LLAMA_MTP_PREFILL_SCRATCH");
        prefill_overlap = prefill_overlap || (scratch && std::strcmp(scratch, "1") == 0);
        if (prefill_overlap) {
            SPC_INF("%s", "GGML_MTP_PREFILL_OVERLAP=1: coalescing target hidden-state reads\n");
        }
        batch = llama_batch_init(/*n_tokens=*/ batch_capacity, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * batch_capacity);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        {
            // default on since PQ1 shipped (2026-09-10): LLAMA_SPEC_PQ=0 restores the id-match rule
            const char * env = getenv("LLAMA_SPEC_PQ");
            pq_enabled = env == nullptr || std::strcmp(env, "0") != 0;
            q_smpls.resize(n_seq);
            q_params.resize(n_seq);
            if (pq_enabled) {
                SPC_INF("%s", "exact p/q drafting enabled (default; LLAMA_SPEC_PQ=0 disables): draft tokens are sampled at the request's temperature and verified by rejection sampling\n");
            } else {
                SPC_INF("%s", "exact p/q drafting disabled (LLAMA_SPEC_PQ=0): drafts are verified by identity match\n");
            }
        }

        const char * qwen_chain = std::getenv("LLAMA_MTP_QWEN4EXP_CHAIN");
        char arch_name[32] = {};
        chain_qwen4exp = qwen_chain && std::strcmp(qwen_chain, "1") == 0 &&
                llama_model_meta_val_str(llama_get_model(ctx_dft), "general.architecture", arch_name, sizeof(arch_name)) >= 0 &&
                std::strcmp(arch_name, "qwen4exp") == 0;
        const bool chain_enabled = common_speculative_mtp_chain_enabled(this->params) || chain_qwen4exp;

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        same_position_draft = is_mem_shared && llama_model_uses_shared_position_draft(llama_get_model(ctx_dft));
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
        chain_graph   = !is_mem_shared && !chain_heads && chain_enabled &&
                (!chain_qwen4exp || (n_seq == 1 && this->params.n_max > 0 && this->params.n_max <= LLAMA_MTP_QWEN4EXP_CHAIN_MAX)) &&
                llama_model_supports_mtp_chain(llama_get_model(ctx_dft));

        // fork: Stage prompt hidden rows once, then feed the logical catch-up batch by backend copies.
        prefill_device = !is_mem_shared && !chain_heads && llama_mtp_prefill_init(ctx_tgt);
        handoff = !is_mem_shared && !chain_heads && !prefill_device && llama_mtp_handoff_init(ctx_tgt);
        prefill_overlap = prefill_overlap || prefill_device;
        if (prefill_device) {
            std::memset(batch.embd, 0, (size_t) batch_capacity * n_embd * sizeof(float));
        }

        if (chain_enabled && !is_mem_shared && !chain_heads && !chain_graph) {
            SPC_WRN("%s", "chained MTP is not supported for this model; using sequential MTP\n");
        }

        // offload draft sampling to the backend (chained drafting outputs several
        // rows per sequence, which backend sampling does not support)
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !chain_graph) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // draft all n_max tokens with one chained decode (in-graph argmax
        // feeds each next step); replaces n_max sequential draft decodes
        if (chain_graph) {
            // the chain decode absorbs the deferred catch-up rows, one eval per round
            defer_enabled = true;
            if (chain_qwen4exp) {
                SPC_INF("%s", "Qwen4exp bounded MTP chain graph enabled (LLAMA_MTP_QWEN4EXP_CHAIN)\n");
            }

            const char * env_s = getenv("LLAMA_SPEC_CHAIN_SAMPLED");
            if (env_s != nullptr && std::strcmp(env_s, "0") != 0) {
                if (pq_enabled) {
                    chain_sampled = true;
                    chain_rng.resize(n_seq);
                    if (chain_qwen4exp) {
                        chain_rng_qwen.resize(n_seq);
                    }
                    chain_rng_seeded.assign(n_seq, 0);
                    SPC_INF("sampled draft chain enabled (LLAMA_SPEC_CHAIN_SAMPLED): steps draw in-graph over top_k <= %d, "
                            "the host re-derives each step\n", LLAMA_MTP_CHAIN_TOP_K_MAX);
                } else {
                    SPC_WRN("%s", "LLAMA_SPEC_CHAIN_SAMPLED needs exact p/q drafting (LLAMA_SPEC_PQ=0 is set); the chain stays argmax\n");
                }
            }
        }

        // [DRAFTFAST] LLAMA_SPEC_DEFER_CATCHUP=1: the sequential drafter also absorbs the verified catch-up rows
        // into its first draft decode, the way the chain decode does: one helper decode (and one helper graph
        // build) fewer per round. The first draft batch changes shape, so the draft's own numbers can move by
        // rounding; the verifier still checks every draft token against the target, so the output distribution
        // is unchanged. Off by default.
        if (!chain_graph && !is_mem_shared && !chain_heads) {
            const char * env_d = getenv("LLAMA_SPEC_DEFER_CATCHUP");
            if (env_d != nullptr && atoi(env_d) != 0) {
                defer_enabled = true;
                SPC_INF("%s", "sequential MTP: catch-up rows ride along with the first draft decode (LLAMA_SPEC_DEFER_CATCHUP)\n");
            }
        }

        device_feedback = n_seq == 1 && !chain_heads && !chain_graph && !defer_enabled &&
            !same_position_draft && llama_mtp_device_feedback_init(ctx_dft);
        if (device_feedback) {
            SPC_INF("%s", "sequential MTP: helper hidden feedback stays on device (LLAMA_MTP_DEVICE_FEEDBACK)\n");
        }
        mtp_pipe = device_feedback && llama_mtp_pipe_init(ctx_dft);
        if (mtp_pipe) {
            SPC_INF("%s", "sequential MTP: next guess step queued before its token is drawn (LLAMA_MTP_PIPELINE)\n");
        }

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        if (adaptive) {
            if (this->params.n_min_adaptive < 1 || this->params.n_min_adaptive > this->params.n_max) {
                GGML_ABORT("%s: invalid adaptive draft range: n_min_adaptive=%d, n_max=%d (n_min_adaptive must be in [1, n_max])",
                        __func__, this->params.n_min_adaptive, this->params.n_max);
            }

            adaptive_ctrl.assign(n_seq, common_speculative_adaptive());
            for (uint32_t s = 0; s < n_seq; ++s) {
                // start at the floor max(1, n_min_adaptive), bounded by n_max;
                // the controller climbs from there once acceptance feedback arrives
                adaptive_ctrl[s].reset(this->params.n_max, this->params.n_min_adaptive);
            }
            SPC_TRC("%s", "adaptive draft depth enabled (draft-mtp-adaptive)\n");

            if (this->params.depth_policy == "cdepth") {
                std::string err;
                bool ok;
                if (this->params.depth_model.empty()) {
                    std::istringstream builtin(common_speculative_cdepth_builtin_text());
                    ok = cd_model.parse(builtin, err);
                } else {
                    ok = cd_model.load(this->params.depth_model, err);
                }
                if (!ok) {
                    GGML_ABORT("%s: --spec-depth-policy cdepth: bad model (%s): %s", __func__,
                            this->params.depth_model.empty() ? "built-in" : this->params.depth_model.c_str(), err.c_str());
                }
                cdepth = true;
                cd_state.assign(n_seq, common_speculative_cdepth_state());
                LOG_INF("%s: CDEPTH depth policy on: draft %d..%d tokens per round, window %d, stages %zu, theta x%.2f, model %s\n",
                        __func__, this->params.n_min_adaptive, this->params.n_max, cd_model.window, cd_model.coef.size(),
                        (double) this->params.depth_theta,
                        this->params.depth_model.empty() ? "built-in" : this->params.depth_model.c_str());
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h_beg.assign(n_seq, 0);
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        if (chain_sampled) {
            chain_log_stats();
        }

        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    // decode deferred catch-up rows standalone (no logits); used when no draft decode
    // follows to absorb them
    bool flush_deferred() {
        if (defer.tok.empty()) {
            return true;
        }

        // defer only ever grows on the !is_mem_shared path (deferral requires
        // chain_graph, which requires !is_mem_shared). Keep that invariant local:
        // the trim below on shared memory would delete target KV.
        GGML_ASSERT(!is_mem_shared);

        auto * ctx_dft = this->params.ctx_dft;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // The draft KV can retain cells from a position rewind (context shift,
        // rollback, or a new prompt) for a sequence that had no rows in the last
        // process() batch, so its stale cells were not trimmed there. Drop them
        // here, keyed on the flush batch's own per-seq positions, before decode,
        // or the position consistency check (X < Y) fails.
        auto * mem_dft = llama_get_memory(ctx_dft);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            llama_pos pos_min = -1;
            for (size_t k = 0; k < defer.pos.size(); ++k) {
                if (defer.seq[k] == seq_id) {
                    pos_min = pos_min < 0 ? defer.pos[k] : std::min(pos_min, defer.pos[k]);
                }
            }
            if (pos_min < 0) {
                continue;
            }
            // In the normal flow the flush rows start right past the draft KV max,
            // so an overlap means a stale-state producer this trim is papering over.
            // Log it: a silently firing trim leaves no forensic signal.
            const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_id);
            if (pos_max_pre >= pos_min) {
                SPC_WRN("stale draft KV for seq %d at deferred flush: pos_max %d >= flush start %d, trimming\n",
                        (int) seq_id, (int) pos_max_pre, (int) pos_min);
            }
            if (!llama_memory_seq_rm(mem_dft, seq_id, pos_min, -1)) {
                SPC_ERR("failed to trim draft memory for sequence %d at deferred flush\n", (int) seq_id);
                return false;
            }
        }

        // fork: Restoring several sequences can exceed one decode batch's deferred-row capacity.
        const size_t capacity = std::min(batch_capacity, ubatch_capacity);
        for (size_t k0 = 0; k0 < defer.tok.size(); k0 += capacity) {
            common_batch_clear(batch);
            const size_t end = std::min(k0 + capacity, defer.tok.size());
            for (size_t k = k0; k < end; ++k) {
                common_batch_add(batch, defer.tok[k], defer.pos[k], { defer.seq[k] }, false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, defer.embd.data() + k * (size_t) n_embd, row_bytes);
            }
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) deferred flush failed rc=%d\n", (int) rc);
                return false;
            }
        }
        defer.tok.clear();
        defer.pos.clear();
        defer.seq.clear();
        defer.embd.clear();
        return true;
    }

    // drop deferred rows of seq_id with pos >= pos_from; returns true if any dropped
    bool drop_deferred_from(llama_seq_id seq_id, llama_pos pos_from) {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        size_t w = 0;
        for (size_t k = 0; k < defer.tok.size(); ++k) {
            if (defer.seq[k] == seq_id && defer.pos[k] >= pos_from) {
                continue;
            }
            if (w != k) {
                defer.tok[w] = defer.tok[k];
                defer.pos[w] = defer.pos[k];
                defer.seq[w] = defer.seq[k];
                std::memmove(defer.embd.data() + w * (size_t) n_embd,
                             defer.embd.data() + k * (size_t) n_embd, row_bytes);
            }
            w++;
        }

        const bool dropped = w < defer.tok.size();
        defer.tok.resize(w);
        defer.pos.resize(w);
        defer.seq.resize(w);
        defer.embd.resize(w * (size_t) n_embd);

        return dropped;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        uint32_t n_defer = 0;
        for (const auto s : defer.seq) {
            n_defer += s == seq_id;
        }
        const common_speculative_adaptive ctrl = adaptive ? adaptive_ctrl[seq_id] : common_speculative_adaptive();
        const uint32_t header[] = {
            0x3150544d, (uint32_t) n_embd, n_defer,
            (uint32_t) ctrl.n_cur, (uint32_t) ctrl.n_climb, (uint32_t) ctrl.n_drop,
            (uint32_t) n_last[seq_id], (adaptive ? 1u : 0u) | (defer_enabled ? 2u : 0u),
        };
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        data.resize(sizeof(header) + row_bytes + n_defer * (sizeof(llama_token) + sizeof(llama_pos) + row_bytes));
        size_t off = 0;
        auto put = [&](const void * src, size_t n) {
            std::memcpy(data.data() + off, src, n);
            off += n;
        };
        put(header, sizeof(header));
        put(pending_h[seq_id].data(), row_bytes);
        // Keep deferred rows separate from KV so restoring preserves the next decode's batch shape.
        for (size_t k = 0; k < defer.seq.size(); ++k) {
            if (defer.seq[k] == seq_id) {
                put(&defer.tok[k], sizeof(llama_token));
                put(&defer.pos[k], sizeof(llama_pos));
                put(defer.embd.data() + k * (size_t) n_embd, row_bytes);
            }
        }
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        uint32_t header[8];
        if (data.size() < sizeof(header)) {
            return;
        }
        std::memcpy(header, data.data(), sizeof(header));
        const uint32_t flags = (adaptive ? 1u : 0u) | (defer_enabled ? 2u : 0u);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        const size_t retained = std::count_if(defer.seq.begin(), defer.seq.end(), [seq_id](llama_seq_id id) { return id != seq_id; });
        // Each restored sequence may carry a full deferred tail; flush_deferred splits the combined rows.
        if (retained + header[2] > (size_t) n_seq * defer_max) {
            return;
        }
        if (header[0] != 0x3150544d || header[1] != (uint32_t) n_embd || header[2] > (uint32_t) defer_max || header[7] != flags ||
                header[6] > (uint32_t) params.n_max ||
                data.size() != sizeof(header) + row_bytes + header[2] * (sizeof(llama_token) + sizeof(llama_pos) + row_bytes)) {
            return;
        }
        const uint32_t cap = std::max(1, params.n_max);
        const uint32_t floor = std::min<uint32_t>(std::max(1, params.n_min_adaptive), cap);
        if (adaptive && (header[3] < floor || header[3] > cap ||
                header[4] >= (uint32_t) common_speculative_adaptive::climb_threshold(header[3]) ||
                header[5] >= (uint32_t) common_speculative_adaptive::drop_pressure(header[3]))) {
            return;
        }
        if ((!adaptive && (header[3] || header[4] || header[5])) || (!defer_enabled && header[2])) {
            return;
        }

        // Validate the complete snapshot before changing this sequence or its deferred rows.
        size_t off = sizeof(header) + row_bytes;
        auto finite_row = [&](size_t start) {
            for (int32_t i = 0; i < n_embd; ++i) {
                float value;
                std::memcpy(&value, data.data() + start + (size_t) i*sizeof(value), sizeof(value));
                if (!std::isfinite(value)) { return false; }
            }
            return true;
        };
        if (!finite_row(sizeof(header))) { return; }
        llama_pos prev_pos = -1;
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(params.ctx_dft)));
        for (uint32_t i = 0; i < header[2]; ++i) {
            llama_token tok;
            llama_pos pos;
            std::memcpy(&tok, data.data() + off, sizeof(tok));
            std::memcpy(&pos, data.data() + off + sizeof(tok), sizeof(pos));
            if (tok < 0 || tok >= n_vocab || pos < 0 || (uint32_t) pos >= llama_n_ctx(params.ctx_dft) ||
                    (i > 0 && pos != prev_pos + 1) || !finite_row(off + sizeof(tok) + sizeof(pos))) {
                return;
            }
            prev_pos = pos;
            off += sizeof(tok) + sizeof(pos) + row_bytes;
        }

        drop_deferred_from(seq_id, 0);
        llama_mtp_prefill_reset(params.ctx_tgt, seq_id);
        std::memcpy(pending_h[seq_id].data(), data.data() + sizeof(header), row_bytes);
        if (adaptive) {
            adaptive_ctrl[seq_id].n_cur   = header[3];
            adaptive_ctrl[seq_id].n_climb = header[4];
            adaptive_ctrl[seq_id].n_drop  = header[5];
        }
        n_last[seq_id] = header[6];
        verify_h_beg[seq_id] = 0;
        verify_h_rows[seq_id] = 0;

        off = sizeof(header) + row_bytes;
        for (uint32_t i = 0; i < header[2]; ++i) {
            llama_token tok;
            llama_pos pos;
            std::memcpy(&tok, data.data() + off, sizeof(tok));
            off += sizeof(tok);
            std::memcpy(&pos, data.data() + off, sizeof(pos));
            off += sizeof(pos);
            defer.tok.push_back(tok);
            defer.pos.push_back(pos);
            defer.seq.push_back(seq_id);
            const size_t k = defer.embd.size();
            defer.embd.resize(k + n_embd);
            std::memcpy(defer.embd.data() + k, data.data() + off, row_bytes);
            off += row_bytes;
        }
    }

    bool get_resume_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || n_embd <= 0 || n_embd > 65536) { return false; }
        uint32_t n_rows = 0;
        for (auto seq : defer.seq) { n_rows += seq == seq_id; }
        if (n_rows > defer_max) { return false; }
        const uint32_t hdr[] = { 0x4d545052, 1, (uint32_t) n_embd, (uint32_t) n_mtp_layers, n_rows };
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        data.resize(sizeof(hdr) + row_bytes + n_rows * (sizeof(llama_token) + sizeof(llama_pos) + row_bytes));
        uint8_t * out = data.data();
        auto put = [&](const void * src, size_t n) { std::memcpy(out, src, n); out += n; };
        put(hdr, sizeof(hdr));
        put(pending_h[seq_id].data(), row_bytes);
        for (size_t i = 0; i < defer.seq.size(); ++i) {
            if (defer.seq[i] != seq_id) { continue; }
            put(&defer.tok[i], sizeof(llama_token));
            put(&defer.pos[i], sizeof(llama_pos));
            put(defer.embd.data() + i * n_embd, row_bytes);
        }
        return true;
    }

    bool set_resume_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) { return false; }
        if (data.empty()) {
            std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
            drop_deferred_from(seq_id, 0);
            verify_h_rows[seq_id] = 0;
            verify_h_beg[seq_id] = -1;
            return true;
        }
        if (data.size() < 5 * sizeof(uint32_t)) { return false; }
        uint32_t hdr[5];
        std::memcpy(hdr, data.data(), sizeof(hdr));
        if (hdr[0] != 0x4d545052 || hdr[1] != 1 || hdr[2] != (uint32_t) n_embd ||
                hdr[3] != (uint32_t) n_mtp_layers || hdr[4] > defer_max || n_embd <= 0 || n_embd > 65536) { return false; }
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        const size_t row_size = sizeof(llama_token) + sizeof(llama_pos) + row_bytes;
        if (data.size() != sizeof(hdr) + row_bytes + hdr[4] * row_size) { return false; }
        size_t other_rows = 0;
        for (auto seq : defer.seq) { other_rows += seq != seq_id; }
        if (other_rows > defer_max || hdr[4] > defer_max - other_rows ||
                other_rows + hdr[4] > (size_t) std::min(batch_capacity, ubatch_capacity)) { return false; }
        // Validate every row before mutating the live carryover.
        for (uint32_t i = 0; i < hdr[4]; ++i) {
            const uint8_t * row = data.data() + sizeof(hdr) + row_bytes + i * row_size;
            llama_token tok; llama_pos pos;
            std::memcpy(&tok, row, sizeof(tok));
            std::memcpy(&pos, row + sizeof(tok), sizeof(pos));
            if (tok < 0 || tok >= llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(params.ctx_dft))) || pos < 0) { return false; }
        }
        drop_deferred_from(seq_id, 0);
        const size_t k0 = defer.tok.size();
        defer.tok.resize(k0 + hdr[4]); defer.pos.resize(k0 + hdr[4]); defer.seq.resize(k0 + hdr[4]);
        defer.embd.resize((k0 + hdr[4]) * (size_t) n_embd);
        std::memcpy(pending_h[seq_id].data(), data.data() + sizeof(hdr), row_bytes);
        for (uint32_t i = 0; i < hdr[4]; ++i) {
            const uint8_t * row = data.data() + sizeof(hdr) + row_bytes + i * row_size;
            std::memcpy(&defer.tok[k0 + i], row, sizeof(llama_token));
            std::memcpy(&defer.pos[k0 + i], row + sizeof(llama_token), sizeof(llama_pos));
            defer.seq[k0 + i] = seq_id;
            std::memcpy(defer.embd.data() + (k0 + i) * n_embd, row + sizeof(llama_token) + sizeof(llama_pos), row_bytes);
        }
        // These indices reference target output buffers, which are not restored.
        verify_h_rows[seq_id] = 0;
        verify_h_beg[seq_id] = -1;
        return true;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // note: the server calls begin() after the prefill decode, so stale defer
        // rows are already handled by the position-rewind trim in process(). Rows
        // that remain here belong to this prompt and feed the next draft decode.

        // new request: reseed the p/q draft RNG once, then it advances across draft rounds
        if (q_smpls[seq_id]) {
            common_sampler_reset(q_smpls[seq_id].get());
        }
        if (seq_id < (llama_seq_id) chain_rng_seeded.size()) {
            chain_rng_seeded[seq_id] = 0; // [#69] same for the sampled chain's uniform stream
        }

        // [#196] new request: restart the adaptive depth controller of this seq at its floor, so the
        // verify widths (and with them the temp-0 output) depend only on this request, not on whatever
        // the previous request on the slot left behind. LLAMA_SPEC_ADAPTIVE_CARRY=1 keeps the old
        // carry-over across requests (A/B arm for the speed gate).
        if (adaptive && seq_id >= 0 && seq_id < (llama_seq_id) adaptive_ctrl.size()) {
            static const bool carry = [] { const char * e = getenv("LLAMA_SPEC_ADAPTIVE_CARRY"); return e && atoi(e) != 0; }();
            if (!carry) {
                adaptive_ctrl[seq_id].reset(params.n_max, params.n_min_adaptive);
            }
        }
        if (cdepth && seq_id >= 0 && seq_id < (llama_seq_id) cd_state.size()) {
            cd_state[seq_id].reset_history(); // the target-confidence history belongs to one request
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max_mem = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        llama_pos pos_max_defer = -1;
        if (defer_enabled) {
            for (size_t k = 0; k < defer.pos.size(); ++k) {
                if (defer.seq[k] == seq_id) {
                    pos_max_defer = std::max(pos_max_defer, defer.pos[k]);
                }
            }
        }
        const llama_pos pos_max = std::max(pos_max_mem, pos_max_defer);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        return process_range(batch_in, 0);
    }

    bool process_range(const llama_batch & batch_in, int32_t output_offset) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // A sequence that starts again at position 0 has no previous target row. Clear the
        // carry the last request on this slot left behind, so a reused server drafts the
        // same way as a fresh one (a fresh server starts with zeros).
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] >= 0 && batch_in.pos[i_batch_beg[seq_id]] == 0) {
                std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
            }
        }

        // The draft memory can retain speculative rows after a rewind (context
        // shift, rollback, or a new prompt). Trim them before the next decode.
        if (!is_mem_shared) {
            auto * mem_dft = llama_get_memory(ctx_dft);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                const llama_pos pos_min_in = batch_in.pos[i_batch_beg[seq_id]];
                // The server trims draft KV to the target's pre-draft max after every
                // draft, so overlap here means an unidentified stale-state producer.
                // Log it: a silently firing trim leaves no forensic signal.
                const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_id);
                if (pos_max_pre >= pos_min_in) {
                    SPC_WRN("stale draft KV for seq %d at process: pos_max %d >= batch start %d, trimming\n",
                            (int) seq_id, (int) pos_max_pre, (int) pos_min_in);
                }
                if (!llama_memory_seq_rm(mem_dft, seq_id, pos_min_in, -1)) {
                    SPC_ERR("failed to trim draft memory for sequence %d from position %d\n",
                            seq_id, (int) pos_min_in);
                    return false;
                }
            }
        }

        // Deferred rows at or past the incoming batch positions are stale: either a
        // new prompt rewound the sequence, or they hold candidates a later verify
        // rejected. Drop them before any decode.
        if (defer_enabled && !defer.tok.empty()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                drop_deferred_from(seq_id, batch_in.pos[i_batch_beg[seq_id]]);
            }
        }

        // [PREFILLFAST] the target staged this range's hidden rows on its device during its decode; the host has
        // only the range's last row (non-final ubatch), so every chunk must read the staged rows
        bool one_seq = true;
        for (int k = 1; k < n_tokens && one_seq; ++k) {
            one_seq = batch_in.seq_id[k][0] == batch_in.seq_id[0][0];
        }
        const bool use_handoff = handoff && one_seq && llama_mtp_handoff_ready(ctx_tgt, output_offset, n_tokens);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        const int32_t defer_cap = defer_capacity();
        if (!is_mem_shared && defer_enabled && n_tokens <= defer_cap && !use_handoff) {
            // defer the catch-up rows; the first draft decode absorbs them, which saves
            // one eval per round. Flush first if rows from an undrafted round remain.
            if (!defer.tok.empty() && (int32_t) defer.tok.size() + n_tokens > defer_cap) {
                if (!flush_deferred()) {
                    return false;
                }
            }

            const size_t k0 = defer.tok.size();
            defer.tok.resize(k0 + n_tokens);
            defer.pos.resize(k0 + n_tokens);
            defer.seq.resize(k0 + n_tokens);
            defer.embd.resize((k0 + n_tokens) * (size_t) n_embd);

            for (int k = 0; k < n_tokens; ++k) {
                defer.tok[k0 + k] = batch_in.token[k];
                defer.pos[k0 + k] = batch_in.pos[k];
                defer.seq[k0 + k] = batch_in.seq_id[k][0];
            }
            // Shift the target rows right by one. The deferred buffer owns this
            // bounded copy; subsequent helper decodes no longer need its source pages.
            if (n_tokens > 1 &&
                    (!llama_copy_embeddings_nextn(ctx_tgt, defer.embd.data() + (k0 + 1) * (size_t) n_embd,
                            output_offset, n_tokens - 1) ||
                     !llama_release_embeddings_nextn(ctx_tgt, output_offset, n_tokens - 1))) {
                return false;
            }
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                std::memcpy(defer.embd.data() + (k0 + i_batch_beg[seq_id]) * (size_t) n_embd, pending_h[seq_id].data(), row_bytes);
            }
        } else if (!is_mem_shared) {
            if (!flush_deferred()) {
                return false;
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers && ok; ++head) {
                if (chain_heads) {
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] >= 0) {
                            llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                        }
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                for (int32_t first = 0; first < n_tokens; first += batch_capacity) {
                    const int32_t count = std::min(batch_capacity, n_tokens - first);
                    common_batch_clear(batch);
                    for (int32_t i = 0; i < count; ++i) {
                        const int32_t k = first + i;
                        common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
                    }
                    const int32_t skip = first == 0 ? 1 : 0;
                    if (use_handoff) {
                        const llama_seq_id s0 = batch_in.seq_id[0][0];
                        const int32_t rc = llama_mtp_handoff_decode(ctx_dft, ctx_tgt, batch, output_offset + first,
                                first == 0 ? pending_h[s0].data() : nullptr);
                        if (rc != 0) {
                            SPC_ERR("handoff llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", (int) rc, (int) batch.pos[0]);
                            ok = false;
                            break;
                        }
                        continue;
                    }
                    // Each helper batch owns a copy of the preceding target rows.
                    // Release those file-backed pages before decoding this batch,
                    // rather than retaining every refault across the whole prompt.
                    // Later heads and accept() can refault the original bytes.
                    if (count > skip &&
                            (!llama_copy_embeddings_nextn(ctx_tgt, batch.embd + (size_t) skip * n_embd,
                                    output_offset + first + skip - 1, count - skip) ||
                             !llama_release_embeddings_nextn(ctx_tgt, output_offset + first + skip - 1, count - skip))) {
                        ok = false;
                        break;
                    }
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        const int32_t i = i_batch_beg[seq_id] - first;
                        if (i >= 0 && i < count) {
                            std::memcpy(batch.embd + (size_t) i * n_embd, pending_h[seq_id].data(), row_bytes);
                        }
                    }
                    int32_t rc = -4;
                    if (prefill_device && !chain_heads && first == 0 && count == n_tokens && output_offset == 0) {
                        std::vector<const float *> carry(n_seq);
                        for (uint32_t seq = 0; seq < n_seq; ++seq) { carry[seq] = pending_h[seq].data(); }
                        rc = llama_mtp_prefill_decode(ctx_dft, ctx_tgt, batch, carry.data(), n_seq);
                    }
                    if (rc == -4) { rc = llama_decode(ctx_dft, batch); }
                    if (rc != 0) {
                        SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n", head, (int) rc, (int) batch.pos[0]);
                        ok = false;
                        break;
                    }
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        // The target's NextN rows are unmasked and reordered by the bulk getter. Avoid one backend sync per token while the draft runs.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            // Target outputs remain valid until the next target decode. Accept reads one row.
            verify_h_rows[seq_id] = n_rows;
            verify_h_beg[seq_id] = output_offset + i_batch_beg[seq_id];

            if (use_handoff) {
                // the target copied this row to the host and released its pages after the ubatch
                if (!llama_mtp_handoff_row(ctx_tgt, pending_h[seq_id].data(), output_offset + i_batch_end[seq_id])) { return false; }
                continue;
            }
            if (!llama_copy_embeddings_nextn(ctx_tgt, pending_h[seq_id].data(), output_offset + i_batch_end[seq_id], 1) ||
                    !llama_release_embeddings_nextn(ctx_tgt, output_offset + i_batch_end[seq_id], 1)) { return false; }
        }

        if (use_handoff) {
            return true;
        }
        // Also release untouched boundary rows and the shared-KV path's rows.
        return llama_release_embeddings_nextn(ctx_tgt, output_offset, n_tokens);
    }

    // [#72] LLAMA_SPEC_DRAFT_TEMP_MULT=<x>: the p/q drafter samples at x times the request's temperature (default 1).
    // Exactness does not depend on it: the verifier's rejection test uses the q the draft token was actually drawn
    // from (the sampler's final candidates, temperature included), so any q > 0 keeps the output distribution. Only
    // the acceptance rate moves: a draft head flatter than the target gains from x < 1, a sharper one from x > 1.
    // With p_min > 0 the confidence cutoff sees the rescaled top probability too (production runs p_min 0).
    static float pq_temp_mult() {
        static const float v = [] {
            const char * e = getenv("LLAMA_SPEC_DRAFT_TEMP_MULT");
            float x = e ? (float) atof(e) : 1.0f;
            if (!(x > 0.0f)) {
                x = 1.0f;
            }
            x = std::min(std::max(x, 0.05f), 4.0f);
            if (e) {
                LOG_INF("spec: LLAMA_SPEC_DRAFT_TEMP_MULT = %.3f (p/q draft temperature = %.3f x request temperature)\n", x, x);
            }
            return x;
        }();
        return v;
    }

    static bool pq_params_ok(const common_params_sampling & tgt) {
        return tgt.temp > 0.0f && tgt.mirostat == 0;
    }

    // the drafter-side sampler that mirrors the target's request: rebuilt only when those params change
    common_sampler * pq_sampler(llama_seq_id seq_id, const common_params_sampling & tgt) {
        auto & cur  = q_params[seq_id];
        auto & smpl = q_smpls[seq_id];

        const bool same = smpl &&
            cur.temp == tgt.temp && cur.dynatemp_range == tgt.dynatemp_range && cur.dynatemp_exponent == tgt.dynatemp_exponent &&
            cur.top_k == tgt.top_k && cur.top_p == tgt.top_p && cur.min_p == tgt.min_p && cur.min_keep == tgt.min_keep &&
            cur.seed == tgt.seed && cur.samplers == tgt.samplers;
        if (same) {
            return smpl.get();
        }

        common_params_sampling sp;
        sp.no_perf           = false;
        sp.temp              = tgt.temp * pq_temp_mult(); // [#72]
        sp.dynatemp_range    = tgt.dynatemp_range;
        sp.dynatemp_exponent = tgt.dynatemp_exponent;
        sp.top_k             = tgt.top_k;
        sp.top_p             = tgt.top_p;
        sp.min_p             = tgt.min_p;
        sp.min_keep          = tgt.min_keep;
        // the draft draw must be independent of the target chain's draw: a different stream
        sp.seed              = tgt.seed == LLAMA_DEFAULT_SEED ? LLAMA_DEFAULT_SEED : (tgt.seed ^ 0x27d4eb2fu);
        sp.backend_sampling  = false;
        sp.samplers.clear();
        bool has_temp = false;
        for (const auto t : tgt.samplers) {
            switch (t) {
                case COMMON_SAMPLER_TYPE_TOP_K:
                case COMMON_SAMPLER_TYPE_TOP_P:
                case COMMON_SAMPLER_TYPE_MIN_P:
                    sp.samplers.push_back(t);
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    sp.samplers.push_back(t);
                    has_temp = true;
                    break;
                default:
                    break; // typical / top-n-sigma / xtc / dry / penalties: q simply differs a little from p
            }
        }
        if (!has_temp) {
            sp.samplers.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
        }

        smpl.reset(common_sampler_init(llama_get_model(params.ctx_dft), sp));
        cur = tgt;
        if (chain_qwen4exp && !chain_rng_qwen.empty()) {
            // Use the core sampler's resolved seed, including LLAMA_DEFAULT_SEED, whenever it rebuilds.
            chain_rng_qwen[seq_id].seed(common_sampler_get_seed(smpl.get()));
            chain_rng_seeded[seq_id] = 1;
        }

        SPC_INF("seq %d: p/q draft sampler rebuilt: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f\n",
                (int) seq_id, sp.temp, sp.top_k, sp.top_p, sp.min_p);

        return smpl.get();
    }

    // [#69] the chain graph mirrors the draft sampler chain top_k -> top_p -> min_p -> temperature (the pq_sampler
    // subset of the request's samplers, in the request's order). backend_k is the support limit applied before
    // that request sampler. Qwen4exp falls back to sequential drafting when this exact shape is unsupported.
    static bool chain_sampled_params(const common_params_sampling & tgt, int32_t & k, float & temp, float & top_p, float & min_p,
                                     int32_t backend_k = 0, bool strict = false) {
        if (tgt.dynatemp_range > 0.0f || tgt.min_keep > 1 || tgt.min_p >= 1.0f) {
            return false;
        }
        static const common_sampler_type order[] = {
            COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE,
        };
        int  at    = 0;
        bool has_k = false;
        bool has_p = false;
        bool has_m = false;
        bool seen[4] = {};
        for (const auto t : tgt.samplers) {
            int idx = -1;
            for (int i = 0; i < 4; ++i) {
                if (order[i] == t) {
                    idx = i;
                }
            }
            if (idx < 0) {
                continue; // not mirrored by the p/q draft sampler either
            }
            if (idx < at || (strict && seen[idx])) {
                return false;
            }
            seen[idx] = true;
            at = idx;
            has_k = has_k || idx == 0;
            has_p = has_p || idx == 1;
            has_m = has_m || idx == 2;
        }
        if (!has_k && backend_k <= 0) {
            return false;
        }
        // Preserve the CPU candidate order used for its inverse CDF. Backend top-k is not
        // required to return sorted candidates; an active CPU top-k or top-p sorts them.
        if (strict && backend_k > 0 && !(has_k && tgt.top_k > 0) && !(has_p && tgt.top_p < 1.0f)) {
            return false;
        }
        k = backend_k > 0 ? (has_k && tgt.top_k > 0 ? std::min(backend_k, tgt.top_k) : backend_k) : tgt.top_k;
        if (k <= 0 || k > LLAMA_MTP_CHAIN_TOP_K_MAX) {
            return false;
        }
        temp  = tgt.temp * pq_temp_mult();
        top_p = has_p ? tgt.top_p : 1.0f;
        min_p = has_m ? tgt.min_p : 0.0f;
        return temp > 0.0f && (!strict || (std::isfinite(temp) && std::isfinite(top_p) && top_p > 0.0f &&
                                         std::isfinite(min_p)));
    }

    void repair_qwen_chain(llama_seq_id seq, common_speculative_draft_params & dp,
            common_sampler * sampler, std::vector<float> hidden, common_spec_host_phase_call & hp) {
        auto * ctx = params.ctx_dft;
        struct sampler_scope {
            llama_context * ctx;
            llama_seq_id seq;
            llama_sampler * chain = nullptr;
            ~sampler_scope() {
                if (chain) {
                    llama_synchronize(ctx);
                    llama_set_sampler(ctx, seq, nullptr);
                    llama_sampler_free(chain);
                }
            }
        } backend { ctx, seq };
        auto fail = [&](const char * reason) {
            ++chain_st.repair_failed;
            SPC_ERR("Qwen4exp chain suffix repair failed for seq %d: %s\n", (int) seq, reason);
        };
        if (params.backend_sampling) {
            backend.chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(backend.chain, llama_sampler_init_top_k(10));
            if (!llama_set_sampler(ctx, seq, backend.chain)) {
                fail("cannot attach the sequential top-k sampler");
                return;
            }
        }
        const llama_pos first = dp.pos0 + (llama_pos) dp.result->size();
        if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, first, -1)) {
            fail("cannot discard provisional KV suffix");
            return;
        }
        ++chain_st.repaired;
        while ((int) dp.result->size() < n_cap[seq]) {
            common_batch_clear(batch);
            common_batch_add(batch, dp.result->back(), dp.pos0 + (llama_pos) dp.result->size(), { seq }, true);
            std::memcpy(batch.embd, hidden.data(), (size_t) n_embd*sizeof(float));
            const int64_t t0 = hp.on ? ggml_time_us() : 0;
            const int ret = llama_decode(ctx, batch);
            if (hp.on) {
                const int64_t t1 = ggml_time_us();
                llama_synchronize(ctx);
                hp.acc.decode_us += t1 - t0;
                hp.acc.sync_us += ggml_time_us() - t1;
                ++hp.acc.n_steps;
            }
            if (ret != 0) {
                fail("sequential helper decode failed");
                return;
            }
            ++chain_st.repair_steps;
            const int64_t ts = hp.on ? ggml_time_us() : 0;
            const llama_token id = common_sampler_sample(sampler, ctx, 0, true);
            llama_mtp_chain_qwen_commit(chain_rng_qwen[seq], 1);
            if (hp.on) {
                hp.acc.sample_us += ggml_time_us() - ts;
            }
            const auto * candidates = common_sampler_get_candidates(sampler, true);
            if (candidates->size == 0) {
                fail("sequential sampler returned no candidates");
                return;
            }
            if (candidates->data[0].p < params.p_min) {
                return;
            }
            common_sampler_accept(sampler, id, true);
            dp.result->push_back(id);
            dp.result_q->emplace_back(candidates->data, candidates->data + candidates->size);
            ++chain_st.steps;
            if ((int) dp.result->size() < n_cap[seq]) {
                const float * row = llama_get_embeddings_nextn_ith(ctx, 0);
                if (row == nullptr) {
                    fail("sequential helper returned no raw residual");
                    return;
                }
                hidden.assign(row, row + n_embd);
            }
        }
    }

    void chain_log_stats() const {
        SPC_INF("sampled chain: %llu rounds, %llu draft tokens, %llu GPU/host disagreements, %llu malformed; %llu repaired rounds, %llu repair steps, %llu repair failures\n",
                (unsigned long long) chain_st.rounds, (unsigned long long) chain_st.steps,
                (unsigned long long) chain_st.cut, (unsigned long long) chain_st.bad,
                (unsigned long long) chain_st.repaired, (unsigned long long) chain_st.repair_steps,
                (unsigned long long) chain_st.repair_failed);
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_spec_host_phase_call hp; // [#47]

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        bool any_drafting = false;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            any_drafting = any_drafting || dparams[seq_id].drafting;
        }

        if (chain_graph && chain_qwen4exp && any_drafting) {
            bool supported = chain_sampled;
            for (llama_seq_id seq_id = 0; supported && seq_id < (llama_seq_id) n_seq; ++seq_id) {
                const auto & dp = dparams[seq_id];
                if (!dp.drafting) {
                    continue;
                }
                int32_t k;
                float temp, top_p, min_p;
                // Preserve the ordinary backend top10 support and order before the CPU filters, including ties.
                supported = params.backend_sampling && dp.sampling && dp.result_q && pq_params_ok(*dp.sampling) &&
                    chain_sampled_params(*dp.sampling, k, temp, top_p, min_p, 10, true) && k == 10;
                if (supported) {
                    auto * cpu_chain = common_sampler_get(pq_sampler(seq_id, *dp.sampling));
                    const int32_t n = llama_sampler_chain_n(cpu_chain);
                    supported = n > 0 && std::strcmp(llama_sampler_name(llama_sampler_chain_get(cpu_chain, n - 1)), "dist") == 0;
                }
            }
            if (!supported) {
                SPC_WRN("%s", "Qwen4exp chain cannot mirror this request's draft sampler; using sequential MTP for this context\n");
                chain_graph = false;
                // Retire any carried accepted prefix before returning to the ordinary sequential path.
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (dparams[seq_id].drafting) {
                        drop_deferred_from(seq_id, dparams[seq_id].pos0);
                    }
                }
                if (!flush_deferred()) {
                    return;
                }
                // The flushed input rows remain in the shared batch.
                common_batch_clear(batch);
                const char * env_d = getenv("LLAMA_SPEC_DEFER_CATCHUP");
                defer_enabled = env_d != nullptr && atoi(env_d) != 0;
                if (params.backend_sampling) {
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                        llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
                        if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                            SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                            llama_sampler_free(chain);
                            chain = nullptr;
                        }
                        backend_chains[seq_id] = chain;
                    }
                }
            }
        }

        // chained drafting: one decode drafts n_max tokens for a single sequence.
        // this block must run before the generic defer merge below - it consumes
        // the deferred rows itself and prepends them to the chain batch
        if (chain_graph && any_drafting) {
            llama_seq_id seq_one = -1;
            int n_seq_drafting = 0;
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (dparams[s].drafting) {
                    n_seq_drafting++;
                    seq_one = s;
                }
            }

            if (n_seq_drafting == 1) {
                auto & dp = dparams[seq_one];
                auto * smpl = smpls[seq_one].get();
                common_sampler_reset(smpl);

                // effective draft cap: adaptive depth (or n_max), then clamped by the
                // per-call context bound from the server (same as the sequential path)
                n_cap[seq_one] = adaptive ? adaptive_ctrl[seq_one].n_cur : params.n_max;
                if (cdepth && !cd_chain_warned) {
                    cd_chain_warned = true;
                    LOG_WRN("%s: --spec-depth-policy cdepth applies to the per-token draft loop; the chained-graph draft path keeps the streak controller\n", __func__);
                }
                if (dp.n_max > 0 && dp.n_max < n_cap[seq_one]) {
                    n_cap[seq_one] = dp.n_max;
                }
                n_own[seq_one] = n_cap[seq_one];
                const int n_chain = n_cap[seq_one];
                if (chain_qwen4exp && n_chain <= 0) {
                    dp.result->clear();
                    if (dp.result_q) {
                        dp.result_q->clear();
                    }
                    n_last[seq_one] = 0;
                    return;
                }
                if (dp.conf_depth) {
                    *dp.conf_depth = n_chain;
                }

                // deferred rows at or past n_past hold candidates the verify
                // rejected; the committed prefix ends at n_past - 1. Drop them.
                drop_deferred_from(seq_one, dp.pos0);

                // rows of other sequences cannot join a single-sequence chain
                // batch; decode all remaining rows standalone in that case
                bool defer_other = false;
                for (size_t k = 0; k < defer.tok.size(); ++k) {
                    defer_other = defer_other || defer.seq[k] != seq_one;
                }
                if ((defer_other || defer.tok.size() + n_chain > (size_t) std::min(batch_capacity, ubatch_capacity)) && !flush_deferred()) {
                    return;
                }

                common_batch_clear(batch);

                // deferred catch-up rows ride along, before the chain rows
                const int n_catchup = (int) defer.tok.size();
                for (int k = 0; k < n_catchup; ++k) {
                    common_batch_add(batch, defer.tok[k], defer.pos[k], { defer.seq[k] }, false);
                    std::memcpy(batch.embd + (size_t) k * n_embd, defer.embd.data() + (size_t) k * n_embd, row_bytes);
                }
                defer.tok.clear();
                defer.pos.clear();
                defer.seq.clear();
                defer.embd.clear();

                for (int j = 0; j < n_chain; ++j) {
                    common_batch_add(batch, j == 0 ? dp.id_last : 0, dp.pos0 + j, { seq_one }, true);
                    if (j == 0) {
                        std::memcpy(batch.embd + (size_t) n_catchup * n_embd, pending_h[seq_one].data(), row_bytes);
                    } else {
                        std::memset(batch.embd + (size_t) (n_catchup + j) * n_embd, 0, row_bytes);
                    }
                }

                // The chain batch starts at the deferred catch-up rows, which can sit
                // below the draft KV max from an earlier/rejected draft (position
                // rewind or a repeated decode). Drop the draft cells at/above the
                // batch's own minimum position before decoding so the position
                // consistency check (X < Y) holds; the chain re-writes that region.
                if (batch.n_tokens > 0) {
                    llama_pos pos_min = batch.pos[0];
                    for (int k = 1; k < batch.n_tokens; ++k) {
                        pos_min = std::min(pos_min, batch.pos[k]);
                    }
                    auto * mem_dft = llama_get_memory(ctx_dft);
                    // Normal flow: the chain batch starts right past the draft KV max,
                    // so an overlap means a stale-state producer this trim is papering
                    // over. Log it: a silently firing trim leaves no forensic signal.
                    const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_one);
                    if (pos_max_pre >= pos_min) {
                        SPC_WRN("stale draft KV for seq %d before chain decode: pos_max %d >= batch start %d, trimming\n",
                                (int) seq_one, (int) pos_max_pre, (int) pos_min);
                    }
                    if (!llama_memory_seq_rm(mem_dft, seq_one, pos_min, -1)) {
                        SPC_ERR("failed to trim draft memory for sequence %d before chain decode\n", (int) seq_one);
                        return;
                    }
                }

                if (dp.result_q) {
                    dp.result_q->clear(); // the argmax chain leaves it empty: id-match verification
                }

                // [#69] sampled chain: the round's uniforms, one per chain step, drawn before the decode
                int32_t s_k     = 0;
                float   s_temp  = 0.0f;
                float   s_top_p = 1.0f;
                float   s_min_p = 0.0f;
                const bool sampled = chain_sampled && dp.sampling && dp.result_q && pq_params_ok(*dp.sampling) &&
                        chain_sampled_params(*dp.sampling, s_k, s_temp, s_top_p, s_min_p,
                                chain_qwen4exp && params.backend_sampling ? 10 : 0, chain_qwen4exp);
                common_sampler * qwen_cpu_sampler = nullptr;
                if (sampled) {
                    if (chain_qwen4exp) {
                        qwen_cpu_sampler = pq_sampler(seq_one, *dp.sampling);
                        if (!chain_rng_seeded[seq_one]) {
                            chain_rng_qwen[seq_one].seed(common_sampler_get_seed(qwen_cpu_sampler));
                            chain_rng_seeded[seq_one] = 1;
                        }
                        llama_mtp_chain_qwen_uniforms(chain_rng_qwen[seq_one], n_chain, chain_u_host, chain_u);
                    } else {
                        auto & rng = chain_rng[seq_one];
                        if (!chain_rng_seeded[seq_one]) {
                            // independent of the target's and the p/q verifier's streams
                            const uint32_t seed = dp.sampling->seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : (dp.sampling->seed ^ 0x6a09e667u);
                            rng.seed(seed);
                            chain_rng_seeded[seq_one] = 1;
                        }
                        chain_u.resize(n_chain);
                        for (auto & x : chain_u) {
                            x = (float) ((double) (rng() >> 40) * 0x1.0p-24); // Qwen35's existing stream
                        }
                    }
                    llama_set_mtp_chain_sampling(ctx_dft, s_k, s_temp, s_top_p, s_min_p, chain_u.data(), n_chain);
                }

                // TODO(mtp-chain): llama_set_mtp_chain() is provided by the llama API
                // backport (llama-ext.h / llama-context.cpp); the chain decode depends
                // on its in-graph chained sampling writing [token, prob] pairs to logits
                const bool chain_timing = hp.on && chain_qwen4exp;
                const int64_t t_dec0 = chain_timing ? ggml_time_us() : 0;
                llama_set_mtp_chain(ctx_dft, true);
                const int ret = llama_decode(ctx_dft, batch);
                if (chain_timing) {
                    const int64_t t_dec1 = ggml_time_us();
                    llama_synchronize(ctx_dft);
                    hp.acc.decode_us += t_dec1 - t_dec0;
                    hp.acc.sync_us += ggml_time_us() - t_dec1;
                    hp.acc.n_steps += n_chain;
                }
                llama_set_mtp_chain(ctx_dft, false);
                if (sampled) {
                    llama_set_mtp_chain_sampling(ctx_dft, 0, 0.0f, 0.0f, 0.0f, nullptr, 0);
                }

                if (ret != 0) {
                    SPC_ERR("llama_decode(chain) returned %d\n", ret);
                    return;
                }

                // the chain samples greedily in-graph and emits [token id, top prob]
                // pairs as 2-float rows, packed from the start of the logits buffer;
                // no host-side sampling pass runs over the draft logits
                const float * lp = llama_get_logits(ctx_dft);

                auto & result = *dp.result;
                const int64_t t_smp0 = chain_timing ? ggml_time_us() : 0;

                if (sampled) {
                    // [#69] rows of LLAMA_MTP_CHAIN_ROW(s_k) floats: re-derive every step on the host; the tokens and
                    // the q rows handed to the verifier are the host's, so p/q stays exact whatever the GPU drew.
                    // Later rows depend on the GPU token. Qwen repairs a disagreement with ordinary
                    // sampler steps; the existing Qwen35 path stops at that valid prefix.
                    const int32_t W = LLAMA_MTP_CHAIN_ROW(s_k);
                    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
                    chain_st.rounds++;
                    int32_t n_consumed = 0;
                    std::vector<float> repair_hidden;
                    for (int j = 0; j < n_chain; ++j) {
                        const float * row = lp + (size_t) W*j;
                        std::vector<llama_token_data> q;
                        const int32_t pick = chain_qwen4exp
                            ? llama_mtp_chain_rederive_qwen(row, s_k, common_sampler_get(qwen_cpu_sampler), n_vocab, q)
                            : llama_mtp_chain_rederive(row, s_k, s_temp, s_top_p, s_min_p, (double) chain_u[j], n_vocab, q);
                        if (pick < 0) {
                            chain_st.bad++;
                            break;
                        }
                        ++n_consumed; // the core dist draws before the confidence check, including size-one support
                        if (q[0].p < params.p_min) {
                            break;
                        }
                        const llama_token id = (llama_token) row[2 + pick];
                        float selected_p = 0.0f;
                        for (const auto & candidate : q) {
                            if (candidate.id == id) {
                                selected_p = candidate.p;
                                break;
                            }
                        }

                        SPC_DBG(" - seq_id %d, sampled chain step %3d: %6d (q %8.3f, u %.6f)%s '%s'\n",
                                seq_one, j, id, selected_p, chain_u[j],
                                (chain_qwen4exp ? row[0] == (float) id : (int32_t) row[1] == pick) ? "" : " [disagreement]",
                                common_token_to_piece(ctx_dft, id).c_str());

                        result.push_back(id);
                        if (qwen_cpu_sampler) {
                            common_sampler_accept(qwen_cpu_sampler, id, true);
                        }
                        if (dp.conf_p1) {
                            float p1 = 0.0f;
                            float p2 = 0.0f;
                            for (const auto & c : q) {
                                if (c.p > p1) {
                                    p2 = p1;
                                    p1 = c.p;
                                } else if (c.p > p2) {
                                    p2 = c.p;
                                }
                            }
                            spec_conf_push(dp, p1, p2);
                        }
                        dp.result_q->push_back(std::move(q));
                        chain_st.steps++;

                        const bool mismatch = chain_qwen4exp ? row[0] != (float) id : (int32_t) row[1] != pick;
                        if (mismatch) {
                            chain_st.cut++;
                            if (chain_qwen4exp && j + 1 < n_chain) {
                                const float * h = llama_get_embeddings_nextn_ith(ctx_dft, n_catchup + j);
                                if (h) {
                                    repair_hidden.assign(h, h + n_embd);
                                } else {
                                    ++chain_st.repair_failed;
                                    SPC_ERR("Qwen4exp chain suffix repair has no raw residual for seq %d step %d\n", (int) seq_one, j);
                                }
                            }
                            break;
                        }
                    }

                    if (chain_qwen4exp) {
                        llama_mtp_chain_qwen_commit(chain_rng_qwen[seq_one], n_consumed);
                    }
                    if (chain_timing) {
                        hp.acc.sample_us += ggml_time_us() - t_smp0;
                    }
                    if (!repair_hidden.empty()) {
                        repair_qwen_chain(seq_one, dp, qwen_cpu_sampler, std::move(repair_hidden), hp);
                    }
                    if ((chain_st.rounds & 1023) == 0) {
                        chain_log_stats();
                    }

                    n_last[seq_one] = (int) result.size();
                    if (!adaptive && dp.result->size() < (size_t) (dp.n_min >= 0 ? dp.n_min : params.n_min)) {
                        dp.result->clear();
                        dp.result_q->clear();
                        if (dp.conf_p1) {
                            spec_conf_clear(dp);
                        }
                    }
                    return;
                }

                const int stride = spec_conflog_enabled() ? 3 : 2;
                for (int j = 0; j < n_chain; ++j) {
                    const llama_token id = (llama_token) lp[stride*j + 0];
                    const float       p  =               lp[stride*j + 1];

                    SPC_DBG(" - seq_id %d, chain candidate %3d: %6d (%8.3f) '%s'\n",
                            seq_one, j, id, p,
                            common_token_to_piece(ctx_dft, id).c_str());

                    if (p < params.p_min) {
                        break;
                    }

                    result.push_back(id);
                    if (dp.conf_p1) {
                        spec_conf_push(dp, p, lp[stride*j + 2]);
                    }

                    if ((int) result.size() >= n_chain) {
                        break;
                    }
                }

                if (chain_timing) {
                    hp.acc.sample_us += ggml_time_us() - t_smp0;
                }
                n_last[seq_one] = (int) result.size();

                // the adaptive controller decides its own depth, so the generic n_min
                // draft cutoff does not apply to it
                if (!adaptive && dp.result->size() < (size_t) (dp.n_min >= 0 ? dp.n_min : params.n_min)) {
                    dp.result->clear();
                    if (dp.conf_p1) {
                        spec_conf_clear(dp);
                    }
                }
                return;
            }
        }

        // deferred catch-up rows ride along with the first draft decode. They carry
        // earlier positions, so they must precede the draft rows: recurrent state
        // advances in batch order. If nothing drafts this round they stay deferred
        // and process() flushes them later.
        //
        // For every seq drafting this round, deferred rows at or past its n_past are
        // candidates the verify just rejected (the committed prefix ends at
        // n_past - 1, same as the chain path). Merging them would decode a rejected
        // token and this round's first draft row at the same position in the same
        // batch: duplicate-position draft KV cells, no X < Y error, silent quality
        // loss. Only reachable at n_parallel > 1 (a single drafting seq takes the
        // chain path, which already drops these).
        if (any_drafting && !defer.tok.empty()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (dparams[seq_id].drafting) {
                    drop_deferred_from(seq_id, dparams[seq_id].pos0);
                }
            }
        }
        if (any_drafting && defer.tok.size() > (size_t) defer_capacity()) {
            if (!flush_deferred()) {
                return;
            }
            common_batch_clear(batch);
        }
        if (any_drafting && !defer.tok.empty()) {
            for (size_t k = 0; k < defer.tok.size(); ++k) {
                common_batch_add(batch, defer.tok[k], defer.pos[k], { defer.seq[k] }, false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, defer.embd.data() + k * (size_t) n_embd, row_bytes);
            }
            defer.tok.clear();
            defer.pos.clear();
            defer.seq.clear();
            defer.embd.clear();
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());
            // do not reset q_smpls here: reset reseeds the RNG, so each round would reuse the same draws (reset is in begin())
            if (dp.result_q) {
                dp.result_q->clear();
            }

            // effective draft cap for this step: adaptive depth (or the user n_max),
            // then clamped by the per-call context bound from the server
            n_cap[seq_id] = (adaptive && !cdepth) ? adaptive_ctrl[seq_id].n_cur : params.n_max;
            if (cdepth) {
                cd_state[seq_id].begin_round();
            }
            if (dp.n_max > 0 && dp.n_max < n_cap[seq_id]) {
                n_cap[seq_id] = dp.n_max;
            }

            if (dp.conf_depth) {
                *dp.conf_depth = n_cap[seq_id];
            }
            n_own[seq_id] = n_cap[seq_id];

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
            spec_ple_hint(params.ctx_tgt, dp, nullptr);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        // [PARW] equal-depth multi-seq drafting: the target verifies every slot in one batch, and a hybrid/recurrent
        // target splits that batch into one ubatch (a full pass over the weights) per distinct draft length
        // (llama_batch_allocr::split_equal). With per-slot adaptive depths (3 vs 4) --parallel 4 ran 2-4 weight passes
        // per round. Draft every seq to the round's deepest cap (each still bounded by its own context limit dp.n_max);
        // the extra tokens are verified like any draft (lossless), and the adaptive controller still sees only the
        // seq's own depth (accept()). LLAMA_SPEC_EQUAL_DEPTH=0 turns it off.
        if (n_drafting > 1 && equal_depth_enabled()) {
            int c_max = 0;
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (drafting[seq_id]) {
                    c_max = std::max(c_max, n_cap[seq_id]);
                }
            }
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (drafting[seq_id]) {
                    const int bound = dparams[seq_id].n_max > 0 ? dparams[seq_id].n_max : c_max;
                    n_cap[seq_id] = std::max(n_cap[seq_id], std::min(c_max, bound));
                    if (dparams[seq_id].conf_depth) {
                        *dparams[seq_id].conf_depth = n_cap[seq_id];
                    }
                }
            }
        }

        int i = 0;

        // [RTPORT] pipelined guess steps: while the card runs step i, the host already queues step i + 1 (batch,
        // memory slot, graph reuse, inputs, launch); that step's first kernel waits on the card for the token row the
        // host publishes right after drawing token i. Only when step i + 1 is certain to run (fixed cap this round, no
        // confidence stop), so no card time is spent on a step that gets thrown away.
        llama_seq_id pipe_seq = 0;
        if (mtp_pipe) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (drafting[seq_id]) {
                    pipe_seq = seq_id;
                }
            }
        }
        const bool pipe_round = mtp_pipe && n_drafting == 1 && !cdepth && !(params.p_min > 0.0f);
        bool pipe_queued = false; // step i is already on the card (queued in the previous iteration)

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            const int64_t t_dec0 = hp.on ? ggml_time_us() : 0;
            int ret = 0;
            if (pipe_round) {
                if (!pipe_queued) {
                    ret = llama_mtp_pipe_decode(ctx_dft, batch, i > 0, false);
                    if (ret == -4) {
                        ret = llama_mtp_device_feedback_decode(ctx_dft, batch, i > 0);
                    }
                }
                pipe_queued = false;
                if (ret == 0 && i + 1 < n_cap[pipe_seq]) {
                    // the token is a placeholder: the step reads its row from the host once drawn below
                    common_batch_clear(batch);
                    common_batch_add(batch, 0, dparams[pipe_seq].pos0 + i + 1, { pipe_seq }, true);
                    pipe_queued = llama_mtp_pipe_decode(ctx_dft, batch, true, true) == 0;
                }
                if (ret == 0 && !llama_mtp_pipe_step_ok(ctx_dft)) {
                    // never seen: step i waited too long for its token. Drop it and everything after it.
                    llama_mtp_pipe_cancel(ctx_dft);
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), pipe_seq, dparams[pipe_seq].pos0 + i, -1);
                    break;
                }
            } else {
                ret = device_feedback ? llama_mtp_device_feedback_decode(ctx_dft, batch, i > 0) : llama_decode(ctx_dft, batch);
            }
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                if (pipe_queued) {
                    llama_mtp_pipe_cancel(ctx_dft);
                    pipe_queued = false;
                }
                break;
            }
            if (hp.on) {
                const int64_t t_dec1 = ggml_time_us();
                llama_synchronize(ctx_dft); // [#47] separates the GPU wait from host sampling; the sampler syncs anyway
                const int64_t t_dec2 = ggml_time_us();
                hp.acc.decode_us += t_dec1 - t_dec0;
                hp.acc.sync_us   += t_dec2 - t_dec1;
                hp.acc.n_steps   += 1;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                // exact p/q: sample the draft token from the request-matched distribution and keep that
                // distribution for the verifier; otherwise the drafter's argmax under its own top-k chain
                bool pq = pq_enabled && dp.sampling && dp.result_q && pq_params_ok(*dp.sampling);

                auto * smpl = pq ? pq_sampler(seq_id, *dp.sampling) : smpls[seq_id].get();

                const int64_t t_smp0 = hp.on ? ggml_time_us() : 0;
                llama_token id = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                if (hp.on) {
                    hp.acc.sample_us += ggml_time_us() - t_smp0;
                }
                const float * h_row = device_feedback ? nullptr : llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                if (pq && llama_get_sampled_token_ith(ctx_dft, i_last[seq_id]) != LLAMA_TOKEN_NULL) {
                    // the backend chain selected the token itself, so the recorded distribution would not be
                    // the one it was drawn from: this position (and the rest of the draft) is verified by id-match
                    if (!pq_backend_warned) {
                        SPC_WRN("%s", "draft backend sampler returns tokens; p/q drafting falls back to argmax\n");
                        pq_backend_warned = true;
                    }
                    pq = false;
                }

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                if (!pq) {
                    // add drafted token for each sequence
                    id = cur_p->data[0].id;
                }

                // only collect very high-confidence draft tokens (p/q: the top candidate's probability under
                // the request-matched distribution is the same confidence measure)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                if (dp.result_q) {
                    if (pq) {
                        dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                    } else {
                        dp.result_q->emplace_back(); // id-match row, keeps result_q aligned with result
                    }
                }

                result.push_back(id);
                spec_ple_hint(params.ctx_tgt, dp, &result);
                if (dp.conf_p1) {
                    spec_conf_push_candidates(dp, cur_p, ctx_dft, i_last[seq_id]);
                }
                {
                    // [RTPORT] LLAMA_MTP_Q_DUMP=<file> (diagnostic, default off): every guess step's sampling
                    // distribution, to measure the KL of a valid-math helper change between two same-seed runs
                    static FILE * q_dump = [] {
                        const char * path = getenv("LLAMA_MTP_Q_DUMP");
                        return path && *path ? fopen(path, "w") : nullptr;
                    }();
                    if (q_dump) {
                        fprintf(q_dump, "%d %d %d %d", seq_id, dp.pos0 + i, i, id);
                        for (size_t k = 0; k < cur_p->size; ++k) {
                            if (cur_p->data[k].p > 0.0f) {
                                fprintf(q_dump, " %d:%.9g", cur_p->data[k].id, cur_p->data[k].p);
                            }
                        }
                        fputc('\n', q_dump);
                        fflush(q_dump);
                    }
                }

                if (cdepth) {
                    // CDEPTH: stop before the next draft step when one more token is not expected to pay for itself.
                    // Uses only the helper's own distributions and already verified history (lossless).
                    const float p1 = cur_p->data[0].p;
                    const float p2 = cur_p->size > 1 ? cur_p->data[1].p : 0.0f;
                    const bool go = cd_state[seq_id].keep_going(cd_model, (int) result.size(), params.n_min_adaptive, p1, p2, params.depth_theta);
                    if (!go && n_cap[seq_id] > (int) result.size()) {
                        n_cap[seq_id] = (int) result.size();
                    }
                }

                if (n_cap[seq_id] <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.pos0 + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (same_position_draft) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.pos0, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    // is_mem_shared models with a real trained NextN head (qwen35, qwen4exp, ...)
                    // still draft at incrementing positions, same as the non-shared-memory path
                    common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
                    if (!device_feedback) {
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                    }
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (pipe_queued) {
                if (drafting[pipe_seq] && batch.n_tokens == 1) {
                    llama_mtp_pipe_supply(ctx_dft, batch.token[0]);
                } else {
                    // stopped after all: release the queued step and drop the K/V row it writes
                    llama_mtp_pipe_cancel(ctx_dft);
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), pipe_seq, dparams[pipe_seq].pos0 + i + 1, -1);
                    pipe_queued = false;
                }
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            n_last[seq_id] = (int) dp.result->size();

            // the adaptive controller decides its own depth, so the generic n_min
            // draft cutoff does not apply to it
            if (!adaptive && dp.result->size() < (size_t) (dp.n_min >= 0 ? dp.n_min : params.n_min)) {
                dp.result->clear();
                if (dp.conf_p1) {
                    spec_conf_clear(dp);
                }
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // update the adaptive controller only when this implementation produced the
        // accepted draft; on is_other the stats belong to a different speculator
        if (adaptive && !is_other) {
            const int depth_before = adaptive_ctrl[seq_id].n_cur;
            // [PARW] judge the controller at the seq's own depth: tokens drafted past it (equal-depth lift) are a
            // bonus the controller did not ask for, and verification is sequential, so the accepted prefix within
            // the own depth is what a draft of that depth would have returned
            const int own   = n_own[seq_id] > 0 ? std::min(n_last[seq_id], n_own[seq_id]) : n_last[seq_id];
            int n_acc = std::min<int>(n_accepted, own);
            if (n_accepted_ctrl >= 0) {
                n_acc = std::min<int>(n_acc, n_accepted_ctrl); // [MBFIX] exact-rule count under mode-biased MTP
            }
            adaptive_ctrl[seq_id].update(own, n_acc, params.n_max, params.n_min_adaptive);
            if (adaptive_ctrl[seq_id].n_cur != depth_before) {
                SPC_DBG("adaptive draft depth seq %d: %d -> %d (n_draft=%d, n_accepted=%d)\n",
                        (int) seq_id, depth_before, adaptive_ctrl[seq_id].n_cur, n_last[seq_id], n_accepted);
            }
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        GGML_ASSERT(llama_copy_embeddings_nextn(params.ctx_tgt, pending_h[seq_id].data(), verify_h_beg[seq_id] + i_h, 1));
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

// Self-speculative decoding with the 3-level n-gram cache of common/ngram-cache.h:
//   context: n-grams of the tokens of the current sequence, rebuilt when the sequence shrinks
//   dynamic: n-grams accumulated over previous requests; with save_dynamic also over previous runs of the
//            server, through the file given by --lookup-cache-dynamic
//   static:  n-grams of a corpus prepared with llama-lookup-create, read-only
struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    // draft length used when --spec-ngram-cache-n-max is not given
    static constexpr uint16_t N_DRAFT_DEFAULT = 8;

    // the draft length the implementation will use for these params; the single place that resolves the
    // default, so the server sizes its per-sequence buffers (common_speculative_n_max) for the same length
    static uint16_t n_draft_for(const common_params_speculative_ngram_cache & params) {
        return params.n_max > 0 ? (uint16_t) params.n_max : N_DRAFT_DEFAULT;
    }

    uint16_t n_draft;

    // shared across all sequences
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    // writeback of the dynamic cache, see sync() and checkpoint_poll()
    bool               save_dynamic = false; // params.save_dynamic and a path to write to
    common_ngram_cache ngram_cache_delta;    // what was added to ngram_cache_dynamic and is not in the file yet
    bool               file_present = false; // the file has been loaded or written by this process
    int64_t            t_last_sync_ms = 0;
    size_t             n_max_dynamic = 0;    // 0 = unlimited

    common_ngram_cache_vocab_id vocab_id;

    // Periodic checkpoints are written by a worker thread so that the thread driving inference never waits
    // for the file lock, the disk or the prune. The inference thread hands the delta over (an O(1) swap),
    // the worker merges it into the file, and the inference thread takes the merged result back as the
    // new in-memory cache at a later request boundary.
    //
    // Locking: everything named ckpt_* is shared with the worker and only touched with mutex_ckpt held,
    // by either side. The worker also reads params, n_max_dynamic and vocab_id, which are constant once
    // the constructor has started it. It never touches anything else of this struct: ngram_cache_dynamic,
    // ngram_cache_delta, sinfos etc. belong to the inference thread alone. The mutex is only ever held
    // for the O(1) handovers, never during the write, so the inference thread never blocks on it for
    // longer than that; it uses try_lock anyway and simply retries at the next request.
    //
    // Ownership of the delta: in CKPT_QUEUED and CKPT_WRITING the worker owns ckpt_delta. On success the
    // worker returns ckpt_cache = the file's content after the write and an empty ckpt_delta; the delta
    // is then in the file and nowhere else. On failure the worker returns ckpt_delta untouched and the
    // inference thread merges it back into ngram_cache_delta, so the next checkpoint (or the shutdown
    // sync) retries it. Nothing is written twice and nothing is dropped.
    //
    // Only one checkpoint is ever in flight: a tick that finds the worker busy does nothing, the delta
    // keeps accumulating in ngram_cache_delta and goes out with the next checkpoint.
    enum ckpt_status_t {
        CKPT_IDLE,    // nothing in flight, ckpt_delta and ckpt_cache are empty
        CKPT_QUEUED,  // ckpt_delta is handed over, the worker has not picked it up yet
        CKPT_WRITING, // the worker holds the delta and is writing the file (outside the mutex)
        CKPT_DONE,    // the write finished: ckpt_ok, ckpt_err, ckpt_cache, ckpt_delta hold the result
        CKPT_RETIRE,  // ckpt_cache holds the previous in-memory cache, for the worker to free
    };

    std::thread             thread_ckpt;
    std::mutex              mutex_ckpt;
    std::condition_variable cv_ckpt;
    ckpt_status_t           ckpt_status = CKPT_IDLE;
    bool                    ckpt_stop   = false;
    common_ngram_cache      ckpt_delta;
    common_ngram_cache      ckpt_cache;
    bool                    ckpt_ok        = false;
    bool                    ckpt_from_disk = false;
    std::string             ckpt_err;
    int64_t                 ckpt_t_us = 0;

    struct seq_info {
        size_t cache_size = 0; // number of tokens of the sequence in ngram_cache_context

        common_ngram_cache ngram_cache_context;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
    {
        const std::string & path_static  = this->params.lookup_cache_static;
        const std::string & path_dynamic = this->params.lookup_cache_dynamic;

        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, n_min=%d, cache_static=%s, cache_dynamic=%s, save_dynamic=%d\n",
                n_draft, this->params.n_min,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str(),
                this->params.save_dynamic ? 1 : 0);

        // a minimum above the draft length would discard every draft
        if (this->params.n_min > (int32_t) n_draft) {
            SPC_WRN("ngram-cache n_min=%d exceeds the draft length %d -- clamping to %d\n",
                    this->params.n_min, n_draft, n_draft);
            this->params.n_min = std::min(this->params.n_min, (int32_t) n_draft);
        }

        sinfos.resize(n_seq);

        vocab_id.n_tokens = this->params.vocab_n_tokens;
        vocab_id.hash     = this->params.vocab_hash;

        // a cache file that cannot be used is skipped, never fatal: the drafter then starts cold, as without the file
        if (!path_static.empty()) {
            std::string err;
            if (common_ngram_cache_load_file(path_static, ngram_cache_static, vocab_id, err) == COMMON_NGRAM_CACHE_LOAD_OK) {
                SPC_INF("static lookup cache loaded: %s (%zu n-grams)\n", path_static.c_str(), ngram_cache_static.size());
            } else {
                SPC_WRN("static lookup cache ignored: %s\n", err.c_str());
            }
        }

        auto status_dynamic = COMMON_NGRAM_CACHE_LOAD_MISSING;
        if (!path_dynamic.empty()) {
            std::string err;
            status_dynamic = common_ngram_cache_load_file(path_dynamic, ngram_cache_dynamic, vocab_id, err);
            if (status_dynamic == COMMON_NGRAM_CACHE_LOAD_OK) {
                SPC_INF("dynamic lookup cache loaded: %s (%zu n-grams)\n", path_dynamic.c_str(), ngram_cache_dynamic.size());
                file_present = true;
            } else {
                SPC_WRN("dynamic lookup cache ignored, starting cold: %s\n", err.c_str());
            }
        }

        if (this->params.save_dynamic) {
            if (path_dynamic.empty()) {
                SPC_WRN("%s", "--lookup-cache-dynamic-save needs --lookup-cache-dynamic, the dynamic lookup cache will not be saved\n");
            } else if (status_dynamic == COMMON_NGRAM_CACHE_LOAD_FOREIGN) {
                // the file is a valid cache of another model: saving would overwrite it with this model's n-grams
                SPC_WRN("dynamic lookup cache will not be saved: %s belongs to another model\n", path_dynamic.c_str());
            } else {
                save_dynamic  = true;
                n_max_dynamic = this->params.max_ngrams_dynamic > 0 ? (size_t) this->params.max_ngrams_dynamic : 0;

                const size_t n_loaded = ngram_cache_dynamic.size();
                common_ngram_cache_prune(ngram_cache_dynamic, n_max_dynamic);
                if (ngram_cache_dynamic.size() < n_loaded) {
                    SPC_INF("dynamic lookup cache pruned to %zu n-grams (max %zu)\n", ngram_cache_dynamic.size(), n_max_dynamic);
                }
                if (this->params.save_dynamic_interval > 0) {
                    try {
                        thread_ckpt = std::thread([this]() { checkpoint_worker(); });
                        SPC_INF("dynamic lookup cache will be saved to %s (every %d s and at shutdown)\n",
                                path_dynamic.c_str(), this->params.save_dynamic_interval);
                    } catch (const std::exception & e) {
                        SPC_WRN("dynamic lookup cache will be saved to %s at shutdown only, no checkpoint thread: %s\n",
                                path_dynamic.c_str(), e.what());
                    }
                } else {
                    SPC_INF("dynamic lookup cache will be saved to %s (at shutdown only)\n", path_dynamic.c_str());
                }
            }
        }

        t_last_sync_ms = ggml_time_ms();
    }

    // the destructor saves, a copy would save twice
    common_speculative_impl_ngram_cache(const common_speculative_impl_ngram_cache &) = delete;
    common_speculative_impl_ngram_cache & operator=(const common_speculative_impl_ngram_cache &) = delete;

    ~common_speculative_impl_ngram_cache() override {
        // The server destroys the speculative context on clean shutdown and when entering the sleeping state.
        // A process killed with SIGKILL, or with a second Ctrl+C, never gets here - hence the periodic
        // checkpoints from begin().
        //
        // Both steps block, which is fine here: a checkpoint in flight is completed by the worker (never
        // abandoned half-written: the file is replaced atomically or not at all), its result is collected,
        // and what is still unwritten goes out synchronously.
        checkpoint_stop();
        sync("shutdown");
    }

    // Write the dynamic cache back to its file, merged with whatever another process wrote there meanwhile.
    // Synchronous: only used at shutdown. Never throws: this runs from the destructor.
    void sync(const char * reason) noexcept {
        if (!save_dynamic) {
            return;
        }
        if (ngram_cache_delta.empty()) {
            t_last_sync_ms = ggml_time_ms();
            return;
        }

        try {
            const int64_t t_start_us = ggml_time_us();

            std::string err;
            if (common_ngram_cache_sync_file(params.lookup_cache_dynamic, ngram_cache_dynamic, ngram_cache_delta,
                        n_max_dynamic, vocab_id, err)) {
                SPC_INF("dynamic lookup cache saved (%s): %s, %zu n-grams, %.1f ms\n",
                        reason, params.lookup_cache_dynamic.c_str(), ngram_cache_dynamic.size(),
                        (ggml_time_us() - t_start_us) / 1000.0);
                file_present = true;
            } else {
                SPC_WRN("dynamic lookup cache not saved (%s), will retry later: %s\n", reason, err.c_str());
            }
        } catch (const std::exception & e) {
            SPC_WRN("dynamic lookup cache not saved (%s): %s\n", reason, e.what());
        } catch (...) {
            SPC_WRN("dynamic lookup cache not saved (%s)\n", reason);
        }

        t_last_sync_ms = ggml_time_ms();
    }

    // Body of thread_ckpt. Waits for a delta, writes it, hands the result back, frees retired caches.
    // Exits when ckpt_stop is set and nothing is queued, so a checkpoint in flight at shutdown completes.
    // Never lets an exception escape (that would terminate the process).
    void checkpoint_worker() noexcept {
        std::unique_lock<std::mutex> lock(mutex_ckpt);
        while (true) {
            cv_ckpt.wait(lock, [this]() {
                return ckpt_stop || ckpt_status == CKPT_QUEUED || ckpt_status == CKPT_RETIRE;
            });

            if (ckpt_status == CKPT_RETIRE) {
                // free the previous in-memory cache here, not on the inference thread: with up to
                // n_max_dynamic n-grams, each with its own map of continuations, that is millions of frees
                common_ngram_cache retired;
                std::swap(retired, ckpt_cache);
                ckpt_status = CKPT_IDLE;

                lock.unlock();
                retired.clear();
                lock.lock();
                continue;
            }

            if (ckpt_status == CKPT_QUEUED) {
                common_ngram_cache delta;
                std::swap(delta, ckpt_delta);
                ckpt_status = CKPT_WRITING;
                lock.unlock();

                // The inference thread keeps drafting from ngram_cache_dynamic meanwhile, so the worker cannot
                // read it. The delta itself serves as the fallback content for a missing or unusable file
                // (see common_ngram_cache_sync_file): the file is then recreated from what was learned since
                // the last save, and the in-memory cache follows the file when the result is collected.
                const int64_t t_start_us = ggml_time_us();

                common_ngram_cache cache;
                bool        ok        = false;
                bool        from_disk = false;
                std::string err;
                try {
                    cache = delta;
                    ok = common_ngram_cache_sync_file(params.lookup_cache_dynamic, cache, delta,
                            n_max_dynamic, vocab_id, err, &from_disk);
                } catch (const std::exception & e) {
                    ok  = false;
                    err = e.what();
                } catch (...) {
                    ok  = false;
                    err = "unknown error";
                }
                if (!ok) {
                    // the delta is untouched on failure and goes back to the inference thread
                    cache.clear();
                }

                lock.lock();
                std::swap(ckpt_cache, cache);
                std::swap(ckpt_delta, delta);
                ckpt_ok        = ok;
                ckpt_from_disk = from_disk;
                ckpt_err       = std::move(err);
                ckpt_t_us      = ggml_time_us() - t_start_us;
                ckpt_status    = CKPT_DONE;
                continue;
            }

            // ckpt_stop, nothing left to do
            return;
        }
    }

    // Take the result of a finished checkpoint. Called with mutex_ckpt held (or after the worker was joined).
    // Success: the in-memory cache becomes the file's content plus what was learned since the handover,
    // so it stays bounded by n_max_dynamic and picks up what other processes wrote to the file. The old
    // cache is left for the worker to free.
    // Failure: the delta is merged back into ngram_cache_delta and retried by the next sync.
    void checkpoint_collect() noexcept {
        if (ckpt_status != CKPT_DONE) {
            return;
        }

        if (ckpt_ok) {
            if (file_present && !ckpt_from_disk) {
                SPC_WRN("dynamic lookup cache %s was missing or unusable and has been recreated with the n-grams "
                        "learned since the last save, the earlier ones are gone\n", params.lookup_cache_dynamic.c_str());
            }
            file_present = true;

            try {
                // the live delta was learned after the handover, it is not in the file yet and stays a delta
                common_ngram_cache_merge(ckpt_cache, ngram_cache_delta);
                std::swap(ngram_cache_dynamic, ckpt_cache);
            } catch (const std::exception & e) {
                // out of memory during the merge: keep the current in-memory cache (it contains the delta
                // that was just written, and the live one), and let the worker free the half-merged copy
                SPC_WRN("dynamic lookup cache saved, but the in-memory cache was not refreshed from it: %s\n", e.what());
            }

            SPC_INF("dynamic lookup cache saved (checkpoint): %s, %zu n-grams, %.1f ms\n",
                    params.lookup_cache_dynamic.c_str(), ngram_cache_dynamic.size(), ckpt_t_us / 1000.0);

            ckpt_status = CKPT_RETIRE;
            cv_ckpt.notify_one();
            return;
        }

        SPC_WRN("dynamic lookup cache not saved (checkpoint), will retry later: %s\n", ckpt_err.c_str());

        // ckpt_delta is the whole last interval, ngram_cache_delta only what came after the handover:
        // add the small one to the big one and take that back
        try {
            common_ngram_cache_merge(ckpt_delta, ngram_cache_delta);
        } catch (const std::exception & e) {
            // out of memory part way through: a retry would count the merged part twice, so the rest of
            // the live delta (a few seconds of n-grams) is dropped instead
            SPC_WRN("dynamic lookup cache: n-grams learned since the last checkpoint dropped: %s\n", e.what());
        }
        std::swap(ngram_cache_delta, ckpt_delta);
        ckpt_delta.clear();
        ckpt_status = CKPT_IDLE;
    }

    // Called from begin(), on the inference thread. Collects a finished checkpoint and starts the next one
    // when it is due. Never blocks: if the worker happens to hold the mutex, this request is skipped.
    void checkpoint_poll() {
        if (!thread_ckpt.joinable()) {
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_ckpt, std::try_to_lock);
        if (!lock.owns_lock()) {
            return;
        }

        if (ckpt_status == CKPT_DONE) {
            checkpoint_collect();
        }

        if (ckpt_status == CKPT_IDLE && !ngram_cache_delta.empty() &&
                ggml_time_ms() - t_last_sync_ms >= (int64_t) params.save_dynamic_interval * 1000) {
            // ckpt_delta is empty in CKPT_IDLE, so this empties ngram_cache_delta
            std::swap(ckpt_delta, ngram_cache_delta);
            ckpt_status    = CKPT_QUEUED;
            t_last_sync_ms = ggml_time_ms();
            cv_ckpt.notify_one();
        }
    }

    // Stop the worker: a checkpoint in flight is completed and collected first. Blocks, only used at shutdown.
    void checkpoint_stop() noexcept {
        if (!thread_ckpt.joinable()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_ckpt);
            ckpt_stop = true;
        }
        cv_ckpt.notify_one();

        try {
            thread_ckpt.join();
        } catch (const std::exception & e) {
            SPC_WRN("dynamic lookup cache checkpoint thread not joined: %s\n", e.what());
            return;
        }

        // the worker is gone, the mutex is free; a retired cache is freed with this object
        std::lock_guard<std::mutex> lock(mutex_ckpt);
        checkpoint_collect();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        auto & sinfo = sinfos[seq_id];
        if (prompt.size() < sinfo.cache_size) {
            // The sequence got shorter (a reasoning block was dropped, or the slot serves another conversation).
            // The context cache can only be appended to (see common_ngram_cache_update), so it is rebuilt from
            // scratch at the next draft.
            sinfo.ngram_cache_context.clear();
            sinfo.cache_size = 0;
        }

        // checkpoint at request boundaries, so the in-memory cache is never swapped in the middle of a draft
        if (save_dynamic) {
            checkpoint_poll();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        inp.insert(inp.end(), prompt.begin(), prompt.end());
        inp.push_back(dparams.id_last);

        if (sinfo.cache_size < inp.size()) {
            // Pass the whole sequence together with the number of new tokens, so that the n-grams spanning
            // the boundary between the already cached tokens and the new ones are counted as well.
            const int nnew = (int) (inp.size() - sinfo.cache_size);

            common_ngram_cache_update(sinfo.ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, nnew, false);

            if (save_dynamic) {
                // the dynamic cache learns from every sequence; the delta is what sync() writes back
                common_ngram_cache_update(ngram_cache_dynamic, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, nnew, false);
                common_ngram_cache_update(ngram_cache_delta,   LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, nnew, false);
            }

            sinfo.cache_size = inp.size();
        }

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }

        // an empty result leaves the sequence drafting, so the next implementation gets its turn
        if (result.size() < (size_t) params.n_min) {
            result.clear();
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE: return "draft-mtp-adaptive";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) common_speculative_impl_ngram_cache::n_draft_for(spec->ngram_cache));
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) {
                return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;
            });

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    // an MTP draft's prompt pass is split by llama_decode into ubatches of the draft's own size, so a smaller draft ubatch
    // only shrinks its compute buffer (about 0.8 MiB per row for Flash-Next); a separate MTP draft file defaults to 2048 rows
    const int32_t ub_cap = params_spec.n_ubatch >= 0 ? params_spec.n_ubatch : (spec_mtp && has_draft ? 2048 : 0);
    if (spec_mtp && ub_cap > 0) {
        result.n_ubatch = std::min(params.n_ubatch == 0 ? params.n_batch : params.n_ubatch, ub_cap);
    }

    const char * lean = std::getenv("LLAMA_MTP_DRAFT_COMPUTE_LEAN");
    if (lean && std::strcmp(lean, "1") == 0 && spec_mtp && !has_draft && params.n_parallel == 1) {
        // Catch-up still needs the full logical batch; decode splits it into smaller microbatches.
        result.n_ubatch = std::min(params.n_ubatch == 0 ? params.n_batch : params.n_ubatch, 64);
    }

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    // draft KV cache types: an unset flag (GGML_TYPE_COUNT) inherits the main context's -ctk / -ctv, so the drafter
    // sees the same quantization error as the verifier (exception: an MTP drafter over a SJ-KVaRN trunk, below); an explicit --spec-draft-type-k/-v wins, K and V independently.
    // result.speculative.draft keeps the unresolved values so the draft context creation can log which were inherited
    result.cache_type_k  = params_spec.cache_type_k != GGML_TYPE_COUNT ? params_spec.cache_type_k : params.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v != GGML_TYPE_COUNT ? params_spec.cache_type_v : params.cache_type_v;
    // A SJ-KVaRN trunk cannot be inherited (see below) and would leave the drafter on the plain F16 cache, so an unset
    // draft type defaults to COMMON_SJKVARN_MTP_DRAFT_KV there (tq5_0 K / turbo4 V: the fused attention path at every
    // MTP width 1-8; tq6_0/tq6_0 at width 5 takes the slow f16-convert route unless GGML_SLOWKV is set).
    // The fused route needs D = 256. A built-in MTP head has that size, and so does a separate MTP file
    // (-md plus draft-mtp / draft-mtp-adaptive: that context is LLAMA_CONTEXT_TYPE_MTP). Flash-Next loads the
    // head as -md, so has_draft is true and the old !has_draft test left the draft cache at the plain f16
    // -ctk sjkvarnN stores. A non-MTP separate draft (EAGLE3 / DFlash, or -md with no MTP type) keeps the plain
    // cache, because its head size may not be 256. A trunk that is SJ-KVaRN on one side only fails the pair
    // check before this runs.
    const bool sj_kvarn_mtp_default = spec_mtp || !has_draft;
    if (sj_kvarn_mtp_default && params.sj_kvarn_bits_k > 0 && params_spec.cache_type_k == GGML_TYPE_COUNT) {
        result.cache_type_k = COMMON_SJKVARN_MTP_DRAFT_KV[0];
        result.speculative.draft.cache_type_sj_kvarn_default = true;
    }
    if (sj_kvarn_mtp_default && params.sj_kvarn_bits_v > 0 && params_spec.cache_type_v == GGML_TYPE_COUNT) {
        result.cache_type_v = COMMON_SJKVARN_MTP_DRAFT_KV[1];
        result.speculative.draft.cache_type_sj_kvarn_default = true;
    }
    // A quantized trunk must not leave an unset draft cache at f16 (q8_0/q8_0 trunk -> q8_0 draft).
    // Inheritance above already copies that trunk type. This is the backstop if a later default wrote f16.
    // Explicit -ctkd/-ctvd f16 still wins (the flag is not GGML_TYPE_COUNT). SJ-KVaRN's plain trunk type is
    // f16, and ggml_is_quantized(F16) is false, so this does not replace the tq5_0/turbo4 default.
    if (params_spec.cache_type_k == GGML_TYPE_COUNT && result.cache_type_k == GGML_TYPE_F16 && ggml_is_quantized(params.cache_type_k)) {
        result.cache_type_k = params.cache_type_k;
    }
    if (params_spec.cache_type_v == GGML_TYPE_COUNT && result.cache_type_v == GGML_TYPE_F16 && ggml_is_quantized(params.cache_type_v)) {
        result.cache_type_v = params.cache_type_v;
    }
    // The first block-streaming implementation owns only the target cache.
    // MTP keeps its ordinary cache until both contexts can share one pool.
    result.kv_stream_arena_mib = 0;
    if (params_spec.sj_kvarn) {
        if (!spec_mtp || has_draft || params.n_parallel != 1 || params.sj_kvarn_bits_k == 0 || params.sj_kvarn_bits_v == 0) {
            throw std::invalid_argument("--spec-draft-sjkvarn requires single-sequence MTP and a SJ-KVaRN trunk");
        }
        if (params_spec.cache_type_k != GGML_TYPE_COUNT || params_spec.cache_type_v != GGML_TYPE_COUNT) {
            throw std::invalid_argument("--spec-draft-sjkvarn cannot be combined with -ctkd/-ctvd");
        }
        if (params_spec.n_max < 0 || params.sj_kvarn_tail < (uint32_t) params_spec.n_max + 1) {
            throw std::invalid_argument("--spec-draft-sjkvarn requires a tail larger than the maximum draft width");
        }
        // opt-in: the drafter keeps the trunk's SJ-KVaRN regions instead of the tq5_0/turbo4 default above
        result.cache_type_k = params.sj_kvarn_staging_type;
        result.cache_type_v = params.sj_kvarn_staging_type;
        result.speculative.draft.cache_type_sj_kvarn_default = false;
    } else {
        // the draft/MTP context shares cells with the target (or is a separate small cache): never SJ-KVaRN.
        // A SJ-KVaRN trunk (-ctk/-ctv sj_kvarnN) leaves cache_type_k/v at the plain type -ctk/-ctv sj_kvarnN set (F16),
        // so an unset draft type is resolved above to tq5_0/turbo4 instead of that plain cache.
        result.sj_kvarn_bits_k  = 0;
        result.sj_kvarn_bits_v  = 0;
        result.sj_kvarn_tail_max = 0;
        result.sj_kvarn_flush_chunk = 0;
    }
    // The nextn layer has no trunk edge tier, and no WUSH transform (the file covers trunk layers only).
    result.sj_kvarn_edge_layers = 0;
    result.sj_kvarn_transform.clear();
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // the MoE cache is only used by the target context
    result.moe_cache_size = 0;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    // chained MTP drafting outputs logits for every chain step in one decode
    const char * qwen_chain = std::getenv("LLAMA_MTP_QWEN4EXP_CHAIN");
    if (common_speculative_mtp_chain_enabled(params_spec) || (qwen_chain && std::strcmp(qwen_chain, "1") == 0)) {
        const int32_t per_seq = std::max(1, params_spec.n_max);
        result.n_outputs_max = std::max(result.n_outputs_max, params.n_parallel * per_seq);
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end() ||
                          std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) != params.speculative.types.end();
    GGML_ASSERT(has_draft || spec_mtp);

    const char * shared_experts = getenv("GGML_MOE_CACHE_MTP_SHARED");
    if (has_draft && spec_mtp && shared_experts && strcmp(shared_experts, "1") == 0) {
        if (!params.tensor_buft_overrides.empty() && params.tensor_buft_overrides.back().pattern == nullptr) {
            params.tensor_buft_overrides.pop_back();
        }
        params.tensor_buft_overrides.insert(params.tensor_buft_overrides.begin(), llm_ffn_exps_cpu_override());
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
        LOG_INF("%s: MTP expert weights use host storage and the target expert cache\n", __func__);
    }

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        if (!has_draft) {
            common_speculative_mtp_cparams(params, cparams);
            if (cparams.draft_attn_window > 0) {
                LOG_INF("%s: MTP draft attention window = %d, sink = %d\n",
                        __func__, cparams.draft_attn_window, cparams.draft_attn_sink);
            }
        }
        if (!params.speculative.draft.vocab_map.empty() && params.speculative.draft.vocab_map != "none") {
            cparams.draft_vocab_map = params.speculative.draft.vocab_map.c_str();
            cparams.draft_vocab_hot = params.speculative.draft.vocab_hot;
            LOG_INF("%s: MTP draft context uses the draft-only vocabulary shortlist '%s'\n", __func__, cparams.draft_vocab_map);
            if (cparams.draft_vocab_hot > 0) {
                LOG_INF("%s: the last %d entries of the shortlist are adaptive hot slots\n", __func__, cparams.draft_vocab_hot);
            }
        }
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // SJ-KVaRN trunk with --kv-unified and several slots: every target sequence may address the whole context
    // (one shared record pool of n_ctx - sink - tail tokens, plus a per-sequence sink and ring), so the slots
    // together can hold up to n_ctx + n_seq*(sink + ring) tokens. A plain unified draft cache of n_ctx cells would
    // run out first (decode: failed to find a memory slot); it gets that many cells instead, still unified
    // (one n_ctx stream per sequence would cost n_seq*n_ctx cells and a matching compute buffer).
    if (cparams.kv_unified && cparams.n_seq_max > 1 && cparams.sj_kvarn_bits_k == 0 &&
        llama_sj_kvarn_sealed_end(ctx_tgt) >= 0 && llama_n_ctx_seq(ctx_tgt) == llama_n_ctx(ctx_tgt)) {
        const uint32_t unsealed = (uint32_t) std::max(0, llama_sj_kvarn_seq_unsealed_max(ctx_tgt));
        const uint32_t n_ctx_draft = GGML_PAD(llama_n_ctx(ctx_tgt) + cparams.n_seq_max*unsealed, 256);
        LOG_INF("%s: SJ-KVaRN shared pool: unified draft KV cache of %u cells (n_ctx %u + %u slots x %u sink + ring rows)\n",
                __func__, n_ctx_draft, llama_n_ctx(ctx_tgt), cparams.n_seq_max, unsealed);
        cparams.n_ctx = n_ctx_draft;
    }

    // params comes from common_base_params_to_speculative: cache_type_k/v hold the resolved types,
    // speculative.draft.cache_type_k/v are GGML_TYPE_COUNT when the flag was not given
    const char * unset_src = params.speculative.draft.cache_type_sj_kvarn_default ? "SJ-KVaRN trunk default" : nullptr;
    LOG_INF("%s: draft KV cache: K = %s (%s), V = %s (%s)\n", __func__,
            ggml_type_name(cparams.type_k), params.speculative.draft.cache_type_k == GGML_TYPE_COUNT ? (unset_src ? unset_src : "inherited from -ctk") : "explicit -ctkd",
            ggml_type_name(cparams.type_v), params.speculative.draft.cache_type_v == GGML_TYPE_COUNT ? (unset_src ? unset_src : "inherited from -ctv") : "explicit -ctvd");

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        if (spec_mtp) {
            const char * borrow = getenv("LLAMA_MTP_BORROW_WEIGHTS");
            if (!borrow || strcmp(borrow, "0") != 0) { mparams.tensor_borrow_from = model_tgt; }
        }
        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

void common_speculative_observe(common_speculative_init_result * spec, const llama_tokens & toks) {
    if (spec == nullptr || toks.empty()) {
        return;
    }
    llama_context * ctx = spec->context();
    if (ctx != nullptr) {
        llama_draft_vocab_observe(ctx, toks.data(), toks.size());
    }
}

int32_t common_speculative_refresh(common_speculative_init_result * spec) {
    if (spec == nullptr) {
        return 0;
    }
    llama_context * ctx = spec->context();
    return ctx == nullptr ? 0 : llama_draft_vocab_refresh(ctx);
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) && params.draft.ctx_dft != nullptr;
        bool has_draft_mtp    = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP))    && params.draft.ctx_dft != nullptr;
        bool has_draft_mtp_adaptive = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE)) && params.draft.ctx_dft != nullptr;
        bool has_draft_dflash = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) && params.draft.ctx_dft != nullptr;
        bool has_draft_dspark = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) && params.draft.ctx_dft != nullptr;

        // If --dflash or --eagle3 flags are set, enable the corresponding type
        if (!has_draft_dflash && params.draft.dflash && params.draft.ctx_dft != nullptr) {
            has_draft_dflash = true;
        }
        if (!has_draft_eagle3 && params.draft.eagle3 && params.draft.ctx_dft != nullptr) {
            has_draft_eagle3 = true;
        }


        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 12);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_draft_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_mtp_adaptive) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, params));
        }
        if (has_draft_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params));
        }
        if (has_draft_dspark) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq, /*adaptive=*/ false));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq, /*adaptive=*/ true));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                const uint16_t n_draft = common_speculative_impl_ngram_cache::n_draft_for(config.params.ngram_cache);

                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(config.params, n_seq, n_draft));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

// The drafter implementations read the incoming target batch through the flat llama_batch layout
// (token/pos/seq_id/logits arrays, one contiguous embd block). Upstream moved the server to
// common_batch (llama_batch_ext); this adapter renders a common_batch into that layout so the
// drafter code paths stay byte-for-byte the ones the v0.4/v0.5 gates measured.
namespace {
struct spec_batch_view {
    std::vector<llama_token>    token;
    std::vector<float>          embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_flat;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t>         logits;

    llama_batch render(const common_batch & cb) {
        const int32_t n = cb.size();
        const bool has_tok  = cb.has_token();
        const bool has_embd = cb.has_embd();

        token.resize(has_tok ? n : 0);
        pos.resize(n);
        n_seq_id.resize(n);
        logits.resize(n);
        seq_id.resize(n);

        size_t n_seq_total = 0;
        size_t n_embd = 0;
        for (int32_t i = 0; i < n; ++i) {
            n_seq_total += 1 + cb.tokens[i].seq_ids_extra.size();
            if (has_embd) {
                n_embd = std::max(n_embd, cb.tokens[i].embd.n_embd);
            }
        }
        seq_flat.resize(n_seq_total);
        embd.resize(has_embd ? (size_t) n * n_embd : 0);

        size_t o = 0;
        for (int32_t i = 0; i < n; ++i) {
            const auto & t = cb.tokens[i];
            if (has_tok) {
                token[i] = t.id;
            }
            if (has_embd && t.embd.data != nullptr) {
                std::memcpy(embd.data() + (size_t) i * n_embd, t.embd.data, t.embd.n_embd * sizeof(float));
            }
            pos[i]      = t.pos[0];
            logits[i]   = t.output ? 1 : 0;
            n_seq_id[i] = 1 + (int32_t) t.seq_ids_extra.size();
            seq_id[i]   = seq_flat.data() + o;
            seq_flat[o++] = t.seq_id;
            for (const auto s : t.seq_ids_extra) {
                seq_flat[o++] = s;
            }
        }

        llama_batch b = {};
        b.n_tokens = n;
        b.token    = has_tok  ? token.data() : nullptr;
        b.embd     = has_embd ? embd.data()  : nullptr;
        b.pos      = pos.data();
        b.n_seq_id = n_seq_id.data();
        b.seq_id   = seq_id.data();
        b.logits   = logits.data();
        return b;
    }
};
}

bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    static thread_local spec_batch_view view;
    const llama_batch b = view.render(batch);

    for (auto & impl : spec->impls) {
        result = result && impl->process(b);
    }

    return result;
}

bool common_speculative_supports_process_range(const common_speculative * spec) {
    return spec && !spec->impls.empty() && std::all_of(spec->impls.begin(), spec->impls.end(), [](const auto & impl) {
        return impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;
    });
}

bool common_speculative_handoff_enabled(const common_speculative * spec) {
    if (!common_speculative_supports_process_range(spec)) {
        return false;
    }
    return std::all_of(spec->impls.begin(), spec->impls.end(), [](const auto & impl) {
        const auto * mtp = dynamic_cast<const common_speculative_impl_draft_mtp *>(impl.get());
        return mtp && mtp->handoff;
    });
}

bool common_speculative_process_range(common_speculative * spec, const common_batch & batch, int32_t first, int32_t count) {
    if (!common_speculative_supports_process_range(spec) || first < 0 || count < 0 || first > batch.size() - count || !batch.has_token() || batch.has_embd()) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    static thread_local spec_batch_view view;
    llama_batch b = view.render(batch);
    b.n_tokens = count;
    b.token += first;
    b.pos += first;
    b.n_seq_id += first;
    b.seq_id += first;
    b.logits += first;
    for (auto & impl : spec->impls) {
        if (!impl->process_range(b, first)) {
            return false;
        }
    }
    return true;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);

                        // trim the candidates only if the drafter produced them (n-gram drafters do not)
                        if (dp.result_q && !dp.result_q->empty()) {
                            dp.result_q->resize(dp.n_max);
                        }
                    }
                }

                if (dp.conf_p1 && dp.conf_p1->size() > result.size()) {
                    dp.conf_p1->resize(result.size());
                }
                if (dp.conf_p2 && dp.conf_p2->size() > result.size()) {
                    dp.conf_p2->resize(result.size());
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_note_target_conf(common_speculative * spec, llama_seq_id seq_id, const float * p, size_t n) {
    if (spec == nullptr || p == nullptr || n == 0) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->note_target_conf(seq_id, p, n);
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted, int32_t n_accepted_ctrl) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->n_accepted_ctrl = n_accepted_ctrl;
        impl->accept(seq_id, n_accepted, false);
        impl->n_accepted_ctrl = -1;
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

bool common_speculative_get_resume_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    data.clear();
    if (spec) { for (auto & impl : spec->impls) { if (impl->get_resume_state(seq_id, data)) { return true; } } }
    return false;
}

bool common_speculative_set_resume_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    bool restored = false;
    if (spec) { for (auto & impl : spec->impls) { restored = impl->set_resume_state(seq_id, data) || restored; } }
    return restored;
}

bool common_speculative_restore_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    std::vector<uint8_t> restored;
    if (data.empty()) {
        return !common_speculative_get_state(spec, seq_id, restored);
    }
    common_speculative_set_state(spec, seq_id, data);
    return common_speculative_get_state(spec, seq_id, restored) && restored == data;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}

std::vector<int32_t> common_speculative_row_budget(const std::vector<int32_t> & requested,
        const std::vector<int32_t> & retained, int32_t capacity, size_t first,
        const std::vector<std::vector<double>> & acceptance) {
    GGML_ASSERT(requested.size() == retained.size());
    GGML_ASSERT(acceptance.empty() || acceptance.size() == requested.size());
    std::vector<int32_t> budget(requested.size(), -1);
    if (requested.empty()) {
        return budget;
    }
    if (requested.size() == 1 || capacity <= 0) {
        for (size_t i = 0; i < requested.size(); ++i) {
            budget[i] = retained[i] >= 0 ? retained[i] : std::max(0, requested[i]);
        }
        return budget;
    }
    first %= requested.size();
    // A replay made in the solo path can exceed the shared budget. Serve it alone without truncation.
    if (retained[first] >= capacity) {
        budget[first] = retained[first];
        return budget;
    }
    int32_t left = capacity;
    for (size_t k = 0; k < requested.size(); ++k) {
        const size_t i = (first + k) % requested.size();
        const int32_t rows = retained[i] >= 0 ? retained[i] : 0;
        if (rows < left) {
            budget[i] = rows;
            left -= rows + 1;
        }
    }
    while (left > 0) {
        size_t best = requested.size();
        double best_value = -1.0;
        for (size_t k = 0; k < requested.size(); ++k) {
            const size_t i = (first + k) % requested.size();
            if (budget[i] < 0 || retained[i] >= 0 || budget[i] >= requested[i]) {
                continue;
            }
            double value = 1.0;
            for (int32_t d = 0; d <= budget[i]; ++d) {
                const double p = !acceptance.empty() && (size_t) d < acceptance[i].size() ? acceptance[i][d] : 0.85;
                value *= std::isfinite(p) ? std::max(0.0, std::min(1.0, p)) : 0.85;
            }
            if (value > best_value || (value == best_value && best < requested.size() && budget[i] < budget[best])) {
                best = i;
                best_value = value;
            }
        }
        if (best == requested.size() || best_value <= 0.0) {
            break;
        }
        ++budget[best];
        --left;
    }
    return budget;
}
