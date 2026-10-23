#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "../src/llama-ext.h" // staging sampler APIs

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define COMMON_TOPK_AVX2 1
#else
#define COMMON_TOPK_AVX2 0
#endif

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

// Inspect only complete blocks, stopping before the first block that can alter
// the heap. The caller still visits every winning candidate in original ID order.
#if COMMON_TOPK_AVX2
__attribute__((target("avx2")))
static int32_t common_sampler_topk_skip_avx2(const float * logits, int32_t i, int32_t end, float threshold) {
    const __m256 t = _mm256_set1_ps(threshold);
    while (end - i >= 32) {
        const __m256 a = _mm256_cmp_ps(_mm256_loadu_ps(logits + i), t, _CMP_GT_OQ);
        const __m256 b = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 8), t, _CMP_GT_OQ);
        const __m256 c = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 16), t, _CMP_GT_OQ);
        const __m256 d = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 24), t, _CMP_GT_OQ);
        if (_mm256_movemask_ps(_mm256_or_ps(_mm256_or_ps(a, b), _mm256_or_ps(c, d)))) {
            break;
        }
        i += 32;
    }
    return i;
}
#endif

// The same result as llama_sampler_top_k (std::partial_sort with comp a.logit > b.logit over the full
// {id, logit, 0} array), ids and order included for equal logits: the heap select is replayed step by step
// (libstdc++ __heap_select / __adjust_heap / __push_heap). Banned ids are sorted, unique and in range;
// they have -inf in the initial heap and cannot replace its top during the subsequent scan.
static void common_sampler_topk_exact_impl(
        const float * logits, int32_t n, int32_t k, std::vector<llama_token_data> & cur,
        const llama_token * banned, size_t n_banned) {
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };
    cur.resize(k);
    llama_token_data * h = cur.data();
    for (int32_t i = 0; i < k; ++i) {
        h[i] = llama_token_data{i, logits[i], 0.0f};
    }
    size_t next_ban = 0;
    while (next_ban < n_banned && banned[next_ban] < k) {
        h[banned[next_ban++]].logit = -INFINITY;
    }
    std::make_heap(h, h + k, comp);

    // replace the heap top by v and restore the heap, as libstdc++ __pop_heap(first, middle, i) does
    const auto replace_top = [&](const llama_token_data & v) {
        const int64_t len = k;
        int64_t hole = 0;
        int64_t child = 0;
        while (child < (len - 1) / 2) {
            child = 2 * (child + 1);
            if (comp(h[child], h[child - 1])) {
                child--;
            }
            h[hole] = h[child];
            hole = child;
        }
        if ((len & 1) == 0 && child == (len - 2) / 2) {
            child = 2 * (child + 1);
            h[hole] = h[child - 1];
            hole = child - 1;
        }
        int64_t parent = (hole - 1) / 2;
        while (hole > 0 && comp(h[parent], v)) {
            h[hole] = h[parent];
            hole = parent;
            parent = (hole - 1) / 2;
        }
        h[hole] = v;
    };

#if COMMON_TOPK_AVX2
    static const bool avx2 = [] {
        const char * enabled = std::getenv("LLAMA_BOOKKEEP_TOPK_AVX2");
        return (!enabled || std::atoi(enabled) != 0) && __builtin_cpu_supports("avx2");  // default on (owner 2026-10-22); 0 turns it off
    }();
#endif
    // A block is skipped only when none of its logits beats the current heap top.
    // Splitting at bans changes no heap operations: -inf never beats that top, even if it is NaN.
    const auto scan_range = [&](int32_t i, const int32_t end) {
        constexpr int32_t B = 32;
        while (i < end) {
#if COMMON_TOPK_AVX2
            if (avx2) {
                i = common_sampler_topk_skip_avx2(logits, i, end, h[0].logit);
                if (i == end) { break; }
            }
#endif
            if (end - i >= B) {
#if COMMON_TOPK_AVX2
                if (!avx2) {
#endif
                const float t = h[0].logit;
                int c = 0;
                for (int32_t j = 0; j < B; ++j) {
                    c += logits[i + j] > t ? 1 : 0;
                }
                if (c == 0) {
                    i += B;
                    continue;
                }
#if COMMON_TOPK_AVX2
                }
#endif
                for (const int32_t block_end = i + B; i < block_end; ++i) {
                    if (logits[i] > h[0].logit) {
                        replace_top(llama_token_data{i, logits[i], 0.0f});
                    }
                }
            } else {
                if (logits[i] > h[0].logit) {
                    replace_top(llama_token_data{i, logits[i], 0.0f});
                }
                ++i;
            }
        }
    };
    int32_t begin = k;
    for (; next_ban < n_banned; ++next_ban) {
        scan_range(begin, banned[next_ban]);
        begin = banned[next_ban] + 1;
    }
    scan_range(begin, n);
    std::sort_heap(h, h + k, comp);
}

void common_sampler_topk_exact(const float * logits, int32_t n, int32_t k, std::vector<llama_token_data> & cur) {
    common_sampler_topk_exact_impl(logits, n, k, cur, nullptr, 0);
}

bool common_sampler_topk_biases_supported(const std::vector<llama_logit_bias> & biases) {
    std::unordered_map<llama_token, float> seen;
    for (const auto & lb : biases) {
        const auto entry = seen.emplace(lb.token, lb.bias);
        if (!entry.second && (entry.first->second != -INFINITY || lb.bias != -INFINITY)) {
            return false;
        }
    }
    return true;
}

bool common_sampler_topk_exact_biased(
        const float * logits, int32_t n, int32_t k, const std::vector<llama_logit_bias> & biases,
        std::vector<float> & scratch, std::vector<llama_token_data> & cur) {
    // Repeated bans are idempotent only when their inputs do not produce a NaN.
    bool only_bans = true;
    for (const auto & lb : biases) {
        only_bans = only_bans && lb.bias == -INFINITY;
        if (lb.token >= 0 && lb.token < n && lb.bias == -INFINITY &&
                !std::isfinite(logits[lb.token]) && logits[lb.token] != -INFINITY) {
            return false;
        }
    }
    if (only_bans) {
        std::vector<llama_token> banned;
        banned.reserve(biases.size());
        for (const auto & lb : biases) {
            if (lb.token >= 0 && lb.token < n) {
                banned.push_back(lb.token);
            }
        }
        std::sort(banned.begin(), banned.end());
        banned.erase(std::unique(banned.begin(), banned.end()), banned.end());
        common_sampler_topk_exact_impl(logits, n, k, cur, banned.data(), banned.size());
    } else {
        scratch.resize(n);
        std::memcpy(scratch.data(), logits, n * sizeof(float));
        for (const auto & lb : biases) {
            if (lb.token >= 0 && lb.token < n) {
                scratch[lb.token] += lb.bias;
            }
        }
        common_sampler_topk_exact(scratch.data(), n, k, cur);
    }
    // The chain still applies its bias sampler to the compact candidates.
    for (auto & c : cur) {
        c.logit = logits[c.id];
    }
    return true;
}

static bool common_sampler_output_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_BOOKKEEP_OUTPUT");
        return !e || atoi(e) != 0;  // default on (owner 2026-10-22); 0 turns it off
    }();
    return enabled;
}

static bool common_sampler_bookkeep_timing() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_BOOKKEEP_TIMING");
        return e && atoi(e) != 0;
    }();
    return enabled;
}

struct pq_record {
    int64_t at;
    llama_token token;
    double kl;
    double lr;
    bool rescued;
    llama_token y;
    double py;
    bool ey;
    bool eo;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    void reset() {
        prev.clear();
        sampled_backend = false;

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx, const llama_sampler_output * output = nullptr) {
        const bool timing = common_sampler_bookkeep_timing();
        const auto prepare_timer = common_time_meas(prep_perf.prepare_us, !timing);
        if (timing) { ++prep_perf.prepare_calls; }
        const float *       sampled_probs  = output ? output->probs      : llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = output ? output->logits     : llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = output ? output->candidates : llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = output ? output->probs_count : llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = output ? output->logits_count : llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = output ? output->raw_logits : llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            if (fused_top_k > 0 && fused_top_k < n_vocab && grmr == nullptr && rbudget == nullptr) {
                const auto select_timer = common_time_meas(prep_perf.fused_topk_us, !timing);
                if (timing) { ++prep_perf.fused_topk_calls; }
                bool fused;
                if (fused_topk_bias.empty()) {
                    common_sampler_topk_exact(logits, n_vocab, fused_top_k, cur);
                    fused = true;
                } else {
                    fused = common_sampler_topk_exact_biased(logits, n_vocab, fused_top_k, fused_topk_bias, fused_topk_scratch, cur);
                }
                if (fused) {
                    cur_p = { cur.data(), cur.size(), -1, true };
                    return;
                }
            }
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    // LLAMA_SAMPLER_TOPK_FUSED=1: exact leading top-k (0 = off). LLAMA_BOOKKEEP_SAMPLER=1 also permits logit bias.
    int32_t fused_top_k = 0;
    std::vector<llama_logit_bias> fused_topk_bias;
    std::vector<float> fused_topk_scratch;

    // Status of the most recent sample, refreshed before the p/q acceptance check.
    bool sampled_backend = false;
    common_sampler_prep_perf prep_perf;


    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
    llama_sampler * greedy_backend = nullptr;
    // true when greedy_backend is [logit-bias, greedy] rather than a bare [greedy]: a biased chain really
    // runs a backend graph, so it has to have been offloaded before it can be trusted
    bool greedy_backend_biased = false;

    // exact p/q verification: its own generator (independent of the chain's dist sampler, whose draws the
    // proof needs untouched) and per-request counters
    uint32_t pq_seed = 0x9e3779b9u;
    std::mt19937 pq_rng{pq_seed};
    common_sampler_pq_stats pq_stats = {};
    std::vector<pq_record> pq_pending = {};

    // target top-1 probability (request-matched distribution, after the sampler chain) of every position the last
    // sample_and_accept_n_pq call looked at; a free by-product used by the CDEPTH draft-depth policy. -1 = unknown
    std::vector<float> top1_trace;
    int32_t first_rescue = -1; // [MBFIX] position of the last pq verify call's first mode-biased rescue (-1 = none)
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    std::vector<llama_logit_bias> merged_bias = params.logit_bias;

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged_bias.push_back({ suppress[i], -INFINITY });
        }

        if (!merged_bias.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged_bias.size(), merged_bias.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
    };

    {
        // the p/q generator must not replay the chain's dist draws: derive a different stream
        uint32_t seed = params.seed;
        if (seed == LLAMA_DEFAULT_SEED) {
            seed = std::random_device{}();
        }
        result->pq_seed = seed ^ 0x5bd1e995u;
        result->pq_rng.seed(result->pq_seed);
    }

    {
        static const bool topk_fused_env = [] {
            // default on (exact: same kept set and order as the unfused top-k); LLAMA_SAMPLER_TOPK_FUSED=0 turns it off
            const char * e = getenv("LLAMA_SAMPLER_TOPK_FUSED");
            return e == nullptr || atoi(e) != 0;
        }();
        static const bool bookkeep_sampler_env = [] {
            const char * e = getenv("LLAMA_BOOKKEEP_SAMPLER");
            return !e || atoi(e) != 0;  // default on (owner 2026-10-22); 0 turns it off
        }();
        // LLAMA_HOSTCPU_TOPK_SKIP_EMPTY (default on): disabled samplers ("?penalties", "?dry", "?top-n-sigma", ...: their
        // apply and accept do nothing) may sit before top-k. The default sampler order puts three of them
        // first, which kept the fused top-k off for every plain temp/top-k/top-p request.
        // Default on (exact: the skipped samplers do nothing); LLAMA_HOSTCPU_TOPK_SKIP_EMPTY=0 turns it off.
        static const bool hostcpu_skip_empty_env = [] {
            const char * e = getenv("LLAMA_HOSTCPU_TOPK_SKIP_EMPTY");
            return e == nullptr || atoi(e) != 0;
        }();
        if (topk_fused_env && params.top_k > 0 && params.top_k <= 128) {
            const int bias_idx = merged_bias.empty() ? 0 : 1;
            int top_k_idx = bias_idx;
            if (hostcpu_skip_empty_env) {
                while (top_k_idx < llama_sampler_chain_n(chain) &&
                        llama_sampler_name(llama_sampler_chain_get(chain, top_k_idx))[0] == '?') {
                    ++top_k_idx;
                }
            }
            // Repeated bans are exact; other duplicate biases differ after a candidate shuffle.
            if ((bias_idx == 0 || bookkeep_sampler_env) && common_sampler_topk_biases_supported(merged_bias) &&
                    llama_sampler_chain_n(chain) > top_k_idx &&
                    (bias_idx == 0 || strcmp(llama_sampler_name(llama_sampler_chain_get(chain, 0)), "logit-bias") == 0) &&
                    strcmp(llama_sampler_name(llama_sampler_chain_get(chain, top_k_idx)), "top-k") == 0) {
                result->fused_top_k = params.top_k;
                result->fused_topk_bias = std::move(merged_bias);
            }
        }
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);
    llama_sampler_free(gsmpl->greedy_backend);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
    if (gsmpl->params.spec_lossy) {
        gsmpl->pq_rng.seed(gsmpl->pq_seed);
        gsmpl->pq_stats = {};
    }
    gsmpl->pq_pending.clear();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
    };
    if (gsmpl->params.spec_lossy) {
        result->cur_p.data = gsmpl->cur_p.data ? result->cur.data() + (gsmpl->cur_p.data - gsmpl->cur.data()) : nullptr;
    }
    result->pq_pending = gsmpl->pq_pending;
    result->t_total_us = gsmpl->t_total_us;
    result->fused_top_k = gsmpl->fused_top_k;
    result->fused_topk_bias = gsmpl->fused_topk_bias;
    result->sampled_backend = gsmpl->sampled_backend;
    result->prep_perf = gsmpl->prep_perf;
    result->pq_seed    = gsmpl->pq_seed;
    result->pq_rng     = gsmpl->pq_rng;   // a restored clone must replay the same accept decisions
    result->pq_stats   = gsmpl->pq_stats;
    return result;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() + (src->params.spec_lossy ? src->cur_p.data - src->cur.data() : 0) : nullptr;
    dst->rng        = src->rng;
    if (src->params.spec_lossy) {
        dst->pq_seed  = src->pq_seed;
        dst->pq_rng   = src->pq_rng;
        dst->pq_stats = src->pq_stats;
    }
    dst->pq_pending = src->pq_pending;
    dst->t_total_us = src->t_total_us;
    dst->fused_top_k = src->fused_top_k;
    dst->fused_topk_bias = src->fused_topk_bias;
    dst->sampled_backend = src->sampled_backend;
    dst->prep_perf = src->prep_perf;
}

common_sampler_prep_perf common_sampler_get_prep_perf(const common_sampler * gsmpl) {
    return gsmpl ? gsmpl->prep_perf : common_sampler_prep_perf{};
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

bool common_sampler_supports_greedy_backend(const common_params_sampling & params) {
    // logit biases (including the EOG biases that a request with ignore_eos carries) are not listed here:
    // they are applied exactly on the backend, by the same logit-bias sampler the CPU chain uses, in front
    // of the argmax - see common_sampler_get_greedy_backend()
    if (!(params.temp <= 0.0f) || params.dynatemp_range != 0.0f || params.mirostat != 0 ||
            params.n_probs != 0 || !common_grammar_value(params.grammar).empty()) {
        return false;
    }

    bool temperature = false;
    for (const auto type : params.samplers) {
        switch (type) {
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                temperature = true;
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
                // Active truncation can reorder tied maxima before temperature runs.
                if (params.top_k > 0) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (!(params.top_p >= 1.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (!(params.min_p <= 0.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (!(params.typ_p >= 1.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (!(params.top_n_sigma < 0.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (!(params.xtc_probability <= 0.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (params.dry_multiplier != 0.0f && params.dry_penalty_last_n != 0) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (params.penalty_last_n != 0 && (params.penalty_repeat != 1.0f ||
                        params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) { return false; }
                break;
            default:
                return false;
        }
    }
    return temperature;
}

// the biases the CPU chain applies before its greedy argmax: the request's logit_bias (which is where the
// server puts the EOG biases of ignore_eos) followed by the model's suppress tokens at -INFINITY. The set
// is exactly the one common_sampler_init() builds for the CPU chain.
//
// Repeated tokens are summed here. The CPU logit-bias sampler adds every entry to the candidate's logit,
// while the backend one writes the entries into a bias row with ggml_set_rows (last write wins) and adds
// the row once, so without folding a repeated token would silently lose all but its last bias. With it the
// two paths apply the same total bias (a repeated finite bias can still round differently, since the CPU
// adds it in steps). Suppressed ids stay -INFINITY under any sum with a finite bias, and a token that is
// both biased and suppressed is suppressed on both paths.
std::vector<llama_logit_bias> common_sampler_greedy_biases(
        const common_params_sampling & params, const struct llama_vocab * vocab) {
    std::vector<llama_logit_bias> merged;
    std::unordered_map<llama_token, size_t> pos;

    auto add = [&](llama_token token, float bias) {
        const auto it = pos.find(token);
        if (it == pos.end()) {
            pos[token] = merged.size();
            merged.push_back({ token, bias });
        } else {
            merged[it->second].bias += bias;
        }
    };

    for (const auto & lb : params.logit_bias) {
        add(lb.token, lb.bias);
    }

    if (vocab) {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            add(suppress[i], -INFINITY);
        }
    }

    return merged;
}

struct llama_sampler * common_sampler_get_greedy_backend(struct common_sampler * gsmpl, const struct llama_model * model) {
    if (!gsmpl || gsmpl->grmr || gsmpl->rbudget || !common_sampler_supports_greedy_backend(gsmpl->params)) {
        return nullptr;
    }
    if (!gsmpl->greedy_backend) {
        const struct llama_vocab * vocab = llama_model_get_vocab(model);

        auto params = llama_sampler_chain_default_params();
        params.no_perf = gsmpl->params.no_perf;
        gsmpl->greedy_backend = llama_sampler_chain_init(params);

        // suppressed / biased ids are forced to -INFINITY in the logit rows before the argmax, which is
        // exactly what the CPU chain does; both sides break ties on the lowest token id (the CPU greedy
        // sampler keeps the first maximum of a candidate list built in id order, and ggml_argmax keeps
        // the lowest column index), so the selected token is identical
        const auto biases = common_sampler_greedy_biases(gsmpl->params, vocab);
        if (!biases.empty()) {
            llama_sampler_chain_add(gsmpl->greedy_backend,
                    llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), biases.size(), biases.data()));
            gsmpl->greedy_backend_biased = true;
        }

        llama_sampler_chain_add(gsmpl->greedy_backend, llama_sampler_init_greedy());
    }
    return gsmpl->greedy_backend;
}

bool common_sampler_greedy_backend_ready(const struct common_sampler * gsmpl) {
    if (!gsmpl || !gsmpl->greedy_backend) {
        return false;
    }
    // a bare [greedy] chain never runs a backend graph of its own: build_sampling emits one argmax over all
    // output rows for it, whether or not the chain was offloaded
    if (!gsmpl->greedy_backend_biased) {
        return true;
    }
    return llama_sampler_backend_rows_ready(gsmpl->greedy_backend);
}

bool common_sampler_supports_backend_verify(const common_params_sampling & params) {
    if (params.spec_lossy) {
        return false; // [LA3] beat needs the target probabilities and residual draw on the CPU (from 1559f5410)
    }
    if (params.temp <= 0.0f) {
        return false; // greedy chains use common_sampler_get_greedy_backend
    }
    if (params.mirostat != 0 || params.n_probs != 0 || params.min_keep > 1 || !common_grammar_value(params.grammar).empty()) {
        return false;
    }

    for (const auto type : params.samplers) {
        switch (type) {
            case COMMON_SAMPLER_TYPE_TEMPERATURE: // temp and dynamic temp run on the backend
            case COMMON_SAMPLER_TYPE_TOP_K:       // truncations run on the backend for any value
            case COMMON_SAMPLER_TYPE_TOP_P:
            case COMMON_SAMPLER_TYPE_MIN_P:
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (!(params.typ_p >= 1.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (!(params.top_n_sigma < 0.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (!(params.xtc_probability <= 0.0f)) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (params.dry_multiplier != 0.0f && params.dry_penalty_last_n != 0) { return false; }
                break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                // active penalties need the per-row token history (row r sees the draft tokens accepted before it)
                if (params.penalty_last_n != 0 && (params.penalty_repeat != 1.0f ||
                        params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) { return false; }
                break;
            default:
                return false;
        }
    }
    return true;
}

struct llama_sampler * common_sampler_get_backend_verify(struct common_sampler * gsmpl) {
    if (!gsmpl || gsmpl->grmr || gsmpl->rbudget || !common_sampler_supports_backend_verify(gsmpl->params)) {
        return nullptr;
    }
    if (!llama_sampler_backend_supports_rows(gsmpl->chain)) {
        return nullptr;
    }
    return gsmpl->chain;
}

bool common_sampler_backend_verify_ready(const struct common_sampler * gsmpl) {
    return gsmpl && llama_sampler_chain_is_backend(gsmpl->chain);
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_sampler_output snapshot;
    const llama_sampler_output * output = nullptr;
    if (common_sampler_output_enabled()) {
        snapshot = llama_get_sampler_output_ith(ctx, idx);
        output = &snapshot;
    } else {
        llama_synchronize(ctx);
    }

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = output ? output->token : llama_get_sampled_token_ith(ctx, idx);
        gsmpl->sampled_backend = id != LLAMA_TOKEN_NULL;

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            if (output ? !output->logits && !output->probs :
                    !llama_get_sampled_logits_ith(ctx, idx) && !llama_get_sampled_probs_ith(ctx, idx)) {
                // Compact greedy output has no candidate distribution to materialize.
                gsmpl->cur.clear();
                cur_p = { nullptr, 0, -1, false };
                return id;
            }

            gsmpl->set_logits(ctx, idx, output);
            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    gsmpl->set_logits(ctx, idx, output);

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    // a chain that was offloaded to a backend skips all of its samplers here, so cur_p.selected would stay
    // unset: that means the backend returned no token for this row and there is no valid CPU fallback
    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx, output);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // Stop before a bonus draw after a committed EOG.
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && (gsmpl->params.spec_lossy || i + 1 < draft.size()))) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

bool common_sampler_pq_supported(const struct common_sampler * gsmpl) {
    if (!gsmpl || gsmpl->grmr || gsmpl->rbudget) {
        return false;
    }
    const auto & params = gsmpl->params;
    return params.temp > 0.0f && params.mirostat == 0 &&
        (!params.spec_lossy || std::find(params.samplers.begin(), params.samplers.end(), COMMON_SAMPLER_TYPE_ADAPTIVE_P) == params.samplers.end());
}

const common_sampler_pq_stats & common_sampler_get_pq_stats(const struct common_sampler * gsmpl) {
    return gsmpl->pq_stats;
}

common_sampler_pq_result common_sampler_verify_pq(const llama_token_data_array & p, const std::vector<llama_token_data> & q,
        llama_token d, llama_token x, std::mt19937 & rng, bool beat, common_sampler_pq_stats & stats,
        const llama_vocab * vocab, const std::function<bool(llama_token)> & is_stop) {
    auto q_of = [&](llama_token id) -> double {
        for (const auto & c : q) {
            if (c.id == id) {
                return c.p;
            }
        }
        return 0.0;
    };
    double p_d = 0.0, sum_min = 0.0;
    for (size_t k = 0; k < p.size; ++k) {
        const auto & c = p.data[k];
        if (c.id == d) {
            p_d = c.p;
        }
        sum_min += std::min((double) c.p, q_of(c.id));
    }
    const double q_d = q_of(d);
    stats.n_pos++;
    stats.sum_p_d += p_d;
    stats.sum_min += sum_min;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    common_sampler_pq_result res{d};
    res.accepted = x == d;
    if (res.accepted) {
        stats.n_match++;
    } else if (p_d > 0.0 && q_d > 0.0 && p_d < 1.0) {
        const double a = std::min(1.0, p_d / q_d);
        const double r = (a - p_d) / (1.0 - p_d);
        if (r > 0.0 && uniform(rng) < r) {
            res.accepted = true;
            stats.n_extra++;
        }
    }
    if (res.accepted) {
        return res;
    }

    double total = 0.0;
    for (size_t k = 0; k < p.size; ++k) {
        total += std::max(0.0, (double) p.data[k].p - q_of(p.data[k].id));
    }
    res.id = x;
    if (total > 0.0) {
        const double target = uniform(rng) * total;
        double run = 0.0;
        for (size_t k = 0; k < p.size; ++k) {
            const auto & c = p.data[k];
            const double mass = std::max(0.0, (double) c.p - q_of(c.id));
            if (beat && mass <= 0.0) {
                continue;
            }
            // Keep the last positive entry as a rounding fallback. Zero residual mass is never selected.
            if (beat) {
                res.id = c.id;
                res.p_replacement = c.p;
            }
            run += mass;
            if (beat ? run > target : run >= target) {
                res.id = c.id;
                res.p_replacement = c.p;
                break;
            }
        }
    } else {
        res.p_replacement = prob_of(p.data, p.size, x);
    }
    res.replacement = res.id;
    const auto ends = [&](llama_token id) {
        return (vocab && llama_vocab_is_eog(vocab, id)) || (is_stop && is_stop(id));
    };
    if (beat && !ends(d) && !ends(res.replacement) && p_d >= res.p_replacement) {
        res.id = d;
        res.accepted = true;
        res.rescued = true;
        stats.n_lossy++;
    } else {
        stats.n_rej++;
    }
    return res;
}

// V2 keeps the old P prefix and appends tagged fields. Only actual positions are logged; no shadow sampling.
struct pq_logger {
    FILE * f = nullptr;
    std::mutex mutex;
    uint64_t round = 0;
    pq_logger() {
        const char * path = getenv("LLAMA_SPEC_PQ_LOG");
        if (path && *path) {
            f = fopen(path, "a");
            if (f) {
                setvbuf(f, nullptr, _IOFBF, 1 << 20);
                fprintf(f, "V\t2\n");
            }
        }
    }
    ~pq_logger() {
        if (f) {
            fclose(f);
        }
    }
};

static pq_logger * pq_log_get() {
    static pq_logger lg;
    return lg.f ? &lg : nullptr;
}

static void pq_log_pos(pq_logger * lg, common_sampler * gsmpl, const llama_vocab * vocab, int32_t seq, int64_t at,
        size_t j, llama_token d, llama_token x, const llama_token_data_array & p, const std::vector<llama_token_data> & q,
        const common_sampler_pq_result & res, const std::function<bool(llama_token)> & is_stop) {
    auto p_of = [&](llama_token id) -> double { return prob_of(p.data, p.size, id); };
    auto q_of = [&](llama_token id) -> double { return prob_of(q.data(), q.size(), id); };
    double total = 0.0, p_top = 0.0, entropy = 0.0;
    for (size_t k = 0; k < p.size; ++k) {
        const double pk = p.data[k].p;
        total += std::max(0.0, pk - q_of(p.data[k].id));
        p_top = std::max(p_top, pk);
        if (pk > 0.0) {
            entropy -= pk * std::log(pk);
        }
    }
    std::vector<double> law(p.size);
    for (size_t k = 0; k < p.size; ++k) {
        law[k] = p.data[k].p;
    }
    const auto ends = [&](llama_token id) {
        return llama_vocab_is_eog(vocab, id) || (is_stop && is_stop(id));
    };
    // Do not move probability mass to or from an answer boundary.
    if (total > 0.0) {
        for (const auto & a : q) {
            const double pa = p_of(a.id), excess = std::max(0.0, (double) a.p - pa);
            if (excess == 0.0 || ends(a.id)) {
                continue;
            }
            for (size_t k = 0; k < p.size; ++k) {
                const auto & y = p.data[k];
                if (pa < y.p || ends(y.id)) {
                    continue;
                }
                const double moved = excess * std::max(0.0, (double) y.p - q_of(y.id)) / total;
                law[k] -= moved;
                for (size_t z = 0; z < p.size; ++z) {
                    if (p.data[z].id == a.id) {
                        law[z] += moved;
                        break;
                    }
                }
            }
        }
    }
    double kl = 0.0, lr = 0.0;
    for (size_t k = 0; k < p.size; ++k) {
        if (law[k] > 0.0 && p.data[k].p > 0.0) {
            const double log_ratio = std::log(law[k] / p.data[k].p);
            kl += law[k] * log_ratio;
            if (p.data[k].id == res.id) {
                lr = log_ratio;
            }
        }
    }
    const bool ey = res.replacement != LLAMA_TOKEN_NULL && ends(res.replacement);
    const bool eo = ends(res.id);
    gsmpl->pq_pending.push_back({at, res.id, gsmpl->params.spec_lossy ? kl : 0.0,
            gsmpl->params.spec_lossy ? lr : 0.0, res.rescued, res.replacement, res.p_replacement, ey, eo});
    const char outcome = x == d ? 'M' : (res.rescued ? 'L' : (res.accepted ? 'E' : 'R'));
    fprintf(lg->f, "P\t%llu\t%zu\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%c\t%zu",
            (unsigned long long) lg->round, j, d, p_of(d), q_of(d), p_top, entropy, outcome, q.size());
    for (const auto & c : q) {
        fprintf(lg->f, "\t%d:%.9g:%.9g", c.id, c.p, p_of(c.id));
    }
    fprintf(lg->f, "\ty=%d\tpy=%.9g\tout=%d\tkl=%.12g\tlr=%.12g\tey=%d\teo=%d\tseq=%d\tat=%lld\tp=",
            res.replacement, res.p_replacement, res.id, kl, lr, ey, eo, seq, (long long) at);
    for (size_t k = 0; k < p.size; ++k) {
        fprintf(lg->f, "%s%d:%.9g", k ? "," : "", p.data[k].id, p.data[k].p);
    }
    fputc('\n', lg->f);
}

void common_sampler_pq_commit(common_sampler * gsmpl, int32_t seq, int64_t at, llama_token token, bool eog) {
    auto * lg = pq_log_get();
    if (!lg) {
        return;
    }
    std::lock_guard<std::mutex> lock(lg->mutex);
    pq_record row{at, token, 0.0, 0.0, false, LLAMA_TOKEN_NULL, -1.0, false, eog};
    for (const auto & r : gsmpl->pq_pending) {
        if (r.at == at && r.token == token) {
            row = r;
            break;
        }
    }
    fprintf(lg->f, "C\tseq=%d\tat=%lld\ttoken=%d\tkl=%.12g\tlr=%.12g\trescued=%d\ty=%d\tpy=%.9g\tey=%d\teo=%d\n",
            seq, (long long) at, token, row.kl, row.lr, row.rescued, row.y, row.py, row.ey, row.eo);
    fflush(lg->f);
}

std::vector<llama_token> common_sampler_sample_and_accept_n_pq(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q,
        bool grammar_first, bool replay, int32_t seq, int64_t generated,
        const std::function<bool(llama_token, const llama_tokens &)> & is_stop) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    if (!common_sampler_pq_supported(gsmpl)) {
        return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
    }
    GGML_ASSERT((!replay || !gsmpl->params.spec_lossy) && "beat replay must restore the saved post-verification sampler");
    std::vector<llama_token> result;
    result.reserve(idxs.size());
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    // top-1 probability of the distribution just sampled from (cur_p holds p over the surviving candidates)
    gsmpl->top1_trace.clear();
    gsmpl->first_rescue = -1;
    auto trace_top1 = [&]() {
        float top = -1.0f;
        if (llama_get_sampled_token_ith(ctx, idxs[std::min(gsmpl->top1_trace.size(), idxs.size() - 1)]) == LLAMA_TOKEN_NULL) {
            for (size_t k = 0; k < gsmpl->cur_p.size; ++k) {
                top = std::max(top, gsmpl->cur_p.data[k].p);
            }
        }
        gsmpl->top1_trace.push_back(top);
    };
    auto * lg = pq_log_get();
    std::unique_lock<std::mutex> lock;
    const int64_t log_start = lg ? ggml_time_us() : 0;
    int64_t log_us = 0;
    if (lg) {
        lock = std::unique_lock<std::mutex>(lg->mutex);
        gsmpl->pq_pending.clear();
        fprintf(lg->f, "R\t%llu\t%zu\tseq=%d\tat=%lld\n", (unsigned long long) ++lg->round,
                draft.size(), seq, (long long) generated);
    }
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        // x ~ p, the target chain's own draw; it also leaves cur_p holding p over the surviving candidates
        const llama_token x = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        trace_top1();
        if (replay && i + 1 == draft.size()) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }
        const auto & p = gsmpl->cur_p;
        const bool have_q = i < draft_q.size() && !draft_q[i].empty();
        const bool cpu_p  = p.size > 0 && p.selected >= 0 && (common_sampler_output_enabled() ?
                !gsmpl->sampled_backend : llama_get_sampled_token_ith(ctx, idxs[i]) == LLAMA_TOKEN_NULL);
        const auto ends = [&](llama_token id) { return is_stop && is_stop(id, result); };
        common_sampler_pq_result res{x};
        if (have_q && cpu_p) {
            res = common_sampler_verify_pq(p, draft_q[i], draft[i], x, gsmpl->pq_rng, gsmpl->params.spec_lossy, gsmpl->pq_stats, vocab, ends);
            if (res.rescued && gsmpl->first_rescue < 0) {
                gsmpl->first_rescue = (int32_t) i;
            }
            if (lg) {
                const int64_t t0 = ggml_time_us();
                pq_log_pos(lg, gsmpl, vocab, seq, generated + i, i, draft[i], x, p, draft_q[i], res, ends);
                log_us += ggml_time_us() - t0;
            }
        } else {
            // id-match for this and the remaining positions
            res.accepted = x == draft[i];
        }
        const bool stopped = gsmpl->params.spec_lossy && (llama_vocab_is_eog(vocab, res.id) || ends(res.id));
        common_sampler_accept(gsmpl, res.id, true);
        result.push_back(res.id);
        if (!res.accepted || stopped) {
            break;
        }
    }
    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        trace_top1();
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
    }
    if (lg) {
        fprintf(lg->f, "D\t%llu\tlog_us=%lld\tverify_us=%lld\n", (unsigned long long) lg->round, (long long) log_us, (long long) (ggml_time_us() - log_start));
    }
    return result;
}

const std::vector<float> & common_sampler_get_top1_trace(const struct common_sampler * gsmpl) {
    return gsmpl->top1_trace;
}

int32_t common_sampler_get_first_rescue(const struct common_sampler * gsmpl) {
    return gsmpl ? gsmpl->first_rescue : -1;
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
