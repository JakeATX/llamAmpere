#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_sampler_backend_supports_rows

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

static void test_greedy_argmax_rows(bool cpu_only) {
    constexpr int cols = 2081;
    constexpr int rows = 7;
    std::vector<float> logits(cols * rows, -10.0f);
    std::fill_n(logits.data(), cols, 2.0f);
    logits[cols + 5] = logits[cols + 32] = logits[2 * cols - 1] = 3.0f;
    std::fill_n(logits.data() + 2 * cols, cols, -INFINITY);
    logits[3 * cols] = std::numeric_limits<float>::quiet_NaN();
    logits[3 * cols + 7] = 1.0f;
    logits[4 * cols + 5] = 3.0f;
    logits[4 * cols + 17] = std::numeric_limits<float>::quiet_NaN();
    logits[5 * cols + 5] = logits[5 * cols + 32] = INFINITY;
    std::fill_n(logits.data() + 6 * cols, cols, -std::numeric_limits<float>::max());
    const std::vector<llama_token> expected = {0, 5, 0, 0, 5, 5, 0};

    std::vector<ggml_backend_t> backends;
    if (cpu_only) {
        backends.push_back(ggml_backend_cpu_init());
    } else {
        ggml_backend_load_all();
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
            auto dev = ggml_backend_dev_get(d);
            const std::string name = ggml_backend_dev_name(dev);
            // These backends implement the first-index tie convention used by greedy verification.
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && name.find("CUDA") != 0) { continue; }
            backends.push_back(ggml_backend_dev_init(dev, nullptr));
        }
    }
    int tested = 0;
    for (auto backend : backends) {
        GGML_ASSERT(backend);
        const std::string name = ggml_backend_name(backend);
        ggml_init_params init = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(init);
        GGML_ASSERT(ctx);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        ggml_tensor * result = ggml_argmax(ctx, input);
        GGML_ASSERT(ggml_backend_supports_op(backend, result));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        ggml_backend_tensor_set(input, logits.data(), 0, logits.size() * sizeof(float));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<llama_token> actual(rows);
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(llama_token));
        GGML_ASSERT(actual == expected);
        printf("greedy argmax ties/nonfinite rows: %s PASSED\n", name.c_str());
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        ++tested;
    }
    GGML_ASSERT(tested > 0);
}

static void test_greedy_backend_truncation_ties() {
    int divergent = 0;
    for (int n : {32, 64, 128}) {
        for (int k : {2, 20}) {
            std::vector<llama_token_data> data;
            for (int i = 0; i < n; ++i) { data.push_back({i, 2.0f, 0.0f}); }
            llama_token_data_array row{data.data(), data.size(), -1, false};
            auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(0));
            llama_sampler_apply(chain, &row);
            GGML_ASSERT(row.selected >= 0);
            const llama_token selected = row.data[row.selected].id;
            printf("top-k tie witness: n=%d k=%d cpu=%d raw_argmax=0\n", n, k, selected);
            divergent += selected != 0;
            llama_sampler_free(chain);
        }
    }
    printf("top-k tie witnesses: %d/6 differ on this standard library\n", divergent);
    common_params_sampling params;
    params.temp = 0.0f;
    params.top_k = 20;
    params.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
}

// Run [logit-bias, greedy] (or bare [greedy] when there is no bias) on the CPU backend over a
// [n_vocab, n_rows] logit block, the way speculative verification does, and return the sampled ids.
static std::vector<llama_token> run_backend_greedy_rows(
        const std::vector<llama_logit_bias> & biases,
        const std::vector<float>            & logits,
        int                                   n_vocab,
        int                                   n_rows) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);

    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!biases.empty()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, biases.size(), biases.data()));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    GGML_ASSERT(llama_sampler_backend_supports_rows(chain));
    GGML_ASSERT(chain->iface->backend_init(chain, ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(llama_sampler_backend_rows_ready(chain));

    ggml_init_params init = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(init);
    GGML_ASSERT(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_rows);
    ggml_set_input(inp);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    llama_sampler_data data = { inp, nullptr, nullptr, nullptr };
    chain->iface->backend_apply(chain, ctx, gf, &data);
    GGML_ASSERT(data.sampled != nullptr);
    ggml_build_forward_expand(gf, data.sampled);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    ggml_backend_tensor_set(inp, logits.data(), 0, logits.size()*sizeof(float));
    chain->iface->backend_set_input(chain);

    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<llama_token> out(n_rows);
    GGML_ASSERT(ggml_nelements(data.sampled) == n_rows);
    ggml_backend_tensor_get(data.sampled, out.data(), 0, out.size()*sizeof(llama_token));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    llama_sampler_free(chain);
    ggml_backend_free(backend);

    return out;
}

// The CPU chain common_sampler_init() builds for a greedy request: the raw (undeduplicated) bias list
// followed by greedy, applied one row at a time.
static std::vector<llama_token> run_cpu_greedy_rows(
        const std::vector<llama_logit_bias> & biases,
        const std::vector<float>            & logits,
        int                                   n_vocab,
        int                                   n_rows) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!biases.empty()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, biases.size(), biases.data()));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    std::vector<llama_token> out;
    for (int r = 0; r < n_rows; ++r) {
        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            cur.push_back({ i, logits[r*n_vocab + i], 0.0f });
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(chain, &cur_p);
        GGML_ASSERT(cur_p.selected >= 0);
        out.push_back(cur_p.data[cur_p.selected].id);
    }

    llama_sampler_free(chain);
    return out;
}

// Suppressed tokens must not be selected by the backend greedy path.
//
// Model suppress tokens and request logit biases reach the greedy chain through the same list
// (common_sampler_greedy_biases appends the suppressed ids to the request's biases at -INFINITY), so the
// -INFINITY entries below stand for either source. The reference is the CPU chain common_sampler_init()
// builds from the raw list, including a duplicated token, which the CPU sampler adds twice.
static void test_greedy_backend_suppressed_argmax() {
    constexpr int n_vocab = 257;
    constexpr int n_rows  = 5;

    const std::vector<llama_token> suppressed = { 3, 11, 250 };

    std::vector<float> logits(n_vocab*n_rows);
    for (int r = 0; r < n_rows; ++r) {
        for (int i = 0; i < n_vocab; ++i) {
            logits[r*n_vocab + i] = -10.0f + 0.001f*((i*37 + r*11) % 97);
        }
    }
    // row 0: the argmax is a suppressed id, the runner-up is not
    logits[0*n_vocab + 3]   =  5.0f;
    logits[0*n_vocab + 40]  =  4.0f;
    // row 1: the two best ids are both suppressed
    logits[1*n_vocab + 250] =  9.0f;
    logits[1*n_vocab + 11]  =  8.0f;
    logits[1*n_vocab + 7]   =  7.0f;
    // row 2: nothing suppressed is anywhere near the top
    logits[2*n_vocab + 5]   =  3.0f;
    // row 3: a tie between a suppressed id and two others - the lowest surviving id wins
    logits[3*n_vocab + 3]   =  2.0f;
    logits[3*n_vocab + 64]  =  2.0f;
    logits[3*n_vocab + 65]  =  2.0f;
    // row 4: only the finitely biased id 40 is on top, so the bias decides
    logits[4*n_vocab + 40]  =  1.0f;
    logits[4*n_vocab + 41]  =  0.5f;

    const std::vector<llama_token> expected = { 40, 7, 5, 64, 41 };

    // as the server builds it: a repeated user bias (summed by the CPU sampler) plus the suppressed ids
    common_params_sampling params;
    params.temp = 0.0f;
    params.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    params.logit_bias.push_back({ 40, -0.4f });
    params.logit_bias.push_back({ 40, -0.4f });
    for (auto id : suppressed) {
        params.logit_bias.push_back({ id, -INFINITY });
    }
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));

    const auto merged = common_sampler_greedy_biases(params, nullptr);
    GGML_ASSERT(merged.size() == params.logit_bias.size() - 1); // the duplicate was folded
    GGML_ASSERT(merged[0].token == 40 && merged[0].bias == -0.8f);

    const auto backend = run_backend_greedy_rows(merged,             logits, n_vocab, n_rows);
    const auto cpu     = run_cpu_greedy_rows    (params.logit_bias,  logits, n_vocab, n_rows);

    for (int r = 0; r < n_rows; ++r) {
        printf("suppressed greedy row %d: backend=%d cpu=%d expected=%d\n", r, backend[r], cpu[r], expected[r]);
    }
    GGML_ASSERT(backend == cpu);
    GGML_ASSERT(backend == expected);

    // and with no biases at all the path is unchanged: the raw argmax of every row
    common_params_sampling plain;
    plain.temp = 0.0f;
    const auto none = common_sampler_greedy_biases(plain, nullptr);
    GGML_ASSERT(none.empty());

    const auto backend_plain = run_backend_greedy_rows(none, logits, n_vocab, n_rows);
    const auto cpu_plain     = run_cpu_greedy_rows    (none, logits, n_vocab, n_rows);
    const std::vector<llama_token> expected_plain = { 3, 250, 5, 3, 40 };
    GGML_ASSERT(backend_plain == cpu_plain);
    GGML_ASSERT(backend_plain == expected_plain);

    printf("greedy backend suppress tokens: PASSED\n");
}

static void test_greedy_backend_eligibility() {
    common_params_sampling params;
    params.temp = 0.0f;
    params.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.temp = 0.1f;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.temp = 0.0f;
    params.n_probs = 1;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.n_probs = 0;
    // logit biases (the shape ignore_eos takes) are applied on the backend in front of the argmax
    params.logit_bias.push_back({0, -INFINITY});
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.logit_bias.clear();
    params.samplers.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.penalty_repeat = 1.1f;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.penalty_repeat = 1.0f;
    // Active truncation must not be bypassed, including its tie ordering.
    params.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K);
    params.top_k = 40;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.top_k = 0;
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_P);
    params.top_p = 0.95f;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.top_p = 1.0f;
    params.samplers.push_back(COMMON_SAMPLER_TYPE_MIN_P);
    params.min_p = 0.05f;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.min_p = 0.0f;
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.samplers.push_back(COMMON_SAMPLER_TYPE_TYPICAL_P);
    params.typ_p = 0.9f;
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
    params.typ_p = 1.0f;
    GGML_ASSERT(common_sampler_supports_greedy_backend(params));
    params.samplers.pop_back();
    params.samplers.push_back(COMMON_SAMPLER_TYPE_INFILL);
    GGML_ASSERT(!common_sampler_supports_greedy_backend(params));
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc == 2 && std::strcmp(argv[1], "--cpu-only") == 0;
    if (argc != 1 && !cpu_only) {
        fprintf(stderr, "usage: %s [--cpu-only]\n", argv[0]);
        return 1;
    }
    test_greedy_backend_truncation_ties();
    test_greedy_backend_eligibility();
    test_greedy_backend_suppressed_argmax();
    test_greedy_argmax_rows(cpu_only);
    ggml_time_init();

    test_dist_singleton_rng();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
