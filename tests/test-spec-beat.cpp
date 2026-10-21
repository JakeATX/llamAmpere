#include "sampling.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<llama_token_data> probs(std::initializer_list<float> p) {
    std::vector<llama_token_data> result;
    for (float value : p) {
        result.push_back({(llama_token) result.size(), std::log(value), value});
    }
    return result;
}

static void test_identity_and_ties() {
    auto p = probs({0.05f, 0.95f}), q = probs({0.8f, 0.2f});
    llama_token_data_array arr{p.data(), p.size(), 1, false};
    std::mt19937 e(7300), b(e);
    common_sampler_pq_stats es, bs;
    for (int i = 0; i < 10000; ++i) {
        auto exact = common_sampler_verify_pq(arr, q, 0, 1, e, false, es);
        auto beat = common_sampler_verify_pq(arr, q, 0, 1, b, true, bs);
        assert(exact.id == beat.id && exact.accepted == beat.accepted);
        assert(exact.replacement == beat.replacement && exact.p_replacement == beat.p_replacement);
        assert(!beat.rescued && e == b);
    }
    assert(es.n_rej > 9000 && bs.n_lossy == 0);
    p = probs({0.5f, 0.5f}); q = probs({0.75f, 0.25f});
    arr.data = p.data();
    bool tied = false;
    for (int i = 0; i < 100; ++i) {
        e.seed(i); b = e;
        auto exact = common_sampler_verify_pq(arr, q, 0, 1, e, false, es);
        auto beat = common_sampler_verify_pq(arr, q, 0, 1, b, true, bs);
        assert(e == b);
        if (!exact.accepted) {
            assert(exact.id == 1 && beat.id == 0 && beat.rescued && beat.accepted);
            assert(beat.replacement == 1 && beat.p_replacement == 0.5);
            tied = true;
        }
    }
    assert(tied);
    puts("no-rescue identity/RNG, equality rescue and actual Y: PASS");
}

static void test_laws() {
    auto p = probs({0.50f, 0.30f, 0.20f}), q = probs({0.75f, 0.10f, 0.15f});
    llama_token_data_array arr{p.data(), p.size(), 0, false};
    std::discrete_distribution<int> pd({.5, .3, .2}), qd({.75, .1, .15});
    const double expected[2][3] = {{.5, .3, .2}, {.75, .1, .15}};
    for (int arm = 0; arm < 2; ++arm) {
        std::mt19937 draws(123), rng(456);
        common_sampler_pq_stats stats;
        int counts[3] = {};
        constexpr int n = 400000;
        for (int i = 0; i < n; ++i) {
            const int d = qd(draws), x = pd(draws);
            auto res = common_sampler_verify_pq(arr, q, d, x, rng, arm != 0, stats);
            counts[res.id]++;
        }
        for (int k = 0; k < 3; ++k) {
            assert(std::abs((double) counts[k] / n - expected[arm][k]) < .004);
        }
        assert(stats.n_pos == n);
        assert(stats.n_match + stats.n_extra + stats.n_rej + stats.n_lossy == n);
    }
    // p=0 draft cannot beat any positive residual; absent q support keeps target tail intact.
    p = probs({0.0f, 0.4f, 0.6f}); q = probs({1.0f}); arr.data = p.data();
    std::mt19937 rng(42);
    common_sampler_pq_stats stats;
    for (int i = 0; i < 1000; ++i) {
        auto r = common_sampler_verify_pq(arr, q, 0, 2, rng, true, stats);
        assert(!r.rescued && !r.accepted && r.id > 0 && r.p_replacement > 0.0);
    }
    puts("400K-draw exact and beat laws, counters, zero/absent support: PASS");
}

static void test_model_state(const char * path) {
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    mp.devices = devices;
    auto * model = llama_model_load_from_file(path, mp);
    assert(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 8; cp.n_ubatch = 8; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.n_rs_seq = 8;
    auto * ctx = llama_init_from_model(model, cp);
    assert(ctx);
    llama_token input[5] = {1, 2, 3, 4, 5};
    auto batch = llama_batch_init(5, 0, 1);
    batch.n_tokens = 5;
    for (int i = 0; i < 5; ++i) {
        batch.token[i] = input[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; batch.logits[i] = true;
    }
    assert(llama_decode(ctx, batch) == 0);
    const auto * vocab = llama_model_get_vocab(model);
    const int n = llama_vocab_n_tokens(vocab);
    const llama_token eog = llama_vocab_eos(vocab);
    assert(eog >= 0 && eog != 20 && eog != 21);
    auto set_p = [&](bool stop) {
        for (int row = 0; row < 5; ++row) {
            float * logits = llama_get_logits_ith(ctx, row);
            for (int k = 0; k < n; ++k) { logits[k] = -INFINITY; }
            logits[20] = std::log(.5f); logits[21] = std::log(.3f); logits[22] = std::log(.2f);
            if (stop) {
                logits[20] = logits[21] = logits[22] = -INFINITY; logits[eog] = 0;
            }
        }
    };
    common_params_sampling params;
    params.seed = 7300; params.temp = 1; params.top_k = 3; params.top_p = 1; params.min_p = 0;
    params.spec_lossy = true;
    auto * s = common_sampler_init(model, params);
    std::vector<std::vector<llama_token_data>> qs(4, {{20, 0, .75f}, {21, 0, .1f}, {22, 0, .15f}});
    auto step = [&](common_sampler * sampler) {
        set_p(false);
        return common_sampler_sample_and_accept_n_pq(sampler, ctx, {0,1,2,3,4}, {20,20,20,20}, qs);
    };
    step(s);
    auto * clone = common_sampler_clone(s);
    auto * copied = common_sampler_init(model, params);
    common_sampler_copy(s, copied);
    for (int i = 0; i < 64; ++i) {
        const auto expected = step(s);
        assert(step(clone) == expected && step(copied) == expected);
        assert(common_sampler_get_pq_stats(s).n_pos == common_sampler_get_pq_stats(copied).n_pos);
    }
    assert(common_sampler_get_pq_stats(s).n_lossy > 0);
    common_sampler_reset(copied);
    auto * fresh = common_sampler_init(model, params);
    for (int i = 0; i < 32; ++i) { assert(step(copied) == step(fresh)); }
    // A committed stop must terminate verification, with no bonus or suffix sampler draws.
    set_p(true);
    auto before = common_sampler_get_pq_stats(s).n_pos;
    auto out = common_sampler_sample_and_accept_n_pq(s, ctx, {0,1,2}, {eog,20}, {qs[0],qs[0]});
    assert(out == llama_tokens{eog});
    assert(common_sampler_get_pq_stats(s).n_pos == before + 1);
    out = common_sampler_sample_and_accept_n(s, ctx, std::vector<int>{0,1}, llama_tokens{eog});
    assert(out == llama_tokens{eog});
    // Rejected EOS and request stop tokens must retain the exact decision and RNG state.
    auto end_p = std::vector<llama_token_data>{{20, 0, .6f}, {eog, 0, .4f}};
    auto end_q = std::vector<llama_token_data>{{20, 0, 1.0f}};
    llama_token_data_array end_arr{end_p.data(), end_p.size(), 1, false};
    int endings = 0;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 exact_rng(i), beat_rng(i);
        common_sampler_pq_stats es, bs;
        auto exact = common_sampler_verify_pq(end_arr, end_q, 20, eog, exact_rng, false, es, vocab);
        auto beat = common_sampler_verify_pq(end_arr, end_q, 20, eog, beat_rng, true, bs, vocab);
        assert(exact.id == beat.id && exact_rng == beat_rng && !beat.rescued);
        endings += beat.id == eog;
    }
    assert(endings > 0);
    // The server stop predicate includes the preceding output, including the accepted draft prefix.
    end_p[1].id = 21;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 rng(i);
        common_sampler_pq_stats stats;
        auto res = common_sampler_verify_pq(end_arr, end_q, 20, 21, rng, true, stats, vocab,
                [](llama_token id) { return id == 21; });
        assert(!res.rescued);
    }
    // A draft EOS cannot rescue a rejected normal replacement either.
    end_p[0].id = eog; end_p[1].id = 20; end_q[0].id = eog;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 rng(i);
        common_sampler_pq_stats stats;
        auto res = common_sampler_verify_pq(end_arr, end_q, eog, 20, rng, true, stats, vocab);
        assert(!res.rescued);
    }
    for (int row = 0; row < 3; ++row) {
        float * logits = llama_get_logits_ith(ctx, row);
        for (int k = 0; k < n; ++k) { logits[k] = -INFINITY; }
        logits[20] = 0;
    }
    out = common_sampler_sample_and_accept_n_pq(s, ctx, {0,1,2}, {20,20}, {{{20,0,1}},{{20,0,1}}},
            false, false, 0, 0, [](llama_token id, const llama_tokens & prefix) { return id == 20 && prefix.empty(); });
    assert(out == llama_tokens{20});
    out = common_sampler_sample_and_accept_n_pq(s, ctx, {0,1,2}, {20,20}, {{{20,0,1}},{{20,0,1}}},
            false, false, 0, 0, [](llama_token id, const llama_tokens & prefix) { return id == 20 && prefix.size() == 1; });
    assert(out == (llama_tokens{20,20}));
    // XTC can leave cur_p pointing inside cur. Snapshots must own that same shifted view.
    params.xtc_probability = 1.0f; params.xtc_threshold = .1f;
    auto * shifted = common_sampler_init(model, params);
    set_p(false);
    common_sampler_sample(shifted, ctx, 0);
    auto * view = common_sampler_get_candidates(shifted, false);
    assert(view->size == 1 && view->data[0].id == 22);
    auto * shifted_clone = common_sampler_clone(shifted);
    auto * shifted_copy = common_sampler_init(model, params);
    common_sampler_copy(shifted, shifted_copy);
    assert(common_sampler_get_candidates(shifted_clone, false)->data[0].id == 22);
    assert(common_sampler_get_candidates(shifted_copy, false)->data[0].id == 22);
    view->data[0].id = 999;
    common_sampler_free(shifted);
    assert(common_sampler_get_candidates(shifted_clone, false)->data[0].id == 22);
    assert(common_sampler_get_candidates(shifted_copy, false)->data[0].id == 22);
    common_sampler_free(shifted_clone); common_sampler_free(shifted_copy);
    puts("real sampler clone/copy/reset, repeated rescues, EOG without bonus: PASS");
    common_sampler_free(s); common_sampler_free(clone); common_sampler_free(copied); common_sampler_free(fresh);
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
}

static void bench() {
    for (int support : {3, 20, 40}) {
        std::vector<llama_token_data> p;
        double norm = 0.0;
        for (int i = 0; i < support; ++i) { norm += 1.0 / (i + 1); }
        std::vector<llama_token_data> q;
        for (int i = 0; i < support; ++i) {
            const float mass = (float) (1.0 / (i + 1) / norm);
            p.push_back({i, 0.0f, mass});
            q.push_back({i, 0.0f, mass});
        }
        q[0].p += p[1].p / 2; q[1].p /= 2;
        llama_token_data_array arr{p.data(), p.size(), 1, false};
        for (int repeat = 0; repeat < 8; ++repeat) {
            double us[2] = {};
            for (int k = 0; k < 2; ++k) {
                const int arm = k ^ (repeat & 1);
                std::mt19937 rng(7300); common_sampler_pq_stats stats;
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 100000; ++i) {
                    common_sampler_verify_pq(arr, q, 0, 1, rng, arm != 0, stats);
                }
                const auto end = std::chrono::steady_clock::now();
                us[arm] = std::chrono::duration<double, std::micro>(end - start).count() / 100000 * 4;
                assert(stats.n_pos == 100000);
            }
            printf("BENCH support=%d repeat=%d exact_us_per_4pos=%.6f beat_us_per_4pos=%.6f delta_us=%.6f\n",
                   support, repeat, us[0], us[1], us[1]-us[0]);
        }
    }
}

int main(int argc, char ** argv) {
    test_identity_and_ties(); test_laws();
    if (argc == 2) { test_model_state(argv[1]); }
    bench();
    puts("test-spec-beat: PASS");
}
