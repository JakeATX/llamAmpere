#include "sampling.h"
#include "arg.h"
#include "spec-defaults.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

static void test_model_mtp_default() {
#ifndef _WIN32
    char path[] = "/tmp/test-spec-default-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    FILE * file = fdopen(fd, "wb");
    assert(file);
    auto * metadata = gguf_init_empty();
    gguf_set_val_str(metadata, "general.architecture", "qwen35");
    gguf_set_val_u32(metadata, "qwen35.block_count", 2);
    gguf_set_val_u32(metadata, "qwen35.nextn_predict_layers", 1);
    ggml_init_params gp = {ggml_tensor_overhead(), nullptr, true};
    auto * ctx = ggml_init(gp);
    assert(ctx);
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tensor, "blk.1.nextn.eh_proj.weight");
    gguf_add_tensor(metadata, tensor);
    assert(gguf_write_to_file_ptr(metadata, file, true));
    assert(fclose(file) == 0);
    gguf_free(metadata);
    ggml_free(ctx);

    const struct {
        std::vector<std::string> options;
        bool automatic;
        bool mode_biased;
    } cases[] = {
        {{}, true, true},
        {{"--no-spec-mb-mtp"}, true, false},
        {{"--spec-mb-mtp"}, true, true},
        {{"--spec-type", "draft-mtp"}, false, true},
        {{"--spec-type", "draft-mtp-adaptive"}, false, true},
        {{"--spec-type", "draft-mtp", "--no-spec-mb-mtp"}, false, false},
        {{"--spec-type", "draft-mtp-adaptive", "--no-spec-mb-mtp"}, false, false},
        {{"--spec-type", "none"}, false, false},
        {{"--spec-draft-n-max", "0"}, false, false},
    };
    for (const auto & test : cases) {
        std::vector<std::string> args = {"test-spec-beat", "-m", path};
        args.insert(args.end(), test.options.begin(), test.options.end());
        std::vector<char *> argv;
        for (auto & arg : args) { argv.push_back(&arg[0]); }
        common_params params;
        assert(common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
        if (test.automatic) {
            assert(params.speculative.types == std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        }
        assert(common_speculative_apply_model_default(params));
        assert(params.sampling.spec_lossy == test.mode_biased);
        if (test.automatic) {
            assert(params.speculative.types.back() == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE);
        }
        assert(common_speculative_apply_model_default(params));
        assert(params.sampling.spec_lossy == test.mode_biased);
    }
    assert(unlink(path) == 0);
    puts("model-header automatic MTP, explicit choices and repeated resolution: PASS");
#endif
}

static std::vector<llama_token_data> probs(std::initializer_list<float> p) {
    std::vector<llama_token_data> result;
    for (float value : p) {
        result.push_back({(llama_token) result.size(), std::log(value), value});
    }
    return result;
}

static common_sampler_pq_result verify_pq_reference(const llama_token_data_array & p, const std::vector<llama_token_data> & q,
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
        res.p_replacement = 0.0;
        for (size_t k = 0; k < p.size; ++k) {
            if (p.data[k].id == x) { res.p_replacement = p.data[k].p; break; }
        }
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

static void test_verifier_identity() {
    std::mt19937 inputs(90817);
    size_t checked = 0;
    for (int support : {1, 3, 4, 5, 20, 40, 64, 65, 129}) {
        for (int shape = 0; shape < 3; ++shape) {
            std::vector<llama_token_data> p, q;
            double ps = 0.0, qs = 0.0;
            for (int k = 0; k < support; ++k) {
                const float mass = k % 7 == 1 ? 0.0f : 1.0f + inputs() % 97;
                p.push_back({3*k + 7, 0.0f, mass}); ps += mass;
                if (shape != 2 || k % 3 != 1) {
                    const float qm = k % 5 == 2 ? 0.0f : 1.0f + inputs() % 89;
                    q.push_back({shape == 1 && k % 4 == 0 ? 10000 + k : 3*k + 7, 0.0f, qm}); qs += qm;
                }
            }
            for (auto & c : p) { c.p /= ps; }
            for (auto & c : q) { c.p /= qs; }
            if (shape != 0) { std::shuffle(p.begin(), p.end(), inputs); std::shuffle(q.begin(), q.end(), inputs); }
            llama_token_data_array arr{p.data(), p.size(), 0, false};
            for (bool beat : {false, true}) {
                std::mt19937 actual(717), reference(actual);
                common_sampler_pq_stats as, rs;
                const auto stop = [](llama_token id) { return id % 11 == 7; };
                for (int i = 0; i < 512; ++i) {
                    const llama_token d = i % 17 == 0 ? 77777 : q[inputs() % q.size()].id;
                    const llama_token x = p[inputs() % p.size()].id;
                    const auto a = common_sampler_verify_pq(arr, q, d, x, actual, beat, as, nullptr, stop);
                    const auto r = verify_pq_reference(arr, q, d, x, reference, beat, rs, nullptr, stop);
                    assert(a.id == r.id && a.replacement == r.replacement && a.p_replacement == r.p_replacement);
                    assert(a.accepted == r.accepted && a.rescued == r.rescued && actual == reference);
                    assert(as.n_pos == rs.n_pos && as.n_match == rs.n_match && as.n_extra == rs.n_extra);
                    assert(as.n_rej == rs.n_rej && as.n_lossy == rs.n_lossy);
                    assert(as.sum_p_d == rs.sum_p_d && as.sum_min == rs.sum_min);
                    ++checked;
                }
            }
        }
    }
    printf("cached verifier identity: %zu token/counter/RNG cases PASS\n", checked);
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

static void test_mode_distribution() {
    auto p = probs({.50f, .30f, .20f}), q = probs({.75f, .10f, .15f});
    llama_token_data_array arr{p.data(), p.size(), 0, false};
    auto law = common_sampler_pq_mode_distribution(arr, q);
    for (size_t k = 0; k < law.size(); ++k) { assert(std::abs(law[k] - q[k].p) < 1e-6); }
    // Both sides of the ending veto must preserve the exact law for this single excess token.
    for (int ending : {0, 1, 2}) {
        const auto stop = [ending](llama_token id) { return id == ending; };
        law = common_sampler_pq_mode_distribution(arr, q, stop);
        std::mt19937 draws(391), rng(729);
        std::discrete_distribution<int> pd({.5, .3, .2}), qd({.75, .1, .15});
        int counts[3] = {}; common_sampler_pq_stats stats;
        for (int i = 0; i < 100000; ++i) {
            const int d = qd(draws), x = pd(draws);
            counts[common_sampler_verify_pq(arr, q, d, x, rng, true, stats, nullptr, stop).id]++;
        }
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
            assert(law[k] >= 0.0 && std::abs(counts[k]/100000.0 - law[k]) < .006);
            sum += law[k];
        }
        assert(std::abs(sum - 1.0) < 1e-6);
    }
    // Equal probabilities rescue; missing q support and zero p support are valid.
    p = probs({.5f, .5f, 0.f}); q = probs({.75f, 0.f, .25f}); arr.data = p.data();
    law = common_sampler_pq_mode_distribution(arr, q);
    assert(std::abs(law[0] - .75) < 1e-6 && std::abs(law[1] - .25) < 1e-6 && law[2] == 0.0);
    q = p; law = common_sampler_pq_mode_distribution(arr, q);
    for (size_t k = 0; k < law.size(); ++k) { assert(law[k] == p[k].p); }
    puts("mode-biased local law, ending veto, ties, missing/zero support: PASS");
}

static void test_model_state(const char * path, bool mode_biased) {
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
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
    params.spec_lossy = mode_biased;
    auto * s = common_sampler_init(model, params);
    std::vector<std::vector<llama_token_data>> qs(4, {{20, 0, .75f}, {21, 0, .1f}, {22, 0, .15f}});
    auto step = [&](common_sampler * sampler) {
        set_p(false);
        return common_sampler_sample_and_accept_n_pq(sampler, ctx, {0,1,2,3,4}, {20,20,20,20}, qs);
    };
    // Both verification modes use the p/q RNG, including exact rejection/residual draws.
    for (bool lossy : {false, true}) {
        auto request = params;
        request.spec_lossy = lossy;
        auto * source = common_sampler_init(model, request);
        for (int i = 0; i < 7; ++i) { step(source); }
        const auto consumed = common_sampler_get_pq_stats(source);
        assert(consumed.n_extra + consumed.n_rej + consumed.n_lossy > 0);
        auto * snapshot = common_sampler_clone(source);
        auto * restored = common_sampler_init(model, request);
        common_sampler_copy(source, restored);
        assert(common_sampler_get_pq_stats(restored).n_pos == consumed.n_pos);
        for (int i = 0; i < 64; ++i) {
            const auto expected = step(source);
            assert(step(snapshot) == expected && step(restored) == expected);
            assert(common_sampler_get_pq_stats(restored).n_pos == common_sampler_get_pq_stats(source).n_pos);
        }
        common_sampler_reset(restored);
        assert(common_sampler_get_pq_stats(restored).n_pos == 0);
        auto * fresh_request = common_sampler_init(model, request);
        for (int i = 0; i < 64; ++i) { assert(step(restored) == step(fresh_request)); }
        common_sampler_free(source); common_sampler_free(snapshot);
        common_sampler_free(restored); common_sampler_free(fresh_request);
    }
    puts("exact and mode-biased p/q clone/copy/reset request streams: PASS");
    step(s);
    auto * clone = common_sampler_clone(s);
    auto * copied = common_sampler_init(model, params);
    common_sampler_copy(s, copied);
    for (int i = 0; i < 64; ++i) {
        const auto expected = step(s);
        assert(step(clone) == expected && step(copied) == expected);
        assert(common_sampler_get_pq_stats(s).n_pos == common_sampler_get_pq_stats(copied).n_pos);
    }
    assert((common_sampler_get_pq_stats(s).n_lossy > 0) == mode_biased);
    common_sampler_reset(copied);
    auto * fresh = common_sampler_init(model, params);
    for (int i = 0; i < 32; ++i) { assert(step(copied) == step(fresh)); }
    common_sampler * slots[3], * controls[3];
    for (int i = 0; i < 3; ++i) {
        params.seed = 8100 + i;
        slots[i] = common_sampler_init(model, params);
        controls[i] = common_sampler_init(model, params);
    }
    for (int round = 0; round < 64; ++round) {
        for (int i = 0; i < 3; ++i) {
            // A slot may finish, reset, or replay while the other slots keep their state.
            if (round == 32 && i == 1) {
                common_sampler_reset(slots[i]);
                common_sampler_reset(controls[i]);
            }
            auto * saved = common_sampler_clone(slots[i]);
            const auto expected = step(controls[i]);
            assert(step(slots[i]) == expected);
            const auto rescue = common_sampler_get_first_rescue(slots[i]);
            const auto trace = common_sampler_get_top1_trace(slots[i]);
            auto * verified = common_sampler_clone(slots[i]);
            assert(common_sampler_get_first_rescue(verified) == rescue);
            assert(common_sampler_get_top1_trace(verified) == trace);
            common_sampler_copy(saved, slots[i]);
            common_sampler_free(saved);
            assert(step(slots[i]) == expected);
            assert(common_sampler_get_first_rescue(slots[i]) == rescue);
            assert(common_sampler_get_pq_stats(slots[i]).n_pos == common_sampler_get_pq_stats(controls[i]).n_pos);
            // Restore the post-verify snapshot after rollback; do not draw from the replay logits.
            common_sampler_reset(slots[i]);
            common_sampler_copy(verified, slots[i]);
            common_sampler_free(verified);
            assert(common_sampler_get_first_rescue(slots[i]) == rescue);
            assert(common_sampler_get_top1_trace(slots[i]) == trace);
            assert(step(slots[i]) == step(controls[i]));
        }
    }
    for (int i = 0; i < 3; ++i) {
        common_sampler_free(slots[i]);
        common_sampler_free(controls[i]);
    }
    puts("three independent sampler states, per-slot reset and checkpoint replay: PASS");
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
    // Unsupported p/q chains and empty-q verification must obey the same stop prefix.
    for (const bool fallback : {false, true}) {
        auto stop_params = params;
        stop_params.mirostat = fallback ? 2 : 0;
        for (const bool pq : {false, true}) {
            for (size_t prefix_size = 0; prefix_size < 2; ++prefix_size) {
                auto * stopped = common_sampler_init(model, stop_params);
                auto * control = common_sampler_init(model, stop_params);
                assert(common_sampler_pq_supported(stopped) != fallback);
                auto stop = [prefix_size](llama_token id, const llama_tokens & prefix) {
                    return id == 20 && prefix.size() == prefix_size;
                };
                out = pq
                    ? common_sampler_sample_and_accept_n_pq(stopped, ctx, {0,1,2}, {20,20}, {{{20,0,1}},{{20,0,1}}}, false, false, 0, 0, stop)
                    : common_sampler_sample_and_accept_n(stopped, ctx, llama_tokens{20,20}, false, stop);
                assert(out == llama_tokens(prefix_size + 1, 20));
                for (size_t j = 0; j <= prefix_size; ++j) {
                    const auto id = common_sampler_sample(control, ctx, j);
                    assert(id == 20);
                    common_sampler_accept(control, id, true);
                }
                if (fallback || !pq) {
                    // Compare later random draws, not just the forced stop tokens.
                    set_p(false);
                    for (int j = 0; j < 8; ++j) {
                        const auto id = common_sampler_sample(control, ctx, 0);
                        assert(common_sampler_sample(stopped, ctx, 0) == id);
                        common_sampler_accept(control, id, true);
                        common_sampler_accept(stopped, id, true);
                    }
                }
                common_sampler_free(stopped);
                common_sampler_free(control);
                for (int row = 0; row < 3; ++row) {
                    float * logits = llama_get_logits_ith(ctx, row);
                    for (int k = 0; k < n; ++k) { logits[k] = -INFINITY; }
                    logits[20] = 0;
                }
            }
        }
    }
    puts("ordinary and Mirostat fallback stop prefixes without extra sampler draws: PASS");
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
    for (int support : {3, 20, 40, 65}) {
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
        for (bool beat : {false, true}) {
            for (int repeat = 0; repeat < 4; ++repeat) {
                double us[2] = {};
                for (int k = 0; k < 2; ++k) {
                    const int arm = k ^ (repeat & 1);
                    std::mt19937 rng(7300); common_sampler_pq_stats stats;
                    const auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < 100000; ++i) {
                        if (arm) { common_sampler_verify_pq(arr, q, 0, 1, rng, beat, stats); }
                        else { verify_pq_reference(arr, q, 0, 1, rng, beat, stats, nullptr, {}); }
                    }
                    const auto end = std::chrono::steady_clock::now();
                    us[arm] = std::chrono::duration<double, std::micro>(end - start).count() / 100000 * 4;
                    assert(stats.n_pos == 100000);
                }
                printf("BENCH support=%d beat=%d repeat=%d reference_us_per_4pos=%.6f cached_us_per_4pos=%.6f\n",
                       support, (int) beat, repeat, us[0], us[1]);
            }
        }
    }
}

int main(int argc, char ** argv) {
    test_model_mtp_default();
    test_verifier_identity(); test_identity_and_ties(); test_laws(); test_mode_distribution();
    if (argc == 2) { test_model_state(argv[1], false); test_model_state(argv[1], true); }
    bench();
    puts("test-spec-beat: PASS");
}
