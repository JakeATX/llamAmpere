// [DRAFTFAST] common_sampler_topk_exact must give the same ids, logits and order as the real top-k sampler on the
// full {id, logit, 0} array, equal logits, -inf and NaN included, for every k <= 128.
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool candidates_equal(const llama_token_data_array & a, const llama_token_data_array & b) {
    if (a.size != b.size || a.selected != b.selected || a.sorted != b.sorted) {
        return false;
    }
    for (size_t i = 0; i < a.size; ++i) {
        if (a.data[i].id != b.data[i].id ||
                std::memcmp(&a.data[i].logit, &b.data[i].logit, sizeof(float)) != 0 ||
                std::memcmp(&a.data[i].p, &b.data[i].p, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

static std::vector<llama_token_data> full_candidates(const std::vector<float> & logits) {
    std::vector<llama_token_data> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = { (llama_token) i, logits[i], 0.0f };
    }
    return result;
}

static int test_topk() {
    std::mt19937 rng(20261020);
    int n_bad = 0;
    int n_test = 0;
    for (int t = 0; t < 4000; ++t) {
        int n = 1 + (int) (rng() % 6000);
        if (t % 20 == 0) {
            n = 248320;
        }
        int k = 1 + (int) (rng() % 128);
        if (k > n) {
            k = n;
        }
        std::vector<float> logits(n);
        const int mode = t % 5;
        for (auto & x : logits) {
            switch (mode) {
                case 0:  x = std::normal_distribution<float>(0.0f, 3.0f)(rng); break;
                case 1:  x = (float) (rng() % 7); break;                         // heavy ties
                case 2:  x = (float) (rng() % 50) * 0.5f; break;
                case 3:  x = (rng() % 3 == 0) ? -INFINITY : (float) (rng() % 20); break;
                default: x = (rng() % 997 == 0) ? NAN : std::normal_distribution<float>(0.0f, 2.0f)(rng); break;
            }
        }
        if (mode == 0) {
            for (int j = 0; j < 300 && n > 1000; ++j) {
                logits[rng() % n] = std::normal_distribution<float>(14.0f, 3.0f)(rng);
            }
        }

        std::vector<llama_token_data> ref(n);
        for (int i = 0; i < n; ++i) {
            ref[i] = llama_token_data{ i, logits[i], 0.0f };
        }
        llama_token_data_array ref_p = { ref.data(), ref.size(), -1, false };
        llama_sampler * top_k = llama_sampler_init_top_k(k);
        llama_sampler_apply(top_k, &ref_p);
        llama_sampler_free(top_k);

        std::vector<llama_token_data> got;
        common_sampler_topk_exact(logits.data(), n, k, got);

        ++n_test;
        bool ok = ref_p.size == got.size();
        for (size_t i = 0; ok && i < got.size(); ++i) {
            ok = ref_p.data[i].id == got[i].id && std::memcmp(&ref_p.data[i].logit, &got[i].logit, sizeof(float)) == 0;
        }
        if (!ok) {
            ++n_bad;
            fprintf(stderr, "mismatch: test %d n %d k %d mode %d\n", t, n, k, mode);
        }
    }
    printf("test-sampler-topk-exact: %d cases, %d mismatches\n", n_test, n_bad);
    return n_bad;
}

static int test_topk_biased() {
    std::mt19937 rng(20261021);
    int n_bad = 0;
    int n_test = 0;
    std::vector<float> scratch;
    std::vector<llama_token_data> got;

    assert(common_sampler_topk_biases_supported({}));
    assert(common_sampler_topk_biases_supported({ { -1, 1.0f }, { 0, -INFINITY }, { 248320, 2.0f } }));
    assert(common_sampler_topk_biases_supported({ { 20, -INFINITY }, { 20, -INFINITY } }));
    assert(!common_sampler_topk_biases_supported({ { 20, 1.0f }, { 20, -INFINITY } }));
    assert(!common_sampler_topk_biases_supported({ { 20, -INFINITY }, { 20, 1.0f } }));
    assert(!common_sampler_topk_biases_supported({ { 20, NAN }, { 20, NAN } }));
    assert(!common_sampler_topk_biases_supported({ { -1, 1.0f }, { -1, 2.0f } }));

    for (const int n : { 41, 127, 248320 }) {
        for (const int k_request : { 1, 40, 128 }) {
            const int k = std::min(n, k_request);
            for (int mode = 0; mode < 8; ++mode) {
                std::vector<float> logits(n);
                for (auto & x : logits) {
                    x = mode % 2 == 0 ? (float) (rng() % 7) : std::normal_distribution<float>(0.0f, 3.0f)(rng);
                }
                std::vector<llama_logit_bias> biases = {
                    { 0, -INFINITY }, { 1, mode < 2 ? 8.0f : INFINITY }, { n / 2, -8.0f },
                    { n - 1, 0.0f }, { -1, INFINITY }, { n, -INFINITY },
                };
                if (mode >= 3) {
                    biases.push_back({ 0, -INFINITY });
                }
                if (mode == 4) logits[n / 2] = NAN;
                if (mode == 5) logits[0] = INFINITY;
                if (mode == 6) logits[0] = NAN;
                if (mode == 7) logits[0] = -INFINITY;
                auto ref = full_candidates(logits);
                llama_token_data_array ref_p = { ref.data(), ref.size(), -1, false };
                auto * bias = llama_sampler_init_logit_bias(n, biases.size(), biases.data());
                auto * top_k = llama_sampler_init_top_k(k);
                llama_sampler_apply(bias, &ref_p);
                llama_sampler_apply(top_k, &ref_p);

                const bool fused = common_sampler_topk_exact_biased(logits.data(), n, k, biases, scratch, got);
                assert(fused == (mode != 5 && mode != 6));
                if (!fused) {
                    got = full_candidates(logits);
                }
                for (const auto & c : got) {
                    assert(std::memcmp(&c.logit, &logits[c.id], sizeof(float)) == 0);
                }
                llama_token_data_array got_p = { got.data(), got.size(), -1, fused };
                llama_sampler_apply(bias, &got_p);
                llama_sampler_apply(top_k, &got_p);
                if (!candidates_equal(ref_p, got_p)) {
                    ++n_bad;
                    fprintf(stderr, "biased top-k mismatch: n %d k %d mode %d\n", n, k, mode);
                }
                ++n_test;
                llama_sampler_free(bias);
                llama_sampler_free(top_k);
            }
        }
    }
    printf("test-sampler-topk-exact biased: %d cases, %d mismatches\n", n_test, n_bad);
    return n_bad;
}

// Ban-only selection must leave the full-vocabulary scratch untouched, including when
// the heap initially contains bans or all surviving values tie at -inf.
static int test_topk_bans() {
    int n_bad = 0;
    int n_test = 0;
    for (const int n : { 1, 41, 73, 248320 }) {
        for (const int k_request : { 1, 5, 40, 128 }) {
            const int k = std::min(n, k_request);
            for (int mode = 0; mode < 5; ++mode) {
                if (mode == 2 && n > 73) { continue; }
                std::vector<float> logits(n);
                for (int i = 0; i < n; ++i) {
                    logits[i] = mode == 3 ? -INFINITY : float((i * 7) % 11);
                }
                std::vector<llama_logit_bias> biases;
                const auto add = [&](int id) {
                    if (id >= 0 && id < n) { biases.push_back({ id, -INFINITY }); }
                };
                for (const int id : { n - 1, k + 64, k + 63, k + 32, k + 31, k, k - 1, 0 }) { add(id); }
                if (mode == 1 || mode == 2) {
                    for (int id = 0; id < (mode == 1 ? k : n); ++id) { add(id); }
                }
                if (mode == 4) {
                    // Unbanned nonfinite values retain the original comparator/heap behavior.
                    for (int id = 1; id < n; id += 13) {
                        if (std::none_of(biases.begin(), biases.end(), [&](const llama_logit_bias & b) { return b.token == id; })) {
                            logits[id] = id % 2 ? NAN : INFINITY;
                        }
                    }
                }
                // Invalid bans and repeated valid bans must be ignored/deduplicated only for the scan.
                biases.push_back({ -1, -INFINITY });
                biases.push_back({ n, -INFINITY });
                add(0);
                assert(common_sampler_topk_biases_supported(biases));
                auto ref = full_candidates(logits);
                llama_token_data_array ref_p = { ref.data(), ref.size(), -1, false };
                auto * bias = llama_sampler_init_logit_bias(n, biases.size(), biases.data());
                auto * top_k = llama_sampler_init_top_k(k);
                llama_sampler_apply(bias, &ref_p);
                llama_sampler_apply(top_k, &ref_p);
                std::vector<float> scratch = { 314.0f, -0.0f };
                const auto scratch_before = scratch;
                std::vector<llama_token_data> got;
                assert(common_sampler_topk_exact_biased(logits.data(), n, k, biases, scratch, got));
                assert(scratch.size() == scratch_before.size() &&
                        std::memcmp(scratch.data(), scratch_before.data(), scratch.size() * sizeof(float)) == 0);
                for (const auto & c : got) { assert(std::memcmp(&c.logit, &logits[c.id], sizeof(float)) == 0); }
                llama_token_data_array got_p = { got.data(), got.size(), -1, true };
                llama_sampler_apply(bias, &got_p);
                llama_sampler_apply(top_k, &got_p);
                if (!candidates_equal(ref_p, got_p)) {
                    ++n_bad;
                    fprintf(stderr, "ban-only top-k mismatch: n %d k %d mode %d\n", n, k, mode);
                }
                llama_sampler_free(bias);
                llama_sampler_free(top_k);
                ++n_test;
            }
        }
    }
    for (const float invalid : { NAN, INFINITY }) {
        std::vector<float> logits(73, 0.0f), scratch = { 314.0f };
        logits[0] = invalid;
        std::vector<llama_token_data> got = { { 7, 1.0f, 0.5f } };
        assert(!common_sampler_topk_exact_biased(logits.data(), logits.size(), 5,
                { { 0, -INFINITY }, { 0, -INFINITY } }, scratch, got));
        assert(scratch.size() == 1 && scratch[0] == 314.0f);
        assert(got.size() == 1 && got[0].id == 7 && got[0].logit == 1.0f && got[0].p == 0.5f);
    }
    printf("test-sampler-topk-exact bans: %d cases, %d mismatches\n", n_test, n_bad);
    return n_bad;
}

static llama_sampler * sampling_chain(int n, int k, const std::vector<llama_logit_bias> & biases) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n, biases.size(), biases.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(20261021));
    return chain;
}

static int test_biased_chain() {
    std::mt19937 rng(20261022);
    int n_bad = 0;
    int n_test = 0;
    std::vector<float> scratch;
    std::vector<llama_token_data> got;

    for (const int n : { 96, 248320 }) {
        const int k = 40;
        std::vector<llama_logit_bias> biases = {
            { 0, -INFINITY }, { 1, -1.5f }, { n / 2, 4.0f }, { n - 1, 0.5f }, { -1, 20.0f }, { n, 20.0f },
        };
        if (n == 248320) {
            biases.clear();
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (llama_token token : { 248044, 248046, 248063, 248064, 248065 }) {
                    biases.push_back({ token, -INFINITY });
                }
            }
        }
        assert(common_sampler_topk_biases_supported(biases));
        for (int mode = 0; mode < 3; ++mode) {
            auto * ref_chain = sampling_chain(n, k, biases);
            auto * got_chain = sampling_chain(n, k, biases);
            for (int step = 0; step < 16; ++step) {
                std::vector<float> logits(n);
                for (auto & x : logits) {
                    x = mode == 0 ? std::normal_distribution<float>(0.0f, 3.0f)(rng) : (float) (rng() % 7);
                }
                if (mode == 2) {
                    for (int i = 2; i < n; i += 7) {
                        logits[i] = -INFINITY;
                    }
                }
                logits[n == 248320 ? 248044 : 0] = 100.0f;
                auto ref = full_candidates(logits);
                llama_token_data_array ref_p = { ref.data(), ref.size(), -1, false };
                const bool fused = common_sampler_topk_exact_biased(logits.data(), n, k, biases, scratch, got);
                assert(fused);
                llama_token_data_array got_p = { got.data(), got.size(), -1, true };
                llama_sampler_apply(ref_chain, &ref_p);
                llama_sampler_apply(got_chain, &got_p);
                if (!candidates_equal(ref_p, got_p)) {
                    ++n_bad;
                    fprintf(stderr, "biased chain mismatch: n %d k %d mode %d step %d\n", n, k, mode, step);
                }
                assert(ref_p.selected >= 0 && got_p.selected >= 0);
                const auto ref_token = ref_p.data[ref_p.selected].id;
                const auto got_token = got_p.data[got_p.selected].id;
                if (ref_token != got_token) {
                    ++n_bad;
                    fprintf(stderr, "biased RNG mismatch: n %d mode %d step %d\n", n, mode, step);
                }
                llama_sampler_accept(ref_chain, ref_token);
                llama_sampler_accept(got_chain, got_token);
                if (step == 7) {
                    auto * ref_copy = llama_sampler_clone(ref_chain);
                    auto * got_copy = llama_sampler_clone(got_chain);
                    llama_sampler_free(ref_chain);
                    llama_sampler_free(got_chain);
                    ref_chain = ref_copy;
                    got_chain = got_copy;
                }
                ++n_test;
            }
            llama_sampler_free(ref_chain);
            llama_sampler_free(got_chain);
        }
    }
    printf("test-sampler-topk-exact biased chain: %d draws, %d mismatches\n", n_test, n_bad);
    return n_bad;
}

int main() {
    const int n_bad = test_topk() + test_topk_biased() + test_topk_bans() + test_biased_chain();
    return n_bad == 0 ? 0 : 1;
}
