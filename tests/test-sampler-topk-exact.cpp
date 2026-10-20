// [DRAFTFAST] common_sampler_topk_exact must give the same ids, logits and order as the real top-k sampler on the
// full {id, logit, 0} array, equal logits, -inf and NaN included, for every k <= 128.
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
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
    return n_bad == 0 ? 0 : 1;
}
