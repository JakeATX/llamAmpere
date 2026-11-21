// [HOSTGAP08] Target top-K verify (LLAMA_MTP_TOPK_VERIFY, default on): the card sends the K = top_k best logits of each
// verification row and the CPU chain runs on those K candidates instead of on the full row. This test checks, on the CPU,
// that the sampler law is unchanged:
//   1. exactness: for the preset chain (top-k 20, top-p 0.95, min-p 0, temp 1, dist), the K candidates in the order the
//      card returns them (value desc, lower id first; or any order) give the same token as the shipped path (heap top-k
//      of the full row, sorted) for the same seed, on every row without equal logits among the kept ones;
//   2. law: the no-dist chain on the K candidates keeps the same tokens with the same logits as on the full row, and
//      draws through the candidate path follow the full row's truncated softmax (chi-square), with and without request
//      bans (-INFINITY logit bias, applied on the card before the K best are kept and again by the CPU chain).
// The card's candidate set is emulated on the CPU (its kernels are checked against the CPU top-k in test-backend-ops).
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
#include <functional>
#include <map>
#include <random>
#include <vector>

static llama_sampler * make_chain(uint32_t seed, bool with_dist, const std::vector<llama_logit_bias> & bias = {}, int n_vocab = 0) {
    auto * c = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!bias.empty()) {
        llama_sampler_chain_add(c, llama_sampler_init_logit_bias(n_vocab, (int32_t) bias.size(), bias.data()));
    }
    llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_min_p(0.0f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_temp_ext(1.0f, 0.0f, 1.0f));
    if (with_dist) {
        llama_sampler_chain_add(c, llama_sampler_init_dist(seed));
    }
    return c;
}

static std::vector<float> make_row(std::mt19937 & rng, int n, int mode) {
    std::vector<float> x(n);
    for (auto & v : x) {
        switch (mode) {
            case 0:  v = std::normal_distribution<float>(0.0f, 3.0f)(rng); break;      // LM-like body ...
            case 1:  v = (float) (rng() % 9); break;                                  // heavy ties
            default: v = std::normal_distribution<float>(-2.0f, 1.5f)(rng); break;
        }
    }
    if (mode != 1) {  // ... with a head of a few dozen likely tokens
        for (int j = 0; j < 40; ++j) {
            x[rng() % n] = std::normal_distribution<float>(mode == 0 ? 14.0f : 9.0f, 2.0f)(rng);
        }
    }
    return x;
}

// the card's K best: value desc, lower id first on equal values (ggml-cuda top-k split / sort order)
static std::vector<llama_token_data> card_topk(const std::vector<float> & x, int k) {
    std::vector<int> id(x.size());
    for (size_t i = 0; i < id.size(); ++i) {
        id[i] = (int) i;
    }
    std::partial_sort(id.begin(), id.begin() + k, id.end(), [&](int a, int b) {
        return x[a] > x[b] || (x[a] == x[b] && a < b);
    });
    std::vector<llama_token_data> c(k);
    for (int i = 0; i < k; ++i) {
        c[i] = { id[i], x[id[i]], 0.0f };
    }
    return c;
}

static bool kept_has_ties(const std::vector<float> & x, int k) {
    std::vector<float> s(x);
    std::partial_sort(s.begin(), s.begin() + k + 1, s.end(), std::greater<float>());
    for (int i = 0; i < k; ++i) {
        if (s[i] == s[i + 1]) {
            return true;
        }
    }
    return false;
}

static llama_token sample(llama_sampler * chain, std::vector<llama_token_data> cand, bool sorted) {
    llama_token_data_array p = { cand.data(), cand.size(), -1, sorted };
    llama_sampler_apply(chain, &p);
    assert(p.selected >= 0 && p.selected < (int64_t) p.size);
    return p.data[p.selected].id;
}

static int test_exact() {
    std::mt19937 rng(20261121);
    int n_rows = 0, n_tie_rows = 0, n_draw = 0, n_diff = 0, n_diff_ties = 0;
    for (int t = 0; t < 300; ++t) {
        const int n    = t % 10 == 0 ? 248320 : 2000 + (int) (rng() % 30000);
        const int mode = t % 3;
        const auto x   = make_row(rng, n, mode);
        const bool ties = kept_has_ties(x, 20);
        ++n_rows;
        n_tie_rows += ties;

        std::vector<llama_token_data> ref;
        common_sampler_topk_exact(x.data(), n, 20, ref);   // the shipped CPU path: heap top-k of the full row, sorted
        auto cand = card_topk(x, 20);
        auto perm = cand;
        std::shuffle(perm.begin(), perm.end(), rng);

        for (int s = 0; s < 40; ++s) {
            const uint32_t seed = 1000u * t + s;
            llama_sampler * a = make_chain(seed, true);
            llama_sampler * b = make_chain(seed, true);
            llama_sampler * c = make_chain(seed, true);
            const llama_token ta = sample(a, ref,  true);
            const llama_token tb = sample(b, cand, false);
            const llama_token tc = sample(c, perm, false);
            llama_sampler_free(a);
            llama_sampler_free(b);
            llama_sampler_free(c);
            ++n_draw;
            if (ta != tb || ta != tc) {
                if (ties) {
                    ++n_diff_ties;
                } else {
                    ++n_diff;
                    fprintf(stderr, "exact: row %d n %d mode %d seed %u: shipped %d card %d card-shuffled %d\n",
                            t, n, mode, seed, ta, tb, tc);
                }
            }
        }
    }
    printf("test-sampler-topk-law exact: %d rows (%d with equal kept logits), %d draws: %d differ on rows without ties, "
           "%d on rows with ties (allowed: equal logits, equal probability)\n", n_rows, n_tie_rows, n_draw, n_diff, n_diff_ties);
    return n_diff;
}

// the no-dist chain on a candidate list: kept (id, logit) sorted by id
static std::vector<std::pair<llama_token, float>> kept(std::vector<llama_token_data> c, const std::vector<llama_logit_bias> & bias, int n) {
    llama_token_data_array p = { c.data(), c.size(), -1, false };
    llama_sampler * nd = make_chain(0, false, bias, n);
    llama_sampler_apply(nd, &p);
    llama_sampler_free(nd);
    std::vector<std::pair<llama_token, float>> r;
    for (size_t i = 0; i < p.size; ++i) {
        r.emplace_back(p.data[i].id, p.data[i].logit);
    }
    std::sort(r.begin(), r.end());
    return r;
}

static int test_law() {
    std::mt19937 rng(7);
    int n_bad = 0;
    for (int mode = 0; mode < 6; ++mode) {
        const int n = 248320;
        auto x = make_row(rng, n, mode % 3);
        std::vector<llama_logit_bias> bias;
        if (mode >= 3) {  // ban the current best two tokens and a few others, like the end-of-turn ids of a request
            auto top = card_topk(x, 2);
            bias = { { top[0].id, -INFINITY }, { top[1].id, -INFINITY }, { 248044, -INFINITY }, { 248046, -INFINITY } };
        }
        // card path: bias on the card, K best of the biased row, original-to-biased logits exported, then the CPU chain
        // (whose logit-bias sampler applies the bans again)
        std::vector<float> xb = x;
        for (const auto & b : bias) {
            xb[b.token] += b.bias;
        }
        const auto cand = card_topk(xb, 20);
        // full-row path
        std::vector<llama_token_data> full(n);
        for (int i = 0; i < n; ++i) {
            full[i] = { i, x[i], 0.0f };
        }
        const auto ref = kept(full, bias, n);
        const auto got = kept(cand, bias, n);
        bool same = ref.size() == got.size();
        for (size_t i = 0; same && i < ref.size(); ++i) {
            same = ref[i].first == got[i].first && std::memcmp(&ref[i].second, &got[i].second, sizeof(float)) == 0;
        }
        const bool ties = kept_has_ties(xb, 20);
        const auto & lawsrc = ties ? got : ref;  // with ties at the cut the law is checked on the card's own set

        double mx = -INFINITY;
        for (const auto & r : lawsrc) {
            mx = std::max(mx, (double) r.second);
        }
        std::map<llama_token, double> law;
        double z = 0.0;
        for (const auto & r : lawsrc) {
            z += std::exp((double) r.second - mx);
        }
        for (const auto & r : lawsrc) {
            law[r.first] = std::exp((double) r.second - mx) / z;
        }

        const int M = 200000;
        std::map<llama_token, int> hist;
        llama_sampler * chain = make_chain(12345u + mode, true, bias, n);
        for (int i = 0; i < M; ++i) {
            ++hist[sample(chain, cand, false)];
        }
        llama_sampler_free(chain);

        double chi2 = 0.0;
        int    outside = 0;
        for (const auto & [id, c] : hist) {
            if (law.find(id) == law.end()) {
                outside += c;
            }
        }
        for (const auto & [id, p] : law) {
            const double e = p * M;
            const double o = hist.count(id) ? hist[id] : 0;
            chi2 += (o - e) * (o - e) / e;
        }
        const int    dof = std::max<int>(1, (int) law.size() - 1);
        const double zs  = (chi2 - dof) / std::sqrt(2.0 * dof);
        // rows with equal logits at the cut may keep a different (equal-logit) token than the full row: law only
        const bool   ok  = (same || ties) && outside == 0 && zs < 6.0;
        n_bad += !ok;
        printf("test-sampler-topk-law law: row kind %d%s, kept %zu (full row) / %zu (card), same tokens+logits %s%s, %d draws: "
               "chi2 %.2f dof %d (z %.2f), outside the law %d: %s\n",
               mode % 3, bias.empty() ? "" : " + bans", ref.size(), got.size(), same ? "yes" : "no", ties ? " (ties at the cut)" : "",
               M, chi2, dof, zs, outside, ok ? "ok" : "FAIL");
    }
    return n_bad;
}

int main() {
    const int bad = test_exact() + test_law();
    printf("test-sampler-topk-law: %s\n", bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 1;
}
