// [DSPINE-V07] lattice walk, verify-length planner and p/q acceptance (CPU, no model).
//  1. cost table parse + interpolation, calibration update
//  2. planner length choice (step cost at 9 rows, flat cost, floors)
//  3. the walk with the planner off is the pre-DSPINE-V07 inline walk, token for token and q for q (greedy, p_min, sampled)
//  4. T=1 exactness end to end: lattice walk (sampled, planner on) + common_sampler_verify_pq, repeated rounds, must
//     reproduce the target's joint law of the first 3 tokens (chi-square); a negative control that truncates on the
//     drawn token must be detectably biased
#include "sampling.h"
#include "spec-lattice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

// the DFlash2 walk as it was inline in common/speculative.cpp before DSPINE-V07 (reference copy)
static void ref_walk(const float * lattice, size_t stride, int32_t n_block_tokens, int32_t top_k, float temp, float p_min,
        bool sample, std::mt19937 * rng, std::vector<llama_token> & result, std::vector<std::vector<llama_token_data>> * result_q) {
    std::vector<float>   probs;
    std::vector<int32_t> sorted_idx;
    int32_t predecessor = 0;
    for (int32_t i = 1; i < n_block_tokens; ++i) {
        const float * row = lattice + (size_t) i * stride;
        const float * scores = row + top_k + (size_t) predecessor * top_k;
        predecessor = (int32_t) std::distance(scores, std::max_element(scores, scores + top_k));
        if (p_min > 0.0f || sample) {
            probs.resize(top_k);
            float sum = 0.0f;
            for (int32_t k = 0; k < top_k; ++k) {
                probs[k] = std::exp((scores[k] - scores[predecessor]) / temp);
                sum += probs[k];
            }
            if (1.0f / sum < p_min) {
                break;
            }
            for (int32_t k = 0; k < top_k; ++k) {
                probs[k] /= sum;
            }
        }
        if (sample) {
            sorted_idx.resize(top_k);
            for (int32_t k = 0; k < top_k; ++k) {
                sorted_idx[k] = k;
            }
            std::sort(sorted_idx.begin(), sorted_idx.end(), [row](int32_t a, int32_t b) { return row[a] < row[b]; });
            double u = std::uniform_real_distribution<double>(0.0, 1.0)(*rng);
            for (const int32_t k : sorted_idx) {
                u -= probs[k];
                if (u < 0.0) {
                    predecessor = k;
                    break;
                }
            }
            auto & q = result_q->emplace_back();
            for (int32_t k = 0; k < top_k; ++k) {
                q.push_back({ (llama_token) row[k], scores[k], probs[k] });
            }
        }
        result.push_back((llama_token) row[predecessor]);
    }
}

// a lattice of n_pos rows (row 0 unused) over a vocab of n_vocab, top_k distinct candidates per row, seeded
static std::vector<float> make_lattice(int32_t n_pos, int32_t top_k, int32_t n_vocab, size_t stride, uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> lat((size_t) (n_pos + 1) * stride, 0.0f);
    std::vector<int32_t> ids(n_vocab);
    for (int32_t i = 1; i <= n_pos; ++i) {
        float * row = lat.data() + (size_t) i * stride;
        for (int32_t v = 0; v < n_vocab; ++v) {
            ids[v] = v;
        }
        std::shuffle(ids.begin(), ids.end(), g);
        for (int32_t k = 0; k < top_k; ++k) {
            row[k] = (float) ids[k];
        }
        for (int32_t j = 0; j < top_k; ++j) {
            for (int32_t k = 0; k < top_k; ++k) {
                // row 1: every predecessor row is the anchor's
                row[top_k + j * top_k + k] = i == 1 && j > 0 ? row[top_k + k] : nd(g);
            }
        }
    }
    return lat;
}

static void test_cost_and_calib() {
    common_lattice_cost c;
    CHECK(common_lattice_cost::parse("5:31.2,6:32.7,8:36.4,9:49.6,16:54.3", c), "parse default table");
    CHECK(std::fabs(c.verify_ms(1) - 31.2f) < 1e-4f, "flat below: %f", c.verify_ms(1));
    CHECK(std::fabs(c.verify_ms(7) - 34.55f) < 1e-3f, "interp 7: %f", c.verify_ms(7));
    CHECK(std::fabs(c.verify_ms(9) - 49.6f) < 1e-4f, "point 9");
    CHECK(std::fabs(c.verify_ms(40) - 54.3f) < 1e-4f, "flat above");
    common_lattice_cost bad = common_lattice_cost::defaults();
    CHECK(!common_lattice_cost::parse("5:31,5:32", bad), "rows must increase");
    CHECK(!common_lattice_cost::parse("5:31,x", bad), "garbage");
    CHECK(!common_lattice_cost::parse("0:31", bad), "rows >= 1");
    CHECK(!common_lattice_cost::parse("5:-1", bad), "ms > 0");
    CHECK(!common_lattice_cost::parse("", bad), "empty");
    CHECK(bad.pts.size() == 5, "failed parse leaves out untouched");

    common_lattice_calib cal;
    CHECK(std::fabs(cal.beta(0) - 1.0f) < 1e-6f, "prior beta 1");
    const float conf[4] = { 0.9f, 0.8f, 0.5f, 0.5f };
    cal.update(conf, 4, 1); // pos 0 accepted, pos 1 rejected, 2..3 unchecked
    CHECK(std::fabs(cal.beta(0) - (1.0 + 4.0) / (0.9 + 4.0)) < 1e-5, "beta0 %f", cal.beta(0));
    CHECK(std::fabs(cal.beta(1) - (0.0 + 4.0) / (0.8 + 4.0)) < 1e-5, "beta1 %f", cal.beta(1));
    CHECK(std::fabs(cal.beta(2) - 1.0f) < 1e-6f, "unreached beta stays at prior");
    cal.update(conf, 4, 4); // all accepted: positions 0..3 reached and accepted
    CHECK(cal.beta(3) > 1.0f, "accepted pos raises beta");
    cal.reset();
    CHECK(std::fabs(cal.beta(1) - 1.0f) < 1e-6f, "reset");
}

static void test_best_len() {
    const common_lattice_cost c = common_lattice_cost::defaults();
    std::vector<float> a(15, 0.75f);
    const int32_t l_mid = common_lattice_best_len(a.data(), 15, 0, c);
    CHECK(l_mid >= 1 && l_mid <= 7, "a=0.75 with the 9-row step stays <= 8 rows, got %d", l_mid);
    std::fill(a.begin(), a.end(), 0.995f);
    CHECK(common_lattice_best_len(a.data(), 15, 0, c) == 15, "a~1 takes the full block");
    std::fill(a.begin(), a.end(), 0.05f);
    CHECK(common_lattice_best_len(a.data(), 15, 0, c) <= 4, "low confidence drafts little");
    CHECK(common_lattice_best_len(a.data(), 15, 6, c) >= 6, "floor respected");
    common_lattice_cost flat;
    flat.pts = { { 1, 10.0f } };
    std::fill(a.begin(), a.end(), 0.01f);
    CHECK(common_lattice_best_len(a.data(), 15, 0, flat) == 15, "free rows: draft everything");
    // hand check: a = {1, 0} with cost 10 ms flat, draft 0: L=1 gives 2 tokens / 10 > L=0 gives 1 / 10
    common_lattice_cost f2;
    f2.pts = { { 1, 10.0f } };
    f2.draft_ms = 0.0f;
    const float a2[2] = { 1.0f, 0.0f };
    CHECK(common_lattice_best_len(a2, 2, 0, f2) == 1, "ties keep the shorter draft");
}

static void test_walk_matches_reference() {
    const int32_t top_k = 16;
    const size_t  stride = top_k + top_k * top_k + 8;
    for (uint32_t seed = 1; seed <= 200; ++seed) {
        const auto lat = make_lattice(15, top_k, 300, stride, seed, seed % 2 ? 1.0f : 3.0f);
        for (int mode = 0; mode < 3; ++mode) {
            const bool  sample = mode == 2;
            const float p_min  = mode == 1 ? 0.3f : 0.0f;
            const float temp   = sample ? 0.7f : 1.0f;
            std::mt19937 r1(seed * 7 + 1), r2(seed * 7 + 1);
            std::vector<llama_token> a, b;
            std::vector<std::vector<llama_token_data>> qa, qb;
            ref_walk(lat.data(), stride, 16, top_k, temp, p_min, sample, &r1, a, &qa);
            common_lattice_walk_params wp;
            wp.top_k = top_k; wp.temp = temp; wp.p_min = p_min; wp.sample = sample;
            common_lattice_walk(lat.data(), stride, 15, wp, sample ? &r2 : nullptr, b, sample ? &qb : nullptr, nullptr);
            CHECK(a == b, "seed %u mode %d: tokens differ from the reference walk", seed, mode);
            CHECK(qa.size() == qb.size(), "seed %u mode %d: q rows", seed, mode);
            for (size_t i = 0; i < std::min(qa.size(), qb.size()); ++i) {
                for (size_t k = 0; k < qa[i].size(); ++k) {
                    CHECK(qa[i][k].id == qb[i][k].id && qa[i][k].p == qb[i][k].p && qa[i][k].logit == qb[i][k].logit,
                          "seed %u: q row %zu entry %zu", seed, i, k);
                }
            }
            if (sample) {
                CHECK(r1() == r2(), "seed %u: rng streams diverged", seed);
            }
        }
    }
}

static void test_planner_walk() {
    const int32_t top_k = 8;
    const size_t  stride = top_k + top_k * top_k;
    const auto lat = make_lattice(15, top_k, 100, stride, 99, 6.0f);
    common_lattice_cost flat;
    flat.pts = { { 1, 10.0f } };
    common_lattice_walk_params wp;
    wp.top_k = top_k; wp.cost = &flat;
    std::vector<llama_token> out, ref;
    std::vector<float> conf;
    common_lattice_walk(lat.data(), stride, 15, wp, nullptr, out, nullptr, &conf);
    ref_walk(lat.data(), stride, 16, top_k, 1.0f, 0.0f, false, nullptr, ref, nullptr);
    CHECK(out == ref, "flat cost: planner walk = full greedy walk");
    CHECK(conf.size() == out.size(), "one conf per token");
    // a steep cost per row: the planner must stop early, and its tokens are a prefix of the greedy walk
    common_lattice_cost steep;
    steep.pts = { { 1, 1.0f }, { 16, 1000.0f } };
    steep.draft_ms = 0.0f;
    wp.cost = &steep;
    std::vector<llama_token> out2;
    common_lattice_walk(lat.data(), stride, 15, wp, nullptr, out2, nullptr, nullptr);
    CHECK(out2.size() < 15, "steep cost stops early (%zu)", out2.size());
    CHECK(std::equal(out2.begin(), out2.end(), ref.begin()), "planner output is a prefix of the walk");
    wp.l_min = 15;
    std::vector<llama_token> out3;
    common_lattice_walk(lat.data(), stride, 15, wp, nullptr, out3, nullptr, nullptr);
    CHECK(out3.size() == 15, "l_min floor keeps every position");
}

// ---- end-to-end exactness at temperature 1 ----
// target: a Markov chain over V tokens, p(x_t | x_{t-1}); the drafter's lattice for a round is a seeded function of
// the round's anchor (the last emitted token). One round: walk (sampled, planner on) -> p/q verification of each
// position (target draw x from p, common_sampler_verify_pq) -> on a rejection emit the replacement and end the round,
// after a full acceptance emit a bonus draw from p. Rounds repeat until 3 tokens are out.
static constexpr int32_t V = 6;

static std::vector<double> target_row(int32_t prev) {
    static std::vector<std::vector<double>> P;
    if (P.empty()) {
        std::mt19937 g(4242);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        P.assign(V + 1, std::vector<double>(V));
        for (int32_t a = 0; a <= V; ++a) {
            double s = 0.0;
            for (int32_t v = 0; v < V; ++v) {
                P[a][v] = (v == (a + 2) % V) ? 0.0 : std::pow(u(g), 2.0) + 0.02; // one zero-probability token per row
                s += P[a][v];
            }
            for (auto & x : P[a]) {
                x /= s;
            }
        }
    }
    return P[prev < 0 ? V : prev];
}

static int32_t draw(const std::vector<double> & p, std::mt19937 & g) {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(g);
    for (int32_t v = 0; v < (int32_t) p.size(); ++v) {
        u -= p[v];
        if (u < 0.0) {
            return v;
        }
    }
    return (int32_t) p.size() - 1;
}

static double run_exactness(bool biased, double * chi2_out) {
    const int32_t top_k = 4;
    const int32_t n_pos = 4;
    const size_t  stride = top_k + top_k * top_k;
    common_lattice_cost cost;
    cost.pts = { { 1, 10.0f }, { 2, 10.5f }, { 3, 11.0f }, { 4, 16.0f }, { 5, 16.5f } };
    cost.draft_ms = 2.0f;
    common_lattice_calib calib; // fixed, uneven betas so the stopping rule varies
    {
        const float c[4] = { 0.4f, 0.6f, 0.2f, 0.9f };
        calib.update(c, 4, 3);
        calib.update(c, 4, 1);
    }
    std::vector<std::vector<float>> lat_by_anchor;
    for (int32_t a = 0; a <= V; ++a) {
        lat_by_anchor.push_back(make_lattice(n_pos, top_k, V, stride, 1000 + a, 1.5f));
    }

    std::mt19937 g_draft(11), g_tgt(22), g_ver(33);
    common_sampler_pq_stats stats;
    const int64_t n_trials = 300000;
    std::map<int32_t, int64_t> counts;
    std::vector<llama_token_data> pd(V);
    int64_t len_hist[8] = {};
    for (int64_t t = 0; t < n_trials; ++t) {
        std::vector<int32_t> outv;
        int32_t prev = -1;
        while (outv.size() < 3) {
            const auto & lat = lat_by_anchor[prev < 0 ? V : prev];
            common_lattice_walk_params wp;
            wp.top_k = top_k; wp.temp = 1.0f; wp.sample = true; wp.cost = &cost; wp.calib = &calib;
            std::vector<llama_token> d;
            std::vector<std::vector<llama_token_data>> q;
            std::vector<float> conf;
            common_lattice_walk(lat.data(), stride, n_pos, wp, &g_draft, d, &q, &conf);
            if (biased) {
                // negative control: drop a position after seeing the drawn token (not a stopping time)
                for (size_t i = 0; i < d.size(); ++i) {
                    if (conf[i] < 0.35f) {
                        d.resize(i);
                        q.resize(i);
                        break;
                    }
                }
            }
            len_hist[std::min<size_t>(d.size(), 7)]++;
            bool all = true;
            for (size_t i = 0; i < d.size() && outv.size() < 3; ++i) {
                const auto p = target_row(prev);
                for (int32_t v = 0; v < V; ++v) {
                    pd[v] = { v, 0.0f, (float) p[v] };
                }
                llama_token_data_array arr = { pd.data(), (size_t) V, -1, false };
                const int32_t x = draw(p, g_tgt);
                const auto res = common_sampler_verify_pq(arr, q[i], d[i], x, g_ver, false, stats);
                outv.push_back(res.id);
                prev = res.id;
                if (!res.accepted) {
                    all = false;
                    break;
                }
            }
            if (all && outv.size() < 3) {
                const int32_t x = draw(target_row(prev), g_tgt);
                outv.push_back(x);
                prev = x;
            }
        }
        counts[outv[0] * V * V + outv[1] * V + outv[2]]++;
    }
    double chi2 = 0.0;
    double tv   = 0.0;
    int    cells = 0;
    for (int32_t a = 0; a < V; ++a) {
        for (int32_t b = 0; b < V; ++b) {
            for (int32_t c = 0; c < V; ++c) {
                const double pe = target_row(-1)[a] * target_row(a)[b] * target_row(b)[c];
                const double n  = (double) counts[a * V * V + b * V + c];
                tv += std::fabs(n / n_trials - pe);
                if (pe * n_trials >= 5.0) {
                    chi2 += (n - pe * n_trials) * (n - pe * n_trials) / (pe * n_trials);
                    cells++;
                } else {
                    CHECK(pe > 0.0 || n == 0, "a zero-probability sequence was emitted");
                }
            }
        }
    }
    fprintf(stderr, "exactness%s: chi2 %.1f over %d cells (df %d), TV %.4f, draft lengths 0..4: %lld %lld %lld %lld %lld\n",
            biased ? " (biased control)" : "", chi2, cells, cells - 1, tv / 2, (long long) len_hist[0], (long long) len_hist[1],
            (long long) len_hist[2], (long long) len_hist[3], (long long) len_hist[4]);
    *chi2_out = chi2;
    return (double) cells - 1;
}

static void test_exactness() {
    double chi2 = 0.0;
    const double df = run_exactness(false, &chi2);
    // mean df, sd sqrt(2 df); 6 sd is a fixed-seed bound with no false alarms in practice
    CHECK(chi2 < df + 6.0 * std::sqrt(2.0 * df), "planned sampled walk + p/q verify is not exact: chi2 %.1f df %.0f", chi2, df);
    double chi2_b = 0.0;
    const double df_b = run_exactness(true, &chi2_b);
    if (!(chi2_b > df_b + 6.0 * std::sqrt(2.0 * df_b))) {
        fprintf(stderr, "WARN: the biased control was not detected (chi2 %.1f): the exactness check has little power here\n", chi2_b);
    }
}

int main() {
    test_cost_and_calib();
    test_best_len();
    test_walk_matches_reference();
    test_planner_walk();
    test_exactness();
    if (n_fail) {
        fprintf(stderr, "test-spec-lattice: %d FAIL\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-spec-lattice: PASS\n");
    return 0;
}
