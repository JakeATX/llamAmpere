// [DSPINE-V07] lattice walk, verify-length planner and p/q acceptance (CPU, no model).
//  1. cost table parse + interpolation, calibration update
//  2. planner length choice (step cost at 9 rows, flat cost, floors)
//  3. the walk with the planner off is the pre-DSPINE-V07 inline walk, token for token and q for q (greedy, p_min, sampled)
//  4. T=1 exactness end to end: lattice walk (sampled) + common_sampler_verify_pq, repeated rounds, must reproduce the
//     target's joint law of tokens 1-3 and 3-5 (chi-square) with the planner (fixed calibration, the accept() feedback
//     loop, cost tables that give draft lengths 0..4), with p_min, and with n_min + p_min under the F1 rule (a sampled
//     draft is never dropped). Two negative controls (truncate on the drawn token; the pre-F1 n_min discard) must be
//     detected.
//  5. [DSPINE2] mb (mode biasing, the mb-MTP accept rule, common_sampler_verify_pq beat=true) on the DSpine lattice path:
//     the first emitted token follows the mb local law common_sampler_pq_mode_distribution(p, q) (chi-square), and
//     the joint law of tokens 1-3 differs from the target's (the exact chi-square must catch it). The mb tolerance is
//     documented as in mb-MTP: a local law per position, its TV from p printed; the server's LLAMA_SPEC_MB_KL audit
//     lines (mb-mtp local-law) report the same per-position divergence on real text.
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
    CHECK(bad.pts.size() == common_lattice_cost::defaults().pts.size() && bad.pts.front().first == 1, "failed parse leaves out untouched");

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
    CHECK(common_lattice_best_len(a.data(), 15, 0, c) <= 1, "low confidence drafts little (the default table prices 1..4 rows, F6)");
    std::fill(a.begin(), a.end(), 0.45f);
    const int32_t l_low = common_lattice_best_len(a.data(), 15, 0, c);
    CHECK(l_low >= 1 && l_low <= 3, "a=0.45 drafts 1..3 with the default table, got %d", l_low);
    CHECK(common_lattice_best_len(a.data(), 15, 6, c) >= 6, "floor respected");
    common_lattice_cost flat;
    flat.pts = { { 1, 10.0f } };
    // a = 0.5 keeps every extra position's gain (0.5^15 ~ 3e-5) above double round-off, so each row strictly helps
    std::fill(a.begin(), a.end(), 0.5f);
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
// after a full acceptance emit a bonus draw from p. Rounds repeat until N_OUT tokens are out.
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

// one exactness configuration (DSPINE-V07 review F2)
struct exact_cfg {
    const char * name     = "";
    bool   plan           = true;  // planner on
    bool   mixed_cost     = false; // per round, a cost table drawn from an independent RNG (covers lengths 0..4)
    bool   feedback       = false; // calibration updated from each round's accept count (reset per trial), as accept() does
    float  p_min          = 0.0f;
    int    n_min          = 0;     // engine floor: planner l_min; with discard_short the pre-fix "too short -> empty draft"
    bool   discard_short  = false; // negative control: the pre-F1 rule (drop a sampled draft shorter than n_min)
    bool   trunc_on_draw  = false; // negative control: drop a position after seeing the drawn token
    bool   beat           = false; // [DSPINE2] mb accept rule: not exact by design (expected to DIFFER)
    double * tv123_out    = nullptr; // [DSPINE2] TV of the tokens 1-3 joint law from the target's
};

static constexpr int32_t N_OUT = 5;

// chi-square of observed counts vs expected probabilities, cells with expectation < 5 left out (checked for 0 instead)
static void chi2_add(double pe, double n, int64_t n_trials, double & chi2, int & cells, double & tv) {
    tv += std::fabs(n / n_trials - pe);
    if (pe * n_trials >= 5.0) {
        chi2 += (n - pe * n_trials) * (n - pe * n_trials) / (pe * n_trials);
        cells++;
    } else {
        CHECK(pe > 0.0 || n == 0, "a zero-probability sequence was emitted");
    }
}

// returns false if the emitted law differs from the target's (two chi-squares: tokens 1-3 and tokens 3-5)
static bool run_exactness(const exact_cfg & cfg, int64_t * len_hist_out = nullptr) {
    const int32_t top_k = 4;
    const int32_t n_pos = 4;
    const size_t  stride = top_k + top_k * top_k;
    common_lattice_cost cost;
    cost.pts = { { 1, 10.0f }, { 2, 10.5f }, { 3, 11.0f }, { 4, 16.0f }, { 5, 16.5f } };
    cost.draft_ms = 2.0f;
    // forced tables: flat up to k + 1 rows, then a wall, so the planner drafts up to k (k = 0..4)
    std::vector<common_lattice_cost> forced(5);
    for (int32_t k = 0; k < 5; ++k) {
        forced[k].pts = { { k + 1, 10.0f }, { k + 2, 1000.0f } };
        forced[k].draft_ms = 0.0f;
    }
    common_lattice_calib calib0; // fixed, uneven betas so the stopping rule varies
    {
        const float c[4] = { 0.4f, 0.6f, 0.2f, 0.9f };
        calib0.update(c, 4, 3);
        calib0.update(c, 4, 1);
    }
    std::vector<std::vector<float>> lat_by_anchor;
    for (int32_t a = 0; a <= V; ++a) {
        lat_by_anchor.push_back(make_lattice(n_pos, top_k, V, stride, 1000 + a, 1.5f));
    }

    std::mt19937 g_draft(11), g_tgt(22), g_ver(33), g_cost(44);
    common_sampler_pq_stats stats;
    const int64_t n_trials = 300000;
    std::map<int32_t, int64_t> c123, c345;
    std::vector<llama_token_data> pd(V);
    int64_t len_hist[8] = {};
    int64_t n_rounds = 0;
    for (int64_t t = 0; t < n_trials; ++t) {
        std::vector<int32_t> outv;
        int32_t prev = -1;
        common_lattice_calib calib = calib0;
        if (cfg.feedback) {
            calib.reset();
        }
        while ((int32_t) outv.size() < N_OUT) {
            const auto & lat = lat_by_anchor[prev < 0 ? V : prev];
            common_lattice_walk_params wp;
            wp.top_k = top_k; wp.temp = 1.0f; wp.sample = true; wp.p_min = cfg.p_min;
            if (cfg.plan) {
                wp.cost  = &cost;
                wp.calib = &calib;
                wp.l_min = cfg.n_min;
                if (cfg.mixed_cost) {
                    // independent of the draft: half the rounds use the data-dependent table, half a forced length
                    const int32_t r = std::uniform_int_distribution<int32_t>(0, 9)(g_cost);
                    if (r < 5) {
                        wp.cost = &forced[r];
                    }
                }
            }
            std::vector<llama_token> d;
            std::vector<std::vector<llama_token_data>> q;
            std::vector<float> conf;
            common_lattice_walk(lat.data(), stride, n_pos, wp, &g_draft, d, &q, cfg.plan ? &conf : nullptr);
            if (cfg.trunc_on_draw) {
                for (size_t i = 0; i < d.size(); ++i) {
                    if (q[i].empty()) {
                        break;
                    }
                    float qd = 0.0f;
                    for (const auto & e : q[i]) {
                        qd = e.id == d[i] ? e.p : qd;
                    }
                    if (qd < 0.35f) {
                        d.resize(i);
                        q.resize(i);
                        break;
                    }
                }
            }
            if (cfg.discard_short && (int) d.size() < cfg.n_min) {
                d.clear();
                q.clear();
            }
            len_hist[std::min<size_t>(d.size(), 7)]++;
            n_rounds++;
            bool all = true;
            int32_t n_acc = 0;
            for (size_t i = 0; i < d.size() && (int32_t) outv.size() < N_OUT; ++i) {
                const auto p = target_row(prev);
                for (int32_t v = 0; v < V; ++v) {
                    pd[v] = { v, 0.0f, (float) p[v] };
                }
                llama_token_data_array arr = { pd.data(), (size_t) V, -1, false };
                const int32_t x = draw(p, g_tgt);
                const auto res = common_sampler_verify_pq(arr, q[i], d[i], x, g_ver, cfg.beat, stats);
                outv.push_back(res.id);
                prev = res.id;
                if (!res.accepted) {
                    all = false;
                    break;
                }
                n_acc++;
            }
            if (all && (int32_t) outv.size() < N_OUT) {
                const int32_t x = draw(target_row(prev), g_tgt);
                outv.push_back(x);
                prev = x;
            }
            if (cfg.feedback && cfg.plan) {
                // as accept(): the verified width is the whole draft
                conf.resize(d.size());
                calib.update(conf.data(), (int32_t) d.size(), n_acc);
            }
        }
        c123[outv[0] * V * V + outv[1] * V + outv[2]]++;
        c345[outv[2] * V * V + outv[3] * V + outv[4]]++;
    }
    // marginal of token 3 under the target chain
    std::vector<double> m(V, 0.0);
    for (int32_t a = 0; a < V; ++a) {
        for (int32_t b = 0; b < V; ++b) {
            for (int32_t c = 0; c < V; ++c) {
                m[c] += target_row(-1)[a] * target_row(a)[b] * target_row(b)[c];
            }
        }
    }
    double chi2_a = 0.0, chi2_b = 0.0, tv_a = 0.0, tv_b = 0.0;
    int    cells_a = 0, cells_b = 0;
    for (int32_t a = 0; a < V; ++a) {
        for (int32_t b = 0; b < V; ++b) {
            for (int32_t c = 0; c < V; ++c) {
                const int32_t key = a * V * V + b * V + c;
                chi2_add(target_row(-1)[a] * target_row(a)[b] * target_row(b)[c], (double) c123[key], n_trials, chi2_a, cells_a, tv_a);
                chi2_add(m[a] * target_row(a)[b] * target_row(b)[c], (double) c345[key], n_trials, chi2_b, cells_b, tv_b);
            }
        }
    }
    const double df_a = cells_a - 1, df_b = cells_b - 1;
    // mean df, sd sqrt(2 df); 6 sd is a fixed-seed bound with no false alarms in practice
    const bool ok = chi2_a < df_a + 6.0 * std::sqrt(2.0 * df_a) && chi2_b < df_b + 6.0 * std::sqrt(2.0 * df_b);
    fprintf(stderr, "exactness %-28s: tokens 1-3 chi2 %8.1f (df %3.0f) TV %.4f | tokens 3-5 chi2 %8.1f (df %3.0f) TV %.4f | "
            "draft lengths 0..4 (%%): %.1f %.1f %.1f %.1f %.1f -> %s\n",
            cfg.name, chi2_a, df_a, tv_a / 2, chi2_b, df_b, tv_b / 2,
            100.0 * len_hist[0] / n_rounds, 100.0 * len_hist[1] / n_rounds, 100.0 * len_hist[2] / n_rounds,
            100.0 * len_hist[3] / n_rounds, 100.0 * len_hist[4] / n_rounds, ok ? "matches target" : "DIFFERS from target");
    if (cfg.tv123_out) {
        *cfg.tv123_out = tv_a / 2;
    }
    if (len_hist_out) {
        for (int i = 0; i < 8; ++i) {
            len_hist_out[i] = len_hist[i] * 1000 / std::max<int64_t>(1, n_rounds); // per mille
        }
    }
    return ok;
}

static void test_exactness() {
    exact_cfg c;
    c.name = "planner, fixed calib";
    CHECK(run_exactness(c), "planned sampled walk + p/q verify is not exact (%s)", c.name);

    // (a) lengths 0..4 all exercised
    exact_cfg cm;
    cm.name = "planner, mixed cost tables";
    cm.mixed_cost = true;
    int64_t h[8] = {};
    CHECK(run_exactness(cm, h), "not exact (%s)", cm.name);
    for (int L = 0; L <= 4; ++L) {
        CHECK(h[L] >= 10, "mixed cost tables: draft length %d in only %lld per mille of rounds", L, (long long) h[L]);
    }

    // (b) the calibration feedback loop of accept()
    exact_cfg cf;
    cf.name = "planner, calib feedback";
    cf.feedback = true;
    cf.mixed_cost = true;
    CHECK(run_exactness(cf), "not exact (%s)", cf.name);

    // (c) p_min stop (planner off, and planner on)
    exact_cfg cp;
    cp.name = "p_min 0.5, no planner";
    cp.plan = false;
    cp.p_min = 0.5f;
    CHECK(run_exactness(cp), "not exact (%s)", cp.name);
    exact_cfg cpp = cp;
    cpp.name = "p_min 0.5 + planner";
    cpp.plan = true;
    cpp.feedback = true;
    CHECK(run_exactness(cpp), "not exact (%s)", cpp.name);

    // (d) n_min 2 with p_min: the fixed rule keeps short sampled drafts (planner floor l_min = n_min)
    exact_cfg cn = cpp;
    cn.name = "n_min 2 + p_min 0.5 (fixed)";
    cn.n_min = 2;
    CHECK(run_exactness(cn), "not exact (%s)", cn.name);

    // negative controls: each must be detected, else the check has no power for that failure
    exact_cfg b1;
    b1.name = "CONTROL truncate on draw";
    b1.trunc_on_draw = true;
    CHECK(!run_exactness(b1), "biased control (truncate on the drawn token) was not detected");
    exact_cfg b2 = cn;
    b2.name = "CONTROL n_min discard (pre-F1)";
    b2.plan = false;
    b2.feedback = false;
    b2.discard_short = true;
    CHECK(!run_exactness(b2), "biased control (pre-F1 n_min discard of a sampled draft) was not detected");
}

// [DSPINE2] mb on the lattice path. (a) First position: walk the anchor lattice (planner off), verify position 0 with
// beat on; the emitted token must follow common_sampler_pq_mode_distribution(p, q0), the mb-MTP local law, and not p.
// (b) Whole rounds with the planner: mb is detected as non-exact by the tokens 1-3 / 3-5 chi-square; its TV is printed.
static void test_mb() {
    const int32_t top_k = 4, n_pos = 4;
    const size_t  stride = top_k + top_k * top_k;
    const auto lat = make_lattice(n_pos, top_k, V, stride, 1000 + V, 1.5f);
    const auto p = target_row(-1);
    std::vector<llama_token_data> pd(V);
    for (int32_t v = 0; v < V; ++v) {
        pd[v] = { v, 0.0f, (float) p[v] };
    }
    llama_token_data_array arr = { pd.data(), (size_t) V, -1, false };
    std::mt19937 g_draft(5), g_tgt(6), g_ver(7);
    common_sampler_pq_stats stats;
    const int64_t n = 400000;
    for (int beat = 0; beat < 2; ++beat) {
        std::vector<int64_t> cnt(V, 0);
        std::vector<llama_token_data> q0;
        for (int64_t t = 0; t < n; ++t) {
            common_lattice_walk_params wp;
            wp.top_k = top_k; wp.temp = 1.0f; wp.sample = true;
            std::vector<llama_token> d;
            std::vector<std::vector<llama_token_data>> q;
            common_lattice_walk(lat.data(), stride, n_pos, wp, &g_draft, d, &q, nullptr);
            if (t == 0) {
                q0 = q[0];
            }
            const auto res = common_sampler_verify_pq(arr, q[0], d[0], draw(p, g_tgt), g_ver, beat != 0, stats);
            cnt[res.id]++;
        }
        const auto law = common_sampler_pq_mode_distribution(arr, q0);
        double chi_law = 0.0, chi_p = 0.0, tv = 0.0;
        int cells = 0;
        for (int32_t v = 0; v < V; ++v) {
            const double e = beat ? law[v] : p[v]; // exact emits p; mb emits its local law
            tv += std::fabs(law[v] - p[v]) / 2;
            if (e * n >= 5.0) { chi_law += std::pow(cnt[v] - e * n, 2) / (e * n); cells++; }
            else { CHECK(cnt[v] == 0 || e > 0.0, "a token with zero expected mass was emitted"); }
            if (p[v] * n >= 5.0) { chi_p += std::pow(cnt[v] - p[v] * n, 2) / (p[v] * n); }
        }
        const double df = cells - 1, bound = df + 6.0 * std::sqrt(2.0 * df);
        fprintf(stderr, "mb lattice pos 0 (beat %d): chi2 vs expected law %.1f, vs p %.1f (df %.0f, bound %.1f); "
                "predicted law TV from p %.4f; kept by mb %zu\n", beat, chi_law, chi_p, df, bound, tv, stats.n_lossy);
        CHECK(chi_law < bound, "lattice + verify_pq (beat %d) does not follow the predicted local law", beat);
        if (beat) {
            CHECK(tv > 0.01 && chi_p > bound, "mb lattice: the mb law is not distinguishable from p (no power)");
            CHECK(stats.n_lossy > 0, "mb lattice: no draft kept by the mb rule");
        } else {
            CHECK(chi_p < bound, "exact lattice: emitted law differs from p");
        }
    }
    // (b) whole rounds with the planner and calibration feedback
    exact_cfg cb;
    cb.name = "mb planner, calib feedback";
    cb.feedback = true;
    cb.mixed_cost = true;
    cb.beat = true;
    double tv = 0.0;
    cb.tv123_out = &tv;
    CHECK(!run_exactness(cb), "mb was not detected as non-exact (no power)");
    fprintf(stderr, "mb planner rounds: tokens 1-3 joint TV from the target %.4f (mb tolerance, documented; not exact)\n", tv);
}

int main() {
    test_cost_and_calib();
    test_best_len();
    test_walk_matches_reference();
    test_planner_walk();
    test_exactness();
    test_mb();
    if (n_fail) {
        fprintf(stderr, "test-spec-lattice: %d FAIL\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-spec-lattice: PASS\n");
    return 0;
}
