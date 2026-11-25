#pragma once

// DFlash2 / DSpine lattice walk and the opt-in verify-length planner (LLAMA_SPEC_LATTICE_PLAN=1).
//
// Lattice row layout (one row per draft position i = 1..n_pos, n_embd floats apart):
//   [ cand[0..k) as float token ids, scores[k * k] predecessor-major ]
// scores + j * k are the k candidate logits of this position given candidate j of the previous position
// (row 1: every predecessor row holds the anchor's scores, so predecessor 0 is used).
//
// Exactness (temperature > 0, sampled walk, p/q verification): every quantity the planner reads before it decides
// whether to draft position i is a function of the drafter's lattice, of the tokens already drafted at positions < i
// and of the calibration state left by earlier rounds. It never reads the token it is about to draw, and it draws no
// random numbers. So the draft length is a stopping time of the draft process, each kept position i is drawn from the
// q_i recorded for it, and p/q verification emits the target distribution at every position (see
// S/DSPINE/V07/DESIGN_dspine_v07.md, "T=1 exactness").

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

// verify cost model: milliseconds of one target verify of `rows` rows (draft + 1), piecewise linear over the given
// points, flat outside them; draft_ms is the fixed per-round drafting cost
struct common_lattice_cost {
    std::vector<std::pair<int32_t, float>> pts; // (rows, ms), rows strictly increasing
    float draft_ms = 5.5f;

    float verify_ms(int32_t rows) const {
        if (pts.empty()) {
            return 1.0f;
        }
        if (rows <= pts.front().first) {
            return pts.front().second;
        }
        if (rows >= pts.back().first) {
            return pts.back().second;
        }
        for (size_t i = 1; i < pts.size(); ++i) {
            if (rows <= pts[i].first) {
                const auto & a = pts[i - 1];
                const auto & b = pts[i];
                const float t = float(rows - a.first) / float(b.first - a.first);
                return a.second + t * (b.second - a.second);
            }
        }
        return pts.back().second;
    }

    // "rows:ms,rows:ms,..." with rows strictly increasing and ms > 0; false (out untouched) on any error
    static bool parse(const char * s, common_lattice_cost & out) {
        if (s == nullptr || *s == '\0') {
            return false;
        }
        std::vector<std::pair<int32_t, float>> pts;
        std::string str(s);
        size_t pos = 0;
        while (pos <= str.size()) {
            const size_t end = std::min(str.find(',', pos), str.size());
            const std::string item = str.substr(pos, end - pos);
            const size_t colon = item.find(':');
            if (colon == std::string::npos) {
                return false;
            }
            char * e1 = nullptr;
            char * e2 = nullptr;
            const std::string r = item.substr(0, colon);
            const std::string m = item.substr(colon + 1);
            const long  rows = std::strtol(r.c_str(), &e1, 10);
            const float ms   = std::strtof(m.c_str(), &e2);
            if (r.empty() || m.empty() || *e1 != '\0' || *e2 != '\0' || rows < 1 || rows > 1024 || !(ms > 0.0f) ||
                    !std::isfinite(ms) || (!pts.empty() && rows <= pts.back().first)) {
                return false;
            }
            pts.emplace_back((int32_t) rows, ms);
            pos = end + 1;
        }
        out.pts = std::move(pts);
        return true;
    }

    // DSPINE 11-09 measurement (Taichi 3090, Swift 27B IQ4_XS-M, SJ-KVaRN 4/4, coding 64K, verify ms by rows 5..16);
    // the 1-row point is extrapolated from the 5..8 slope (about 1.6 ms per row) so that short drafts are priced too.
    // Only the shape matters to the planner. Set LLAMA_SPEC_VERIFY_COST (measured, stage A1) for another file, card or
    // context.
    static common_lattice_cost defaults() {
        common_lattice_cost c;
        c.pts = { { 1, 24.8f }, { 5, 31.2f }, { 6, 32.7f }, { 8, 36.4f }, { 9, 49.6f }, { 16, 54.3f } };
        c.draft_ms = 5.5f;
        return c;
    }
};

// per-sequence acceptance calibration: the planner predicts the chance that draft position j is accepted, given
// that every earlier one was, as min(1, beta_j * conf_j), conf_j = the drafter's probability of the drafted token.
// beta_j = (accepted_j + prior) / (sum of conf_j over rounds that reached j + prior), decayed per round.
struct common_lattice_calib {
    static constexpr int32_t N = 32;
    double acc [N] = {};
    double conf[N] = {};
    double prior = 4.0;
    double decay = 0.98;

    void reset() {
        std::fill(acc,  acc  + N, 0.0);
        std::fill(conf, conf + N, 0.0);
    }

    float beta(int32_t j) const {
        if (j < 0 || j >= N) {
            return 1.0f;
        }
        return (float) ((acc[j] + prior) / (conf[j] + prior));
    }

    // c[0..n_drafted) = conf of the drafted tokens; the first n_accepted were accepted, position n_accepted (if
    // drafted) was rejected, later ones were never checked
    void update(const float * c, int32_t n_drafted, int32_t n_accepted) {
        if (c == nullptr || n_drafted <= 0) {
            return;
        }
        n_accepted = std::max(0, std::min(n_accepted, n_drafted));
        const int32_t reached = std::min(N, std::min(n_drafted, n_accepted + 1));
        for (int32_t j = 0; j < N; ++j) {
            acc [j] *= decay;
            conf[j] *= decay;
        }
        for (int32_t j = 0; j < reached; ++j) {
            conf[j] += c[j];
            acc [j] += j < n_accepted ? 1.0 : 0.0;
        }
    }
};

// the draft length L in [l_min, n_pos] that maximises (expected tokens per round) / (draft_ms + verify_ms(L + 1)),
// expected tokens = 1 + sum_{j <= L} prod_{m <= j} a_m; a[0..n_pos) are the per-position acceptance predictions.
// Ties keep the shorter draft.
static inline int32_t common_lattice_best_len(const float * a, int32_t n_pos, int32_t l_min, const common_lattice_cost & cost) {
    l_min = std::max(0, std::min(l_min, n_pos));
    double e    = 1.0;
    double prod = 1.0;
    for (int32_t j = 0; j < l_min; ++j) {
        prod *= a[j];
        e    += prod;
    }
    int32_t best   = l_min;
    double  best_r = e / (cost.draft_ms + cost.verify_ms(l_min + 1));
    for (int32_t j = l_min; j < n_pos; ++j) {
        prod *= a[j];
        e    += prod;
        const double r = e / (cost.draft_ms + cost.verify_ms(j + 2));
        if (r > best_r) {
            best_r = r;
            best   = j + 1;
        }
    }
    return best;
}

struct common_lattice_walk_params {
    int32_t top_k  = 0;
    float   temp   = 1.0f;  // softmax temperature of the candidate scores (the request's when sampling)
    float   p_min  = 0.0f;  // stop before a position whose top candidate has a lower probability
    bool    sample = false; // sample each position and record q, else take the argmax

    // planner (nullptr = off: the walk is the plain DFlash2 walk)
    const common_lattice_cost  * cost  = nullptr;
    const common_lattice_calib * calib = nullptr;
    int32_t l_min = 0;      // never plan fewer drafts than this
};

// softmax(scores / temp) probability of the argmax, and the argmax
static inline float common_lattice_top_prob(const float * scores, int32_t k, float temp, int32_t * arg) {
    const int32_t am = (int32_t) std::distance(scores, std::max_element(scores, scores + k));
    float sum = 0.0f;
    for (int32_t c = 0; c < k; ++c) {
        sum += std::exp((scores[c] - scores[am]) / temp);
    }
    if (arg) {
        *arg = am;
    }
    return 1.0f / sum;
}

// walk lattice rows 1..n_pos (row i at lattice + i * stride). Appends the drafted tokens to `out`; when sampling, one
// q row per drafted token to *q_out; when planning, the drafted tokens' probabilities to *conf_out.
static inline void common_lattice_walk(const float * lattice, size_t stride, int32_t n_pos,
        const common_lattice_walk_params & wp, std::mt19937 * rng,
        std::vector<llama_token> & out,
        std::vector<std::vector<llama_token_data>> * q_out,
        std::vector<float> * conf_out) {
    const int32_t top_k = wp.top_k;
    const float   temp  = wp.temp;
    const bool    plan  = wp.cost != nullptr;

    std::vector<float>   probs;
    std::vector<int32_t> sorted_idx;
    std::vector<float>   a;      // planner: acceptance predictions for positions 1..n_pos (0-based)

    if (plan) {
        a.assign(n_pos, 0.0f);
    }

    int32_t predecessor = 0;
    for (int32_t i = 1; i <= n_pos; ++i) {
        const float * row    = lattice + (size_t) i * stride;
        const float * scores = row + top_k + (size_t) predecessor * top_k;

        if (plan) {
            // forecast positions i..n_pos along the argmax chain from the token drafted at i - 1; positions < i keep
            // the predictions of the tokens actually drafted there
            int32_t pf = predecessor;
            for (int32_t j = i; j <= n_pos; ++j) {
                const float * rj = lattice + (size_t) j * stride;
                int32_t am = 0;
                const float c = common_lattice_top_prob(rj + top_k + (size_t) pf * top_k, top_k, temp, &am);
                a[j - 1] = std::min(1.0f, (wp.calib ? wp.calib->beta(j - 1) : 1.0f) * c);
                pf = am;
            }
            const int32_t best = common_lattice_best_len(a.data(), n_pos, std::max(i - 1, wp.l_min), *wp.cost);
            if (best < i) {
                break;
            }
        }

        predecessor = (int32_t) std::distance(scores, std::max_element(scores, scores + top_k));
        if (wp.p_min > 0.0f || wp.sample || plan) {
            // softmax(scores / temp) at the argmax, i.e. 1 / sum(exp((s_k - s_max) / temp))
            probs.resize(top_k);
            float sum = 0.0f;
            for (int32_t k = 0; k < top_k; ++k) {
                probs[k] = std::exp((scores[k] - scores[predecessor]) / temp);
                sum += probs[k];
            }
            if (1.0f / sum < wp.p_min) {
                break;
            }
            for (int32_t k = 0; k < top_k; ++k) {
                probs[k] /= sum;
            }
        }
        if (wp.sample) {
            // ggml_top_k leaves candidates unordered, so walk them by token id to keep a seed's draft stable
            sorted_idx.resize(top_k);
            for (int32_t k = 0; k < top_k; ++k) {
                sorted_idx[k] = k;
            }
            std::sort(sorted_idx.begin(), sorted_idx.end(), [row](int32_t x, int32_t y) { return row[x] < row[y]; });

            // a round-off overshoot keeps the argmax, which has q > 0
            double u = std::uniform_real_distribution<double>(0.0, 1.0)(*rng);
            for (const int32_t k : sorted_idx) {
                u -= probs[k];
                if (u < 0.0) {
                    predecessor = k;
                    break;
                }
            }

            if (q_out) {
                auto & q = q_out->emplace_back();
                q.reserve(top_k);
                for (int32_t k = 0; k < top_k; ++k) {
                    q.push_back({ (llama_token) row[k], scores[k], probs[k] });
                }
            }
        }
        if (plan) {
            const float c = probs[predecessor];
            a[i - 1] = std::min(1.0f, (wp.calib ? wp.calib->beta(i - 1) : 1.0f) * c);
            if (conf_out) {
                conf_out->push_back(c);
            }
        }
        out.push_back((llama_token) row[predecessor]);
    }
}
