#pragma once

// CDEPTH: confidence-driven draft depth for MTP speculative decoding (--spec-depth-policy cdepth).
//
// The streak controller (speculative-adaptive.h) picks the depth of the NEXT round from how the last rounds went.
// CDEPTH decides INSIDE a round, token by token: after j draft tokens exist it predicts the chance that draft token
// j+1 would be accepted, from
//   * how confident the big (target) model has recently been (rolling mean of its top-1 probability over the last
//     `window` committed tokens, free by-product of exact p/q verification),
//   * how confident the helper (draft) model is at the token it just drafted (top-1 probability and top-1 minus
//     top-2 margin),
//   * the product of the helper's confidences over the whole draft so far.
// and keeps drafting only while that chance is at least a per-depth threshold (the break-even point: the chance at
// which one more draft step pays for itself in expected extra tokens). The floor (--spec-draft-n-min-adaptive) is
// always drafted; --spec-draft-n-max is the ceiling.
//
// Lossless: the rule only decides how many tokens the helper proposes. It looks at the helper's own distributions and
// at already verified history, never at the tokens the verifier has yet to judge, so the exact p/q verification still
// returns the target distribution whatever depth is chosen.
//
// The coefficients come from tools/cdepth/cdepth_sim.py (`fit-causal`), which writes a small text file for
// --spec-depth-model.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct common_speculative_cdepth_model {
    static constexpr int N_FEAT = 5; // 1, rolling target top-1, draft top-1, draft margin, sum of log draft top-1

    int   window = 16;     // committed tokens in the rolling target-confidence mean
    float cold   = 0.5f;   // rolling mean before any history exists

    // stage j (1-based): the decision to draft token j+1 once j draft tokens exist
    std::vector<std::array<float, N_FEAT>> coef;
    std::vector<float>                     theta;
    bool                                   valid = false;

    // chance that draft token j+1 is accepted; stage is clamped to the last fitted stage
    float chance(int j, float mean, float p1, float p2, float sum_log_p1) const {
        const size_t s = (size_t) std::min<int>(std::max(j, 1), (int) coef.size()) - 1;
        const auto & c = coef[s];
        const float x[N_FEAT] = { 1.0f, mean, p1, p1 - p2, std::max(sum_log_p1, -12.0f) };
        float z = 0.0f;
        for (int i = 0; i < N_FEAT; ++i) {
            z += c[i] * x[i];
        }
        return 1.0f / (1.0f + std::exp(-z));
    }

    float threshold(int j, float scale) const {
        const size_t s = (size_t) std::min<int>(std::max(j, 1), (int) theta.size()) - 1;
        return theta[s] * scale;
    }

    // text format (one record per line, '#' starts a comment):
    //   cdepth-model 1
    //   window 16
    //   cold 0.5
    //   stage <j> <c0> <c1> <c2> <c3> <c4> <theta>
    bool load(const std::string & path, std::string & err) {
        std::ifstream f(path);
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        return parse(f, err);
    }

    bool parse(std::istream & f, std::string & err) {
        std::string line;
        bool header = false;
        coef.clear();
        theta.clear();
        while (std::getline(f, line)) {
            const size_t hash = line.find('#');
            if (hash != std::string::npos) {
                line.resize(hash);
            }
            std::istringstream is(line);
            std::string key;
            if (!(is >> key)) {
                continue;
            }
            if (key == "cdepth-model") {
                int version = 0;
                if (!(is >> version) || version != 1) {
                    err = "unsupported cdepth-model version";
                    return false;
                }
                header = true;
            } else if (key == "window") {
                if (!(is >> window) || window < 1 || window > 256) {
                    err = "bad window";
                    return false;
                }
            } else if (key == "cold") {
                if (!(is >> cold) || !(cold >= 0.0f && cold <= 1.0f)) {
                    err = "bad cold";
                    return false;
                }
            } else if (key == "stage") {
                int j = 0;
                std::array<float, N_FEAT> c{};
                float t = 0.0f;
                if (!(is >> j)) {
                    err = "bad stage line";
                    return false;
                }
                for (int i = 0; i < N_FEAT; ++i) {
                    if (!(is >> c[i]) || !std::isfinite(c[i])) {
                        err = "bad stage coefficient";
                        return false;
                    }
                }
                if (!(is >> t) || !std::isfinite(t)) {
                    err = "bad stage threshold";
                    return false;
                }
                if (j != (int) coef.size() + 1) {
                    err = "stages must be listed 1,2,3,...";
                    return false;
                }
                coef.push_back(c);
                theta.push_back(t);
            } else {
                err = "unknown key '" + key + "'";
                return false;
            }
        }
        if (!header || coef.empty()) {
            err = "missing header or stages";
            return false;
        }
        valid = true;
        return true;
    }
};

// per-sequence state: committed target-confidence history plus the draft-so-far features of the current round
struct common_speculative_cdepth_state {
    std::deque<float> hist;
    double            hist_sum = 0.0;
    double            sum_log_p1 = 0.0; // current round

    void reset_history() {
        hist.clear();
        hist_sum = 0.0;
        sum_log_p1 = 0.0;
    }

    void begin_round() {
        sum_log_p1 = 0.0;
    }

    // committed tokens' target top-1 probabilities (entries < 0 are unknown and skipped)
    void note_target(const float * p, size_t n, int window) {
        for (size_t i = 0; i < n; ++i) {
            if (!(p[i] >= 0.0f && p[i] <= 1.0f)) {
                continue;
            }
            hist.push_back(p[i]);
            hist_sum += p[i];
            while ((int) hist.size() > window) {
                hist_sum -= hist.front();
                hist.pop_front();
            }
        }
    }

    float mean(const common_speculative_cdepth_model & m) const {
        return hist.empty() ? m.cold : (float) (hist_sum / (double) hist.size());
    }

    // called right after draft token number j (1-based) was added with top-1 / top-2 probabilities p1 / p2.
    // returns true when drafting should continue to token j+1. j < floor always continues; the caller handles the
    // ceiling.
    bool keep_going(const common_speculative_cdepth_model & m, int j, int floor, float p1, float p2, float scale) {
        sum_log_p1 += std::log(std::max(p1, 1e-6f));
        if (j < floor) {
            return true;
        }
        return m.chance(j, mean(m), p1, p2, (float) sum_log_p1) >= m.threshold(j, scale);
    }
};

// built-in coefficient table used when --spec-depth-policy cdepth is given without --spec-depth-model.
// PLACEHOLDER until the Flash-Next fit lands: stages 1-2 never stop (the floor is drafted anyway).
inline const char * common_speculative_cdepth_builtin_text() {
    return
        "cdepth-model 1\n"
        "window 16\n"
        "cold 0.5\n"
        "stage 1 0 0 0 0 0 0\n"
        "stage 2 0 0 0 0 0 0\n";
}
