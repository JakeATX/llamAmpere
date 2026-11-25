#pragma once

// Cost-model draft depth (LLAMA_RTB_COSTDEPTH=1, default off; the server's BL6a controller LLAMA_SPEC_ADAPT_COST
// wins when both are set).
//
// Acceptance: per draft position k, decayed counts of rounds that reached k and of rounds that accepted k, read with
// a prior of PRIOR_W rounds: p_k = (acc_k + PRIOR_W * prior_k) / (reach_k + PRIOR_W), prior_k = min(P0, 0.9 p_{k-1}).
// Counts decay by LLAMA_RTB_COSTDEPTH_DECAY (0.97) per round, so the estimate follows the text.
// Cost: round wall time (draft + verify + sampling) is fitted by least squares with exponential forgetting and a
// small ridge on t = a + b * rows + c * rows * ctx / 1024, so the cost of a depth that has not been run lately is extrapolated
// from the ones that were, and the optimum moves down at deep context where every verify row reads more KV.
// Depth: argmax over d of (1 + sum_{L <= d} prod_{k < L} p_k) / t(d + 1), within [floor, n_max]. A neighbouring
// depth is tried every EXPLORE rounds so the slope stays identifiable. Past rounds only: with exact verification the
// output law is unchanged.
//
// Per-request mode (LLAMA_RTB_REQDEPTH=1, default off; 24DENSE-RAG): the depth window of the MTP streak controller
// ([n_min_adaptive, n_max], 3..7 under --preset 24dense at one slot) gets a ceiling chosen per request from that
// request's own measured acceptance. begin_request() keeps the cost fit (hardware and context, not content) and
// scales the acceptance counts by LLAMA_RTB_REQDEPTH_CARRY (0.25), so a new request starts from a weak prior and its
// own rounds take over; a depth the request has not reached yet is assumed as good as the deepest one it has (an
// optimistic prior, so the ceiling probes upward and measures instead of staying low); inside a request the counts decay slowly (LLAMA_RTB_REQDEPTH_DECAY, 0.995, ~200 rounds) so
// the rarely reached deep positions keep enough samples. A pick only moves when it beats the current ceiling by
// LLAMA_RTB_REQDEPTH_HYST (2%). The ceiling never goes under the window floor (n_min_adaptive, LLAMA_RTB_REQDEPTH_FLOOR
// overrides). It caps the streak depth (the streak controller still climbs and drops under it), so it can only make a
// round shallower than the drafter's own choice. Known: while capped, full accepts at the cap still count toward the
// streak climb, so the streak depth can sit above the cap and the first rounds after the ceiling rises draft deeper.
// Output: with exact verification the output law is unchanged; under mb-MTP the drafted depth decides which positions
// mode bias may keep, so the text differs as with any depth change (a shallower draft is closer to exact sampling).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace rtb {

struct cost_depth {
    static constexpr double PRIOR_W = 4.0;
    static constexpr double P0      = 0.7;
    static constexpr int    EXPLORE = 16;

    double decay  = 0.97;
    double forget = 0.995;
    int    floor_d = 1;

    std::vector<double> acc, reach; // [n_max]
    double S[3][3] = {};             // decayed sum of x x^T
    double R[3]    = {};             // decayed sum of x y
    double theta[3] = { 0.0, 0.0, 0.0 };
    int    n_obs   = 0;
    int    rounds  = 0;
    int    cur     = 0;
    int    explore_dir = 1;
    int    n_max   = 0;
    double hyst    = 0.0;            // relative rate gain a new depth needs over the current one (0 = plain argmax)
    bool   prior_flat = false;       // prior_k = p_{k-1} (optimistic: an unreached depth looks as good as the last one)
    int    n_req   = 0;              // requests begun (begin_request)
    std::vector<int> req_picks;      // [n_max + 1] depths picked in this request

    bool active() const { return n_max > 0; }

    void reset(int n_max_) {
        n_max = std::max(0, n_max_);
        acc.assign(n_max, 0.0);
        reach.assign(n_max, 0.0);
        for (int i = 0; i < 3; ++i) {
            theta[i] = 0.0;
            R[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                S[i][j] = 0.0;
            }
        }
        n_obs = 0;
        rounds = 0;
        cur = n_max;
        explore_dir = 1;
        n_req = 0;
        req_picks.assign(n_max + 1, 0);
    }

    // a new request: keep the cost fit, keep `carry` of the acceptance evidence (0 = this request only)
    void begin_request(double carry) {
        if (!active()) {
            return;
        }
        carry = std::max(0.0, std::min(1.0, carry));
        for (int k = 0; k < n_max; ++k) {
            acc[k]   *= carry;
            reach[k] *= carry;
        }
        req_picks.assign(n_max + 1, 0);
        n_req++;
    }

    double p_acc(int k) const {
        double prior = P0;
        for (int j = 1; j <= k; ++j) {
            const double prev = (acc[j - 1] + PRIOR_W * prior) / (reach[j - 1] + PRIOR_W);
            prior = prior_flat ? prev : std::min(P0, 0.9 * prev);
        }
        return (acc[k] + PRIOR_W * prior) / (reach[k] + PRIOR_W);
    }

    double expected_tokens(int d) const {
        double e = 1.0, r = 1.0;
        for (int k = 0; k < d && k < n_max; ++k) {
            r *= p_acc(k);
            e += r;
        }
        return e;
    }

    static void feat(double rows, double ctx, double x[3]) {
        x[0] = 1.0;
        x[1] = rows;
        x[2] = rows * ctx / 1024.0;
    }

    double cost_us(int d, double ctx) const {
        double x[3];
        feat(d + 1, ctx, x);
        return theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[2];
    }

    // theta = (S + ridge)^-1 R; the ridge only decides directions the data does not (constant context: the split
    // between b and c), the prediction at the observed contexts stays the least-squares one
    void solve() {
        double A[3][4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A[i][j] = S[i][j] + (i == j ? 1e-4 * (S[i][i] + 1.0) : 0.0);
            }
            A[i][3] = R[i];
        }
        for (int c = 0; c < 3; ++c) {
            int piv = c;
            for (int r = c + 1; r < 3; ++r) {
                if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) {
                    piv = r;
                }
            }
            if (std::fabs(A[piv][c]) < 1e-300) {
                return;
            }
            for (int j = 0; j < 4; ++j) {
                std::swap(A[c][j], A[piv][j]);
            }
            for (int r = 0; r < 3; ++r) {
                if (r == c) {
                    continue;
                }
                const double f = A[r][c] / A[c][c];
                for (int j = c; j < 4; ++j) {
                    A[r][j] -= f * A[c][j];
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            theta[i] = A[i][3] / A[i][i];
        }
    }

    // one verified round: n_draft drafted, n_accepted accepted, round_us wall time, ctx = context position
    void observe(int n_draft, int n_accepted, int64_t round_us, double ctx) {
        if (!active() || n_draft <= 0) {
            return;
        }
        for (int k = 0; k < n_max; ++k) {
            acc[k]   *= decay;
            reach[k] *= decay;
        }
        for (int k = 0; k < n_draft && k < n_max; ++k) {
            reach[k] += 1.0;
            if (k < n_accepted) {
                acc[k] += 1.0;
            } else {
                break;
            }
        }
        rounds++;
        if (round_us <= 0) {
            return;
        }
        const double y = (double) round_us;
        if (n_obs >= 4) {
            const double pred = cost_us(n_draft, ctx);
            if (pred > 0.0 && y > 3.0 * pred) {
                return; // a stall or a graph being recorded
            }
        }
        double x[3];
        feat(n_draft + 1, ctx, x);
        for (int i = 0; i < 3; ++i) {
            R[i] = forget * R[i] + x[i] * y;
            for (int j = 0; j < 3; ++j) {
                S[i][j] = forget * S[i][j] + x[i] * x[j];
            }
        }
        n_obs++;
        solve();
    }

    // depth for the next draft
    int pick(double ctx) {
        if (!active()) {
            return 0;
        }
        const int lo = std::min(std::max(1, floor_d), n_max);
        cur = std::max(lo, std::min(cur, n_max));
        if (n_obs < 8) {
            return cur;
        }
        if (rounds % EXPLORE == 0) {
            int cand = cur + explore_dir;
            explore_dir = -explore_dir;
            if (cand < lo || cand > n_max) {
                cand = cur - (cand - cur);
            }
            if (cand >= lo && cand <= n_max) {
                return cand;
            }
        }
        int    best = cur;
        double best_rate = -1.0;
        for (int d = lo; d <= n_max; ++d) {
            const double t = cost_us(d, ctx);
            if (t <= 0.0) {
                continue;
            }
            const double rate = expected_tokens(d) / t;
            if (rate > best_rate) {
                best_rate = rate;
                best = d;
            }
        }
        if (hyst > 0.0 && best != cur) {
            const double t_cur = cost_us(cur, ctx);
            if (t_cur > 0.0 && best_rate <= (1.0 + hyst) * expected_tokens(cur) / t_cur) {
                best = cur;
            }
        }
        cur = best;
        return cur;
    }

    // count a pick toward this request's depth histogram (summary only)
    void note_pick(int d) {
        if (d >= 0 && d < (int) req_picks.size()) {
            req_picks[d]++;
        }
    }

    std::string summary() const {
        std::string out = "depth=" + std::to_string(cur) + " rounds=" + std::to_string(rounds) + " p=(";
        char buf[64];
        for (int k = 0; k < n_max; ++k) {
            snprintf(buf, sizeof(buf), "%s%.3f", k ? " " : "", p_acc(k));
            out += buf;
        }
        snprintf(buf, sizeof(buf), ") t_us=%.0f+%.1f*r+%.3f*r*ctxK", theta[0], theta[1], theta[2]);
        out += buf;
        bool any = false;
        for (int d = 0; d < (int) req_picks.size(); ++d) {
            if (req_picks[d] > 0) {
                snprintf(buf, sizeof(buf), "%sd%d:%d", any ? " " : " request picks=(", d, req_picks[d]);
                out += buf;
                any = true;
            }
        }
        if (any) {
            out += ")";
        }
        return out;
    }
};

} // namespace rtb
