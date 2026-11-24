#pragma once

// Copy-first controller pieces, each behind its own environment flag, all default OFF. With every flag unset the
// copy-first path in speculative.cpp is the old one (span doubling + cool-down, prompt-only 16-token match).
//
//   LLAMA_RTB_COPYCTL=1     cost-model copy controller. Every round's copy proposal is scored against the tokens
//                           that were committed afterwards, also on rounds that did not copy (shadow scoring), so
//                           the controller knows how often a copy would have been followed without spending a probe
//                           pass. Acceptance is kept per backward-match-length bucket (4-7 / 8-15 / 16-31 / 32+) for
//                           the first copied token plus one continuation rate, separately for greedy and sampled
//                           requests. Round wall time is measured per verify row count for copy rounds and per
//                           emitted token for drafter rounds. A span of W tokens is verified only when its expected
//                           tokens per ms, (1 + q1 * (1 + qn + .. + qn^(W-1))) / t(W + 1), beats margin / (drafter ms
//                           per token). Knobs: LLAMA_RTB_COPYCTL_MARGIN (1.05), LLAMA_RTB_COPYCTL_ROWK (0.04, the
//                           per-row cost slope of the prior before a row count is measured), LLAMA_RTB_COPYCTL_SPAN
//                           (32, the copy-first cap the flag turns on when copy-first is on auto).
//   LLAMA_RTB_COPYCURSOR=1  copy cursor + re-anchor. While the output keeps following a source, the next proposal
//                           continues from there with no lookup and no minimum-match gate. After a break, a 4-gram
//                           index over prompt + reply is asked for the nearest candidate at or after the place where
//                           the copy stopped (while that stop is <= 256 tokens behind), unless its backward match is
//                           shorter than min(8, the longest one); otherwise the longest backward match, newest on
//                           ties. Source: prompt + reply (LLAMA_RTB_COPYCURSOR_SRC=prompt keeps the prompt only).
//   LLAMA_RTB_EDITCOPY=1    edit copy. The committed output is aligned against the prompt source it copies from; a
//                           mismatch that re-joins the source within 6 tokens on each side (on a 4-token anchor)
//                           after >= 4 unchanged tokens becomes a replacement rule. A rule seen at >= 2 different
//                           source places is applied (longest source text first) inside later copy spans; it stops
//                           when unchanged copies of its source text outnumber a quarter of its uses or after 2
//                           rejected first edited tokens. 3 rounds without edit spans after a miss.
//
// The verify is unchanged (point-mass id match on every span position), so none of this changes the output law.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtb {

inline bool env_on(const char * name) {
    const char * s = std::getenv(name);
    if (s == nullptr || s[0] == '\0') {
        return false;
    }
    return std::strcmp(s, "1") == 0 || std::strcmp(s, "on") == 0 || std::strcmp(s, "true") == 0 || std::strcmp(s, "yes") == 0;
}

inline double env_num(const char * name, double def, double lo, double hi) {
    const char * s = std::getenv(name);
    if (s == nullptr || s[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s) {
        return def;
    }
    return std::max(lo, std::min(hi, v));
}

// The token stream the controller works on: history h[0, n) followed by the last sampled token (index n).
struct stream_view {
    const llama_token * h = nullptr;
    int32_t             n = 0;
    llama_token         last = 0;

    int32_t size() const { return n + 1; }
    llama_token operator[](int32_t i) const { return i < n ? h[i] : last; }
};

// ---------------------------------------------------------------------------------------------------------------
// 4-gram anchor index. Each position p >= 3 of the stream is filed under the key of the 4 tokens ending at p;
// positions with the same key are linked newest first. Keys are hashed; a hash collision shows up as a backward
// match shorter than 4 and is skipped.
struct anchor_index {
    static constexpr int32_t GRAM = 4;

    std::unordered_map<uint64_t, int32_t> newest; // key -> newest indexed position
    std::vector<int32_t>                  older;  // position -> next older position with the same key (-1 = none)
    int32_t                               upto = 0; // positions < upto are indexed

    static uint64_t key_at(const stream_view & s, int32_t p) {
        uint64_t k = 0x6a09e667f3bcc909ull;
        for (int32_t j = p - GRAM + 1; j <= p; ++j) {
            k = (k ^ (uint64_t) (uint32_t) s[j]) * 0x100000001b3ull;
            k ^= k >> 31;
            k *= 0xd6e8feb86659fd93ull;
        }
        return k ^ (k >> 29);
    }

    void clear() {
        newest.clear();
        older.clear();
        upto = 0;
    }

    // index positions [upto, end)
    void extend(const stream_view & s, int32_t end) {
        end = std::min(end, s.size());
        if (end <= upto) {
            return;
        }
        older.resize(end, -1);
        for (int32_t p = upto; p < end; ++p) {
            if (p < GRAM - 1) {
                older[p] = -1;
                continue;
            }
            auto ins = newest.emplace(key_at(s, p), p);
            if (ins.second) {
                older[p] = -1;
            } else {
                older[p] = ins.first->second;
                ins.first->second = p;
            }
        }
        upto = end;
    }

    struct hit {
        int32_t p   = -1; // source position matching s[F] (the copy continues with s[p + 1])
        int32_t len = 0;  // backward match length (capped at max_back)
    };

    // Candidates p < min(F, p_end) whose last 4 tokens equal s[F-3 .. F]. prefer_after >= 0: the nearest candidate
    // at or after it wins unless its match is shorter than min(8, the longest); otherwise the longest match, the
    // newest on ties.
    hit find(const stream_view & s, int32_t F, int32_t prefer_after, int32_t p_end, int32_t max_visit, int32_t max_back) const {
        hit best;
        if (F < GRAM - 1 || F >= s.size()) {
            return best;
        }
        auto it = newest.find(key_at(s, F));
        if (it == newest.end()) {
            return best;
        }
        hit near;
        int32_t visited = 0;
        int32_t steps   = 0;
        for (int32_t p = it->second; p >= 0 && visited < max_visit && steps < 4 * max_visit; p = older[p], ++steps) {
            if (p >= F || (p_end >= 0 && p >= p_end)) {
                continue;
            }
            int32_t l = 0;
            while (l < max_back && p - l >= 0 && s[p - l] == s[F - l]) {
                ++l;
            }
            if (l < GRAM) {
                continue; // a different 4-gram in the same hash slot
            }
            ++visited;
            if (l > best.len) {
                best.p   = p;
                best.len = l;
            }
            if (prefer_after >= 0 && p >= prefer_after) {
                near.p   = p; // newest first: the last one seen is the closest to prefer_after
                near.len = l;
            }
        }
        if (near.p >= 0 && near.len >= std::min(best.len, (int32_t) 8)) {
            return near;
        }
        return best;
    }
};

// Copy cursor: where the next copy continues while the output follows a source, and where the last copy broke.
struct copy_cursor {
    int32_t cur      = -1; // stream index of the next source token (the source position matching s[F] is cur - 1)
    int32_t stop_src = -1; // source index where the last copy broke
    int32_t stop_at  = -1; // output index where it broke
    int32_t prop_F   = -1; // F of the last proposal
    int32_t prop_src = -1; // stream index of the first token of the last proposal (-1 = none)

    void clear() { cur = stop_src = stop_at = prop_F = prop_src = -1; }

    // the stream grew past the last proposal: follow it, or record where it broke
    void observe(const stream_view & s) {
        const int32_t F = s.size() - 1;
        if (prop_src < 0 || prop_F < 0 || F <= prop_F) {
            if (prop_F >= 0 && F <= prop_F) {
                return; // nothing committed since (should not happen): keep the state
            }
            cur = -1;
            return;
        }
        const int32_t ne = F - prop_F;
        int32_t k = 0;
        while (k < ne && prop_src + k < s.size() && s[prop_src + k] == s[prop_F + 1 + k]) {
            ++k;
        }
        if (k == ne) {
            cur = prop_src + ne;
        } else {
            cur      = -1;
            stop_src = prop_src + k;
            stop_at  = prop_F + 1 + k;
        }
        prop_src = -1;
    }
};

inline int32_t match_bucket(int32_t len) {
    return len < 8 ? 0 : (len < 16 ? 1 : (len < 32 ? 2 : 3));
}

// ---------------------------------------------------------------------------------------------------------------
// Controller: learned acceptance + measured costs -> span length (0 = let the drafter draft).
struct copy_ctl {
    static constexpr int32_t ROWS_MAX = 160;

    struct rates {
        double first[4] = { 0.35, 0.5, 0.65, 0.8 }; // P(first proposed token followed) by match bucket
        double next     = 0.85;                      // P(a later token followed | the ones before it were)
        double edit     = 0.5;                       // P(an edit span's first edited token is taken)
    };
    rates r[2]; // [0] greedy, [1] sampled

    double t_ms[ROWS_MAX + 1] = {};   // EWMA round ms of copy rounds by verify rows
    int    t_n [ROWS_MAX + 1] = {};   // copy rounds seen by rows (the first records a graph and is not counted)
    double alt_tok_ms   = -1.0;       // drafter rounds: ms per emitted token
    double alt_round_ms = -1.0;       // drafter rounds: ms per round
    double alt_rows     = -1.0;       // drafter rounds: verify rows
    int    alt_n        = 0;

    double margin = 1.05;
    double row_k  = 0.04;

    static constexpr double BETA_FIRST = 0.15;
    static constexpr double BETA_NEXT  = 0.05;
    static constexpr double BETA_EDIT  = 0.125;
    static constexpr double BETA_T     = 0.2;
    static constexpr double BETA_ALT   = 0.1;

    static constexpr int32_t LADDER[] = { 1, 3, 7, 15, 23, 31, 47, 63, 95, 127 };

    static double clampq(double q) { return q < 0.02 ? 0.02 : (q > 0.995 ? 0.995 : q); }

    double row_scale(double rows) const { return 1.0 + row_k * std::max(0.0, rows - 1.0); }

    // ms of a copy round with `rows` verify rows: measured, else scaled from the nearest measured row count, else
    // from the drafter rounds; < 0 when nothing is known yet
    double pass_ms(int32_t rows) const {
        rows = std::max(1, std::min(rows, ROWS_MAX));
        if (t_n[rows] > 1) {
            return t_ms[rows];
        }
        int32_t near = -1;
        for (int32_t d = 1; d <= ROWS_MAX; ++d) {
            if (rows - d >= 1 && t_n[rows - d] > 1) { near = rows - d; break; }
            if (rows + d <= ROWS_MAX && t_n[rows + d] > 1) { near = rows + d; break; }
        }
        if (near > 0) {
            return t_ms[near] * row_scale(rows) / row_scale(near);
        }
        if (alt_round_ms > 0.0 && alt_rows > 0.0) {
            return alt_round_ms * row_scale(rows) / row_scale(alt_rows);
        }
        return -1.0;
    }

    // tokens per ms the drafter path gives (< 0: unknown)
    double alt_rate() const { return alt_tok_ms > 0.0 ? 1.0 / alt_tok_ms : -1.0; }

    // geometric sum 1 + q + .. + q^(m-1)
    static double geo(double q, int32_t m) {
        double s = 0.0, t = 1.0;
        for (int32_t j = 0; j < m; ++j) {
            s += t;
            t *= q;
        }
        return s;
    }

    // best span length <= n_avail (0 = no copy); `rate_out` gets its expected tokens per ms
    int32_t choose(int32_t n_avail, int32_t bucket, bool cursor, bool greedy, double * rate_out = nullptr) const {
        const rates & q = r[greedy ? 0 : 1];
        const double alt = alt_rate();
        if (alt <= 0.0 || n_avail <= 0) {
            return 0;
        }
        const double q1 = cursor ? q.next : q.first[std::max(0, std::min(bucket, 3))];
        int32_t best_w    = 0;
        double  best_rate = margin * alt;
        for (const int32_t W : LADDER) {
            if (W > n_avail) {
                break;
            }
            const double t = pass_ms(W + 1);
            if (t <= 0.0) {
                continue;
            }
            const double rate = (1.0 + q1 * geo(q.next, W)) / t;
            if (rate > best_rate) {
                best_rate = rate;
                best_w    = W;
            }
        }
        if (rate_out) {
            *rate_out = best_w > 0 ? best_rate : 0.0;
        }
        return best_w;
    }

    // an edit span of n tokens whose first edited token is at index f (tokens before it are a plain copy)
    double edit_rate(int32_t n, int32_t f, bool cursor, int32_t bucket, bool greedy) const {
        const rates & q = r[greedy ? 0 : 1];
        const double t = pass_ms(n + 1);
        if (t <= 0.0 || n <= 0 || f < 0 || f >= n) {
            return 0.0;
        }
        const double q1    = cursor ? q.next : q.first[std::max(0, std::min(bucket, 3))];
        const double reach = f == 0 ? 1.0 : q1 * std::pow(q.next, (double) (f - 1)); // P(all f plain tokens taken)
        const double plain = f == 0 ? 0.0 : q1 * geo(q.next, f);                   // expected plain tokens taken
        const double after = reach * q.edit * geo(q.next, n - f);                   // edited token and beyond
        return (1.0 + plain + after) / t;
    }

    // `trials` proposed tokens compared with the output, the first `matched` equal (matched < trials: the next differed)
    void learn(int32_t matched, int32_t trials, int32_t bucket, bool cursor, bool greedy) {
        if (trials <= 0) {
            return;
        }
        rates & q = r[greedy ? 0 : 1];
        const bool ok1 = matched >= 1;
        double & f = cursor ? q.next : q.first[std::max(0, std::min(bucket, 3))];
        f = clampq(f + (cursor ? BETA_NEXT : BETA_FIRST) * ((ok1 ? 1.0 : 0.0) - f));
        if (!ok1) {
            return;
        }
        double n = q.next;
        for (int32_t j = 1; j < trials; ++j) {
            const bool ok = j < matched;
            n += BETA_NEXT * ((ok ? 1.0 : 0.0) - n);
            if (!ok) {
                break;
            }
        }
        q.next = clampq(n);
    }

    void learn_edit(bool taken, bool greedy) {
        double & e = r[greedy ? 0 : 1].edit;
        e = clampq(e + BETA_EDIT * ((taken ? 1.0 : 0.0) - e));
    }

    void time_copy(int32_t rows, double ms) {
        if (rows < 1 || rows > ROWS_MAX || ms <= 0.0) {
            return;
        }
        if (++t_n[rows] == 1) {
            return; // the first round of a row count may record a graph
        }
        if (t_n[rows] == 2) {
            t_ms[rows] = ms;
            return;
        }
        t_ms[rows] += BETA_T * (std::min(ms, 2.0 * t_ms[rows]) - t_ms[rows]); // a stall counts as twice the usual
    }

    void time_alt(double ms, int32_t tokens, int32_t rows) {
        if (ms <= 0.0 || tokens <= 0) {
            return;
        }
        if (++alt_n == 1) {
            return;
        }
        const double per = ms / tokens;
        if (alt_tok_ms < 0.0) {
            alt_tok_ms   = per;
            alt_round_ms = ms;
            alt_rows     = rows;
            return;
        }
        if (ms > 3.0 * alt_round_ms) {
            return; // a graph being recorded or a stall
        }
        alt_tok_ms   += BETA_ALT * (per - alt_tok_ms);
        alt_round_ms += BETA_ALT * (ms - alt_round_ms);
        alt_rows     += BETA_ALT * (rows - alt_rows);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Edit copy: replacement rules learned from committed output vs the immutable source s[0, src_end).
struct edit_learner {
    static constexpr int32_t GAP    = 6;  // tokens skipped on each side at most
    static constexpr int32_t ANCHOR = 4;  // tokens that must agree after a skip (and before an edit counts)
    static constexpr int32_t RULES  = 12;
    static constexpr int32_t KEPT_RATIO = 4;
    static constexpr int32_t MISS_MAX   = 2;
    static constexpr int32_t PAUSE      = 3;

    struct rule {
        llama_token from[GAP + 1] = {};
        llama_token to  [GAP + 1] = {};
        int32_t nf = 0, nt = 0;
        int32_t places = 0;     // distinct source places that showed this edit (capped at 7)
        int32_t last   = -1;    // the last such place
        int32_t n_edit = 0;     // places where the edit was applied by the target
        int32_t n_kept = 0;     // places where its source text was copied unchanged
        int32_t miss   = 0;     // rejected first edited tokens in a row
        int32_t miss_place = -1;

        bool usable() const { return places >= 2 && n_kept * KEPT_RATIO <= n_edit && miss < MISS_MAX; }
    };

    rule    rules[RULES];
    int32_t n_rules = 0;
    int32_t src     = -1; // aligned pair: s[src] is expected at output index out
    int32_t out     = -1;
    int32_t lit     = 0;  // unchanged tokens since the last edit (capped at ANCHOR)
    int32_t run     = 0;  // unchanged tokens in a row
    int32_t src_end = 0;
    int32_t pause   = 0;

    // the last proposal
    int32_t prop_rule  = -1;
    int32_t prop_place = -1;
    int32_t prop_first = -1;

    void clear() {
        n_rules = 0;
        src = out = -1;
        lit = run = 0;
        src_end = 0;
        pause = 0;
        prop_rule = prop_place = prop_first = -1;
    }

    bool any_usable() const {
        for (int32_t i = 0; i < n_rules; ++i) {
            if (rules[i].usable()) {
                return true;
            }
        }
        return false;
    }

    void note_edit(const stream_view & s, int32_t ns, int32_t no) {
        if (ns == 0 || no == 0) {
            // a pure insertion or deletion keeps the next shared token on both sides, so the rule has source text
            ++ns;
            ++no;
        }
        if (ns > GAP + 1 || no > GAP + 1 || src + ns > src_end) {
            return;
        }
        const int32_t place = src;
        for (int32_t i = 0; i < n_rules; ++i) {
            rule & r = rules[i];
            if (r.nf != ns) {
                continue;
            }
            bool same_from = true;
            for (int32_t j = 0; j < ns && same_from; ++j) {
                same_from = r.from[j] == s[src + j];
            }
            if (!same_from) {
                continue;
            }
            bool same_to = r.nt == no;
            for (int32_t j = 0; j < no && same_to; ++j) {
                same_to = r.to[j] == s[out + j];
            }
            if (same_to) {
                if (r.last != place) {
                    r.places = std::min(7, r.places + 1);
                    r.n_edit++;
                    r.miss = std::max(0, r.miss - 1);
                }
                r.last = place;
                return;
            }
            if (r.last != place) {
                r.places = std::max(0, r.places - 1); // the same source text edited differently here
            }
        }
        int32_t slot = -1;
        if (n_rules < RULES) {
            slot = n_rules++;
        } else {
            for (int32_t i = 0; i < n_rules; ++i) {
                if (rules[i].places < 2) {
                    slot = i;
                    break;
                }
            }
            if (slot < 0) {
                return;
            }
        }
        rule & r = rules[slot];
        r = rule();
        r.nf = ns;
        r.nt = no;
        for (int32_t j = 0; j < ns; ++j) r.from[j] = s[src + j];
        for (int32_t j = 0; j < no; ++j) r.to[j]   = s[out + j];
        r.places = 1;
        r.last   = place;
        r.n_edit = 1;
    }

    void note_kept(const stream_view & s) {
        for (int32_t i = 0; i < n_rules; ++i) {
            rule & r = rules[i];
            if (run < ANCHOR + r.nf || src - r.nf < 0) {
                continue;
            }
            bool eq = true;
            for (int32_t j = 0; j < r.nf && eq; ++j) {
                eq = r.from[j] == s[src - r.nf + j];
            }
            if (eq) {
                r.n_kept++;
            }
        }
    }

    // Follow the committed output s[src_end, end). seed >= 0: where a fresh alignment may start (a copy source
    // position in the prompt whose token is expected next).
    void advance(const stream_view & s, int32_t end, int32_t seed) {
        if (src >= 0) {
            while (src < src_end && out < end) {
                if (s[src] == s[out]) {
                    ++src;
                    ++out;
                    lit = std::min(ANCHOR, lit + 1);
                    ++run;
                    if (n_rules > 0) {
                        note_kept(s);
                    }
                    continue;
                }
                // smallest total skip (ds + dout) after which ANCHOR tokens agree again
                int32_t ds = -1, dout = -1;
                for (int32_t cost = 1; cost <= 2 * GAP && ds < 0; ++cost) {
                    for (int32_t a = std::max(0, cost - GAP); a <= std::min(GAP, cost); ++a) {
                        const int32_t b = cost - a;
                        if (src + a + ANCHOR > src_end || out + b + ANCHOR > end) {
                            continue;
                        }
                        bool eq = true;
                        for (int32_t j = 0; j < ANCHOR && eq; ++j) {
                            eq = s[src + a + j] == s[out + b + j];
                        }
                        if (eq) {
                            ds = a;
                            dout = b;
                            break;
                        }
                    }
                }
                if (ds >= 0) {
                    if (lit >= ANCHOR) {
                        note_edit(s, ds, dout);
                    }
                    src += ds;
                    out += dout;
                    lit = 0;
                    run = 0;
                    continue;
                }
                if (end - out < GAP + ANCHOR) {
                    return; // wait for more output before giving up on this alignment
                }
                src = -1;
                break;
            }
            if (src >= 0 && src < src_end) {
                return;
            }
        }
        src = -1;
        if (seed >= 0 && seed < src_end && end - src_end >= ANCHOR) {
            src = seed;
            out = end;
            lit = ANCHOR;
            run = ANCHOR;
        }
    }

    int32_t match_rule(const stream_view & s, int32_t at) const {
        int32_t best = -1;
        for (int32_t i = 0; i < n_rules; ++i) {
            const rule & r = rules[i];
            if (!r.usable() || at + r.nf > src_end || s[at] != r.from[0]) {
                continue;
            }
            bool eq = true;
            for (int32_t j = 1; j < r.nf && eq; ++j) {
                eq = s[at + j] == r.from[j];
            }
            if (eq && (best < 0 || r.nf > rules[best].nf || (r.nf == rules[best].nf && r.places > rules[best].places))) {
                best = i;
            }
        }
        return best;
    }

    // Proposal from the aligned source with the rules applied: up to `limit` tokens into `prop`. When the alignment
    // lags behind the output (an edit too recent to re-anchor on), the walk first replays the rules over the
    // committed tokens s[out, end) and gives up on any difference. *src_at_end gets the source position the output
    // continues from (-1 when the walk did not reach `end` on a rule boundary). Returns the index of the first
    // edited token in `prop`, or -1 when no usable rule applies (the plain copy is the same) or during a pause.
    int32_t propose(const stream_view & s, int32_t end, int32_t limit, std::vector<llama_token> & prop, int32_t * src_at_end) {
        prop.clear();
        prop_rule = prop_place = prop_first = -1;
        *src_at_end = -1;
        if (src < 0 || out > end) {
            return -1;
        }
        const bool apply = pause == 0 && any_usable() && limit > 0;
        int32_t at = src;
        int32_t o  = out;
        while (at < src_end) {
            if (o == end && *src_at_end < 0) {
                *src_at_end = at;
                if (!apply) {
                    break;
                }
            }
            if (o >= end && (int32_t) prop.size() >= limit) {
                break;
            }
            const int32_t ri = apply ? match_rule(s, at) : -1;
            const llama_token * data = ri >= 0 ? rules[ri].to : nullptr;
            const int32_t count = ri >= 0 ? rules[ri].nt : 1;
            for (int32_t j = 0; j < count; ++j) {
                const llama_token t = data ? data[j] : s[at];
                if (o < end) {
                    if (s[o] != t) {
                        prop.clear();
                        prop_rule = prop_place = prop_first = -1;
                        *src_at_end = -1;
                        return -1; // the output left the rules' prediction
                    }
                    ++o;
                    continue;
                }
                if ((int32_t) prop.size() >= limit) {
                    break;
                }
                if (data && prop_first < 0) {
                    prop_first = (int32_t) prop.size();
                    prop_rule  = ri;
                    prop_place = at;
                }
                prop.push_back(t);
                ++o;
            }
            at += ri >= 0 ? rules[ri].nf : 1;
        }
        if (prop_first < 0) {
            prop.clear();
        }
        return prop_first;
    }

    // the target's verdict on the last proposal: `accepted` tokens of it were kept
    // returns true when the first edited token was taken
    bool feedback(int32_t accepted) {
        if (prop_rule < 0 || prop_first < 0) {
            return false;
        }
        rule & r = rules[prop_rule];
        const bool hit = accepted > prop_first;
        if (hit) {
            r.miss = 0;
        } else {
            pause = PAUSE;
            if (accepted == prop_first && r.miss_place != prop_place) {
                r.miss++;
                r.miss_place = prop_place;
                r.last = prop_place;
            }
        }
        prop_rule = prop_place = prop_first = -1;
        return hit;
    }

    void tick() {
        if (pause > 0) {
            --pause;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Per-sequence state: per request (index, cursor, edits, pending shadow proposals) + persistent controller.
struct copy_seq {
    struct pending {
        int32_t F = -1;
        std::vector<llama_token> prop;
        int32_t bucket = 0;
        bool    cursor = false;
        bool    greedy = false;
    };

    anchor_index         index;
    copy_cursor          cursor;
    edit_learner         edits;
    std::vector<pending> pend;
    int32_t              n_hist_last = 0;

    copy_ctl ctl; // survives requests

    // the round in flight
    enum kind_t { ROUND_NONE = 0, ROUND_DRAFTER, ROUND_COPY, ROUND_EDIT };
    kind_t  kind = ROUND_NONE;
    int32_t rows = 0;
    bool    greedy = false;
    int64_t t_last_us = 0;
    bool    acc_seen  = false;

    // statistics (per request)
    uint64_t n_shadow = 0, n_copy = 0, n_copy_tok = 0, n_copy_acc = 0, n_edit = 0, n_edit_hit = 0, n_cursor = 0;

    void reset_request() {
        index.clear();
        cursor.clear();
        edits.clear();
        pend.clear();
        n_hist_last = 0;
        kind = ROUND_NONE;
        rows = 0;
        t_last_us = 0;
        acc_seen = false;
        n_shadow = n_copy = n_copy_tok = n_copy_acc = n_edit = n_edit_hit = n_cursor = 0;
    }

    // the target's verdict on this round's span (n_accepted span tokens kept)
    void on_verified(int32_t n_accepted) {
        if (kind == ROUND_COPY) {
            n_copy_acc += std::max(0, n_accepted);
        } else if (kind == ROUND_EDIT) {
            n_copy_acc += std::max(0, n_accepted);
            const bool had = edits.prop_first >= 0;
            const bool hit = edits.feedback(n_accepted);
            if (had) {
                n_edit_hit += hit ? 1 : 0;
                ctl.learn_edit(hit, greedy);
            }
        }
    }

    // score pending proposals against the committed stream
    void resolve(const stream_view & s) {
        const int32_t F = s.size() - 1;
        for (size_t i = 0; i < pend.size();) {
            pending & p = pend[i];
            const int32_t avail = F - p.F;
            int32_t k = 0;
            const int32_t lim = std::min<int32_t>((int32_t) p.prop.size(), std::max(0, avail));
            while (k < lim && p.prop[k] == s[p.F + 1 + k]) {
                ++k;
            }
            const bool done = k < lim || k == (int32_t) p.prop.size();
            if (done) {
                ctl.learn(k, std::min<int32_t>(k + 1, (int32_t) p.prop.size()), p.bucket, p.cursor, p.greedy);
                n_shadow++;
                pend.erase(pend.begin() + (std::ptrdiff_t) i);
            } else {
                ++i;
            }
        }
        if (pend.size() > 8) {
            pend.erase(pend.begin(), pend.begin() + (std::ptrdiff_t) (pend.size() - 8));
        }
    }
};

struct copy_flags {
    bool ctl    = false;
    bool cursor = false;
    bool edit   = false;
    bool cursor_src_all = true;

    bool any() const { return ctl || cursor || edit; }

    static copy_flags from_env() {
        copy_flags f;
        f.ctl    = env_on("LLAMA_RTB_COPYCTL");
        f.cursor = env_on("LLAMA_RTB_COPYCURSOR");
        f.edit   = env_on("LLAMA_RTB_EDITCOPY");
        const char * src = std::getenv("LLAMA_RTB_COPYCURSOR_SRC");
        f.cursor_src_all = !(src != nullptr && std::strcmp(src, "prompt") == 0);
        return f;
    }
};

// One copy-first round decision. s = history + id_last; cap = most span tokens allowed (n_first_max); n_src = prompt
// length (source bound for edit copy and, without the cursor or with LLAMA_RTB_COPYCURSOR_SRC=prompt, for lookups);
// legacy_min / legacy_want: the old match gate and span when the controller is off.
// Returns the span to verify (empty = let the drafter draft).
inline std::vector<llama_token> copy_round(const copy_flags & fl, copy_seq & cs, const stream_view & s, int32_t cap,
        int32_t n_src, bool greedy, int32_t legacy_min, int32_t legacy_want) {
    std::vector<llama_token> span;
    const int32_t F = s.size() - 1;
    if (s.n < cs.n_hist_last) {
        // the history shrank (context shift): positions moved
        const copy_ctl keep = cs.ctl;
        cs.reset_request();
        cs.ctl = keep;
    }
    cs.n_hist_last = s.n;
    cs.greedy = greedy;
    cs.kind   = copy_seq::ROUND_DRAFTER; // unless a span is chosen below
    cs.rows   = 0;
    cs.edits.tick();

    if (fl.ctl) {
        cs.resolve(s);
    }
    if (fl.cursor) {
        cs.cursor.observe(s);
    }
    if (cap <= 0) {
        return span;
    }

    const int32_t src_bound = (fl.cursor && fl.cursor_src_all) ? -1 : (n_src >= 0 ? n_src : -1);

    // lookup: the cursor first, then the index
    anchor_index::hit h;
    bool by_cursor = false;
    if (fl.cursor && cs.cursor.cur > 0 && cs.cursor.cur <= F && s[cs.cursor.cur - 1] == s[F]) {
        const int32_t q = cs.cursor.cur - 1;
        int32_t l = 0;
        while (l < 64 && q - l >= 0 && s[q - l] == s[F - l]) {
            ++l;
        }
        if (l >= anchor_index::GRAM) {
            h.p = q;
            h.len = l;
            by_cursor = true;
            cs.n_cursor++;
        }
    }
    if (h.p < 0) {
        cs.index.extend(s, F);
        int32_t after = -1;
        if (fl.cursor && cs.cursor.stop_src >= 0 && F - cs.cursor.stop_at <= 256) {
            after = cs.cursor.stop_src;
        }
        h = cs.index.find(s, F, after, src_bound, 256, 64);
    }

    // edit copy: follow the output against the prompt source; an aligned source is also the best plain source
    // (it has walked through earlier edits, where the cursor broke)
    std::vector<llama_token> eprop;
    int32_t efirst = -1;
    if (fl.edit) {
        cs.edits.src_end = n_src >= 0 ? std::min(n_src, s.n) : s.n;
        const int32_t seed = (h.p >= 0 && h.len >= 6 && h.p + 1 < cs.edits.src_end) ? h.p + 1 : -1;
        cs.edits.advance(s, F + 1, seed);
        int32_t src_at_end = -1;
        efirst = cs.edits.propose(s, F + 1, cap, eprop, &src_at_end);
        if (efirst >= 0 && efirst >= (int32_t) eprop.size()) {
            efirst = -1;
        }
        if (!by_cursor && src_at_end > 0 && src_at_end < cs.edits.src_end && src_at_end - 1 != h.p) {
            h.p = src_at_end - 1; // the copy continues with s[src_at_end]
            h.len = 8;
            by_cursor = true; // a continuation: scored with the continuation rate
        }
    }

    std::vector<llama_token> prop;
    if (h.p >= 0) {
        prop.reserve(cap);
        for (int32_t i = 0; i < cap; ++i) {
            const int32_t k = h.p + 1 + i;
            prop.push_back(k <= F ? s[k] : prop[k - F - 1]); // a match that runs into its own span keeps repeating
        }
        if (fl.cursor) {
            cs.cursor.prop_F   = F;
            cs.cursor.prop_src = h.p + 1;
        }
        if (fl.ctl) {
            copy_seq::pending p;
            p.F      = F;
            p.prop   = prop;
            p.bucket = match_bucket(h.len);
            p.cursor = by_cursor;
            p.greedy = greedy;
            cs.pend.push_back(std::move(p));
        }
    } else if (fl.cursor) {
        cs.cursor.prop_F   = F;
        cs.cursor.prop_src = -1;
    }

    const int32_t bucket = match_bucket(h.len);
    if (fl.ctl) {
        // with an edit proposal, the plain copy is expected to break at the edit: it is only weighed up to it
        double rate_plain = 0.0;
        const int32_t n_plain = efirst >= 0 ? std::min<int32_t>((int32_t) prop.size(), efirst) : (int32_t) prop.size();
        const int32_t W = n_plain <= 0 ? 0 : cs.ctl.choose(n_plain, bucket, by_cursor, greedy, &rate_plain);
        if (efirst >= 0) {
            // an edit span replaces the plain one when it is expected to pay more (and beats the drafter)
            const int32_t n_all = (int32_t) eprop.size();
            int32_t n_e = 0;
            double  re  = 0.0;
            for (const int32_t w : copy_ctl::LADDER) {
                if (w > n_all) {
                    break;
                }
                if (w <= efirst) {
                    continue;
                }
                const double r = cs.ctl.edit_rate(w, efirst, by_cursor, bucket, greedy);
                if (r > re) {
                    re  = r;
                    n_e = w;
                }
            }
            if (n_e == 0) {
                n_e = n_all; // shorter than the first rung past the edit: the whole proposal
                re  = cs.ctl.edit_rate(n_e, efirst, by_cursor, bucket, greedy);
            }
            const double alt = cs.ctl.alt_rate();
            if (alt > 0.0 && re > cs.ctl.margin * alt && re >= rate_plain) {
                eprop.resize(n_e);
                cs.kind = copy_seq::ROUND_EDIT;
                cs.rows = n_e + 1;
                cs.n_edit++;
                return eprop;
            }
        }
        if (W > 0) {
            prop.resize(W);
            cs.kind = copy_seq::ROUND_COPY;
            cs.rows = W + 1;
            cs.n_copy++;
            cs.n_copy_tok += W;
            cs.edits.prop_rule = cs.edits.prop_first = cs.edits.prop_place = -1;
            return prop;
        }
        cs.edits.prop_rule = cs.edits.prop_first = cs.edits.prop_place = -1;
        cs.kind = copy_seq::ROUND_DRAFTER;
        return span;
    }

    // no controller: the old span length and match gate (the cursor is exempt from the gate)
    if (efirst >= 0 && legacy_want > 0) {
        eprop.resize(std::min<int32_t>((int32_t) eprop.size(), std::max(legacy_want, efirst + 1)));
        cs.kind = copy_seq::ROUND_EDIT;
        cs.rows = (int32_t) eprop.size() + 1;
        cs.n_edit++;
        return eprop;
    }
    cs.edits.prop_rule = cs.edits.prop_first = cs.edits.prop_place = -1;
    if (legacy_want > 0 && !prop.empty() && (by_cursor || h.len >= legacy_min)) {
        prop.resize(std::min<int32_t>((int32_t) prop.size(), legacy_want));
        cs.kind = copy_seq::ROUND_COPY;
        cs.rows = (int32_t) prop.size() + 1;
        cs.n_copy++;
        cs.n_copy_tok += prop.size();
        return prop;
    }
    cs.kind = copy_seq::ROUND_DRAFTER;
    return span;
}

} // namespace rtb
