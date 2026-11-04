#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine with a climb counter and a weighted drop-pressure
// accumulator. The depth N climbs one step after N_CLIMB(N) consecutive verifies
// that accepted every drafted token. The climb cost is low at the floor and at
// depth, high in the middle: 2 at depth 1, 4 at depth 2, 6 at depth 3, then
// 5/4/3/2 from depth 4 upward. Getting from the floor to depth 3 needs only 6
// full accepts, but pushing past 3 (where prose acceptance collapses) costs 6
// full accepts of 3-token drafts, which predictable content clears quickly and
// marginal content never does. Any miss adds (n_draft - n_accepted) to a
// drop-pressure accumulator; when it reaches depth * 5 the depth drops one step
// and the pressure resets. A near miss (n_draft-1) adds 1, a total miss adds
// n_draft, so high depths fall quickly while low depths hold. The drop budget
// scales with depth but never drops below 20, so shallow depths shed bad content
// quickly without collapsing to the floor on a few bad rounds; deep depths hold
// a little longer. At the floor no pressure accumulates at all. The depth starts
// at the floor max(1, --spec-draft-n-min-adaptive) and stays in
// [floor, n_max]; --spec-draft-n-max bounds the upper end of the adaptive
// range.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    // consecutive full accepts needed to climb one step from depth N; low at the
    // floor and at depth, high in the middle where acceptance is marginal
    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 6;
            case 4: return 5;
            case 5: return 4;
            case 6: return 3;
            default: return 2; // depth >= 7
        }
    }

    // accumulated (n_draft - n_accepted) needed to drop one step from depth N;
    // scaled by depth, with a floor so shallow depths do not collapse too fast
    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    // feed one verification result: n_draft is the number of tokens this
    // implementation drafted, n_accepted the number the target accepted
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (n_accepted == n_draft) {
            n_drop = 0;

            // full acceptance: reset the drop pressure, accumulate the climb streak
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            // any miss adds (n_draft - n_accepted) to the drop pressure; drop one
            // step when the accumulated pressure reaches the depth-scaled budget
            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }
};

// fork AIPFIX POSGATE (LLAMA_SPEC_POS_GATE=<ratio>, default off): a cap on top of the streak controller from
// per-depth acceptance. For each depth d it keeps, over rounds drafted exactly d deep, a running mean of
//   gain_d = 1 if the d-th draft token was accepted, else 0      (the extra token position d buys)
//   base_d = 1 + min(n_accepted, d - 1)                          (tokens the same round gives without it)
// and caps the depth at d - 1 while gain_d < ratio * base_d, ratio = the cost of one more verify row as a fraction
// of a round (whole-round w5/w4 on the Taichi trace: ~0.13). Every `probe` gated rounds one round runs at the
// streak depth so gain_d keeps being measured. Lossless: it only reads verified history to choose a depth.
// Its state is not part of the prompt-cache snapshot; it relearns after a restore or a new request.
struct common_speculative_posgate {
    static constexpr int N_DEPTH = 9;
    float gain[N_DEPTH] = {};
    float base[N_DEPTH] = {};
    int   seen[N_DEPTH] = {};
    int   gated = 0;

    struct config {
        float ratio = 0.0f;  // 0 = off
        float alpha = 0.05f; // running-mean weight of the newest round
        int   min_seen = 16; // rounds at a depth before it can be gated
        int   probe = 32;    // gated rounds between probe rounds
    };

    static const config & cfg() {
        static const config c = [] {
            config r;
            auto f = [](const char * name, float def) { const char * v = getenv(name); return v && *v ? (float) atof(v) : def; };
            r.ratio = std::max(0.0f, f("LLAMA_SPEC_POS_GATE", 0.0f));
            r.alpha = std::min(1.0f, std::max(0.001f, f("LLAMA_SPEC_POS_GATE_ALPHA", 0.05f)));
            r.min_seen = std::max(1, (int) f("LLAMA_SPEC_POS_GATE_MIN", 16.0f));
            r.probe = std::max(1, (int) f("LLAMA_SPEC_POS_GATE_PROBE", 32.0f));
            if (r.ratio > 0.0f) {
                fprintf(stderr, "[pos-gate] on: ratio=%.4f alpha=%.3f min=%d probe=%d\n", r.ratio, r.alpha, r.min_seen, r.probe);
            }
            return r;
        }();
        return c;
    }

    static bool enabled() { return cfg().ratio > 0.0f; }

    void reset() { *this = common_speculative_posgate(); }

    // one verified round drafted n_draft deep with n_accepted accepted (both at the controller's own depth)
    void update(int n_draft, int n_accepted) {
        if (n_draft < 1 || n_draft >= N_DEPTH) {
            return;
        }
        const config & c = cfg();
        const float g = n_accepted >= n_draft ? 1.0f : 0.0f;
        const float b = 1.0f + (float) std::min(n_accepted, n_draft - 1);
        const float a = seen[n_draft] == 0 ? 1.0f : c.alpha;
        gain[n_draft] += a * (g - gain[n_draft]);
        base[n_draft] += a * (b - base[n_draft]);
        seen[n_draft]++;
    }

    // depth to draft this round given the streak depth n_cur and the floor
    int cap(int n_cur, int floor) {
        const config & c = cfg();
        int d = n_cur;
        while (d > floor && d < N_DEPTH && seen[d] >= c.min_seen && gain[d] < c.ratio * base[d]) {
            --d;
        }
        if (d == n_cur) {
            gated = 0;
            return d;
        }
        if (++gated >= c.probe) {
            gated = 0;
            return n_cur; // probe round keeps gain[n_cur] fresh
        }
        return d;
    }
};

// fork AIPFIX TPM (LLAMA_SPEC_TPM=1, default off): pick the draft depth that maximizes expected tokens per ms.
// Per position k it keeps a running mean of the conditional acceptance p_k = P(draft token k accepted | 1..k-1
// accepted), learned from every verified round that drafted at least k tokens (the MTP draft chain does not depend
// on the depth chosen, so p_k is shared by all depths >= k). Per drafted count d it keeps a running mean of the
// wall time of a whole round (draft start to the next draft start: draft + verify + accept), so the cost includes
// the extra expert reads of a wider verify on the card it runs on. Expected tokens of a depth-d round:
//   E(d) = 1 + sum_{k=1..d} prod_{j<=k} p_j            (the target always gives one token)
// and the depth is argmax E(d) / cost(d) over [floor, n_max], moving off the current depth only for a > margin gain.
// Depths without a measured cost use the nearest measured depth scaled by (1 + slope x row difference); positions
// without enough samples are reached by a probe round (current best + 1) every `probe` rounds. LLAMA_SPEC_TPM_COST
// = "c1,c2,..." fixes the per-depth costs (ms) instead of measuring them. Lossless: it only chooses a depth from
// verified history. Its state is per request and is not part of the prompt-cache snapshot.
struct common_speculative_tpm {
    static constexpr int N_DEPTH = 9;
    float p[N_DEPTH]     = {}; // conditional acceptance of position k (1-based)
    int   tried[N_DEPTH] = {};
    float cost[N_DEPTH]  = {}; // ms per round drafted d deep
    int   timed[N_DEPTH] = {};
    int   rounds = 0;
    int   since_probe = 0;
    int   cur = 0;          // last chosen depth
    int   last_d = 0;       // drafted count of the round in flight (set by update)
    long long t_start = -1; // us, start of the round in flight
    long long n_pick[N_DEPTH] = {};

    struct config {
        bool  on = false;
        float alpha = 0.03f;
        float margin = 0.01f;
        float slope = 0.10f; // prior cost per extra draft token, fraction of the measured neighbor's round
        int   min_seen = 16;
        int   probe = 24;
        int   warm = 32;     // rounds on the streak depth before the controller takes over
        int   log_every = 0; // rounds between [spec-tpm] state lines (0 = only at reset)
        float fixed[N_DEPTH] = {};
    };

    static const config & cfg() {
        static const config c = [] {
            config r;
            auto f = [](const char * name, float def) { const char * v = getenv(name); return v && *v ? (float) atof(v) : def; };
            r.on = f("LLAMA_SPEC_TPM", 0.0f) != 0.0f;
            r.alpha = std::min(1.0f, std::max(0.001f, f("LLAMA_SPEC_TPM_ALPHA", 0.03f)));
            r.margin = std::max(0.0f, f("LLAMA_SPEC_TPM_MARGIN", 0.01f));
            r.slope = std::max(0.0f, f("LLAMA_SPEC_TPM_SLOPE", 0.10f));
            r.min_seen = std::max(1, (int) f("LLAMA_SPEC_TPM_MIN", 16.0f));
            r.probe = std::max(2, (int) f("LLAMA_SPEC_TPM_PROBE", 24.0f));
            r.warm = std::max(0, (int) f("LLAMA_SPEC_TPM_WARM", 32.0f));
            r.log_every = std::max(0, (int) f("LLAMA_SPEC_TPM_LOG", 0.0f));
            if (const char * s = getenv("LLAMA_SPEC_TPM_COST")) {
                int d = 1;
                for (const char * q = s; *q && d < N_DEPTH; ++d) {
                    r.fixed[d] = (float) atof(q);
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
            }
            if (r.on) {
                fprintf(stderr, "[spec-tpm] on: alpha=%.3f margin=%.3f slope=%.3f min=%d probe=%d warm=%d fixed=%s\n",
                        r.alpha, r.margin, r.slope, r.min_seen, r.probe, r.warm, r.fixed[1] > 0.0f ? "yes" : "no");
            }
            return r;
        }();
        return c;
    }

    static bool enabled() { return cfg().on; }

    void log_state(const char * why) const {
        fprintf(stderr, "[spec-tpm] %s rounds=%d p=", why, rounds);
        for (int k = 1; k < N_DEPTH && tried[k] > 0; ++k) fprintf(stderr, "%s%.3f", k > 1 ? "," : "", p[k]);
        fprintf(stderr, " ms=");
        for (int d = 1; d < N_DEPTH; ++d) if (timed[d] > 0) fprintf(stderr, "d%d:%.2f(%d) ", d, cost[d], timed[d]);
        fprintf(stderr, "picks=");
        for (int d = 1; d < N_DEPTH; ++d) if (n_pick[d] > 0) fprintf(stderr, "d%d:%lld ", d, n_pick[d]);
        fprintf(stderr, "\n");
    }

    void reset() {
        if (rounds > 0) log_state("end");
        *this = common_speculative_tpm();
    }

    // one verified round: n_draft tokens drafted (verify width n_draft + 1), n_accepted of them accepted
    void update(int n_draft, int n_accepted) {
        if (n_draft < 1) return;
        const config & c = cfg();
        n_accepted = std::min(n_accepted, n_draft);
        for (int k = 1; k <= std::min(n_accepted + 1, n_draft) && k < N_DEPTH; ++k) {
            const float g = k <= n_accepted ? 1.0f : 0.0f;
            p[k] += (tried[k] == 0 ? 1.0f : c.alpha) * (g - p[k]);
            tried[k]++;
        }
        last_d = std::min(n_draft, N_DEPTH - 1);
        rounds++;
        if (c.log_every > 0 && rounds % c.log_every == 0) log_state("state");
    }

    float cost_of(int d) const {
        const config & c = cfg();
        if (c.fixed[d] > 0.0f) return c.fixed[d];
        if (timed[d] >= c.min_seen / 2) return cost[d];
        int best = -1;
        for (int m = 1; m < N_DEPTH; ++m) {
            if (timed[m] >= c.min_seen / 2 && (best < 0 || std::abs(m - d) < std::abs(best - d))) best = m;
        }
        if (best < 0) return 1.0f + c.slope * d; // relative only; the argmax needs ratios
        return cost[best] * std::max(0.1f, 1.0f + c.slope * (d - best));
    }

    float expected(int d) const {
        float e = 1.0f, run = 1.0f;
        for (int k = 1; k <= d; ++k) { run *= p[k]; e += run; }
        return e;
    }

    // depth for the round that starts now; streak = the streak controller's depth (used during warm-up)
    int pick(int streak, int floor, int cap, long long now_us) {
        const config & c = cfg();
        cap = std::min(cap, N_DEPTH - 1);
        floor = std::max(1, std::min(floor, cap));
        if (t_start >= 0 && last_d > 0) {
            const float ms = (float) (now_us - t_start) * 1e-3f;
            const float ref = timed[last_d] > 0 ? cost[last_d] : 0.0f;
            if (ms > 0.0f && ms < 1000.0f && (ref == 0.0f || ms < 4.0f * ref)) { // drop gaps between requests
                cost[last_d] += (timed[last_d] == 0 ? 1.0f : cfg().alpha) * (ms - cost[last_d]);
                timed[last_d]++;
            }
        }
        t_start = now_us;
        last_d = 0;
        int d;
        if (rounds < c.warm) {
            d = std::max(floor, std::min(streak, cap));
        } else {
            int best = cur > 0 ? std::max(floor, std::min(cur, cap)) : std::max(floor, std::min(streak, cap));
            float best_r = expected(best) / cost_of(best);
            for (int e = floor; e <= cap; ++e) {
                bool known = true;
                for (int k = 1; k <= e; ++k) known = known && tried[k] >= c.min_seen;
                if (!known) continue;
                const float r = expected(e) / cost_of(e);
                if (r > best_r * (1.0f + c.margin)) { best = e; best_r = r; }
            }
            d = best;
            if (++since_probe >= c.probe) { // one deeper round keeps p and cost of the next depth measured
                since_probe = 0;
                if (d < cap) d++;
            } else {
                cur = d;
            }
        }
        n_pick[d]++;
        return d;
    }
};
