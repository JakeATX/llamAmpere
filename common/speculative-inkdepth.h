#pragma once

// INKDEPTH: per-round draft depth for the Inkling DSpark drafter (sampled mb walk).
//
// Method:
//   * every round the drafter walks W slots in one graph; for each slot k the drafter's top probability qmax_k at that
//     slot (known before the slot's token is drawn) is mapped to an acceptance rate a_k learned online (decayed counts per slot and q bin, shrunk to
//     the q-bin rate across slots, shrunk to a prior at q);
//   * the expected tokens of keeping n slots is E(n) = 1 + sum_{j<=n} prod_{i<=j} a_i (the verifier always gives one);
//   * the cost of a round that keeps n slots is the measured wall time of such rounds on this card, per context bucket
//     (< 16K, < 64K, longer): draft time + the rest of the round (verify of n+1 rows, whose MoE expert reads grow with
//     the union of experts the rows touch, plus accept/sampling), learned per (bucket, n); n values not measured yet use
//     a prior shape scaled by the bucket's observed/prior ratio. A width change costs a graph rebuild; that penalty is
//     learned per bucket and charged when n differs from the last round's n;
//   * slots are decided in order: slot k is kept while E/cost at k, or at k+1/k+2 with mean per-slot rates, still
//     beats the best E/cost so far; the first stop is final ("stop when more drafts cannot beat the best tokens/ms");
//   * W itself moves between rounds: up when the pick sits at W often, down when it rarely does.
// Exact under p/q: n is a stopping time of the walk. Whether slot k is kept depends only on verified history, the
// tokens drawn at slots < k and slot k's distribution (qmax_k), never on the token drawn at k or later, and never on the
// verifier's draws (keeping a slot because of its own drawn token would skew that token away from q). State is per server (shared by sequences), cost timing only from single-sequence rounds.
//
// Env: LLAMA_INKDEPTH=0 off (fixed depth, the old path)   LLAMA_INKDEPTH_MAX=7 walk ceiling   LLAMA_INKDEPTH_MIN=1
//      LLAMA_INKDEPTH_SWEEP=1 measurement mode (fixed-n blocks, half the rounds at n=W)   LLAMA_INKDEPTH_TRACE=<file>
//      LLAMA_INKDEPTH_LOG=N state line every N rounds   LLAMA_INKDEPTH_WALK=0 keeps W at the ceiling

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

struct common_inkdepth {
    static constexpr int MAXD = 8;  // slots 1..7 (DSpark block 8)
    static constexpr int NQB  = 10; // q bins of width 0.1
    static constexpr int NB   = 3;  // context buckets

    struct config {
        bool  on        = false;
        int   max_depth = 7;
        int   min_depth = 1;
        bool  sweep     = false;
        bool  walk_adapt = true;
        int   log_every = 0;
        int   probe     = 64;   // rounds between probe picks of an unmeasured neighbor depth
        int   min_cost  = 8;    // rounds before a (bucket, n) cell uses its own mean
        float decay     = 1.0f - 1.0f/512.0f;
        float n0_cell   = 8.0f; // prior strength (pseudo counts) of a (slot, bin) cell toward the bin rate
        float n0_bin    = 4.0f; // prior strength of a bin toward its prior
        std::string trace;
    };

    static int env_i(const char * n, int d) { const char * v = getenv(n); return v && *v ? atoi(v) : d; }

    static const config & cfg() {
        static const config c = [] {
            config r;
            r.on         = env_i("LLAMA_INKDEPTH", 0) != 0;
            r.max_depth  = std::max(1, std::min(MAXD - 1, env_i("LLAMA_INKDEPTH_MAX", 7)));
            r.min_depth  = std::max(1, std::min(r.max_depth, env_i("LLAMA_INKDEPTH_MIN", 1)));
            r.sweep      = env_i("LLAMA_INKDEPTH_SWEEP", 0) != 0;
            r.walk_adapt = env_i("LLAMA_INKDEPTH_WALK", 1) != 0;
            r.log_every  = std::max(0, env_i("LLAMA_INKDEPTH_LOG", 0));
            if (const char * t = getenv("LLAMA_INKDEPTH_TRACE")) {
                r.trace = t;
            }
            return r;
        }();
        return c;
    }

    // prior of the rest-of-round cost by kept slots n (relative to n = 1); measured on a 3090 (AI4 fold, 3K/32K/128K),
    // see lanes/INKDEPTH.md. Only the shape matters: the bucket scale is learned.
    static float rest_shape(int n) {
        static const float s[MAXD] = { 1.00f, 1.00f, 1.08f, 1.16f, 1.24f, 1.32f, 1.40f, 1.48f };
        return s[std::max(0, std::min(MAXD - 1, n))];
    }
    // prior acceptance of a slot whose drawn token had drafter probability q (mb accept; learned online)
    static float accept_prior(int qb) {
        const float q = (qb + 0.5f) / NQB;
        return std::min(0.97f, 0.30f + 0.65f*q);
    }

    // accept model
    float ok[MAXD][NQB] = {}, cnt[MAXD][NQB] = {};
    float okb[NQB] = {}, cntb[NQB] = {};
    // cost model (ms)
    float rest[NB][MAXD] = {};  int nrest[NB][MAXD] = {};
    float scale[NB]      = {};  int nscale[NB]      = {};
    float pen[NB]        = {};  int npen[NB]        = {};
    float dms[MAXD]      = {};  int ndms[MAXD]      = {};
    // walk
    int   walk = 0;
    float hit  = 0.15f;
    // round in flight
    bool    pend = false;
    int64_t t_round = -1;
    int     r_b = 0, r_w = 0, r_n = 0, r_acc = -1, r_ctx = 0;
    bool    r_sw = false;
    float   r_dms = 0.0f;
    float   r_q[MAXD] = {};
    int     prev_n = 0;
    // stats
    long long rounds = 0, n_pick[MAXD] = {}, n_walk[MAXD] = {};
    FILE * ftrace = nullptr;

    common_inkdepth() {
        walk = cfg().max_depth;
        if (!cfg().trace.empty()) {
            ftrace = fopen(cfg().trace.c_str(), "a");
        }
    }
    ~common_inkdepth() {
        log_state("end");
        if (ftrace) {
            fclose(ftrace);
        }
    }
    common_inkdepth(const common_inkdepth &) = delete;
    common_inkdepth & operator=(const common_inkdepth &) = delete;

    static int bucket(int ctx) { return ctx < 16384 ? 0 : ctx < 65536 ? 1 : 2; }
    static int qbin(float q)   { return std::max(0, std::min(NQB - 1, (int) (q * NQB))); }

    float rate(int k, float q) const {
        const config & c = cfg();
        const int b = qbin(q);
        const float rb = (okb[b] + c.n0_bin*accept_prior(b)) / (cntb[b] + c.n0_bin);
        k = std::max(1, std::min(MAXD - 1, k));
        return (ok[k][b] + c.n0_cell*rb) / (cnt[k][b] + c.n0_cell);
    }

    // rest-of-round cost (ms) of keeping n slots in bucket b; < 0 when nothing is known yet (no timing in this bucket)
    float rest_ms(int b, int n) const {
        const config & c = cfg();
        if (nrest[b][n] >= c.min_cost) {
            return rest[b][n];
        }
        if (nscale[b] == 0) {
            return -1.0f;
        }
        return scale[b] * rest_shape(n);
    }

    // sweep: blocks of 32 rounds at walk = kept = 7,1,7,2,...,7,6 (the n = 7 rounds give the full acceptance of every
    // slot, the others the cost of each depth without width changes inside a block)
    int sweep_n() const {
        static const int order[12] = { 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6 };
        return order[(rounds / 32) % 12];
    }

    // walk for the round that starts now (single drafting sequence)
    int pick_walk() {
        const config & c = cfg();
        if (c.sweep) {
            return std::min(c.max_depth, sweep_n());
        }
        if (!c.walk_adapt) {
            return c.max_depth;
        }
        return std::max(c.min_depth, std::min(c.max_depth, walk));
    }

    // a round starts: close the previous one (its accept already came) and time it
    void begin_round(int64_t now_us) {
        if (pend && r_acc >= 0 && t_round >= 0) {
            const float ms   = (float) (now_us - t_round) * 1e-3f;
            const float rst  = ms - r_dms;
            const float ref  = rest_ms(r_b, r_n);
            const bool  sane = ms > 0.0f && ms < 1000.0f && rst > 0.0f && (ref <= 0.0f || rst < 4.0f*ref + 4.0f);
            if (sane) {
                const float a = 0.05f;
                if (!r_sw) {
                    float & m = rest[r_b][r_n];
                    m += (nrest[r_b][r_n] == 0 ? 1.0f : a) * (rst - m);
                    nrest[r_b][r_n]++;
                    const float sc = rst / rest_shape(r_n);
                    scale[r_b] += (nscale[r_b] == 0 ? 1.0f : a) * (sc - scale[r_b]);
                    nscale[r_b]++;
                } else if (nrest[r_b][r_n] >= cfg().min_cost) {
                    const float d = std::max(0.0f, rst - rest[r_b][r_n]);
                    pen[r_b] += (npen[r_b] == 0 ? 1.0f : a) * (d - pen[r_b]);
                    npen[r_b]++;
                }
            }
            if (ftrace) {
                fprintf(ftrace, "%d %d %d %d %.3f %.3f %d", r_ctx, r_w, r_n, r_sw ? 1 : 0, r_dms, sane ? ms : -1.0f, r_acc);
                for (int k = 1; k <= r_w; ++k) {
                    fprintf(ftrace, " %.4f", r_q[k]);
                }
                fprintf(ftrace, "\n");
            }
            rounds++;
            if (cfg().log_every > 0 && rounds % cfg().log_every == 0) {
                log_state("state");
            }
        }
        pend    = false;
        r_acc   = -1;
        t_round = now_us;
    }

    // mean acceptance of slot k over all q bins (history + prior), for lookahead past the slot being decided
    float slot_rate(int k) const {
        k = std::max(1, std::min(MAXD - 1, k));
        float o = 0.0f, n = 0.0f;
        for (int b = 0; b < NQB; ++b) { o += ok[k][b]; n += cnt[k][b]; }
        return (o + 8.0f*0.75f) / (n + 8.0f);
    }

    // probe depth: a neighbor of the last verified width whose cost cell is still a prior (0 = none)
    int probe_n(int floor_n, int w) const {
        if (prev_n <= 0) return 0;
        const int up = std::min(w, prev_n + 1), dn = std::max(floor_n, prev_n - 1);
        if (nrest[r_b][up] < cfg().min_cost && up != prev_n) return up;
        if (nrest[r_b][dn] < cfg().min_cost && dn != prev_n) return dn;
        return 0;
    }

    // the walk is read: q[1..w] = each slot's pre-draw feature (the drafter's top probability qmax_k at that slot, which
    // depends only on the earlier slots); returns the slots to keep in [1, w]. legacy = fixed depth before any timing.
    int legacy = 4;
    int choose(int ctx, int w, const float * q, int floor_n, int64_t now_us) {
        const config & c = cfg();
        w = std::max(1, std::min(MAXD - 1, w));
        floor_n = std::max(1, std::min(floor_n, w));
        r_ctx = ctx;
        r_b   = bucket(ctx);
        r_sw  = false;
        r_w   = w;
        r_dms = t_round >= 0 ? (float) (now_us - t_round) * 1e-3f : 0.0f;
        if (t_round >= 0) {
            dms[w] += (ndms[w] == 0 ? 1.0f : 0.05f) * (r_dms - dms[w]);
            ndms[w]++;
        }
        for (int k = 1; k <= w; ++k) {
            r_q[k] = q[k];
        }
        int n = w;
        if (c.sweep) {
            n = w; // the walk is the sweep depth
        } else if (c.probe > 0 && rounds % c.probe == c.probe - 1 && probe_n(floor_n, w) > 0) {
            n = probe_n(floor_n, w); // fixed by history only (an unmeasured neighbor of the last pick)
        } else {
            // a stopping time: slot k is kept or the walk stops using only the slots before it (their kept tokens and
            // slot k's own pre-draw feature qmax_k), never the token drawn at k or any later slot, and a stop is final
            // (no cut back to an earlier, better prefix). Lookahead past k uses the per-slot mean rate (history only).
            const float p = npen[r_b] > 0 ? pen[r_b] : 0.0f;
            const auto cost = [&](int j) {
                const float r = rest_ms(r_b, j);
                return r <= 0.0f ? -1.0f : r_dms + r + (j != prev_n && prev_n > 0 ? p : 0.0f);
            };
            float e = 1.0f, run = 1.0f, best_v = -1.0f;
            n = 0;
            for (int k = 1; k <= w; ++k) {
                const float ck = cost(k);
                if (ck <= 0.0f) {
                    n = std::min(w, legacy); // nothing timed in this bucket yet: the old fixed depth (history only)
                    break;
                }
                const float rk = run * rate(k, q[k]);
                const float ek = e + rk;
                const float vk = ek / ck;
                if (k > floor_n) {
                    float look = vk, e2 = ek, r2 = rk;
                    for (int j = k + 1; j <= std::min(w, k + 2); ++j) {
                        const float cj = cost(j);
                        if (cj <= 0.0f) break;
                        r2 *= slot_rate(j);
                        e2 += r2;
                        look = std::max(look, e2 / cj);
                    }
                    if (look <= best_v) {
                        break;
                    }
                }
                e = ek; run = rk; n = k;
                best_v = std::max(best_v, vk);
            }
            n = std::max(n, std::min(floor_n, w));
            // walk for the next rounds
            hit += 0.05f * ((n >= w ? 1.0f : 0.0f) - hit);
            if (hit > 0.30f && walk < c.max_depth) {
                walk++;
                hit = 0.15f;
            } else if (hit < 0.03f && walk > std::max(c.min_depth, 2)) {
                walk--;
                hit = 0.15f;
            }
        }
        r_n    = n;
        pend   = true;
        n_pick[n]++;
        n_walk[w]++;
        return n;
    }

    // verified: n_drafted kept slots went to the verifier (after any caller truncation), n_acc of them were accepted
    void accept(int n_drafted, int n_acc) {
        if (!pend) {
            return;
        }
        const config & c = cfg();
        r_n   = std::max(0, std::min(r_n, n_drafted));
        r_acc = std::max(0, std::min(n_acc, r_n));
        if (r_n > 0) {
            r_sw   = prev_n > 0 && r_n != prev_n; // the verify width that really ran
            prev_n = r_n;
        }
        for (int k = 1; k <= std::min(r_acc + 1, r_n); ++k) {
            const float g = k <= r_acc ? 1.0f : 0.0f;
            const int   b = qbin(r_q[k]);
            ok[k][b]  = ok[k][b]*c.decay + g;
            cnt[k][b] = cnt[k][b]*c.decay + 1.0f;
            okb[b]    = okb[b]*c.decay + g;
            cntb[b]   = cntb[b]*c.decay + 1.0f;
        }
        if (r_n == 0) {
            pend = false;
        }
    }

    // a round that never reached the verifier (or a request boundary): drop its timing
    void cancel() {
        pend    = false;
        r_acc   = -1;
        t_round = -1;
    }

    void log_state(const char * why) const {
        if (rounds == 0) {
            return;
        }
        fprintf(stderr, "[inkdepth] %s rounds=%lld walk=%d picks=", why, rounds, walk);
        for (int n = 1; n < MAXD; ++n) if (n_pick[n]) fprintf(stderr, "n%d:%lld ", n, n_pick[n]);
        fprintf(stderr, "walks=");
        for (int n = 1; n < MAXD; ++n) if (n_walk[n]) fprintf(stderr, "w%d:%lld(%.2fms) ", n, n_walk[n], dms[n]);
        for (int b = 0; b < NB; ++b) {
            if (nscale[b] == 0) continue;
            fprintf(stderr, "| b%d rest=", b);
            for (int n = 1; n < MAXD; ++n) if (nrest[b][n]) fprintf(stderr, "%d:%.2f(%d) ", n, rest[b][n], nrest[b][n]);
            fprintf(stderr, "pen=%.2f(%d) ", pen[b], npen[b]);
        }
        fprintf(stderr, "| rate(k=1) by q:");
        for (int qb = 0; qb < NQB; ++qb) fprintf(stderr, " %.2f", rate(1, (qb + 0.5f)/NQB));
        fprintf(stderr, "\n");
    }
};
