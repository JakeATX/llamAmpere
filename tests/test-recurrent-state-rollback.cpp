#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-context.h"
#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

static llama_context * make_ctx_n(const common_params & params, llama_model * model, uint32_t n_seq_max) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    return make_ctx_n(params, model, 1);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// ---------------------------------------------------------------------------
// RB1b gates. RB1 (3772c377e) bounded the snapshot WRITER; RB1b bounds the
// READER. `set_rs_idx` clamped a rollback request to n_rs_seq -- the ring
// CAPACITY -- which says nothing about whether those slots were ever filled for
// this sequence, so a rollback into a fresh, recycled or restored sequence was
// accepted and silently restored a state that never existed. rs_valid[seq]
// tracks how many slots behind the head really hold this sequence's data.
//
// Note on reachability: in a clean single-sequence history rs_valid can never
// bind, because seq_rm's own position guard (0 < p0) already caps a rollback at
// cell.pos, and a sequence that decoded cell.pos+1 tokens has at least that many
// valid snapshots. The bound only matters on the paths where the ring holds data
// that is not this sequence's: after a state restore, after seq_cp, and after
// repeated rollbacks have consumed it. Those are exactly the cases below, and
// they are why production never tripped the old unbounded reader.
// ---------------------------------------------------------------------------

static bool decode_one_seq(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { seq }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_tokens_seq(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count, llama_seq_id seq) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { seq }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// Roll a sequence back by `r` tokens from its current head at `head_pos`.
static bool rollback_by(llama_context * ctx, llama_seq_id seq, llama_pos head_pos, uint32_t r) {
    return llama_memory_seq_rm(llama_get_memory(ctx), seq, head_pos - (llama_pos) r + 1, -1);
}

// Replay `n` tokens starting at `from` and capture the logits of each step.
static bool replay_capture(llama_context * ctx, const std::vector<llama_token> & tokens,
                           llama_pos from, uint32_t n, llama_seq_id seq, int n_vocab,
                           std::vector<std::vector<float>> & out) {
    out.assign(n, {});
    for (uint32_t i = 0; i < n; ++i) {
        const llama_pos pos = from + (llama_pos) i;
        if (!decode_one_seq(ctx, tokens[pos], pos, seq)) {
            return false;
        }
        const float * lg = llama_get_logits_ith(ctx, 0);
        if (lg == nullptr) {
            return false;
        }
        out[i].assign(lg, lg + n_vocab);
    }
    return true;
}

static bool logits_match(const std::vector<std::vector<float>> & a,
                         const std::vector<std::vector<float>> & b, float eps, const char * what) {
    if (a.size() != b.size()) {
        fprintf(stderr, "rb1b : %s replay length mismatch (%zu != %zu)\n", what, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            fprintf(stderr, "rb1b : %s vocab mismatch at step %zu\n", what, i);
            return false;
        }
        for (size_t t = 0; t < a[i].size(); ++t) {
            if (std::fabs(a[i][t] - b[i][t]) > eps) {
                fprintf(stderr, "rb1b : %s logits mismatch at step %zu, token %zu (%g != %g)\n",
                        what, i, t, (double) a[i][t], (double) b[i][t]);
                return false;
            }
        }
    }
    return true;
}

static int run_rb1b_gates(const common_params & params, llama_model * model,
                          const std::vector<llama_token> & tokens, uint32_t n_rs_seq, int n_vocab) {
    constexpr float eps = 1e-5f;
    const uint32_t  n_tokens = tokens.size();
    const llama_pos head     = (llama_pos) n_tokens - 1;
    int failures = 0;

    // --- Gate A: the depth boundary on a short history, and a correction to the plan.
    // RB1b's plan phrased case (a) as "rollback on a sequence with fewer than n_rs_seq decoded
    // tokens must be REFUSED". That is not reachable, and it is not what correct behaviour would
    // be. A sequence that decoded k tokens has k valid snapshots and may legally roll back k-1 of
    // them regardless of ring capacity. And a DEEPER request cannot even reach the rs_valid bound:
    // rolling back r from head h means seq_rm(p0 = h-r+1), and the rollback branch is guarded by
    // `0 < p0`, so r = k (p0 = 0) is not a rollback at all -- it is a full erase of the sequence,
    // which falls through to the tail-invalidation path and correctly returns true. There is no
    // over-deep rollback expressible through seq_rm on a clean single-sequence history.
    //
    // So rs_valid can only ever bind where the ring holds data that is NOT this sequence's:
    // after a state restore (gate D), after seq_cp (gate C), or after rollbacks have consumed it
    // (gate B). That is exactly why the unbounded reader never tripped in production. This gate
    // pins the one thing that is checkable here -- the deepest legal rollback is still accepted,
    // i.e. rs_valid is not under-counting and silently disabling legitimate rollbacks.
    {
        const uint32_t k = std::min<uint32_t>(4, n_tokens);
        if (k < 3 || k >= n_rs_seq) {
            fprintf(stderr, "rb1b : gate A SKIP -- need 3 <= k < n_rs_seq (k=%u n_rs_seq=%u)\n", k, n_rs_seq);
        } else {
            llama_context * ctx_s = make_ctx(params, model);
            if (ctx_s == nullptr) {
                fprintf(stderr, "rb1b : gate A context init failed\n");
                return 1;
            }
            const llama_pos h = (llama_pos) k - 1;
            if (!decode_tokens_seq(ctx_s, tokens, k, 0)) {
                fprintf(stderr, "rb1b : gate A FAIL -- short decode failed\n");
                failures++;
            } else if (!rollback_by(ctx_s, 0, h, k - 1)) {
                fprintf(stderr, "rb1b : gate A FAIL -- the deepest legal rollback (%u on a %u-token "
                                "history) was refused; rs_valid is under-counting\n", k - 1, k);
                failures++;
            } else {
                fprintf(stderr, "rb1b : gate A PASS -- %u-token history accepts its deepest legal "
                                "rollback of %u\n", k, k - 1);
            }
            llama_free(ctx_s);
        }
    }

    // --- Gate B: rollbacks COMPOSE. Two seq_rm of 1 before any decode must land
    // exactly where one seq_rm of 2 lands. The old code assigned rs_idx instead
    // of adding to it, so the second rollback restored a state one token too new.
    {
        std::vector<std::vector<float>> lg_two, lg_one;
        bool ok = true;
        {   // 1 + 1, then free before the second context exists
            llama_context * ctx_two = make_ctx(params, model);
            if (ctx_two == nullptr) { fprintf(stderr, "rb1b : gate B context init failed\n"); return 1; }
            ok = decode_tokens_seq(ctx_two, tokens, n_tokens, 0) &&
                 rollback_by(ctx_two, 0, head,     1) &&
                 rollback_by(ctx_two, 0, head - 1, 1) &&
                 replay_capture(ctx_two, tokens, head - 1, 2, 0, n_vocab, lg_two);
            llama_free(ctx_two);
        }
        if (ok) {   // one shot of 2
            llama_context * ctx_one = make_ctx(params, model);
            if (ctx_one == nullptr) { fprintf(stderr, "rb1b : gate B context init failed\n"); return 1; }
            ok = decode_tokens_seq(ctx_one, tokens, n_tokens, 0) &&
                 rollback_by(ctx_one, 0, head, 2) &&
                 replay_capture(ctx_one, tokens, head - 1, 2, 0, n_vocab, lg_one);
            llama_free(ctx_one);
        }
        if (!ok) {
            fprintf(stderr, "rb1b : gate B FAIL -- a rollback that should be legal was refused, "
                            "or its replay failed\n");
            failures++;
        } else if (!logits_match(lg_two, lg_one, eps, "gate B (1+1 vs 2)")) {
            failures++;
        } else {
            fprintf(stderr, "rb1b : gate B PASS -- 1+1 composes to 2\n");
        }
    }

    // --- Gate D: a sequence restored from a state blob has NO valid snapshots of
    // its own, so a rollback must be REFUSED, not silently served from whatever
    // the ring happened to contain. After refilling the ring it must work again.
    {
        std::vector<uint8_t> blob;
        size_t got = 0;
        {
            llama_context * ctx_a = make_ctx(params, model);
            if (ctx_a == nullptr) { fprintf(stderr, "rb1b : gate D context init failed\n"); return 1; }
            if (decode_tokens_seq(ctx_a, tokens, n_tokens, 0)) {
                blob.resize(llama_state_seq_get_size(ctx_a, 0));
                got = llama_state_seq_get_data(ctx_a, blob.data(), blob.size(), 0);
            }
            llama_free(ctx_a);
        }
        llama_context * ctx_b = make_ctx(params, model);
        if (ctx_b == nullptr) {
            fprintf(stderr, "rb1b : gate D context init failed\n");
            return 1;
        }
        if (got == 0) {
            fprintf(stderr, "rb1b : gate D FAIL -- source decode or state save failed\n");
            failures++;
        } else {
            const size_t set = llama_state_seq_set_data(ctx_b, blob.data(), got, 0);
            if (set == 0) {
                fprintf(stderr, "rb1b : gate D SKIP -- state seq restore unavailable (got=%zu set=%zu)\n", got, set);
            } else if (rollback_by(ctx_b, 0, head, 2)) {
                fprintf(stderr, "rb1b : gate D FAIL -- rollback accepted on a freshly restored sequence "
                                "with no snapshots of its own\n");
                failures++;
            } else {
                // Refill the ring, then the same rollback must be accepted.
                bool ok = true;
                for (uint32_t i = 0; i < n_rs_seq && ok; ++i) {
                    ok = decode_one_seq(ctx_b, tokens[i % n_tokens], head + 1 + (llama_pos) i, 0);
                }
                if (!ok) {
                    fprintf(stderr, "rb1b : gate D FAIL -- refill decode failed\n");
                    failures++;
                } else if (!rollback_by(ctx_b, 0, head + (llama_pos) n_rs_seq, 2)) {
                    fprintf(stderr, "rb1b : gate D FAIL -- rollback still refused after refilling the ring\n");
                    failures++;
                } else {
                    fprintf(stderr, "rb1b : gate D PASS -- refused after restore, accepted after refill\n");
                }
            }
        }
        llama_free(ctx_b);
    }

    // --- Gate C: seq_cp must carry the rollback bookkeeping. A branch created by
    // seq_cp inherits the source's tail cell and therefore its snapshot ring, so
    // rolling the DESTINATION back must give the same thing as rolling the SOURCE
    // back. Before RB1b the destination inherited rs_valid == 0 and could not roll
    // back at all.
    {
        std::vector<std::vector<float>> lg_dst, lg_ref;
        bool ok = true, skip = false;
        {
            llama_context * ctx_cp = make_ctx_n(params, model, 2);
            if (ctx_cp == nullptr) {
                fprintf(stderr, "rb1b : gate C SKIP -- could not init a 2-sequence context\n");
                skip = true;
            } else {
                ok = decode_tokens_seq(ctx_cp, tokens, n_tokens, 0);
                if (ok) {
                    llama_memory_seq_cp(llama_get_memory(ctx_cp), 0, 1, -1, -1);
                }
                ok = ok && rollback_by(ctx_cp, 1, head, 2) &&
                     replay_capture(ctx_cp, tokens, head - 1, 2, 1, n_vocab, lg_dst);
                llama_free(ctx_cp);
            }
        }
        if (!skip && ok) {
            llama_context * ctx_ref = make_ctx(params, model);
            if (ctx_ref == nullptr) { fprintf(stderr, "rb1b : gate C context init failed\n"); return 1; }
            ok = decode_tokens_seq(ctx_ref, tokens, n_tokens, 0) &&
                 rollback_by(ctx_ref, 0, head, 2) &&
                 replay_capture(ctx_ref, tokens, head - 1, 2, 0, n_vocab, lg_ref);
            llama_free(ctx_ref);
        }
        if (skip) {
            // already reported
        } else if (!ok) {
            fprintf(stderr, "rb1b : gate C FAIL -- rollback on the seq_cp destination (or the "
                            "reference) was refused, or its replay failed\n");
            failures++;
        } else if (!logits_match(lg_dst, lg_ref, eps, "gate C (seq_cp dst vs src)")) {
            failures++;
        } else {
            fprintf(stderr, "rb1b : gate C PASS -- seq_cp destination rolls back like the source\n");
        }
    }

    // --- Gate E: the negative seq_id wildcard. seq_rm(-1, -1, -1) means "every
    // sequence, whole range" and must succeed; a partial-range wildcard
    // (seq_rm(-1, p0 > 0, -1)) is a rollback that cannot be applied per sequence
    // and must be refused. Before the fix, seq_id == -1 fell through the
    // `seq_id >= n_seq_max` guard as an unsigned compare and returned false.
    {
        llama_context * ctx_e = make_ctx(params, model);
        if (ctx_e == nullptr) { fprintf(stderr, "rb1b : gate E context init failed\n"); return 1; }
        bool ok = decode_tokens_seq(ctx_e, tokens, n_tokens, 0);
        llama_memory_t mem = llama_get_memory(ctx_e);
        const bool partial_refused = ok && !llama_memory_seq_rm(mem, -1, 1, -1);
        const bool full_ok         = ok && llama_memory_seq_rm(mem, -1, -1, -1);
        const llama_pos pos_after  = ok ? llama_memory_seq_pos_max(mem, 0) : 0;
        // the context must be usable again afterwards
        const bool redecode_ok = ok && decode_tokens_seq(ctx_e, tokens, n_tokens, 0);
        llama_free(ctx_e);
        if (!ok) {
            fprintf(stderr, "rb1b : gate E FAIL -- decode failed\n");
            failures++;
        } else if (!partial_refused) {
            fprintf(stderr, "rb1b : gate E FAIL -- partial wildcard seq_rm(-1, 1, -1) was accepted\n");
            failures++;
        } else if (!full_ok || pos_after != -1) {
            fprintf(stderr, "rb1b : gate E FAIL -- seq_rm(-1, -1, -1) returned %s, pos_max after = %d\n",
                full_ok ? "true" : "false", (int) pos_after);
            failures++;
        } else if (!redecode_ok) {
            fprintf(stderr, "rb1b : gate E FAIL -- decode after seq_rm(-1, -1, -1) failed\n");
            failures++;
        } else {
            fprintf(stderr, "rb1b : gate E PASS -- seq_rm(-1,-1,-1) clears, partial wildcard refused\n");
        }
    }

    // --- Gate F: a rollback after a run of 1-token decodes restores the EXACT state. Gates A-D roll back within
    // one multi-token ubatch; here every snapshot behind the head comes from a separate 1-token ubatch, so the ring
    // must have accumulated them (DSV4 once wrote the pre-ubatch state into every plane deeper than the ubatch and
    // accepted the rollback). Prefill P tokens, decode n_single more one at a time, roll back r (in one seq_rm, and
    // as r-1 then 1), replay, and compare bitwise against a fresh context that never went past the rollback point.
    // Depths 1..max_depth and n_rs_seq must be accepted; one depth past the ring capacity (n_rs_seq + 1, still inside the
    // decoded history) must be refused, in one call and as n_rs_seq then 1 -- not skipped.
    {
        constexpr uint32_t n_single   = 8;
        constexpr uint32_t n_replay_f = 3;
        constexpr uint32_t max_depth  = 4;

        std::vector<llama_token> toks(9 + n_single + n_replay_f);
        for (size_t i = 0; i < toks.size(); ++i) {
            toks[i] = (llama_token) ((7*i + 3) % (size_t) n_vocab);
        }

        // prefill `n_prefill` in one batch, then single tokens up to (excluding) position `end`
        const auto build = [&](llama_context * ctx, uint32_t n_prefill, llama_pos end) {
            if (!decode_tokens_seq(ctx, toks, n_prefill, 0)) {
                return false;
            }
            for (llama_pos p = (llama_pos) n_prefill; p < end; ++p) {
                if (!decode_one_seq(ctx, toks[p], p, 0)) {
                    return false;
                }
            }
            return true;
        };

        int n_checked = 0;
        int n_refused = 0;
        const int failures_before = failures;
        for (uint32_t n_prefill : { 3u, 9u }) {
            const llama_pos end = (llama_pos) (n_prefill + n_single); // head = end - 1
            std::vector<uint32_t> depths;
            for (uint32_t r = 1; r <= max_depth; ++r) {
                depths.push_back(r);
            }
            if (n_rs_seq > max_depth && (llama_pos) n_rs_seq < end) {
                depths.push_back(n_rs_seq);     // the deepest plane, reached only through n_single shifts
            }
            if ((llama_pos) (n_rs_seq + 1) < end) {
                depths.push_back(n_rs_seq + 1); // past the ring, p0 = end - r still > 0
            }
            for (uint32_t r : depths) {
                const bool expect_ok = r <= n_rs_seq;

                std::vector<std::vector<float>> lg_ref;
                if (expect_ok) {
                    llama_context * ctx_ref = make_ctx(params, model);
                    if (ctx_ref == nullptr) { fprintf(stderr, "rb1b : gate F context init failed\n"); return 1; }
                    const bool ok = build(ctx_ref, n_prefill, end - (llama_pos) r) &&
                                    replay_capture(ctx_ref, toks, end - (llama_pos) r, n_replay_f, 0, n_vocab, lg_ref);
                    llama_free(ctx_ref);
                    if (!ok) {
                        fprintf(stderr, "rb1b : gate F FAIL -- reference decode failed (P=%u r=%u)\n", n_prefill, r);
                        failures++;
                        continue;
                    }
                }

                for (bool two_step : { false, true }) {
                    if (two_step && r < 2) {
                        continue;
                    }
                    llama_context * ctx_rb = make_ctx(params, model);
                    if (ctx_rb == nullptr) { fprintf(stderr, "rb1b : gate F context init failed\n"); return 1; }

                    bool ok = build(ctx_rb, n_prefill, end);
                    bool accepted = false;
                    if (ok) {
                        llama_memory_t mem = llama_get_memory(ctx_rb);
                        const llama_pos p0 = end - (llama_pos) r;
                        accepted = two_step ? llama_memory_seq_rm(mem, 0, p0 + 1, -1) && llama_memory_seq_rm(mem, 0, p0, -1)
                                            : llama_memory_seq_rm(mem, 0, p0, -1);
                    }

                    std::vector<std::vector<float>> lg_rb;
                    const char * mode = two_step ? "two-step" : "single";
                    if (!ok) {
                        fprintf(stderr, "rb1b : gate F FAIL -- decode failed (P=%u r=%u %s)\n", n_prefill, r, mode);
                        failures++;
                    } else if (accepted != expect_ok) {
                        fprintf(stderr, "rb1b : gate F FAIL -- rollback of %u after %u single decodes was %s "
                                        "(P=%u %s, n_rs_seq=%u)\n", r, n_single, accepted ? "accepted" : "refused",
                                        n_prefill, mode, n_rs_seq);
                        failures++;
                    } else if (!accepted) {
                        n_refused++;
                    } else if (!replay_capture(ctx_rb, toks, end - (llama_pos) r, n_replay_f, 0, n_vocab, lg_rb)) {
                        fprintf(stderr, "rb1b : gate F FAIL -- replay failed (P=%u r=%u %s)\n", n_prefill, r, mode);
                        failures++;
                    } else {
                        char what[96];
                        snprintf(what, sizeof(what), "gate F (P=%u r=%u %s vs fresh)", n_prefill, r, mode);
                        if (!logits_match(lg_rb, lg_ref, 0.0f, what)) {
                            failures++;
                        } else {
                            n_checked++;
                        }
                    }
                    llama_free(ctx_rb);
                }
            }
        }
        if (failures == failures_before) {
            fprintf(stderr, "rb1b : gate F PASS -- %d rollbacks after 1-token decodes match a fresh context exactly, "
                            "%d refused beyond the ring\n", n_checked, n_refused);
        }
    }

    // --- Gate G: two sequences decoding 1 token each in one shared ubatch, with different pending rollbacks.
    // In the shift layout the ubatch moves the older snapshot groups of every lane by one shared amount; if that
    // amount follows the deepest pending rollback in the ubatch (seq A), the other lane (seq B, nothing pending)
    // keeps groups its rs_valid still credits, and a deep rollback on B later restores a stale state.
    // Schedule: prefill A and B together, n_single joint 1-token decodes, roll A back by rb_a, one more joint
    // 1-token decode (A replays, B advances), then roll B back by each depth in depths_b (and A by its full
    // remaining credit, plus one past it, which must be refused), replay 3 tokens of that sequence alone, and
    // compare bitwise against a fresh context that ran the same joint schedule only up to the rollback point.
    {
        constexpr uint32_t n_prefill  = 9;
        constexpr uint32_t n_single   = 8;
        constexpr uint32_t n_replay_g = 3;
        constexpr uint32_t rb_a       = 6;

        const auto tok_of = [&](llama_seq_id seq, llama_pos pos) {
            return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq + 3) % (uint32_t) n_vocab);
        };
        std::vector<std::vector<llama_token>> toks_g(2, std::vector<llama_token>(n_prefill + n_single + n_replay_g + 1));
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            for (size_t i = 0; i < toks_g[seq].size(); ++i) {
                toks_g[seq][i] = tok_of(seq, (llama_pos) i);
            }
        }

        const auto make_ctx_g = [&]() {
            auto cparams = common_context_params_to_llama(params);
            cparams.n_seq_max  = 2;
            cparams.n_rs_seq   = n_rs_seq;
            cparams.n_ctx      = 256;
            cparams.n_batch    = 64;
            cparams.n_ubatch   = 64;
            cparams.kv_unified = false;
            return llama_init_from_model(model, cparams);
        };

        // one batch holding (seq, first pos, count) runs; only the first token is an output, so the
        // batch is not all-output and both sequences share each equal-split ubatch
        struct run { llama_seq_id seq; llama_pos p0; uint32_t n; };
        const auto decode_joint = [&](llama_context * ctx, std::initializer_list<run> runs) {
            uint32_t n_tok = 0;
            for (const run & r : runs) {
                n_tok += r.n;
            }
            llama_batch batch = llama_batch_init(n_tok, 0, 1);
            bool first = true;
            for (const run & r : runs) {
                for (uint32_t i = 0; i < r.n; ++i) {
                    const llama_pos pos = r.p0 + (llama_pos) i;
                    common_batch_add(batch, toks_g[r.seq][pos], pos, { r.seq }, first);
                    first = false;
                }
            }
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            return ok;
        };

        // joint prefill, then joint 1-token decodes for steps [0, n_steps)
        const auto build = [&](llama_context * ctx, uint32_t n_steps) {
            if (!decode_joint(ctx, { { 0, 0, n_prefill }, { 1, 0, n_prefill } })) {
                return false;
            }
            for (uint32_t t = 0; t < n_steps; ++t) {
                const llama_pos pos = (llama_pos) (n_prefill + t);
                if (!decode_joint(ctx, { { 0, pos, 1 }, { 1, pos, 1 } })) {
                    return false;
                }
            }
            return true;
        };

        // after build(n_single) both heads are at end - 1; A rolls back rb_a, then one joint step:
        // A's head is end - rb_a, B's head is end
        const llama_pos end    = (llama_pos) (n_prefill + n_single);
        const llama_pos head_a = end - (llama_pos) rb_a;
        const llama_pos head_b = end;

        struct check { llama_seq_id seq; uint32_t depth; };
        std::vector<check> checks;
        for (uint32_t d : { 2u, 4u, n_rs_seq }) {
            checks.push_back({ 1, d });
        }
        // A: rs_valid = n_rs_seq - rb_a + 1 after the joint step; one deeper is refused
        checks.push_back({ 0, n_rs_seq - rb_a + 1 });
        checks.push_back({ 0, n_rs_seq - rb_a + 2 });

        int n_checked = 0;
        int n_refused = 0;
        const int failures_before = failures;
        for (const check & c : checks) {
            const char    what_seq  = c.seq == 0 ? 'A' : 'B';
            const llama_pos head    = c.seq == 0 ? head_a : head_b;
            const llama_pos p0      = head - (llama_pos) c.depth + 1;
            const bool    expect_ok = c.seq == 1 || c.depth <= n_rs_seq - rb_a + 1;

            std::vector<std::vector<float>> lg_ref;
            if (expect_ok) {
                // the reference stops the joint schedule where the checked sequence's history ends at p0
                // (both sequences share the position line until A's rollback)
                llama_context * ctx_ref = make_ctx_g();
                if (ctx_ref == nullptr) { fprintf(stderr, "rb1b : gate G context init failed\n"); return 1; }
                const bool ok = p0 >= (llama_pos) n_prefill &&
                                build(ctx_ref, (uint32_t) (p0 - (llama_pos) n_prefill)) &&
                                replay_capture(ctx_ref, toks_g[c.seq], p0, n_replay_g, c.seq, n_vocab, lg_ref);
                llama_free(ctx_ref);
                if (!ok) {
                    fprintf(stderr, "rb1b : gate G FAIL -- reference decode failed (seq %c depth %u)\n", what_seq, c.depth);
                    failures++;
                    continue;
                }
            }

            llama_context * ctx_rb = make_ctx_g();
            if (ctx_rb == nullptr) { fprintf(stderr, "rb1b : gate G context init failed\n"); return 1; }
            llama_memory_t mem = llama_get_memory(ctx_rb);
            bool ok = build(ctx_rb, n_single) &&
                      rollback_by(ctx_rb, 0, end - 1, rb_a) &&
                      decode_joint(ctx_rb, { { 0, head_a, 1 }, { 1, head_b, 1 } });
            const bool accepted = ok && llama_memory_seq_rm(mem, c.seq, p0, -1);

            std::vector<std::vector<float>> lg_rb;
            if (!ok) {
                fprintf(stderr, "rb1b : gate G FAIL -- joint schedule failed (seq %c depth %u)\n", what_seq, c.depth);
                failures++;
            } else if (accepted != expect_ok) {
                fprintf(stderr, "rb1b : gate G FAIL -- rollback of seq %c by %u after a shared ubatch with seq A's "
                                "pending rollback of %u was %s\n", what_seq, c.depth, rb_a, accepted ? "accepted" : "refused");
                failures++;
            } else if (!accepted) {
                n_refused++;
            } else if (!replay_capture(ctx_rb, toks_g[c.seq], p0, n_replay_g, c.seq, n_vocab, lg_rb)) {
                fprintf(stderr, "rb1b : gate G FAIL -- replay failed (seq %c depth %u)\n", what_seq, c.depth);
                failures++;
            } else {
                char what[96];
                snprintf(what, sizeof(what), "gate G (seq %c depth %u vs fresh)", what_seq, c.depth);
                if (!logits_match(lg_rb, lg_ref, 0.0f, what)) {
                    failures++;
                } else {
                    n_checked++;
                }
            }
            llama_free(ctx_rb);
        }
        if (failures == failures_before) {
            fprintf(stderr, "rb1b : gate G PASS -- %d rollbacks after a shared ubatch with different pending rollbacks "
                            "match a fresh context exactly, %d refused past the credit\n", n_checked, n_refused);
        }
    }

    fprintf(stderr, "rb1b : %d gate failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
// true when every backend the context schedules on is a CPU device (no GPU, no accelerator)
static bool ctx_all_cpu(llama_context * ctx) {
    ggml_backend_sched_t sched = ctx->get_sched();
    if (sched == nullptr) {
        return false;
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_dev_t dev = ggml_backend_get_device(ggml_backend_sched_get_backend(sched, i));
        if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return false;
        }
    }
    return true;
}

static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // upstream refuses any second partial removal while one is pending (single-use rs_idx);
        // in this tree rollbacks compose (rb1b gate B), bounded by what the ring still holds.
        // Here 8 slots were filled and 3 are pending, so a further removal of 8 (deeper than
        // the 5 remaining) must be refused and must leave the pending rollback untouched.
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s,
                                        p0 - (llama_pos) llama_n_rs_seq(ctx_roll), -1);
        ok = ok && llama_memory_seq_pos_max(llama_get_memory(ctx_roll), (llama_seq_id) s) == (llama_pos) p0 - 1;
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // Both contexts run the identical prefill and the identical replay batch, so the only
    // difference is where ctx_roll's replay starts from: the restored rollback snapshot.
    // Exact on CPU: same ubatch shapes, same kernels, so the logits must match bit for bit.
    // On any other backend the bound is nmse <= 1e-10, which is provisional until a GPU run
    // confirms exactness (the queue owner runs it). It must stay far below what a stale
    // snapshot does here: the tiny test models barely lean on the recurrent state, and the
    // nemotron-h rollback before the snapshot shift moved the logits by only 2e-4 (nmse 3.6e-8),
    // which the old nmse <= 1e-5 bound passed.
    const bool   exact    = ctx_all_cpu(ctx_roll) && ctx_all_cpu(ctx_ref);
    const double nmse_eps = 1e-10;
    const auto mismatch = [&](float diff, double nmse_v) {
        return exact ? diff > 0.0f : !(nmse_v <= nmse_eps);
    };
    const char * rule = exact ? "exact, CPU" : "nmse <= 1e-10, non-CPU backend";

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (mismatch(diff_max, nmse_val)) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (%s; max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, rule, (double) diff_max, nmse_val, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (%s; max diff %g, nmse %g)\n", __func__, rule, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || mismatch(diff_tail, nmse_tail)) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (%s; ok=%d, max diff %g, nmse %g)\n",
                __func__, rule, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (%s; max diff %g, nmse %g)\n", __func__, rule, (double) diff_tail, nmse_tail);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // TODO: use smart pointers
    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                fprintf(stderr, "%s : %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);

    // RB1b reader-bound gates. Run after the original checkpoint test so a failure here is
    // unambiguously the new bookkeeping and not the pre-existing path -- and only after the
    // three contexts above are freed, because on a 27B model each rs cache is over a GiB and
    // holding the originals alive alongside the gates' own is an out-of-memory, not a result.
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    // RB1b reader-bound gates (the contexts above are freed): n_rs_seq matches make_ctx's cparams
    if (run_rb1b_gates(params, model, tokens, 8, n_vocab) != 0) {
        return 1;
    }

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return 0;
}
