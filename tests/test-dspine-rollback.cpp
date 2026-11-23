// [DSPINE-V07] recurrent-state rollback for DSpine-width verify rounds (CPU, generated hybrid model, e.g.
// test-llama-archs -a qwen35 -> qwen35-dense.gguf).
//
// A DSpine round verifies 1 + 15 rows and keeps 1 + a of them (a = accepted drafts, 0..15). With n_rs_seq = 15 every
// a must roll back in place. For the default ring layout (one full state plane per snapshot) and for --gdn-replay
// (one checkpoint plane + per-token ingredients), and for several rounds in a row, the next-token logits after the
// rollback must match a context that decoded only the kept tokens. Also 4 sequences interleaved in one context (F8).
//
// usage: test-dspine-rollback -m qwen35-dense.gguf
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static constexpr uint32_t N_RS  = 15;
static constexpr int32_t  BLOCK = 16;

static llama_context * make_ctx(const common_params & params, llama_model * model, bool gdn_replay, uint32_t n_seq = 1,
        uint32_t n_rs = N_RS) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq;
    cparams.n_ctx      = 512 * n_seq;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.n_rs_seq   = n_rs;
    cparams.gdn_replay = gdn_replay;
    return llama_init_from_model(model, cparams);
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0, bool all_logits) {
    if (toks.empty()) {
        return true;
    }
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, all_logits || i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

static double nmse(const float * a, const float * b, int n) {
    double e = 0.0, r = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        e += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        r += (double) a[i] * a[i];
    }
    return r > 0.0 ? e / r : e;
}

// rounds: kept[r] = accepted drafts of round r (0..15). worst = the worst nmse, or +inf on an error.
// [DSPINE4] seg_worst: the same probe vs a rollback-free reference decoded in the tested context's segment shapes
// (prefix, then per round the 1 + a kept rows as one batch and the 1-row probe); floor_worst: one-batch reference vs that
// segmented reference (the batch-shape rounding between two contexts that never roll back).
struct run_result {
    double worst = 0.0, seg_worst = 0.0, floor_worst = 0.0;
};

static run_result run(const common_params & params, llama_model * model, bool gdn_replay, const std::vector<int32_t> & kept,
        uint32_t seed, int n_vocab) {
    run_result res;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<llama_token>> segs; // the tested context's kept segments, in decode order
    std::mt19937 g(seed);
    std::uniform_int_distribution<int32_t> tok(1, std::max(2, std::min(n_vocab - 1, 1000)));
    std::vector<llama_token> prefix(12);
    for (auto & t : prefix) {
        t = tok(g);
    }

    llama_context * ctx = make_ctx(params, model, gdn_replay);
    if (!ctx) {
        fprintf(stderr, "context init failed (gdn_replay=%d)\n", gdn_replay);
        res.worst = inf;
        return res;
    }
    if (llama_n_rs_seq(ctx) < N_RS) {
        fprintf(stderr, "n_rs_seq %u < %u\n", llama_n_rs_seq(ctx), N_RS);
        llama_free(ctx);
        res.worst = inf;
        return res;
    }

    std::vector<llama_token> history = prefix; // tokens the sequence really holds
    double worst = 0.0;
    bool ok = decode(ctx, prefix, 0, false);
    segs.push_back(prefix);
    for (size_t r = 0; ok && r < kept.size(); ++r) {
        const int32_t a = kept[r];
        std::vector<llama_token> blk(BLOCK);
        for (auto & t : blk) {
            t = tok(g);
        }
        const llama_pos p0 = (llama_pos) history.size();
        ok = decode(ctx, blk, p0, true);
        // keep the round's first token (the target's own) plus a accepted drafts
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), 0, p0 + 1 + a, -1);
        ok = ok && llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == p0 + a;
        if (!ok) {
            fprintf(stderr, "round %zu (a=%d, gdn_replay=%d): verify/rollback refused\n", r, a, gdn_replay);
            break;
        }
        history.insert(history.end(), blk.begin(), blk.begin() + 1 + a);
        segs.emplace_back(blk.begin(), blk.begin() + 1 + a);

        // next token on the rolled-back context vs a fresh context that decoded only the kept tokens
        const llama_token nxt = tok(g);
        ok = decode(ctx, { nxt }, (llama_pos) history.size(), true);
        llama_context * ref = make_ctx(params, model, gdn_replay);
        ok = ok && ref && decode(ref, history, 0, false) && decode(ref, { nxt }, (llama_pos) history.size(), true);
        if (!ok) {
            fprintf(stderr, "round %zu: next-token decode failed\n", r);
            if (ref) {
                llama_free(ref);
            }
            break;
        }
        const double e = nmse(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ref, -1), n_vocab);
        // segmented, rollback-free reference: same segment shapes as ctx (kept rows, then the 1-row probe)
        double es = inf, ef = inf;
        llama_context * seg = make_ctx(params, model, gdn_replay);
        if (seg) {
            bool sok = true;
            llama_pos pos = 0;
            for (const auto & sg : segs) {
                sok = sok && decode(seg, sg, pos, false);
                pos += (llama_pos) sg.size();
            }
            sok = sok && decode(seg, { nxt }, pos, true);
            if (sok) {
                es = nmse(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(seg, -1), n_vocab);
                ef = nmse(llama_get_logits_ith(ref, -1), llama_get_logits_ith(seg, -1), n_vocab);
            }
            llama_free(seg);
        }
        fprintf(stderr, "  gdn_replay=%d round %zu kept 1+%2d: nmse %.3e (same-shape ref %.3e, one-batch vs same-shape floor %.3e)\n",
                gdn_replay, r, a, e, es, ef);
        worst = std::max(worst, e);
        res.seg_worst   = std::max(res.seg_worst, es);
        res.floor_worst = std::max(res.floor_worst, ef);
        history.push_back(nxt);
        segs.push_back({ nxt });
        llama_free(ref);
    }
    llama_free(ctx);
    res.worst = ok ? worst : inf;
    return res;
}

// [DSPINE4] shape-matched probe. run() probes each rollback with a 1-row decode, while its reference decodes that row inside
// a long batch; on CUDA the 1-row and the many-row matmul paths round differently, so later rounds differ at ~1e-5 even
// when the state is right. Here every row of the rolled-back context is decoded inside a 16-row verify block (as the
// server does: the next round's row 0 is the target's own token), and the probe is that block's row-0 logits. The
// reference decodes history + row 0 in one batch. A state mix-up would show here as it does in run(); rounding from
// batch shape should not. ctrl = the same reference with the last kept token dropped (must be far above tol, so the
// probe can see a one-token state error).
struct matched_result {
    double worst = 0.0;
    double ctrl_min = std::numeric_limits<double>::infinity();
};

static matched_result run_matched(const common_params & params, llama_model * model, bool gdn_replay,
        const std::vector<int32_t> & kept, uint32_t seed, int n_vocab) {
    matched_result res;
    std::mt19937 g(seed);
    std::uniform_int_distribution<int32_t> tok(1, std::max(2, std::min(n_vocab - 1, 1000)));
    std::vector<llama_token> history(12);
    for (auto & t : history) {
        t = tok(g);
    }
    llama_context * ctx = make_ctx(params, model, gdn_replay);
    if (!ctx || llama_n_rs_seq(ctx) < N_RS) {
        if (ctx) {
            llama_free(ctx);
        }
        res.worst = std::numeric_limits<double>::infinity();
        return res;
    }
    const size_t n_prefix = history.size();
    bool ok = decode(ctx, history, 0, false);
    for (size_t r = 0; ok && r <= kept.size(); ++r) {
        std::vector<llama_token> blk(BLOCK);
        for (auto & t : blk) {
            t = tok(g);
        }
        const llama_pos p0 = (llama_pos) history.size();
        ok = decode(ctx, blk, p0, true);
        if (!ok) {
            break;
        }
        // probe: row 0 of this block vs a fresh context that decoded history + row 0 in one batch
        std::vector<llama_token> seq = history;
        seq.push_back(blk[0]);
        llama_context * ref = make_ctx(params, model, gdn_replay);
        ok = ref && decode(ref, seq, 0, true);
        if (ok) {
            const double e = nmse(llama_get_logits_ith(ctx, 0), llama_get_logits_ith(ref, -1), n_vocab);
            fprintf(stderr, "  matched gdn_replay=%d round %zu: nmse %.3e\n", gdn_replay, r, e);
            res.worst = std::max(res.worst, e);
        }
        if (ref) {
            llama_free(ref);
        }
        if (ok && history.size() > n_prefix) {
            std::vector<llama_token> bad(history.begin(), history.end() - 1);
            bad.push_back(blk[0]);
            llama_context * rc = make_ctx(params, model, gdn_replay);
            if (rc && decode(rc, bad, 0, true)) {
                const double c = nmse(llama_get_logits_ith(ctx, 0), llama_get_logits_ith(rc, -1), n_vocab);
                res.ctrl_min = std::min(res.ctrl_min, c);
            }
            if (rc) {
                llama_free(rc);
            }
        }
        if (!ok || r == kept.size()) {
            break;
        }
        const int32_t a = kept[r];
        ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, p0 + 1 + a, -1) &&
             llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == p0 + a;
        history.insert(history.end(), blk.begin(), blk.begin() + 1 + a);
    }
    llama_free(ctx);
    if (!ok) {
        res.worst = std::numeric_limits<double>::infinity();
    }
    return res;
}

// [review F8] 4 sequences in one context: every round verifies a 16-row block per sequence in ONE batch, then rolls each
// sequence back to its own depth, then decodes one next token per sequence in one batch. Each sequence's logits must
// match a fresh single-sequence context that decoded only that sequence's kept tokens.
static double run_multi(const common_params & params, llama_model * model, bool gdn_replay,
        const std::vector<std::vector<int32_t>> & kept_by_seq, uint32_t seed, int n_vocab) {
    const int32_t n_seq = (int32_t) kept_by_seq.size();
    std::mt19937 g(seed);
    std::uniform_int_distribution<int32_t> tok(1, std::max(2, std::min(n_vocab - 1, 1000)));

    llama_context * ctx = make_ctx(params, model, gdn_replay, (uint32_t) n_seq);
    if (!ctx || llama_n_rs_seq(ctx) < N_RS) {
        fprintf(stderr, "multi: context init failed or n_rs_seq too small (gdn_replay=%d)\n", gdn_replay);
        if (ctx) {
            llama_free(ctx);
        }
        return std::numeric_limits<double>::infinity();
    }
    auto * mem = llama_get_memory(ctx);

    std::vector<std::vector<llama_token>> hist(n_seq);
    llama_batch b = llama_batch_init(BLOCK * n_seq, 0, 1);
    bool ok = true;

    // prefixes of different lengths, one batch
    common_batch_clear(b);
    for (int32_t s = 0; s < n_seq; ++s) {
        hist[s].resize(9 + 2 * s);
        for (size_t i = 0; i < hist[s].size(); ++i) {
            hist[s][i] = tok(g);
            common_batch_add(b, hist[s][i], (llama_pos) i, { s }, i + 1 == hist[s].size());
        }
    }
    ok = llama_decode(ctx, b) == 0;

    double worst = 0.0;
    const size_t n_rounds = kept_by_seq[0].size();
    for (size_t r = 0; ok && r < n_rounds; ++r) {
        // one 16-row verify block per sequence, all in one batch
        std::vector<std::vector<llama_token>> blk(n_seq, std::vector<llama_token>(BLOCK));
        common_batch_clear(b);
        for (int32_t s = 0; s < n_seq; ++s) {
            for (int32_t i = 0; i < BLOCK; ++i) {
                blk[s][i] = tok(g);
                common_batch_add(b, blk[s][i], (llama_pos) (hist[s].size() + i), { s }, true);
            }
        }
        ok = llama_decode(ctx, b) == 0;
        for (int32_t s = 0; ok && s < n_seq; ++s) {
            const int32_t a = kept_by_seq[s][r];
            const llama_pos p0 = (llama_pos) hist[s].size();
            ok = llama_memory_seq_rm(mem, s, p0 + 1 + a, -1) && llama_memory_seq_pos_max(mem, s) == p0 + a;
            if (!ok) {
                fprintf(stderr, "multi round %zu seq %d (a=%d, gdn_replay=%d): rollback refused\n", r, s, a, gdn_replay);
            }
            hist[s].insert(hist[s].end(), blk[s].begin(), blk[s].begin() + 1 + a);
        }
        if (!ok) {
            break;
        }
        // one next token per sequence, one batch
        std::vector<llama_token> nxt(n_seq);
        common_batch_clear(b);
        for (int32_t s = 0; s < n_seq; ++s) {
            nxt[s] = tok(g);
            common_batch_add(b, nxt[s], (llama_pos) hist[s].size(), { s }, true);
        }
        ok = llama_decode(ctx, b) == 0;
        for (int32_t s = 0; ok && s < n_seq; ++s) {
            // [DSPINE4] the reference never rolls back: one snapshot plane (the 4-seq ring case OOMed next to a 14 GB model
            // with 15); n_rs_seq 0 would switch the GDN op and turn gdn_replay off, so it stays above 0
            llama_context * ref = make_ctx(params, model, gdn_replay, 1, 1);
            ok = ref && decode(ref, hist[s], 0, false) && decode(ref, { nxt[s] }, (llama_pos) hist[s].size(), true);
            if (ok) {
                const double e = nmse(llama_get_logits_ith(ctx, s), llama_get_logits_ith(ref, -1), n_vocab);
                fprintf(stderr, "  multi gdn_replay=%d round %zu seq %d kept 1+%2d: nmse %.3e\n", gdn_replay, r, s, kept_by_seq[s][r], e);
                worst = std::max(worst, e);
            } else {
                fprintf(stderr, "multi round %zu seq %d: reference decode failed\n", r, s);
            }
            if (ref) {
                llama_free(ref);
            }
            hist[s].push_back(nxt[s]);
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return ok ? worst : std::numeric_limits<double>::infinity();
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

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_hybrid(model) && !llama_model_is_recurrent(model)) {
        fprintf(stderr, "skipping: not a recurrent/hybrid model\n");
        llama_model_free(model);
        return 0;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // every depth once, then a few multi-round mixes (full block, empty, mid)
    const std::vector<std::vector<int32_t>> plans = {
        { 0 }, { 1 }, { 3 }, { 7 }, { 8 }, { 14 }, { 15 },
        { 4, 15, 0, 7, 2 },
        { 15, 15, 15 },
        { 0, 0, 1, 0 },
    };
    // CPU float order differs between a 16-row and a short batch; this bound is far below any state mix-up
    const double tol = 1e-6;
    int fails = 0;
    // [DSPINE4] verdict per plan:
    //   FAIL  the shape-matched probe (every row inside a 16-row block) is above tol, or the 1-row probe is above tol vs
    //         a rollback-free reference decoded in the same segment shapes (a state error);
    //   PASS  the 1-row probe is within tol of the one-batch reference;
    //   PASS(ROUNDING)  it is not, but it is within tol of the same-shape reference: the excess is the batch-shape
    //         rounding between one batch and the segments (printed as the floor), not the rollback.
    for (const bool replay : { false, true }) {
        for (size_t i = 0; i < plans.size(); ++i) {
            const run_result e = run(params, model, replay, plans[i], 100 + (uint32_t) i, n_vocab);
            const matched_result m = run_matched(params, model, replay, plans[i], 500 + (uint32_t) i, n_vocab);
            const char * verdict = !(m.worst <= tol) || !(e.seg_worst <= tol) ? "FAIL" : e.worst <= tol ? "PASS" : "PASS(ROUNDING)";
            fprintf(stderr, "%s layout=%s plan %zu: worst nmse %.3e (1-row probe), %.3e (same-shape ref), floor %.3e, "
                    "%.3e (shape-matched probe), one-token ctrl %.3e\n", verdict, replay ? "gdn_replay" : "ring", i,
                    e.worst, e.seg_worst, e.floor_worst, m.worst, m.ctrl_min);
            fails += verdict[0] == 'P' ? 0 : 1;
        }
    }
    // 4 sequences, different accept counts per sequence and round (incl. 0 and the full block)
    const std::vector<std::vector<int32_t>> multi = {
        { 0, 15,  3,  7 },
        { 15, 0,  8,  1 },
        { 4,  4, 14,  0 },
        { 7,  2,  0, 15 },
    };
    for (const bool replay : { false, true }) {
        const double e = run_multi(params, model, replay, multi, 900, n_vocab);
        const bool pass = e <= tol;
        fprintf(stderr, "%s layout=%s 4-seq interleaved: worst nmse %.3e\n", pass ? "PASS" : "FAIL", replay ? "gdn_replay" : "ring", e);
        fails += pass ? 0 : 1;
    }
    llama_model_free(model);
    fprintf(stderr, "test-dspine-rollback: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
