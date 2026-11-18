// [DSPINE-V07] recurrent-state rollback for DSpine-width verify rounds (CPU, generated hybrid model, e.g.
// test-llama-archs -a qwen35 -> qwen35-dense.gguf).
//
// A DSpine round verifies 1 + 15 rows and keeps 1 + a of them (a = accepted drafts, 0..15). With n_rs_seq = 15 every
// a must roll back in place. For the default ring layout (one full state plane per snapshot) and for --gdn-replay
// (one checkpoint plane + per-token ingredients), and for several rounds in a row, the next-token logits after the
// rollback must match a context that decoded only the kept tokens.
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

static llama_context * make_ctx(const common_params & params, llama_model * model, bool gdn_replay) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 1;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.n_rs_seq   = N_RS;
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

// rounds: kept[r] = accepted drafts of round r (0..15). Returns the worst nmse, or +inf on an error.
static double run(const common_params & params, llama_model * model, bool gdn_replay, const std::vector<int32_t> & kept,
        uint32_t seed, int n_vocab) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int32_t> tok(1, std::max(2, std::min(n_vocab - 1, 1000)));
    std::vector<llama_token> prefix(12);
    for (auto & t : prefix) {
        t = tok(g);
    }

    llama_context * ctx = make_ctx(params, model, gdn_replay);
    if (!ctx) {
        fprintf(stderr, "context init failed (gdn_replay=%d)\n", gdn_replay);
        return std::numeric_limits<double>::infinity();
    }
    if (llama_n_rs_seq(ctx) < N_RS) {
        fprintf(stderr, "n_rs_seq %u < %u\n", llama_n_rs_seq(ctx), N_RS);
        llama_free(ctx);
        return std::numeric_limits<double>::infinity();
    }

    std::vector<llama_token> history = prefix; // tokens the sequence really holds
    double worst = 0.0;
    bool ok = decode(ctx, prefix, 0, false);
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
        fprintf(stderr, "  gdn_replay=%d round %zu kept 1+%2d: nmse %.3e\n", gdn_replay, r, a, e);
        worst = std::max(worst, e);
        history.push_back(nxt);
        llama_free(ref);
    }
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
    for (const bool replay : { false, true }) {
        for (size_t i = 0; i < plans.size(); ++i) {
            const double e = run(params, model, replay, plans[i], 100 + (uint32_t) i, n_vocab);
            const bool pass = e <= tol;
            fprintf(stderr, "%s layout=%s plan %zu: worst nmse %.3e\n", pass ? "PASS" : "FAIL", replay ? "gdn_replay" : "ring", i, e);
            fails += pass ? 0 : 1;
        }
    }
    llama_model_free(model);
    fprintf(stderr, "test-dspine-rollback: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
