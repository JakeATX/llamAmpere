#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

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

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
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

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
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

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = logit_diff(l_roll[t], l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

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
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

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

    constexpr float eps = 1e-5f;
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
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

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

        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
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
