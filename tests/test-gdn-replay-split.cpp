// [TAG_GDN_REPLAY_SPLIT] gdn_replay builds one replay subtree per ubatch, so the ubatch split must never put
// sequences with different (replay_len, ckpt_span, s_stale) into one ubatch. No model: this drives
// llama_batch_allocr::split_equal with llama_memory_recurrent::replay_split the way init_batch does and checks
//   - lanes that agree still share a ubatch (no throughput lost when nothing differs),
//   - lanes that disagree are split, including after earlier ubatches of the same batch changed them,
//   - every token is used and each seq's trailing n_keep_tail tokens stay in one ubatch,
//   - with the constraint off (fn() == nullptr) the split is the old one.
#include "../src/llama-batch.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-vocab.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

struct lane {
    uint32_t n_tokens;
    uint32_t replay_len;
    uint32_t ckpt_span;
    uint8_t  s_stale;
};

struct split_result {
    std::vector<std::vector<llama_seq_id>> seqs; // per ubatch, its seqs
    uint32_t n_used   = 0;
    uint32_t n_tokens = 0;
};

// runs the init_batch loop; fails the process on any broken invariant
static split_result run_split(const std::vector<lane> & lanes, uint32_t n_ubatch, uint32_t n_rs_seq, bool constrain) {
    const uint32_t n_seq = lanes.size();

    // token order: all of seq 0, then seq 1, ... (the server fills batches slot by slot)
    std::vector<float>          embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_ids;
    std::vector<int8_t>         logits;
    for (uint32_t s = 0; s < n_seq; ++s) {
        for (uint32_t t = 0; t < lanes[s].n_tokens; ++t) {
            embd.push_back(0.0f);
            pos.push_back(100 + t);
            n_seq_id.push_back(1);
            seq_ids.push_back(s);
            logits.push_back(1);
        }
    }
    const int32_t n_tokens = (int32_t) pos.size();
    std::vector<llama_seq_id *> seq_id_ptrs(n_tokens + 1, nullptr);
    for (int32_t i = 0; i < n_tokens; ++i) {
        seq_id_ptrs[i] = &seq_ids[i];
    }

    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd     = embd.data();
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id_ptrs.data();
    batch.logits   = logits.data();

    llama_vocab vocab;
    llama_batch_allocr balloc(1);
    GGML_ASSERT(balloc.init(batch, vocab, nullptr, 1, n_seq, false));

    llama_memory_recurrent::replay_split rs;
    rs.active   = constrain;
    rs.n_rs_seq = n_rs_seq;
    for (const lane & l : lanes) {
        rs.replay_len.push_back(l.replay_len);
        rs.ckpt_span.push_back(l.ckpt_span);
        rs.s_stale.push_back(l.s_stale);
    }
    if (!constrain) {
        GGML_ASSERT(rs.fn() == nullptr);
    }

    const uint32_t n_keep_tail = n_rs_seq > 0 ? n_rs_seq + 1 : 0;

    split_result res;
    res.n_tokens = n_tokens;
    std::map<llama_seq_id, uint32_t> seen;      // tokens of each seq emitted so far
    std::map<llama_seq_id, size_t>   last_ub;   // ubatch holding each seq's last token
    std::map<std::pair<size_t, llama_seq_id>, uint32_t> per_ub;

    balloc.split_reset();
    while (true) {
        llama_ubatch ub = balloc.split_equal(n_ubatch, true, n_keep_tail, rs.fn());
        if (ub.n_tokens == 0) {
            break;
        }
        GGML_ASSERT(ub.n_tokens <= n_ubatch);

        std::vector<llama_seq_id> ids(ub.seq_id_unq, ub.seq_id_unq + ub.n_seqs_unq);
        if (constrain) {
            // the values the graph would see for this ubatch agree across its lanes
            for (llama_seq_id s : ids) {
                GGML_ASSERT(rs.compat(ids[0], s));
            }
        }
        const size_t k = res.seqs.size();
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_seq_id s = ub.seq_id[i][0];
            seen[s]++;
            per_ub[std::make_pair(k, s)]++;
            if (seen[s] == lanes[s].n_tokens) {
                last_ub[s] = k;
            }
        }
        rs.advance(ub);
        res.seqs.push_back(ids);
    }
    res.n_used = balloc.get_n_used();

    GGML_ASSERT(res.n_used == res.n_tokens);
    for (uint32_t s = 0; s < n_seq; ++s) {
        GGML_ASSERT(seen[s] == lanes[s].n_tokens);
        if (n_keep_tail > 0) {
            // the trailing n_keep_tail tokens (or the whole seq) share the last ubatch
            const uint32_t n_tail = per_ub[std::make_pair(last_ub[s], (llama_seq_id) s)];
            GGML_ASSERT(n_tail >= std::min(lanes[s].n_tokens, n_keep_tail));
        }
    }
    if (constrain) {
        // every lane left the batch consumed
        for (uint32_t s = 0; s < n_seq; ++s) {
            GGML_ASSERT(rs.replay_len[s] == 0 && rs.s_stale[s] == 0);
        }
    }
    return res;
}

int main() {
    const uint32_t n_rs_seq = 3; // MTP depth 3-4 verify: K = 4 snapshot groups

    // two verify lanes of 4 tokens that agree: one ubatch with both
    {
        auto r = run_split({ { 4, 2, 3, 0 }, { 4, 2, 3, 0 } }, 512, n_rs_seq, true);
        GGML_ASSERT(r.seqs.size() == 1 && r.seqs[0].size() == 2);
        printf("agree: 1 ubatch, 2 seqs\n");
    }
    // different accept lengths (replay_len 1 vs 2): one ubatch each
    {
        auto r = run_split({ { 4, 1, 3, 0 }, { 4, 2, 3, 0 } }, 512, n_rs_seq, true);
        GGML_ASSERT(r.seqs.size() == 2 && r.seqs[0].size() == 1 && r.seqs[1].size() == 1);
        printf("replay_len differs: 2 ubatches\n");
    }
    // same replay, different checkpoint span; and stale only
    {
        auto r = run_split({ { 4, 1, 2, 0 }, { 4, 1, 3, 0 } }, 512, n_rs_seq, true);
        GGML_ASSERT(r.seqs.size() == 2);
        r = run_split({ { 4, 0, 3, 1 }, { 4, 0, 3, 0 } }, 512, n_rs_seq, true);
        GGML_ASSERT(r.seqs.size() == 2);
        printf("span / stale differ: 2 ubatches each\n");
    }
    // constraint off: the old split (both lanes in one ubatch whatever they hold)
    {
        auto r = run_split({ { 4, 1, 3, 0 }, { 4, 2, 3, 0 } }, 512, n_rs_seq, false);
        GGML_ASSERT(r.seqs.size() == 1 && r.seqs[0].size() == 2);
        printf("constraint off: 1 ubatch\n");
    }
    // A, B, A: the sequential split cannot skip seq 1, and seq 2 must not join seq 0
    {
        auto r = run_split({ { 4, 1, 3, 0 }, { 4, 2, 3, 0 }, { 4, 1, 3, 0 } }, 512, n_rs_seq, true);
        GGML_ASSERT(r.seqs.size() == 3);
        printf("A,B,A: 3 ubatches\n");
    }
    // lanes that agree at the start and drift apart inside the batch: seq 0 runs a ubatch alone (seq 1 is
    // deferred by the tail rule), after which the two no longer agree
    {
        auto r = run_split({ { 20, 1, 3, 0 }, { 5, 1, 3, 0 } }, 8, n_rs_seq, true);
        printf("drift: %zu ubatches:", r.seqs.size());
        for (const auto & u : r.seqs) {
            printf(" {");
            for (llama_seq_id s : u) {
                printf(" %d", s);
            }
            printf(" }");
        }
        printf("\n");
    }

    // random batches: the invariants are checked inside run_split
    std::mt19937 rng(1234);
    int n_split_cases = 0;
    for (int it = 0; it < 20000; ++it) {
        const uint32_t n_seq = 1 + rng() % 4;
        std::vector<lane> lanes(n_seq);
        for (auto & l : lanes) {
            l.n_tokens   = 1 + rng() % 24;
            l.ckpt_span  = rng() % (n_rs_seq + 1);
            l.replay_len = rng() % (l.ckpt_span + 1);
            l.s_stale    = (rng() % 8) == 0;
        }
        const uint32_t n_ubatch = n_rs_seq + 2 + rng() % 40;
        auto r = run_split(lanes, n_ubatch, n_rs_seq, true);
        auto r0 = run_split(lanes, n_ubatch, n_rs_seq, false);
        n_split_cases += r.seqs.size() > r0.seqs.size();
    }
    printf("random: 20000 batches passed, %d needed extra ubatches\n", n_split_cases);
    printf("OK\n");
    return 0;
}
