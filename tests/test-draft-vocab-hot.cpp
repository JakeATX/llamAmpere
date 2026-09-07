// Unit test for the adaptive tail of the MTP draft vocabulary shortlist (llama_mtp_hot_vocab).
// CPU only: the policy is header-only and needs neither a model nor a backend.

#include "../src/llama-mtp-vocab.h"
#include "ggml.h"

#include <cstdio>
#include <random>

static llama_mtp_hot_vocab make(int64_t n_vocab, int32_t n_static, int32_t n_hot) {
    std::vector<int32_t> ids;
    for (int32_t i = 0; i < n_static + n_hot; ++i) {
        ids.push_back(i);
    }
    llama_mtp_hot_vocab hv;
    hv.init(ids, n_vocab, n_hot);
    return hv;
}

static void observe(llama_mtp_hot_vocab & hv, const std::vector<int32_t> & toks) {
    hv.observe(toks.data(), toks.size());
}

// (a) no duplicates anywhere and (b) a static id never occupies a hot slot
static void check_invariants(const llama_mtp_hot_vocab & hv) {
    std::vector<uint8_t> resident(hv.n_vocab, 0);
    for (int32_t i = 0; i < hv.n_hot; ++i) {
        const int32_t id = hv.slots[i];
        GGML_ASSERT(id >= 0 && id < hv.n_vocab);
        GGML_ASSERT(!hv.static_mask[id]);
        GGML_ASSERT(!resident[id]);
        GGML_ASSERT(hv.slot_of[id] == i);
        resident[id] = 1;
    }
    for (int64_t id = 0; id < hv.n_vocab; ++id) {
        GGML_ASSERT((hv.slot_of[id] >= 0) == (resident[id] != 0));
    }
}

int main() {
    // ids 0..11 are static, 12..15 are the four hot slots
    {
        llama_mtp_hot_vocab hv = make(64, 12, 4);
        GGML_ASSERT(hv.slots == std::vector<int32_t>({12, 13, 14, 15}));
        check_invariants(hv);

        // (b) static ids are never counted, so nothing can displace a hot slot
        observe(hv, {0, 1, 2, 0});
        GGML_ASSERT(hv.cands.empty());
        GGML_ASSERT(hv.select().empty());
        check_invariants(hv);

        // the file-provided hot ids have score 0 and go first, in slot order
        observe(hv, {20, 20, 20, 20, 21, 21, 21, 22, 22, 23});
        auto rep = hv.select();
        GGML_ASSERT(rep.size() == 4);
        GGML_ASSERT(hv.slots == std::vector<int32_t>({20, 21, 22, 23}));
        check_invariants(hv);

        // (d) an immediate second select() changes nothing: resident ids win ties
        GGML_ASSERT(hv.select().empty());
        check_invariants(hv);

        // (e) with every slot taken, the lowest-scoring resident (23) is the one evicted
        observe(hv, {24, 24, 24, 24, 24});
        rep = hv.select();
        GGML_ASSERT(rep.size() == 1 && rep[0].slot == 3 && rep[0].id == 24);
        GGML_ASSERT(hv.slots == std::vector<int32_t>({20, 21, 22, 24}));
        GGML_ASSERT(hv.slot_of[23] == -1);
        check_invariants(hv);
    }

    // (c) recurrence beats a single huge document once the one-shot burst has decayed
    {
        llama_mtp_hot_vocab hv = make(64, 12, 4);
        const int32_t burst = 30; // seen 5000 times in one request
        const int32_t recur = 31; // seen once in each of three later requests

        observe(hv, std::vector<int32_t>(5000, burst));
        GGML_ASSERT(hv.score_of(burst) <= hv.count_cap + 1e-4f); // the per-request cap holds

        const std::vector<int32_t> none;
        for (int i = 0; i < 12; ++i) {
            observe(hv, none);
        }
        for (int i = 0; i < 3; ++i) {
            observe(hv, {recur});
        }
        GGML_ASSERT(hv.score_of(recur) > hv.score_of(burst));

        auto rep = hv.select();
        GGML_ASSERT(hv.slot_of[recur] >= 0);
        GGML_ASSERT(rep.size() == 2); // both ids are resident now, the other two slots stay
        check_invariants(hv);
    }

    // (f) n_hot == 0 is a no-op: the map stays a plain static shortlist
    {
        llama_mtp_hot_vocab hv = make(64, 16, 0);
        observe(hv, {20, 20, 21});
        GGML_ASSERT(hv.cands.empty() && hv.slots.empty() && hv.select().empty());
        GGML_ASSERT(hv.req_counter == 0);
        for (int32_t i = 0; i < 16; ++i) {
            GGML_ASSERT(hv.static_mask[i]);
        }
    }

    // rejected configurations
    {
        llama_mtp_hot_vocab hv;
        bool failed = false;
        try { hv.init({1, 2, 3}, 64, 3); } catch (const std::runtime_error &) { failed = true; }
        GGML_ASSERT(failed);
        failed = false;
        try { hv.init({1, 2, 3}, 64, -1); } catch (const std::runtime_error &) { failed = true; }
        GGML_ASSERT(failed);
    }

    // soak: random traffic must keep the invariants and the candidate table bounded
    {
        const int64_t n_vocab = 4096;
        llama_mtp_hot_vocab hv = make(n_vocab, 512, 64);
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int32_t> tok(0, (int32_t) n_vocab - 1);
        size_t n_replaced = 0;
        for (int req = 0; req < 400; ++req) {
            std::vector<int32_t> toks;
            for (int i = 0, n = 1 + (int) (rng() % 512); i < n; ++i) {
                toks.push_back(tok(rng));
            }
            observe(hv, toks);
            n_replaced += hv.select().size();
            check_invariants(hv);
            GGML_ASSERT((int64_t) hv.cands.size() <= n_vocab);
        }
        GGML_ASSERT(n_replaced > 0);
        printf("soak: %zu slot replacements over 400 requests, %zu candidates\n", n_replaced, hv.cands.size());
    }

    // the candidate table is bounded: entries that decayed away are dropped, residents are kept
    {
        llama_mtp_hot_vocab hv = make(4096, 512, 8);
        std::vector<int32_t> toks;
        for (int32_t i = 0; i < 200; ++i) {
            toks.push_back(1000 + i);
        }
        observe(hv, toks);
        GGML_ASSERT(hv.cands.size() == 200);
        GGML_ASSERT(hv.select().size() == 8);
        const std::vector<int32_t> none;
        for (int i = 0; i < 45; ++i) {
            observe(hv, none);
        }
        GGML_ASSERT(hv.cands.size() == 8); // only the resident ids survive
        check_invariants(hv);
    }

    printf("%s: OK\n", __func__);
    return 0;
}
