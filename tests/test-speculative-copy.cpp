// asserts must stay live in Release builds: undefine NDEBUG before <cassert>
#undef NDEBUG

#include "speculative-copy.h"

#include <cassert>
#include <cstdio>
#include <random>

typedef std::vector<llama_token> toks;

// reference: every earlier occurrence, longest suffix match by direct comparison of the full history
static common_speculative_copy_match ref_find(const common_speculative_copy_params & cp, toks h, llama_token id_last) {
    h.push_back(id_last);
    const int n = (int) h.size();
    common_speculative_copy_match best;
    for (int e = n - 1; e >= 1; --e) { // e = one past the end of the earlier occurrence
        int len = 0;
        while (len < cp.long_min && e - 1 - len >= 0 && h[e - 1 - len] == h[n - 1 - len]) {
            len++;
        }
        if (len > best.len) {
            best.next = e;
            best.len  = len;
        }
        if (best.len >= cp.long_min) {
            break;
        }
    }
    return best.len >= cp.short_min ? best : common_speculative_copy_match();
}

static void test_find_matches_reference(void) {
    common_speculative_copy_params cp;
    std::mt19937 rng(1234);
    for (int it = 0; it < 2000; ++it) {
        // a small alphabet gives many short matches, overlaps and repeats
        const int n = 2 + (int) (rng() % 120);
        toks h(n);
        for (auto & t : h) {
            t = (llama_token) (rng() % (it % 2 ? 3 : 6));
        }
        if (it % 5 == 0 && n > 40) { // planted long copy
            std::copy(h.begin() + 3, h.begin() + 18, h.end() - 15);
        }
        const llama_token id_last = h.back();
        h.pop_back();
        const auto a = common_speculative_copy_find(cp, h.data(), (int32_t) h.size(), id_last);
        const auto b = ref_find(cp, h, id_last);
        assert(a.next == b.next && a.len == b.len);
    }
}

static void test_gate(void) {
    common_speculative_copy_params cp;
    //            0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    toks src = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };

    // long match (8 tokens 10..17 repeated): block 18,19 agrees -> tail continues the copy
    toks h = src;
    h.insert(h.end(), { 99, 10, 11, 12, 13, 14, 15, 16 });
    toks tail = common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 17, { 18, 19 }, 4);
    assert((tail == toks{ 20, 21, 22, 23 }));
    // a long match needs no agreement count: a one-token block is enough
    tail = common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 17, { 18 }, 3);
    assert((tail == toks{ 19, 20, 21 }));
    // a block that departs from the copy drops the tail, long match or not
    assert(common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 17, { 18, 77 }, 4).empty());
    // no model block, no tail
    assert(common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 17, {}, 4).empty());

    // short match (4 tokens): needs a block of >= agree tokens that repeats the copy
    h = src;
    h.insert(h.end(), { 99, 12, 13, 14 });
    assert(common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 15, { 16 }, 4).empty());
    tail = common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 15, { 16, 17 }, 4);
    assert((tail == toks{ 18, 19, 20, 21 }));
    // below short_min: nothing
    h = src;
    h.insert(h.end(), { 99, 98, 13, 14 });
    assert(common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 15, { 16, 17 }, 4).empty());

    // a match that overlaps its own suffix continues the period through the block and the tail
    h = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2 };
    tail = common_speculative_copy_propose(cp, h.data(), (int32_t) h.size(), 3, { 1, 2 }, 5);
    assert((tail == toks{ 3, 1, 2, 3, 1 }));
}

static void test_point_mass_rows(void) {
    // sampled block with candidate rows: tail rows are empty = id match in the p/q verify
    toks result = { 5, 6 };
    std::vector<std::vector<llama_token_data>> q = { { { 5, 0.0f, 0.9f } }, { { 6, 0.0f, 0.8f } } };
    common_speculative_copy_append(result, &q, { 7, 8, 9 });
    assert(result.size() == 5 && q.size() == 5);
    assert(q[0].size() == 1 && q[1].size() == 1 && q[2].empty() && q[3].empty() && q[4].empty());

    // id-match block (no rows): stays without rows
    result = { 5, 6 };
    q.clear();
    common_speculative_copy_append(result, &q, { 7 });
    assert(result.size() == 3 && q.empty());
    common_speculative_copy_append(result, nullptr, { 8 });
    assert(result.size() == 4);
}

static void test_trigger(void) {
    common_speculative_copy_state cs;
    cs.reset();
    assert(!cs.wide());

    // one full round is not enough, a miss restarts the streak
    cs.on_draft(7, 0);
    assert(cs.on_accept(7) == 7 && !cs.wide());
    cs.on_draft(7, 0);
    assert(cs.on_accept(3) == 3 && !cs.wide());
    cs.on_draft(7, 0);
    cs.on_accept(7);
    cs.on_draft(5, 0);
    cs.on_accept(5);
    assert(cs.wide());

    // wide round, all accepted: the drafter is credited with its block only
    cs.on_draft(7, 8);
    assert(cs.on_accept(15) == 7 && cs.wide() && cs.n_acc == 8);
    // the copy ends inside the tail: wide survives one such round
    cs.on_draft(7, 8);
    assert(cs.on_accept(9) == 7 && cs.wide() && cs.n_acc == 10);
    // repeated tail misses leave wide through the controller's drop pressure
    for (int i = 0; i < 3 && cs.wide(); ++i) {
        cs.on_draft(7, 8);
        cs.on_accept(7);
    }
    assert(!cs.wide());

    // a miss inside the block of a wide round leaves wide at once
    cs.reset();
    for (int i = 0; i < 2; ++i) {
        cs.on_draft(7, 0);
        cs.on_accept(7);
    }
    assert(cs.wide());
    cs.on_draft(7, 8);
    assert(cs.on_accept(4) == 4 && !cs.wide() && cs.n_miss_blk == 1);
}

int main(void) {
    test_find_matches_reference();
    test_gate();
    test_point_mass_rows();
    test_trigger();
    printf("test-speculative-copy: all tests OK\n");
    return 0;
}
