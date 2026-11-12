#include <algorithm>
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
    // default: no short rule, a tail needs a match of long_min tokens
    assert(cp.short_min == cp.long_min);
    {
        toks s = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 99, 12, 13, 14 };
        assert(common_speculative_copy_propose(cp, s.data(), (int32_t) s.size(), 15, { 16, 17, 18, 19 }, 4).empty());
        s = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 99, 11, 12, 13, 14, 15, 16, 17 }; // 8 with id_last
        assert((common_speculative_copy_propose(cp, s.data(), (int32_t) s.size(), 18, { 19 }, 2) == toks{ 20, 21 }));
    }
    cp.short_min = 4; // the rest exercises the short rule (LLAMA_SPEC_COPY_SHORT=4)
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

    // a wide round whose lookup proposed no tail: a block miss still leaves wide at once (not counted as a tail miss)
    for (int i = 0; i < 2; ++i) {
        cs.on_draft(7, 0);
        cs.on_accept(7);
    }
    assert(cs.wide());
    cs.on_draft(7, 0);
    assert(cs.on_accept(5) == 5 && !cs.wide() && cs.n_miss_blk == 1);
}

// tail credit: a tail accepted for less than half starts a cool-down that doubles per poor tail in a row
static void test_cooldown(void) {
    common_speculative_copy_state cs;
    cs.cool_base = 4;
    cs.reset();
    assert(cs.cool_base == 4); // configuration survives the per-request reset
    for (int i = 0; i < 2; ++i) {
        cs.on_draft(4, 0);
        cs.on_accept(4);
    }
    assert(cs.wide());

    // half of the tail accepted: good enough, no cool-down
    cs.on_draft(4, 4);
    assert(cs.on_accept(6) == 4 && cs.cool == 0 && cs.n_poor == 0);
    // the whole tail accepted
    cs.on_draft(4, 4);
    assert(cs.on_accept(8) == 4 && cs.cool == 0 && cs.wide());

    // one of four: poor, no tail for 4 rounds even though the blocks keep being accepted in full
    cs.on_draft(4, 4);
    assert(cs.on_accept(5) == 4 && cs.cool == 4 && cs.n_poor == 1 && !cs.wide());
    for (int i = 0; i < 4; ++i) {
        assert(!cs.wide());
        cs.on_draft(4, 0);
        cs.on_accept(4);
    }
    assert(cs.cool == 0 && cs.wide() && cs.n_cooled == 4);

    // a second poor tail in a row doubles it, a third doubles again
    cs.on_draft(4, 4);
    cs.on_accept(4);
    assert(cs.cool == 8 && cs.n_fail == 2);
    for (int i = 0; i < 8; ++i) {
        cs.on_draft(4, 0);
        cs.on_accept(4);
    }
    assert(cs.wide());
    cs.on_draft(4, 4);
    cs.on_accept(4);
    assert(cs.cool == 16);
    for (int i = 0; i < 16; ++i) {
        cs.on_draft(4, 0);
        cs.on_accept(4);
    }
    // a good tail clears the count: the next poor one starts at the base again
    assert(cs.wide());
    cs.on_draft(4, 4);
    cs.on_accept(8);
    assert(cs.n_fail == 0 && cs.cool == 0);
    if (!cs.wide()) { // regain the streak if the controller dropped
        for (int i = 0; i < 4 && !cs.wide(); ++i) {
            cs.on_draft(4, 0);
            cs.on_accept(4);
        }
    }
    assert(cs.wide());
    cs.on_draft(4, 4);
    cs.on_accept(5);
    assert(cs.cool == 4);

    // the cool-down is capped
    cs.reset();
    for (int k = 0; k < 12; ++k) {
        for (int i = 0; i < 200 && !cs.wide(); ++i) {
            cs.on_draft(4, 0);
            cs.on_accept(4);
        }
        assert(cs.wide());
        cs.on_draft(4, 4);
        cs.on_accept(4);
        assert(cs.cool <= common_speculative_copy_params::COOL_MAX);
    }
    assert(cs.cool == common_speculative_copy_params::COOL_MAX);

    // a block miss on a tail round is a poor tail too, and it also ends the streak
    cs.reset();
    for (int i = 0; i < 2; ++i) {
        cs.on_draft(4, 0);
        cs.on_accept(4);
    }
    cs.on_draft(4, 4);
    assert(cs.on_accept(2) == 2 && cs.cool == 4 && cs.n_miss_blk == 1 && !cs.wide());

    // cool_base 0 (the state's default): no cool-down at all
    common_speculative_copy_state c0;
    c0.reset();
    for (int i = 0; i < 2; ++i) {
        c0.on_draft(4, 0);
        c0.on_accept(4);
    }
    c0.on_draft(4, 4);
    c0.on_accept(4);
    assert(c0.cool == 0 && c0.n_poor == 1);
}

// the server's rollback rule (server-context.cpp): with the tail capped at n_rs_seq, a wide round whose block was
// accepted in full never rejects more than n_rs_seq tokens, so it rolls back in place; only a block miss is deeper
static void test_rollback_depth(void) {
    for (int n_rs = 1; n_rs <= 8; ++n_rs) {
        for (int block = 1; block <= n_rs; ++block) {
            for (int tail = 1; tail <= n_rs; ++tail) {
                const int n_draft = block + tail;
                for (int n_acc = 0; n_acc <= n_draft; ++n_acc) {
                    const int n_rollback = n_draft - n_acc; // = spec_draft.size() + 1 - accepted.size()
                    const bool in_place = n_rollback <= n_rs;
                    if (n_acc >= block) {
                        assert(in_place);
                    }
                    if (!in_place) {
                        assert(n_acc < block);
                    }
                }
            }
        }
    }
}

// the verify-row cap: block + tail + 1 never passes n_rows, whatever the block, the tail ask and the tail cap
static void test_rows_cap(void) {
    assert(common_speculative_copy_total_cap(12, 0) == 12);
    assert(common_speculative_copy_total_cap(12, 8) == 7);
    assert(common_speculative_copy_total_cap(4, 8) == 4);
    for (int rows = 1; rows <= 16; ++rows) {
        for (int block = 1; block <= 8; ++block) {
            for (int tail = 0; tail <= 15; ++tail) {
                for (int tail_cap = 0; tail_cap <= 8; ++tail_cap) {
                    const int total = common_speculative_copy_total_cap(block + tail, rows);
                    // speculative.cpp: room = min(tail, n_max_total - n_block, n_tail_max)
                    const int room = std::max(0, std::min(std::min(tail, total - block), tail_cap));
                    if (block + 1 <= rows) {
                        assert(block + room + 1 <= rows);
                    } else {
                        assert(room == 0);
                    }
                }
            }
        }
    }
    // ship shape: block 4, tail ask 8, tail cap n_rs_seq 4, 8 rows -> tail 3
    assert(std::min(std::min(8, common_speculative_copy_total_cap(12, 8) - 4), 4) == 3);
}

// keep-base rule (server-context.cpp + llama-memory-recurrent.cpp): a wide round of n = draft + 1 rows keeps groups
// 0..K-2 and the pre-round plane, so a reject is rolled back in place when <= n_rs - 1 deep, else the whole batch goes
static void test_keep_base_rule(void) {
    for (int n_rs = 2; n_rs <= 8; ++n_rs) {
        const int K = n_rs + 1;
        for (int block = 1; block <= n_rs; ++block) {
            for (int tail = 1; tail <= n_rs - 1; ++tail) {
                const int n_draft = block + tail;
                if (n_draft <= n_rs || n_draft + 1 > 2 * K) {
                    continue; // narrow round, or wider than the keep-base window
                }
                for (int n_acc = 0; n_acc <= n_draft; ++n_acc) {
                    const int n_rollback = n_draft - n_acc;
                    const bool in_place = n_rollback + 1 <= n_rs;
                    if (n_acc >= block) {
                        assert(in_place);          // a tail miss never needs the base plane
                    }
                    if (!in_place) {
                        assert(n_acc < block);     // only a block miss does
                        assert(n_acc + 1 <= n_rs); // and its replay (sampled + accepted) is a narrow batch
                    }
                }
            }
        }
    }
}

int main(void) {
    test_keep_base_rule();
    test_rows_cap();
    test_find_matches_reference();
    test_gate();
    test_point_mass_rows();
    test_trigger();
    test_cooldown();
    test_rollback_depth();
    printf("test-speculative-copy: all tests OK\n");
    return 0;
}
