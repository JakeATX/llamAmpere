// llama_kv_cells position orders and the cells that llama_kv_cache::seq_rm scans (fork).
//
// llama_kv_cells tracks two orders of the positions of its used cells: identity (used cell i holds position i) and
// sorted (the positions never decrease along the used cells). pos_range(p0, p1) uses them to return the cells that
// can hold a position of [p0, p1), and seq_rm scans only those. This test applies random operations to a small set of
// cells, the ones llama_kv_cache uses (fill, remove a sequence by position, clear a cell, shift, divide, save and
// restore with cp/set, reset), and after each one checks:
//   1. pos_is_identity() and pos_is_sorted() are true only when the order holds
//   2. for random [p0, p1), every cell that holds a position of [p0, p1) is inside pos_range(p0, p1), and removing
//      a sequence by scanning only pos_range removes the same cells as scanning all cells
// Fixed cases check that the short scans are taken: text filled in order with suffixes removed keeps identity,
// image tokens that share a position keep sorted, two sequences in one stream fall back to the full scan, and
// clearing every cell restores both orders.

#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static const llama_pos POS_MAX = std::numeric_limits<llama_pos>::max();

// the orders, computed from the cells
static void check_orders(const llama_kv_cells & cells, const std::string & where) {
    bool identity = true;
    bool sorted   = true;
    llama_pos p_prev = -1;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        const llama_pos p = cells.pos_get(i);
        identity = identity && p == (llama_pos) i;
        sorted   = sorted   && p >= p_prev;
        p_prev = p;
    }
    require(!cells.pos_is_identity() || identity, where + ": pos_is_identity() but a used cell i does not hold position i");
    require(!cells.pos_is_sorted()   || sorted,   where + ": pos_is_sorted() but the positions decrease");
}

// every cell holding a position of [p0, p1) is inside pos_range(p0, p1)
static void check_range(const llama_kv_cells & cells, llama_pos p0, llama_pos p1, const std::string & where) {
    uint32_t i_beg = 0;
    uint32_t i_end = 0;
    cells.pos_range(p0, p1, i_beg, i_end);
    require(i_end <= cells.size(), where + ": pos_range past the last cell");
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.pos_in(i, p0, p1)) {
            require(i_beg <= i && i < i_end, where + ": cell " + std::to_string(i) + " (position " +
                    std::to_string(cells.pos_get(i)) + ") outside pos_range(" + std::to_string(p0) + ", " +
                    std::to_string(p1) + ") = [" + std::to_string(i_beg) + ", " + std::to_string(i_end) + ")");
        }
    }
}

// remove sequence seq_id from the positions [p0, p1) as llama_kv_cache::seq_rm does, scanning only pos_range, and
// check that the cells removed are the ones a scan of all cells removes. Returns the number of cells scanned.
static uint32_t seq_rm(llama_kv_cells & cells, llama_seq_id seq_id, llama_pos p0, llama_pos p1, const std::string & where) {
    std::vector<uint32_t> want;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.pos_in(i, p0, p1) && (seq_id < 0 || cells.seq_has(i, seq_id))) {
            want.push_back(i);
        }
    }
    uint32_t i_beg = 0;
    uint32_t i_end = 0;
    cells.pos_range(p0, p1, i_beg, i_end);
    std::vector<uint32_t> got;
    for (uint32_t i = i_beg; i < i_end; ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }
        if (seq_id < 0) {
            cells.rm(i);
            got.push_back(i);
        } else if (cells.seq_has(i, seq_id)) {
            cells.seq_rm(i, seq_id);
            got.push_back(i);
        }
    }
    require(got == want, where + ": seq_rm over pos_range removed " + std::to_string(got.size()) + " cells, a full scan " +
            std::to_string(want.size()));
    return i_end > i_beg ? i_end - i_beg : 0;
}

// fill the cells after the last used one with positions p, p + step, ... for sequence seq_id
static void append(llama_kv_cells & cells, llama_seq_id seq_id, llama_pos p, uint32_t n, llama_pos step) {
    const uint32_t i0 = cells.used_max_p1();
    for (uint32_t k = 0; k < n && i0 + k < cells.size(); ++k) {
        cells.pos_set(i0 + k, p + (llama_pos) k*step);
        cells.seq_add(i0 + k, seq_id);
    }
}

static llama_pos next_pos(const llama_kv_cells & cells) {
    const uint32_t n = cells.used_max_p1();
    return n == 0 ? 0 : cells.pos_get(n - 1) + 1;
}

static void fixed_cases() {
    llama_kv_cells cells;
    cells.resize(1024);

    // text: 300 tokens, a suffix of 3 removed and 3 tokens appended, as one round of speculative decoding does
    append(cells, 0, 0, 300, 1);
    require(cells.pos_is_identity(), "text: identity after filling in order");
    uint32_t n = seq_rm(cells, 0, 297, POS_MAX, "text");
    require(n == 3, "text: suffix of 3 scanned " + std::to_string(n) + " cells");
    append(cells, 0, next_pos(cells), 3, 1);
    require(cells.pos_is_identity(), "text: identity after a suffix removal and an append");
    check_orders(cells, "text");

    // image: text, 48 image tokens at one position, text that continues at a larger position
    cells.reset();
    append(cells, 0, 0, 100, 1);
    append(cells, 0, 100, 48, 0);
    append(cells, 0, 108, 50, 1);
    require(!cells.pos_is_identity() && cells.pos_is_sorted(), "image: sorted, not identity");
    uint32_t i_beg = 0;
    uint32_t i_end = 0;
    cells.pos_range(100, 108, i_beg, i_end);
    require(i_beg == 100 && i_end == 148, "image: pos_range(100, 108) = [" + std::to_string(i_beg) + ", " + std::to_string(i_end) + ")");
    n = seq_rm(cells, 0, 150, POS_MAX, "image");
    require(n == 8, "image: suffix of 8 scanned " + std::to_string(n) + " cells");
    n = seq_rm(cells, 0, 100, POS_MAX, "image");
    require(n == 90, "image: cut at the image start scanned " + std::to_string(n) + " cells");
    require(cells.pos_is_sorted(), "image: still sorted after the cuts");
    check_orders(cells, "image");

    // two sequences in one stream (--kv-unified): the second one starts again at position 0, so the full scan is used
    cells.reset();
    append(cells, 0, 0, 200, 1);
    append(cells, 1, 0, 100, 1);
    require(!cells.pos_is_identity() && !cells.pos_is_sorted(), "two sequences: no order");
    n = seq_rm(cells, 1, 97, POS_MAX, "two sequences");
    require(n == 300, "two sequences: scanned " + std::to_string(n) + " cells, want the 300 used ones");
    seq_rm(cells, 0, 0, POS_MAX, "two sequences");
    seq_rm(cells, 1, 0, POS_MAX, "two sequences");
    require(cells.get_used() == 0 && cells.pos_is_identity() && cells.pos_is_sorted(), "two sequences: both orders after clearing every cell");

    // a hole (a cell removed below the last used one): sorted alone no longer gives the range, identity still does
    cells.reset();
    append(cells, 0, 0, 100, 1);
    cells.rm(40);
    n = seq_rm(cells, 0, 90, POS_MAX, "hole");
    require(n == 10, "hole: identity scanned " + std::to_string(n) + " cells");

    // K-shift: the positions move, both orders are dropped
    cells.reset();
    append(cells, 0, 0, 100, 1);
    for (uint32_t i = 50; i < 100; ++i) {
        cells.pos_add(i, -10);
    }
    require(!cells.pos_is_identity() && !cells.pos_is_sorted(), "K-shift: no order");
    check_range(cells, 40, 60, "K-shift");

    printf("fixed cases OK\n");
}

static void random_ops(uint32_t seed, int n_ops) {
    std::mt19937 rng(seed);
    auto rnd = [&](uint32_t n) { return (uint32_t) (rng() % n); };

    llama_kv_cells cells;
    cells.resize(96);
    int n_identity = 0;
    int n_sorted   = 0;

    for (int op = 0; op < n_ops; ++op) {
        const std::string where = "seed " + std::to_string(seed) + " op " + std::to_string(op);
        const uint32_t k = rnd(100);
        const llama_seq_id s = (llama_seq_id) rnd(3);
        if (k < 30) {
            // text in order
            append(cells, s, next_pos(cells), 1 + rnd(6), 1);
        } else if (k < 38) {
            // image tokens: one position, then the text continues at a larger position
            const llama_pos p = next_pos(cells);
            const uint32_t w = 1 + rnd(8);
            append(cells, s, p, w, 0);
            append(cells, s, p + (llama_pos) w, 1 + rnd(3), 1);
        } else if (k < 42) {
            // another sequence starts at position 0 in the same cells
            append(cells, s, 0, 1 + rnd(4), 1);
        } else if (k < 46) {
            // an empty cell anywhere gets any position (holes refilled, SWA reuse)
            const uint32_t i = rnd(cells.size());
            if (cells.is_empty(i)) {
                cells.pos_set(i, (llama_pos) rnd(120));
                cells.seq_add(i, s);
            }
        } else if (k < 50) {
            // a used cell joins another sequence (seq_cp)
            const uint32_t i = rnd(cells.size());
            if (!cells.is_empty(i) && !cells.seq_has(i, s)) {
                cells.seq_add(i, s);
            }
        } else if (k < 68) {
            // remove a sequence from a position on, the speculative rollback
            seq_rm(cells, s, (llama_pos) rnd(130), POS_MAX, where);
        } else if (k < 74) {
            // remove a range of positions, of one sequence or of all of them
            const llama_pos p0 = (llama_pos) rnd(130);
            seq_rm(cells, rnd(2) ? s : -1, p0, p0 + (llama_pos) rnd(40), where);
        } else if (k < 78) {
            // clear one used cell, leaving a hole
            const uint32_t i = rnd(cells.size());
            if (!cells.is_empty(i)) {
                cells.rm(i);
            }
        } else if (k < 81) {
            // K-shift of the cells from a position on
            const llama_pos p0 = (llama_pos) rnd(120);
            const llama_pos d  = (llama_pos) rnd(21) - 10;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.pos_get(i) >= p0) {
                    cells.pos_add(i, d);
                }
            }
            cells.reset_shift(); // the cache applies the shift to K before the next ubatch
        } else if (k < 82) {
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.pos_get(i) >= 60) {
                    cells.pos_div(i, 2);
                }
            }
            cells.reset_shift();
        } else if (k < 89) {
            // a ubatch is placed and rolled back: save the cells with cp, change them, restore them with set
            // (cp/set of a range of cells, or of a list of cells as find_slot returns for a non-contiguous ubatch)
            std::vector<uint32_t> idxs;
            const bool contiguous = rnd(2) == 0;
            const uint32_t n  = 1 + rnd(8);
            const uint32_t i0 = rnd(cells.size() - n + 1);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t i = contiguous ? i0 + j : rnd(cells.size());
                if (std::find(idxs.begin(), idxs.end(), i) == idxs.end()) {
                    idxs.push_back(i);
                }
            }
            std::vector<llama_pos> before;
            for (uint32_t i : idxs) {
                before.push_back(cells.is_empty(i) ? -1 : cells.pos_get(i));
            }
            const llama_kv_cells bak = contiguous ? cells.cp(i0, n) : cells.cp(idxs);
            for (uint32_t i : idxs) {
                if (!cells.is_empty(i)) {
                    cells.rm(i);
                }
                if (rnd(2)) {
                    cells.pos_set(i, (llama_pos) rnd(120));
                    cells.seq_add(i, s);
                }
            }
            check_orders(cells, where + " (placed)");
            if (contiguous) {
                cells.set(i0, bak);
            } else {
                cells.set(idxs, bak);
            }
            for (size_t j = 0; j < idxs.size(); ++j) {
                const llama_pos p = cells.is_empty(idxs[j]) ? -1 : cells.pos_get(idxs[j]);
                require(p == before[j], where + ": set did not restore cell " + std::to_string(idxs[j]));
            }
        } else if (k < 94) {
            // set writes the contents of other cells (cp of a range or of a list of cells) into a range or a list
            const uint32_t n  = 1 + rnd(8);
            const uint32_t i0 = rnd(cells.size() - n + 1);
            const uint32_t j0 = rnd(cells.size() - n + 1);
            std::vector<uint32_t> idxs;
            std::vector<uint32_t> jdxs;
            for (uint32_t j = 0; j < n; ++j) {
                idxs.push_back(i0 + j);
                jdxs.push_back(rnd(2) ? j0 + j : rnd(cells.size()));
            }
            const llama_kv_cells src = rnd(2) ? cells.cp(j0, n) : cells.cp(jdxs);
            if (rnd(2)) {
                cells.set(i0, src);
            } else {
                std::shuffle(idxs.begin(), idxs.end(), rng);
                cells.set(idxs, src);
            }
        } else if (k < 96) {
            cells.reset();
        } else {
            // clear every cell, as a new conversation does
            seq_rm(cells, -1, 0, POS_MAX, where);
            require(cells.get_used() == 0 && cells.pos_is_identity() && cells.pos_is_sorted(), where + ": both orders after clearing every cell");
        }

        check_orders(cells, where);
        for (int r = 0; r < 4; ++r) {
            const llama_pos p0 = (llama_pos) rnd(130);
            check_range(cells, p0, rnd(4) ? p0 + (llama_pos) rnd(40) : POS_MAX, where);
        }
        n_identity += cells.pos_is_identity() && cells.get_used() > 0;
        n_sorted   += cells.pos_is_sorted()   && cells.get_used() > 0;
    }
    printf("seed %u: %d ops OK (identity after %d, sorted after %d)\n", seed, n_ops, n_identity, n_sorted);
}

int main() {
    try {
        fixed_cases();
        for (uint32_t seed = 1; seed <= 20; ++seed) {
            random_ops(seed, 20000);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    printf("OK\n");
    return 0;
}
