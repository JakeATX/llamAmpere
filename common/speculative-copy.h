#pragma once

// Copy mode (--spec-copy-tail N, default 0 = off): a context-lookup tail fused onto the model drafter's block.
//
// When the text being generated repeats an earlier stretch of the request (quoting, applying an edit to a file,
// reproducing code), the tokens after the drafter's block can be read from the earlier occurrence at no drafter
// cost. The tail is appended after the block and verified in the same target pass. Tail positions carry no draft
// distribution: the target samples its own token and the tail token is kept only if it is the same one
// (point mass), so the output law is the target's at any temperature.
//
// Gating (a wide verify costs rows, see the 8 -> 9 row step on the dense 27B):
//   - match:   the most recent earlier occurrence of the last tokens; >= long_min tokens is used on its own,
//              a shorter one (>= short_min) only when the drafter's block repeats it for >= agree tokens;
//   - block:   a block that departs from the copy anywhere drops the tail (the tail could not be reached, or the
//              copy has ended);
//   - trigger: the tail is scheduled only after full-accept rounds, counted by the adaptive depth controller
//              (narrow = depth 1, wide = depth 2); a miss inside the block of a wide round leaves wide at once.
// A match may overlap its own suffix, which makes a periodic repeat continue.
//
// Rollback on a recurrent/hybrid target: a draft longer than n_rs_seq is verified under a checkpoint, and a
// partial accept of it restores and replays. By default copy mode keeps n_rs_seq (no VRAM), so only a wide
// round that is not fully accepted pays that; --spec-copy-rs sizes the snapshots for block + tail instead.

#include "llama.h"
#include "speculative-adaptive.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

struct common_speculative_copy_params {
    int32_t tail      = 0;  // lookup tokens appended per round (0 = off)
    int32_t long_min  = 8;  // LLAMA_SPEC_COPY_LONG:  match length that needs no agreement
    int32_t short_min = 4;  // LLAMA_SPEC_COPY_SHORT: shortest match considered at all
    int32_t agree     = 2;  // LLAMA_SPEC_COPY_AGREE: block tokens that must repeat a short match
    int32_t scan_max  = 0;  // LLAMA_SPEC_COPY_SCAN:  history tokens searched, newest first (0 = all)

    static constexpr int32_t TAIL_MAX = 16;

    static int32_t env_int(const char * name, int32_t def, int32_t lo, int32_t hi) {
        const char * s = std::getenv(name);
        return s && s[0] ? std::max(lo, std::min(hi, (int32_t) std::atoi(s))) : def;
    }

    static common_speculative_copy_params from_env(int32_t tail) {
        common_speculative_copy_params p;
        p.tail      = std::max(0, std::min(tail, TAIL_MAX));
        p.long_min  = env_int("LLAMA_SPEC_COPY_LONG",  p.long_min,  2, 64);
        p.short_min = env_int("LLAMA_SPEC_COPY_SHORT", p.short_min, 1, p.long_min);
        p.agree     = env_int("LLAMA_SPEC_COPY_AGREE", p.agree,     1, 16);
        p.scan_max  = env_int("LLAMA_SPEC_COPY_SCAN",  p.scan_max,  0, INT32_MAX);
        return p;
    }
};

struct common_speculative_copy_match {
    int32_t next = -1; // index of the token that followed the earlier occurrence (-1 = no match)
    int32_t len  = 0;  // matched tokens, counted back from the end of the history, capped at long_min
};

// History = hist[0, n) followed by id_last. Finds an earlier occurrence of its suffix: the most recent one of
// at least long_min tokens, else the longest one of at least short_min tokens (the most recent on a tie).
inline common_speculative_copy_match common_speculative_copy_find(
        const common_speculative_copy_params & cp, const llama_token * hist, int32_t n, llama_token id_last) {
    common_speculative_copy_match best;
    const int32_t stop = cp.scan_max > 0 ? std::max(0, n - cp.scan_max) : 0;
    for (int32_t p = n - 1; p >= stop; --p) {
        if (hist[p] != id_last) {
            continue;
        }
        int32_t len = 1;
        while (len < cp.long_min && p - len >= 0 && hist[p - len] == hist[n - len]) {
            len++;
        }
        if (len > best.len) {
            best.next = p + 1;
            best.len  = len;
            if (len >= cp.long_min) {
                break;
            }
        }
    }
    if (best.len < cp.short_min) {
        return {};
    }
    return best;
}

// The tail to append after `draft` (the model drafter's block), at most n_tail tokens; empty when the gate fails.
// The continuation is read from the virtual stream hist ++ id_last ++ draft ++ tail, so a match that runs into
// the tokens being proposed keeps repeating.
inline std::vector<llama_token> common_speculative_copy_propose(
        const common_speculative_copy_params & cp, const llama_token * hist, int32_t n, llama_token id_last,
        const std::vector<llama_token> & draft, int32_t n_tail) {
    std::vector<llama_token> tail;
    if (n_tail <= 0 || n <= 0 || draft.empty()) {
        return tail;
    }
    const auto m = common_speculative_copy_find(cp, hist, n, id_last);
    if (m.next < 0) {
        return tail;
    }
    const int32_t n_block = (int32_t) draft.size();
    if (m.len < cp.long_min && n_block < cp.agree) {
        return tail;
    }
    const auto at = [&](int32_t i) -> llama_token {
        if (i < n) {
            return hist[i];
        }
        if (i == n) {
            return id_last;
        }
        const int32_t j = i - n - 1;
        return j < n_block ? draft[j] : tail[j - n_block];
    };
    // the block must repeat the copy; with full agreement required, a short match is backed by >= agree tokens
    for (int32_t i = 0; i < n_block; ++i) {
        if (at(m.next + i) != draft[i]) {
            return tail;
        }
    }
    tail.reserve(n_tail);
    for (int32_t i = 0; i < n_tail; ++i) {
        tail.push_back(at(m.next + n_block + i));
    }
    return tail;
}

// Appends the tail. Tail positions get an empty candidate row = id match (point mass) in the exact p/q verify;
// a drafter that recorded no rows keeps none (the whole draft is then verified by id match).
inline void common_speculative_copy_append(
        std::vector<llama_token> & result, std::vector<std::vector<llama_token_data>> * result_q,
        const std::vector<llama_token> & tail) {
    result.insert(result.end(), tail.begin(), tail.end());
    if (result_q && !result_q->empty()) {
        result_q->resize(result.size());
    }
}

// Per-sequence wide trigger and round bookkeeping. The streak rule is the adaptive depth controller on a
// two-step range: depth 1 = narrow, depth 2 = wide (two full-accept rounds in a row climb, see climb_threshold(1)).
struct common_speculative_copy_state {
    common_speculative_adaptive ctrl;
    int32_t n_block = 0; // model drafter tokens in the current draft
    int32_t n_tail  = 0; // lookup tokens appended to the current draft

    uint64_t n_rounds   = 0; // rounds that carried a tail
    uint64_t n_gen      = 0; // tail tokens proposed
    uint64_t n_acc      = 0; // tail tokens accepted
    uint64_t n_miss_blk = 0; // wide rounds that missed inside the block

    void reset() {
        ctrl.reset(2, 1);
        n_block = 0;
        n_tail  = 0;
    }

    bool wide() const {
        return ctrl.n_cur >= 2;
    }

    void on_draft(int32_t block, int32_t tail) {
        n_block = block;
        n_tail  = tail;
        if (tail > 0) {
            n_rounds++;
            n_gen += tail;
        }
    }

    // feed one verified round; returns the accepted count that belongs to the model drafter's block
    int32_t on_accept(int32_t n_accepted) {
        const int32_t acc_block = std::min(n_accepted, n_block);
        if (n_block <= 0) {
            return n_accepted;
        }
        if (n_tail > 0) {
            n_acc += std::max(0, std::min(n_accepted - n_block, n_tail));
        }
        if (n_tail > 0 && acc_block < n_block) {
            n_miss_blk++;
            ctrl.reset(2, 1);
        } else {
            ctrl.update(n_block + n_tail, std::min(n_accepted, n_block + n_tail), 2, 1);
        }
        return acc_block;
    }
};
