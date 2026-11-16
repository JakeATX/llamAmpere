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
//              (narrow = depth 1, wide = depth 2); any miss inside the block leaves wide at once.
// A match may overlap its own suffix, which makes a periodic repeat continue.
//
//   - credit:  a tail of which less than half was accepted starts a cool-down (no tail for cool, 2x cool, ...
//              rounds, doubling per poor tail in a row, up to COOL_MAX); a tail accepted for at least half clears it.
//
// Rollback on a recurrent/hybrid target (the server side, see server-context.cpp): the recurrent state can be
// rolled back in place by up to n_rs_seq tokens from the end of the verify batch. The server caps the tail at
// n_rs_seq, so a miss inside the tail always rolls back in place; only a miss inside the block of a wide round
// is deeper, and that one restores a checkpoint and replays. The checkpoint is not taken every wide round: one
// taken at most LLAMA_SPEC_COPY_CKPT_STALE tokens ago is reused, and the replay then re-decodes from it.
// --spec-copy-rs sizes the snapshots for block + tail instead (more VRAM).

#include "llama.h"
#include "speculative-adaptive.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

struct common_speculative_copy_params {
    int32_t tail      = 0;  // lookup tokens appended per round (0 = off)
    int32_t long_min  = 8;  // LLAMA_SPEC_COPY_LONG:  match length that needs no agreement
    int32_t short_min = 8;  // LLAMA_SPEC_COPY_SHORT: shortest match considered at all (= long_min: no short rule)
    int32_t agree     = 2;  // LLAMA_SPEC_COPY_AGREE: block tokens that must repeat a short match
    int32_t scan_max  = 0;  // LLAMA_SPEC_COPY_SCAN:  history tokens searched, newest first (0 = all)
    int32_t cool      = 4;  // LLAMA_SPEC_COPY_COOL:  rounds without a tail after a poor tail (0 = no cool-down)

    static constexpr int32_t TAIL_MAX = 16;
    static constexpr int32_t COOL_MAX = 64;

    static int32_t env_int(const char * name, int32_t def, int32_t lo, int32_t hi) {
        const char * s = std::getenv(name);
        return s && s[0] ? std::max(lo, std::min(hi, (int32_t) std::atoi(s))) : def;
    }

    static int32_t resolve_tail(int32_t requested, bool eligible, const char * env) {
        if (requested >= 0) {
            return requested;
        }
        if (!eligible || env == nullptr || env[0] == '\0') {
            return 0;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        // Keep the old boolean opt-in at tail 8.
        if (value == "1" || value == "on" || value == "true" || value == "yes") {
            return 8;
        }
        char * end = nullptr;
        const long tail = std::strtol(value.c_str(), &end, 10);
        return end != value.c_str() && *end == '\0' && tail > 0 ? (int32_t) std::min(tail, (long) TAIL_MAX) : 0;
    }

    static common_speculative_copy_params from_env(int32_t tail) {
        common_speculative_copy_params p;
        p.tail      = std::max(0, std::min(tail, TAIL_MAX));
        p.long_min  = env_int("LLAMA_SPEC_COPY_LONG",  p.long_min,  2, 64);
        p.short_min = env_int("LLAMA_SPEC_COPY_SHORT", p.short_min, 1, p.long_min);
        p.agree     = env_int("LLAMA_SPEC_COPY_AGREE", p.agree,     1, 16);
        p.scan_max  = env_int("LLAMA_SPEC_COPY_SCAN",  p.scan_max,  0, INT32_MAX);
        p.cool      = env_int("LLAMA_SPEC_COPY_COOL",  p.cool,      0, COOL_MAX);
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
// verify-row cap of a copy round: rows = drafted tokens (block + tail) + the sampled token. Returns the most
// tokens the round may draft; n_rows <= 0 means no cap.
inline int32_t common_speculative_copy_total_cap(int32_t n_draft_max, int32_t n_rows) {
    if (n_rows <= 0) {
        return n_draft_max;
    }
    return n_draft_max < n_rows - 1 ? n_draft_max : n_rows - 1;
}

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
    uint64_t n_poor     = 0; // tails accepted for less than half (each starts a cool-down)
    uint64_t n_cooled   = 0; // rounds on a full-accept streak that carried no tail because of a cool-down

    // tail credit: cool-down after a poor tail (cool_base is configuration, it survives reset())
    int32_t cool_base = 0; // 0 = no cool-down
    int32_t cool      = 0; // rounds left without a tail
    int32_t n_fail    = 0; // poor tails in a row

    // per request (common_speculative_begin): the statistics line printed at the end of a request covers that request
    void reset() {
        ctrl.reset(2, 1);
        n_block    = 0;
        n_tail     = 0;
        n_rounds   = 0;
        n_gen      = 0;
        n_acc      = 0;
        n_miss_blk = 0;
        n_poor     = 0;
        n_cooled   = 0;
        cool       = 0;
        n_fail     = 0;
    }

    bool wide() const {
        return ctrl.n_cur >= 2 && cool <= 0;
    }

    void on_draft(int32_t block, int32_t tail) {
        n_block = block;
        n_tail  = tail;
        if (tail > 0) {
            n_rounds++;
            n_gen += tail;
        } else if (cool > 0 && ctrl.n_cur >= 2) {
            n_cooled++;
        }
    }

    // feed one verified round; returns the accepted count that belongs to the model drafter's block
    int32_t on_accept(int32_t n_accepted) {
        if (n_block <= 0) {
            return n_accepted;
        }
        const int32_t acc_block = std::min(n_accepted, n_block);
        if (n_tail > 0) {
            const int32_t acc_tail = std::max(0, std::min(n_accepted - n_block, n_tail));
            n_acc += acc_tail;
            if (acc_tail * 2 >= n_tail) {
                n_fail = 0;
            } else {
                n_poor++;
                if (cool_base > 0) {
                    n_fail = std::min(n_fail + 1, 8);
                    cool   = std::min(common_speculative_copy_params::COOL_MAX, cool_base << (n_fail - 1));
                }
            }
        } else if (cool > 0) {
            cool--;
        }
        if (acc_block < n_block) {
            // a miss inside the block ends the full-accept streak at once, with or without a tail this round
            // (without this a wide round whose lookup found nothing would only add drop pressure on a block miss)
            if (n_tail > 0) {
                n_miss_blk++;
            }
            ctrl.reset(2, 1);
        } else {
            ctrl.update(n_block + n_tail, std::min(n_accepted, n_block + n_tail), 2, 1);
        }
        return acc_block;
    }
};
