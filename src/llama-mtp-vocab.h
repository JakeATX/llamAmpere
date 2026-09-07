#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_mtp_vocab_map {
    int64_t n_vocab = 0;
    std::vector<int32_t> ids;
};

inline llama_mtp_vocab_map llama_mtp_vocab_read(std::istream & in, int64_t max_size = 32768) {
    llama_mtp_vocab_map result;
    std::string magic;
    int64_t count = 0;
    if (!(in >> magic >> result.n_vocab >> count) || magic != "llama-mtp-vocab-v1" ||
            result.n_vocab <= 0 || result.n_vocab > (1 << 24) || count <= 0 ||
            count > max_size || count > result.n_vocab) {
        throw std::runtime_error("invalid MTP vocabulary header or shortlist exceeds LLAMA_SPEC_CHAIN_SUB");
    }
    std::vector<bool> seen(result.n_vocab, false);
    result.ids.reserve(count);
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = -1;
        if (!(in >> id) || id < 0 || id >= result.n_vocab || seen[id]) {
            throw std::runtime_error("invalid, duplicate or missing MTP vocabulary token ID");
        }
        seen[id] = true;
        result.ids.push_back((int32_t) id);
    }
    std::string extra;
    if (in >> extra) {
        throw std::runtime_error("unexpected trailing MTP vocabulary data");
    }
    return result;
}

// Adaptive tail of the draft shortlist: the last `n_hot` entries of a loaded map are slots the
// runtime may repoint at token ids observed in recent requests (prompts and finished responses).
// Ranking is frequency across requests with an exponential recency decay; the per-request count is
// capped so that recurrence beats a single long document. Updated at request boundaries only, so
// the decode path and the graph shape are untouched. Header-only, so it can be tested without a model.
struct llama_mtp_hot_vocab {
    struct entry {
        float   score     = 0.0f; // score as of `last_seen`
        int64_t last_seen = 0;    // request index the score was last updated at
    };

    struct replacement {
        int32_t slot; // index into `slots`
        int32_t id;   // token id that now owns the slot
    };

    // tunables
    float   half_life   = 8.0f;  // requests after which an untouched score halves
    float   count_cap   = 8.0f;  // maximum contribution of a single request
    float   prune_score = 0.05f; // decayed score below which a candidate may be dropped
    int32_t prune_ratio = 4;     // the candidate table is pruned above prune_ratio*n_hot entries

    int64_t n_vocab     = 0;
    int32_t n_hot       = 0;
    int64_t req_counter = 0;

    std::vector<uint8_t> static_mask;             // [n_vocab] 1 = permanent part of the shortlist
    std::vector<int32_t> slot_of;                 // [n_vocab] hot slot holding the id, -1 if not resident
    std::vector<int32_t> slots;                   // [n_hot]   token id of each hot slot
    std::unordered_map<int32_t, entry> cands;     // observed non-static ids

    // `ids` is the full shortlist; its last `n_hot` entries become the hot slots
    void init(const std::vector<int32_t> & ids, int64_t vocab_size, int32_t hot) {
        if (vocab_size <= 0 || hot < 0 || (size_t) hot >= ids.size()) {
            throw std::runtime_error("invalid MTP hot vocabulary tail size");
        }
        n_vocab     = vocab_size;
        n_hot       = hot;
        req_counter = 0;
        static_mask.assign(n_vocab, 0);
        slot_of.assign(n_vocab, -1);
        slots.assign(n_hot, -1);
        cands.clear();

        const size_t n_static = ids.size() - (size_t) n_hot;
        for (size_t i = 0; i < n_static; ++i) {
            static_mask[ids[i]] = 1;
        }
        for (int32_t i = 0; i < n_hot; ++i) {
            slots[i]              = ids[n_static + i];
            slot_of[slots[i]]     = i;
        }
    }

    float decay(int64_t delta) const {
        return delta <= 0 ? 1.0f : std::exp2(-(float) delta / half_life);
    }

    float score_of(int32_t id) const {
        const auto it = cands.find(id);
        return it == cands.end() ? 0.0f : it->second.score * decay(req_counter - it->second.last_seen);
    }

    // one call per request boundary (prompt, or a finished response)
    void observe(const int32_t * toks, size_t n) {
        if (n_hot <= 0) {
            return;
        }
        std::unordered_map<int32_t, int32_t> counts;
        for (size_t i = 0; i < n; ++i) {
            const int32_t id = toks[i];
            if (id < 0 || id >= n_vocab || static_mask[id]) {
                continue; // out of range, media placeholder, or already permanently shortlisted
            }
            counts[id]++;
        }
        for (const auto & c : counts) {
            entry & e   = cands[c.first];
            e.score     = e.score * decay(req_counter - e.last_seen) + std::min((float) c.second, count_cap);
            e.last_seen = req_counter;
        }
        req_counter++;
        prune();
    }

    // keep the n_hot best ids, fill the freed slots with the best non-resident ones and report the
    // changes. Resident ids that stay keep their slot, so the caller only ever rewrites the tail.
    std::vector<replacement> select() {
        std::vector<replacement> out;
        if (n_hot <= 0) {
            return out;
        }

        struct ranked {
            float   score;
            int32_t slot; // -1 when the id is not resident
            int32_t id;
        };

        std::vector<ranked> pool;
        pool.reserve(slots.size() + cands.size());
        for (int32_t i = 0; i < n_hot; ++i) {
            pool.push_back({ slots[i] < 0 ? -1.0f : score_of(slots[i]), i, slots[i] });
        }
        for (const auto & c : cands) {
            if (slot_of[c.first] < 0) {
                pool.push_back({ c.second.score * decay(req_counter - c.second.last_seen), -1, c.first });
            }
        }
        if (pool.size() <= (size_t) n_hot) {
            return out; // every candidate fits, nothing to evict
        }

        std::sort(pool.begin(), pool.end(), [](const ranked & a, const ranked & b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            const bool ra = a.slot >= 0;
            const bool rb = b.slot >= 0;
            if (ra != rb) {
                return ra; // on a tie keep what is already resident: no churn
            }
            if (ra) {
                return a.slot > b.slot; // equal residents: the earliest slots are evicted first
            }
            return a.id < b.id;
        });

        std::vector<int32_t> incoming;
        std::vector<uint8_t> keep(n_hot, 0);
        for (int32_t i = 0; i < n_hot; ++i) {
            if (pool[i].slot >= 0) {
                keep[pool[i].slot] = 1;
            } else {
                incoming.push_back(pool[i].id);
            }
        }
        if (incoming.empty()) {
            return out;
        }

        size_t next = 0;
        for (int32_t i = 0; i < n_hot && next < incoming.size(); ++i) {
            if (keep[i]) {
                continue;
            }
            if (slots[i] >= 0) {
                slot_of[slots[i]] = -1;
            }
            const int32_t id = incoming[next++];
            slots[i]         = id;
            slot_of[id]      = i;
            out.push_back({ i, id });
        }
        return out;
    }

private:
    // bound the candidate table: drop the non-resident ids whose score has decayed away
    void prune() {
        if ((int64_t) cands.size() <= (int64_t) prune_ratio * n_hot) {
            return;
        }
        for (auto it = cands.begin(); it != cands.end(); ) {
            if (slot_of[it->first] < 0 && it->second.score * decay(req_counter - it->second.last_seen) < prune_score) {
                it = cands.erase(it);
            } else {
                ++it;
            }
        }
    }
};
