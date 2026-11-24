#include "llama-moe-cache.h"
#include "llama-moe-cache-policy.h"
#include "llama-moe-ramlock.h"
#include "llama-moe-prefetch.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-moe-nvme.h"
#include "llama-moe-onegraph.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    // fork (INKMISS, R3 staged admission; LLAMA_MOE_CACHE_ROUND=1, off by default). Once the cache is full, a miss of a
    // decode ubatch is uploaded into one of n_stage staging slots (after the zero slots) and computed from there, unless
    // it is admitted: its CRF beats the decayed CRF of its victim by admit_margin, and the window's admission budget is
    // not spent. A staged expert does not enter the LRU and evicts nothing. admit_budget < 0: no admission control.
    int32_t n_stage      = 0;
    int32_t stage_base   = 0;     // first staging slot (n_slots + n_zero)
    float   admit_margin = 0.0f;  // S2-SWAP-POLICY pair swaps (LLAMA_MOE_CACHE_PAIR): hysteresis between candidate and victim
    int32_t admit_budget = -1;
    bool    declined_budget = false; // a miss beat its victim but was staged because the budget was spent
    std::vector<int32_t> declined;   // their keys (S2-NOWAIT promotes them in the background)
    // fork (INKMISS, S2-NOWAIT): busy[s] == busy_gen: slot s is being written on the copy stream, it is never a victim
    std::vector<uint32_t> busy;
    uint32_t busy_gen = 0;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    bool is_busy(int32_t s) const {
        return busy_gen != 0 && busy[s] == busy_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s]) && !is_busy(s)) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    // fork (R4, NEXTPF): victim() that also skips the slots for which keep(s) is true; -1 if every slot is kept
    template <typename K>
    int32_t victim_keep(int32_t il, K && keep) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            for (int32_t s = head[c]; s >= 0; s = next[s]) {
                if (is_selected(il, key_of[s]) || keep(s)) {
                    continue;
                }
                const int32_t key = key_of[s];
                if (key < 0) {
                    return s;
                }
                const double score = std::ldexp((double) (clock[key_layer(key)] - last_use[key]), -c);
                if (score > res_score) {
                    res       = s;
                    res_score = score;
                }
                break; // the least recently used candidate of the list, as candidate()
            }
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    // few_rows > 0 (fork SURGE-PROMPT, LLAMA_MOE_CACHE_FEWROW_CPU): instead of n_cpu, the misses that this ubatch routes to
    // at most few_rows tokens are left to the CPU (lowest CRF first if the zero slots run out); the others are uploaded
    std::vector<int32_t> miss_rows; // [n_expert] tokens routed to each expert, only with few_rows > 0
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit, bool advance = true, int32_t few_rows = 0) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        size_t n_c = 0;
        if (few_rows > 0) {
            miss_rows.assign(n_expert, 0);
            for (size_t i = 0; i < n_ids; ++i) {
                miss_rows[ids[i]]++;
            }
            for (int32_t e : misses) {
                n_c += miss_rows[e] <= few_rows ? 1 : 0;
            }
        } else {
            n_c = std::min(n_cpu(misses.size()), misses.size());
        }
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                if (few_rows > 0) {
                    const bool few_a = miss_rows[a] <= few_rows;
                    const bool few_b = miss_rows[b] <= few_rows;
                    if (few_a != few_b) {
                        return few_a; // the few-row misses first: only they go to the CPU
                    }
                }
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        int32_t n_staged = 0;
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (admit_budget >= 0 && n_filled == n_slots) {
                // fork (INKMISS, R3): admit, or stage while staging slots last
                const int32_t key = make_key(il, e);
                const int32_t old = key_of[s];
                const bool better = old < 0 || crf[key] > decayed_crf(key_layer(old), old) + admit_margin;
                const bool admit  = better && admit_budget > 0;
                if (!admit && n_staged < n_stage) {
                    if (better) {
                        declined_budget = true;
                        declined.push_back(key);
                    }
                    slots[e] = stage_base + n_staged;
                    fills.push_back({ e, stage_base + n_staged });
                    n_staged++;
                    continue;
                }
                if (admit) {
                    admit_budget--;
                }
            }
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        if (advance) {
            clock[il] += n_tokens; // fork (INKOFFLOAD): a looped layer's later visits do not age the clock again
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

// fork (INKSMALL2): bytes of host-resident experts the cache could hold, each looped (shared) layer counted once.
// Used by the context to size the cache when --moe-cache-mib is not given. 0 = no layer keeps all experts on the host.
size_t llama_moe_cache_host_expert_bytes(const llama_model & model) {
    if (model.hparams.n_expert == 0) {
        return 0;
    }
    size_t total = 0;
    std::vector<std::vector<ggml_tensor *>> seen;
    for (const auto & layer : model.layers) {
        auto experts = llama_moe_cache_layer_experts(layer);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight) ||
            std::find(seen.begin(), seen.end(), experts) != seen.end()) {
            continue;
        }
        seen.push_back(experts);
        for (const ggml_tensor * t : experts) {
            total += t->nb[1] * t->ne[1] * t->ne[2];
        }
    }
    return total;
}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        bool elastic = false; // fork (PLOAN): buf is a VMM buffer whose pages can be unmapped
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        int32_t n_canon = 0;              // layers that own their experts (looped layers reuse a canonical layer's)
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    // fork (INKOFFLOAD): LLAMA_MOE_CACHE_MAX_BATCH raises the largest ubatch that runs through the cache (default 32).
    // Without op offload (small cards with looped layers), larger prompt ubatches would otherwise run on the CPU.
    const int64_t max_batch = [] {
        const char * env = getenv("LLAMA_MOE_CACHE_MAX_BATCH");
        const long long v = env ? atoll(env) : 32;
        return (int64_t) std::max(1LL, std::min(v, 65536LL));
    }();

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    // fork (INKOFFLOAD): canon[il] is the first layer with the same expert tensors as il. Looped models (Inkling 2)
    // run one physical layer several times per token; its visits share one LRU key space and one slot array, so a
    // physical expert is cached once. LLAMA_MOE_CACHE_LOOP_SHARE=0 keys every visit on its own (the old behaviour).
    std::vector<int32_t> canon;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // fork (#130, RAMLOCK): LLAMA_MOE_RAM_LOCK_GIB, nullptr when unset (see llama-moe-ramlock.h)
    std::unique_ptr<llama_moe_ramlock> ramlock;

    // fork (R4, NEXTPF): LLAMA_MOE_CACHE_NEXT_PREFETCH, nullptr when unset (see llama-moe-prefetch.h); stopped in ~impl()
    std::unique_ptr<llama_moe_prefetch> prefetch;
    std::vector<uint32_t> pf_pred;    // [n_expert] == pf_gen: predicted for the next layer
    uint32_t pf_gen = 0;
    std::vector<int32_t> pf_experts;
    std::vector<moe_cache_lru::fill> pf_fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    bool is_suspended = false; // fork (INKSTREAM): banks freed for a streamed prompt

    // fork (PLOAN, LLAMA_INK_PLOAN=1, off by default): the banks sit in one VMM buffer (one device). A streamed prompt
    // then keeps the first keep_n[g] slots of each bank of group g mapped (suspend_keep) and the streamer copies those
    // experts device to device; resume() maps the rest again and uploads only the dropped slots.
    using elastic_buft_fn   = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t);
    using elastic_set_fn    = bool   (*)(ggml_backend_buffer_t, const size_t *, const size_t *, int);
    using elastic_mapped_fn = size_t (*)(ggml_backend_buffer_t);
    const bool ploan_on = [] { const char * e = getenv("LLAMA_INK_PLOAN"); return e && atoi(e) == 1; }();
    ggml_backend_buffer_type_t buft_elastic = nullptr; // set only with ploan_on, one device
    elastic_set_fn    el_set    = nullptr;
    elastic_mapped_fn el_mapped = nullptr;
    bool is_partial = false;
    std::vector<int32_t> keep_n; // [group] kept slots while partial
    bool elastic_warned = false;
    int32_t res_n_up = 0, res_n_all = 0; // the last resume(): slots uploaded, of the filled slots
    size_t  res_bytes = 0;

    // fork (SURGE-cache, round admission; LLAMA_MOE_CACHE_ROUND=1, off by default). Once the cache is full, a decode
    // miss is computed on the CPU from the host experts instead of being uploaded on the spot (the graph's CPU part,
    // as with --moe-cache-cpu-share 1). After the last layer of the window (one decode ubatch) has been served, the
    // experts the CPU computed are ranked by their decayed use count and the hottest are uploaded in place of the
    // coldest cached ones: at most LLAMA_MOE_CACHE_ROUND_SWAPS (16) per window, LLAMA_MOE_CACHE_ROUND_BOOST (4) times
    // that while a backlog lasts (the previous round left candidates that would still beat the coldest slot).
    // An expert routed once qualifies when its count beats the coldest cached expert's decayed count.
    bool     round_on    = false;
    int32_t  round_swaps = 16;
    int32_t  round_boost = 4;
    int32_t  round_prev_il = -1;               // layer of the previous lookup: a lower or equal one starts a window
    std::vector<std::vector<int32_t>> round_cand; // [group] keys the CPU computed in this window
    std::vector<uint32_t> round_mark;          // [n_layer*n_expert] == round_gen when the key is in round_cand
    uint32_t round_gen = 1;
    std::vector<uint8_t> round_backlog;        // [group]
    size_t   round_n = 0, round_n_swaps = 0, round_n_backlog = 0;

    // fork (INKMISS, R3): LLAMA_MOE_CACHE_ROUND=1 is staged admission (see moe_cache_lru::n_stage): decode misses are
    // uploaded as before, on the spot, but only the ones that beat their victim enter the cache, at most
    // LLAMA_MOE_CACHE_ROUND_SWAPS per window (x LLAMA_MOE_CACHE_ROUND_BOOST after a window that ran out); the others go
    // to LLAMA_MOE_CACHE_ROUND_STAGE (default 2 x n_expert_used) staging slots. No CPU part, so no extra graph splits
    // and no CPU expert compute (the old design, 4.8 vs ~29 tok/s, is LLAMA_MOE_CACHE_ROUND=2: round_on below).
    bool     stage_on = false;
    int32_t  n_stage  = 0;
    float    pair_margin = 0.0f;                // S2-SWAP-POLICY: LLAMA_MOE_CACHE_PAIR=<CRF margin>
    int32_t  stage_first_il = -1;               // a window starts at the lookup of this layer

    // fork (INKMISS, S2-NOWAIT; LLAMA_MOE_CACHE_NOWAIT=1 with ROUND=1, off by default): the misses a window staged only
    // because its budget was spent are uploaded into the cache at the next window start on a copy stream; the slots
    // being written are never victims, and a lookup that selects one makes the compute stream wait for the copy event
    // (a device-side wait, the host never blocks; one batch in flight, synchronized at the following window start).
    // Fast with page-locked host experts (LLAMA_INK_HOST_PIN): from file pages each copy is staged by the driver.
    bool     nowait_on = false;
    ggml_backend_t       nw_backend = nullptr;  // copy stream on the cache's device
    ggml_backend_event_t nw_event   = nullptr;
    bool     nw_pending = false;
    size_t   nw_n_batches = 0, nw_n_swaps = 0, nw_n_waits = 0;
    size_t   stage_n_staged = 0, stage_n_admitted = 0;

    // fork (INKMISS, S2-SWAP-POLICY seeding): LLAMA_MOE_CACHE_SEED=<file> fills the empty cache at boot with the hottest
    // experts of a saved profile; LLAMA_MOE_CACHE_SEED_SAVE=<file> writes the decayed use counts at exit.
    std::string seed_save;

    // fork (SURGE-cache, MCDET; LLAMA_MOE_CACHE_DET=1, off by default): no miss is computed on the CPU (its last bits
    // differ from the GPU's), so a request's output does not depend on what earlier requests left in the cache.
    bool     det_on = false;

    // fork (SURGE-PROMPT, PROMPTPOL mid-size prompt policy): LLAMA_MOE_CACHE_FEWROW_CPU=R (1..64, unset or 0 = off).
    // In a ubatch of at least LLAMA_MOE_CACHE_FEWROW_MIN_TOKENS rows (default 17: prompt chunks, not decode or verify),
    // a missed expert that only R or fewer of its tokens route to is computed on the CPU from the host experts (where
    // it lives) instead of being uploaded over PCIe into the cache, which also keeps it from evicting the experts that
    // decode uses. Misses with more rows are uploaded as before. It needs the zero slots of the CPU-share path, so it
    // adds n_expert_used zero slots per group (the cache holds that many fewer experts). Smaller ubatches keep the graph
    // without a CPU part, so decode runs the same kernels as with the flag off. Rounding-level: a CPU-computed expert
    // sums in another order than the GPU one.
    const int32_t fewrow_r = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_FEWROW_CPU");
        const long v = e && e[0] ? strtol(e, nullptr, 10) : 0;
        return (int32_t) std::clamp(v, 0l, 64l);
    }();
    const int64_t fewrow_min = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_FEWROW_MIN_TOKENS");
        const long v = e && e[0] ? strtol(e, nullptr, 10) : 17;
        return (int64_t) std::clamp(v, 2l, 65536l);
    }();
    bool fewrow_op_offload = true; // the scheduler offloads large host-weight ops (see set_op_offload)

    // fork (INKCACHEUX): GS1 controls, GS4 heat profile, #249/GS9 miss split, GS8 boundary (llama-moe-cache-policy.h)
    llama_moe_cache_controls ux;
    llama_moe_cache_heat     heat;
    llama_moe_split_policy   split;
    stats req_small, req_large; // stats at the previous request end
    // fork (NVMETIER, LLAMA_MOE_NVME_TIER=1): the uploads read the host experts from the RAM tier, which reads the
    // missing ones from the model file; nullptr (the default) reads them from the mapping as before
    llama_moe_nvme_tier * nvme = llama_moe_nvme_global();
    // host bytes of an upload: the RAM tier copy when the tier is on
    const uint8_t * host_src(const ggml_tensor * w, size_t offs, size_t size) const {
        const uint8_t * p = (const uint8_t *) w->data + offs;
        if (nvme != nullptr) {
            if (const void * q = nvme->ensure(p, size)) {
                return (const uint8_t *) q;
            }
        }
        return p;
    }
    // fork (ONEGRAPH, LLAMA_MOE_ONEGRAPH=1, off by default): decode lookups on the GPU, misses served by a host worker
    // (llama-moe-onegraph.h). Declared last: it stops its worker before the rest of the cache goes away.
    int32_t og_n_expert = 0;
    std::vector<int32_t> og_ids;
    std::vector<moe_cache_lru::fill> og_fills, og_cpu;
    std::unique_ptr<llama_moe_og> og;


    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        ux.from_env((int32_t) model.layers.size()); // fork (INKCACHEUX) GS1
        const char * loop_share_env = getenv("LLAMA_MOE_CACHE_LOOP_SHARE");
        const bool loop_share = loop_share_env == nullptr || strcmp(loop_share_env, "0") != 0;
        canon.resize(model.layers.size());
        for (size_t il = 0; il < model.layers.size(); ++il) {
            canon[il] = (int32_t) il;
            if (!loop_share) {
                continue;
            }
            const auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            for (size_t jl = 0; jl < il && !experts.empty(); ++jl) {
                if (llama_moe_cache_layer_experts(model.layers[jl]) == experts) {
                    canon[il] = (int32_t) jl;
                    break;
                }
            }
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        const char * cpu_exl3 = getenv("GGML_CPU_EXL3");
        const bool native_cpu_exl3 = cpu_exl3 && strcmp(cpu_exl3, "1") == 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() ||
                (native_cpu_exl3 && std::any_of(experts.begin(), experts.end(), [](const ggml_tensor * t) { return ggml_exl3_bits(t->type) != 0; })) ||
                !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight) ||
                !ux.layer_selected(canon[il])) { // fork (INKCACHEUX) GS1: a looped layer's visits follow its first visit
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            if (canon[il] != (int32_t) il) {
                // a later visit of a looped layer: same host experts, counted once
                const auto it_c = std::find(it->layers.begin(), it->layers.end(), canon[il]);
                if (it_c == it->layers.end()) {
                    // canonical layer cached in another group (another device): its bindings own these host
                    // tensors, so this visit is not cached
                    it->layers.pop_back();
                    if (it->layers.empty()) {
                        groups.erase(it);
                    }
                }
                continue;
            }
            it->n_canon++;
            for (const ggml_tensor * t : experts) {
                // EXL3 experts: nb[2] can exceed nb[1]*ne[1], so size an expert by its rows
                const size_t bytes = t->nb[1] * t->ne[1] * t->ne[2];
                it->host_bytes         += bytes;
                devices[id].host_bytes += bytes;
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        {
            const char * e = getenv("LLAMA_MOE_CACHE_ROUND");
            const int round_mode = e ? atoi(e) : 0;
            round_on = round_mode == 2;                 // fork (INKMISS): the old CPU-computed design
            stage_on = round_mode != 0 && round_mode != 2; // 1 (any other non-zero value too)
            const char * d = getenv("LLAMA_MOE_CACHE_DET");
            det_on = d && atoi(d) != 0;
            if (det_on) {
                round_on = false;
            }
            if (stage_on) {
                const char * st = getenv("LLAMA_MOE_CACHE_ROUND_STAGE");
                n_stage = std::clamp(st && st[0] ? atoi(st) : 2*n_expert_used, n_expert_used, 64*n_expert_used);
                if (const char * pm = getenv("LLAMA_MOE_CACHE_PAIR"); pm && pm[0]) {
                    pair_margin = std::max(0.0f, (float) atof(pm));
                }
                const char * nw = getenv("LLAMA_MOE_CACHE_NOWAIT");
                nowait_on = nw && atoi(nw) != 0;
            }
            if (const char * ss = getenv("LLAMA_MOE_CACHE_SEED_SAVE"); ss && ss[0]) {
                seed_save = ss;
            }
            if (const char * w = getenv("LLAMA_MOE_CACHE_ROUND_SWAPS"); w && w[0]) {
                round_swaps = std::max(1, atoi(w));
            }
            if (const char * b = getenv("LLAMA_MOE_CACHE_ROUND_BOOST"); b && b[0]) {
                round_boost = std::max(1, atoi(b));
            }
        }
        n_zero = (cpu_share > 0.0f && !det_on) || round_on || fewrow_r > 0 ? std::min<int32_t>(n_expert_used, 64) : 0; // merge: cache + prompt zero-slot users
        if (stage_on) {
            LLAMA_LOG_WARN("%s: [MCROUND] staged admission on: %d staging slots per group, %d admissions per window (x%d after "
                    "a window that ran out), pair margin %.2f%s\n", __func__, n_stage, round_swaps, round_boost, pair_margin,
                    nowait_on ? ", background promotion on a copy stream (NOWAIT)" : "");
        }
        if (round_on) {
            round_cand.assign(groups.size(), {});
            round_backlog.assign(groups.size(), 0);
            round_mark.assign(model.layers.size()*(size_t) model.hparams.n_expert, 0);
            LLAMA_LOG_WARN("%s: [MCROUND] round admission on: decode misses on the CPU, %d swaps per window (x%d under a backlog)\n",
                    __func__, round_swaps, round_boost);
        }
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD((t->nb[1]*t->ne[1])*(n_slots + n_zero + n_stage + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.n_canon*n_expert;
            if (ux.slots > 0) {
                g.n_slots = std::min(ux.slots, max_slots); // fork (INKCACHEUX) GS1: slot-count form
                if (alloc_size(g, g.n_slots) > budget) {
                    LLAMA_LOG_WARN("%s: [CACHEUX] %d slots for %zu layers take %.1f MiB, over the %.1f MiB the fit gave this group "
                            "(the slot count bypasses the fit; the context may not fit)\n", __func__, g.n_slots, g.layers.size(),
                            alloc_size(g, g.n_slots)/1048576.0, budget/1048576.0);
                }
            } else if (const double mib = ux.dev_budget_mib(g.id); mib >= 0.0) {
                const size_t b = (size_t) (mib*1024.0*1024.0*g.host_bytes/d.host_bytes); // GS1: MiB per GPU
                while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= b) {
                    g.n_slots++;
                }
            } else {
                while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                    g.n_slots++;
                }
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            init_stage(g);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + n_stage + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[1] * t->ne[1]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size() && canon[il] == il; ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero + n_stage, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        for (size_t il = 0; stage_on && il < layers.size() && stage_first_il < 0; ++il) {
            stage_first_il = layers[il].ig >= 0 ? (int32_t) il : -1; // fork (INKMISS, R3): window start
        }

        if (ploan_on && !no_alloc) {
            ploan_init();
        }
        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            if (buft_elastic && alloc_elastic(d)) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        if (canon[il] == il) {
                            g.lru.slot_map[il] = slots; // visits of a looped layer plan and upload this one
                        }
                    }
                }
            }
            if (const char * sp = getenv("LLAMA_MOE_CACHE_SEED"); sp && sp[0]) {
                seed_load(sp);
            }
        }

        if (!no_alloc) {
            ramlock = llama_moe_ramlock::from_env((int32_t) layers.size(), (int32_t) model.hparams.n_expert);
            if (ramlock) {
                for (const group & g : groups) {
                    for (int32_t il : g.layers) {
                        if (canon[il] == il) {
                            ramlock->add_layer(il, layers[il].experts);
                        }
                    }
                }
                ramlock->start();
            }
            prefetch = llama_moe_prefetch::from_env((int32_t) layers.size());
            if (prefetch && (round_on || det_on)) {
                LLAMA_LOG_WARN("%s: [NEXTPF] off: not with LLAMA_MOE_CACHE_ROUND / LLAMA_MOE_CACHE_DET\n", __func__);
                prefetch.reset();
            }
            if (prefetch) {
                pf_pred.assign(model.hparams.n_expert, 0);
                LLAMA_LOG_WARN("%s: [NEXTPF] decode lookups upload up to %d experts of the next layer ahead (LLAMA_MOE_CACHE_NEXT_PREFETCH)\n",
                        __func__, prefetch->budget());
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ux_init(model, no_alloc); // fork (INKCACHEUX)

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            if (fewrow_r > 0) {
                LLAMA_LOG_WARN("%s: [FEWROW] misses routed to <= %d rows of a ubatch of >= %lld rows are computed on the CPU "
                        "(LLAMA_MOE_CACHE_FEWROW_CPU), %d zero slots per group\n", __func__, fewrow_r, (long long) fewrow_min, n_zero);
            }
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.n_canon*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                    if (g.n_canon != (int32_t) g.layers.size()) {
                        LLAMA_LOG_INFO("%s: %2zu layer visits share the experts of %d physical layers (LLAMA_MOE_CACHE_LOOP_SHARE=0 disables)\n",
                            __func__, g.layers.size(), g.n_canon);
                    }
                }
            }
        }
        og_init(model, no_alloc);
    }

    // fork (ONEGRAPH): one device, every group sized, no zero slots (CPU share, round admission, few-row tier) and no trace
    void og_init(const llama_model & model, bool no_alloc) {
        if (!llama_moe_og::requested() || no_alloc || n_zero > 0 || round_on || fewrow_r > 0 || trace != nullptr) {
            if (llama_moe_og::requested() && !no_alloc) {
                LLAMA_LOG_WARN("%s: [ONEGRAPH] off: needs no CPU share, round admission, few-row tier or trace\n", __func__);
            }
            return;
        }
        int32_t n_dev = 0, id = -1;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].ctx) {
                n_dev++;
                id = (int32_t) i;
            }
        }
        og_n_expert = (int32_t) model.hparams.n_expert;
        bool ok = n_dev == 1;
        for (const group & g : groups) {
            ok = ok && g.ref[0]->ne[2] == og_n_expert;
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: [ONEGRAPH] off: one device and one expert count only\n", __func__);
            return;
        }
        // the host experts the GPU copies misses from (looped visits share their canonical layer's tensors)
        std::vector<std::pair<void *, size_t>> pins;
        std::set<const void *> seen;
        for (const layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (const ggml_tensor * w : l.experts) {
                if (seen.insert(w->data).second) {
                    pins.emplace_back(w->data, ggml_nbytes(w));
                }
            }
        }
        // CUDA pins whole pages: ranges that share a page (adjacent tensors) are registered as one
        std::sort(pins.begin(), pins.end());
        std::vector<std::pair<void *, size_t>> merged;
        for (const auto & p : pins) {
            const uintptr_t b = (uintptr_t) p.first;
            if (!merged.empty()) {
                const uintptr_t mb = (uintptr_t) merged.back().first;
                const uintptr_t me = mb + merged.back().second;
                if (b < me + 65536) {
                    merged.back().second = std::max(me, b + p.second) - mb;
                    continue;
                }
            }
            merged.push_back(p);
        }
        og.reset(llama_moe_og::create(devices[id].backend, devices[id].buft, og_n_expert, (int32_t) layers.size(), merged,
            [this](const ggml_moe_og_rec & rec, std::vector<ggml_moe_og_copy> & copies) { return og_serve(rec, copies); },
            [this](int32_t * table) { og_snapshot(table); }));
    }

    // fork (ONEGRAPH): the worker's half of copy(): same plan, the GPU does the copies (no CPU part)
    int og_serve(const ggml_moe_og_rec & rec, std::vector<ggml_moe_og_copy> & copies) {
        const int32_t il = rec.layer;
        if (is_suspended || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || rec.n_used <= 0 ||
                rec.n_ids <= 0 || rec.n_ids % rec.n_used != 0) {
            return 0;
        }
        const int32_t ci = canon[il];
        layer & l = layers[il];
        group & g = groups[l.ig];
        if (g.n_slots == 0) {
            return 0;
        }
        og_ids.assign(rec.ids, rec.ids + rec.n_ids);
        size_t n_hit = 0;
        auto no_cpu = [](size_t) { return (size_t) 0; };
        if (!g.lru.plan(ci, og_ids.data(), og_ids.size(), rec.n_ids / rec.n_used, rec.n_used, n_zero, no_cpu, og_fills, og_cpu, n_hit, ci == il, 0)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = bank->nb[2];
            for (size_t i = 0; i < og_fills.size();) {
                size_t n = 1;
                while (w->nb[2] == expert_size && i + n < og_fills.size() && og_fills[i + n].expert == og_fills[i].expert + (int32_t) n &&
                        og_fills[i + n].slot == og_fills[i].slot + (int32_t) n) {
                    n++;
                }
                copies.push_back({ (uint64_t) (uintptr_t) ((const uint8_t *) w->data + og_fills[i].expert*w->nb[2]),
                                   (uint64_t) (uintptr_t) ((uint8_t *) bank->data + og_fills[i].slot*expert_size), (uint64_t) (n*expert_size) });
                bytes += n*expert_size;
                i += n;
            }
        }
        stats & st = og_ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += og_fills.size();
        st.bytes  += bytes;
        return og_fills.empty() ? 0 : 1;
    }

    // fork (ONEGRAPH): the slot maps of the canonical layers, row il = layer il (looped visits read their canonical row)
    void og_snapshot(int32_t * table) const {
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                if (canon[il] == il && g.lru.slot_map[il] != nullptr) {
                    memcpy(table + (size_t) il*og_n_expert, g.lru.slot_map[il], (size_t) og_n_expert*sizeof(int32_t));
                }
            }
        }
    }

    ggml_tensor * og_slots(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_used) const {
        if (!og || llama_moe_og::mode() != 1 || is_suspended || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || n_tokens < 1 || n_tokens > 8 ||
                n_tokens*n_used > GGML_MOE_OG_MAX_IDS || ggml_nelements(ids) != n_tokens*n_used) {
            return nullptr;
        }
        const group & g = groups[layers[il].ig];
        return g.n_slots > 0 ? og->slots(ctx, ids, canon[il], il, g.n_slots, (int32_t) n_used) : nullptr;
    }

    // fork (ONEGRAPH): the regular paths change the maps or the banks only with the worker idle and its uploads done
    std::unique_lock<std::mutex> og_hold(bool changes) {
        if (!og) {
            return {};
        }
        og->quiesce();
        if (changes) {
            og->mark_dirty();
        }
        return std::unique_lock<std::mutex>(og->mu);
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // fork (INKMISS): staging and admission state of a group's LRU (after every lru.init)
    void init_stage(group & g) {
        if (!stage_on) {
            return;
        }
        g.lru.n_stage      = n_stage;
        g.lru.stage_base   = g.n_slots + n_zero;
        g.lru.admit_margin = pair_margin;
        g.lru.admit_budget = round_swaps;
        g.lru.declined_budget = false;
        g.lru.declined.clear();
        if (nowait_on) {
            g.lru.busy.assign(g.n_slots, 0);
            g.lru.busy_gen = 0;
        }
    }

    // fork (INKMISS, S2-NOWAIT): the background batch in flight is done (host wait; it was queued a window ago)
    void nowait_drain() {
        if (!nw_pending) {
            return;
        }
        ggml_backend_event_synchronize(nw_event);
        nw_pending = false;
        for (group & g : groups) {
            if (!g.lru.busy.empty() && ++g.lru.busy_gen == 0) {
                std::fill(g.lru.busy.begin(), g.lru.busy.end(), 0);
                g.lru.busy_gen = 1;
            }
        }
    }

    bool nowait_init() {
        if (nw_backend) {
            return true;
        }
        int32_t n_dev = 0;
        for (const device & d : devices) {
            n_dev += d.ctx ? 1 : 0;
        }
        ggml_backend_dev_t dev = n_dev == 1 ? ggml_backend_get_device(devices[groups[0].id].backend) : nullptr;
        ggml_backend_dev_props props;
        if (dev) {
            ggml_backend_dev_get_props(dev, &props);
        }
        if (!dev || !props.caps.events || !props.caps.async) {
            LLAMA_LOG_WARN("%s: [MCROUND] NOWAIT needs one device with events and async copies: off\n", __func__);
            nowait_on = false;
            return false;
        }
        nw_backend = ggml_backend_dev_init(dev, nullptr);
        nw_event   = nw_backend ? ggml_backend_event_new(dev) : nullptr;
        if (!nw_event) {
            LLAMA_LOG_WARN("%s: [MCROUND] NOWAIT: no copy stream or event: off\n", __func__);
            if (nw_backend) {
                ggml_backend_free(nw_backend);
                nw_backend = nullptr;
            }
            nowait_on = false;
            return false;
        }
        return true;
    }

    // fork (INKMISS, R3 + S2-NOWAIT): a new window: reset each group's budget (boosted after a window that ran out) and
    // promote in the background what the last window had to stage for lack of budget. Called right after copy() has
    // synchronized the compute stream, so no queued kernel still reads a victim slot.
    void stage_window() {
        nowait_drain();
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            moe_cache_lru & lru = g.lru;
            const bool backlog = lru.declined_budget;
            lru.admit_budget    = backlog ? round_swaps*round_boost : round_swaps;
            lru.declined_budget = false;
            if (!nowait_on || g.n_slots == 0 || lru.declined.empty() || is_suspended || !nowait_init()) {
                lru.declined.clear();
                continue;
            }
            auto crf_now = [&](int32_t key) { return lru.decayed_crf(lru.key_layer(key), key); };
            std::vector<std::pair<float, int32_t>> rank;
            std::sort(lru.declined.begin(), lru.declined.end());
            lru.declined.erase(std::unique(lru.declined.begin(), lru.declined.end()), lru.declined.end());
            for (int32_t key : lru.declined) {
                if (lru.slot_map[lru.key_layer(key)][lru.key_expert(key)] >= lru.n_slots) {
                    rank.push_back({ crf_now(key), key });
                }
            }
            lru.declined.clear();
            std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            if (++lru.seen_gen == 0) { // nothing is selected now
                std::fill(lru.seen.begin(), lru.seen.end(), 0);
                lru.seen_gen = 1;
            }
            if (++lru.busy_gen == 0) {
                std::fill(lru.busy.begin(), lru.busy.end(), 0);
                lru.busy_gen = 1;
            }
            int32_t n_swap = 0;
            for (size_t i = 0; i < rank.size() && n_swap < round_swaps; ++i) {
                const int32_t key = rank[i].second;
                const int32_t s   = lru.victim(lru.key_layer(key));
                const int32_t old = lru.key_of[s];
                if (old >= 0 && crf_now(old) + pair_margin >= rank[i].first) {
                    break; // the coldest slot is as hot: so is every later candidate's victim
                }
                if (old >= 0) {
                    lru.slot_map[lru.key_layer(old)][lru.key_expert(old)] = lru.empty;
                } else {
                    lru.n_filled++;
                }
                const int32_t il = lru.key_layer(key);
                const int32_t e  = lru.key_expert(key);
                lru.key_of[s]       = key;
                lru.slot_map[il][e] = s;
                lru.move(s, lru.class_of(key));
                lru.busy[s] = lru.busy_gen;
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    ggml_backend_tensor_set_async(nw_backend, bank, (const uint8_t *) w->data + e*w->nb[2], s*bank->nb[2], bank->nb[2]);
                    stats_small.bytes += bank->nb[2];
                }
                n_swap++;
            }
            if (n_swap > 0) {
                ggml_backend_event_record(nw_event, nw_backend);
                nw_pending = true;
                nw_n_batches++;
                nw_n_swaps += n_swap;
            }
        }
    }

    // fork (INKMISS, S2-SWAP-POLICY seeding): profile = "MCS1", u32 n_layer, u32 n_records, then (i32 layer, i32 expert,
    // f32 decayed use count) per record
    void seed_write(const std::string & path) const {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            LLAMA_LOG_WARN("%s: [MCSEED] cannot write %s\n", __func__, path.c_str());
            return;
        }
        struct rec { int32_t il, e; float v; };
        std::vector<rec> recs;
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                if (canon[il] != il) {
                    continue;
                }
                for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                    const int32_t key = g.lru.make_key(il, e);
                    if (g.lru.last_use[key] >= 0) {
                        const float v = g.lru.decayed_crf(il, key);
                        if (v > 0.0f) {
                            recs.push_back({ il, e, v });
                        }
                    }
                }
            }
        }
        const uint32_t hdr[2] = { (uint32_t) layers.size(), (uint32_t) recs.size() };
        fwrite("MCS1", 1, 4, f);
        fwrite(hdr, sizeof(uint32_t), 2, f);
        for (const rec & r : recs) {
            fwrite(&r.il, sizeof(int32_t), 1, f);
            fwrite(&r.e,  sizeof(int32_t), 1, f);
            fwrite(&r.v,  sizeof(float),   1, f);
        }
        fclose(f);
        LLAMA_LOG_INFO("%s: [MCSEED] %zu expert use counts written to %s\n", __func__, recs.size(), path.c_str());
    }

    void seed_load(const char * path) {
        FILE * f = fopen(path, "rb");
        if (!f) {
            LLAMA_LOG_WARN("%s: [MCSEED] no profile at %s, the cache starts empty\n", __func__, path);
            return;
        }
        char magic[4] = {};
        uint32_t hdr[2] = {};
        bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "MCS1", 4) == 0 && fread(hdr, sizeof(uint32_t), 2, f) == 2 &&
                hdr[0] == (uint32_t) layers.size();
        size_t n_seeded = 0;
        for (uint32_t i = 0; ok && i < hdr[1]; ++i) {
            int32_t il = -1, e = -1;
            float v = 0.0f;
            ok = fread(&il, sizeof(il), 1, f) == 1 && fread(&e, sizeof(e), 1, f) == 1 && fread(&v, sizeof(v), 1, f) == 1;
            if (!ok || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || canon[il] != il || !(v > 0.0f)) {
                continue;
            }
            group & g = groups[layers[il].ig];
            if (g.n_slots == 0 || e < 0 || e >= g.lru.n_expert) {
                continue;
            }
            const int32_t key = g.lru.make_key(il, e);
            g.lru.crf[key]      = v;
            g.lru.last_use[key] = 0;
        }
        fclose(f);
        if (!ok) {
            LLAMA_LOG_WARN("%s: [MCSEED] %s is not a profile of this model, ignored\n", __func__, path);
            for (group & g : groups) {
                std::fill(g.lru.crf.begin(), g.lru.crf.end(), 0.0f);
                std::fill(g.lru.last_use.begin(), g.lru.last_use.end(), -1);
            }
            return;
        }
        // the hottest keys of each group fill its slots, in key order (consecutive experts in consecutive slots)
        for (group & g : groups) {
            moe_cache_lru & lru = g.lru;
            if (g.n_slots == 0 || lru.n_filled > 0) {
                continue;
            }
            std::vector<std::pair<float, int32_t>> rank;
            for (int32_t il : g.layers) {
                for (int32_t e = 0; canon[il] == il && e < lru.n_expert; ++e) {
                    const int32_t key = lru.make_key(il, e);
                    if (lru.last_use[key] == 0) {
                        rank.push_back({ lru.crf[key], key });
                    }
                }
            }
            std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            rank.resize(std::min(rank.size(), (size_t) g.n_slots));
            std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
            for (size_t k = 0; k < rank.size(); ++k) {
                const int32_t key = rank[k].second;
                const int32_t s   = (int32_t) k;
                lru.crf[key]  = std::max(1.0f, lru.crf[key]); // a cached key has class >= 0
                lru.key_of[s] = key;
                lru.slot_map[lru.key_layer(key)][lru.key_expert(key)] = s;
                lru.move(s, lru.class_of(key));
                lru.n_filled++;
                const layer & l = layers[lru.key_layer(key)];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    ggml_backend_tensor_set(bank, (const uint8_t *) w->data + lru.key_expert(key)*w->nb[2], s*bank->nb[2], bank->nb[2]);
                }
            }
            n_seeded += rank.size();
        }
        LLAMA_LOG_WARN("%s: [MCSEED] %zu experts seeded from %s (LLAMA_MOE_CACHE_SEED)\n", __func__, n_seeded, path);
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        if (prefetch) {
            prefetch->wait();
            LLAMA_LOG_WARN("llama_moe_cache: [NEXTPF] %zu experts uploaded ahead in %zu rounds (%.2f MiB), %zu of them used (%.1f%%)\n",
                    prefetch->n_prefetched, prefetch->n_rounds, prefetch->n_bytes/1024.0/1024.0, prefetch->n_used,
                    prefetch->n_prefetched ? 100.0*prefetch->n_used/prefetch->n_prefetched : 0.0);
            prefetch.reset();
        }
        if (!seed_save.empty()) {
            seed_write(seed_save);
        }
        if (nw_backend) {
            nowait_drain();
            ggml_backend_synchronize(nw_backend);
            ggml_backend_event_free(nw_event);
            ggml_backend_free(nw_backend);
        }
        log_stats();
        split.log_summary();
        if (heat.on()) {
            heat.save();
        }
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (is_suspended || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        if (fewrow_r > 0 && cpu_share == 0.0f) {
            // few-row tier only: no CPU part below the prompt-chunk size, and with op offload off the scheduler keeps
            // the CPU part on the CPU whatever the batch size
            if (n_tokens < fewrow_min) {
                return nullptr;
            }
            if (!fewrow_op_offload) {
                return layers[il].cpu_map;
            }
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        if (is_suspended) {
            return nullptr;
        }
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        if (is_suspended) {
            return 0;
        }
        nowait_drain(); // fork (INKMISS): a large batch reads the banks without the lookup's event wait
        if (prefetch) {
            prefetch->wait(); // the banks it reads may be in flight
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }
        std::unique_lock<std::mutex> og_lock;
        if (og) {
            og_lock = std::unique_lock<std::mutex>(og->mu); // fork (ONEGRAPH): reads the slot maps the worker writes
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const int32_t ci      = canon[il];
        const bool    has_cpu = it->second.second;
        if (prefetch) {
            prefetch->wait(); // uploads ahead write the banks this lookup plans over
        }
        if (nvme != nullptr) {
            nvme->epoch(); // the uploads of earlier lookups have left the host buffers (pageable copies)
        }
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_moe_cache_slots

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);
        if (split.active()) {
            split.on_ids(il, sel->src[0]->ne[1]); // fork (INKCACHEUX) #249: layer interval, ids to ids
        }
        const std::unique_lock<std::mutex> og_lock = og_hold(true); // fork (ONEGRAPH): no-op unless on

        if (trace != nullptr) {
            write_trace(il, n_used);
        }
        if (ramlock) {
            ramlock->observe(ci, ids.data(), ids.size());
        }
        const bool pf_decode = prefetch && sel->src[0]->ne[1] <= 8;
        if (pf_decode) {
            prefetch->account(ci, ids.data(), ids.size());
            prefetch->record(ci, ids.data(), ids.size());
        }

        // fork (SURGE-cache): a window starts at a lookup of a layer at or below the previous one; rank the last one first
        if (round_on) {
            if (il <= round_prev_il) {
                run_rounds();
            }
            round_prev_il = il;
        }
        // fork (INKMISS, R3): same windows for staged admission; only decode ubatches (<= 8 rows) are admission-controlled
        const bool staged = stage_on && sel->src[0]->ne[1] <= 8;
        if (stage_on && il == stage_first_il) {
            stage_window(); // the first cached layer in graph order (looped models revisit lower layers mid-token)
        }
        const int32_t budget_keep = g.lru.admit_budget;
        if (stage_on && !staged) {
            g.lru.admit_budget = -1;
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        // (round admission: all of them, as far as zero slots allow; MCDET: none)
        // (round admission: all decode misses, ubatches of <= 8 rows; larger ones upload as before)
        const uint32_t share_full = round_on ? (sel->src[0]->ne[1] <= 8 ? 65536u : 0u) : det_on ? 0u : cpu_share_q;
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? share_full : 0;
        const bool split_on = split.active() && share_q > 0 && !round_on; // fork (INKCACHEUX) #249/GS9
        auto n_cpu = [&](size_t n_miss) {
            if (split_on) {
                return split.n_cpu(n_miss, l.cpu_acc);
            }
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        const int32_t few_rows = has_cpu && fewrow_r > 0 && sel->src[0]->ne[1] >= fewrow_min &&
                g.lru.n_filled == g.lru.n_slots ? fewrow_r : 0;
        if (!g.lru.plan(ci, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit, ci == il, few_rows)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (stage_on) {
            if (staged) {
                for (const auto & f : fills) {
                    const bool st = f.slot >= g.lru.stage_base;
                    stage_n_staged   += st ? 1 : 0;
                    stage_n_admitted += st ? 0 : 1;
                }
            } else {
                g.lru.admit_budget = budget_keep;
            }
            // fork (INKMISS, S2-NOWAIT): a selected expert whose slot the copy stream is still writing: the compute stream
            // waits for the batch (on the device)
            if (nw_pending) {
                const int32_t * slots = g.lru.slot_map[ci];
                for (int32_t e : g.lru.uniq) {
                    const int32_t sl = slots[e];
                    if (sl < g.lru.n_slots && g.lru.is_busy(sl)) {
                        ggml_backend_event_wait(backend, nw_event);
                        nw_n_waits++;
                        break;
                    }
                }
            }
        }
        if (split_on) {
            split.record(cpu.size(), fills.size());
        }
        if (heat.on() && sel->src[0]->ne[1] <= 8) {
            heat.observe(ci, ids.data(), ids.size()); // fork (INKCACHEUX) GS4: decode rows only
            heat.add_rows(ci, sel->src[0]->ne[1]);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
                if (round_on && sel->src[0]->ne[1] <= 8) { // decode misses only (a prompt tier may share the CPU part)
                    const int32_t key = g.lru.make_key(ci, f.expert);
                    if (round_mark[key] != round_gen) {
                        round_mark[key] = round_gen;
                        round_cand[l.ig].push_back(key);
                    }
                }
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        if (nvme != nullptr && !fills.empty()) {
            // fork (NVMETIER): read every missing expert of the layer into the RAM tier at once (parallel reads)
            std::vector<std::pair<const void *, size_t>> want;
            for (const ggml_tensor * w : l.experts) {
                for (const auto & f : fills) {
                    want.emplace_back((const uint8_t *) w->data + f.expert*w->nb[2], (size_t) w->nb[2]);
                }
            }
            nvme->ensure_many(want);
        }
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = bank->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (w->nb[2] == expert_size && i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                const uint8_t * src = nvme ? host_src(w, fills[i].expert*w->nb[2], n*expert_size) : (const uint8_t *) w->data + fills[i].expert*w->nb[2];
                ggml_backend_tensor_set_async(backend, bank, src, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, g.lru.slot_map[ci], 0, ggml_nbytes(l.slot_map));

        if (pf_decode) {
            prefetch_next(il, ci, g);
        }

        return true;
    }

    // fork (R4, NEXTPF): see llama-moe-prefetch.h. Runs after the plan of layer il (seen = its selection), so victims
    // skip its selected experts; the uploads go to slots that nothing on the device reads before the next lookup.
    void prefetch_next(int32_t il, int32_t ci, group & g) {
        const int32_t nl = il + 1;
        if (nl >= (int32_t) layers.size() || layers[nl].ig != layers[il].ig) {
            return;
        }
        const int32_t cn = canon[nl];
        int32_t * slots = g.lru.slot_map[cn];
        const std::vector<int32_t> & pred = prefetch->predicted(cn);
        if (cn == ci || slots == nullptr || pred.empty()) {
            return;
        }
        if (++pf_gen == 0) {
            std::fill(pf_pred.begin(), pf_pred.end(), 0);
            pf_gen = 1;
        }
        for (int32_t e : pred) {
            pf_pred[e] = pf_gen;
        }
        moe_cache_lru & lru = g.lru;
        auto keep = [&](int32_t s) {
            const int32_t key = lru.key_of[s];
            return key >= 0 && lru.key_layer(key) == cn && pf_pred[lru.key_expert(key)] == pf_gen;
        };
        pf_fills.clear();
        pf_experts.clear();
        for (int32_t e : pred) {
            if (slots[e] < g.n_slots) {
                continue; // cached (a slot >= n_slots is empty or a zero slot of a CPU miss)
            }
            if ((int32_t) pf_fills.size() >= prefetch->budget()) {
                break;
            }
            const int32_t s = lru.victim_keep(ci, keep);
            if (s < 0) {
                break;
            }
            if (lru.key_of[s] >= 0) {
                lru.slot_map[lru.key_layer(lru.key_of[s])][lru.key_expert(lru.key_of[s])] = lru.empty;
            } else {
                lru.n_filled++;
            }
            const int32_t key = lru.make_key(cn, e);
            lru.key_of[s] = key;
            slots[e]      = s;
            lru.move(s, lru.crf[key] >= 1.0f ? lru.class_of(key) : 0);
            pf_fills.push_back({ e, s });
            pf_experts.push_back(e);
        }
        if (pf_fills.empty()) {
            return;
        }
        std::vector<llama_moe_prefetch::job> jobs;
        const layer & ln = layers[nl];
        for (size_t ip = 0; ip < ln.experts.size(); ++ip) {
            const ggml_tensor * w    = ln.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = bank->nb[2];
            for (size_t i = 0; i < pf_fills.size();) {
                size_t n = 1;
                while (w->nb[2] == expert_size && i + n < pf_fills.size() && pf_fills[i + n].expert == pf_fills[i].expert + (int32_t) n &&
                        pf_fills[i + n].slot == pf_fills[i].slot + (int32_t) n) {
                    n++;
                }
                jobs.push_back({ bank, (const uint8_t *) w->data + pf_fills[i].expert*w->nb[2], pf_fills[i].slot*expert_size, n*expert_size });
                i += n;
            }
        }
        prefetch->n_prefetched += pf_fills.size();
        prefetch->mark(cn, pf_experts);
        prefetch->submit(std::move(jobs));
    }

    // fork (SURGE-cache, round admission): see round_on. Runs at the first lookup of a window, when the device has
    // finished the previous window (copy() synchronizes before it reads the ids), so the banks can be written.
    void run_rounds() {
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            std::vector<int32_t> & cand = round_cand[ig];
            if (g.n_slots == 0 || cand.empty()) {
                round_backlog[ig] = 0;
                cand.clear();
                continue;
            }
            moe_cache_lru & lru = g.lru;
            auto crf_now = [&](int32_t key) { return lru.decayed_crf(lru.key_layer(key), key); };
            std::vector<std::pair<float, int32_t>> rank;
            rank.reserve(cand.size());
            for (int32_t key : cand) {
                if (lru.slot_map[lru.key_layer(key)][lru.key_expert(key)] >= lru.n_slots) {
                    rank.push_back({ crf_now(key), key });
                }
            }
            cand.clear();
            // hottest first; ties by key, so the order does not depend on the order of the lookups
            std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            const int32_t budget = round_backlog[ig] ? round_swaps*round_boost : round_swaps;
            // nothing is selected during a round: a fresh generation clears is_selected() for every slot
            if (++lru.seen_gen == 0) {
                std::fill(lru.seen.begin(), lru.seen.end(), 0);
                lru.seen_gen = 1;
            }
            ggml_backend_t backend = devices[g.id].backend;
            int32_t n_swap = 0;
            size_t  i = 0;
            for (; i < rank.size() && n_swap < budget; ++i) {
                const int32_t key = rank[i].second;
                const int32_t s   = lru.victim(lru.key_layer(key));
                const int32_t old = lru.key_of[s];
                if (old >= 0 && crf_now(old) + pair_margin >= rank[i].first) {
                    break; // the coldest cached expert is at least as hot: so is every later candidate's victim
                }
                if (old >= 0) {
                    lru.slot_map[lru.key_layer(old)][lru.key_expert(old)] = lru.empty;
                } else {
                    lru.n_filled++;
                }
                const int32_t il = lru.key_layer(key);
                const int32_t e  = lru.key_expert(key);
                lru.key_of[s]       = key;
                lru.slot_map[il][e] = s;
                lru.move(s, lru.class_of(key));
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    const uint8_t * src = nvme ? host_src(w, e*w->nb[2], bank->nb[2]) : (const uint8_t *) w->data + e*w->nb[2];
                    ggml_backend_tensor_set_async(backend, bank, src, s*bank->nb[2], bank->nb[2]);
                    stats_small.bytes += bank->nb[2];
                }
                n_swap++;
            }
            // a backlog: the budget ran out while the next candidate would still have beaten the coldest slot
            bool backlog = false;
            if (n_swap == budget && i < rank.size()) {
                const int32_t s   = lru.victim(lru.key_layer(rank[i].second));
                const int32_t old = lru.key_of[s];
                backlog = old < 0 || crf_now(old) < rank[i].first;
            }
            round_backlog[ig] = backlog ? 1 : 0;
            round_n++;
            round_n_swaps   += n_swap;
            round_n_backlog += backlog ? 1 : 0;
        }
        if (++round_gen == 0) {
            std::fill(round_mark.begin(), round_mark.end(), 0);
            round_gen = 1;
        }
    }

    // fork (PLOAN): the VMM buffer type of the one device that holds banks and its page calls; else PLOAN stays off
    void ploan_init() {
        const device * one = nullptr;
        int n_dev = 0;
        for (const device & d : devices) {
            if (d.ctx) {
                one = &d;
                n_dev++;
            }
        }
        ggml_backend_dev_t dev = one ? ggml_backend_get_device(one->backend) : nullptr;
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto get_buft = reg ? (elastic_buft_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_elastic_buffer_type") : nullptr;
        el_set    = reg ? (elastic_set_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_elastic_set_ranges") : nullptr;
        el_mapped = reg ? (elastic_mapped_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_elastic_mapped") : nullptr;
        buft_elastic = n_dev == 1 && get_buft && el_set && el_mapped ? get_buft(dev) : nullptr;
        if (buft_elastic) {
            LLAMA_LOG_WARN("%s: [PLOAN] on: MoE cache banks in a VMM buffer; a streamed prompt keeps a mapped prefix of each bank\n", __func__);
        } else {
            LLAMA_LOG_WARN("%s: [PLOAN] off: needs one CUDA device with VMM for the banks (%d devices hold banks)\n", __func__, n_dev);
        }
    }

    // fork (PLOAN): the banks of d in a VMM buffer; false (logged once): the caller takes the plain buffer type
    bool alloc_elastic(device & d) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), buft_elastic);
        d.elastic = buf != nullptr;
        if (!buf) {
            if (!elastic_warned) {
                elastic_warned = true;
                LLAMA_LOG_WARN("%s: [PLOAN] the VMM banks did not allocate: plain banks (prompts lend the whole cache)\n", __func__);
            }
            return false;
        }
        d.buf.reset(buf);
        return true;
    }

    // fork (PLOAN): bytes of one slot of a group (all its banks)
    static size_t slot_bytes(const group & g) {
        size_t res = 0;
        for (const ggml_tensor * t : g.ref) {
            res += t->nb[1]*t->ne[1];
        }
        return res;
    }

    size_t suspend_keep(size_t keep_bytes) {
        if (is_suspended) {
            return 0;
        }
        device * one = nullptr;
        int n_dev = 0;
        for (device & d : devices) {
            if (d.ctx) {
                one = &d;
                n_dev++;
            }
        }
        if (keep_bytes == 0 || n_dev != 1 || !one->buf || !one->elastic || !el_set || !el_mapped) {
            return suspend();
        }
        nowait_drain(); // no background copy into the banks while the streamer reads them
        if (prefetch) {
            prefetch->wait();
        }
        // each group keeps the same share of its slots, about keep_bytes in all
        size_t all = 0;
        for (const group & g : groups) {
            all += (size_t) g.n_slots*slot_bytes(g);
        }
        const double f = all > 0 ? std::min(1.0, (double) keep_bytes/all) : 0.0;
        keep_n.assign(groups.size(), 0);
        std::vector<size_t> off, len;
        const char * base = (const char *) ggml_backend_buffer_get_base(one->buf.get());
        int32_t n_kept = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            keep_n[ig] = std::min(g.n_slots, (int32_t) (f*g.n_slots));
            n_kept += keep_n[ig];
            for (const ggml_tensor * bank : g.banks) {
                off.push_back((size_t) ((const char *) bank->data - base));
                len.push_back((size_t) keep_n[ig]*bank->nb[2]);
            }
        }
        const size_t before = el_mapped(one->buf.get());
        if (n_kept == 0 || !el_set(one->buf.get(), off.data(), len.data(), (int) off.size())) {
            keep_n.clear();
            return suspend();
        }
        // the banks keep their pointers and the kept slots their experts; readers see the cache suspended
        is_suspended = true;
        is_partial   = true;
        return before - el_mapped(one->buf.get());
    }

    size_t kept_bytes() const {
        size_t res = 0;
        for (const device & d : devices) {
            res += is_partial && d.buf && d.elastic ? el_mapped(d.buf.get()) : 0;
        }
        return res;
    }

    const int32_t * kept_slots(const ggml_tensor * w, ggml_tensor ** bank, int32_t * n_keep) const {
        if (!is_partial) {
            return nullptr;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const binding & b = it->second;
        const int32_t ig = layers[b.il].ig;
        if (ig < 0 || ig >= (int32_t) keep_n.size() || keep_n[ig] <= 0 || groups[ig].lru.slot_map[b.il] == nullptr) {
            return nullptr;
        }
        *bank   = groups[ig].banks[b.ip];
        *n_keep = keep_n[ig];
        return groups[ig].lru.slot_map[b.il];
    }

    size_t suspend() {
        if (ploan_on && is_suspended && !is_partial) { // port: PLOAN only, off = the cand base's path
            return 0;
        }
        nowait_drain(); // fork (INKMISS): no background copy into banks about to be freed
        if (prefetch) {
            prefetch->wait();
        }
        const std::unique_lock<std::mutex> og_lock = og_hold(true); // fork (ONEGRAPH): no-op unless on
        size_t freed = 0;
        for (device & d : devices) {
            if (!d.buf || !d.ctx) {
                continue;
            }
            // fork (PLOAN): a partial loan frees the pages it kept mapped
            freed += is_partial && d.elastic ? el_mapped(d.buf.get()) : ggml_backend_buffer_get_size(d.buf.get());
            d.buf.reset();
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                t->data   = nullptr;
                t->buffer = nullptr;
            }
        }
        is_suspended = true;
        is_partial   = false;
        return freed;
    }

    // fork (CACHELOAN): bytes of the banks of a group with n slots
    size_t group_bytes(const group & g, int32_t n) const {
        const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
        size_t res = 0;
        for (const ggml_tensor * t : g.ref) {
            res += GGML_PAD((t->nb[1]*t->ne[1])*(n + n_zero + n_stage + 1), alignment);
        }
        return res;
    }

    size_t bank_bytes() const {
        size_t res = 0;
        for (const group & g : groups) {
            res += g.n_slots > 0 ? group_bytes(g, g.n_slots) : 0;
        }
        return res;
    }

    // fork (CACHELOAN): while suspended, give each group the share of `size` it has now and keep its experts with the
    // lowest eviction score (the ones victim() would take last). The banks are rebuilt, so the views change: the graphs
    // that read them must be rebuilt. One device, no CPU share (zero slots), no trace: otherwise false and nothing changes.
    bool resize(size_t size) {
        // fork (SURGE-cache): zero slots (CPU misses) are fine under round admission: the banks are cleared on resume
        // the few-row tier keeps its zero slots through a resize: resume() clears the new banks, so they read zero again
        if (!is_suspended || (n_zero > 0 && !round_on && !(fewrow_r > 0 && cpu_share == 0.0f)) || trace != nullptr || groups.empty()) {
            return false;
        }
        nowait_drain();
        const std::unique_lock<std::mutex> og_lock = og_hold(true); // fork (ONEGRAPH): no-op unless on
        int32_t n_dev = 0;
        for (const device & d : devices) {
            n_dev += d.ctx ? 1 : 0;
        }
        if (n_dev != 1) {
            return false;
        }
        const size_t cur = bank_bytes();
        std::vector<int32_t> want(groups.size());
        bool change = false;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.n_slots == 0) {
                return false;
            }
            const size_t  budget    = (size_t) ((double) size*group_bytes(g, g.n_slots)/cur);
            const int32_t max_slots = g.n_canon*(int32_t) g.ref[0]->ne[2];
            int32_t n = n_expert_used;
            if (group_bytes(g, n) > budget) {
                return false;
            }
            while (n < max_slots && group_bytes(g, n + 1) <= budget) {
                n++;
            }
            want[ig] = n;
            change = change || n != g.n_slots;
        }
        if (!change) {
            return true; // fork (PLOAN): a partial loan stays, resume() refills only its dropped slots
        }
        if (is_partial) {
            suspend(); // fork (PLOAN): the slots move, so the kept pages go too
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            moe_cache_lru & lru = g.lru;
            // the filled slots in list order (classes low to high, each least recently used first) and their scores
            std::vector<int32_t> order;
            std::vector<std::pair<double, int32_t>> score;
            for (int c = 0; c < moe_cache_lru::n_class; ++c) {
                for (int32_t s = lru.head[c]; s >= 0; s = lru.next[s]) {
                    const int32_t key = lru.key_of[s];
                    if (key >= 0) {
                        order.push_back(key);
                        const int64_t age = lru.clock[lru.key_layer(key)] - lru.last_use[key];
                        score.push_back({ std::ldexp((double) age, -c), key });
                    }
                }
            }
            const size_t n_keep = std::min(order.size(), (size_t) want[ig]);
            std::sort(score.begin(), score.end());
            std::vector<int32_t> keep(n_keep);
            for (size_t k = 0; k < n_keep; ++k) {
                keep[k] = score[k].second;
            }
            std::sort(keep.begin(), keep.end()); // consecutive experts of a layer in consecutive slots: one upload

            std::vector<int32_t *> maps     = lru.slot_map;
            std::vector<float>     crf      = std::move(lru.crf);
            std::vector<int64_t>   last_use = std::move(lru.last_use);
            std::vector<int64_t>   clock    = std::move(lru.clock);
            lru.init((int32_t) maps.size(), lru.n_expert, want[ig]);
            lru.slot_map = maps;
            lru.crf      = std::move(crf);
            lru.last_use = std::move(last_use);
            lru.clock    = std::move(clock);
            for (int32_t il : g.layers) {
                int32_t * slots = (int32_t *) layers[il].slot_map->data;
                std::fill(slots, slots + layers[il].slot_map->ne[1], lru.empty);
            }
            for (size_t k = 0; k < n_keep; ++k) {
                lru.key_of[k] = keep[k];
                lru.slot_map[lru.key_layer(keep[k])][lru.key_expert(keep[k])] = (int32_t) k;
            }
            lru.n_filled = (int32_t) n_keep;
            for (int32_t key : order) {
                const int32_t s = lru.slot_map[lru.key_layer(key)][lru.key_expert(key)];
                if (s < lru.n_slots) {
                    lru.move(s, lru.class_of(key)); // same order within each class as before
                }
            }
            g.n_slots = want[ig];
            init_stage(g); // fork (INKMISS): the staging slots follow the new slot count
            if (round_on) {
                round_cand[ig].clear(); // the candidates' slot state was rebuilt
                round_backlog[ig] = 0;
                // cpu_experts stay: the next lookup of each layer resets their cpu_map entries before it sets new ones
            }
        }

        // new banks and views on the device
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!d.ctx) {
                continue;
            }
            size_t n_tensors = 0;
            for (const group & g : groups) {
                n_tensors += g.id == (int32_t) id ? g.ref.size()*(1 + g.layers.size()) : 0;
            }
            ggml_init_params params = { n_tensors*ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            d.buf_size = 0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.id != (int32_t) id) {
                    continue;
                }
                g.banks.clear();
                for (const ggml_tensor * t : g.ref) {
                    ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + n_stage + 1);
                    GGML_ASSERT(bank->nb[2] == t->nb[1] * t->ne[1]);
                    ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                    g.banks.push_back(bank);
                }
                for (int32_t il : g.layers) {
                    const layer & l = layers[il];
                    for (size_t ip = 0; ip < l.experts.size() && canon[il] == il; ++ip) {
                        ggml_tensor * bank   = g.banks[ip];
                        ggml_tensor * cached = ggml_view_3d(ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero + n_stage, bank->nb[1], bank->nb[2], 0);
                        ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                        bindings[l.experts[ip]].cached = cached;
                    }
                }
                d.buf_size += group_bytes(g, g.n_slots);
            }
            d.ctx = std::move(ctx);
        }
        return true;
    }

    bool resume(const llama_moe_cache::uploader & up) {
        if (!is_suspended) {
            return true;
        }
        const std::unique_lock<std::mutex> og_lock = og_hold(true); // fork (ONEGRAPH): no-op unless on
        // fork (PLOAN): a partial loan maps its dropped pages again (they read the clear value, zero) and uploads only
        // the slots from keep_n on; a failed map leaves it as it was (still suspended)
        std::vector<int32_t> first(groups.size(), 0);
        if (is_partial) {
            for (device & d : devices) {
                if (!d.buf || !d.ctx || !d.elastic) {
                    continue;
                }
                const size_t off = 0, len = ggml_backend_buffer_get_size(d.buf.get());
                if (!el_set(d.buf.get(), &off, &len, 1)) {
                    LLAMA_LOG_WARN("%s: [PLOAN] mapping the lent bank pages again failed (retried later)\n", __func__);
                    return false;
                }
            }
            first = keep_n;
        }
        for (device & d : devices) {
            if (d.buf || !d.ctx) {
                continue;
            }
            ggml_backend_buffer_t buf = buft_elastic && alloc_elastic(d) ? d.buf.release() :
                ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft);
            if (!buf) {
                // free what was allocated so far, the context stays suspended
                for (device & o : devices) {
                    if (o.buf && o.ctx) {
                        o.buf.reset();
                        for (ggml_tensor * t = ggml_get_first_tensor(o.ctx.get()); t; t = ggml_get_next_tensor(o.ctx.get(), t)) {
                            t->data   = nullptr;
                            t->buffer = nullptr;
                        }
                    }
                }
                return false;
            }
            d.buf.reset(buf);
            d.buf_size = ggml_backend_buffer_get_size(buf);
            ggml_backend_buffer_clear(buf, 0);
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        if (up) {
            // the clear above is a memset on the legacy default stream, which the uploader's non-blocking copy stream
            // does not wait for: finish it first (a read on the per-thread stream orders after it), or it can zero
            // experts already uploaded. The synchronous copies below are ordered after it anyway.
            for (device & d : devices) {
                ggml_tensor * t = d.buf && d.ctx ? ggml_get_first_tensor(d.ctx.get()) : nullptr;
                if (t && ggml_nbytes(t) > 0) {
                    uint8_t b;
                    ggml_backend_tensor_get(t, &b, 0, 1);
                }
            }
        }
        // the experts the slots held, consecutive slots of consecutive experts of one layer in one upload; with an
        // uploader (REFILLPIPE) the runs go to it as one list, else (or when it declines) one synchronous copy each
        std::vector<llama_moe_cache::copy_op> ops;
        bool use_up = (bool) up;
        auto put = [&](ggml_tensor * bank, const ggml_tensor * w, const void * src, size_t offs, size_t len) {
            if (use_up && w->buffer && ggml_backend_buffer_is_host(w->buffer)) {
                ops.push_back({ bank, offs, src, len });
            } else {
                use_up = false;
                ggml_backend_tensor_set(bank, src, offs, len);
            }
        };
        if (ploan_on) {
            res_n_up = res_n_all = 0;
            res_bytes = 0;
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            const moe_cache_lru & lru = g.lru;
            for (int32_t s = first[ig]; s < g.n_slots; ) {
                const int32_t key = lru.key_of[s];
                if (key < 0) {
                    s++;
                    continue;
                }
                const int32_t il = lru.key_layer(key);
                const int32_t e  = lru.key_expert(key);
                int32_t n = 1;
                while (s + n < g.n_slots && lru.key_of[s + n] == key + n && lru.key_layer(key + n) == il) {
                    n++;
                }
                if (ploan_on) {
                    res_n_up  += n;
                    res_bytes += (size_t) n*slot_bytes(g);
                }
                const layer & l = layers[il];
                if (nvme != nullptr) {
                    // fork (NVMETIER): queued, read into the RAM tier in parallel batches below
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        for (int32_t k = 0; k < n; ++k) {
                            nvme_refill.push_back({ l.experts[ip], g.banks[ip], e + k, s + k });
                            nvme_refill_bytes += g.banks[ip]->nb[2];
                        }
                    }
                    if (nvme_refill_bytes >= nvme_refill_batch) {
                        nvme_refill_flush();
                    }
                    s += n;
                    continue;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    const size_t expert_size = bank->nb[2];
                    if (w->nb[2] == expert_size) {
                        put(bank, w, (const uint8_t *) w->data + e*w->nb[2], s*expert_size, n*expert_size);
                    } else {
                        for (int32_t k = 0; k < n; ++k) {
                            put(bank, w, (const uint8_t *) w->data + (e + k)*w->nb[2], (s + k)*expert_size, expert_size);
                        }
                    }
                }
                s += n;
            }
        }
        if (nvme != nullptr) {
            nvme_refill_flush();
        }
        if (!ops.empty() && !(use_up && up(ops))) {
            for (const auto & op : ops) {
                ggml_backend_tensor_set(op.dst, op.src, op.offs, op.len);
            }
        }
        if (ploan_on) {
            for (const group & g : groups) {
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    res_n_all += g.lru.key_of[s] >= 0 ? 1 : 0;
                }
            }
        }
        is_suspended = false;
        is_partial   = false;
        return true;
    }

    // fork (NVMETIER): resume() uploads from the RAM tier, in batches of up to 256 MiB read in parallel (synchronous
    // uploads, so a batch may leave the tier once it is on the device)
    struct nvme_upload { const ggml_tensor * w; ggml_tensor * bank; int32_t e; int32_t s; };
    std::vector<nvme_upload> nvme_refill;
    size_t nvme_refill_bytes = 0;
    static constexpr size_t nvme_refill_batch = 256u << 20;
    void nvme_refill_flush() {
        if (nvme_refill.empty()) {
            return;
        }
        nvme->epoch();
        std::vector<std::pair<const void *, size_t>> want;
        want.reserve(nvme_refill.size());
        for (const auto & u : nvme_refill) {
            want.emplace_back((const uint8_t *) u.w->data + u.e*u.w->nb[2], (size_t) u.bank->nb[2]);
        }
        nvme->ensure_many(want);
        for (const auto & u : nvme_refill) {
            const size_t expert_size = u.bank->nb[2];
            ggml_backend_tensor_set(u.bank, host_src(u.w, u.e*u.w->nb[2], expert_size), u.s*expert_size, expert_size);
        }
        nvme_refill.clear();
        nvme_refill_bytes = 0;
        nvme->epoch();
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stage_on) {
            LLAMA_LOG_INFO("llama_moe_cache: [MCROUND] staged admission: %zu decode misses admitted, %zu staged; NOWAIT %zu batches, "
                "%zu background swaps, %zu device waits\n", stage_n_admitted, stage_n_staged, nw_n_batches, nw_n_swaps, nw_n_waits);
        }
        if (round_n > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: [MCROUND] %zu rounds, %zu swaps (%.2f per round), %zu rounds under a backlog\n",
                round_n, round_n_swaps, (double) round_n_swaps/round_n, round_n_backlog);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (nvme != nullptr) {
            nvme->print_stats("MoE cache");
        }
    }

    // fork (INKCACHEUX): end of the constructor. GS1 log, #249/GS9 policy, GS4 profile load and warm slots.
    void ux_init(const llama_model & model, bool no_alloc) {
        if (ux.any() || ux.req_stats || ux.boundary) {
            std::string sel;
            for (size_t il = 0; il < ux.layers.size(); ++il) {
                sel += ux.layers[il] ? '1' : '0';
            }
            LLAMA_LOG_WARN("%s: [CACHEUX] controls: slots %d, per-GPU MiB %s, layers %s, request stats %d, boundary %d\n", __func__,
                    ux.slots, ux.dev_mib.empty() ? "off" : "list", sel.empty() ? "all" : sel.c_str(), ux.req_stats, ux.boundary);
        }
        split.init(cpu_share);
        if (!getenv("LLAMA_MOE_CACHE_HEAT") || groups.empty()) {
            return;
        }
        llama_moe_cache_fp fp;
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                if (g.n_slots > 0 && canon[il] == il) {
                    for (const ggml_tensor * t : layers[il].experts) {
                        fp.add_tensor(il, t);
                    }
                }
            }
        }
        heat.init((int32_t) layers.size(), (int32_t) model.hparams.n_expert, fp.h);
        if (no_alloc || !heat.load()) {
            return;
        }
        const std::vector<float> rate = heat.rates();
        // a key used with probability p per decode row settles at a CRF of about p/(1 - 2^(-1/half-life))
        const float crf_per_rate = 1.0f/(1.0f - std::exp2(-1.0f/moe_cache_lru::crf_half_life));
        size_t n_warm = 0, bytes = 0;
        for (group & g : groups) {
            moe_cache_lru & lru = g.lru;
            if (g.n_slots == 0 || lru.n_expert != (int32_t) model.hparams.n_expert) {
                continue;
            }
            std::vector<std::pair<float, int32_t>> rank;
            for (int32_t il : g.layers) {
                if (canon[il] != il || lru.slot_map[il] == nullptr) {
                    continue;
                }
                for (int32_t e = 0; e < lru.n_expert; ++e) {
                    const int32_t key = lru.make_key(il, e);
                    if (rate[key] > 0.0f) {
                        lru.crf[key]      = std::max(1.0f, rate[key]*crf_per_rate);
                        lru.last_use[key] = 0;
                        rank.push_back({ rate[key], key });
                    }
                }
            }
            std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            std::vector<int32_t> keep;
            for (size_t i = 0; i < rank.size() && keep.size() < (size_t) g.n_slots; ++i) {
                keep.push_back(rank[i].second);
            }
            std::sort(keep.begin(), keep.end()); // consecutive experts of a layer in consecutive slots: one upload
            for (size_t s = 0; s < keep.size(); ++s) {
                const int32_t key = keep[s];
                lru.key_of[s] = key;
                lru.slot_map[lru.key_layer(key)][lru.key_expert(key)] = (int32_t) s;
                lru.move((int32_t) s, lru.class_of(key));
            }
            lru.n_filled = (int32_t) keep.size();
            for (size_t s = 0; s < keep.size(); ) {
                const int32_t key = keep[s];
                const int32_t il  = lru.key_layer(key);
                const int32_t e   = lru.key_expert(key);
                size_t n = 1;
                while (s + n < keep.size() && keep[s + n] == key + (int32_t) n && lru.key_layer(key + (int32_t) n) == il) {
                    n++;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w    = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    const size_t expert_size = bank->nb[2];
                    for (size_t k = 0; k < n; ) {
                        size_t run = w->nb[2] == expert_size ? n - k : 1;
                        ggml_backend_tensor_set(bank, (const uint8_t *) w->data + (e + k)*w->nb[2], (s + k)*expert_size, run*expert_size);
                        bytes += run*expert_size;
                        k += run;
                    }
                }
                s += n;
            }
            n_warm += keep.size();
        }
        LLAMA_LOG_WARN("%s: [HEAT] warm start: %zu experts placed from the profile, %.2f MiB uploaded\n", __func__,
                n_warm, bytes/1024.0/1024.0);
    }

    // fork (INKCACHEUX) GS8: a decode step has been submitted; run the window work now, while the device computes the
    // tail of the step, instead of at the first lookup of the next step. Same decisions: nothing plans in between.
    // The bank writes queue on the device stream behind the step that is still running.
    void window_end() {
        if (!ux.boundary || is_suspended) {
            return;
        }
        if (round_on && round_prev_il >= 0) {
            run_rounds();
            round_prev_il = -1; // the next lookup starts a window without running the rounds again
        }
        if (split.active()) {
            split.window_end();
        }
    }

    // fork (INKCACHEUX): a request ended (server). GS1 per-request line, GS4 periodic save.
    void request_end() {
        if (ux.req_stats) {
            auto line = [](const char * name, const stats & now, stats & prev) {
                const size_t hits = now.hits - prev.hits, misses = now.misses - prev.misses;
                const size_t cpu  = now.cpu - prev.cpu, bytes = now.bytes - prev.bytes;
                prev = now;
                if (hits + misses == 0) {
                    return;
                }
                LLAMA_LOG_WARN("llama_moe_cache: request %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                        name, hits, misses, cpu, 100.0*hits/(hits + misses), bytes/1024.0/1024.0);
            };
            line("decode", stats_small, req_small);
            line("prompt", stats_large, req_large);
        }
        if (heat.on()) {
            heat.maybe_save();
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_map(il, n_tokens);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

size_t llama_moe_cache::suspend() {
    return pimpl->suspend();
}

bool llama_moe_cache::resume(const uploader & up) {
    return pimpl->resume(up);
}

bool llama_moe_cache::resize(size_t size) {
    return pimpl->resize(size);
}

size_t llama_moe_cache::size() const {
    return pimpl->bank_bytes();
}

bool llama_moe_cache::suspended() const {
    return pimpl->is_suspended;
}

size_t llama_moe_cache::suspend_keep(size_t keep_bytes) {
    return pimpl->suspend_keep(keep_bytes);
}

bool llama_moe_cache::partial() const {
    return pimpl->is_partial;
}

const int32_t * llama_moe_cache::kept_slots(const ggml_tensor * w, ggml_tensor ** bank, int32_t * n_keep) const {
    return pimpl->kept_slots(w, bank, n_keep);
}

size_t llama_moe_cache::kept_bytes() const {
    return pimpl->kept_bytes();
}

void llama_moe_cache::last_resume(int32_t * n_up, int32_t * n_filled, size_t * bytes) const {
    *n_up     = pimpl->res_n_up;
    *n_filled = pimpl->res_n_all;
    *bytes    = pimpl->res_bytes;
}

void llama_moe_cache::set_op_offload(bool on) {
    pimpl->fewrow_op_offload = on;
}

void llama_moe_cache::window_end() {
    pimpl->window_end();
}

bool llama_moe_cache::boundary() const {
    return pimpl->ux.boundary;
}

void llama_moe_cache::request_end() {
    pimpl->request_end();
}

ggml_tensor * llama_moe_cache::og_slots(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->og_slots(ctx, ids, il, n_tokens, n_expert_used);
}

void llama_moe_cache::og_prepare() {
    if (pimpl->og) {
        pimpl->og->prepare();
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
