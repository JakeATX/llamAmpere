#include "spec-tree.h"

#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <queue>
#include <set>

int32_t common_spec_tree_paths() {
    static const int32_t v = [] {
        const char * e = std::getenv("LLAMA_SPEC_TREE");
        return e ? std::max(0, std::min(8, std::atoi(e))) : 0;
    }();
    return v;
}

bool common_spec_tree_posrng() {
    static const bool v = [] {
        const char * e = std::getenv("LLAMA_SPEC_TREE_POSRNG");
        return e && std::atoi(e) != 0;
    }();
    return v;
}

static uint64_t spec_tree_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

double common_spec_tree_uniform(uint32_t seed, llama_pos pos, int32_t node) {
    uint64_t h = spec_tree_mix((uint64_t) seed);
    h = spec_tree_mix(h ^ (uint64_t) (uint32_t) pos);
    h = spec_tree_mix(h ^ ((uint64_t) (uint32_t) node << 32));
    return (double) (h >> 11) * (1.0 / 9007199254740992.0);
}

// inverse-CDF draw from the sampler's final candidates (their p after the chain); NULL if they are not a distribution
static llama_token spec_tree_draw(struct common_sampler * gsmpl, double u) {
    const llama_token_data_array * cur_p = common_sampler_get_candidates(gsmpl, false);
    if (!cur_p || cur_p->size == 0) {
        return LLAMA_TOKEN_NULL;
    }
    double sum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = cur_p->data[i].p;
        if (!(p >= 0.0f) || !std::isfinite(p)) {
            return LLAMA_TOKEN_NULL;
        }
        sum += p;
    }
    if (!(sum > 0.99 && sum < 1.01)) {
        return LLAMA_TOKEN_NULL;
    }
    const double t = u * sum;
    double acc = 0.0;
    size_t last = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p <= 0.0f) {
            continue;
        }
        last = i;
        acc += cur_p->data[i].p;
        if (t < acc) {
            return cur_p->data[i].id;
        }
    }
    return cur_p->data[last].id;
}

int32_t common_spec_tree_branch() {
    static const int32_t v = [] {
        const char * e = std::getenv("LLAMA_SPEC_TREE_BRANCH");
        return e ? std::max(2, std::min(8, std::atoi(e))) : 2;
    }();
    return v;
}

std::vector<std::vector<llama_token>> common_spec_tree_build(const common_spec_tree_marg & m, size_t n_depth,
        int32_t n_paths, int32_t n_branch) {
    std::vector<std::vector<llama_token>> out;
    if (n_paths <= 0 || n_depth == 0 || m.chain.size() < n_depth || m.cand.size() < n_depth) {
        return out;
    }

    // candidates per depth by probability, high first; log p of each
    std::vector<std::vector<llama_token_data>> cand(n_depth);
    for (size_t i = 0; i < n_depth; ++i) {
        cand[i] = m.cand[i];
        std::stable_sort(cand[i].begin(), cand[i].end(), [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
        bool has_chain = false;
        for (const auto & c : cand[i]) {
            has_chain = has_chain || c.id == m.chain[i];
        }
        if (cand[i].empty() || !has_chain) {
            return out;
        }
    }
    auto lp = [&](size_t i, int32_t k) { return std::log(std::max(1e-30, (double) cand[i][k].p)); };

    // trie of the kept paths (chain first): children of each prefix
    std::map<std::vector<llama_token>, std::set<llama_token>> kids;
    auto fits = [&](const std::vector<llama_token> & t) {
        std::vector<llama_token> pre;
        for (size_t i = 0; i < t.size(); ++i) {
            auto it = kids.find(pre);
            if (it != kids.end() && !it->second.count(t[i]) && (int32_t) it->second.size() >= n_branch) {
                return false;
            }
            pre.push_back(t[i]);
        }
        return true;
    };
    auto keep = [&](const std::vector<llama_token> & t) {
        std::vector<llama_token> pre;
        for (size_t i = 0; i < t.size(); ++i) {
            kids[pre].insert(t[i]);
            pre.push_back(t[i]);
        }
    };
    const std::vector<llama_token> chain(m.chain.begin(), m.chain.begin() + n_depth);
    keep(chain);

    // best-first over the rank vectors: the successors of a vector raise one depth's rank by one
    using item = std::pair<double, std::vector<int32_t>>;
    std::priority_queue<item> pq;
    std::set<std::vector<int32_t>> seen;
    {
        std::vector<int32_t> r0(n_depth, 0);
        double s = 0.0;
        for (size_t i = 0; i < n_depth; ++i) {
            s += lp(i, 0);
        }
        pq.push({ s, r0 });
        seen.insert(r0);
    }
    for (int32_t n_pop = 0; !pq.empty() && (int32_t) out.size() < n_paths && n_pop < 1024; ++n_pop) {
        const item it = pq.top();
        pq.pop();
        std::vector<llama_token> t(n_depth);
        for (size_t i = 0; i < n_depth; ++i) {
            t[i] = cand[i][it.second[i]].id;
        }
        if (t != chain && fits(t)) {
            keep(t);
            out.push_back(t);
        }
        for (size_t i = 0; i < n_depth; ++i) {
            if (it.second[i] + 1 >= (int32_t) cand[i].size()) {
                continue;
            }
            std::vector<int32_t> r = it.second;
            r[i]++;
            if (seen.insert(r).second) {
                pq.push({ it.first - lp(i, it.second[i]) + lp(i, r[i]), r });
            }
        }
    }
    return out;
}

std::vector<llama_token> common_spec_tree_extend(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<llama_token> & chain, const std::vector<common_spec_tree_path> & paths, size_t n0, llama_token x,
        const std::vector<llama_token> & committed, llama_pos pos_row0, float temp,
        const std::function<bool(llama_token, const std::vector<llama_token> &)> & is_stop, int32_t & path_out) {
    std::vector<llama_token> ext;
    path_out = -1;
    const size_t L = chain.size();
    if (n0 >= L) {
        return ext;
    }

    // paths that hold the committed prefix: the chain's first n0 tokens, then x
    std::vector<int32_t> live;
    for (int32_t k = 0; k < (int32_t) paths.size(); ++k) {
        const auto & p = paths[k];
        if (p.tokens.size() != L || p.rows.size() != L + 1 || p.tokens[n0] != x) {
            continue;
        }
        if (std::equal(chain.begin(), chain.begin() + n0, p.tokens.begin())) {
            live.push_back(k);
        }
    }
    if (live.empty()) {
        return ext;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<llama_token> prefix = committed;
    const bool     posrng = common_spec_tree_posrng() && temp > 0.0f;
    const uint32_t seed   = posrng ? common_sampler_get_seed(gsmpl) : 0;

    // row r of a live path holds tokens[r - 1]; its output is the target's token at depth r
    for (size_t r = n0 + 1; r <= L; ++r) {
        const int32_t k = live.front();
        llama_token y = common_sampler_sample(gsmpl, ctx, paths[k].rows[r]);
        if (posrng) {
            // UZU-3: this node's own uniform; the node index is the path's, the depth is in the position
            const llama_token yp = spec_tree_draw(gsmpl, common_spec_tree_uniform(seed, pos_row0 + (llama_pos) r, k));
            if (yp != LLAMA_TOKEN_NULL) {
                y = yp;
            }
        }
        common_sampler_accept(gsmpl, y, true);
        const bool stopped = llama_vocab_is_eog(vocab, y) || (is_stop && is_stop(y, prefix));
        ext.push_back(y);
        prefix.push_back(y);
        path_out = k;
        if (stopped || r == L) {
            break;
        }
        std::vector<int32_t> next;
        for (const int32_t j : live) {
            if (paths[j].tokens[r] == y) {
                next.push_back(j);
            }
        }
        if (next.empty()) {
            break;
        }
        live = std::move(next);
    }
    return ext;
}
