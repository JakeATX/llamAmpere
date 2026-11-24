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
        const std::vector<llama_token> & committed, llama_pos pos_row0,
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

    // row r of a live path holds tokens[r - 1]; its output is the target's token at depth r
    for (size_t r = n0 + 1; r <= L; ++r) {
        const int32_t k = live.front();
        const llama_token y = common_sampler_sample(gsmpl, ctx, paths[k].rows[r]);
        GGML_UNUSED(pos_row0);
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
