#pragma once

// UZU-1T tree verification for block drafters ("trees from marginals"), opt-in: LLAMA_SPEC_TREE=<P>.
//
// A block drafter (Inkling2 DSpark) gives K candidates with their probabilities at every draft depth. Next to the
// drafter's own chain, P more full-depth paths are built best-first by the product of the per-depth probabilities
// (a node has at most LLAMA_SPEC_TREE_BRANCH children, default 2). The server verifies every extra path as its own
// target sequence in the same batch as the chain; when the target's token at the chain's first miss is the next
// token of an extra path, the verify walk continues on that path's rows. Every committed token is still drawn from
// the target's own sampler (temp 0: argmax), so the output is the chain's.
//
// Unset or 0: off, nothing is reserved and no code path changes.

#include "llama.h"

#include <cstdint>
#include <functional>
#include <vector>

// number of extra paths per round (LLAMA_SPEC_TREE, 0 = off, at most 8)
int32_t common_spec_tree_paths();

// max children per tree node, chain child included (LLAMA_SPEC_TREE_BRANCH, default 2, 2..8)
int32_t common_spec_tree_branch();

// per-depth candidate marginals of the last block draft of one sequence
struct common_spec_tree_marg {
    std::vector<std::vector<llama_token_data>> cand; // [depth] candidates (id, p), any order
    std::vector<llama_token>                   chain; // the drafted tokens (one per depth)
};

// up to n_paths extra full-depth paths (each n_depth tokens, n_depth <= chain size), distinct from the chain's first
// n_depth tokens and from each other, best-first by the product of the candidate probabilities, with at most n_branch
// children per node. Empty when the marginals do not hold the chain.
std::vector<std::vector<llama_token>> common_spec_tree_build(const common_spec_tree_marg & m, size_t n_depth,
        int32_t n_paths, int32_t n_branch);

// one extra path of a verify batch
struct common_spec_tree_path {
    std::vector<llama_token> tokens; // d_1..d_L
    std::vector<int32_t>     rows;   // batch index of row r: r = 0 holds the anchor, r >= 1 holds tokens[r - 1]
    llama_seq_id             seq = -1;
};

struct common_sampler;

// After the chain walk accepted n0 drafted tokens and committed x at depth n0 (x != chain[n0]): continue on an extra
// path whose first n0 tokens are the chain's and whose token n0 is x. Each further token is sampled from the target
// row of that path and accepted into gsmpl; the walk follows a path while its next token is the sampled one.
// Returns the tokens committed after x (the last one is the new bonus token; empty: no path matched), path_out = the
// index of the path whose rows hold them. committed = the tokens this round committed before (for is_stop).
// pos_row0 = absolute position of row 0 (for LLAMA_SPEC_TREE_POSRNG).
std::vector<llama_token> common_spec_tree_extend(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<llama_token> & chain, const std::vector<common_spec_tree_path> & paths, size_t n0, llama_token x,
        const std::vector<llama_token> & committed, llama_pos pos_row0,
        const std::function<bool(llama_token, const std::vector<llama_token> &)> & is_stop, int32_t & path_out);
