#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);

// Stateless greedy chains can sample every output row in one graph.
bool llama_sampler_is_greedy_chain(const llama_sampler * sampler);

// Chains whose backend graph produces only token ids: [greedy], or [logit-bias..., greedy].
// A bare greedy chain shares one argmax over all output rows; a biased one needs its own row block,
// because the bias has to be added to its rows before the argmax.
bool llama_sampler_is_argmax_chain(const llama_sampler * sampler);
