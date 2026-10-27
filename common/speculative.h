#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

// MTP drafter KV cache types over a SJ-KVaRN trunk when --spec-draft-type-k/-v are not given ({K, V}).
// Provisional: one line to change (tq6_0/tq6_0 once its fused verify-width kernel lands).
inline constexpr ggml_type COMMON_SJKVARN_MTP_DRAFT_KV[2] = { GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0 };

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // (optional) exact p/q drafting: the target's request sampling parameters, and where an implementation
    // that samples its draft from a matching distribution records that distribution per draft position
    // (candidate ids with normalised probabilities, aligned with *result; an empty row means id-match)
    const common_params_sampling * sampling = nullptr;
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // upstream #27694 fields (probabilistic draft sampling); kept for API compatibility. In this tree the
    // exact p/q path above (LLAMA_SPEC_PQ=1, `sampling`) is the sampled-draft implementation.
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // A shared verify budget can require fewer drafts than the configured minimum.
    int32_t n_min = -1;

};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// Process MTP prompt ranges while target hidden rows still cover the full batch.
bool common_speculative_supports_process_range(const common_speculative * spec);
bool common_speculative_process_range(common_speculative * spec, const common_batch & batch, int32_t first, int32_t count);
// [PREFILLFAST] LLAMA_MTP_B1_HANDOFF=1: the MTP helper can catch up per ubatch from the target's device rows
bool common_speculative_handoff_enabled(const common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
// n_accepted_ctrl >= 0 caps what the adaptive depth controller sees (mode-biased MTP passes the exact rule's count, so
// the depth choices, and with them the work per round, stay those of exact acceptance); -1 = n_accepted.
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted, int32_t n_accepted_ctrl = -1);

// target top-1 probabilities (request-matched) of the tokens one verify round committed; feeds the CDEPTH history.
// Call once per verify round, not for the re-decode after a checkpoint restore. Unknown entries are negative.
void common_speculative_note_target_conf(common_speculative * spec, llama_seq_id seq_id, const float * p, size_t n);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);
// Full slot carryover; does not affect partial-checkpoint eligibility.
bool common_speculative_get_resume_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_set_resume_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);


// Restore a snapshot only if the current implementation accepts its complete state.
bool common_speculative_restore_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

// adaptive draft-vocabulary tail (--spec-draft-vocab-hot): fold one request's tokens into the hot
// statistics, and re-rank/upload the hot slots. Both are no-ops without a hot tail and must be
// called from the decoding thread at a request boundary.
void    common_speculative_observe(common_speculative_init_result * spec, const llama_tokens & toks);
int32_t common_speculative_refresh(common_speculative_init_result * spec);

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);

// Each admitted sequence needs one target row. Retained drafts must stay whole; -1 means a fresh draft.
// A result of -1 defers the sequence. A zero result must skip drafting, not set draft_params.n_max to zero.
std::vector<int32_t> common_speculative_row_budget(const std::vector<int32_t> & requested,
        const std::vector<int32_t> & retained, int32_t capacity, size_t first,
        const std::vector<std::vector<double>> & acceptance = {});
