#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4
#define LLAMA_NGRAM_STATIC 2

// Data structures to map n-grams to empirical token probabilities:

struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];

    common_ngram() {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    common_ngram(const llama_token * input, const int ngram_size) {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    bool operator==(const common_ngram & other) const {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

struct common_token_hash_function {
    size_t operator()(const llama_token token) const {
        // see https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
        return token * 11400714819323198485llu;
    }
};

struct common_ngram_hash_function {
    size_t operator()(const common_ngram & ngram) const {
        size_t hash = common_token_hash_function{}(ngram.tokens[0]);
        for (int i = 1; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= common_token_hash_function{}(ngram.tokens[i]);
        }
        return hash;
    }
};

// token -> number of times token has been seen
typedef std::unordered_map<llama_token, int32_t> common_ngram_cache_part;

// n-gram -> empirical distribution of following tokens
typedef std::unordered_map<common_ngram, common_ngram_cache_part, common_ngram_hash_function> common_ngram_cache;


// Update an ngram cache with tokens.
// ngram_cache:         the cache to modify.
// ngram_min/ngram_max: the min/max size of the ngrams to extract from inp_data.
// inp_data:            the token sequence with which to update ngram_cache.
// nnew:                how many new tokens have been appended to inp_data since the last call to this function.
// print_progress:      whether to print progress to stderr.
//
// In order to get correct results inp_data can ONLY BE APPENDED TO.
// Changes in the middle need a complete rebuild.
void common_ngram_cache_update(
    common_ngram_cache & ngram_cache, int ngram_min, int ngram_max, std::vector<llama_token> & inp_data, int nnew, bool print_progress);

// Try to draft tokens from ngram caches.
// inp:                the tokens generated so far.
// draft:              the token sequence to draft. Expected to initially contain the previously sampled token.
// n_draft:            maximum number of tokens to add to draft.
// ngram_min/gram_max: the min/max size of the ngrams in nc_context and nc_dynamic.
// nc_context:         ngram cache based on current context.
// nc_dynamic:         ngram cache based on previous user generations.
// nc_static:          ngram cache generated from a large text corpus, used for validation.
void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static);

// Persistence
//
// WHAT A CACHE FILE CONTAINS - read this before pointing --lookup-cache-dynamic-save at a shared location:
//
//   A cache file is a list of token n-grams (1 to LLAMA_NGRAM_MAX tokens) and, for each, the tokens that
//   followed it together with how often they did. The tokens come verbatim from everything the model saw:
//   system prompts, user messages, retrieved documents, tool output and the generated replies. Detokenized,
//   fragments of that text can be reconstructed from the file. It is NOT opaque telemetry; treat it with the
//   same care as a log of the conversations that produced it. Writing it is opt-in and only ever goes to a
//   path given explicitly on the command line.
//
// File layout (all integers little-endian, native sizes):
//
//   magic        char[4]  "LNGC"
//   version      u32      2
//   n_vocab      u32      number of tokens in the vocab the cache was built with (0 = unknown)
//   vocab_hash   u64      FNV-1a of all token texts of that vocab (0 = unknown)
//   n_ngrams     u64      number of n-gram entries in the payload
//   payload_size u64      size of the payload in bytes
//   payload_hash u64      FNV-1a of the payload bytes
//   payload:     n_ngrams entries of
//     ngram      i32[LLAMA_NGRAM_MAX]   the n-gram, padded with LLAMA_TOKEN_NULL
//     n_tokens   i32                    number of (token, count) pairs following (> 0)
//     token      i32 } n_tokens times   a token that followed the n-gram
//     count      i32 }                  how often it did (> 0)
//
// Files without the header (written by older versions of llama-lookup-create) hold the payload only and are
// still accepted. Any load problem - unreadable file, bad magic/version, vocab mismatch, truncation, hash
// mismatch, out of range value - is reported through the return value and never aborts.

// Identity of the vocabulary a cache was built with. A cache is only meaningful for the tokenizer that
// produced it, so it is stored in the file and checked at load. n_tokens = 0 / hash = 0 mean "unknown" and
// disable the check for that side.
struct common_ngram_cache_vocab_id {
    uint32_t n_tokens = 0;
    uint64_t hash     = 0;
};

// Compute the identity of a vocab (hash of all token texts).
common_ngram_cache_vocab_id common_ngram_cache_get_vocab_id(const llama_vocab * vocab);

enum common_ngram_cache_load_status {
    COMMON_NGRAM_CACHE_LOAD_OK,
    COMMON_NGRAM_CACHE_LOAD_MISSING, // the file cannot be opened
    COMMON_NGRAM_CACHE_LOAD_CORRUPT, // truncated, checksum mismatch, out of range values: not a usable cache
    COMMON_NGRAM_CACHE_LOAD_FOREIGN, // a valid cache, but for another vocab or a newer format: must not be overwritten
};

// Load an ngram cache file into ngram_cache.
// vocab_id: identity of the current vocab; a file that records a different vocab is refused.
// returns:  COMMON_NGRAM_CACHE_LOAD_OK on success. Otherwise ngram_cache is left empty and err describes the problem.
common_ngram_cache_load_status common_ngram_cache_load_file(
    const std::string & filename, common_ngram_cache & ngram_cache,
    const common_ngram_cache_vocab_id & vocab_id, std::string & err);

// Save an ngram cache to a file, written to a temporary file next to it and renamed into place, so a reader
// or a crash never sees a partially written file.
// returns: true on success, otherwise err describes the problem and the previous file (if any) is untouched.
bool common_ngram_cache_save_file(
    const common_ngram_cache & ngram_cache, const std::string & filename,
    const common_ngram_cache_vocab_id & vocab_id, std::string & err);

// Evict the lowest-utility n-grams until at most n_max remain (n_max = 0: no limit).
// See the implementation for the ranking.
void common_ngram_cache_prune(common_ngram_cache & ngram_cache, size_t n_max);

// Read-modify-write of a cache file shared between processes:
// merges ngram_cache_delta into the file's current content (under an advisory lock where available),
// prunes to n_max, writes the result atomically and, on success, replaces ngram_cache with the merged
// result and clears ngram_cache_delta.
// If the file is missing or corrupt, ngram_cache (which must already contain the delta) is written instead.
// A file that belongs to another vocab or a newer format is never overwritten: the sync fails.
// ngram_cache and ngram_cache_delta must be distinct objects.
// returns: true on success, otherwise err describes the problem and nothing is modified.
bool common_ngram_cache_sync_file(
    const std::string & filename, common_ngram_cache & ngram_cache, common_ngram_cache & ngram_cache_delta,
    size_t n_max, const common_ngram_cache_vocab_id & vocab_id, std::string & err);

// Save an ngram cache to a file (vocab identity unknown). Logs on failure.
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename);

// Load an ngram cache saved with common_ngram_cache_save.
// filename: the path from which to load the ngram cache.
// returns:  an ngram cache containing the information saved to filename.
// throws:   std::ifstream::failure if the file cannot be opened or is not a usable cache file.
common_ngram_cache common_ngram_cache_load(const std::string & filename);

// Merge two ngram caches, counts of shared (n-gram, token) pairs are summed.
// ngram_cache_target: the ngram cache to which to add the information from ngram_cache_add.
// ngram_cache_add:    the ngram cache to add to ngram_cache_target.
void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add);
