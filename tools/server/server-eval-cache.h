#pragma once

// Eval cache: a read-mostly folder of pre-built prompt states (--eval-cache DIR).
//
// Evals re-send the same long prompts. An eval-cache folder holds the prefilled state of each prompt (or of the
// document that several prompts share), built once with tools/eval-cache/llama-eval-cache.py. A server started
// with --eval-cache DIR restores the longest entry that is a prefix of an incoming prompt and only prefills the
// rest. Entries are never consumed or evicted.
//
// Folder layout:
//   DIR/eval-cache.json         index: format version, the configuration key, one record per entry
//   DIR/entries/<id>.evh        entry header: key, prompt tokens, context checkpoints, speculative state, checksum
//   DIR/entries/<id>.tgt        target model sequence state (llama_state_seq_save_file format, streamed on load)
//   DIR/entries/<id>.dft        drafter sequence state, when the server runs a drafter with its own context
// <id> = 16 hex digits of the FNV-1a hash of the entry's token ids, so rebuilding a prompt gives the same file.
//
// The key describes the model file and every setting that changes the layout or meaning of the stored state
// (KV types, SJ-KVaRN settings, recurrent-state type, drafter, flash attention, rope). It deliberately leaves out
// the build, the context size and the slot count, so a folder built once serves any --ctx-size / --parallel.
// A server whose key differs from the folder's refuses to start; an entry whose header key differs is skipped.
//
// .evh layout (little-endian):
//   magic "LEVC", u32 version, u64 key_hash, u32 key_len, char key[key_len], u64 n_tokens, i32 tokens[n_tokens],
//   u32 n_checkpoints, per checkpoint: i64 n_tokens, i32 pos_min, i32 pos_max, u64 len, tgt, u64 len, dft, u64 len, spec,
//   u64 spec_len, spec state, u8 has_dft, u64 checksum (FNV-1a over every byte before it)

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#define EVAL_CACHE_FORMAT_VERSION 2

struct server_eval_cache_entry {
    std::string  id;
    llama_tokens tokens;
    uint64_t     bytes   = 0; // all files of the entry
    bool         has_dft = false;
};

struct server_eval_cache_record {
    llama_tokens                        tokens;
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t>                spec;
    bool                                has_dft = false;
};

struct server_eval_cache {
    std::string dir;
    std::string key;
    uint64_t    key_hash = 0;
    bool        writable = false; // --eval-cache-build: the server may add entries

    std::vector<server_eval_cache_entry> entries;

    size_t n_skipped = 0; // entries ignored at open (bad header, key mismatch)

    // open the folder. Read-only mode requires an existing index whose key equals `key`; build mode creates the
    // folder and index when missing and still refuses a folder built for another key. An unsupported version disables the read-only cache with a warning; build mode refuses it. `err` says why on failure.
    bool open(const std::string & dir, const std::string & key, bool writable, std::string & err);

    bool enabled() const { return !dir.empty(); }

    uint64_t total_bytes() const;

    // the longest entry whose tokens are all a prefix of `tokens` and longer than `n_min`; -1 = none
    int find(const llama_tokens & tokens, size_t n_min) const;

    std::string path_hdr(const std::string & id) const;
    std::string path_tgt(const std::string & id) const;
    std::string path_dft(const std::string & id) const;

    // read an entry header (tokens, checkpoints, speculative state); verifies key and checksum
    bool read_header(const std::string & id, server_eval_cache_record & out, std::string & err) const;

    // write the header of a new entry, then add it to the index (the state files must already be in place)
    bool add(const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints,
             const std::vector<uint8_t> & spec, bool has_dft, std::string & id_out, std::string & err);

    static std::string entry_id(const llama_tokens & tokens);

    bool write_index(std::string & err) const;
};
