#pragma once

// Disk tier of the server prompt cache.
//
// RAM evictions (and, at shutdown or sleep, every RAM entry) are written here as one file per prompt and
// consulted on a RAM miss. Entries are keyed: the key string describes the model and every setting that
// changes the layout or the meaning of the serialized KV / recurrent state, and an entry whose key does not
// match is never restored. Several servers (different models, different KV settings) can share a directory:
// foreign entries count toward the size limit and the FIFO order, but are never read.
//
// file layout (little-endian), version 2:
//   magic "LCPC", u32 version, u64 key_hash, u32 key_len, char key[key_len],
//   u64 n_tokens, i32 tokens[n_tokens], u64 payload_len, payload[payload_len], u64 checksum
// payload:
//   u64 main_len, bytes, u64 drft_len, bytes, u32 n_checkpoints, then per checkpoint:
//   i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max, u64 len_tgt, bytes, u64 len_dft, bytes, u64 len_spec, bytes
// checksum: word-wise FNV-1a over every byte before it
//
// Writes go to a ".pc-*.tmp" file that is renamed into place, so a crash never leaves a readable partial
// entry. At startup, stale temp files, files of another format version, truncated files and files whose
// header does not parse are deleted; a checksum mismatch found at read time deletes the file too.

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <vector>

struct server_prompt_disk_entry {
    std::string  file;
    llama_tokens tokens;          // only loaded for entries whose key matches
    size_t       size  = 0;       // bytes on disk
    int64_t      mtime = 0;       // write time (ns since the epoch), defines the FIFO order
    bool         match = false;   // key matches this server's model + KV configuration
};

struct server_prompt_disk_record {
    llama_tokens                        tokens;
    std::vector<uint8_t>                main;
    std::vector<uint8_t>                drft;
    std::list<common_prompt_checkpoint> checkpoints;
};

uint64_t server_prompt_disk_hash(const void * data, size_t n, uint64_t seed = 0xcbf29ce484222325ull);

struct server_prompt_disk_tier {
    // free-space query, injectable for tests: returns false when the filesystem cannot be queried
    using space_fn_t = std::function<bool(const std::string & dir, uint64_t & capacity, uint64_t & available)>;

    std::string dir;
    std::string key;
    uint64_t    key_hash = 0;
    size_t      limit    = 0; // bytes, 0 = no limit

    // writes are skipped while the filesystem would have less than max(guard_frac * capacity, guard_abs) free
    double      guard_frac = 0.10;
    uint64_t    guard_abs  = 8ull*1024*1024*1024;
    space_fn_t  space_fn;

    // temp files older than this are leftovers of a crashed writer
    int64_t     tmp_stale_s = 120;

    // all entries in the directory, oldest first
    std::list<server_prompt_disk_entry> entries;

    // counters (for logs and tests)
    size_t n_guard_skips   = 0;
    size_t n_removed_bad   = 0;
    bool   guard_active    = false;

    // create the directory (mkdir -p), check that it is writable and index it; on failure `err` says why
    bool open(const std::string & dir, const std::string & key, size_t limit_bytes, std::string & err);

    bool enabled() const { return !dir.empty(); }

    size_t size() const;
    size_t n_match() const;

    // write one prompt atomically; skipped (returns false) under the free-space guard, when the entry alone
    // exceeds the limit, or when a matching entry already contains this prompt (returns true, nothing written).
    // matching entries that are a prefix of the new prompt are superseded and removed once it is on disk.
    bool write(const llama_tokens & tokens,
               const std::vector<uint8_t> & main,
               const std::vector<uint8_t> & drft,
               const std::list<common_prompt_checkpoint> & checkpoints);

    // read a matching entry; a file that fails to parse or to verify is deleted
    bool read(std::list<server_prompt_disk_entry>::iterator it, server_prompt_disk_record & out);

    void remove(std::list<server_prompt_disk_entry>::iterator it);

    // drop index entries whose file is gone, pick up files written by other processes, then evict the
    // oldest entries (by write time) until the directory fits the limit
    void enforce_limit();

private:
    uint64_t seq = 0;

    bool parse_header(const std::string & file, server_prompt_disk_entry & e, bool & bad) const;
    void scan(bool initial);
    bool guard_allows(uint64_t need);
};
