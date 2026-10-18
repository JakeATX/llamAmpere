#pragma once

// Disk tier of the server prompt cache.
//
// RAM evictions (and, at shutdown or sleep, every RAM entry) are written here as one file per prompt and
// consulted on a RAM miss. Entries are keyed: the key string describes the model and every setting that
// changes the layout or the meaning of the serialized KV / recurrent state, and an entry whose key does not
// match is never restored. Several servers (different models, different KV settings) can share a directory:
// foreign entries count toward the size limit and the FIFO order, but are never read.
//
// file layout (little-endian), version 4:
//   magic "LCPC", u32 version, u64 key_hash, u32 key_len, char key[key_len],
//   u64 n_tokens, i32 tokens[n_tokens], u64 payload_len, payload[payload_len], u64 checksum
// payload:
//   u64 main_len, bytes, u64 drft_len, bytes, u64 spec_len, bytes, u32 n_checkpoints, then per checkpoint:
//   i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max, u64 len_tgt, bytes, u64 len_dft, bytes, u64 len_spec, bytes
//   then u32 n_lora, f32 lora_scales[n_lora] (the adapter scales the state was computed with)
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
    std::vector<uint8_t>                spec;
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<float>                  lora_scales;
};

uint64_t server_prompt_disk_hash(const void * data, size_t n, uint64_t seed = 0xcbf29ce484222325ull);

// the checkpoints of an entry with n_tokens tokens that a keep-newest-plus-doubling-distance policy keeps: the
// ckpt_keep newest, then one each time the distance from the end has doubled; in list order. 0 = all. Used by
// the RAM tier (--cache-ram-checkpoints).
std::vector<const common_prompt_checkpoint *> server_prompt_keep_checkpoints(
        const std::list<common_prompt_checkpoint> & checkpoints, size_t n_tokens, size_t ckpt_keep);

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

    // while writes are being skipped, the warning is repeated at most once per this many seconds (0 = every skip)
    double      guard_warn_interval_s = 60.0;
    // monotonic clock in microseconds for the warning rate limit, injectable for tests (default ggml_time_us)
    std::function<int64_t()> clock_us_fn;

    // temp files older than this are leftovers of a crashed writer
    int64_t     tmp_stale_s = 120;

    // all entries in the directory, oldest first
    std::list<server_prompt_disk_entry> entries;

    // counters (for logs and tests)
    size_t   n_guard_skips      = 0; // writes skipped by the free-space guard since start
    uint64_t bytes_guard_skips  = 0; // bytes those writes would have written
    size_t   n_guard_warnings   = 0; // guard warnings logged (first one + rate-limited repeats)
    size_t   n_removed_bad   = 0;
    bool     guard_active    = false;

    // free space against the guard's floor
    enum space_state { SPACE_UNKNOWN, SPACE_OK, SPACE_NEAR, SPACE_BELOW };
    struct space_info {
        space_state state     = SPACE_UNKNOWN;
        uint64_t    capacity  = 0;
        uint64_t    available = 0;
        uint64_t    reserve   = 0; // the floor: max(guard_frac * capacity, guard_abs)
        uint64_t    want      = 0; // headroom above the floor the tier wants (room left to its limit, else guard_abs)
    };
    // SPACE_BELOW: writes are skipped now; SPACE_NEAR: the floor will stop writes before the tier reaches its limit
    space_info space_check() const;

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
               const std::list<common_prompt_checkpoint> & checkpoints,
               const std::vector<uint8_t> & spec = {},
               const std::vector<float> & lora_scales = {});

    // read a matching entry; a file that fails to parse or to verify is deleted
    bool read(std::list<server_prompt_disk_entry>::iterator it, server_prompt_disk_record & out);

    void remove(std::list<server_prompt_disk_entry>::iterator it);

    // drop index entries whose file is gone, pick up files written by other processes, then evict the
    // oldest entries (by write time) until the directory fits the limit
    void enforce_limit();

private:
    uint64_t seq = 0;

    int64_t guard_t_warn      = 0; // time of the last guard warning (clock_us_fn)
    size_t  guard_skips_warn  = 0; // n_guard_skips at the last guard warning

    int64_t clock_us() const;

    bool parse_header(const std::string & file, server_prompt_disk_entry & e, bool & bad) const;
    void scan(bool initial);
    bool guard_allows(uint64_t need);
};
