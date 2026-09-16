#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>
#endif

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            common_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                common_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                common_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static llama_token try_draft(common_ngram_cache & nc_static, const common_ngram ngram_static) {
    common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const common_ngram_cache_part part_static = part_static_it->second;

    int max_count_static  = 0;
    int sum_count_static  = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (std::pair<llama_token, int> token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static  = token_count_static.second;

        if (count_static > max_count_static) {
            max_token        = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*max_count_static < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    common_ngram_cache & nc_primary, const std::vector<common_ngram> & ngrams_primary, common_ngram_cache_part & part_static,
    const int * min_sample_size, const int * min_percent) {

    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (int i = ngrams_primary.size()-1; i >= 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const common_ngram ngram_primary = ngrams_primary[i];

        common_ngram_cache::iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const common_ngram_cache_part part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            common_ngram_cache_part::iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;

            if (count_primary*count_static > max_count_primary*max_count_static) {
                max_token         = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[i]) {
            continue;
        }
        if (100*max_count_primary < min_percent[i]*sum_count_primary) {
            continue;;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static
) {
    GGML_ASSERT(draft.size() == 1);
    const int inp_size = inp.size();

    if (inp_size < LLAMA_NGRAM_STATIC) {
        return;
    }

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + draft.size()-1;
        common_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
        }
        common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
        common_ngram_cache_part part_static;
        if (part_static_it != nc_static.end()) {
            part_static = part_static_it->second;
        }

        // cd = context + dynamic
        std::vector<common_ngram> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size-ngram_size_cd + draft.size()-1;
            common_ngram ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ngrams_cd.push_back(ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static, draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static, draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_static, ngram_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        LOG_DBG(" - draft candidate: token=%d\n", drafted_token);
        draft.push_back(drafted_token);
    }
}

//
// persistence
//
// See ngram-cache.h for the file layout and for what the file contains.
//

static const char     COMMON_NGRAM_CACHE_MAGIC[4] = { 'L', 'N', 'G', 'C' };
static const uint32_t COMMON_NGRAM_CACHE_VERSION  = 2;

// upper bound on the number of (token, count) pairs of one n-gram, only used to reject garbage early
static const int32_t COMMON_NGRAM_CACHE_MAX_PART = 1 << 24;

static uint64_t common_ngram_cache_fnv1a(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static const uint64_t COMMON_NGRAM_CACHE_FNV_INIT = 14695981039346656037ULL;

common_ngram_cache_vocab_id common_ngram_cache_get_vocab_id(const llama_vocab * vocab) {
    common_ngram_cache_vocab_id id;
    if (vocab == nullptr) {
        return id;
    }

    const int32_t n_tokens = llama_vocab_n_tokens(vocab);

    uint64_t hash = COMMON_NGRAM_CACHE_FNV_INIT;
    for (llama_token token = 0; token < n_tokens; ++token) {
        const char * text = llama_vocab_get_text(vocab, token);
        if (text != nullptr) {
            hash = common_ngram_cache_fnv1a(hash, text, strlen(text));
        }
        hash = common_ngram_cache_fnv1a(hash, "", 1); // separator, includes the terminating 0
    }

    id.n_tokens = (uint32_t) n_tokens;
    id.hash     = hash != 0 ? hash : 1; // 0 is reserved for "unknown"
    return id;
}

// A token is plausible if it is in the vocab (when the vocab size is known) or, for n-gram padding, LLAMA_TOKEN_NULL.
static bool common_ngram_cache_token_ok(llama_token token, uint32_t n_vocab, bool allow_null) {
    if (token == LLAMA_TOKEN_NULL) {
        return allow_null;
    }
    if (token < 0) {
        return false;
    }
    return n_vocab == 0 || (uint32_t) token < n_vocab;
}

// Sequential reader with exact-size reads and a running hash of everything read.
struct common_ngram_cache_reader {
    std::ifstream & file;
    uint64_t hash = COMMON_NGRAM_CACHE_FNV_INIT;
    uint64_t n_read = 0;

    common_ngram_cache_reader(std::ifstream & file) : file(file) {}

    // restart the hash and the byte count (used to hash the payload separately from the header)
    void reset() {
        hash   = COMMON_NGRAM_CACHE_FNV_INIT;
        n_read = 0;
    }

    bool get(void * dst, size_t size) {
        if (!file.read(static_cast<char *>(dst), size)) {
            return false;
        }
        hash = common_ngram_cache_fnv1a(hash, dst, size);
        n_read += size;
        return true;
    }

    template <typename T> bool get(T & value) {
        return get(&value, sizeof(T));
    }
};

// Parse n_ngrams entries (or, if n_ngrams is SIZE_MAX, until the end of the file) of payload into ngram_cache.
static bool common_ngram_cache_read_payload(
        common_ngram_cache_reader & reader, common_ngram_cache & ngram_cache,
        uint64_t n_ngrams, uint32_t n_vocab, std::string & err) {
    const bool until_eof = n_ngrams == UINT64_MAX;

    for (uint64_t i = 0; until_eof || i < n_ngrams; ++i) {
        common_ngram ngram;
        if (!reader.get(ngram.tokens, sizeof(ngram.tokens))) {
            if (until_eof && reader.file.eof() && reader.file.gcount() == 0) {
                break; // clean end of a header-less file
            }
            err = "truncated at n-gram " + std::to_string(i);
            return false;
        }

        // the n-gram is a prefix of real tokens followed by LLAMA_TOKEN_NULL padding
        bool in_padding = false;
        for (int j = 0; j < LLAMA_NGRAM_MAX; ++j) {
            const llama_token token = ngram.tokens[j];
            if (!common_ngram_cache_token_ok(token, n_vocab, true)) {
                err = "n-gram " + std::to_string(i) + " has token " + std::to_string(token) + " outside the vocab";
                return false;
            }
            if (token == LLAMA_TOKEN_NULL) {
                in_padding = true;
            } else if (in_padding) {
                err = "n-gram " + std::to_string(i) + " has a token after its padding";
                return false;
            }
        }
        if (ngram.tokens[0] == LLAMA_TOKEN_NULL) {
            err = "n-gram " + std::to_string(i) + " is empty";
            return false;
        }

        int32_t n_tokens = 0;
        if (!reader.get(n_tokens)) {
            err = "truncated at n-gram " + std::to_string(i);
            return false;
        }
        if (n_tokens <= 0 || n_tokens > COMMON_NGRAM_CACHE_MAX_PART) {
            err = "n-gram " + std::to_string(i) + " has an implausible number of continuations: " + std::to_string(n_tokens);
            return false;
        }

        common_ngram_cache_part part;
        part.reserve(n_tokens);
        for (int32_t j = 0; j < n_tokens; ++j) {
            llama_token token = LLAMA_TOKEN_NULL;
            int32_t     count = 0;
            if (!reader.get(token) || !reader.get(count)) {
                err = "truncated at n-gram " + std::to_string(i);
                return false;
            }
            if (!common_ngram_cache_token_ok(token, n_vocab, false)) {
                err = "n-gram " + std::to_string(i) + " has continuation token " + std::to_string(token) + " outside the vocab";
                return false;
            }
            if (count <= 0) {
                err = "n-gram " + std::to_string(i) + " has a non-positive count";
                return false;
            }
            part[token] += count; // a duplicated token in a corrupt or hand-merged file is folded, not rejected
        }

        ngram_cache[ngram] = std::move(part); // a duplicated n-gram keeps the last entry
    }

    return true;
}

common_ngram_cache_load_status common_ngram_cache_load_file(
        const std::string & filename, common_ngram_cache & ngram_cache,
        const common_ngram_cache_vocab_id & vocab_id, std::string & err) {
    ngram_cache.clear();
    err.clear();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        err = "cannot open " + filename;
        return COMMON_NGRAM_CACHE_LOAD_MISSING;
    }

    common_ngram_cache_reader reader(file);

    char magic[4] = { 0, 0, 0, 0 };
    if (!reader.get(magic, sizeof(magic))) {
        if (file.eof() && file.gcount() == 0) {
            return COMMON_NGRAM_CACHE_LOAD_OK; // empty file: empty cache
        }
        err = filename + ": truncated header";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    if (memcmp(magic, COMMON_NGRAM_CACHE_MAGIC, sizeof(magic)) != 0) {
        // no header: a file written by an older llama-lookup-create; the payload starts at byte 0
        file.clear();
        file.seekg(0, std::ios::beg);
        if (!file) {
            err = filename + ": cannot rewind";
            return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
        }
        reader.reset();

        std::string err_payload;
        if (!common_ngram_cache_read_payload(reader, ngram_cache, UINT64_MAX, vocab_id.n_tokens, err_payload)) {
            ngram_cache.clear();
            err = filename + ": not a lookup cache file (" + err_payload + ")";
            return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
        }
        LOG_INF("%s: %s has no header (older format), vocab identity not checked, %zu n-grams\n",
                __func__, filename.c_str(), ngram_cache.size());
        return COMMON_NGRAM_CACHE_LOAD_OK;
    }

    uint32_t version      = 0;
    uint32_t n_vocab      = 0;
    uint64_t vocab_hash   = 0;
    uint64_t n_ngrams     = 0;
    uint64_t payload_size = 0;
    uint64_t payload_hash = 0;
    if (!reader.get(version) || !reader.get(n_vocab) || !reader.get(vocab_hash) ||
        !reader.get(n_ngrams) || !reader.get(payload_size) || !reader.get(payload_hash)) {
        err = filename + ": truncated header";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    if (version != COMMON_NGRAM_CACHE_VERSION) {
        err = filename + ": unsupported format version " + std::to_string(version) +
              " (expected " + std::to_string(COMMON_NGRAM_CACHE_VERSION) + ")";
        return COMMON_NGRAM_CACHE_LOAD_FOREIGN;
    }

    // vocab identity: mismatch means the file was built for another tokenizer and would draft garbage
    if (vocab_id.n_tokens != 0 && n_vocab != 0 && vocab_id.n_tokens != n_vocab) {
        err = filename + ": built for a vocab of " + std::to_string(n_vocab) +
              " tokens, current model has " + std::to_string(vocab_id.n_tokens);
        return COMMON_NGRAM_CACHE_LOAD_FOREIGN;
    }
    if (vocab_id.hash != 0 && vocab_hash != 0 && vocab_id.hash != vocab_hash) {
        err = filename + ": built for a different vocab (same size, different token texts)";
        return COMMON_NGRAM_CACHE_LOAD_FOREIGN;
    }

    // the smallest possible entry is one n-gram, n_tokens and one (token, count) pair
    const uint64_t min_entry_size = sizeof(common_ngram) + sizeof(int32_t) + sizeof(llama_token) + sizeof(int32_t);
    if (n_ngrams > payload_size / min_entry_size) {
        err = filename + ": header claims " + std::to_string(n_ngrams) + " n-grams in " + std::to_string(payload_size) + " bytes";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    const uint32_t n_vocab_check = vocab_id.n_tokens != 0 ? vocab_id.n_tokens : n_vocab;

    reader.reset(); // the payload hash does not cover the header

    std::string err_payload;
    if (!common_ngram_cache_read_payload(reader, ngram_cache, n_ngrams, n_vocab_check, err_payload)) {
        ngram_cache.clear();
        err = filename + ": " + err_payload;
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    if (reader.n_read != payload_size) {
        ngram_cache.clear();
        err = filename + ": payload size mismatch (" + std::to_string(reader.n_read) + " read, " + std::to_string(payload_size) + " expected)";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }
    if (reader.hash != payload_hash) {
        ngram_cache.clear();
        err = filename + ": payload checksum mismatch";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    char trailing = 0;
    if (file.read(&trailing, 1)) {
        ngram_cache.clear();
        err = filename + ": trailing data after the payload";
        return COMMON_NGRAM_CACHE_LOAD_CORRUPT;
    }

    return COMMON_NGRAM_CACHE_LOAD_OK;
}

static void common_ngram_cache_put(std::vector<char> & buf, const void * data, size_t size) {
    const char * bytes = static_cast<const char *>(data);
    buf.insert(buf.end(), bytes, bytes + size);
}

template <typename T> static void common_ngram_cache_put(std::vector<char> & buf, const T & value) {
    common_ngram_cache_put(buf, &value, sizeof(T));
}

// Force the contents of a file that was written and closed through a stream to stable storage.
// Closing the stream only hands the data to the page cache: without this a crash or power loss after the
// rename can leave the final name pointing at an empty or truncated file, the one case the temporary file
// is meant to prevent. On failure errno describes the problem.
static bool common_ngram_cache_fsync_file(const std::string & filename) {
#ifdef _WIN32
    const int fd = _open(filename.c_str(), _O_WRONLY | _O_BINARY);
    if (fd < 0) {
        return false;
    }
    const bool ok = _commit(fd) == 0;
    const int err_saved = errno;
    _close(fd);
    errno = err_saved;
    return ok;
#else
    const int fd = open(filename.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const bool ok = fsync(fd) == 0;
    const int err_saved = errno;
    close(fd);
    errno = err_saved;
    return ok;
#endif
}

// Make a completed rename durable: the new directory entry is metadata of the directory and is not covered
// by the fsync of the file. Best effort - the file is already complete and visible, so if the directory
// cannot be synced the worst case is that a crash brings back the previous version.
static void common_ngram_cache_fsync_dir(const std::string & filename) {
#ifndef _WIN32
    std::filesystem::path dir = std::filesystem::path(filename).parent_path();
    if (dir.empty()) {
        dir = ".";
    }
    const int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        return;
    }
    fsync(fd);
    close(fd);
#else
    GGML_UNUSED(filename);
#endif
}

bool common_ngram_cache_save_file(
        const common_ngram_cache & ngram_cache, const std::string & filename,
        const common_ngram_cache_vocab_id & vocab_id, std::string & err) {
    err.clear();

    // serialize the payload in memory first, so the hash and the sizes are known before anything is written
    std::vector<char> payload;
    uint64_t n_ngrams = 0;
    for (const auto & item : ngram_cache) {
        const common_ngram & ngram = item.first;
        const common_ngram_cache_part & part = item.second;
        if (part.empty()) {
            continue; // never written: the reader rejects n-grams without continuations
        }

        int32_t n_tokens = 0;
        for (const auto & token_count : part) {
            if (token_count.second > 0) {
                ++n_tokens;
            }
        }
        if (n_tokens == 0) {
            continue;
        }

        common_ngram_cache_put(payload, ngram.tokens, sizeof(ngram.tokens));
        common_ngram_cache_put(payload, n_tokens);
        for (const auto & token_count : part) {
            if (token_count.second <= 0) {
                continue;
            }
            common_ngram_cache_put(payload, token_count.first);
            common_ngram_cache_put(payload, token_count.second);
        }
        ++n_ngrams;
    }

    const uint64_t payload_size = payload.size();
    const uint64_t payload_hash = common_ngram_cache_fnv1a(COMMON_NGRAM_CACHE_FNV_INIT, payload.data(), payload.size());

    std::vector<char> header;
    common_ngram_cache_put(header, COMMON_NGRAM_CACHE_MAGIC, sizeof(COMMON_NGRAM_CACHE_MAGIC));
    common_ngram_cache_put(header, COMMON_NGRAM_CACHE_VERSION);
    common_ngram_cache_put(header, vocab_id.n_tokens);
    common_ngram_cache_put(header, vocab_id.hash);
    common_ngram_cache_put(header, n_ngrams);
    common_ngram_cache_put(header, payload_size);
    common_ngram_cache_put(header, payload_hash);

    // write to a temporary file in the same directory and rename it into place: a reader either sees the
    // complete previous file or the complete new one, never a partial write
    const std::string filename_tmp = filename + ".tmp." + std::to_string(
#ifdef _WIN32
            (long long) _getpid()
#else
            (long long) getpid()
#endif
            );

    {
        std::ofstream file(filename_tmp, std::ios::binary | std::ios::trunc);
        if (!file) {
            err = "cannot create " + filename_tmp;
            return false;
        }
        file.write(header.data(),  header.size());
        file.write(payload.data(), payload.size());
        file.flush();
        if (!file) {
            err = "write to " + filename_tmp + " failed";
        }
    }

    // the stream is closed but the data may still be in the page cache only: it has to be on disk before
    // the rename makes it the current file
    if (err.empty() && !common_ngram_cache_fsync_file(filename_tmp)) {
        err = "cannot sync " + filename_tmp + " to disk: " + strerror(errno);
    }

    if (err.empty()) {
        std::error_code ec;
        std::filesystem::rename(filename_tmp, filename, ec);
        if (ec) {
            err = "cannot rename " + filename_tmp + " to " + filename + ": " + ec.message();
        } else {
            common_ngram_cache_fsync_dir(filename);
        }
    }

    if (!err.empty()) {
        std::error_code ec;
        std::filesystem::remove(filename_tmp, ec);
        return false;
    }

    return true;
}

// Eviction ranking for common_ngram_cache_prune.
//
// An n-gram is only worth keeping if it is likely to be looked up again AND, when it is, the lookup produces
// a draft. common_ngram_cache_draft drafts the most frequent continuation only if it holds at least
// draft_min_percent of all observations after the n-gram. So for an n-gram with continuation counts c_i,
// max = max(c_i) and sum = sum(c_i):
//
//   score = max * max / sum  =  max * (max / sum)
//
// i.e. the frequency with which the winning continuation was seen, scaled by how dominant it is. An n-gram
// seen 100 times with 100 different continuations (max = 1, sum = 100) scores 0.01 and goes first; one seen
// 10 times with the same continuation every time scores 10. Plain frequency (sum) would keep the useless
// high-entropy n-grams (e.g. after common single tokens) and evict the rare but reliable ones.
// Acceptance feedback is not available for this drafter (accept() is a no-op), so the ranking is computed
// from the counts alone.
static double common_ngram_cache_score(const common_ngram_cache_part & part) {
    int64_t max_count = 0;
    int64_t sum_count = 0;
    for (const auto & token_count : part) {
        const int64_t count = token_count.second;
        max_count  = std::max(max_count, count);
        sum_count += count;
    }
    if (sum_count <= 0) {
        return 0.0;
    }
    return (double) max_count * (double) max_count / (double) sum_count;
}

void common_ngram_cache_prune(common_ngram_cache & ngram_cache, size_t n_max) {
    if (n_max == 0 || ngram_cache.size() <= n_max) {
        return;
    }

    struct scored {
        double score;
        const common_ngram * ngram;
    };

    std::vector<scored> ranking;
    ranking.reserve(ngram_cache.size());
    for (const auto & item : ngram_cache) {
        ranking.push_back({ common_ngram_cache_score(item.second), &item.first });
    }

    std::nth_element(ranking.begin(), ranking.begin() + n_max, ranking.end(),
            [](const scored & a, const scored & b) { return a.score > b.score; });

    // erasing one key does not invalidate pointers to the other keys of an unordered_map
    for (size_t i = n_max; i < ranking.size(); ++i) {
        ngram_cache.erase(*ranking[i].ngram);
    }
}

// Advisory inter-process lock on <filename>.lock, held while a cache file is read, merged and rewritten.
// The lock file itself is never renamed or deleted, so every process locks the same inode.
// Best effort: unavailable on Windows, and a lock that cannot be taken is reported, not waited for.
struct common_ngram_cache_file_lock {
#ifndef _WIN32
    int fd = -1;
#endif

    bool acquire(const std::string & filename, int n_tries) {
#ifndef _WIN32
        const std::string filename_lock = filename + ".lock";
        fd = open(filename_lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) {
            return false;
        }
        for (int i = 0; i < n_tries; ++i) {
            if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
                return true;
            }
            if (errno != EWOULDBLOCK && errno != EINTR) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        close(fd);
        fd = -1;
        return false;
#else
        GGML_UNUSED(filename);
        GGML_UNUSED(n_tries);
        return false;
#endif
    }

    ~common_ngram_cache_file_lock() {
#ifndef _WIN32
        if (fd >= 0) {
            flock(fd, LOCK_UN);
            close(fd);
        }
#endif
    }
};

// Remove <filename>.tmp.<pid> files left behind by a writer that died between writing and renaming.
// Must be called with the lock on <filename>.lock held: every cooperating writer creates and renames its
// temporary file under that lock, so no temporary of a live writer can be in flight while we hold it.
// A writer that gave up on the lock is the exception, which is why a temporary is only removed if its pid
// is demonstrably gone (kill(pid, 0) fails with ESRCH); a pid that exists, or that cannot be signalled,
// keeps its file. Best effort: nothing here is reported as an error.
static void common_ngram_cache_remove_stale_tmp(const std::string & filename) {
#ifndef _WIN32
    const std::filesystem::path path(filename);
    const std::string prefix = path.filename().string() + ".tmp.";

    std::filesystem::path dir = path.parent_path();
    if (dir.empty()) {
        dir = ".";
    }

    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    for (; !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }

        const std::string pid_str = name.substr(prefix.size());
        if (pid_str.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        errno = 0;
        const long long pid = strtoll(pid_str.c_str(), nullptr, 10);
        if (errno != 0 || pid <= 0 || pid > INT_MAX) {
            continue;
        }
        if (pid == (long long) getpid()) {
            continue; // ours: either not written yet or already renamed away
        }

        std::error_code ec_type;
        if (!it->is_regular_file(ec_type)) {
            continue;
        }

        if (kill((pid_t) pid, 0) == 0 || errno != ESRCH) {
            continue; // alive, or exists but belongs to another user (EPERM)
        }

        std::error_code ec_rm;
        if (std::filesystem::remove(it->path(), ec_rm)) {
            LOG_INF("%s: removed %s, left behind by a writer (pid %lld) that is gone\n",
                    __func__, it->path().string().c_str(), pid);
        }
    }
#else
    GGML_UNUSED(filename);
#endif
}

bool common_ngram_cache_sync_file(
        const std::string & filename, common_ngram_cache & ngram_cache, common_ngram_cache & ngram_cache_delta,
        size_t n_max, const common_ngram_cache_vocab_id & vocab_id, std::string & err, bool * from_disk) {
    GGML_ASSERT(&ngram_cache != &ngram_cache_delta);
    err.clear();
    if (from_disk) {
        *from_disk = false;
    }

    common_ngram_cache_file_lock lock;
    const bool locked = lock.acquire(filename, /*n_tries=*/ 40); // ~2 s
    if (locked) {
        common_ngram_cache_remove_stale_tmp(filename);
    } else {
        LOG_WRN("%s: could not lock %s.lock, writing without lock (another writer may lose its last update)\n",
                __func__, filename.c_str());
    }

    // start from what is on disk now (another process may have written since we loaded it) and add our delta;
    // if the file is missing or unusable, our in-memory cache already contains everything we know
    common_ngram_cache merged;
    bool loaded = false;
    {
        std::string err_load;
        switch (common_ngram_cache_load_file(filename, merged, vocab_id, err_load)) {
            case COMMON_NGRAM_CACHE_LOAD_OK:
                loaded = true;
                break;
            case COMMON_NGRAM_CACHE_LOAD_MISSING:
                break; // first save, or the file was removed meanwhile
            case COMMON_NGRAM_CACHE_LOAD_CORRUPT:
                LOG_WRN("%s: existing %s is unusable and will be replaced: %s\n", __func__, filename.c_str(), err_load.c_str());
                break;
            case COMMON_NGRAM_CACHE_LOAD_FOREIGN:
                // a valid cache of another model (or a newer version): overwriting it would destroy someone's data
                err = err_load + " - not overwriting it";
                return false;
        }
    }

    if (loaded) {
        common_ngram_cache_merge(merged, ngram_cache_delta);
    } else {
        merged = ngram_cache;
    }

    common_ngram_cache_prune(merged, n_max);

    if (!common_ngram_cache_save_file(merged, filename, vocab_id, err)) {
        return false;
    }

    ngram_cache = std::move(merged);
    ngram_cache_delta.clear();
    if (from_disk) {
        *from_disk = loaded;
    }
    return true;
}

void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename) {
    std::string err;
    if (!common_ngram_cache_save_file(ngram_cache, filename, common_ngram_cache_vocab_id{}, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
    }
}

common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    common_ngram_cache ngram_cache;
    std::string err;
    if (common_ngram_cache_load_file(filename, ngram_cache, common_ngram_cache_vocab_id{}, err) != COMMON_NGRAM_CACHE_LOAD_OK) {
        throw std::ifstream::failure(err);
    }
    return ngram_cache;
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add) {
    for (std::pair<common_ngram, common_ngram_cache_part> ngram_part : ngram_cache_add) {
        const common_ngram      ngram = ngram_part.first;
        common_ngram_cache_part  part = ngram_part.second;

        common_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            common_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
