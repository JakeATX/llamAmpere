#include "server-prompt-cache-disk.h"
#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

#ifdef _WIN32
#include <process.h>
static int pc_getpid() { return _getpid(); }
#else
#include <unistd.h>
static int pc_getpid() { return (int) getpid(); }
#endif

namespace fs = std::filesystem;

static const char     PC_MAGIC[4] = {'L', 'C', 'P', 'C'};
static const uint32_t PC_VERSION  = 2;

static constexpr uint64_t PC_MAX_KEY_LEN  = 1u << 20;
static constexpr uint64_t PC_MAX_N_TOKENS = 1u << 26;
static constexpr uint32_t PC_MAX_N_CKPT   = 1u << 16;

// word-wise FNV-1a: 8 bytes per multiply, so hashing a multi-GiB entry costs a fraction of writing it
struct pc_hasher {
    uint64_t h = 0xcbf29ce484222325ull;
    uint8_t  buf[8];
    size_t   nb = 0;

    static constexpr uint64_t PRIME = 0x100000001b3ull;

    void word(const uint8_t * p) {
        uint64_t w;
        memcpy(&w, p, 8);
        h ^= w;
        h *= PRIME;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        if (nb > 0) {
            while (nb < 8 && n > 0) {
                buf[nb++] = *p++;
                --n;
            }
            if (nb < 8) {
                return;
            }
            word(buf);
            nb = 0;
        }
        while (n >= 8) {
            word(p);
            p += 8;
            n -= 8;
        }
        while (n > 0) {
            buf[nb++] = *p++;
            --n;
        }
    }

    uint64_t final() const {
        uint64_t r = h;
        for (size_t i = 0; i < nb; ++i) {
            r ^= buf[i];
            r *= PRIME;
        }
        return r;
    }
};

uint64_t server_prompt_disk_hash(const void * data, size_t n, uint64_t seed) {
    pc_hasher hs;
    hs.h = seed;
    hs.update(data, n);
    return hs.final();
}

struct pc_writer {
    FILE *    f  = nullptr;
    pc_hasher hs;
    bool      ok = true;

    void bytes(const void * p, size_t n) {
        if (!ok || n == 0) {
            return;
        }
        if (fwrite(p, 1, n, f) != n) {
            ok = false;
            return;
        }
        hs.update(p, n);
    }

    template <typename T> void val(const T & v) { bytes(&v, sizeof(T)); }

    void blob(const std::vector<uint8_t> & v) {
        val<uint64_t>(v.size());
        bytes(v.data(), v.size());
    }
};

struct pc_reader {
    FILE *    f  = nullptr;
    pc_hasher hs;
    uint64_t  remaining = 0; // bytes left in the file
    bool      ok = true;

    bool bytes(void * p, size_t n) {
        if (!ok) {
            return false;
        }
        if (n > remaining || (n > 0 && fread(p, 1, n, f) != n)) {
            ok = false;
            return false;
        }
        remaining -= n;
        hs.update(p, n);
        return true;
    }

    template <typename T> bool val(T & v) { return bytes(&v, sizeof(T)); }

    bool blob(std::vector<uint8_t> & v) {
        uint64_t n = 0;
        if (!val(n) || n > remaining) {
            ok = false;
            return false;
        }
        v.resize(n);
        return bytes(v.data(), n);
    }
};

static int64_t pc_mtime_ns(const std::string & file) {
    std::error_code ec;
    const auto t = fs::last_write_time(file, ec);
    if (ec) {
        return 0;
    }
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

static bool pc_starts_with(const std::string & s, const char * p) {
    return s.rfind(p, 0) == 0;
}

static bool pc_ends_with(const std::string & s, const char * p) {
    const size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

static size_t pc_common_prefix(const llama_tokens & a, const llama_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

size_t server_prompt_disk_tier::size() const {
    size_t res = 0;
    for (const auto & e : entries) {
        res += e.size;
    }
    return res;
}

size_t server_prompt_disk_tier::n_match() const {
    size_t res = 0;
    for (const auto & e : entries) {
        res += e.match ? 1 : 0;
    }
    return res;
}

bool server_prompt_disk_tier::open(const std::string & dir_in, const std::string & key_in, size_t limit_bytes, std::string & err) {
    dir.clear();
    entries.clear();

    if (dir_in.empty()) {
        err = "no directory";
        return false;
    }

    std::error_code ec;
    fs::create_directories(dir_in, ec);
    if (ec || !fs::is_directory(dir_in, ec)) {
        err = "cannot create directory " + dir_in + (ec ? ": " + ec.message() : "");
        return false;
    }

    // write probe: a directory that exists but cannot be written would fail on every spill
    {
        const std::string probe = (fs::path(dir_in) / (".pc-probe-" + std::to_string(pc_getpid()) + ".tmp")).string();
        FILE * f = fopen(probe.c_str(), "wb");
        bool ok = f != nullptr;
        if (f) {
            ok = fwrite("p", 1, 1, f) == 1;
            ok = (fclose(f) == 0) && ok;
        }
        fs::remove(probe, ec);
        if (!ok) {
            err = "directory " + dir_in + " is not writable";
            return false;
        }
    }

    dir      = dir_in;
    key      = key_in;
    key_hash = server_prompt_disk_hash(key.data(), key.size());
    limit    = limit_bytes;

    if (!space_fn) {
        space_fn = [](const std::string & d, uint64_t & capacity, uint64_t & available) {
            std::error_code ec2;
            const auto si = fs::space(d, ec2);
            if (ec2) {
                return false;
            }
            capacity  = si.capacity;
            available = si.available;
            return true;
        };
    }

    scan(true);

    if (limit > 0) {
        while (!entries.empty() && size() > limit) {
            remove(entries.begin());
        }
    }

    return true;
}

bool server_prompt_disk_tier::parse_header(const std::string & file, server_prompt_disk_entry & e, bool & bad) const {
    bad = false;

    std::error_code ec;
    const uint64_t fsize = fs::file_size(file, ec);
    if (ec) {
        return false; // vanished or unreadable: not ours to delete
    }

    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }

    char     magic[4];
    uint32_t version  = 0;
    uint64_t khash    = 0;
    uint32_t key_len  = 0;
    uint64_t n_tokens = 0;
    uint64_t payload  = 0;

    f.read(magic, 4);
    f.read((char *) &version, sizeof(version));
    if (!f || memcmp(magic, PC_MAGIC, 4) != 0 || version != PC_VERSION) {
        bad = true; // includes v1 files: they carry no model/KV key and can never be restored safely
        return false;
    }

    f.read((char *) &khash, sizeof(khash));
    f.read((char *) &key_len, sizeof(key_len));
    if (!f || key_len > PC_MAX_KEY_LEN) {
        bad = true;
        return false;
    }

    std::string fkey(key_len, '\0');
    f.read(fkey.data(), key_len);
    f.read((char *) &n_tokens, sizeof(n_tokens));
    if (!f || n_tokens > PC_MAX_N_TOKENS) {
        bad = true;
        return false;
    }

    const bool match = khash == key_hash && fkey == key;

    llama_tokens tokens;
    if (match) {
        tokens.resize(n_tokens);
        f.read((char *) tokens.data(), n_tokens * sizeof(llama_token));
    } else {
        f.seekg((std::streamoff) (n_tokens * sizeof(llama_token)), std::ios::cur);
    }
    f.read((char *) &payload, sizeof(payload));
    if (!f) {
        bad = true;
        return false;
    }

    const uint64_t header = 4 + 4 + 8 + 4 + (uint64_t) key_len + 8 + n_tokens * sizeof(llama_token) + 8;
    if (header + payload + 8 != fsize) {
        bad = true; // truncated or trailing garbage
        return false;
    }

    e.file   = file;
    e.tokens = std::move(tokens);
    e.size   = fsize;
    e.mtime  = pc_mtime_ns(file);
    e.match  = match;

    return true;
}

void server_prompt_disk_tier::scan(bool initial) {
    std::error_code ec;

    if (!initial) {
        for (auto it = entries.begin(); it != entries.end();) {
            if (!fs::exists(it->file, ec)) {
                it = entries.erase(it); // removed by another server sharing the directory
            } else {
                ++it;
            }
        }
    }

    const auto now = fs::file_time_type::clock::now();

    for (fs::directory_iterator di(dir, ec), end; !ec && di != end; di.increment(ec)) {
        const auto & de = *di;
        std::error_code ec2;
        if (!de.is_regular_file(ec2)) {
            continue;
        }

        const std::string name = de.path().filename().string();
        const std::string path = de.path().string();

        if (pc_starts_with(name, ".pc-") && pc_ends_with(name, ".tmp")) {
            // a writer that crashed (or was killed) mid-write leaves its temp file behind
            const auto t = fs::last_write_time(de.path(), ec2);
            if (initial && !ec2 && now - t > std::chrono::seconds(tmp_stale_s)) {
                SRV_WRN("prompt cache disk tier: removing stale partial entry %s\n", path.c_str());
                fs::remove(de.path(), ec2);
                n_removed_bad++;
            }
            continue;
        }

        if (!pc_starts_with(name, "pc-") || !pc_ends_with(name, ".bin")) {
            continue;
        }

        if (!initial && std::any_of(entries.begin(), entries.end(), [&](const auto & e) { return e.file == path; })) {
            continue;
        }

        server_prompt_disk_entry e;
        bool bad = false;
        if (parse_header(path, e, bad)) {
            entries.push_back(std::move(e));
        } else if (bad) {
            SRV_WRN("prompt cache disk tier: removing corrupt or outdated entry %s\n", path.c_str());
            fs::remove(de.path(), ec2);
            n_removed_bad++;
        }
    }

    // FIFO order = write order, rebuilt from the file modification times
    entries.sort([](const server_prompt_disk_entry & a, const server_prompt_disk_entry & b) {
        return a.mtime != b.mtime ? a.mtime < b.mtime : a.file < b.file;
    });
}

void server_prompt_disk_tier::remove(std::list<server_prompt_disk_entry>::iterator it) {
    std::error_code ec;
    fs::remove(it->file, ec);
    entries.erase(it);
}

void server_prompt_disk_tier::enforce_limit() {
    scan(false);

    if (limit == 0) {
        return;
    }

    while (!entries.empty() && size() > limit) {
        SRV_INF("prompt cache disk tier: limit %.1f MiB reached, removing oldest entry (%.1f MiB%s)\n",
                limit / (1024.0 * 1024.0), entries.front().size / (1024.0 * 1024.0), entries.front().match ? "" : ", other model/config");
        remove(entries.begin());
    }
}

bool server_prompt_disk_tier::guard_allows(uint64_t need) {
    uint64_t capacity = 0, available = 0;
    if (!space_fn || !space_fn(dir, capacity, available)) {
        return true; // cannot query: the write itself will fail cleanly if the disk is full
    }

    const uint64_t reserve = std::max<uint64_t>((uint64_t) (guard_frac * (double) capacity), guard_abs);

    if (available < need + reserve) {
        n_guard_skips++;
        if (!guard_active) {
            SRV_WRN("prompt cache disk tier: skipping writes, %.1f GiB free of %.1f GiB (keeping at least %.1f GiB free: max(%.0f%%, %.0f GiB))\n",
                    available / (1024.0*1024.0*1024.0), capacity / (1024.0*1024.0*1024.0), reserve / (1024.0*1024.0*1024.0),
                    guard_frac * 100.0, guard_abs / (1024.0*1024.0*1024.0));
            guard_active = true;
        }
        return false;
    }

    if (guard_active) {
        SRV_INF("prompt cache disk tier: free space recovered (%.1f GiB free), resuming writes\n", available / (1024.0*1024.0*1024.0));
        guard_active = false;
    }

    return true;
}

bool server_prompt_disk_tier::write(const llama_tokens & tokens,
                                    const std::vector<uint8_t> & main,
                                    const std::vector<uint8_t> & drft,
                                    const std::list<common_prompt_checkpoint> & checkpoints) {
    if (!enabled() || tokens.empty() || main.empty()) {
        return false;
    }

    // already on disk (this prompt is a prefix of, or equal to, a stored one)
    for (const auto & e : entries) {
        if (e.match && pc_common_prefix(e.tokens, tokens) == tokens.size()) {
            SRV_TRC("prompt cache disk tier: prompt with %zu tokens is already on disk\n", tokens.size());
            return true;
        }
    }

    uint64_t payload = 8 + main.size() + 8 + drft.size() + 4;
    for (const auto & c : checkpoints) {
        payload += 8 + 4 + 4 + 4 + 8 + c.data_tgt.size() + 8 + c.data_dft.size() + 8 + c.data_spec.size();
    }
    const uint64_t header = 4 + 4 + 8 + 4 + key.size() + 8 + tokens.size() * sizeof(llama_token) + 8;
    const uint64_t total  = header + payload + 8;

    if (limit > 0 && total > limit) {
        SRV_WRN("prompt cache disk tier: entry of %.1f MiB exceeds the disk limit of %.1f MiB, not writing\n",
                total / (1024.0 * 1024.0), limit / (1024.0 * 1024.0));
        return false;
    }

    if (!guard_allows(total)) {
        return false;
    }

    // make room first, so the directory never exceeds the limit
    scan(false);
    if (limit > 0) {
        while (!entries.empty() && size() + total > limit) {
            SRV_INF("prompt cache disk tier: making room, removing oldest entry (%.1f MiB%s)\n",
                    entries.front().size / (1024.0 * 1024.0), entries.front().match ? "" : ", other model/config");
            remove(entries.begin());
        }
    }

    const int64_t t_start = ggml_time_us();

    const int      pid  = pc_getpid();
    const uint64_t s    = seq++;
    const auto     t_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    char name[96];
    snprintf(name, sizeof(name), "pc-%016" PRIx64 "-%d-%" PRIu64 ".bin", (uint64_t) t_us, pid, s);
    const std::string file = (fs::path(dir) / name).string();
    const std::string tmp  = (fs::path(dir) / (std::string(".") + name + ".tmp")).string();

    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        SRV_WRN("prompt cache disk tier: cannot open %s for writing\n", tmp.c_str());
        return false;
    }
    std::vector<char> iobuf(4u << 20);
    setvbuf(f, iobuf.data(), _IOFBF, iobuf.size());

    pc_writer w;
    w.f = f;
    w.bytes(PC_MAGIC, 4);
    w.val<uint32_t>(PC_VERSION);
    w.val<uint64_t>(key_hash);
    w.val<uint32_t>((uint32_t) key.size());
    w.bytes(key.data(), key.size());
    w.val<uint64_t>(tokens.size());
    w.bytes(tokens.data(), tokens.size() * sizeof(llama_token));
    w.val<uint64_t>(payload);
    w.blob(main);
    w.blob(drft);
    w.val<uint32_t>((uint32_t) checkpoints.size());
    for (const auto & c : checkpoints) {
        w.val<int64_t>(c.n_tokens);
        w.val<int32_t>(c.id_task);
        w.val<int32_t>(c.pos_min);
        w.val<int32_t>(c.pos_max);
        w.blob(c.data_tgt);
        w.blob(c.data_dft);
        w.blob(c.data_spec);
    }
    const uint64_t checksum = w.hs.final();
    if (w.ok && fwrite(&checksum, 1, sizeof(checksum), f) != sizeof(checksum)) {
        w.ok = false;
    }
    const bool closed = fclose(f) == 0;

    std::error_code ec;
    if (!w.ok || !closed) {
        SRV_WRN("prompt cache disk tier: failed to write %s\n", tmp.c_str());
        fs::remove(tmp, ec);
        return false;
    }

    fs::rename(tmp, file, ec);
    if (ec) {
        SRV_WRN("prompt cache disk tier: failed to rename %s: %s\n", tmp.c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return false;
    }

    // superseded: stored prompts that are a prefix of this one
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->match && pc_common_prefix(it->tokens, tokens) == it->tokens.size()) {
            SRV_TRC("prompt cache disk tier: removing superseded entry with %zu tokens\n", it->tokens.size());
            auto cur = it++;
            remove(cur);
        } else {
            ++it;
        }
    }

    server_prompt_disk_entry e;
    e.file   = file;
    e.tokens = tokens;
    e.size   = total;
    e.mtime  = pc_mtime_ns(file);
    e.match  = true;
    entries.push_back(std::move(e));

    SRV_INF("prompt cache disk tier: wrote %zu tokens, %zu checkpoints, %.1f MiB in %.1f ms (%zu entries, %.1f MiB on disk)\n",
            tokens.size(), checkpoints.size(), total / (1024.0 * 1024.0), (ggml_time_us() - t_start) / 1000.0,
            entries.size(), size() / (1024.0 * 1024.0));

    return true;
}

bool server_prompt_disk_tier::read(std::list<server_prompt_disk_entry>::iterator it, server_prompt_disk_record & out) {
    const std::string file = it->file;

    std::error_code ec;
    const uint64_t fsize = fs::file_size(file, ec);

    FILE * f = ec ? nullptr : fopen(file.c_str(), "rb");
    if (!f) {
        SRV_WRN("prompt cache disk tier: cannot open %s\n", file.c_str());
        entries.erase(it); // gone (another server evicted it)
        return false;
    }
    std::vector<char> iobuf(4u << 20);
    setvbuf(f, iobuf.data(), _IOFBF, iobuf.size());

    pc_reader r;
    r.f = f;
    r.remaining = fsize;

    const char * why     = nullptr;
    bool         foreign = false;

    char        magic[4];
    uint32_t    version  = 0;
    uint64_t    khash    = 0;
    uint32_t    key_len  = 0;
    uint64_t    n_tokens = 0;
    uint64_t    payload  = 0;
    std::string fkey;

    if (!r.bytes(magic, 4) || memcmp(magic, PC_MAGIC, 4) != 0 || !r.val(version) || version != PC_VERSION) {
        why = "bad magic or version";
    } else if (!r.val(khash) || !r.val(key_len) || key_len > PC_MAX_KEY_LEN) {
        why = "bad key";
    } else {
        fkey.resize(key_len);
        if (!r.bytes(fkey.data(), key_len)) {
            why = "bad key";
        } else if (khash != key_hash || fkey != key) {
            why     = "key mismatch";
            foreign = true;
        } else if (!r.val(n_tokens) || n_tokens > PC_MAX_N_TOKENS) {
            why = "bad token count";
        } else {
            out.tokens.resize(n_tokens);
            if (!r.bytes(out.tokens.data(), n_tokens * sizeof(llama_token)) || !r.val(payload) || payload + 8 != r.remaining) {
                why = "truncated";
            }
        }
    }

    uint32_t n_ckpt = 0;
    if (!why && (!r.blob(out.main) || !r.blob(out.drft) || !r.val(n_ckpt) || n_ckpt > PC_MAX_N_CKPT)) {
        why = "truncated state";
    }

    out.checkpoints.clear();
    for (uint32_t i = 0; !why && i < n_ckpt; ++i) {
        common_prompt_checkpoint c;
        if (!r.val(c.n_tokens) || !r.val(c.id_task) || !r.val(c.pos_min) || !r.val(c.pos_max) ||
            !r.blob(c.data_tgt) || !r.blob(c.data_dft) || !r.blob(c.data_spec)) {
            why = "truncated checkpoint";
            break;
        }
        out.checkpoints.push_back(std::move(c));
    }

    if (!why) {
        const uint64_t expected = r.hs.final();
        uint64_t stored = 0;
        if (r.remaining != 8 || fread(&stored, 1, 8, f) != 8) {
            why = "truncated checksum";
        } else if (stored != expected) {
            why = "checksum mismatch";
        }
    }

    fclose(f);

    if (foreign) {
        // replaced by another server's entry with the same name; never ours to restore or delete
        SRV_WRN("prompt cache disk tier: %s belongs to another model or KV configuration, not restoring\n", file.c_str());
        out = server_prompt_disk_record();
        it->match = false;
        it->tokens.clear();
        return false;
    }

    if (why) {
        SRV_WRN("prompt cache disk tier: removing unreadable entry %s (%s)\n", file.c_str(), why);
        out = server_prompt_disk_record();
        n_removed_bad++;
        remove(it);
        return false;
    }

    return true;
}
