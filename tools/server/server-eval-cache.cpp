#include "server-eval-cache.h"
#include "server-prompt-cache-disk.h" // server_prompt_disk_hash
#include "log.h"

#include <nlohmann/json.hpp>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static const char     EVC_MAGIC[4]   = {'L', 'E', 'V', 'C'};
static constexpr uint64_t EVC_MAX_KEY_LEN  = 1u << 20;
static constexpr uint64_t EVC_MAX_N_TOKENS = 1u << 26;
static constexpr uint32_t EVC_MAX_N_CKPT   = 1u << 16;
static constexpr uint64_t EVC_MAX_BLOB     = 1ull << 36;

static const char * EVC_INDEX = "eval-cache.json";

namespace {

struct evc_buf {
    std::vector<uint8_t> b;

    void bytes(const void * p, size_t n) {
        const uint8_t * q = (const uint8_t *) p;
        b.insert(b.end(), q, q + n);
    }
    template <typename T> void val(const T & v) { bytes(&v, sizeof(T)); }
    void blob(const std::vector<uint8_t> & v) {
        val<uint64_t>(v.size());
        bytes(v.data(), v.size());
    }
};

struct evc_reader {
    std::ifstream f;
    uint64_t      remaining = 0;
    uint64_t      h         = 0xcbf29ce484222325ull;
    bool          hash      = true;
    bool          ok        = true;

    // the checksum is a chain of per-read FNV-1a hashes: cheap to compute while streaming, order sensitive
    bool bytes(void * p, size_t n) {
        if (!ok || n > remaining) {
            ok = false;
            return false;
        }
        if (n > 0) {
            f.read((char *) p, (std::streamsize) n);
            if (!f) {
                ok = false;
                return false;
            }
        }
        remaining -= n;
        if (hash) {
            h = server_prompt_disk_hash(p, n, h);
        }
        return true;
    }
    template <typename T> bool val(T & v) { return bytes(&v, sizeof(T)); }
    bool blob(std::vector<uint8_t> & v) {
        uint64_t n = 0;
        if (!val(n) || n > remaining || n > EVC_MAX_BLOB) {
            ok = false;
            return false;
        }
        v.resize(n);
        return bytes(v.data(), n);
    }
};

// same chaining as evc_reader, applied to a buffer written in the same pieces
struct evc_chain {
    uint64_t h = 0xcbf29ce484222325ull;
    void add(const void * p, size_t n) { h = server_prompt_disk_hash(p, n, h); }
};

} // namespace

std::string server_eval_cache::entry_id(const llama_tokens & tokens) {
    const uint64_t h = server_prompt_disk_hash(tokens.data(), tokens.size() * sizeof(llama_token));
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

std::string server_eval_cache::path_hdr(const std::string & id) const { return dir + "/entries/" + id + ".evh"; }
std::string server_eval_cache::path_tgt(const std::string & id) const { return dir + "/entries/" + id + ".tgt"; }
std::string server_eval_cache::path_dft(const std::string & id) const { return dir + "/entries/" + id + ".dft"; }

uint64_t server_eval_cache::total_bytes() const {
    uint64_t res = 0;
    for (const auto & e : entries) {
        res += e.bytes;
    }
    return res;
}

int server_eval_cache::find(const llama_tokens & tokens, size_t n_min) const {
    int    best   = -1;
    size_t best_n = n_min;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & et = entries[i].tokens;
        if (et.size() <= best_n || et.size() > tokens.size()) {
            continue;
        }
        if (std::equal(et.begin(), et.end(), tokens.begin())) {
            best   = (int) i;
            best_n = et.size();
        }
    }
    return best;
}

// parse the header up to the token list; `full` also reads checkpoints + speculative state and verifies the checksum
static bool evc_parse(const std::string & file, uint64_t key_hash, const std::string & key, bool full,
        server_eval_cache_record & out, std::string & err) {
    evc_reader r;
    r.f.open(file, std::ios::binary);
    if (!r.f) {
        err = "cannot open " + file;
        return false;
    }
    std::error_code ec;
    r.remaining = fs::file_size(file, ec);
    if (ec) {
        err = "cannot stat " + file;
        return false;
    }

    char     magic[4];
    uint32_t version = 0;
    uint64_t khash   = 0;
    uint32_t klen    = 0;
    if (!r.bytes(magic, 4) || memcmp(magic, EVC_MAGIC, 4) != 0 || !r.val(version)) {
        err = file + " is not an eval-cache entry";
        return false;
    }
    if (version != EVAL_CACHE_FORMAT_VERSION) {
        err = file + string_format(" has format version %u, this server reads version %d", version, EVAL_CACHE_FORMAT_VERSION);
        return false;
    }
    if (!r.val(khash) || !r.val(klen) || klen > EVC_MAX_KEY_LEN) {
        err = file + " has a truncated header";
        return false;
    }
    std::string k(klen, '\0');
    if (!r.bytes(k.data(), klen)) {
        err = file + " has a truncated header";
        return false;
    }
    if (khash != key_hash || k != key) {
        err = file + " was built for another model or KV configuration";
        return false;
    }

    uint64_t n_tokens = 0;
    if (!r.val(n_tokens) || n_tokens > EVC_MAX_N_TOKENS || n_tokens == 0) {
        err = file + " has a bad token count";
        return false;
    }
    out.tokens.resize(n_tokens);
    if (!r.bytes(out.tokens.data(), n_tokens * sizeof(llama_token))) {
        err = file + " is truncated";
        return false;
    }

    if (!full) {
        return true;
    }

    uint32_t n_ckpt = 0;
    if (!r.val(n_ckpt) || n_ckpt > EVC_MAX_N_CKPT) {
        err = file + " is truncated";
        return false;
    }
    out.checkpoints.clear();
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        common_prompt_checkpoint c;
        int64_t n = 0;
        int32_t pmin = 0;
        int32_t pmax = 0;
        if (!r.val(n) || !r.val(pmin) || !r.val(pmax) || !r.blob(c.data_tgt) || !r.blob(c.data_dft) || !r.blob(c.data_spec)) {
            err = file + " has a truncated checkpoint";
            return false;
        }
        c.n_tokens = n;
        c.pos_min  = pmin;
        c.pos_max  = pmax;
        out.checkpoints.push_back(std::move(c));
    }
    uint8_t has_dft = 0;
    if (!r.blob(out.spec) || !r.val(has_dft)) {
        err = file + " is truncated";
        return false;
    }
    out.has_dft = has_dft != 0;

    const uint64_t h = r.h;
    r.hash = false;
    uint64_t checksum = 0;
    if (!r.val(checksum) || checksum != h || r.remaining != 0) {
        err = file + " failed its checksum";
        return false;
    }
    return true;
}

bool server_eval_cache::read_header(const std::string & id, server_eval_cache_record & out, std::string & err) const {
    return evc_parse(path_hdr(id), key_hash, key, true, out, err);
}

bool server_eval_cache::write_index(std::string & err) const {
    json j;
    j["format"]         = "llamampere-eval-cache";
    j["format_version"] = EVAL_CACHE_FORMAT_VERSION;
    j["key_hash"]       = string_format("%016" PRIx64, key_hash);
    j["key"]            = key;
    json arr = json::array();
    for (const auto & e : entries) {
        arr.push_back({
            {"id",       e.id},
            {"n_tokens", e.tokens.size()},
            {"bytes",    e.bytes},
            {"has_dft",  e.has_dft},
        });
    }
    j["entries"] = arr;

    const std::string tmp = dir + "/." + EVC_INDEX + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << j.dump(1) << "\n";
        f.close();
        if (!f) {
            err = "cannot write " + tmp;
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, dir + "/" + EVC_INDEX, ec);
    if (ec) {
        err = "cannot rename " + tmp + ": " + ec.message();
        return false;
    }
    return true;
}

bool server_eval_cache::open(const std::string & dir_in, const std::string & key_in, bool writable_in, std::string & err) {
    dir      = dir_in;
    key      = key_in;
    key_hash = server_prompt_disk_hash(key.data(), key.size());
    writable = writable_in;
    entries.clear();
    n_skipped = 0;

    while (dir.size() > 1 && dir.back() == '/') {
        dir.pop_back();
    }

    std::error_code ec;
    const std::string index = dir + "/" + EVC_INDEX;

    if (!fs::exists(index, ec)) {
        if (!writable) {
            err = "no " + std::string(EVC_INDEX) + " in " + dir + " (build the folder with tools/eval-cache/llama-eval-cache.py build)";
            dir.clear();
            return false;
        }
        fs::create_directories(dir + "/entries", ec);
        if (ec) {
            err = "cannot create " + dir + "/entries: " + ec.message();
            dir.clear();
            return false;
        }
        if (!write_index(err)) {
            dir.clear();
            return false;
        }
        return true;
    }

    json j;
    try {
        std::ifstream f(index);
        j = json::parse(f);
    } catch (const std::exception & e) {
        err = index + " does not parse: " + e.what();
        dir.clear();
        return false;
    }

    const int ver = j.value("format_version", -1);
    if (j.value("format", std::string()) != "llamampere-eval-cache" || ver != EVAL_CACHE_FORMAT_VERSION) {
        err = index + string_format(" has format version %d, this server reads version %d; rebuild the folder", ver, EVAL_CACHE_FORMAT_VERSION);
        dir.clear();
        if (!writable && j.value("format", std::string()) == "llamampere-eval-cache" && ver != EVAL_CACHE_FORMAT_VERSION) {
            LOG_WRN("eval cache: %s; skipping read-only cache, using fresh prefill\n", err.c_str());
            err.clear();
            return true;
        }
        return false;
    }

    const std::string key_dir = j.value("key", std::string());
    if (key_dir != key) {
        // name the first differing field so the user can see which flag differs
        std::string diff;
        size_t a = 0;
        size_t b = 0;
        while (a < key.size() || b < key_dir.size()) {
            const size_t ea = key.find(';', a);
            const size_t eb = key_dir.find(';', b);
            const std::string fa = key.substr(a, ea == std::string::npos ? std::string::npos : ea - a);
            const std::string fb = key_dir.substr(b, eb == std::string::npos ? std::string::npos : eb - b);
            if (fa != fb) {
                diff = "this server has '" + fa + "', the folder was built with '" + fb + "'";
                break;
            }
            if (ea == std::string::npos || eb == std::string::npos) {
                break;
            }
            a = ea + 1;
            b = eb + 1;
        }
        err = dir + " was built for another model or KV configuration (" + (diff.empty() ? std::string("keys differ") : diff) +
              "); start the server with the settings the folder was built with, or build a folder for these settings";
        dir.clear();
        return false;
    }

    if (j.contains("entries") && j["entries"].is_array()) {
        for (const auto & je : j["entries"]) {
            server_eval_cache_entry e;
            e.id      = je.value("id", std::string());
            e.has_dft = je.value("has_dft", false);
            if (e.id.empty()) {
                ++n_skipped;
                continue;
            }

            server_eval_cache_record rec;
            std::string perr;
            if (!evc_parse(path_hdr(e.id), key_hash, key, false, rec, perr)) {
                LOG_WRN("eval cache: skipping entry %s: %s\n", e.id.c_str(), perr.c_str());
                ++n_skipped;
                continue;
            }
            if (!fs::exists(path_tgt(e.id), ec) || (e.has_dft && !fs::exists(path_dft(e.id), ec))) {
                LOG_WRN("eval cache: skipping entry %s: state file missing\n", e.id.c_str());
                ++n_skipped;
                continue;
            }
            e.tokens = std::move(rec.tokens);
            e.bytes  = fs::file_size(path_hdr(e.id), ec) + fs::file_size(path_tgt(e.id), ec) +
                       (e.has_dft ? fs::file_size(path_dft(e.id), ec) : 0);
            entries.push_back(std::move(e));
        }
    }

    if (writable) {
        fs::create_directories(dir + "/entries", ec);
    }

    return true;
}

bool server_eval_cache::add(const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints,
        const std::vector<uint8_t> & spec, bool has_dft, std::string & id_out, std::string & err) {
    if (!writable) {
        err = "the eval cache is read-only (start the server with --eval-cache-build to add entries)";
        return false;
    }

    const std::string id = entry_id(tokens);
    id_out = id;

    evc_buf   w;
    evc_chain hs;
    auto put = [&](const void * p, size_t n) {
        w.bytes(p, n);
        hs.add(p, n);
    };
    auto put_blob = [&](const std::vector<uint8_t> & v) {
        const uint64_t n = v.size();
        put(&n, sizeof(n));
        put(v.data(), v.size());
    };

    const uint32_t version = EVAL_CACHE_FORMAT_VERSION;
    const uint32_t klen    = (uint32_t) key.size();
    const uint64_t ntok    = tokens.size();
    const uint32_t nckpt   = (uint32_t) checkpoints.size();
    const uint8_t  hd      = has_dft ? 1 : 0;

    put(EVC_MAGIC, 4);
    put(&version, sizeof(version));
    put(&key_hash, sizeof(key_hash));
    put(&klen, sizeof(klen));
    put(key.data(), key.size());
    put(&ntok, sizeof(ntok));
    put(tokens.data(), tokens.size() * sizeof(llama_token));
    put(&nckpt, sizeof(nckpt));
    for (const auto & c : checkpoints) {
        const int64_t n    = c.n_tokens;
        const int32_t pmin = c.pos_min;
        const int32_t pmax = c.pos_max;
        put(&n, sizeof(n));
        put(&pmin, sizeof(pmin));
        put(&pmax, sizeof(pmax));
        put_blob(c.data_tgt);
        put_blob(c.data_dft);
        put_blob(c.data_spec);
    }
    put_blob(spec);
    put(&hd, sizeof(hd));
    w.val<uint64_t>(hs.h);

    const std::string file = path_hdr(id);
    const std::string tmp  = file + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write((const char *) w.b.data(), (std::streamsize) w.b.size());
        f.close();
        if (!f) {
            err = "cannot write " + tmp;
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, file, ec);
    if (ec) {
        err = "cannot rename " + tmp + ": " + ec.message();
        return false;
    }

    server_eval_cache_entry e;
    e.id      = id;
    e.tokens  = tokens;
    e.has_dft = has_dft;
    e.bytes   = fs::file_size(file, ec) + fs::file_size(path_tgt(id), ec) + (has_dft ? fs::file_size(path_dft(id), ec) : 0);

    bool replaced = false;
    for (auto & old : entries) {
        if (old.id == id) {
            old = e;
            replaced = true;
        }
    }
    if (!replaced) {
        entries.push_back(std::move(e));
    }

    return write_index(err);
}
