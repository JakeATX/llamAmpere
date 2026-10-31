// Unit tests of the eval-cache folder (--eval-cache DIR), no model needed:
//   folder creation in build mode and the refusal of a missing folder in read-only mode, entry header round trip
//   (tokens, checkpoints, speculative state), longest-prefix lookup, the refusal of a folder built for another key
//   (naming the differing field), the format-version check, and skipping of corrupt / incomplete entries.

#include "server-eval-cache.h"
#include "server-prompt-cache-disk.h"

#include "common.h"
#include "fit.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace allocation_probe {
static thread_local bool active = false;
static thread_local size_t bytes = 0;
static thread_local size_t hits = 0;
static thread_local size_t last = 0;
static thread_local size_t largest = 0;

static void start(size_t size) {
    bytes = size;
    hits = last = largest = 0;
    active = true;
}
static size_t stop() {
    active = false;
    return hits;
}
}

// Count the real vector/list requests, including requests later freed on failure.
void * operator new(size_t n) {
    if (allocation_probe::active) {
        allocation_probe::last = n;
        allocation_probe::largest = std::max(allocation_probe::largest, n);
        allocation_probe::hits += n == allocation_probe::bytes;
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }

static int n_fail = 0;
static int n_pass = 0;

#define CHECK(cond) do { \
    if (cond) { n_pass++; } else { n_fail++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

static const std::string KEY_A = "fmt=llamampere-eval-cache-v1;model=abc;ctk=q8_0;ctv=q8_0;fa=1";
static const std::string KEY_B = "fmt=llamampere-eval-cache-v1;model=abc;ctk=tq5_0;ctv=turbo4;fa=1";

static llama_tokens seq(int from, int n) {
    llama_tokens t(n);
    for (int i = 0; i < n; ++i) {
        t[i] = from + i;
    }
    return t;
}

static void touch(const std::string & path, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(n, 'x');
    f.write(buf.data(), (std::streamsize) buf.size());
}

static common_prompt_checkpoint ckpt(int64_t n, int fill) {
    common_prompt_checkpoint c;
    c.n_tokens = n;
    c.pos_min  = (llama_pos) n - 1;
    c.pos_max  = (llama_pos) n - 1;
    c.data_tgt.assign(1000 + fill, (uint8_t) fill);
    c.data_dft.assign(fill, (uint8_t) (fill + 1));
    c.data_spec.assign(3, (uint8_t) (fill + 2));
    return c;
}

static bool add_entry(server_eval_cache & ec, const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & ck,
        const std::vector<uint8_t> & spec, bool has_dft, std::string & id) {
    const std::string pid = server_eval_cache::entry_id(tokens);
    touch(ec.path_tgt(pid), 4096);
    if (has_dft) {
        touch(ec.path_dft(pid), 1024);
    }
    std::string err;
    const bool ok = ec.add(tokens, ck, spec, has_dft, id, err);
    if (!ok) {
        fprintf(stderr, "add: %s\n", err.c_str());
    }
    return ok;
}

static void test_open_and_roundtrip(const std::string & root) {
    const std::string dir = root + "/a";
    std::string err;

    // read-only open of a folder that was never built: refused, with a hint
    {
        server_eval_cache ec;
        CHECK(!ec.open(dir, KEY_A, false, err));
        CHECK(err.find("llama-eval-cache.py build") != std::string::npos);
        CHECK(!ec.enabled());
    }

    // build mode creates the folder and index
    std::string id_long;
    std::string id_short;
    {
        server_eval_cache ec;
        CHECK(ec.open(dir, KEY_A, true, err));
        CHECK(ec.enabled());
        CHECK(fs::exists(dir + "/eval-cache.json"));
        CHECK(fs::is_directory(dir + "/entries"));
        CHECK(ec.entries.empty());

        std::list<common_prompt_checkpoint> ck = { ckpt(40, 1), ckpt(80, 2) };
        CHECK(add_entry(ec, seq(1, 100), ck, {9, 8, 7}, true, id_long));
        CHECK(add_entry(ec, seq(1, 50), {}, {}, false, id_short));
        CHECK(ec.entries.size() == 2);
        CHECK(id_long == server_eval_cache::entry_id(seq(1, 100)));
        CHECK(id_long != id_short);
        CHECK(id_long.size() == 16);

        // re-adding the same prompt replaces the entry, it does not duplicate it
        std::string id_again;
        CHECK(add_entry(ec, seq(1, 50), {}, {}, false, id_again));
        CHECK(id_again == id_short);
        CHECK(ec.entries.size() == 2);
    }

    // read-only reopen with the same key sees both entries
    {
        server_eval_cache ec;
        CHECK(ec.open(dir, KEY_A, false, err));
        CHECK(ec.entries.size() == 2);
        CHECK(ec.n_skipped == 0);
        CHECK(ec.total_bytes() > 2 * 4096);

        // longest prefix wins
        const int i_long = ec.find(seq(1, 120), 0);
        CHECK(i_long >= 0 && ec.entries[i_long].id == id_long);
        // an entry longer than the prompt is not a prefix
        const int i_short = ec.find(seq(1, 60), 0);
        CHECK(i_short >= 0 && ec.entries[i_short].id == id_short);
        // nothing beats what the slot already has
        CHECK(ec.find(seq(1, 120), 100) == -1);
        CHECK(ec.find(seq(1, 120), 99) == i_long);
        // a prompt that differs inside the stored prefix does not match
        llama_tokens other = seq(1, 120);
        other[10] = 999999;
        CHECK(ec.find(other, 0) == -1);
        // exact length match is a prefix
        CHECK(ec.find(seq(1, 100), 0) == i_long);

        // header round trip
        server_eval_cache_record rec;
        CHECK(ec.read_header(id_long, rec, err));
        CHECK(rec.tokens == seq(1, 100));
        CHECK(rec.has_dft);
        CHECK(rec.spec == std::vector<uint8_t>({9, 8, 7}));
        CHECK(rec.checkpoints.size() == 2);
        if (rec.checkpoints.size() == 2) {
            const auto & c0 = rec.checkpoints.front();
            const auto & c1 = rec.checkpoints.back();
            const auto e0 = ckpt(40, 1);
            const auto e1 = ckpt(80, 2);
            CHECK(c0.n_tokens == 40 && c0.pos_min == e0.pos_min && c0.pos_max == e0.pos_max);
            CHECK(c0.data_tgt == e0.data_tgt && c0.data_dft == e0.data_dft && c0.data_spec == e0.data_spec);
            CHECK(c1.n_tokens == 80 && c1.data_tgt == e1.data_tgt);
        }

        // read-only folders cannot grow
        std::string id;
        CHECK(!ec.add(seq(5, 10), {}, {}, false, id, err));
    }

    // a server with other KV settings refuses the folder and names the differing field
    {
        server_eval_cache ec;
        CHECK(!ec.open(dir, KEY_B, false, err));
        CHECK(err.find("ctk=tq5_0") != std::string::npos);
        CHECK(err.find("ctk=q8_0") != std::string::npos);
        CHECK(!ec.enabled());
        // build mode refuses too: a folder holds one configuration
        server_eval_cache eb;
        CHECK(!eb.open(dir, KEY_B, true, err));
    }
}

static void test_record_ram_budget(const std::string & root) {
    server_eval_cache ec;
    std::string err, id;
    CHECK(ec.open(root + "/ram", KEY_A, true, err));
    std::list<common_prompt_checkpoint> checkpoints(128);
    CHECK(add_entry(ec, seq(1, 100), checkpoints, {1, 2, 3}, false, id));
    const uint64_t file_bytes = fs::file_size(ec.path_hdr(id));
    server_eval_cache_record rec;
    CHECK(ec.read_header(id, rec, err));
    CHECK(rec.checkpoints.size() == 128);
    CHECK(!ec.read_header(id, rec, err, 0));
    CHECK(err.find("RAM fit budget") != std::string::npos);
    CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());
    CHECK(!ec.read_header(id, rec, err, file_bytes - 1));
    CHECK(!ec.read_header(id, rec, err, file_bytes + 65536 + 4096));
    CHECK(err.find("checkpoints exceed") != std::string::npos);
    CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());
    CHECK(ec.read_header(id, rec, err, file_bytes + 1024 * 1024));
    CHECK(rec.tokens == seq(1, 100) && rec.checkpoints.size() == 128);
    CHECK(rec.spec == std::vector<uint8_t>({1, 2, 3}));

    // Calibrate the node size, then prove the node cap runs before list growth.
    allocation_probe::start(0);
    std::list<common_prompt_checkpoint> probe;
    probe.emplace_back();
    allocation_probe::stop();
    const size_t node_bytes = allocation_probe::last;
    CHECK(node_bytes >= sizeof(common_prompt_checkpoint));
    allocation_probe::start(node_bytes);
    const bool nodes_ok = ec.read_header(id, rec, err);
    const size_t nodes_seen = allocation_probe::stop();
    CHECK(nodes_ok && nodes_seen >= 128);
    allocation_probe::start(node_bytes);
    const bool nodes_rejected = !ec.read_header(id, rec, err, file_bytes + 65536 + 4096);
    const size_t rejected_nodes = allocation_probe::stop();
    CHECK(nodes_rejected && rejected_nodes == 0);
    CHECK(err.find("checkpoints exceed") != std::string::npos);
    for (uint64_t host_gib : {16, 32, 64, 128}) {
        const uint64_t budget = common_fit_eval_cache_ram_bytes((host_gib - 6) << 30, true);
        CHECK(ec.read_header(id, rec, err, budget));
    }
    CHECK(!ec.read_header(id, rec, err, common_fit_eval_cache_ram_bytes(5ull << 30, true)));
    CHECK(ec.read_header(id, rec, err, common_fit_eval_cache_ram_bytes(0, false)));

    // A sparse header must be rejected before token or checkpoint allocation.
    fs::resize_file(ec.path_hdr(id), 1ull << 36);
    CHECK(!ec.read_header(id, rec, err, 1024 * 1024));
    CHECK(err.find("RAM fit budget") != std::string::npos);
    CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());
    fs::resize_file(ec.path_hdr(id), file_bytes);
    CHECK(ec.read_header(id, rec, err));
}

static void test_record_allocations(const std::string & root) {
    server_eval_cache ec;
    std::string err, id;
    CHECK(ec.open(root + "/alloc", KEY_A, true, err));
    const size_t payload = 1u << 20;
    common_prompt_checkpoint c{};
    c.data_tgt.assign(payload, 7);
    CHECK(add_entry(ec, seq(1, payload / sizeof(llama_token)), {c}, std::vector<uint8_t>(payload, 8), false, id));
    server_eval_cache_record rec;
    allocation_probe::start(payload);
    const bool loaded = ec.read_header(id, rec, err);
    const size_t good_allocations = allocation_probe::stop();
    CHECK(loaded && good_allocations == 3); // tokens, checkpoint target and spec
    const uint64_t file_bytes = fs::file_size(ec.path_hdr(id));
    allocation_probe::start(payload);
    const bool rejected = !ec.read_header(id, rec, err, file_bytes - 1);
    const size_t rejected_allocations = allocation_probe::stop();
    CHECK(rejected && rejected_allocations == 0);
    CHECK(err.find("RAM fit budget") != std::string::npos);
    CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());

    // Keep a small real file; corrupt each declared payload size in turn.
    CHECK(add_entry(ec, {1}, {}, {2}, false, id));
    const auto token_offset = uint64_t(4 + 4 + 8 + 4 + KEY_A.size());
    const auto count_offset = token_offset + 8 + sizeof(llama_token);
    const auto blob_offset = count_offset + 4;
    auto overwrite = [&](uint64_t offset, const void * value, size_t n) {
        std::fstream f(ec.path_hdr(id), std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(std::streamoff(offset));
        f.write(static_cast<const char *>(value), std::streamsize(n));
        f.close();
        CHECK(bool(f));
    };
    const uint64_t bad_tokens = payload / sizeof(llama_token);
    const uint64_t bad_blob = payload;
    for (bool tokens : {true, false}) {
        CHECK(add_entry(ec, {1}, {}, {2}, false, id));
        overwrite(tokens ? token_offset : blob_offset, tokens ? &bad_tokens : &bad_blob, 8);
        allocation_probe::start(payload);
        const bool bad = !ec.read_header(id, rec, err);
        const size_t allocations = allocation_probe::stop();
        CHECK(bad && allocations == 0);
        CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());
    }
    CHECK(add_entry(ec, {1}, {}, {2}, false, id));
    const uint32_t bad_count = 128;
    overwrite(count_offset, &bad_count, sizeof(bad_count));
    CHECK(!ec.read_header(id, rec, err));
    CHECK(err.find("bad checkpoint count") != std::string::npos);

    // A token payload without the required footer must not allocate first.
    CHECK(add_entry(ec, seq(1, payload / sizeof(llama_token)), {}, {}, false, id));
    fs::resize_file(ec.path_hdr(id), token_offset + sizeof(uint64_t) + payload);
    allocation_probe::start(payload);
    const bool missing_tail = !ec.read_header(id, rec, err);
    const size_t tail_allocations = allocation_probe::stop();
    CHECK(missing_tail && tail_allocations == 0);
    CHECK(err.find("bad token count") != std::string::npos);
    CHECK(rec.tokens.empty() && rec.checkpoints.empty() && rec.spec.empty());
}

static void test_streamed_header(const std::string & root) {
    server_eval_cache ec;
    std::string err, id;
    CHECK(ec.open(root + "/stream", KEY_A, true, err));
    const llama_tokens tokens = {4, 5};
    const std::list<common_prompt_checkpoint> checkpoints = {ckpt(2, 3)};
    const std::vector<uint8_t> spec = {7, 8, 9};
    CHECK(add_entry(ec, tokens, checkpoints, spec, true, id));

    // Encode the existing wire layout in memory as a byte-for-byte control.
    std::vector<uint8_t> expected;
    uint64_t hash = 0xcbf29ce484222325ull;
    auto put = [&](const void * p, size_t n) {
        const auto * bytes = static_cast<const uint8_t *>(p);
        if (n) {
            expected.insert(expected.end(), bytes, bytes + n);
        }
        hash = server_prompt_disk_hash(p, n, hash);
    };
    const uint32_t version = EVAL_CACHE_FORMAT_VERSION, key_len = KEY_A.size(), count = 1;
    const uint64_t ntok = tokens.size();
    const uint8_t has_dft = 1;
    put("LEVC", 4);
    put(&version, sizeof(version));
    put(&ec.key_hash, sizeof(ec.key_hash));
    put(&key_len, sizeof(key_len));
    put(KEY_A.data(), KEY_A.size());
    put(&ntok, sizeof(ntok));
    put(tokens.data(), tokens.size() * sizeof(llama_token));
    put(&count, sizeof(count));
    const auto & c = checkpoints.front();
    const int64_t n = c.n_tokens;
    const int32_t pmin = c.pos_min, pmax = c.pos_max;
    put(&n, sizeof(n));
    put(&pmin, sizeof(pmin));
    put(&pmax, sizeof(pmax));
    for (const auto * blob : {&c.data_tgt, &c.data_dft, &c.data_spec, &spec}) {
        const uint64_t size = blob->size();
        put(&size, sizeof(size));
        put(blob->data(), blob->size());
    }
    put(&has_dft, sizeof(has_dft));
    const auto * checksum = reinterpret_cast<const uint8_t *>(&hash);
    expected.insert(expected.end(), checksum, checksum + sizeof(hash));
    std::ifstream f(ec.path_hdr(id), std::ios::binary);
    const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    CHECK(actual == expected);

    const size_t payload = 1u << 20;
    std::list<common_prompt_checkpoint> big(1);
    big.front().data_tgt.assign(payload, 6);
    big.front().data_dft.assign(payload, 7);
    big.front().data_spec.assign(payload, 8);
    const std::vector<uint8_t> big_spec(payload, 9);
    allocation_probe::start(0);
    const bool saved = ec.add(tokens, big, big_spec, true, id, err);
    allocation_probe::stop();
    CHECK(saved && allocation_probe::largest < payload);
    server_eval_cache_record rec;
    CHECK(ec.read_header(id, rec, err));
    CHECK(rec.tokens == tokens && rec.has_dft && rec.spec == big_spec);
    CHECK(rec.checkpoints.size() == 1);
    if (!rec.checkpoints.empty()) {
        CHECK(rec.checkpoints.front().data_tgt == big.front().data_tgt);
        CHECK(rec.checkpoints.front().data_dft == big.front().data_dft);
        CHECK(rec.checkpoints.front().data_spec == big.front().data_spec);
    }

    const std::string index_path = ec.dir + "/eval-cache.json";
    auto index_bytes = [&]() {
        std::ifstream file(index_path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    const std::string before_index = index_bytes();
    CHECK(!before_index.empty());
    // An open failure must leave the existing entry and its index intact.
    fs::create_directory(ec.path_hdr(id) + ".tmp");
    CHECK(!ec.add(tokens, checkpoints, spec, false, id, err));
    CHECK(index_bytes() == before_index);
    CHECK(ec.entries.size() == 1 && ec.entries.front().has_dft);
    CHECK(ec.read_header(id, rec, err) && rec.spec == big_spec && rec.has_dft);
#ifdef __linux__
    // /dev/full opens successfully but rejects writes; exercise failure after open.
    fs::remove(ec.path_hdr(id) + ".tmp");
    fs::create_symlink("/dev/full", ec.path_hdr(id) + ".tmp");
    {
        std::ofstream probe(ec.path_hdr(id) + ".tmp", std::ios::binary);
        CHECK(probe.is_open() && probe.good());
    }
    std::ifstream before_file(ec.path_hdr(id), std::ios::binary);
    const std::vector<uint8_t> before{std::istreambuf_iterator<char>(before_file),
                                     std::istreambuf_iterator<char>()};
    // The large payload forces the stream buffer to flush before close.
    CHECK(!ec.add(tokens, big, big_spec, false, id, err));
    CHECK(index_bytes() == before_index);
    std::ifstream after_file(ec.path_hdr(id), std::ios::binary);
    const std::vector<uint8_t> after{std::istreambuf_iterator<char>(after_file),
                                    std::istreambuf_iterator<char>()};
    CHECK(before == after && !before.empty());
    CHECK(ec.entries.size() == 1 && ec.entries.front().has_dft);
    CHECK(ec.read_header(id, rec, err) && rec.spec == big_spec && rec.has_dft);
    CHECK(rec.checkpoints.size() == 1);
    if (!rec.checkpoints.empty()) {
        CHECK(rec.checkpoints.front().data_tgt == big.front().data_tgt);
        CHECK(rec.checkpoints.front().data_dft == big.front().data_dft);
        CHECK(rec.checkpoints.front().data_spec == big.front().data_spec);
    }
    fs::remove(ec.path_hdr(id) + ".tmp");
#endif
}

static void test_borrowed_checkpoints(const std::string & root) {
    server_eval_cache ec;
    std::string err, id;
    CHECK(ec.open(root + "/borrowed", KEY_A, true, err));
    const llama_tokens tokens = seq(1, 8);
    const std::vector<uint8_t> spec = {41, 42};
    const size_t payload = 1u << 20;
    std::list<common_prompt_checkpoint> checkpoints;
    // Deliberately unordered: selection must not stop at the first excluded entry.
    for (const int64_t n : {5, 10, 7, 11}) {
        checkpoints.push_back(ckpt(n, (int) n));
        checkpoints.back().data_tgt.assign(payload, (uint8_t) n);
        checkpoints.back().data_dft.assign(payload, (uint8_t) (n + 1));
        checkpoints.back().data_spec.assign(payload, (uint8_t) (n + 2));
    }
    CHECK(add_entry(ec, tokens, {}, spec, true, id));
    auto header_bytes = [&]() {
        std::ifstream f(ec.path_hdr(id), std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    for (const int64_t limit : {int64_t(0), int64_t(5), int64_t(7), INT64_MAX}) {
        // Reproduce the old caller's deep-copy selection as a positive allocation control.
        std::list<common_prompt_checkpoint> selected;
        allocation_probe::start(payload);
        for (const auto & c : checkpoints) {
            if (c.n_tokens <= limit) {
                selected.push_back(c);
            }
        }
        const size_t copies = allocation_probe::stop();
        CHECK(copies == 3 * selected.size());
        CHECK(ec.add(tokens, selected, spec, true, id, err));
        const std::string expected = header_bytes();

        allocation_probe::start(payload);
        const bool saved = ec.add(tokens, checkpoints, spec, true, id, err, limit);
        const size_t borrowed_copies = allocation_probe::stop();
        CHECK(saved && borrowed_copies == 0 && allocation_probe::largest < payload);
        CHECK(header_bytes() == expected);
        server_eval_cache_record rec;
        CHECK(ec.read_header(id, rec, err));
        CHECK(rec.tokens == tokens && rec.spec == spec && rec.has_dft);
        CHECK(rec.checkpoints.size() == selected.size());
        auto actual = rec.checkpoints.begin();
        for (const auto & c : selected) {
            if (actual == rec.checkpoints.end()) {
                break;
            }
            CHECK(actual->n_tokens == c.n_tokens && actual->pos_min == c.pos_min && actual->pos_max == c.pos_max);
            CHECK(actual->data_tgt == c.data_tgt && actual->data_dft == c.data_dft && actual->data_spec == c.data_spec);
            ++actual;
        }
    }
    // Legacy callers that omit the cutoff still save all checkpoints.
    CHECK(ec.add(tokens, checkpoints, spec, true, id, err));
    server_eval_cache_record rec;
    CHECK(ec.read_header(id, rec, err) && rec.checkpoints.size() == checkpoints.size());
}

static void test_streamed_index(const std::string & root) {
    using json = nlohmann::ordered_json;
    server_eval_cache ec;
    std::string err;
    const std::string key = KEY_A + ";escaped=\"\\\n\t\x01";
    CHECK(ec.open(root + "/stream-index", key, true, err));
    const std::string path = ec.dir + "/eval-cache.json";
    auto bytes = [&]() {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    auto control = [&]() {
        json j;
        j["format"] = "llamampere-eval-cache";
        j["format_version"] = EVAL_CACHE_FORMAT_VERSION;
        j["key_hash"] = string_format("%016" PRIx64, ec.key_hash);
        j["key"] = ec.key;
        json arr = json::array();
        for (const auto & e : ec.entries) {
            arr.push_back({{"id", e.id}, {"n_tokens", e.tokens.size()},
                           {"bytes", e.bytes}, {"has_dft", e.has_dft}});
        }
        j["entries"] = arr;
        return j.dump(1) + "\n";
    };
    CHECK(bytes() == control());
    // Use real new requests to catch whole-tree and full-text allocations.
    for (int i = 0; i < 4096; ++i) {
        server_eval_cache_entry e;
        e.id = std::to_string(i) + "\"\\\n";
        e.tokens = seq(i, i % 3);
        e.bytes = UINT64_MAX - i;
        e.has_dft = (i % 2) != 0;
        ec.entries.push_back(std::move(e));
        if (i == 0) {
            CHECK(ec.write_index(err));
            CHECK(bytes() == control());
        }
    }
    allocation_probe::start(0);
    const bool saved = ec.write_index(err);
    allocation_probe::stop();
    CHECK(saved && allocation_probe::largest < 65536);
    const std::string before = bytes();
    CHECK(before == control() && before.size() > 65536);

    const std::string tmp = ec.dir + "/.eval-cache.json.tmp";
    fs::create_directory(tmp);
    CHECK(!ec.write_index(err));
    CHECK(bytes() == before);
    fs::remove(tmp);
#ifdef __linux__
    fs::create_symlink("/dev/full", tmp);
    CHECK(!ec.write_index(err));
    CHECK(bytes() == before);
    fs::remove(tmp);
#endif
}

static void test_index_growth_budget(const std::string & root) {
    const std::string dir = root + "/index-growth";
    const std::string index = dir + "/eval-cache.json";
    auto contents = [](const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    server_eval_cache writer, reader;
    std::string err, id;
    CHECK(writer.open(dir, KEY_A, true, err, 256 * 1024));
    const llama_tokens first = seq(0, 2048);
    CHECK(add_entry(writer, first, {}, {}, false, id));
    const std::string first_id = id;
    const std::string first_header = contents(writer.path_hdr(id));
    CHECK(writer.add(first, {}, {}, false, id, err));
    CHECK(writer.entries.size() == 1 && contents(writer.path_hdr(id)) == first_header);

    bool refused = false;
    llama_tokens refused_tokens;
    for (int i = 1; i < 128; ++i) {
        const llama_tokens tokens = seq(i * 2048, 2048);
        const std::string next_id = server_eval_cache::entry_id(tokens);
        const std::string before = contents(index);
        const size_t count = writer.entries.size();
        touch(writer.path_tgt(next_id), 4096);
        allocation_probe::start(tokens.size() * sizeof(llama_token));
        const bool saved = writer.add(tokens, {}, {}, false, id, err);
        const size_t copies = allocation_probe::stop();
        if (!saved) {
            CHECK(err.find("RAM fit budget") != std::string::npos);
            CHECK(copies == 0 && allocation_probe::largest < tokens.size() * sizeof(llama_token));
            CHECK(writer.entries.size() == count && contents(index) == before);
            CHECK(!fs::exists(writer.path_hdr(next_id)) && !fs::exists(writer.path_hdr(next_id) + ".tmp"));
            CHECK(contents(writer.path_hdr(first_id)) == first_header);
            refused_tokens = tokens;
            refused = true;
            break;
        }
    }
    CHECK(refused && writer.entries.size() > 1);
    CHECK(reader.open(dir, KEY_A, false, err));
    CHECK(reader.entries.size() == writer.entries.size());
    server_eval_cache_record rec;
    CHECK(reader.read_header(first_id, rec, err) && rec.tokens == first);
    if (!refused) {
        return;
    }

    // An unbounded reopen must accept the same tokens and make the real copy.
    CHECK(writer.open(dir, KEY_A, true, err));
    allocation_probe::start(refused_tokens.size() * sizeof(llama_token));
    const bool saved = writer.add(refused_tokens, {}, {}, false, id, err);
    const size_t copies = allocation_probe::stop();
    CHECK(saved && copies > 0);
    CHECK(writer.read_header(id, rec, err) && rec.tokens == refused_tokens);
}

static void test_index_ram_budget(const std::string & root) {
    const std::string dir = root + "/index-ram";
    const std::string index = dir + "/eval-cache.json";
    server_eval_cache writer, reader;
    std::string err, id;
    CHECK(writer.open(dir, KEY_A, true, err));
    for (int i = 0; i < 16; ++i) {
        CHECK(add_entry(writer, seq(i * 10000, 8192), {}, {}, false, id));
    }
    const uint64_t index_bytes = fs::file_size(index);
    auto index_contents = [&]() {
        std::ifstream f(index, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    const std::string original_index = index_contents();
    CHECK(reader.open(dir, KEY_A, false, err, 2 * 1024 * 1024));
    CHECK(reader.enabled() && reader.entries.size() == 16);
    CHECK(reader.find(seq(0, 8192), 0) >= 0);

    // A small index can refer to more tokens than the whole index budget.
    CHECK(reader.open(dir, KEY_A, false, err, 256 * 1024));
    CHECK(!reader.enabled() && reader.entries.empty() && reader.entries.capacity() == 0);
    CHECK(err.empty());
    CHECK(!reader.open(dir, KEY_A, true, err, 256 * 1024));
    CHECK(err.find("RAM fit budget") != std::string::npos);
    CHECK(fs::file_size(index) == index_bytes);
    CHECK(index_contents() == original_index);
    CHECK(reader.open(dir, KEY_A, false, err, 0));
    CHECK(!reader.enabled());
    for (uint64_t host_gib : {16, 32, 64, 128}) {
        CHECK(reader.open(dir, KEY_A, false, err, common_fit_eval_cache_ram_bytes((host_gib - 6) << 30, true)));
        CHECK(reader.enabled() && reader.entries.size() == 16);
    }
    CHECK(reader.open(dir, KEY_A, false, err, common_fit_eval_cache_ram_bytes(0, false)));
    CHECK(reader.entries.size() == 16);

    // Reject a large snapshot before parsing, then prove the source is untouched.
    fs::resize_file(index, 1ull << 26);
    CHECK(reader.open(dir, KEY_A, false, err, 1024 * 1024));
    CHECK(!reader.enabled() && reader.entries.empty());
    CHECK(!reader.open(dir, KEY_A, true, err, 1024 * 1024));
    CHECK(fs::file_size(index) == (1ull << 26));
    CHECK(writer.write_index(err));

    // Many short JSON values need a node budget even when file bytes fit.
    {
        std::ofstream f(index, std::ios::trunc);
        f << "{\"extra\":[0";
        for (int i = 0; i < 5000; ++i) {
            f << ",0";
        }
        f << "]}";
    }
    CHECK(fs::file_size(index) < 16384);
    CHECK(reader.open(dir, KEY_A, false, err, 1024 * 1024));
    CHECK(!reader.enabled());
    {
        std::ofstream f(index, std::ios::trunc);
        f << std::string(72, '[') << '0' << std::string(72, ']');
    }
    CHECK(reader.open(dir, KEY_A, false, err, 1024 * 1024));
    CHECK(!reader.enabled());
    CHECK(writer.write_index(err));
    CHECK(reader.open(dir, KEY_A, false, err, 2 * 1024 * 1024));
    CHECK(reader.enabled() && reader.entries.size() == 16);
}

static void test_corruption(const std::string & root) {
    const std::string dir = root + "/b";
    std::string err;
    std::string id_ok;
    std::string id_bad;
    std::string id_trunc;
    std::string id_nostate;
    {
        server_eval_cache ec;
        CHECK(ec.open(dir, KEY_A, true, err));
        CHECK(add_entry(ec, seq(1, 30), { ckpt(10, 3) }, {}, false, id_ok));
        CHECK(add_entry(ec, seq(2, 30), { ckpt(10, 4) }, {}, false, id_bad));
        CHECK(add_entry(ec, seq(3, 30), {}, {}, false, id_trunc));
        CHECK(add_entry(ec, seq(4, 30), {}, {}, true, id_nostate));
    }

    // flip a byte in a checkpoint blob: the entry still lists (tokens are fine) but its header fails the checksum
    {
        const std::string p = dir + "/entries/" + id_bad + ".evh";
        std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(-31, std::ios::end); // inside the drafter blob of the checkpoint
        f.put('\x55');
    }
    // truncate a header inside the token list
    fs::resize_file(dir + "/entries/" + id_trunc + ".evh", 40);
    // remove a state file
    fs::remove(dir + "/entries/" + id_nostate + ".dft");

    server_eval_cache ec;
    CHECK(ec.open(dir, KEY_A, false, err));
    CHECK(ec.entries.size() == 2);
    CHECK(ec.n_skipped == 2);

    server_eval_cache_record rec;
    CHECK(ec.read_header(id_ok, rec, err));
    CHECK(!ec.read_header(id_bad, rec, err));
    CHECK(err.find("checksum") != std::string::npos);

    // a header from another key (copied in by hand) is rejected even though the index lists it
    {
        server_eval_cache eb;
        CHECK(eb.open(root + "/c", KEY_B, true, err));
        std::string id;
        CHECK(add_entry(eb, seq(7, 20), {}, {}, false, id));
        fs::copy_file(root + "/c/entries/" + id + ".evh", dir + "/entries/" + id + ".evh");
        CHECK(!ec.read_header(id, rec, err));
        CHECK(err.find("another model") != std::string::npos);
    }
}

static void test_index_version(const std::string & root) {
    const std::string dir = root + "/d";
    std::string err;
    {
        server_eval_cache ec;
        CHECK(ec.open(dir, KEY_A, true, err));
    }
    server_eval_cache ec;
    for (int version : {1, 99}) {
        {
            std::ofstream f(dir + "/eval-cache.json", std::ios::trunc);
            f << "{\"format\": \"llamampere-eval-cache\", \"format_version\": " << version << ", \"key\": \"" << KEY_A << "\", \"entries\": []}\n";
        }
        CHECK(ec.open(dir, KEY_A, false, err));
        CHECK(err.empty());
        CHECK(!ec.enabled());
        CHECK(!ec.writable);
        CHECK(ec.entries.empty());
        CHECK(ec.find(seq(1, 100), 0) == -1);
        std::string id;
        CHECK(!ec.add(seq(1, 100), {}, {}, false, id, err));
        CHECK(!ec.open(dir, KEY_A, true, err));
        CHECK(err.find("version " + std::to_string(version)) != std::string::npos);
        CHECK(!ec.enabled());
    }

    {
        std::ofstream f(dir + "/eval-cache.json", std::ios::trunc);
        f << "{\"format\": \"another-cache\", \"format_version\": 1}\n";
    }
    CHECK(!ec.open(dir, KEY_A, false, err));

    {
        std::ofstream f(dir + "/eval-cache.json", std::ios::trunc);
        f << "not json";
    }
    CHECK(!ec.open(dir, KEY_A, true, err));
}

int main() {
    common_log_set_verbosity_thold(-1);

    char tmpl[] = "/tmp/test-eval-cache-XXXXXX";
    const char * root_c = mkdtemp(tmpl);
    if (!root_c) {
        fprintf(stderr, "mkdtemp failed\n");
        return 1;
    }
    const std::string root = root_c;

    test_open_and_roundtrip(root);
    test_corruption(root);
    test_record_ram_budget(root);
    test_record_allocations(root);
    test_streamed_header(root);
    test_borrowed_checkpoints(root);
    test_streamed_index(root);
    test_index_ram_budget(root);
    test_index_growth_budget(root);
    test_index_version(root);

    std::error_code ec;
    fs::remove_all(root, ec);

    printf("test-eval-cache: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
