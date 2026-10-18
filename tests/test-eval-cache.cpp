// Unit tests of the eval-cache folder (--eval-cache DIR), no model needed:
//   folder creation in build mode and the refusal of a missing folder in read-only mode, entry header round trip
//   (tokens, checkpoints, speculative state), longest-prefix lookup, the refusal of a folder built for another key
//   (naming the differing field), the format-version check, and skipping of corrupt / incomplete entries.

#include "server-eval-cache.h"

#include "common.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

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
    test_index_version(root);

    std::error_code ec;
    fs::remove_all(root, ec);

    printf("test-eval-cache: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
