// Unit tests of the server prompt-cache defaults and disk tier (no model needed):
//   RAM tier selection from mocked host memory, FIFO eviction (including the order rebuilt from mtimes after a
//   restart), the free-space guard, key mismatch, corrupt / partial / outdated entry cleanup, and the
//   supersede / dedupe rules.

#include "server-prompt-cache-disk.h"

#include "common.h"
#include "log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const uint64_t GiB = 1024ull*1024*1024;
static const uint64_t MiB = 1024ull*1024;

static void test_ram_tiers() {
    std::string why;

    // nominal sizes as the kernel reports them (a "64 GB" box shows ~60.7 GiB)
    CHECK(common_prompt_cache_ram_auto_mib( 8*GiB,   6*GiB, why) == 2048);
    CHECK(common_prompt_cache_ram_auto_mib(15*GiB,  12*GiB, why) == 4096);   // a 16 GB box
    CHECK(common_prompt_cache_ram_auto_mib(31*GiB,  20*GiB, why) == 8192);   // a 32 GB box
    CHECK(common_prompt_cache_ram_auto_mib(63660456ull*1024, 40*GiB, why) == 12288); // this host's MemTotal
    CHECK(common_prompt_cache_ram_auto_mib(94*GiB,  80*GiB, why) == 16384);
    CHECK(common_prompt_cache_ram_auto_mib(125*GiB, 100*GiB, why) == 20480);
    CHECK(common_prompt_cache_ram_auto_mib(512*GiB, 400*GiB, why) == 20480);
    CHECK(common_prompt_cache_ram_auto_mib(12*GiB,  10*GiB, why) == 2048);

    // clamp to half of MemAvailable
    CHECK(common_prompt_cache_ram_auto_mib(63*GiB, 10*GiB, why) == 5120);
    CHECK(why.find("clamped") != std::string::npos);
    CHECK(common_prompt_cache_ram_auto_mib(63*GiB, 40*GiB, why) == 12288);
    CHECK(why.find("clamped") == std::string::npos);

    // unknown host memory
    CHECK(common_prompt_cache_ram_auto_mib(0, 0, why) == 2048);

    // the live query works on this platform
    uint64_t total = 0, avail = 0;
    CHECK(common_host_memory(total, avail));
    CHECK(total > 0 && avail > 0 && avail <= total);

    // the default directory follows XDG_CACHE_HOME
#ifndef _WIN32
    const char * old = getenv("XDG_CACHE_HOME");
    const std::string old_s = old ? old : "";
    setenv("XDG_CACHE_HOME", "/tmp/xdg-test", 1);
    CHECK(common_prompt_cache_default_dir() == fs::path("/tmp/xdg-test/llamampere/prompt-cache"));
    unsetenv("XDG_CACHE_HOME");
    CHECK(common_prompt_cache_default_dir().string().find(".cache/llamampere/prompt-cache") != std::string::npos);
    if (old) {
        setenv("XDG_CACHE_HOME", old_s.c_str(), 1);
    }
#endif
}

struct entry_data {
    llama_tokens tokens;
    std::vector<uint8_t> main, drft;
    std::list<common_prompt_checkpoint> ckpts;
};

static entry_data make_entry(int id, size_t n_tokens, size_t main_bytes, int n_ckpt = 1) {
    entry_data e;
    for (size_t i = 0; i < n_tokens; ++i) {
        e.tokens.push_back((llama_token) (id * 100000 + i));
    }
    e.main.resize(main_bytes);
    for (size_t i = 0; i < main_bytes; ++i) {
        e.main[i] = (uint8_t) (i * 31 + id);
    }
    e.drft.resize(main_bytes / 8 + 3);
    for (size_t i = 0; i < e.drft.size(); ++i) {
        e.drft[i] = (uint8_t) (i * 7 + id);
    }
    for (int c = 0; c < n_ckpt; ++c) {
        common_prompt_checkpoint ck;
        ck.n_tokens = 10 + c;
        ck.id_task  = c;
        ck.pos_min  = c;
        ck.pos_max  = 100 + c;
        ck.data_tgt = std::vector<uint8_t>(1000 + c, (uint8_t) (id + c));
        ck.data_dft = std::vector<uint8_t>(17, (uint8_t) c);
        ck.data_spec = {};
        e.ckpts.push_back(std::move(ck));
    }
    return e;
}

static bool write_entry(server_prompt_disk_tier & t, const entry_data & e) {
    return t.write(e.tokens, e.main, e.drft, e.ckpts);
}

static server_prompt_disk_tier make_tier(const std::string & dir, const std::string & key, size_t limit) {
    server_prompt_disk_tier t;
    // plenty of room unless a test says otherwise
    t.space_fn = [](const std::string &, uint64_t & cap, uint64_t & avail) { cap = 1000*GiB; avail = 900*GiB; return true; };
    std::string err;
    const bool ok = t.open(dir, key, limit, err);
    if (!ok) {
        fprintf(stderr, "open failed: %s\n", err.c_str());
    }
    CHECK(ok);
    return t;
}

static std::list<server_prompt_disk_entry>::iterator find_entry(server_prompt_disk_tier & t, const llama_tokens & tokens) {
    for (auto it = t.entries.begin(); it != t.entries.end(); ++it) {
        if (it->match && it->tokens == tokens) {
            return it;
        }
    }
    return t.entries.end();
}

static bool same(const server_prompt_disk_record & r, const entry_data & e) {
    if (r.tokens != e.tokens || r.main != e.main || r.drft != e.drft || r.checkpoints.size() != e.ckpts.size()) {
        return false;
    }
    auto a = r.checkpoints.begin();
    auto b = e.ckpts.begin();
    for (; a != r.checkpoints.end(); ++a, ++b) {
        if (a->n_tokens != b->n_tokens || a->id_task != b->id_task || a->pos_min != b->pos_min || a->pos_max != b->pos_max ||
            a->data_tgt != b->data_tgt || a->data_dft != b->data_dft || a->data_spec != b->data_spec) {
            return false;
        }
    }
    return true;
}

static size_t count_files(const std::string & dir, const char * suffix) {
    size_t n = 0;
    for (const auto & de : fs::directory_iterator(dir)) {
        const std::string name = de.path().filename().string();
        if (name.size() >= strlen(suffix) && name.compare(name.size() - strlen(suffix), strlen(suffix), suffix) == 0) {
            n++;
        }
    }
    return n;
}

static void set_mtime_offset(const std::string & file, int seconds) {
    fs::last_write_time(file, fs::file_time_type::clock::now() + std::chrono::seconds(seconds));
}

static void test_dir_creation(const std::string & root) {
    // a missing nested directory is created
    const std::string dir = root + "/a/b/c";
    server_prompt_disk_tier t = make_tier(dir, "k", 0);
    CHECK(fs::is_directory(dir));
    CHECK(t.enabled());

#ifndef _WIN32
    // an unwritable directory is refused (RAM-only fallback in the server)
    if (getuid() != 0) {
        const std::string ro = root + "/ro";
        fs::create_directories(ro);
        fs::permissions(ro, fs::perms::owner_read | fs::perms::owner_exec);
        server_prompt_disk_tier t2;
        std::string err;
        CHECK(!t2.open(ro, "k", 0, err));
        CHECK(!t2.enabled());
        CHECK(!err.empty());
        fs::permissions(ro, fs::perms::owner_all);
    }
    // a path under a regular file cannot be created
    {
        std::ofstream(root + "/file") << "x";
        server_prompt_disk_tier t3;
        std::string err;
        CHECK(!t3.open(root + "/file/sub", "k", 0, err));
    }
#endif
}

static void test_roundtrip_and_key(const std::string & root) {
    const std::string dir = root + "/key";
    const entry_data a = make_entry(1, 500, 300000, 2);

    {
        server_prompt_disk_tier t = make_tier(dir, "model=A;ctk=tq5_0;ctv=turbo4;", 0);
        CHECK(write_entry(t, a));
        CHECK(t.n_match() == 1);
    }
    {
        // another KV configuration on the same directory: indexed, never restorable, not deleted
        server_prompt_disk_tier t = make_tier(dir, "model=A;ctk=q8_0;ctv=turbo4;", 0);
        CHECK(t.entries.size() == 1);
        CHECK(t.n_match() == 0);
        CHECK(t.entries.front().tokens.empty());
        CHECK(count_files(dir, ".bin") == 1);
        // and its writes do not supersede the other config's entry
        CHECK(write_entry(t, a));
        CHECK(t.n_match() == 1);
        CHECK(count_files(dir, ".bin") == 2);
    }
    {
        // a different model: no match either
        server_prompt_disk_tier t = make_tier(dir, "model=B;ctk=tq5_0;ctv=turbo4;", 0);
        CHECK(t.n_match() == 0);
        CHECK(t.entries.size() == 2);
    }
    {
        // the original key restores the data bit-exactly, and the file stays on disk after the read
        server_prompt_disk_tier t = make_tier(dir, "model=A;ctk=tq5_0;ctv=turbo4;", 0);
        CHECK(t.n_match() == 1);
        auto it = find_entry(t, a.tokens);
        CHECK(it != t.entries.end());
        server_prompt_disk_record r;
        CHECK(t.read(it, r));
        CHECK(same(r, a));
        CHECK(t.n_match() == 1);
        CHECK(count_files(dir, ".bin") == 2);
    }
}

static void test_fifo(const std::string & root) {
    const std::string dir = root + "/fifo";

    const entry_data e1 = make_entry(11, 100, 1*MiB, 0);
    const entry_data e2 = make_entry(12, 100, 1*MiB, 0);
    const entry_data e3 = make_entry(13, 100, 1*MiB, 0);
    const entry_data e4 = make_entry(14, 100, 1*MiB, 0);

    // room for two entries (each ~1.13 MiB)
    const size_t limit = (size_t) (2.5 * MiB);

    {
        server_prompt_disk_tier t = make_tier(dir, "k", limit);
        CHECK(write_entry(t, e1));
        CHECK(write_entry(t, e2));
        CHECK(t.n_match() == 2);
        CHECK(write_entry(t, e3)); // evicts e1, the oldest write
        CHECK(t.n_match() == 2);
        CHECK(find_entry(t, e1.tokens) == t.entries.end());
        CHECK(find_entry(t, e2.tokens) != t.entries.end());
        CHECK(find_entry(t, e3.tokens) != t.entries.end());
        CHECK(t.size() <= limit);
        CHECK(count_files(dir, ".bin") == 2);

        // make e3 look older than e2 on disk: the restart must order by mtime, not by name
        set_mtime_offset(find_entry(t, e3.tokens)->file, -100);
        set_mtime_offset(find_entry(t, e2.tokens)->file, -10);
    }
    {
        server_prompt_disk_tier t = make_tier(dir, "k", limit);
        CHECK(t.entries.size() == 2);
        CHECK(t.entries.front().tokens == e3.tokens);
        CHECK(write_entry(t, e4)); // evicts e3 (oldest mtime)
        CHECK(find_entry(t, e3.tokens) == t.entries.end());
        CHECK(find_entry(t, e2.tokens) != t.entries.end());
        CHECK(find_entry(t, e4.tokens) != t.entries.end());
    }
    {
        // reopening with a smaller limit trims the oldest at startup
        server_prompt_disk_tier t = make_tier(dir, "k", (size_t) (1.5 * MiB));
        CHECK(t.entries.size() == 1);
        CHECK(t.entries.front().tokens == e4.tokens);
    }
    {
        // an entry larger than the whole limit is refused and evicts nothing
        server_prompt_disk_tier t = make_tier(dir, "k", (size_t) (1.5 * MiB));
        const entry_data big = make_entry(15, 100, 2*MiB, 0);
        CHECK(!write_entry(t, big));
        CHECK(t.entries.size() == 1);
    }
    {
        // limit 0 = no limit (only the free-space guard applies)
        server_prompt_disk_tier t = make_tier(dir, "k", 0);
        for (int i = 0; i < 5; ++i) {
            CHECK(write_entry(t, make_entry(20 + i, 100, 1*MiB, 0)));
        }
        CHECK(t.n_match() == 6);
    }
}

static void test_guard(const std::string & root) {
    const std::string dir = root + "/guard";

    server_prompt_disk_tier t = make_tier(dir, "k", 0);

    uint64_t cap = 100*GiB, avail = 9*GiB;
    t.space_fn = [&](const std::string &, uint64_t & c, uint64_t & a) { c = cap; a = avail; return true; };

    // 100 GiB disk: reserve = max(10 GiB, 8 GiB) = 10 GiB
    CHECK(!write_entry(t, make_entry(1, 10, 1000)));
    CHECK(t.n_guard_skips == 1);
    CHECK(t.guard_active);
    CHECK(!write_entry(t, make_entry(2, 10, 1000)));
    CHECK(t.n_guard_skips == 2);
    CHECK(t.entries.empty());
    CHECK(count_files(dir, ".bin") == 0);

    avail = 11*GiB;
    CHECK(write_entry(t, make_entry(3, 10, 1000)));
    CHECK(!t.guard_active);

    // small disk: the absolute 8 GiB floor dominates
    cap = 40*GiB; avail = 7*GiB;
    CHECK(!write_entry(t, make_entry(4, 10, 1000)));
    avail = 9*GiB;
    CHECK(write_entry(t, make_entry(5, 10, 1000)));

    // big disk: 10% dominates (910 GiB disk with 88 GiB free = skip)
    cap = 910*GiB; avail = 88*GiB;
    CHECK(!write_entry(t, make_entry(6, 10, 1000)));
    avail = 104*GiB;
    CHECK(write_entry(t, make_entry(7, 10, 1000)));

    // the entry's own size counts toward the room needed
    t.guard_abs  = 0;
    t.guard_frac = 0.0;
    avail = 500; // bytes, less than one ~1.1 KiB entry
    CHECK(!write_entry(t, make_entry(8, 10, 1000)));
    avail = 1*MiB;
    CHECK(write_entry(t, make_entry(9, 10, 1000)));
}

static void test_corrupt_cleanup(const std::string & root) {
    const std::string dir = root + "/corrupt";

    const entry_data a = make_entry(1, 200, 100000);
    const entry_data b = make_entry(2, 200, 100000);
    const entry_data c = make_entry(3, 200, 100000);
    std::string file_a, file_b, file_c;
    {
        server_prompt_disk_tier t = make_tier(dir, "k", 0);
        CHECK(write_entry(t, a));
        CHECK(write_entry(t, b));
        CHECK(write_entry(t, c));
        file_a = find_entry(t, a.tokens)->file;
        file_b = find_entry(t, b.tokens)->file;
        file_c = find_entry(t, c.tokens)->file;
    }

    // a: truncated (partial copy left by an older writer)
    fs::resize_file(file_a, fs::file_size(file_a) - 1000);
    // b: flip one payload byte (header intact, so only the checksum can catch it)
    {
        std::fstream f(file_b, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff) (fs::file_size(file_b) - 5000));
        char ch = 0;
        f.read(&ch, 1);
        ch ^= 0x5a;
        f.seekp((std::streamoff) (fs::file_size(file_b) - 5000));
        f.write(&ch, 1);
    }
    // garbage named like an entry
    std::ofstream(dir + "/pc-garbage.bin") << "not a prompt cache file";
    // a version-1 file (unkeyed, can never be restored safely)
    {
        std::ofstream f(dir + "/pc-0000000000000001.bin", std::ios::binary);
        f.write("LCPC", 4);
        const uint32_t v = 1;
        f.write((const char *) &v, 4);
        const uint64_t n = 0;
        f.write((const char *) &n, 8);
    }
    // a stale temp file of a crashed writer, and a fresh one of a live writer
    std::ofstream(dir + "/.pc-stale.bin.tmp") << "partial";
    set_mtime_offset(dir + "/.pc-stale.bin.tmp", -3600);
    std::ofstream(dir + "/.pc-live.bin.tmp") << "partial";
    // an unrelated file is left alone
    std::ofstream(dir + "/README") << "user file";

    server_prompt_disk_tier t = make_tier(dir, "k", 0);

    CHECK(!fs::exists(file_a));
    CHECK(!fs::exists(dir + "/pc-garbage.bin"));
    CHECK(!fs::exists(dir + "/pc-0000000000000001.bin"));
    CHECK(!fs::exists(dir + "/.pc-stale.bin.tmp"));
    CHECK(fs::exists(dir + "/.pc-live.bin.tmp"));
    CHECK(fs::exists(dir + "/README"));
    CHECK(t.n_removed_bad == 4);

    // b passes the header scan, fails the checksum at read time and is deleted then
    CHECK(t.n_match() == 2);
    auto it_b = find_entry(t, b.tokens);
    CHECK(it_b != t.entries.end());
    server_prompt_disk_record r;
    CHECK(!t.read(it_b, r));
    CHECK(!fs::exists(file_b));
    CHECK(t.n_match() == 1);

    auto it_c = find_entry(t, c.tokens);
    CHECK(it_c != t.entries.end());
    CHECK(t.read(it_c, r));
    CHECK(same(r, c));

    // an entry deleted behind our back (another server sharing the directory) fails cleanly
    fs::remove(file_c);
    it_c = find_entry(t, c.tokens);
    CHECK(!t.read(it_c, r));
    CHECK(t.n_match() == 0);
}

static void test_supersede_dedupe(const std::string & root) {
    const std::string dir = root + "/supersede";
    server_prompt_disk_tier t = make_tier(dir, "k", 0);

    entry_data a = make_entry(1, 300, 10000);
    entry_data a_short = a;
    a_short.tokens.resize(200);
    entry_data a_long = a;
    for (int i = 0; i < 50; ++i) {
        a_long.tokens.push_back(7);
    }

    CHECK(write_entry(t, a));
    // a prefix of a stored prompt is already covered: nothing written
    CHECK(write_entry(t, a_short));
    CHECK(t.n_match() == 1);
    CHECK(count_files(dir, ".bin") == 1);
    // a longer prompt with the stored one as prefix supersedes it once written
    CHECK(write_entry(t, a_long));
    CHECK(t.n_match() == 1);
    CHECK(t.entries.front().tokens == a_long.tokens);
    CHECK(count_files(dir, ".bin") == 1);
    // a sibling (diverging) prompt is kept alongside
    entry_data sib = a;
    sib.tokens.back() = 99;
    CHECK(write_entry(t, sib));
    CHECK(t.n_match() == 2);
}

int main() {
    const std::string root = (fs::temp_directory_path() / ("llama.cpp-pcdisk-test-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    fs::create_directories(root);

    test_ram_tiers();
    test_dir_creation(root);
    test_roundtrip_and_key(root);
    test_fifo(root);
    test_guard(root);
    test_corrupt_cleanup(root);
    test_supersede_dedupe(root);

    std::error_code ec;
    fs::remove_all(root, ec);

    printf("test-prompt-cache-disk: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
