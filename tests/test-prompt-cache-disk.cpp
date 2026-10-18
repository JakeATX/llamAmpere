// Unit tests of the server prompt-cache defaults and disk tier (no model needed):
//   RAM tier selection from mocked host memory, FIFO eviction (including the order rebuilt from mtimes after a
//   restart), the free-space guard, key mismatch, corrupt / partial / outdated entry cleanup, and the
//   supersede / dedupe rules.

#include "server-prompt-cache-disk.h"
#include "server-task.h"

#include "common.h"
#include "log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdint>
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
    std::vector<uint8_t> main, drft, spec;
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
    e.spec = std::vector<uint8_t>(31, (uint8_t) id);
    for (int c = 0; c < n_ckpt; ++c) {
        common_prompt_checkpoint ck;
        ck.n_tokens = 10 + c;
        ck.id_task  = c;
        ck.pos_min  = c;
        ck.pos_max  = 100 + c;
        ck.data_tgt = std::vector<uint8_t>(1000 + c, (uint8_t) (id + c));
        ck.data_dft = std::vector<uint8_t>(17, (uint8_t) c);
        ck.data_spec = std::vector<uint8_t>(13 + c, (uint8_t) (id + c));
        e.ckpts.push_back(std::move(ck));
    }
    return e;
}

static bool write_entry(server_prompt_disk_tier & t, const entry_data & e) {
    return t.write(e.tokens, e.main, e.drft, e.ckpts, e.spec);
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
    if (r.tokens != e.tokens || r.main != e.main || r.drft != e.drft || r.spec != e.spec || r.checkpoints.size() != e.ckpts.size()) {
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

// the guard must be visible: a startup state that says writes are (about to be) skipped, a rate-limited
// repeated warning while skipping, and counters for /metrics
static void test_guard_visibility(const std::string & root) {
    const std::string dir = root + "/guard-vis";

    server_prompt_disk_tier t = make_tier(dir, "k", 16*GiB);

    uint64_t cap = 1000*GiB, avail = 900*GiB;
    bool     queryable = true;
    t.space_fn = [&](const std::string &, uint64_t & c, uint64_t & a) { c = cap; a = avail; return queryable; };

    int64_t now = 1000000;
    t.clock_us_fn = [&]() { return now; };
    t.guard_warn_interval_s = 60.0;

    // startup states: floor = max(10% of 1000 GiB, 8 GiB) = 100 GiB, the tier wants its 16 GiB limit above it
    auto st = t.space_check();
    CHECK(st.state == server_prompt_disk_tier::SPACE_OK);
    CHECK(st.reserve == 100*GiB && st.want == 16*GiB);
    avail = 110*GiB;   // 10 GiB above the floor < 16 GiB wanted
    CHECK(t.space_check().state == server_prompt_disk_tier::SPACE_NEAR);
    avail = 100*GiB;   // exactly at the floor (the PDLA case: 91 GiB free with a 90.9 GiB floor is the same)
    CHECK(t.space_check().state == server_prompt_disk_tier::SPACE_BELOW);
    avail = 50*GiB;
    CHECK(t.space_check().state == server_prompt_disk_tier::SPACE_BELOW);
    queryable = false;
    CHECK(t.space_check().state == server_prompt_disk_tier::SPACE_UNKNOWN);
    queryable = true;

    // no size limit: "near" means less than guard_abs (8 GiB) above the floor
    {
        server_prompt_disk_tier u = make_tier(dir + "-nolimit", "k", 0);
        u.space_fn = t.space_fn;
        avail = 105*GiB;
        CHECK(u.space_check().state == server_prompt_disk_tier::SPACE_NEAR);
        avail = 109*GiB;
        CHECK(u.space_check().state == server_prompt_disk_tier::SPACE_OK);
    }

    // a percentage floor (--cache-disk-min-free 5%): guard_frac 0.05, guard_abs 0 -> floor 50 GiB
    t.guard_frac = 0.05;
    t.guard_abs  = 0;
    avail = 60*GiB;
    st = t.space_check();
    CHECK(st.reserve == 50*GiB && st.state == server_prompt_disk_tier::SPACE_NEAR);
    CHECK(write_entry(t, make_entry(1, 10, 1000)));
    avail = 49*GiB;
    CHECK(!write_entry(t, make_entry(2, 10, 1000)));
    CHECK(t.guard_active);
    t.guard_frac = 0.10;
    t.guard_abs  = 8*GiB;
    avail = 500*GiB;
    CHECK(write_entry(t, make_entry(3, 10, 1000)));   // recovered: the next skip starts a new episode
    CHECK(!t.guard_active);

    // skipping: one warning at the start, then at most one per interval, every skip counted with its bytes
    const size_t n_skip0 = t.n_guard_skips;
    const size_t n_warn0 = t.n_guard_warnings;
    avail = 90*GiB;
    for (int i = 0; i < 10; ++i) {
        CHECK(!write_entry(t, make_entry(10 + i, 10, 1000)));
        now += 5*1000000ll; // 5 s apart: 50 s in total, inside one interval
    }
    CHECK(t.n_guard_skips == n_skip0 + 10);
    CHECK(t.n_guard_warnings == n_warn0 + 1);
    CHECK(t.guard_active);
    CHECK(t.bytes_guard_skips > 10 * 1000);
    CHECK(count_files(dir, ".bin") == 2);

    now += 20*1000000ll; // 70 s after the first warning
    CHECK(!write_entry(t, make_entry(30, 10, 1000)));
    CHECK(t.n_guard_warnings == n_warn0 + 2);
    CHECK(!write_entry(t, make_entry(31, 10, 1000)));
    CHECK(t.n_guard_warnings == n_warn0 + 2);

    // interval 0 = warn on every skip
    t.guard_warn_interval_s = 0.0;
    CHECK(!write_entry(t, make_entry(32, 10, 1000)));
    CHECK(!write_entry(t, make_entry(33, 10, 1000)));
    CHECK(t.n_guard_warnings == n_warn0 + 4);
    t.guard_warn_interval_s = 60.0;

    // recovery clears the active flag but keeps the totals; a new episode warns again at once
    const size_t   n_skip1 = t.n_guard_skips;
    const uint64_t b_skip1 = t.bytes_guard_skips;
    avail = 500*GiB;
    CHECK(write_entry(t, make_entry(40, 10, 1000)));
    CHECK(!t.guard_active);
    CHECK(t.n_guard_skips == n_skip1 && t.bytes_guard_skips == b_skip1);
    avail = 90*GiB;
    now += 1000000ll;
    CHECK(!write_entry(t, make_entry(41, 10, 1000)));
    CHECK(t.n_guard_warnings == n_warn0 + 5);

    // the RAM tier's shutdown flush goes through the same guard and counts its skips
    {
        server_prompt_cache pc(1024, 0);
        std::string err;
        pc.disk.space_fn    = t.space_fn;
        pc.disk.clock_us_fn = t.clock_us_fn;
        CHECK(pc.disk_open(dir + "-flush", "k", 0, err)); // logs the BELOW state
        server_prompt p;
        p.tokens = server_tokens(llama_tokens{1, 2, 3, 4}, false);
        auto * en = pc.alloc(p, 1*MiB, 0);
        CHECK(en != nullptr && en->data.main.size() == 1*MiB);
        pc.flush_to_disk();
        CHECK(pc.disk.n_guard_skips == 1);
        CHECK(pc.disk.n_guard_warnings == 1);
        CHECK(count_files(dir + "-flush", ".bin") == 0);
    }
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

// [#187] recycled RAM-tier state buffers: reuse without a fresh allocation, bounded count, trimmed to the limit
static void test_spare_pool() {
    server_prompt_cache pc(1024, 0);
    std::vector<uint8_t> a = pc.take_buffer(64*MiB);
    CHECK(a.size() == 64*MiB && a.capacity() >= 72*MiB); // 1/8 headroom
    a[0] = 42;
    const uint8_t * pa = a.data();
    pc.recycle(a);
    CHECK(a.empty() && pc.spare.size() == 1 && pc.spare_size() >= 72*MiB);

    // a slightly longer save of the same conversation fits the recycled buffer: same memory, no fresh allocation
    std::vector<uint8_t> b = pc.take_buffer(70*MiB);
    CHECK(b.data() == pa && b.size() == 70*MiB && pc.spare.empty());

    // too large for every spare: a fresh buffer, the spare stays
    pc.recycle(b);
    std::vector<uint8_t> c = pc.take_buffer(100*MiB);
    CHECK(c.data() != pa && pc.spare.size() == 1);
    // the smallest fitting spare is chosen
    pc.recycle(c);
    std::vector<uint8_t> d = pc.take_buffer(50*MiB);
    CHECK(d.data() == pa && pc.spare.size() == 1);
    d.clear(); d.shrink_to_fit();

    // small buffers are not kept; empty requests take nothing
    std::vector<uint8_t> small(1*MiB);
    pc.recycle(small);
    CHECK(pc.spare.size() == 1 && small.empty());
    CHECK(pc.take_buffer(0).empty() && pc.spare.size() == 1);

    // at most spare_max_n buffers: the smallest is dropped
    pc.spare.clear();
    for (size_t mb : {20, 30, 40, 50, 60}) {
        std::vector<uint8_t> v(mb*MiB);
        pc.recycle(v);
    }
    CHECK(pc.spare.size() == 4);
    size_t min_cap = SIZE_MAX;
    for (const auto & v : pc.spare) {
        min_cap = std::min(min_cap, v.capacity());
    }
    CHECK(min_cap == 30*MiB);

    // spares count against the RAM limit and the largest go first
    pc.limit_size = 100*MiB;
    pc.trim_spare(0);
    CHECK(pc.spare_size() <= 100*MiB && pc.spare.size() == 2); // 30 + 40 left
    pc.trim_spare(50*MiB);
    CHECK(pc.spare.size() == 1 && pc.spare_size() == 30*MiB);
    pc.update();
    CHECK(pc.spare.size() == 1);
}

// [#187] alloc(.., hand_over) moves the checkpoints into the entry instead of copying them; holds() tracks the entry
static void test_hand_over() {
    server_prompt_cache pc(1024, 0);
    server_prompt p;
    p.tokens = server_tokens(llama_tokens{1, 2, 3, 4, 5, 6, 7, 8}, false);
    for (int i = 0; i < 3; ++i) {
        auto & c = p.checkpoints.emplace_back();
        c.n_tokens = 2*(i + 1); c.pos_min = 0; c.pos_max = 2*(i + 1) - 1;
        c.data_tgt.assign(1*MiB, (uint8_t) i);
    }
    const uint8_t * d0 = p.checkpoints.front().data_tgt.data();

    auto * e = pc.alloc(p, 4*MiB, 0, &p.checkpoints);
    CHECK(e != nullptr && pc.holds(e));
    CHECK(p.checkpoints.empty() && e->prompt.checkpoints.size() == 3);
    CHECK(e->prompt.checkpoints.front().data_tgt.data() == d0); // moved, not copied
    CHECK(e->size() == 4*MiB + 3*MiB && pc.size() == 7*MiB);

    // without hand_over the slot keeps its own copy
    server_prompt q;
    q.tokens = server_tokens(llama_tokens{9, 10, 11}, false);
    q.checkpoints.emplace_back().data_tgt.assign(1*MiB, 7);
    auto * f = pc.alloc(q, 1*MiB, 0);
    CHECK(f != nullptr && q.checkpoints.size() == 1 && f->prompt.checkpoints.size() == 1);
    CHECK(f->prompt.checkpoints.front().data_tgt.data() != q.checkpoints.front().data_tgt.data());

    // an evicted entry is no longer held
    const server_prompt_cache_state * none = nullptr;
    CHECK(!pc.holds(none) && pc.holds(f));
    pc.limit_size = 2*MiB;
    pc.update();
    CHECK(!pc.holds(e));
}

// --cache-ram-checkpoints: the keep-newest-plus-doubling selection, and alloc() thinning a RAM entry (the slot
// keeps its full list, also when alloc is asked to hand the checkpoints over by move)
static void test_ram_ckpt_keep() {
    // 31 checkpoints, one every 900 tokens, entry of 28000 tokens
    entry_data e = make_entry(7, 28000, 4096, 31);
    int c = 0;
    for (auto & ck : e.ckpts) {
        ck.pos_min = 0;
        ck.pos_max = 900 * (c + 1) - 1;
        ck.n_tokens = 900 * (c + 1);
        c++;
    }
    CHECK(server_prompt_keep_checkpoints(e.ckpts, e.tokens.size(), 0).size() == 31);
    CHECK(server_prompt_keep_checkpoints(e.ckpts, e.tokens.size(), 31).size() == 31);
    const auto k = server_prompt_keep_checkpoints(e.ckpts, e.tokens.size(), 4);
    std::vector<int> pos;
    for (auto * p : k) {
        pos.push_back(p->pos_max);
    }
    CHECK(pos.size() >= 5 && pos.size() <= 8);
    CHECK(std::is_sorted(pos.begin(), pos.end()));      // list order
    for (int i = 0; i < 4; ++i) {
        CHECK(pos[pos.size() - 1 - i] == 900 * (31 - i) - 1); // the 4 newest
    }
    CHECK(pos.front() < 15000);                        // something near the middle survives for branches
    CHECK(server_prompt_keep_checkpoints(e.ckpts, e.tokens.size(), 8).size() > k.size());

    server_prompt p;
    p.tokens = server_tokens(e.tokens, false);
    for (const auto & ck : e.ckpts) {
        p.checkpoints.push_back(ck);
    }
    size_t kept_size = 0;
    for (auto * ck : k) {
        kept_size += ck->size();
    }
    {
        server_prompt_cache pc(1024, 0);
        pc.ram_ckpt_keep = 4;
        auto * en = pc.alloc(p, 1*MiB, 0, &p.checkpoints);
        CHECK(en != nullptr && en->prompt.checkpoints.size() == k.size());
        CHECK(p.checkpoints.size() == 31);             // thinned: copied, the slot keeps all
        CHECK(en->size() == 1*MiB + kept_size);
        CHECK(pc.n_ram_ckpt_dropped == 31 - k.size());
        CHECK(en->prompt.checkpoints.back().pos_max == 900 * 31 - 1);
    }
    {
        server_prompt_cache pc(1024, 0);               // 0 = all, hand-over still moves
        auto * en = pc.alloc(p, 1*MiB, 0, &p.checkpoints);
        CHECK(en != nullptr && en->prompt.checkpoints.size() == 31 && p.checkpoints.empty());
        CHECK(pc.n_ram_ckpt_dropped == 0);
    }
}

// [#35] with --mmproj loaded every prompt has has_mtmd set; a text-only prompt (no media chunks) must still reach
// the disk tier, both from a RAM entry and through the over-limit staging area
static void test_mtmd_text_only_spills(const std::string & root) {
    const std::string dir = root + "/mtmd";
    server_prompt_cache pc(1024, 0);
    pc.disk = make_tier(dir, "k", 0);

    server_prompt p;
    p.tokens = server_tokens(make_entry(3, 300, 0).tokens, /* has_mtmd = */ true);
    CHECK(p.tokens.has_mtmd && !p.tokens.has_media());

    auto * e = pc.alloc(p, 1*MiB, 0, nullptr, 31);
    CHECK(e != nullptr && e->prompt.tokens.has_mtmd);
    e->data.spec.assign(31, 42);
    CHECK(e->size() == 1*MiB + 31);
    pc.flush_to_disk();
    CHECK(pc.disk.n_match() == 1);
    server_prompt_disk_record record;
    CHECK(pc.disk.read(pc.disk.entries.begin(), record));
    CHECK(record.spec == e->data.spec);

    // over the RAM limit: staged for the disk tier instead of skipped
    server_prompt q;
    q.tokens = server_tokens(make_entry(4, 300, 0).tokens, /* has_mtmd = */ true);
    pc.limit_size = 1*MiB;
    auto * st = pc.alloc(q, 2*MiB, 0, nullptr, 17);
    CHECK(st != nullptr && pc.staging_pending);
    st->data.spec.assign(17, 7);
    pc.update();
    CHECK(!pc.staging_pending && pc.disk.n_match() == 2);
    CHECK(pc.disk.read(find_entry(pc.disk, q.tokens.get_text_tokens()), record));
    CHECK(record.spec == std::vector<uint8_t>(17, 7));
}

// a cache entry keeps the adapter scales its KV state was computed with: in RAM, through the staging area, and on disk
static void test_lora_scales(const std::string & root) {
    const std::string dir = root + "/lora";
    server_prompt_cache pc(1024, 0);
    pc.disk = make_tier(dir, "k", 0);

    server_prompt p;
    p.tokens      = server_tokens(make_entry(5, 300, 0).tokens, false);
    p.lora_scales = {0.0f, 0.5f};

    auto * e = pc.alloc(p, 1*MiB, 0);
    CHECK(e != nullptr && e->prompt.lora_scales == p.lora_scales);
    CHECK(e->prompt.clone().lora_scales == p.lora_scales);
    pc.flush_to_disk();

    server_prompt q;
    q.tokens      = server_tokens(make_entry(6, 300, 0).tokens, false);
    q.lora_scales = {1.0f};
    pc.limit_size = 1*MiB;
    auto * st = pc.alloc(q, 2*MiB, 0);
    CHECK(st != nullptr && pc.staging_pending && st->prompt.lora_scales == q.lora_scales);
    pc.update();

    server_prompt_disk_tier t = make_tier(dir, "k", 0);
    CHECK(t.n_match() == 2);
    server_prompt_disk_record r;
    CHECK(t.read(find_entry(t, p.tokens.get_text_tokens()), r) && r.lora_scales == p.lora_scales);
    CHECK(t.read(find_entry(t, q.tokens.get_text_tokens()), r) && r.lora_scales == q.lora_scales);

    // no adapters: an empty list round-trips
    const entry_data n = make_entry(8, 50, 4096);
    CHECK(write_entry(t, n));
    CHECK(t.read(find_entry(t, n.tokens), r) && r.lora_scales.empty() && same(r, n));
}

int main() {
    const std::string root = (fs::temp_directory_path() / ("llama.cpp-pcdisk-test-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    fs::create_directories(root);

    test_ram_tiers();
    test_dir_creation(root);
    test_roundtrip_and_key(root);
    test_fifo(root);
    test_guard(root);
    test_guard_visibility(root);
    test_corrupt_cleanup(root);
    test_supersede_dedupe(root);
    test_spare_pool();
    test_hand_over();
    test_ram_ckpt_keep();
    test_mtmd_text_only_spills(root);
    test_lora_scales(root);

    std::error_code ec;
    fs::remove_all(root, ec);

    printf("test-prompt-cache-disk: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
