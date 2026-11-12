#include "llama-moe-arena.h"
#include "llama-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

void llama_copy_moe_output(void * dst, const void * src, size_t bytes) {
    memcpy(dst, src, bytes);
}

void llama_write_moe_output(void * dst, const void * src, size_t bytes) {
    memcpy(dst, src, bytes);
}

int llama_compare_moe_output(const void * lhs, const void * rhs, size_t bytes) {
    return memcmp(lhs, rhs, bytes);
}

llama_moe_output::llama_moe_output(size_t bytes, const char * directory) : size(bytes) {
#ifndef _WIN32
    std::string name = std::string(directory) + "/llama-moe-output-XXXXXX";
    std::vector<char> path(name.begin(), name.end());
    path.push_back(0);
    const int fd = mkstemp(path.data());
    if (fd < 0) { throw std::runtime_error("cannot create MoE temporary output file"); }
    unlink(path.data());
    bool reserved = bytes && bytes <= (size_t) INT64_MAX;
#ifdef __linux__
    reserved = reserved && posix_fallocate(fd, 0, bytes) == 0;
#else
    reserved = reserved && ftruncate(fd, bytes) == 0;
#endif
    if (!reserved) {
        close(fd);
        throw std::runtime_error("cannot reserve MoE temporary output file space");
    }
    data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) { close(fd); data = nullptr; throw std::runtime_error("cannot map MoE temporary output file"); }
    file = fd;
    madvise(data, size, MADV_SEQUENTIAL);
#else
    (void)directory;
    throw std::runtime_error("MoE temporary output file requires mmap");
#endif
}

void llama_moe_output::flush() {
#ifndef _WIN32
    // [PREFILLFAST] LLAMA_MOE_OUTPUT_NO_MSYNC=1: the file is unlinked and mapped MAP_SHARED in this process
    // only, so every read of the mapping already sees the bytes the backend wrote. msync only forces them to
    // disk (about 86 ms and 112.6 MiB of SSD writes per in-call checkpoint). Skipping it changes no data.
    static const bool no_msync = [] {
        const char * v = getenv("LLAMA_MOE_OUTPUT_NO_MSYNC");
        return v == nullptr || atoi(v) != 0;
    }();
    if (dirty && !no_msync && msync(data, size, MS_SYNC) != 0) {
        throw std::runtime_error("cannot flush MoE temporary output file");
    }
    dirty = false;
#endif
}

void llama_moe_output::release(size_t offset, size_t bytes) {
    if (offset > size || bytes > size - offset) {
        throw std::out_of_range("MoE temporary output release exceeds allocation");
    }
    if (!bytes) { return; }
#ifndef _WIN32
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { throw std::runtime_error("cannot query page size"); }
    const size_t page = (size_t) page_size;
    const size_t padding = offset % page ? page - offset % page : 0;
    if (padding >= bytes) { return; }
    const size_t begin = offset + padding;
    const size_t end = offset + bytes - (offset + bytes) % page;
    if (end <= begin) { return; }
    if (end > (size_t) INT64_MAX) {
        throw std::out_of_range("MoE temporary output page range overflow");
    }
    const size_t length = end - begin;
    // Boundary pages can contain live logits or neighboring activation rows.
    // Call only after the backend has completed all writes to these pages.
    if (msync((char *) data + begin, length, MS_SYNC) != 0) {
        throw std::runtime_error("cannot flush completed MoE output pages");
    }
#if defined(__linux__) && defined(MADV_PAGEOUT)
    // Reclaim mapped file pages before dropping PTEs; neighboring mapped pages can block fadvise eviction.
    if (madvise((char *) data + begin, length, MADV_PAGEOUT) != 0 && errno != EINVAL && errno != ENOSYS) {
        throw std::runtime_error("cannot reclaim completed MoE output pages");
    }
#endif
    if (madvise((char *) data + begin, length, MADV_DONTNEED) != 0) {
        throw std::runtime_error("cannot release completed MoE output pages");
    }
#ifdef __linux__
    if (posix_fadvise(file, begin, length, POSIX_FADV_DONTNEED) != 0) {
        throw std::runtime_error("cannot release MoE output file cache");
    }
#endif
#endif
}

// [PREFILLFAST] LLAMA_MOE_OUTPUT_READAHEAD=1: B1 writes each ubatch of hidden rows to this file and
// drops the pages, so the MTP catch-up reads every chunk back from disk on page faults. A
// WILLNEED hint on the next chunk lets the kernel read it while the current chunk computes.
// Advisory only: it never changes the bytes read.
void llama_moe_output::readahead(size_t offset, size_t bytes) {
#ifdef __linux__
    if (file < 0 || offset >= size) { return; }
    bytes = std::min(bytes, size - offset);
    if (bytes) { (void) posix_fadvise(file, (off_t) offset, (off_t) bytes, POSIX_FADV_WILLNEED); }
#else
    (void) offset; (void) bytes;
#endif
}

llama_moe_output::~llama_moe_output() {
#ifndef _WIN32
    if (data) { munmap(data, size); }
    if (file >= 0) { close(file); }
#endif
}

#ifndef _WIN32
static size_t arena_env_size(const char * name, size_t fallback, size_t lo, size_t hi) {
    const char * v = getenv(name);
    if (!v || !*v) { return fallback; }
    const long long n = atoll(v);
    return n < (long long) lo ? lo : (size_t) n > hi ? hi : (size_t) n;
}
#endif

void llama_moe_arena_parallel_load(const std::vector<llama_moe_arena_copy> & copies,
        bool (*progress)(size_t done, void * user), void * user, size_t ram_floor,
        llama_moe_arena_load_stats & stats) {
#ifndef _WIN32
    const auto start = std::chrono::steady_clock::now();
    constexpr size_t align = 4096; // covers the logical block size of NVMe/SATA filesystems
    const size_t chunk = arena_env_size("LLAMA_MOE_ARENA_CHUNK_MIB", 8, 1, 256) << 20;
    const int threads = (int) arena_env_size("LLAMA_MOE_ARENA_THREADS", 8, 1, 64);
    const bool want_direct = arena_env_size("LLAMA_MOE_ARENA_DIRECT", 1, 0, 1) != 0;

    // One direct descriptor per distinct buffered descriptor, opened through /proc so the
    // loader's own file stays untouched; -1 means buffered reads for that file.
    std::vector<std::pair<int, int>> direct_fd; // (buffered, direct)
    auto direct_of = [&](int fd) {
        for (const auto & p : direct_fd) { if (p.first == fd) { return p.second; } }
        int d = -1;
#ifdef O_DIRECT
        if (want_direct) {
            const std::string path = "/proc/self/fd/" + std::to_string(fd);
            d = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        }
#endif
        direct_fd.emplace_back(fd, d);
        return d;
    };
    struct item { int copy; size_t first, last; }; // aligned file byte range [first, last)
    std::vector<item> items;
    size_t total = 0;
    for (int i = 0; i < (int) copies.size(); ++i) {
        const auto & c = copies[i];
        direct_of(c.fd);
        const size_t bytes = c.expert_bytes * (size_t) c.n_expert;
        total += bytes;
        const size_t first = c.src_offset & ~(align - 1);
        const size_t last = (c.src_offset + bytes + align - 1) & ~(align - 1);
        for (size_t a = first; a < last; a += chunk) {
            items.push_back({ i, a, std::min(last, a + chunk) });
        }
    }
    std::atomic<size_t> next{0}, done{0}, read_total{0};
    std::atomic<bool> stop{false};
    std::mutex error_mu;
    std::string error;
    std::atomic<int> direct_failed{0};
    auto fail = [&](const std::string & what) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (error.empty()) { error = what; }
        stop = true;
    };
    auto worker = [&]() {
        void * raw = llama_aligned_alloc(align, chunk);
        if (!raw) { fail("cannot allocate arena read buffer"); return; }
        char * buf = (char *) raw;
        while (!stop.load(std::memory_order_relaxed)) {
            const size_t k = next.fetch_add(1);
            if (k >= items.size()) { break; }
            const auto & it = items[k];
            const auto & c = copies[it.copy];
            const size_t bytes = c.expert_bytes * (size_t) c.n_expert;
            const size_t need_end = std::min(it.last, c.src_offset + bytes);
            int fd = direct_of(c.fd);
            const bool direct = fd >= 0;
            if (!direct) { fd = c.fd; }
            size_t got = 0;
            const size_t len = it.last - it.first;
            while (got < len) {
                const ssize_t r = pread(fd, buf + got, len - got, (off_t) (it.first + got));
                if (r < 0 && errno == EINTR) { continue; }
                if (r < 0 && direct && errno == EINVAL && got == 0) {
                    // the filesystem refused this direct read: use buffered reads for it
                    direct_failed = 1;
                    fd = c.fd;
                    continue;
                }
                if (r <= 0) { break; }
                got += (size_t) r;
            }
            if (it.first + got < need_end) {
                fail("short read while loading the expert arena");
                break;
            }
            read_total += got;
            const size_t s0 = std::max(it.first, c.src_offset);
            for (size_t s = s0; s < need_end;) {
                const size_t rel = s - c.src_offset;
                const size_t e = rel / c.expert_bytes, w = rel % c.expert_bytes;
                const size_t n = std::min(c.expert_bytes - w, need_end - s);
                memcpy(c.dst + e * c.dst_stride + w, buf + (s - it.first), n);
                s += n;
            }
            if (fd == c.fd) {
#ifdef __linux__
                posix_fadvise(c.fd, (off_t) it.first, (off_t) got, POSIX_FADV_DONTNEED);
#endif
            }
            done += need_end - s0;
        }
        llama_aligned_free(buf);
    };
    // the direct descriptors are all opened above, so workers only read direct_fd
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int i = 0; i < threads; ++i) { pool.emplace_back(worker); }
    std::string stop_reason;
    for (;;) {
        bool finished = done.load() >= total || stop.load();
        if (!finished) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (progress && !progress(done.load(), user)) { stop_reason = "expert arena loading canceled"; stop = true; }
        if (llama_moe_arena::available_memory() < ram_floor) {
            stop_reason = "expert arena stopped: less than the RAM floor of MemAvailable"; stop = true;
        }
        if (finished || stop.load()) { break; }
    }
    for (auto & t : pool) { t.join(); }
    for (const auto & p : direct_fd) { if (p.second >= 0) { close(p.second); } }
    if (!stop_reason.empty()) { throw std::runtime_error(stop_reason); }
    if (!error.empty()) { throw std::runtime_error(error); }
    if (done.load() != total) { throw std::runtime_error("expert arena load incomplete"); }
    bool any_direct = false;
    for (const auto & p : direct_fd) { any_direct = any_direct || p.second >= 0; }
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stats.bytes_read = read_total.load();
    stats.threads = threads;
    stats.chunk = chunk;
    stats.direct = any_direct && !direct_failed.load();
#else
    (void) copies; (void) progress; (void) user; (void) ram_floor; (void) stats;
    throw std::runtime_error("parallel expert arena load requires POSIX");
#endif
}

bool llama_moe_arena::is_expert(const char * name) {
    const char * suffixes[] = { ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" };
    const size_t length = strlen(name);
    for (const char * suffix : suffixes) {
        const size_t n = strlen(suffix);
        if (length >= n && strcmp(name + length - n, suffix) == 0) { return true; }
    }
    return false;
}

bool llama_moe_arena::is_trunk_expert(const char * name, uint32_t trunk_layers) {
    int layer = -1;
    return is_expert(name) && sscanf(name, "blk.%d.", &layer) == 1 &&
            layer >= 0 && (uint32_t) layer < trunk_layers;
}

// The "<key> <number>" lines of a meminfo / memory.stat style file, in one pass: out[i] is set for each key found.
// Returns how many keys were found.
static int ram_file_keys(const std::string & path, const char * const * keys, uint64_t * out, int n_keys) {
    std::ifstream input(path);
    std::string line;
    int found = 0;
    while (found < n_keys && std::getline(input, line)) {
        for (int i = 0; i < n_keys; ++i) {
            const size_t n = strlen(keys[i]);
            if (line.compare(0, n, keys[i]) == 0 && line.size() > n && (line[n] == ' ' || line[n] == '\t')) {
                char * end = nullptr;
                const char * p = line.c_str() + n;
                const unsigned long long v = strtoull(p, &end, 10);
                if (end != p) { out[i] = v; ++found; }
                break;
            }
        }
    }
    return found;
}

// A single-number cgroup file. "max" (no limit) and a missing file both give false.
static bool ram_file_value(const std::string & path, uint64_t & out) {
    std::ifstream input(path);
    std::string word;
    if (!(input >> word) || word.empty() || word[0] < '0' || word[0] > '9') { return false; }
    out = strtoull(word.c_str(), nullptr, 10);
    return true;
}

llama_moe_arena::ram_view llama_moe_arena::read_ram_view(const std::string & meminfo_path,
        const std::string & cgroup_root, const std::string & cgroup_rel) {
    ram_view view;
    static const char * const mem_keys[] = { "MemAvailable:", "MemTotal:" };
    uint64_t kib[2] = { UINT64_MAX, UINT64_MAX }; // UINT64_MAX = not found
    ram_file_keys(meminfo_path, mem_keys, kib, 2);
    if (kib[0] == UINT64_MAX || kib[0] > UINT64_MAX / 1024) { return view; }
    view.meminfo = kib[0] * 1024;
    view.memtotal = kib[1] <= UINT64_MAX / 1024 ? kib[1] * 1024 : 0;
    // Walk from our group up to the root: a parent limit binds its children, and MemAvailable sees none of them.
    std::string rel = cgroup_rel;
    while (!rel.empty() && rel[0] == '/') {
        const std::string dir = cgroup_root + (rel == "/" ? "" : rel);
        uint64_t limit = UINT64_MAX, high = 0, current = 0;
        ram_file_value(dir + "/memory.max", limit); // "max" or no file leaves UINT64_MAX
        if (ram_file_value(dir + "/memory.high", high) && high < limit) { limit = high; }
        if (limit < view.limit) { view.limit = limit; } // RAMSAFE: caps total_memory(); does not touch available
        if (limit != UINT64_MAX && ram_file_value(dir + "/memory.current", current)) {
            // The group's file LRU (active + inactive) is page cache the kernel reclaims at the limit, as MemAvailable
            // assumes system-wide. shmem sits on the anon LRU in cgroup v2 and mlocked pages on unevictable, so both
            // stay counted as used.
            static const char * const stat_keys[] = { "inactive_file", "active_file" };
            uint64_t file[2] = { 0, 0 };
            ram_file_keys(dir + "/memory.stat", stat_keys, file, 2);
            const uint64_t cache = file[0] + file[1];
            const uint64_t used = current > cache ? current - cache : 0;
            const uint64_t room = limit > used ? limit - used : 0;
            if (room < view.cgroup) { view.cgroup = room; view.group = dir; }
        }
        if (rel == "/") { break; }
        const size_t cut = rel.find_last_of('/');
        rel = cut == 0 ? "/" : rel.substr(0, cut);
    }
    view.available = std::min(view.meminfo, view.cgroup);
    view.total = std::min(view.memtotal, view.limit);
    if (view.cgroup >= view.meminfo) { view.group.clear(); }
    return view;
}

#ifdef __linux__
// The group path cannot change under us without a restart of the unit: read it once. A path that is gone or not
// under /sys/fs/cgroup (no cgroup namespace in a container) has no limit files, which reads as no limit: today's
// behavior.
static const std::string & arena_cgroup_rel() {
    static const std::string rel = [] {
        const char * source = getenv("LLAMA_RAM_AVAIL_SOURCE");
        if (source && strcmp(source, "meminfo") == 0) { return std::string(); }
        std::ifstream input("/proc/self/cgroup");
        std::string line;
        while (std::getline(input, line)) {
            if (line.compare(0, 3, "0::") == 0) {
                std::string path = line.substr(3);
                const std::string deleted = " (deleted)";
                if (path.size() > deleted.size() && path.compare(path.size() - deleted.size(), deleted.size(), deleted) == 0) {
                    path.resize(path.size() - deleted.size());
                }
                return path;
            }
        }
        return std::string(); // cgroup v1 or none: MemAvailable only, as before
    }();
    return rel;
}
#endif

size_t llama_moe_arena::total_memory() {
#ifdef __linux__
    const uint64_t total = read_ram_view("/proc/meminfo", "/sys/fs/cgroup", arena_cgroup_rel()).total;
    return total > SIZE_MAX ? SIZE_MAX : (size_t) total;
#elif defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) { return (size_t) st.ullTotalPhys; }
    return 0;
#else
    return 0;
#endif
}

// LLAMA_MOE_RAM_HEADROOM_MIB as bytes (never below the 5 GiB floor); 0 when unset, empty or not a plain MiB count.
static size_t arena_headroom_mib_override() {
    constexpr unsigned long long floor_mib = 5120, max_mib = 1ull << 20;
    const char * v = getenv("LLAMA_MOE_RAM_HEADROOM_MIB");
    if (!v || !*v) { return 0; }
    unsigned long long mib = 0;
    bool ok = true;
    for (const char * c = v; ok && *c; ++c) {
        ok = *c >= '0' && *c <= '9' && (mib = mib * 10 + (unsigned long long) (*c - '0')) <= max_mib;
    }
    if (ok) { return (size_t) (std::max(mib, floor_mib) << 20); }
    LLAMA_LOG_WARN("%s: LLAMA_MOE_RAM_HEADROOM_MIB='%s' is not a plain MiB count; using the default\n", __func__, v);
    return 0;
}

static bool arena_headroom_auto() {
    const char * v = getenv("LLAMA_MOE_RAM_HEADROOM");
    return v && (strcmp(v, "auto") == 0 || strcmp(v, "1") == 0);
}

bool llama_moe_arena::ram_headroom_custom() {
    return arena_headroom_mib_override() != 0 || arena_headroom_auto();
}

size_t llama_moe_arena::ram_headroom() {
    if (const size_t bytes = arena_headroom_mib_override()) { return bytes; }
    if (arena_headroom_auto()) { return ram_headroom_for(total_memory()); }
    return size_t(7ull << 30); // default: the fixed 7 GiB startup headroom
}

size_t llama_moe_arena::available_memory() {
#ifdef __linux__
    const ram_view view = read_ram_view("/proc/meminfo", "/sys/fs/cgroup", arena_cgroup_rel());
    // Say it once, the first time the group limit is the tighter figure: from then on every RAM gate may decide
    // differently from what MemAvailable alone would give.
    static std::atomic<bool> said { false };
    if (!view.group.empty() && !said.exchange(true)) {
        LLAMA_LOG_WARN("%s: [RAM] usable RAM is %.1f GiB, not MemAvailable %.1f GiB: cgroup limit in %s "
                "(LLAMA_RAM_AVAIL_SOURCE=meminfo ignores the group)\n", __func__,
                view.available / double(1ull << 30), view.meminfo / double(1ull << 30), view.group.c_str());
    }
    // 0 means "unknown" to the callers. A group at or over its limit is known and exhausted, so it reports 1 byte and
    // every RAM gate refuses instead of turning itself off.
    if (view.meminfo > 0 && view.available == 0) { return 1; }
    return view.available > SIZE_MAX ? SIZE_MAX : (size_t) view.available;
#elif defined(_WIN32)
    // Windows has no /proc/meminfo: without this every RAM-fit / prefetch budget sees 0 bytes available, so
    // read-ahead is disabled and cold expert pages fault in one by one. ullAvailPhys counts free + standby,
    // close to MemAvailable's meaning for the planner's floors.
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) { return (size_t) st.ullAvailPhys; }
    return 0;
#else
    return 0;
#endif
}

static ggml_backend_reg_t arena_cuda_reg() {
    return ggml_backend_reg_by_name("CUDA");
}

bool llama_moe_arena::pin_supported() {
#ifndef _WIN32
    auto * reg = arena_cuda_reg();
    return reg && ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
#else
    return false;
#endif
}

// LLAMA_MOE_ARENA_THP=1 (default unset): back the arena with 2 MiB transparent huge pages. The CPU expert pool streams
// whole experts (~1.7 MB each); with 4 KB pages that is ~430 TLB misses per expert, and the hardware prefetcher stops at
// every 4 KB boundary. "1" skips the request when under half the arena could be backed by free 2 MiB blocks (compaction
// on a fragmented box can stall the load for minutes); "2" asks anyway.
#ifndef _WIN32
static size_t arena_free_huge_bytes() {
    std::ifstream f("/proc/buddyinfo");
    std::string line;
    size_t blocks = 0;
    while (std::getline(f, line)) {
        // "Node 0, zone   Normal  4688 601 ..." : skip the zone name, then one count per order
        const size_t zone = line.find("zone");
        if (zone == std::string::npos) { continue; }
        const char * c = line.c_str() + zone + 4;
        while (*c == ' ') { ++c; }
        while (*c && *c != ' ') { ++c; }
        for (int order = 0; *c; ++order) {
            char * end = nullptr;
            const unsigned long n = strtoul(c, &end, 10);
            if (end == c) { break; }
            c = end;
            if (order >= 9) { blocks += (size_t) n << (order - 9); }
        }
    }
    return blocks << 21;
}

static size_t arena_thp_kb(const void * base, size_t bytes) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const uintptr_t lo = (uintptr_t) base, hi = lo + bytes;
    bool inside = false;
    size_t kb = 0;
    while (std::getline(f, line)) {
        uintptr_t a = 0, b = 0;
        if (sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) { inside = a < hi && b > lo; continue; }
        size_t v = 0;
        if (inside && sscanf(line.c_str(), "AnonHugePages: %zu kB", &v) == 1) { kb += v; }
    }
    return kb;
}
#endif

llama_moe_arena::llama_moe_arena(size_t bytes) : size(bytes) {
#ifndef _WIN32
    const char * thp_env = getenv("LLAMA_MOE_ARENA_THP");
    const int thp = thp_env ? atoi(thp_env) : 0;
    constexpr size_t huge = 2ull << 20;
    if (thp > 0) {
        const size_t padded = size + huge;
        void * raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { throw std::bad_alloc(); }
        const uintptr_t start = (uintptr_t) raw;
        const uintptr_t aligned = (start + huge - 1) & ~(uintptr_t) (huge - 1);
        if (aligned > start) { munmap(raw, aligned - start); }
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        const uintptr_t end = aligned + ((size + page - 1) & ~(page - 1)), raw_end = start + padded;
        if (raw_end > end) { munmap((void *) end, raw_end - end); }
        data = (void *) aligned;
        const size_t free_huge = arena_free_huge_bytes();
        if (thp == 1 && free_huge < size / 2) {
            LLAMA_LOG_WARN("%s: THP not requested: %zu GiB of free 2 MiB blocks for a %zu GiB arena (LLAMA_MOE_ARENA_THP=2 asks anyway)\n",
                    __func__, free_huge >> 30, size >> 30);
        } else if (madvise(data, size, MADV_HUGEPAGE) == 0) {
            thp_requested = true;
            LLAMA_LOG_INFO("%s: THP requested for %zu GiB arena (%zu GiB free 2 MiB blocks)\n", __func__, size >> 30, free_huge >> 30);
        } else {
            LLAMA_LOG_WARN("%s: madvise(MADV_HUGEPAGE) failed: %s\n", __func__, strerror(errno));
        }
    } else {
        data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data == MAP_FAILED) { data = nullptr; throw std::bad_alloc(); }
    }
    buffer.reset(ggml_backend_cpu_buffer_from_ptr(data, size));
    if (!buffer) { munmap(data, size); data = nullptr; throw std::bad_alloc(); }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#else
    throw std::runtime_error("expert arena requires anonymous mmap");
#endif
}

void llama_moe_arena::report_thp() const {
#ifndef _WIN32
    if (thp_requested && data) {
        const size_t kb = arena_thp_kb(data, size);
        LLAMA_LOG_INFO("%s: %.2f of %.2f GiB of the arena is on 2 MiB pages\n", __func__, kb / double(1 << 20), size / double(1ull << 30));
    }
#endif
}

void llama_moe_arena::pin() {
    pin_ranges.reserve(1);
    auto * reg = arena_cuda_reg();
    pin_range = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
    unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
    is_unpinned = (bool (*)(const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_is_unpinned");
    if (!pin_ranges.empty() || !pin_range || !unpin || !is_unpinned || !pin_range(data, size)) {
        throw std::runtime_error("expert arena CUDA host registration failed");
    }
    pin_ranges.emplace_back(0, size);
}

size_t llama_moe_arena::discard(const std::vector<std::pair<size_t, size_t>> & ranges) {
#ifndef _WIN32
    if (pin_ranges.size() != 1 || pin_ranges[0] != std::make_pair(size_t(0), size) || !missing_ranges.empty()) {
        throw std::runtime_error("expert arena discard requires its original full registration");
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { throw std::runtime_error("cannot query arena page size"); }
    const size_t page = (size_t) page_size;
    std::vector<std::pair<size_t, size_t>> covered;
    for (const auto & range : ranges) {
        if (range.first > size || range.second > size - range.first) {
            throw std::out_of_range("resident expert span exceeds arena");
        }
        if (range.second) { covered.emplace_back(range.first, range.first + range.second); }
    }
    std::sort(covered.begin(), covered.end());
    std::vector<std::pair<size_t, size_t>> merged;
    for (const auto & range : covered) {
        if (!merged.empty() && range.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    std::vector<std::pair<size_t, size_t>> holes, cold;
    size_t cursor = 0, dropped = 0;
    for (const auto & range : merged) {
        const size_t padding = range.first % page ? page - range.first % page : 0;
        if (padding > range.second - range.first) { continue; }
        const size_t begin = range.first + padding;
        const size_t end = range.second - range.second % page;
        if (begin >= end) { continue; }
        if (cursor < begin) { cold.emplace_back(cursor, begin - cursor); }
        holes.emplace_back(begin, end - begin);
        dropped += end - begin;
        cursor = end;
    }
    if (!dropped) { return 0; }
    if (cursor < size) { cold.emplace_back(cursor, size - cursor); }
    pin_ranges.reserve(cold.size() + holes.size());
    // The GPU owner must preserve every covered byte before this transition.
    unpin(data);
    GGML_ASSERT(is_unpinned(data, size));
    pin_ranges.clear();
    for (const auto & range : cold) {
        if (!pin_range((char *) data + range.first, range.second)) {
            for (const auto & pinned : pin_ranges) { unpin((char *) data + pinned.first); }
            pin_ranges.clear();
            if (pin_range(data, size)) { pin_ranges.emplace_back(0, size); }
            throw std::runtime_error("cannot pin cold arena pages; no expert bytes discarded");
        }
        pin_ranges.push_back(range);
    }
    missing_ranges = std::move(holes);
    for (const auto & range : missing_ranges) {
        GGML_ASSERT(is_unpinned((char *) data + range.first, range.second));
        if (madvise((char *) data + range.first, range.second, MADV_DONTNEED) != 0) {
            throw std::runtime_error("cannot discard GPU-backed arena pages");
        }
    }
    return dropped;
#else
    (void) ranges;
    throw std::runtime_error("expert arena discard requires mmap");
#endif
}

void llama_moe_arena::pin_restored() {
    const size_t original = pin_ranges.size();
    pin_ranges.reserve(original + missing_ranges.size());
    try {
        for (const auto & range : missing_ranges) {
            if (!pin_range((char *) data + range.first, range.second)) {
                throw std::runtime_error("cannot pin restored arena pages");
            }
            pin_ranges.push_back(range);
        }
    } catch (...) {
        for (size_t i = original; i < pin_ranges.size(); ++i) { unpin((char *) data + pin_ranges[i].first); }
        pin_ranges.resize(original);
        throw;
    }
    missing_ranges.clear();
}

llama_moe_arena::~llama_moe_arena() {
    for (const auto & range : pin_ranges) { unpin((char *) data + range.first); }
    GGML_ASSERT(!is_unpinned || is_unpinned(data, size));
    buffer.reset();
#ifndef _WIN32
    if (data) { munmap(data, size); }
#endif
}

void llama_moe_arena_apply_flashnext_defaults() {
    // one master switch; later loads (fit pass, helper) find everything already set
    if (const char * master = getenv("LLAMA_FLASHNEXT_DEFAULTS"); master && strcmp(master, "0") == 0) { return; }
    static bool done = false;
    if (done) { return; }
    done = true;
    struct flashnext_default { const char * name; const char * value; };
    static const flashnext_default table[] = {
        // expert cache on the card, filled from measured usage, missed experts copied once per layer
        { "GGML_MOE_CACHE_DEVICE",                "1" },
        { "GGML_MOE_CACHE_DEVICE_CAPTURE",        "1" },
        { "GGML_MOE_CACHE_DEVICE_STATS",          "1" },   // route counts; hit-first staging stays inactive without them
        { "GGML_MOE_CACHE_SWAP",                  "1" },
        { "GGML_MOE_MISSJOIN",                    "stage" },
        // startup profile knobs: ship together (PRIOR_PCT=100 without PRIOR_DECAY=1 loses hits)
        { "GGML_MOE_CACHE_PROFILE_GLOBAL",        "1" },
        { "GGML_MOE_CACHE_PROFILE_PRIOR_PCT",     "100" },
        { "GGML_MOE_CACHE_PROFILE_PRIOR_DECAY",   "1" },
        // RAM and card memory lent during the prompt
        { "LLAMA_MOE_OUTPUT_RECLAIM",             "1" },
        { "LLAMA_MOE_CACHE_RELEASE_PROMPT",       "1" },
        // whole prompt read one layer at a time (prompts of 16,384 tokens or more)
        { "LLAMA_B1",                             "1" },
        { "LLAMA_B1_CACHE_WARM",                  "1" },
        { "LLAMA_B1_LAYER_RESERVE",               "1" },
        { "LLAMA_B1_MOE_CACHE_SUSPEND",           "1" },
        { "LLAMA_B1_EARLY_CACHE_SUSPEND",         "1" },
        // exact prompt and decode kernels
        { "GGML_CUDA_CONCAT_TILE",                "1" },
        { "GGML_CUDA_NORM_CAST",                  "1" },
        { "GGML_CUDA_MMQ_ID_CAP",                 "2" },
        { "GGML_CUDA_MMID_PAR",                   "1" },
        { "GGML_CUDA_LID_EXACT",                  "1" },
        { "GGML_CUDA_GET_ROWS_NARROW",            "1" },
        { "GGML_CUDA_GDN_R16",                    "1" },
        { "GGML_CUDA_TOPK_SELECT",                "1" },
        { "GGML_CUDA_FA_SPARSE_G16",              "4" },
        { "LLAMA_HC_BF16_ONCE",                   "1" },
        { "LLAMA_QSA_POOL_RANGE",                 "1" },
        { "LLAMA_QSA_SEL_DIRECT",                 "1" },
        { "LLAMA_PREFETCH_THREADS",               "8" },
        // verify-step matrix kernels (MOEGAP/AIFAST trio, 100K, split on in both arms: +6.3% words/s, every task
        // faster at 3 and 4 words). KQ_TC: valid math (tensor-core sums), K-quants only, IQ3_S stays opt-in by name.
        // FROWS: valid math (router rows kernel).
        // KQ_ROWS is exact but has no in-model number (never ran at width 5), so it stays opt-in.
        { "GGML_CUDA_KQ_TC",                      "q4_K,q5_K,q6_K" },
        { "GGML_CUDA_KQ_TC_WARPS",                "4" },
        { "GGML_CUDA_FROWS",                      "1" },
        // router weights stored as BF16 (exact: loader checks every value; F32 sum order; =0 keeps F32)
        { "GGML_CUDA_FROWS_BF16_F32_ORDER",       "1" },
        { "LLAMA_QWEN_ROUTER_BF16",               "1" },
        // CPU expert split (fork MOEGAP, gated 2026-10-23: 97.5 tok/s plain vs 97.4 recipe, no loops, probe < 75 us).
        // Host threads compute the experts that are not on the card instead of copying their weights over PCIe. The
        // text differs run to run with the split on (CPU and card kernels round differently): valid math, not exact.
        // Set here, not in the kernel code, so only Flash-Next models get it; --no-moe-cpu-split or =0 turns it off.
        { "GGML_MOE_CPU_SPLIT",                   "1" },
        { "GGML_MOE_CPU_SPLIT_Q2_FAST",           "1" },
        { "GGML_MOE_CPU_SPLIT_ROWS",              "1" },
        { "GGML_MOE_CPU_SPLIT_PF",                "2048" },
        { "GGML_MOE_CPU_SPLIT_MAX_THREADS",       "1" },
        { "GGML_MOE_CPU_SPLIT_AVX512",            "1" },
        { "GGML_MOE_CPU_SPLIT_REUSE8",            "1" },
        { "GGML_MOE_CPU_SPLIT_Q8_REUSE",           "1" },
        { "GGML_MOE_CPU_SPLIT_DOWN_REUSE",         "1" },
        { "GGML_CUDA_GRAPH_NODE_PRIORITY",        "1" },   // scheduling only: hit-first copies overlap the matrix-vector product
        // expert kernels that every speed measurement ran with (each also takes =check, which runs the old path next
        // to the new one and stops on any bit difference; =0 turns one off). AI23_*: AI+ banked experts only.
        { "GGML_MOE_HIT_FUSED",                   "1" },
        { "GGML_MOE_MISS_LIST",                   "1" },      // as measured; "auto" keeps it off on AI+ bank layers
        { "GGML_MOE_DOWN_SW",                     "1" },
        { "GGML_MOE_Q8_SHARE",                    "1" },
        { "GGML_MOE_PLAN_COUNT",                  "1" },
        { "GGML_CUDA_HC_UP_Q8",                   "1" },
        { "GGML_CUDA_KQ_ROWS",                    "1" },
        { "GGML_CUDA_AI23_ID_REF",                "0" },
        { "GGML_CUDA_AI23_DIRECT",                "6" },
        { "GGML_CUDA_AI23_VERIFY_COMPACT",        "0" },
        // prompt memory handling the measured runs used
        { "LLAMA_B1_LEND_SCRATCH",                "1" },
        { "LLAMA_B1_REPLAN",                      "1" },
        { "LLAMA_B1_MALLOC_TRIM",                 "1" },
        { "LLAMA_MALLOC_TRIM",                    "1" },
        { "LLAMA_QSA_POOL_COMPACT",               "1" },
        // fork: short prompts need a checkpoint before the restore-grid boundary for prefix reuse.
        { "LLAMA_SJKVARN_EXACT_CKPT",              "1" },
        // helper (MTP) memory and launch savings
        { "LLAMA_MTP_PIN_BORROWED_HOST",          "1" },
        { "LLAMA_MTP_COMPAT_REUSE",               "1" },
        { "LLAMA_MTP_EH_BATCH",                   "1" },
        { "LLAMA_MTP_DRAFT_COMPUTE_LEAN",         "1" },
    };
    int applied = 0;
    for (const auto & entry : table) {
        if (getenv(entry.name) != nullptr) { continue; }
#ifdef _WIN32
        if (_putenv_s(entry.name, entry.value) == 0) { applied++; }
#else
        if (setenv(entry.name, entry.value, 0) == 0) { applied++; }
#endif
    }
    LLAMA_LOG_INFO("%s: Flash-Next defaults: %d switches turned on (NAME=0 turns one off; set LLAMA_FLASHNEXT_DEFAULTS=0 for none)\n",
            __func__, applied);
}
