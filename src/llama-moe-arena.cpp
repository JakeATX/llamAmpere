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
    if (msync((char *) data + begin, length, MS_SYNC) != 0 ||
            madvise((char *) data + begin, length, MADV_DONTNEED) != 0) {
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
        void * raw = nullptr;
        if (posix_memalign(&raw, align, chunk) != 0 || !raw) { fail("cannot allocate arena read buffer"); return; }
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
        free(buf);
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

uint64_t llama_moe_arena_checksum(const void * data, size_t bytes) {
    // FNV-style mix over 8-byte words per 64 MiB piece, pieces combined in order
    const size_t piece = (size_t) 64 << 20;
    const size_t n = (bytes + piece - 1) / piece;
    std::vector<uint64_t> part(n, 0);
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (size_t k = next++; k < n; k = next++) {
            const size_t a = k * piece, len = std::min(piece, bytes - a);
            const unsigned char * p = (const unsigned char *) data + a;
            uint64_t h = 1469598103934665603ull ^ k;
            size_t i = 0;
            for (; i + 8 <= len; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 1099511628211ull; }
            for (; i < len; ++i) { h = (h ^ p[i]) * 1099511628211ull; }
            part[k] = h;
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < 8; ++i) { pool.emplace_back(worker); }
    for (auto & t : pool) { t.join(); }
    uint64_t h = 0;
    for (uint64_t v : part) { h = (h ^ v) * 1099511628211ull + 0x9e3779b97f4a7c15ull; }
    return h;
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

size_t llama_moe_arena::available_memory() {
#ifdef __linux__
    std::ifstream input("/proc/meminfo");
    std::string key, unit;
    size_t kib;
    while (input >> key >> kib >> unit) {
        if (key == "MemAvailable:" && kib <= SIZE_MAX / 1024) { return kib * 1024; }
    }
#endif
    return 0;
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

llama_moe_arena::llama_moe_arena(size_t bytes) : size(bytes) {
#ifndef _WIN32
    data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) { data = nullptr; throw std::bad_alloc(); }
    buffer.reset(ggml_backend_cpu_buffer_from_ptr(data, size));
    if (!buffer) { munmap(data, size); data = nullptr; throw std::bad_alloc(); }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#else
    throw std::runtime_error("expert arena requires anonymous mmap");
#endif
}

void llama_moe_arena::pin() {
    pin_ranges.reserve(1);
    auto * reg = arena_cuda_reg();
    pin_range = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
    unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
    if (!pin_ranges.empty() || !pin_range || !unpin || !pin_range(data, size)) {
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
    for (const auto & range : missing_ranges) {
        if (!pin_range((char *) data + range.first, range.second)) {
            throw std::runtime_error("cannot pin restored arena pages");
        }
        pin_ranges.push_back(range);
    }
    missing_ranges.clear();
}

llama_moe_arena::~llama_moe_arena() {
    buffer.reset();
    for (const auto & range : pin_ranges) { unpin((char *) data + range.first); }
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
        // exact prompt and decode kernels
        { "GGML_CUDA_CONCAT_TILE",                "1" },
        { "GGML_CUDA_NORM_CAST",                  "1" },
        { "GGML_CUDA_MMQ_ID_CAP",                 "2" },
        { "GGML_CUDA_MMID_PAR",                   "1" },
        { "GGML_CUDA_LID_EXACT",                  "1" },
        { "GGML_CUDA_GET_ROWS_NARROW",            "1" },
        { "GGML_CUDA_GDN_R16",                    "1" },
        { "GGML_CUDA_TOPK_SELECT",                "1" },
        { "LLAMA_HC_BF16_ONCE",                   "1" },
        { "LLAMA_QSA_POOL_RANGE",                 "1" },
        { "LLAMA_QSA_SEL_DIRECT",                 "1" },
        { "LLAMA_PREFETCH_THREADS",               "8" },
        // verify-step matrix kernels (STRATAGAP/AIFAST trio, 100K, split on in both arms: +6.3% words/s, every task
        // faster at 3 and 4 words). KQ_TC: valid math (tensor-core sums), K-quants only, IQ3_S stays opt-in by name.
        // FROWS: valid math (router rows kernel).
        // KQ_ROWS is exact but has no in-model number (never ran at width 5), so it stays opt-in.
        { "GGML_CUDA_KQ_TC",                      "q4_K,q5_K,q6_K" },
        { "GGML_CUDA_KQ_TC_WARPS",                "4" },
        { "GGML_CUDA_FROWS",                      "1" },
        // CPU expert split (fork STRATAGAP, gated 2026-10-23: 97.5 tok/s plain vs 97.4 recipe, no loops, probe < 75 us).
        // Host threads compute the experts that are not on the card instead of copying their weights over PCIe. The
        // text differs run to run with the split on (CPU and card kernels round differently): valid math, not exact.
        // Set here, not in the kernel code, so only Flash-Next models get it; --no-moe-cpu-split or =0 turns it off.
        { "GGML_MOE_CPU_SPLIT",                   "1" },
        { "GGML_MOE_CPU_SPLIT_Q2_FAST",           "1" },
        { "GGML_MOE_CPU_SPLIT_ROWS",              "1" },
        { "GGML_CUDA_GRAPH_NODE_PRIORITY",        "1" },   // scheduling only: hit-first copies overlap the matrix-vector product
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
