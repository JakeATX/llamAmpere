#include "llama-moe-nvme.h"

// fork (NVMETIER, backlog #161): see llama-moe-nvme.h. POSIX only; elsewhere the tier is never enabled.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#define LLAMA_MOE_NVME_POSIX 1
#endif

static long nvme_env_long(const char * name, long def, long lo, long hi) {
    const char * v = getenv(name);
    if (v == nullptr || *v == '\0') {
        return def;
    }
    return std::clamp(strtol(v, nullptr, 10), lo, hi);
}

bool llama_moe_nvme_enabled() {
#ifdef LLAMA_MOE_NVME_POSIX
    static const bool on = nvme_env_long("LLAMA_MOE_NVME_TIER", 0, 0, 1) == 1;
    return on;
#else
    return false;
#endif
}

#ifdef LLAMA_MOE_NVME_POSIX

namespace {

constexpr size_t nvme_align = 4096; // page and O_DIRECT alignment

enum : uint8_t { ST_ABSENT = 0, ST_LOADING = 1, ST_RESIDENT = 2 };

struct nvme_entry {
    uint8_t  state = ST_ABSENT;
    uint32_t pins  = 0;     // read() copies in progress
    uint64_t epoch = 0;     // epoch of the last ensure
    int64_t  prev  = -1;    // LRU links over resident entries, global ids
    int64_t  next  = -1;
};

struct nvme_region {
    const char * host = nullptr;
    size_t nbytes = 0;      // n_expert*stride
    size_t stride = 0;
    int64_t n_expert = 0;
    size_t file_offset = 0;
    int fd = -1;            // O_DIRECT when `direct`
    bool direct = false;
    char * map_base = nullptr;
    size_t map_size = 0;
    char * shadow = nullptr; // map_base + file_offset % nvme_align: same alignment as the file
    const void * owner = nullptr;
    int64_t first = 0;      // global id of expert 0
    bool live = true;
};

// small fixed pool: run(jobs) runs every job once, on the pool and the calling thread
struct nvme_pool {
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv, cv_done;
    const std::vector<std::function<void()>> * jobs = nullptr;
    std::atomic<size_t> next{0};
    size_t n_done = 0;
    int n_busy = 0; // workers holding the current batch: run() returns only when none does
    uint64_t gen = 0;
    bool stop = false;

    explicit nvme_pool(int n) {
        for (int i = 0; i < n - 1; ++i) {
            threads.emplace_back([this] { worker(); });
        }
    }
    ~nvme_pool() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }
    size_t drain(const std::vector<std::function<void()>> * js) {
        size_t mine = 0;
        for (size_t i = next.fetch_add(1); i < js->size(); i = next.fetch_add(1)) {
            (*js)[i]();
            ++mine;
        }
        return mine;
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            const std::vector<std::function<void()>> * js = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
                if (jobs == nullptr) {
                    continue; // woke after its batch was done
                }
                js = jobs;
                n_busy++;
            }
            const size_t mine = drain(js);
            std::lock_guard<std::mutex> lk(mtx);
            n_done += mine;
            n_busy--;
            cv_done.notify_all();
        }
    }
    void run(const std::vector<std::function<void()>> & js) {
        if (js.size() <= 1 || threads.empty()) {
            for (const auto & j : js) {
                j();
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            jobs = &js;
            next.store(0);
            n_done = 0;
            ++gen;
        }
        cv.notify_all();
        const size_t mine = drain(&js);
        std::unique_lock<std::mutex> lk(mtx);
        n_done += mine;
        cv_done.wait(lk, [&] { return n_done == js.size() && n_busy == 0; });
        jobs = nullptr; // a worker that wakes late sees no batch
    }
};

} // namespace

struct llama_moe_nvme_tier::impl {
    size_t budget;
    mutable std::mutex mtx;
    std::condition_variable cv_loaded;
    std::vector<nvme_region> regions;
    std::map<uintptr_t, size_t> by_host; // host begin -> region index
    std::vector<nvme_entry> entries;
    int64_t lru_head = -1, lru_tail = -1;
    std::vector<size_t> entry_region; // global id -> region index
    uint64_t cur_epoch = 2;
    llama_moe_nvme_stats st;
    nvme_pool pool;
    // serializes the pool between ensure_many callers (one batch at a time)
    std::mutex mtx_pool;

    impl(size_t budget, int n_threads) : budget(budget), pool(std::max(1, n_threads)) { st.budget = budget; }

    ~impl() {
        for (auto & r : regions) {
            release_region(r);
        }
    }

    static void release_region(nvme_region & r) {
        if (!r.live) {
            return;
        }
        if (r.map_base) {
            munmap(r.map_base, r.map_size);
        }
        if (r.fd >= 0) {
            close(r.fd);
        }
        r.live = false;
        r.map_base = nullptr;
        r.fd = -1;
    }

    // region holding [p, p + len), or nullptr
    nvme_region * find(const void * p, size_t len) {
        const uintptr_t a = (uintptr_t) p;
        auto it = by_host.upper_bound(a);
        if (it == by_host.begin()) {
            return nullptr;
        }
        --it;
        nvme_region & r = regions[it->second];
        if (!r.live || a + len > (uintptr_t) r.host + r.nbytes || a < (uintptr_t) r.host) {
            return nullptr;
        }
        return &r;
    }

    void lru_unlink(int64_t id) {
        nvme_entry & en = entries[id];
        if (en.prev >= 0) { entries[en.prev].next = en.next; } else { lru_head = en.next; }
        if (en.next >= 0) { entries[en.next].prev = en.prev; } else { lru_tail = en.prev; }
        en.prev = en.next = -1;
    }
    void lru_push(int64_t id) {
        nvme_entry & en = entries[id];
        en.prev = lru_tail;
        en.next = -1;
        if (lru_tail >= 0) { entries[lru_tail].next = id; } else { lru_head = id; }
        lru_tail = id;
    }

    // [lo, hi): the aligned shadow span of expert e, the bytes a read of it writes and its charge
    static void span(const nvme_region & r, int64_t e, size_t & lo, size_t & hi) {
        // offsets relative to r.map_base, which is aligned; shadow - map_base == file_offset % align
        const size_t base = r.file_offset % nvme_align;
        const size_t vs = base + (size_t) e*r.stride;
        const size_t ve = vs + r.stride;
        lo = vs & ~(nvme_align - 1);
        hi = std::min((ve + nvme_align - 1) & ~(nvme_align - 1), r.map_size);
    }

    void evict(int64_t id) {
        nvme_region & r = regions[entry_region[id]];
        const int64_t e = id - r.first;
        size_t lo, hi;
        span(r, e, lo, hi);
        const size_t charge = hi - lo;
        // a boundary page shared with a neighbour that is not absent holds its bytes: keep it
        const size_t base = r.file_offset % nvme_align;
        const size_t vs = base + (size_t) e*r.stride, ve = vs + r.stride;
        size_t dlo = lo, dhi = hi;
        if ((vs % nvme_align) != 0 && e > 0 && entries[id - 1].state != ST_ABSENT) {
            dlo += nvme_align;
        }
        if ((ve % nvme_align) != 0 && e + 1 < r.n_expert && entries[id + 1].state != ST_ABSENT) {
            dhi -= nvme_align;
        }
        if (dhi > dlo) {
            madvise(r.map_base + dlo, dhi - dlo, MADV_DONTNEED);
        }
        lru_unlink(id);
        entries[id].state = ST_ABSENT;
        st.resident -= charge;
        st.evictions++;
    }

    // make room for `charge` bytes; experts ensured in this or the previous epoch and pinned ones stay
    void make_room(size_t charge) {
        int64_t id = lru_head;
        while (st.resident + charge > budget && id >= 0) {
            const int64_t nx = entries[id].next;
            if (entries[id].pins == 0 && entries[id].epoch + 1 < cur_epoch) {
                evict(id);
            }
            id = nx;
        }
        if (st.resident + charge > budget) {
            st.overflows++;
        }
    }

    // read the aligned span of expert e into the shadow; false on failure
    bool read_span(nvme_region & r, int64_t e, uint64_t & nread) {
        size_t lo, hi;
        span(r, e, lo, hi);
        const size_t base = r.file_offset % nvme_align;
        // file offset of shadow byte at map offset x: file_offset - base + x
        const size_t f0 = r.file_offset - base;
        const size_t need_end = base + (size_t) (e + 1)*r.stride; // map offset the read must reach
        size_t done = 0;
        while (lo + done < hi) {
            const ssize_t n = pread(r.fd, r.map_base + lo + done, hi - lo - done, (off_t) (f0 + lo + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (n == 0) {
                break; // end of file
            }
            done += (size_t) n;
        }
        nread += done;
        if (!r.direct && done > 0) {
            posix_fadvise(r.fd, (off_t) (f0 + lo), (off_t) done, POSIX_FADV_DONTNEED);
        }
        return lo + done >= need_end;
    }

    // a fallback that is always right: the mapped bytes
    static void copy_mapped(nvme_region & r, int64_t e) {
        memcpy(r.shadow + (size_t) e*r.stride, r.host + (size_t) e*r.stride, r.stride);
    }

    void ensure_ids(const std::vector<int64_t> & want) {
        std::vector<int64_t> load, wait;
        {
            std::lock_guard<std::mutex> lk(mtx);
            for (int64_t id : want) {
                nvme_entry & en = entries[id];
                if (en.state == ST_RESIDENT) {
                    if (en.epoch != cur_epoch) {
                        en.epoch = cur_epoch;
                    }
                    lru_unlink(id);
                    lru_push(id);
                    st.hits++;
                } else if (en.state == ST_LOADING) {
                    en.epoch = cur_epoch;
                    wait.push_back(id);
                } else {
                    nvme_region & r = regions[entry_region[id]];
                    size_t lo, hi;
                    span(r, id - r.first, lo, hi);
                    make_room(hi - lo);
                    en.state = ST_LOADING;
                    en.epoch = cur_epoch;
                    st.resident += hi - lo;
                    st.misses++;
                    load.push_back(id);
                }
            }
        }
        if (!load.empty()) {
            std::lock_guard<std::mutex> lp(mtx_pool);
            std::atomic<uint64_t> nread{0}, nfail{0};
            std::vector<std::function<void()>> jobs;
            jobs.reserve(load.size());
            for (int64_t id : load) {
                nvme_region * r = &regions[entry_region[id]];
                const int64_t e = id - r->first;
                jobs.emplace_back([this, r, e, &nread, &nfail] {
                    uint64_t n = 0;
                    if (!read_span(*r, e, n)) {
                        copy_mapped(*r, e);
                        nfail++;
                    }
                    nread += n;
                });
            }
            const auto t0 = std::chrono::steady_clock::now();
            pool.run(jobs);
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::lock_guard<std::mutex> lk(mtx);
            for (int64_t id : load) {
                entries[id].state = ST_RESIDENT;
                lru_push(id);
            }
            st.bytes_read += nread.load();
            st.fallbacks  += nfail.load();
            st.read_s     += dt;
        }
        if (!load.empty()) {
            cv_loaded.notify_all();
        }
        if (!wait.empty()) {
            std::unique_lock<std::mutex> lk(mtx);
            for (int64_t id : wait) {
                cv_loaded.wait(lk, [&] { return entries[id].state == ST_RESIDENT; });
            }
        }
    }

    // global ids of the experts overlapping [p, p + len) of r
    static void ids_of(const nvme_region & r, const void * p, size_t len, std::vector<int64_t> & out) {
        if (len == 0) {
            return;
        }
        const size_t o = (size_t) ((const char *) p - r.host);
        const int64_t e0 = (int64_t) (o / r.stride);
        const int64_t e1 = (int64_t) ((o + len - 1) / r.stride);
        for (int64_t e = e0; e <= e1; ++e) {
            out.push_back(r.first + e);
        }
    }
};

llama_moe_nvme_tier::llama_moe_nvme_tier(size_t budget, int n_threads) : pimpl(std::make_unique<impl>(budget, n_threads)) {}

llama_moe_nvme_tier::~llama_moe_nvme_tier() = default;

bool llama_moe_nvme_tier::add(const void * host, int fd, size_t file_offset, size_t stride, int64_t n_expert, const void * owner) {
    if (host == nullptr || fd < 0 || n_expert <= 0 || stride < 2*nvme_align) {
        return false; // an expert smaller than two pages would share a page with more than one neighbour
    }
    // a different open file description: O_DIRECT is per description, and the mapping's fd stays as it was
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    bool direct = true;
    int dfd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (dfd < 0) {
        direct = false;
        dfd = open(path, O_RDONLY | O_CLOEXEC);
        if (dfd < 0) {
            return false;
        }
    }
    nvme_region r;
    r.host = (const char *) host;
    r.stride = stride;
    r.n_expert = n_expert;
    r.nbytes = stride*(size_t) n_expert;
    r.file_offset = file_offset;
    r.fd = dfd;
    r.direct = direct;
    r.owner = owner;
    const size_t base = file_offset % nvme_align;
    r.map_size = (base + r.nbytes + nvme_align - 1) & ~(nvme_align - 1);
    void * m = mmap(nullptr, r.map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) {
        close(dfd);
        return false;
    }
#ifdef MADV_NOHUGEPAGE
    madvise(m, r.map_size, MADV_NOHUGEPAGE); // the budget counts 4 KiB pages
#endif
    r.map_base = (char *) m;
    r.shadow = r.map_base + base;

    // one probe read decides whether this file takes O_DIRECT
    if (direct) {
        const size_t f0 = file_offset - base;
        if (pread(dfd, r.map_base, nvme_align, (off_t) f0) < 0 && errno == EINVAL) {
            close(dfd);
            dfd = open(path, O_RDONLY | O_CLOEXEC);
            if (dfd < 0) {
                munmap(m, r.map_size);
                return false;
            }
            r.fd = dfd;
            r.direct = false;
        }
        madvise(r.map_base, nvme_align, MADV_DONTNEED);
    }

    std::lock_guard<std::mutex> lk(pimpl->mtx);
    auto & P = *pimpl;
    if (P.find(host, 1) != nullptr) {
        munmap(m, r.map_size);
        close(r.fd);
        return false; // already registered
    }
    r.first = (int64_t) P.entries.size();
    P.entries.resize(P.entries.size() + n_expert);
    P.entry_region.resize(P.entries.size(), P.regions.size());
    P.by_host[(uintptr_t) host] = P.regions.size();
    P.regions.push_back(r);
    P.st.direct_files += r.direct ? 1 : 0;
    return true;
}

void llama_moe_nvme_tier::remove(const void * owner) {
    std::lock_guard<std::mutex> lp(pimpl->mtx_pool);
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    auto & P = *pimpl;
    for (size_t i = 0; i < P.regions.size(); ++i) {
        nvme_region & r = P.regions[i];
        if (!r.live || r.owner != owner) {
            continue;
        }
        for (int64_t e = 0; e < r.n_expert; ++e) {
            const int64_t id = r.first + e;
            if (P.entries[id].state == ST_RESIDENT) {
                size_t lo, hi;
                impl::span(r, e, lo, hi);
                P.lru_unlink(id);
                P.st.resident -= hi - lo;
            }
            P.entries[id].state = ST_ABSENT;
        }
        P.by_host.erase((uintptr_t) r.host);
        impl::release_region(r);
    }
}

bool llama_moe_nvme_tier::empty() const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    return pimpl->by_host.empty();
}

const void * llama_moe_nvme_tier::ensure(const void * host, size_t len) {
    std::vector<int64_t> ids;
    const char * shadow = nullptr;
    {
        std::lock_guard<std::mutex> lk(pimpl->mtx);
        nvme_region * r = pimpl->find(host, std::max<size_t>(len, 1));
        if (r == nullptr) {
            return nullptr;
        }
        impl::ids_of(*r, host, len, ids);
        shadow = r->shadow + ((const char *) host - r->host);
    }
    pimpl->ensure_ids(ids);
    return shadow;
}

void llama_moe_nvme_tier::ensure_many(const std::vector<std::pair<const void *, size_t>> & ranges) {
    std::vector<int64_t> ids;
    {
        std::lock_guard<std::mutex> lk(pimpl->mtx);
        for (const auto & [p, len] : ranges) {
            if (nvme_region * r = pimpl->find(p, std::max<size_t>(len, 1))) {
                impl::ids_of(*r, p, len, ids);
            }
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    pimpl->ensure_ids(ids);
}

bool llama_moe_nvme_tier::read(const void * host, size_t len, void * dst) {
    auto & P = *pimpl;
    struct seg { size_t o; size_t n; bool ram; };
    std::vector<seg> segs;
    std::vector<int64_t> pinned;
    nvme_region * r = nullptr;
    {
        std::lock_guard<std::mutex> lk(P.mtx);
        r = P.find(host, std::max<size_t>(len, 1));
        if (r == nullptr) {
            return false;
        }
        if (len == 0) {
            return true;
        }
        const size_t o0 = (size_t) ((const char *) host - r->host);
        for (size_t o = o0; o < o0 + len;) {
            const int64_t e = (int64_t) (o / r->stride);
            const size_t end = std::min(o0 + len, (size_t) (e + 1)*r->stride);
            nvme_entry & en = P.entries[r->first + e];
            const bool ram = en.state == ST_RESIDENT;
            if (ram) {
                en.pins++;
                pinned.push_back(r->first + e);
                P.st.hits++;
            }
            if (!segs.empty() && !segs.back().ram && !ram && segs.back().o + segs.back().n == o) {
                segs.back().n += end - o;
            } else {
                segs.push_back({ o, end - o, ram });
            }
            o = end;
        }
    }
    const size_t o0 = (size_t) ((const char *) host - r->host);
    uint64_t nread = 0, ndirect = 0, nfail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    thread_local std::vector<char> bounce_store;
    for (const seg & s : segs) {
        char * out = (char *) dst + (s.o - o0);
        if (s.ram) {
            memcpy(out, r->shadow + s.o, s.n);
            continue;
        }
        // aligned superset of the file bytes, through a per-thread aligned bounce buffer
        const size_t f  = r->file_offset + s.o;
        const size_t fa = f & ~(nvme_align - 1);
        const size_t fe = (f + s.n + nvme_align - 1) & ~(nvme_align - 1);
        const size_t need = fe - fa + nvme_align;
        if (bounce_store.size() < need) {
            bounce_store.assign(need, 0);
        }
        char * bounce = (char *) (((uintptr_t) bounce_store.data() + nvme_align - 1) & ~(uintptr_t) (nvme_align - 1));
        size_t done = 0;
        bool ok = true;
        while (fa + done < f + s.n) {
            const ssize_t n = pread(r->fd, bounce + done, fe - fa - done, (off_t) (fa + done));
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                ok = false;
                break;
            }
            done += (size_t) n;
        }
        if (ok) {
            memcpy(out, bounce + (f - fa), s.n);
            if (!r->direct) {
                posix_fadvise(r->fd, (off_t) fa, (off_t) done, POSIX_FADV_DONTNEED);
            }
            nread   += done;
            ndirect += s.n;
        } else {
            memcpy(out, r->host + s.o, s.n);
            nfail++;
        }
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(P.mtx);
    for (int64_t id : pinned) {
        P.entries[id].pins--;
    }
    P.st.bytes_read   += nread;
    P.st.direct_reads += ndirect;
    P.st.fallbacks    += nfail;
    P.st.read_s       += dt;
    return true;
}

void llama_moe_nvme_tier::epoch() {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    pimpl->cur_epoch++;
}

llama_moe_nvme_stats llama_moe_nvme_tier::stats() const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    return pimpl->st;
}

void llama_moe_nvme_tier::print_stats(const char * tag) const {
    const llama_moe_nvme_stats s = stats();
    const double gib = 1024.0*1024.0*1024.0;
    fprintf(stderr, "llama_moe_nvme: %s: hits %llu misses %llu evictions %llu overflows %llu fallbacks %llu, "
            "read %.2f GiB (%.2f GiB streamed) in %.2f s = %.2f GiB/s, resident %.0f / %.0f MiB\n", tag,
            (unsigned long long) s.hits, (unsigned long long) s.misses, (unsigned long long) s.evictions,
            (unsigned long long) s.overflows, (unsigned long long) s.fallbacks,
            s.bytes_read/gib, s.direct_reads/gib, s.read_s, s.read_s > 0 ? s.bytes_read/gib/s.read_s : 0.0,
            s.resident/1048576.0, s.budget/1048576.0);
}

bool llama_moe_nvme_tier::resident(const void * host, int64_t e) const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    nvme_region * r = pimpl->find(host, 1);
    return r && e >= 0 && e < r->n_expert && pimpl->entries[r->first + e].state == ST_RESIDENT;
}

const void * llama_moe_nvme_tier::shadow(const void * host) const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    nvme_region * r = pimpl->find(host, 1);
    return r ? r->shadow + ((const char *) host - r->host) : nullptr;
}

#else // not POSIX: never enabled, every call is a no-op

struct llama_moe_nvme_tier::impl {};
llama_moe_nvme_tier::llama_moe_nvme_tier(size_t, int) {}
llama_moe_nvme_tier::~llama_moe_nvme_tier() = default;
bool llama_moe_nvme_tier::add(const void *, int, size_t, size_t, int64_t, const void *) { return false; }
void llama_moe_nvme_tier::remove(const void *) {}
bool llama_moe_nvme_tier::empty() const { return true; }
const void * llama_moe_nvme_tier::ensure(const void *, size_t) { return nullptr; }
void llama_moe_nvme_tier::ensure_many(const std::vector<std::pair<const void *, size_t>> &) {}
bool llama_moe_nvme_tier::read(const void *, size_t, void *) { return false; }
void llama_moe_nvme_tier::epoch() {}
llama_moe_nvme_stats llama_moe_nvme_tier::stats() const { return {}; }
void llama_moe_nvme_tier::print_stats(const char *) const {}
bool llama_moe_nvme_tier::resident(const void *, int64_t) const { return false; }
const void * llama_moe_nvme_tier::shadow(const void *) const { return nullptr; }

#endif

static std::mutex g_nvme_mtx;
static std::unique_ptr<llama_moe_nvme_tier> g_nvme_tier;

llama_moe_nvme_tier * llama_moe_nvme_global() {
    if (!llama_moe_nvme_enabled()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_nvme_mtx);
    return g_nvme_tier && !g_nvme_tier->empty() ? g_nvme_tier.get() : nullptr;
}

llama_moe_nvme_tier * llama_moe_nvme_global_create() {
    if (!llama_moe_nvme_enabled()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_nvme_mtx);
    if (!g_nvme_tier) {
        const size_t mib = (size_t) nvme_env_long("LLAMA_MOE_NVME_RAM_MIB", 4096, 64, 1l << 24);
        const int n_threads = (int) nvme_env_long("LLAMA_MOE_NVME_THREADS", 8, 1, 64);
        g_nvme_tier = std::make_unique<llama_moe_nvme_tier>(mib << 20, n_threads);
    }
    return g_nvme_tier.get();
}

void llama_moe_nvme_global_release(const void * owner) {
    if (!llama_moe_nvme_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_nvme_mtx);
    if (!g_nvme_tier) {
        return;
    }
    g_nvme_tier->remove(owner);
    if (g_nvme_tier->empty()) {
        if (nvme_env_long("LLAMA_MOE_NVME_STATS", 0, 0, 1)) {
            g_nvme_tier->print_stats("exit");
        }
        g_nvme_tier.reset();
    }
}
