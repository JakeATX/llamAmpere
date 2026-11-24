// fork (NVMETIER, backlog #161): CPU-only test of the NVMe expert tier (src/llama-moe-nvme.cpp).
// A model-like file of experts at an unaligned offset: O_DIRECT reads into the RAM tier, LRU eviction under the
// budget, protection of the current window, streamed reads, concurrent use; every byte equal to the file.
#include "llama-moe-nvme.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    const char * tmp = getenv("TMPDIR");
    std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/test-moe-nvme-tier-XXXXXX";
    std::vector<char> name(path.begin(), path.end());
    name.push_back('\0');
    const int wfd = mkstemp(name.data());
    if (wfd < 0) {
        perror("mkstemp");
        return 1;
    }
    const size_t head = 777, stride = (1u << 20) + 4100, tail = 3000;
    const int64_t n_expert = 24;
    const size_t size = head + stride*n_expert + tail;
    std::vector<char> bytes(size);
    std::mt19937_64 rng(42);
    for (auto & c : bytes) {
        c = (char) rng();
    }
    if (write(wfd, bytes.data(), size) != (ssize_t) size) {
        perror("write");
        return 1;
    }
    fsync(wfd);
    close(wfd);
    const int fd = open(name.data(), O_RDONLY);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    char * map = (char *) mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(map != MAP_FAILED);
    const char * host = map + head;
    auto file_bytes = [&](size_t o) { return bytes.data() + head + o; };

    const size_t charge = ((stride + 8191) / 4096) * 4096; // an expert's aligned span, at most
    const size_t budget = 6*charge;
    llama_moe_nvme_tier tier(budget, 4);
    CHECK(!tier.add(host, fd, head, 4096, n_expert, &tier)); // experts under two pages are refused
    CHECK(tier.add(host, fd, head, stride, n_expert, &tier));
    CHECK(!tier.add(host, fd, head, stride, n_expert, &tier)); // twice
    CHECK(tier.ensure(map, 16) == nullptr);                    // before the tensor
    CHECK(tier.ensure(host + stride*n_expert, 1) == nullptr);  // after it
    const llama_moe_nvme_stats st0 = tier.stats();
    printf("O_DIRECT: %s\n", st0.direct_files ? "yes" : "no (buffered + DONTNEED)");

    // random windows of 1..3 experts: bytes equal the file, resident stays under the budget
    std::uniform_int_distribution<int> pick(0, (int) n_expert - 1);
    for (int round = 0; round < 200; ++round) {
        tier.epoch();
        std::vector<std::pair<const void *, size_t>> want;
        std::vector<int> es;
        const int k = 1 + round % 3;
        for (int i = 0; i < k; ++i) {
            es.push_back(pick(rng));
            want.emplace_back(host + es.back()*stride, stride);
        }
        tier.ensure_many(want);
        for (int e : es) {
            CHECK(tier.resident(host, e));
            const char * p = (const char *) tier.ensure(host + e*stride, stride);
            CHECK(p != nullptr && p != host + e*stride);
            CHECK(memcmp(p, file_bytes(e*stride), stride) == 0);
        }
        CHECK(tier.stats().resident <= budget);
    }
    llama_moe_nvme_stats st = tier.stats();
    CHECK(st.evictions > 0);
    CHECK(st.overflows == 0);
    CHECK(st.fallbacks == 0);
    CHECK(st.misses > 0 && st.hits > 0);

    // evicted experts give their pages back: mincore on the interior of a non-resident expert
    {
        const char * sh = (const char *) tier.shadow(host);
        int64_t absent = -1;
        for (int64_t e = 1; e + 1 < n_expert && absent < 0; ++e) {
            if (!tier.resident(host, e) && !tier.resident(host, e - 1) && !tier.resident(host, e + 1)) {
                absent = e;
            }
        }
        if (absent >= 0) {
            const uintptr_t a = ((uintptr_t) (sh + absent*stride) + 4095) & ~(uintptr_t) 4095;
            const uintptr_t b = ((uintptr_t) (sh + (absent + 1)*stride)) & ~(uintptr_t) 4095;
            std::vector<unsigned char> vec((b - a)/4096);
            CHECK(mincore((void *) a, b - a, vec.data()) == 0);
            size_t in = 0;
            for (unsigned char v : vec) {
                in += v & 1;
            }
            CHECK(in == 0);
        }
    }

    // one window larger than the budget: admitted above it (all protected), bytes right; later windows shrink it
    {
        tier.epoch();
        tier.epoch();
        std::vector<std::pair<const void *, size_t>> want;
        for (int e = 0; e < 9; ++e) {
            want.emplace_back(host + e*stride, stride);
        }
        tier.ensure_many(want);
        for (int e = 0; e < 9; ++e) {
            CHECK(tier.resident(host, e));
            CHECK(memcmp(tier.ensure(host + e*stride, stride), file_bytes(e*stride), stride) == 0);
        }
        CHECK(tier.stats().overflows > 0);
        tier.epoch();
        tier.epoch();
        tier.epoch();
        tier.ensure(host + 20*stride, stride);
        CHECK(tier.stats().resident <= budget);
    }

    // read(): any range, resident parts from RAM, the others from the file; the tier is not filled
    for (int i = 0; i < 300; ++i) {
        const size_t o = rng() % (stride*n_expert - 1);
        const size_t n = 1 + rng() % std::min<size_t>(3*stride, stride*n_expert - o);
        std::vector<char> out(n);
        const uint64_t misses = tier.stats().misses;
        CHECK(tier.read(host + o, n, out.data()));
        CHECK(memcmp(out.data(), file_bytes(o), n) == 0);
        CHECK(tier.stats().misses == misses);
    }
    std::vector<char> one(8);
    CHECK(!tier.read(map, 8, one.data())); // not registered

    // concurrent ensure + read on several threads
    {
        std::vector<std::thread> th;
        std::atomic<int> bad{0};
        for (int t = 0; t < 4; ++t) {
            th.emplace_back([&, t] {
                std::mt19937_64 r(t + 7);
                for (int i = 0; i < 100; ++i) {
                    const int e = (int) (r() % n_expert);
                    if (t % 2 == 0) {
                        const char * p = (const char *) tier.ensure(host + e*stride, stride);
                        // another thread may evict it two epochs later; compare at once
                        if (p == nullptr || memcmp(p, file_bytes(e*stride), stride) != 0) {
                            bad++;
                        }
                    } else {
                        const size_t o = e*stride + r() % stride;
                        const size_t n = std::min<size_t>(1 + r() % (2*stride), stride*n_expert - o);
                        std::vector<char> out(n);
                        if (!tier.read(host + o, n, out.data()) || memcmp(out.data(), file_bytes(o), n) != 0) {
                            bad++;
                        }
                    }
                }
            });
        }
        for (auto & t : th) {
            t.join();
        }
        CHECK(bad.load() == 0);
    }

    st = tier.stats();
    tier.print_stats("test");
    tier.remove(&tier);
    CHECK(tier.empty());
    CHECK(tier.stats().resident == 0);
    munmap(map, size);
    close(fd);
    unlink(name.data());
    printf("%s (%llu misses, %llu evictions, %.2f MiB read)\n", n_fail ? "FAILED" : "OK",
           (unsigned long long) st.misses, (unsigned long long) st.evictions, st.bytes_read/1048576.0);
    return n_fail ? 1 : 0;
}
#else
int main() {
    printf("skipped: POSIX only\n");
    return 0;
}
#endif
