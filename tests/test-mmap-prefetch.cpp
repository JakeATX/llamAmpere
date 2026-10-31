#include "../src/llama-mmap.h"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

struct advice_call {
    void * addr;
    size_t size;
    int kind;
};

static int mapped_flags;
static int failures = 0;
static std::vector<advice_call> advice;

extern "C" void * __real_mmap(void *, size_t, int, int, int, off_t);
extern "C" void * __wrap_mmap(void * addr, size_t size, int prot, int flags, int fd, off_t offset) {
    mapped_flags = flags;
    return __real_mmap(addr, size, prot, flags, fd, offset);
}

extern "C" int __wrap_posix_madvise(void * addr, size_t size, int kind) {
    advice.push_back({addr, size, kind});
    return 0;
}

static void expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 1;
    }
    const size_t page = size_t(page_size);
    std::vector<uint8_t> bytes(16 * page, 0x5a);
    FILE * temporary = std::tmpfile();
    if (!temporary) {
        return 1;
    }
    if (std::fwrite(bytes.data(), 1, bytes.size(), temporary) != bytes.size() || std::fflush(temporary)) {
        std::fclose(temporary);
        return 1;
    }
    {
        llama_file file(temporary);
        for (size_t prefix : {size_t(0), size_t(1), page, 2 * page, bytes.size() - 1, bytes.size(), SIZE_MAX}) {
            for (bool numa : {false, true}) {
                for (bool lazy : {false, true}) {
                    const llama_mmap::ranges lazy_ranges = lazy ? llama_mmap::ranges{{page, 3 * page}} : llama_mmap::ranges{};
                    advice.clear();
                    mapped_flags = 0;
                    llama_mmap mapping(&file, prefix, numa, lazy_ranges);
                    const bool populate = !numa && !lazy && prefix >= bytes.size();
                    expect(bool(mapped_flags & MAP_POPULATE) == populate, "whole-map population requires a full-file budget and no lazy ranges");
                    const size_t limit = numa ? 0 : (prefix >= bytes.size() ? bytes.size() : ((prefix + page - 1) / page) * page);
                    size_t advised = 0;
                    for (const auto & call : advice) {
                        if (call.kind != POSIX_MADV_WILLNEED) {
                            continue;
                        }
                        const size_t begin = size_t((char *) call.addr - (char *) mapping.addr());
                        expect(begin <= limit && call.size <= limit - begin, "prefetch advice stays within the page-rounded prefix");
                        expect(!lazy || begin >= 3 * page || begin + call.size <= page, "prefetch advice skips lazy pages");
                        advised += call.size;
                    }
                    expect(advised <= limit, "total prefetched bytes respect the finite budget");
                    expect(!limit || advised > 0, "nonzero prefetch reaches the advice syscall");
                    const auto * data = (const uint8_t *) mapping.addr();
                    expect(mapping.size() == bytes.size() && data[0] == 0x5a && data[bytes.size() - 1] == 0x5a,
                            "finite prefetch still maps every model byte");
                }
            }
        }
    }
    std::fclose(temporary);
    std::printf("%s mmap prefetch: 28 actual syscall cases; no model or GPU\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
