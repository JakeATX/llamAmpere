#include "../src/llama-mmap.h"
#include "../src/llama-ple-prefetch.h"
#include "../src/llama-moe-arena.h"
#include "../src/models/models.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

struct advice_call {
    void * addr;
    size_t size;
    int kind;
};

static std::atomic<size_t> fake_available{16ull << 30};
size_t llama_moe_arena::available_memory() { return fake_available.load(); }

static int mapped_flags;
static int failures = 0;
static std::vector<advice_call> advice;
static std::mutex advice_mutex;
static std::atomic<size_t> file_read_bytes{0}, file_advice_bytes{0}, mapped_advice_bytes{0};
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void * dst, size_t bytes, off_t offset) {
    file_read_bytes.fetch_add(bytes);
    return __real_pread(fd, dst, bytes, offset);
}
extern "C" int __wrap_posix_fadvise(int, off_t, off_t bytes, int kind) {
    if (kind == POSIX_FADV_WILLNEED) { file_advice_bytes.fetch_add(size_t(bytes)); }
    return 0;
}

extern "C" void * __real_mmap(void *, size_t, int, int, int, off_t);
extern "C" void * __wrap_mmap(void * addr, size_t size, int prot, int flags, int fd, off_t offset) {
    mapped_flags = flags;
    return __real_mmap(addr, size, prot, flags, fd, offset);
}

extern "C" int __wrap_posix_madvise(void * addr, size_t size, int kind) {
    std::lock_guard<std::mutex> lock(advice_mutex);
    advice.push_back({addr, size, kind});
    return 0;
}

extern "C" int __wrap_madvise(void * addr, size_t size, int kind) {
    std::lock_guard<std::mutex> lock(advice_mutex);
    advice.push_back({addr, size, kind});
    if (kind == MADV_WILLNEED
#if defined(MADV_POPULATE_READ)
            || kind == MADV_POPULATE_READ
#endif
    ) { mapped_advice_bytes.fetch_add(size); }
    return 0;
}

static void expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static void test_ple_readers() {
    constexpr uint64_t GiB = 1ull << 30;
    constexpr size_t page = 4096, rows = 64, bytes = rows*page;
    char path[] = "/tmp/llama-ple-fit-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, bytes)) { std::abort(); }
    void * data = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { std::abort(); }
    auto * buffer = ggml_backend_cpu_buffer_from_ptr(data, bytes);
    ggml_context * ctx = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
    auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, page/sizeof(float), rows);
    table->data = data;
    table->buffer = buffer;
    auto params = llama_model_default_params();
    params.fit_ram = true;
    {
        llama_model_qwen4exp model(params);
        model.arch = LLM_ARCH_QWEN4EXP;
        model.per_layer_tok_embd = table;
        model.can_prefetch.insert(table);
        auto & hp = model.hparams;
        hp.ple_ngram_size = 2;
        hp.ple_n_heads = hp.ple_heads_per_ngram = 1;
        hp.ple_head_vocab_sizes[0] = rows;
        hp.ple_head_offsets[0] = 0;
        hp.ple_eos_token_id = 999;
        hp.ple_layer_multipliers[0] = 1;
        hp.ple_layer_multipliers[1] = 0;
        setenv("LLAMA_PLE_PREFETCH", "2", 1);
        setenv("LLAMA_PLE_PF_WINDOW_UBATCHES", "0", 1);
        setenv("LLAMA_PLE_PF_COLD_THREADS", "0", 1);
        setenv("LLAMA_PLE_PF_TEST_DROP", "0", 1);
        setenv("LLAMA_PLE_DECODE", "0", 1);
        setenv("LLAMA_PLE_PF_MIN", "1", 1);
        setenv("LLAMA_PLE_PF_AHEAD", "1000000", 1);
        setenv("LLAMA_PLE_PF_POPULATE", "0", 1);
        setenv("LLAMA_PLE_PF_VERIFY", "2", 1);
        std::vector<llama_token> tokens(rows);
        for (size_t i = 0; i < rows; ++i) { tokens[i] = i; }
        // Protected pages trap any unbudgeted fault reader or decode-ahead load.
        for (const char * mode : {"fault", "pread", "advise", "coalesce"}) {
            setenv("LLAMA_PLE_PF_MODE", mode, 1);
            for (size_t credit : {size_t(0), 3*page}) {
                model.prefetch_remaining = credit;
                fake_available = 16ull << 30;
                if (mprotect(data, bytes, PROT_NONE) || (credit && mprotect(data, credit, PROT_READ))) { std::abort(); }
                auto first = llama_ple_prefetch::create(model, rows);
                auto second = llama_ple_prefetch::create(model, rows);
                expect(bool(first) && bool(second), "two actual PLE contexts use the same model");
                if (!first || !second) { std::abort(); }
                file_read_bytes = file_advice_bytes = mapped_advice_bytes = 0;
                first->hint(tokens.data(), tokens.size(), 0);
                second->hint(tokens.data(), tokens.size(), 0);
                expect(first->consume(rows, nullptr, 0) && second->consume(rows, nullptr, 0), "production readers drain all completion counters even with exhausted credit");
                first->stop();
                second->stop();
                expect(file_read_bytes + file_advice_bytes <= credit, "actual pread and file-advice syscalls stay within the shared page credit");
                expect(mapped_advice_bytes <= credit, "mapped advice stays within the shared page credit");
                expect(model.prefetch_remaining == 0, "PLE contexts share and exhaust the actual model counter");
                // Use one context after exhaustion, with every page inaccessible.
                if (mprotect(data, bytes, PROT_NONE)) { std::abort(); }
                file_read_bytes = file_advice_bytes = mapped_advice_bytes = 0;
                first->hint(tokens.data(), tokens.size(), 0);
                expect(first->consume(rows, nullptr, 0), "exhausted production readers cannot touch protected pages");
                first->stop();
                expect(file_read_bytes == 0 && file_advice_bytes == 0, "exhausted actual PLE readers make no file IO calls");
                expect(mapped_advice_bytes == 0, "exhausted actual PLE readers make no mapped read-ahead calls");
            }
        }
        // Coalescing must not read the holes between credited sparse pages.
        setenv("LLAMA_PLE_PF_MODE", "coalesce", 1);
        model.prefetch_remaining = 2*page;
        const llama_token sparse[] = {0, 2, 4};
        auto holes = llama_ple_prefetch::create(model, 3);
        if (!holes) { std::abort(); }
        file_read_bytes = 0;
        holes->hint(sparse, 3, 0);
        expect(holes->consume(3, nullptr, 0), "sparse production reader drains completion counters");
        holes->stop();
        expect(file_read_bytes == 2*page && model.prefetch_remaining == 0, "coalesced production reads cannot charge two pages then read a third hole");
        setenv("LLAMA_PLE_PF_MODE", "fault", 1);
        for (uint64_t available : {uint64_t(0), 4*GiB, 5*GiB}) {
            fake_available = available;
            model.prefetch_remaining = SIZE_MAX;
            auto pf = llama_ple_prefetch::create(model, rows);
            if (!pf) { std::abort(); }
            pf->hint(tokens.data(), tokens.size(), 0);
            expect(pf->consume(rows, nullptr, 0), "unknown and low RAM finish the actual worker without reads");
            pf->stop();
            expect(model.prefetch_remaining == 0, "production PLE reader reduces its quota on low or unknown RAM");
        }
        // A demand gather remains usable; the next optional decode must not read row one.
        fake_available = 16ull << 30;
        model.prefetch_remaining = 0;
        if (mprotect(data, page, PROT_READ)) { std::abort(); }
        setenv("LLAMA_PLE_DECODE", "1", 1);
        setenv("LLAMA_PLE_DECODE_PINNED", "0", 1);
        auto pf = llama_ple_prefetch::create(model, 1);
        if (!pf || !pf->decode_enabled()) { std::abort(); }
        pf->hint(tokens.data(), 2, 0);
        const int32_t row = 0;
        expect(pf->consume(1, &row, 1), "verification cannot refill an exhausted production quota");
        std::vector<float> output(page/sizeof(float), 1.0f);
        auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output.size(), 1);
        auto * out_buffer = ggml_backend_cpu_buffer_from_ptr(output.data(), page);
        dst->data = output.data();
        dst->buffer = out_buffer;
        pf->gather(1, &row, 1, dst);
        pf->stop();
        expect(std::all_of(output.begin(), output.end(), [](float value) { return value == 0.0f; }) && model.prefetch_remaining == 0,
                "demand decode stays exact while exhausted decode-ahead cannot touch row one");
        ggml_backend_buffer_free(out_buffer);
        pf.reset();
        // Fit OFF keeps the requested fault readers active and leaves quota unchanged.
        unsetenv("LLAMA_PLE_DECODE");
        model.params.fit_ram = false;
        if (mprotect(data, bytes, PROT_READ)) { std::abort(); }
        auto legacy = llama_ple_prefetch::create(model, rows);
        if (!legacy) { std::abort(); }
        legacy->hint(tokens.data(), tokens.size(), 0);
        expect(legacy->consume(rows, nullptr, 0), "fit-off retains production PLE fault readers");
        legacy->stop();
        expect(model.prefetch_remaining == 0, "fit-off does not debit the RAM-fit quota");
    }
    fake_available = 16ull << 30;
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    munmap(data, bytes);
    close(fd);
    unlink(path);
}

int main() {
    setenv("LLAMA_PREFETCH_THREADS", "4", 1);
    constexpr uint64_t GiB = 1ull << 30;
    constexpr size_t MiB = 1u << 20;
    for (uint64_t host : {16, 32, 64, 128}) {
        const uint64_t available = (host - 6) * GiB;
        const size_t expected = size_t(((host - 11) * 1024 * 4 / 5) / 10) * MiB;
        size_t budget = llama_mmap_prefetch_budget(available);
        expect(budget == expected, "fake host prefetch uses independently calculated eight percent of RAM above floor");
        const size_t first = llama_mmap_prefetch_take(budget, 1);
        const size_t second = llama_mmap_prefetch_take(budget, SIZE_MAX);
        expect(first == 1 && second == expected - MiB && budget == 0, "two shards share one page-rounded budget");
    }
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB}) {
        expect(llama_mmap_prefetch_budget(available) == 0, "unknown or low RAM disables optional prefetch");
    }
    size_t huge = llama_mmap_prefetch_budget(UINT64_MAX);
    const size_t huge_start = huge;
    expect(llama_mmap_prefetch_take(huge, SIZE_MAX) == huge_start && huge == 0, "extreme RAM cannot overflow the quota or debit");

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
    // A real mapping must exist before querying the policy, and the policy
    // must disable whole-map population even when the initial request is full.
    {
        llama_file file(temporary);
        struct policy_state { size_t remaining; int calls; bool mapped; } state{((bytes.size() + MiB - 1) / MiB) * MiB, 0, false};
        const auto limit = [](size_t requested, void * opaque) noexcept -> size_t {
            auto & state = *static_cast<policy_state *>(opaque);
            state.mapped = mapped_flags != 0;
            state.calls++;
            return llama_mmap_prefetch_take(state.remaining, requested);
        };
        for (int shard = 0; shard < 2; ++shard) {
            advice.clear();
            mapped_flags = 0;
            llama_mmap mapping(&file, SIZE_MAX, false, {}, +limit, &state);
            expect(!(mapped_flags & MAP_POPULATE), "post-map policy prevents eager whole-map population");
            size_t advised = 0;
            for (const auto & call : advice) {
                if (call.kind == POSIX_MADV_WILLNEED) { advised += call.size; }
            }
            expect(shard == 0 ? advised == bytes.size() : advised == 0, "next shard cannot reuse first shard's page-rounded quota");
        }
        expect(state.calls == 2 && state.mapped && state.remaining == 0, "policy runs after each actual mmap and shares one budget");
        advice.clear();
        llama_mmap mapping(&file, 0);
        std::atomic<size_t> later{state.remaining};
        llama_prefetch({{mapping.addr(), bytes.size()}}, &later, 128 * GiB);
        expect(advice.empty() && later == 0, "model handoff cannot reuse quota already spent during mapping");
    }
    {
        llama_file file(temporary);
        llama_mmap mapping(&file, 0);
        auto * base = static_cast<char *>(mapping.addr());
        auto total = [&]() {
            size_t bytes = 0;
            for (const auto & call : advice) {
                if (call.kind == MADV_WILLNEED) {
                    const uintptr_t start = uintptr_t(call.addr);
                    expect(start >= uintptr_t(base) && start <= uintptr_t(base) + mapping.size(), "later advice belongs to this mapping");
                    expect(call.size <= uintptr_t(base) + mapping.size() - start, "later advice does not cross the mapping");
                    expect(start % page == 0 && call.size % page == 0, "later advice accounts for whole pages");
                    bytes += call.size;
                }
            }
            return bytes;
        };
        advice.clear();
        std::atomic<size_t> quota{3 * page};
        llama_prefetch({{base + 8 * page + 1, page}, {base + 1, page}, {base + page, page}}, &quota, 16 * GiB);
        expect(total() == 3 * page && quota == 0, "sorted overlapping and sparse rows share one exact physical-page quota");
        llama_prefetch({{base, 8 * page}}, &quota, 128 * GiB);
        expect(total() == 3 * page && quota == 0, "later calls and higher free RAM cannot refill spent quota");
        advice.clear();
        quota = 2 * page;
        llama_prefetch({{base, page}}, &quota, 16 * GiB);
        llama_prefetch({{base + 4 * page, 4 * page}}, &quota, 16 * GiB);
        expect(total() == 2 * page && quota == 0, "separate ubatches cannot reuse a prior call's quota");
        for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB}) {
            advice.clear();
            quota = SIZE_MAX;
            llama_prefetch({{base, 4 * page}}, &quota, available);
            expect(total() == 0 && quota == 0, "unknown or low current RAM stops later advice before any syscall");
        }
        advice.clear();
        llama_prefetch({{base, 4 * page}});
        expect(total() == 4 * page, "fit-off keeps legacy unlimited advice");
        for (uint64_t host : {16, 32, 64, 128}) {
            advice.clear();
            quota = SIZE_MAX;
            const uint64_t available = (host - 6) * GiB;
            const size_t expected = size_t(((host - 11) * 1024 * 4 / 5) / 10) * MiB;
            llama_prefetch({{base, page}}, &quota, available);
            expect(total() == page && quota == expected - page, "later quota uses the independent fake-host budget before its first syscall");
        }
        advice.clear();
        quota = 3 * page;
        auto read_ahead = [&]() { llama_prefetch({{base, 4 * page}}, &quota, 16 * GiB); };
        std::thread first(read_ahead), second(read_ahead);
        first.join();
        second.join();
        expect(total() == 3 * page && quota == 0, "concurrent contexts cannot double spend the model quota");
        advice.clear();
        llama_prefetch({{reinterpret_cast<void *>(UINTPTR_MAX - 1), 16}});
        expect(advice.empty(), "overflowing ranges cannot issue advice to unrelated memory");
        expect(base[0] == char(0x5a) && base[bytes.size() - 1] == char(0x5a), "capping optional read-ahead preserves all model bytes");
    }
    std::fclose(temporary);
    test_ple_readers();
    std::printf("%s mmap prefetch: baseline syscalls + shared startup/later quotas + concurrent contexts + fake RAM hosts; no model or GPU\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
