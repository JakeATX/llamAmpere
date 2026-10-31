#include "fit.h"
#include "arg.h"
#include "common.h"
#include "ngram-cache.h"
#include "speculative.h"
#include "../src/llama-moe-arena.h"
#include "../src/llama-b1.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <utility>

#ifdef __linux__
#include <unistd.h>
#endif

namespace allocation_probe {
static thread_local bool active = false;
static thread_local size_t largest = 0;
static thread_local size_t total = 0;
}

void * operator new(size_t n) {
    if (allocation_probe::active) {
        allocation_probe::largest = std::max(allocation_probe::largest, n);
        allocation_probe::total += n;
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

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static std::vector<char> ngram_old_file_bytes(const common_ngram_cache & cache, common_ngram_cache_vocab_id vocab) {
    std::vector<char> payload;
    uint64_t n_ngrams = 0;
    const auto append = [](std::vector<char> & dst, const void * data, size_t size) {
        const char * bytes = static_cast<const char *>(data);
        dst.insert(dst.end(), bytes, bytes + size);
    };
    for (const auto & item : cache) {
        int32_t n_tokens = 0;
        for (const auto & token : item.second) {
            n_tokens += token.second > 0;
        }
        if (!n_tokens) {
            continue;
        }
        append(payload, item.first.tokens, sizeof(item.first.tokens));
        append(payload, &n_tokens, sizeof(n_tokens));
        for (const auto & token : item.second) {
            if (token.second > 0) {
                append(payload, &token.first, sizeof(token.first));
                append(payload, &token.second, sizeof(token.second));
            }
        }
        ++n_ngrams;
    }
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : payload) {
        hash = (hash ^ byte) * 1099511628211ULL;
    }
    std::vector<char> result = {'L', 'N', 'G', 'C'};
    const uint32_t version = 2;
    const uint64_t size = payload.size();
    append(result, &version, sizeof(version));
    append(result, &vocab.n_tokens, sizeof(vocab.n_tokens));
    append(result, &vocab.hash, sizeof(vocab.hash));
    append(result, &n_ngrams, sizeof(n_ngrams));
    append(result, &size, sizeof(size));
    append(result, &hash, sizeof(hash));
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

static void test_ngram_ram_load_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    common_params_speculative_ngram_cache params;
    params.lookup_cache_static = "static";
    params.lookup_cache_dynamic = "dynamic";
    uint64_t previous = 0;
    for (uint64_t ram : {16ull, 32ull, 64ull, 128ull}) {
        common_fit_ngram_cache_load_limits(ram * GiB, true, params);
        expect(params.load_max_entries_static > previous &&
                params.load_max_entries_static == params.load_max_entries_dynamic &&
                params.load_max_continuations_static == params.load_max_continuations_dynamic,
                "automatic lookup load quotas scale with fake machine RAM and split across both files");
        expect(params.load_max_entries_static * 512 + params.load_max_continuations_static * 128 <=
                (ram * GiB - 5 * GiB) / 2,
                "lookup load count estimates fit within spare RAM");
        previous = params.load_max_entries_static;
    }
    const uint64_t both_entries = params.load_max_entries_static;
    params.max_ngrams_dynamic = 0;
    params.max_ngrams_dynamic_explicit = true;
    common_fit_ngram_cache_load_limits(128 * GiB, true, params);
    expect(params.max_ngrams_dynamic == 0 && params.load_max_entries_dynamic == UINT64_MAX &&
            params.load_max_continuations_dynamic == UINT64_MAX && params.load_max_entries_static > both_entries,
            "explicit unlimited dynamic size wins while the static file and context stay automatic");
    params.max_ngrams_dynamic = 17;
    common_fit_ngram_cache_load_limits(0, true, params);
    expect(params.max_ngrams_dynamic == 17 && params.load_max_entries_dynamic == UINT64_MAX &&
            params.load_max_entries_static == 0, "explicit positive dynamic size wins even with unknown RAM");
    params.max_ngrams_dynamic_explicit = false;
    for (uint64_t ram : std::array<uint64_t, 3>{0, 4 * GiB, 5 * GiB}) {
        common_fit_ngram_cache_load_limits(ram, true, params);
        expect(params.load_max_entries_static == 0 && params.load_max_entries_dynamic == 0 &&
                params.load_max_continuations_static == 0 && params.load_max_continuations_dynamic == 0,
                "low or unknown RAM refuses nonempty automatic lookup loads");
    }
    common_fit_ngram_cache_load_limits(0, false, params);
    expect(params.load_max_entries_static == UINT64_MAX && params.load_max_entries_dynamic == UINT64_MAX &&
            params.load_max_continuations_static == UINT64_MAX && params.load_max_continuations_dynamic == UINT64_MAX,
            "fit off resets prior quotas and preserves unlimited lookup loading");
    common_fit_ngram_cache_load_limits(UINT64_MAX, true, params);
    expect(params.load_max_entries_static > 0 && params.load_max_entries_static < UINT64_MAX / 1024 &&
            params.load_max_continuations_static < UINT64_MAX / 256,
            "lookup count planning handles the largest available-RAM value without overflow");

    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-ram-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create private RAM lookup load test directory");
    if (!created) {
        return;
    }
    params.lookup_cache_static.clear();
    params.lookup_cache_dynamic = (dir / "dynamic").string();
    common_fit_ngram_cache_load_limits(5 * GiB + (64ull << 20), true, params);
    expect(params.load_max_entries_dynamic > 0 && params.load_max_entries_dynamic < 4096,
            "small spare RAM gives a bounded nonzero lookup load quota");
    if (params.load_max_entries_dynamic == 0 || params.load_max_entries_dynamic >= 4096) {
        fs::remove_all(dir);
        return;
    }
    common_ngram_cache source;
    for (llama_token token = 0; uint64_t(token) < params.load_max_entries_dynamic; ++token) {
        source[common_ngram(&token, 1)] = {{1, 2}};
    }
    const common_ngram_cache_vocab_id vocab = {100000, 123};
    std::string err;
    expect(common_ngram_cache_save_file(source, params.lookup_cache_dynamic, vocab, err), "write exact-quota lookup file");
    common_ngram_cache loaded;
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
            params.load_max_entries_dynamic, params.load_max_continuations_dynamic) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == source,
            "RAM-derived key quota accepts a real file exactly at its limit");
    const llama_token extra_token = llama_token(params.load_max_entries_dynamic);
    source[common_ngram(&extra_token, 1)] = {{1, 2}};
    expect(common_ngram_cache_save_file(source, params.lookup_cache_dynamic, vocab, err), "write over-quota lookup file");
    const auto read = [&]() {
        std::ifstream in(params.lookup_cache_dynamic, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in), {});
    };
    const auto before = read();
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
            params.load_max_entries_dynamic, params.load_max_continuations_dynamic) == COMMON_NGRAM_CACHE_LOAD_LIMIT && loaded.empty(),
            "RAM-derived quotas refuse a real oversized file before map insertion");
    const llama_token token = 0;
    common_ngram_cache live = {{common_ngram(&token, 1), {{1, 3}}}};
    common_ngram_cache delta = {{common_ngram(&token, 1), {{1, 1}}}};
    const auto live_before = live;
    const auto delta_before = delta;
    expect(!common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, nullptr,
            params.load_max_entries_dynamic, params.load_max_continuations_dynamic) &&
            live == live_before && delta == delta_before && read() == before,
            "RAM quota refusal on checkpoint preserves the entire file, live cache and pending counts");
    common_fit_ngram_cache_load_limits(0, false, params);
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
            params.load_max_entries_dynamic, params.load_max_continuations_dynamic) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == source,
            "the same file still loads with fit off");
    fs::remove_all(dir);
}

static void test_ngram_context_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> requests = {};
    for (auto & request : requests) {
        request.bytes = UINT64_MAX;
    }
    uint64_t previous = 0;
    for (uint64_t ram : {16ull, 32ull, 64ull, 128ull}) {
        common_params_speculative_ngram_cache params;
        params.lookup_cache_static = "static";
        params.lookup_cache_dynamic = "dynamic";
        common_fit_ngram_cache_load_limits(ram * GiB, true, params, 4);
        expect(params.max_entries_context > previous &&
                params.max_entries_context == params.load_max_entries_static &&
                params.max_entries_context == params.load_max_entries_dynamic &&
                params.max_continuations_context == params.load_max_continuations_dynamic,
                "RAM lookup share is split across both files and all four slots");
        const uint64_t estimate = params.max_entries_context * 512 + params.max_continuations_context * 128;
        expect(estimate * 6 <= common_fit_ram_plan(ram * GiB, requests).bytes[COMMON_FIT_RAM_NGRAM],
                "all file and slot count estimates together stay within the ngram share");
        previous = params.max_entries_context;
    }

    common_params_speculative_ngram_cache params;
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 1);
    const uint64_t single = params.max_entries_context;
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 4);
    expect(params.max_entries_context > 0 && params.max_entries_context < single / 3 &&
            params.load_max_entries_dynamic == UINT64_MAX && params.load_max_entries_static == UINT64_MAX,
            "context-only lookup fitting scales with slots without loading files");
    params.max_ngrams_dynamic_explicit = true;
    params.lookup_cache_dynamic = "dynamic";
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 4);
    expect(params.load_max_entries_dynamic == UINT64_MAX && params.max_entries_context < single / 3,
            "explicit dynamic size does not remove automatic context admission");
    for (uint64_t available : std::array<uint64_t, 3>{0, 4 * GiB, 5 * GiB}) {
        common_fit_ngram_cache_load_limits(available, true, params, 4);
        expect(params.max_entries_context == 0 && params.max_continuations_context == 0,
                "low or unknown RAM disables new automatic context pairs");
    }
    common_fit_ngram_cache_load_limits(0, false, params, 4);
    expect(params.max_entries_context == UINT64_MAX && params.max_continuations_context == UINT64_MAX,
            "fit off resets both context quotas");
    common_fit_ngram_cache_load_limits(UINT64_MAX, true, params, UINT32_MAX);
    expect(params.max_entries_context < single && params.max_continuations_context < UINT64_MAX,
            "extreme available RAM and slot counts do not overflow quota arithmetic");
}

static void test_ngram_checkpoint_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> requests = {};
    for (auto & request : requests) {
        request.bytes = UINT64_MAX;
    }
    for (uint64_t ram : {16ull, 32ull, 64ull, 128ull}) {
        const uint64_t budget = common_fit_ram_plan(ram * GiB, requests).bytes[COMMON_FIT_RAM_NGRAM];
        uint64_t previous = UINT64_MAX;
        for (int mode : {0, 1, 2}) {
            common_params_speculative_ngram_cache params;
            params.lookup_cache_static = "static";
            params.lookup_cache_dynamic = "dynamic";
            params.save_dynamic = mode > 0;
            params.save_dynamic_interval = mode == 2 ? 300 : 0;
            common_fit_ngram_cache_load_limits(ram * GiB, true, params, 4);
            expect(params.load_max_entries_dynamic > 0 && params.load_max_entries_dynamic < previous &&
                    params.load_max_entries_dynamic == params.load_max_entries_static &&
                    params.load_max_entries_dynamic == params.max_entries_context &&
                    params.load_max_continuations_dynamic == params.load_max_continuations_static &&
                    params.load_max_continuations_dynamic == params.max_continuations_context,
                    "saving modes reduce per-map quotas and retain equal file and slot shares");
            // Live, delta and merged maps coexist at shutdown; periodic saves also retain a worker delta.
            const uint64_t maps = mode == 0 ? 6 : mode == 1 ? 8 : 9;
            const uint64_t estimate = params.load_max_entries_dynamic * 512 +
                    params.load_max_continuations_dynamic * 128 + 8192;
            expect(estimate <= budget / maps,
                    "all simultaneous checkpoint, file and slot estimates fit the ngram RAM share");
            previous = params.load_max_entries_dynamic;
        }
    }

    common_params_speculative_ngram_cache params;
    params.lookup_cache_dynamic = "dynamic";
    params.save_dynamic = true;
    params.save_dynamic_interval = 300;
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 0);
    const uint64_t periodic_entries = params.load_max_entries_dynamic;
    const uint64_t budget = common_fit_ram_plan(16 * GiB, requests).bytes[COMMON_FIT_RAM_NGRAM];
    expect(periodic_entries > 0 && params.max_entries_context == 0 &&
            (periodic_entries * 512 + params.load_max_continuations_dynamic * 128 + 8192) <= budget / 4,
            "dynamic-only periodic saving budgets every map without static files or slots");
    params.save_dynamic_interval = -1;
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 0);
    expect(params.load_max_entries_dynamic > periodic_entries,
            "nonpositive checkpoint interval uses the shutdown-only memory budget");

    params.lookup_cache_dynamic.clear();
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 4);
    const uint64_t context_only = params.max_entries_context;
    params.save_dynamic = false;
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 4);
    expect(params.max_entries_context == context_only && params.load_max_entries_dynamic == UINT64_MAX,
            "saving without a dynamic path does not reserve phantom checkpoint maps");
    common_fit_ngram_cache_load_limits(16 * GiB, true, params, 0);
    expect(params.load_max_entries_dynamic == UINT64_MAX && params.max_entries_context == UINT64_MAX,
            "no files and no slots preserve the no-consumer path");

    params.lookup_cache_dynamic = "dynamic";
    params.save_dynamic = true;
    params.max_ngrams_dynamic_explicit = true;
    for (uint32_t explicit_size : {0u, 17u}) {
        params.max_ngrams_dynamic = explicit_size;
        common_fit_ngram_cache_load_limits(16 * GiB, true, params, 4);
        expect(params.max_ngrams_dynamic == explicit_size && params.load_max_entries_dynamic == UINT64_MAX &&
                params.load_max_continuations_dynamic == UINT64_MAX && params.max_entries_context == context_only,
                "explicit dynamic sizes preserve their override without charging automatic slots for their copies");
    }
    params.max_ngrams_dynamic_explicit = false;
    params.save_dynamic_interval = 300;
    for (uint64_t ram : std::array<uint64_t, 3>{0, 4 * GiB, 5 * GiB}) {
        common_fit_ngram_cache_load_limits(ram, true, params, 4);
        expect(params.load_max_entries_dynamic == 0 && params.load_max_continuations_dynamic == 0 &&
                params.max_entries_context == 0 && params.max_continuations_context == 0,
                "low or unknown RAM disables automatic checkpoint and context growth");
    }
    common_fit_ngram_cache_load_limits(0, false, params, 4);
    expect(params.load_max_entries_dynamic == UINT64_MAX && params.load_max_continuations_dynamic == UINT64_MAX &&
            params.max_entries_context == UINT64_MAX && params.max_continuations_context == UINT64_MAX,
            "fit off resets checkpoint and context quotas to unlimited");
    common_fit_ngram_cache_load_limits(UINT64_MAX, true, params, UINT32_MAX);
    const uint64_t maps = uint64_t(UINT32_MAX) + 4;
    const uint64_t estimate = params.max_entries_context * 512 + params.max_continuations_context * 128 + 8192;
    expect(params.load_max_entries_dynamic > 0 && params.load_max_entries_dynamic == params.max_entries_context &&
            estimate <= common_fit_ram_plan(UINT64_MAX, requests).bytes[COMMON_FIT_RAM_NGRAM] / maps,
            "maximum RAM and slot values keep checkpoint aggregate estimates within budget");
}

static void test_ngram_context_admission() {
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_CACHE};
    params.ngram_cache.max_entries_context = 1;
    params.ngram_cache.max_continuations_context = 1;
    common_speculative_ptr spec(common_speculative_init(params, 2));
    expect(spec != nullptr, "context-only production lookup drafter initializes without a model");
    if (!spec) {
        return;
    }
    const auto draft = [&](llama_seq_id seq, const llama_tokens & prompt, llama_tokens & result) {
        result.clear();
        common_speculative_begin(spec.get(), seq, prompt);
        auto & dp = common_speculative_get_draft_params(spec.get(), seq);
        dp.drafting = true;
        dp.id_last = prompt.back();
        dp.prompt = &prompt;
        dp.result = &result;
        dp.n_max = 8;
        common_speculative_draft(spec.get());
    };
    llama_tokens prompt(20, 7), result;
    draft(0, prompt, result);
    expect(!result.empty() && std::all_of(result.begin(), result.end(), [](llama_token t) { return t == 7; }),
            "retained context pair can draft at capacity");
    llama_tokens other(20, 9);
    draft(1, other, result);
    expect(!result.empty() && result.front() == 9, "slots have independent context credit");
    prompt.assign(12, 11);
    draft(0, prompt, result);
    expect(!result.empty() && result.front() == 11, "shortened prompt resets context and successor credit");

    const auto allocated = [&](uint64_t entries, uint64_t continuations) {
        params.ngram_cache.max_entries_context = entries;
        params.ngram_cache.max_continuations_context = continuations;
        spec.reset(common_speculative_init(params, 2));
        llama_tokens large(4097);
        for (size_t i = 0; i < large.size(); ++i) {
            large[i] = llama_token(i + 100);
        }
        result.clear();
        allocation_probe::total = 0;
        allocation_probe::active = true;
        draft(0, large, result);
        allocation_probe::active = false;
        return allocation_probe::total;
    };
    const size_t bounded = allocated(1, 1);
    expect(bounded < 131072, "long production context draft admits only its bounded map nodes");
    expect(allocated(0, UINT64_MAX) < 131072 && allocated(UINT64_MAX, 0) < 131072,
            "either zero context quota prevents map growth");
    expect(allocated(UINT64_MAX, UINT64_MAX) > bounded * 2,
            "unlimited positive control allocates the long context's map nodes");
}

static void test_ngram_checkpoint_load_quotas() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-checkpoint-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create private lookup checkpoint quota test directory");
    if (!created) {
        return;
    }
    common_params_speculative_ngram_cache params;
    params.lookup_cache_dynamic = (dir / "dynamic").string();
    common_fit_ngram_cache_load_limits(5ull * 1024 * 1024 * 1024 + (64ull << 20), true, params);
    const uint64_t max_entries = params.load_max_entries_dynamic;
    expect(max_entries > 0 && max_entries < 4096, "checkpoint regression uses real bounded RAM plan");
    if (max_entries == 0 || max_entries >= 4096) {
        fs::remove_all(dir);
        return;
    }
    common_ngram_cache live;
    for (llama_token token = 0; uint64_t(token) < max_entries; ++token) {
        live[common_ngram(&token, 1)] = {{1, 1}};
    }
    const common_ngram_cache_vocab_id vocab = {100000, 123};
    std::string err;
    expect(common_ngram_cache_save_file(live, params.lookup_cache_dynamic, vocab, err), "seed exact-quota checkpoint");
    const llama_token token = llama_token(max_entries);
    const common_ngram hot(&token, 1);
    const common_ngram retained = live.begin()->first;
    common_ngram_cache delta = {{hot, {{1, 10}}}, {retained, {{1, 2}}}};
    common_ngram_cache_merge(live, delta);
    bool from_disk = false;
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 1000000, vocab, err, &from_disk,
            max_entries, params.load_max_continuations_dynamic) && from_disk && delta.empty() &&
            live.size() == max_entries && !live.count(hot) && live.at(retained).at(1) == 3,
            "first checkpoint bounds new nodes but retains count learning at the automatic load quota");
    common_ngram_cache loaded;
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
            max_entries, params.load_max_continuations_dynamic) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == live,
            "first checkpoint is reloadable under the same RAM plan");
    delta = {{retained, {{1, 1}}}};
    common_ngram_cache_merge(live, delta);
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 1000000, vocab, err, &from_disk,
            max_entries, params.load_max_continuations_dynamic) && from_disk && delta.empty() &&
            live.at(retained).at(1) == 4,
            "second checkpoint still saves new learning instead of failing LOAD_LIMIT");
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
            max_entries, params.load_max_continuations_dynamic) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == live,
            "second key-limited checkpoint reloads with the same quotas");

    // Unlimited controls reproduce the old oversized output, then the bounded reader refuses it.
    delta = {{common_ngram(&token, 1), {{2, 3}, {3, 3}, {4, 3}}}};
    common_ngram_cache_merge(live, delta);
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err),
            "explicit unlimited checkpoint preserves prior behavior");
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err, UINT64_MAX, max_entries) ==
            COMMON_NGRAM_CACHE_LOAD_LIMIT, "unlimited output is a positive control for continuation overflow");
    fs::remove(params.lookup_cache_dynamic);
    const llama_token other_token = 0;
    const common_ngram other(&other_token, 1);
    live = {{hot, {{1, 2}, {2, 2}}}, {other, {{1, 1}}}};
    expect(common_ngram_cache_save_file(live, params.lookup_cache_dynamic, vocab, err), "seed exact continuation quota");
    delta = {{hot, {{3, 1}}}};
    common_ngram_cache_merge(live, delta);
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, &from_disk, 2, 3) &&
            from_disk && delta.empty(), "checkpoint bounds continuations after merging new successors");
    expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err, 2, 3) == COMMON_NGRAM_CACHE_LOAD_OK &&
            loaded == live && !loaded.empty(), "continuation-limited checkpoint reloads with retained learning");
    if (!live.empty()) {
        delta = {{live.begin()->first, {{1, 1}}}};
        common_ngram_cache_merge(live, delta);
        expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, &from_disk, 2, 3) &&
                from_disk && delta.empty(), "second continuation-limited checkpoint accepts an existing successor");
        expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err, 2, 3) == COMMON_NGRAM_CACHE_LOAD_OK &&
                loaded == live, "second continuation-limited output also reloads");
    }
    fs::remove(params.lookup_cache_dynamic);
    live = {{hot, {{1, 1}}}};
    delta = live;
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, nullptr, 0, 0) &&
            live.empty() && delta.empty(), "zero automatic quotas publish an empty cache instead of unlimited entries");
    live = {{hot, {{1, 1}}}};
    delta = live;
    expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, &from_disk, 0, 0) &&
            from_disk && live.empty() && delta.empty(), "zero-quota output stays reloadable at the next checkpoint");
    const std::pair<uint64_t, uint64_t> zero_quotas[] = {{0, UINT64_MAX}, {UINT64_MAX, 0}};
    for (const auto limits : zero_quotas) {
        fs::remove(params.lookup_cache_dynamic);
        live = {{hot, {{1, 1}}}};
        delta = live;
        expect(common_ngram_cache_sync_file(params.lookup_cache_dynamic, live, delta, 0, vocab, err, nullptr,
                limits.first, limits.second) && live.empty() && delta.empty(), "each zero quota independently clears saved output");
        expect(common_ngram_cache_load_file(params.lookup_cache_dynamic, loaded, vocab, err,
                limits.first, limits.second) == COMMON_NGRAM_CACHE_LOAD_OK && loaded.empty(), "each independent zero-quota output reloads");
    }
    fs::remove_all(dir);
}

static void test_ngram_checkpoint_merge_allocations() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-merge-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create private checkpoint allocation test directory");
    if (!created) {
        return;
    }
    const std::string path = (dir / "dynamic").string();
    const common_ngram_cache_vocab_id vocab = {100000, 123};
    const llama_token token = 0;
    const common_ngram key(&token, 1);
    const common_ngram_cache seed = {{key, {{1, 10}}}};
    common_ngram_cache source;
    for (llama_token t = 0; t < 64; ++t) {
        auto & part = source[common_ngram(&t, 1)];
        for (llama_token next = 1000; next < 1256; ++next) {
            part.emplace(next, 1);
        }
    }
    source[key].emplace(1, 3);
    std::string err;
    size_t bounded_max_bytes = 0;
    {
        // The first warning in the process sets up the logger; keep that one-time cost out of the probe.
        std::ofstream corrupt(path, std::ios::binary);
        corrupt << "bad";
        corrupt.close();
        common_ngram_cache warm_live = seed;
        common_ngram_cache warm_delta = seed;
        common_ngram_cache_sync_file(path, warm_live, warm_delta, 0, vocab, err, nullptr, 1, 1);
    }
    for (uint64_t max_continuations : {uint64_t(1), UINT64_MAX}) {
        for (int mode = 0; mode < 4; ++mode) {
            fs::remove(path);
            if (mode == 0) {
                expect(common_ngram_cache_save_file(seed, path, vocab, err), "seed full checkpoint allocation control");
            } else if (mode == 2) {
                std::ofstream corrupt(path, std::ios::binary);
                corrupt << "bad";
            } else if (mode == 3) {
                auto bytes = ngram_old_file_bytes(seed, vocab);
                bytes.back() ^= 1;
                std::ofstream corrupt(path, std::ios::binary);
                corrupt.write(bytes.data(), bytes.size());
            }
            common_ngram_cache live = source;
            common_ngram_cache delta = source;
            bool from_disk = false;
            allocation_probe::total = 0;
            allocation_probe::active = true;
            const bool ok = common_ngram_cache_sync_file(path, live, delta, 0, vocab, err, &from_disk, 1, max_continuations);
            allocation_probe::active = false;
            const size_t allocated = allocation_probe::total;
            bounded_max_bytes = std::max(bounded_max_bytes, allocated);
            expect(ok && delta.empty() && live.size() == 1 && live.begin()->second.size() <= max_continuations && from_disk == (mode == 0),
                    "loaded, missing and corrupt checkpoints bound merge nodes before publication");
            expect(allocated < 128 * 1024, "bounded checkpoint allocates less than 128 KiB despite 16385 proposed pairs");
            if (mode == 0) {
                expect(live.at(key).at(1) == 13, "full saved quota still learns retained counts exactly once");
            }
            common_ngram_cache reloaded;
            expect(common_ngram_cache_load_file(path, reloaded, vocab, err, 1, max_continuations) == COMMON_NGRAM_CACHE_LOAD_OK && reloaded == live,
                    "bounded merge output reloads under the same quotas");
        }
    }
    fs::remove(path);
    common_ngram_cache live = source;
    common_ngram_cache delta = source;
    allocation_probe::total = 0;
    allocation_probe::active = true;
    const bool ok = common_ngram_cache_sync_file(path, live, delta, 0, vocab, err);
    allocation_probe::active = false;
    expect(ok && delta.empty() && live == source && allocation_probe::total > 2 * bounded_max_bytes,
            "unlimited positive control retains all pairs and shows the old full-copy allocation cost");
    fs::remove_all(dir);
}

static void test_ngram_worker_fallback() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-worker-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create private worker fallback test directory");
    if (!created) {
        return;
    }
    const std::string path = (dir / "dynamic").string();
    const common_ngram_cache_vocab_id vocab = {100000, 123};
    const llama_token token = 0;
    const common_ngram key(&token, 1);
    const common_ngram_cache seed = {{key, {{1, 10}}}};
    common_ngram_cache source = {{key, {{1, 3}}}};
    for (llama_token t = 100; t < 16484; ++t) {
        source[common_ngram(&t, 1)].emplace(1, 1);
    }
    std::string err;
    for (uint64_t quota : {uint64_t(1), UINT64_MAX}) {
        for (int mode = 0; mode < 3; ++mode) {
            size_t copied_bytes = 0;
            for (bool borrow_delta : {false, true}) {
                fs::remove(path);
                if (mode == 0) {
                    expect(common_ngram_cache_save_file(seed, path, vocab, err), "seed worker fallback file");
                } else if (mode == 2) {
                    auto bytes = ngram_old_file_bytes(seed, vocab);
                    bytes.back() ^= 1;
                    std::ofstream corrupt(path, std::ios::binary);
                    corrupt.write(bytes.data(), bytes.size());
                }
                common_ngram_cache cache;
                common_ngram_cache delta = source;
                bool from_disk = false;
                allocation_probe::total = 0;
                allocation_probe::active = true;
                if (!borrow_delta) {
                    cache = delta;
                }
                const bool ok = common_ngram_cache_sync_file(path, cache, delta, 0, vocab, err, &from_disk,
                        quota, quota, borrow_delta);
                allocation_probe::active = false;
                const size_t allocated = allocation_probe::total;
                expect(ok && delta.empty() && from_disk == (mode == 0), "worker fallback saves loaded, missing and late-corrupt files");
                if (quota == 1) {
                    expect(cache.size() == 1 && cache.begin()->second.size() == 1, "worker fallback respects both quotas");
                    if (mode == 0) {
                        expect(cache == common_ngram_cache({{key, {{1, 13}}}}), "worker adds known counts exactly once");
                    }
                    if (borrow_delta) {
                        expect(allocated < 128 * 1024 && copied_bytes > 2 * allocated,
                                "worker avoids its full input copy before bounded merge; copied control allocates more");
                    }
                } else {
                    common_ngram_cache expected = source;
                    if (mode == 0) {
                        expected[key][1] += 10;
                    }
                    expect(cache == expected, "unlimited worker fallback retains every pair with exact counts");
                }
                if (!borrow_delta) {
                    copied_bytes = allocated;
                }
                common_ngram_cache reloaded;
                expect(common_ngram_cache_load_file(path, reloaded, vocab, err, quota, quota) == COMMON_NGRAM_CACHE_LOAD_OK &&
                        reloaded == cache, "worker output reloads at the same quotas");
            }
        }
    }
#ifdef __linux__
    expect(common_ngram_cache_save_file(seed, path, vocab, err), "seed worker failure control");
    const auto bytes_before = ngram_old_file_bytes(seed, vocab);
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    expect(fs::create_directory(tmp), "block worker temporary-file creation");
    common_ngram_cache cache = seed;
    common_ngram_cache delta = source;
    expect(!common_ngram_cache_sync_file(path, cache, delta, 0, vocab, err, nullptr, 1, 1, true) &&
            !err.empty() && cache == seed && delta == source, "failed worker write preserves result and retry delta");
    std::ifstream saved(path, std::ios::binary);
    expect(std::vector<char>(std::istreambuf_iterator<char>(saved), {}) == bytes_before, "failed worker write preserves the old file");
    saved.close();
    expect(fs::remove(tmp), "remove worker temporary-file blocker");
    expect(common_ngram_cache_sync_file(path, cache, delta, 0, vocab, err, nullptr, 1, 1, true) &&
            delta.empty() && cache.size() == 1 && cache.at(key).at(1) == 13,
            "retry writes the retained worker count exactly once");
    common_ngram_cache reloaded;
    expect(common_ngram_cache_load_file(path, reloaded, vocab, err, 1, 1) == COMMON_NGRAM_CACHE_LOAD_OK &&
            reloaded == cache, "successful worker retry reloads with the same exact count");
#endif
    fs::remove_all(dir);
}

static void test_ngram_live_successor_budget() {
    const llama_token a_token = 1;
    const llama_token b_token = 2;
    const common_ngram a(&a_token, 1);
    const common_ngram b(&b_token, 1);
    const auto successors = [](const common_ngram_cache & cache) {
        uint64_t result = 0;
        for (const auto & entry : cache) {
            result += entry.second.size();
        }
        return result;
    };
    common_ngram_cache live = {{a, {{2, 10}}}};
    common_ngram_cache_limits limits(live, 2, 3);
    common_ngram_cache delta;
    std::vector<llama_token> tokens = {1, 2, 1, 3, 1, 4, 1, 2};
    common_ngram_cache unlimited = live;
    common_ngram_cache_update(unlimited, 1, 1, tokens, tokens.size(), false);
    common_ngram_cache_update(live, 1, 1, tokens, tokens.size(), false, 0, nullptr, &limits);
    common_ngram_cache_update(delta, 1, 1, tokens, tokens.size(), false, 0, &live);
    expect(live.size() == 2 && successors(live) == 3 && limits.continuations_left == 0,
            "live admission consumes one credit per new successor and enforces the entry cap");
    expect(live.at(a).at(2) == 12 && live.at(a).at(3) == 1 && live.at(a).count(4) == 0,
            "exhausted successor budget retains increments for known pairs");
    expect(unlimited.at(a).count(4) && unlimited.size() > live.size(), "unlimited update reproduces unbounded successor growth");
    expect(delta.at(a).at(2) == 2 && delta.at(a).at(3) == 1 && delta.at(a).count(4) == 0,
            "writeback delta rejects successors refused by live admission");

    const common_ngram_cache before = live;
    std::vector<llama_token> rejected = {8, 9, 10};
    allocation_probe::largest = 0;
    allocation_probe::active = true;
    common_ngram_cache_update(live, 1, 1, rejected, rejected.size(), false, 0, nullptr, &limits);
    allocation_probe::active = false;
    expect(live == before && allocation_probe::largest == 0, "rejected live pairs allocate no map nodes");
    for (const uint64_t cap : {uint64_t(0), uint64_t(1), UINT64_MAX}) {
        common_ngram_cache warm = before;
        common_ngram_cache_limits exhausted(warm, 2, cap);
        expect(exhausted.continuations_left == (cap == UINT64_MAX ? UINT64_MAX : 0),
                "warm-cache credit saturates without unsigned wrap and preserves unlimited mode");
    }
    const std::pair<uint64_t, uint64_t> zero_quotas[] = {{0, UINT64_MAX}, {UINT64_MAX, 0}};
    for (const auto caps : zero_quotas) {
        common_ngram_cache empty;
        common_ngram_cache_limits zero(empty, caps.first, caps.second);
        common_ngram_cache_update(empty, 1, 1, tokens, tokens.size(), false, 0, nullptr, &zero);
        expect(empty.empty(), "either zero live quota refuses a cold cache");
    }

    // An external checkpoint fills the budget before the live delta is collected.
    common_ngram_cache saved = {{a, {{2, 20}}}, {b, {{1, 30}, {3, 40}}}};
    delta = {{a, {{2, 2}, {3, 1}}}, {b, {{1, 1}}}};
    common_ngram_cache_refresh(live, saved, delta, 0, 2, 3);
    expect(successors(live) == 3 && live.at(a).at(2) == 22 && live.at(b).at(1) == 31 && !live.at(a).count(3),
            "checkpoint collection caps successors but still merges retained counts");
    expect(delta.at(a).size() == 1 && delta.at(a).at(2) == 2 && delta.at(b).at(1) == 1,
            "collection retires only delta successors absent from the new live cache");
    limits = common_ngram_cache_limits(live, 2, 3);
    common_ngram_cache_update(live, 1, 1, tokens, tokens.size(), false, 0, nullptr, &limits);
    expect(successors(live) == 3 && live.at(a).at(2) == 24 && limits.continuations_left == 0,
            "next request uses refreshed credit and still learns retained pairs");

    common_ngram_cache add = {{a, {{2, 1}, {3, 1}, {4, 1}}}, {b, {{1, 1}}}};
    common_ngram_cache merged;
    limits = common_ngram_cache_limits(merged, 1, 2);
    common_ngram_cache_merge(merged, add, 0, &limits);
    expect(merged.size() == 1 && successors(merged) <= 2 && limits.continuations_left == 2 - successors(merged),
            "bounded merge handles both partial new parts and entry limits");
    common_ngram_cache large_part = {{a, {{2, 1}, {3, 1}, {4, 1}}}};
    merged.clear();
    limits = common_ngram_cache_limits(merged, 1, 2);
    common_ngram_cache_merge(merged, large_part, 0, &limits);
    expect(merged.size() == 1 && successors(merged) == 2 && limits.continuations_left == 0,
            "one new part is clipped before inserting excess successor nodes");
    common_ngram_cache all;
    common_ngram_cache_limits unlimited_limits;
    common_ngram_cache_merge(all, add, 0, &unlimited_limits);
    expect(all == add && unlimited_limits.continuations_left == UINT64_MAX,
            "default unlimited merge preserves every pair and count");
}

static void test_ngram_stream_save() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-stream-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create private lookup stream test directory");
    if (!created) {
        return;
    }
    const std::string path = (dir / "cache").string();
    const common_ngram_cache_vocab_id vocab = {100000, 123};
    const auto read = [&]() {
        std::ifstream file(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    std::string err;
    common_ngram_cache cache;
    for (int n : {0, 1, 8192}) {
        cache.clear();
        common_ngram_cache filtered;
        for (llama_token token = 1; token <= n; ++token) {
            const common_ngram key(&token, 1);
            cache[key] = {{0, token}, {1, 0}, {2, -1}};
            filtered[key] = {{0, token}};
        }
        const llama_token skipped = 90000;
        cache[common_ngram(&skipped, 1)] = {{0, 0}};
        const llama_token empty_key[2] = {90000, 90001};
        cache[common_ngram(empty_key, 2)] = {};
        const auto before = cache;
        allocation_probe::largest = 0;
        allocation_probe::active = true;
        const auto expected = ngram_old_file_bytes(cache, vocab);
        allocation_probe::active = false;
        if (n == 8192) {
            expect(allocation_probe::largest > 65536, "old full-payload control allocates above 64 KiB");
        }
        allocation_probe::largest = 0;
        allocation_probe::active = true;
        const bool saved = common_ngram_cache_save_file(cache, path, vocab, err);
        allocation_probe::active = false;
        expect(saved, "streamed lookup save succeeds");
        expect(allocation_probe::largest < 65536, "streamed lookup save has no allocation above 64 KiB");
        expect(cache == before && read() == expected, "streamed save preserves source and complete prior-format bytes");
        common_ngram_cache loaded;
        expect(common_ngram_cache_load_file(path, loaded, vocab, err) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == filtered,
                "streamed hash and size reload all positive counts, skipping empty or nonpositive parts");
    }
#ifdef __linux__
    const auto before = read();
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    expect(fs::create_directory(tmp), "block temporary-file creation");
    expect(!common_ngram_cache_save_file(cache, path, vocab, err) && !err.empty() && read() == before,
            "failed temporary open preserves the complete old file");
    fs::remove(tmp);
    std::error_code ec;
    fs::create_symlink("/dev/full", tmp, ec);
    expect(!ec, "create a write-failure fixture");
    if (!ec) {
        expect(!common_ngram_cache_save_file(cache, path, vocab, err) && !err.empty() && read() == before,
                "failed streamed write preserves the complete old file");
        expect(fs::symlink_status(tmp, ec).type() == fs::file_type::not_found,
                "failed write removes its temporary name");
    }
#endif
    fs::remove_all(dir);
}

static void test_ngram_load_limits() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-load-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create a private lookup load test directory");
    if (!created) {
        return;
    }
    const std::string path = (dir / "cache").string();
    const common_ngram_cache_vocab_id vocab = {100, 123};
    std::vector<llama_token> tokens = {0, 4};
    const common_ngram a(tokens.data(), 1);
    const common_ngram b(tokens.data() + 1, 1);
    const common_ngram_cache source = {{a, {{1, 7}, {2, 8}}}, {b, {{3, 9}}}};
    const auto read_file = [](const std::string & name) {
        std::ifstream file(name, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    std::string err;
    expect(common_ngram_cache_save_file(source, path, vocab, err), "write a versioned lookup load fixture");
    const std::string before = read_file(path);
    common_ngram_cache loaded;
    expect(common_ngram_cache_load_file(path, loaded, vocab, err, 2, 3) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == source,
            "exact record and continuation quotas load all counts");
    for (const auto limits : {std::pair<uint64_t, uint64_t>{0, 3}, {1, 3}, {2, 0}, {2, 2}}) {
        loaded = source;
        expect(common_ngram_cache_load_file(path, loaded, vocab, err, limits.first, limits.second) == COMMON_NGRAM_CACHE_LOAD_LIMIT,
                "one-under and zero quotas refuse nonempty versioned input");
        expect(loaded.empty() && !err.empty() && read_file(path) == before,
                "load refusal clears partial results and preserves the file");
    }
    common_ngram_cache live = source;
    live[a][1]++;
    common_ngram_cache delta = {{a, {{1, 1}}}};
    const auto live_before = live;
    const auto delta_before = delta;
    bool from_disk = true;
    expect(!common_ngram_cache_sync_file(path, live, delta, 2, vocab, err, &from_disk, 2, 2),
            "an over-quota existing file cannot be recreated by sync");
    expect(!from_disk && live == live_before && delta == delta_before && read_file(path) == before,
            "quota refusal preserves live counts, pending delta and on-disk data");
    expect(common_ngram_cache_sync_file(path, live, delta, 2, vocab, err, &from_disk, 2, 3) && from_disk,
            "same-input sync succeeds at the exact load quotas");
    expect(live == live_before && delta.empty(), "successful sync applies the retained delta exactly once");
    expect(common_ngram_cache_load_file(path, loaded, vocab, err) == COMMON_NGRAM_CACHE_LOAD_OK && loaded == live,
            "default unlimited load preserves the prior file behavior");

    const std::string legacy = (dir / "legacy").string();
    for (int records : {1, 2}) {
        {
            std::ofstream file(legacy, std::ios::binary | std::ios::trunc);
            for (int record = 0; record < records; ++record) {
                const int32_t count = 3;
                file.write(reinterpret_cast<const char *>(a.tokens), sizeof(a.tokens));
                file.write(reinterpret_cast<const char *>(&count), sizeof(count));
                for (int i = 0; i < count; ++i) {
                    const llama_token token = 1;
                    const int32_t one = 1;
                    file.write(reinterpret_cast<const char *>(&token), sizeof(token));
                    file.write(reinterpret_cast<const char *>(&one), sizeof(one));
                }
            }
            expect(bool(file), "write duplicate records and continuations in a legacy fixture");
        }
        expect(common_ngram_cache_load_file(legacy, loaded, vocab, err, records, records * 3) == COMMON_NGRAM_CACHE_LOAD_OK &&
                loaded.size() == 1 && loaded.at(a).at(1) == 3, "legacy exact quotas retain duplicate folding semantics");
        expect(common_ngram_cache_load_file(legacy, loaded, vocab, err, records, records * 3 - 1) == COMMON_NGRAM_CACHE_LOAD_LIMIT && loaded.empty(),
                "duplicate continuations consume quota before temporary-map allocation");
        expect(common_ngram_cache_load_file(legacy, loaded, vocab, err, records - 1, UINT64_MAX) == COMMON_NGRAM_CACHE_LOAD_LIMIT && loaded.empty(),
                "duplicate legacy keys cannot bypass the record quota");
    }
    expect(common_ngram_cache_save_file({}, path, vocab, err) &&
            common_ngram_cache_load_file(path, loaded, vocab, err, 0, 0) == COMMON_NGRAM_CACHE_LOAD_OK && loaded.empty(),
            "empty versioned caches need no record or continuation quota");
    fs::remove_all(dir);
}

static void test_ngram_live_limit() {
    std::vector<llama_token> tokens;
    for (llama_token token = 0; token < 4096; ++token) {
        tokens.push_back(token);
    }
    common_ngram_cache unlimited;
    common_ngram_cache limited;
    common_ngram_cache explicit_unlimited;
    common_ngram_cache_update(unlimited, 1, 4, tokens, tokens.size(), false);
    common_ngram_cache_update(explicit_unlimited, 1, 4, tokens, tokens.size(), false, 0);
    common_ngram_cache_update(limited, 1, 4, tokens, tokens.size(), false, 8);
    expect(unlimited.size() > 16000, "unlimited positive control learns the complete long sequence");
    expect(unlimited == explicit_unlimited, "zero key limit preserves the default update result");
    expect(limited.size() == 8, "live update stops at the key limit within a long batch");
    for (const auto & entry : limited) {
        expect(entry.second == unlimited.at(entry.first), "retained keys keep the same learned counts");
    }

    std::vector<llama_token> repeat = {0, 1, 0, 1};
    common_ngram zero(repeat.data(), 1);
    common_ngram one(repeat.data() + 1, 1);
    common_ngram_cache_update(limited, 1, 1, repeat, repeat.size(), false, 8);
    expect(limited.size() == 8 && limited.at(zero).at(1) == 3,
            "a full cache still learns an existing successor");
    expect(limited.at(one).at(0) == 1, "a full cache still learns a new successor for a retained key");
    const auto before = limited;
    std::vector<llama_token> more = {9000, 9001, 9002};
    common_ngram_cache_update(limited, 1, 4, more, more.size(), false, 8);
    expect(limited == before, "full-cache rejection leaves all entries unchanged");

    const auto source_before = unlimited;
    common_ngram_cache_merge(limited, unlimited, 8);
    expect(limited.size() == 8 && limited.at(zero).at(1) == 4,
            "checkpoint collection merges retained counts without new keys");
    expect(limited.at(one).at(0) == 1, "checkpoint collection preserves other retained successors");
    common_ngram_cache collected;
    common_ngram_cache_merge(collected, unlimited, 8);
    expect(collected.size() == 8, "checkpoint collection also caps an initially empty cache");
    for (const auto & entry : collected) {
        expect(entry.second == unlimited.at(entry.first), "capped merge keeps complete source counts");
    }
    common_ngram_cache_merge(collected, unlimited, 0);
    expect(collected.size() == unlimited.size(), "zero key limit keeps unlimited merge behavior");
    expect(unlimited == source_before, "bounded and unlimited merges leave the source unchanged");
}

static void test_ngram_delta_admission() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-ngram-fit-" + std::to_string(ggml_time_us()));
    const bool created = fs::create_directory(dir);
    expect(created, "create a private ngram save test directory");
    if (!created) {
        return;
    }
    const std::string path = (dir / "cache").string();
    const common_ngram_cache_vocab_id vocab = {100, 123};
    std::string err;
    std::vector<llama_token> tokens = {2, 3, 0, 1};
    common_ngram a(tokens.data() + 2, 1);
    common_ngram b(tokens.data(), 1);
    const common_ngram_cache disk = {{a, {{1, 10}}}};

    // The old independent cap learns B in the delta and loses the retained A increment at save.
    common_ngram_cache old_live = disk;
    common_ngram_cache old_delta;
    common_ngram_cache_update(old_live, 1, 1, tokens, tokens.size(), false, 1);
    common_ngram_cache_update(old_delta, 1, 1, tokens, tokens.size(), false, 1);
    expect(old_live.at(a).at(1) == 11 && old_delta.count(b) == 1 && old_delta.count(a) == 0,
            "independent admission positive control reproduces different live and delta keys");
    expect(common_ngram_cache_save_file(disk, path, vocab, err), "write independent-admission control");
    expect(common_ngram_cache_sync_file(path, old_live, old_delta, 1, vocab, err), "save independent-admission control");
    expect(old_live.at(a).at(1) == 10, "independent-admission control loses the retained increment");

    common_ngram_cache live = disk;
    common_ngram_cache delta;
    common_ngram_cache_update(live, 1, 1, tokens, tokens.size(), false, 1);
    common_ngram_cache_update(delta, 1, 1, tokens, tokens.size(), false, 1, &live);
    expect(live.size() == 1 && delta.size() == 1 && delta.at(a).at(1) == 1,
            "the bounded delta records only keys admitted by the live cache");
    expect(common_ngram_cache_save_file(disk, path, vocab, err), "write coordinated-admission base");
    expect(common_ngram_cache_sync_file(path, live, delta, 1, vocab, err), "save coordinated delta");
    common_ngram_cache restored;
    expect(common_ngram_cache_load_file(path, restored, vocab, err) == COMMON_NGRAM_CACHE_LOAD_OK,
            "restore coordinated save");
    expect(live == restored && live.at(a).at(1) == 11 && delta.empty(),
            "save and reload keep the retained key increment exactly once");

    std::vector<llama_token> next = {4, 5};
    common_ngram c(next.data(), 1);
    common_ngram_cache saved = {{c, {{5, 100}}}};
    delta = {{a, {{1, 1}}}};
    common_ngram_cache_refresh(live, saved, delta, 1);
    expect(live.size() == 1 && live.count(c) == 1 && delta.empty() && saved.count(a) == 1,
            "checkpoint collection drops retired delta keys and hands the old live cache back");
    const auto checkpoint_disk = live;
    common_ngram_cache_update(live, 1, 1, next, next.size(), false, 1);
    common_ngram_cache_update(delta, 1, 1, next, next.size(), false, 1, &live);
    expect(delta.size() == 1 && delta.at(c).at(5) == 1, "retired keys cannot fill the next checkpoint delta");
    expect(common_ngram_cache_save_file(checkpoint_disk, path, vocab, err), "write changed checkpoint base");
    expect(common_ngram_cache_sync_file(path, live, delta, 1, vocab, err), "save after a different-key checkpoint");
    expect(live.size() == 1 && live.at(c).at(5) == 101 && delta.empty(),
            "the next saved checkpoint also keeps its retained increment");

    common_ngram_cache unlimited = disk;
    common_ngram_cache unlimited_delta;
    common_ngram_cache_update(unlimited, 1, 1, tokens, tokens.size(), false);
    common_ngram_cache_update(unlimited_delta, 1, 1, tokens, tokens.size(), false, 0, &unlimited);
    common_ngram_cache expected = disk;
    common_ngram_cache_update(expected, 1, 1, tokens, tokens.size(), false);
    common_ngram_cache refreshed = disk;
    common_ngram_cache_refresh(unlimited, refreshed, unlimited_delta, 0);
    expect(unlimited == expected && unlimited_delta.size() == 3,
            "unlimited checkpoint collection keeps all learned keys and counts");

    common_ngram_cache failed_delta = {{a, {{1, 2}}}};
    common_ngram_cache since_handover = {{c, {{5, 3}}}};
    common_ngram_cache_merge(failed_delta, since_handover, 2);
    expect(failed_delta.size() == 2 && failed_delta.at(a).at(1) == 2 && failed_delta.at(c).at(5) == 3,
            "failed checkpoint collection keeps disjoint admitted delta keys");
    fs::remove_all(dir);
}

static void test_ram_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    constexpr uint64_t MiB = 1ull << 20;
    const char * stage_names[] = {"LLAMA_B1_THREADS", "LLAMA_B1_PIECE_MIB", "LLAMA_B1_STAGE_MIB", "LLAMA_B1_DIRECT_PIECE_MIB"};
    std::string saved_stage[4];
    bool had_stage[4];
    auto set_stage = [](const char * name, const char * value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
    };
    for (size_t i = 0; i < 4; ++i) {
        const char * value = std::getenv(stage_names[i]);
        had_stage[i] = value != nullptr;
        if (value) { saved_stage[i] = value; }
        set_stage(stage_names[i], nullptr);
    }
    expect(!llama_b1::staging_explicit(), "absent staging settings retain automatic fitting");
    for (const char * name : stage_names) {
        set_stage(name, "");
        expect(!llama_b1::staging_explicit(), "empty staging settings retain automatic fitting");
        set_stage(name, nullptr);
    }
    for (const char * value : {"0", "-1", "bad", "", " 0"}) {
        set_stage("LLAMA_B1_STAGE_MIB", value);
        expect(!llama_b1::staging_explicit(), "ineffective stage sizes cannot bypass the RAM cap");
        expect(llama_b1::fit_staging_piece(5 * GiB + 128 * MiB, 8, 16 * MiB, 4096, 4096,
                true, llama_b1::staging_explicit()) == 0, "ineffective override cannot allocate staging below the floor");
    }
    set_stage("LLAMA_B1_STAGE_MIB", nullptr);
    for (const char * name : {stage_names[0], stage_names[1], stage_names[2]}) {
        set_stage(name, "4");
        expect(llama_b1::staging_explicit(), "effective explicit sizes retain precedence");
        set_stage(name, nullptr);
    }
    set_stage("LLAMA_B1_DIRECT_PIECE_MIB", "64");
    expect(!llama_b1::staging_explicit(), "direct GPU upload chunks do not override host staging sizes");
    for (size_t i = 0; i < 4; ++i) { set_stage(stage_names[i], had_stage[i] ? saved_stage[i].c_str() : nullptr); }
    const uint64_t stage_required = 24 * (16 * MiB + 16384) + 4096;
    expect(!llama_b1::staging_floor_unmet(5 * GiB + stage_required, 8, 16 * MiB, 4096, 4096),
            "explicit staging may exactly meet the RAM floor");
    expect(llama_b1::staging_floor_unmet(5 * GiB + stage_required - 1, 8, 16 * MiB, 4096, 4096) &&
            llama_b1::staging_floor_unmet(0, 8, 16 * MiB, 4096, 4096), "unsafe or unknown explicit staging needs a warning");
    expect(llama_b1::staging_floor_unmet(UINT64_MAX, 64, SIZE_MAX, 4096, 4096) &&
            llama_b1::staging_floor_unmet(UINT64_MAX, 64, MiB, SIZE_MAX, 4096),
            "explicit staging warning arithmetic cannot overflow");
    std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> requests;
    for (auto & request : requests) {
        request.bytes = UINT64_MAX;
    }
    uint64_t previous_prompt = 0;
    for (const uint64_t host_gib : {16, 32, 64, 128}) {
        // Fake a mapped model and context that already use six GiB.
        const uint64_t available = (host_gib - 6) * GiB;
        const auto plan = common_fit_ram_plan(available, requests);
        const size_t expert_budget = llama_moe_arena::fit_memory_budget(available);
        const uint64_t expected_expert_mib = (((host_gib - 11) * 1024 * 4 / 5) / 2);
        expect(expert_budget == expected_expert_mib * MiB && expert_budget == plan.bytes[COMMON_FIT_RAM_EXPERTS],
                "post-map expert arena quota agrees with the common RAM-fit plan on fake hosts");
        expect(llama_moe_arena::fits_ram_budget(expert_budget, available, true, false), "exact expert quota can pin");
        expect(!llama_moe_arena::fits_ram_budget(expert_budget + 1, available, true, false),
                "one byte beyond expert quota selects the existing unpinned fallback");
        expect(llama_moe_arena::fits_ram_budget(expert_budget + 1, available, true, true),
                "explicit arena selection wins over the automatic share");
        expect(llama_moe_arena::fits_ram_budget(expert_budget + 1, available, false, false),
                "fit off preserves legacy arena admission");
        const size_t stage_piece = llama_b1::fit_staging_piece(available, 8, 16 * MiB, 4096, 4096, true, false);
        expect(stage_piece > 0 && stage_piece <= 16 * MiB, "fake hosts retain bounded prefill staging");
        expect(24 * (stage_piece + 3 * 4096 + 4096) + 4096 <= plan.bytes[COMMON_FIT_RAM_STAGING],
                "staging share covers two pinned slots, disk bounce and padding per worker");
        uint64_t sum = 0;
        for (uint64_t bytes : plan.bytes) {
            expect(bytes <= available - sum, "RAM plan cannot overflow available memory");
            sum += bytes;
        }
        expect(plan.floor_met && plan.floor == 5 * GiB, "all host sizes retain the five GiB floor");
        expect(sum <= available - 5 * GiB, "optional consumers leave five GiB free");
        const uint64_t expected_budget = ((available / MiB - 5120) * 4 / 5) * MiB;
        expect(plan.automatic_budget == expected_budget, "reported budget keeps independently calculated twenty percent reserve");
        expect(sum <= expected_budget, "automatic allocations keep independently calculated twenty percent reserve");
        expect(plan.bytes[COMMON_FIT_RAM_PROMPT] > previous_prompt, "cache defaults grow with available host RAM");
        expect(plan.bytes[COMMON_FIT_RAM_EXPERTS] > plan.bytes[COMMON_FIT_RAM_PROMPT], "expert storage gets priority");
        previous_prompt = plan.bytes[COMMON_FIT_RAM_PROMPT];
        std::printf("RAM fit fake %llu GiB host: available=%llu MiB optional=%llu MiB floor=5120 MiB\n",
                (unsigned long long) host_gib, (unsigned long long) (available / MiB), (unsigned long long) (sum / MiB));
    }
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB}) {
        expect(llama_moe_arena::fit_memory_budget(available) == 0 &&
                !llama_moe_arena::fits_ram_budget(MiB, available, true, false),
                "unknown or low RAM disables automatic pinned expert storage");
    }
    expect(llama_moe_arena::fit_memory_budget(UINT64_MAX) <= UINT64_MAX - 5 * GiB,
            "extreme host RAM cannot overflow the expert budget");
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB}) {
        expect(llama_b1::fit_staging_piece(available, 8, 16 * MiB, 4096, 4096, true, false) == 0,
                "unknown or low RAM disables automatic staging before host allocation");
        expect(llama_b1::fit_staging_piece(available, 8, 16 * MiB, 4096, 4096, false, false) == 16 * MiB &&
                llama_b1::fit_staging_piece(available, 8, 16 * MiB, 4096, 4096, true, true) == 16 * MiB,
                "fit off and explicit staging retain legacy sizes");
    }
    const uint64_t tight_ram = 5 * GiB + GiB;
    const size_t tight_piece = llama_b1::fit_staging_piece(tight_ram, 8, 16 * MiB, 4096, 4096, true, false);
    expect(tight_piece >= MiB && tight_piece < 16 * MiB, "tight RAM shrinks automatic staging");
    std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> staging_requests = {};
    for (auto & request : staging_requests) { request.bytes = UINT64_MAX; }
    const size_t stage_quota = common_fit_ram_plan(tight_ram, staging_requests).bytes[COMMON_FIT_RAM_STAGING];
    expect(24 * (tight_piece + 16384) + 4096 <= stage_quota, "reduced stage and disk bounce fit their quota");
    expect(llama_b1::fit_staging_piece(tight_ram, 8, 16 * MiB, 4096, stage_quota, true, false) == 0,
            "zero buffer exhausting the quota disables staging");
    expect(llama_b1::fit_staging_piece(UINT64_MAX, 64, 16 * MiB, 4096, 4096, true, false) == 16 * MiB,
            "extreme RAM does not overflow staging quota");
    expect(llama_b1::fit_staging_piece(16 * GiB, 8, 16 * MiB, SIZE_MAX, 4096, true, false) == 0 &&
            llama_b1::fit_staging_piece(16 * GiB, 0, 16 * MiB, 4096, 4096, true, false) == 0,
            "invalid alignment and worker counts cannot overflow staging arithmetic");
    requests[COMMON_FIT_RAM_PROMPT] = {123 * MiB, true};
    requests[COMMON_FIT_RAM_EXPERTS].bytes = 0;
    const auto explicit_plan = common_fit_ram_plan(16 * GiB, requests);
    const uint64_t explicit_budget = ((16 * 1024 - 5120 - 123) * 4ull / 5) * MiB;
    uint64_t explicit_auto = 0;
    for (size_t i = 0; i < explicit_plan.bytes.size(); ++i) {
        if (i != COMMON_FIT_RAM_PROMPT) { explicit_auto += explicit_plan.bytes[i]; }
    }
    expect(explicit_plan.automatic_budget == explicit_budget && explicit_auto <= explicit_budget,
            "explicit bytes are debited before independently calculated eighty percent budget");
    expect(explicit_plan.bytes[COMMON_FIT_RAM_PROMPT] == 123 * MiB, "explicit prompt limit wins exactly");
    expect(explicit_plan.bytes[COMMON_FIT_RAM_EXPERTS] == 0, "unused expert bank stays off");
    expect(explicit_plan.bytes[COMMON_FIT_RAM_STAGING] > explicit_plan.bytes[COMMON_FIT_RAM_PREFETCH], "unused first-priority quota goes to staging");
    requests[COMMON_FIT_RAM_PROMPT] = {0, true};
    expect(common_fit_ram_plan(16 * GiB, requests).bytes[COMMON_FIT_RAM_PROMPT] == 0, "explicit disable wins");
    for (const uint64_t available : std::array<uint64_t, 3>{0, 4 * GiB, 5 * GiB}) {
        const auto plan = common_fit_ram_plan(available, requests);
        expect(plan.automatic_budget == 0, "no automatic allocations at or below the floor");
        for (uint64_t bytes : plan.bytes) {
            expect(bytes == 0, "optional pieces turn off when there is no safe RAM");
        }
    }
    requests[COMMON_FIT_RAM_PROMPT] = {12 * GiB, true};
    const auto too_large = common_fit_ram_plan(16 * GiB, requests);
    expect(!too_large.floor_met && too_large.automatic_budget == 0, "unsafe explicit demand cannot claim a safe fit");
    expect(too_large.bytes[COMMON_FIT_RAM_PROMPT] == 12 * GiB, "unsafe explicit demand is reported unchanged");
    const auto odd = common_fit_ram_plan(5 * GiB + 11 * MiB + 999, {});
    expect(odd.automatic_budget == 8 * MiB, "fractional MiB and fifths round down before budgeting");
    const auto large = common_fit_ram_plan(UINT64_MAX, requests);
    uint64_t left = UINT64_MAX - large.floor;
    for (uint64_t bytes : large.bytes) {
        expect(bytes <= left, "large-memory arithmetic does not wrap");
        left -= bytes;
    }
    expect(common_fit_vram_floor(false) == 300 * MiB, "headless VRAM floor is 300 MiB");
    expect(common_fit_vram_floor(true) == 800 * MiB, "display VRAM floor is 800 MiB");
}

static void test_ram_eval_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    constexpr uint64_t MiB = 1ull << 20;
    for (uint64_t host_gib : {16, 32, 64, 128}) {
        const uint64_t available = (host_gib - 6) * GiB;
        const uint64_t budget_mib = ((available - 5 * GiB) / MiB) * 4 / 5;
        const uint64_t actual = common_fit_eval_cache_ram_bytes(available, true);
        expect(actual == budget_mib * 5 / 100 * MiB, "eval read keeps its share without borrowing future cache growth");
        expect(actual <= available - 5 * GiB, "eval read preserves the RAM floor");
    }
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB, 5 * GiB + MiB - 1}) {
        expect(common_fit_eval_cache_ram_bytes(available, true) == 0, "low or unknown RAM disables eval record reads");
        expect(common_fit_eval_cache_ram_bytes(available, false) == UINT64_MAX, "fit off keeps prior eval read limits");
    }
    expect(common_fit_eval_cache_ram_bytes(UINT64_MAX, true) < UINT64_MAX / 20,
            "extreme available RAM cannot overflow eval quota");

    for (uint64_t host_gib : {16, 32, 64, 128}) {
        const uint64_t available = (host_gib - 6) * GiB;
        const uint64_t budget = common_fit_eval_cache_ram_bytes(available, true);
        expect(common_fit_eval_cache_load_bytes(available, true, 100000) == budget - 100000 * sizeof(llama_token),
                "eval restore reserves a token buffer beside the parsed record");
        const uint64_t max_tokens = budget / sizeof(llama_token);
        expect(common_fit_eval_cache_load_bytes(available, true, max_tokens) == 0 &&
                common_fit_eval_cache_load_bytes(available, true, max_tokens + 1) == 0,
                "a full or oversized token buffer leaves no room for an eval record");
        expect(common_fit_eval_cache_load_bytes(available, true, max_tokens - 1) == sizeof(llama_token),
                "restore token reservation keeps the exact boundary");
        expect(common_fit_eval_cache_load_bytes(available, true, UINT64_MAX) == 0,
                "restore token count cannot overflow byte admission");
        expect(common_fit_eval_cache_load_bytes(available, true, 0) == budget,
                "an empty request reserves no token storage");
    }
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB, 5 * GiB + MiB - 1}) {
        expect(common_fit_eval_cache_load_bytes(available, true, 0) == 0 &&
                common_fit_eval_cache_load_bytes(available, true, 1) == 0,
                "low or unknown RAM refuses restore before request extraction");
        expect(common_fit_eval_cache_load_bytes(available, false, UINT64_MAX) == UINT64_MAX,
                "fit off keeps prior restore admission without token arithmetic");
    }
    expect(common_fit_eval_cache_load_bytes(UINT64_MAX, true, UINT64_MAX) == 0,
            "extreme RAM and token counts cannot wrap restore admission");

    for (uint64_t host_gib : {16, 32, 64, 128}) {
        const uint64_t available = (host_gib - 6) * GiB;
        const uint64_t budget = common_fit_eval_cache_ram_bytes(available, true);
        const uint64_t n_tokens = 100000;
        const uint64_t spec_bytes = budget - 65536 - n_tokens * sizeof(llama_token);
        expect(common_fit_eval_cache_save_fits(available, true, n_tokens, spec_bytes),
                "token and speculative copies at the live eval quota fit");
        expect(!common_fit_eval_cache_save_fits(available, true, n_tokens, spec_bytes + 1),
                "one byte over the live save quota is refused before either copy");
        expect(!common_fit_eval_cache_save_fits(available, true, n_tokens + 1, spec_bytes),
                "the token copy is charged along with the speculative state");
        expect(common_fit_eval_cache_save_fits(available, true, 1024, 0) &&
                !common_fit_eval_cache_save_fits(available, true, budget / sizeof(llama_token) + 1, 0),
                "a saved prefix can fit while the full slot token copy cannot");
        expect(!common_fit_eval_cache_save_fits(available, true, UINT64_MAX, 0) &&
                !common_fit_eval_cache_save_fits(available, true, 0, UINT64_MAX),
                "hostile copy sizes cannot wrap save admission arithmetic");
    }
    for (uint64_t available : {uint64_t(0), 4 * GiB, 5 * GiB, 5 * GiB + MiB - 1}) {
        expect(!common_fit_eval_cache_save_fits(available, true, 0, 0),
                "low or unknown current RAM disables eval saves before extraction");
        expect(common_fit_eval_cache_save_fits(available, false, 100000, 8 * MiB),
                "fit off preserves valid eval save sizes without a RAM probe");
    }
}

static void test_ram_prompt_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    std::string why;
    for (uint64_t host_gib : {16, 32, 64, 128}) {
        const uint64_t available = (host_gib - 6) * GiB;
        const int32_t actual = common_fit_prompt_cache_ram_mib(host_gib * GiB, available, -2, true, why);
        const uint64_t budget_mib = (available / (1ull << 20) - 5120) * 4 / 5;
        const int32_t tier = host_gib == 16 ? 4096 : host_gib == 32 ? 8192 : host_gib == 64 ? 12288 : 20480;
        // Only prompt cache is pending here: earlier allocated bank/staging are
        // already in MemAvailable, and their unused quotas flow down to prompt.
        const uint64_t prompt_quota = budget_mib * 50 / 100 + budget_mib * 10 / 100 +
                                      budget_mib * 10 / 100 + budget_mib * 20 / 100;
        expect(actual == (int32_t) std::min<uint64_t>(tier, prompt_quota), "startup prompt size follows shared budget after model/context use");
        expect(available - (uint64_t(actual) << 20) >= 5 * GiB, "startup prompt preserves five GiB floor");
    }
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 5 * GiB, -2, true, why) == 0,
            "startup disables automatic prompt cache at RAM floor");
    expect(common_fit_prompt_cache_ram_mib(0, 0, -2, true, why) == 0, "unknown available RAM disables automatic prompt cache under fit");
    expect(why.find("5 GiB") != std::string::npos, "low-RAM log explains prompt cache reduction and floor");
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 4 * GiB, 123, true, why) == 123,
            "explicit prompt limit wins even when RAM floor cannot be met");
    expect(why.find("cannot") != std::string::npos, "unsafe explicit prompt limit reports infeasible floor");
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 8 * GiB, 0, true, why) == 0, "explicit cache disable wins");
    for (bool fit : {false, true}) {
        // Linux CPU backend reports total RAM here, unlike MemAvailable.
        expect(common_fit_prompt_cache_ram_mib(16 * GiB, 8 * GiB, -1, fit, why, 16 * GiB) == 8192,
                "explicit half-free limit uses the same CPU query as the release baseline");
        expect(common_fit_prompt_cache_ram_mib(16 * GiB, 8 * GiB, -1, fit, why, 6 * GiB) == 3072,
                "explicit half-free limit follows backend free RAM when it differs from total RAM");
        expect(common_fit_prompt_cache_ram_mib(16 * GiB, 8 * GiB, -1, fit, why, 0) == 8192,
                "missing CPU query keeps the baseline eight GiB explicit fallback");
    }
    expect(common_fit_prompt_cache_ram_mib(64 * GiB, 0, -2, true, why) == 0,
            "known total with unknown available RAM disables automatic cache under fit");
    expect(common_fit_prompt_cache_ram_mib(64 * GiB, 0, -2, false, why) == 12288,
            "unknown available RAM with fit off keeps the baseline total-memory tier");
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 6 * GiB, -2, false, why) == 3072,
            "fit off keeps prior tier and half-available clamp");
}

static void test_vram_fit_floor() {
    constexpr size_t MiB = 1024 * 1024;
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("test-vram-fit-" + std::to_string(ggml_time_us()));
    fs::create_directories(dir);
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "empty DRM inventory cannot prove headless operation");
    fs::create_directories(dir / "card0");
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "card without connector status keeps the unknown display floor");
    fs::create_directories(dir / "card0-DP-1");
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "connector without status cannot prove headless operation");
    const auto status = dir / "card0-DP-1" / "status";
    std::ofstream(status) << "disconnected\n";
    expect(common_fit_vram_floor_detect(dir.string(), false) == 300 * MiB, "disconnected display uses headless floor");
    fs::create_directories(dir / "card1-HDMI-A-1");
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "disconnected plus missing connector status keeps the unknown display floor");
    std::ofstream(dir / "card1-HDMI-A-1" / "status") << "disconnected\n";
    expect(common_fit_vram_floor_detect(dir.string(), false) == 300 * MiB, "all known disconnected connectors allow the headless floor");
    fs::remove(status);
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "missing status on the other connector also keeps the unknown floor");
    std::ofstream(status) << "disconnected\n";
    expect(common_fit_vram_floor_detect(dir.string(), true) == 800 * MiB, "display session keeps larger floor even without a connector");
    fs::create_directories(dir / "card1-HDMI-A-1");
    std::ofstream(dir / "card1-HDMI-A-1" / "status") << "connected\n";
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "any connected display keeps larger floor for multi-device fit");
    fs::remove_all(dir / "card1-HDMI-A-1");
    for (const std::string text : {"unknown\n", "bad\n", ""}) {
        std::ofstream(status) << text;
        expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "unknown or unreadable connector state keeps larger floor");
    }
    fs::remove_all(dir);
    expect(common_fit_vram_floor_detect(dir.string(), false) == 800 * MiB, "missing DRM inventory keeps larger floor");

    for (size_t floor : {300 * MiB, 800 * MiB}) {
        for (size_t headroom : {size_t(0), floor - 1, floor, floor + 1}) {
            const auto status = common_fit_vram_margin_status(1024 * MiB, 1024 * MiB - headroom, floor);
            const bool fits = headroom >= floor;
            expect((status == COMMON_PARAMS_FIT_STATUS_SUCCESS) == fits, "zero-layer buffers must leave the requested margin before fit succeeds");
            expect(common_fit_verify_compact_safe(status, {floor}, floor) == fits, "compact verify must reject actual positive headroom below its requested floor");
        }
        common_params params;
        const auto automatic = common_fit_vram_targets(params, floor);
        expect(std::all_of(automatic.begin(), automatic.end(),
            [floor](size_t margin) { return margin == floor; }), "automatic fit uses detected floor on every device");
        expect(params.fit_params_target[0] == 1024 * MiB, "automatic resolution does not overwrite the request reused by a later model load");
        params.fit_params_target = {0, 128 * MiB, 1024 * MiB};
        params.fit_params_target_explicit = true;
        const auto explicit_targets = params.fit_params_target;
        expect(common_fit_vram_targets(params, floor) == explicit_targets, "explicit zero, small and stock-size margins all win");
        params.fit_params_target_explicit = false;
        expect(common_fit_vram_targets(params, floor) == std::vector<size_t>({0, 128 * MiB, floor}), "programmatic nondefault margins stay unchanged");
        params.fit_params_target = {1024 * MiB};
        params.fit_params = false;
        expect(common_fit_vram_targets(params, floor)[0] == 1024 * MiB, "fit off keeps prior margins");
        params.n_cpu_moe_auto = true;
        expect(common_fit_vram_targets(params, floor)[0] == floor, "explicit automatic expert placement still uses memory fit");
        expect(common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_SUCCESS, {floor}, floor), "compact verify accepts fitted floor");
        expect(!common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_SUCCESS, {floor, floor - 1}, floor), "compact verify rejects one device below its floor");
        expect(!common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_FAILURE, {floor}, floor), "failed fit cannot enable compact verify");
        expect(!common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_SUCCESS, {}, floor), "empty fit cannot enable compact verify");
    }
    expect(common_fit_vram_margin_status(1024 * MiB, 1024 * MiB + 1, 0) == COMMON_PARAMS_FIT_STATUS_NO_ROOM, "buffers exceeding free bytes remain a hard no-room result");
    expect(common_fit_vram_margin_status(-1, 0, 0) == COMMON_PARAMS_FIT_STATUS_NO_ROOM, "negative free bytes cannot fit");
    expect(common_fit_vram_margin_status(INT64_MAX, SIZE_MAX, 0) == COMMON_PARAMS_FIT_STATUS_NO_ROOM, "large buffer sizes cannot wrap the headroom subtraction");
    expect(common_fit_vram_margin_status(INT64_MAX, 0, SIZE_MAX) == COMMON_PARAMS_FIT_STATUS_FAILURE, "large requested margins cannot wrap into a successful fit");
    expect(common_fit_vram_margin_status(1024 * MiB, 1024 * MiB, 0) == COMMON_PARAMS_FIT_STATUS_SUCCESS, "explicit zero margin accepts the exact allocation boundary");
    expect(!common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_SUCCESS, {300 * MiB}, 800 * MiB), "display floor disables compact verify with an explicit headless margin");
    expect(!common_fit_verify_compact_safe(COMMON_PARAMS_FIT_STATUS_SUCCESS, {1}, 0), "compact verify retains its minimum safety floor");

    const std::string old_target = common_get_env("LLAMA_ARG_FIT_TARGET");
    common_set_env("LLAMA_ARG_FIT_TARGET", "");
    char program[] = "test-fit-extra";
    char option[] = "--fit-target";
    char single[] = "1024";
    char multi[] = "0,512";
    for (char * value : {single, multi}) {
        common_params params;
        char * argv[] = {program, option, value};
        const bool parsed = common_params_parse(3, argv, params, LLAMA_EXAMPLE_SERVER);
        expect(parsed && params.fit_params_target_explicit, "fit-target parser marks explicit scalar and vector margins");
        const auto targets = params.fit_params_target;
        expect(common_fit_vram_targets(params, 300 * MiB) == targets, "parsed explicit target is preserved by automatic fit");
        expect(params.fit_params_target[0] == (value == single ? 1024 * MiB : 0), "parsed explicit margin retains its requested bytes");
    }
    const uint64_t max_mib = std::min<uint64_t>(SIZE_MAX, INT64_MAX) / MiB;
    for (const std::string & value : {std::to_string(max_mib), "0," + std::to_string(max_mib)}) {
        common_params params;
        std::vector<char> input(value.begin(), value.end());
        input.push_back('\0');
        char * argv[] = {program, option, input.data()};
        expect(common_params_parse(3, argv, params, LLAMA_EXAMPLE_SERVER), "largest supported scalar or vector target parses without wrapping");
        expect(params.fit_params_target[value[0] == '0' ? 1 : 0] == max_mib * MiB, "largest fit target retains its byte value");
    }
    for (const std::string & value : {std::to_string(max_mib + 1), std::string("17592186044416"),
            std::string("0,17592186044416"), std::string("18446744073709551616"), std::string("-1"), std::string("12x")}) {
        common_params params;
        const auto before = params.fit_params_target;
        std::vector<char> input(value.begin(), value.end());
        input.push_back('\0');
        char * argv[] = {program, option, input.data()};
        expect(!common_params_parse(3, argv, params, LLAMA_EXAMPLE_SERVER), "parser rejects oversized or malformed fit targets before multiplication");
        expect(params.fit_params_target == before && !params.fit_params_target_explicit, "invalid vector targets do not partially change the fit request");
    }
    expect(common_fit_vram_margin_surplus(4096 * MiB, {300 * MiB, 800 * MiB}) == 2996 * MiB, "aggregate fit surplus subtracts every device margin");
    expect(common_fit_vram_margin_surplus(-1, {INT64_MAX - 1}) == -INT64_MAX, "largest negatable deficit keeps its exact value");
    expect(common_fit_vram_margin_surplus(INT64_MAX, {INT64_MAX}) == 0, "exact largest aggregate boundary is valid");
    for (const auto & margins : std::vector<std::vector<int64_t>>{{INT64_MAX, INT64_MAX}, {INT64_MAX, 1}, {-1}}) {
        bool rejected = false;
        try {
            common_fit_vram_margin_surplus(-1, margins);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        expect(rejected, "production aggregate accounting rejects overflow and negative margins");
    }
    if (SIZE_MAX > UINT32_MAX) {
        common_params params;
        char large[] = "8796093022207,8796093022207";
        char * argv[] = {program, option, large};
        expect(common_params_parse(3, argv, params, LLAMA_EXAMPLE_SERVER), "large per-device margins remain individually representable");
        bool rejected = false;
        try {
            common_fit_vram_margin_surplus(24ll << 30,
                {int64_t(params.fit_params_target[0]), int64_t(params.fit_params_target[1])});
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        expect(rejected, "parsed two-device targets cannot overflow the production aggregate subtraction");
    }
    common_set_env("LLAMA_ARG_FIT_TARGET", "1024");
    common_params from_env;
    char * argv[] = {program};
    expect(common_params_parse(1, argv, from_env, LLAMA_EXAMPLE_SERVER) && from_env.fit_params_target_explicit, "environment fit target is an explicit override");
    expect(common_fit_vram_targets(from_env, 300 * MiB)[0] == 1024 * MiB, "environment target preserves stock-size request");
    common_set_env("LLAMA_ARG_FIT_TARGET", "17592186044416");
    common_params bad_env;
    expect(!common_params_parse(1, argv, bad_env, LLAMA_EXAMPLE_SERVER), "environment target uses the same checked conversion as the CLI");
    common_set_env("LLAMA_ARG_FIT_TARGET", old_target);
}

static void test_vram_layer_step() {
    constexpr int64_t GiB = 1ll << 30;
    const auto normal = common_fit_vram_layer_step(49, GiB, 16 * GiB, 7 * GiB, 23 * GiB);
    expect(normal.status == COMMON_PARAMS_FIT_STATUS_SUCCESS && normal.step == 19,
            "layer interpolation keeps a normal measured placement guess");
    expect(common_fit_vram_layer_step(49, GiB, 16 * GiB, GiB, 23 * GiB).step == 1,
            "an exact zero-layer target still advances by one layer");
    expect(common_fit_vram_layer_step(UINT32_MAX, 0, INT64_MAX, INT64_MAX - 1, INT64_MAX).step == UINT32_MAX - 1,
            "large valid interpolation bounds cannot overflow the layer product");
    expect(common_fit_vram_layer_step(49, INT64_MAX - 2, INT64_MAX, INT64_MAX - 1, INT64_MAX).step == 24,
            "nearby large memory bounds retain their difference before floating conversion");
    for (uint32_t delta : {uint32_t(0), uint32_t(1), uint32_t(49)}) {
        const auto rejected = common_fit_vram_layer_step(delta, GiB, 16 * GiB,
                23 * GiB - int64_t(8796093022207ull << 20), 23 * GiB);
        expect(rejected.status == COMMON_PARAMS_FIT_STATUS_FAILURE && rejected.step == 0,
                "reported large margin is a recoverable failure before interpolation");
        const auto no_room = common_fit_vram_layer_step(delta, 2 * GiB, 4 * GiB, GiB, GiB);
        expect(no_room.status == COMMON_PARAMS_FIT_STATUS_NO_ROOM && no_room.step == 0,
                "reported zero-layer overflow remains hard NO_ROOM before interpolation");
    }
    for (const auto & bounds : std::vector<std::vector<int64_t>>{{-1, 16 * GiB, 7 * GiB},
            {8 * GiB, 16 * GiB, 7 * GiB}, {GiB, GiB, GiB}, {GiB, 16 * GiB, 16 * GiB}}) {
        const auto rejected = common_fit_vram_layer_step(49, bounds[0], bounds[1], bounds[2], 23 * GiB);
        expect(rejected.status == COMMON_PARAMS_FIT_STATUS_FAILURE && rejected.step == 0,
                "invalid measured interpolation bounds fail before arithmetic");
    }
}

int main() {
    test_vram_layer_step();
    test_vram_fit_floor();
    test_ngram_ram_load_fit();
    test_ngram_context_fit();
    test_ngram_checkpoint_fit();
    test_ngram_context_admission();
    test_ngram_checkpoint_load_quotas();
    test_ngram_checkpoint_merge_allocations();
    test_ngram_worker_fallback();
    test_ngram_live_successor_budget();
    test_ngram_stream_save();
    test_ngram_load_limits();
    test_ngram_live_limit();
    test_ngram_delta_admission();
    test_ram_fit();
    test_ram_prompt_fit();
    test_ram_eval_fit();
    constexpr size_t MiB = 1024*1024;

    // Qwen3.5 27B MTP draft with f16 KV, 4 sequences, unified KV: one nextn layer of 4 KV heads x 256 dims,
    // 4 KiB per token; measured at the minimum context of 4096 tokens with a 126 MiB compute buffer
    const common_fit_extra_memory mtp = {0, 16*MiB, 126*MiB};

    {
        // the fit's probe context n_ctx_train * n_seq_max, next to a main compute buffer of 1187 MiB
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 1048576, 1187*MiB, true);
        expect(at.model == 0, "shared weights should stay uncounted");
        expect(at.context == 4096*MiB, "the draft KV cache should scale linearly to 1M tokens (4 KiB per token)");
        expect(at.compute == 0, "a compute buffer that fits in the main context's buffer should add nothing");
    }

    {
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 53248, 300*MiB, true);
        expect(at.context == 208*MiB, "the draft KV cache at 53248 tokens should be 208 MiB");
        expect(at.compute == 0, "the draft compute should ride on the main context's buffer");
    }

    {
        // a draft graph larger than the main one grows the shared buffer to the larger graph (ggml_gallocr donor rule)
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 4096, 100*MiB, true);
        expect(at.context == 16*MiB, "the measured context should be kept unscaled");
        expect(at.compute == 26*MiB, "a compute buffer larger than the donor's should add only the growth of the shared buffer");
    }

    {
        // SJ-KVaRN 3/3 target at 110592 tokens: main graph 460.03 MiB, MTP draft graph 632.03 MiB, both measured at the
        // probed context -> the shared buffer grows by 172 MiB
        const common_fit_extra_memory sj_kvarn_mtp = {0, 165*MiB + MiB/2, 632*MiB + 32*1024};
        const common_fit_extra_memory at = common_fit_extra_memory_at(sj_kvarn_mtp, 110592, 110592, 460*MiB + 32*1024, true);
        expect(at.context == 165*MiB + MiB/2, "a measurement at the probed context should be used as is");
        expect(at.compute == 172*MiB, "the shared buffer should grow by the difference of the two graphs");
    }

    {
        // equal sizes still share: the donor's buffer is adopted when it is large enough
        const common_fit_extra_memory at = common_fit_extra_memory_at(mtp, 4096, 8192, 126*MiB, true);
        expect(at.compute == 0, "a compute buffer equal to the donor's should be shared");
        expect(at.context == 32*MiB, "doubling the context should double the draft KV cache");
    }

    {
        // separate draft models (or LLAMA_SHARED_COMPUTE=0) keep their own buffers and are measured at each context
        const common_fit_extra_memory draft = {512*MiB, 64*MiB, 200*MiB};
        const common_fit_extra_memory at = common_fit_extra_memory_at(draft, 16384, 16384, 1000*MiB, false);
        expect(at.model == 512*MiB, "a separate draft model should keep its weights");
        expect(at.context == 64*MiB, "a measurement at the same context should be used as is");
        expect(at.compute == 200*MiB, "an unshared compute buffer should always be counted");
    }

    {
        // scaling rounds up so the fit never under-reserves
        const common_fit_extra_memory odd = {0, 1001, 0};
        const common_fit_extra_memory at = common_fit_extra_memory_at(odd, 3, 4, 0, true);
        expect(at.context == 1335, "a scaled context should be rounded up");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d fit extra model tests failed\n", failures);
        return 1;
    }
    std::printf("fit extra model tests passed\n");
    return 0;
}
