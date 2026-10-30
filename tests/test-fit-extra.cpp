#include "fit.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static void test_ram_fit() {
    constexpr uint64_t GiB = 1ull << 30;
    constexpr uint64_t MiB = 1ull << 20;
    std::array<common_fit_ram_request, COMMON_FIT_RAM_COUNT> requests;
    for (auto & request : requests) {
        request.bytes = UINT64_MAX;
    }
    uint64_t previous_prompt = 0;
    for (const uint64_t host_gib : {16, 32, 64, 128}) {
        // Fake a mapped model and context that already use six GiB.
        const uint64_t available = (host_gib - 6) * GiB;
        const auto plan = common_fit_ram_plan(available, requests);
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
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 8 * GiB, -1, true, why) == 4096, "explicit half-free setting keeps its meaning");
    expect(common_fit_prompt_cache_ram_mib(16 * GiB, 6 * GiB, -2, false, why) == 3072,
            "fit off keeps prior tier and half-available clamp");
}

int main() {
    test_ram_fit();
    test_ram_prompt_fit();
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
