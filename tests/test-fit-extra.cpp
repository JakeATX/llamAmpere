#include "fit.h"
#include "../src/llama-moe-arena.h"
#include "../src/llama-b1.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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
        const uint64_t n_tokens = 100000;
        const uint64_t spec_bytes = budget - 65536 - n_tokens * sizeof(llama_token);
        expect(common_fit_eval_cache_save_fits(available, true, n_tokens, spec_bytes),
                "token and speculative copies at the live eval quota fit");
        expect(!common_fit_eval_cache_save_fits(available, true, n_tokens, spec_bytes + 1),
                "one byte over the live save quota is refused before either copy");
        expect(!common_fit_eval_cache_save_fits(available, true, n_tokens + 1, spec_bytes),
                "the token copy is charged along with the speculative state");
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

int main() {
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
