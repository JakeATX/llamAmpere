// CPU test: the card-cache profile planner sizes slots from each expert's real bytes, so a file whose routed
// experts are AI+ (route 2) fits more experts in the same budget than a 3XS file, and a mixed file lands between.
// Numbers are the home fire2 receipt: budget 18,986,500,096 B, AI+ bundle = AIT2_U gate + up (435,200 B each)
// + AIT3_U down (640,000 B); the 3XS-like bundle is 2 x 524,800 + 627,200 B. 48 layers x 512 experts, ranked mode.
// Also pins the harness catch: GGML_MOE_CACHE_PROFILE_RANKED_LIMIT=12000 clips the all-AI+ plan (12,570 fit).
#include <algorithm>
#include <cstdio>
#include <vector>

#include "../ggml/src/ggml-cuda/moe-cache.cuh"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static const size_t BUDGET = 18986500096ull;
static const int LAYERS = 48, EXPERTS = 512;

// ai_layers: layers [0, ai_layers) hold AI+ bytes (AIT2_U gate/up 435,200 B, AIT3_U down 640,000 B), the rest
// 3XS-like (524,800 / 627,200 B). Each (type, bytes) is its own pool, as in moe-cache.cu; a pure file has 2 pools.
// Counts are distinct and fall with (layer, expert), so the ranked order is layer 0 first.
static size_t plan_bundles(int ai_layers, size_t limit, size_t * bytes_out) {
    std::vector<ggml_cuda_moe_profile_item> items;
    std::vector<size_t> pool_size;
    auto pool_of = [&](size_t bytes) {
        for (size_t p = 0; p < pool_size.size(); ++p) { if (pool_size[p] == bytes) { return p; } }
        pool_size.push_back(bytes);
        return pool_size.size() - 1;
    };
    for (int l = 0; l < LAYERS; ++l) {
        const bool ai = l < ai_layers;
        for (int e = 0; e < EXPERTS; ++e) {
            const uint64_t count = (uint64_t)(LAYERS * EXPERTS - (l * EXPERTS + e));
            for (int t = 0; t < 3; ++t) {
                const size_t bytes = ai ? (t < 2 ? 435200 : 640000) : (t < 2 ? 524800 : 627200);
                items.push_back({(size_t)(l * 3 + t), e, (size_t)l, pool_of(bytes), bytes, count});
            }
        }
    }
    const std::vector<size_t> caps(pool_size.size(), (size_t)2 * LAYERS * EXPERTS);
    ggml_cuda_moe_profile_plan plan;
    if (!ggml_cuda_moe_cache_profile_plan(items, caps, BUDGET, 64, plan, false, limit)) { return 0; }
    if (bytes_out) { *bytes_out = plan.bytes; }
    return plan.bundles;
}

int main() {
    const size_t ai = 2 * 435200 + 640000, x3 = 2 * 524800 + 627200;
    size_t bytes = 0;

    // pure 3XS-like file: floor(budget / bundle) = 11,323; the 12,000 product cap never binds
    const size_t all_x3 = plan_bundles(0, 24576, &bytes);
    CHECK(all_x3 == BUDGET / x3 && all_x3 == 11323, "3XS-like: got %zu want 11323", all_x3);
    CHECK(plan_bundles(0, 12000, nullptr) == all_x3, "3XS-like under limit 12000 changed");

    // pure AI+ file (route 2): floor(budget / AI+ bundle) = 12,570 = +1,247 experts on card (+11.0%)
    const size_t all_ai = plan_bundles(LAYERS, 24576, &bytes);
    CHECK(all_ai == BUDGET / ai && all_ai == 12570, "AI+: got %zu want 12570", all_ai);
    CHECK(bytes <= BUDGET, "AI+ bytes %zu > budget", bytes);
    // harness catch: the product env's RANKED_LIMIT=12000 clips the route-2 plan
    const size_t all_ai12 = plan_bundles(LAYERS, 12000, nullptr);
    CHECK(all_ai12 == 12000, "limit 12000 must clip the AI+ plan: got %zu", all_ai12);

    // the smoke file (one AI+ layer, ranked first): 512 AI+ bundles, then 3XS-like until the budget runs out
    const size_t mixed = plan_bundles(1, 24576, &bytes);
    const size_t mixed_want = 512 + (BUDGET - 512 * ai) / x3;
    CHECK(mixed == mixed_want, "1 AI+ layer: got %zu want %zu", mixed, mixed_want);
    CHECK(all_x3 < mixed && mixed < all_ai, "ordering 3XS %zu < mixed %zu < AI+ %zu", all_x3, mixed, all_ai);

    printf("route-2 plan: 3XS-like %zu, 1 AI+ layer %zu, all AI+ %zu (limit 12000: %zu) -> %s\n",
        all_x3, mixed, all_ai, all_ai12, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
