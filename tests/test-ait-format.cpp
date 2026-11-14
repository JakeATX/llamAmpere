// Format-version and independent exhaustive CPU level checks. No GPU needed.
#include "../src/llama-ait-format.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "../ggml/src/ggml-ai4.h"
#include "ggml-ai4-decode.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-aif175-trellis.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <vector>

static int checks = 0;
static void require(bool ok, const char * message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
static int aif25_level_reference(const uint8_t * tile, int index) {
    uint32_t state = 0;
    for (int j = 0; j < 16; ++j) {
        const int bit = (5*(index/2 + 1) - 16 + j + 320)%320;
        const int byte = (bit/32)*4 + 3 - (bit%32)/8;
        state = (state << 1) | ((tile[byte] >> (7-bit%8)) & 1);
    }
    const uint32_t mul = index%2 ? uint32_t(uint64_t(0x87D8D12Du)*0x87D8D12Du) : 0x87D8D12Du;
    const uint32_t x = state*mul;
    const unsigned sum = (x & 255)+((x >> 8) & 255)+((x >> 16) & 255)+(x >> 24);
    return int((sum+2)/4)-128;
}
static void check_aif25_windows() {
    for (int offset = 0; offset < 4; ++offset) {
        uint8_t guarded[46];
        uint8_t * tile = guarded + 1 + offset;
        for (int pattern = 0; pattern < 354; ++pattern) {
            std::memset(guarded, 0xa5, sizeof(guarded));
            uint32_t random = uint32_t(pattern+1);
            for (int b = 0; b < 40; ++b) {
                random = random*1664525u + 1013904223u;
                tile[b] = pattern == 321 ? 255 : pattern > 321 ? uint8_t(random >> 24) : 0;
            }
            if (pattern < 320) { tile[pattern/8] = uint8_t(1u << (pattern%8)); }
            uint8_t before[46];
            std::memcpy(before, guarded, sizeof(guarded));
            for (int index = 0; index < 128; ++index) {
                require(aif25_level(tile, index) == aif25_level_reference(tile, index),
                        "2.5 word windows match independent circular bits at every pair and alignment");
            }
            require(std::memcmp(before, guarded, sizeof(guarded)) == 0, "2.5 word windows preserve tile and guards");
        }
    }
}
static int allocation_calls = 0;
static int release_calls = 0;
static void * deny_allocation(size_t) {
    ++allocation_calls;
    return nullptr;
}
static void count_release(void * ptr) {
    require(ptr == nullptr, "denied encoder allocation has no storage to release");
    ++release_calls;
}
static int fail_allocation_at = 0;
static int live_allocations = 0;
static void * fail_allocation(size_t size) {
    if (++allocation_calls == fail_allocation_at) { return nullptr; }
    void * ptr = std::malloc(size);
    require(ptr != nullptr, "encoder failure fixture allocates small scratch buffers");
    ++live_allocations;
    return ptr;
}
static void release_allocation(void * ptr) {
    ++release_calls;
    if (ptr) { --live_allocations; }
    std::free(ptr);
}
static void check_aif175_predecessors() {
    std::vector<float> current(65536);
    std::vector<float> best(8192+2, -1234.0f);
    std::vector<uint8_t> trace(8192+2, 0xa5);
    std::vector<uint8_t> levels(2*65536);
    aif175_level_map(levels.data());
    float error0[256], error1[256];
    for (int level = 0; level < 256; ++level) {
        error0[level] = (level%13)*0.0625f;
        error1[level] = (level%17)*0.03125f;
    }
    for (int width : {3, 4}) {
        const int tails = 1 << (16-width), heads = 1 << width;
        for (int head = 0; head < heads; ++head) {
            for (int tail = 0; tail < tails; ++tail) {
                const int winner = (tail*7)%heads;
                current[head*tails+tail] = tail%11 == 0 ? INFINITY :
                    head == winner || (tail%7 == 0 && head == (winner+1)%heads) ? 0.0f :
                    1.0f + ((head+tail)%23)*0.25f;
                if (head && tail%31 == 0) { current[head*tails+tail] = NAN; }
                if (head && tail%37 == 0) { current[head*tails+tail] = -INFINITY; }
                if (head == winner && tail%13 == 0 && tail%11 != 0 && tail%31 != 0 && tail%37 != 0) {
                    current[head*tails+tail] = -0.0f;
                }
            }
        }
        const auto full = current;
        for (int count : {8, 16, 128, 1024, tails/2, tails, 2*tails, 4*tails, 65536}) {
            for (int first : {0, (32768/count)*count, 65536-count}) {
                current = full;
                for (int s = 0; s < first; ++s) { current[s] = INFINITY; }
                for (int s = first+count; s < 65536; ++s) { current[s] = INFINITY; }
                const auto unchanged = current;
                std::fill(best.begin(), best.end(), -1234.0f);
                std::fill(trace.begin(), trace.end(), 0xa5);
                aif175_predecessors(current.data(), width, first, count, best.data()+1, trace.data()+1);
                // Original scalar recurrence is independent of the new loop order.
                for (int tail = 0; tail < tails; ++tail) {
                    const int begin = first%tails;
                    if (count < tails && (tail < begin || tail >= begin+count)) {
                        require(best[tail+1] == -1234.0f && trace[tail+1] == 0xa5,
                                "1.75 predecessor leaves inactive tail scratch unchanged");
                        continue;
                    }
                    float expected = current[tail];
                    int winner_head = 0;
                    for (int head = 1; head < heads; ++head) {
                        if (current[head*tails+tail] < expected) {
                            expected = current[head*tails+tail];
                            winner_head = head;
                        }
                    }
                    require(std::memcmp(&best[tail+1], &expected, sizeof(float)) == 0,
                            "1.75 predecessor minimum retains exact signed zero and infinity bits");
                    require(trace[tail+1] == winner_head, "1.75 predecessor retains first tied head");
                }
                require(best[0] == -1234.0f && trace[0] == 0xa5, "1.75 predecessor leading canaries");
                for (size_t j = tails+1; j < best.size(); ++j) {
                    require(best[j] == -1234.0f && trace[j] == 0xa5, "1.75 predecessor tail bounds");
                }
                if (count == 65536) {
                    for (int state = 0; state < 65536; ++state) {
                        const int tail = state >> width;
                        const float error = (state%37)*0.03125f;
                        uint8_t last_trace[3] = {0xa5, 0xa5, 0xa5};
                        const float last = aif175_terminal_cost(current.data(), width, state, error, last_trace+1);
                        require(last == best[tail+1] + error, "1.75 fixed end matches full scalar state cost");
                        require(last_trace[1] == trace[tail+1], "1.75 fixed end keeps first ties and infinite heads");
                        require(last_trace[0] == 0xa5 && last_trace[2] == 0xa5, "1.75 fixed end writes only its trace byte");
                    }
                    for (bool infinite : {false, true}) {
                        std::vector<float> next(65536), all_infinite(infinite ? 65536 : 0, INFINITY);
                        const float * input = infinite ? all_infinite.data() : current.data();
                        for (int s = 0; s < 65536; ++s) {
                            const float error = error0[levels[2*s]] + error1[levels[2*s+1]];
                            next[s] = infinite ? INFINITY : best[(s >> width)+1] + error;
                        }
                        const int last_width = 7-width, last_tails = 1 << (16-last_width);
                        std::vector<float> second(65536), next_best(last_tails);
                        std::vector<uint8_t> next_heads(last_tails, 0);
                        for (int tail = 0; tail < last_tails; ++tail) {
                            next_best[tail] = next[tail];
                            for (int head = 1; head < (1 << last_width); ++head) {
                                if (next[head*last_tails+tail] < next_best[tail]) {
                                    next_best[tail] = next[head*last_tails+tail]; next_heads[tail] = uint8_t(head);
                                }
                            }
                        }
                        for (int s = 0; s < 65536; ++s) {
                            const float error = error1[levels[2*s]] + error0[levels[2*s+1]];
                            second[s] = next_best[s >> last_width] + error;
                        }
                        for (int state : {0, 1, 7, 15, 31, 8191, 32768, 65535}) {
                            std::vector<uint8_t> saved_trace(8192+2, 0xa5), expected_trace = saved_trace;
                            const int tail = state >> last_width;
                            float expected = next[tail];
                            int winner = 0;
                            for (int head = 0; head < (1 << last_width); ++head) {
                                const int s = head*last_tails+tail;
                                if (next[s] < expected) { expected = next[s]; winner = head; }
                                expected_trace[(s >> width)+1] = infinite ? 0 : trace[(s >> width)+1];
                            }
                            const float error = (state%37)*0.03125f;
                            uint8_t last_trace[3] = {0xa5, 0xa5, 0xa5};
                            const float last = aif175_fixed_tail_cost(input, levels.data(), width, state,
                                                                     error0, error1, error, saved_trace.data()+1, last_trace+1);
                            require(last == expected + error, "1.75 fixed two-step tail matches full scalar recurrence");
                            require(last_trace[1] == winner, "1.75 fixed two-step tail retains first tied head");
                            require(last_trace[0] == 0xa5 && last_trace[2] == 0xa5, "1.75 fixed two-step final trace bounds");
                            require(saved_trace == expected_trace, "1.75 fixed two-step writes only needed earlier traces");
                            std::fill(saved_trace.begin(), saved_trace.end(), 0xa5);
                            std::fill(expected_trace.begin(), expected_trace.end(), 0xa5);
                            std::vector<uint8_t> middle_trace(8192+2, 0xa5), expected_middle = middle_trace;
                            const int final_tail = state >> width;
                            float expected_three = second[final_tail];
                            int final_winner = 0;
                            for (int head = 0; head < heads; ++head) {
                                const int s = head*tails+final_tail;
                                if (second[s] < expected_three) { expected_three = second[s]; final_winner = head; }
                                expected_middle[(s >> last_width)+1] = next_heads[s >> last_width];
                                for (int previous = 0; previous < (1 << last_width); ++previous) {
                                    const int before = previous*last_tails+(s >> last_width);
                                    expected_trace[(before >> width)+1] = infinite ? 0 : trace[(before >> width)+1];
                                }
                            }
                            const float three = aif175_fixed_three_cost(input, levels.data(), width, state,
                                error0, error1, error1, error0, error, saved_trace.data()+1, middle_trace.data()+1, last_trace+1);
                            require(three == expected_three + error && last_trace[1] == final_winner,
                                    "1.75 fixed three-step tail matches full scalar costs and tie order");
                            require(saved_trace == expected_trace && middle_trace == expected_middle && last_trace[0] == 0xa5 && last_trace[2] == 0xa5,
                                    "1.75 fixed three-step writes only required traces and keeps guards");
                        }
                    }
                }
                require(std::memcmp(current.data(), unchanged.data(), current.size()*sizeof(float)) == 0,
                        "1.75 predecessor keeps all input costs unchanged");
                const auto expected_best = best;
                const auto expected_trace = trace;
                for (int s = 0; s < first; ++s) { current[s] = -INFINITY; }
                for (int s = first+count; s < 65536; ++s) { current[s] = -INFINITY; }
                const auto poisoned = current;
                std::fill(best.begin(), best.end(), -1234.0f);
                std::fill(trace.begin(), trace.end(), 0xa5);
                aif175_predecessors(current.data(), width, first, count, best.data()+1, trace.data()+1);
                require(best == expected_best && trace == expected_trace, "1.75 predecessor ignores stale costs outside the live range");
                require(std::memcmp(current.data(), poisoned.data(), current.size()*sizeof(float)) == 0,
                        "1.75 predecessor keeps stale costs and NaN source bits unchanged");
            }
        }
    }
    // These errors round differently when added to the prior one at a time.
    const float prior = 1e8f, error0_value = 4.01f, error1_value = 4.01f;
    const float pair_error = error0_value + error1_value;
    const float expected = prior + pair_error;
    const float split_first = prior + error0_value;
    require(split_first + error1_value != expected, "1.75 non-dyadic fixture detects reassociation");
    std::fill(current.begin(), current.end(), prior);
    std::fill(error0, error0+256, error0_value);
    std::fill(error1, error1+256, error1_value);
    std::fill(best.begin(), best.end(), prior);
    std::vector<float> costs(65536);
    for (int width : {3, 4}) {
        aif175_state_costs(error0, error1, levels.data(), width, 0, 65536, false, best.data(), costs.data());
        for (float cost : costs) { require(cost == expected, "1.75 state cost keeps paired-error rounding"); }
        std::fill(trace.begin(), trace.end(), 0xa5);
        uint8_t last_trace[3] = {0xa5, 0xa5, 0xa5};
        const float final_error = 0.1f;
        const float last = aif175_fixed_tail_cost(current.data(), levels.data(), width, 0,
                                                 error0, error1, final_error, trace.data()+1, last_trace+1);
        require(last == expected + final_error, "1.75 fixed tail keeps non-dyadic rounding");
        require(last_trace[1] == 0, "1.75 non-dyadic tail keeps first tied head");
        require(last_trace[0] == 0xa5 && last_trace[2] == 0xa5 && trace.front() == 0xa5 && trace.back() == 0xa5,
                "1.75 non-dyadic tail trace canaries");
        std::vector<uint8_t> middle_trace(8192+2, 0xa5);
        float three_errors[256];
        std::fill(three_errors, three_errors+256, 2.005f);
        const float three_pair = three_errors[0] + three_errors[0];
        const float first_expected = prior + three_pair;
        const float second_expected = first_expected + three_pair;
        require(second_expected != prior + (three_pair+three_pair), "1.75 three-step fixture detects merging step errors");
        const float three = aif175_fixed_three_cost(current.data(), levels.data(), width, 0,
            three_errors, three_errors, three_errors, three_errors, final_error, trace.data()+1, middle_trace.data()+1, last_trace+1);
        require(three == second_expected + final_error && last_trace[1] == 0, "1.75 three-step tail keeps paired rounding and first ties");
        require(middle_trace.front() == 0xa5 && middle_trace.back() == 0xa5 && trace.front() == 0xa5 && trace.back() == 0xa5 &&
                last_trace[0] == 0xa5 && last_trace[2] == 0xa5, "1.75 three-step rounded tail trace guards");
    }
}
static void check_aif175_trace_bounds() {
    require(AIF175_TRACE_BYTES == 393216, "1.75 trace holds 32 steps of each width");
    struct guarded_workspace { uint32_t before; aif175_workspace data; uint32_t after; };
    std::vector<guarded_workspace> storage(1);
    auto & workspace = storage[0];
    workspace.before = workspace.after = 0x76543210;
    uint8_t * levels = workspace.data.levels;
    aif175_level_map(levels);
    std::vector<uint8_t> unchanged_levels(levels, levels+2*65536);
    const std::vector<float> unchanged_rotated(16*512, -1234.0f);
    std::memcpy(workspace.data.rotated, unchanged_rotated.data(), sizeof(workspace.data.rotated));
    float * costs = workspace.data.costs;
    float * best = workspace.data.best;
    aif175_errors * tables = workspace.data.tables;
    std::vector<uint8_t> trace(393216+2, 0xa5);
    float target[128], metric[128];
    for (int j = 0; j < 128; ++j) { target[j] = -128; metric[j] = 1; }
    for (int p = 0; p < 64; ++p) { aif175_error_tables(target + 2*p, metric + 2*p, tables[p].values); }
    std::vector<aif175_errors> unchanged_tables(tables, tables+64);
    for (int offset : {0, 31}) {
        uint16_t path[66];
        for (int j = 0; j < 66; ++j) { path[j] = 0xbeef; }
        std::fill(trace.begin(), trace.end(), 0xa5);
        std::fill(costs, costs+65536, NAN);
        std::fill(best, best+8192, NAN);
        // State zero has exactly zero error; every other cost is nonnegative.
        require(aif175_path(target, metric, tables, levels, offset, offset ? -1 : 0,
                            costs, best, trace.data()+1, path+1), "1.75 bounded trace finds the zero-cost ring");
        require(std::memcmp(levels, unchanged_levels.data(), sizeof(workspace.data.levels)) == 0 &&
                std::memcmp(workspace.data.rotated, unchanged_rotated.data(), sizeof(workspace.data.rotated)) == 0,
                "1.75 zero ring preserves adjacent levels and rotated weights");
        require(std::memcmp(tables, unchanged_tables.data(), sizeof(workspace.data.tables)) == 0,
                "1.75 zero ring does not overwrite adjacent cached tables");
        require(workspace.before == 0x76543210 && workspace.after == 0x76543210, "1.75 zero ring keeps workspace guards");
        require(trace.front() == 0xa5 && trace.back() == 0xa5, "1.75 both trace parities keep byte guards");
        require(path[0] == 0xbeef && path[65] == 0xbeef, "1.75 bounded traceback keeps state guards");
        for (int p = 1; p <= 64; ++p) { require(path[p] == 0, "1.75 bounded traceback keeps known optimal states"); }
    }
    // A one-to-one test map gives each pair one zero-cost state in a nonzero ring.
    uint8_t tile[28];
    uint32_t rng = 229;
    for (auto & byte : tile) { rng = rng*1664525u+1013904223u; byte = uint8_t(rng >> 24); }
    uint16_t expected[64];
    for (int p = 0; p < 64; ++p) {
        const int end = 7*(p/2)+(p%2 ? 7 : 3);
        uint16_t state = 0;
        for (int b = end-16; b < end; ++b) {
            const int bit = (b+224)%224;
            const int byte = bit/32*4+3-bit%32/8;
            state = uint16_t((state << 1) | ((tile[byte] >> (7-bit%8)) & 1));
        }
        expected[p] = state;
        target[2*p] = int(state >> 8)-128;
        target[2*p+1] = int(state & 255)-128;
        metric[2*p] = 1; metric[2*p+1] = 0.25f;
    }
    for (int s = 0; s < 65536; ++s) { levels[2*s] = uint8_t(s >> 8); levels[2*s+1] = uint8_t(s); }
    unchanged_levels.assign(levels, levels+2*65536);
    for (int p = 0; p < 64; ++p) { aif175_error_tables(target + 2*p, metric + 2*p, tables[p].values); }
    unchanged_tables.assign(tables, tables+64);
    uint8_t * owned_trace = workspace.data.trace;
    for (int offset : {0, 31}) {
        for (bool fixed : {false, true}) {
            uint16_t path[66];
            for (auto & state : path) { state = 0xbeef; }
            std::fill(costs, costs+65536, NAN);
            std::fill(best, best+8192, NAN);
            std::memset(owned_trace, 0xa5, sizeof(workspace.data.trace));
            const int boundary = fixed ? expected[(offset+63)%64] : -1;
            require(aif175_path(target, metric, tables, levels, offset, boundary,
                                costs, best, owned_trace, path+1), "1.75 nonzero ring is reachable for both start parities");
            require(std::memcmp(levels, unchanged_levels.data(), sizeof(workspace.data.levels)) == 0 &&
                    std::memcmp(workspace.data.rotated, unchanged_rotated.data(), sizeof(workspace.data.rotated)) == 0,
                    "1.75 nonzero ring preserves adjacent levels and rotated weights");
            require(std::memcmp(tables, unchanged_tables.data(), sizeof(workspace.data.tables)) == 0,
                    "1.75 nonzero ring reuses adjacent read-only tables across both boundaries and offsets");
            require(workspace.before == 0x76543210 && workspace.after == 0x76543210, "1.75 nonzero ring keeps workspace guards");
            require(std::memcmp(path+1, expected, sizeof(expected)) == 0, "1.75 traceback finds each unique nonzero optimal state");
            require(path[0] == 0xbeef && path[65] == 0xbeef, "1.75 nonzero traceback keeps state guards");
            uint8_t packed[28];
            require(aif175_pack(path+1, packed) && std::memcmp(packed, tile, sizeof(tile)) == 0,
                    "1.75 nonzero optimal ring keeps independent literal bytes");
        }
    }
}
static void check_aif175_errors() {
    std::vector<uint8_t> guarded_levels(2*65536+2, 0xa5);
    uint8_t * levels = guarded_levels.data()+1;
    const double rms = aif175_level_map(levels);
    double square_sum = 0;
    for (uint32_t state = 0; state < 65536; ++state) {
        for (int odd = 0; odd < 2; ++odd) {
            const uint32_t c = odd ? uint32_t(uint64_t(0x87D8D12Du)*0x87D8D12Du) : 0x87D8D12Du;
            const uint32_t z = state*c;
            const unsigned sum = (z & 255)+((z >> 8) & 255)+((z >> 16) & 255)+(z >> 24);
            const int level = int((sum+2)/4)-128;
            require(aif175_level_state(state, c) == level, "1.75 packed byte sums match independent levels for both multipliers");
            require(levels[2*state+odd] == level+128, "1.75 production map matches independent MAGIC2 bytes");
            if (!odd) { square_sum += double(level)*level; }
        }
    }
    require(rms == std::sqrt(square_sum/65536), "1.75 production map keeps level RMS");
    require(guarded_levels.front() == 0xa5 && guarded_levels.back() == 0xa5, "1.75 production map bounds");
    const auto unchanged_levels = guarded_levels;
    const float targets[8][2] = {{0, 0}, {-1.125f, 53.75f}, {149.5f, -217.25f},
                                {0.1f, -17.3f}, {127.0001f, -127.9999f}, {-0.000031f, 0.000071f}, {-0.0f, 0}, {0, -0.0f}};
    const float metrics[8][2] = {{0, 1}, {0.125f, 0.75f}, {1, 1.0f/65536},
                                {0.13f, 0.59f}, {0.3333333f, 0.000017f}, {1e-20f, 0.97f}, {0, -0.0f}, {-0.0f, -0.0f}};
    for (int test = 0; test < 8; ++test) {
        struct { float before; float tables[2][256]; float after; } guarded;
        guarded.before = guarded.after = -1234.0f;
        aif175_error_tables(targets[test], metrics[test], guarded.tables);
        for (uint32_t state = 0; state < 65536; ++state) {
            float scalar[2];
            unsigned index[2];
            double exact_pair = 0;
            for (int odd = 0; odd < 2; ++odd) {
                index[odd] = levels[2*state+odd];
                const float delta = float(int(index[odd])-128) - targets[test][odd];
                scalar[odd] = (delta*delta)*metrics[test][odd];
                require(test >= 3 || guarded.tables[odd][index[odd]] == scalar[odd], "1.75 table matches scalar error for every dyadic state");
                const double exact_delta = double(int(index[odd])-128) - double(targets[test][odd]);
                const double exact = exact_delta*exact_delta*double(metrics[test][odd]);
                exact_pair += exact;
                require(std::isfinite(guarded.tables[odd][index[odd]]) &&
                        std::fabs(double(guarded.tables[odd][index[odd]])-exact) <= 2e-6*exact+1e-38,
                        "1.75 production map error stays within float budget against double oracle");
            }
            const float pair = guarded.tables[0][index[0]]+guarded.tables[1][index[1]];
            require(test >= 3 || pair == scalar[0]+scalar[1],
                    "1.75 table preserves paired error with zero and unequal metrics");
            require(std::fabs(double(pair)-exact_pair) <= 3e-6*exact_pair+1e-38,
                    "1.75 paired production lookup stays within float budget against double oracle");
        }
        require(guarded.before == -1234.0f && guarded.after == -1234.0f, "1.75 error table canaries");
        std::vector<float> costs(65536+2), best(8192);
        for (int width : {3, 4}) {
            for (int tail = 0; tail < (65536 >> width); ++tail) {
                best[tail] = tail%11 == 0 ? INFINITY : (tail%37)*0.25f;
            }
            best[1] = -0.0f; best[2] = -INFINITY;
            const auto unchanged_best = best;
            for (int count : {1 << width, 128, 1024, 8192, 65536}) {
                for (int first : {0, (32768/count)*count, 65536-count}) {
                    for (bool initial : {false, true}) {
                        std::fill(costs.begin(), costs.end(), -1234.0f);
                        aif175_state_costs(guarded.tables[0], guarded.tables[1], levels, width, first, count,
                                          test == 6, initial ? nullptr : best.data(), costs.data()+1);
                        for (int s = 0; s < 65536; ++s) {
                            float expected = -1234.0f;
                            if (s >= first && s < first+count) {
                                const float error = guarded.tables[0][levels[2*s]] + guarded.tables[1][levels[2*s+1]];
                                expected = initial ? error : best[s >> width] + error;
                            }
                            require(std::memcmp(&costs[s+1], &expected, sizeof(float)) == 0,
                                    "1.75 grouped and padded state costs match scalar bits in live range");
                        }
                        require(costs.front() == -1234.0f && costs.back() == -1234.0f, "1.75 grouped state cost guards");
                        require(best == unchanged_best, "1.75 grouped state costs leave prior minima unchanged");
                    }
                }
            }
        }
    }
    require(guarded_levels == unchanged_levels, "1.75 error lookup leaves production map and guards unchanged");
    for (float target : {0.0f, -0.0f, 1.0f, 0x1.fffffep127f}) {
        for (float metric : {0.0f, -0.0f}) {
            const float pair_target[2] = {target, -target}, pair_metric[2] = {metric, -metric};
            struct { float before; float tables[2][256]; float after; } guarded;
            guarded.before = guarded.after = -1234.0f;
            aif175_error_tables(pair_target, pair_metric, guarded.tables);
            for (int j = 0; j < 2; ++j) {
                for (int level = 0; level < 256; ++level) {
                    const float delta = float(level-128) - pair_target[j];
                    const float expected = (delta*delta)*pair_metric[j];
                    const float actual = guarded.tables[j][level];
                    require(std::isnan(expected) ? std::isnan(actual) : std::memcmp(&actual, &expected, sizeof(float)) == 0,
                            "1.75 zero metric tables keep scalar signed zeros and overflowing nonzero targets");
                }
            }
            require(guarded.before == -1234.0f && guarded.after == -1234.0f, "1.75 zero metric table bounds");
        }
    }
}
static void check_aif175_windows() {
    uint8_t storage[28+4];
    for (int alignment = 0; alignment < 4; ++alignment) {
        uint8_t * tile = storage + alignment;
        // Every stream bit, and its complement, test each wrap and word split.
        for (int pattern = -2; pattern < 2*224; ++pattern) {
            std::memset(storage, 0xa5, sizeof(storage));
            std::memset(tile, pattern == -1 || pattern >= 224 ? 255 : 0, 28);
            if (pattern >= 0) {
                const int bit = pattern%224;
                tile[4*(bit/32)+3-(bit%32)/8] ^= uint8_t(1u << (7-bit%8));
            }
            const std::vector<uint8_t> before(storage, storage+sizeof(storage));
            uint16_t path[64];
            for (int pair = 0; pair < 64; ++pair) {
                const int end = 7*(pair/2) + (pair%2 ? 7 : 3);
                uint32_t expected = 0;
                for (int j = end-16; j < end; ++j) {
                    const int bit = (j+224)%224;
                    expected = 2*expected + ((tile[4*(bit/32)+3-(bit%32)/8] >> (7-bit%8)) & 1);
                }
                path[pair] = uint16_t(expected);
                require(aif175_state(tile, pair) == expected, "1.75 word window matches every single-bit basis");
                for (int odd = 0; odd < 2; ++odd) {
                    const uint32_t c = odd ? uint32_t(uint64_t(0x87D8D12Du)*0x87D8D12Du) : 0x87D8D12Du;
                    const uint32_t z = expected*c;
                    const int sum = int(z & 255)+int((z >> 8) & 255)+int((z >> 16) & 255)+int(z >> 24);
                    require(aif175_level(tile, 2*pair+odd) == (sum+2)/4-128, "1.75 word window keeps both decoded levels");
                }
            }
            uint32_t words[7] = {};
            for (int w = 0; w < 7; ++w) {
                for (int byte = 3; byte >= 0; --byte) { words[w] = 256*words[w] + tile[4*w+byte]; }
            }
            for (int row = 0; row < 16; ++row) {
                const int pair = 4*row;
                const uint32_t expected = (uint32_t(path[pair]) << 11) | (uint32_t(path[pair+1] & 15u) << 7) |
                    (uint32_t(path[pair+2] & 7u) << 4) | (path[pair+3] & 15u);
                const uint32_t states = aif175_state_quad(words, row);
                require(states == expected, "1.75 four-state window matches independent stream bits");
                require((states >> 11) == path[pair] && ((states >> 7) & 0xffffu) == path[pair+1] &&
                        ((states >> 4) & 0xffffu) == path[pair+2] && (states & 0xffffu) == path[pair+3],
                        "1.75 four-state window retains all complete states");
            }
            uint8_t packed[28+2];
            std::memset(packed, 0xa5, sizeof(packed));
            const std::vector<uint16_t> path_before(path, path+64);
            require(aif175_pack(path, packed+1), "1.75 pack accepts independent closed bit paths");
            require(std::memcmp(packed+1, tile, 28) == 0, "1.75 packing retains independent literal tile bytes");
            require(packed[0] == 0xa5 && packed[29] == 0xa5, "1.75 packing keeps unaligned output guards");
            require(std::memcmp(path, path_before.data(), sizeof(path)) == 0, "1.75 packing keeps input states unchanged");
            for (int b = 0; b < 28; ++b) { packed[b+1] = uint8_t(~tile[b]); }
            const std::vector<uint8_t> packed_before(packed, packed+sizeof(packed));
            if (alignment == 0 && (pattern == -2 || pattern == -1 || pattern == 121)) {
                for (int offset : {0, 1, 31, 65, 100}) {
                    uint16_t shared[64+16];
                    std::memset(shared, 0xa5, sizeof(shared));
                    std::memcpy(shared+1, path, sizeof(path));
                    const auto * raw = reinterpret_cast<const uint8_t *>(shared);
                    const std::vector<uint8_t> shared_before(raw, raw+sizeof(shared));
                    auto * dst = reinterpret_cast<uint8_t *>(shared+1)+offset;
                    require(aif175_pack(shared+1, dst), "1.75 pair pack reads all states before overlapping writes");
                    require(std::memcmp(dst, tile, 28) == 0, "1.75 overlapping pair pack retains literal bytes");
                    for (size_t j = 0; j < sizeof(shared); ++j) {
                        if (raw+j < dst || raw+j >= dst+28) {
                            require(raw[j] == shared_before[j], "1.75 overlapping pack writes only its output span");
                        }
                    }
                    std::memcpy(shared+1, path, sizeof(path));
                    shared[64] ^= 0x8000;
                    const std::vector<uint8_t> invalid_shared(raw, raw+sizeof(shared));
                    require(!aif175_pack(shared+1, dst), "1.75 overlapping pack rejects a broken final link");
                    require(std::memcmp(shared, invalid_shared.data(), sizeof(shared)) == 0,
                            "1.75 rejected overlapping pack keeps every source and destination byte");
                }
                for (int pair = 0; pair < 64; ++pair) {
                    for (int bit = 0; bit < 16; ++bit) {
                        path[pair] ^= uint16_t(1u << bit);
                        const std::vector<uint16_t> invalid_before(path, path+64);
                        require(!aif175_pack(path, packed+1), "1.75 pack rejects each changed state bit at both widths and wrap");
                        require(std::memcmp(packed, packed_before.data(), sizeof(packed)) == 0,
                                "1.75 every rejected state bit keeps output and canaries");
                        require(std::memcmp(path, invalid_before.data(), sizeof(path)) == 0,
                                "1.75 path check keeps rejected states unchanged");
                        path[pair] ^= uint16_t(1u << bit);
                    }
                }
                require(std::memcmp(path, path_before.data(), sizeof(path)) == 0, "1.75 corruption fixture restores independent closed states");
            }
            // High state bits are not emitted symbols, so this breaks only closure.
            path[(pattern+2)%64] ^= 0x8000;
            require(!aif175_pack(path, packed+1), "1.75 pack rejects an open path at every pair");
            require(std::memcmp(packed, packed_before.data(), sizeof(packed)) == 0, "1.75 rejected path leaves output and guards unchanged");
            require(std::memcmp(storage, before.data(), sizeof(storage)) == 0, "1.75 window leaves tile and guards unchanged");
        }
    }
}
// Types 128 / 129 / 130: the AIT2_U2 / AIT3_U2 / AIT4_U2 tiles with one FP16 row scale per 512 columns (four 128-column chunks per block).
static void check_ai4_u2_512() {
    for (ggml_type T : {GGML_TYPE_AIT2_U2_512, GGML_TYPE_AIT3_U2_512, GGML_TYPE_AIT4_U2_512}) {
        const int k = T == GGML_TYPE_AIT2_U2_512 ? 2 : T == GGML_TYPE_AIT3_U2_512 ? 3 : 4;
        // the 640-column type of the same rate (122 / 123 / 124) and the per-128 unit type (115 / 116 / 117)
        const ggml_type wide = k == 2 ? GGML_TYPE_AIT2_U2 : k == 3 ? GGML_TYPE_AIT3_U2 : GGML_TYPE_AIT4_U2;
        const ggml_type unit = k == 2 ? GGML_TYPE_AIT2_U : k == 3 ? GGML_TYPE_AIT3_U : GGML_TYPE_AIT4_U;
        const size_t cb = size_t(256)*k, bb = 32 + 4*cb, wb = 32 + 5*cb, ub = 32 + cb;
        require(ggml_blck_size(T) == 512 && ggml_row_size(T, 512) == size_t(2 + 64*k) && 16*ggml_row_size(T, 512) == bb,
                "row512 2/3-bit exact stored bytes");
        require(ggml_ai4_rate(T) == k && ggml_ai4v2_rate(T) == k && ggml_ai4v2_chunks(T) == 4 &&
                ggml_ai4v2_chunks(wide) == 5 && ggml_ai4v2_chunks(unit) == 0 && ggml_ai4v2_chunks(GGML_TYPE_F32) == 0,
                "row512 2/3-bit rate and chunk count");
        require(ggml_ai4_trellis_mul(T) == ggml_ai4_trellis_mul(wide) && ggml_ai4v2_trellis_mul(T) == ggml_ai4_trellis_mul(wide) &&
                ggml_ai4_scale_mul(T) == ggml_ai4_scale_mul(wide), "row512 2/3-bit constants equal the 640 type");
        for (int64_t K : {int64_t(0), int64_t(-512), int64_t(128), int64_t(256), int64_t(640), int64_t(1280)}) {
            require(ggml_ai4_slab_nbytes(T, K) == 0, "row512 2/3-bit rejects partial or invalid widths");
        }
        // Widths of the dense 27B body that do not divide by 640 (3072, 4096, 6144, 9216, 17408) and some that do.
        for (int K : {512, 1024, 2560, 3072, 4096, 6144, 9216, 17408}) {
            // Per-128 unit form with random tiles; every unit of a 512 block carries that block's 16 row scales.
            std::vector<uint8_t> units(size_t(K/128)*ub), packed(size_t(K/512)*bb + 1, 0x5a);
            uint32_t state = 4100u + 7u*uint32_t(k) + uint32_t(K);
            for (auto & b : units) { state = state*1664525u + 1013904223u; b = state >> 24; }
            for (int block = 0; block < K/512; ++block) {
                uint8_t * p = packed.data() + size_t(block)*bb;
                for (int r = 0; r < 16; ++r) {
                    const uint16_t scale = uint16_t(0x2800 + ((17*block + 5*r + k) & 0x3ff));
                    p[2*r] = uint8_t(scale); p[2*r+1] = uint8_t(scale >> 8);
                }
                for (int u = 0; u < 4; ++u) {
                    uint8_t * q = units.data() + size_t(4*block + u)*ub;
                    std::memcpy(q, p, 32);
                    std::memcpy(p + 32 + u*cb, q + 32, cb);
                }
            }
            const auto before = packed;
            std::vector<float> a(size_t(16)*K, 1.0f), b(size_t(16)*K + 1, 123.0f);
            require(ggml_ai4_slab_nbytes(T, K) == packed.size() - 1 && ggml_ai4_slab_nbytes(unit, K) == units.size(),
                    "row512 2/3-bit slab bytes");
            require(ggml_ai4_dequantize_slab(unit, units.data(), K, 0, a.data()), "per-128 unit decode for the row512 oracle");
            require(ggml_ai4_dequantize_slab(T, packed.data(), K, 0, b.data()), "row512 2/3-bit public slab decoder");
            require(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0,
                    "row512 2/3-bit matches the per-128 unit decode exactly");
            bool nonzero = false;
            for (float v : a) { nonzero = nonzero || v != 0.0f; }
            require(nonzero && b.back() == 123.0f && packed == before, "row512 2/3-bit decode is non-trivial and preserves canaries and packed bytes");
        }
        // GATE: on a width both layouts can hold, the same tiles and the same row scales decode to the same bits
        // through the 640-column type (122 / 123) and the 512-column type (128 / 129).
        for (int K : {2560, 5120, 7680}) {
            const int chunks = K/128;
            std::vector<uint8_t> tiles(size_t(chunks)*cb), scales(32);
            uint32_t state = 911u + 3u*uint32_t(k) + uint32_t(K);
            for (auto & b : tiles) { state = state*1664525u + 1013904223u; b = state >> 24; }
            for (int r = 0; r < 16; ++r) {
                const uint16_t scale = uint16_t(0x2c00 + 37*r);
                scales[2*r] = uint8_t(scale); scales[2*r+1] = uint8_t(scale >> 8);
            }
            std::vector<uint8_t> w640(size_t(K/640)*wb), w512(size_t(K/512)*bb);
            for (int c = 0; c < chunks; ++c) {
                uint8_t * w = w640.data() + size_t(c/5)*wb;
                uint8_t * n = w512.data() + size_t(c/4)*bb;
                std::memcpy(w, scales.data(), 32); std::memcpy(n, scales.data(), 32);
                std::memcpy(w + 32 + size_t(c%5)*cb, tiles.data() + size_t(c)*cb, cb);
                std::memcpy(n + 32 + size_t(c%4)*cb, tiles.data() + size_t(c)*cb, cb);
            }
            require(ggml_ai4_slab_nbytes(wide, K) == w640.size() && ggml_ai4_slab_nbytes(T, K) == w512.size(),
                    "row640 and row512 2/3-bit slab sizes on a shared width");
            std::vector<float> a(size_t(16)*K, 1.0f), b(size_t(16)*K, 2.0f);
            require(ggml_ai4_dequantize_slab(wide, w640.data(), K, 0, a.data()), "row640 2/3-bit decode on a shared width");
            require(ggml_ai4_dequantize_slab(T, w512.data(), K, 0, b.data()), "row512 2/3-bit decode on a shared width");
            require(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0,
                    "row512 2/3-bit decodes bit for bit like row640 for equal tiles and scales");
        }
    }
}

// Type 127: the 2.5-bit pair stream of AIF25_U2 with one FP16 row scale per 512 columns (four chunks per block).
static void check_aif25_u2_512() {
    const ggml_type T = GGML_TYPE_AIF25_U2_512;
    require(ggml_blck_size(T) == 512 && ggml_row_size(T, 512) == 162 && 16*162 == AIF25_V2_512_BLOCK_BYTES,
            "2.5 row512 exact 2.53125-bpw bytes");
    require(ggml_ai4_rate(T) == 0, "2.5 row512 stays outside integer-rate kernels");
    for (int64_t K : {int64_t(0), int64_t(-512), int64_t(128), int64_t(256), int64_t(640), int64_t(1280)}) {
        require(ggml_ai4_slab_nbytes(T, K) == 0, "2.5 row512 rejects partial or invalid widths");
    }
    // Widths of the dense 27B body that do not divide by 640 (3072, 4096, 6144, 9216, 17408) and some that do.
    for (int K : {512, 1024, 2560, 3072, 4096, 6144, 9216, 17408}) {
        // Expand each stored row scale to four unit headers; the decoded matrix must stay exact,
        // including the rotation that crosses the boundary between 512-column blocks.
        const size_t sb = size_t(K/512)*AIF25_V2_512_BLOCK_BYTES;
        std::vector<uint8_t> packed(2*sb + 1);
        std::vector<uint8_t> expanded(size_t(K/128)*AIF25_UNIT_BYTES);
        std::vector<float> oracle(size_t(16)*K);
        uint32_t state = 7300u + uint32_t(K);
        for (auto & b : packed) { state = state*1664525u + 1013904223u; b = state >> 24; }
        uint8_t * src = packed.data() + 1 + sb;
        for (int block = 0; block < K/512; ++block) {
            uint8_t * p = src + size_t(block)*AIF25_V2_512_BLOCK_BYTES;
            for (int r = 0; r < 16; ++r) {
                const uint16_t scale = r == 6 ? 0 : r == 7 ? 0x8000 : r == 8 ? 0xb800 :
                    r == 9 ? 0x0001 : r == 10 ? 0x03ff : 0x2800 + 16*block + r;
                p[2*r] = uint8_t(scale); p[2*r+1] = uint8_t(scale >> 8);
                for (int k = 0; k < 512; ++k) {
                    oracle[size_t(r)*K + block*512 + k] =
                        ai4_fp16_to_fp32(scale)*aif25_level_reference(p + 32 + (k/8)*40, r*8 + k%8);
                }
            }
            for (int u = 0; u < 4; ++u) {
                uint8_t * q = expanded.data() + size_t(4*block + u)*AIF25_UNIT_BYTES;
                std::memcpy(q, p, 32);
                std::memcpy(q + 32, p + 32 + u*640, 640);
            }
        }
        const auto before = packed;
        require(ggml_ai4_slab_nbytes(T, K) == sb, "2.5 row512 slab bytes");
        require(ggml_validate_row_data(T, src, sb), "2.5 row512 validates finite scales in unaligned multi-block storage");
        require(!ggml_validate_row_data(T, src, sb-162), "2.5 row512 validation rejects a partial interleaved block");
        require(!ggml_validate_row_data(T, nullptr, sb), "2.5 row512 validation rejects missing block data");
        std::vector<float> a(size_t(16)*K), b(size_t(16)*K+2, 123.0f);
        const auto output_before = b;
        for (int edge = 0; edge < (K == 512 ? 1 : 2); ++edge) {
            const int block = edge ? K/512-1 : 0;
            for (int r : {0, 15}) {
                uint8_t * scale = src + size_t(block)*AIF25_V2_512_BLOCK_BYTES + 2*r;
                const uint8_t lo = scale[0], hi = scale[1];
                for (uint16_t bad : {0x7c00, 0xfc00, 0x7e00, 0xfc01}) {
                    scale[0] = uint8_t(bad); scale[1] = uint8_t(bad >> 8);
                    require(!ggml_validate_row_data(T, src, sb), "2.5 row512 validation rejects nonfinite scales at edge rows and blocks");
                    require(!ggml_ai4_dequantize_slab(T, packed.data()+1, K, 1, b.data()+1),
                            "2.5 row512 direct decode rejects nonfinite scales in the selected slab");
                    require(b == output_before, "2.5 row512 rejected decode leaves all output and canaries unchanged");
                }
                scale[0] = lo; scale[1] = hi;
            }
        }
        require(packed == before && ggml_validate_row_data(T, src, sb), "2.5 row512 validation preserves payload");
        const int rb = ai4u_rot_size(K);
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < K; k += rb) { ai4_unrotate_n(oracle.data() + size_t(r)*K + k, rb); }
        }
        aif25_dequantize_slab(expanded.data(), a.data(), K);
        require(ggml_ai4_dequantize_slab(T, packed.data()+1, K, 1, b.data()+1), "2.5 row512 public slab decoder");
        require(std::memcmp(a.data(), b.data()+1, a.size()*sizeof(float)) == 0,
                "2.5 row512 matches expanded independent scale headers exactly");
        require(std::memcmp(oracle.data(), b.data()+1, oracle.size()*sizeof(float)) == 0,
                "2.5 row512 decoded scales match canonical values independent of host byte order");
        require(b.front() == 123.0f && b.back() == 123.0f && packed == before, "2.5 row512 valid decode preserves canaries and packed bytes");
    }
    // GATE: on a width both layouts can hold, the same pair chunks and the same row scales decode to the same bits
    // through type 125 (one scale per 640 columns) and type 127 (one scale per 512 columns).
    for (int K : {2560, 5120, 7680}) {
        const int chunks = K/128;
        std::vector<uint8_t> payload(size_t(chunks)*640), scales(32);
        uint32_t state = 91u + uint32_t(K);
        for (auto & b : payload) { state = state*1664525u + 1013904223u; b = state >> 24; }
        for (int r = 0; r < 16; ++r) {
            const uint16_t scale = r == 3 ? 0 : r == 4 ? 0x8000 : uint16_t(0x2c00 + 37*r);
            scales[2*r] = uint8_t(scale); scales[2*r+1] = uint8_t(scale >> 8);
        }
        std::vector<uint8_t> wide(size_t(K/640)*AIF25_V2_BLOCK_BYTES), narrow(size_t(K/512)*AIF25_V2_512_BLOCK_BYTES);
        for (int c = 0; c < chunks; ++c) {
            uint8_t * w = wide.data() + size_t(c/5)*AIF25_V2_BLOCK_BYTES;
            uint8_t * n = narrow.data() + size_t(c/4)*AIF25_V2_512_BLOCK_BYTES;
            std::memcpy(w, scales.data(), 32); std::memcpy(n, scales.data(), 32);
            std::memcpy(w + 32 + size_t(c%5)*640, payload.data() + size_t(c)*640, 640);
            std::memcpy(n + 32 + size_t(c%4)*640, payload.data() + size_t(c)*640, 640);
        }
        require(ggml_ai4_slab_nbytes(GGML_TYPE_AIF25_U2, K) == wide.size() && ggml_ai4_slab_nbytes(T, K) == narrow.size(),
                "2.5 row640 and row512 slab sizes on a shared width");
        std::vector<float> a(size_t(16)*K, 1.0f), b(size_t(16)*K, 2.0f);
        require(ggml_ai4_dequantize_slab(GGML_TYPE_AIF25_U2, wide.data(), K, 0, a.data()), "2.5 row640 decode on a shared width");
        require(ggml_ai4_dequantize_slab(T, narrow.data(), K, 0, b.data()), "2.5 row512 decode on a shared width");
        require(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0,
                "2.5 row512 decodes bit for bit like row640 for equal chunks and scales");
    }
}

static void check_fractional_copy(ggml_type type, int64_t K, int64_t M) {
#if defined(__linux__)
    FILE * input = std::tmpfile(), * output = std::tmpfile();
    require(input && output, "fractional copy creates private temporary files");
    char input_path[64], output_path[64];
    std::snprintf(input_path, sizeof(input_path), "/proc/self/fd/%d", fileno(input));
    std::snprintf(output_path, sizeof(output_path), "/proc/self/fd/%d", fileno(output));
    ggml_init_params init = {ggml_tensor_overhead(), nullptr, true};
    auto * ctx = ggml_init(init);
    auto * tensor = ggml_new_tensor_2d(ctx, type, K, M);
    ggml_set_name(tensor, "blk.0.attn_q.weight");
    std::vector<uint8_t> bytes(ggml_nbytes(tensor), 0x5a);
    const size_t block_bytes = type == GGML_TYPE_AIF175_U2 ? AIF175_BLOCK_BYTES :
        type == GGML_TYPE_AIF25_U2_512 ? AIF25_V2_512_BLOCK_BYTES : AIF25_V2_BLOCK_BYTES;
    for (size_t block = 0; block < bytes.size(); block += block_bytes) {
        for (int row = 0; row < 16; ++row) {
            bytes[block+2*row] = 0; bytes[block+2*row+1] = 0x3c;
        }
    }
    tensor->data = bytes.data();
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_u32(meta, "general.file_type", LLAMA_FTYPE_ALL_F32);
    gguf_set_val_u32(meta, "llama.context_length", 512);
    gguf_set_val_u32(meta, "llama.embedding_length", K);
    gguf_set_val_u32(meta, "llama.block_count", 1);
    gguf_set_val_u32(meta, "llama.feed_forward_length", 11008);
    gguf_set_val_u32(meta, "llama.attention.head_count", 32);
    gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    if (type == GGML_TYPE_AIF175_U2) {
        gguf_set_val_str(meta, "ait.fractional.175.contract", AIF175_CONTRACT);
    }
    gguf_add_tensor(meta, tensor);
    require(gguf_write_to_file_ptr(meta, input, false) && std::fflush(input) == 0,
            "fractional copy writes finite interleaved fixture");
    auto params = llama_model_quantize_default_params();
    params.only_copy = true;
    params.nthread = 1;
    for (size_t limit : {size_t(1), size_t(1024*1024), size_t(2*1024*1024)}) {
        params.max_buf_size = limit;
        require(llama_model_quantize(input_path, output_path, &params) == 0,
                "fractional actual COPY accepts tiny and odd-row chunk limits");
        ggml_context * copied_ctx = nullptr;
        gguf_init_params read = {false, &copied_ctx};
        auto * copied = gguf_init_from_file(output_path, read);
        require(copied && copied_ctx && llama_ait_format_error(copied) == nullptr,
                "fractional actual COPY output parses with its versioned contract");
        auto * actual = ggml_get_tensor(copied_ctx, "blk.0.attn_q.weight");
        require(actual && actual->type == tensor->type && ggml_are_same_shape(actual, tensor) &&
                ggml_nbytes(actual) == bytes.size() && actual->data && std::memcmp(actual->data, bytes.data(), bytes.size()) == 0,
                "fractional actual COPY retains every stored scale and payload byte");
        gguf_free(copied); ggml_free(copied_ctx);
    }
    bytes[bytes.size()-block_bytes+31] = 0x7c;
    std::rewind(input);
    require(gguf_write_to_file_ptr(meta, input, false) && std::fflush(input) == 0,
            "fractional copy writes final-block nonfinite scale");
    require(llama_model_quantize(input_path, output_path, &params) != 0,
            "fractional actual COPY still rejects a nonfinite scale in its final chunk");
    gguf_free(meta); ggml_free(ctx);
    std::fclose(input); std::fclose(output);
#else
    (void) type; (void) K; (void) M;
#endif
}
static gguf_context * fixture(int type, int64_t K = 1024, int64_t M = 1024, int64_t n2 = 1, int64_t n3 = 1) {
    auto * meta = gguf_init_empty();
    if (type >= 0) {
        ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
        auto * ctx = ggml_init(params);
        auto * tensor = ggml_new_tensor_4d(ctx, (ggml_type) type, K, M, n2, n3);
        ggml_set_name(tensor, "test.weight");
        gguf_add_tensor(meta, tensor);
        ggml_free(ctx);
    }
    return meta;
}
static void stamp(gguf_context * meta) {
    gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 2);
    gguf_set_val_u32(meta, "ait.multiplier.110", 0xC3CCD12Du);
    gguf_set_val_u32(meta, "ait.multiplier.111", 0x87D8D12Du);
    gguf_set_val_u32(meta, "ait.multiplier.112", 0xAA73876Bu);
    gguf_set_val_u32(meta, "ait.multiplier.114", 0xAA73876Bu);
    gguf_set_val_u32(meta, "ait.multiplier.115", 0xC3CCD12Du);
    gguf_set_val_u32(meta, "ait.multiplier.116", 0x87D8D12Du);
    gguf_set_val_u32(meta, "ait.multiplier.117", 0xAA73876Bu);
    gguf_set_val_str(meta, AIT2P_HASH_KEY, "quadratic20-table1024-v1");
}
int main() {
    check_aif25_windows();
    check_aif175_predecessors();
    check_aif175_trace_bounds();
    check_aif175_errors();
    check_aif175_windows();
    check_fractional_copy(GGML_TYPE_AIF175_U2, 4096, 4096);
    check_fractional_copy(GGML_TYPE_AIF25_U2, 5120, 2048);
    check_fractional_copy(GGML_TYPE_AIF25_U2_512, 6144, 2048);
    check_aif25_u2_512();
    check_ai4_u2_512();
    const uint32_t multipliers[3] = {0xC3CCD12Du, 0x87D8D12Du, 0xAA73876Bu};
    const char * keys[3] = {"ait.multiplier.110", "ait.multiplier.111", "ait.multiplier.112"};
    for (int k = 2; k <= 4; ++k) {
        require(AIT_MUL_FOR_TYPE(108+k) == multipliers[k-2], "fixed type multiplier");
        for (uint32_t state = 0; state < 65536; ++state) {
            uint32_t value = (uint32_t) ((uint64_t) state * multipliers[k-2]);
            int sum = 0;
            for (int b = 0; b < 4; ++b) { sum += (value >> (8*b)) & 255; }
            require(ait_level(state, k) == ((sum+2)/4)-128, "independent byte-sum level");
            require(aik_level_c(state, ggml_ai4_trellis_mul((ggml_type) (113+k))) == ((sum+2)/4)-128, "independent unit byte-sum level");
        }
    }
    for (int type = 110; type <= 117; ++type) {
        auto * meta = fixture(type);
        require(llama_ait_format_error(meta) != nullptr, "unversioned rejected");
        gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 1);
        require(llama_ait_format_error(meta) != nullptr, "old version rejected");
        gguf_set_val_str(meta, AIT_FORMAT_VERSION_KEY, "2");
        require(llama_ait_format_error(meta) != nullptr, "string version rejected");
        gguf_set_val_u32(meta, AIT_FORMAT_VERSION_KEY, 2);
        require(llama_ait_format_error(meta) != nullptr, "missing decoder metadata rejected");
        stamp(meta);
        require(llama_ait_format_error(meta) == nullptr, "valid format accepted");
        if (type < 113) {
            gguf_set_val_u32(meta, keys[type-110], multipliers[(type-109)%3]);
            require(llama_ait_format_error(meta) != nullptr, "cross-tier multiplier rejected");
            gguf_set_val_str(meta, keys[type-110], "wrong type");
            require(llama_ait_format_error(meta) != nullptr, "string multiplier rejected");
        } else if (type >= 115) {
            const int k = type - 113;
            const auto t = (ggml_type) type;
            require(ggml_ai4_rate(t) == k && ggml_ai4_layout(t) == 1, "unit rate and layout");
            require(ggml_ait_rate(t) == 0, "unit is not tiled");
            require(ggml_ai4_trellis_mul(t) == multipliers[k-2], "unit own multiplier");
            require(AIT_MUL_FOR_TYPE(type) == multipliers[k-2], "unit metadata multiplier");
            require(ggml_blck_size(t) == 128 && ggml_row_size(t, 640) == 5*(2+16*k), "unit H128 row bytes");
            require(std::fabs(ggml_ai4_scale_mul(t) - (k == 2 ? 1.00f : k == 3 ? 1.05f : 1.09f)) < 1e-6f, "unit scale seed");
            char key[32];
            std::snprintf(key, sizeof(key), "ait.multiplier.%d", type);
            gguf_set_val_u32(meta, key, multipliers[(k-1)%3]);
            require(llama_ait_format_error(meta) != nullptr, "cross-tier unit multiplier rejected");
            gguf_set_val_str(meta, key, "wrong type");
            require(llama_ait_format_error(meta) != nullptr, "unit string multiplier rejected");
        } else if (type == 114) {
            require(ggml_ai4_rate((ggml_type) type) == 5, "AIT5_U rate");
            require(ggml_ai4_trellis_mul((ggml_type) type) == 0xAA73876Bu, "AIT5_U own multiplier");
            require(ggml_blck_size((ggml_type) type) == 128 && ggml_row_size((ggml_type) type, 1024) == 8*82, "AIT5_U row bytes");
            gguf_set_val_u32(meta, "ait.multiplier.114", 0);
            require(llama_ait_format_error(meta) != nullptr, "wrong AIT5_U multiplier rejected");
        } else {
            gguf_set_val_str(meta, AIT2P_HASH_KEY, "wrong table");
            require(llama_ait_format_error(meta) != nullptr, "wrong pair table rejected");
        }
        gguf_free(meta);
    }
    {
        // AIT2P_U: unit bytes, pair-table metadata, no multiplier
        const auto t = GGML_TYPE_AIT2P_U;
        require(ggml_ai4_rate(t) == 2 && ggml_ai4_layout(t) == 1 && ggml_ai4_code(t) == 1, "AIT2P_U rate, layout, code");
        require(ggml_ai4_code(GGML_TYPE_AIT2_U) == 0 && ggml_ai4_code(GGML_TYPE_AIT2P) == -1, "unit codes");
        require(ggml_ai4_trellis_mul(t) == 0 && ggml_ait_rate(t) == 0, "AIT2P_U has no multiplier, is not tiled");
        require(ggml_blck_size(t) == 128 && ggml_row_size(t, 640) == 5*34, "AIT2P_U row bytes");
        auto * meta = fixture((int) t);
        require(llama_ait_format_error(meta) != nullptr, "unversioned AIT2P_U rejected");
        stamp(meta);
        require(llama_ait_format_error(meta) == nullptr, "valid AIT2P_U accepted");
        gguf_set_val_str(meta, AIT2P_HASH_KEY, "wrong table");
        require(llama_ait_format_error(meta) != nullptr, "AIT2P_U wrong pair table rejected");
        gguf_free(meta);
        // independent decode of one random tile: bit-serial MSB-first stream, pair windows of 20 bits ending at 4p + 4
        uint32_t wd[8];
        uint32_t x = 0x12345679u;
        for (int rep = 0; rep < 64; ++rep) {
            for (uint32_t & w : wd) { x = x * 1664525u + 1013904223u; w = x ^ (x >> 15); }
            for (int n = 0; n < 128; ++n) {
                const int p = n / 2;
                uint32_t win = 0;
                for (int b = 4*p + 4 - 20; b < 4*p + 4; ++b) {
                    const int bb = (b + 256) % 256;
                    win = (win << 1) | ((wd[bb / 32] >> (31 - bb % 32)) & 1u);
                }
                const uint32_t e = ((win * (win + 1u)) >> 10) & 1023u;
                require(aik_level_n(wd, 2, 0, n) == ait2p_tab[2*e + n % 2], "independent AIT2P_U pair level");
            }
        }
    }
    for (int type : {-1, (int) GGML_TYPE_F16}) {
        auto * meta = fixture(type);
        require(llama_ait_format_error(meta) == nullptr, "non-AIT model unaffected");
        gguf_free(meta);
    }
    auto * fractional = fixture(GGML_TYPE_AIF25_U);
    require(ggml_row_size(GGML_TYPE_AIF25_U,128) == 42, "fractional exact 2.625-bpw bytes");
    require(ggml_ai4_rate(GGML_TYPE_AIF25_U) == 0, "fractional never enters integer-rate dispatch");
    require(llama_ait_format_error(fractional) != nullptr, "unversioned fractional bytes rejected");
    gguf_set_val_str(fractional, "ait.fractional.contract", "wrong");
    require(llama_ait_format_error(fractional) != nullptr, "wrong fractional contract rejected");
    gguf_set_val_str(fractional, "ait.fractional.contract", AIF25_CONTRACT);
    require(llama_ait_format_error(fractional) == nullptr, "fractional own-constant contract accepted");
    gguf_free(fractional);
    require(ggml_row_size(GGML_TYPE_AIF25_U2, 640) == 202, "fractional U2 exact 2.525-bpw bytes");
    require(ggml_ai4_rate(GGML_TYPE_AIF25_U2) == 0, "fractional U2 stays outside integer-rate kernels");
    require(ggml_ai4_slab_nbytes(GGML_TYPE_AIF25_U2, 128) == 0, "fractional U2 rejects partial blocks");
    {
        // Query routing only: unsupported callbacks must never be scheduled on CPU.
        ggml_backend_load_all();
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu != nullptr, "CPU backend available for support checks");
        auto * ctx = ggml_init({64*ggml_tensor_overhead(), nullptr, true});
        require(ctx != nullptr, "CPU support-check context");
        for (auto type : {GGML_TYPE_AIF25_U2, GGML_TYPE_AIF175_U2, GGML_TYPE_AIF25_U2_512}) {
            const int K = type == GGML_TYPE_AIF25_U2 ? 640 : 512;
            auto * packed = ggml_new_tensor_2d(ctx, type, K, 16);
            auto * plain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 16);
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 16);
            require(!ggml_backend_dev_supports_op(cpu, ggml_cast(ctx, plain, type)),
                    "CPU refuses cast into fractional U2");
            require(!ggml_backend_dev_supports_op(cpu, ggml_cpy(ctx, plain, packed)),
                    "CPU refuses copy into fractional U2");
            require(!ggml_backend_dev_supports_op(cpu, ggml_set_rows(ctx, packed, plain, ids)),
                    "CPU refuses fractional U2 row destination");
            require(!ggml_backend_dev_supports_op(cpu, ggml_cast(ctx, packed, GGML_TYPE_F32)),
                    "CPU refuses cast from fractional U2");
            require(!ggml_backend_dev_supports_op(cpu, ggml_mul_mat(ctx, packed, plain)),
                    "CPU refuses fractional U2 matrix product");
            require(ggml_backend_dev_supports_op(cpu, packed), "CPU accepts fractional U2 leaf metadata");
            auto * view = ggml_view_2d(ctx, packed, K, 16, packed->nb[1], 0);
            require(view->op == GGML_OP_VIEW, "CPU support check uses a real view operation");
            require(ggml_backend_dev_supports_op(cpu, view),
                    "CPU accepts fractional U2 view metadata");
            require(ggml_backend_dev_supports_op(cpu, ggml_reshape_2d(ctx, packed, K, 16)),
                    "CPU accepts fractional U2 reshape metadata");
            require(ggml_backend_dev_supports_op(cpu, ggml_cast(ctx, plain, GGML_TYPE_F16)),
                    "CPU still accepts ordinary casts");
            require(ggml_backend_dev_supports_op(cpu, ggml_set_rows(ctx, plain, plain, ids)),
                    "CPU still accepts ordinary row destinations");
        }
        ggml_free(ctx);
    }
    for (int K : {640, 1280, 1920, 5120}) {
        // Expand each stored row scale to five headers; the decoded matrix must stay exact,
        // including the H256 rotation that crosses the boundary between 640-column blocks.
        const size_t sb = (K/640)*AIF25_V2_BLOCK_BYTES;
        std::vector<uint8_t> packed(2*sb + 1);
        std::vector<uint8_t> expanded((K/128)*AIF25_UNIT_BYTES);
        std::vector<float> oracle(16*K);
        uint32_t state = 7300;
        for (auto & b : packed) { state = state*1664525u + 1013904223u; b = state >> 24; }
        uint8_t * src = packed.data() + 1 + sb;
        for (int block = 0; block < K/640; ++block) {
            uint8_t * p = src + block*AIF25_V2_BLOCK_BYTES;
            for (int r = 0; r < 16; ++r) {
                const uint16_t scale = r == 6 ? 0 : r == 7 ? 0x8000 : r == 8 ? 0xb800 :
                    r == 9 ? 0x0001 : r == 10 ? 0x03ff : 0x2800 + 16*block + r;
                p[2*r] = uint8_t(scale); p[2*r+1] = uint8_t(scale >> 8);
                for (int k = 0; k < 640; ++k) {
                    oracle[r*K + block*640 + k] = ai4_fp16_to_fp32(scale)*aif25_level_reference(p + 32 + (k/8)*40, r*8 + k%8);
                }
            }
            for (int u = 0; u < 5; ++u) {
                uint8_t * q = expanded.data() + (5*block + u)*AIF25_UNIT_BYTES;
                std::memcpy(q, p, 32);
                std::memcpy(q + 32, p + 32 + u*640, 640);
            }
        }
        const auto before = packed;
        require(ggml_validate_row_data(GGML_TYPE_AIF25_U2, src, sb), "2.5 validates finite scales in unaligned multi-block storage");
        require(!ggml_validate_row_data(GGML_TYPE_AIF25_U2, src, sb-202), "2.5 validation rejects a partial interleaved block");
        require(!ggml_validate_row_data(GGML_TYPE_AIF25_U2, nullptr, sb), "2.5 validation rejects missing block data");
        std::vector<float> a(16*K), b(16*K+2, 123.0f);
        const auto output_before = b;
        for (int edge = 0; edge < (K == 640 ? 1 : 2); ++edge) {
            const int block = edge ? K/640-1 : 0;
            for (int r = 0; r < 16; ++r) {
                uint8_t * scale = src + block*AIF25_V2_BLOCK_BYTES + 2*r;
                const uint8_t lo = scale[0], hi = scale[1];
                for (uint16_t bad : {0x7c00, 0xfc00, 0x7e00, 0xfc01}) {
                    scale[0] = uint8_t(bad); scale[1] = uint8_t(bad >> 8);
                    require(!ggml_validate_row_data(GGML_TYPE_AIF25_U2, src, sb), "2.5 validation rejects nonfinite scales at every row and edge block");
                    if (r == 0 || r == 15) {
                        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF25_U2, packed.data()+1, K, 1, b.data()+1),
                                "2.5 direct decode rejects nonfinite scales in the selected slab");
                        require(b == output_before, "2.5 rejected decode leaves all output and canaries unchanged");
                        require(scale[0] == uint8_t(bad) && scale[1] == uint8_t(bad >> 8), "2.5 rejected decode preserves invalid scale bytes");
                    }
                }
                scale[0] = lo; scale[1] = hi;
            }
        }
        require(packed == before && ggml_validate_row_data(GGML_TYPE_AIF25_U2, src, sb), "2.5 validation preserves payload and accepts restored finite scales");
        const int rb = ai4u_rot_size(K);
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < K; k += rb) { ai4_unrotate_n(oracle.data() + r*K + k, rb); }
        }
        aif25_dequantize_slab(expanded.data(), a.data(), K);
        require(ggml_ai4_slab_nbytes(GGML_TYPE_AIF25_U2, K) == sb, "fractional U2 slab bytes");
        require(ggml_ai4_dequantize_slab(GGML_TYPE_AIF25_U2, packed.data()+1, K, 1, b.data()+1),
                "fractional U2 public slab decoder");
        require(std::memcmp(a.data(), b.data()+1, a.size()*sizeof(float)) == 0,
                "fractional U2 matches expanded independent scale headers exactly");
        require(std::memcmp(oracle.data(), b.data()+1, oracle.size()*sizeof(float)) == 0,
                "2.5 decoded scales match canonical values independent of host byte order");
        require(b.front() == 123.0f && b.back() == 123.0f && packed == before, "2.5 valid decode preserves canaries and packed bytes");
    }
    {
        auto * meta = fixture(GGML_TYPE_AIF175_U2);
        require(llama_ait_format_error(meta) != nullptr, "1.75 missing contract rejected");
        gguf_set_val_u32(meta, "ait.fractional.175.contract", 1);
        require(llama_ait_format_error(meta) != nullptr, "1.75 mistyped contract rejected");
        gguf_set_val_str(meta, "ait.fractional.175.contract", "wrong");
        require(llama_ait_format_error(meta) != nullptr, "1.75 wrong contract rejected");
        gguf_set_val_str(meta, "ait.fractional.175.contract", AIF175_CONTRACT);
        require(llama_ait_format_error(meta) == nullptr, "1.75 versioned contract accepted");
        gguf_free(meta);
    }
    for (auto type : {GGML_TYPE_AIF25_U2, GGML_TYPE_AIF175_U2, GGML_TYPE_AIF25_U2_512}) {
        const int64_t columns = type == GGML_TYPE_AIF25_U2 ? 640 : 512;
        const int64_t widths[] = {columns, 2*columns, 3*columns, 8*columns, 10*columns, 12*columns, 34*columns};
        for (int64_t M : {1, 15, 17, 31, 1025}) {
            auto * meta = fixture(type, columns, M);
            if (type == GGML_TYPE_AIF175_U2) { gguf_set_val_str(meta, "ait.fractional.175.contract", AIF175_CONTRACT); }
            require(llama_ait_format_error(meta) != nullptr, "fractional U2 rejects partial stored row slabs");
            gguf_free(meta);
        }
        for (int64_t K : widths) {
            for (int M : {16, 32, 4096}) {
                auto * meta = fixture(type, K, M);
                if (type == GGML_TYPE_AIF175_U2) { gguf_set_val_str(meta, "ait.fractional.175.contract", AIF175_CONTRACT); }
                require(llama_ait_format_error(meta) == nullptr, "fractional U2 accepts complete stored row slabs");
                gguf_free(meta);
            }
        }
        const int64_t shapes[][4] = {
            {columns, 16, 2, 1}, {columns, 16, 1, 2},
            {(INT32_MAX/columns+1)*columns, 16, 1, 1}, {columns, int64_t(INT32_MAX)+1, 1, 1},
        };
        for (const auto & shape : shapes) {
            auto * meta = fixture(type, shape[0], shape[1], shape[2], shape[3]);
            if (type == GGML_TYPE_AIF175_U2) { gguf_set_val_str(meta, "ait.fractional.175.contract", AIF175_CONTRACT); }
            require(llama_ait_format_error(meta) != nullptr, "fractional U2 rejects batched or oversized stored matrices");
            gguf_free(meta);
        }
    }
    {
        auto * meta = fixture(GGML_TYPE_F16, 640, 17, 2);
        require(llama_ait_format_error(meta) == nullptr, "fractional shape rules leave ordinary batched tensors unchanged");
        gguf_free(meta);
    }
    require(ggml_row_size(GGML_TYPE_AIF175_U2, 512) == 114, "1.75 stored row bytes");
    require(ggml_ai4_rate(GGML_TYPE_AIF175_U2) == 0, "1.75 avoids integer-rate dispatch");
    for (int K : {512, 1024, 4096, 5120, 6144, 17408}) {
        const size_t sb = (K/512)*AIF175_BLOCK_BYTES;
        std::vector<uint8_t> packed(2*sb + 1);
        uint32_t state = 7300;
        for (auto & b : packed) { state = state*1664525u + 1013904223u; b = state >> 24; }
        uint8_t * src = packed.data() + 1 + sb;
        for (int block = 0; block < K/512; ++block) {
            // An interior zero block must not stop the next nonzero block.
            const bool zero_block = K > 512 && (block == K/512-1 || (K >= 4096 && block == 1));
            const bool full_block = K > 512 && block == 0;
            for (int r = 0; r < 16; ++r) {
                const uint16_t scale = zero_block ? (r%2 ? 0x8000 : 0) :
                    full_block ? 0x2800 + r :
                    r == 6 && block%2 == 0 ? 0 : r == 7 ? 0 : r == 8 ? 0x8000 : r == 9 ? 0xb800 :
                    r == 10 ? 0x0001 : r == 11 ? 0x03ff : 0x2800 + 16*block + r;
                src[block*AIF175_BLOCK_BYTES + 2*r] = uint8_t(scale);
                src[block*AIF175_BLOCK_BYTES + 2*r+1] = uint8_t(scale >> 8);
            }
        }
        const auto before = packed;
        require(ggml_validate_row_data(GGML_TYPE_AIF175_U2, src, sb),
                "1.75 validates finite scales in unaligned multi-block storage");
        require(!ggml_validate_row_data(GGML_TYPE_AIF175_U2, src, sb-114),
                "1.75 validation rejects a partial interleaved block");
        require(!ggml_validate_row_data(GGML_TYPE_AIF175_U2, nullptr, sb),
                "1.75 validation rejects missing block data");
        std::vector<float> invalid_output(16*K+2, 123.0f);
        const auto invalid_before = invalid_output;
        for (int edge = 0; edge < (K == 512 ? 1 : 2); ++edge) {
            const int block = edge ? K/512-1 : 0;
            for (int r = 0; r < 16; ++r) {
                uint8_t * scale = src + block*AIF175_BLOCK_BYTES + 2*r;
                const uint8_t lo = scale[0], hi = scale[1];
                for (uint16_t bad : {0x7c00, 0xfc00, 0x7e00, 0xfc01}) {
                    scale[0] = uint8_t(bad); scale[1] = uint8_t(bad >> 8);
                    require(!ggml_validate_row_data(GGML_TYPE_AIF175_U2, src, sb),
                            "1.75 validation rejects nonfinite scales at every row and edge block");
                    if (r == 0 || r == 15) {
                        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data()+1, K, 1, invalid_output.data()+1),
                                "1.75 direct decode rejects nonfinite scales in the selected slab");
                        require(invalid_output == invalid_before,
                                "1.75 rejected scale leaves all output and canaries unchanged");
                        require(scale[0] == uint8_t(bad) && scale[1] == uint8_t(bad >> 8),
                                "1.75 rejected decode preserves the invalid scale bytes");
                    }
                }
                scale[0] = lo; scale[1] = hi;
            }
        }
        require(packed == before && ggml_validate_row_data(GGML_TYPE_AIF175_U2, src, sb),
                "1.75 validation preserves payload and accepts restored finite scales");
        std::vector<float> expected(16*K), actual(16*K+2, 123.0f);
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < K; ++k) {
                const uint8_t * block = src + (k/512)*AIF175_BLOCK_BYTES;
                const uint8_t * tile = block + 32 + ((k%512)/128)*448 + ((k%128)/8)*28;
                const int n = 8*r + k%8, p = n/2;
                const int end = (p/2)*7 + (p%2 ? 7 : 3);
                uint32_t bits = 0;
                for (int j = end-16; j < end; ++j) {
                    const int bit = (j + 224)%224;
                    const int byte = bit/32*4+3-bit%32/8;
                    bits = (bits << 1) | ((tile[byte] >> (7-bit%8)) & 1);
                }
                const uint32_t C = n%2 ? uint32_t(uint64_t(AIF175_MUL)*AIF175_MUL) : AIF175_MUL;
                const uint32_t z = bits*C;
                int sum = 0;
                for (int j = 0; j < 4; ++j) { sum += (z >> (8*j)) & 255; }
                const int level = ((sum + 2) >> 2) - 128;
                require(aif175_level(tile, n) == level, "1.75 independent bit window and level");
                const uint16_t h = uint16_t(block[2*r]) | (uint16_t(block[2*r+1]) << 8);
                expected[r*K + k] = ai4_fp16_to_fp32(h)*level;
            }
            for (int k = 0; k < K; k += 256) { ai4_unrotate_n(expected.data() + r*K + k, 256); }
        }
        require(ggml_ai4_slab_nbytes(GGML_TYPE_AIF175_U2, K) == sb, "1.75 whole slab size");
        require(ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data()+1, K, 1, actual.data()+1),
                "1.75 unaligned later-slab public decode");
        for (int block = 0; block < K/512; ++block) {
            for (int r = 0; r < 16; ++r) {
                const uint8_t * scale = src + block*AIF175_BLOCK_BYTES + 2*r;
                const bool zero_scale = (scale[0] | ((scale[1] & 0x7f) << 8)) == 0;
                const size_t begin = r*K + block*512;
                if (zero_scale) {
                    for (int k = 0; k < 512; ++k) {
                        require(expected[begin+k] == 0.0f && actual[begin+k+1] == 0.0f && !std::signbit(actual[begin+k+1]),
                                "1.75 both zero scales produce canonical zeros with arbitrary payload");
                    }
                } else {
                    require(std::memcmp(expected.data()+begin, actual.data()+begin+1, 512*sizeof(float)) == 0,
                            "1.75 nonzero scale decode keeps exact independent results");
                }
            }
        }
        require(actual.front() == 123.0f && actual.back() == 123.0f && packed == before,
                "1.75 canaries and packed input unchanged");
        const auto unchanged = actual;
        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data(), K, -1, actual.data()),
                "1.75 rejects negative slab");
        require(actual == unchanged, "1.75 negative slab preserves output");
        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, nullptr, K, 0, actual.data()),
                "1.75 rejects null input");
        require(actual == unchanged, "1.75 null input preserves output");
        const int64_t overflow_slab = int64_t(SIZE_MAX/sb);
        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data(), K, overflow_slab, actual.data()),
                "1.75 rejects a slab whose start fits but end overflows");
        require(!ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data(), K, 0, nullptr),
                "1.75 rejects null output");
        require(actual == unchanged && packed == before, "1.75 rejected decode changes no input or output");
        require(actual.front() == 123.0f, "1.75 rejected decode writes nothing");
    }
    for (int64_t K : {int64_t(0), int64_t(-512), int64_t(128), int64_t(640), INT64_MAX}) {
        require(ggml_ai4_slab_nbytes(GGML_TYPE_AIF175_U2, K) == 0, "1.75 rejects partial or invalid widths");
    }
    for (int K : {512, 6144}) {
        std::vector<float> zero(16*K);
        std::vector<uint8_t> bytes(ggml_ai4_slab_nbytes(GGML_TYPE_AIF175_U2, K)+2, 0x5a);
        for (size_t i = 1; i < zero.size(); i += 2) { zero[i] = -0.0f; }
        const auto zero_before = zero;
        allocation_calls = release_calls = 0;
        require(aif175_quantize_slab_impl(zero.data(), K, bytes.data()+1, bytes.size()-2,
                deny_allocation, count_release), "1.75 zero slab needs no trellis allocation");
        require(allocation_calls == 0 && release_calls == 0, "1.75 zero slab skips all scratch allocation and cleanup");
        require(std::memcmp(zero.data(), zero_before.data(), zero.size()*sizeof(float)) == 0,
                "1.75 zero slab preserves source signed zeros");
        require(bytes.front() == 0x5a && bytes.back() == 0x5a, "1.75 zero fast path output canaries");
        for (size_t i = 1; i+1 < bytes.size(); ++i) { require(bytes[i] == 0, "1.75 zero fast path canonical bytes"); }
        std::fill(bytes.begin()+1, bytes.end()-1, 0x5a);
        require(ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-2), "1.75 native zero encode");
        require(bytes.front() == 0x5a && bytes.back() == 0x5a, "1.75 encoder output canaries");
        for (size_t i = 1; i+1 < bytes.size(); ++i) { require(bytes[i] == 0, "1.75 canonical zero payload"); }
        std::fill(bytes.begin()+1, bytes.end()-1, 0xa7);
        const auto before = bytes;
        zero.back() = 1.0f;
        const auto trailing_before = zero;
        allocation_calls = release_calls = 0;
        require(!aif175_quantize_slab_impl(zero.data(), K, bytes.data()+1, bytes.size()-2,
                deny_allocation, count_release), "1.75 trailing finite nonzero enters ordinary encoder");
        require(allocation_calls == 1 && release_calls == 1 && bytes == before,
                "1.75 trailing nonzero cannot take the zero fast path");
        require(std::memcmp(zero.data(), trailing_before.data(), zero.size()*sizeof(float)) == 0,
                "1.75 trailing nonzero failure keeps source bytes");
        zero.back() = 0.0f;
        for (size_t index : {size_t(0), zero.size()/2, zero.size()-1}) {
            for (float invalid : {INFINITY, -INFINITY, NAN}) {
                // A nonzero first block must not hide a bad weight in a later block.
                zero[0] = 1.0f;
                zero[index] = invalid;
                const auto invalid_before = zero;
                allocation_calls = release_calls = 0;
                require(!aif175_quantize_slab_impl(zero.data(), K, bytes.data()+1, bytes.size()-2,
                        deny_allocation, count_release), "1.75 rejects nonfinite weights before trellis allocation");
                require(allocation_calls == 0 && release_calls == 0, "1.75 invalid input needs no scratch or cleanup");
                require(bytes == before && std::memcmp(zero.data(), invalid_before.data(), zero.size()*sizeof(float)) == 0,
                        "1.75 nonfinite preflight preserves source and output bytes");
                zero[index] = 0.0f;
                zero[0] = 0.0f;
            }
        }
        zero[3] = INFINITY;
        require(!ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-2), "1.75 rejects nonfinite encode");
        require(bytes == before, "1.75 infinite encode leaves bytes unchanged");
        zero[3] = NAN;
        require(!ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-2), "1.75 rejects NaN encode");
        require(bytes == before, "1.75 NaN encode leaves bytes unchanged");
        zero[3] = 0;
        require(!ggml_aif175_quantize_slab(nullptr, K, bytes.data()+1, bytes.size()-2), "1.75 rejects null encode input");
        require(!ggml_aif175_quantize_slab(zero.data(), K, nullptr, bytes.size()-2), "1.75 rejects null encode output");
        require(!ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-3), "1.75 rejects short encode output");
        require(!ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-1), "1.75 rejects long encode output");
        require(!ggml_aif175_quantize_slab(zero.data(), K-1, bytes.data()+1, bytes.size()-2), "1.75 rejects partial encode width");
        for (int64_t invalid : {int64_t(0), int64_t(-512), INT64_MAX}) {
            require(!ggml_aif175_quantize_slab(zero.data(), invalid, bytes.data()+1, bytes.size()-2), "1.75 rejects invalid encode dimensions");
        }
        require(bytes == before, "1.75 failed encode leaves all bytes unchanged");
        std::fill(zero.begin(), zero.end(), 1e30f);
        require(!ggml_aif175_quantize_slab(zero.data(), K, bytes.data()+1, bytes.size()-2), "1.75 rejects finite FP16 scale overflow");
        require(bytes == before, "1.75 scale overflow leaves bytes unchanged");
    }
    {
        constexpr int K = 1024;
        std::vector<float> src(16*K), decoded(16*K);
        uint32_t rng = 7300;
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < K; ++k) {
                rng = rng*1664525u + 1013904223u;
                const bool zero_row = r == 7 || (r == 0 && k < 512) || (r == 15 && k >= 512);
                src[r*K+k] = zero_row ? (k%2 ? -0.0f : 0.0f) : (int(rng >> 16)-32768)/32768.0f * (r+1) * (k < 512 ? 0.1f : 0.3f);
            }
        }
        const auto before = src;
        std::vector<uint8_t> packed(ggml_ai4_slab_nbytes(GGML_TYPE_AIF175_U2, K)+2, 0x5a);
        require(ggml_aif175_quantize_slab(src.data(), K, packed.data()+1, packed.size()-2), "1.75 native trellis encode");
        require(std::memcmp(src.data(), before.data(), src.size()*sizeof(float)) == 0 && packed.front() == 0x5a && packed.back() == 0x5a,
                "1.75 encode source signed zeros and canaries unchanged");
        for (int block = 0; block < K/512; ++block) {
            for (int r : {0, 7, 15}) {
                if (r == 7 || (r == 0 && block == 0) || (r == 15 && block == 1)) {
                    const size_t scale = 1 + block*1824 + 2*r;
                    require(packed[scale] == 0 && packed[scale+1] == 0, "1.75 zero row keeps canonical zero scale in each block");
                }
            }
            const int r = block == 0 ? 15 : 0;
            const size_t scale = 1 + block*1824 + 2*r;
            require(packed[scale] != 0 || packed[scale+1] != 0, "1.75 opposite block keeps nonzero row scale");
        }
        require(ggml_ai4_dequantize_slab(GGML_TYPE_AIF175_U2, packed.data()+1, K, 0, decoded.data()), "1.75 encoded native decode");
        // Independent stream reader and dense inverse H256, separate from native decoder helpers.
        const uint32_t signs[8] = {0x9E3779B9, 0x7F4A7C15, 0xF39CC060, 0x5CED2A1B,
                                    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A};
        double energy = 0, error = 0;
        for (int r = 0; r < 16; ++r) {
            for (int base = 0; base < K; base += 256) {
                double rotated[256];
                for (int k = 0; k < 256; ++k) {
                    const int col = base+k, pair = (8*r+col%8)/2;
                    const uint8_t * block = packed.data()+1+(col/512)*1824;
                    const uint8_t * tile = block+32+(col%512/128)*448+(col%128/8)*28;
                    const int end = 7*(pair/2)+(pair%2 ? 7 : 3);
                    uint32_t state = 0;
                    for (int b = end-16; b < end; ++b) {
                        const int bit = (b+224)%224, offset = bit/32*4+3-bit%32/8;
                        state = (state << 1) | ((tile[offset] >> (7-bit%8)) & 1);
                    }
                    const uint32_t c = col%2 ? uint32_t(uint64_t(0x87D8D12Du)*0x87D8D12Du) : 0x87D8D12Du;
                    const uint32_t z = state*c;
                    const int level = (int((z & 255)+((z >> 8) & 255)+((z >> 16) & 255)+(z >> 24))+2)/4-128;
                    const uint16_t h = uint16_t(block[2*r]) | (uint16_t(block[2*r+1]) << 8);
                    rotated[k] = double(ai4_fp16_to_fp32(h))*level;
                }
                for (int k = 0; k < 256; ++k) {
                    double sum = 0;
                    for (int j = 0; j < 256; ++j) {
                        unsigned parity = unsigned(k & j); parity ^= parity >> 4; parity ^= parity >> 2; parity ^= parity >> 1;
                        sum += parity & 1 ? -rotated[j] : rotated[j];
                    }
                    const double value = sum/16*((signs[k/32] >> (k%32)) & 1 ? -1 : 1);
                    require(std::abs(value-decoded[r*K+base+k]) < 2e-5*(1+std::abs(value)), "1.75 encoded independent original-basis oracle");
                    const double d = value-src[r*K+base+k]; error += d*d; energy += double(src[r*K+base+k])*src[r*K+base+k];
                    if (r == 7 || (r == 0 && base < 512) || (r == 15 && base >= 512)) {
                        require(value == 0, "1.75 native encoder preserves zero row");
                    }
                }
            }
        }
        require(error/energy < 0.5, "1.75 native encode nontrivial distortion bound");
        // The first nonzero block finishes its trellis before the second block's scale overflows.
        for (int r = 0; r < 16; ++r) {
            std::fill(src.begin()+r*K+512, src.begin()+(r+1)*K, 1e30f);
        }
        std::fill(packed.begin()+1, packed.end()-1, 0xa7);
        const auto failed_before = packed;
        const auto source_before = src;
        allocation_calls = release_calls = live_allocations = fail_allocation_at = 0;
        require(!aif175_quantize_slab_impl(src.data(), K, packed.data()+1, packed.size()-2,
                fail_allocation, release_allocation), "1.75 rejects later-block scale overflow");
        require(allocation_calls == 1 && release_calls == 1 && live_allocations == 0,
                "1.75 late failure releases its single owned buffer");
        require(packed == failed_before && src == source_before, "1.75 late failure preserves every input and output byte");
    }
    for (auto type : {GGML_TYPE_AIF25_U2, GGML_TYPE_AIF175_U2, GGML_TYPE_AIF25_U2_512}) {
        const int64_t width = type == GGML_TYPE_AIF25_U2 ? 640 : 512;
        const size_t sb = ggml_ai4_slab_nbytes(type, width);
        const int64_t large_width = int64_t(SIZE_MAX/(16*sizeof(float))/width + 1)*width;
        const int64_t packed_overflow = INT64_MAX - INT64_MAX%width;
        std::vector<uint8_t> packed(8, 0xa5);
        std::vector<float> output(8, 123.0f);
        const auto packed_before = packed;
        const auto output_before = output;
        require(sb > 0, "fractional U2 valid slab size");
        require(ggml_ai4_slab_nbytes(type, packed_overflow) == 0, "fractional U2 packed size overflow rejected");
        require(ggml_ai4_slab_nbytes(type, large_width) > 0, "fractional U2 output overflow has valid packed size");
        if (type == GGML_TYPE_AIF175_U2) {
            const auto source_before = output;
            allocation_calls = release_calls = 0;
            require(!aif175_quantize_slab_impl(output.data(), large_width, packed.data(),
                    ggml_ai4_slab_nbytes(type, large_width), deny_allocation, count_release),
                    "1.75 instrumented encoder rejects input span overflow");
            require(allocation_calls == 0 && release_calls == 0,
                    "1.75 input span guard runs before every allocation");
            require(output == source_before && packed == packed_before,
                    "1.75 instrumented span rejection preserves buffers");
            require(!ggml_aif175_quantize_slab(output.data(), large_width, packed.data(),
                    ggml_ai4_slab_nbytes(type, large_width)),
                    "1.75 encoder rejects input span overflow before allocation or read");
            require(output == source_before && packed == packed_before,
                    "1.75 encoder input overflow preserves buffers");
            const std::vector<float> nonzero(16*width, 123.0f);
            require(!aif175_quantize_slab_impl(nonzero.data(), width, packed.data(), sb,
                    deny_allocation, count_release), "1.75 denied allocation returns failure");
            require(allocation_calls == 1 && release_calls == 1,
                    "1.75 valid width reaches the instrumented allocator and cleanup");
            require(output == source_before && packed == packed_before,
                    "1.75 denied allocation preserves buffers");
            for (float value : nonzero) { require(value == 123.0f, "1.75 denied allocation preserves nonzero source"); }
            std::vector<uint8_t> failed_bytes(sb+2, 0xa7);
            const auto failed_before = failed_bytes;
            fail_allocation_at = 1;
            allocation_calls = release_calls = live_allocations = 0;
            require(!aif175_quantize_slab_impl(nonzero.data(), width, failed_bytes.data()+1, sb,
                    fail_allocation, release_allocation), "1.75 scratch allocation failure returns false");
            require(allocation_calls == 1 && release_calls == 1 && live_allocations == 0,
                    "1.75 allocation failure stops requests and leaves no live buffer");
            require(failed_bytes == failed_before, "1.75 allocation failure preserves output and canaries");
            for (float value : nonzero) { require(value == 123.0f, "1.75 allocation failure preserves source"); }
        }
        require(!ggml_ai4_dequantize_slab(type, packed.data(), large_width, 0, output.data()),
                "fractional U2 output span overflow rejected before memory access");
        require(output == output_before && packed == packed_before, "fractional U2 output overflow preserves buffers");
        require(!ggml_ai4_dequantize_slab(type, packed.data(), width, int64_t(SIZE_MAX/sb), output.data()),
                "fractional U2 input span overflow rejected before memory access");
        require(output == output_before && packed == packed_before, "fractional U2 input overflow preserves buffers");
        for (int64_t K : {int64_t(0), int64_t(-1), width - 1, packed_overflow}) {
            require(!ggml_ai4_dequantize_slab(type, packed.data(), K, 0, output.data()),
                    "fractional U2 invalid width rejected before memory access");
            require(output == output_before && packed == packed_before, "fractional U2 invalid width preserves buffers");
        }
    }
    std::printf("PASS %d checks\n", checks);
    return 0;
}
