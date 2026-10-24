// asserts must stay live in Release builds: undefine NDEBUG before <cassert>
#undef NDEBUG

#include "speculative-adaptive.h"
#include "speculative.h"

#include <cassert>
#include <cstdio>

static void test_reset(void) {
    common_speculative_adaptive ctrl;

    // cold start at the floor max(1, n_min_adaptive)
    ctrl.reset(8, 1);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_climb == 0);
    assert(ctrl.n_drop == 0);

    // the default adaptive floor of 3 starts the controller at depth 3
    ctrl.reset(8, 3);
    assert(ctrl.n_cur == 3);

    // the ceiling clamps the cold start to n_max
    ctrl.reset(1, 3);
    assert(ctrl.n_cur == 1);
}

static void test_climb(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1); // ceiling 8, cold start at the floor 1

    // depth 1 climbs after 2 consecutive full accepts
    ctrl.update(1, 1, 8, 1);
    assert(ctrl.n_cur == 1);
    ctrl.update(1, 1, 8, 1);
    assert(ctrl.n_cur == 2);

    // a miss resets the climb streak
    ctrl.update(2, 1, 8, 1); // near miss
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_climb == 0);

    // depth 2 climbs after 4 consecutive full accepts
    for (int i = 0; i < 3; ++i) {
        ctrl.update(2, 2, 8, 1);
        assert(ctrl.n_cur == 2);
    }
    ctrl.update(2, 2, 8, 1);
    assert(ctrl.n_cur == 3);

    // depth 3 is the barrier: 6 consecutive full accepts to reach depth 4
    for (int i = 0; i < 5; ++i) {
        ctrl.update(3, 3, 8, 1);
        assert(ctrl.n_cur == 3);
    }
    ctrl.update(3, 3, 8, 1);
    assert(ctrl.n_cur == 4);

    // a full accept of a draft truncated below the depth (e.g. clamped by the
    // server context bound) counts as a full accept, not as a miss
    ctrl.update(3, 3, 8, 1); // depth 4, only 3 tokens drafted, all accepted
    assert(ctrl.n_climb == 1);
    assert(ctrl.n_drop == 0);

    // depth 4-5 need 5 consecutive full accepts
    for (int i = 0; i < 4; ++i) {
        ctrl.update(4, 4, 8, 1);
    }
    assert(ctrl.n_cur == 5);

    // depth 5 needs 4 consecutive full accepts
    for (int i = 0; i < 3; ++i) {
        ctrl.update(5, 5, 8, 1);
        assert(ctrl.n_cur == 5);
    }
    ctrl.update(5, 5, 8, 1);
    assert(ctrl.n_cur == 6);

    // depth 6 needs 3 consecutive full accepts
    for (int i = 0; i < 2; ++i) {
        ctrl.update(6, 6, 8, 1);
        assert(ctrl.n_cur == 6);
    }
    ctrl.update(6, 6, 8, 1);
    assert(ctrl.n_cur == 7);

    // depth 7+ needs 2 consecutive full accepts
    ctrl.update(7, 7, 8, 1);
    assert(ctrl.n_cur == 7);
    ctrl.update(7, 7, 8, 1);
    assert(ctrl.n_cur == 8);

    // the ceiling blocks further climbs
    for (int i = 0; i < 8; ++i) {
        ctrl.update(8, 8, 8, 1);
    }
    assert(ctrl.n_cur == 8);

    // no feedback for a zero-length draft
    ctrl.update(0, 0, 8, 1);
    assert(ctrl.n_cur == 8);
    assert(ctrl.n_climb == 0);
}

static void test_drop(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1); // cold start at the floor

    // at the floor no pressure accumulates at all
    for (int i = 0; i < 100; ++i) {
        ctrl.update(1, 0, 8, 1);
    }
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_drop == 0);

    // climb to depth 3 (2 + 4 full accepts)
    for (int i = 0; i < 2; ++i) {
        ctrl.update(1, 1, 8, 1);
    }
    for (int i = 0; i < 4; ++i) {
        ctrl.update(2, 2, 8, 1);
    }
    assert(ctrl.n_cur == 3);

    // at depth 3 the drop budget is floored at 20: a total miss adds 3, so
    // 7 misses drop one step
    for (int i = 0; i < 6; ++i) {
        ctrl.update(3, 0, 8, 1);
        assert(ctrl.n_cur == 3);
    }
    assert(ctrl.n_drop == 18);
    ctrl.update(3, 0, 8, 1);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_drop == 0);

    // at depth 2 the budget is floored at 20: a near miss adds 1, so 20 near
    // misses drop one step (the depth-1 collapse needs real sustained failure)
    for (int i = 0; i < 19; ++i) {
        ctrl.update(2, 1, 8, 1);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_drop == 19);
    ctrl.update(2, 1, 8, 1);
    assert(ctrl.n_cur == 1);

    // back at the floor, misses no longer accumulate pressure
    for (int i = 0; i < 100; ++i) {
        ctrl.update(1, 0, 8, 1);
    }
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_drop == 0);

    // deep depths hold a little longer: at depth 5 the budget is 25, a total
    // miss adds 5, so 5 misses drop one step
    ctrl.reset(8, 1);
    ctrl.n_cur = 5; // simulate a controller that already climbed to 5
    for (int i = 0; i < 4; ++i) {
        ctrl.update(5, 0, 8, 1);
        assert(ctrl.n_cur == 5);
    }
    assert(ctrl.n_drop == 20);
    ctrl.update(5, 0, 8, 1); // 20 + 5 = 25 -> drop
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_drop == 0);
}

static void test_full_accept_resets_pressure(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1);
    ctrl.n_cur = 3;

    // accumulate pressure, then a full accept wipes it out
    for (int i = 0; i < 5; ++i) {
        ctrl.update(3, 1, 8, 1); // near miss: +2 pressure at depth 3
    }
    assert(ctrl.n_drop == 10);
    ctrl.update(3, 3, 8, 1);
    assert(ctrl.n_drop == 0);

    // the miss pressure uses the drafted count, not the depth: a truncated
    // draft (2 tokens at depth 3) with 1 accepted adds 1, not 2
    ctrl.update(2, 1, 8, 1);
    assert(ctrl.n_drop == 1);
}

static void test_floor(void) {
    common_speculative_adaptive ctrl;

    // with the floor at 2 the depth never drops below 2, no matter how bad
    // the content gets
    ctrl.reset(8, 2);
    for (int i = 0; i < 1000; ++i) {
        ctrl.update(2, 0, 8, 2);
    }
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_drop == 0);

    // climbs still work from the floor
    for (int i = 0; i < 4; ++i) {
        ctrl.update(2, 2, 8, 2);
    }
    assert(ctrl.n_cur == 3);

    // and drops stop at the floor, not below it
    for (int i = 0; i < 100; ++i) {
        ctrl.update(3, 0, 8, 2);
    }
    assert(ctrl.n_cur == 2);
}

static void test_row_budget(void) {
    auto plan = [](std::vector<int32_t> wants, std::vector<int32_t> retained, int cap, size_t first) {
        return common_speculative_row_budget(wants, retained, cap, first);
    };
    assert(plan({4, 4}, {-1, -1}, 6, 0) == std::vector<int32_t>({2, 2}));
    assert(plan({4, 4}, {-1, -1}, 8, 0) == std::vector<int32_t>({3, 3}));
    assert(plan({4, 4}, {-1, -1}, 5, 0) == std::vector<int32_t>({2, 1}));
    assert(plan({4, 4}, {-1, -1}, 5, 1) == std::vector<int32_t>({1, 2}));
    assert(plan({0, 4}, {-1, -1}, 6, 0) == std::vector<int32_t>({0, 4}));
    assert(plan({4, 4}, {4, -1}, 6, 0) == std::vector<int32_t>({4, 0}));
    assert(plan({4, 4}, {4, 4}, 6, 0) == std::vector<int32_t>({4, -1}));
    assert(plan({4, 4}, {4, 4}, 6, 1) == std::vector<int32_t>({-1, 4}));
    assert(plan({4, 4, 4}, {-1, -1, -1}, 2, 1) == std::vector<int32_t>({-1, 0, 0}));
    assert(plan({8}, {-1}, 6, 0) == std::vector<int32_t>({8}));
    assert(plan({8, 8}, {8, 8}, 6, 1) == std::vector<int32_t>({-1, 8}));
    assert(plan({8, 4}, {8, -1}, 6, 0) == std::vector<int32_t>({8, -1}));
    assert(plan({8, 4}, {8, -1}, 6, 1) == std::vector<int32_t>({-1, 4}));
    assert(plan({}, {}, 6, 0).empty());
    for (int cap = 1; cap <= 8; ++cap) {
        for (size_t first = 0; first < 4; ++first) {
            const auto budget = plan({4, 3, 0, 2}, {-1, -1, -1, -1}, cap, first);
            int rows = 0;
            for (auto n : budget) { if (n >= 0) { rows += n + 1; } }
            assert(rows <= cap && rows > 0);
        }
    }
    assert(common_speculative_row_budget({4, 4}, {-1, -1}, 6, 0, {{.99, .99, .99, .99}, {.1, .1, .1, .1}}) == std::vector<int32_t>({4, 0}));
    assert(common_speculative_row_budget({4, 4}, {-1, -1}, 6, 0, {{.1}, {.99, .99, .99, .99}}) == std::vector<int32_t>({0, 4}));
    assert(common_speculative_row_budget({4, 4}, {-1, -1}, 8, 0, {{0}, {0}}) == std::vector<int32_t>({0, 0}));
    for (int g = 1; g <= 3; ++g) {
        for (int a = 0; a <= 4; ++a) {
            for (int b = 0; b <= 4; ++b) {
                std::vector<int32_t> wishes(g, a), retained(g, -1);
                wishes.back() = b;
                for (int first = 0; first < g; ++first) {
                    for (double p : {0.0, 0.5, 1.0}) {
                        const auto rows = common_speculative_row_budget(wishes, retained, 8, first, std::vector<std::vector<double>>(g, std::vector<double>(4, p)));
                        int total = 0;
                        for (int i = 0; i < g; ++i) {
                            assert(rows[i] >= 0 && rows[i] <= wishes[i]);
                            total += 1 + rows[i];
                        }
                        assert(total <= 8);
                        if (g == 1) { assert(rows == wishes); }
                    }
                }
            }
        }
    }
    printf("test-speculative-adaptive: row budget OK\n");
}

int main(void) {
    test_row_budget();
    test_reset();
    test_climb();
    test_drop();
    test_full_accept_resets_pressure();
    test_floor();

    printf("test-speculative-adaptive: all tests OK\n\n");

    return 0;
}
