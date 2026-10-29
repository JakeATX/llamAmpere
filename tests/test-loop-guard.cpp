#include "server-loop-guard.h"
#include "server-schema.h"
#include "server-task.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); std::exit(1); } } while (0)

static void periodic(int p, int reps, int span, int stop_at, int group) {
    server_loop_guard guard;
    int n = 0;
    bool stopped = false;
    while (!stopped && n < stop_at + group) {
        // Accepted drafts and the target token go through accept in output order.
        for (int j = 0; j < group; ++j) {
            stopped = guard.accept(n % p, reps, span);
            ++n;
            if (stopped) { break; }
            CHECK(n < stop_at);
        }
    }
    CHECK(n == stop_at);
    CHECK(guard.period == p);
    CHECK(guard.repeats == double(stop_at)/p);
}

int main() {
    for (int p : {1, 2, 6, 18, 64}) {
        int stop_at = std::min(100*p, std::max(400, 4*p));
        for (int group : {1, 3, 8}) {
            periodic(p, 100, 400, stop_at, group);
        }
        periodic(p, 100, INT_MAX, 100*p, 5);
    }
    periodic(1, 4096, 12802, 4096, 8);
    periodic(18, 4096, 12802, 12802, 8);
    server_loop_guard guard;
    for (int i = 0; i < 198; ++i) { CHECK(!guard.accept(i % 2, 100, 400)); }
    CHECK(guard.accept(0, 100, 400) == false);
    CHECK(guard.accept(1, 100, 400));
    guard = {};
    for (int i = 0; i < 99; ++i) { CHECK(!guard.accept(7, 100, 400)); }
    CHECK(!guard.accept(8, 100, 400)); // mismatch resets the stretch
    for (int i = 0; i < 99; ++i) { CHECK(!guard.accept(7, 100, 400)); }
    guard = {};
    for (int i = 0; i < 10000; ++i) { CHECK(!guard.accept(7, 0, 1)); }
    for (int i = 0; i < 10000; ++i) { CHECK(!guard.accept(i, 100, 400)); }
    server_loop_guard other;
    guard = {};
    for (int i = 0; i < 99; ++i) {
        CHECK(!guard.accept(7, 100, 400));
        CHECK(!other.accept(i, 100, 400));
    }
    CHECK(guard.accept(7, 100, 400));
    CHECK(!other.accept(7, 100, 400));

    common_params base;
    base.loop_guard_reps = 123;
    const auto params = server_schema::eval_llama_cmpl_schema(nullptr, base, {}, {{"loop_guard", false}, {"loop_guard_reps", 0}});
    CHECK(!params.loop_guard && params.loop_guard_reps == 0);
    CHECK(server_schema::eval_llama_cmpl_schema(nullptr, base, {}, {}).loop_guard_reps == 123);
    bool rejected = false;
    try { server_schema::eval_llama_cmpl_schema(nullptr, base, {}, {{"loop_guard_reps", -1}}); }
    catch (const std::exception &) { rejected = true; }
    CHECK(rejected);

    for (auto type : {TASK_RESPONSE_TYPE_NONE, TASK_RESPONSE_TYPE_OAI_CMPL, TASK_RESPONSE_TYPE_OAI_CHAT}) {
        for (bool stream : {false, true}) {
            server_task_result_cmpl_final res{};
            res.res_type = type;
            res.stream = stream;
            res.include_usage = stream;
            res.is_updated = true;
            res.stop = STOP_TYPE_LOOP;
            res.stats.stopped_loop = true;
            res.stats.loop_period = 18;
            res.stats.loop_repeats = 400.0/18;
            auto data = res.to_json();
            auto last = data.is_array() ? data.back() : data;
            CHECK(last.at("stopped_loop") == true);
            CHECK(last.at("loop_period") == 18);
            CHECK(last.at("loop_repeats") == 400.0/18);
            CHECK(res.stats.to_json().at("stopped_loop") == true);
            if (type == TASK_RESPONSE_TYPE_NONE) {
                CHECK(last.at("finish_reason") == "stop");
                CHECK(last.at("stop_type") == "loop");
            } else {
                const auto & finished = data.is_array() ? data.at(data.size() - 2) : data;
                CHECK(finished.at("choices").at(0).at("finish_reason") == "stop");
            }
        }
    }
    std::puts("loop guard tests passed");
}
