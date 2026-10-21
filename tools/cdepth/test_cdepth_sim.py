import contextlib
from collections import deque
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.cdepth import cdepth_sim as sim
from tools.cdepth.check_conflog import check, function_body


ROOT = Path(__file__).resolve().parents[2]


def make_round(i, accepted, depth=6, stream=("fixture", 0, 0, 0), n_past=None):
    return sim.Round(stream, i * 7 if n_past is None else n_past, depth, depth, accepted,
                     [0.7] * depth, [0.2] * depth, [0.8] * (depth + 1), 2 + depth)


def fixture_model(depth=6):
    return {"version": 1, "window": 16, "positions": [
        {"k": k + 1, "coef": [math.log(0.7 / 0.3), 0, 0, 0]} for k in range(depth)],
        "cost": {"a": 2.0, "b": 1.0, "identified": True}}


def serialize(row, request=1, slot=0):
    return dict(slot=slot, request=request, n_past=row.n_past, depth=row.depth,
                drafted=row.drafted, accepted=row.accepted, draft_p1=row.draft_p1,
                draft_p2=row.draft_p2, target_p1=row.target_p1, t_round_ms=row.ms)


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1729)
        cls.coef = np.array([-1.0, 2.4, 1.9, 1.1])
        history = deque(maxlen=16)
        cls.rounds = []
        past = 100000
        for i in range(50000):
            mean = float(np.mean(history)) if history else 0.5
            p1 = rng.uniform(0.3, 0.95, 6)
            p2 = np.minimum(p1, 1 - p1) * rng.uniform(0.05, 0.95, 6)
            row = make_round(i, 0, n_past=past)
            row.draft_p1, row.draft_p2 = p1.tolist(), p2.tolist()
            for k in range(6):
                p = float(sim.sigmoid(np.asarray(sim.features(row, mean, k)) @ cls.coef))
                if rng.random() >= p:
                    break
                row.accepted += 1
            # Short regimes make the generating 16-token window distinguishable from 8 and 32.
            level = 0.12 if (i // 5) % 2 else 0.88
            row.target_p1 = np.clip(level + rng.normal(0, 0.03, 7), 0, 1).tolist()
            history.extend(row.target_p1[:row.accepted + 1])
            row.ms = 2.0 + 0.7 * row.drafted
            past += row.accepted + 1
            cls.rounds.append(row)
        cls.model = sim.fit(cls.rounds)

    def test_fit_recovers_known_logistic_law(self):
        self.assertEqual(self.model["window"], 16)
        for position in self.model["positions"]:
            np.testing.assert_allclose(position["coef"], self.coef, atol=0.55, rtol=0)

    def test_conditional_risk_set(self):
        rounds = [make_round(0, 0), make_round(1, 1), make_round(2, 6)]
        x, y = sim.conditional_data(rounds, np.ones(3) * 0.5, range(3), 1)
        self.assertEqual(x.shape, (2, 4))
        self.assertEqual(y.tolist(), [0, 1])
        _, y = sim.conditional_data(rounds, np.ones(3), range(3), 5)
        self.assertEqual(y.tolist(), [1])

    def test_cost_fit_and_unidentified_fixed_depth(self):
        rounds = [make_round(i, 1, depth=d) for i, d in enumerate([2, 3, 4, 5, 6] * 3)]
        for row in rounds:
            row.ms = 1.5 + 0.75 * row.drafted
        cost = sim.fit_cost(rounds)
        self.assertTrue(cost["identified"])
        self.assertAlmostEqual(cost["a"], 1.5)
        self.assertAlmostEqual(cost["b"], 0.75)
        cost = sim.fit_cost(self.rounds)
        self.assertFalse(cost["identified"])
        self.assertEqual(cost["b"], 0)
        self.assertAlmostEqual(cost["a"], 6.2)

    def test_degenerate_labels_and_large_logits_are_finite(self):
        x = np.tile([1, 0.5, 0.7, 0.2], (20, 1))
        for labels in (np.zeros(20), np.ones(20)):
            coef = sim.fit_logistic(x, labels)
            self.assertTrue(np.isfinite(coef).all())
            self.assertLess(sim.log_loss(x, labels, coef), 0.03)
        np.testing.assert_allclose(sim.sigmoid([-10000, 0, 10000]), [0, 0.5, 1])
        with self.assertRaises(ValueError):
            sim.fit([make_round(0, 0)])


class ReplayTests(unittest.TestCase):
    def test_fixed_six_reproduces_logged_acceptance_and_bonus(self):
        rounds = [make_round(i, a) for i, a in enumerate([0, 1, 6, 3, 2, 6, 5])]
        summary = sim.evaluate(rounds, fixture_model(), 3, 6)
        fixed = summary["policies"]["fixed_6"]
        self.assertEqual(fixed["accepted"], sum(r.accepted for r in rounds))
        self.assertAlmostEqual(fixed["ratio_vs_logged"], 1.0)
        self.assertAlmostEqual(fixed["tokens_per_round"], 1 + fixed["accepted_per_round"])
        self.assertAlmostEqual(sim.expected_accepted(rounds[0], 0.5, fixture_model(), 3), 0.7 + 0.7**2 + 0.7**3)

    def test_oracle_bounds_every_policy_on_same_cohort(self):
        rng = np.random.default_rng(90)
        rounds = [make_round(i, int(rng.integers(0, 7))) for i in range(90)]
        result = sim.evaluate(rounds, fixture_model(), 3, 6)["policies"]
        oracle = result["oracle"]["tokens_per_second"]
        for policy in result.values():
            self.assertGreaterEqual(oracle + 1e-9, policy["tokens_per_second"])

    def test_unobservable_rounds_are_never_invented(self):
        rounds = [make_round(i, d, depth=d) for i, d in enumerate([2, 3, 4, 5, 6])]
        result = sim.evaluate(rounds, fixture_model(), 3, 6)
        self.assertEqual(result["eligible_rounds"], 4)
        self.assertEqual(result["policies"]["fixed_6"]["rounds"], 1)
        self.assertEqual(result["policies"]["fixed_6"]["unobservable"], 3)
        for name in ("streak", "cdepth", "oracle"):
            row = result["policies"][name]
            self.assertEqual(row["rounds"], 4)
            # Four distinct ceilings with full acceptance require at least one shallow choice.
            self.assertLessEqual(sum(int(d) * count for d, count in row["depth_counts"].items()), 3 + 4 + 5 + 6)
        with self.assertRaises(ValueError):
            sim.evaluate(rounds, fixture_model(), 0, 6)
        with self.assertRaises(ValueError):
            sim.evaluate(rounds, fixture_model(), 3, 7)

    def test_history_is_causal_and_isolated_by_stream(self):
        first = make_round(0, 0)
        first.target_p1 = [0.2] + [0.99] * 6
        second = make_round(1, 0)
        other = make_round(0, 0, stream=("fixture", 0, 1, 0))
        means = sim.rolling_means([first, other, second], 8)
        np.testing.assert_allclose(means, [0.5, 0.5, 0.2])
        second.target_p1 = [0.99] * 7
        np.testing.assert_allclose(sim.rolling_means([first, other, second], 8), means)

    def test_cdepth_responds_to_confidence(self):
        model = fixture_model()
        model["positions"] = [{"coef": [-6, 0, 12, 0]} for _ in range(6)]
        low, high = make_round(0, 0), make_round(1, 6)
        low.draft_p1 = [0.1] * 6
        high.draft_p1 = [0.99] * 6
        result = sim.evaluate([low, high], model, 3, 6)["policies"]["cdepth"]
        self.assertEqual(result["depth_counts"], {"3": 1, "6": 1})

    def test_checkpoint_replay_bonus_and_history(self):
        row = make_round(0, 1)
        row.emitted = 3
        row.committed_p1 = [.2, .4, .9]
        next_row = make_round(1, 0)
        self.assertEqual(sim.replay_tokens(row, 6), 3)
        self.assertEqual(sim.replay_tokens(row, 1), 2)
        np.testing.assert_allclose(sim.rolling_means([row, next_row], 8), [.5, .5])
        result = sim.evaluate([row], fixture_model(), 3, 6)["policies"]["fixed_6"]
        self.assertEqual(result["tokens_per_round"], 3)
        self.assertAlmostEqual(result["ratio_vs_logged"], 1)

    def test_streak_uses_production_bounds(self):
        rounds = [make_round(i, 6) for i in range(50)]
        default = sim.evaluate(rounds, fixture_model(), 3, 6)
        self.assertEqual(default["streak_range"], [3, 4])
        self.assertEqual(set(default["policies"]["streak"]["depth_counts"]), {"3", "4"})
        expanded = sim.evaluate(rounds, fixture_model(), 3, 6, streak_range=(3, 6))
        self.assertIn("6", expanded["policies"]["streak"]["depth_counts"])

    def test_streak_hand_trace_twenty_rounds(self):
        # Six full accepts climb 3->4; five climb 4->5. Later misses cross pressure 25.
        accepted = [3] * 6 + [4] * 5 + [0] * 4 + [3, 4, 4, 4, 4]
        expected = [(3, i, 0) for i in range(1, 6)] + [(4, 0, 0)]
        expected += [(4, i, 0) for i in range(1, 5)] + [(5, 0, 0)]
        expected += [(5, 0, 5), (5, 0, 10), (5, 0, 15), (5, 0, 20), (5, 0, 22),
                     (5, 0, 23), (5, 0, 24), (4, 0, 0), (4, 1, 0)]
        state = sim.Streak(6, 3)
        trace = []
        for a in accepted:
            state.update(state.n_cur, a)
            trace.append((state.n_cur, state.n_climb, state.n_drop))
        self.assertEqual(trace, expected)
        state.update(0, 0)
        self.assertEqual((state.n_cur, state.n_climb, state.n_drop), expected[-1])
        self.assertEqual([sim.Streak.climb_threshold(d) for d in range(1, 9)], [2, 4, 6, 5, 4, 3, 2, 2])
        self.assertEqual(sim.Streak(2, 4).n_cur, 2)
        self.assertEqual(sim.Streak(0, 0).n_cur, 1)


class IOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build-cpu")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_log(self, name, records):
        path = self.directory / name
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return str(path)

    def test_headers_requests_and_slot_reuse_reset_history(self):
        a = serialize(make_round(0, 0, n_past=100000))
        a["emitted"] = 2
        a["committed_p1"] = [.2, .3]
        same = serialize(make_round(1, 0, n_past=100002))
        b = serialize(make_round(2, 0, n_past=100003), request=2)
        c = serialize(make_round(3, 0, n_past=100004), request=2)
        a["target_p1"][0] = 0.2
        b["target_p1"][0] = 0.4
        header = {"event": "header", "model": "test"}
        shifted = serialize(make_round(4, 0, n_past=128), request=2)
        path = self.write_log("run.jsonl", [header, a, same, b, c, shifted, header, a])
        rows = sim.read_logs([path])
        np.testing.assert_allclose(sim.rolling_means(rows, 8), [0.5, 0.25, 0.5, 0.4, 0.6, 0.5])
        self.assertEqual(sim.expand_logs([str(self.directory / "*.jsonl"), path]), [path])
        with self.assertRaises(ValueError):
            sim.expand_logs([str(self.directory / "missing*")])

    def test_malformed_and_unavailable_data_fail_loudly(self):
        for key, value in [("accepted", 7), ("drafted", 0), ("target_p1", [0.1]),
                           ("draft_p1", [None] * 6), ("t_round_ms", float("nan")),
                           ("draft_p2", [0.9] * 6), ("request", "bad")]:
            with self.subTest(key=key):
                record = serialize(make_round(0, 1))
                record[key] = value
                path = self.write_log("bad.jsonl", [record])
                with self.assertRaisesRegex(ValueError, "bad.jsonl:1"):
                    sim.read_logs([path])
        for value in ([], 1, None):
            path = self.write_log("bad.jsonl", [value])
            with self.assertRaisesRegex(ValueError, "each log line must be an object"):
                sim.read_logs([path])
        path = self.write_log("empty.jsonl", [{"event": "header"}])
        with self.assertRaises(ValueError):
            sim.read_logs([path])

    def test_fit_eval_cli_writes_json_and_markdown(self):
        records = []
        past = 100000
        for i in range(160):
            row = make_round(i, i % 7, n_past=past)
            records.append(serialize(row))
            past += row.accepted + 1
        path = self.write_log("run.jsonl", [{"event": "header"}] + records)
        model, out = self.directory / "model.json", self.directory / "summary.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sim.main(["fit", "--logs", path, "--out", str(model)]), 0)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(sim.main(["eval", "--logs", path, "--model", str(model),
                                       "--range", "3", "6", "--out", str(out)]), 0)
        self.assertIn("| Policy |", printed.getvalue())
        self.assertIn("unobservable", printed.getvalue())
        self.assertEqual(json.loads(out.read_text())["policies"]["fixed_6"]["accepted"], sum(i % 7 for i in range(160)))

    def test_disabled_cpp_log_guard(self):
        self.assertIn("PASS", check(ROOT))

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is needed")
    def test_cpp_writer_guards_buffering_and_slot_records(self):
        source = (ROOT / "tools/server/server-context.cpp").read_text()
        path_body = function_body(source, "static const char * spec_conflog_path()")
        methods = source[source.index("    FILE * spec_conflog_open()"):
                         source.index("    void destroy()")]
        # Count opens at the I/O boundary while running the production methods.
        methods = methods.replace("std::fopen(path,", "counted_fopen(path,")
        code = r'''#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include "vendor/nlohmann/json.hpp"
using json = nlohmann::json;
#define SRV_ERR(...) ((void) 0)
static int opens = 0;
static FILE * counted_fopen(const char * path, const char * mode) { ++opens; return std::fopen(path, mode); }
static int64_t ggml_time_us() { return 10000; }
static float spec_conflog_row_p1(void *, int32_t idx) { return .1f * (idx + 1); }
static const std::vector<float> & common_sampler_get_top1_trace(void *) { static const std::vector<float> v = {.5f, .6f}; return v; }
static const char * spec_conflog_path() {''' + path_body + r'''}
struct task { int id = 7; };
struct server_slot {
    int id = 0;
    struct task storage;
    struct task * task = &storage;
    void * ctx_tgt = nullptr;
    struct { void * get() const { return nullptr; } } smpl;
    std::vector<int32_t> spec_i_batch = {5, 6, 7};
    std::vector<float> spec_conf_p1 = {.7f, .8f}, spec_conf_p2 = {.2f, .1f};
    bool spec_is_replay = false;
    int spec_conf_depth = 2;
    int spec_conf_pos = 100000;
    int64_t spec_conf_start = 9000;
    json spec_conf_pending;
};
static std::string common_speculative_type_to_str(int) { return "draft-mtp"; }
struct logger {
    FILE * conflog_fp = nullptr;
    std::mutex conflog_mu;
    bool conflog_failed = false;
    uint32_t conflog_lines = 0;
    struct {
        struct { std::string path = "test\"model"; } model;
        struct {
            struct { int n_max = 6; int n_min_adaptive = 3; } draft;
            std::vector<int> types = {1};
        } speculative;
    } params_base;
''' + methods + r'''};
int main() {
    logger log;
    if (!spec_conflog_path()) {
        if (log.spec_conflog_open() != nullptr || opens != 0) return 1;
        log.spec_conflog_close();
        return 0;
    }
    server_slot a, b;
    b.id = 1;
    log.spec_conflog_capture(a, 2, 1, 5);
    log.spec_conflog_capture(b, 2, 0, 5);
    if (a.spec_conf_pending["accepted"] != 1 || b.spec_conf_pending["accepted"] != 0) return 2;
    if (a.spec_conf_pending["target_p1"].size() != 3) return 3;
    if (a.spec_conf_pending["target_ps1"].size() != 2) return 9;
    if (std::abs(a.spec_conf_pending["target_p1"][0].get<float>() - .1f) > 1e-6) return 4;
    a.spec_is_replay = true;
    log.spec_conflog_capture(a, 2, 2, 5);
    if (a.spec_conf_pending["accepted"] != 1 || a.spec_conf_pending["committed_p1"].size() != 3) return 8;
    a.spec_is_replay = false;
    for (int i = 0; i < 64; ++i) {
        server_slot & slot = i % 2 ? b : a;
        log.spec_conflog_capture(slot, 2, i % 2 ? 0 : 1, 5);
        log.spec_conflog_write(slot, slot.spec_conf_pending["accepted"].get<size_t>() + 1);
        if (!slot.spec_conf_pending.is_null()) return 5;
    }
    if (opens != 1) return 6;
    // Header plus 63 round lines were explicitly flushed at the 64-line boundary.
    std::ifstream input(spec_conflog_path());
    std::string line;
    int lines = 0;
    while (std::getline(input, line)) ++lines;
    if (lines < 64) return 7;
    log.spec_conflog_close();
}
'''
        program = self.directory / "writer.cpp"
        binary = self.directory / "writer"
        program.write_text(code)
        subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(program), "-o", str(binary)], check=True)
        env = os.environ.copy()
        env.pop("LLAMA_SPEC_CONFLOG", None)
        subprocess.run([str(binary)], env=env, check=True)
        env["LLAMA_SPEC_CONFLOG"] = ""
        subprocess.run([str(binary)], env=env, check=True)
        log = self.directory / "confidence.jsonl"
        env["LLAMA_SPEC_CONFLOG"] = str(log)
        subprocess.run([str(binary)], env=env, check=True)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(records), 65)
        self.assertEqual(records[0]["event"], "header")
        for i, row in enumerate(records[1:]):
            self.assertEqual(row["slot"], i % 2)
            self.assertEqual(row["accepted"], 0 if i % 2 else 1)
            self.assertEqual(row["drafted"], 2)
            self.assertEqual(row["t_round_ms"], 1)
            np.testing.assert_allclose(row["target_p1"], [.1, .2, .3], atol=1e-6)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is needed")
    def test_cpp_top2_probabilities_and_streak_port(self):
        # Compile only host arithmetic and the controller header. No llama backend is linked.
        source = (ROOT / "common/speculative.cpp").read_text()
        top2 = function_body(source, "static void spec_conf_top2_from_row(")
        program = self.directory / "check.cpp"
        program.write_text('''#include <cmath>
#include <cstdint>
#include <cstdio>
#include "common/speculative-adaptive.h"
static void top2(const float * row, int32_t n, float & p1, float & p2) {''' + top2 + '''}
int main() {
    float row[] = {std::log(0.1f), std::log(0.6f), std::log(0.3f)};
    float p1, p2;
    top2(row, 3, p1, p2);
    if (std::abs(p1 - .6f) > 1e-6 || std::abs(p2 - .3f) > 1e-6) return 1;
    float ties[] = {10000, 10000, -INFINITY};
    top2(ties, 3, p1, p2);
    if (p1 != .5f || p2 != .5f) return 2;
    top2(row, 1, p1, p2);
    if (p1 != 1 || p2 != 0) return 3;
    top2(nullptr, 0, p1, p2);
    if (!std::isnan(p1) || !std::isnan(p2)) return 4;
    float invalid[] = {NAN, 1};
    top2(invalid, 2, p1, p2);
    if (!std::isnan(p1) || !std::isnan(p2)) return 5;
    common_speculative_adaptive state;
    state.reset(8, 1);
    for (int i = 0; i < 150; ++i) {
        int n = i % 19 == 0 ? 0 : state.n_cur;
        int a = i < 85 ? n : (i % 11 < 8 ? 0 : n);
        state.update(n, a, 8, 1);
        std::printf("%d %d %d\\n", state.n_cur, state.n_climb, state.n_drop);
    }
}
''')
        binary = self.directory / "check"
        subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(program), "-o", str(binary)], check=True)
        actual = subprocess.check_output([str(binary)], text=True).splitlines()
        state = sim.Streak(8, 1)
        expected = []
        for i in range(150):
            n = 0 if i % 19 == 0 else state.n_cur
            a = n if i < 85 else (0 if i % 11 < 8 else n)
            state.update(n, a)
            expected.append(f"{state.n_cur} {state.n_climb} {state.n_drop}")
        self.assertEqual(actual, expected)


class CausalTests(unittest.TestCase):
    """The on-device (causal) policy: fit, replay, and exact agreement between speculative-cdepth.h and the simulator."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.cost = {3: 70.0, 4: 72.0, 5: 74.5, 6: 77.0}
        cls.rounds = []
        history = deque(maxlen=16)
        for i in range(20000):
            mean = float(np.mean(history)) if history else 0.5
            p1 = rng.uniform(0.3, 0.97, 6)
            p2 = np.minimum(p1, 1 - p1) * rng.uniform(0.05, 0.95, 6)
            row = make_round(i, 0)
            row.draft_p1, row.draft_p2 = p1.tolist(), p2.tolist()
            for k in range(6):
                z = -2.2 + 1.5 * mean + 2.8 * p1[k]
                if rng.random() >= 1 / (1 + math.exp(-z)):
                    break
                row.accepted += 1
            row.target_ps1 = np.clip(rng.normal(0.35 + 0.5 * (i // 7 % 2), 0.05, row.accepted + 1), 0, 1).tolist()
            history.extend(row.target_ps1)
            cls.rounds.append(row)

    def test_fit_orders_thresholds_and_beats_fixed_floor(self):
        model = sim.fit_causal(self.rounds, 3, 6, self.cost)
        self.assertEqual(len(model["stages"]), 5)
        self.assertTrue(all(st["theta"] >= 0 for st in model["stages"]))
        result = sim.evaluate_causal(self.rounds, model, self.cost, 3, 6)
        policies = result["policies"]
        self.assertGreater(policies["cdepth_causal_x1"]["tokens_per_second"], policies["fixed_3"]["tokens_per_second"])
        self.assertLessEqual(policies["cdepth_causal_x1"]["tokens_per_second"], policies["oracle"]["tokens_per_second"] * 1.0001)

    def test_missing_ps1_is_rejected(self):
        rows = [make_round(i, 2) for i in range(10)]
        with self.assertRaises(ValueError):
            sim.fit_causal(rows, 3, 6, self.cost)

    def test_header_matches_simulator(self):
        model = sim.fit_causal(self.rounds, 3, 6, self.cost)
        directory = ROOT / "build-cpu"
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            tmp = Path(tmp)
            (tmp / "m.cdm").write_text(sim.causal_text(model))
            (tmp / "t.cpp").write_text(r'''
#include "common/speculative-cdepth.h"
#include <cstdio>
int main(int, char ** argv) {
    common_speculative_cdepth_model m; std::string err;
    if (!m.load(argv[1], err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
    std::printf("window %d stages %zu\n", m.window, m.coef.size());
    for (int j = 1; j <= 5; ++j) {
        for (float mean : {0.2f, 0.8f}) {
            std::printf("%d %.6f %.6f\n", j, (double) m.chance(j, mean, 0.7f, 0.1f, -1.3f), (double) m.threshold(j, 1.0f));
        }
    }
    common_speculative_cdepth_state s;
    const float p[3] = {0.5f, -1.0f, 0.9f};
    s.note_target(p, 3, m.window);
    std::printf("mean %.6f\n", (double) s.mean(m));
    common_speculative_cdepth_model b; std::istringstream is(common_speculative_cdepth_builtin_text());
    std::printf("builtin %d\n", (int) b.parse(is, err));
}
''')
            subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(tmp / "t.cpp"), "-o", str(tmp / "t")], check=True)
            lines = subprocess.check_output([str(tmp / "t"), str(tmp / "m.cdm")], text=True).splitlines()
        self.assertEqual(lines[0], f"window {model['window']} stages 5")
        k = 1
        for j in range(1, 6):
            for mean in (0.2, 0.8):
                _, chance, theta = lines[k].split()
                k += 1
                x = [1.0, mean, 0.7, 0.7 - 0.1, -1.3]
                self.assertAlmostEqual(float(chance), sim.causal_probability(model["stages"][j - 1]["coef"], x), places=5)
                self.assertAlmostEqual(float(theta), model["stages"][j - 1]["theta"], places=5)
        self.assertEqual(lines[k], "mean 0.700000")  # the unknown (-1) entry is skipped
        self.assertEqual(lines[k + 1], "builtin 1")


if __name__ == "__main__":
    unittest.main()
