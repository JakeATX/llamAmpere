#!/usr/bin/env python3
"""Fit conditional acceptance curves and replay confidence-driven MTP depths."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
import sys

import numpy as np


@dataclass
class Round:
    stream: tuple
    n_past: int
    depth: int
    drafted: int
    accepted: int
    draft_p1: list
    draft_p2: list
    target_p1: list
    ms: float
    emitted: int = 0
    committed_p1: list | None = None
    target_ps1: list | None = None  # request-matched (post-sampler) target top-1 of the committed rows; may be absent


def expand_logs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"no log files match {pattern!r}")
        for path in matches:
            if path not in paths:
                paths.append(path)
    return paths


def read_logs(paths):
    rounds = []
    for path in paths:
        session = 0
        requests = {}
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("each log line must be an object")
                    if row.get("event") == "header":
                        session += 1
                        requests.clear()
                        continue
                    integers = ("slot", "n_past", "depth", "drafted", "accepted")
                    for key in integers:
                        if type(row[key]) is not int or row[key] < 0:
                            raise ValueError(f"{key} must be a nonnegative integer")
                    n, a = row["drafted"], row["accepted"]
                    if n < 1 or a > n or row["depth"] < n:
                        raise ValueError("need 0 <= accepted <= drafted <= depth, drafted > 0")
                    for key, size in (("draft_p1", n), ("draft_p2", n), ("target_p1", n + 1)):
                        values = row[key]
                        if not isinstance(values, list) or len(values) != size:
                            raise ValueError(f"{key} must have {size} entries")
                        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                            raise ValueError(f"{key} needs finite probabilities; null means confidence was unavailable")
                    if any(p2 > p1 or p1 + p2 > 1.00001 for p1, p2 in zip(row["draft_p1"], row["draft_p2"])):
                        raise ValueError("invalid top-2 probabilities")
                    ms = float(row["t_round_ms"])
                    if not math.isfinite(ms) or ms <= 0:
                        raise ValueError("t_round_ms must be positive and finite")
                    emitted = row.get("emitted", a + 1)
                    if type(emitted) is not int or not a + 1 <= emitted <= n + 1:
                        raise ValueError("invalid emitted token count")
                    committed = row.get("committed_p1", row["target_p1"][:a + 1])
                    if not isinstance(committed, list) or len(committed) != emitted or any(
                            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in committed):
                        raise ValueError("committed_p1 must describe every emitted token")
                    slot = row["slot"]
                    # Request ids disambiguate a reused slot even when its next prompt is longer.
                    request = row.get("request")
                    if request is not None and type(request) is not int:
                        raise ValueError("request must be an integer")
                    previous = requests.get(slot)
                    if previous is None:
                        epoch = 0
                    else:
                        old_request, expected_past, epoch = previous
                        if request != old_request or (request is None and row["n_past"] != expected_past):
                            epoch += 1
                    requests[slot] = (request, row["n_past"] + emitted, epoch)
                    stream = (str(Path(path).resolve()), session, slot, epoch)
                    ps1 = row.get("target_ps1")
                    if ps1 is not None:
                        if not isinstance(ps1, list) or any(
                                v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1)
                                for v in ps1):
                            raise ValueError("target_ps1 must be a list of probabilities or null")
                    rounds.append(Round(stream, row["n_past"], row["depth"], n, a,
                                        row["draft_p1"], row["draft_p2"], row["target_p1"], ms, emitted, committed, ps1))
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{path}:{line_no}: {e}") from e
    if not rounds:
        raise ValueError("logs contain no verify rounds")
    return rounds


def rolling_means(rounds, window):
    histories = defaultdict(lambda: deque(maxlen=window))
    result = []
    for row in rounds:
        history = histories[row.stream]
        result.append(float(np.mean(history)) if history else 0.5)
        # Rows beyond the first rejection have a false prefix. Do not add them.
        history.extend(row.committed_p1 if row.committed_p1 is not None else row.target_p1[:row.accepted + 1])
    return np.asarray(result)


def features(row, mean, k):
    return [1.0, mean, row.draft_p1[k], row.draft_p1[k] - row.draft_p2[k]]


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def log_loss(x, y, coef):
    z = x @ coef
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_logistic(x, y):
    """Damped Newton steps; standardize for conditioning, return raw coefficients."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) == 0:
        raise ValueError("no conditional observations for a draft position")
    if np.all(y == y[0]):
        p = (float(y.sum()) + 0.5) / (len(y) + 1)
        return np.concatenate(([math.log(p / (1 - p))], np.zeros(x.shape[1] - 1)))
    center = x[:, 1:].mean(axis=0)
    scale = x[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = x.copy()
    z[:, 1:] = (z[:, 1:] - center) / scale
    beta = np.zeros(x.shape[1])
    penalty = np.diag([0.0] + [1e-6] * (x.shape[1] - 1))

    def objective(b):
        return log_loss(z, y, b) * len(y) + 0.5 * b @ penalty @ b

    for _ in range(80):
        p = sigmoid(z @ beta)
        gradient = z.T @ (p - y) + penalty @ beta
        hessian = z.T @ ((p * (1 - p))[:, None] * z) + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        before = objective(beta)
        rate = 1.0
        while rate > 1e-8 and objective(beta - rate * step) > before:
            rate *= 0.5
        beta -= rate * step
        if np.max(np.abs(rate * step)) < 1e-8:
            break
    coef = beta.copy()
    coef[1:] = beta[1:] / scale
    coef[0] -= coef[1:] @ center
    return coef


def conditional_data(rounds, means, indices, k):
    # Position k is observed only if all preceding positions were accepted.
    eligible = [i for i in indices if rounds[i].drafted > k and rounds[i].accepted >= k]
    x = np.asarray([features(rounds[i], means[i], k) for i in eligible], dtype=float).reshape(-1, 4)
    y = np.asarray([float(rounds[i].accepted > k) for i in eligible])
    return x, y


def fit_cost(rounds):
    depth = np.asarray([r.drafted for r in rounds], dtype=float)
    times = np.asarray([r.ms for r in rounds])
    if len(set(depth)) == 1:
        return {"a": float(times.mean()), "b": 0.0, "identified": False,
                "warning": "One observed depth cannot identify a slope. Using constant cost; collect separate mixed-depth cost logs."}
    x = np.column_stack((np.ones(len(depth)), depth))
    coef = np.linalg.lstsq(x, times, rcond=None)[0]
    candidates = [np.array([times.mean(), 0.0]), np.array([0.0, depth @ times / (depth @ depth)])]
    if np.all(coef >= 0):
        candidates.append(coef)
    a, b = min(candidates, key=lambda c: float(np.sum((x @ c - times) ** 2)))
    return {"a": float(a), "b": float(b), "identified": True,
            "depth_min": int(depth.min()), "depth_max": int(depth.max())}


def fit(rounds, cost_rounds=None):
    streams = defaultdict(list)
    for i, row in enumerate(rounds):
        streams[row.stream].append(i)
    train, heldout = [], []
    for indices in streams.values():
        cut = max(1, int(len(indices) * 0.8))
        train.extend(indices[:cut])
        heldout.extend(indices[cut:])
    if not heldout:
        raise ValueError("need at least two consecutive rounds in a stream for held-out selection")
    n_depth = max(r.drafted for r in rounds)
    losses = {}
    for window in (8, 16, 32):
        means = rolling_means(rounds, window)
        total, count = 0.0, 0
        for k in range(n_depth):
            x, y = conditional_data(rounds, means, train, k)
            coef = fit_logistic(x, y)
            xv, yv = conditional_data(rounds, means, heldout, k)
            if len(yv):
                total += log_loss(xv, yv, coef) * len(yv)
                count += len(yv)
        if not count:
            raise ValueError("no held-out conditional observations")
        losses[window] = total / count
    window = min(losses, key=lambda window: losses[window])
    means = rolling_means(rounds, window)
    positions = []
    for k in range(n_depth):
        x, y = conditional_data(rounds, means, range(len(rounds)), k)
        positions.append({"k": k + 1, "coef": fit_logistic(x, y).tolist(), "observations": len(y)})
    return {"version": 1, "window": window,
            "features": ["intercept", "rolling_target_p1", "draft_p1", "draft_margin"],
            "heldout_log_loss": {str(k): v for k, v in losses.items()},
            "split": "chronological last 20% of each stream; refit selected window on all fit data",
            "positions": positions, "cost": fit_cost(rounds if cost_rounds is None else cost_rounds)}


class Streak:
    """Port of common/speculative-adaptive.h, including cap/floor edge behavior."""

    def __init__(self, n_max, n_min):
        self.cap = max(1, n_max)
        self.floor = max(1, n_min)
        self.n_cur = min(self.floor, self.cap)
        self.n_climb = 0
        self.n_drop = 0

    @staticmethod
    def climb_threshold(depth):
        return {1: 2, 2: 4, 3: 6, 4: 5, 5: 4, 6: 3}.get(depth, 2)

    def update(self, n_draft, n_accepted):
        if n_draft <= 0:
            return
        if n_accepted == n_draft:
            self.n_drop = 0
            if self.n_cur < self.cap:
                self.n_climb += 1
                if self.n_climb >= self.climb_threshold(self.n_cur):
                    self.n_cur += 1
                    self.n_climb = 0
        else:
            self.n_climb = 0
            if self.n_cur > self.floor:
                self.n_drop += n_draft - n_accepted
                if self.n_drop >= max(self.n_cur * 5, 20):
                    self.n_cur -= 1
                    self.n_drop = 0


def validate_model(model):
    if model.get("version") != 1 or model.get("window") not in (8, 16, 32):
        raise ValueError("invalid model version or rolling window")
    if not model.get("positions"):
        raise ValueError("model has no positions")
    for position in model["positions"]:
        coef = position["coef"]
        if len(coef) != 4 or not all(math.isfinite(v) for v in coef):
            raise ValueError("invalid model coefficients")
    a, b = model["cost"]["a"], model["cost"]["b"]
    if not math.isfinite(a + b) or a < 0 or b < 0 or a + b <= 0:
        raise ValueError("invalid cost coefficients")


def cost(model, depth):
    return model["cost"]["a"] + model["cost"]["b"] * depth


def expected_accepted(row, mean, model, depth):
    cumulative = 1.0
    tokens = 0.0
    for k in range(depth):
        cumulative *= float(sigmoid(np.asarray(features(row, mean, k)) @ np.asarray(model["positions"][k]["coef"])))
        tokens += cumulative
    return tokens


def replay_tokens(row, depth):
    # Preserve an extra replay bonus only when reproducing the logged depth.
    extra = max(0, row.emitted - row.accepted - 1) if depth == row.drafted else 0
    return min(row.accepted, depth) + 1 + extra


def oracle_choices(rounds, options, model):
    # Fractional optimization gives an aggregate throughput upper bound on this cohort.
    # Maximizing each round's ratio independently need not bound the aggregate ratio.
    rate = 0.0
    choices = []
    for _ in range(100):
        choices = [max(ds, key=lambda d: (replay_tokens(row, d) - rate * cost(model, d), -d))
                   for row, ds in zip(rounds, options)]
        tokens = sum(replay_tokens(row, d) for row, d in zip(rounds, choices))
        ms = sum(cost(model, d) for d in choices)
        new_rate = tokens / ms
        if abs(new_rate - rate) < 1e-12:
            break
        rate = new_rate
    return choices


def evaluate(rounds, model, low, high, streak_range=(3, 4)):
    validate_model(model)
    if low < 1 or high < low or high > len(model["positions"]):
        raise ValueError("range must be positive, ordered, and covered by fitted positions")
    streak_low, streak_high = streak_range
    if not low <= streak_low <= streak_high <= high:
        raise ValueError("streak range must be inside the replay range")
    means = rolling_means(rounds, model["window"])
    eligible = [(row, mean) for row, mean in zip(rounds, means) if row.drafted >= low]
    if not eligible:
        raise ValueError("no observable rounds in requested range")
    rows = [row for row, _ in eligible]
    options = [list(range(low, min(high, row.drafted) + 1)) for row in rows]
    policies = {}
    for depth in range(low, high + 1):
        policies[f"fixed_{depth}"] = [depth if depth <= row.drafted else None for row in rows]
    states = {}
    choices = []
    for row, ds in zip(rows, options):
        state = states.setdefault(row.stream, Streak(streak_high, streak_low))
        depth = min(state.n_cur, ds[-1])
        choices.append(depth)
        state.update(depth, min(row.accepted, depth))
    policies["streak"] = choices
    policies["cdepth"] = [max(ds, key=lambda d: (expected_accepted(row, mean, model, d) / cost(model, d), -d))
                          for (row, mean), ds in zip(eligible, options)]
    policies["oracle"] = oracle_choices(rows, options, model)
    summary = {}
    for name, choices in policies.items():
        pairs = [(row, d) for row, d in zip(rows, choices) if d is not None]
        if not pairs:
            summary[name] = {"rounds": 0, "unobservable": len(rows)}
            continue
        accepted = sum(min(row.accepted, d) for row, d in pairs)
        emitted = sum(replay_tokens(row, d) for row, d in pairs)
        ms = sum(cost(model, d) for _, d in pairs)
        baseline_tokens = sum(row.emitted or row.accepted + 1 for row, _ in pairs)
        baseline_ms = sum(row.ms for row, _ in pairs)
        summary[name] = {"rounds": len(pairs), "unobservable": len(rows) - len(pairs),
                         "accepted": accepted, "accepted_per_round": accepted / len(pairs),
                         "tokens_per_round": emitted / len(pairs), "ms_per_token": ms / emitted,
                         "tokens_per_second": 1000 * emitted / ms,
                         "ratio_vs_logged": (emitted / ms) / (baseline_tokens / baseline_ms),
                         "depth_counts": {str(d): choices.count(d) for d in sorted({d for d in choices if d is not None})}}
    return {"policies": summary, "input_rounds": len(rounds), "eligible_rounds": len(rows),
            "range": [low, high], "streak_range": list(streak_range), "cost": model["cost"],
            "notes": [
                "Offline replay at logged round starts, not a new token trajectory. Outcomes use min(accepted, depth).",
                f"Streak uses bounds {streak_low}..{streak_high}; its thresholds match common/speculative-adaptive.h.",
                "Every policy is bounded by actual drafted depth, even for a fixed-max run. Larger depths are unobservable.",
                "Fixed policies skip unobservable rounds; compare policies only on equal cohorts. Use fixed depth 6 (dense) or 5 (sparse).",
                "Accepted counts are realized replay estimates. Throughput includes replacement/bonus tokens and recorded checkpoint-replay bonuses.",
                "Target history uses only prior committed prefixes. Cdepth can inspect all logged draft confidences; this is an optimistic offline policy.",
                "Cdepth maximizes predicted accepted draft tokens per fitted millisecond. Reported throughput also counts replacement/bonus tokens.",
                "Costs use the fitted line for all policies. Ratios use actual logged time on each policy's cohort, including logging overhead.",
                "Oracle maximizes aggregate hindsight throughput on its cohort. It is an upper bound under the fitted cost model.",
                "Parallel round times include shared work; use parallel=1 for cost calibration. No on-device controller is changed."]}


def markdown(summary):
    lines = ["| Policy | Rounds | Accepted/round | Tokens/round | ms/token | tok/s vs logged |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in summary["policies"].items():
        if not row["rounds"]:
            lines.append(f"| {name} | 0 | - | - | - | - |")
        else:
            lines.append(f"| {name} | {row['rounds']} | {row['accepted_per_round']:.4f} | {row['tokens_per_round']:.4f} | {row['ms_per_token']:.4f} | {row['ratio_vs_logged']:.4f}x |")
    lines.extend("\n" + note for note in summary["notes"])
    if not summary["cost"]["identified"]:
        lines.append("\nCost warning: " + summary["cost"]["warning"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Causal (on-device) policy. After j draft tokens exist, decide whether to draft token j+1 using only what the
# device knows at that moment: the rolling mean of the target's request-matched top-1 over recent committed
# tokens, the helper's top-1 and top-1 minus top-2 at the last drafted token, and the sum of log helper top-1 over
# the draft so far. Keep drafting while P(token j+1 accepted) >= theta_j, where theta_j = R * (cost(j+1) - cost(j))
# is the break-even chance (R = tokens per ms of the whole policy, solved by fixed-point iteration).
# Mirrors common/speculative-cdepth.h exactly (same features, same clamp).
# ---------------------------------------------------------------------------------------------------------------

CAUSAL_FEATURES = ["intercept", "rolling_target_ps1", "draft_p1", "draft_margin", "sum_log_draft_p1"]
CAUSAL_VARIANTS = {  # feature columns kept (ablation)
    "all": [0, 1, 2, 3, 4],
    "no_target_history": [0, 2, 3, 4],
    "no_draft_product": [0, 1, 2, 3],
    "target_history_only": [0, 1],
    "draft_only": [0, 2, 3, 4],
    "intercept_only": [0],
}


def parse_cost_table(text):
    table = {}
    for item in text.split(","):
        depth, ms = item.split(":")
        table[int(depth)] = float(ms)
    if len(table) < 2 or any(v <= 0 or not math.isfinite(v) for v in table.values()):
        raise ValueError("cost table needs at least two positive depth:ms entries, e.g. 3:71.5,4:72.0,5:75.0")
    return table


def table_cost(table, depth):
    if depth in table:
        return table[depth]
    ds = sorted(table)
    if depth < ds[0]:
        a, b = ds[0], ds[1]
    elif depth > ds[-1]:
        a, b = ds[-2], ds[-1]
    else:
        b = next(d for d in ds if d > depth)
        a = max(d for d in ds if d < depth)
    slope = (table[b] - table[a]) / (b - a)
    return table[a] + slope * (depth - a)


def rolling_means_ps(rounds, window):
    histories = defaultdict(lambda: deque(maxlen=window))
    result = []
    for row in rounds:
        history = histories[row.stream]
        result.append(float(np.mean(history)) if history else 0.5)
        if row.target_ps1 is not None:
            history.extend(v for v in row.target_ps1 if v is not None)
    return np.asarray(result)


def causal_features(row, mean, j):
    """Features after j (1-based) draft tokens exist."""
    p1, p2 = row.draft_p1[j - 1], row.draft_p2[j - 1]
    slog = sum(math.log(max(v, 1e-6)) for v in row.draft_p1[:j])
    return [1.0, mean, p1, p1 - p2, max(slog, -12.0)]


def causal_data(rounds, means, indices, j, columns):
    eligible = [i for i in indices if rounds[i].drafted >= j + 1]
    x = np.asarray([causal_features(rounds[i], means[i], j) for i in eligible], dtype=float).reshape(-1, 5)[:, columns]
    y = np.asarray([float(rounds[i].accepted >= j + 1) for i in eligible])
    return x, y


def causal_probability(coef, x):
    return float(sigmoid(np.asarray(x) @ np.asarray(coef)))


def causal_choices(rounds, means, coefs, thetas, floor, cap, scale=1.0):
    choices = []
    for row, mean in zip(rounds, means):
        top = min(cap, row.drafted)
        j = min(max(floor, 1), top)
        while j < top:
            if causal_probability(coefs[j - 1], causal_features(row, mean, j)) < thetas[j - 1] * scale:
                break
            j += 1
        choices.append(j)
    return choices


def policy_rate(rounds, choices, cost_fn):
    tokens = sum(replay_tokens(row, d) for row, d in zip(rounds, choices))
    ms = sum(cost_fn(d) for d in choices)
    return tokens / ms, tokens, ms


def solve_thresholds(rounds, means, coefs, floor, cap, cost_fn):
    """Fixed point of theta_j = R * (cost(j+1) - cost(j)) with R the policy's own tokens per ms."""
    deltas = [max(cost_fn(j + 1) - cost_fn(j), 1e-6) for j in range(1, cap)]
    rate = policy_rate(rounds, [min(max(floor, 1), min(cap, r.drafted)) for r in rounds], cost_fn)[0]
    best = (rate, [rate * d for d in deltas])
    for _ in range(60):
        thetas = [rate * d for d in deltas]
        choices = causal_choices(rounds, means, coefs, thetas, floor, cap)
        new_rate = policy_rate(rounds, choices, cost_fn)[0]
        if new_rate > best[0]:
            best = (new_rate, thetas)
        if abs(new_rate - rate) < 1e-10:
            break
        rate = new_rate
    return best[1], best[0]


def fit_causal(rounds, floor, cap, cost_table, variant="all", windows=(8, 16, 32)):
    if any(r.target_ps1 is None for r in rounds):
        raise ValueError("logs have no target_ps1; collect them with the CDEPTH-capable server")
    columns = CAUSAL_VARIANTS[variant]
    cost_fn = lambda d: table_cost(cost_table, d)
    streams = defaultdict(list)
    for i, row in enumerate(rounds):
        streams[row.stream].append(i)
    train, heldout = [], []
    for indices in streams.values():
        cut = max(1, int(len(indices) * 0.8))
        train.extend(indices[:cut])
        heldout.extend(indices[cut:])
    if not heldout:
        raise ValueError("need at least two consecutive rounds in a stream for held-out selection")
    losses = {}
    for window in windows:
        means = rolling_means_ps(rounds, window)
        total, count = 0.0, 0
        for j in range(max(floor, 1), cap):
            x, y = causal_data(rounds, means, train, j, columns)
            coef = fit_logistic(x, y)
            xv, yv = causal_data(rounds, means, heldout, j, columns)
            if len(yv):
                total += log_loss(xv, yv, coef) * len(yv)
                count += len(yv)
        if not count:
            raise ValueError("no held-out observations")
        losses[window] = total / count
    window = min(losses, key=lambda w: losses[w])
    means = rolling_means_ps(rounds, window)
    coefs, observations = [], []
    for j in range(1, cap):
        x, y = causal_data(rounds, means, range(len(rounds)), j, columns)
        full = np.zeros(5)
        if len(y):
            full[columns] = fit_logistic(x, y)
        coefs.append(full.tolist())
        observations.append(int(len(y)))
    cohort = [i for i, r in enumerate(rounds) if r.drafted >= cap]
    thetas, rate = solve_thresholds([rounds[i] for i in cohort], means[cohort], coefs, floor, cap, cost_fn)
    for j in range(1, floor):
        thetas[j - 1] = 0.0
    return {"version": 2, "kind": "causal", "variant": variant, "features": CAUSAL_FEATURES, "window": window,
            "floor": floor, "cap": cap, "cold": 0.5, "cost_table": {str(k): v for k, v in sorted(cost_table.items())},
            "heldout_log_loss": {str(k): v for k, v in losses.items()}, "fit_rounds": len(rounds),
            "fit_cohort_rounds": len(cohort), "fit_rate_tokens_per_ms": rate,
            "stages": [{"j": j + 1, "coef": coefs[j], "theta": thetas[j], "observations": observations[j]} for j in range(cap - 1)]}


def causal_text(model):
    lines = ["cdepth-model 1", f"# variant {model['variant']} floor {model['floor']} cap {model['cap']} cohort {model['fit_cohort_rounds']}",
             f"window {model['window']}", f"cold {model['cold']}"]
    for st in model["stages"]:
        lines.append("stage %d %s %.8g" % (st["j"], " ".join("%.8g" % c for c in st["coef"]), st["theta"]))
    return "\n".join(lines) + "\n"


def evaluate_causal(rounds, model, cost_table, floor, cap, scales=(1.0,), streak_range=(3, 4)):
    if any(r.target_ps1 is None for r in rounds):
        raise ValueError("logs have no target_ps1")
    cost_fn = lambda d: table_cost(cost_table, d)
    means = rolling_means_ps(rounds, model["window"])
    cohort = [i for i, r in enumerate(rounds) if r.drafted >= cap]
    rows = [rounds[i] for i in cohort]
    cmeans = means[cohort]
    if not rows:
        raise ValueError("no rounds drafted the full cap")
    coefs = [st["coef"] for st in model["stages"]]
    thetas = [st["theta"] for st in model["stages"]]
    policies = {}
    for d in range(floor, cap + 1):
        policies[f"fixed_{d}"] = [d] * len(rows)
    state = {}
    choices = []
    for row in rows:
        s = state.setdefault(row.stream, Streak(streak_range[1], streak_range[0]))
        d = min(s.n_cur, cap)
        choices.append(d)
        s.update(d, min(row.accepted, d))
    policies[f"streak_{streak_range[0]}_{streak_range[1]}"] = choices
    state = {}
    choices = []
    for row in rows:
        s = state.setdefault(row.stream, Streak(cap, floor))
        d = min(s.n_cur, cap)
        choices.append(d)
        s.update(d, min(row.accepted, d))
    policies[f"streak_{floor}_{cap}"] = choices
    for scale in scales:
        policies[f"cdepth_causal_x{scale:g}"] = causal_choices(rows, cmeans, coefs, thetas, floor, cap, scale)
    options = [list(range(floor, cap + 1))] * len(rows)
    rate = 0.0
    oracle = []
    for _ in range(100):
        oracle = [max(ds, key=lambda d: (replay_tokens(row, d) - rate * cost_fn(d), -d)) for row, ds in zip(rows, options)]
        new_rate = sum(replay_tokens(r, d) for r, d in zip(rows, oracle)) / sum(cost_fn(d) for d in oracle)
        if abs(new_rate - rate) < 1e-12:
            break
        rate = new_rate
    policies["oracle"] = oracle
    out = {}
    for name, choice in policies.items():
        tokens = sum(replay_tokens(r, d) for r, d in zip(rows, choice))
        ms = sum(cost_fn(d) for d in choice)
        out[name] = {"rounds": len(rows), "tokens_per_round": tokens / len(rows), "ms_per_round": ms / len(rows),
                     "tokens_per_second": 1000.0 * tokens / ms, "accepted_per_round": sum(min(r.accepted, d) for r, d in zip(rows, choice)) / len(rows),
                     "depth_counts": {str(d): choice.count(d) for d in sorted(set(choice))}}
    base = out[f"streak_{streak_range[0]}_{streak_range[1]}"]["tokens_per_second"]
    for v in out.values():
        v["ratio_vs_streak"] = v["tokens_per_second"] / base
    return {"policies": out, "cohort_rounds": len(rows), "input_rounds": len(rounds), "floor": floor, "cap": cap,
            "cost_table": cost_table, "window": model["window"], "variant": model.get("variant")}


def causal_markdown(result):
    lines = ["| Policy | Tokens/round | ms/round (table) | tok/s (table) | vs streak | Depth counts |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, v in result["policies"].items():
        lines.append(f"| {name} | {v['tokens_per_round']:.4f} | {v['ms_per_round']:.2f} | {v['tokens_per_second']:.3f} | {v['ratio_vs_streak']:.4f}x | {v['depth_counts']} |")
    lines.append(f"\ncohort {result['cohort_rounds']} of {result['input_rounds']} rounds drafted the full cap {result['cap']}; window {result['window']}; variant {result['variant']}")
    return "\n".join(lines)


def main_causal(args):
    rounds = read_logs(expand_logs(args.logs))
    cost_table = parse_cost_table(args.cost_table)
    if args.command == "fit-causal":
        model = fit_causal(rounds, args.floor, args.cap, cost_table, args.variant)
        Path(args.out).write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.out_text:
            Path(args.out_text).write_text(causal_text(model), encoding="utf-8")
        print(f"Fitted causal CDEPTH stages 1..{args.cap - 1}, window={model['window']}, variant={args.variant}; "
              f"held-out log loss {model['heldout_log_loss']}")
        print("thresholds:", [round(st["theta"], 4) for st in model["stages"]])
    else:
        with open(args.model, encoding="utf-8") as f:
            model = json.load(f)
        scales = [float(s) for s in args.theta_scales.split(",")]
        result = evaluate_causal(rounds, model, cost_table, args.floor, args.cap, scales, tuple(args.streak_range))
        print(causal_markdown(result))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--logs", nargs="+", required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.add_argument("--cost-logs", nargs="+", help="separate varied-depth logs for the timing fit")
    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--logs", nargs="+", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--range", nargs=2, type=int, default=[3, 6], metavar=("MIN", "MAX"))
    eval_parser.add_argument("--streak-range", nargs=2, type=int, default=[3, 4], metavar=("MIN", "MAX"),
                             help="existing controller bounds (default: production 3 4)")
    eval_parser.add_argument("--out", help="JSON summary path; otherwise print JSON after the table")
    for name in ("fit-causal", "eval-causal"):
        sub = commands.add_parser(name, help="on-device (causal) CDEPTH policy: fit coefficients / replay")
        sub.add_argument("--logs", nargs="+", required=True)
        sub.add_argument("--cost-table", required=True, help="measured ms per round by drafted width, e.g. 3:71.5,4:72,5:75,6:78")
        sub.add_argument("--floor", type=int, default=3)
        sub.add_argument("--cap", type=int, default=6)
        sub.add_argument("--out", help="JSON path")
        if name == "fit-causal":
            sub.add_argument("--variant", choices=sorted(CAUSAL_VARIANTS), default="all")
            sub.add_argument("--out-text", help="text model for --spec-depth-model")
        else:
            sub.add_argument("--model", required=True)
            sub.add_argument("--theta-scales", default="1.0")
            sub.add_argument("--streak-range", nargs=2, type=int, default=[3, 4], metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)
    if args.command in ("fit-causal", "eval-causal"):
        try:
            main_causal(args)
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as e:
            parser.error(str(e))
        return 0
    try:
        rounds = read_logs(expand_logs(args.logs))
        if args.command == "fit":
            cost_rounds = read_logs(expand_logs(args.cost_logs)) if args.cost_logs else None
            result = fit(rounds, cost_rounds)
            Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Fitted {len(result['positions'])} positions, window={result['window']}; wrote {args.out}")
            if not result["cost"]["identified"]:
                print(result["cost"]["warning"], file=sys.stderr)
        else:
            with open(args.model, encoding="utf-8") as f:
                model = json.load(f)
            result = evaluate(rounds, model, *args.range, streak_range=args.streak_range)
            print(markdown(result))
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.out:
                Path(args.out).write_text(payload, encoding="utf-8")
            else:
                print("\n" + payload)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as e:
        parser.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
