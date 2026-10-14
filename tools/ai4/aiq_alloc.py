#!/usr/bin/env python3
"""aiq_alloc.py (AIQ3, private): byte-budget tier allocator for AI+ models.

Picks one tier (= one already-encoded candidate GGUF, see llama-ai4-errtab --cand TAG=FILE) per tensor so that the
model's bytes in --scope fit a target, minimizing the summed measured error of the chosen tiers (multiple-choice
knapsack). Tensors that are not in the error tables keep the --base GGUF's bytes (fixed cost).

  error of (tensor, tier) = metric (wsse: imatrix-weighted output error energy, default; hsse: full Hessian; sse; wrel)
                            x the first matching --weight REGEX=W (default 1; sensitivity per tensor group, see --fit)
Solvers: dp (exact up to --quantum-kib, then greedy fill of the rounding slack; default), lagrange (convex-hull
Lagrangian + greedy fill), greedy
(start cheapest, best error drop per byte first). Heuristic baselines: --rule REGEX=TAG (fixed map, first match wins;
budget only reported) and --ladder REGEX=TAG (promote in that order while the budget allows, EXL3-style priority list).

--fit: fit group sensitivities from measured KL arms. FILE lines "arm  kl_ratio_vs_base  alloc.tsv"; the base arm
(all tensors at --base-tag) has ratio 1. Model: kl_ratio - 1 = sum_g s_g * (E_g(arm) - E_g(base)) / E_base, s_g >= 0
(NNLS). Prints --weight arguments. Groups: --group REGEX=NAME (default: tensor kind, e.g. attn_qkv, ffn_down).

Outputs: --out alloc.tsv (tensor tag type bytes err), --tensor-list FILE (llama-ai4-quantize --tensor-list format),
--json summary. Feed alloc.tsv to aiq_assemble.py to build the mixed-tier GGUF from the per-tier encodes.
"""
import argparse, json, math, os, re, sys
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))

MIB = 1 << 20

# AI+ tier types -> the llama-ai4-quantize --type-/--tensor-list target name (ggml type name == target name here).
# Rates k = 2..6 of the unit-128 layout plus the private A/B control constants; anything else becomes "keep".
AI_TYPE_TARGETS = {
    "ai4": "ai4", "ai4u": "ai4u", "ai2u": "ai2u", "ai3u": "ai3u", "ai5u": "ai5u", "ai6u": "ai6u",
    "ai6uo": "ai6uo", "ai5un": "ai5un",
}


def gguf_tensor_bytes(path):
    """name -> (bytes, type name) for every tensor of a GGUF (header only)."""
    from gguf import GGUFReader
    r = GGUFReader(path, "r")
    return OrderedDict((t.name, (int(t.n_bytes), t.tensor_type.name.lower())) for t in r.tensors)


SCOPES = {
    "all": lambda n: True,
    "gpu": lambda n: n != "token_embd.weight",
    "body": lambda n: n not in ("token_embd.weight", "output.weight"),
}


def in_scope_fn(scope, scope_re):
    if scope_re:
        rx = re.compile(scope_re)
        return lambda n: rx.search(n) is not None
    return SCOPES[scope]


def read_tables(paths):
    """-> {tensor: {tag: row}} with row = dict(type, bytes, n, sse, ref, wsse, wref, hsse)."""
    tab = defaultdict(dict)
    for p in paths:
        hdr = None
        for line in open(p):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = f
                continue
            row = dict(zip(hdr, f))
            tab[row["tensor"]][row["tag"]] = dict(type=row["type"], bytes=int(row["bytes"]), n=int(row["n_weights"]),
                                                  sse=float(row["sse"]), ref=float(row["ref"]), wsse=float(row["wsse"]),
                                                  wref=float(row["wref"]), hsse=float(row["hsse"]))
    return tab


def metric_of(row, metric):
    if metric == "wrel":
        return row["wsse"] / row["wref"] if row["wref"] > 0 else -1.0
    return row[metric]


def parse_kv(items, cast=str):
    out = []
    for it in items or []:
        k, _, v = it.rpartition("=")
        if not k:
            raise SystemExit(f"bad REGEX=VALUE: {it}")
        out.append((re.compile(k), cast(v)))
    return out


def first_match(rules, name, default=None):
    for rx, v in rules:
        if rx.search(name):
            return v
    return default


class Problem:
    """Per tensor: options [(tag, bytes, err)], fixed bytes of everything else in scope."""

    def __init__(self, tab, base, args):
        self.in_scope = in_scope_fn(args.scope, args.scope_re)
        weights = parse_kv(args.weight, float)
        allow = [(rx, set(v.split(","))) for rx, v in parse_kv(args.allow)]
        self.names, self.opts, self.meta = [], [], {}
        missing = []
        for name in sorted(tab):
            if name not in base:
                raise SystemExit(f"{name} is in the error tables but not in the base GGUF")
            if not self.in_scope(name):
                continue
            ok_tags = first_match(allow, name)
            w = first_match(weights, name, 1.0)
            o = []
            tags = [t for t in tab[name] if ok_tags is None or t in ok_tags]
            if tags == [args.base_tag]:
                continue   # no choice: fixed at the base bytes (so a missing metric there is harmless, e.g. MTP layer, no imatrix)
            for tag, row in tab[name].items():
                if ok_tags is not None and tag not in ok_tags:
                    continue
                e = metric_of(row, args.metric)
                if e < 0:
                    missing.append(f"{name}/{tag}")
                    continue
                o.append((tag, row["bytes"], e * w))
                self.meta[(name, tag)] = row["type"]
            if not o:
                continue
            o.sort(key=lambda x: (x[1], x[2]))
            self.names.append(name)
            self.opts.append(o)
        if missing:
            raise SystemExit(f"metric {args.metric} missing for {len(missing)} rows (no imatrix / Hessian), e.g. {missing[:4]}; "
                             f"use --metric sse or --allow to drop them")
        chosen = set(self.names)
        self.fixed = sum(b for n, (b, _) in base.items() if self.in_scope(n) and n not in chosen)

    def min_bytes(self):
        return self.fixed + sum(o[0][1] for o in self.opts)

    def total(self, pick):
        return self.fixed + sum(self.opts[i][j][1] for i, j in enumerate(pick))

    def err(self, pick):
        return sum(self.opts[i][j][2] for i, j in enumerate(pick))


def hull(o):
    """indices of the lower convex hull of (bytes, err), bytes ascending, dominated options removed."""
    pts = []
    for j, (_, b, e) in enumerate(o):
        if pts and e >= o[pts[-1]][2]:
            continue  # more bytes, no less error
        while pts and o[pts[-1]][1] == b:
            pts.pop()
        while len(pts) >= 2:
            j1, j2 = pts[-2], pts[-1]
            b1, e1, b2, e2 = o[j1][1], o[j1][2], o[j2][1], o[j2][2]
            if (e1 - e2) * (b - b2) <= (e2 - e) * (b2 - b1):  # slope j1->j2 no steeper than j2->j
                pts.pop()
            else:
                break
        pts.append(j)
    return pts


def solve_greedy(P, budget, start=None):
    """best error-drop-per-byte upgrades along each tensor's hull, then any remaining upgrade that fits."""
    H = [hull(o) for o in P.opts]
    pick = list(start) if start else [h[0] for h in H]
    tot = P.total(pick)
    if tot > budget:
        return pick, tot
    while True:
        best, bi, bj = 0.0, -1, -1
        for i, o in enumerate(P.opts):
            cj = pick[i]
            for j in range(len(o)):
                db, de = o[j][1] - o[cj][1], o[cj][2] - o[j][2]
                if db <= 0 or de <= 0 or tot + db > budget:
                    continue
                r = de / db
                if r > best:
                    best, bi, bj = r, i, j
        if bi < 0:
            return pick, tot
        tot += P.opts[bi][bj][1] - P.opts[bi][pick[bi]][1]
        pick[bi] = bj


def solve_lagrange(P, budget):
    H = [hull(o) for o in P.opts]

    def pick_for(mu):
        return [min(h, key=lambda j: (o[j][2] + mu * o[j][1], o[j][1])) for o, h in zip(P.opts, H)]

    # the optimum multiplier is one of the hull slopes; pick_for(mu) uses fewer bytes as mu grows
    slopes = sorted({(o[h[k - 1]][2] - o[h[k]][2]) / (o[h[k]][1] - o[h[k - 1]][1]) for o, h in zip(P.opts, H) for k in range(1, len(h))})
    mus = [0.0] + slopes + [math.inf]
    lo, hi = 0, len(mus) - 1          # invariant: total(mus[hi]) <= budget (mu = inf -> cheapest hull points)
    if P.total(pick_for(0.0)) <= budget:
        hi = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if P.total(pick_for(mus[mid])) <= budget:
            hi = mid
        else:
            lo = mid
    start = pick_for(mus[hi]) if mus[hi] != math.inf else [h[0] for h in H]
    return solve_greedy(P, budget, start)


def solve_dp(P, budget, quantum):
    base_min = P.min_bytes()
    cap = (budget - base_min) // quantum
    if cap < 0:
        return [0] * len(P.opts), P.total([0] * len(P.opts))
    cap = int(cap)
    INF = np.inf
    dp = np.full(cap + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((len(P.opts), cap + 1), dtype=np.int16)
    for i, o in enumerate(P.opts):
        b0 = o[0][1]
        new = np.full(cap + 1, INF)
        ch = np.zeros(cap + 1, dtype=np.int16)
        for j, (_, b, e) in enumerate(o):
            c = -(-(b - b0) // quantum)  # ceil: never over budget
            if c > cap:
                continue
            cand = np.full(cap + 1, INF)
            cand[c:] = dp[:cap + 1 - c] + e
            better = cand < new
            new[better] = cand[better]
            ch[better] = j
        dp, choice[i] = new, ch
    k = int(np.argmin(dp))
    pick = [0] * len(P.opts)
    for i in range(len(P.opts) - 1, -1, -1):
        j = int(choice[i, k])
        pick[i] = j
        b0 = P.opts[i][0][1]
        k -= -(-(P.opts[i][j][1] - b0) // quantum)
    return pick, P.total(pick)


def solve_rules(P, rules, default_tag):
    pick = []
    for name, o in zip(P.names, P.opts):
        tag = first_match(rules, name, default_tag)
        js = [j for j, x in enumerate(o) if x[0] == tag]
        if not js:
            raise SystemExit(f"--rule/base tag {tag} has no row for {name}")
        pick.append(js[0])
    return pick, P.total(pick)


def solve_ladder(P, steps, budget, default_tag):
    pick, tot = solve_rules(P, [], default_tag)
    for rx, tag in steps:
        for i, name in enumerate(P.names):
            if not rx.search(name):
                continue
            js = [j for j, x in enumerate(P.opts[i]) if x[0] == tag]
            if not js:
                continue
            db = P.opts[i][js[0]][1] - P.opts[i][pick[i]][1]
            if tot + db <= budget:
                pick[i], tot = js[0], tot + db
    return pick, tot


def kind_of(name):
    m = re.match(r"blk\.\d+\.([^.]+)\.", name)
    return m.group(1) if m else name


def fit_calib(args, tab, base):
    groups = parse_kv(args.group)
    gname = lambda n: first_match(groups, n, kind_of(n))
    arms = []
    for line in open(args.fit):
        if line.startswith("#") or not line.strip():
            continue
        arm, ratio, path = line.split()
        alloc = {f[0]: f[1] for f in (l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#") and l.strip()) if f[0] != "tensor"}
        arms.append((arm, float(ratio), alloc))
    tab = {n: r for n, r in tab.items() if len(r) > 1}   # single-tier tensors (no choice) carry no signal
    gs = sorted({gname(n) for n in tab})
    ebase = {g: 0.0 for g in gs}
    for n, rows in tab.items():
        if args.base_tag not in rows:
            raise SystemExit(f"--fit needs the base tag {args.base_tag} for every tensor ({n})")
        ebase[gname(n)] += metric_of(rows[args.base_tag], args.metric)
    etot = sum(ebase.values())
    A, y = [], []
    for arm, ratio, alloc in arms:
        d = {g: 0.0 for g in gs}
        for n, rows in tab.items():
            tag = alloc.get(n, args.base_tag)
            d[gname(n)] += metric_of(rows[tag], args.metric) - metric_of(rows[args.base_tag], args.metric)
        A.append([d[g] / etot for g in gs])
        y.append(ratio - 1.0)
    A, y = np.array(A), np.array(y)
    keep = np.abs(A).sum(0) > 0
    s = np.zeros(len(gs))
    s[keep] = nnls(A[:, keep], y)
    res = A @ s - y
    for (arm, ratio, _), r in zip(arms, res):
        print(f"arm {arm}: measured ratio {ratio:.4f} fitted {ratio + r:.4f}")
    for g, v, k in zip(gs, s, keep):
        print(f"group {g}: s = {v:.4f}" + ("" if k else " (no arm moves this group: unconstrained, set to 0)"))
    print("weights (relative to the largest):")
    mx = s.max() if s.max() > 0 else 1.0
    rx_of = {name: rx.pattern for rx, name in groups}
    print(" ".join(f"--weight '{rx_of.get(g, 'blk[.][0-9]+[.]' + g + '[.]')}={v / mx:.4g}'" for g, v, k in zip(gs, s, keep) if k))


def nnls(A, y, iters=2000):
    """small Lawson-Hanson NNLS."""
    m, n = A.shape
    x = np.zeros(n)
    P = np.zeros(n, bool)
    w = A.T @ (y - A @ x)
    for _ in range(iters):
        if P.all() or w[~P].max(initial=-1) <= 1e-15:
            break
        j = np.argmax(np.where(P, -np.inf, w))
        P[j] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], y, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            q = P & (z <= 0)
            alpha = np.min(x[q] / (x[q] - z[q]))
            x = x + alpha * (z - x)
            P &= x > 1e-15
        w = A.T @ (y - A @ x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", required=True, help="llama-ai4-errtab TSV (repeatable)")
    ap.add_argument("--base", required=True, help="GGUF whose bytes count for tensors not in the tables")
    ap.add_argument("--base-tag", default="base", help="tier tag of the base GGUF in the tables (default base)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--target-mib", type=float)
    g.add_argument("--target-bytes", type=int)
    g.add_argument("--match-gguf", help="target = this GGUF's bytes in --scope")
    ap.add_argument("--scope", choices=sorted(SCOPES), default="body", help="tensors counted toward the target (default body = all but token_embd/output)")
    ap.add_argument("--scope-re", help="regex overriding --scope")
    ap.add_argument("--metric", choices=["wsse", "hsse", "sse", "wrel"], default="wsse")
    ap.add_argument("--weight", action="append", help="REGEX=W error multiplier (first match wins)")
    ap.add_argument("--allow", action="append", help="REGEX=TAG1,TAG2 tiers allowed for matching tensors (first match wins)")
    ap.add_argument("--solver", choices=["lagrange", "dp", "greedy"], default="dp")
    ap.add_argument("--quantum-kib", type=int, default=16, help="dp byte quantum (default 16 KiB; slack refilled greedily)")
    ap.add_argument("--rule", action="append", help="heuristic baseline: REGEX=TAG fixed map, others --base-tag")
    ap.add_argument("--ladder", action="append", help="heuristic baseline: REGEX=TAG promoted in order while it fits")
    ap.add_argument("--fit", help="fit group sensitivities from KL arms (see doc) and exit")
    ap.add_argument("--group", action="append", help="--fit grouping REGEX=NAME (default tensor kind)")
    ap.add_argument("--out", help="allocation TSV")
    ap.add_argument("--tensor-list", help="llama-ai4-quantize --tensor-list file (AI+ tiers; others keep)")
    ap.add_argument("--json", help="summary JSON")
    args = ap.parse_args(argv)

    tab = read_tables(args.table)
    base = gguf_tensor_bytes(args.base)
    if args.fit:
        fit_calib(args, tab, base)
        return 0
    P = Problem(tab, base, args)
    if args.target_mib is not None:
        budget = int(args.target_mib * MIB)
    elif args.target_bytes is not None:
        budget = args.target_bytes
    elif args.match_gguf:
        budget = sum(b for n, (b, _) in gguf_tensor_bytes(args.match_gguf).items() if P.in_scope(n))
    else:
        budget = None
    if args.rule:
        mode = "rule"
        pick, tot = solve_rules(P, parse_kv(args.rule), args.base_tag)
    else:
        if budget is None:
            raise SystemExit("need --target-mib / --target-bytes / --match-gguf (or --rule)")
        if budget < P.min_bytes():
            raise SystemExit(f"target {budget / MIB:.1f} MiB below the cheapest allocation {P.min_bytes() / MIB:.1f} MiB")
        if args.ladder:
            mode = "ladder"
            pick, tot = solve_ladder(P, parse_kv(args.ladder), budget, args.base_tag)
        else:
            mode = args.solver
            pick, tot = {"lagrange": lambda: solve_lagrange(P, budget), "greedy": lambda: solve_greedy(P, budget),
                         "dp": lambda: solve_greedy(P, budget, solve_dp(P, budget, args.quantum_kib * 1024)[0])}[args.solver]()
    err = P.err(pick)
    base_pick = [next((j for j, x in enumerate(o) if x[0] == args.base_tag), None) for o in P.opts]
    base_err = P.err(base_pick) if None not in base_pick else None
    cnt = defaultdict(int)
    for i, j in enumerate(pick):
        cnt[(kind_of(P.names[i]), P.opts[i][j][0])] += 1
    summ = dict(mode=mode, metric=args.metric, scope=args.scope_re or args.scope, budget_bytes=budget, total_bytes=tot,
                fixed_bytes=P.fixed, n_tensors=len(P.names), err=err, base_err=base_err,
                counts={f"{k}:{t}": v for (k, t), v in sorted(cnt.items())})
    print(f"{mode}: {len(P.names)} tensors, scope bytes {tot / MIB:.1f} MiB" + (f" / target {budget / MIB:.1f} MiB" if budget else "")
          + f" (fixed {P.fixed / MIB:.1f}), error {err:.6e}" + (f" = {err / base_err:.4f} x base" if base_err else ""))
    for k, v in summ["counts"].items():
        print(f"  {v:4d}  {k}")
    if args.out:
        with open(args.out + ".tmp", "w") as f:
            f.write(f"# aiq_alloc {mode} metric={args.metric} scope={summ['scope']} total={tot} budget={budget} err={err:.9e}\n")
            f.write("tensor\ttag\ttype\tbytes\terr\n")
            for i, j in enumerate(pick):
                tag, b, e = P.opts[i][j]
                f.write(f"{P.names[i]}\t{tag}\t{P.meta[(P.names[i], tag)]}\t{b}\t{e:.9e}\n")
        os.replace(args.out + ".tmp", args.out)
    if args.tensor_list:
        amap = AI_TYPE_TARGETS
        with open(args.tensor_list, "w") as f:
            f.write("# aiq_alloc tensor list for llama-ai4-quantize --tensor-list (non-AI+ tiers: keep = template type)\n")
            for i, j in enumerate(pick):
                f.write(f"{P.names[i]} {amap.get(P.meta[(P.names[i], P.opts[i][j][0])], 'keep')}\n")
    if args.json:
        json.dump(summ, open(args.json, "w"), indent=1)
    if budget is not None and tot > budget:
        print(f"WARNING: allocation {tot / MIB:.1f} MiB is over the target {budget / MIB:.1f} MiB")
        return 3 if mode not in ("rule",) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
