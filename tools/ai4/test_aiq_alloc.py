#!/usr/bin/env python3
"""AIQ3 tests: allocator solvers (no binaries) + tiny-model end to end (llama-ai4-errtab CPU encode -> error table ->
aiq_alloc -> aiq_assemble -> error table of the assembled file).

usage: test_aiq_alloc.py [--bin BUILD/bin] [--keep DIR]     (without --bin only the solver tests run)
"""
import argparse, itertools, json, os, random, re, shutil, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "gguf-py"))
import aiq_alloc  # noqa: E402
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT  # noqa: E402

PASS, FAIL = 0, 0


def check(cond, what):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {what}")


def mk_problem(opts, fixed=0):
    P = aiq_alloc.Problem.__new__(aiq_alloc.Problem)
    P.names = [f"t{i}" for i in range(len(opts))]
    P.opts = [sorted(o, key=lambda x: (x[1], x[2])) for o in opts]
    P.fixed = fixed
    return P


def brute(P, budget):
    best = (None, None)
    for pick in itertools.product(*[range(len(o)) for o in P.opts]):
        if P.total(pick) <= budget:
            e = P.err(pick)
            if best[0] is None or e < best[0] - 1e-15:
                best = (e, pick)
    return best


def test_solvers():
    rng = random.Random(20261013)
    n_opt_lag = 0
    N = 300
    for it in range(N):
        n = rng.randint(2, 7)
        opts = []
        for i in range(n):
            k = rng.randint(1, 4)
            bytes_ = sorted(rng.sample(range(10, 200), k))
            errs = sorted((rng.uniform(0.01, 10) for _ in range(k)), reverse=True)
            if rng.random() < 0.2 and k > 1:   # a dominated option (more bytes, more error)
                errs[-1] = errs[0] + 1
            opts.append([(f"x{j}", b, e) for j, (b, e) in enumerate(zip(bytes_, errs))])
        P = mk_problem(opts, fixed=rng.randint(0, 50))
        lo, hi = P.min_bytes(), P.fixed + sum(o[-1][1] for o in P.opts)
        budget = rng.randint(lo, hi + 20)
        eb, _ = brute(P, budget)
        pd, td = aiq_alloc.solve_dp(P, budget, 1)
        check(td <= budget and abs(P.err(pd) - eb) <= 1e-9 * max(1, eb), f"dp optimal (it {it}: {P.err(pd)} vs {eb})")
        pl, tl = aiq_alloc.solve_lagrange(P, budget)
        check(tl <= budget, f"lagrange within budget (it {it})")
        check(P.err(pl) >= eb - 1e-9, f"lagrange not below optimum (it {it})")
        n_opt_lag += abs(P.err(pl) - eb) <= 1e-9 * max(1, eb)
        pg, tg = aiq_alloc.solve_greedy(P, budget)
        check(tg <= budget, f"greedy within budget (it {it})")
        # quantized dp stays feasible
        pq, tq = aiq_alloc.solve_dp(P, budget, 7)
        check(tq <= budget, f"dp quantum 7 feasible (it {it})")
    print(f"solvers: lagrange+fill optimal on {n_opt_lag}/{N} random instances")
    check(n_opt_lag >= 0.85 * N, f"lagrange optimal on >= 85% of instances ({n_opt_lag}/{N})")
    # hull: dominated and non-convex points removed
    o = [("a", 10, 10.0), ("b", 20, 9.9), ("c", 30, 1.0), ("d", 40, 2.0)]
    check(aiq_alloc.hull(o) == [0, 2], f"hull {aiq_alloc.hull(o)}")
    # dp monotone in budget
    P = mk_problem([[("a", 10, 5.0), ("b", 20, 1.0)], [("a", 10, 4.0), ("b", 30, 0.5)]])
    errs = [P.err(aiq_alloc.solve_dp(P, b, 1)[0]) for b in range(20, 51)]
    check(all(x >= y for x, y in zip(errs, errs[1:])), "dp error non-increasing in budget")
    # nnls
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x = aiq_alloc.nnls(A, A @ np.array([2.0, 0.5]))
    check(np.allclose(x, [2.0, 0.5]), f"nnls exact {x}")
    x = aiq_alloc.nnls(A, np.array([1.0, -1.0, 0.0]))
    check((x >= 0).all(), f"nnls nonneg {x}")


# ---------------------------------------------------------------- tiny model end to end
TENSORS = {   # name: (ne0 = K, ne1 = M[, ne2 = E])
    "token_embd.weight": (256, 64),
    "blk.0.attn_qkv.weight": (256, 384),
    "blk.0.ssm_out.weight": (256, 256),
    "blk.0.ffn_down.weight": (512, 256),
    "blk.0.attn_norm.weight": (256,),
    "blk.1.attn_q.weight": (256, 512),
    "blk.1.ffn_gate.weight": (256, 512),
    "blk.1.ffn_up_exps.weight": (256, 128, 4),
    "output.weight": (256, 64),
}


def write_tiny(d):
    rng = np.random.default_rng(7)
    w = GGUFWriter(os.path.join(d, "bf16.gguf"), "qwen35")
    w.add_string("aiq3.test", "tiny")
    w.add_uint32("aiq3.answer", 42)
    W = {}
    for n, ne in TENSORS.items():
        shp = tuple(reversed(ne))
        a = rng.standard_normal(shp).astype(np.float32) * (0.02 if len(ne) > 1 else 1.0)
        if len(ne) > 1:   # heavy-ish tails + a few outlier columns, like real weights
            a *= np.exp(0.5 * rng.standard_normal(shp[-1])).astype(np.float32)
        bf = (a.view(np.uint32) >> 16).astype(np.uint16)
        W[n] = (bf.astype(np.uint32) << 16).view(np.float32)
        w.add_tensor(n, bf, raw_dtype=QT.BF16)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    im = GGUFWriter(os.path.join(d, "imatrix.gguf"), "")
    im.add_string("general.type", "imatrix")
    lam = {}
    for n, ne in TENSORS.items():
        if len(ne) < 2 or n == "token_embd.weight":
            continue
        E = ne[2] if len(ne) > 2 else 1
        l = rng.gamma(0.7, 1.0, size=(E, ne[0])).astype(np.float32) + 1e-3
        cnt = np.full(E, 10.0, np.float32)
        lam[n] = l
        im.add_tensor(n + ".in_sum2", (l * cnt[:, None]).astype(np.float32))
        im.add_tensor(n + ".counts", cnt.reshape(E, 1))
    im.write_header_to_file(); im.write_kv_data_to_file(); im.write_tensors_to_file(); im.close()
    return W, lam


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd), "\n", r.stdout[-3000:], r.stderr[-3000:])
    return r


def read_tab(p):
    return aiq_alloc.read_tables([p])


def test_tiny(bindir, d):
    et = os.path.join(bindir, "llama-ai4-errtab")
    W, lam = write_tiny(d)
    bf, imx = os.path.join(d, "bf16.gguf"), os.path.join(d, "imatrix.gguf")
    lin = r"^blk\..*\.weight$"
    tiers = {"base": "ai4u", "ai3": "ai3u", "ai5": "ai5u", "q8": "q8_0", "q4k": "q4_K"}
    for tag, ty in tiers.items():
        r = run([et, "--cpu-encode", ty, "--bf16", bf, "--imatrix", imx, "--re", lin, "--threads", "4", "--out", os.path.join(d, f"{tag}.gguf")])
        check(r.returncode == 0, f"cpu-encode {ty}")
    for tag, ty in tiers.items():
        rd = GGUFReader(os.path.join(d, f"{tag}.gguf"))
        types_ = {t.name: t.tensor_type.name.lower() for t in rd.tensors}
        check(types_["blk.0.attn_qkv.weight"].replace("_", "") == ty.lower().replace("_", ""), f"{tag}: attn_qkv type {types_['blk.0.attn_qkv.weight']}")
        check(types_["token_embd.weight"] == "bf16" and types_["blk.0.attn_norm.weight"] == "bf16", f"{tag}: untouched tensors stay bf16")
    tab = os.path.join(d, "table.tsv")
    r = run([et, "--bf16", bf, "--imatrix", imx, "--re", lin, "--threads", "4", "--out", tab]
            + [x for t in tiers for x in ("--cand", f"{t}={os.path.join(d, t + '.gguf')}")])
    check(r.returncode == 0, "errtab table")
    T = read_tab(tab)
    lin_names = [n for n, ne in TENSORS.items() if len(ne) > 1 and n.startswith("blk.")]
    check(sorted(T) == sorted(lin_names), f"table tensors {sorted(T)}")
    check(all(len(T[n]) == len(tiers) for n in T), "one row per tier")
    # independent numpy check of the q8_0 / q4_K rows (gguf-py dequantize)
    for tag in ("q8", "q4k"):
        rd = GGUFReader(os.path.join(d, f"{tag}.gguf"))
        for t in rd.tensors:
            if t.name not in T:
                continue
            R = gguf.dequantize(t.data, t.tensor_type).reshape(W[t.name].shape)
            e = (W[t.name].astype(np.float64) - R)
            sse, ws = (e * e).sum(), 0.0
            l = lam[t.name]
            E = l.shape[0]
            ws = sum(((e.reshape(E, -1, e.shape[-1])[i] ** 2) * l[i]).sum() for i in range(E))
            row = T[t.name][tag]
            check(abs(row["sse"] - sse) <= 1e-6 * sse, f"{tag} {t.name} sse {row['sse']} vs numpy {sse}")
            check(abs(row["wsse"] - ws) <= 1e-6 * ws, f"{tag} {t.name} wsse {row['wsse']} vs numpy {ws}")
            check(row["bytes"] == int(t.n_bytes), f"{tag} {t.name} bytes")
    # the numpy twin (aiq_errtab.py: Python port of the AI+ slab decoders) must reproduce every row of the C++ table
    tabp = os.path.join(d, "table_py.tsv")
    r = run([sys.executable, os.path.join(HERE, "aiq_errtab.py"), "--bf16", bf, "--imatrix", imx, "--re", lin, "--out", tabp]
            + [x for t in tiers for x in ("--cand", f"{t}={os.path.join(d, t + '.gguf')}")])
    check(r.returncode == 0, "aiq_errtab.py table")
    TP = read_tab(tabp)
    for n in T:
        for t in T[n]:
            a_, b_ = T[n][t], TP.get(n, {}).get(t)
            check(b_ is not None and a_["bytes"] == b_["bytes"] and a_["type"] == b_["type"]
                  and all(abs(a_[c] - b_[c]) <= 1e-5 * abs(a_[c]) + 1e-30 for c in ("sse", "ref", "wsse", "wref")),
                  f"numpy twin {n} {t}: {b_} vs C++ {a_}")
    for n in T:
        rel = {t: T[n][t]["wsse"] / T[n][t]["wref"] for t in T[n]}
        check(rel["q8"] < rel["ai5"] < rel["base"], f"{n}: error order q8 < ai5u < ai4u {rel}")
        check(0.15 < rel["ai5"] / rel["base"] < 0.45, f"{n}: ai5u/ai4u weighted rel MSE {rel['ai5'] / rel['base']:.3f} (CPU ref ~0.25)")
        check(rel["base"] < 0.02, f"{n}: ai4u weighted rel MSE {rel['base']:.4f} sane")
    # allocate between all-ai4u and all-ai5u (q8 / q4k excluded), scope = body
    a_py = os.path.join(HERE, "aiq_alloc.py")
    sz = {t: sum(T[n][t]["bytes"] for n in T) for t in tiers}
    fixed = sum(int(t.n_bytes) for t in GGUFReader(os.path.join(d, "base.gguf")).tensors if t.name not in T and t.name not in ("token_embd.weight", "output.weight"))
    target = fixed + (sz["base"] + sz["ai5"]) // 2
    outs = {}
    for solver in ("lagrange", "dp", "greedy"):
        o = os.path.join(d, f"alloc_{solver}.tsv")
        r = run([sys.executable, a_py, "--table", tab, "--base", os.path.join(d, "base.gguf"), "--target-bytes", str(target),
                 "--allow", ".*=base,ai5", "--solver", solver, "--quantum-kib", "1", "--out", o, "--tensor-list", o + ".list",
                 "--json", o + ".json"])
        check(r.returncode == 0, f"alloc {solver}")
        js = json.load(open(o + ".json"))
        outs[solver] = js
        check(js["total_bytes"] <= target, f"{solver} total {js['total_bytes']} <= {target}")
        e_all4 = sum(T[n]["base"]["wsse"] for n in T)
        e_all5 = sum(T[n]["ai5"]["wsse"] for n in T)
        check(e_all5 <= js["err"] <= e_all4, f"{solver} err between all-ai5u and all-ai4u")
        lst = [l.split() for l in open(o + ".list") if not l.startswith("#")]
        check(len(lst) == len(T) and all(t in ("ai4u", "ai5u") for _, t in lst), f"{solver} tensor list")
    check(outs["dp"]["err"] <= outs["lagrange"]["err"] * (1 + 1e-9), "dp <= lagrange")
    print(f"tiny alloc err: dp {outs['dp']['err']:.6e} lagrange {outs['lagrange']['err']:.6e} greedy {outs['greedy']['err']:.6e}")
    # rule baseline + ladder
    r = run([sys.executable, a_py, "--table", tab, "--base", os.path.join(d, "base.gguf"), "--rule", "attn=ai5", "--out", os.path.join(d, "rule.tsv")])
    rl = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(d, "rule.tsv")) if not l.startswith(("#", "tensor"))}
    check(r.returncode == 0 and rl["blk.0.attn_qkv.weight"] == "ai5" and rl["blk.0.ffn_down.weight"] == "base", "rule mode")
    r = run([sys.executable, a_py, "--table", tab, "--base", os.path.join(d, "base.gguf"), "--target-bytes", str(target),
             "--ladder", "ffn=ai5", "--ladder", "attn=ai5", "--out", os.path.join(d, "ladder.tsv"), "--json", os.path.join(d, "ladder.json")])
    check(r.returncode == 0 and json.load(open(os.path.join(d, "ladder.json")))["total_bytes"] <= target, "ladder within budget")
    # fit: two synthetic arms whose KL ratios follow known group sensitivities
    al = os.path.join(d, "alloc_dp.tsv")
    fit_in = os.path.join(d, "fit.txt")
    e_base = sum(T[n]["base"]["wsse"] for n in T)
    s_true = {"attn": 2.0, "other": 0.5}
    grp = lambda n: "attn" if "attn" in n else "other"
    arms = []
    for arm, rx in (("A", "attn"), ("B", "ffn"), ("C", ".")):
        p = os.path.join(d, f"arm{arm}.tsv")
        with open(p, "w") as f:
            f.write("tensor\ttag\n")
            for n in T:
                if re.search(rx, n):
                    f.write(f"{n}\tai5\n")
        dk = sum(s_true[grp(n)] * (T[n]["ai5"]["wsse"] - T[n]["base"]["wsse"]) for n in T if re.search(rx, n)) / e_base
        arms.append(f"{arm} {1 + dk:.9f} {p}")
    open(fit_in, "w").write("\n".join(arms) + "\n")
    r = run([sys.executable, a_py, "--table", tab, "--base", os.path.join(d, "base.gguf"), "--fit", fit_in,
             "--group", "attn=attn", "--group", ".=other", "--allow", ".*=base,ai5"])
    ok = r.returncode == 0 and "group attn: s = 2.0000" in r.stdout and "group other: s = 0.5000" in r.stdout
    check(ok, f"fit recovers sensitivities\n{r.stdout}")
    # assemble the dp allocation, verify bytes/types/KV, then re-measure the assembled file
    asm = os.path.join(HERE, "aiq_assemble.py")
    mixed = os.path.join(d, "mixed.gguf")
    r = run([sys.executable, asm, "--base", os.path.join(d, "base.gguf"), "--src", f"ai5={os.path.join(d, 'ai5.gguf')}",
             "--alloc", al, "--out", mixed])
    check(r.returncode == 0, "assemble")
    alloc = {l.split("\t")[0]: l.split("\t")[1] for l in open(al) if not l.startswith(("#", "tensor"))}
    rm, rb = GGUFReader(mixed), GGUFReader(os.path.join(d, "base.gguf"))
    r5 = GGUFReader(os.path.join(d, "ai5.gguf"))
    t5 = {t.name: t for t in r5.tensors}
    check([t.name for t in rm.tensors] == [t.name for t in rb.tensors], "assembled tensor order = base")
    for t, tb in zip(rm.tensors, rb.tensors):
        src = t5[t.name] if alloc.get(t.name) == "ai5" else tb
        check(t.tensor_type == src.tensor_type and bytes(t.data.tobytes()) == bytes(src.data.tobytes()), f"assembled bytes {t.name}")
    for k, f in rb.fields.items():
        if k.startswith("GGUF."):
            continue
        check(k in rm.fields and b"".join(p.tobytes() for p in rm.fields[k].parts) == b"".join(p.tobytes() for p in f.parts), f"KV {k} copied")
    check("aiq3.allocation" in rm.fields, "aiq3.allocation KV")
    n5 = sum(1 for v in alloc.values() if v == "ai5")
    check(0 < n5 < len(alloc), f"mixed allocation uses both tiers ({n5}/{len(alloc)})")
    # assemble twice (from an assembled base) is idempotent in content and keeps one aiq3.allocation key
    mixed2 = os.path.join(d, "mixed2.gguf")
    r = run([sys.executable, asm, "--base", mixed, "--alloc", al, "--src", f"ai5={os.path.join(d, 'ai5.gguf')}", "--out", mixed2])
    check(r.returncode == 0 and sum(1 for k in GGUFReader(mixed2).fields if k == "aiq3.allocation") == 1, "re-assemble")
    # the same tier as two per-layer overlay files gives a byte-identical assembly
    ovs = []
    for L in (0, 1):
        o = os.path.join(d, f"ai5_ov{L}.gguf")
        r = run([et, "--cpu-encode", "ai5u", "--bf16", bf, "--imatrix", imx, "--re", rf"^blk\.{L}\..*\.weight$", "--overlay-only",
                 "--threads", "4", "--out", o])
        check(r.returncode == 0 and all(t.name.startswith(f"blk.{L}.") for t in GGUFReader(o).tensors), f"overlay-only encode L{L}")
        ovs += ["--src", f"ai5={o}"]
    mixed3 = os.path.join(d, "mixed3.gguf")
    r = run([sys.executable, asm, "--base", os.path.join(d, "base.gguf"), "--alloc", al, "--out", mixed3] + ovs)
    check(r.returncode == 0 and open(mixed3, "rb").read() == open(mixed, "rb").read(), "multi-file tier assembly identical")
    tab2 = os.path.join(d, "table_mixed.tsv")
    r = run([et, "--bf16", bf, "--imatrix", imx, "--re", lin, "--cand", f"mix={mixed}", "--out", tab2])
    check(r.returncode == 0, "errtab on assembled file (C++ gguf loader accepts it)")
    T2 = read_tab(tab2)
    for n in T:
        check(T2[n]["mix"]["wsse"] == T[n][alloc[n]]["wsse"], f"assembled {n} error == chosen tier error")
    check(abs(sum(T2[n]["mix"]["wsse"] for n in T) - outs["dp"]["err"]) <= 1e-6 * outs["dp"]["err"], "assembled total error == allocator prediction")
    # overlay-style tier (only some tensors) + a missing tensor is refused
    bad = os.path.join(d, "bad.tsv")
    open(bad, "w").write("tensor\ttag\nblk.0.attn_qkv.weight\tnope\n")
    r = run([sys.executable, asm, "--base", os.path.join(d, "base.gguf"), "--alloc", bad, "--out", os.path.join(d, "bad.gguf")])
    check(r.returncode != 0 and not os.path.exists(os.path.join(d, "bad.gguf")), "unknown tier refused, no output left")


def test_menu():
    """The tensor-list menu must name every AI+ rate, so a tier of any rate survives --tensor-list."""
    m = aiq_alloc.AI_TYPE_TARGETS
    for t in ("ai4", "ai4u", "ai2u", "ai3u", "ai5u", "ai6u", "ai6uo", "ai5un"):
        check(m.get(t) == t, f"menu: {t} -> {m.get(t)}")
    check(m.get("q8_0") is None, "menu: non-AI+ types are not AI+ targets")
    print(f"menu: {len(m)} AI+ tier types")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin")
    ap.add_argument("--keep")
    a = ap.parse_args()
    test_solvers()
    test_menu()
    if a.bin:
        d = a.keep or tempfile.mkdtemp(prefix="aiq3test-")
        os.makedirs(d, exist_ok=True)
        try:
            test_tiny(a.bin, d)
        finally:
            if not a.keep:
                shutil.rmtree(d, ignore_errors=True)
    print(f"test_aiq_alloc: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
