#!/usr/bin/env python3
"""AIBEST (private tool; never ships): KL-driven per-tensor bit plan for AI+ from direct KL probes.
Private branch copy of frontier/BACKLOG_RUN_20261013/AIBEST/tools/kl_plan.py (2026-10-14).

Banks (rental packet QUEUE/running/aibest-ladder.md, stage B):
  - three full-body AI+ encodes with --scale-mode row (one scale per row; the S format stores it at k + 0.0156 bpw);
  - they follow turboderp's SC_4.00 per-tensor bits shifted -1 / 0 / +1 bit (sc3r / sc4r / sc5r), body clamped to 2..6
    and k=5 = ai5u (the shipped constant); sc5r is SC4+1, NOT EXL3's exact SC5 recipe;
  - heads ai4u / ai5u / ai6u, plus one extra head-only ai3u encode (head3).
  On the box each bank is 4 chunk overlays BANK/<bank>_c{0..3}.gguf plus BANK/<bank>_head.gguf.
Probes (KL arms; same 3 texts, generated region): base AIB_SC4R. Each slice = tensor group x 16-layer chunk.
  - PD_<G>_c<N> takes that slice from sc3r (one bit down);
  - PU_<G>_c<N> takes it from sc5r (one bit up);
  - head probes PD_HEAD (ai4u), PD2_HEAD (ai3u), PU_HEAD (ai6u).
Model: per tensor t in slice s, moving t from its SC4 tier k4 to tier k changes KL by
  dKL_t(k) = f_dir * a_s,dir * n_t * (m(k) - m(k4)),  dir = down if k < k4 else up,
where m(k) is the per-tier relative weight MSE (CPU, imatrix-weighted: 4x per bit) and n_t is the parameter count.
  - a_s,dir is fitted from the slice probe.
  - f_dir is the additivity factor: measured whole-file SC3R (or SC5R) minus SC4R, divided by the sum of the probes.
  - Head options are measured directly.
  - Tiers k4-2 (one below the sc3r bank) are extrapolated with m(k) and need a fill encode (fill.list).
Plan: minimize the predicted KL subject to GPU bytes <= each EXL3 target file. GPU bytes = template tensors that are
not re-encoded + every listed tensor at S-format bytes (rows * K/1024 * (128k+2)), token_embd excluded, as the ladder does.
The fit uses the even 200-token blocks (--fit-half even, default); `score` reports the plan arms on the odd (held-out)
blocks as well as all blocks.

usage:
  kl_plan.py exact-list REF_LIST OUT_LIST --head-bits N   copy exact reference tiers, including an explicit head
  kl_plan.py lists OUTDIR                    bank tensor lists sc3r/sc4r/sc5r/head3 (llama-ai4-quantize --tensor-list, U types)
  kl_plan.py arms                            stage-B KL arm lines with @T@ and @BANK@ placeholders (sed them on the box)
  kl_plan.py fit KLDIR [--fit-half H]        slice sensitivities + additivity checks
  kl_plan.py plan KLDIR [--fit-half H] [--out DIR]   knapsack to every target: arm lines (arms_plan.txt), plan lists
                                             (S types), aiq_assemble.py alloc tsvs, fill.list, plan.json
  kl_plan.py score KLDIR ARM:REF ...         paired block-bootstrap KL ratios, all blocks and odd (held-out) blocks
  kl_plan.py xcheck KLDIR [PLANDIR]          split-half check: fit on one half, test probes and plan arms on the other
                                             (both directions, raw and shrunk probes)
  kl_plan.py bytes                           S-format bytes of the whole banks vs the EXL3 targets (no KL needed)
  kl_plan.py k6 KLDIR                        the 6-bit scale decision (below) on its own
  plan/k6 take --k6 auto|row|group|span (default auto).
  fit/plan/k6 --shrink: shrink each probe dKL toward 0 by its noise (AIFIX 2026-10-14): dk * max(0, 1 - se^2/dk^2),
  se = paired block-bootstrap SE on the fit half. Off by default: on stage B/C xcheck it raised the even->odd probe
  correlation 0.967 -> 0.983 but lowered odd->even 0.967 -> 0.962, and moved no plan arm's held-out miss.

6-bit scales (box selftest 15:25: ai6u row scales 1.082x the weight error of group scales on blk.8.attn_qkv; ai2u-ai5u
0.978-1.022): the 70 ai6u tensors of sc5r (69 body + head) are encoded a second time with --scale-mode group
(BANK/sc5g6_c{0..3}.gguf, sc5g6_head.gguf) and scored as AIB_SC5G6 = AIB_SC5R + those overlays. Group scales cannot use
the S format, so they cost the U bytes (+0.109 bpw on those tensors, +0.0159 bpw on the sc5r file). auto picks group
only if the paired 95% CI of KL(AIB_SC5G6)/KL(AIB_SC5R) lies wholly below the equal-bytes break-even
exp(-SLOPE56 * extra bpw), SLOPE56 = 1.086 ln-KL per bpw (EXL3 SC 5.0 -> SC 6.0, measured); otherwise row (S bytes).
With group, every tier-6 option of the plan costs U bytes, comes from sc5g6, and gets its share of the measured gain
(shares by the model's 6-bit KL contribution f_up * a * n * m(6); the head's from PU_HEAD).
span (17:26, bl/aibest 92246e8de --scale-mode span: one scale per row and 1024 K, storable in S at no extra bytes):
the same 70 tensors as BANK/sc5s6_*.gguf, arm AIB_SC5S6 (packet B-3k, only if group won). Its break-even is 1.0 (equal bytes).
auto takes, among the measured modes whose CI clears their own break-even, the lowest ratio / break-even; rank-1 (AIB_SC5K6)
is informative only: rank-1 scales vary inside a 1024 span, so S cannot store them.
"""
import csv, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.environ.get("AIB_FRONTIER", "/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier")
AIB = os.environ.get("AIB_ROOT", f"{FR}/BACKLOG_RUN_20261013/AIBEST")
sys.path.insert(0, HERE)
# on a rental box: AIB_TEMPLATE=/path/T.gguf AIB_SC4_LIST=/path/aisc_SC_4.00.list (gguf_bytes.py next to this file)
TEMPLATE = os.environ.get("AIB_TEMPLATE", "/home/jake-k/qwen38-bench/models/d2_d003/Qwen3.8-27B-D003-D2.gguf")
SC4_LIST = os.environ.get("AIB_SC4_LIST", f"{AIB}/recipes/aisc_SC_4.00.list")  # turboderp SC_4.00bpw_H5 bits (k=5 named ai5un)
NP = 26_049_299_456
CUT = {"coding_gate": 3214, "agentic_gate": 5334, "rag_gate": 6334}
BLK = 200
M_K = {2: 7.34e-2, 3: 1.86e-2, 4: 4.61e-3, 5: 1.17e-3, 6: 2.97e-4}   # CPU relative MSE per tier (AIBEST STATUS 12:10)
GROUPS = {"FFG": "ffn_gate", "FFU": "ffn_up", "FFD": "ffn_down", "QKV": "attn_qkv", "GATE": "attn_gate",
          "SSO": "ssm_out", "FA": "attn_q|attn_k|attn_v|attn_output"}
CHUNKS = [range(0, 16), range(16, 32), range(32, 48), range(48, 64)]
BANKS = {"sc3r": -1, "sc4r": 0, "sc5r": +1}
HEAD_TIER = {"sc3r": 4, "sc4r": 5, "sc5r": 6, "head3": 3}
HEAD_ARM = {3: "PD2_HEAD", 4: "PD_HEAD", 6: "PU_HEAD"}
SLOPE56 = 1.086   # ln-KL per bpw between EXL3 SC 5.0 (KL 0.00102, 5.047 bpw) and SC 6.0 (0.00037, 5.981 bpw), stage A
SHRINK = {"on": False}   # --shrink turns it on
K6MODE = {"mode": "row"}   # set by k6_decide(); read by options()
K6ALT = {"group": ("AIB_SC5G6", "sc5g6", "u"), "span": ("AIB_SC5S6", "sc5s6", "s")}   # 6-bit scale alternatives to row
# EXL3 targets: GPU bytes (all tensors except token_embd) and where their per-position KL csvs live at home
KLA = f"{FR}/RELEASE_v0.6/AIQ_20261011/kl/kl.{{a}}.{{p}}.csv"
EVO = f"{FR}/RELEASE_v0.6/EXL3_VS_OURS_20261011/results/kl/kl.{{a}}.{{p}}.csv"
BITS = f"{FR}/BITS_kld_sweep/csv/{{a}}.on_atx4xs.{{p}}.csv"
TARGETS = {   # plan arm: (EXL3 file it is paired with, its GPU bytes, size source, csv pattern)
    "AIB_P30":  ("EXL3-SC3.0", 9_981_433_856, "stage A header", KLA.format(a="AIB_XSC30", p="{p}")),
    "AIB_P35":  ("EXL3-3.5", round(12.32 * (1 << 30)) - 1_350_860_800, "file GiB (2 dp) - Q8_0 embd, +-5 MB",
                 BITS.format(a="exl3_3.5", p="{p}")),
    "AIB_P40":  ("EXL3-SC4.0", 12_621 * (1 << 20), "EXL3_VS_OURS table MiB", EVO.format(a="EXL3_SC40", p="{p}")),
    "AIB_P445": ("EXL3-4.44", 14_730_172_416, "stage A header", KLA.format(a="AIB_X444", p="{p}")),
}


def tensors():
    """per listed tensor: (name, dims K x M, params, SC4 tier, group, chunk) + bytes of every template tensor not listed"""
    import gguf_bytes
    hdr = {t["name"]: t for t in gguf_bytes.parse(TEMPLATE)["tensors"]}
    k4 = {}
    for l in open(SC4_LIST):
        if l.strip() and not l.startswith("#"):
            n, ty = l.split()
            k4[n] = min(6, max(2, int(re.match(r"ai(\d)", ty).group(1))))
    k4["output.weight"] = 5
    rows = []
    for n, k in k4.items():
        K, M = hdr[n]["dims"]
        assert K % 1024 == 0, n
        if n == "output.weight":
            g = c = None
        else:
            L = int(n.split(".")[1]); kind = n.split(".")[2]
            g = next(x for x, rx in GROUPS.items() if re.fullmatch(rx, kind))
            c = next(i for i, r in enumerate(CHUNKS) if L in r)
        rows.append(dict(name=n, K=K, M=M, n=K * M, k4=k, group=g, chunk=c))
    rest = sum(t["bytes"] for n, t in hdr.items() if n not in k4 and n != "token_embd.weight")
    return rows, rest


def sbytes(t, k):
    return t["M"] * t["K"] // 1024 * (128 * k + 2)


def ubytes(t, k):
    return t["M"] * t["K"] // 128 * (16 * k + 2)


def bank_tier(t, bank):
    if t["name"] == "output.weight":
        return HEAD_TIER[bank]
    return min(6, max(2, t["k4"] + BANKS[bank]))


def cmd_lists(out):
    rows, _ = tensors()
    os.makedirs(out, exist_ok=True)
    for b in BANKS:
        with open(f"{out}/{b}.list", "w") as f:
            f.write(f"# AIBEST bank {b}: turboderp SC_4.00 bits shifted {BANKS[b]:+d} (body 2..6, k5 = ai5u), head ai{HEAD_TIER[b]}u;"
                    f" encode with --scale-mode row\n")
            f.writelines(f"{t['name']} ai{bank_tier(t, b)}u\n" for t in rows)
    with open(f"{out}/sc5r_k6.list", "w") as f:
        f.write("# AIBEST the ai6u tensors of sc5r (69 body + head), encode with --scale-mode group (sc5g6), span (sc5s6) or rank1 (sc5k6)\n")
        f.writelines(f"{t['name']} ai6u\n" for t in rows if bank_tier(t, "sc5r") == 6)
    with open(f"{out}/head3.list", "w") as f:
        f.write("# AIBEST head-only ai3u (encode from the head BF16 file with --scale-mode row)\noutput.weight ai3u\n")
    # cross-check: SC4-1 vs turboderp's own SC_3.00 list (informative)
    sc3 = {}
    for l in open(f"{AIB}/recipes/aisc_SC_3.00.list"):
        if l.strip() and not l.startswith("#"):
            n, ty = l.split(); sc3[n] = int(re.match(r"ai(\d)", ty).group(1))
    diff = sum(1 for t in rows if t["name"] != "output.weight" and sc3.get(t["name"]) != bank_tier(t, "sc3r"))
    print(f"wrote {out}/{{sc3r,sc4r,sc5r,head3}}.list; sc3r body vs turboderp SC_3.00 list: {diff} tensors differ")


def plan_template():
    import verify_bit_plan as verifier
    reader = verifier.GGUFReader(TEMPLATE, "r")
    arch = reader.get_field("general.architecture").contents()
    blocks = int(reader.get_field(f"{arch}.block_count").contents())
    nextn = reader.get_field(f"{arch}.nextn_predict_layers")
    shapes = {t.name: tuple(int(d) for d in t.shape) for t in reader.tensors}
    body_layers = blocks - (int(nextn.contents()) if nextn else 0)
    if body_layers <= 0:
        raise ValueError("template has no main-model blocks")
    return shapes, body_layers


def cmd_exact_list(reference, out, head_bits):
    import verify_bit_plan as verifier
    if head_bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError("exact-list requires --head-bits in 2..6 or 8")
    plan = verifier.read_plan(reference, head_bits)
    shapes, body_layers = plan_template()
    checked = verifier.compare(plan, plan, shapes, body_layers)
    if not checked["exact"]:
        raise ValueError("reference plan is incomplete: " + "; ".join(checked["errors"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as dest:
        dest.write(f"# Exact tiers from {reference}; head {head_bits} bits; encode with --scale-mode row\n")
        dest.writelines(f"{name} {'q8_0' if bits == 8 else f'ai{bits}u'}\n" for name, bits in plan.items())
    print(f"wrote exact reference plan {out}: {len(plan)} tensors, head {head_bits} bits")


def slice_re(g, c=None):
    lay = "[0-9]+" if c is None else "(" + "|".join(str(i) for i in CHUNKS[c]) + ")"
    return f"blk\\.{lay}\\.({GROUPS[g]})\\.weight"


def bank_files(b):
    return [f"@BANK@/{b}_c{c}.gguf" for c in range(4)] + [f"@BANK@/{b}_head.gguf"]


def cmd_arms():
    base = bank_files("sc4r")
    line = lambda name, ovs, res: f"{name};@T@;{','.join(ovs)};{',,'.join(res)}"
    for b in BANKS:
        print(line(f"AIB_{b.upper()}", bank_files(b), [".*"] * 5))
    for name, b6 in (("AIB_SC5G6", "sc5g6"), ("AIB_SC5S6", "sc5s6"), ("AIB_SC5K6", "sc5k6")):   # 6-bit scale arms (group, span, rank-1)
        print(line(name, bank_files("sc5r") + bank_files(b6), [".*"] * 10))
    for d, b in (("PD", "sc3r"), ("PU", "sc5r")):
        for g in GROUPS:
            for c in range(4):
                print(line(f"{d}_{g}_c{c}", base + [f"@BANK@/{b}_c{c}.gguf"], [".*"] * 5 + [slice_re(g)]))
    for k, name in HEAD_ARM.items():
        hb = {3: "head3", 4: "sc3r", 6: "sc5r"}[k]
        print(line(name, base + [f"@BANK@/{hb}_head.gguf"], [".*"] * 5 + ["output\\.weight"]))


# ---------------------------------------------------------------- KL reading
class KLBlocks(dict):
    def __init__(self):
        super().__init__()
        self.identity = {}


def block_sums(path_pat):
    out = KLBlocks()
    for p in CUT:
        d = {}
        path = path_pat.format(p=p)
        with open(path) as source:
            for r in csv.DictReader(source):
                pos = int(r["pos"])
                if pos in d:
                    raise ValueError(f"{path}: duplicate position {pos}")
                value, nll = float(r["kld"]), float(r["nll_base"])
                if not np.isfinite(value) or not np.isfinite(nll):
                    raise ValueError(f"{path}: non-finite score at {pos}")
                d[pos] = (value, int(r["tok"]), nll)
        pos = sorted(i for i in d if i >= CUT[p])
        if not pos or pos != list(range(CUT[p], pos[-1] + 1)):
            raise ValueError(f"{path}: missing generated positions or stride is not 1")
        out.identity[p] = tuple((i, d[i][1], d[i][2]) for i in pos)
        x = np.array([d[i][0] for i in pos]); nb = len(x) // BLK
        if nb == 0:
            raise ValueError(f"{path}: fewer than {BLK} generated scores")
        out[p] = x[:nb * BLK].reshape(nb, BLK).sum(1)
    return out


def assert_paired(a, b):
    for p in CUT:
        if a.identity[p] != b.identity[p] or len(a[p]) != len(b[p]):
            raise ValueError(f"{p}: position/token/reference NLL mismatch between KL arms")


def half_indices(bs, half):
    if half not in ("all", "even", "odd"):
        raise ValueError(f"invalid fit half: {half}")
    sel = {p: np.arange(len(bs[p])) if half == "all" else np.arange(0 if half == "even" else 1, len(bs[p]), 2) for p in CUT}
    if any(len(v) == 0 for v in sel.values()):
        raise ValueError(f"no complete blocks in {half} half")
    return sel


def kl_of(bs, half="all"):
    sel = half_indices(bs, half)
    tot = n = 0.0
    for p, v in bs.items():
        tot += v[sel[p]].sum(); n += len(sel[p]) * BLK
    return tot / n


def load_arm(d, a):
    pat = f"{d}/kl.{a}.{{p}}.csv"
    present = [os.path.exists(pat.format(p=p)) for p in CUT]
    if any(present) and not all(present):
        raise ValueError(f"{a}: incomplete KL arm in {d}")
    return block_sums(pat) if all(present) else None


def dk_se(bs, base, half, B=500):
    """dKL of a probe vs base on `half` and its paired block-bootstrap SE (same blocks for both arms, within each text)"""
    rng = np.random.default_rng(7)
    assert_paired(bs, base)
    sel = half_indices(base, half)
    n = sum(len(v) for v in sel.values()) * BLK
    dif = {p: bs[p][sel[p]] - base[p][sel[p]] for p in CUT}
    boot = [sum(dif[p][rng.integers(0, len(dif[p]), len(dif[p]))].sum() for p in CUT) / n for _ in range(B)]
    return sum(v.sum() for v in dif.values()) / n, float(np.std(boot))


def shrink(dk, se):
    return dk * max(0.0, 1.0 - se * se / (dk * dk)) if SHRINK["on"] and dk != 0 else dk


def fit(d, half):
    rows, rest = tensors()
    base = load_arm(d, "AIB_SC4R")
    if base is None:
        sys.exit(f"{d}: no AIB_SC4R csvs")
    K0 = kl_of(base, half)
    A, rep = {}, []
    pooled = {}
    for dirn, pre, b in (("down", "PD", "sc3r"), ("up", "PU", "sc5r")):
        for g in GROUPS:
            for c in range(4):
                bs = load_arm(d, f"{pre}_{g}_c{c}")
                ts = [t for t in rows if t["group"] == g and t["chunk"] == c]
                den = sum(t["n"] * (M_K[bank_tier(t, b)] - M_K[t["k4"]]) for t in ts)
                if bs is None or den == 0:
                    continue
                dk, se = dk_se(bs, base, half)
                dk = shrink(dk, se)
                A[(g, c, dirn)] = dk / den
                pooled.setdefault((g, dirn), [0.0, 0.0])
                pooled[(g, dirn)][0] += dk; pooled[(g, dirn)][1] += den
                rep.append((f"{pre}_{g}_c{c}", dk, dk / K0))
    # floor noisy slices at 10% of their group's pooled sensitivity (a probe can come out <= 0 by noise)
    for (g, c, dirn), a in list(A.items()):
        pa = pooled[(g, dirn)][0] / pooled[(g, dirn)][1]
        A[(g, c, dirn)] = max(a, 0.1 * pa) if pa > 0 else max(a, 0.0)
    head = {5: 0.0}
    for k, name in HEAD_ARM.items():
        bs = load_arm(d, name)
        if bs is not None:
            head[k] = shrink(*dk_se(bs, base, half))
    f = {}
    for dirn, b in (("down", "sc3r"), ("up", "sc5r")):
        bs = load_arm(d, f"AIB_{b.upper()}")
        pred = sum(A.get((t["group"], t["chunk"], dirn), 0.0) * t["n"] * (M_K[bank_tier(t, b)] - M_K[t["k4"]])
                   for t in rows if t["name"] != "output.weight")
        pred += head.get(HEAD_TIER[b], 0.0)
        if bs is not None:
            assert_paired(bs, base)
        meas = kl_of(bs, half) - K0 if bs is not None else None
        f[dirn] = (meas / pred) if (meas is not None and pred != 0) else 1.0
        rep.append((f"WHOLE_{b}", meas, pred))
    return dict(rows=rows, rest=rest, K0=K0, A=A, head=head, f=f, rep=rep, half=half, base=base)


def cmd_fit(d, half):
    F = fit(d, half)
    print(f"fit half={half}: KL(AIB_SC4R) = {F['K0']:.6f}")
    for name, dk, x in F["rep"]:
        if name.startswith("WHOLE"):
            print(f"  {name:12s} measured dKL {dk if dk is None else f'{dk:+.6f}'}  sum of probes {x:+.6f}")
        else:
            print(f"  {name:12s} dKL {dk:+.6f} ({100 * x:+.1f}% of base)")
    print(f"  head dKL by tier: {', '.join(f'ai{k}: {v:+.6f}' for k, v in sorted(F['head'].items()))}")
    print(f"  additivity factor f_down {F['f']['down']:.3f}, f_up {F['f']['up']:.3f} (1 = probes add up exactly)")
    return F


def options(t, F):
    """[(tier, S bytes, predicted dKL, source bank or 'fill')]"""
    if t["name"] == "output.weight":
        src = {3: "head3", 4: "sc3r", 5: "sc4r", 6: "sc5r"}
        o = [(k, sbytes(t, k), F["head"][k], src[k]) for k in sorted(F["head"])]
        if K6MODE["mode"] != "row":
            m = K6MODE["mode"]
            o = [(k, (ubytes if m == "group" else sbytes)(t, k), d - K6MODE["gain"].get("output.weight", 0.0), K6ALT[m][1])
                 if k == 6 else (k, b, d, s) for k, b, d, s in o]
        return o
    out = []
    for k in range(max(2, t["k4"] - 2), min(6, t["k4"] + 1) + 1):
        if k == t["k4"]:
            out.append((k, sbytes(t, k), 0.0, "sc4r")); continue
        dirn = "down" if k < t["k4"] else "up"
        a = F["A"].get((t["group"], t["chunk"], dirn))
        if a is None:
            continue
        dk = F["f"][dirn] * a * t["n"] * (M_K[k] - M_K[t["k4"]])
        src = "sc3r" if k == bank_tier(t, "sc3r") else "sc5r" if k == bank_tier(t, "sc5r") else "fill"
        if k == 6 and src == "sc5r" and K6MODE["mode"] != "row":
            m = K6MODE["mode"]
            out.append((k, (ubytes if m == "group" else sbytes)(t, k), dk - K6MODE["gain"].get(t["name"], 0.0), K6ALT[m][1]))
            continue
        out.append((k, sbytes(t, k), dk, src))
    return out


def solve(F, budget):
    rows = F["rows"]; opts = [options(t, F) for t in rows]
    avail = budget - F["rest"]

    def pick(lam):
        ch = [min(o, key=lambda x: x[2] + lam * x[1]) for o in opts]
        return ch, sum(x[1] for x in ch)
    lo, hi = 0.0, 1e-6
    while pick(hi)[1] > avail:
        hi *= 4
        if hi > 1e6:
            sys.exit("budget below the smallest plan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[1] > avail:
            lo = mid
        else:
            hi = mid
    ch, used = pick(hi)
    ch = list(ch)
    while True:   # spend the slack: best dKL saved per byte among upgrades that still fit
        best = None
        for i, o in enumerate(opts):
            for x in o:
                db = x[1] - ch[i][1]; dk = ch[i][2] - x[2]
                if db > 0 and dk > 0 and used + db <= avail and (best is None or dk / db > best[0]):
                    best = (dk / db, i, x)
        if best is None:
            break
        used += best[2][1] - ch[best[1]][1]; ch[best[1]] = best[2]
    return [(t, x) for t, x in zip(rows, ch)], used + F["rest"]


def compact_re(names):
    by = {}
    for n in names:
        if n == "output.weight":
            by.setdefault("output", []).append(None); continue
        L, kind = n.split(".")[1], n.split(".")[2]
        by.setdefault(kind, []).append(int(L))
    parts = []
    for kind, ls in sorted(by.items()):
        if kind == "output":
            parts.append("output\\.weight")
        else:
            parts.append(f"blk\\.({'|'.join(str(x) for x in sorted(ls))})\\.{kind}\\.weight")
    return "|".join(parts)


def k6_decide(d, F, choice="auto", B=4000):
    """6-bit scale mode for the plan (see the docstring); sets K6MODE and returns it"""
    six = [t for t in F["rows"] if bank_tier(t, "sc5r") == 6]
    extra = sum(ubytes(t, 6) - sbytes(t, 6) for t in six) * 8 / NP
    Rr = load_arm(d, "AIB_SC5R")
    info = dict(mode="row", fit_half=F["half"], extra_bpw=extra, cand={}, gain={})
    gains = {}
    for mode, (arm, bank, _) in K6ALT.items():
        be = float(np.exp(-SLOPE56 * extra)) if mode == "group" else 1.0   # span = S bytes = row bytes
        G = load_arm(d, arm)
        c = dict(arm=arm, break_even=be, ratio=None, ci=None, passes=False)
        if G is not None and Rr is not None:
            assert_paired(G, Rr)
            sel = half_indices(Rr, F["half"])
            rng = np.random.default_rng(1)
            pt = kl_of(G, F["half"]) / kl_of(Rr, F["half"])
            bs = []
            for _ in range(B):
                ix = {p: rng.choice(sel[p], len(sel[p])) for p in CUT}
                bs.append(sum(G[p][ix[p]].sum() for p in CUT) / sum(Rr[p][ix[p]].sum() for p in CUT))
            lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
            c.update(ratio=float(pt), ci=[lo, hi], passes=hi < be)
            # gain over row on the sc5r ai6 set, shared by each tensor's modelled 6-bit KL contribution
            dK = kl_of(Rr, F["half"]) - kl_of(G, F["half"])
            w = {}
            for t in six:
                if t["name"] == "output.weight":
                    h6 = F["head"].get(6)
                    if h6 is not None and h6 < 0:
                        w[t["name"]] = -h6 * M_K[6] / (M_K[5] - M_K[6])
                else:
                    a = F["A"].get((t["group"], t["chunk"], "up"))
                    if a is not None:
                        w[t["name"]] = F["f"]["up"] * a * t["n"] * M_K[6]
            tot = sum(w.values())
            gains[mode] = {n: dK * v / tot for n, v in w.items()} if tot > 0 else {}
        info["cand"][mode] = c
    if choice == "auto":
        ok = [m for m, c in info["cand"].items() if c["passes"]]
        if ok:   # lowest byte-adjusted ratio among the modes whose CI clears their own break-even
            info["mode"] = min(ok, key=lambda m: info["cand"][m]["ratio"] / info["cand"][m]["break_even"])
    else:
        if choice != "row" and info["cand"][choice]["ratio"] is None:
            sys.exit(f"--k6 {choice} needs {K6ALT[choice][0]} and AIB_SC5R csvs")
        info["mode"] = choice
    info["gain"] = gains.get(info["mode"], {})
    K6MODE.clear(); K6MODE.update(info)
    for m, c in info["cand"].items():
        r = "not measured" if c["ratio"] is None else f"{c['ratio']:.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}]"
        cost = f"+{extra:.4f} bpw on sc5r" if m == "group" else "no extra bytes (S format)"
        print(f"6-bit scales: KL({c['arm']})/KL(AIB_SC5R) = {r}; {m} costs {cost} -> break-even {c['break_even']:.4f}")
    print(f"6-bit scales: mode {info['mode']} ({choice})")
    return info


def cmd_plan(d, half, out, k6="auto"):
    F = cmd_fit(d, half)
    k6i = k6_decide(d, F, k6)
    os.makedirs(out, exist_ok=True)
    arms, fill, summary = [], {}, {}
    for name, (ref, budget, src, _) in TARGETS.items():
        plan, used = solve(F, budget)
        pred = F["K0"] + sum(x[2] for _, x in plan)
        ovs, res = bank_files("sc4r"), [".*"] * 5
        for b in ("sc3r", "sc5r", "sc5g6", "sc5s6", "head3", "fill"):
            for c in list(range(4)) + ["head"]:
                sel = [t["name"] for t, x in plan if x[3] == b and (t["chunk"] == c or (c == "head" and t["chunk"] is None))]
                if sel:
                    ovs.append(f"@BANK@/{b}_{'c' + str(c) if c != 'head' else 'head'}.gguf"); res.append(compact_re(sel))
        for t, x in plan:
            if x[3] == "fill":
                fill[t["name"]] = min(fill.get(t["name"], 9), x[0])
        arms.append(f"{name};@T@;{','.join(ovs)};{',,'.join(res)}")
        with open(f"{out}/{name}.list", "w") as f:
            f.write(f"# {name}: KL plan (fit half {half}) at GPU bytes <= {ref} ({budget:,} B, {src}); S-format types"
                    f" except group-scaled 6-bit (ai6u, from sc5g6; repack the other banks only)\n")
            f.writelines(f"{t['name']} ai{x[0]}{'u' if x[3] == 'sc5g6' else 's'} {x[3]}\n" for t, x in plan)
        with open(f"{out}/{name}.tsv", "w") as f:   # aiq_assemble.py --alloc (tags = bank names; --src TAG=BANK/<bank>_c*.gguf)
            f.write(f"# {name} KL plan, tags = AIBEST bank overlays\ntensor\ttag\n")
            f.writelines(f"{t['name']}\t{x[3]}\n" for t, x in plan)
        hist = {}
        for t, x in plan:
            hist[x[0]] = hist.get(x[0], 0) + t["n"]
        body_bits = sum(x[0] * t["n"] for t, x in plan if t["name"] != "output.weight") / sum(
            t["n"] for t in F["rows"] if t["name"] != "output.weight")
        summary[name] = dict(ref=ref, budget=budget, bytes=used, bpw=used * 8 / NP, pred_kl=pred, pred_vs_sc4r=pred / F["K0"],
                             body_bits=body_bits, head=[x[0] for t, x in plan if t["name"] == "output.weight"][0],
                             params_by_tier={k: v / 1e9 for k, v in sorted(hist.items())},
                             fill_tensors=sum(1 for _, x in plan if x[3] == "fill"))
        print(f"{name}: <= {ref} {budget:,} B -> {used:,} B ({used * 8 / NP:.3f} bpw), body {body_bits:.3f} bits, head ai{summary[name]['head']}, "
              f"predicted KL {pred:.6f} = {pred / F['K0']:.3f}x SC4R, fill tensors {summary[name]['fill_tensors']}")
    with open(f"{out}/arms_plan.txt", "w") as f:
        f.write("\n".join(arms) + "\n")
    with open(f"{out}/fill.list", "w") as f:
        f.write("# AIBEST fill encodes (tiers below the sc3r bank), --scale-mode row, out BANK/fill_c{0..3}.gguf\n")
        f.writelines(f"{n} ai{k}u\n" for n, k in sorted(fill.items()))
    k6j = {k: v for k, v in k6i.items() if k != "gain"}
    json.dump(dict(fit_half=half, shrink=SHRINK["on"], K0=F["K0"], f=F["f"], k6=k6j, plans=summary), open(f"{out}/plan.json", "w"), indent=1)
    print(f"wrote {out}/arms_plan.txt, {out}/AIB_P*.list, {out}/fill.list ({len(fill)} tensors), {out}/plan.json")


def cmd_score(d, pairs, B=4000):
    rng = np.random.default_rng(1)
    for pr in pairs:
        a, r = pr.split(":")
        A = load_arm(d, a)
        R = load_arm(d, r)
        if R is None:
            R = block_sums(next(v[3] for v in TARGETS.values() if v[0] == r))
        assert_paired(A, R)
        for half in ("all", "odd"):
            sel = {p: (np.arange(len(A[p])) if half == "all" else np.arange(1, len(A[p]), 2)) for p in CUT}
            pt = sum(A[p][sel[p]].sum() for p in CUT) / sum(R[p][sel[p]].sum() for p in CUT)
            bs = []
            for _ in range(B):
                ix = {p: rng.choice(sel[p], len(sel[p])) for p in CUT}
                bs.append(sum(A[p][ix[p]].sum() for p in CUT) / sum(R[p][ix[p]].sum() for p in CUT))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{a}/{r} [{half} blocks] {pt:.3f} [{lo:.3f},{hi:.3f}]")


def pred_dkl(F, plan):
    """model dKL vs SC4R of a fixed assignment {tensor: tier} (row 6-bit scales)"""
    tot = 0.0
    for t in F["rows"]:
        k = plan.get(t["name"], t["k4"])
        if t["name"] == "output.weight":
            tot += F["head"].get(k, 0.0)
        elif k != t["k4"]:
            dirn = "down" if k < t["k4"] else "up"
            tot += F["f"][dirn] * F["A"].get((t["group"], t["chunk"], dirn), 0.0) * t["n"] * (M_K[k] - M_K[t["k4"]])
    return tot


def cmd_xcheck(d, plandir):
    """fit on one half, test on the other: every probe's dKL, and the plan arms' KL ratio vs SC4R (AIFIX 2026-10-14)"""
    plans = {}
    for fn in sorted(os.listdir(plandir)) if plandir else []:
        if re.fullmatch(r"AIB_P\w+\.list", fn):
            plans[fn[:-5]] = {l.split()[0]: int(l.split()[1][2]) for l in open(f"{plandir}/{fn}") if l.strip() and l[0] != "#"}
    for fh, th in (("even", "odd"), ("odd", "even")):
        for sh in (False, True):
            SHRINK["on"] = sh
            F = fit(d, fh); base = F["base"]
            pr, me = [], []
            for dirn, pre, b in (("down", "PD", "sc3r"), ("up", "PU", "sc5r")):
                for g in GROUPS:
                    for c in range(4):
                        bs = load_arm(d, f"{pre}_{g}_c{c}")
                        if bs is None or (g, c, dirn) not in F["A"]:
                            continue
                        ts = [t for t in F["rows"] if t["group"] == g and t["chunk"] == c]
                        pr.append(F["A"][(g, c, dirn)] * sum(t["n"] * (M_K[bank_tier(t, b)] - M_K[t["k4"]]) for t in ts))
                        me.append(dk_se(bs, base, th)[0])
            pr, me = np.array(pr), np.array(me)
            top = np.argsort(pr)[:8]   # the 8 probes the fit half rates as the biggest KL cuts (what a planner buys first)
            print(f"fit {fh} -> test {th}, shrink {'on ' if sh else 'off'}: {len(pr)} probes, corr {np.corrcoef(pr, me)[0, 1]:.3f}, "
                  f"mean |pred - test| {np.abs(pr - me).mean():.6f}; top-8 cuts predicted {pr[top].sum():+.6f}, "
                  f"test half {me[top].sum():+.6f}")
            K0t = kl_of(base, th)
            for name, plan in plans.items():
                bs = load_arm(d, name)
                if bs is not None:
                    assert_paired(bs, base)
                meas = f"{kl_of(bs, th) / K0t:.3f}" if bs is not None else "n/a"
                print(f"    {name}: predicted {(F['K0'] + pred_dkl(F, plan)) / F['K0']:.3f}x SC4R, test half {meas}x")
    SHRINK["on"] = False


def cmd_bytes():
    rows, rest = tensors()
    print(f"template tensors kept (not re-encoded, no token_embd): {rest:,} B")
    for b in BANKS:
        s = rest + sum(sbytes(t, bank_tier(t, b)) for t in rows)
        u = rest + sum(ubytes(t, bank_tier(t, b)) for t in rows)
        print(f"  {b}: S format {s:,} B ({s * 8 / NP:.3f} bpw); as encoded (U, per-128 fp16 kept) {u:,} B ({u * 8 / NP:.3f} bpw)")
    for name, (ref, budget, src, _) in TARGETS.items():
        print(f"  target {name}: {ref} {budget:,} B ({budget * 8 / NP:.3f} bpw; {src})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--shrink" in a:
        SHRINK["on"] = True; a.remove("--shrink")
    opt = lambda k, dflt: a[a.index(k) + 1] if k in a else dflt
    if not a:
        print(__doc__)
    elif a[0] == "exact-list":
        if len(a) < 3:
            sys.exit("exact-list requires REF_LIST OUT_LIST --head-bits N")
        try:
            cmd_exact_list(a[1], a[2], int(opt("--head-bits", "0")))
        except ValueError as error:
            sys.exit(str(error))
    elif a[0] == "lists":
        cmd_lists(a[1])
    elif a[0] == "arms":
        cmd_arms()
    elif a[0] == "fit":
        cmd_fit(a[1], opt("--fit-half", "even"))
    elif a[0] == "plan":
        cmd_plan(a[1], opt("--fit-half", "even"), opt("--out", f"{AIB}/recipes/plan"), opt("--k6", "auto"))
    elif a[0] == "k6":
        k6_decide(a[1], fit(a[1], opt("--fit-half", "even")), opt("--k6", "auto"))
    elif a[0] == "score":
        cmd_score(a[1], [x for x in a[2:] if ":" in x])
    elif a[0] == "xcheck":
        cmd_xcheck(a[1], a[2] if len(a) > 2 and not a[2].startswith("--") else None)
    elif a[0] == "bytes":
        cmd_bytes()
    else:
        print(__doc__)
