#!/usr/bin/env python3
"""Figures for the v0.2 release write-up.

Reads the raw CSV/JSON produced by the S1 speed program and writes every PNG
into img/ next to this script. Re-runnable. If an input file is missing (for
example an arm that is still running) the series is skipped with a printed
note instead of crashing.

Run:  python3 make_figures.py
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

DPI = 150
SKIPPED = []
WRITTEN = []

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
})

# One color per configuration, held constant across every figure.
# Palette slots validated with the data-viz validator in this order:
#   #2a78d6, #eda100, #4a3aa7, #eb6834, #1baf7a
C_STOCK = "#2a78d6"   # blue    stock TurboQuant+
C_V01OFF = "#eda100"  # yellow  v0.1 with the fused MMA path off
C_V01 = "#4a3aa7"     # violet  v0.1 as documented
C_W7 = "#eb6834"      # orange  W7 kernels
C_V02 = "#1baf7a"     # aqua    v0.2
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

# ---------------------------------------------------------------- loaders

def note_skip(what, path):
    msg = "SKIP  %s: missing %s" % (what, path)
    print(msg)
    SKIPPED.append(msg)


def read_csv_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def series5k(relpath, label):
    """Return (tokens, cum_tok_s, window_tok_s) from a *_per5k.csv."""
    path = os.path.join(ROOT, relpath)
    rows = read_csv_rows(path)
    if rows is None:
        note_skip(label, path)
        return None
    tok = [int(r["tokens"]) for r in rows]
    cum = [float(r["cum_tok_s"]) for r in rows]
    win = [float(r["chk_tok_s"]) for r in rows]
    return tok, cum, win


def series1k(relpath, label):
    path = os.path.join(ROOT, relpath)
    rows = read_csv_rows(path)
    if rows is None:
        note_skip(label, path)
        return None
    tok = [int(r["tokens"]) for r in rows]
    cum = [float(r["cum_tok_s"]) for r in rows]
    win = [float(r["window_tok_s"]) for r in rows]
    return tok, cum, win


def gputrace(relpath, label):
    path = os.path.join(ROOT, relpath)
    rows = read_csv_rows(path)
    if rows is None:
        note_skip(label, path)
        return None
    t = [float(r["t"]) / 60.0 for r in rows]
    pw = [float(r["power_w"]) for r in rows]
    mhz = [float(r["sm_mhz"]) for r in rows]
    return t, pw, mhz


def save(fig, name):
    out = os.path.join(IMG, name)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote %s" % out)
    WRITTEN.append(name)


def kfmt(v, _pos=None):
    return "%gK" % (v / 1000.0)


def endlabel(ax, x, y, text, color, dx=1400, dy=0.0, va="center"):
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=8.5, va=va, ha="left", annotation_clip=False)


def endlabels(ax, items, dx=1600, gap=0.9):
    """items: list of (x, y, text, color). Spread labels that would collide."""
    items = sorted(items, key=lambda it: it[1])
    ys = [it[1] for it in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (x, y, text, color), yy in zip(items, ys):
        ax.annotate(text, xy=(x, y), xytext=(x + dx, yy), color=color,
                    fontsize=8.5, va="center", ha="left", annotation_clip=False)


def below_legend(ax, ncol=3):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol)


# Arms of the 100K generation run at temperature 1.0.
TEMP1_ARMS = [
    ("stock TurboQuant+", "W11/tps_100k/stock_per5k.csv", "W11/tps_100k/stock_per1k.csv",
     "W11/tps_100k/stock_gpu.csv", C_STOCK, "-"),
    ("v0.1, fused MMA off", "W11/tps_100k/checkpoint_per5k.csv", "W11/tps_100k/checkpoint_per1k.csv",
     "W11/tps_100k/checkpoint_gpu.csv", C_V01OFF, "-"),
    ("v0.1 as documented", "W13/tps_100k_fused/checkpoint_fused_per5k.csv",
     "W13/tps_100k_fused/checkpoint_fused_per1k.csv",
     "W13/tps_100k_fused/checkpoint_fused_gpu.csv", C_V01, "-"),
    ("W7 kernels only", "W13/tps_100k_fused/w7_fused_per5k.csv",
     "W13/tps_100k_fused/w7_fused_per1k.csv",
     "W13/tps_100k_fused/w7_fused_gpu.csv", C_W7, "-"),
    ("v0.2", "W11/tps_100k/latest_per5k.csv", "W11/tps_100k/latest_per1k.csv",
     "W11/tps_100k/latest_gpu.csv", C_V02, "-"),
]

# Arms of the 100K generation run at temperature 0 (greedy). The last entry is
# the arm that may still be running when this script is executed.
TEMP0_ARMS = [
    ("stock TurboQuant+, greedy", "W12/tps_100k_greedy/stock_per5k.csv", C_STOCK, "-"),
    ("v0.1 fused MMA off, greedy", "W12/tps_100k_greedy/checkpoint_per5k.csv", C_V01OFF, "-"),
    ("v0.1 as documented, greedy", "W13/tps_100k_greedy_fused/checkpoint_fused_per5k.csv", C_V01, "-"),
    ("v0.2, greedy", "W12/tps_100k_greedy/latest_per5k.csv", C_V02, "-"),
    ("v0.2 + GPU greedy verifier", "W12/tps_100k_greedy_w12/latest_w12_per5k.csv", C_V02, "--"),
]


# ------------------------------------------------------- fig01 cumulative t1

def fig01():
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    ends = []
    for label, p5, _p1, _g, color, ls in TEMP1_ARMS:
        s = series5k(p5, "fig01 " + label)
        if s is None:
            continue
        tok, cum, _ = s
        ax.plot(tok, cum, color=color, linestyle=ls, label=label)
        ends.append((tok[-1], cum[-1], "%.1f" % cum[-1], color))
    if not ends:
        return
    endlabels(ax, ends)
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("cumulative tokens/s")
    ax.set_title("Cumulative decode rate over 102,400 generated tokens, temperature 1.0")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlim(0, 114000)
    below_legend(ax, 3)
    save(fig, "fig01_cumulative_temp1.png")


# ------------------------------------------------------- fig02 5K window t1

def fig02():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    drew = 0
    for label, p5, _p1, _g, color, ls in TEMP1_ARMS:
        s = series5k(p5, "fig02 " + label)
        if s is None:
            continue
        tok, _, win = s
        ax.plot(tok, win, color=color, linestyle=ls, label=label)
        drew += 1
    if not drew:
        return
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("tokens/s over the preceding 5,000 tokens")
    ax.set_title("Instantaneous decode rate, 5,000-token windows, temperature 1.0")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    below_legend(ax, 3)
    save(fig, "fig02_window5k_temp1.png")


# ------------------------------------------------------- fig03 1K window t1

def fig03():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    drew = 0
    for label, _p5, p1, _g, color, ls in TEMP1_ARMS:
        s = series1k(p1, "fig03 " + label)
        if s is None:
            continue
        tok, _, win = s
        ax.plot(tok, win, color=color, linestyle=ls, linewidth=0.9, alpha=0.9,
                label=label)
        drew += 1
    if not drew:
        return
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("tokens/s over the preceding 1,000 tokens")
    ax.set_title("Instantaneous decode rate, 1,000-token windows, temperature 1.0")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    save(fig, "fig03_window1k_temp1.png")


# ------------------------------------------------------- fig04 ratios

def fig04():
    base_stock = series5k("W11/tps_100k/stock_per5k.csv", "fig04 stock")
    base_v01 = series5k("W13/tps_100k_fused/checkpoint_fused_per5k.csv", "fig04 v0.1")
    v02 = series5k("W11/tps_100k/latest_per5k.csv", "fig04 v0.2")
    if v02 is None or (base_stock is None and base_v01 is None):
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    tok, cum, _ = v02
    for base, color, label in ((base_stock, C_STOCK, "v0.2 / stock TurboQuant+"),
                               (base_v01, C_V01, "v0.2 / v0.1 as documented")):
        if base is None:
            continue
        bt, bc, _ = base
        n = min(len(tok), len(bt))
        ratio = [cum[i] / bc[i] for i in range(n)]
        ax.plot(tok[:n], ratio, color=color, label=label)
        endlabel(ax, tok[n - 1], ratio[n - 1], "%.2fx" % ratio[n - 1], color)
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("ratio of cumulative tokens/s")
    ax.set_title("How far ahead v0.2 is at every 5,000-token checkpoint, temperature 1.0")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlim(0, 114000)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.975), hi)
    ax.text(2000, 1.0, "1.00, no change", color=MUTED, fontsize=7.5,
            va="bottom", ha="left")
    below_legend(ax, 2)
    save(fig, "fig04_ratio_temp1.png")


# ------------------------------------------------------- fig05 cumulative t0

def fig05():
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    ends = []
    for label, p5, color, ls in TEMP0_ARMS:
        s = series5k(p5, "fig05 " + label)
        if s is None:
            continue
        tok, cum, _ = s
        ax.plot(tok, cum, color=color, linestyle=ls, label=label)
        ends.append((tok[-1], cum[-1], "%.1f" % cum[-1], color))
    if not ends:
        return
    endlabels(ax, ends, gap=1.6)
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("cumulative tokens/s")
    ax.set_title("Greedy decoding, temperature 0, MTP depth 4 (not the headline configuration)")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlim(0, 114000)
    below_legend(ax, 2)
    save(fig, "fig05_cumulative_temp0.png")


# ------------------------------------------------------- fig06 real prompts

REAL_ARMS = [("map_d3", "64K shortlist, depth 3", C_V02),
             ("map_d4", "64K shortlist, depth 4", C_W7),
             ("hot_d3", "64K + 6K adaptive tail, depth 3", C_V01)]
DOMAIN_LABEL = {"tb": "terminal-bench", "scicode": "SciCode",
                "chat": "chat", "gdpval": "GDPval"}


def load_real_accept():
    path = os.path.join(ROOT, "W11/real_accept/results.jsonl")
    if not os.path.exists(path):
        note_skip("fig06 real-prompt pool", path)
        return None
    agg = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["arm"], r["domain"])
            a = agg.setdefault(k, [0, 0.0, 0, 0])
            a[0] += r["predicted_n"]
            a[1] += r["predicted_ms"]
            a[2] += r["draft_n"]
            a[3] += r["draft_n_accepted"]
    return agg


def fig06():
    agg = load_real_accept()
    if not agg:
        return
    domains = ["tb", "scicode", "chat", "gdpval"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    width = 0.26
    xs = list(range(len(domains)))
    for i, (arm, label, color) in enumerate(REAL_ARMS):
        off = (i - 1) * width
        vals, accs = [], []
        for d in domains:
            a = agg.get((arm, d))
            vals.append(a[0] / (a[1] / 1000.0) if a else 0.0)
            accs.append(a[3] / a[2] if a else 0.0)
        bars = ax.bar([x + off for x in xs], vals, width * 0.92, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate("%.1f" % v, (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.0, color=INK2)
        ax2.bar([x + off for x in xs], accs, width * 0.92, color=color)
        for x, v in zip(xs, accs):
            ax2.annotate("%.2f" % v, (x + off, v), ha="center", va="bottom",
                         fontsize=7.0, color=INK2)
    for a, ttl, ylab in ((ax, "Decode rate on real prompts", "tokens/s"),
                         (ax2, "Draft acceptance on the same runs", "accepted / drafted")):
        a.set_xticks(xs)
        a.set_xticklabels([DOMAIN_LABEL[d] for d in domains], fontsize=8.6)
        a.set_title(ttl, fontsize=10.5)
        a.set_ylabel(ylab)
        a.grid(axis="x", visible=False)
    ax.set_ylim(0, 98)
    ax2.set_ylim(0, 0.80)
    ax.legend(loc="upper center", bbox_to_anchor=(1.08, -0.13), ncol=3)
    fig.suptitle("Real prompts, four domains, up to 3,072 generated tokens each, temperature 1.0",
                 y=1.02, fontsize=11.5)
    save(fig, "fig06_real_prompt_domains.png")


# ------------------------------------------------------- fig07 depth 3 vs 4

# Values from <S1>/W4/ANALYSIS.md stages 2 and 3 (temperature 0, exact, n=3 per arm).
# The temperature-1.0 pool comes from <S1>/W11/real_accept/results.jsonl and the
# RAG chain from gate G4.6 in <S1>/STATE.yaml (content-confounded across depths).
# and from W11's real-prompt pool / rag chain re-test at temperature 1.0.
DEPTH_T0_FULL = [("coding\n104K ctx", 75.59, 76.27), ("agentic\n72K ctx", 66.55, 67.01)]
DEPTH_T0_MAP = [("coding\n104K ctx", 74.73, 78.51), ("agentic\n72K ctx", 69.89, 70.97),
                ("RAG\n144K ctx", 52.42, 61.32)]
DEPTH_T1_MAP = [("real prompts\npooled", 78.11, 76.69), ("RAG chain\n140K ctx", 67.12, 59.94)]


def _depth_panel(ax, data, title):
    xs = list(range(len(data)))
    width = 0.34
    for i, (color, label, idx) in enumerate(((C_V02, "depth 3", 1), (C_W7, "depth 4", 2))):
        vals = [row[idx] for row in data]
        off = (i - 0.5) * width
        bars = ax.bar([x + off for x in xs], vals, width * 0.9, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate("%.1f" % v, (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([row[0] for row in data], fontsize=8.4)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("tokens/s")
    ax.set_ylim(0, 92)
    ax.grid(axis="x", visible=False)


def fig07():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3, 2]})
    _depth_panel(axes[0], DEPTH_T0_FULL, "temp 0, full draft head")
    _depth_panel(axes[1], DEPTH_T0_MAP, "temp 0, 64K shortlist head")
    _depth_panel(axes[2], DEPTH_T1_MAP, "temp 1.0, 64K shortlist head")
    for a in axes[1:]:
        a.set_ylabel("")
    axes[0].legend(loc="upper left")
    fig.suptitle("MTP draft depth 3 against depth 4: the fixtures say 4, real traffic says 3",
                 y=1.03, fontsize=11.5)
    save(fig, "fig07_depth3_vs_4.png")


# ------------------------------------------------------- fig08 verify census

# From <frontier>/D5_atx4xs_census/ANALYSIS.md (one level above <S1>), coding
# fixture at 100K context, MTP-3,
# temperature 1.0, main 26e7bc523, GGML_Q8_TURBO3_MMA_FUSED=1.
CENSUS = [
    ("body matrix-vector kernels", 21.8, "left alone"),
    ("grouped verify attention", 9.5, "W1 loader, W6 staging"),
    ("singleton attention", 6.0, "W1 loader, W6 staging"),
    ("copies, norms, quantize, GDN", 5.9, "left alone"),
    ("output head, drafter (width 1)", 3.7, "W3 64K shortlist"),
    ("output head, verify (width 4)", 1.4, "left alone"),
    ("width-2 launches", 0.7, "left alone"),
    ("GPU idle between rounds", 6.3, "W2a buffer, W12 verifier"),
]


def fig08():
    fig, ax = plt.subplots(figsize=(7.9, 3.9))
    labels = [c[0] for c in CENSUS][::-1]
    vals = [c[1] for c in CENSUS][::-1]
    tags = [c[2] for c in CENSUS][::-1]
    colors = []
    for lab in labels:
        if "head, drafter" in lab:
            colors.append(C_V02)
        elif "attention" in lab:
            colors.append(C_W7)
        elif "idle" in lab:
            colors.append(C_V01)
        else:
            colors.append("#c3c2b7")
    ys = list(range(len(labels)))
    ax.barh(ys, vals, 0.62, color=colors)
    total = sum(vals)
    for y, v, t in zip(ys, vals, tags):
        ax.annotate("%.1f ms   %s" % (v, t), (v + 0.4, y), va="center",
                    fontsize=8.2, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 34)
    ax.set_xlabel("milliseconds per verify pass (56.4 ms wall, 49.4 ms of it kernels)")
    ax.set_title("Where one verify pass went before this round, and what v0.2 aimed at",
                 fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "fig08_verify_pass_census.png")


# ------------------------------------------------------- fig09 coverage

DOMAIN_ORDER = ["chat", "gdpval", "scicode", "tb", "rag_analysis",
                "document_generation", "coding", "agentic", "creative",
                "mixed", "stem_qa"]
COV_ARMS = [("prefix64k", "64K by raw frequency", C_STOCK),
            ("atx_32k", "32K shortlist", C_V01OFF),
            ("atx_64k", "64K shortlist (shipped)", C_V02)]


def fig09():
    path = os.path.join(ROOT, "W3/data/report.json")
    rep = read_json(path)
    if rep is None or "eval_by_domain" not in rep:
        note_skip("fig09 shortlist coverage", path)
        return
    per = rep["eval_by_domain"]
    doms = [d for d in DOMAIN_ORDER if d in per] + \
           [d for d in sorted(per) if d not in DOMAIN_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    ys = list(range(len(doms)))[::-1]
    for ax, metric, title in ((axes[0], "coverage", "single tokens inside the shortlist"),
                              (axes[1], "win3_full", "all three drafted tokens inside it")):
        for y, dom in zip(ys, doms):
            vals = [per[dom][arm][metric] * 100.0 for arm, _l, _c in COV_ARMS]
            ax.plot([min(vals), max(vals)], [y, y], color=MUTED, linewidth=1.1, zorder=1)
            for (arm, label, color), v in zip(COV_ARMS, vals):
                ax.scatter([v], [y], s=46, color=color, zorder=3,
                           label=label if y == ys[0] else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("percent of held-out tokens")
        ax.set_xlim(76, 102)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([d.replace("_", " ") for d in doms])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.06, -0.13), ncol=3)
    fig.suptitle("What the draft head can still propose after the shortlist, by domain",
                 y=1.02, fontsize=11.5)
    save(fig, "fig09_shortlist_coverage.png")


# ------------------------------------------------------- fig10 VRAM at 220K

VRAM_SOURCES = [
    ("v0.1 (depth 3, no shortlist)",
     "W7/raw/capacity/W7_220k_atx4xs_c225280_p220000_n16_b4096_ub1024.json", C_V01),
    ("v0.2 (depth 4 + 64K shortlist)",
     "W9/raw/capacity/W9_220k_d4_map_atx4xs_c225280_p220000_n16_b4096_ub1024.json", C_V02),
]


def fig10():
    bars = []
    for label, rel, color in VRAM_SOURCES:
        path = os.path.join(ROOT, rel)
        d = read_json(path)
        if d is None:
            note_skip("fig10 " + label, path)
            continue
        bars.append((label, d["ready_vram_mib"], d["peak_vram_mib"], color))
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = list(range(len(bars)))
    for y, (label, ready, peak, color) in zip(ys, bars):
        ax.barh(y, peak, 0.46, color=color)
        ax.annotate("%s MiB peak, %s at ready" % (format(peak, ","), format(ready, ",")),
                    (peak - 320, y), va="center", ha="right", fontsize=8.6,
                    color="#fcfcfb", weight="bold")
    ax.axvline(24564, color="#d03b3b", linewidth=1.5, linestyle="--")
    ax.annotate("24 GB card, 24,564 MiB", (24300, len(bars) - 0.42), color="#d03b3b",
                fontsize=8.4, ha="right", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars])
    ax.set_xlim(0, 26200)
    ax.set_ylim(-0.6, len(bars) - 0.25)
    ax.set_xlabel("VRAM in use during a 220,000-token prompt, MiB")
    ax.set_title("A populated 220K context still fits after v0.2", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_vram_220k.png")


# ------------------------------------------------------- fig11 power / clock

def _rolling_median(vals, w=9):
    out = []
    half = w // 2
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        window = sorted(vals[lo:hi])
        out.append(window[len(window) // 2])
    return out


def fig11():
    fig, (axp, axc) = plt.subplots(2, 1, figsize=(6.9, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.5]})
    drew = 0
    for label, _p5, _p1, gp, color, ls in TEMP1_ARMS:
        s = gputrace(gp, "fig11 " + label)
        if s is None:
            continue
        t, pw, mhz = s
        keep = [i for i in range(len(t)) if t[i] > 0.6]
        t = [t[i] for i in keep]
        pw = [pw[i] for i in keep]
        mhz = _rolling_median([mhz[i] for i in keep])
        axp.plot(t, pw, color=color, linestyle=ls, linewidth=1.2, label=label)
        axc.plot(t, mhz, color=color, linestyle=ls, linewidth=1.4)
        drew += 1
    if not drew:
        return
    axp.set_ylabel("board power, W")
    axp.set_ylim(336, 366)
    axp.set_yticks([340, 345, 350, 355])
    axp.axhline(350, color=MUTED, linestyle=":", linewidth=1.0)
    axp.annotate("350 W limit", (0.8, 356.5), fontsize=8, color=MUTED, va="bottom")
    axp.set_title("Every arm ran pinned to the same 350 W limit in the same clock band",
                  fontsize=11)
    axc.set_ylabel("SM clock, MHz (9-sample median)")
    axc.set_xlabel("minutes into the run")
    axc.set_ylim(1550, 1900)
    handles, labels = axp.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False)
    save(fig, "fig11_power_clock.png")


# ------------------------------------------------------- fig12 accept by pos

ACCEPT_SOURCES_T1 = [
    ("stock, v0.1 and W7 (identical stream)", "W13/tps_100k_fused/summary.json",
     "checkpoint_fused", C_V01),
    ("v0.2", "W11/tps_100k/summary.json", "latest", C_V02),
]
ACCEPT_SOURCES_T0 = [
    ("stock and v0.1, greedy", "W12/tps_100k_greedy/summary.json", "stock", C_V01),
    ("v0.2, greedy", "W12/tps_100k_greedy/summary.json", "latest", C_V02),
]


def _accept_panel(ax, sources, title, npos):
    width = 0.36
    xs = list(range(npos))
    for i, (label, rel, key, color) in enumerate(sources):
        path = os.path.join(ROOT, rel)
        d = read_json(path)
        if d is None or key not in d:
            note_skip("fig12 " + label, path)
            continue
        acc = d[key].get("accepted_per_pos") or {}
        drafted = d[key]["timings"]["draft_n"]
        rounds = drafted / float(npos)
        vals = [acc.get(str(p), 0.0) / rounds for p in range(npos)]
        off = (i - 0.5) * width
        bars = ax.bar([x + off for x in xs], vals, width * 0.9, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate("%.2f" % v, (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.6, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["slot %d" % (p + 1) for p in range(npos)])
    ax.set_ylabel("fraction of draft rounds accepted")
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", fontsize=8.4)


def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    _accept_panel(axes[0], ACCEPT_SOURCES_T1, "temperature 1.0, depth 3", 3)
    _accept_panel(axes[1], ACCEPT_SOURCES_T0, "temperature 0, depth 4", 4)
    fig.suptitle("How deep the draft got accepted, by slot, over 102,400 tokens",
                 y=1.03, fontsize=11.5)
    save(fig, "fig12_accept_by_position.png")


FIGURES = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
           fig10, fig11, fig12]


def main():
    for fn in FIGURES:
        try:
            fn()
        except Exception as exc:  # keep going: a bad arm must not kill the run
            msg = "SKIP  %s raised %s: %s" % (fn.__name__, type(exc).__name__, exc)
            print(msg)
            SKIPPED.append(msg)
    print("\n%d figures written to %s" % (len(WRITTEN), IMG))
    for name in WRITTEN:
        print("  " + name)
    if SKIPPED:
        print("\n%d note(s):" % len(SKIPPED))
        for msg in SKIPPED:
            print("  " + msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
