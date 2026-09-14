#!/usr/bin/env python3
"""Figures for the v0.3 release write-up.

Every value is transcribed from the tables in ARTICLE.md (themselves sourced to the SP2 ladder,
SP2 Phase 2 and SP3 engine-comparison logs), so the script needs no raw data. Writes PNGs into
img/ next to this script.

Run:  python3 make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.family": "sans-serif", "font.size": 9.5,
    "axes.titlesize": 11.5, "axes.labelsize": 10, "legend.fontsize": 8.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})

# Fixed per arm across every figure.
C = {"stock": "#8a8f98", "tq": "#2a78d6", "v02": "#1baf7a", "v03": "#eb6834",
     "vllm4": "#4a3aa7", "vllm3": "#9b8fe0", "sglang": "#eda100"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote img/" + name)


# fig01: the four-arm ladder, decode tok/s at temperature 1, MTP-3.
workloads = ["Agentic (3 fixtures)", "Coding (1 fixture)", "All clean fixtures"]
ladder = [("Stock llama.cpp", "stock", [68.6, 65.7, 67.9]),
          ("TurboQuant", "tq", [66.1, 51.3, 62.4]),
          ("llamAmpere v0.2", "v02", [81.3, 67.8, 77.9]),
          ("llamAmpere v0.3", "v03", [100.9, 94.9, 99.4])]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
w = 0.2
for i, (label, key, vals) in enumerate(ladder):
    xs = [x + (i - 1.5) * w for x in range(len(workloads))]
    bars = ax.bar(xs, vals, w, label=label, color=C[key])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(workloads)), workloads)
ax.set_ylabel("decode tok/s")
ax.set_ylim(0, 115)
ax.set_title("Decode speed at temperature 1, MTP depth 3, 20,480 generated tokens")
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
save(fig, "fig01_ladder.png")

# fig02: speed over one long session, tok/s by KV depth after each 5,000-token window.
depth = [56222, 100201, 149743, 156515, 187185, 197174, 206851]
curves = [("llamAmpere v0.3", "v03", [95.09, 100.45, 87.94, 92.49, 90.82, 87.12, 85.16]),
          ("llamAmpere v0.2", "v02", [75.34, 75.01, 78.35, 78.07, 73.49, 71.57, None]),
          ("TurboQuant", "tq", [63.87, 56.94, 54.37, 53.82, 48.30, None, None]),
          ("Stock llama.cpp", "stock", [68.86, 59.11, 50.09, 49.83, None, None, None])]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for label, key, vals in curves:
    pts = [(d / 1000, v) for d, v in zip(depth, vals) if v is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4, color=C[key], label=label)
    ax.annotate(f"{pts[-1][1]:.1f}", pts[-1], textcoords="offset points", xytext=(5, -3), fontsize=7.5,
                color=C[key])
ax.set_xlabel("KV depth after window (thousand tokens)")
ax.set_ylabel("decode tok/s in window")
ax.set_ylim(40, 110)
ax.set_title("One server per arm, 5,000-token turns from a 51K agentic prompt, temperature 1")
ax.legend(loc="lower left", frameon=False)
save(fig, "fig02_long_context.png")

# fig03: against vLLM and SGLang, single stream, peak VRAM under the 23,552 MiB cap.
ctxs = ["32K", "64K"]
eng = [("llamAmpere v0.3, MTP-3", "v03", [112.4, 100.7]),
       ("vLLM tuned, MTP-4", "vllm4", [101.7, 90.3]),
       ("vLLM tuned, MTP-3", "vllm3", [90.2, 83.1]),
       ("SGLang tuned, NEXTN", "sglang", [68.6, 64.4])]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (label, key, vals) in enumerate(eng):
    xs = [x + (i - 1.5) * w for x in range(len(ctxs))]
    bars = ax.bar(xs, vals, w, label=label, color=C[key])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(ctxs)), [c + " context" for c in ctxs])
ax.set_ylabel("decode tok/s")
ax.set_ylim(0, 125)
ax.set_title("Single-stream decode against serving engines, temperature 1, 2,048 tokens")
ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
save(fig, "fig03_engines.png")
