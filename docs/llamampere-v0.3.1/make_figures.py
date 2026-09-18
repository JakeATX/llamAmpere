#!/usr/bin/env python3
"""Figures for the llamAmpere v0.3.1 highlights document.

Every number this script draws comes from `highlights_data.json` next to it, and every number in
that file carries a `source` string naming the file it was read out of. Nothing is computed here
except ratios that the JSON marks as derived, and nothing is invented: a cell whose `value` is
null is drawn as a hatched "pending" bar and labelled `pending`, so the same script renders the
document before and after a measurement lands. Fill a null in, re-run, and the bar becomes real.

Writes 1600x900 PNGs at 200 DPI into img/ next to this script.

Run:  python3 make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
DATA = os.path.join(HERE, "highlights_data.json")
os.makedirs(IMG, exist_ok=True)

# 1600 x 900 px at 200 DPI.
FIGSIZE = (8.0, 4.5)
DPI = 200

SURFACE = "#1a1a19"
PANEL = "#111110"
INK = "#ffffff"
INK2 = "#c3c2b7"
INK3 = "#8a8980"
GRID = "#38383a"

# Categorical slots, dark-surface steps, assigned in fixed order and never cycled.
# Colour follows the format (the entity), not its rank, and is the same in every figure.
SLOT = {
    1: "#3987e5",  # blue
    2: "#d95926",  # orange
    3: "#199e70",  # aqua
    4: "#c98500",  # yellow
    5: "#d55181",  # magenta
    7: "#9085e9",  # violet
    8: "#e66767",  # red
}

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.7,
    "hatch.color": INK3,
    "hatch.linewidth": 1.0,
})

with open(DATA, "r", encoding="utf-8") as fh:
    D = json.load(fh)

FORMATS = D["formats"]
ORDER = D["format_order"]


def colour(fmt_key):
    return SLOT[FORMATS[fmt_key]["color_slot"]]


def val(cell):
    """A data cell is {"value": <number or null>, "source": "...", ...}. None means pending."""
    if cell is None:
        return None
    return cell.get("value")


def fmt_num(x, digits=1):
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def footer(fig, spec):
    fig.text(0.010, 0.010, spec["footer"], color=INK3, fontsize=7.4, ha="left", va="bottom",
             linespacing=1.5)
    fig.text(0.990, 0.972, D["meta"]["stamp"], color=INK3, fontsize=8.2, ha="right", va="top")


def header(fig, spec):
    """Headline and one-line message, pinned to the left edge of the canvas, not to the axes."""
    fig.text(0.012, 0.945, spec["title"], color=INK, fontsize=15, fontweight="bold", ha="left",
             va="bottom")
    fig.text(0.012, 0.893, spec["message"], color=INK2, fontsize=10.5, ha="left", va="bottom")


def frame(fig, ax, spec, ylabel=None):
    """Title block, footer conditions line and axis chrome shared by every figure."""
    header(fig, spec)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    footer(fig, spec)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def bar_group(ax, xs, values, width, face, pending_h, horizontal=False, digits=1,
              suffix="", label_size=11, pending_size=8.5, pending_text=True):
    """Draw one row of bars. None values become hatched 'pending' bars at a nominal height."""
    for x, v in zip(xs, values):
        if v is None:
            if horizontal:
                ax.barh(x, pending_h, width, color=PANEL, edgecolor=INK3, hatch="//",
                        linewidth=1.0, zorder=3)
                ax.text(pending_h * 1.04, x, "pending", va="center", ha="left",
                        color=INK3, fontsize=9, style="italic", zorder=5)
            else:
                ax.bar(x, pending_h, width, color=PANEL, edgecolor=INK3, hatch="//",
                       linewidth=1.0, zorder=3)
                if pending_text:
                    ax.text(x, pending_h * 1.05, "pending", ha="center", va="bottom",
                            color=INK3, fontsize=pending_size, style="italic", zorder=5)
            continue
        if horizontal:
            ax.barh(x, v, width, color=face, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.text(v * 1.012, x, fmt_num(v, digits) + suffix, va="center", ha="left",
                    color=INK, fontsize=label_size, fontweight="bold", zorder=5)
        else:
            ax.bar(x, v, width, color=face, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.text(x, v, fmt_num(v, digits) + suffix, ha="center", va="bottom",
                    color=INK, fontsize=label_size, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.75))


def legend_formats(ax, keys, ncols=None, y=-0.16, pending=True):
    handles = [Patch(facecolor=colour(k), edgecolor=SURFACE, label=FORMATS[k]["label"])
               for k in keys]
    if pending:
        handles.append(Patch(facecolor=PANEL, edgecolor=INK3, hatch="//", label="not measured yet"))
    ax.legend(handles=handles, ncols=ncols or len(handles), loc="upper center",
              bbox_to_anchor=(0.5, y), frameon=False, labelcolor=INK2, handlelength=1.4,
              columnspacing=1.4)


def top(values, fallback, pad=1.32):
    real = [v for v in values if v is not None]
    return max(real) * pad if real else fallback


def save(fig, name):
    # No bbox_inches="tight": the canvas must stay exactly FIGSIZE * DPI = 1600 x 900 px.
    fig.savefig(os.path.join(IMG, name), facecolor=SURFACE, dpi=DPI)
    plt.close(fig)
    print("wrote img/" + name)


# ---------------------------------------------------------------- fig01: decode at 16-20K depth
def fig01():
    spec = D["figures"]["fig01"]
    keys = [b["format"] for b in spec["bars"]]
    arms = spec["arms"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.19
    allv = []
    for arm in ("t1", "mtp"):
        allv += [val(b[arm]) for b in spec["bars"]]
    ymax = top(allv, spec["axis_max_fallback"])
    pend = ymax * 0.13
    for i, b in enumerate(spec["bars"]):
        xs = [j + (i - (len(keys) - 1) / 2) * w for j in range(len(arms))]
        cells = [b["t1"], b["mtp"]]
        bar_group(ax, xs, [val(c) for c in cells], w, colour(b["format"]), pend)
        for x, c in zip(xs, cells):
            if val(c) is not None:
                ax.text(x, -ymax * 0.025, c.get("label", ""), ha="center", va="top", color=INK3,
                        fontsize=7.2, linespacing=1.4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=13, color=INK, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=34)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    legend_formats(ax, keys, y=-0.37)
    frame(fig, ax, spec, "decode tok/s")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.33)
    save(fig, "fig01_decode_16k_20k.png")


# ---------------------------------------------------------------- fig02: decode at 64K and 100K
def fig02():
    spec = D["figures"]["fig02"]
    keys = [b["format"] for b in spec["panels"][0]["bars"]]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    allv = []
    for p in spec["panels"]:
        for b in p["bars"]:
            allv += [val(b["t1"]), val(b["mtp"])]
    ymax = top(allv, spec["axis_max_fallback"])
    pend = ymax * 0.13
    w = 0.19
    for ax, arm, arm_label in ((axes[0], "t1", spec["arms"][0]), (axes[1], "mtp", spec["arms"][1])):
        for i, key in enumerate(keys):
            xs = [j + (i - (len(keys) - 1) / 2) * w for j in range(len(spec["panels"]))]
            cells = [[b for b in p["bars"] if b["format"] == key][0][arm]
                     for p in spec["panels"]]
            # Four narrow bars per group: no room for a "pending" word (the legend carries it),
            # and the condition tags are staggered onto two rows so neighbours cannot touch.
            bar_group(ax, xs, [val(c) for c in cells], w, colour(key), pend, label_size=9.5,
                      pending_text=False)
            # Four narrow bars per group leave no room for a tag on every bar, so the panel
            # label carries the common condition and only the cells that depart from it are
            # tagged ("show_label": true in the JSON). The rest are spelled out in the footer.
            for x, c in zip(xs, cells):
                v = val(c)
                if v is not None and c.get("show_label"):
                    # Offset to the right of its own bar: centred it would sit over the
                    # neighbouring bar and become unreadable.
                    ax.text(x + 0.105, v + ymax * 0.02, c.get("label", ""), ha="left",
                            va="bottom", color=INK3, fontsize=7.6, linespacing=1.4)
        ax.set_xticks(range(len(spec["panels"])))
        ax.set_xticklabels([p["label"] for p in spec["panels"]], fontsize=12, color=INK,
                           fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.55, len(spec["panels"]) - 0.45)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.set_facecolor(SURFACE)
        ax.text(0.5, 1.01, arm_label, transform=ax.transAxes, ha="center", va="bottom",
                color=INK, fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("decode tok/s", color=INK2)
    legend_formats(axes[0], keys, y=-0.14)
    axes[0].get_legend().set_bbox_to_anchor((1.04, -0.14), transform=axes[0].transAxes)
    header(fig, spec)
    footer(fig, spec)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.27, wspace=0.08)
    save(fig, "fig02_decode_64k_100k.png")


# ---------------------------------------------------------------- fig03: bits per weight vs size
def fig03():
    spec = D["figures"]["fig03"]
    rows = spec["bars"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ys = list(range(len(rows)))[::-1]
    sizes = [val(r["file_gib"]) for r in rows]
    xmax = top(sizes, spec["axis_max_fallback"], pad=1.18)
    for y, r in zip(ys, rows):
        bar_group(ax, [y], [val(r["file_gib"])], 0.58, colour(r["format"]), xmax * 0.13,
                  horizontal=True, digits=2, suffix=" GiB", label_size=13)
        bpw = val(r["bpw"])
        tag = f"{bpw} bpw" if bpw is not None else "bpw pending"
        ax.text(xmax * 0.012, y, tag, va="center", ha="left", color="#0d0d0c", fontsize=11,
                fontweight="bold", zorder=6)
    ax.set_yticks(ys)
    ax.set_yticklabels([FORMATS[r["format"]]["label"] for r in rows], fontsize=12, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("file size on disk (GiB)", color=INK2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    frame(fig, ax, spec)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.80, bottom=0.165)
    save(fig, "fig03_bpw_vs_size.png")


# ---------------------------------------------------------------- fig04: peak VRAM at depth
def fig04():
    spec = D["figures"]["fig04"]
    keys = [b["format"] for b in spec["panels"][0]["bars"]]
    ceiling = spec["ceiling_mib"]["value"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.19
    allv = []
    for p in spec["panels"]:
        allv += [val(b["peak_mib"]) for b in p["bars"]]
    ymax = max(top(allv, spec["axis_max_fallback"], pad=1.22), ceiling * 1.18)
    pend = ymax * 0.13
    for i, key in enumerate(keys):
        xs = [j + (i - (len(keys) - 1) / 2) * w for j in range(len(spec["panels"]))]
        vals = []
        for p in spec["panels"]:
            cell = [b for b in p["bars"] if b["format"] == key][0]
            vals.append(val(cell["peak_mib"]))
        bar_group(ax, xs, vals, w, colour(key), pend, digits=0, label_size=9.5)
        for x, v in zip(xs, vals):
            if v is not None and v > ceiling:
                ax.text(x, v + ymax * 0.055, "does not fit", ha="center", va="bottom",
                        color=SLOT[8], fontsize=9, fontweight="bold", zorder=6)
    ax.axhline(ceiling, color=SLOT[8], linestyle="--", linewidth=1.6, zorder=4)
    ax.text(len(spec["panels"]) - 0.5, ceiling + ymax * 0.012, spec["ceiling_label"], ha="right",
            va="bottom", color=SLOT[8], fontsize=10, fontweight="bold", zorder=6)
    ax.set_xticks(range(len(spec["panels"])))
    ax.set_xticklabels([p["label"] for p in spec["panels"]], fontsize=12, color=INK)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.55, len(spec["panels"]) - 0.45)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    legend_formats(ax, keys, y=-0.17)
    frame(fig, ax, spec, "peak whole-card VRAM (MiB)")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.25)
    save(fig, "fig04_vram_100k_200k.png")


# ---------------------------------------------------------------- fig05: speed-up vs alternates
def fig05():
    spec = D["figures"]["fig05"]
    rows = spec["bars"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ys = list(range(len(rows)))[::-1]
    ratios = [val(r["ratio"]) for r in rows]
    xmax = top(ratios, spec["axis_max_fallback"], pad=1.30)
    for y, r in zip(ys, rows):
        v = val(r["ratio"])
        bar_group(ax, [y], [v], 0.6, colour(r["format"]), xmax * 0.13, horizontal=True,
                  digits=2, suffix="x", label_size=13)
        ours, theirs = val(r["ours_tok_s"]), val(r["theirs_tok_s"])
        if ours is not None and theirs is not None:
            ax.text(xmax * 0.015, y, f"{fmt_num(ours)} vs {fmt_num(theirs)} tok/s", va="center",
                    ha="left", color="#0d0d0c", fontsize=10, fontweight="bold", zorder=6)
    ax.axvline(1.0, color=INK3, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(1.0, len(rows) - 0.35, "parity", ha="center", va="bottom", color=INK3, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.2, color=INK,
                       linespacing=1.3)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("decode tok/s, ours divided by theirs (higher is better for us)", color=INK2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    legend_formats(ax, spec["legend_formats"], y=-0.19, pending=False)
    frame(fig, ax, spec)
    fig.subplots_adjust(left=0.255, right=0.985, top=0.80, bottom=0.24)
    save(fig, "fig05_speedup_vs_alternates.png")


# ---------------------------------------------------------------- fig06: at a glance card
def fig06():
    spec = D["figures"]["fig06"]
    cols = spec["columns"]
    rows = spec["rows"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    name_w = 0.215
    xs = [name_w + (1.0 - name_w) * (i + 0.5) / len(cols) for i in range(len(cols))]
    head_y = 0.905
    row_h = 0.885 / (len(rows) + 0.1)

    for x, c in zip(xs, cols):
        ax.text(x, head_y, c, ha="center", va="center", color=INK2, fontsize=8.4,
                fontweight="bold", linespacing=1.4)
    ax.plot([0.0, 1.0], [head_y - 0.075, head_y - 0.075], color=GRID, lw=1.2)

    for i, row in enumerate(rows):
        y = head_y - 0.105 - (i + 0.5) * row_h
        ax.add_patch(plt.Rectangle((0.0, y - row_h * 0.44), 1.0, row_h * 0.88,
                                   facecolor=PANEL if i % 2 else SURFACE, edgecolor="none",
                                   zorder=0))
        ax.add_patch(plt.Rectangle((0.0, y - row_h * 0.44), 0.009, row_h * 0.88,
                                   facecolor=colour(row["format"]), edgecolor="none", zorder=2))
        ax.text(0.024, y, FORMATS[row["format"]]["label"], ha="left", va="center", color=INK,
                fontsize=11.5, fontweight="bold", zorder=3)
        for x, cell in zip(xs, row["cells"]):
            v = val(cell)
            if v is None:
                ax.text(x, y, "pending", ha="center", va="center", color=INK3, fontsize=9,
                        style="italic", zorder=3)
            else:
                ax.text(x, y, str(v), ha="center", va="center", color=INK, fontsize=10.2,
                        zorder=3, linespacing=1.35)

    header(fig, spec)
    footer(fig, spec)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.855, bottom=0.075)
    save(fig, "fig06_at_a_glance.png")


if __name__ == "__main__":
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    print("all figures written to " + IMG)
