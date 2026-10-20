"""Figures for the technical note: possession timelines.

    python scripts/make_figures.py out/Yamal_Fan_Boys.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import data, score  # noqa: E402
import possession_metrics as pm  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "figures")
# validated categorical slots 1-2 (blue, orange) + a neutral for dead ball
COLOURS = {0: "#2a78d6", 1: "#eb6834", 2: "#c3c2b7"}
NAMES = {0: "Barcelona in possession", 1: "Athletic in possession", 2: "Dead ball"}
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def strip(ax, codes, start, stop, y, height, gap):
    """Draw the H/A/D segments of codes[start:stop] as one horizontal strip at row y."""
    runs = [(start, stop)]
    filled = pm.fill_abstentions(codes, runs)
    for a, b, k in pm.segments(filled, runs):
        if k not in COLOURS:
            continue
        x0 = (a - start) / 1500.0  # minutes
        w = (b - a) / 1500.0
        ax.add_patch(plt.Rectangle((x0 + gap / 2, y - height / 2), max(w - gap, 0.0005), height,
                                   facecolor=COLOURS[k], edgecolor="none"))


def legend(fig):
    handles = [Patch(facecolor=COLOURS[k], label=NAMES[k]) for k in (0, 1, 2)]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 0.99), handlelength=1.2, labelcolor=INK2)


def public_window(pred):
    gt = score.public_gt()
    a, b = 11185, 18685
    fig, ax = plt.subplots(figsize=(8.0, 1.9), dpi=200)
    strip(ax, gt, a, b, 1, 0.62, 0.004)
    strip(ax, pred, a, b, 0, 0.62, 0.004)
    ax.set_yticks([1, 0], ["Ground truth", "Our prediction"])
    ax.set_xlim(0, (b - a) / 1500.0)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("minutes since kick-off (public window)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = os.path.join(OUT, "public_window.png")
    fig.savefig(path)
    return path


def full_match(pred):
    fig, ax = plt.subplots(figsize=(8.0, 1.9), dpi=200)
    longest = max(b - a for a, b in data.HALVES) / 1500.0
    for y, (a, b) in zip((1, 0), data.HALVES):
        strip(ax, pred, a, b + 1, y, 0.62, 0.0)
    ax.set_yticks([1, 0], ["1st half", "2nd half"])
    ax.set_xlim(0, longest)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("minutes since the start of each graded half")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = os.path.join(OUT, "full_match.png")
    fig.savefig(path)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pred = data.read_labels(sys.argv[1] if len(sys.argv) > 1 else "out/submission.csv")
    print(public_window(pred))
    print(full_match(pred))
