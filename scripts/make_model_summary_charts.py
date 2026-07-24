"""Regenerate the model-comparison charts embedded in MODEL_SUMMARY.md / README.

Design follows the dataviz method: a CVD-validated blue/orange categorical pair, a light
chart surface, recessive grid, direct value labels (relief + secondary encoding), a legend
for >=2 series, and an honest zero baseline on every bar. Layout is stacked vertically
(title / legend / plot / footnote) so nothing overlaps.

All numbers are the measured evaluation values from `gs://qi-2026summer-avocado/outputs/`
and the AutoML Vision evaluation console — edit them here if a run is re-measured.

    python scripts/make_model_summary_charts.py   # writes assets/model-summary/*.png
"""
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parents[1] / "assets" / "model-summary"
OUT.mkdir(parents=True, exist_ok=True)

# dataviz reference palette (light mode)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, BLUE, ORANGE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": MUTED, "font.size": 11,
})


def _labels(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK_2)


def grouped(fname, title, groups, series, colors, fmt, footnote, pct=False, figsize=(8.6, 4.9)):
    fig, ax = plt.subplots(figsize=figsize)
    wrapped = textwrap.wrap(footnote, 118)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.78, bottom=0.11 + 0.032 * len(wrapped))

    n = len(series)
    x = np.arange(len(groups))
    w = 0.8 / n
    for i, (name, vals) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        _labels(ax, ax.bar(x + off, vals, w * 0.9, label=name, color=colors[i], zorder=3), fmt)

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    fig.text(0.02, 0.955, title, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    if n > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0.018, 0.865), bbox_transform=fig.transFigure,
                  ncol=n, frameon=False, fontsize=10, handlelength=1.1, columnspacing=1.8)
    fig.text(0.02, 0.02, "\n".join(wrapped), color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print("wrote", OUT / fname)


d3 = lambda v: f"{v:.3f}"
pc1 = lambda v: f"{v * 100:.1f}%"

# 1) HERO — the one metric all four tracks share
grouped(
    "cross_model_precision_recall.png",
    "Precision & Recall across model tracks",
    ["ResNet-18", "AutoML Raw", "AutoML Balanced", "GMM (color)"],
    [("Precision", [0.807, 0.828, 0.843, 0.650]),
     ("Recall",    [0.801, 0.784, 0.799, 0.634])],
    [BLUE, ORANGE], d3,
    "ResNet/GMM: macro precision/recall (argmax) on the curated General test split.  "
    "AutoML: precision/recall at 0.5 confidence on its own split.  Directional — protocols differ (summary §1).",
)

# 2) ResNet (CV) vs GMM color baseline — ordinal metrics, higher = better
grouped(
    "resnet_vs_gmm_ordinal.png",
    "ResNet-18 vs GMM color baseline  (same General data, higher = better)",
    ["Exact acc", "Within-1 stage", "QWK", "Macro-F1"],
    [("ResNet-18 (5-fold CV)", [0.794, 0.995, 0.946, 0.790]),
     ("GMM-rich (single split)", [0.650, 0.964, 0.889, 0.637])],
    [BLUE, ORANGE], d3,
    "ResNet: 5-fold CV mean.  GMM: single fixed 70/15/15 split.  "
    "Stage MAE (lower = better), not shown: ResNet 0.212 vs GMM 0.386.",
)

# 3) AutoML Raw vs Balanced — per-class Average Precision
grouped(
    "automl_raw_vs_balanced_ap.png",
    "AutoML Vision — per-class Average Precision (Raw vs Balanced)",
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    [("Raw (natural dist.)", [0.977, 0.863, 0.798, 0.865, 0.942]),
     ("Balanced (4k/class)", [0.974, 0.896, 0.823, 0.848, 0.949])],
    [BLUE, ORANGE], d3,
    "Vertex AI AutoML Vision.  Balancing lifts Stage 3 (0.798→0.823) but dips Stage 4 (0.865→0.848).",
)

# 4) ResNet per-stage correct rate (confusion-matrix diagonal / recall)
grouped(
    "resnet_per_stage.png",
    "ResNet-18 per-stage correct rate  (5-fold CV, confusion diagonal)",
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    [("Correct rate", [0.902, 0.760, 0.766, 0.711, 0.817])],
    [BLUE], pc1,
    "Stage 4 (peak, end of shelf life) is the weakest class; Stage 1 the strongest.",
    pct=True, figsize=(8.6, 4.6),
)

print("done ->", OUT)
