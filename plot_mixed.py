"""plot_mixed.py -- figures from mixed_models.R (word-on contrast; run that first).

  latency   lobe x lobe matrix of high-gamma latency differences (mixed model), lobes in
            latency order; cell = row lobe minus earlier column lobe (ms); stars: Holm p
            -> figures/latency/hg_latency_word_on_mixed.png
  bands     lobe x lobe matrix of the alpha-vs-high-gamma pattern difference (lobe x band
            interaction contrast, |d|); stars: Holm p
            -> figures/burke_roi_power/roi_power_word_on_mixed.png

    python plot_mixed.py
"""
from __future__ import annotations

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc

LAT = join("figures", "latency", "hg_latency_word_on_mixed")
BANDS = join("figures", "burke_roi_power", "roi_power_word_on_mixed")


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


def latency() -> None:
    emm = pd.read_csv(fc.table_path(f"{LAT}_lobes.csv")).sort_values("emmean")
    pairs = pd.read_csv(fc.table_path(f"{LAT}_pairs.csv"))
    lobes = list(emm["lobe"])
    diff = np.full((len(lobes), len(lobes)), np.nan)
    q = diff.copy()
    for contrast, est, pval in zip(pairs["contrast"], pairs["estimate"], pairs["p.value"]):
        idx_a, idx_b = (lobes.index(lobe) for lobe in contrast.split(" - "))
        i, j = max(idx_a, idx_b), min(idx_a, idx_b)   # later lobe (row) - earlier lobe (column)
        diff[i, j], q[i, j] = est if idx_a > idx_b else -est, pval
    diff, q, rows, cols = diff[1:, :-1], q[1:, :-1], lobes[1:], lobes[:-1]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(diff, cmap="Reds", vmin=0)
    for (i, j), val in np.ndenumerate(diff):
        if np.isfinite(val):
            ax.annotate(f"{val:.0f}{fc.stars(q[i, j])}", (j, i), ha="center", va="center", fontsize=12,
                        color="white" if val > np.nanmax(diff) / 2 else "black")
    ax.set_xticks(range(len(cols)), [lb.capitalize() for lb in cols], rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(rows)), [lb.capitalize() for lb in rows], fontsize=12)
    for tick, lb in zip([*ax.get_xticklabels(), *ax.get_yticklabels()], cols + rows):
        tick.set_color(fc.LOBE_COLORS[lb])
        tick.set_fontweight("bold")
    ax.set_xlabel("Earlier lobe", fontsize=12)
    ax.set_ylabel("Later lobe", fontsize=12)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Latency difference (ms)", fontsize=12)
    save(fig, LAT)


def bands() -> None:
    pairs = pd.read_csv(fc.table_path(f"{BANDS}_pairs.csv"))
    lobes = list(fc.LOBES)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    diff = np.full((len(lobes), len(lobes)), np.nan)
    q = diff.copy()
    for pair, est, pval in zip(pairs["lobe_pairwise"], pairs["estimate"], pairs["p.value"]):
        idx_a, idx_b = (lobes.index(lobe) for lobe in pair.split(" - "))
        i, j = max(idx_a, idx_b), min(idx_a, idx_b)
        diff[i, j], q[i, j] = abs(est), pval   # lower triangle: size of the pattern difference
    diff, q = diff[1:, :-1], q[1:, :-1]   # drop the empty first row / last column
    rows, cols = lobes[1:], lobes[:-1]
    im = ax.imshow(diff, cmap="magma_r", vmin=0)
    for (i, j), pval in np.ndenumerate(q):
        if np.isfinite(pval) and fc.stars(pval):
            ax.annotate(fc.stars(pval), (j, i), ha="center", va="center", fontsize=10,
                        color="white" if diff[i, j] > np.nanmax(diff) / 2 else "black")
    ax.set_xticks(range(len(cols)), [lb.capitalize() for lb in cols], rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(rows)), [lb.capitalize() for lb in rows], fontsize=11)
    for tick, lb in zip([*ax.get_xticklabels(), *ax.get_yticklabels()], cols + rows):
        tick.set_color(fc.LOBE_COLORS[lb])
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Alpha vs high-gamma pattern difference (|d|)", fontsize=10)
    save(fig, BANDS)


if __name__ == "__main__":
    latency()
    bands()
