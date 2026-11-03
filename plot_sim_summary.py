"""plot_sim_summary.py -- the null, equal-connectivity and strong simulations side by side.

Reads the per-tag tables of the plot_*.py scripts (results/simulations/<tag>/...).
One figure per analysis: power (Cohen's d), synchrony (on - off change), recall decoder
(AUC - permutation chance), network replicability (plot_reliability.py's boxes over splits);
one panel per tag, sharing the y-axis, so a null is read on the strong scale; test-retest
across sessions as plot_reliability.py's boxes (power and synchrony in one panel).
Power and synchrony: one row per band, the paper figures' per-ROI box plots of subjects
(fc.roi_panel); the decoder: frontal
(planted in strong) in black, the other lobes grey.
-> figures/simulations/sim_summary_{power,synchrony,decoder,replicability,test_retest}.png

    python plot_sim_summary.py [--tags hg_null sync_equal strong]
"""
from __future__ import annotations

import argparse
from os.path import exists, getsize, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc

TITLES = {"hg_null": "Null (no connectivity)", "sync_equal": "Null (equal connectivity)",
          "strong": "Strong (frontal, 200-600 ms)"}


def read(tag: str, path: str) -> pd.DataFrame | None:
    f = join("results", "simulations", tag, path)
    return pd.read_csv(f) if exists(f) and getsize(f) > 1 else None   # old empty tables: 1 byte


def roi_boxes(path: str, col: str):
    """Panel: the paper figures' per-ROI box plots (fc.roi_panel) of one per-subject table."""
    def panel(ax, tag) -> bool:
        t = read(tag, path)
        if t is not None:
            fc.roi_panel(ax, t, col, np.random.default_rng(0), stats=fc.roi_stats(t, col))
        return t is not None
    return panel


def decoder(ax, tag) -> bool:
    dec = read(tag, "decoding/recall_auc_time_word_on_stats.csv")
    if dec is not None:
        for lobe, g in dec.groupby("lobe"):
            y, s = g["mean"] - g["null"], g["sem"]
            c, z = ("k", 3) if lobe == "frontal" else ("0.65", 2)
            ax.plot(g["window_ms"] + 50, y, color=c, lw=2 if lobe == "frontal" else 1, zorder=z,
                    label="Frontal" if lobe == "frontal" else None)
            if lobe == "frontal":
                ax.fill_between(g["window_ms"] + 50, y - s, y + s, color="k", alpha=0.15, lw=0)
        ax.plot([], [], color="0.65", label="Other lobes")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Time from word onset (ms)", fontsize=9)
    return dec is not None


def replicability(ax, tag) -> bool:
    rep = read(tag, "reliability/profile_replication_word_on.csv")
    if rep is not None:
        x = np.arange(len(fc.LOBES))
        for k, (m, col) in enumerate(zip(["Alpha ciPLV", "High-gamma AEC-c"], ("C0", "C3"))):
            r = rep[rep["measure"] == m]   # as plot_reliability.py: boxes over splits, whiskers central 95%
            ax.boxplot([r.loc[r["lobe"] == l, "r"] for l in fc.LOBES], positions=x + (k - 0.5) * 0.38,
                       widths=0.32, patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                       boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="0.15"))
            ax.plot([], [], "s", color=col, alpha=0.6, ms=9, label=m)
        ax.set_xticks(x, [l.capitalize() for l in fc.LOBES], rotation=30, fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    return rep is not None


def retest(ax, tag) -> bool:
    """Test-retest across sessions, as plot_reliability.py."""
    rt = read(tag, "reliability/test_retest_word_on.csv")
    ok = rt is not None and len(rt) > 0
    x = [0, 1, 2.4, 3.4]
    for xi, (k, col) in zip(x, RETEST.items()):
        v = rt[k].dropna().to_numpy() if ok else []
        if len(v):   # subjects; whiskers: central 95% (as plot_reliability.py)
            ax.boxplot([v], positions=[xi], widths=0.6,
                       patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                       boxprops=dict(facecolor=col[1], alpha=0.6), medianprops=dict(color="0.15"))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, [c[0] for c in RETEST.values()], rotation=30, fontsize=8)
    ax.set_xlabel(f"n = {len(rt) if ok else 0} subjects", fontsize=9)
    return ok


RETEST = {"alpha": ("Alpha d", "C0"), "high_gamma": ("High-gamma d", "C3"),
          "ciplv": ("Alpha ciPLV", "C0"), "aec_c": ("High-gamma AEC-c", "C3")}


# {figure: [(y label, panel)] one row each}
P, Q = "burke_roi_power/roi_power_word_on_{}_per_subject.csv", "burke_roi_synchrony/roi_synchrony_word_on_{}_per_subject.csv"
ANALYSES = {"power": [("Alpha\nCohen's d", roi_boxes(P.format("alpha"), "cohens_d")),
                      ("High gamma\nCohen's d", roi_boxes(P.format("high_gamma"), "cohens_d"))],
            "synchrony": [("Alpha\nciPLV (on − off)", roi_boxes(Q.format("alpha_ciplv"), "sync_diff")),
                          ("High gamma\nAEC-c (on − off)", roi_boxes(Q.format("high_gamma_aec_c"), "sync_diff"))],
            "decoder": [("Recall AUC − chance", decoder)], "replicability": [("Network replicability r", replicability)],
            "test_retest": [("Test-retest r within subjects", retest)]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=list(TITLES))
    p.add_argument("--out-dir", default=join("figures", "simulations"))
    args = p.parse_args()
    for name, rows in ANALYSES.items():
        fig, axes = plt.subplots(len(rows), len(args.tags), figsize=(6.5 * len(args.tags), 3.6 * len(rows)),
                                 sharey="row", squeeze=False)
        for i, (ylabel, panel) in enumerate(rows):
            for ax, tag in zip(axes[i], args.tags):
                if i == 0:
                    ax.set_title(TITLES.get(tag, tag), fontsize=12)
                if not panel(ax, tag):
                    ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="0.5")
                ax.spines[["top", "right"]].set_visible(False)
            axes[i, 0].set_ylabel(ylabel, fontsize=10)
            if axes[i, 0].get_legend_handles_labels()[0]:
                axes[i, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        out = join(args.out_dir, f"sim_summary_{name}")
        fig.savefig(f"{out}.png", dpi=200)
        fig.savefig(f"{out}.pdf")
        plt.close(fig)
        print(f"[saved] {out}.png")

if __name__ == "__main__":
    main()
