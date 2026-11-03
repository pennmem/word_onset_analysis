"""plot_sim_summary.py -- the null, equal-coupling and ceiling simulations side by side.

Reads the per-tag tables of the plot_*.py scripts (results/simulations/<tag>/...).
One figure per analysis: power (Cohen's d), synchrony (on - off change), recall decoder
(AUC - permutation chance), network replicability (median r, central 95% of splits);
one panel per tag, sharing the y-axis, so a null is read on the ceiling's scale; test-retest
across sessions as plot_reliability.py's boxes (power and synchrony in one panel).
Power and synchrony: mean +/- 95% CI across subjects per ROI; the decoder: frontal
(planted in ceiling) in black, the other lobes grey.
-> figures/simulations/sim_summary_{power,synchrony,decoder,replicability,test_retest}.png

    python plot_sim_summary.py [--tags hg_null sync_equal ceiling]
"""
from __future__ import annotations

import argparse
from os.path import exists, getsize, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc

TITLES = {"hg_null": "Null (no coupling)", "sync_equal": "Null (equal coupling)",
          "ceiling": "Ceiling (frontal, 200-600 ms)"}
BANDS = [("alpha", "C0"), ("high_gamma", "C3")]


def read(tag: str, path: str) -> pd.DataFrame | None:
    f = join("results", "simulations", tag, path)
    return pd.read_csv(f) if exists(f) and getsize(f) > 1 else None   # old empty tables: 1 byte


def roi_panel(ax, tbls, labels) -> None:
    """Mean +/- 95% CI per ROI, one series per band (offset)."""
    x = np.arange(len(fc.ROI_ORDER))
    for k, (t, (lab, col)) in enumerate(zip(tbls, labels)):
        if t is None:
            continue
        t = t.set_index("roi").reindex(fc.ROI_ORDER)
        xx = x + (k - 0.5) * 0.3
        ax.errorbar(xx, t["mean"], yerr=[t["mean"] - t["ci95_lo"], t["ci95_hi"] - t["mean"]],
                    fmt="o", ms=4, color=col, lw=1.2, capsize=0, label=lab)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, [r.replace("hippocampus", "hipp.") for r in fc.ROI_ORDER], rotation=60, fontsize=7)


def power(ax, tag) -> bool:
    pw = [read(tag, f"burke_roi_power/roi_power_word_on_{b}_stats.csv") for b, _ in BANDS]
    roi_panel(ax, [t if t is None else t[t["measure"] == "cohens_d"] for t in pw],
              [("Alpha", "C0"), ("High gamma", "C3")])
    return pw[0] is not None


def synchrony(ax, tag) -> bool:
    t = [read(tag, "burke_roi_synchrony/roi_synchrony_word_on_alpha_ciplv_stats.csv"),
         read(tag, "burke_roi_synchrony/roi_synchrony_word_on_high_gamma_aec_c_stats.csv")]
    roi_panel(ax, t, [("Alpha ciPLV", "C0"), ("High-gamma AEC-c", "C3")])
    return t[0] is not None


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
            g = rep[rep["measure"] == m].groupby("lobe")
            med, lo, hi = (g["r"].median(), g["r"].quantile(0.025), g["r"].quantile(0.975))
            med, lo, hi = (s.reindex(fc.LOBES) for s in (med, lo, hi))
            ax.errorbar(x + (k - 0.5) * 0.3, med, yerr=[med - lo, hi - med], fmt="o", ms=4,
                        color=col, lw=1.2, label=m)
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
ANALYSES = {"power": [("Power (Cohen's d)", power)], "synchrony": [("Synchrony change (on − off)", synchrony)],
            "decoder": [("Recall AUC − chance", decoder)], "replicability": [("Network replicability r", replicability)],
            "test_retest": [("Test-retest r across regions", retest)]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=list(TITLES))
    p.add_argument("--out-dir", default=join("figures", "simulations"))
    args = p.parse_args()
    for name, rows in ANALYSES.items():
        fig, axes = plt.subplots(len(rows), len(args.tags), figsize=(4.2 * len(args.tags), 3.6 * len(rows)),
                                 sharey="row", squeeze=False)
        for i, (ylabel, panel) in enumerate(rows):
            for ax, tag in zip(axes[i], args.tags):
                if i == 0:
                    ax.set_title(TITLES.get(tag, tag), fontsize=12)
                if not panel(ax, tag):
                    ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="0.5")
                ax.spines[["top", "right"]].set_visible(False)
            axes[i, 0].set_ylabel(ylabel, fontsize=10)
            axes[i, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        out = join(args.out_dir, f"sim_summary_{name}")
        fig.savefig(f"{out}.png", dpi=200)
        fig.savefig(f"{out}.pdf")
        plt.close(fig)
        print(f"[saved] {out}.png")

if __name__ == "__main__":
    main()
