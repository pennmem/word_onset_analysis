"""plot_spectrum.py -- word on vs off power per Burke ROI x log-spaced frequency bin.

Per electrode the Cohen's d of multitaper power in 11 log-spaced bins (5-150 Hz,
mains 50/60/100/120 +/- 2 Hz left out; compute.py); sessions averaged per electrode, electrodes
per subject and ROI. As DeHaan et al. 2025: one-sample t vs 0 across subjects per cell (colour),
BH-FDR over all ROI x bin cells (q < .05); outlines enclose connected significant cells.
-> figures/burke_roi_power/power_spectrum_word_on_rois.png
Also per anatomical subregion (hemispheres pooled, >= MIN_SUBJECTS_ROI subjects), rows grouped
by lobe, FDR over all subregion x bin cells -> power_spectrum_word_on_fine.png (supplement)

    python plot_spectrum.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from compute import FREQ_EDGES, power_dir

BEH = "word_on"
CENTRES = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])


def spectrum_table(save_root: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode, bin): d, with its Burke ROI and anatomical label (fine)."""
    rows = []
    for f in fc.session_files(power_dir(save_root, BEH, "spectrum"), "_power.pkl", n_sessions):
        p = fc.load_pickle(str(f))
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for label, r, fine, d_bins in zip(p["labels"], roi, p["reg_full"], p["cohens_d_freq"]):
            rows += [(str(p["sid"][0]), str(label), r, fine if isinstance(fine, str) else None, k, float(d))
                     for k, d in enumerate(d_bins)]
    return pd.DataFrame(rows, columns=["sub", "label", "roi", "fine", "bin", "d"])


def subject_table(d: pd.DataFrame, col: str, min_electrodes: int) -> pd.DataFrame:
    """sessions -> electrode, electrodes -> subject x `col` region x bin mean d."""
    per_el = d.dropna(subset=[col]).groupby(["sub", "label", col, "bin"], as_index=False)["d"].mean()
    per_sub = per_el.groupby(["sub", col, "bin"]).agg(d=("d", "mean"), n=("label", "nunique")).reset_index()
    per_sub = per_sub[per_sub["n"] >= min_electrodes].rename(columns={col: "roi"})
    per_sub["freq_hz"] = CENTRES[per_sub["bin"]]
    return per_sub


def outline(ax, sig: np.ndarray) -> None:
    """Border around connected significant cells: every cell edge between a significant
    cell and a non-significant (or missing) neighbour."""
    from matplotlib.collections import LineCollection
    padded = np.pad(sig, 1)
    segs = []
    for i, j in zip(*np.nonzero(sig)):
        y, x = i - 0.5, j - 0.5
        if not padded[i, j + 1]: segs.append([(x, y), (x + 1, y)])           # top
        if not padded[i + 2, j + 1]: segs.append([(x, y + 1), (x + 1, y + 1)])   # bottom
        if not padded[i + 1, j]: segs.append([(x, y), (x, y + 1)])           # left
        if not padded[i + 1, j + 2]: segs.append([(x + 1, y), (x + 1, y + 1)])   # right
    ax.add_collection(LineCollection(segs, colors="black", linewidths=1.5))


def t_label() -> str:
    con = fc.contrast(BEH)
    return f"t ({con['delta_label']})"


def draw(ax, M: np.ndarray, q: np.ndarray, labels: list, lobes: list, xlabels: list,
         gaps: bool = False, v: float | None = None):
    """t map (symmetric scale, +/- v or max |t|) with outlined q < .05 cells; row labels
    coloured by `lobes`; gaps: white lines between lobes. Returns the image."""
    v = v or float(np.nanmax(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    outline(ax, q < 0.05)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=13)
    ax.set_yticks(range(len(labels)), labels, fontsize=13 if not gaps else 11)
    for tick, lb in zip(ax.get_yticklabels(), lobes):
        tick.set_color(fc.LOBE_COLORS[lb])
        tick.set_fontweight("bold")
    lobes = np.array(lobes)
    for i in np.flatnonzero(lobes[1:] != lobes[:-1]) if gaps else []:
        ax.axhline(i + 0.5, color="white", lw=3)
    return im


def heatmap(M: np.ndarray, q: np.ndarray, labels: list, stem: str, figsize: tuple,
            lobes: list, gaps: bool = False) -> None:
    """ROI x frequency t map (draw), saved to <stem>.png/.pdf."""
    fig, ax = plt.subplots(figsize=figsize)
    im = draw(ax, M, q, labels, lobes, [f"{hz:.0f}" for hz in CENTRES], gaps)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.colorbar(im, ax=ax).set_label(t_label(), fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


def run(g: pd.DataFrame, rows: list, stem: str) -> tuple:
    """Per-cell t across subjects, BH-FDR over all rows x bins; writes <stem>_stats.csv;
    returns (t, q) as rows x bins."""
    stats = fc.bin_stats(g[g["roi"].isin(rows)], "freq_hz", "d", fdr="fdr_bh")
    stats.to_csv(fc.table_path(f"{stem}_stats.csv"), index=False)
    stats["bin"] = np.searchsorted(CENTRES, stats["freq_hz"])
    tmap, q = (stats.pivot(index="roi", columns="bin", values=c).reindex(index=rows, columns=range(len(CENTRES)))
               .to_numpy(float) for c in ("t", "q"))
    print(f"[fdr] {os.path.basename(stem)}: {int(np.nansum(q < 0.05))} of {int(np.isfinite(q).sum())} cells q < .05")
    return tmap, q


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"))
    d = spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [f"B-{lobe}" for lobe in fc.LOBES]   # hemispheres pooled
    stem = join(args.out_dir, f"power_spectrum_{BEH}_rois")
    tmap, q = run(subject_table(d, "roi", args.min_electrodes), rows, stem)
    heatmap(tmap, q, [fc.pretty_roi(roi) for roi in rows], stem, (7.5, 5.2), [r.split("-", 1)[1] for r in rows])

    # anatomical subregions (hemispheres pooled), >= MIN_SUBJECTS_ROI subjects, grouped by lobe
    d["region"] = d["fine"].str.replace(r"^[LR] ", "", regex=True)
    d["lobe"] = d["roi"].str.split("-", n=1).str[1]
    fine_subs = subject_table(d, "region", args.min_electrodes)
    n_subs = fine_subs.groupby("roi")["sub"].nunique()
    lobe_of = d.dropna(subset=["region"]).groupby("region")["lobe"].first()
    rows = [r for lb in fc.LOBES for r in n_subs[n_subs >= fc.MIN_SUBJECTS_ROI].sort_values(ascending=False).index
            if lobe_of[r] == lb]
    stem = join(args.out_dir, f"power_spectrum_{BEH}_fine")
    tmap, q = run(fine_subs, rows, stem)
    heatmap(tmap, q, [f"{r} ({n_subs[r]})" for r in rows], stem, (6.5, 0.24 * len(rows) + 1.3),
            [lobe_of[r] for r in rows], gaps=True)


if __name__ == "__main__":
    main()
