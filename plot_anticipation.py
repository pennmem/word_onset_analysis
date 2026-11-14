"""plot_anticipation.py -- does the high-gamma response depend on how long the
participant waited for the word (the jittered blank before it)?

Responsive electrodes as plot_latency.py. Per electrode, trials are split at the
session's median blank; for short- and long-wait trials, the peak (0-800 ms) of the
trial-mean high-gamma envelope minus the (common) word-off mean. Long - short per
electrode -> subject mean per ROI -> mean +/- SEM, one-sample t vs 0, BH-FDR over
ROIs (fc.roi_stats).
Also before the word: the mean envelope in the word-off window (-700..-100 ms, inside every blank), long - short.
-> figures/latency/hg_anticipation_word_on.png (peak change per ROI; bars), _box.png (subjects) and _pre.png (before the word, lobe means +/- SEM)
   figures/latency/wait_times_word_on.png (ISI durations; central 95% of the session medians shaded, overall median dashed)

    python plot_anticipation.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import half_max, response, responsive

BEH = "word_on"


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "latency"))
    lobe_of = fc.load_burke_maps()
    rows, waits, medians = [], [], []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        x, t = response(p)
        pre = np.nanmean(np.asarray(p["env_lo"], float), -1)   # (events, electrodes) word-off window mean
        blank = np.asarray(p["blank_ms_hi"], float)
        waits.append(blank[np.isfinite(blank)])
        medians.append(np.nanmedian(blank))
        long, short = blank > np.nanmedian(blank), blank < np.nanmedian(blank)
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for c in np.flatnonzero(responsive(p)):
            if roi[c] is None:
                continue
            r = {"sub": str(p["sid"][0]), "label": str(p["labels"][c]), "roi": roi[c]}
            pk_l, pk_s = (half_max(np.nanmean(x[m, c], 0), t)[1] for m in (long, short))
            r["peak"] = pk_l - pk_s
            r["pre"] = np.nanmean(pre[long, c]) - np.nanmean(pre[short, c])
            rows.append(r)
    el = pd.DataFrame(rows).groupby(["sub", "label", "roi"], as_index=False)[["peak", "pre"]].mean()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"hg_anticipation_{BEH}")
    el.to_csv(fc.table_path(f"{stem}_per_electrode.csv"), index=False)

    tbl = el.groupby(["sub", "roi"], as_index=False)[["peak", "pre"]].mean()
    st = fc.roi_stats(tbl, "peak")
    st.to_csv(fc.table_path(f"{stem}_peak_stats.csv"), index=False)
    sp = fc.roi_stats(tbl, "pre")   # pre-stimulus: the word-off window (-700..-100 ms), same units and axis as the peak
    sp.to_csv(fc.table_path(f"{stem}_pre_stats.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")   # before the word: its own scale
    fc.roi_dot_panel(ax, {"End of ISI": sp}, "High gamma power, long − short ISI")
    fig.savefig(f"{stem}_pre.png", dpi=200); fig.savefig(f"{stem}_pre.pdf"); plt.close(fig)
    allsub = el.groupby("sub")["peak"].mean()
    print(f"[anticipation] peak, all electrodes: {allsub.mean():+.4g} ± "
          f"{allsub.std() / np.sqrt(len(allsub)):.3g} SEM (n={len(allsub)} subjects)")
    for name, draw in (("", lambda ax: fc.roi_bar_panel(ax, st)),
                       ("_box", lambda ax: fc.roi_panel(ax, tbl, "peak", np.random.default_rng(0), stats=st,
                                                        ylim=(-0.5, 0.5)))):
        fc.roi_figure([("High-gamma response peak\nlong − short ISI", draw)], args.out_dir, f"hg_anticipation_{BEH}{name}")

    if getattr(args, "simulation_tag", None):   # wait times come from the events: the same in every simulation
        return
    # wait (blank) durations, all words; trials are split at each session's own median
    w = np.concatenate(waits)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lo, hi = np.percentile(w, [0.1, 99.9])   # a few pauses of several seconds would stretch the axis
    # bins of one 60 Hz screen frame: word timing is frame-locked, finer bins alias
    ax.hist(w[(w >= lo) & (w <= hi)], bins=np.arange(lo, hi + 1000 / 60, 1000 / 60), color="0.6", edgecolor="white", lw=0.3)
    lo95, hi95 = np.percentile(medians, [2.5, 97.5])
    ax.axvspan(lo95, hi95, color="tab:red", alpha=0.25, lw=0, zorder=3,
               label=f"95% of session\nmedians ({lo95:.0f}–{hi95:.0f} ms)")
    ax.axvline(np.median(w), color="tab:red", ls="--", lw=1, zorder=4, label=f"overall median ({np.median(w):.0f} ms)")
    ax.set_xlabel("ISI before the word (ms)", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    wstem = join(args.out_dir, f"wait_times_{BEH}")
    fig.savefig(f"{wstem}.png", dpi=300)
    fig.savefig(f"{wstem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png, {wstem}.png")


if __name__ == "__main__":
    main()
