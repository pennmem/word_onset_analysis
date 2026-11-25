"""plot_anticipation.py -- does the high-gamma response depend on how long the
participant waited for the word (the jittered blank before it)?

Responsive electrodes as plot_latency.py. Per electrode, trials are split at the
session's median blank; for short- and long-wait trials, the peak (0-800 ms) of the
trial-mean high-gamma envelope minus the (common) word-off mean. Long - short per
electrode -> subject mean per ROI -> mean +/- SEM, one-sample t vs 0, BH-FDR over
ROIs (fc.roi_stats).
-> figures/latency/hg_anticipation_word_on.png (peak change per ROI; bars; paper). ISI histogram: plot_timing.py.

    python plot_anticipation.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

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
    rows = []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        x, t = response(p)
        blank = np.asarray(p["blank_ms_hi"], float)
        long, short = blank > np.nanmedian(blank), blank < np.nanmedian(blank)
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for c in np.flatnonzero(responsive(p)):
            if roi[c] is None:
                continue
            row = {"sub": str(p["sid"][0]), "label": str(p["labels"][c]), "roi": roi[c]}
            peak_long, peak_short = (half_max(np.nanmean(x[trials, c], 0), t)[1] for trials in (long, short))
            row["peak"] = peak_long - peak_short
            rows.append(row)
    el = pd.DataFrame(rows).groupby(["sub", "label", "roi"], as_index=False)[["peak"]].mean()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"hg_anticipation_{BEH}")
    el.to_csv(fc.table_path(f"{stem}_per_electrode.csv"), index=False)

    per_sub = el.groupby(["sub", "roi"], as_index=False)[["peak"]].mean()
    peak_stats = fc.roi_stats(per_sub, "peak")
    peak_stats.to_csv(fc.table_path(f"{stem}_peak_stats.csv"), index=False)
    sub_peak = el.groupby("sub")["peak"].mean()
    print(f"[anticipation] peak, all electrodes: {sub_peak.mean():+.4g} ± "
          f"{sub_peak.std() / np.sqrt(len(sub_peak)):.3g} SEM (n={len(sub_peak)} subjects)")
    fc.roi_figure([("High-gamma response peak\nlong − short ISI", lambda ax: fc.roi_bar_panel(ax, peak_stats))],
                  args.out_dir, f"hg_anticipation_{BEH}")


if __name__ == "__main__":
    main()
