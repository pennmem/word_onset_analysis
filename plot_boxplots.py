"""plot_boxplots.py -- word on vs word off per Burke ROI, alpha and high gamma rows.

  power      per-electrode Cohen's d of band power (compute.py power pickles; events and units as
             plot_variance_structure.py: mean of the usable events' power responses)
             -> figures/burke_roi_power/roi_power_word_on(_sem).png
  synchrony  per-electrode mean on - off change over all partners (fc.electrode_sync,
             as Rao et al. 2025): alpha ciPLV, high-gamma AEC-c
             -> figures/burke_roi_synchrony/roi_synchrony_word_on(_sem).png

Sessions are averaged per electrode, electrodes per subject and ROI; one-sample t
vs 0 across subjects, BH-FDR over the 12 ROIs (fc.roi_stats). Two versions of each
figure: subject distributions, and bars of the mean +/- SEM (`_sem`).

    python plot_boxplots.py [--measure power synchrony] [--simulation-tag TAG]
"""
from __future__ import annotations

from os.path import join

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc
import helper
from compute import cond_dir
from plot_variance_structure import power_response, usable_sessions

BEH, BANDS = "word_on", ("alpha", "high_gamma")
MEASURES = ("cohens_d", "change_dB")


def power_tables(save_root: str, lobe_of: dict) -> dict[str, pd.DataFrame]:
    """{band: per (subject, electrode) ISI / event power and Cohen's d over the usable events, sessions averaged}."""
    rows = {band: [] for band in BANDS}
    for pkls, usable in usable_sessions(save_root):
        for band, p in pkls.items():
            roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
            lo, hi, d = (np.nanmean(np.where(usable, v, np.nan), 0) for v in
                         (np.asarray(p["trial_pow_lo"], float), np.asarray(p["trial_pow_hi"], float), power_response(p)))
            rows[band] += [(str(p["sid"][0]), str(label), r, str(fine), *vals)
                           for label, r, fine, *vals in zip(p["labels"], roi, p["reg_full"], lo, hi, d)
                           if r is not None and np.isfinite(vals[0]) and np.isfinite(vals[1])]
    tables = {}
    for band, band_rows in rows.items():
        df = pd.DataFrame(band_rows, columns=["sub", "label", "roi", "fine", "lo", "hi", "cohens_d"])
        df = df.groupby(["sub", "label", "roi", "fine"], as_index=False)[["lo", "hi", "cohens_d"]].mean()
        tables[band] = df.assign(change_dB=10 * np.log10(df["hi"] / df["lo"]))
    return tables


def sync_table(save_root: str, band: str, metric: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode): on - off synchrony to all partners, sessions averaged."""
    rows = []
    for f in tqdm(fc.session_files(cond_dir(save_root, BEH, "diff", band), "_fc_mats.pkl", n_sessions),
                  desc=f"synchrony {band}"):
        mat = fc.load_pickle(str(f))
        pairs = helper.get_pairs(fc.dfrow_from_sid(mat["sid"]))
        if pairs is None or mat.get(metric) is None:
            continue
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of, pool=False)
        rows += [(str(mat["sid"][0]), label, r, float(sync))
                 for label, r, sync in zip(pairs["label"].astype(str), roi, fc.electrode_sync(np.asarray(mat[metric], float)))
                 if r is not None and np.isfinite(sync)]
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "sync_diff"])
    return df.groupby(["sub", "label", "roi"], as_index=False)[["sync_diff"]].mean()


def main() -> None:
    args, save_root = fc.plot_args(__doc__, "figures", **{
        "--measure": dict(nargs="+", default=["power", "synchrony"], choices=["power", "synchrony"])})
    lobe_of = fc.load_burke_maps()
    con = fc.contrast(BEH)
    delta_label = f"({con['delta_label']})"
    if "power" in args.measure:
        fig_dir, stem = join(args.out_dir, "burke_roi_power"), f"roi_power_{BEH}_{{band}}"
        for band, el in power_tables(save_root, lobe_of).items():
            sub_roi = fc.subject_roi_means(el, list(MEASURES), min_electrodes=args.min_electrodes)
            stats = {m: fc.roi_stats(sub_roi, m) for m in MEASURES}
            fc.print_roi_stats(stats["cohens_d"], f"{band} power, Cohen's d {delta_label}:")
            fc.write_roi_csvs(fig_dir, stem.format(band=band), sub_roi, el, stats)
        for bars in (False, True):
            fc.band_contrast_figure(fig_dir, stem, "cohens_d", f"Power, Cohen's d {delta_label}", bars=bars)
    if "synchrony" in args.measure:
        fig_dir, stem = join(args.out_dir, "burke_roi_synchrony"), f"roi_synchrony_{BEH}_{{band}}_{{metric}}"
        for band in BANDS:
            metric = fc.RUN_METRIC[band]
            el = sync_table(save_root, band, metric, lobe_of, args.n_sessions)
            sub_roi = fc.subject_roi_means(el, ["sync_diff"], min_electrodes=args.min_electrodes)
            stats = {"sync_diff": fc.roi_stats(sub_roi, "sync_diff")}
            fc.print_roi_stats(stats["sync_diff"], f"{band} {metric}, mean over partners {delta_label}:")
            fc.write_roi_csvs(fig_dir, stem.format(band=band, metric=metric), sub_roi, el, stats)
        for bars in (False, True):
            fc.band_contrast_figure(fig_dir, stem, "sync_diff", f"{{metric}} {delta_label}", bars=bars)


if __name__ == "__main__":
    main()
