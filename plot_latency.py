"""plot_latency.py -- order of activation across ROIs from high-gamma power.

Reads compute.py's high-gamma power pickles (per-trial sub-band Hilbert envelopes,
10 ms steps, each sub-band normalized by its word-off mean). Per electrode:

  responsive  per-trial power, word on (0-600 ms) > word off: one-sided Welch t,
              BH-FDR over the session's electrodes, q < .05
  latency     trial-mean envelope minus its word-off mean, smoothed (30 ms); the
              first time it reaches half its peak within 0-800 ms (half-maximum)

Regions: anatomical labels within the Burke ROIs, hemispheres pooled (--rois: the 12 ROIs).
Sessions are averaged per electrode, each subject contributes its median latency
per region, and regions are compared by the across-subject mean +/- 95% CI (regions
with >= MIN_SUBJECTS_ROI subjects). Figure: regions ordered by latency (bars), and each
region's group-mean response over time (curves, each scaled to its peak).

    python plot_latency.py [--rois] [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import t as tdist, ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir

PEAK_WIN_MS = (0, 800)
SMOOTH_MS = 30


def responsive(p: dict) -> np.ndarray:
    """(C,) word on > word off per-trial power: one-sided Welch t, BH-FDR within session, q < .05."""
    pow_off, pow_on = np.asarray(p["trial_pow_lo"], float), np.asarray(p["trial_pow_hi"], float)
    _, pvals = ttest_ind(pow_on, pow_off, axis=0, equal_var=False, alternative="greater", nan_policy="omit")
    finite = np.isfinite(pvals)
    resp = np.zeros(len(pvals), bool)
    if finite.any():
        resp[finite] = multipletests(pvals[finite], method="fdr_bh")[0]
    return resp


def response(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial high-gamma envelope minus each electrode's word-off mean, smoothed
    (SMOOTH_MS, edge-padded: no dip at the ends): (E, C, T), and times (T,)."""
    step = float(p["env_step_ms"])
    x = np.asarray(p["env_hi"], float) - np.nanmean(np.asarray(p["env_lo"], float), axis=(0, 2))[:, None]
    x = uniform_filter1d(x, max(1, int(round(SMOOTH_MS / step))), axis=-1, mode="nearest")
    return x, p["env_hi_t0_ms"] + step * (np.arange(x.shape[-1]) + 0.5)


def half_max(c: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(latency, peak) along the last axis within PEAK_WIN_MS: latency = first time the
    curve reaches half its peak; NaN where the peak is <= 0."""
    in_win = (times >= PEAK_WIN_MS[0]) & (times < PEAK_WIN_MS[1])
    c, t = c[..., in_win], times[in_win]
    peak = c.max(-1)
    lat = t[np.argmax(c >= peak[..., None] / 2, -1)]
    return np.where(peak > 0, lat, np.nan), peak


def electrode_rows(p: dict, lobe_of: dict) -> list[tuple]:
    """(sub, label, roi, latency_ms, trial-mean response curve, fine label) per responsive electrode."""
    x, times = response(p)
    curve = np.nanmean(x, axis=0)   # (C, T); smoothing is linear, so = smoothed trial mean
    lat, _ = half_max(curve, times)
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    out = []
    for c in np.flatnonzero(responsive(p)):
        if roi[c] is None or np.isnan(lat[c]):
            continue
        out.append((str(p["sid"][0]), str(p["labels"][c]), roi[c], float(lat[c]), curve[c], str(p["reg_full"][c])))
    return out


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "latency"), **{
        "--rois": dict(action="store_true", help="the 12 Burke ROIs instead of anatomical regions (hemispheres pooled)")})
    lobe_of = fc.load_burke_maps()
    os.makedirs(args.out_dir, exist_ok=True)
    rows, times = [], None
    n_elec = 0
    for f in tqdm(fc.session_files(power_dir(save_root, "word_on", "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        n_elec += len(p["labels"])
        if times is None:
            times = p["env_hi_t0_ms"] + p["env_step_ms"] * (np.arange(np.shape(p["env_hi"])[-1]) + 0.5)
        rows += electrode_rows(p, lobe_of)
    # anatomical label without hemisphere, or (--rois) the Burke ROI
    region = [r[2] if args.rois else r[5].split(" ", 1)[1] for r in rows]
    lobe = {reg: r[2].split("-", 1)[1] for reg, r in zip(region, rows)}   # for colour
    el = pd.DataFrame([(r[0], r[1], reg, r[3]) for r, reg in zip(rows, region)],
                      columns=["sub", "label", "roi", "latency_ms"])
    curves = np.array([r[4] for r in rows])
    (pd.DataFrame([(r[0], r[1], r[2], r[5], r[3]) for r in rows], columns=["sub", "label", "roi", "fine", "latency_ms"])
       .groupby(["sub", "label", "roi", "fine"], as_index=False)["latency_ms"].mean()
       .to_csv(fc.table_path(join(args.out_dir, "hg_latency_word_on_electrodes.csv")), index=False))   # mixed_models.R latency
    print(f"[latency] {len(el)} responsive electrode-sessions of {n_elec}, {el['sub'].nunique()} subjects")

    # sessions -> electrode (mean), electrodes -> subject x ROI (median)
    el = el.groupby(["sub", "label", "roi"], as_index=False)["latency_ms"].mean()
    per_sub = el.groupby(["sub", "roi"], as_index=False).agg(latency_ms=("latency_ms", "median"),
                                                              n_elec=("label", "size"))
    stats = []
    for roi, roi_subs in per_sub.groupby("roi"):
        mean, ci = fc.mean_ci(roi_subs["latency_ms"].to_numpy())
        stats.append({"roi": roi, "n_subjects": len(roi_subs), "n_electrodes": int(roi_subs["n_elec"].sum()),
                      "mean_ms": mean, "ci95_ms": ci,
                      "sem_ms": float(roi_subs["latency_ms"].std(ddof=1) / np.sqrt(len(roi_subs))),
                      "median_ms": float(roi_subs["latency_ms"].median())})
    stats = pd.DataFrame(stats).sort_values("mean_ms")
    stem = join(args.out_dir, "hg_latency_word_on" + ("" if args.rois else "_fine_pooled"))
    stats.to_csv(fc.table_path(f"{stem}_stats.csv"), index=False)
    per_sub.to_csv(fc.table_path(f"{stem}_per_subject.csv"), index=False)
    print(stats.round(1).to_string(index=False))

    shown = stats[stats["n_subjects"] >= fc.MIN_SUBJECTS_ROI]
    # electrode curves -> subject mean -> mean over subjects
    curve_df = pd.DataFrame(curves)
    curve_df[["sub", "roi"]] = [(r[0], reg) for r, reg in zip(rows, region)]
    shown_t = (times >= 0) & (times < 1000)
    sub_curves = curve_df.groupby(["roi", "sub"]).mean().loc[:, shown_t]
    by_roi = sub_curves.groupby("roi")
    profile = by_roi.mean().reindex(shown["roi"])
    # pointwise 95% CI across subjects (t-based), descriptive
    ci = (by_roi.std(ddof=1) / np.sqrt(by_roi.size().to_numpy()[:, None])
          * tdist.ppf(0.975, by_roi.size().to_numpy()[:, None] - 1)).reindex(shown["roi"])
    peak = profile.max(axis=1)
    profile, ci = profile.div(peak, axis=0), ci.div(peak, axis=0)

    # bars and curves share the rows (one set of row labels, on the bars)
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(8.5, 0.36 * len(shown) + 1.8), sharey=True,
                                             gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.05})
    y = np.arange(len(shown))
    xmax = float((shown["mean_ms"] + shown["sem_ms"]).max()) * 1.25
    for i, r in enumerate(shown.itertuples()):
        ax_bar.barh(i, r.mean_ms, xerr=r.sem_ms, height=0.7, capsize=3, edgecolor="0.15", ecolor="0.2",
                    color=fc.LOBE_COLORS.get(lobe[r.roi], "0.5"))
        ax_bar.annotate(f"n={r.n_subjects}", (xmax * 0.98, i), ha="right", va="center", fontsize=11, color="0.4")
    ax_bar.set_yticks(y, [fc.pretty_roi(r) for r in shown["roi"]], fontsize=13)
    for tick, r in zip(ax_bar.get_yticklabels(), shown["roi"]):
        tick.set_color(fc.LOBE_COLORS.get(lobe[r], "k"))
        tick.set_fontweight("bold")
    for ax in (ax_bar, ax_curve):
        ax.tick_params(axis="x", labelsize=12)
    ax_bar.set_ylim(len(shown) - 0.5, -0.5)   # first row at the top, for both panels
    ax_bar.set_xlim(0, xmax)
    ax_bar.set_xlabel("Half-max latency (ms)", fontsize=14)
    ax_bar.spines[["top", "right"]].set_visible(False)
    # one curve per row, baseline at the bottom of the row
    t_shown = times[shown_t]
    for i, (r, curve, curve_ci) in enumerate(zip(shown["roi"], profile.to_numpy(), ci.to_numpy())):
        color = fc.LOBE_COLORS.get(lobe[r], "0.5")
        ax_curve.axhline(i + 0.4, color="0.85", lw=0.6, zorder=0)
        ax_curve.fill_between(t_shown, i + 0.4 - 0.8 * (curve - curve_ci), i + 0.4 - 0.8 * (curve + curve_ci),
                              color=color, alpha=0.3, lw=0)
        ax_curve.plot(t_shown, i + 0.4 - 0.8 * curve, color=color, lw=1.5)
    ax_curve.set_xlim(0, 1000)
    ax_curve.tick_params(axis="y", left=False, labelleft=False)
    ax_curve.spines[["top", "right", "left"]].set_visible(False)
    ax_curve.set_xlabel("Time after word onset (ms)", fontsize=14)
    ax_curve.yaxis.set_label_position("right")
    ax_curve.set_ylabel("High-gamma response\n(fraction of peak)", fontsize=13)
    fc.lobe_key(fig, y=0.93, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")   # keeps the row labels
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
