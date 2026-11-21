"""plot_variance_structure.py -- where the word response varies.

Reads compute.py's power pickles (alpha, high gamma). Per word event and electrode-session, the power
response in the paper's units: (word-on power - mean word-off power) / the pooled SD of Cohen's d, whose
mean over the session's word events is the cached Cohen's d. Latency: half-max of plot_latency.response.
Outlier events dropped per electrode (log word-on power or whole-clip peak > OUTLIER_MAD x MAD).
Scored lists, serial positions 2-12 (position 1 follows the countdown and is not in the events).

  1. Variance components (mixed_models.R variance): DeHaan 2026 ch. 3's random intercepts (word, serial
     position, list, subject, subject:word, session, subject:serial position, list:serial position, list in
     session, session:serial position), plus electrode, electrode in session and the word event shared by a
     lobe's electrodes; residual: electrode and event. At most --max-electrodes random electrodes per subject
     and lobe (seed 0; refit: --refit). Stacked shares per lobe, power and latency apart.
  2. Task effects: per serial position and list number, all electrodes (latency: responsive): subject and lobe
     means as deviations from the subject's mean, mean and 95% CI across subjects; per-subject linear slopes
     vs 0 (one-sample t), BH-FDR over lobes x measures x terms.
  3. Alpha vs high gamma: per electrode Pearson r across word events; per subject r across electrodes of
     Cohen's d and of |Cohen's d|; per lobe, subjects' mean Fisher z vs 0, BH-FDR over measures x lobes.
     Prevalence: paired t (word on vs off) over word events per electrode, BH-FDR 0.05.
  4. Recovery from the previous word's offset (not yet in the paper's units): each event's word-off window
     (-700 to -100 ms before the word) re-aligned to time since the previous word disappeared: high-gamma
     envelope vs the electrode's word-off mean (responsive electrodes) and alpha log power vs its session
     mean (window centre); per subject and lobe, mean and 95% CI across subjects.
  5. Blank interval and the neighbouring words (mixed_models.R blank): per lobe and band, the response
     ~ the previous word's response x blank length + the response two words back (z, same electrode and
     list; drift vs carry-over) + section 1's random intercepts. Estimates with 95% Wald CIs; coefficients
     BH-FDR over lobes x bands x terms (a check). Figure: carry-over by correlation, all electrodes: per
     electrode-session, serial-position means out, r between each word's response and the one 1 / 2 words
     back in the same list; r with the previous word after long - short blanks (session median, as
     plot_anticipation). Sessions -> electrode -> subject x lobe; t vs 0, BH-FDR over lobes (fc.roi_stats).
  6. --single (proposal Fig. 1C): the section 1 model within single subjects (mixed_models.R single), occipital,
     subjects with >= 5 electrodes and >= 2 sessions, all their electrodes; shares of the group model's occipital
     row, the mean single subject and the subject with the most electrodes x sessions. Sections 1-5 skipped.
  7. --recall (proposal Fig. 4): per region, recall ~ high-gamma power during the word and during the ISI before it
     + latency (mixed_models.R recall), each z-scored per electrode within session (Rubinstein et al. 2023) and
     averaged over the region's electrodes (latency: responsive electrodes). Sections 1-5 skipped.
-> figures/variance_structure/*.png (tables and model fits: results/variance_structure/)

    python plot_variance_structure.py [--n-subjects N] [--from-csv] [--refit] [--cores C] [--max-electrodes M]
    python plot_variance_structure.py --single --n-subjects 10000 --max-electrodes 100000 --out-dir figures/single_subject
    python plot_variance_structure.py --recall --n-subjects 10000 --out-dir figures/recall
"""
from __future__ import annotations

import os
import subprocess
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist, ttest_1samp
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import _blank_before, power_dir
from plot_latency import half_max, response, responsive

BEH = "word_on"
N_SUBJECTS = 40
OUTLIER_MAD = 10
LMM_MAX_ELECTRODES = 5
SP_RANGE = (2, 12)
LEVELS = 26   # bins for serial position (<= 12) and list number (<= 25)
OFFSET_BIN_MS = 20
OFFSET_MAX_MS = 1300
BANDS = {"alpha": "Alpha power", "high_gamma": "High-gamma power", "latency": "High-gamma latency\n(responsive electrodes)"}
GROUPS = {   # stacked share segments: label -> (model terms, colour)
    "Subject": (["subject"], "#1f4fa3"), "Region": (["region"], "#17becf"), "Electrode": (["electrode"], "#d62728"),
    "Electrode:Session": (["elsess"], "#f2c12e"),
    "Session": (["session"], "#8c564b"), "Session:List": (["sess_list"], "#ff7f0e"),
    "Task (serial position, list, word)": (["sp", "list", "item", "subj_item", "subj_sp", "list_sp", "sess_sp"], "#2ca02c"),
    "Word event": (["event"], "#7b4fa3"),
    "Residual (electrode, event)": (["Residual"], "#d9d9d9")}

def robust_ok(v: np.ndarray) -> np.ndarray:
    """(trials, electrodes) True where the event is not a robust outlier of its electrode."""
    med = np.nanmedian(v, 0); mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    return np.abs(v - med) <= OUTLIER_MAD * mad


def pooled_sd(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """(electrodes,) the pooled SD of helper.cohens_d."""
    return np.sqrt(((len(hi) - 1) * hi.var(0, ddof=1) + (len(lo) - 1) * lo.var(0, ddof=1)) / (len(hi) + len(lo) - 2))


def lag_r(v: np.ndarray, sp: np.ndarray, lst: np.ndarray, lag: int, cur: np.ndarray | None = None) -> float:
    """r between each event's value and the value `lag` words earlier in the same list (current events
    restricted to `cur`); NaN below 10 pairs."""
    now, earlier = v[lag:], v[:-lag]
    paired = (lst[lag:] == lst[:-lag]) & (sp[lag:] == sp[:-lag] + lag) & np.isfinite(now) & np.isfinite(earlier)
    if cur is not None:
        paired &= cur[lag:]
    return np.corrcoef(now[paired], earlier[paired])[0, 1] if paired.sum() >= 10 else np.nan


def xcorr_latency(x: np.ndarray, t: np.ndarray, max_lag_ms: float = 200,
                  tmp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(latency, best r) per event by cross-correlation (Woody 1967): the electrode's average curve (template, zero
    before word onset) slid over each event's curve within 0-800 ms; latency = template half-max latency + the lag
    of the best Pearson r (all events kept; best r saved for cutoff checks); NaN where the template has no half-max.
    `tmp`: another template (e.g. the electrode's average when x holds condition averages)."""
    tmp = np.nanmean(x, 0) if tmp is None else tmp
    template_latency = half_max(tmp, t)[0]
    if not np.isfinite(template_latency):
        return np.full(len(x), np.nan), np.full(len(x), np.nan)
    step = t[1] - t[0]; max_lag = int(round(max_lag_ms / step))
    window = np.flatnonzero((t >= 0) & (t < 800))
    padded = np.r_[np.zeros(max_lag), tmp, np.zeros(max_lag)]
    shifted = np.stack([padded[max_lag + window - lag] for lag in range(-max_lag, max_lag + 1)], 1)
    shifted = (shifted - shifted.mean(0)) / (shifted.std(0) + 1e-12)
    events = x[:, window]; events = (events - events.mean(1, keepdims=True)) / (events.std(1, keepdims=True) + 1e-12)
    r = events @ shifted / len(window)
    best = np.nanargmax(np.where(np.isfinite(r), r, -np.inf), 1)
    return template_latency + (best - max_lag) * step, r[np.arange(len(x)), best]


def task_columns(w: pd.DataFrame, blank: np.ndarray) -> pd.DataFrame:
    """Per event: serial position, list, word, blank length (z within session; NaN where it cannot be
    measured: session median)."""
    list_num = w["trial"].to_numpy(float)
    blank_filled = np.where(np.isfinite(blank), blank, np.nanmedian(blank))
    item = w["item_name"] if "item_name" in w else w["item"]
    return pd.DataFrame({"sp": w["serialpos"].to_numpy(int), "list": list_num.astype(int),
                         "item": item.astype(str).str.upper().to_numpy(),
                         "blank_z": (blank_filled - blank_filled.mean()) / (blank_filled.std() or 1)})


def z_within_session(values: np.ndarray) -> np.ndarray:
    return (values - np.nanmean(values, 0)) / np.nanstd(values, 0)


def recall_figure(tdir: str, out_dir: str) -> None:
    """7. Recall model coefficients (log-odds per SD +/- SE) per predictor and region, BH-FDR over all."""
    fixed = pd.read_csv(join(tdir, "recall_fixed.csv"))
    fixed = fixed[fixed["term"] != "(Intercept)"].copy()
    fixed["q"] = multipletests(2 * norm.sf(np.abs(fixed["est"] / fixed["se"])), method="fdr_bh")[1]
    fixed.to_csv(join(tdir, "recall_stats.csv"), index=False)
    labels = {"word_power": "Word", "isi_power": "ISI", "latency": "Latency"}
    stats = {label: fixed[fixed["term"] == term].assign(roi="B-" + fixed["lobe"], mean=fixed["est"], sem=fixed["se"])
             for term, label in labels.items()}
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
    fc.roi_dot_panel(ax, stats, "Recall, log-odds per SD")
    fig.savefig(join(out_dir, "recall_model.png"), dpi=200); plt.close(fig)
    print(fixed.round(4).to_string(index=False))


def single_subject_figure(tdir: str, out_dir: str) -> None:
    """6. Occipital shares as stacked bars: group model (Fig. 1B's occipital row), mean single subject, example."""
    single = pd.read_csv(join(tdir, "components_single.csv"))
    single["share"] = single["var"] / single.groupby(["subject", "band"])["var"].transform("sum")
    info = single.groupby("subject")[["nel", "nses"]].first()
    example = (info["nel"] * info["nses"]).idxmax()
    group = pd.read_csv(join(os.path.dirname(fc.table_path(join("figures", "variance_structure", "x"))), "components.csv"))
    group = group[group["lobe"] == "occipital"].assign(share=lambda d: d["var"] / d.groupby("band")["var"].transform("sum"))
    rows = {"Group model\n(Fig. 1B)": lambda band: group[group["band"] == band].set_index("term")["share"],
            f"Single subjects\n(mean of {single['subject'].nunique()})":
                lambda band: single[single["band"] == band].pivot_table(index="subject", columns="term", values="share").fillna(0).mean(),
            f"Example subject\n({info.loc[example, 'nel']} electrodes,\n{info.loc[example, 'nses']} sessions)":
                lambda band: single[(single["band"] == band) & (single["subject"] == example)].set_index("term")["share"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), sharey=True)
    for ax, band in zip(axes, ("alpha", "high_gamma")):
        for y, shares_of in enumerate(rows.values()):
            shares, left = shares_of(band), 0.0
            for group_name, (terms, col) in GROUPS.items():
                width = float(shares.reindex(terms).fillna(0).sum())
                ax.barh(y, width, left=left, color=col, height=0.75, edgecolor="w", lw=1.5, label=group_name if y == 0 else None)
                if width >= 0.06:
                    ax.text(left + width / 2, y, f"{width:.0%}", ha="center", va="center", fontsize=11,
                            color="k" if col in ("#f2c12e", "#d9d9d9") else "w")
                left += width
        ax.set_yticks(range(len(rows)), list(rows), fontsize=10); ax.set_xlim(0, 1)
        ax.set_title(f"{BANDS[band]}, occipital", fontsize=13); ax.set_xlabel("Share of variance across word events", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(join(out_dir, "single_subject_shares.png"), dpi=200, bbox_inches="tight")
    print("[single] example subject", example, info.loc[example].to_dict())


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variance_structure"), **{
        "--n-subjects": dict(type=int, default=N_SUBJECTS, help="random subjects (seed 0)"),
        "--from-csv": dict(action="store_true", help="reuse the saved tables and model fits"),
        "--refit": dict(action="store_true", help="refit the single-event mixed models"),
        "--cores": dict(type=int, default=6, help="parallel model fits in R"),
        "--max-electrodes": dict(type=int, default=LMM_MAX_ELECTRODES, help="per subject and lobe, in the models"),
        "--single": dict(action="store_true", help="6. single-subject fits (Fig. 1C) instead of sections 1-5"),
        "--recall": dict(action="store_true", help="7. recall model (Fig. 4) instead of sections 1-5")})
    os.makedirs(args.out_dir, exist_ok=True)
    tdir = os.path.dirname(fc.table_path(join(args.out_dir, "x")))
    lobe_of = fc.load_burke_maps()
    files = {band: {f.name: f for f in fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", None)}
             for band in ("alpha", "high_gamma")}
    subjects = sorted({fname.split("_")[0] for fname in files["high_gamma"]})
    picked = set(np.random.default_rng(0).choice(subjects, min(args.n_subjects, len(subjects)), replace=False))
    names = sorted(fname for fname in files["high_gamma"] if fname.split("_")[0] in picked and fname in files["alpha"])

    if not args.from_csv:
        trial_rows, electrode_rows, recall_rows, offset_sums, level_sums = [], [], [], {}, {}
        for fname in tqdm(names, desc="sessions"):
            hg_pkl, alpha_pkl = (fc.load_pickle(str(files[band][fname])) for band in ("high_gamma", "alpha"))
            sub, sess = str(hg_pkl["sid"][0]), str(hg_pkl["sid"])
            events = fc.load_events(fc.dfrow_from_sid(hg_pkl["sid"]), BEH)
            blank = _blank_before(events)[np.asarray(events.attrs["mask"], bool)]   # on all words: position 2 follows position 1
            events = events[np.asarray(events.attrs["mask"], bool)]
            if len(events) != len(np.asarray(hg_pkl["trial_pow_hi"])):
                continue
            roi = fc.roi_of_reg_full(hg_pkl["reg_full"], lobe_of)
            lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
            curves, t = response(hg_pkl)   # (events, electrodes, time)
            clip_peak = np.nanmax(np.abs(curves), -1)
            log_alpha_on, log_alpha_off = np.log(np.asarray(alpha_pkl["trial_pow_hi"], float)), np.log(np.asarray(alpha_pkl["trial_pow_lo"], float))
            pow_on, pow_off = {}, {}
            for k, pkl in (("a", alpha_pkl), ("g", hg_pkl)):
                pow_on[k], pow_off[k] = np.asarray(pkl["trial_pow_hi"], float), np.asarray(pkl["trial_pow_lo"], float)
            sd_alpha, sd_hg = pooled_sd(pow_on["a"], pow_off["a"]), pooled_sd(pow_on["g"], pow_off["g"])
            alpha_resp, hg_resp = (pow_on["a"] - pow_off["a"].mean(0)) / sd_alpha, (pow_on["g"] - pow_off["g"].mean(0)) / sd_hg   # mean over events: Cohen's d
            alpha_diff, hg_diff = (pow_on["a"] - pow_off["a"]) / sd_alpha, (pow_on["g"] - pow_off["g"]) / sd_hg   # word on - off, same event
            kept = robust_ok(clip_peak) & robust_ok(np.log(pow_on["g"])) & robust_ok(log_alpha_on)
            curves[~kept] = np.nan; alpha_resp[~kept] = np.nan; hg_resp[~kept] = np.nan
            task = task_columns(events, blank)
            use = (task["list"].to_numpy() >= 1) & task["sp"].between(*SP_RANGE).to_numpy()
            is_resp = responsive(hg_pkl)
            if args.recall:
                usable = use[:, None] & kept
                with np.errstate(all="ignore"):
                    word_power = z_within_session(np.where(usable, np.log(pow_on["g"]), np.nan))
                    isi_power = z_within_session(np.where(usable, np.log(pow_off["g"]), np.nan))
                    latency = np.full(word_power.shape, np.nan)
                    for c in np.flatnonzero(is_resp & (lobe != None)):  # noqa: E711
                        if usable[:, c].sum() >= 20:
                            latency[usable[:, c], c] = xcorr_latency(curves[usable[:, c], c], t)[0]
                    latency = z_within_session(latency)
                    for region in set(lobe[lobe != None]):  # noqa: E711
                        in_region = lobe == region
                        recall_rows.append(task[use].assign(
                            subject=sub, session=sess, lobe=region, event=np.flatnonzero(use),
                            recalled=events["correct_recall"].to_numpy(float)[use],
                            word_power=np.nanmean(word_power[use][:, in_region], 1),
                            isi_power=np.nanmean(isi_power[use][:, in_region], 1),
                            latency=np.nanmean(latency[use][:, in_region], 1)))
                continue
            sp, list_num = task["sp"].to_numpy(), task["list"].to_numpy()
            long_blank = blank > np.nanmedian(blank); short_blank = blank < np.nanmedian(blank)   # split as plot_anticipation
            design = "800-1200" if str(hg_pkl["sid"][1]) == "pyFR" else "750-1000"
            off_env = np.asarray(hg_pkl["env_lo"], float)   # (events, electrodes, time)
            if off_env.shape[:2] == curves.shape[:2]:   # 4. recovery
                hg_off = off_env - np.nanmean(off_env, axis=(0, 2))[:, None]
                hg_off[~kept] = np.nan
                alpha_off = log_alpha_off - np.nanmean(log_alpha_off, 0); alpha_off[~kept] = np.nan
                t_off = hg_pkl["env_lo_t0_ms"] + hg_pkl["env_step_ms"] * (np.arange(hg_off.shape[-1]) + 0.5)
                has_blank = use & np.isfinite(blank)
                for lb in fc.LOBES:
                    lobe_els = is_resp & (lobe == lb)
                    if lobe_els.any():
                        for k, vals, since in (("high_gamma", np.nanmean(hg_off[has_blank][:, lobe_els], 1), blank[has_blank, None] + t_off[None]),
                                               ("alpha", np.nanmean(alpha_off[has_blank][:, lobe_els], 1)[:, None], blank[has_blank, None] + t_off.mean())):
                            bins, in_bins = (since // OFFSET_BIN_MS).astype(int).ravel(), np.isfinite(vals.ravel())
                            n_bins = OFFSET_MAX_MS // OFFSET_BIN_MS
                            in_bins &= (bins >= 0) & (bins < n_bins)
                            sums, counts = offset_sums.get((sub, lb, k), (np.zeros(n_bins), np.zeros(n_bins)))
                            offset_sums[(sub, lb, k)] = (sums + np.bincount(bins[in_bins], vals.ravel()[in_bins], n_bins),
                                                         counts + np.bincount(bins[in_bins], minlength=n_bins))
            for c in np.flatnonzero(lobe != None):  # noqa: E711
                el = f"{sub}|{hg_pkl['labels'][c]}"
                ev_mask = use & np.isfinite(hg_resp[:, c]) & np.isfinite(alpha_resp[:, c])
                if ev_mask.sum() < 20:
                    continue
                lat, peak = half_max(curves[ev_mask, c], t) if is_resp[c] else (np.full(ev_mask.sum(), np.nan),) * 2
                lat_xc, r_xc = xcorr_latency(curves[ev_mask, c], t) if is_resp[c] else (np.full(ev_mask.sum(), np.nan),) * 2
                for meas, vals in (("alpha", alpha_resp[ev_mask, c]), ("high_gamma", hg_resp[ev_mask, c]), ("latency", lat)):
                    for term in ("sp", "list"):
                        levels, finite = task.loc[ev_mask, term].to_numpy(), np.isfinite(vals)
                        sums, counts = level_sums.get((sub, lobe[c], meas, term), (np.zeros(LEVELS), np.zeros(LEVELS)))
                        level_sums[(sub, lobe[c], meas, term)] = (sums + np.bincount(levels[finite], vals[finite], LEVELS), counts + np.bincount(levels[finite], minlength=LEVELS))
                subregion = hg_pkl["reg_full"][c]   # hemispheres pooled
                trial_rows.append(task[ev_mask].assign(subject=sub, electrode=el, session=sess, lobe=lobe[c],
                                                       region=subregion[2:] if subregion[:2] in ("L ", "R ") else subregion, event=np.flatnonzero(ev_mask), alpha=alpha_resp[ev_mask, c], high_gamma=hg_resp[ev_mask, c],
                                                       latency=lat, latency_xc=lat_xc, xc_r=r_xc, peak=peak, blank_ms=blank[ev_mask], design=design))
                corrs, resid = {}, {}   # 5. serial-position means out
                for k, series in (("alpha", alpha_resp[:, c]), ("high_gamma", hg_resp[:, c])):
                    series = np.where(ev_mask, series, np.nan)
                    series = series - pd.Series(series).groupby(sp).transform("mean").to_numpy()
                    corrs.update({f"r1_{k}": lag_r(series, sp, list_num, 1), f"r2_{k}": lag_r(series, sp, list_num, 2),
                                  f"r1long_{k}": lag_r(series, sp, list_num, 1, long_blank), f"r1short_{k}": lag_r(series, sp, list_num, 1, short_blank)})
                    resid[k] = series - pd.Series(series).groupby(list_num).transform("mean").to_numpy()   # 3. list means out too
                both = np.isfinite(resid["alpha"]) & np.isfinite(resid["high_gamma"])
                corrs["r_fast"] = np.corrcoef(resid["alpha"][both], resid["high_gamma"][both])[0, 1]
                if is_resp[c]:   # serial-position and list means out
                    lat_dev, peak_dev = np.full(len(ev_mask), np.nan), np.full(len(ev_mask), np.nan); lat_dev[ev_mask], peak_dev[ev_mask] = lat_xc, peak
                    for grouping in (sp, list_num):
                        lat_dev = lat_dev - pd.Series(lat_dev).groupby(grouping).transform("mean").to_numpy()
                        peak_dev = peak_dev - pd.Series(peak_dev).groupby(grouping).transform("mean").to_numpy()
                    both = np.isfinite(lat_dev) & np.isfinite(peak_dev)
                    corrs["r_amp_lat"] = np.corrcoef(peak_dev[both], lat_dev[both])[0, 1] if both.sum() >= 20 else np.nan
                electrode_rows.append({**corrs, "subject": sub, "electrode": el, "session": sess, "lobe": lobe[c],
                                       "d_alpha": alpha_pkl["cohens_d"][c], "d_hg": hg_pkl["cohens_d"][c],
                                       "r_trial": np.corrcoef(alpha_resp[ev_mask, c], hg_resp[ev_mask, c])[0, 1], "n": ev_mask.sum(),
                                       "pa_sum": alpha_diff[ev_mask, c].sum(), "pa_sq": (alpha_diff[ev_mask, c] ** 2).sum(),
                                       "pg_sum": hg_diff[ev_mask, c].sum(), "pg_sq": (hg_diff[ev_mask, c] ** 2).sum()})
        if args.recall:
            pd.concat(recall_rows, ignore_index=True).to_csv(join(tdir, "recall.csv"), index=False)
        else:
            trials = pd.concat(trial_rows, ignore_index=True)
            print(f"[variance] {trials['subject'].nunique()} subjects, {trials['electrode'].nunique()} electrodes, "
                  f"{trials['session'].nunique()} sessions, {len(trials)} electrode-events")
            rng = np.random.default_rng(0)
            pick_electrodes = lambda d: {el for els in d.groupby(["subject", "lobe"])["electrode"].unique() for el in rng.permutation(els)[:args.max_electrodes]}
            keep_power = pick_electrodes(trials)
            keep_lat = pick_electrodes(trials[trials["latency"].notna()])   # responsive electrodes
            trials = trials[trials["electrode"].isin(keep_power | keep_lat)].assign(in_power=lambda d: d["electrode"].isin(keep_power))
            trials.loc[~trials["electrode"].isin(keep_lat), "latency"] = np.nan
            trials.to_csv(join(tdir, "trials.csv"), index=False)
            del trials
            pd.DataFrame(electrode_rows).to_csv(join(tdir, "electrode_sessions.csv"), index=False)
            pd.to_pickle(offset_sums, join(tdir, "offset.pkl")); pd.to_pickle(level_sums, join(tdir, "levels.pkl"))
    for mode, out in ((("single", "components_single.csv"),) if args.single else
                      (("recall", "recall_fixed.csv"),) if args.recall else
                      (("variance", "components.csv"), ("blank", "blank_fixed.csv"))):
        if args.refit or not os.path.exists(join(tdir, out)):
            subprocess.run(["Rscript", "mixed_models.R", mode, tdir, str(args.cores)], check=True)
    if args.single:
        return single_subject_figure(tdir, args.out_dir)
    if args.recall:
        return recall_figure(tdir, args.out_dir)
    el_sess = pd.read_csv(join(tdir, "electrode_sessions.csv"))
    comps = pd.read_csv(join(tdir, "components.csv"))
    comps["share"] = comps["var"] / comps.groupby(["lobe", "band"])["var"].transform("sum")
    lobes = [lb for lb in fc.LOBES if lb in set(comps["lobe"])]

    # 1. variance components
    for name, bands in (("variance_components_power", ["alpha", "high_gamma"]), ("variance_components_latency", ["latency"])):
        fig, axes = plt.subplots(1, len(bands), figsize=(6.5 * len(bands), 4.6), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (b, lab) in zip(axes, [(b, BANDS[b]) for b in bands]):
            shares = comps[comps["band"] == b].pivot(index="lobe", columns="term", values="share").reindex(lobes).fillna(0)
            left = np.zeros(len(shares))
            for group, (terms, col) in GROUPS.items():
                width = shares.reindex(columns=terms, fill_value=0).sum(1).to_numpy()
                ax.barh(range(len(shares)), width, left=left, color=col, height=0.75, edgecolor="w", lw=1.5, label=group)
                for y, (seg_left, seg_width) in enumerate(zip(left, width)):
                    if seg_width >= 0.06:
                        ax.text(seg_left + seg_width / 2, y, f"{seg_width:.0%}", ha="center", va="center", fontsize=11,
                                color="k" if col in ("#f2c12e", "#d9d9d9") else "w")
                left += width
            sizes = comps[comps["band"] == b].groupby("lobe")[["nsub", "nel"]].first().reindex(shares.index)
            for y, row in enumerate(sizes.itertuples()):
                ax.text(1.01, y, f"{row.nsub}/{row.nel}", va="center", fontsize=9, color="0.35")
            ax.set_yticks(range(len(shares)), [f"$\\bf{{{l.capitalize()}}}$" for l in shares.index], fontsize=12)
            ax.set_xlim(0, 1); ax.set_title(lab, fontsize=14); ax.set_xlabel("Share of variance across word events", fontsize=12)
            ax.tick_params(axis="x", labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()   # shared y: once for all panels
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=12, ncol=3 if len(bands) > 1 else 2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
        fig.text(0.995, 0.02, "n = subjects / electrodes", ha="right", fontsize=9, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(join(args.out_dir, f"{name}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # 2. task effects
    level_sums = pd.read_pickle(join(tdir, "levels.pkl"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), sharex="col")
    trend = []
    for i, (b, lab) in enumerate(BANDS.items()):
        for k, (term, xlabel) in enumerate((("sp", "Serial position"), ("list", "List number"))):
            ax = axes[i, k]
            for lb in lobes:
                subj_means = np.array([sm / np.where(n >= 5, n, np.nan) for (s_, l_, b_, t_), (sm, n) in level_sums.items() if l_ == lb and b_ == b and t_ == term])
                subj_means = subj_means[np.isfinite(subj_means).sum(1) >= 3]
                if len(subj_means) < 5:
                    continue
                subj_means -= np.nanmean(subj_means, 1, keepdims=True)
                n_subj = np.isfinite(subj_means).sum(0); shown = n_subj >= 5
                mu = np.nanmean(subj_means[:, shown], 0); ci = np.nanstd(subj_means[:, shown], 0, ddof=1) / np.sqrt(n_subj[shown]) * tdist.ppf(0.975, n_subj[shown] - 1)
                ax.plot(np.flatnonzero(shown), mu, "o-", ms=3, color=fc.LOBE_COLORS[lb], label=lb.capitalize())
                ax.fill_between(np.flatnonzero(shown), mu - ci, mu + ci, color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
                slopes = [np.polyfit(np.flatnonzero(np.isfinite(curve)), curve[np.isfinite(curve)], 1)[0] for curve in subj_means]
                trend.append({"measure": b, "term": term, "lobe": lb, "n": len(slopes), "slope": np.mean(slopes), "p": ttest_1samp(slopes, 0).pvalue})
            ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
            axes[-1, k].set_xlabel(xlabel)
        axes[i, 0].set_ylabel(lab.split("\n")[0] + ("\n(deviation, ms)" if b == "latency" else "\n(deviation, Cohen's d)"))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "task_effects.png"), dpi=200); plt.close(fig)
    trend = pd.DataFrame(trend); trend["q"] = multipletests(trend["p"], method="fdr_bh")[1]
    trend.to_csv(join(tdir, "task_trends.csv"), index=False)

    # 3. alpha vs high gamma
    r_events = el_sess.groupby(["subject", "electrode", "lobe"], as_index=False)["r_trial"].mean().rename(columns={"r_trial": "r"})
    el_d = el_sess.groupby(["subject", "lobe", "electrode"], as_index=False)[["d_alpha", "d_hg"]].mean()
    r_els = pd.DataFrame([{"subject": s_, "lobe": lb, "r": np.corrcoef(sub_df["d_alpha"], sub_df["d_hg"])[0, 1],
                           "r_abs": np.corrcoef(sub_df["d_alpha"].abs(), sub_df["d_hg"].abs())[0, 1]}
                          for (s_, lb), sub_df in el_d.groupby(["subject", "lobe"]) if len(sub_df) >= 5])
    mag = r_els.assign(r=r_els["r_abs"])
    measures = {"across events": r_events, "across electrodes": r_els, "across electrodes, magnitude": mag}
    coupling = pd.DataFrame([{"measure": k, "lobe": lb, "n": len(z), "r": np.tanh(z.mean()), "p": ttest_1samp(z, 0).pvalue}
                             for k, r_table in measures.items() for lb in fc.LOBES
                             if len(z := np.arctanh(r_table[r_table["lobe"] == lb].groupby("subject")["r"].mean().clip(-0.999, 0.999))) >= 5])
    coupling["q"] = multipletests(coupling["p"], method="fdr_bh")[1]
    coupling.to_csv(join(tdir, "alpha_hg_coupling_stats.csv"), index=False)
    # by timescale: levels with >= 5 events, >= 5 levels
    def level_r(term: str) -> pd.DataFrame:
        rows = []
        for (s_, lb, b_, t_), (sum_alpha, n_alpha) in level_sums.items():
            if b_ == "alpha" and t_ == term and (s_, lb, "high_gamma", term) in level_sums:
                sum_hg, n_hg = level_sums[(s_, lb, "high_gamma", term)]
                valid = (n_alpha >= 5) & (n_hg >= 5)
                if valid.sum() >= 5:
                    rows.append({"sub": s_, "roi": f"B-{lb}", "r": np.corrcoef(sum_alpha[valid] / n_alpha[valid], sum_hg[valid] / n_hg[valid])[0, 1]})
        return pd.DataFrame(rows)
    fast = (el_sess.groupby(["subject", "lobe", "electrode"])["r_fast"].mean().groupby(["subject", "lobe"]).mean().reset_index()
            .rename(columns={"subject": "sub", "r_fast": "r"}).assign(roi=lambda d: "B-" + d["lobe"]))
    timescales = {"Word event": fast, "Serial position": level_r("sp"), "List": level_r("list")}
    timescales = {k: fc.roi_stats(d, "r").assign(timescale=k) for k, d in timescales.items()}
    pd.concat(timescales.values()).to_csv(join(tdir, "alpha_hg_timescales_stats.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    fc.roi_dot_panel(ax, timescales, "r (alpha vs high gamma)")
    ax.set_ylim(-0.3, 0.3); fig.savefig(join(args.out_dir, "alpha_hg_timescales.png"), dpi=200); plt.close(fig)
    x0 = np.arange(len(fc.LOBES))
    for name, panels, size in (("alpha_hg_across_electrodes", (("across electrodes", "Signed Cohen's d", "C1"),
                                                               ("across electrodes, magnitude", "|Cohen's d|", "C4")), (8, 3.8)),):
        fig, ax = plt.subplots(figsize=size)
        shifts = (np.arange(len(panels)) - (len(panels) - 1) / 2) * 0.4
        for dx, (measure, lab, col) in zip(shifts, panels):
            r_table = measures[measure]
            fc.box_dots(ax, [r_table.loc[r_table["lobe"] == l, "r"].dropna() for l in fc.LOBES], x0 + dx, [col] * len(fc.LOBES), width=0.34)
            ax.plot([], [], "s", color=col, alpha=0.5, ms=9, label=lab)
            for row in coupling[coupling["measure"] == measure].itertuples():
                stars = "***" if row.q < .001 else "**" if row.q < .01 else "*" if row.q < .05 else "n.s."
                ax.text(fc.LOBES.index(row.lobe) + dx, 1.02, stars, ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x0, [l.capitalize() for l in fc.LOBES], rotation=30)
        ax.set_ylabel("r (alpha vs HG response)"); ax.set_ylim(-1, 1.12)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(join(args.out_dir, f"{name}.png"), dpi=200); plt.close(fig)
    paired_t = el_sess.groupby(["lobe", "electrode"], as_index=False)[["n", "pa_sum", "pa_sq", "pg_sum", "pg_sq"]].sum()
    for k, sm, sq in (("alpha", "pa_sum", "pa_sq"), ("high_gamma", "pg_sum", "pg_sq")):
        tval = (paired_t[sm] / paired_t["n"]) / np.sqrt((paired_t[sq] - paired_t[sm] ** 2 / paired_t["n"]) / (paired_t["n"] - 1) / paired_t["n"])
        q = multipletests(2 * tdist.sf(np.abs(tval), paired_t["n"] - 1), method="fdr_bh")[1]
        paired_t[k] = np.where(q >= 0.05, "none", np.where(tval > 0, "up", "down"))
    prevalence = paired_t.groupby("lobe").apply(lambda d: pd.Series({
        "n": len(d), "neither": ((d["alpha"] == "none") & (d["high_gamma"] == "none")).mean(),
        "alpha_only": ((d["alpha"] != "none") & (d["high_gamma"] == "none")).mean(),
        "hg_only": ((d["alpha"] == "none") & (d["high_gamma"] != "none")).mean(),
        "both": ((d["alpha"] != "none") & (d["high_gamma"] != "none")).mean(),
        "alpha_down_hg_up": ((d["alpha"] == "down") & (d["high_gamma"] == "up")).mean(),
        "alpha_up_hg_up": ((d["alpha"] == "up") & (d["high_gamma"] == "up")).mean()}), include_groups=False)
    prevalence.to_csv(join(tdir, "prevalence.csv"))

    # 4. recovery from the previous word's offset
    offset_sums = pd.read_pickle(join(tdir, "offset.pkl"))
    bin_ms = (np.arange(OFFSET_MAX_MS // OFFSET_BIN_MS) + 0.5) * OFFSET_BIN_MS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (k, lab) in zip(axes, (("high_gamma", "High-gamma envelope\nvs electrode's word-off mean"),
                                   ("alpha", "Alpha log power\nvs session mean (window centre)"))):
        for lb in lobes:
            subj_means = np.array([sm / np.where(n >= 5, n, np.nan) for (s_, l_, k_), (sm, n) in offset_sums.items() if l_ == lb and k_ == k])
            if len(subj_means) < 5:
                continue
            n_subj = np.isfinite(subj_means).sum(0)
            mu = np.nanmean(subj_means, 0); ci = np.nanstd(subj_means, 0, ddof=1) / np.sqrt(n_subj) * tdist.ppf(0.975, np.maximum(n_subj - 1, 1))
            shown = n_subj >= 5
            ax.plot(bin_ms[shown], mu[shown], color=fc.LOBE_COLORS[lb], lw=1.8, label=f"{lb.capitalize()} (n={len(subj_means)})")
            ax.fill_between(bin_ms[shown], (mu - ci)[shown], (mu + ci)[shown], color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
        ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time since the previous word disappeared (ms)"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Word-off window re-aligned to the previous word's offset (responsive electrodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "offset_recovery.png"), dpi=200); plt.close(fig)

    # 5. blank interval and the neighbouring words
    blank_fixed = pd.read_csv(join(tdir, "blank_fixed.csv"))
    terms = {"prev_resp": "Previous word's response", "prev2_resp": "Response two words back",
             "prev_resp:blank_len": "Previous word x blank length"}
    coefs = blank_fixed[blank_fixed["term"].isin(terms)].copy()
    coefs["p"] = 2 * norm.sf(np.abs(coefs["est"] / coefs["se"]))
    coefs["q"] = multipletests(coefs["p"], method="fdr_bh")[1]
    coefs.to_csv(join(tdir, "blank_coefficients.csv"), index=False)   # mixed-model check of the correlations below
    # sessions -> electrode -> subject x lobe
    by_lobe = el_sess.groupby(["subject", "lobe", "electrode"]).mean(numeric_only=True).groupby(["subject", "lobe"]).mean().reset_index()
    by_lobe = by_lobe.rename(columns={"subject": "sub"}).assign(roi=lambda d: "B-" + d["lobe"])
    rows = []
    for b in ("alpha", "high_gamma"):
        by_lobe[f"dlong_{b}"] = by_lobe[f"r1long_{b}"] - by_lobe[f"r1short_{b}"]
        by_lobe[f"d12_{b}"] = by_lobe[f"r1_{b}"] - by_lobe[f"r2_{b}"]
        rows += [fc.roi_stats(by_lobe, f"{m}_{b}").assign(measure=f"{m}_{b}") for m in ("r1", "r2", "dlong", "d12")]
    carry_stats = pd.concat(rows)
    carry_stats.to_csv(join(tdir, "carryover_stats.csv"), index=False)
    panel_labels = {"r1_high_gamma": "Previous word", "r2_high_gamma": "Two words back",
                    "dlong_high_gamma": "Previous word,\nlong − short ISI"}
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    fc.roi_dot_panel(ax, {label: carry_stats[carry_stats["measure"] == m] for m, label in panel_labels.items()}, "r with the current word's\nhigh-gamma response")
    fig.savefig(join(args.out_dir, "carryover.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n[variance components, % of single-event variance]\n",
          (comps.pivot_table(index=["band", "term"], columns="lobe", values="share") * 100)[lobes].round(2).to_string())
    print("\n[task trends]\n", trend.round(4).to_string(index=False))
    print("\n[prevalence, all electrodes]\n", prevalence.round(3).to_string())
    print("\n[alpha-hg r] across events, median by lobe:", r_events.groupby("lobe")["r"].median().round(3).to_dict())
    print("            across electrodes:", r_els.groupby("lobe")["r"].median().round(3).to_dict())
    print("            magnitude:", r_els.groupby("lobe")["r_abs"].median().round(3).to_dict())
    print("            tests (subjects' mean r vs 0):\n", coupling.round(4).to_string(index=False))
    print("\n[blank] coefficients:\n", coefs[["lobe", "band", "term", "est", "se", "q"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
