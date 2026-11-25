"""plot_variance_structure.py -- where the event response varies.

Reads compute.py's power pickles (alpha, high gamma). Every figure's rules (event_mask, imported by the other plot
scripts): scored lists, serial positions 2-12 (position 1 follows the countdown and is not in the events), outlier
events dropped per electrode (whole-clip high-gamma peak, log event high-gamma or alpha power > OUTLIER_MAD x MAD).
Power: the event's power minus the session's mean ISI power, over the pooled SD of Cohen's d (its mean over events
is Cohen's d; both constants from all the session's events). Latency: cross-correlation (xcorr_latency).

  1. Variance components (mixed_models.R variance): DeHaan 2026 ch. 3's random intercepts (word, serial
     position, list, subject, subject:word, session, subject:serial position, list:serial position, list in
     session, session:serial position), plus region, electrode, electrode in session and the event shared by a
     lobe's electrodes; residual: electrode and event. At most --max-electrodes random electrodes per subject
     and lobe (seed 0; refit: --refit). Stacked shares per lobe, power and latency apart.
  2. Alpha vs high gamma: per subject r across electrodes of Cohen's d and of |Cohen's d|; per lobe, subjects' mean
     Fisher z vs 0, BH-FDR over lobes. Prevalence: paired t (event vs ISI) over events per electrode, BH-FDR 0.05.
     Noise ceilings (Aims 1-2): per subject x lobe (>= 2 electrodes), serial-position and list means out within
     electrode-session, electrodes split into random halves, event means per half; reliability = r between halves,
     cross r = mean r of one measure in a half with the other in the other half, corrected r = mean cross r /
     sqrt(mean reliabilities); alpha vs high-gamma power, high-gamma power vs latency.
  3. Carry-over: per electrode-session, serial-position means out, r between each event's response and the one
     1 / 2 events back in the same list; r with the previous event after long - short ISIs (session median).
     Sessions -> electrode -> subject x lobe; t vs 0, BH-FDR over lobes (fc.roi_stats).
  4. --single (proposal Fig. 1C): the section 1 model within single subjects (mixed_models.R single), occipital,
     subjects with >= 5 electrodes and >= 2 sessions, all their electrodes; shares of the group model's occipital
     row, the mean single subject and the subject with the most electrodes x sessions. Sections 1-3 skipped.
  5. --recall (proposal Fig. 4): DeHaan 2026 ch. 3.4.9's subsequent memory effect on high-gamma power during the
     event and during the ISI before it (in the power units above) and latency (ms, responsive electrodes); per
     electrode-session, measure ~ recalled (standard) and ~ recalled + predicted recall (adjusted; mixed_models.R
     recall: his behavioural model). Coefficients averaged per subject and region; t vs 0 across subjects, BH-FDR
     over regions. Sections 1-3 skipped.
-> figures/variance_structure/*.png (tables and model fits: results/variance_structure/)

    python plot_variance_structure.py [--from-csv] [--refit] [--cores C] [--max-electrodes M]
    python plot_variance_structure.py --single --max-electrodes 100000 --out-dir figures/single_subject
    python plot_variance_structure.py --recall --out-dir figures/recall
"""
from __future__ import annotations

import glob
import os
import subprocess
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import _blank_before, power_dir
from plot_latency import half_max, response, responsive

BEH = "word_on"
OUTLIER_MAD = 10
LMM_MAX_ELECTRODES = 5
SP_RANGE = (2, 12)
BANDS = {"alpha": "Alpha power", "high_gamma": "High-gamma power", "latency": "High-gamma latency\n(responsive electrodes)"}
GROUPS = {   # stacked share segments: label -> (model terms, colour)
    "Subject": (["subject"], "#1f4fa3"), "Region": (["region"], "#17becf"), "Electrode": (["electrode"], "#d62728"),
    "Electrode:Session": (["elsess"], "#f2c12e"),
    "Session": (["session"], "#8c564b"), "Session:List": (["sess_list"], "#ff7f0e"),
    "Task (serial position, list, item)": (["sp", "list", "item", "subj_item", "subj_sp", "list_sp", "sess_sp"], "#2ca02c"),
    "Event": (["event"], "#7b4fa3"),
    "Residual (electrode, event)": (["Residual"], "#d9d9d9")}

def robust_ok(v: np.ndarray) -> np.ndarray:
    """(trials, electrodes) True where the event is not a robust outlier of its electrode."""
    med = np.nanmedian(v, 0); mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    return np.abs(v - med) <= OUTLIER_MAD * mad


def event_mask(events: pd.DataFrame, hg_pkl: dict, alpha_pkl: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every figure's events: (in_task (events,), kept (events, electrodes)) -- scored list and serial position in
    SP_RANGE; not an outlier of its electrode (robust_ok)."""
    in_task = (events["trial"].to_numpy(float) >= 1) & events["serialpos"].between(*SP_RANGE).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = (robust_ok(np.nanmax(np.abs(response(hg_pkl)[0]), -1)) & robust_ok(np.log(np.asarray(hg_pkl["trial_pow_hi"], float)))
                & robust_ok(np.log(np.asarray(alpha_pkl["trial_pow_hi"], float))))
    return in_task, kept


def session_events(hg_pkl: dict) -> pd.DataFrame | None:
    """The session's word events in the pickle's order; None if they do not match it."""
    events = fc.load_events(fc.dfrow_from_sid(hg_pkl["sid"]), BEH)
    events = events[np.asarray(events.attrs["mask"], bool)].reset_index(drop=True)
    return events if len(events) == len(np.asarray(hg_pkl["trial_pow_hi"])) else None


def usable_sessions(save_root: str):
    """Per session with both bands: ({band: power pickle}, usable (events, electrodes): in_task & kept)."""
    dirs = {band: power_dir(save_root, BEH, band) for band in ("alpha", "high_gamma")}
    for f in tqdm(fc.session_files(dirs["high_gamma"], "_power.pkl", None), desc="sessions"):
        if not (dirs["alpha"] / f.name).exists():
            continue
        pkls = {band: fc.load_pickle(str(d / f.name)) for band, d in dirs.items()}
        events = session_events(pkls["high_gamma"])
        if events is not None:
            in_task, kept = event_mask(events, pkls["high_gamma"], pkls["alpha"])
            yield pkls, in_task[:, None] & kept


def power_response(pkl: dict) -> np.ndarray:
    """(events, electrodes) event power minus the session's mean ISI power, over the pooled SD of Cohen's d."""
    on, off = np.asarray(pkl["trial_pow_hi"], float), np.asarray(pkl["trial_pow_lo"], float)
    return (on - off.mean(0)) / pooled_sd(on, off)


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


SME_MEASURES = {"event_power": "Event power (Cohen's d units)", "isi_power": "ISI power (Cohen's d units)", "latency": "Latency (ms)"}


def recall_sme_figure(tdir: str, out_dir: str) -> None:
    """5. Per electrode-session OLS of each measure on recalled (standard SME) and on recalled + predicted recall
    (adjusted SME); per subject and region mean over electrode-sessions; t vs 0, BH-FDR over regions."""
    words = pd.read_csv(join(tdir, "behaviour.csv")).merge(pd.read_csv(join(tdir, "recall_phat.csv")), on=["session", "event"])
    words_by_session = {sess: w.set_index("event") for sess, w in words.groupby("session")}
    rows = []
    for path in tqdm(sorted(glob.glob(join(tdir, "sme", "*.npz"))), desc="SME"):
        arrays = np.load(path)
        if str(arrays["session"]) not in words_by_session:   # its experiment's behavioural model did not fit
            continue
        session_words = words_by_session[str(arrays["session"])].reindex(arrays["event"])
        recalled, predicted = session_words["recalled"].to_numpy(), session_words["phat"].to_numpy()
        for measure in SME_MEASURES:
            for c, lobe in enumerate(arrays["lobe"]):
                values = arrays[measure][:, c]; finite = np.isfinite(values) & np.isfinite(recalled)
                if finite.sum() < 20 or recalled[finite].std() == 0:
                    continue
                ones = np.ones(finite.sum())
                standard = np.linalg.lstsq(np.c_[ones, recalled[finite]], values[finite], rcond=None)[0][1]
                adjusted = np.linalg.lstsq(np.c_[ones, recalled[finite], predicted[finite]], values[finite], rcond=None)[0][1]
                rows.append({"sub": str(arrays["subject"]), "roi": f"B-{lobe}", "measure": measure,
                             "standard": standard, "adjusted": adjusted})
    per_subject = pd.DataFrame(rows).groupby(["sub", "roi", "measure"], as_index=False)[["standard", "adjusted"]].mean()
    stats = pd.concat([fc.roi_stats(per_subject[per_subject["measure"] == m], sme).assign(measure=m, sme=sme)
                       for sme in ("standard", "adjusted") for m in SME_MEASURES], ignore_index=True)
    per_subject.to_csv(join(tdir, "sme_subjects.csv"), index=False); stats.to_csv(join(tdir, "sme_stats.csv"), index=False)
    print(stats[["sme", "measure", "roi", "n_subjects", "mean", "sem", "q"]].round(4).to_string(index=False))
    # mean +/- SEM dots as 3C: power (event, ISI) | latency
    adjusted = {m: stats[(stats["sme"] == "adjusted") & (stats["measure"] == m)] for m in SME_MEASURES}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), layout="constrained")
    fc.roi_dot_panel(axes[0], {"Event": adjusted["event_power"], "ISI": adjusted["isi_power"]},
                     "High-gamma power (Cohen's d units),\nrecalled − not recalled")
    fc.roi_dot_panel(axes[1], {"Latency": adjusted["latency"]}, "Latency, recalled − not recalled (ms)")
    axes[1].get_legend().remove()
    fig.savefig(join(out_dir, "recall_sme.png"), dpi=200); plt.close(fig)


def single_subject_figure(tdir: str, out_dir: str) -> None:
    """4. Occipital shares as stacked bars: group model (Fig. 1B's occipital row), mean single subject, example."""
    single = pd.read_csv(join(tdir, "components_single.csv"))
    single["share"] = single["var"] / single.groupby(["subject", "band"])["var"].transform("sum")
    info = single.groupby("subject")[["nel", "nses"]].first()
    example = (info["nel"] * info["nses"]).idxmax()
    group = pd.read_csv(join(os.path.dirname(fc.table_path(join(os.path.dirname(out_dir), "variance_structure", "x"))), "components.csv"))
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
        ax.set_title(f"{BANDS[band]}, occipital", fontsize=13); ax.set_xlabel("Share of variance across events", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(join(out_dir, "single_subject_shares.png"), dpi=200, bbox_inches="tight")
    print("[single] example subject", example, info.loc[example].to_dict())


def noise_ceilings(trials_csv: str) -> pd.DataFrame:
    """2. Split-half reliability, cross r and noise-corrected r of measure pairs, per lobe (cross r vs 0: t across
    subjects, BH-FDR over lobes)."""
    pairs = (("alpha", "high_gamma", "in_power"), ("high_gamma", "latency", "latency"))
    t = pd.read_csv(trials_csv, usecols=["subject", "electrode", "session", "event", "sp", "list", "lobe", "alpha",
                                         "high_gamma", "latency", "in_power"])
    for c in ("alpha", "high_gamma", "latency"):
        for k in ("sp", "list"):
            t[c] = t[c] - t.groupby(["electrode", "session", k])[c].transform("mean")
    rng, rows = np.random.default_rng(0), []
    for m1, m2, sel in pairs:
        pair_trials = t[(t[sel] if sel == "in_power" else t[sel].notna()) & t[m1].notna() & t[m2].notna()]
        for (sub, lb), d in pair_trials.groupby(["subject", "lobe"]):
            electrodes = rng.permutation(d["electrode"].unique())
            if len(electrodes) < 2:
                continue
            half = {e: i for i, chunk in enumerate(np.array_split(electrodes, 2)) for e in chunk}
            m = d.assign(h=d["electrode"].map(half)).groupby(["session", "event", "h"])[[m1, m2]].mean().unstack("h").dropna()
            if len(m) < 50:
                continue
            r = lambda a, b: np.corrcoef(m[a], m[b])[0, 1]
            rows.append({"pair": f"{m1} vs {m2}", "roi": f"B-{lb}", "sub": sub, "rel1": r((m1, 0), (m1, 1)),
                         "rel2": r((m2, 0), (m2, 1)), "cross": (r((m1, 0), (m2, 1)) + r((m2, 0), (m1, 1))) / 2})
    rows = pd.DataFrame(rows)
    out = []
    for pair, d in rows.groupby("pair"):
        stats = fc.roi_stats(d, "cross").set_index("roi")[["n_subjects", "mean", "q"]].rename(columns={"mean": "cross"})
        rel = d.groupby("roi")[["rel1", "rel2"]].mean()
        out.append(stats.join(rel).assign(pair=pair, corrected=lambda g: g["cross"] / np.sqrt(g["rel1"].clip(lower=1e-3) * g["rel2"].clip(lower=1e-3))))
    return pd.concat(out).reset_index().set_index(["pair", "roi"])


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variance_structure"), **{
        "--from-csv": dict(action="store_true", help="reuse the saved tables and model fits"),
        "--refit": dict(action="store_true", help="refit the single-event mixed models"),
        "--cores": dict(type=int, default=6, help="parallel model fits in R"),
        "--max-electrodes": dict(type=int, default=LMM_MAX_ELECTRODES, help="per subject and lobe, in the models"),
        "--single": dict(action="store_true", help="4. single-subject fits (Fig. 1C) instead of sections 1-3"),
        "--recall": dict(action="store_true", help="5. subsequent memory effect (Fig. 4) instead of sections 1-3")})
    os.makedirs(args.out_dir, exist_ok=True)
    tdir = os.path.dirname(fc.table_path(join(args.out_dir, "x")))
    lobe_of = fc.load_burke_maps()
    files = {band: {f.name: f for f in fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", None)}
             for band in ("alpha", "high_gamma")}
    names = sorted(fname for fname in files["high_gamma"] if fname in files["alpha"])

    if not args.from_csv:
        trial_rows, electrode_rows, recall_rows = [], [], []
        if args.recall:
            os.makedirs(join(tdir, "sme"), exist_ok=True)
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
            alpha_resp, hg_resp = power_response(alpha_pkl), power_response(hg_pkl)   # mean over events: Cohen's d
            pow_on, pow_off = {}, {}
            for k, pkl in (("a", alpha_pkl), ("g", hg_pkl)):
                pow_on[k], pow_off[k] = np.asarray(pkl["trial_pow_hi"], float), np.asarray(pkl["trial_pow_lo"], float)
            sd_hg = pooled_sd(pow_on["g"], pow_off["g"])
            alpha_diff, hg_diff = (pow_on["a"] - pow_off["a"]) / pooled_sd(pow_on["a"], pow_off["a"]), (pow_on["g"] - pow_off["g"]) / sd_hg   # event - ISI, same event
            use, kept = event_mask(events, hg_pkl, alpha_pkl)
            curves[~kept] = np.nan; alpha_resp[~kept] = np.nan; hg_resp[~kept] = np.nan
            task = task_columns(events, blank)
            is_resp = responsive(hg_pkl)
            if args.recall:
                usable = use[:, None] & kept
                isi_power = np.where(usable, (pow_off["g"] - pow_off["g"].mean(0)) / sd_hg, np.nan)
                latency = np.full(hg_resp.shape, np.nan)
                for c in np.flatnonzero(is_resp & (lobe != None)):  # noqa: E711
                    if usable[:, c].sum() >= 20:
                        latency[usable[:, c], c] = xcorr_latency(curves[usable[:, c], c], t)[0]
                recall_rows.append(task[use].assign(subject=sub, session=sess, experiment=str(hg_pkl["sid"][1]),
                                                    event=np.flatnonzero(use),
                                                    recalled=events["correct_recall"].to_numpy(float)[use]))
                in_lobe = lobe != None  # noqa: E711
                np.savez(join(tdir, "sme", fname.replace("_power.pkl", ".npz")), session=sess, subject=sub,
                         event=np.flatnonzero(use), lobe=lobe[in_lobe].astype(str),
                         **{k: v[use][:, in_lobe].astype(np.float32) for k, v in
                            (("event_power", hg_resp), ("isi_power", isi_power), ("latency", latency))})
                continue
            sp, list_num = task["sp"].to_numpy(), task["list"].to_numpy()
            long_blank = blank > np.nanmedian(blank); short_blank = blank < np.nanmedian(blank)   # split as plot_timing
            for c in np.flatnonzero(lobe != None):  # noqa: E711
                el = f"{sub}|{hg_pkl['labels'][c]}"
                ev_mask = use & np.isfinite(hg_resp[:, c]) & np.isfinite(alpha_resp[:, c])
                if ev_mask.sum() < 20:
                    continue
                lat, r_xc = xcorr_latency(curves[ev_mask, c], t) if is_resp[c] else (np.full(ev_mask.sum(), np.nan),) * 2
                subregion = hg_pkl["reg_full"][c]   # hemispheres pooled
                trial_rows.append(task[ev_mask].assign(subject=sub, electrode=el, session=sess, lobe=lobe[c],
                                                       region=subregion[2:] if subregion[:2] in ("L ", "R ") else subregion, event=np.flatnonzero(ev_mask), alpha=alpha_resp[ev_mask, c], high_gamma=hg_resp[ev_mask, c],
                                                       latency=lat, xc_r=r_xc))
                corrs = {}   # 3. serial-position means out
                for k, series in (("alpha", alpha_resp[:, c]), ("high_gamma", hg_resp[:, c])):
                    series = np.where(ev_mask, series, np.nan)
                    series = series - pd.Series(series).groupby(sp).transform("mean").to_numpy()
                    corrs.update({f"r1_{k}": lag_r(series, sp, list_num, 1), f"r2_{k}": lag_r(series, sp, list_num, 2),
                                  f"r1long_{k}": lag_r(series, sp, list_num, 1, long_blank), f"r1short_{k}": lag_r(series, sp, list_num, 1, short_blank)})
                if is_resp[c]:   # serial-position and list means out
                    lat_dev, power_dev = np.full(len(ev_mask), np.nan), np.where(ev_mask, hg_resp[:, c], np.nan); lat_dev[ev_mask] = lat
                    for grouping in (sp, list_num):
                        lat_dev = lat_dev - pd.Series(lat_dev).groupby(grouping).transform("mean").to_numpy()
                        power_dev = power_dev - pd.Series(power_dev).groupby(grouping).transform("mean").to_numpy()
                    both = np.isfinite(lat_dev) & np.isfinite(power_dev)
                    corrs["r_power_lat"] = np.corrcoef(power_dev[both], lat_dev[both])[0, 1] if both.sum() >= 20 else np.nan
                electrode_rows.append({**corrs, "subject": sub, "electrode": el, "session": sess, "lobe": lobe[c],
                                       "d_alpha": alpha_resp[ev_mask, c].mean(), "d_hg": hg_resp[ev_mask, c].mean(), "n": ev_mask.sum(),
                                       "pa_sum": alpha_diff[ev_mask, c].sum(), "pa_sq": (alpha_diff[ev_mask, c] ** 2).sum(),
                                       "pg_sum": hg_diff[ev_mask, c].sum(), "pg_sq": (hg_diff[ev_mask, c] ** 2).sum()})
        if args.recall:
            pd.concat(recall_rows, ignore_index=True).to_csv(join(tdir, "behaviour.csv"), index=False)
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
    for mode, out in ((("single", "components_single.csv"),) if args.single else
                      (("recall", "recall_phat.csv"),) if args.recall else
                      (("variance", "components.csv"),)):
        if args.refit or not os.path.exists(join(tdir, out)):
            subprocess.run(["Rscript", "mixed_models.R", mode, tdir, str(args.cores)], check=True)
    if args.single:
        return single_subject_figure(tdir, args.out_dir)
    if args.recall:
        return recall_sme_figure(tdir, args.out_dir)
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
            ax.set_xlim(0, 1); ax.set_title(lab, fontsize=14); ax.set_xlabel("Share of variance across events", fontsize=12)
            ax.tick_params(axis="x", labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()   # shared y: once for all panels
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=12, ncol=3 if len(bands) > 1 else 2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
        fig.text(0.995, 0.02, "n = subjects / electrodes", ha="right", fontsize=9, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(join(args.out_dir, f"{name}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # 2. alpha vs high gamma
    el_d = el_sess.groupby(["subject", "lobe", "electrode"], as_index=False)[["d_alpha", "d_hg"]].mean()
    r_els = pd.DataFrame([{"subject": s_, "lobe": lb, "r": np.corrcoef(sub_df["d_alpha"], sub_df["d_hg"])[0, 1],
                           "r_abs": np.corrcoef(sub_df["d_alpha"].abs(), sub_df["d_hg"].abs())[0, 1]}
                          for (s_, lb), sub_df in el_d.groupby(["subject", "lobe"]) if len(sub_df) >= 5])
    measures = {"across electrodes": r_els, "across electrodes, magnitude": r_els.assign(r=r_els["r_abs"])}
    coupling = pd.concat([fc.roi_stats(r_table.assign(sub=r_table["subject"], roi="B-" + r_table["lobe"],
                                                      z=np.arctanh(r_table["r"].clip(-0.999, 0.999))), "z").assign(measure=k)
                          for k, r_table in measures.items()], ignore_index=True)
    coupling["lobe"] = coupling["roi"].str[2:]; coupling["r"] = np.tanh(coupling["mean"])
    coupling.to_csv(join(tdir, "alpha_hg_coupling_stats.csv"), index=False)
    x0 = np.arange(len(fc.LOBES))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    panels = (("across electrodes", "Signed Cohen's d", "C1"), ("across electrodes, magnitude", "|Cohen's d|", "C4"))
    for dx, (measure, lab, col) in zip((-0.2, 0.2), panels):
        r_table = measures[measure]
        fc.box_dots(ax, [r_table.loc[r_table["lobe"] == l, "r"].dropna() for l in fc.LOBES], x0 + dx, [col] * len(fc.LOBES), width=0.34)
        ax.plot([], [], "s", color=col, alpha=0.5, ms=9, label=lab)
        for row in coupling[coupling["measure"] == measure].itertuples():
            ax.text(fc.LOBES.index(row.lobe) + dx, 1.02, fc.stars(row.q) or ("n.s." if np.isfinite(row.q) else ""), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x0, [l.capitalize() for l in fc.LOBES], rotation=30)
    ax.set_ylabel("r (alpha vs high-gamma response)"); ax.set_ylim(-1, 1.12)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "alpha_hg_across_electrodes.png"), dpi=200); plt.close(fig)
    paired_t = el_sess.groupby(["lobe", "electrode"], as_index=False)[["n", "pa_sum", "pa_sq", "pg_sum", "pg_sq"]].sum()
    for k, sm, sq in (("alpha", "pa_sum", "pa_sq"), ("high_gamma", "pg_sum", "pg_sq")):
        tval = (paired_t[sm] / paired_t["n"]) / np.sqrt((paired_t[sq] - paired_t[sm] ** 2 / paired_t["n"]) / (paired_t["n"] - 1) / paired_t["n"])
        q = multipletests(2 * tdist.sf(np.abs(tval), paired_t["n"] - 1), method="fdr_bh")[1]
        paired_t[k] = np.where(q >= 0.05, "none", np.where(tval > 0, "up", "down"))
    prevalence = paired_t.groupby("lobe").apply(lambda d: pd.Series({
        "n": len(d), "neither": ((d["alpha"] == "none") & (d["high_gamma"] == "none")).mean(),
        "alpha_only": ((d["alpha"] != "none") & (d["high_gamma"] == "none")).mean(),
        "hg_only": ((d["alpha"] == "none") & (d["high_gamma"] != "none")).mean(),
        "both": ((d["alpha"] != "none") & (d["high_gamma"] != "none")).mean()}), include_groups=False)
    prevalence.to_csv(join(tdir, "prevalence.csv"))
    ceilings = noise_ceilings(join(tdir, "trials.csv"))
    ceilings.to_csv(join(tdir, "noise_ceilings.csv"))

    # 3. carry-over: sessions -> electrode -> subject x lobe
    by_lobe = el_sess.groupby(["subject", "lobe", "electrode"]).mean(numeric_only=True).groupby(["subject", "lobe"]).mean().reset_index()
    by_lobe = by_lobe.rename(columns={"subject": "sub"}).assign(roi=lambda d: "B-" + d["lobe"])
    rows = []
    for b in ("alpha", "high_gamma"):
        by_lobe[f"dlong_{b}"] = by_lobe[f"r1long_{b}"] - by_lobe[f"r1short_{b}"]
        by_lobe[f"d12_{b}"] = by_lobe[f"r1_{b}"] - by_lobe[f"r2_{b}"]
        rows += [fc.roi_stats(by_lobe, f"{m}_{b}").assign(measure=f"{m}_{b}") for m in ("r1", "r2", "dlong", "d12")]
    carry_stats = pd.concat(rows)
    carry_stats.to_csv(join(tdir, "carryover_stats.csv"), index=False)
    panel_labels = {"r1_high_gamma": "Previous event", "r2_high_gamma": "Two events back",
                    "dlong_high_gamma": "Previous event,\nlong − short ISI"}
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    fc.roi_dot_panel(ax, {label: carry_stats[carry_stats["measure"] == m] for m, label in panel_labels.items()}, "r with the current event's\nhigh-gamma response")
    fig.savefig(join(args.out_dir, "carryover.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n[variance components, % of single-event variance]\n",
          (comps.pivot_table(index=["band", "term"], columns="lobe", values="share") * 100)[lobes].round(2).to_string())
    print("\n[prevalence, all electrodes]\n", prevalence.round(3).to_string())
    print("\n[alpha-hg r across electrodes, subjects' mean]\n", coupling[["measure", "lobe", "n_subjects", "r", "q"]].round(3).to_string(index=False))
    print("\n[noise ceilings]\n", ceilings.round(3).to_string())
    print("\n[carry-over]\n", carry_stats[["measure", "roi", "n_subjects", "mean", "sem", "q"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
