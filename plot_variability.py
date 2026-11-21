"""plot_variability.py -- preliminary results for the variability / event-modulator proposal.

Reads compute.py's per-event power (alpha: multitaper; high gamma: sub-band Hilbert, each
normalized by its word-off mean), word on = 0-600 ms. Per electrode and session, the event
response y = log word-on power, robust outliers dropped (|y - median| > OUTLIER_MAD x MAD:
e.g. R1323T session 3 event 90, a broadband transient ~5000x baseline on 82 of 148 electrodes),
then z-scored over the session's events. Each event's own word-off window is not used: after
a short blank it can contain the previous word's offset response.

  1. Electrode-pattern stability: per subject with >= 2 sessions, sessions alternately in two
     halves; per lobe (>= 5 electrodes) r across electrodes of Cohen's d between the halves.
     (Variance components, task effects, serial position: plot_variance_structure.py.)
  2. Memory (on hold): per subject, logistic recall ~ the same covariates; per subject and lobe, the
     subsequent memory effect (Cohen's d of y, recalled vs not) before and after regressing y on
     the covariates (serial position also as dummies for positions 2-3: primacy).
-> figures/variability/*.png, results tables alongside

    python plot_variability.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir

BEH = "word_on"
BANDS = {"alpha": "Alpha", "high_gamma": "High gamma"}
COVS = ["serialpos", "list", "blank_ms"]
OUTLIER_MAD = 10


def covariates(g: pd.DataFrame) -> pd.DataFrame:
    """COVS z-scored within session. The events start at serial position 2 (position 1 follows the
    countdown: no blank baseline), whose blank is undefined: the session median, so it stays in."""
    X = g[COVS].copy()
    X["blank_ms"] = X["blank_ms"].fillna(g.groupby("session")["blank_ms"].transform("median"))
    return X.groupby(g["session"]).transform(lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0)


def zs(v: np.ndarray) -> np.ndarray:
    return (v - np.nanmean(v, 0)) / np.nanstd(v, 0)


def session_tables(p: dict, lobe_of: dict) -> tuple[list, pd.DataFrame | None]:
    """(electrode rows, event x lobe table) of one session."""
    y_off, y_on = np.log(np.asarray(p["trial_pow_lo"], float)), np.log(np.asarray(p["trial_pow_hi"], float))
    for logpow in (y_off, y_on):   # robust outliers per electrode
        med = np.nanmedian(logpow, 0); mad = np.nanmedian(np.abs(logpow - med), 0) * 1.4826
        logpow[np.abs(logpow - med) > OUTLIER_MAD * mad] = np.nan
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
    sub = str(p["sid"][0])
    el = [{"sub": sub, "session": str(p["sid"]), "label": label, "lobe": lb, "d": d}
          for label, lb, d in zip(p["labels"], lobe, p["cohens_d"]) if lb]
    events = fc.load_events(fc.dfrow_from_sid(p["sid"]), BEH)
    events = events[np.asarray(events.attrs["mask"], bool)]
    if len(events) != len(y_on):
        return el, None
    y = zs(y_on)
    trials = pd.DataFrame({"sub": sub, "session": str(p["sid"]), "serialpos": events["serialpos"].to_numpy(float),
                           "list": events["trial"].to_numpy(float), "blank_ms": np.asarray(p["blank_ms_hi"], float),
                           "recalled": np.asarray(p["recalled_hi"], float)})
    for lb in fc.LOBES:
        if (lobe == lb).any():
            trials[lb] = np.nanmean(y[:, lobe == lb], 1)
    return el, trials


def sme(tr: pd.DataFrame) -> pd.DataFrame:
    """Per subject x lobe: Cohen's d recalled vs not, raw and after removing the covariates."""
    def cohens_d(resp, recalled):
        rec, nonrec = resp[recalled == 1], resp[recalled == 0]
        return (rec.mean() - nonrec.mean()) / np.sqrt((rec.var(ddof=1) + nonrec.var(ddof=1)) / 2)
    out = []
    for sub, sub_tr in tr.groupby("sub"):
        X = covariates(sub_tr)
        X["sp2"], X["sp3"] = (sub_tr["serialpos"] == 2).astype(float), (sub_tr["serialpos"] == 3).astype(float)   # primacy: the first two positions in the events
        recalled = sub_tr["recalled"].to_numpy()
        for lb in fc.LOBES:
            if lb not in sub_tr:
                continue
            valid = (sub_tr[lb].notna() & X.notna().all(1) & np.isfinite(recalled)).to_numpy()
            if valid.sum() < 50 or min((recalled[valid] == 1).sum(), (recalled[valid] == 0).sum()) < 10:
                continue
            resp = sub_tr[lb].to_numpy()[valid]
            design = np.c_[X.to_numpy()[valid], np.ones(valid.sum())]
            resid = resp - design @ np.linalg.lstsq(design, resp, rcond=None)[0]
            out.append({"sub": sub, "lobe": lb, "d_raw": cohens_d(resp, recalled[valid]),
                        "d_controlled": cohens_d(resid, recalled[valid])})
    return pd.DataFrame(out)


def recall_behaviour(tr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sub, sub_tr in tr.drop_duplicates(["session", "serialpos", "list"]).groupby("sub"):
        X = covariates(sub_tr)
        valid = X.notna().all(1) & sub_tr["recalled"].notna()
        if sub_tr.loc[valid, "recalled"].nunique() < 2:
            continue
        model = LogisticRegression(penalty=None).fit(X[valid], sub_tr.loc[valid, "recalled"].astype(int))
        out.append({"sub": sub, **dict(zip(COVS, model.coef_[0]))})
    return pd.DataFrame(out)


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variability"), **{
        "--from-csv": dict(action="store_true", help="reuse the saved electrode / trial tables")})
    os.makedirs(args.out_dir, exist_ok=True)
    table = lambda n: fc.table_path(join(args.out_dir, n))
    lobe_of = fc.load_burke_maps()
    electrodes, trials = {}, {}
    for band in BANDS:
        if args.from_csv:
            electrodes[band] = pd.read_csv(table(f"electrodes_{band}.csv"))
            trials[band] = pd.read_csv(table(f"trials_{band}.csv"))
            continue
        el_rows, tr_tables = [], []
        for f in tqdm(fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", args.n_sessions), desc=band):
            p = fc.load_pickle(str(f))
            el, tr = session_tables(p, lobe_of)
            el_rows += el
            if tr is not None:
                tr_tables.append(tr)
        electrodes[band], trials[band] = pd.DataFrame(el_rows), pd.concat(tr_tables, ignore_index=True)
        os.makedirs(os.path.dirname(table("x")), exist_ok=True)
        electrodes[band].to_csv(table(f"electrodes_{band}.csv"), index=False)
        trials[band].to_csv(table(f"trials_{band}.csv"), index=False)

    # event analyses: scored lists only (catFR1 list -1 is an unscored practice list); serial position <= 12
    trials = {b: t[(t["list"] >= 1) & (t["serialpos"] <= 12)] for b, t in trials.items()}

    # 1. do electrode differences replicate across sessions?
    reliability = []
    for band, band_els in electrodes.items():
        for sub, sub_els in band_els.groupby("sub"):
            sessions = sorted(sub_els["session"].unique())
            if len(sessions) < 2:
                continue
            sub_els = sub_els.assign(half=sub_els["session"].map({s: i % 2 for i, s in enumerate(sessions)}))
            d_halves = sub_els.groupby(["lobe", "label", "half"])["d"].mean().unstack("half").dropna()
            for lb, lobe_d in d_halves.groupby(level="lobe"):
                if len(lobe_d) >= 5:
                    reliability.append({"band": band, "sub": sub, "lobe": lb,
                                        "r": np.corrcoef(lobe_d[0], lobe_d[1])[0, 1], "n": len(lobe_d)})
    reliability = pd.DataFrame(reliability); reliability.to_csv(table("electrode_reliability_per_subject.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(fc.LOBES))
    for k, (band, color) in enumerate(zip(BANDS, ("C0", "C3"))):
        band_rel = reliability[reliability["band"] == band]
        fc.box_dots(ax, [band_rel.loc[band_rel["lobe"] == lb, "r"].dropna() for lb in fc.LOBES], x + (k - 0.5) * 0.38,
                    [color] * len(fc.LOBES), width=0.32)
        ax.plot([], [], "s", color=color, alpha=0.5, ms=9, label=BANDS[band])
    ax.set_ylabel("Electrode test-retest r")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x, [lb.capitalize() for lb in fc.LOBES], rotation=30)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "electrode_reliability.png"), dpi=200); plt.close(fig)
    rel_stats = reliability.groupby(["band", "lobe"])["r"].agg(["median", "size"]).reset_index()
    rel_stats.to_csv(table("electrode_reliability_stats.csv"), index=False)

    # 2. memory
    recall_cov = recall_behaviour(trials["high_gamma"])
    recall_cov.to_csv(table("recall_covariates_per_subject.csv"), index=False)
    recall_stats = pd.DataFrame([{"covariate": c, "n": len(recall_cov), "beta": recall_cov[c].mean(),
                                  "sem": recall_cov[c].sem(), "p": ttest_1samp(recall_cov[c], 0).pvalue} for c in COVS])
    recall_stats.to_csv(table("recall_covariates_stats.csv"), index=False)
    sme_stats = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (band, tr) in zip(axes, trials.items()):
        sub_sme = sme(tr); sub_sme.to_csv(table(f"sme_controlled_{band}_per_subject.csv"), index=False)
        x = np.arange(len(fc.LOBES))
        for k, measure in enumerate(["d_raw", "d_controlled"]):
            by_lobe = sub_sme.groupby("lobe")[measure]
            mu, se = by_lobe.mean().reindex(fc.LOBES), by_lobe.sem().reindex(fc.LOBES)
            ax.bar(x + (k - 0.5) * 0.38, mu, 0.36, yerr=se, color=[fc.LOBE_COLORS[lb] for lb in fc.LOBES],
                   alpha=1 if k == 0 else 0.45, capsize=2,
                   label="Raw" if k == 0 else "Covariates removed")
        for lb, lobe_sme in sub_sme.groupby("lobe"):
            sme_stats.append({"band": band, "lobe": lb, "n": len(lobe_sme), "d_raw": lobe_sme["d_raw"].mean(),
                              "d_controlled": lobe_sme["d_controlled"].mean(),
                              "p_raw": ttest_1samp(lobe_sme["d_raw"], 0).pvalue,
                              "p_change": ttest_rel(lobe_sme["d_raw"], lobe_sme["d_controlled"]).pvalue})
        ax.axhline(0, color="0.3", lw=0.8); ax.set_xticks(x, [lb.capitalize() for lb in fc.LOBES], rotation=30)
        ax.set_title(BANDS[band]); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Subsequent memory effect\n(Cohen's d, recalled vs not)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "sme_controlled.png"), dpi=200); plt.close(fig)
    sme_stats = pd.DataFrame(sme_stats); sme_stats.to_csv(table("sme_controlled_stats.csv"), index=False)

    pd.set_option("display.width", 200)
    for name, t in [("electrode reliability", rel_stats), ("recall ~ covariates", recall_stats),
                    ("SME raw vs controlled", sme_stats)]:
        print(f"\n[{name}]\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
