"""plot_variability.py -- preliminary results for the variability / trial-modulator proposal.

Reads compute.py's per-trial power (alpha: multitaper; high gamma: sub-band Hilbert, each
normalized by its word-off mean), word on = 0-600 ms. Per electrode and session, the trial
response y = log word-on power, robust outliers dropped (|y - median| > OUTLIER_MAD x MAD:
e.g. R1323T session 3 trial 90, a broadband transient ~5000x baseline on 82 of 148 electrodes),
then z-scored over the session's trials. Each trial's own word-off window is not used: after
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
    """(electrode rows, trial x lobe table) of one session."""
    lo, hi = np.log(np.asarray(p["trial_pow_lo"], float)), np.log(np.asarray(p["trial_pow_hi"], float))
    for x in (lo, hi):   # robust outliers per electrode
        med = np.nanmedian(x, 0); mad = np.nanmedian(np.abs(x - med), 0) * 1.4826
        x[np.abs(x - med) > OUTLIER_MAD * mad] = np.nan
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
    sub = str(p["sid"][0])
    el = [{"sub": sub, "session": str(p["sid"]), "label": l, "lobe": lb, "d": d}
          for l, lb, d in zip(p["labels"], lobe, p["cohens_d"]) if lb]
    w = fc.load_events(fc.dfrow_from_sid(p["sid"]), BEH)
    w = w[np.asarray(w.attrs["mask"], bool)]
    if len(w) != len(hi):
        return el, None
    y = zs(hi)
    tr = pd.DataFrame({"sub": sub, "session": str(p["sid"]), "serialpos": w["serialpos"].to_numpy(float),
                       "list": w["trial"].to_numpy(float), "blank_ms": np.asarray(p["blank_ms_hi"], float),
                       "recalled": np.asarray(p["recalled_hi"], float)})
    for lb in fc.LOBES:
        if (lobe == lb).any():
            tr[lb] = np.nanmean(y[:, lobe == lb], 1)
    return el, tr


def sme(tr: pd.DataFrame) -> pd.DataFrame:
    """Per subject x lobe: Cohen's d recalled vs not, raw and after removing the covariates."""
    def d(v, r):
        a, b = v[r == 1], v[r == 0]
        return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    out = []
    for sub, g in tr.groupby("sub"):
        X = covariates(g)
        X["sp2"], X["sp3"] = (g["serialpos"] == 2).astype(float), (g["serialpos"] == 3).astype(float)   # primacy: the first two positions in the events
        r = g["recalled"].to_numpy()
        for lb in fc.LOBES:
            if lb not in g:
                continue
            ok = (g[lb].notna() & X.notna().all(1) & np.isfinite(r)).to_numpy()
            if ok.sum() < 50 or min((r[ok] == 1).sum(), (r[ok] == 0).sum()) < 10:
                continue
            v = g[lb].to_numpy()[ok]
            A = np.c_[X.to_numpy()[ok], np.ones(ok.sum())]
            res = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
            out.append({"sub": sub, "lobe": lb, "d_raw": d(v, r[ok]), "d_controlled": d(res, r[ok])})
    return pd.DataFrame(out)


def recall_behaviour(tr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sub, g in tr.drop_duplicates(["session", "serialpos", "list"]).groupby("sub"):
        X = covariates(g)
        ok = X.notna().all(1) & g["recalled"].notna()
        if g.loc[ok, "recalled"].nunique() < 2:
            continue
        m = LogisticRegression(penalty=None).fit(X[ok], g.loc[ok, "recalled"].astype(int))
        out.append({"sub": sub, **dict(zip(COVS, m.coef_[0]))})
    return pd.DataFrame(out)


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variability"), **{
        "--from-csv": dict(action="store_true", help="reuse the saved electrode / trial tables")})
    os.makedirs(args.out_dir, exist_ok=True)
    stem = lambda n: fc.table_path(join(args.out_dir, n))
    lobe_of = fc.load_burke_maps()
    E, T = {}, {}
    for band in BANDS:
        if args.from_csv:
            E[band], T[band] = pd.read_csv(stem(f"electrodes_{band}.csv")), pd.read_csv(stem(f"trials_{band}.csv"))
            continue
        els, trs = [], []
        for f in tqdm(fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", args.n_sessions), desc=band):
            p = fc.load_pickle(str(f))
            el, tr = session_tables(p, lobe_of)
            els += el
            if tr is not None:
                trs.append(tr)
        E[band], T[band] = pd.DataFrame(els), pd.concat(trs, ignore_index=True)
        os.makedirs(os.path.dirname(stem("x")), exist_ok=True)
        E[band].to_csv(stem(f"electrodes_{band}.csv"), index=False); T[band].to_csv(stem(f"trials_{band}.csv"), index=False)

    # trial analyses: scored lists only (catFR1 list -1 is an unscored practice list); serial position <= 12
    T = {b: t[(t["list"] >= 1) & (t["serialpos"] <= 12)] for b, t in T.items()}

    # 1. do electrode differences replicate across sessions?
    rel = []
    for band, e in E.items():
        for sub, g in e.groupby("sub"):
            ss = sorted(g["session"].unique())
            if len(ss) < 2:
                continue
            g = g.assign(half=g["session"].map({s: i % 2 for i, s in enumerate(ss)}))
            w = g.groupby(["lobe", "label", "half"])["d"].mean().unstack("half").dropna()
            for lb, h in w.groupby(level="lobe"):
                if len(h) >= 5:
                    rel.append({"band": band, "sub": sub, "lobe": lb, "r": np.corrcoef(h[0], h[1])[0, 1], "n": len(h)})
    rel = pd.DataFrame(rel); rel.to_csv(stem("electrode_reliability_per_subject.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(fc.LOBES))
    for k, (band, col) in enumerate(zip(BANDS, ("C0", "C3"))):
        r = rel[rel["band"] == band]
        fc.box_dots(ax, [r.loc[r["lobe"] == l, "r"].dropna() for l in fc.LOBES], x + (k - 0.5) * 0.38,
                    [col] * len(fc.LOBES), width=0.32)
        ax.plot([], [], "s", color=col, alpha=0.5, ms=9, label=BANDS[band])
    ax.set_ylabel("Electrode test-retest r")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x, [l.capitalize() for l in fc.LOBES], rotation=30)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "electrode_reliability.png"), dpi=200); plt.close(fig)
    rst = rel.groupby(["band", "lobe"])["r"].agg(["median", "size"]).reset_index()
    rst.to_csv(stem("electrode_reliability_stats.csv"), index=False)

    # 2. memory
    beh = recall_behaviour(T["high_gamma"])
    beh.to_csv(stem("recall_covariates_per_subject.csv"), index=False)
    bst = pd.DataFrame([{"covariate": c, "n": len(beh), "beta": beh[c].mean(), "sem": beh[c].sem(),
                         "p": ttest_1samp(beh[c], 0).pvalue} for c in COVS])
    bst.to_csv(stem("recall_covariates_stats.csv"), index=False)
    sst = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (band, tr) in zip(axes, T.items()):
        s = sme(tr); s.to_csv(stem(f"sme_controlled_{band}_per_subject.csv"), index=False)
        x = np.arange(len(fc.LOBES))
        for k, col in enumerate(["d_raw", "d_controlled"]):
            g = s.groupby("lobe")[col]; mu, se = g.mean().reindex(fc.LOBES), g.sem().reindex(fc.LOBES)
            ax.bar(x + (k - 0.5) * 0.38, mu, 0.36, yerr=se, color=[fc.LOBE_COLORS[l] for l in fc.LOBES],
                   alpha=1 if k == 0 else 0.45, capsize=2,
                   label="Raw" if k == 0 else "Covariates removed")
        for lb, g in s.groupby("lobe"):
            sst.append({"band": band, "lobe": lb, "n": len(g), "d_raw": g["d_raw"].mean(),
                        "d_controlled": g["d_controlled"].mean(), "p_raw": ttest_1samp(g["d_raw"], 0).pvalue,
                        "p_change": ttest_rel(g["d_raw"], g["d_controlled"]).pvalue})
        ax.axhline(0, color="0.3", lw=0.8); ax.set_xticks(x, [l.capitalize() for l in fc.LOBES], rotation=30)
        ax.set_title(BANDS[band]); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Subsequent memory effect\n(Cohen's d, recalled vs not)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "sme_controlled.png"), dpi=200); plt.close(fig)
    sst = pd.DataFrame(sst); sst.to_csv(stem("sme_controlled_stats.csv"), index=False)

    pd.set_option("display.width", 200)
    for name, t in [("electrode reliability", rst), ("recall ~ covariates", bst), ("SME raw vs controlled", sst)]:
        print(f"\n[{name}]\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
