"""plot_variability.py -- preliminary results for the variability / trial-modulator proposal.

Reads compute.py's per-trial power (alpha: multitaper; high gamma: sub-band Hilbert, each
normalized by its word-off mean), word on = 0-600 ms. Per electrode and session, the trial
response y = log word-on power, robust outliers dropped (|y - median| > OUTLIER_MAD x MAD:
e.g. R1323T session 3 trial 90, a broadband transient ~5000x baseline on 82 of 148 electrodes),
then z-scored over the session's trials. Each trial's own word-off window is not used: after
a short blank it can contain the previous word's offset response.

  1. Electrode heterogeneity: per electrode-session, word on vs word off (Welch t on log power,
     both directions, BH-FDR within session) -> fraction of electrodes up / down per lobe.
  2. Where the variability sits: electrode Cohen's d (mean over sessions); per lobe, the share
     of variance between subjects (subject means) vs between electrodes within subjects.
  3. Trial modulators (scored lists, serial positions 2-12; position 1 has no blank baseline
     and is not in the events): per subject and lobe (electrodes averaged), OLS of y on serial position,
     list number and blank interval (each z-scored within session); betas across subjects,
     one-sample t, BH-FDR over band x lobe x covariate. Also y by serial position / list / blank bin.
  4. Memory: per subject, logistic recall ~ the same covariates; per subject and lobe, the
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
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir

BEH = "word_on"
BANDS = {"alpha": "Alpha", "high_gamma": "High gamma"}
COVS = ["serialpos", "list", "blank_ms"]
COV_LABEL = {"serialpos": "Serial position", "list": "List number", "blank_ms": "Blank interval"}
OUTLIER_MAD = 10
N_BLANK_BINS = 8   # quantile bins, plotted at their median


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
    t, pv = ttest_ind(hi, lo, axis=0, equal_var=False, nan_policy="omit")
    ok = np.isfinite(pv); sig = np.zeros(len(pv), bool)
    if ok.any():
        sig[ok] = multipletests(pv[ok], method="fdr_bh")[0]
    sub = str(p["sid"][0])
    el = [{"sub": sub, "label": l, "lobe": lb, "d": d, "dir": (np.sign(tt) if s else 0)}
          for l, lb, d, tt, s in zip(p["labels"], lobe, p["cohens_d"], t, sig) if lb]
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


def modulators(tr: pd.DataFrame) -> pd.DataFrame:
    """Per subject x lobe OLS betas of y on the z-scored covariates (within session)."""
    out = []
    for sub, g in tr.groupby("sub"):
        X = covariates(g)
        for lb in fc.LOBES:
            if lb not in g or g[lb].notna().sum() < 50:
                continue
            ok = (g[lb].notna() & X.notna().all(1)).to_numpy()
            A = np.c_[X[ok].to_numpy(), np.ones(ok.sum())]
            v = g[lb].to_numpy()[ok]
            b, *_ = np.linalg.lstsq(A, v, rcond=None)
            r2 = 1 - np.var(v - A @ b) / np.var(v)
            out.append({"sub": sub, "lobe": lb, **dict(zip(COVS, b[:3])), "r2": r2})
    return pd.DataFrame(out)


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
            el, tr = session_tables(fc.load_pickle(str(f)), lobe_of)
            els += el
            if tr is not None:
                trs.append(tr)
        E[band], T[band] = pd.DataFrame(els), pd.concat(trs, ignore_index=True)
        os.makedirs(os.path.dirname(stem("x")), exist_ok=True)
        E[band].to_csv(stem(f"electrodes_{band}.csv"), index=False); T[band].to_csv(stem(f"trials_{band}.csv"), index=False)

    # trial analyses: scored lists only (catFR1 list -1 is an unscored practice list); serial position <= 12
    T = {b: t[(t["list"] >= 1) & (t["serialpos"] <= 12)] for b, t in T.items()}

    # 1-2. heterogeneity and variance shares
    het, var = [], []
    for band, e in E.items():
        for lb, g in e.groupby("lobe"):
            het.append({"band": band, "lobe": lb, "n_electrode_sessions": len(g),
                        "frac_up": (g["dir"] > 0).mean(), "frac_down": (g["dir"] < 0).mean()})
            el = g.groupby(["sub", "label"])["d"].mean().reset_index()
            sm = el.groupby("sub")["d"]
            within = (sm.var(ddof=1) * (sm.size() - 1)).sum() / (sm.size() - 1).clip(lower=0).sum()
            between = max(sm.mean().var(ddof=1) - within / sm.size().mean(), 0)
            var.append({"band": band, "lobe": lb, "n_subjects": sm.ngroups, "n_electrodes": len(el),
                        "sd_between_subjects": np.sqrt(between), "sd_within_subjects": np.sqrt(within),
                        "share_between_subjects": between / (between + within)})
    het, var = pd.DataFrame(het), pd.DataFrame(var)
    het.to_csv(stem("electrode_direction.csv"), index=False); var.to_csv(stem("variance_shares.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (band, h) in zip(axes, het.groupby("band", sort=False)):
        h = h.set_index("lobe").reindex(fc.LOBES); x = np.arange(len(fc.LOBES))
        ax.bar(x, h["frac_up"], color=[fc.LOBE_COLORS[l] for l in fc.LOBES], label="Increase")
        ax.bar(x, -h["frac_down"], color=[fc.LOBE_COLORS[l] for l in fc.LOBES], alpha=0.45, hatch="//", label="Decrease")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x, [l.capitalize() for l in fc.LOBES], rotation=30)
        ax.set_title(BANDS[band]); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of electrodes\n(up: increase, down: decrease)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "electrode_direction.png"), dpi=200); plt.close(fig)

    # 3. trial modulators
    st = []
    for band, tr in T.items():
        m = modulators(tr); m["band"] = band
        m.to_csv(stem(f"modulators_{band}_per_subject.csv"), index=False)
        for lb, g in m.groupby("lobe"):
            for c in COVS:
                st.append({"band": band, "lobe": lb, "covariate": c, "n": len(g), "beta": g[c].mean(),
                           "sem": g[c].sem(), "p": ttest_1samp(g[c], 0).pvalue, "median_r2": g["r2"].median()})
    st = pd.DataFrame(st); st["q"] = multipletests(st["p"], method="fdr_bh")[1]
    st.to_csv(stem("modulators_stats.csv"), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for i, (band, tr) in enumerate(T.items()):
        tr = tr.assign(blank_bin=tr.groupby(pd.qcut(tr["blank_ms"], N_BLANK_BINS, duplicates="drop"), observed=True)
                       ["blank_ms"].transform("median"), list_c=tr["list"].clip(upper=25))
        for ax, (c, xcol) in zip(axes[i], [("serialpos", "serialpos"), ("list", "list_c"), ("blank_ms", "blank_bin")]):
            for lb in fc.LOBES:
                if lb not in tr:
                    continue
                s = tr.groupby(["sub", xcol])[lb].mean().groupby(xcol)
                mu, se = s.mean(), s.sem()
                ax.plot(mu.index, mu, "o-", ms=3, color=fc.LOBE_COLORS[lb], label=lb.capitalize())
                ax.fill_between(mu.index, mu - se, mu + se, color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
            ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel(COV_LABEL[c] + (" (ms)" if c == "blank_ms" else ""))
        axes[i, 0].set_ylabel(f"{BANDS[band]} word-on power\n(z within session)")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "trial_modulators.png"), dpi=200); plt.close(fig)

    # 4. memory
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
                   alpha=1 if k == 0 else 0.45, hatch=None if k == 0 else "//", capsize=2,
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
    for name, t in [("variance shares", var), ("electrode direction", het), ("modulators (q<.05)", st[st.q < .05]),
                    ("recall ~ covariates", bst), ("SME raw vs controlled", sst)]:
        print(f"\n[{name}]\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
