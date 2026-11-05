"""plot_blank_interval.py -- what the blank screen before a word does to its response.

Reads compute.py's per-trial power (alpha, high gamma; word on 0-600 ms, word off -700 to -100 ms,
the same words row for row). Per electrode and session, log power relative to the session's mean
word-off log power: the response (word on) and the trial's own baseline (word off). Robust outlier
trials dropped (OUTLIER_MAD); lobe = mean over its electrodes; subject = mean over sessions.

  A. Response and baseline by blank length, 50 ms bins, separately for the 750-1000 ms (FR1,
     catFR1) and 800-1200 ms (pyFR) designs; mean +/- SEM across subjects.
  B. High-gamma time course (responsive electrodes) by blank tercile within session.
  C. Per subject and lobe, OLS of the high-gamma response on blank length (z within session) and
     the previous word's response at the same electrode (same list), serial position as dummies:
     one-sample t across subjects.
     The first word in the events (serial position 2) has no blank and is left out.
-> figures/blank_interval/*.png

    python plot_blank_interval.py [--n-sessions N]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import response, responsive

BEH = "word_on"
OUTLIER_MAD = 10
BANDS = {"alpha": "Alpha", "high_gamma": "High gamma"}
DESIGNS = {"750-1000 ms (FR1, catFR1)": (750, 1000), "800-1200 ms (pyFR)": (800, 1200)}
BIN_MS = 50


def clean(v: np.ndarray) -> np.ndarray:
    med = np.nanmedian(v, 0); mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    v = v.copy(); v[np.abs(v - med) > OUTLIER_MAD * mad] = np.nan
    return v


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "blank_interval"))
    os.makedirs(args.out_dir, exist_ok=True)
    lobe_of = fc.load_burke_maps()
    rows, curves = [], []
    for band in BANDS:
        for f in tqdm(fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", args.n_sessions), desc=band):
            p = fc.load_pickle(str(f))
            lo, hi = np.log(np.asarray(p["trial_pow_lo"], float)), np.log(np.asarray(p["trial_pow_hi"], float))
            if lo.shape != hi.shape:
                continue
            ref = np.nanmean(lo, 0)
            lo, hi = clean(lo) - ref, clean(hi) - ref
            roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
            lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
            w = fc.load_events(fc.dfrow_from_sid(p["sid"]), BEH)
            w = w[np.asarray(w.attrs["mask"], bool)]
            if len(w) != len(hi):
                continue
            blank = np.asarray(p["blank_ms_hi"], float)
            lst, sp = w["trial"].to_numpy(float), w["serialpos"].to_numpy(float)
            d = pd.DataFrame({"band": band, "sub": str(p["sid"][0]), "session": str(p["sid"]),
                              "blank": blank, "list": lst, "serialpos": sp})
            for lb in fc.LOBES:
                m = lobe == lb
                if m.any():
                    d[f"r_{lb}"], d[f"b_{lb}"] = np.nanmean(hi[:, m], 1), np.nanmean(lo[:, m], 1)
            rows.append(d)
            if band == "high_gamma":
                x, t = response(p)
                pk = np.nanmax(np.abs(x), -1)   # transient trials: their whole-clip peak
                med = np.nanmedian(pk, 0); mad = np.nanmedian(np.abs(pk - med), 0) * 1.4826
                x[(np.abs(pk - med) > OUTLIER_MAD * mad) | np.isnan(hi)] = np.nan
                ok = np.isfinite(blank)
                terc = np.full(len(blank), -1)
                terc[ok] = np.digitize(blank[ok], np.nanquantile(blank[ok], [1 / 3, 2 / 3]))
                use = responsive(p)
                for lb in fc.LOBES:
                    e = use & (lobe == lb)
                    if e.any():
                        for k in range(3):
                            if (terc == k).sum() > 5:
                                curves.append({"sub": str(p["sid"][0]), "lobe": lb, "tercile": k,
                                               "curve": np.nanmean(x[np.ix_(terc == k, e)], axis=(0, 1)), "t": t})
    D = pd.concat(rows, ignore_index=True)
    D = D[(D["list"] >= 1) & D["blank"].notna()]
    D.to_csv(fc.table_path(join(args.out_dir, "blank_trials.csv")), index=False)

    # A. response and baseline by blank length
    fig, axes = plt.subplots(len(BANDS) * 2, len(DESIGNS), figsize=(11, 12), sharex="col", squeeze=False)
    for i, band in enumerate(BANDS):
        b = D[D["band"] == band]
        for j, (dname, (a0, a1)) in enumerate(DESIGNS.items()):
            g = b[(b["blank"] >= a0 - 5) & (b["blank"] <= a1 + 5)].copy()
            g = g[g.groupby("session")["blank"].transform("max") <= a1 + 5]   # sessions of this design only
            g["bin"] = (np.clip(g["blank"], a0, a1 - 1) - a0) // BIN_MS * BIN_MS + a0 + BIN_MS / 2
            for row, (pre, lab) in enumerate((("r", "response (word on)"), ("b", "baseline (word off)"))):
                ax = axes[2 * i + row, j]
                for lb in fc.LOBES:
                    col = f"{pre}_{lb}"
                    if col not in g:
                        continue
                    s = g.groupby(["sub", "bin"])[col].mean().groupby("bin")
                    mu, se = s.mean(), s.sem()
                    ax.plot(mu.index, mu, "o-", ms=3, color=fc.LOBE_COLORS[lb], label=lb.capitalize())
                    ax.fill_between(mu.index, mu - se, mu + se, color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
                ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
                ax.set_ylabel(f"{BANDS[band]} {lab}\nlog power vs session word-off mean", fontsize=8)
                if i == 0 and row == 0:
                    ax.set_title(dname)
                if 2 * i + row == 2 * len(BANDS) - 1:
                    ax.set_xlabel("Blank before the word (ms)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "blank_response_baseline.png"), dpi=200); plt.close(fig)

    # B. high-gamma time course by blank tercile
    C = pd.DataFrame(curves)
    fig, axes = plt.subplots(1, len(fc.LOBES), figsize=(17, 3), sharex=True)
    for ax, lb in zip(axes, fc.LOBES):
        c = C[C["lobe"] == lb]
        if c.empty:
            ax.set_visible(False); continue
        t = c["t"].iloc[0]; show = (t >= -100) & (t < 1000)
        for k, (lab, a) in enumerate((("shortest", 0.35), ("middle", 0.65), ("longest", 1.0))):
            v = c[c["tercile"] == k].groupby("sub")["curve"].apply(lambda s: np.nanmean(np.stack(s), 0))
            if len(v):
                m = np.stack(v.to_numpy())[:, show]
                ax.plot(t[show], np.nanmean(m, 0), color=fc.LOBE_COLORS[lb], alpha=a, lw=2, label=f"{lab} third")
        ax.axhline(0, color="0.5", lw=0.8); ax.set_title(lb.capitalize(), color=fc.LOBE_COLORS[lb])
        ax.set_xlabel("Time after word onset (ms)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("High-gamma response\n(responsive electrodes)"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "blank_timecourse.png"), dpi=200); plt.close(fig)

    # C. blank length vs the previous word's response
    hg = D[D["band"] == "high_gamma"].sort_values(["session", "list", "serialpos"])
    out = []
    for lb in fc.LOBES:
        col = f"r_{lb}"
        if col not in hg:
            continue
        h = hg[["sub", "session", "list", "serialpos", "blank", col]].dropna().copy()
        h["prev"] = h.groupby(["session", "list"])[col].shift(1)
        h = h.dropna()
        for sub, g in h.groupby("sub"):
            if len(g) < 100:
                continue
            z = g.groupby("session")[["blank", "prev", col]].transform(lambda v: (v - v.mean()) / v.std())
            spd = pd.get_dummies(g["serialpos"].astype(int), drop_first=True).to_numpy(float)   # serial position: both words shrink along the list
            A = np.c_[z[["blank", "prev"]].to_numpy(), spd, np.ones(len(z))]
            beta = np.linalg.lstsq(A, z[col].to_numpy(), rcond=None)[0]
            out.append({"lobe": lb, "sub": sub, "blank": beta[0], "prev": beta[1]})
    O = pd.DataFrame(out)
    st = pd.DataFrame([{"lobe": lb, "covariate": c, "n": len(g), "beta": g[c].mean(), "p": ttest_1samp(g[c], 0).pvalue}
                       for lb, g in O.groupby("lobe") for c in ("blank", "prev")])
    st.to_csv(fc.table_path(join(args.out_dir, "blank_prev_stats.csv")), index=False)
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
