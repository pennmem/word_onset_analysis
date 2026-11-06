"""plot_variance_structure.py -- where the word response varies (preliminary, subset of subjects).

Reads compute.py's power pickles (alpha, high gamma) for N_SUBJECTS random subjects.
Per trial and electrode: the response = word-on minus the electrode-session mean word-off
(alpha: log multitaper power, 0-600 ms; high gamma: log Hilbert power, and plot_latency.response over
time). Outlier trials dropped per electrode (whole-clip peak or window mean > OUTLIER_MAD x MAD).
Scored lists, serial positions 2-12 (position 1 follows the countdown and is not in the events).

  1. Variance components (mixed_models.R variance; DeHaan 2026 ch. 3's random-intercept model, on
     the single-trial response instead of recall): subject, electrode, electrode x session, serial
     position, electrode x serial position, list number, session x list, word (item), the word
     presentation shared by a lobe's electrodes (event), blank length + same category as the
     previous word (fixed), residual (electrode-specific, trial to trial). Quick preliminary fit:
     at most LMM_MAX_ELECTRODES random electrodes per subject and lobe (refit: --refit). Shown as
     stacked shares per lobe, the task terms (serial position, electrode x serial position, list, word,
     blank + category) grouped. The same model for single-trial high-gamma latency (half-max of each
     trial's response, responsive electrodes; plot_latency.half_max).
  2. Task effects: the model's serial-position and list-number estimates per lobe (alpha, high gamma, latency).
  3. Alpha vs high gamma: per electrode Pearson r across trials, vs per subject r across electrodes;
     per lobe, subjects' mean Fisher z vs 0 (one-sample t), BH-FDR over both measures x lobes.
-> figures/variance_structure/*.png (tables and model fits: results/variance_structure/)

    python plot_variance_structure.py [--n-subjects N] [--from-csv] [--refit]
"""
from __future__ import annotations

import os
import subprocess
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
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
BANDS = {"alpha": "Alpha power", "high_gamma": "High-gamma power", "latency": "High-gamma latency\n(responsive electrodes)"}
GROUPS = {   # stacked share segments: label -> (model terms, colour)
    "Subject": (["subject"], "#1f4fa3"), "Electrode": (["electrode"], "#d62728"),
    "Electrode x session": (["elsess"], "#f2c12e"),
    "Task (serial position, list, word, blank, category)": (["sp", "el_sp", "list", "item", "fixed"], "#2ca02c"),
    "Session x list": (["sess_list"], "#ff7f0e"), "Word presentation (shared by a lobe's electrodes)": (["event"], "#7b4fa3"),
    "Residual (electrode, trial)": (["Residual"], "#d9d9d9")}

def robust_ok(v: np.ndarray) -> np.ndarray:
    """(trials, electrodes) True where the trial is not a robust outlier of its electrode."""
    med = np.nanmedian(v, 0); mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    return np.abs(v - med) <= OUTLIER_MAD * mad


def task_columns(w: pd.DataFrame, blank: np.ndarray) -> pd.DataFrame:
    """Per trial: serial position, list, word, blank length (z within session; NaN where it cannot be
    measured: session median), same category as the previous word (catFR1; 0 elsewhere)."""
    lst = w["trial"].to_numpy(float)
    b = np.where(np.isfinite(blank), blank, np.nanmedian(blank))
    same = np.zeros(len(w))
    if "category" in w:
        cat = w["category"].astype(str).to_numpy()
        same = ((cat == np.r_[[""], cat[:-1]]) & (np.r_[[np.nan], lst[:-1]] == lst)).astype(float)
    item = w["item_name"] if "item_name" in w else w["item"]
    return pd.DataFrame({"sp": w["serialpos"].to_numpy(int), "list": lst.astype(int),
                         "item": item.astype(str).str.upper().to_numpy(),
                         "blank_z": (b - b.mean()) / (b.std() or 1), "same": same})


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variance_structure"), **{
        "--n-subjects": dict(type=int, default=N_SUBJECTS, help="random subjects (seed 0)"),
        "--from-csv": dict(action="store_true", help="reuse the saved tables and model fits"),
        "--refit": dict(action="store_true", help="refit the single-trial mixed models")})
    os.makedirs(args.out_dir, exist_ok=True)
    tdir = os.path.dirname(fc.table_path(join(args.out_dir, "x")))
    lobe_of = fc.load_burke_maps()
    files = {b: {f.name: f for f in fc.session_files(power_dir(save_root, BEH, b), "_power.pkl", None)}
             for b in ("alpha", "high_gamma")}
    subs = sorted({n.split("_")[0] for n in files["high_gamma"]})
    pick = set(np.random.default_rng(0).choice(subs, min(args.n_subjects, len(subs)), replace=False))
    names = sorted(n for n in files["high_gamma"] if n.split("_")[0] in pick and n in files["alpha"])

    if not args.from_csv:
        trial_rows, lv_rows = [], []
        for n in tqdm(names, desc="sessions"):
            g, a = (fc.load_pickle(str(files[b][n])) for b in ("high_gamma", "alpha"))
            sub, sess = str(g["sid"][0]), str(g["sid"])
            w = fc.load_events(fc.dfrow_from_sid(g["sid"]), BEH)
            blank = _blank_before(w)[np.asarray(w.attrs["mask"], bool)]   # on all words: position 2 follows position 1
            w = w[np.asarray(w.attrs["mask"], bool)]
            if len(w) != len(np.asarray(g["trial_pow_hi"])):
                continue
            roi = fc.roi_of_reg_full(g["reg_full"], lobe_of)
            lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
            x, t = response(g)                                     # (trials, electrodes, time)
            pk = np.nanmax(np.abs(x), -1)
            ahi, alo = np.log(np.asarray(a["trial_pow_hi"], float)), np.log(np.asarray(a["trial_pow_lo"], float))
            ya = ahi - np.nanmean(alo, 0)                          # alpha response per trial
            ghi, glo = np.log(np.asarray(g["trial_pow_hi"], float)), np.log(np.asarray(g["trial_pow_lo"], float))
            yg = ghi - np.nanmean(glo, 0)
            ok = robust_ok(pk) & robust_ok(yg) & robust_ok(ya)
            x[~ok] = np.nan; ya[~ok] = np.nan; yg[~ok] = np.nan
            tc = task_columns(w, blank)
            use = (tc["list"].to_numpy() >= 1) & tc["sp"].between(*SP_RANGE).to_numpy()
            resp = responsive(g)
            for c in np.flatnonzero(lobe != None):  # noqa: E711
                el = f"{sub}|{g['labels'][c]}"
                e = use & np.isfinite(yg[:, c]) & np.isfinite(ya[:, c])
                if e.sum() < 20:
                    continue
                lat = half_max(x[e, c], t)[0] if resp[c] else np.full(e.sum(), np.nan)   # single-trial half-max
                trial_rows.append(tc[e].assign(subject=sub, electrode=el, session=sess, lobe=lobe[c],
                                               event=np.flatnonzero(e), alpha=ya[e, c], high_gamma=yg[e, c],
                                               latency=lat))
                lv_rows.append({"subject": sub, "electrode": el, "session": sess, "lobe": lobe[c],
                                "d_alpha": a["cohens_d"][c], "d_hg": g["cohens_d"][c],
                                "r_trial": np.corrcoef(ya[e, c], yg[e, c])[0, 1]})
        T = pd.concat(trial_rows, ignore_index=True)
        T.to_csv(join(tdir, "trials.csv"), index=False)
        pd.DataFrame(lv_rows).to_csv(join(tdir, "electrode_sessions.csv"), index=False)
        print(f"[variance] {T['subject'].nunique()} subjects, {T['electrode'].nunique()} electrodes, "
              f"{T['session'].nunique()} sessions, {len(T)} electrode-trials")
        if args.refit or not os.path.exists(join(tdir, "components.csv")):
            subprocess.run(["Rscript", "mixed_models.R", "variance", tdir, str(LMM_MAX_ELECTRODES)], check=True)
    E = pd.read_csv(join(tdir, "electrode_sessions.csv"))
    VC = pd.read_csv(join(tdir, "components.csv"))
    BL, FX = pd.read_csv(join(tdir, "task_blups.csv")), pd.read_csv(join(tdir, "task_fixed.csv"))
    VC["share"] = VC["var"] / VC.groupby(["lobe", "band"])["var"].transform("sum")
    lobes = [lb for lb in fc.LOBES if lb in set(VC["lobe"])]

    # 1. variance components of single trials (one model per lobe and band), stacked
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), sharey=True)
    for ax, (b, lab) in zip(axes, BANDS.items()):
        v = VC[VC["band"] == b].pivot(index="lobe", columns="term", values="share").reindex(lobes).fillna(0)
        left = np.zeros(len(v))
        for g, (terms, col) in GROUPS.items():
            w = v.reindex(columns=terms, fill_value=0).sum(1).to_numpy()
            ax.barh(range(len(v)), w, left=left, color=col, height=0.75, edgecolor="w", lw=1.5, label=g)
            for y, (l0, ww) in enumerate(zip(left, w)):
                if ww >= 0.06:
                    ax.text(l0 + ww / 2, y, f"{ww:.0%}", ha="center", va="center", fontsize=11,
                            color="k" if col in ("#f2c12e", "#d9d9d9") else "w")
            left += w
        nn = VC[VC["band"] == b].groupby("lobe")[["nsub", "nel"]].first().reindex(v.index)
        for y, r_ in enumerate(nn.itertuples()):
            ax.text(1.01, y, f"{r_.nsub}/{r_.nel}", va="center", fontsize=9, color="0.35")
        ax.set_yticks(range(len(v)), [f"$\\bf{{{l.capitalize()}}}$" for l in v.index], fontsize=12); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_title(lab, fontsize=14); ax.set_xlabel("Share of single-trial variance", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.text(0.995, 0.02, "n = subjects / electrodes", ha="right", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(join(args.out_dir, "variance_components.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # 2. task effects: serial position and list number estimates
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), sharex="col")
    for i, (b, lab) in enumerate(BANDS.items()):
        for k, (term, xl) in enumerate((("sp", "Serial position"), ("list", "List number"))):
            ax = axes[i, k]
            for lb in lobes:
                d = BL[(BL["band"] == b) & (BL["lobe"] == lb) & (BL["term"] == term)].sort_values("level")
                ax.plot(d["level"], d["est"], "o-", ms=3, color=fc.LOBE_COLORS[lb], label=lb.capitalize())
                ax.fill_between(d["level"], d["est"] - d["se"], d["est"] + d["se"], color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
            ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
            axes[-1, k].set_xlabel(xl)
        axes[i, 0].set_ylabel(lab.split("\n")[0] + ("\n(deviation, ms)" if b == "latency" else "\n(deviation, log power)"))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "task_effects.png"), dpi=200); plt.close(fig)

    # 3. alpha vs high gamma: trial by trial vs across electrodes
    er = E.groupby(["subject", "electrode", "lobe"], as_index=False)["r_trial"].mean().rename(columns={"r_trial": "r"})
    ae = E.groupby(["subject", "lobe", "electrode"], as_index=False)[["d_alpha", "d_hg"]].mean()
    acr = pd.DataFrame([{"subject": s_, "lobe": lb, "r": np.corrcoef(g["d_alpha"], g["d_hg"])[0, 1]}
                        for (s_, lb), g in ae.groupby(["subject", "lobe"]) if len(g) >= 5])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x0 = np.arange(len(fc.LOBES))
    for k, (dd, lab, col) in enumerate(((er, "Across trials (per electrode)", "C3"),
                                        (acr, "Across electrodes (per subject)", "0.45"))):
        fc.box_dots(ax, [dd.loc[dd["lobe"] == l, "r"].dropna() for l in fc.LOBES], x0 + (k - 0.5) * 0.4,
                    [col] * len(fc.LOBES), width=0.34)
        ax.plot([], [], "s", color=col, alpha=0.5, ms=9, label=lab)
    # per lobe: subjects' mean Fisher z vs 0 (one-sample t), BH-FDR over both measures x lobes
    cst = pd.DataFrame([{"measure": k, "lobe": lb, "n": len(z), "r": np.tanh(z.mean()), "p": ttest_1samp(z, 0).pvalue}
                        for k, dd in (("across trials", er), ("across electrodes", acr)) for lb in fc.LOBES
                        if len(z := np.arctanh(dd[dd["lobe"] == lb].groupby("subject")["r"].mean().clip(-0.999, 0.999))) >= 5])
    cst["q"] = multipletests(cst["p"], method="fdr_bh")[1]
    cst.to_csv(join(tdir, "alpha_hg_coupling_stats.csv"), index=False)
    for k, kk in enumerate(("across trials", "across electrodes")):
        for r_ in cst[cst["measure"] == kk].itertuples():
            stars = "***" if r_.q < .001 else "**" if r_.q < .01 else "*" if r_.q < .05 else "n.s."
            ax.text(fc.LOBES.index(r_.lobe) + (k - 0.5) * 0.4, 1.02, stars, ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x0, [l.capitalize() for l in fc.LOBES], rotation=30)
    ax.set_ylabel("r, alpha vs high-gamma response"); ax.set_ylim(-1, 1.12); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "alpha_hg_coupling.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n[variance components, % of single-trial variance]\n",
          (VC.pivot_table(index=["band", "term"], columns="lobe", values="share") * 100)[lobes].round(2).to_string())
    print("\n[fixed effects]\n", FX[FX["term"] != "(Intercept)"].round(4).to_string(index=False))
    print("\n[alpha-hg r] across trials, median by lobe:", er.groupby("lobe")["r"].median().round(3).to_dict())
    print("            across electrodes:", acr.groupby("lobe")["r"].median().round(3).to_dict())
    print("            tests (subjects' mean r vs 0):\n", cst.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
