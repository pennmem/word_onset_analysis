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
-> figures/variance_structure/*.png (tables and model fits: results/variance_structure/)

    python plot_variance_structure.py [--n-subjects N] [--from-csv] [--refit] [--cores C] [--max-electrodes M]
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
    "Subject": (["subject"], "#1f4fa3"), "Region (within lobe)": (["region"], "#17becf"), "Electrode": (["electrode"], "#d62728"),
    "Electrode in session": (["elsess"], "#f2c12e"),
    "Session": (["session"], "#8c564b"), "List in session": (["sess_list"], "#ff7f0e"),
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
    a, b = v[lag:], v[:-lag]
    ok = (lst[lag:] == lst[:-lag]) & (sp[lag:] == sp[:-lag] + lag) & np.isfinite(a) & np.isfinite(b)
    if cur is not None:
        ok &= cur[lag:]
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 10 else np.nan


def task_columns(w: pd.DataFrame, blank: np.ndarray) -> pd.DataFrame:
    """Per event: serial position, list, word, blank length (z within session; NaN where it cannot be
    measured: session median)."""
    lst = w["trial"].to_numpy(float)
    b = np.where(np.isfinite(blank), blank, np.nanmedian(blank))
    item = w["item_name"] if "item_name" in w else w["item"]
    return pd.DataFrame({"sp": w["serialpos"].to_numpy(int), "list": lst.astype(int),
                         "item": item.astype(str).str.upper().to_numpy(),
                         "blank_z": (b - b.mean()) / (b.std() or 1)})


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variance_structure"), **{
        "--n-subjects": dict(type=int, default=N_SUBJECTS, help="random subjects (seed 0)"),
        "--from-csv": dict(action="store_true", help="reuse the saved tables and model fits"),
        "--refit": dict(action="store_true", help="refit the single-event mixed models"),
        "--cores": dict(type=int, default=6, help="parallel model fits in R"),
        "--max-electrodes": dict(type=int, default=LMM_MAX_ELECTRODES, help="per subject and lobe, in the models")})
    os.makedirs(args.out_dir, exist_ok=True)
    tdir = os.path.dirname(fc.table_path(join(args.out_dir, "x")))
    lobe_of = fc.load_burke_maps()
    files = {b: {f.name: f for f in fc.session_files(power_dir(save_root, BEH, b), "_power.pkl", None)}
             for b in ("alpha", "high_gamma")}
    subs = sorted({n.split("_")[0] for n in files["high_gamma"]})
    pick = set(np.random.default_rng(0).choice(subs, min(args.n_subjects, len(subs)), replace=False))
    names = sorted(n for n in files["high_gamma"] if n.split("_")[0] in pick and n in files["alpha"])

    if not args.from_csv:
        trial_rows, lv_rows, off, lev = [], [], {}, {}
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
            x, t = response(g)                                     # (events, electrodes, time)
            pk = np.nanmax(np.abs(x), -1)
            ahi, alo = np.log(np.asarray(a["trial_pow_hi"], float)), np.log(np.asarray(a["trial_pow_lo"], float))
            rhi, rlo = {}, {}
            for k, p_ in (("a", a), ("g", g)):
                rhi[k], rlo[k] = np.asarray(p_["trial_pow_hi"], float), np.asarray(p_["trial_pow_lo"], float)
            sa, sg = pooled_sd(rhi["a"], rlo["a"]), pooled_sd(rhi["g"], rlo["g"])
            ya, yg = (rhi["a"] - rlo["a"].mean(0)) / sa, (rhi["g"] - rlo["g"].mean(0)) / sg   # mean over events: Cohen's d
            pa, pg = (rhi["a"] - rlo["a"]) / sa, (rhi["g"] - rlo["g"]) / sg                   # word on - off, same event
            ok = robust_ok(pk) & robust_ok(np.log(rhi["g"])) & robust_ok(ahi)
            x[~ok] = np.nan; ya[~ok] = np.nan; yg[~ok] = np.nan
            tc = task_columns(w, blank)
            use = (tc["list"].to_numpy() >= 1) & tc["sp"].between(*SP_RANGE).to_numpy()
            resp = responsive(g)
            sp_, lst_ = tc["sp"].to_numpy(), tc["list"].to_numpy()
            long_ = blank > np.nanmedian(blank); short_ = blank < np.nanmedian(blank)   # split as plot_anticipation
            design = "800-1200" if str(g["sid"][1]) == "pyFR" else "750-1000"
            lo_env = np.asarray(g["env_lo"], float)                # word-off window, (events, electrodes, time)
            if lo_env.shape[:2] == x.shape[:2]:   # 4. time since the previous word's offset
                xl = lo_env - np.nanmean(lo_env, axis=(0, 2))[:, None]
                xl[~ok] = np.nan
                al = alo - np.nanmean(alo, 0); al[~ok] = np.nan
                tl = g["env_lo_t0_ms"] + g["env_step_ms"] * (np.arange(xl.shape[-1]) + 0.5)
                m = use & np.isfinite(blank)
                for lb in fc.LOBES:
                    e = resp & (lobe == lb)
                    if e.any():
                        for k, v, since in (("high_gamma", np.nanmean(xl[m][:, e], 1), blank[m, None] + tl[None]),
                                            ("alpha", np.nanmean(al[m][:, e], 1)[:, None], blank[m, None] + tl.mean())):
                            bi, ok2 = (since // OFFSET_BIN_MS).astype(int).ravel(), np.isfinite(v.ravel())
                            nb = OFFSET_MAX_MS // OFFSET_BIN_MS
                            ok2 &= (bi >= 0) & (bi < nb)
                            s0, n0 = off.get((sub, lb, k), (np.zeros(nb), np.zeros(nb)))
                            off[(sub, lb, k)] = (s0 + np.bincount(bi[ok2], v.ravel()[ok2], nb),
                                                 n0 + np.bincount(bi[ok2], minlength=nb))
            for c in np.flatnonzero(lobe != None):  # noqa: E711
                el = f"{sub}|{g['labels'][c]}"
                e = use & np.isfinite(yg[:, c]) & np.isfinite(ya[:, c])
                if e.sum() < 20:
                    continue
                lat = half_max(x[e, c], t)[0] if resp[c] else np.full(e.sum(), np.nan)   # single-event half-max
                for meas, v in (("alpha", ya[e, c]), ("high_gamma", yg[e, c]), ("latency", lat)):
                    for term in ("sp", "list"):
                        lv, fin = tc.loc[e, term].to_numpy(), np.isfinite(v)
                        s0, n0 = lev.get((sub, lobe[c], meas, term), (np.zeros(LEVELS), np.zeros(LEVELS)))
                        lev[(sub, lobe[c], meas, term)] = (s0 + np.bincount(lv[fin], v[fin], LEVELS), n0 + np.bincount(lv[fin], minlength=LEVELS))
                trial_rows.append(tc[e].assign(subject=sub, electrode=el, session=sess, lobe=lobe[c],
                                               event=np.flatnonzero(e), alpha=ya[e, c], high_gamma=yg[e, c],
                                               latency=lat, blank_ms=blank[e], design=design))
                lags, res = {}, {}   # 5. carry-over: serial-position means out, then r with the word 1 / 2 back
                for k, v in (("alpha", ya[:, c]), ("high_gamma", yg[:, c])):
                    v = np.where(e, v, np.nan)
                    v = v - pd.Series(v).groupby(sp_).transform("mean").to_numpy()
                    lags.update({f"r1_{k}": lag_r(v, sp_, lst_, 1), f"r2_{k}": lag_r(v, sp_, lst_, 2),
                                 f"r1long_{k}": lag_r(v, sp_, lst_, 1, long_), f"r1short_{k}": lag_r(v, sp_, lst_, 1, short_)})
                    res[k] = v - pd.Series(v).groupby(lst_).transform("mean").to_numpy()   # 3. list means out too
                ok_ = np.isfinite(res["alpha"]) & np.isfinite(res["high_gamma"])
                lags["r_fast"] = np.corrcoef(res["alpha"][ok_], res["high_gamma"][ok_])[0, 1]
                lv_rows.append({**lags, "subject": sub, "electrode": el, "session": sess, "lobe": lobe[c],
                                "d_alpha": a["cohens_d"][c], "d_hg": g["cohens_d"][c],
                                "r_trial": np.corrcoef(ya[e, c], yg[e, c])[0, 1], "n": e.sum(),
                                "pa_sum": pa[e, c].sum(), "pa_sq": (pa[e, c] ** 2).sum(),
                                "pg_sum": pg[e, c].sum(), "pg_sq": (pg[e, c] ** 2).sum()})
        T = pd.concat(trial_rows, ignore_index=True)
        print(f"[variance] {T['subject'].nunique()} subjects, {T['electrode'].nunique()} electrodes, "
              f"{T['session'].nunique()} sessions, {len(T)} electrode-events")
        rng = np.random.default_rng(0)   # the models: at most LMM_MAX_ELECTRODES random electrodes per subject and lobe
        pick = lambda d: {el for es in d.groupby(["subject", "lobe"])["electrode"].unique() for el in rng.permutation(es)[:args.max_electrodes]}
        keep = pick(T)                                     # power models
        keep_lat = pick(T[T["latency"].notna()])           # latency model: among responsive electrodes
        T = T[T["electrode"].isin(keep | keep_lat)].assign(in_power=lambda d: d["electrode"].isin(keep))
        T.loc[~T["electrode"].isin(keep_lat), "latency"] = np.nan
        T.to_csv(join(tdir, "trials.csv"), index=False)
        del T
        pd.DataFrame(lv_rows).to_csv(join(tdir, "electrode_sessions.csv"), index=False)
        pd.to_pickle(off, join(tdir, "offset.pkl")); pd.to_pickle(lev, join(tdir, "levels.pkl"))
        for mode, out in (("variance", "components.csv"), ("blank", "blank_fixed.csv")):
            if args.refit or not os.path.exists(join(tdir, out)):
                subprocess.run(["Rscript", "mixed_models.R", mode, tdir, str(args.cores)], check=True)
    E = pd.read_csv(join(tdir, "electrode_sessions.csv"))
    VC = pd.read_csv(join(tdir, "components.csv"))
    VC["share"] = VC["var"] / VC.groupby(["lobe", "band"])["var"].transform("sum")
    lobes = [lb for lb in fc.LOBES if lb in set(VC["lobe"])]

    # 1. variance components of single events (one model per lobe and band), stacked; power and latency apart
    for name, bands in (("variance_components_power", ["alpha", "high_gamma"]), ("variance_components_latency", ["latency"])):
        fig, axes = plt.subplots(1, len(bands), figsize=(6.5 * len(bands), 4.6), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (b, lab) in zip(axes, [(b, BANDS[b]) for b in bands]):
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
            ax.set_yticks(range(len(v)), [f"$\\bf{{{l.capitalize()}}}$" for l in v.index], fontsize=12)
            ax.set_xlim(0, 1); ax.set_title(lab, fontsize=14); ax.set_xlabel("Share of variance across word events", fontsize=12)
            ax.tick_params(axis="x", labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()   # shared y: once for all panels
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=12, ncol=3 if len(bands) > 1 else 2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
        fig.text(0.995, 0.02, "n = subjects / electrodes", ha="right", fontsize=9, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(join(args.out_dir, f"{name}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # 2. task effects: per level, deviations from each subject's own mean; per-subject slopes vs 0
    lev = pd.read_pickle(join(tdir, "levels.pkl"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), sharex="col")
    trend = []
    for i, (b, lab) in enumerate(BANDS.items()):
        for k, (term, xl) in enumerate((("sp", "Serial position"), ("list", "List number"))):
            ax = axes[i, k]
            for lb in lobes:
                v = np.array([sm / np.where(n >= 5, n, np.nan) for (s_, l_, b_, t_), (sm, n) in lev.items() if l_ == lb and b_ == b and t_ == term])
                v = v[np.isfinite(v).sum(1) >= 3]
                if len(v) < 5:
                    continue
                v -= np.nanmean(v, 1, keepdims=True)
                ns = np.isfinite(v).sum(0); sh = ns >= 5
                mu = np.nanmean(v[:, sh], 0); ci = np.nanstd(v[:, sh], 0, ddof=1) / np.sqrt(ns[sh]) * tdist.ppf(0.975, ns[sh] - 1)
                ax.plot(np.flatnonzero(sh), mu, "o-", ms=3, color=fc.LOBE_COLORS[lb], label=lb.capitalize())
                ax.fill_between(np.flatnonzero(sh), mu - ci, mu + ci, color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
                sl = [np.polyfit(np.flatnonzero(np.isfinite(r_)), r_[np.isfinite(r_)], 1)[0] for r_ in v]
                trend.append({"measure": b, "term": term, "lobe": lb, "n": len(sl), "slope": np.mean(sl), "p": ttest_1samp(sl, 0).pvalue})
            ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
            axes[-1, k].set_xlabel(xl)
        axes[i, 0].set_ylabel(lab.split("\n")[0] + ("\n(deviation, ms)" if b == "latency" else "\n(deviation, Cohen's d)"))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "task_effects.png"), dpi=200); plt.close(fig)
    trend = pd.DataFrame(trend); trend["q"] = multipletests(trend["p"], method="fdr_bh")[1]
    trend.to_csv(join(tdir, "task_trends.csv"), index=False)

    # 3. alpha vs high gamma: event by event; across electrodes, signed and in magnitude; prevalence
    er = E.groupby(["subject", "electrode", "lobe"], as_index=False)["r_trial"].mean().rename(columns={"r_trial": "r"})
    ae = E.groupby(["subject", "lobe", "electrode"], as_index=False)[["d_alpha", "d_hg"]].mean()
    acr = pd.DataFrame([{"subject": s_, "lobe": lb, "r": np.corrcoef(g["d_alpha"], g["d_hg"])[0, 1],
                         "r_abs": np.corrcoef(g["d_alpha"].abs(), g["d_hg"].abs())[0, 1]}
                        for (s_, lb), g in ae.groupby(["subject", "lobe"]) if len(g) >= 5])
    mag = acr.assign(r=acr["r_abs"])
    measures = {"across events": er, "across electrodes": acr, "across electrodes, magnitude": mag}
    cst = pd.DataFrame([{"measure": k, "lobe": lb, "n": len(z), "r": np.tanh(z.mean()), "p": ttest_1samp(z, 0).pvalue}
                        for k, dd in measures.items() for lb in fc.LOBES
                        if len(z := np.arctanh(dd[dd["lobe"] == lb].groupby("subject")["r"].mean().clip(-0.999, 0.999))) >= 5])
    cst["q"] = multipletests(cst["p"], method="fdr_bh")[1]
    cst.to_csv(join(tdir, "alpha_hg_coupling_stats.csv"), index=False)
    # alpha vs high gamma by timescale: word event (same electrode, serial-position and list means out), serial-position
    # and list means (subject x lobe, all electrodes; levels with >= 5 events, >= 5 levels); t vs 0 per lobe (fc.roi_stats)
    def level_r(term: str) -> pd.DataFrame:
        rows = []
        for (s_, lb, b_, t_), (sm, n) in lev.items():
            if b_ == "alpha" and t_ == term and (s_, lb, "high_gamma", term) in lev:
                sg, ng = lev[(s_, lb, "high_gamma", term)]
                ok = (n >= 5) & (ng >= 5)
                if ok.sum() >= 5:
                    rows.append({"sub": s_, "roi": f"B-{lb}", "r": np.corrcoef(sm[ok] / n[ok], sg[ok] / ng[ok])[0, 1]})
        return pd.DataFrame(rows)
    fast = (E.groupby(["subject", "lobe", "electrode"])["r_fast"].mean().groupby(["subject", "lobe"]).mean().reset_index()
            .rename(columns={"subject": "sub", "r_fast": "r"}).assign(roi=lambda d: "B-" + d["lobe"]))
    ts = {"Word event": fast, "Serial position": level_r("sp"), "List": level_r("list")}
    ts = {k: fc.roi_stats(d, "r").assign(timescale=k) for k, d in ts.items()}
    pd.concat(ts.values()).to_csv(join(tdir, "alpha_hg_timescales_stats.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    fc.roi_dot_panel(ax, ts, "r (alpha vs high gamma)")
    ax.set_ylim(-0.3, 0.3); fig.savefig(join(args.out_dir, "alpha_hg_timescales.png"), dpi=200); plt.close(fig)
    x0 = np.arange(len(fc.LOBES))
    for name, panels, size in (("alpha_hg_across_electrodes", (("across electrodes", "Signed Cohen's d", "C1"),
                                                               ("across electrodes, magnitude", "|Cohen's d|", "C4")), (8, 3.8)),):
        fig, ax = plt.subplots(figsize=size)
        off_ = (np.arange(len(panels)) - (len(panels) - 1) / 2) * 0.4
        for dx, (kk, lab, col) in zip(off_, panels):
            dd = measures[kk]
            fc.box_dots(ax, [dd.loc[dd["lobe"] == l, "r"].dropna() for l in fc.LOBES], x0 + dx, [col] * len(fc.LOBES), width=0.34)
            ax.plot([], [], "s", color=col, alpha=0.5, ms=9, label=lab)
            for r_ in cst[cst["measure"] == kk].itertuples():
                stars = "***" if r_.q < .001 else "**" if r_.q < .01 else "*" if r_.q < .05 else "n.s."
                ax.text(fc.LOBES.index(r_.lobe) + dx, 1.02, stars, ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x0, [l.capitalize() for l in fc.LOBES], rotation=30)
        ax.set_ylabel("r (alpha vs HG response)"); ax.set_ylim(-1, 1.12)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(join(args.out_dir, f"{name}.png"), dpi=200); plt.close(fig)
    pr = E.groupby(["lobe", "electrode"], as_index=False)[["n", "pa_sum", "pa_sq", "pg_sum", "pg_sq"]].sum()
    for k, sm, sq in (("alpha", "pa_sum", "pa_sq"), ("high_gamma", "pg_sum", "pg_sq")):
        tv = (pr[sm] / pr["n"]) / np.sqrt((pr[sq] - pr[sm] ** 2 / pr["n"]) / (pr["n"] - 1) / pr["n"])
        q = multipletests(2 * tdist.sf(np.abs(tv), pr["n"] - 1), method="fdr_bh")[1]
        pr[k] = np.where(q >= 0.05, "none", np.where(tv > 0, "up", "down"))
    prev = pr.groupby("lobe").apply(lambda d: pd.Series({
        "n": len(d), "neither": ((d["alpha"] == "none") & (d["high_gamma"] == "none")).mean(),
        "alpha_only": ((d["alpha"] != "none") & (d["high_gamma"] == "none")).mean(),
        "hg_only": ((d["alpha"] == "none") & (d["high_gamma"] != "none")).mean(),
        "both": ((d["alpha"] != "none") & (d["high_gamma"] != "none")).mean(),
        "alpha_down_hg_up": ((d["alpha"] == "down") & (d["high_gamma"] == "up")).mean(),
        "alpha_up_hg_up": ((d["alpha"] == "up") & (d["high_gamma"] == "up")).mean()}), include_groups=False)
    prev.to_csv(join(tdir, "prevalence.csv"))

    # 4. recovery from the previous word's offset
    off = pd.read_pickle(join(tdir, "offset.pkl"))
    tb = (np.arange(OFFSET_MAX_MS // OFFSET_BIN_MS) + 0.5) * OFFSET_BIN_MS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (k, lab) in zip(axes, (("high_gamma", "High-gamma envelope\nvs electrode's word-off mean"),
                                   ("alpha", "Alpha log power\nvs session mean (window centre)"))):
        for lb in lobes:
            v = np.array([sm / np.where(n >= 5, n, np.nan) for (s_, l_, k_), (sm, n) in off.items() if l_ == lb and k_ == k])
            if len(v) < 5:
                continue
            ns = np.isfinite(v).sum(0)
            mu = np.nanmean(v, 0); ci = np.nanstd(v, 0, ddof=1) / np.sqrt(ns) * tdist.ppf(0.975, np.maximum(ns - 1, 1))
            sh = ns >= 5
            ax.plot(tb[sh], mu[sh], color=fc.LOBE_COLORS[lb], lw=1.8, label=f"{lb.capitalize()} (n={len(v)})")
            ax.fill_between(tb[sh], (mu - ci)[sh], (mu + ci)[sh], color=fc.LOBE_COLORS[lb], alpha=0.15, lw=0)
        ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time since the previous word disappeared (ms)"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Word-off window re-aligned to the previous word's offset (responsive electrodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "offset_recovery.png"), dpi=200); plt.close(fig)

    # 5. blank interval and the neighbouring words
    BF = pd.read_csv(join(tdir, "blank_fixed.csv"))
    terms = {"prev_resp": "Previous word's response", "prev2_resp": "Response two words back",
             "prev_resp:blank_len": "Previous word x blank length"}
    co = BF[BF["term"].isin(terms)].copy()
    co["p"] = 2 * norm.sf(np.abs(co["est"] / co["se"]))
    co["q"] = multipletests(co["p"], method="fdr_bh")[1]
    co.to_csv(join(tdir, "blank_coefficients.csv"), index=False)   # mixed-model check of the correlations below
    # all electrodes: sessions -> electrode -> subject x lobe; t vs 0 per lobe (fc.roi_stats); r1 vs r2 paired
    L = E.groupby(["subject", "lobe", "electrode"]).mean(numeric_only=True).groupby(["subject", "lobe"]).mean().reset_index()
    L = L.rename(columns={"subject": "sub"}).assign(roi=lambda d: "B-" + d["lobe"])
    rows = []
    for b in ("alpha", "high_gamma"):
        L[f"dlong_{b}"] = L[f"r1long_{b}"] - L[f"r1short_{b}"]
        L[f"d12_{b}"] = L[f"r1_{b}"] - L[f"r2_{b}"]
        rows += [fc.roi_stats(L, f"{m}_{b}").assign(measure=f"{m}_{b}") for m in ("r1", "r2", "dlong", "d12")]
    cs = pd.concat(rows)
    cs.to_csv(join(tdir, "carryover_stats.csv"), index=False)
    lab = {"r1_high_gamma": "Previous word", "r2_high_gamma": "Two words back",
           "dlong_high_gamma": "Previous word,\nlong − short wait"}
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")   # high gamma
    fc.roi_dot_panel(ax, {v: cs[cs["measure"] == m] for m, v in lab.items()}, "r with the current word's\nhigh-gamma response")
    fig.savefig(join(args.out_dir, "carryover.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n[variance components, % of single-event variance]\n",
          (VC.pivot_table(index=["band", "term"], columns="lobe", values="share") * 100)[lobes].round(2).to_string())
    print("\n[task trends]\n", trend.round(4).to_string(index=False))
    print("\n[prevalence, all electrodes]\n", prev.round(3).to_string())
    print("\n[alpha-hg r] across events, median by lobe:", er.groupby("lobe")["r"].median().round(3).to_dict())
    print("            across electrodes:", acr.groupby("lobe")["r"].median().round(3).to_dict())
    print("            magnitude:", acr.groupby("lobe")["r_abs"].median().round(3).to_dict())
    print("            tests (subjects' mean r vs 0):\n", cst.round(4).to_string(index=False))
    print("\n[blank] coefficients:\n", co[["lobe", "band", "term", "est", "se", "q"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
