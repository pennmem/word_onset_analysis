"""plot_timing.py -- high-gamma waveforms and latencies behind the timing claims (proposal Figs. 2B-E, 3A-C, S2).

Per responsive electrode (plot_latency.response), the average curve of each condition of each split (SPLITS);
mean +/- SEM across subjects per lobe (or subregion). Latency difference between two conditions = lag of one
condition's average curve against the other's (xcorr_latency, the other as template). From the variance tables:
r of occipital vs each lobe's latency across word events, r of peak vs latency.

    python plot_timing.py [--from-pkl]
-> figures/timing/*.png, results/timing/*.csv, curves.pkl
"""
from __future__ import annotations

import os
from os.path import join, dirname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import half_max, response, responsive
from plot_variance_structure import xcorr_latency

# split -> (conditions, period): word events split per session; period "pre" = the word-off window (ISI end),
# "post" = after word onset, "both" = the two joined
#   isi: session median ISI; isi_q: ISI quartiles (occipital); region: occipital subregion, all words
#   occ_lat: session median of the word's occipital latency (mean over the subject's responsive occipital electrodes)
#   occ_lat_half: the same split from a random half of the occipital electrodes, curves of the other half
SPLITS = {"isi": ("both",), "isi_q": ("pre",), "occ_lat": ("post",), "occ_lat_half": ("post",), "region": ("post",)}
# waveform figures: name, split, lobes (one panel each), conditions (None: the 3 with most subjects; cuneus, 4th with 16, left out), period,
# x shifted by the condition's mean ISI, legend title
WAVES = [("wait_waveform_occipital", "isi", ["occipital"], ("long", "short"), "both", False, "ISI"),     # 3A
         ("offset_next", "isi_q", ["occipital"], ("0", "1", "2", "3"), "pre", False, "ISI"),                    # 3B
         ("offset_prev", "isi_q", ["occipital"], ("0", "1", "2", "3"), "pre", True, "ISI"),                     # S2
         ("propagation_waveforms", "occ_lat", fc.LOBES, ("early", "late"), "post", False, "occipital latency"),  # 2B
         ("occipital_split_half", "occ_lat_half", ["occipital"], ("early", "late"), "post", False,
          "latency of the other\noccipital electrodes"),
         ("subregion_waveforms", "region", ["occipital"], None, "post", False, None)]                    # 2C
YLAB = "High gamma power"


def halves(v: np.ndarray) -> dict:
    return {"early": v < np.nanmedian(v), "late": v > np.nanmedian(v)}


def collect(save_root: str) -> pd.DataFrame:
    """One row per electrode x split x condition: average curve, its latency and peak, mean ISI."""
    lobe_of, rng, rows = fc.load_burke_maps(), np.random.default_rng(0), []
    for f in tqdm(fc.session_files(power_dir(save_root, "word_on", "high_gamma"), "_power.pkl", None), desc="sessions"):
        p = fc.load_pickle(str(f))
        x, t = response(p)
        x_pre = np.asarray(p["env_lo"], float); x_pre = x_pre - np.nanmean(x_pre, axis=(0, 2))[:, None]
        isi = np.asarray(p["blank_ms_hi"], float)
        lobe = np.array([r.split("-", 1)[1] if r else None for r in fc.roi_of_reg_full(p["reg_full"], lobe_of)], dtype=object)
        resp = responsive(p) & (lobe != None)  # noqa: E711
        occ = np.flatnonzero(resp & (lobe == "occipital"))
        masks = {"isi": {"long": isi > np.nanmedian(isi), "short": isi < np.nanmedian(isi)}}
        if np.isfinite(isi).sum() >= 40:
            quartile = pd.qcut(isi, 4, labels=False, duplicates="drop")
            masks["isi_q"] = {str(k): quartile == k for k in range(4)}
        held_out = set()
        if len(occ):
            occ_lat = np.stack([xcorr_latency(x[:, c], t)[0] for c in occ], 1)
            masks["occ_lat"] = halves(np.nanmean(occ_lat, 1))
            if len(occ) >= 2:
                in_half = rng.permutation(len(occ)) < len(occ) // 2
                masks["occ_lat_half"], held_out = halves(np.nanmean(occ_lat[:, in_half], 1)), set(occ[~in_half])
        for c in np.flatnonzero(resp):
            region = str(p["reg_full"][c]); region = region[2:] if region[:2] in ("L ", "R ") else region
            el_masks = dict(masks, region={region: np.ones(len(x), bool)}) if lobe[c] == "occipital" else masks
            for split, cond_masks in el_masks.items():
                if (split in ("isi_q", "region") and lobe[c] != "occipital") or (split == "occ_lat_half" and c not in held_out):
                    continue
                for cond, mask in cond_masks.items():
                    if mask.sum() < 5:
                        continue
                    post = np.nanmean(x[mask, c], 0)
                    curve = {"pre": np.nanmean(x_pre[mask, c], 0), "post": post}
                    curve = {"both": np.r_[curve["pre"], post], **curve}[SPLITS[split][0]]
                    rows.append({"sub": str(p["sid"][0]), "el": f"{p['sid'][0]}|{p['labels'][c]}", "lobe": lobe[c],
                                 "split": split, "cond": cond, "isi": np.nanmean(isi[mask]), "curve": curve,
                                 "peak": half_max(post, t)[1]})
    t_pre = p["env_lo_t0_ms"] + p["env_step_ms"] * (np.arange(x_pre.shape[-1]) + 0.5)
    return pd.DataFrame(rows), {"pre": t_pre, "post": t, "both": np.r_[t_pre, t]}


def contrast(d: pd.DataFrame, split: str, a, b, col: str, t: np.ndarray | None = None) -> pd.DataFrame:
    """a - b per electrode -> mean per subject x lobe (roi_stats table). col "lat": lag of a's average curve against
    b's (b as the template; the electrode's overall average would contain a and pull the lag toward 0)."""
    rows, idx = d[d["split"] == split], ["sub", "lobe", "el"]
    if col == "lat":
        wide = rows.pivot_table(index=idx, columns="cond", values="curve", aggfunc=lambda c: np.stack(list(c)).mean(0)).dropna()
        diff = pd.Series([xcorr_latency(el[a][-len(t):][None], t, tmp=el[b][-len(t):])[0][0] - half_max(el[b][-len(t):], t)[0]
                          for _, el in wide.iterrows()], index=wide.index)
    else:
        wide = rows.pivot_table(index=idx, columns="cond", values=col); diff = wide[a] - wide[b]
    return diff.dropna().groupby(["sub", "lobe"]).mean().reset_index(name=col).assign(roi=lambda df: "B-" + df["lobe"])


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "timing"), **{
        "--from-pkl": dict(action="store_true", help="reuse curves.pkl instead of reading the pickles")})
    out = args.out_dir; os.makedirs(out, exist_ok=True)
    tab_dir = dirname(fc.table_path(join(out, "x")))
    if args.from_pkl:
        d, times = pd.read_pickle(join(tab_dir, "curves.pkl"))
    else:
        d, times = collect(save_root)
        pd.to_pickle((d, times), join(tab_dir, "curves.pkl"))
    unit = np.where(d["split"] == "region", "occipital", d["lobe"])
    subj_curves = d.assign(unit=unit).groupby(["split", "unit", "cond", "sub"]).agg(
        curve=("curve", lambda c: np.stack(c.to_list()).mean(0)), isi=("isi", "mean"))

    for name, split, lobes, conds, period, shift, title in WAVES:
        grid = len(lobes) > 1
        fig, axes = plt.subplots(2 if grid else 1, 3 if grid else 1, figsize=(11, 5.6) if grid else (5.5, 3.8),
                                 sharey=True, layout="constrained", squeeze=False)
        fs = 15 if grid else 12
        for ax, lb in zip(axes.ravel(), lobes):
            lobe_curves = subj_curves.loc[(split, lb)]
            shown = conds or list(lobe_curves.index.get_level_values("cond").value_counts().index[:3])
            colors = (plt.cm.viridis(np.linspace(0.1, 0.85, 4)) if split == "isi_q" else plt.cm.Dark2(range(4))
                    if split == "region" else ("C3", "C0") if split == "isi" else ("C2", "C1"))
            for k, color in zip(shown, colors):
                curves, isi = np.stack(lobe_curves.loc[k, "curve"].to_list()), lobe_curves.loc[k, "isi"].mean()
                mu, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(curves))
                x = times[period] + (isi if shift else 0)
                ax.plot(x, mu, color=color, lw=1.6, label=f"{isi:.0f} ms" if split == "isi_q" else f"{k} (n={len(curves)})")
                ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.2, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            if period != "pre":
                ax.axvline(0, color="0.5", lw=0.8, ls="--")
            if period == "both":
                ax.axvspan(-100, 0, color="w", zorder=3)
            if period != "pre":
                ax.set_xlim(-700 if period == "both" else 0, 1000)
            nsub = lobe_curves.index.get_level_values("sub").nunique()
            ax.set_title(f"{lb.capitalize()} ({nsub} subjects)" if split == "isi_q" else
                         "Occipital subregions" if split == "region" else lb.capitalize(),
                         color=fc.LOBE_COLORS[lb], fontsize=fs + 2)
            ax.tick_params(labelsize=fs - 2); ax.spines[["top", "right"]].set_visible(False)
        xlabel = ("Time since the previous word's offset (ms)" if shift else "Time to the next word's onset (ms)"
              if period == "pre" else "Time from word onset (ms)")
        for ax in axes[-1]:
            ax.set_xlabel(xlabel, fontsize=fs)
        for ax in axes[:, 0]:
            ax.set_ylabel(YLAB, fontsize=fs)
        axes[0, 0].legend(frameon=False, fontsize=fs - 3, title=title, title_fontsize=fs - 3,
                          loc="upper right" if split == "isi_q" else "best")
        fig.savefig(join(out, f"{name}.png"), dpi=200); plt.close(fig)

    # 2D: lag of each lobe's late vs early average curve (late - early occipital words) against the lobe's average latency
    shift_stats = fc.roi_stats(contrast(d, "occ_lat", "late", "early", "lat", times["post"]), "lat")
    shift_stats.to_csv(join(tab_dir, "propagation_shift_stats.csv"), index=False)
    lag = shift_stats.assign(lobe=shift_stats["roi"].str[2:]).set_index("lobe")
    latency = d[d["split"] == "occ_lat"].groupby(["sub", "lobe"])["lat"].mean().groupby("lobe").agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
    for lobe in fc.LOBES:
        ax.errorbar(latency.loc[lobe, "mean"], lag.loc[lobe, "mean"], xerr=latency.loc[lobe, "sem"], yerr=lag.loc[lobe, "sem"],
                    fmt="o", ms=6, color=fc.LOBE_COLORS[lobe], label=lobe.capitalize())
        ax.annotate(fc.stars(float(lag.loc[lobe, "q"])), (latency.loc[lobe, "mean"], lag.loc[lobe, "mean"] + lag.loc[lobe, "sem"]),
                    textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Region's average latency (ms)"); ax.set_ylabel("Lag of late behind early words (ms)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(join(out, "propagation_shift.png"), dpi=200); plt.close(fig)
    # 3C: long - short ISI, response peak and latency, per electrode
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), layout="constrained")
    for ax, measure, label, ylabel in ((axes[0], "peak", "Response peak", f"{YLAB}, long − short ISI"),
                                       (axes[1], "lat", "Latency", "Latency, long − short ISI (ms)")):
        isi_stats = fc.roi_stats(contrast(d, "isi", "long", "short", measure, times["post"]), measure)
        isi_stats.to_csv(join(tab_dir, f"isi_{measure}_stats.csv"), index=False)
        fc.roi_dot_panel(ax, {label: isi_stats}, ylabel); ax.get_legend().remove()
        print(isi_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig.savefig(join(out, "isi_peak_latency.png"), dpi=200); plt.close(fig)
    print(shift_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(1).to_string())
    print(d[d["split"].str.startswith("occ_lat")].groupby(["split", "cond"])["isi"].mean().round(0).to_string())

    # word events (variance tables): r of occipital latency with each lobe's, same event; ceiling: occipital halves
    var_dir = "results/variance_structure/"
    trials = pd.read_csv(var_dir + "trials.csv", usecols=["subject", "electrode", "session", "event", "sp", "lobe", "latency_xc"])
    trials = trials.dropna(subset=["latency_xc"])
    trials["lat"] = trials["latency_xc"] - trials.groupby(["electrode", "session", "sp"])["latency_xc"].transform("mean")
    rng = np.random.default_rng(0)
    half = {el: h for _, sub_trials in trials.groupby("subject") for h, els in enumerate(np.array_split(
        rng.permutation(sub_trials.loc[sub_trials["lobe"] == "occipital", "electrode"].unique()), 2)) for el in els}
    trials["half"] = trials["electrode"].map(half)
    event_lat = trials.groupby(["subject", "session", "event", "lobe"])["lat"].mean().unstack("lobe")
    occ_half_lat = trials[trials["lobe"] == "occipital"].groupby(["subject", "session", "event", "half"])["lat"].mean().unstack("half")
    rows = []
    for sub, sub_lat in event_lat.groupby(level="subject"):
        for lb in [lb for lb in fc.LOBES if lb != "occipital"]:
            if {"occipital", lb} <= set(sub_lat.columns):
                paired = sub_lat[["occipital", lb]].dropna()
                if len(paired) >= 50:
                    rows.append({"subject": sub, "pair": lb, "r": np.corrcoef(paired["occipital"], paired[lb])[0, 1]})
        if sub in occ_half_lat.index.get_level_values(0) and occ_half_lat.shape[1] == 2:
            paired = occ_half_lat.loc[sub].dropna()
            if len(paired) >= 50:
                rows.append({"subject": sub, "pair": "occipital (split half)", "r": np.corrcoef(paired[0], paired[1])[0, 1]})
    corr = pd.DataFrame(rows)
    corr_stats = corr.groupby("pair")["r"].agg(n="size", mean="mean", sem=lambda v: v.std(ddof=1) / np.sqrt(len(v)))
    corr_stats["p"] = [ttest_1samp(corr.loc[corr["pair"] == k, "r"], 0).pvalue for k in corr_stats.index]
    corr_stats["q"] = multipletests(corr_stats["p"], method="fdr_bh")[1]
    order = ["occipital (split half)"] + [lb for lb in fc.LOBES if lb in corr_stats.index and lb != "occipital"]
    corr_stats = corr_stats.reindex(order); corr_stats.to_csv(join(tab_dir, "propagation_stats.csv"))
    print(corr_stats.round(3).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
    for i, k in enumerate(order):
        lb = "occipital" if k.startswith("occipital") else k
        ax.errorbar(i, corr_stats.loc[k, "mean"], yerr=corr_stats.loc[k, "sem"], fmt="o", ms=6, color=fc.LOBE_COLORS[lb], lw=1.2)
        ax.annotate(fc.stars(corr_stats.loc[k, "q"]), (i, corr_stats.loc[k, "mean"] + corr_stats.loc[k, "sem"]), textcoords="offset points",
                    xytext=(0, 2), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(order)), ["Occipital\n(split half)"] + [k.capitalize() for k in order[1:]], fontsize=9)
    ax.set_ylabel("r with occipital latency\n(same word event)"); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(join(out, "propagation.png"), dpi=200); plt.close(fig)
    # 2E: r of peak vs latency across word events, per electrode-session
    el_sessions = pd.read_csv(var_dir + "electrode_sessions.csv")
    amp_lat = (el_sessions.dropna(subset=["r_amp_lat"]).groupby(["subject", "lobe", "electrode"])["r_amp_lat"].mean()
         .groupby(["subject", "lobe"]).mean().reset_index().rename(columns={"subject": "sub", "r_amp_lat": "r"})
         .assign(roi=lambda v: "B-" + v["lobe"]))
    amp_stats = fc.roi_stats(amp_lat, "r"); amp_stats.to_csv(join(tab_dir, "amp_latency_stats.csv"), index=False)
    print(amp_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")
    fc.roi_dot_panel(ax, {"Word event": amp_stats}, "r (high-gamma peak vs latency)")
    ax.set_ylim(-0.3, 0.3); ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(join(out, "amp_latency.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
