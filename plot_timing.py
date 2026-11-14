"""plot_timing.py -- high-gamma waveforms and latencies behind the timing claims (proposal Figs. 2B-E, 3A-C, S2).

One pass over the high-gamma pickles (responsive electrodes, plot_latency.response): per electrode, the average
curve of each condition of each split (SPLITS below); per subject x lobe (or subregion) -> mean +/- SEM across
subjects. Latency difference between two conditions = lag of one condition's average curve against the other's
(xcorr_latency with the other as template, per electrode).
Waveform figures are rows of WAVES (split, lobes, period). From the variance tables (trials.csv,
electrode_sessions.csv): r of occipital vs each lobe's latency across word events, r of peak vs latency.

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
        lo = np.asarray(p["env_lo"], float); lo = lo - np.nanmean(lo, axis=(0, 2))[:, None]
        isi = np.asarray(p["blank_ms_hi"], float)
        lobe = np.array([r.split("-", 1)[1] if r else None for r in fc.roi_of_reg_full(p["reg_full"], lobe_of)], dtype=object)
        resp = responsive(p) & (lobe != None)  # noqa: E711
        occ = np.flatnonzero(resp & (lobe == "occipital"))
        sp = {"isi": {"long": isi > np.nanmedian(isi), "short": isi < np.nanmedian(isi)}}
        if np.isfinite(isi).sum() >= 40:
            q = pd.qcut(isi, 4, labels=False, duplicates="drop")
            sp["isi_q"] = {str(k): q == k for k in range(4)}
        held = set()
        if len(occ):
            lat = np.stack([xcorr_latency(x[:, c], t)[0] for c in occ], 1)
            sp["occ_lat"] = halves(np.nanmean(lat, 1))
            if len(occ) >= 2:
                h = rng.permutation(len(occ)) < len(occ) // 2
                sp["occ_lat_half"], held = halves(np.nanmean(lat[:, h], 1)), set(occ[~h])
        for c in np.flatnonzero(resp):
            reg = str(p["reg_full"][c]); reg = reg[2:] if reg[:2] in ("L ", "R ") else reg
            conds = dict(sp, region={reg: np.ones(len(x), bool)}) if lobe[c] == "occipital" else sp
            for s, cs in conds.items():
                if (s in ("isi_q", "region") and lobe[c] != "occipital") or (s == "occ_lat_half" and c not in held):
                    continue
                for k, m in cs.items():
                    if m.sum() < 5:
                        continue
                    post = np.nanmean(x[m, c], 0)
                    curve = {"pre": np.nanmean(lo[m, c], 0), "post": post}
                    curve = {"both": np.r_[curve["pre"], post], **curve}[SPLITS[s][0]]
                    rows.append({"sub": str(p["sid"][0]), "el": f"{p['sid'][0]}|{p['labels'][c]}", "lobe": lobe[c],
                                 "split": s, "cond": k, "isi": np.nanmean(isi[m]), "curve": curve,
                                 "peak": half_max(post, t)[1]})
    tl = p["env_lo_t0_ms"] + p["env_step_ms"] * (np.arange(lo.shape[-1]) + 0.5)
    return pd.DataFrame(rows), {"pre": tl, "post": t, "both": np.r_[tl, t]}


def contrast(d: pd.DataFrame, split: str, a, b, col: str, t: np.ndarray | None = None) -> pd.DataFrame:
    """a - b per electrode -> mean per subject x lobe (roi_stats table). col "lat": lag of a's average curve against
    b's (b as the template; the electrode's overall average would contain a and pull the lag toward 0)."""
    s, idx = d[d["split"] == split], ["sub", "lobe", "el"]
    if col == "lat":
        w = s.pivot_table(index=idx, columns="cond", values="curve", aggfunc=lambda c: np.stack(list(c)).mean(0)).dropna()
        v = pd.Series([xcorr_latency(r[a][-len(t):][None], t, tmp=r[b][-len(t):])[0][0] - half_max(r[b][-len(t):], t)[0]
                       for _, r in w.iterrows()], index=w.index)
    else:
        w = s.pivot_table(index=idx, columns="cond", values=col); v = w[a] - w[b]
    return v.dropna().groupby(["sub", "lobe"]).mean().reset_index(name=col).assign(roi=lambda v: "B-" + v["lobe"])


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "timing"), **{
        "--from-pkl": dict(action="store_true", help="reuse curves.pkl instead of reading the pickles")})
    out = args.out_dir; os.makedirs(out, exist_ok=True)
    tab = dirname(fc.table_path(join(out, "x")))
    if args.from_pkl:
        d, T = pd.read_pickle(join(tab, "curves.pkl"))
    else:
        d, T = collect(save_root)
        pd.to_pickle((d, T), join(tab, "curves.pkl"))
    # subject means: electrodes and sessions averaged
    unit = np.where(d["split"] == "region", "occipital", d["lobe"])
    g = d.assign(unit=unit).groupby(["split", "unit", "cond", "sub"]).agg(
        curve=("curve", lambda c: np.stack(c.to_list()).mean(0)), isi=("isi", "mean"))

    for name, split, lobes, conds, period, shift, title in WAVES:
        grid = len(lobes) > 1
        fig, axes = plt.subplots(2 if grid else 1, 3 if grid else 1, figsize=(11, 5.6) if grid else (5.5, 3.8),
                                 sharey=True, layout="constrained", squeeze=False)
        fs = 15 if grid else 12
        for ax, lb in zip(axes.ravel(), lobes):
            gs = g.loc[(split, lb)]
            cs = conds or list(gs.index.get_level_values("cond").value_counts().index[:3])
            cols = (plt.cm.viridis(np.linspace(0.1, 0.85, 4)) if split == "isi_q" else plt.cm.Dark2(range(4))
                    if split == "region" else ("C3", "C0") if split == "isi" else ("C2", "C1"))
            for k, col in zip(cs, cols):
                v, isi = np.stack(gs.loc[k, "curve"].to_list()), gs.loc[k, "isi"].mean()
                mu, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
                x = T[period] + (isi if shift else 0)
                ax.plot(x, mu, color=col, lw=1.6, label=f"{isi:.0f} ms" if split == "isi_q" else f"{k} (n={len(v)})")
                ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            if period != "pre":
                ax.axvline(0, color="0.5", lw=0.8, ls="--")
            if period == "both":
                ax.axvspan(-100, 0, color="w", zorder=3)
            if period != "pre":
                ax.set_xlim(-700 if period == "both" else 0, 1000)
            nsub = gs.index.get_level_values("sub").nunique()
            ax.set_title(f"{lb.capitalize()} ({nsub} subjects)" if split == "isi_q" else
                         "Occipital subregions" if split == "region" else lb.capitalize(),
                         color=fc.LOBE_COLORS[lb], fontsize=fs + 2)
            ax.tick_params(labelsize=fs - 2); ax.spines[["top", "right"]].set_visible(False)
        xl = ("Time since the previous word's offset (ms)" if shift else "Time to the next word's onset (ms)"
              if period == "pre" else "Time from word onset (ms)")
        for ax in axes[-1]:
            ax.set_xlabel(xl, fontsize=fs)
        for ax in axes[:, 0]:
            ax.set_ylabel(YLAB, fontsize=fs)
        axes[0, 0].legend(frameon=False, fontsize=fs - 3, title=title, title_fontsize=fs - 3,
                          loc="upper right" if split == "isi_q" else "best")
        fig.savefig(join(out, f"{name}.png"), dpi=200); plt.close(fig)

    # 2D: lag of each lobe's late vs early average curve (late - early occipital words), per electrode
    ss = fc.roi_stats(contrast(d, "occ_lat", "late", "early", "lat", T["post"]), "lat")
    ss.to_csv(join(tab, "propagation_shift_stats.csv"), index=False)
    o, oc = ss.set_index("roi").loc["B-occipital"], fc.LOBE_COLORS["occipital"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
    ax.axhspan(o["mean"] - o["sem"], o["mean"] + o["sem"], color=oc, alpha=0.15, lw=0)
    ax.axhline(o["mean"], color=oc, ls="--", lw=1.2)
    ax.annotate(f"occipital delay (split), {o['mean']:.0f} ms", (0.98, o["mean"]), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4), ha="right", fontsize=8, color=oc)
    fc.roi_dot_panel(ax, {"Late $-$ early occipital words": ss[ss["roi"] != "B-occipital"]},
                     "Latency shift (ms)", lobes=[lb for lb in fc.LOBES if lb != "occipital"])
    fig.savefig(join(out, "propagation_shift.png"), dpi=200); plt.close(fig)
    # 3C: long - short ISI, response peak and latency, per electrode
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), layout="constrained")
    for ax, col, lab, yl in ((axes[0], "peak", "Response peak", f"{YLAB}, long − short ISI"),
                             (axes[1], "lat", "Latency", "Latency, long − short ISI (ms)")):
        st = fc.roi_stats(contrast(d, "isi", "long", "short", col, T["post"]), col)
        st.to_csv(join(tab, f"isi_{col}_stats.csv"), index=False)
        fc.roi_dot_panel(ax, {lab: st}, yl); ax.get_legend().remove()
        print(st[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig.savefig(join(out, "isi_peak_latency.png"), dpi=200); plt.close(fig)
    print(ss[["roi", "n_subjects", "mean", "sem", "q"]].round(1).to_string())
    print(d[d["split"].str.startswith("occ_lat")].groupby(["split", "cond"])["isi"].mean().round(0).to_string())

    # word events (variance tables): r of occipital latency with each lobe's, same event; ceiling: occipital halves
    R = "results/variance_structure/"
    t = pd.read_csv(R + "trials.csv", usecols=["subject", "electrode", "session", "event", "sp", "lobe", "latency_xc"])
    t = t.dropna(subset=["latency_xc"])
    t["lat"] = t["latency_xc"] - t.groupby(["electrode", "session", "sp"])["latency_xc"].transform("mean")
    rng = np.random.default_rng(0)
    half = {e: h for s_, v in t.groupby("subject") for h, es in enumerate(np.array_split(
        rng.permutation(v.loc[v["lobe"] == "occipital", "electrode"].unique()), 2)) for e in es}
    t["half"] = t["electrode"].map(half)
    ev = t.groupby(["subject", "session", "event", "lobe"])["lat"].mean().unstack("lobe")
    oh = t[t["lobe"] == "occipital"].groupby(["subject", "session", "event", "half"])["lat"].mean().unstack("half")
    rows = []
    for s_, v in ev.groupby(level="subject"):
        for lb in [lb for lb in fc.LOBES if lb != "occipital"]:
            if {"occipital", lb} <= set(v.columns):
                ok = v[["occipital", lb]].dropna()
                if len(ok) >= 50:
                    rows.append({"subject": s_, "pair": lb, "r": np.corrcoef(ok["occipital"], ok[lb])[0, 1]})
        if s_ in oh.index.get_level_values(0) and oh.shape[1] == 2:
            ok = oh.loc[s_].dropna()
            if len(ok) >= 50:
                rows.append({"subject": s_, "pair": "occipital (split half)", "r": np.corrcoef(ok[0], ok[1])[0, 1]})
    r = pd.DataFrame(rows)
    st = r.groupby("pair")["r"].agg(n="size", mean="mean", sem=lambda v: v.std(ddof=1) / np.sqrt(len(v)))
    st["p"] = [ttest_1samp(r.loc[r["pair"] == k, "r"], 0).pvalue for k in st.index]
    st["q"] = multipletests(st["p"], method="fdr_bh")[1]
    order = ["occipital (split half)"] + [lb for lb in fc.LOBES if lb in st.index and lb != "occipital"]
    st = st.reindex(order); st.to_csv(join(tab, "propagation_stats.csv"))
    print(st.round(3).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
    for i, k in enumerate(order):
        lb = "occipital" if k.startswith("occipital") else k
        ax.errorbar(i, st.loc[k, "mean"], yerr=st.loc[k, "sem"], fmt="o", ms=6, color=fc.LOBE_COLORS[lb], lw=1.2)
        ax.annotate(fc.stars(st.loc[k, "q"]), (i, st.loc[k, "mean"] + st.loc[k, "sem"]), textcoords="offset points",
                    xytext=(0, 2), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(order)), ["Occipital\n(split half)"] + [k.capitalize() for k in order[1:]], fontsize=9)
    ax.set_ylabel("r with occipital latency\n(same word event)"); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(join(out, "propagation.png"), dpi=200); plt.close(fig)
    # 2E: r of peak vs latency across word events, per electrode-session
    E = pd.read_csv(R + "electrode_sessions.csv")
    a = (E.dropna(subset=["r_amp_lat"]).groupby(["subject", "lobe", "electrode"])["r_amp_lat"].mean()
         .groupby(["subject", "lobe"]).mean().reset_index().rename(columns={"subject": "sub", "r_amp_lat": "r"})
         .assign(roi=lambda v: "B-" + v["lobe"]))
    sa = fc.roi_stats(a, "r"); sa.to_csv(join(tab, "amp_latency_stats.csv"), index=False)
    print(sa[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")
    fc.roi_dot_panel(ax, {"Word event": sa}, "r (high-gamma peak vs latency)")
    ax.set_ylim(-0.3, 0.3); ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(join(out, "amp_latency.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
