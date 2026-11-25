"""plot_timing.py -- high-gamma waveforms and latencies behind the timing claims (proposal Figs. 2B-D, 3A-C, 3E, S2, S3).

Events and units as plot_variance_structure.py (event_mask; power in Cohen's d units: minus the session's mean ISI
power, over the pooled SD). Per responsive electrode (plot_latency.responsive), the average curve of each condition
of each split (SPLITS); mean +/- SEM across subjects per lobe (or subregion). Latency: cross-correlation
(xcorr_latency); a latency difference between two conditions = lag of one condition's average curve against the
other's. From the variance tables: r of power vs latency.

    python plot_timing.py [--from-pkl]
-> figures/timing/*.png, results/timing/*.csv, curves.pkl
"""
from __future__ import annotations

import os
from os.path import join, dirname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import half_max, response, responsive
from plot_variance_structure import event_mask, pooled_sd, power_response, session_events, xcorr_latency

# split -> (conditions, period): events split per session; period "pre" = the ISI window (its end),
# "post" = after word onset, "both" = the two joined
#   isi: session median ISI; isi_q: ISI quartiles (occipital); region: occipital subregion, all events
#   occ_lat: session median of the event's occipital latency (mean over the subject's responsive occipital electrodes)
SPLITS = {"isi": ("both",), "isi_q": ("pre",), "occ_lat": ("post",), "region": ("post",)}
# waveform figures: name, split, lobes (one panel each), conditions (None: the 3 with most subjects; cuneus, 4th with 16, left out), period,
# x shifted by the condition's mean ISI, legend title
WAVES = [("wait_waveform_occipital", "isi", ["occipital"], ("long", "short"), "both", False, "ISI"),     # 3A
         ("offset_next", "isi_q", ["occipital"], ("0", "1", "2", "3"), "pre", False, "ISI"),                    # 3B
         ("offset_prev", "isi_q", ["occipital"], ("0", "1", "2", "3"), "pre", True, "ISI"),                     # S2
         ("propagation_waveforms", "occ_lat", fc.LOBES, ("early", "late"), "post", False, "occipital latency"),  # 2B
         ("subregion_waveforms", "region", ["occipital"], None, "post", False, None)]                    # S3
YLAB = "High-gamma power\n(Cohen's d units)"


def halves(v: np.ndarray) -> dict:
    return {"early": v < np.nanmedian(v), "late": v > np.nanmedian(v)}


def collect(save_root: str) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """One row per electrode x split x condition: average curve, mean power, mean ISI; the electrode's mean latency.
    Also the ISI before every event and each session's median ISI (3E)."""
    lobe_of, rows, isis, medians = fc.load_burke_maps(), [], [], []
    alpha_dir = power_dir(save_root, "word_on", "alpha")
    for f in tqdm(fc.session_files(power_dir(save_root, "word_on", "high_gamma"), "_power.pkl", None), desc="sessions"):
        p = fc.load_pickle(str(f)); events = session_events(p)
        if events is None or not (alpha_dir / f.name).exists():
            continue
        in_task, kept = event_mask(events, p, fc.load_pickle(str(alpha_dir / f.name)))
        usable = in_task[:, None] & kept
        sd = pooled_sd(np.asarray(p["trial_pow_hi"], float), np.asarray(p["trial_pow_lo"], float))
        x, t = response(p); x = np.where(usable[..., None], x / sd[:, None], np.nan)
        x_pre = np.asarray(p["env_lo"], float); x_pre = np.where(usable[..., None], (x_pre - np.nanmean(x_pre, axis=(0, 2))[:, None]) / sd[:, None], np.nan)
        power = np.where(usable, power_response(p), np.nan)
        isi = np.where(in_task, np.asarray(p["blank_ms_hi"], float), np.nan)
        isis.append(isi[np.isfinite(isi)]); medians.append(np.nanmedian(isi))
        lobe = np.array([r.split("-", 1)[1] if r else None for r in fc.roi_of_reg_full(p["reg_full"], lobe_of)], dtype=object)
        resp = responsive(p) & (lobe != None)  # noqa: E711
        latency = np.full(usable.shape, np.nan)
        for c in np.flatnonzero(resp):
            latency[usable[:, c], c] = xcorr_latency(x[usable[:, c], c], t)[0]
        occ = np.flatnonzero(resp & (lobe == "occipital"))
        masks = {"isi": {"long": isi > np.nanmedian(isi), "short": isi < np.nanmedian(isi)}}
        if np.isfinite(isi).sum() >= 40:
            quartile = pd.qcut(isi, 4, labels=False, duplicates="drop")
            masks["isi_q"] = {str(k): quartile == k for k in range(4)}
        if len(occ):
            masks["occ_lat"] = halves(np.nanmean(latency[:, occ], 1))
        for c in np.flatnonzero(resp):
            region = str(p["reg_full"][c]); region = region[2:] if region[:2] in ("L ", "R ") else region
            el_masks = dict(masks, region={region: in_task}) if lobe[c] == "occipital" else masks
            for split, cond_masks in el_masks.items():
                if split in ("isi_q", "region") and lobe[c] != "occipital":
                    continue
                for cond, mask in cond_masks.items():
                    if (mask & usable[:, c]).sum() < 5:
                        continue
                    post = np.nanmean(x[mask, c], 0)
                    curve = {"pre": np.nanmean(x_pre[mask, c], 0), "post": post}
                    curve = {"both": np.r_[curve["pre"], post], **curve}[SPLITS[split][0]]
                    rows.append({"sub": str(p["sid"][0]), "el": f"{p['sid'][0]}|{p['labels'][c]}", "lobe": lobe[c],
                                 "split": split, "cond": cond, "isi": np.nanmean(isi[mask]), "curve": curve,
                                 "power": np.nanmean(power[mask, c]), "el_lat": np.nanmean(latency[:, c])})
    t_pre = p["env_lo_t0_ms"] + p["env_step_ms"] * (np.arange(x_pre.shape[-1]) + 0.5)
    return pd.DataFrame(rows), {"pre": t_pre, "post": t, "both": np.r_[t_pre, t]}, (np.concatenate(isis), np.array(medians))


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
        d, times, (isi, medians) = pd.read_pickle(join(tab_dir, "curves.pkl"))
    else:
        d, times, (isi, medians) = collect(save_root)
        pd.to_pickle((d, times, (isi, medians)), join(tab_dir, "curves.pkl"))
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
                curves, mean_isi = np.stack(lobe_curves.loc[k, "curve"].to_list()), lobe_curves.loc[k, "isi"].mean()
                mu, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(curves))
                x = times[period] + (mean_isi if shift else 0)
                ax.plot(x, mu, color=color, lw=1.6, label=f"{mean_isi:.0f} ms" if split == "isi_q" else f"{k} (n={len(curves)})")
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

    # 2B (candidate): each lobe's average curve, scaled to its peak
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    for lobe in fc.LOBES:
        mean_curve = np.stack(subj_curves.loc[("occ_lat", lobe)].groupby("sub")["curve"].agg(lambda c: np.stack(c.to_list()).mean(0)).to_list())
        mu = mean_curve.mean(0); peak = mu.max()
        se = mean_curve.std(0, ddof=1) / np.sqrt(len(mean_curve))
        ax.plot(times["post"], mu / peak, color=fc.LOBE_COLORS[lobe], lw=1.6, label=f"{lobe.capitalize()} (n={len(mean_curve)})")
        ax.fill_between(times["post"], (mu - se) / peak, (mu + se) / peak, color=fc.LOBE_COLORS[lobe], alpha=0.15, lw=0)
    ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="0.5", lw=0.8, ls="--"); ax.set_xlim(0, 1000)
    ax.set_xlabel("Time from word onset (ms)"); ax.set_ylabel("High-gamma power\n(fraction of the region's peak)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7)
    fig.savefig(join(out, "region_waveforms.png"), dpi=200); plt.close(fig)

    # 2C: lag of each lobe's late vs early average curve (late - early occipital events) against the lobe's mean latency
    shift_stats = fc.roi_stats(contrast(d, "occ_lat", "late", "early", "lat", times["post"]), "lat")
    shift_stats.to_csv(join(tab_dir, "propagation_shift_stats.csv"), index=False)
    lag = shift_stats.assign(lobe=shift_stats["roi"].str[2:]).set_index("lobe")
    latency = (d[d["split"] == "occ_lat"].groupby(["sub", "lobe", "el"])["el_lat"].first().groupby(["sub", "lobe"]).mean()
               .groupby("lobe").agg(["mean", "sem"]))   # electrodes' mean cross-correlation latency over events
    print(latency.round(1).to_string())
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
    for lobe in fc.LOBES:
        ax.errorbar(latency.loc[lobe, "mean"], lag.loc[lobe, "mean"], xerr=latency.loc[lobe, "sem"], yerr=lag.loc[lobe, "sem"],
                    fmt="o", ms=6, color=fc.LOBE_COLORS[lobe], label=f"{lobe.capitalize()} {fc.stars(float(lag.loc[lobe, 'q'])) or ('n.s.' if np.isfinite(lag.loc[lobe, 'q']) else '')}")
    ax.axhline(0, color="0.5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Region's mean latency (ms)"); ax.set_ylabel("Lag of late behind early events (ms)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(join(out, "propagation_shift.png"), dpi=200); plt.close(fig)
    # 3C: long - short ISI, power and latency, per electrode
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), layout="constrained")
    for ax, measure, label, ylabel in ((axes[0], "power", "Power", "High-gamma power (Cohen's d units),\nlong − short ISI"),
                                       (axes[1], "lat", "Latency", "Latency, long − short ISI (ms)")):
        isi_stats = fc.roi_stats(contrast(d, "isi", "long", "short", measure, times["post"]), measure)
        isi_stats.to_csv(join(tab_dir, f"isi_{measure}_stats.csv"), index=False)
        fc.roi_dot_panel(ax, {label: isi_stats}, ylabel); ax.get_legend().remove()
        print(isi_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig.savefig(join(out, "isi_power_latency.png"), dpi=200); plt.close(fig)
    print(shift_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(1).to_string())
    print(d[d["split"] == "occ_lat"].groupby(["split", "cond"])["isi"].mean().round(0).to_string())

    # 3E: ISI before each event; bins of one 60 Hz screen frame (word timing is frame-locked, finer bins alias)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lo, hi = np.percentile(isi, [0.1, 99.9])   # a few pauses of several seconds would stretch the axis
    ax.hist(isi[(isi >= lo) & (isi <= hi)], bins=np.arange(lo, hi + 1000 / 60, 1000 / 60), color="0.6", edgecolor="white", lw=0.3)
    lo95, hi95 = np.percentile(medians, [2.5, 97.5])
    ax.axvspan(lo95, hi95, color="tab:red", alpha=0.25, lw=0, zorder=3, label=f"95% of session\nmedians ({lo95:.0f}–{hi95:.0f} ms)")
    ax.axvline(np.median(isi), color="tab:red", ls="--", lw=1, zorder=4, label=f"overall median ({np.median(isi):.0f} ms)")
    ax.set_xlabel("ISI before the event (ms)", fontsize=12); ax.set_ylabel("Events", fontsize=12)
    ax.legend(frameon=False, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(out, "isi_hist.png"), dpi=300); plt.close(fig)

    # 2D: r of power vs latency across events, per electrode-session
    el_sessions = pd.read_csv(join(dirname(fc.table_path(join(dirname(out), "variance_structure", "x"))), "electrode_sessions.csv"))
    power_lat = (el_sessions.dropna(subset=["r_power_lat"]).groupby(["subject", "lobe", "electrode"])["r_power_lat"].mean()
                 .groupby(["subject", "lobe"]).mean().reset_index().rename(columns={"subject": "sub", "r_power_lat": "r"})
                 .assign(roi=lambda v: "B-" + v["lobe"]))
    power_lat_stats = fc.roi_stats(power_lat, "r"); power_lat_stats.to_csv(join(tab_dir, "power_latency_stats.csv"), index=False)
    print(power_lat_stats[["roi", "n_subjects", "mean", "sem", "q"]].round(3).to_string())
    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")
    fc.roi_dot_panel(ax, {"Event": power_lat_stats}, "r (high-gamma power vs latency)")
    ax.set_ylim(-0.3, 0.3); ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    fig.savefig(join(out, "power_latency.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
