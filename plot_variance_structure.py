"""plot_variance_structure.py -- where the word response varies (preliminary, subset of subjects).

Reads compute.py's power pickles (alpha, high gamma, spectrum) for N_SUBJECTS random subjects.
Per trial and electrode: the response = word-on minus the electrode-session mean word-off
(alpha: log multitaper power, 0-600 ms; high gamma: plot_latency.response, envelope over time).
Outlier trials dropped per electrode (whole-clip peak or window mean > OUTLIER_MAD x MAD).

  1. Variance partition (nested method of moments: subject > electrode > session > trial) per lobe,
     for the high-gamma response over time (50 ms bins), and per band for the window mean;
     over frequency at electrode-session level (Cohen's d per spectrum bin; subject > electrode > session).
  2. Response types: k-means (K_TYPES) on each electrode's spectral profile (Cohen's d, 11 bins
     5-150 Hz) and high-gamma time profile (Cohen's d, 12 bins), each scaled to unit norm.
  3. Alpha vs high gamma, trial by trial: per electrode-session Pearson r across trials, vs the
     same across electrodes (per subject and lobe).
  4. Gain vs timing: per responsive electrode, high-gamma peak ratio and half-max latency shift,
     serial positions 8-12 vs 2-4.
  5. Example: single-trial high gamma of one occipital electrode, trials sorted by serial position.
-> figures/variance_structure/*.png

    python plot_variance_structure.py [--n-subjects N]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import half_max, response, responsive

BEH = "word_on"
N_SUBJECTS = 40
OUTLIER_MAD = 10
BIN_MS = 50
K_TYPES = 5
LEVELS = ["subject", "electrode", "session", "trial"]
LEVEL_COLORS = {"subject": "#3b5b92", "electrode": "#5fa8a0", "session": "#d9a441", "trial": "#cfcfcf"}


def robust_ok(v: np.ndarray) -> np.ndarray:
    """(E, C) True where the trial is not a robust outlier of its electrode."""
    med = np.nanmedian(v, 0); mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    return np.abs(v - med) <= OUTLIER_MAD * mad


def partition(df: pd.DataFrame, col: str, levels=LEVELS) -> dict:
    """Nested method-of-moments variance components of df[col] over the given levels
    (each a column; the last is the replicate). Returns {level: variance}."""
    keys = levels[:-1]
    out, below = {}, 0.0   # below: variance contributed to a mean by lower levels
    # innermost: variance of replicates within the last grouping
    g = df.groupby(keys)[col]
    n_rep = g.size().mean()
    out[levels[-1]] = g.var(ddof=1).mean()
    below = out[levels[-1]] / n_rep
    means = g.mean().reset_index()
    for i in range(len(keys) - 1, 0, -1):   # session within electrode, electrode within subject
        g = means.groupby(keys[:i])[col]
        n = g.size().mean()
        out[keys[i]] = max(g.var(ddof=1).mean() - below, 0.0)
        below = (out[keys[i]] + below) / n
        means = g.mean().reset_index()
    out[keys[0]] = max(means[col].var(ddof=1) - below, 0.0)
    return out


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variance_structure"), **{
        "--n-subjects": dict(type=int, default=N_SUBJECTS, help="random subjects (seed 0)")})
    os.makedirs(args.out_dir, exist_ok=True)
    lobe_of = fc.load_burke_maps()
    files = {b: {f.name: f for f in fc.session_files(power_dir(save_root, BEH, b), "_power.pkl", None)}
             for b in ("alpha", "high_gamma", "spectrum")}
    subs = sorted({n.split("_")[0] for n in files["high_gamma"]})
    pick = set(np.random.default_rng(0).choice(subs, min(args.n_subjects, len(subs)), replace=False))
    names = sorted(n for n in files["high_gamma"] if n.split("_")[0] in pick and n in files["alpha"] and n in files["spectrum"])

    trial_rows, el_rows, curves, carpet = [], [], [], None
    for n in tqdm(names, desc="sessions"):
        g, a, s = (fc.load_pickle(str(files[b][n])) for b in ("high_gamma", "alpha", "spectrum"))
        sub, sess = str(g["sid"][0]), str(g["sid"])
        roi = fc.roi_of_reg_full(g["reg_full"], lobe_of)
        lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
        x, t = response(g)                                     # (E, C, T)
        pk = np.nanmax(np.abs(x), -1)
        ahi, alo = np.log(np.asarray(a["trial_pow_hi"], float)), np.log(np.asarray(a["trial_pow_lo"], float))
        ya = ahi - np.nanmean(alo, 0)                          # alpha response per trial
        ghi, glo = np.log(np.asarray(g["trial_pow_hi"], float)), np.log(np.asarray(g["trial_pow_lo"], float))
        yg = ghi - np.nanmean(glo, 0)
        ok = robust_ok(pk) & robust_ok(yg) & robust_ok(ya)
        x[~ok] = np.nan; ya[~ok] = np.nan; yg[~ok] = np.nan
        w = fc.load_events(fc.dfrow_from_sid(g["sid"]), BEH)
        w = w[np.asarray(w.attrs["mask"], bool)]
        sp = w["serialpos"].to_numpy(float) if len(w) == len(yg) else np.full(len(yg), np.nan)
        edges = np.arange(0, 1000 + 1e-9, BIN_MS)
        xb = np.stack([np.nanmean(x[..., (t >= e0) & (t < e0 + BIN_MS)], -1) for e0 in edges[:-1]], -1)   # (E, C, B)
        resp = responsive(g)
        for c in np.flatnonzero(lobe != None):  # noqa: E711
            lab = str(g["labels"][c])
            e = np.isfinite(yg[:, c])
            d = pd.DataFrame({"subject": sub, "electrode": f"{sub}|{lab}", "session": sess, "trial": np.arange(e.sum()),
                              "lobe": lobe[c], "alpha": ya[e, c], "high_gamma": yg[e, c]})
            for k, e0 in enumerate(edges[:-1]):
                d[f"t{int(e0)}"] = xb[e, c, k]
            trial_rows.append(d)
            ok2 = np.isfinite(ya[:, c]) & np.isfinite(yg[:, c])
            el_rows.append({"subject": sub, "electrode": f"{sub}|{lab}", "session": sess, "lobe": lobe[c],
                            "label": lab, "responsive": bool(resp[c]),
                            "r_trial": np.corrcoef(ya[ok2, c], yg[ok2, c])[0, 1] if ok2.sum() > 20 else np.nan,
                            "d_alpha": a["cohens_d"][c], "d_hg": g["cohens_d"][c],
                            **{f"f{i}": v for i, v in enumerate(s["cohens_d_freq"][c])},
                            **{f"b{i}": v for i, v in enumerate(g["cohens_d_bins"][c])}})
            if resp[c]:
                for grp, (lo_, hi_) in (("early", (2, 4)), ("late", (8, 12))):
                    m = (sp >= lo_) & (sp <= hi_)
                    if m.sum() > 5:
                        curves.append({"electrode": f"{sub}|{lab}", "lobe": lobe[c], "group": grp,
                                       "curve": np.nanmean(x[m, c], 0)})
            if carpet is None and lobe[c] == "occipital" and resp[c] and g["cohens_d"][c] > 1:
                carpet = (x[:, c].copy(), t, sp, f"{sub} {lab}")
    T = pd.concat(trial_rows, ignore_index=True)
    E = pd.DataFrame(el_rows)
    print(f"[variance] {T['subject'].nunique()} subjects, {E['electrode'].nunique()} electrodes, "
          f"{E['session'].nunique()} sessions, {len(T)} electrode-trials")
    tcols = [c for c in T.columns if c[0] == "t" and c[1:].isdigit()]
    tc = np.array([int(c[1:]) for c in tcols]) + BIN_MS / 2

    # 1. variance partition: high gamma over time, both bands (window mean), frequency (electrode level)
    fig, axes = plt.subplots(2, len(fc.LOBES), figsize=(17, 6.5), sharey=True)
    L3 = ["subject", "electrode", "session"]   # session: includes the session mean's sampling noise
    fcen = np.sqrt(np.geomspace(5, 150, 12)[:-1] * np.geomspace(5, 150, 12)[1:])
    for j, lb in enumerate(fc.LOBES):
        d = T[T["lobe"] == lb]
        if d["subject"].nunique() < 5:
            axes[0, j].set_visible(False); axes[1, j].set_visible(False); continue
        # trial noise dominates single trials; the stack shows the trial-averaged response (electrode-session
        # means), the trial share of the single-trial variance is printed in the panel
        m3 = d.groupby(["subject", "electrode", "session"], as_index=False)[tcols].mean()
        sh = np.array([[partition(m3.dropna(subset=[c]), c, L3)[L] for L in L3] for c in tcols])
        sh = sh / sh.sum(1, keepdims=True)
        axes[0, j].stackplot(tc, sh.T, colors=[LEVEL_COLORS[L] for L in L3], labels=L3, lw=0)
        p4 = partition(d.dropna(subset=["high_gamma"]), "high_gamma")
        axes[0, j].text(0.97, 0.03, f"single trials: {p4['trial'] / sum(p4.values()):.0%} trial-to-trial",
                        transform=axes[0, j].transAxes, ha="right", va="bottom", fontsize=7, color="w")
        axes[0, j].set_title(f"{lb.capitalize()} (n={d['subject'].nunique()})", color=fc.LOBE_COLORS[lb])
        axes[0, j].set_xlabel("Time after word onset (ms)")
        e = E[E["lobe"] == lb]
        fsh = np.array([[partition(e.dropna(subset=[f"f{i}"]), f"f{i}", L3)[L] for L in L3] for i in range(len(fcen))])
        fsh = fsh / fsh.sum(1, keepdims=True)
        axes[1, j].stackplot(fcen, fsh.T, colors=[LEVEL_COLORS[L] for L in L3], lw=0)
        axes[1, j].set_xscale("log"); axes[1, j].set_xticks([5, 10, 30, 100], ["5", "10", "30", "100"])
        axes[1, j].set_xlabel("Frequency (Hz)")
        for ax in axes[:, j]:
            ax.spines[["top", "right"]].set_visible(False); ax.margins(x=0)
    axes[0, 0].set_ylabel("Share of variance\nhigh-gamma response over time")
    axes[1, 0].set_ylabel("Share of variance\nresponse per frequency (Cohen's d)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(join(args.out_dir, "variance_partition.png"), dpi=200); plt.close(fig)
    band = pd.DataFrame([{"lobe": lb, "band": b, **partition(T[T["lobe"] == lb].dropna(subset=[b]), b)}
                         for lb in fc.LOBES for b in ("alpha", "high_gamma") if (T["lobe"] == lb).any()])
    band.to_csv(join(args.out_dir, "variance_partition_bands.csv"), index=False)

    # 2. response types
    el = E.groupby(["electrode", "lobe"], as_index=False)[[c for c in E.columns if c[0] in "fb" and c[1:].isdigit()]].mean().dropna()
    fcols, bcols = [f"f{i}" for i in range(11)], [f"b{i}" for i in range(12)]
    F = np.c_[el[fcols].to_numpy() / np.linalg.norm(el[fcols].to_numpy(), axis=1, keepdims=True),
              el[bcols].to_numpy() / np.linalg.norm(el[bcols].to_numpy(), axis=1, keepdims=True)]
    mag = np.abs(el[fcols]).mean(1).to_numpy()
    act = mag > np.quantile(mag, 0.5)   # the more responsive half; the rest: "weak"
    km = KMeans(K_TYPES, n_init=20, random_state=0).fit(F[act])
    el["type"] = "weak"
    el.loc[act, "type"] = [f"T{k + 1}" for k in km.labels_]
    order = el.loc[act, "type"].value_counts().index.tolist()
    # names read off the centroids of this run (seed 0, N_SUBJECTS); re-read them if the subset changes
    NAMES = {"T5": "Broadband up", "T3": "Alpha/beta down,\nlate high gamma up", "T2": "Broadband down",
             "T4": "Low up, high down", "T1": "Alpha/beta up"}
    el["type"] = el["type"].map(lambda v: NAMES.get(v, v)); order = [NAMES.get(v, v) for v in order]
    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(2, K_TYPES + 1, width_ratios=[1] * K_TYPES + [1.6])
    tbins = np.asarray(fc.load_pickle(str(files["high_gamma"][names[0]]))["bin_centers_ms"])
    for k, ty in enumerate(order):
        m = (el["type"] == ty).to_numpy()
        a1, a2 = fig.add_subplot(gs[0, k]), fig.add_subplot(gs[1, k])
        for ax, cols, xx in ((a1, fcols, fcen), (a2, bcols, tbins)):
            v = el.loc[m, cols].to_numpy()
            ax.plot(xx, np.median(v, 0), color="k", lw=2)
            ax.fill_between(xx, *np.percentile(v, [25, 75], 0), color="0.7", alpha=0.6, lw=0)
            ax.axhline(0, color="0.5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        a1.set_xscale("log"); a1.set_title(f"{ty}\n({m.sum()} electrodes)", fontsize=9); a1.set_xlabel("Frequency (Hz)")
        a2.set_xlabel("High gamma, time (ms)")
        if k == 0:
            a1.set_ylabel("Cohen's d"); a2.set_ylabel("Cohen's d")
    ax = fig.add_subplot(gs[:, -1])
    comp = pd.crosstab(el["lobe"], el["type"], normalize="index").reindex(index=fc.LOBES, columns=order + ["weak"])
    left = np.zeros(len(comp))
    cmap = plt.get_cmap("tab10")
    for k, ty in enumerate(order + ["weak"]):
        ax.barh(range(len(comp)), comp[ty], left=left, color=cmap(k) if ty != "weak" else "0.85", label=ty, height=0.7)
        left += comp[ty].fillna(0).to_numpy()
    ax.set_yticks(range(len(comp)), [l.capitalize() for l in comp.index]); ax.invert_yaxis()
    ax.set_xlabel("Fraction of electrodes"); ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "response_types.png"), dpi=200); plt.close(fig)

    # 3. alpha vs high gamma: trial by trial vs across electrodes
    er = E.groupby(["electrode", "lobe"], as_index=False)["r_trial"].mean()
    ae = E.groupby(["subject", "lobe", "electrode"], as_index=False)[["d_alpha", "d_hg"]].mean()
    acr = pd.DataFrame([{"lobe": lb, "r": np.corrcoef(g["d_alpha"], g["d_hg"])[0, 1]}
                        for (s_, lb), g in ae.groupby(["subject", "lobe"]) if len(g) >= 5])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x0 = np.arange(len(fc.LOBES))
    for k, (dd, lab, col) in enumerate(((er.rename(columns={"r_trial": "r"}), "Trial by trial (per electrode)", "C3"),
                                        (acr, "Across electrodes (per subject)", "0.5"))):
        ax.boxplot([dd.loc[dd["lobe"] == l, "r"].dropna() for l in fc.LOBES], positions=x0 + (k - 0.5) * 0.38,
                   widths=0.32, patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                   boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="0.15"))
        ax.plot([], [], "s", color=col, alpha=0.6, ms=9, label=lab)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x0, [l.capitalize() for l in fc.LOBES], rotation=30)
    ax.set_ylabel("r, alpha vs high-gamma response"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "alpha_hg_coupling.png"), dpi=200); plt.close(fig)

    # 4. gain vs timing, serial positions 8-12 vs 2-4
    C = pd.DataFrame(curves)
    rows = []
    for (e_, lb), g in C.groupby(["electrode", "lobe"]):
        if set(g["group"]) == {"early", "late"}:
            cv = {r.group: np.nanmean(np.stack(g.loc[g["group"] == r.group, "curve"]), 0) for r in g.itertuples()}
            le, pe = half_max(cv["early"][None], t); ll, pl = half_max(cv["late"][None], t)
            if pe[0] > 0 and pl[0] > 0:
                rows.append({"lobe": lb, "dgain": np.log2(pl[0] / pe[0]), "dlat": ll[0] - le[0]})
    G = pd.DataFrame(rows); G.to_csv(join(args.out_dir, "gain_timing.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lb in fc.LOBES:
        g = G[G["lobe"] == lb]
        ax.scatter(g["dlat"], g["dgain"], s=8, alpha=0.5, color=fc.LOBE_COLORS[lb], label=lb.capitalize(), lw=0)
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlim(np.nanpercentile(G["dlat"], [2, 98])); ax.set_ylim(np.nanpercentile(G["dgain"], [2, 98]))
    ax.set_xlabel("Latency shift, positions 8-12 minus 2-4 (ms)"); ax.set_ylabel("Peak ratio, log2 (late / early)")
    ax.legend(frameon=False, fontsize=7, markerscale=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "gain_timing.png"), dpi=200); plt.close(fig)

    # 5. example carpet
    if carpet is not None:
        xx, tt, sp, name = carpet
        o = np.argsort(sp, kind="stable"); keep = np.isfinite(xx[o]).all(1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 4.5), gridspec_kw={"width_ratios": [6, 1]}, sharey=True)
        sh = (tt >= -100) & (tt < 1000)
        a1.imshow(xx[o][keep][:, sh], aspect="auto", cmap="magma", extent=[tt[sh][0], tt[sh][-1], keep.sum(), 0],
                  vmin=0, vmax=np.nanpercentile(xx[o][keep][:, sh], 98))
        a1.set_xlabel("Time after word onset (ms)"); a1.set_ylabel("Trials (sorted by serial position)")
        a1.set_title(f"Single-trial high gamma, {name}", fontsize=9)
        a2.plot(sp[o][keep], np.arange(keep.sum()), color="k"); a2.set_xlabel("Serial position")
        fig.tight_layout(); fig.savefig(join(args.out_dir, "carpet_example.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n[variance shares, window mean]\n", band.assign(**{L: band[L] / band[LEVELS].sum(1) for L in LEVELS}).round(3).to_string(index=False))
    print("\n[response types per lobe]\n", comp.round(2).to_string())
    print("\n[alpha-hg r] trial median by lobe:", er.groupby("lobe")["r_trial"].median().round(3).to_dict())
    print("            across electrodes:", acr.groupby("lobe")["r"].median().round(3).to_dict())
    print("\n[gain-timing] median dgain, dlat by lobe:\n", G.groupby("lobe")[["dgain", "dlat"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
