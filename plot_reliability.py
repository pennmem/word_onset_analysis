"""plot_reliability.py -- is the word-onset response stable across a subject's sessions?

Per session (compute.py power pickles): per-electrode Cohen's d (word on vs off) for alpha
and high gamma. As riley-thesis compute_retest_reliability.py (its channel-level run):
each subject's sessions split at random into two halves (seed 0), >= MIN_SESS_PER_HALF
sessions per half; per session, d averaged per anatomical region (reg_full), then sessions
averaged within each half; test-retest = Pearson r across the subject's regions between the
halves (>= MIN_FINITE regions); box plots of the subjects' r (whiskers: central 95%),
median [IQR] printed; dotted: the no-coupling
simulation's median (results/simulations/hg_null), when present. Synchrony (alpha ciPLV, high-gamma AEC-c;
word on - word off): the same on hub strength (fc.electrode_sync: each electrode's mean
over all partners, as the synchrony summary figure; riley-thesis hub reliability).
Network replicability per lobe across subjects (hemispheres pooled; raw change, no hub averaging): a lobe's profile = its change
on each connection to the 6 lobes (itself included; lobe pairs with >= MIN_SUBJECTS_PAIR
subjects) x the 9 epochs of plot_epoch_network; N_SPLITS times, two disjoint random halves
of the subjects, the same size for every lobe (half the least-covered lobe's subjects), each
r = Pearson between the halves' mean profiles; box plots of the N_SPLITS r per lobe (the
estimate: their median; whiskers: central 95% across splits).
-> figures/reliability/reliability_word_on.png (+ test_retest_word_on.csv,
   profile_replication_word_on.csv)

    python plot_reliability.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import cond_dir, power_dir
from plot_epoch_network import epoch_table

BEH = "word_on"
BANDS = {"alpha": "Alpha d", "high_gamma": "High-gamma d"}
MIN_SESS_PER_HALF = 2   # riley-thesis defaults
MIN_FINITE = 10
SYNC = {"alpha": ("ciplv", "Alpha ciPLV"), "high_gamma": ("aec_c", "High-gamma AEC-c")}
N_SPLITS = 1000


def retest(d: pd.DataFrame, key: str, cols: list[str], order: pd.DataFrame) -> pd.DataFrame:
    """Per subject and column: Pearson r across `key` (region) between the two halves'
    session means; NaN with < MIN_FINITE finite values."""
    s = d.groupby(["sub", "sess", key], as_index=False)[cols].mean()   # per-session region means
    h = s.merge(order, on=["sub", "sess"]).groupby(["sub", "half", key])[cols].mean()
    rows = []
    for sub, g in h.groupby("sub"):
        r = {"sub": sub}
        for c in cols:
            w = g[c].unstack("half").dropna()
            r[c] = pearsonr(w[False], w[True])[0] if len(w) >= MIN_FINITE else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def profile_replication(etbl: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per lobe and split: Pearson r between two disjoint subject halves' mean connection x
    epoch profiles (hemispheres pooled)."""
    ab = etbl["roi"].str.split("|", expand=True)
    pair = ["|".join(sorted((a.split("-", 1)[1], b.split("-", 1)[1]))) for a, b in zip(ab[0], ab[1])]
    e = etbl.assign(roi=pair).groupby(["sub", "roi", "epoch_ms"])["sync_diff"].mean()
    n = e.reset_index().groupby("roi")["sub"].nunique()
    X = e.unstack(["roi", "epoch_ms"])
    X = X.loc[:, X.columns.get_level_values("roi").isin(n.index[n >= fc.MIN_SUBJECTS_PAIR])]
    prof = {}
    for lobe in fc.LOBES:
        v = X.loc[:, [lobe in p.split("|") for p in X.columns.get_level_values("roi")]].to_numpy(float)
        prof[lobe] = v[np.isfinite(v).any(1)]
    half = min(len(v) for v in prof.values()) // 2
    rows = []
    for lobe, v in prof.items():
        for _ in range(N_SPLITS):
            i = rng.permutation(len(v))
            a, b = np.nanmean(v[i[:half]], 0), np.nanmean(v[i[half:2 * half]], 0)
            ok = np.isfinite(a) & np.isfinite(b)
            rows.append((lobe, half, ok.sum(), pearsonr(a[ok], b[ok])[0]))
    return pd.DataFrame(rows, columns=["lobe", "n_half", "n_features", "r"])


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "reliability"))
    el = []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = {b: fc.load_pickle(str(power_dir(save_root, BEH, b) / f.name)) for b in BANDS}
        sub, exp, n = p["alpha"]["sid"]
        reg = np.asarray(p["alpha"]["reg_full"], str)
        fcm = {b: fc.load_pickle(str(cond_dir(save_root, BEH, "diff", b) / f.name.replace("_power.pkl", "_fc_mats.pkl")))
               for b in SYNC}
        el.append(pd.DataFrame({"sub": sub, "sess": f"{exp}_{n}", "region": reg,
                                **{b: p[b]["cohens_d"] for b in BANDS},
                                **{m: fc.electrode_sync(fcm[b][m]) for b, (m, _) in SYNC.items()}}))
    el = pd.concat(el, ignore_index=True)
    el = el[~el["region"].isin(["None", "nan"])]

    rng = np.random.default_rng(0)
    order = el[["sub", "sess"]].drop_duplicates().sort_values(["sub", "sess"])
    order["half"] = order.groupby("sub")["sess"].transform(lambda x: rng.permutation(len(x)) >= len(x) // 2)
    order = order[order.groupby("sub")["half"].transform(lambda x: min(x.sum(), (~x).sum())) >= MIN_SESS_PER_HALF]
    sync_cols = [m for m, _ in SYNC.values()]
    rt = retest(el, "region", list(BANDS) + sync_cols, order)
    if rt.empty:   # no subject with enough sessions (small simulation runs)
        rt = pd.DataFrame(columns=["sub", *BANDS, *sync_cols], dtype=float)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"reliability_{BEH}")
    rt.to_csv(fc.table_path(join(args.out_dir, f"test_retest_{BEH}.csv")), index=False)
    # per measure: median [IQR] across subjects (as riley-thesis)
    labels = {**BANDS, **dict(SYNC.values())}
    ci = {k: rt[k].quantile([0.25, 0.75]).to_numpy() for k in labels}
    print("[reliability] test-retest r, median [IQR]: " + ", ".join(
        f"{k} {rt[k].median():.2f} [{ci[k][0]:.2f}, {ci[k][1]:.2f}]" for k in labels)
          + f"  (n={len(rt)} subjects)")

    lobe_of, rep = fc.load_burke_maps(), []
    for b, (m, label) in SYNC.items():
        etbl, _ = epoch_table(save_root, b, m, lobe_of, args.n_sessions)
        rep.append(profile_replication(etbl, np.random.default_rng(0)).assign(measure=label))
    rep = pd.concat(rep)
    rep.to_csv(fc.table_path(join(args.out_dir, f"profile_replication_{BEH}.csv")), index=False)
    q = rep.groupby(["measure", "lobe"], sort=False)["r"].agg(
        lambda r: f"{r.median():.2f} [{r.quantile(0.025):.2f}, {r.quantile(0.975):.2f}]")
    print(f"[reliability] network profile replication (median r [central 95% of splits]; {rep['n_half'].iloc[0]} subjects per half):\n"
          + q.unstack("lobe")[fc.LOBES].to_string())

    # null reference: the no-coupling simulation's median per measure (real data only)
    nf = join("results", "simulations", "hg_null", "reliability", f"test_retest_{BEH}.csv")
    null = pd.read_csv(nf) if not getattr(args, "simulation_tag", None) and os.path.exists(nf) \
        and os.path.getsize(nf) > 1 else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax, x = axes[0], np.array([0, 1, 2.4, 3.4])   # power | synchrony
    for xi, k, col in zip(x, labels, ("C0", "C3", "C0", "C3")):
        v = rt[k].dropna().to_numpy()
        if len(v):   # subjects; whiskers: central 95%, as the replicability boxes
            ax.boxplot([v], positions=[xi], widths=0.6, patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                       boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="0.15"))
        if null is not None:
            ax.plot([xi - 0.3, xi + 0.3], [null[k].median()] * 2, color="0.4", ls=":", lw=1.5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, [labels[k] for k in labels], rotation=30)
    ax.set_ylim(top=1)
    ax.set_title("Test-retest across sessions", fontsize=12)
    ax.set_ylabel("r (within subject)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax, x = axes[1], np.arange(len(fc.LOBES))
    for k, ((_, label), col) in enumerate(zip(SYNC.values(), ("C0", "C3"))):
        r = rep[rep["measure"] == label]
        ax.boxplot([r.loc[r["lobe"] == l, "r"] for l in fc.LOBES], positions=x + (k - 0.5) * 0.38,
                   widths=0.32, patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                   boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="0.15"))
        ax.plot([], [], "s", color=col, alpha=0.6, ms=9, label=label)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, [l.capitalize() for l in fc.LOBES], rotation=30)
    ax.set_ylim(top=1)
    ax.set_title("Network replicability by lobe", fontsize=12)
    ax.set_ylabel("r (independent subject samples)", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
