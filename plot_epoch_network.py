"""plot_epoch_network.py -- time-resolved synchrony network (Rao et al. 2025 hubs,
Solomon et al. 2017 style), alpha ciPLV and high-gamma AEC-c rows, 200 ms epochs.

Per subject and region pair, the mean per-epoch change (epoch minus the mean of
three word-off epochs; compute.py) over the pair's electrode pairs. Hubs (Rao et al.
2025): a region's mean change over its connections (pairs with >= MIN_SUBJECTS_PAIR
subjects), one-sample t vs 0 across subjects, Holm over regions x epochs (each hub x
epoch is reported as its own result); edges: each hub's top-5 connections
(fc.epoch_network_figure). 12 ROIs, and the fine anatomical regions within them.
Supplement (12 ROIs): relative hubs, a region's mean change minus that over all
connections not involving it (stands out from the network-wide change), as a
ROI x epoch t map, Holm.
-> figures/burke_roi_synchrony/roi_synchrony_epochs_(fine_)word_on.png
   (+ roi_synchrony_epochs_word_on_alpha.png: alpha only, two rows;
    roi_synchrony_epochs_word_on_relative.png: relative hubs)

    python plot_epoch_network.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from compute import cond_dir

BEH, BANDS = "word_on", ("alpha", "high_gamma")


def epoch_table(save_root: str, band: str, metric: str, lobe_of: dict, n_sessions=None, fine: bool = False):
    """(per subject x region pair 'a|b' x epoch start mean change, region centroids
    = mean MNI of their electrodes, with lobe). Regions: 12 ROIs or (fine) reg_full labels."""
    rows, xyz_rows = [], []
    for f in fc.session_files(cond_dir(save_root, BEH, "diff", band), "_fc_mats.pkl", n_sessions):
        mat = fc.load_pickle(str(f))
        E = mat.get(f"{metric}_epochs")
        if E is None:
            continue
        xyz, _ = fc.pair_xyz_lead(fc.dfrow_from_sid(mat["sid"]))
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of)
        lobe = [r and r.split("-", 1)[1] for r in roi]
        if fine:
            roi = [g if r is not None else None for g, r in zip(mat["reg_full"], roi)]
        a, b = np.triu_indices(len(roi), 1)
        ok = np.array([roi[i] is not None and roi[j] is not None for i, j in zip(a, b)], bool)
        a, b = a[ok], b[ok]
        d = pd.DataFrame(E[:, a, b].T, columns=[w[0] for w in fc.EPOCHS])
        d["roi"] = ["|".join(sorted((roi[i], roi[j]))) for i, j in zip(a, b)]
        d = d.groupby("roi").mean().stack().rename("sync_diff").reset_index()
        rows.append(d.rename(columns={"level_1": "epoch_ms"}).assign(sub=str(mat["sid"][0])))
        xyz_rows += [(r, lb, *c) for r, lb, c in zip(roi, lobe, xyz) if r is not None and np.isfinite(c).all()]
    tbl = pd.concat(rows).groupby(["sub", "roi", "epoch_ms"], as_index=False)["sync_diff"].mean()
    cen = (pd.DataFrame(xyz_rows, columns=["roi", "lobe", "x", "y", "z"])
             .groupby("roi").agg(lobe=("lobe", "first"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean")))
    return tbl, cen


def hubs(etbl: pd.DataFrame, relative: bool = False) -> pd.DataFrame:
    """Per subject x epoch, a region's mean change over its connections (pairs with >=
    MIN_SUBJECTS_PAIR subjects); relative: minus the mean change over all connections not
    involving it. t vs 0 across subjects, Holm over regions x epochs."""
    n_pair = etbl.groupby("roi")["sub"].nunique()
    htbl = etbl[etbl["roi"].isin(n_pair.index[n_pair >= fc.MIN_SUBJECTS_PAIR])]
    ab = htbl["roi"].str.split("|", expand=True)
    rel = []
    for r in sorted(set(ab[0]) | set(ab[1])):
        inv = (ab[0] == r) | (ab[1] == r)
        d = htbl[inv].groupby(["sub", "epoch_ms"])["sync_diff"].mean()
        if relative:
            d = d - htbl[~inv].groupby(["sub", "epoch_ms"])["sync_diff"].mean()
        rel.append(d.dropna().reset_index().assign(roi=r))
    return fc.bin_stats(pd.concat(rel), "epoch_ms", "sync_diff", fdr="holm")


def network_csvs(save_root, band, metric, path, lobe_of, n_sessions, fine, tag) -> pd.DataFrame:
    """Write <path>_stats / _centroids / _hubs.csv for one row of the network figure;
    returns the relative hubs (written to <path>_hubs_relative.csv) for the 12 ROIs."""
    etbl, cen = epoch_table(save_root, band, metric, lobe_of, n_sessions, fine)
    fc.bin_stats(etbl, "epoch_ms", "sync_diff", min_subjects=fc.MIN_SUBJECTS_PAIR).to_csv(
        fc.table_path(f"{path}_stats.csv"), index=False)
    cen.to_csv(fc.table_path(f"{path}_centroids.csv"))
    hub = hubs(etbl)
    hub.to_csv(fc.table_path(f"{path}_hubs.csv"), index=False)
    print(f"[epochs {tag}] {int(hub['t'].notna().sum() / len(fc.EPOCHS))} regions tested, "
          f"{int((hub['q'] < 0.05).sum())} hub x epoch cells at Holm p < .05")
    if not fine:
        rel = hubs(etbl, relative=True)
        rel.to_csv(fc.table_path(f"{path}_hubs_relative.csv"), index=False)
        print(f"[epochs {tag} relative] {int((rel['q'] < 0.05).sum())} cells at Holm p < .05")
        return rel


def relative_figure(rel: dict, stem: str) -> None:
    """Supplement: relative hubs, ROI x epoch t per band (one shared scale), Holm cells outlined."""
    import matplotlib.pyplot as plt
    from plot_spectrum import draw
    rows = [f"{h}-{lobe}" for lobe in fc.LOBES for h in "LR"]
    starts = [s for s, _ in fc.EPOCHS]
    mats = {b: tuple(st.pivot(index="roi", columns="epoch_ms", values=c).reindex(index=rows, columns=starts)
                     .to_numpy(float) for c in ("t", "q")) for b, st in rel.items()}
    v = max(float(np.nanmax(np.abs(M))) for M, _ in mats.values()) or 1.0
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 5.2), sharey=True)
    for ax, (band, (M, q)) in zip(np.atleast_1d(axes), mats.items()):
        im = draw(ax, M, q, [fc.pretty_roi(x) for x in rows], [r.split("-", 1)[1] for r in rows],
                  [f"{s:.0f}" if k % 2 == 0 else "" for k, s in enumerate(starts)], v=v)
        ax.set_title(f"{fc.band_label(band)} {fc.METRIC_LABELS[fc.RUN_METRIC[band]]}", fontsize=13)
        ax.set_xlabel("Epoch start (ms)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.8).set_label(f"t, region − rest of network ({fc.contrast(BEH)['delta_label']})", fontsize=12)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.png")


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_synchrony"))
    lobe_of = fc.load_burke_maps()
    c = fc.contrast(BEH)
    os.makedirs(args.out_dir, exist_ok=True)
    for level, fine in (("", False), ("fine_", True)):
        stem = f"roi_synchrony_epochs_{level}{BEH}_{{band}}_{{metric}}"
        rel = {}
        for band in BANDS:
            metric = fc.RUN_METRIC[band]
            rel[band] = network_csvs(save_root, band, metric, join(args.out_dir, stem.format(band=band, metric=metric)),
                                     lobe_of, args.n_sessions, fine, f"{level or 'roi_'}{band}")
        if not fine:
            relative_figure(rel, join(args.out_dir, f"roi_synchrony_epochs_{BEH}_relative"))
        fc.epoch_network_figure(args.out_dir, stem, f"{{metric}} ({c['delta_label']})")
        if not fine:   # paper: alpha only, windows in two rows
            fc.epoch_network_figure(args.out_dir, stem, f"{{metric}} ({c['delta_label']})",
                                    only=("alpha",), ncols=5, tag="_alpha")


if __name__ == "__main__":
    main()
