"""plot_variability.py -- preliminary results for the variability / event-modulator proposal.

Per-electrode Cohen's d of each session (alpha: multitaper; high gamma: sub-band Hilbert), events and units as
plot_variance_structure.py (mean of the usable events' power responses).
Electrode-pattern stability: per subject with >= 2 sessions, sessions alternately in two
halves; per lobe (>= 5 electrodes) r across electrodes of Cohen's d between the halves.
(Variance components, task effects, serial position: plot_variance_structure.py.)
-> figures/variability/electrode_reliability.png, results tables alongside

    python plot_variability.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from plot_variance_structure import power_response, usable_sessions

BEH = "word_on"
BANDS = {"alpha": "Alpha", "high_gamma": "High gamma"}


def electrode_rows(p: dict, usable: np.ndarray, lobe_of: dict) -> list:
    """Per electrode with a lobe: Cohen's d of one session over its usable events."""
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    lobe = np.array([r.split("-", 1)[1] if r else None for r in roi], dtype=object)
    d = np.nanmean(np.where(usable, power_response(p), np.nan), 0)
    return [{"sub": str(p["sid"][0]), "session": str(p["sid"]), "label": label, "lobe": lb, "d": d_el}
            for label, lb, d_el in zip(p["labels"], lobe, d) if lb]


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "variability"), **{
        "--from-csv": dict(action="store_true", help="reuse the saved electrode tables")})
    os.makedirs(args.out_dir, exist_ok=True)
    table = lambda n: fc.table_path(join(args.out_dir, n))
    lobe_of = fc.load_burke_maps()
    if args.from_csv:
        electrodes = {band: pd.read_csv(table(f"electrodes_{band}.csv")) for band in BANDS}
    else:
        el_rows = {band: [] for band in BANDS}
        for pkls, usable in usable_sessions(save_root):
            for band in BANDS:
                el_rows[band] += electrode_rows(pkls[band], usable, lobe_of)
        electrodes = {band: pd.DataFrame(rows) for band, rows in el_rows.items()}
        for band, band_els in electrodes.items():
            band_els.to_csv(table(f"electrodes_{band}.csv"), index=False)

    # do electrode differences replicate across sessions?
    reliability = []
    for band, band_els in electrodes.items():
        for sub, sub_els in band_els.groupby("sub"):
            sessions = sorted(sub_els["session"].unique())
            if len(sessions) < 2:
                continue
            sub_els = sub_els.assign(half=sub_els["session"].map({s: i % 2 for i, s in enumerate(sessions)}))
            d_halves = sub_els.groupby(["lobe", "label", "half"])["d"].mean().unstack("half").dropna()
            for lb, lobe_d in d_halves.groupby(level="lobe"):
                if len(lobe_d) >= 5:
                    reliability.append({"band": band, "sub": sub, "lobe": lb,
                                        "r": np.corrcoef(lobe_d[0], lobe_d[1])[0, 1], "n": len(lobe_d)})
    reliability = pd.DataFrame(reliability); reliability.to_csv(table("electrode_reliability_per_subject.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(fc.LOBES))
    for k, (band, color) in enumerate(zip(BANDS, ("C0", "C3"))):
        band_rel = reliability[reliability["band"] == band]
        fc.box_dots(ax, [band_rel.loc[band_rel["lobe"] == lb, "r"].dropna() for lb in fc.LOBES], x + (k - 0.5) * 0.38,
                    [color] * len(fc.LOBES), width=0.32)
        ax.plot([], [], "s", color=color, alpha=0.5, ms=9, label=BANDS[band])
    ax.set_ylabel("Electrode test-retest r")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x, [lb.capitalize() for lb in fc.LOBES], rotation=30)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(join(args.out_dir, "electrode_reliability.png"), dpi=200); plt.close(fig)
    rel_stats = reliability.groupby(["band", "lobe"])["r"].agg(["median", "size"]).reset_index()
    rel_stats.to_csv(table("electrode_reliability_stats.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"\n[electrode reliability]\n{rel_stats.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
