"""Fig. 1C in Fig. 1B's format: occipital variance shares as stacked bars -- group model (1B's occipital row), mean of
the single-subject fits, and the example subject (most electrodes x sessions)."""
import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd

from plot_variance_structure import GROUPS

out = "figures/single_subject"; os.makedirs(out, exist_ok=True)
c = pd.read_csv("results/single_subject/components_single.csv")   # single_subject_variance.R
c["share"] = c["var"] / c.groupby(["subject", "band"])["var"].transform("sum")
info = c.groupby("subject")[["nel", "nses"]].first()
ex = (info["nel"] * info["nses"]).idxmax()
g = pd.read_csv("results/variance_structure/components.csv")
g = g[g["lobe"] == "occipital"].assign(share=lambda d: d["var"] / d.groupby("band")["var"].transform("sum"))
nsub = c["subject"].nunique()
rows = {"Group model\n(Fig. 1B)": lambda b: g[g["band"] == b].set_index("term")["share"],
        f"Single subjects\n(mean of {nsub})": lambda b: c[c["band"] == b].pivot_table(index="subject", columns="term", values="share").fillna(0).mean(),
        f"Example subject\n({info.loc[ex, 'nel']} electrodes,\n{info.loc[ex, 'nses']} sessions)": lambda b: c[(c["band"] == b) & (c["subject"] == ex)].set_index("term")["share"]}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), sharey=True)
for ax, (b, lab) in zip(axes, (("alpha", "Alpha power"), ("high_gamma", "High-gamma power"))):
    for y, (name, f) in enumerate(rows.items()):
        s, left = f(b), 0.0
        for gname, (terms, col) in GROUPS.items():
            w = float(s.reindex(terms).fillna(0).sum())
            ax.barh(y, w, left=left, color=col, height=0.75, edgecolor="w", lw=1.5, label=gname if y == 0 else None)
            if w >= 0.06:
                ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=11,
                        color="k" if col in ("#f2c12e", "#d9d9d9") else "w")
            left += w
    ax.set_yticks(range(len(rows)), list(rows), fontsize=10); ax.set_xlim(0, 1)
    ax.set_title(f"{lab}, occipital", fontsize=13); ax.set_xlabel("Share of variance across word events", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(join(out, "single_subject_shares.png"), dpi=200, bbox_inches="tight")
print(ex, info.loc[ex].to_dict())
