"""plot_study_figure.py -- Figure 1 B/C: electrode coverage and an example high-gamma response.

  coverage  every analyzed bipolar pair (subject x label, deduplicated over sessions) at
            its MNI centroid on a glass brain, coloured by Burke lobe (hemispheres pooled);
            pairs with no lobe in grey. Legend gives subjects / electrodes per lobe (all
            pairs); pairs > MAX_OUTSIDE_MM outside the MNI152 brain mask (failed MNI
            registration; labels are native-space, so analyses are unaffected) not drawn.
  example   one responsive electrode (plot_latency.responsive; default: the occipital
            electrode with the largest Cohen's d among sessions with >= 200 trials):
            event-mean high-gamma power (/ word-off mean), 95% CI (t-based), word off and word on
            shaded, half-maximum latency marked (plot_latency.half_max).

Electrode table cached to <out-dir>/electrodes.csv (--refresh rebuilds it).

    python plot_study_figure.py [--example SUB_EXP_SESS LABEL] [--refresh]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import half_max, response, responsive

MIN_TRIALS_EXAMPLE = 200
MAX_OUTSIDE_MM = 10


def electrode_table(files: list) -> pd.DataFrame:
    """One row per session x pair: sid, label, MNI centroid, lobe, responsive, Cohen's d."""
    lobe_of, rows = fc.load_burke_maps(), []
    for f in tqdm(files, desc="sessions"):
        p = fc.load_pickle(str(f))
        xyz, _ = fc.pair_xyz_lead(fc.dfrow_from_sid(p["sid"]))
        if len(xyz) != len(p["labels"]):
            continue
        roi, resp = fc.roi_of_reg_full(p["reg_full"], lobe_of), responsive(p)
        for c, label in enumerate(p["labels"]):
            rows.append((f.name, str(p["sid"][0]), str(label), *xyz[c],
                         roi[c].split("-", 1)[1] if roi[c] else "", bool(resp[c]),
                         float(p["cohens_d"][c]), int(p["n_hi"])))
    return pd.DataFrame(rows, columns=["file", "sub", "label", "x", "y", "z", "lobe",
                                       "responsive", "cohens_d", "n_trials"])


def coverage(ax_fig: plt.Figure, el: pd.DataFrame, path: str) -> None:
    el = el.drop_duplicates(["sub", "label"]).dropna(subset=["x", "y", "z"])
    from nilearn.datasets import load_mni152_brain_mask
    from scipy.ndimage import distance_transform_edt
    mask_img = load_mni152_brain_mask(resolution=1)
    dist = distance_transform_edt(mask_img.get_fdata() == 0)           # mm outside the brain
    xyz = el[["x", "y", "z"]].to_numpy(float)
    ijk = np.rint(np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(mask_img.affine).T)[:, :3].astype(int)
    outside = dist[tuple(np.clip(ijk, 0, np.array(dist.shape) - 1).T)] > MAX_OUTSIDE_MM
    display = plotting.plot_glass_brain(None, display_mode="lzr", figure=ax_fig, alpha=0.3)
    for lobe in [""] + fc.LOBES:
        lobe_els = el[(el["lobe"] == lobe) & ~outside]
        display.add_markers(lobe_els[["x", "y", "z"]].to_numpy(float), marker_color=fc.LOBE_COLORS.get(lobe, "0.75"),
                            marker_size=0.4, alpha=0.35 if lobe else 0.2)
    handles = [plt.Line2D([], [], ls="", marker="o", color=fc.LOBE_COLORS[lobe],
                          label=f"{lobe.capitalize()} ({el.loc[el.lobe == lobe, 'sub'].nunique()} subj., "
                                f"{(el.lobe == lobe).sum():,} el.)") for lobe in fc.LOBES]
    ax_fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11, markerscale=1.5, columnspacing=1,
                  bbox_to_anchor=(0.5, -0.16))
    ax_fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    print(f"[coverage] {len(el):,} electrodes, {el['sub'].nunique()} subjects; not drawn (> {MAX_OUTSIDE_MM} mm "
          f"outside the brain): {outside.sum()} from {el.loc[outside, 'sub'].nunique()} subjects -> {path}.png")


def example(el: pd.DataFrame, files: dict, pick: list | None, path: str) -> None:
    if pick:
        row = el[(el["file"].str.startswith(pick[0] + "_")) & (el["label"] == pick[1])].iloc[0]
    else:
        candidates = el[el.responsive & (el.lobe == "occipital") & (el.n_trials >= MIN_TRIALS_EXAMPLE)]
        row = candidates.loc[candidates["cohens_d"].idxmax()]
    p = fc.load_pickle(str(files[row["file"]]))
    ch = list(map(str, p["labels"])).index(row["label"])
    step = float(p["env_step_ms"])
    on = np.asarray(p["env_hi"], float)[:, ch]                       # (E, T), relative to the word-off mean
    off = np.asarray(p["env_lo"], float)[:, ch]
    t_on = p["env_hi_t0_ms"] + step * (np.arange(on.shape[-1]) + 0.5)
    t_off = p["env_lo_t0_ms"] + step * (np.arange(off.shape[-1]) + 0.5)
    x, t = response(p)
    lat, peak = half_max(np.nanmean(x[:, ch], 0), t)
    base = np.nanmean(off)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.axvspan(-700, -100, color="#e8b4a0", alpha=0.4, lw=0)
    ax.axvspan(0, 600, color="#a8e6b8", alpha=0.4, lw=0)
    for times, env in ((t_off, off), (t_on, on)):
        mean, ci = np.nanmean(env, 0), np.array([fc.mean_ci(samples)[1] for samples in env.T])
        ax.fill_between(times, mean - ci, mean + ci, color="0.6", lw=0)
        ax.plot(times, mean, color="k", lw=1.2)
    ax.axhline(base, color="0.4", lw=0.8, ls=":")
    ax.axvline(0, color="k", lw=0.8)
    ylim = ax.get_ylim()
    ax.plot([0, lat], [base + peak / 2] * 2, color="tab:red", lw=0.8, ls="--")
    ax.plot([lat, lat], [ylim[0], base + peak / 2], color="tab:red", lw=0.8, ls="--")
    ax.text(-400, ylim[1], "Word off", ha="center", va="top", fontsize=10)
    ax.text(300, ylim[1], "Word on", ha="center", va="top", fontsize=10)
    ax.text(-10, base + peak / 2, f"half-max\n{lat:.0f} ms", color="tab:red", fontsize=9, ha="right", va="center")
    ax.set_ylim(ylim)
    ax.set(xlabel="Time from word onset (ms)", ylabel="Normalized high-gamma power", xlim=(-700, 1000))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    print(f"[example] {row['file']} {row['label']} ({p['reg_full'][ch]}), d = {row['cohens_d']:.2f}, "
          f"{len(on)} trials, half-max {lat:.0f} ms -> {path}.png")


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "study"), **{
        "--example": dict(nargs=2, metavar=("SUB_EXP_SESS", "LABEL"), help="e.g. R1001P_FR1_0 LO1-LO2"),
        "--refresh": dict(action="store_true", help="rebuild the cached electrode table")})
    os.makedirs(args.out_dir, exist_ok=True)
    files = fc.session_files(power_dir(save_root, "word_on", "high_gamma"), "_power.pkl", args.n_sessions)
    cache = join(args.out_dir, "electrodes.csv")
    if args.refresh or not os.path.exists(cache):
        electrode_table(files).to_csv(cache, index=False)
    el = pd.read_csv(cache, keep_default_na=False, na_values=[""])
    el["lobe"] = el["lobe"].fillna("")
    coverage(plt.figure(figsize=(9, 3.2)), el, join(args.out_dir, "coverage"))
    example(el, {f.name: f for f in files}, args.example, join(args.out_dir, "example_response"))


if __name__ == "__main__":
    main()
