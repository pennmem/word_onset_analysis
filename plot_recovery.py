"""Recovery curves: measured vs planted effect for the simulation sweeps.

Reads the per-session pickles `snakemake simulations` writes for every
`sweeps:` tag in config/simulation_config.yaml and plots, per sweep, the effect
the pipeline measured against the one planted. Target-lobe electrodes should sit
on the identity line; all others at 0.

  power   high gamma post/pre band power (or amplitude, hilbert) in dB; planted 20*log10(gain)
  ciPLV   alpha word on - word off ciPLV between electrode pairs; planted = post phase consistency
  AEC-c   high-gamma word on - word off AEC-c; planted = depth of the shared envelope
Synchrony plants are not on the metric's scale, so their x axis is the true value: the same
metric (fc.compute_prepost_separate) on the planted signal alone, pink noise off, per level on
N_TRUTH sessions (one per subject; synthetic clips of the real montage and events, no EEG needed;
cached in results/simulations/recovery_truth.csv). Measured below the identity line = the noise's
dilution of the coupling (real), not estimator bias.

    python plot_recovery.py [--root-dir DIR] [--out-dir DIR]
"""
import argparse
import os
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fc_comparison_functions as fc
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]
from compute import cond_dir, power_dir
import helper
import simulate_eeg
from simulate_eeg import SWEEPS, _channel_lobes, simulation_parameters

BEH = "word_on"
# data-generating process -> (panel, band, metric; None = power)
KIND = {"hg_power": ("power", "high_gamma", None), "osc_lag": ("ciPLV", "alpha", "ciplv"),
        "hg_envelope": ("AEC-c", "high_gamma", "aec_c")}


def session_values(root, tag):
    """[(target, other)] per session: mean measured effect on / off the plant."""
    params = simulation_parameters[tag]
    _, band, metric = KIND[params["data_generating_process"]]
    sim_root = join(root, "sim", tag)
    out = []
    if metric is None:
        for f in fc.session_files(power_dir(sim_root, BEH, band), "_power.pkl"):
            res = fc.load_pickle(str(f))
            target = np.isin(_channel_lobes(res["reg_full"]), params["target_lobes"])
            db = (20 if res.get("measure") == "amplitude" else 10) * np.log10(res["pow_hi"] / res["pow_lo"])
            out.append((np.nanmean(db[target]), np.nanmean(db[~target])))
    else:
        for f in fc.session_files(cond_dir(sim_root, BEH, "diff", band), "_fc_mats.pkl"):
            res = fc.load_pickle(str(f))
            target = np.isin(_channel_lobes(res["reg_full"]), params["target_lobes"])
            mat = fc.symmetrize_dense(np.asarray(res[metric], float), diag_value=np.nan)
            target_pairs, other_pairs = np.outer(target, target), ~np.outer(target, target)
            np.fill_diagonal(target_pairs, False); np.fill_diagonal(other_pairs, False)
            out.append((np.nanmean(mat[target_pairs]), np.nanmean(mat[other_pairs])))
    return np.array(out)


N_TRUTH = 10
TRUTH_CSV = join("results", "simulations", "recovery_truth.csv")


def synthetic_clips(d, events, window, buf, sim):
    """get_beh_eeg's clips for `sim` without loading EEG: the simulator only needs the shape."""
    sr = float(helper.RESAMPLE_HZ)
    t = np.arange(round(window[0] - buf), round(window[1] + buf) + 1e-9, 1000 / sr)
    labels = helper.get_pairs(d)["label"].astype(str).to_numpy()
    eeg = TimeSeries.create(np.zeros((len(events), len(labels), len(t))), sr, dims=("event", "channel", "time"),
                            coords={"event": events.index.to_numpy(), "channel": labels, "time": t})
    mask = np.asarray(events.attrs["mask"], bool)
    eeg = simulate_eeg.replace_w_simulated_EEG(eeg, d, mask, sim, events)
    return helper.notch_filter(eeg, d["sub"], harmonics_up_to_hz=fc.NOTCH_HARMONICS_UP_TO_HZ), mask


def noise_free(root, sid, tag):
    """Target-pair mean of the metric on one session's planted signal alone (pink noise off)."""
    fc.root_dir = helper.root_dir = root
    params = simulate_eeg.simulation_parameters[tag]   # the module's dict (a global here is pickled as a copy)
    _, band, metric = KIND[params["data_generating_process"]]
    params["pinknoise_amplitude"] = 0   # this worker's process only
    dfrow = fc.dfrow_from_sid(sid)
    events, pairs = fc.load_events(dfrow, BEH), helper.get_pairs(dfrow)
    mat = fc.compute_prepost_separate(dfrow, BEH, events, [metric], *fc.bands[band],
                                      fc.make_overlap_mask(pairs, n_ch=len(pairs)), fc.REAL_DATA_BUFFER_MS,
                                      simulation_tag=tag, load_fn=synthetic_clips)[metric]["diff"]
    target = np.isin(_channel_lobes(fc.session_regions(dfrow["sub"], dfrow["exp"], str(dfrow["sess"]))),
                     params["target_lobes"])
    mat = fc.symmetrize_dense(np.asarray(np.squeeze(mat), float), diag_value=np.nan)
    target_pairs = np.outer(target, target); np.fill_diagonal(target_pairs, False)
    return {"tag": tag, "sid": str(sid), "true": np.nanmean(mat[target_pairs])}


def truth(root, tags):
    """{tag: mean noise-free value over N_TRUTH sessions (first session of the first subjects)}."""
    cached = pd.read_csv(TRUTH_CSV) if exists(TRUTH_CSV) else pd.DataFrame(columns=["tag", "sid", "true"])
    jobs = []
    for tag in tags:
        if (cached["tag"] == tag).sum() >= N_TRUTH:
            continue
        _, band, _ = KIND[simulation_parameters[tag]["data_generating_process"]]
        first = {}
        for f in sorted(fc.session_files(cond_dir(join(root, "sim", tag), BEH, "diff", band), "_fc_mats.pkl")):
            sid = fc.load_pickle(str(f))["sid"]
            first.setdefault(sid[0], sid)
        jobs += [(tag, sid) for sid in list(first.values())[:N_TRUTH]]
    if jobs:
        computed = Parallel(n_jobs=5)(delayed(noise_free)(root, sid, tag) for tag, sid in jobs)   # ~6 GB each
        cached = pd.concat([cached[~cached["tag"].isin({t for t, _ in jobs})], pd.DataFrame(computed)])
        os.makedirs(os.path.dirname(TRUTH_CSV), exist_ok=True)
        cached.to_csv(TRUTH_CSV, index=False)
    return cached.groupby("tag")["true"].mean().to_dict()


def planted(tag):
    params = simulation_parameters[tag]
    return 20 * np.log10(params["planted"]) if params.get("data_generating_process") == "hg_power" else params["planted"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root-dir", default=fc.root_dir)
    ap.add_argument("--out-dir", default=join("figures", "simulations"))
    args = ap.parse_args()

    kinds = {"power": ("power change (dB)", []), "ciPLV": ("ciPLV change", []), "AEC-c": ("AEC-c change", [])}
    for sweep, tags in SWEEPS.items():
        kinds[KIND[simulation_parameters[tags[0]]["data_generating_process"]][0]][1].append((sweep, tags))

    true_vals = truth(args.root_dir, [t for tags in SWEEPS.values() for t in tags
                                      if simulation_parameters[t]["data_generating_process"] != "hg_power"
                                      and len(session_values(args.root_dir, t))])
    kinds = {k: v for k, v in kinds.items()   # only the sweeps that were run
             if any(len(session_values(args.root_dir, t)) for _, tags in v[1] for t in tags)}
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.5 * len(kinds), 5), squeeze=False)
    axes = axes[0]
    for ax, (kind, (ylab, sweeps)) in zip(axes, kinds.items()):
        lim = [np.inf, -np.inf]
        for sweep, tags in sweeps:
            x = np.array([planted(t) if kind == "power" else true_vals.get(t, np.nan) for t in tags])
            sess_vals = [session_values(args.root_dir, t) for t in tags]
            if not any(len(s) for s in sess_vals):   # sweep not run (e.g. no synchrony under longetal)
                continue
            noise = sweep.split("_")[-1] if sweep.endswith(("noisy", "clean")) else ""
            for col, label, ls in ((0, f"Frontal-frontal pairs (planted) {noise}", "-"),
                                 (1, f"All other pairs (nothing planted) {noise}", ":")):
                y = np.array([np.nanmean(s[:, col]) if len(s) else np.nan for s in sess_vals])   # NaN: session without target pairs
                sem = np.array([np.nanstd(s[:, col], ddof=1) / np.sqrt(np.isfinite(s[:, col]).sum())
                                if len(s) > 1 else np.nan for s in sess_vals])
                ax.errorbar(x, y, yerr=sem, marker="o", ls=ls, capsize=3, label=label)
            lim = [min(lim[0], np.nanmin(x)), max(lim[1], np.nanmax(x))]
        if not np.isfinite(lim[0]):   # nothing run for this kind
            ax.set_visible(False)
            continue
        ax.plot(lim, lim, "k--", lw=1, label="Measured = true")
        ax.axhline(0, color="0.6", lw=0.8)
        planted_lab = {"power": "Planted power change (dB)",
                       "ciPLV": "True ciPLV change, word on - word off\n(planted signal alone, no noise)",
                       "AEC-c": "True AEC-c change, word on - word off\n(planted signal alone, no noise)"}[kind]
        title = {"power": "High-gamma power", "ciPLV": "Alpha phase connectivity (ciPLV)",
                 "AEC-c": "High-gamma envelope connectivity (AEC-c)"}[kind]
        ax.set(xlabel=planted_lab, ylabel=f"Measured {ylab}, word on - word off\n(with noise; mean ± SEM over sessions)", title=title)
        ax.legend(fontsize=8)
    fig.suptitle("Does the pipeline recover planted connectivity?\n"
                 "Simulated signals on real montages; points below the line: attenuation by noise", fontsize=11)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(join(args.out_dir, f"recovery.{ext}"), dpi=200, bbox_inches="tight")
    print(f"[saved] {args.out_dir}/recovery.png / .pdf")


if __name__ == "__main__":
    main()
