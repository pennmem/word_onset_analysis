"""compute.py -- single-pass per-session compute: load each session's EEG once,
write every power and synchrony pickle the plot stages read.

Per session the word-off (PRE_WORD, -700..-100 ms) and word-on (WORD, 0..1600 ms)
clips are loaded once, each widened by `real_data_buffer_ms` of real data, and
shared by:

  synchrony  alpha ciPLV (multitaper) and high-gamma AEC-c (70-110 Hz Hilbert),
             main window + 200 ms epochs  -> <save_root>/word_on/fc_mats/diff/<band>/
             (fc.compute_prepost_separate, as build_roi_synchrony)
  power      alpha: multitaper over 8-13 Hz           -> word_on/power/alpha/
             spectrum: multitaper, 10 log bins 5-100   -> word_on/power/spectrum/
             high gamma: Hilbert in 10 Hz sub-bands 70-150 Hz, each normalized by
             its word-off mean, then averaged (broadband HFA with every frequency
             weighted equally)                          -> word_on/power/high_gamma/
             with a 50 ms-bin time course and per-trial envelopes at 10 ms for
             latency analyses

Every power pickle also keeps per-trial power and each word's preceding blank
(ms since the previous word's offset; NaN for a list's first word), for the
baseline (ISI-split) checks.

    python compute.py [--n-subjects K] [--workers N] [--simulation-tag TAG]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import fc_comparison_functions as fc
import helper

FREQ_EDGES = np.geomspace(5, 150, 12)   # log-spaced bins of the power spectrum (5-150 Hz, high gamma included)
MAINS_HZ = (50, 60, 100, 120)            # notched frequencies (+/- 2 Hz) left out of the spectrum


def power_dir(root: str, beh: str, band: str) -> Path:
    return Path(root) / beh / "power" / fc.band_dirname(band, fc.POWER_MODE)


def cond_dir(root: str, beh: str, cond: str, band: str) -> Path:
    return Path(root) / beh / "fc_mats" / cond / band


def band_power(seg: np.ndarray, sf: float, fmin: float, fmax: float, bandwidth: float) -> np.ndarray:
    """(E, C) multitaper power averaged over [fmin, fmax] (same front end as the phase metrics)."""
    from mne.time_frequency import psd_array_multitaper
    psds, _ = psd_array_multitaper(seg, sf, fmin=fmin, fmax=fmax, bandwidth=bandwidth, adaptive=False,
                                   low_bias=False, normalization="full", verbose=False)
    return np.asarray(psds).mean(-1)


def band_power_bins(seg: np.ndarray, sf: float, bandwidth: float,
                    edges: np.ndarray = FREQ_EDGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E, C, n_bins) multitaper power averaged within each [edges[k], edges[k+1]) bin, leaving
    out the notched mains frequencies (MAINS_HZ +/- 2 Hz); + the trial-averaged spectrum at the
    native resolution (C, F) and its frequencies (F) (for specparam)."""
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(seg, sf, fmin=edges[0], fmax=edges[-1], bandwidth=bandwidth,
                                       adaptive=False, low_bias=False, normalization="full", verbose=False)
    psds = np.asarray(psds)
    off_mains = np.all([np.abs(freqs - mains) > 2 for mains in MAINS_HZ], axis=0)
    bin_pow = np.stack([psds[..., (freqs >= f_lo) & (freqs < f_hi) & off_mains].mean(-1)
                        for f_lo, f_hi in zip(edges[:-1], edges[1:])], axis=-1)
    return bin_pow, psds.mean(0), freqs


BEH = "word_on"
SYNC = {"alpha": "ciplv", "high_gamma": "aec_c"}
HG_SUBBANDS = [(f, f + 10) for f in range(70, 150, 10)]
ENV_STEP_MS = 10          # stored per-trial high-gamma envelope resolution
VERSION = "single_pass_8"   # 2: envelopes float32 (float16 overflowed); 3-4: PAC (since removed);
#                            5: epochs 0-1000 ms, 200 ms every 100 ms; 6: recalled_hi; 7: per-trial spectrum bins; 8: spectrum to 150 Hz + native trial-mean spectra


def _crop(eeg: Any, win: tuple[float, float]) -> np.ndarray:
    t = np.asarray(eeg.time, float)
    return np.asarray(eeg.data, float)[..., (t >= win[0]) & (t < win[1])]


def _blank_before(events: pd.DataFrame) -> np.ndarray:
    """ms from the previous word's offset to each word's onset (same list), NaN for the first."""
    timing = events[["trial", "onset", "duration"]].astype(float)
    prev_off = (timing["onset"] + timing["duration"]).groupby(timing["trial"]).shift(1)
    return ((timing["onset"] - prev_off) * 1000).to_numpy()


def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Per-channel Cohen's d (hi vs lo trials) over the trial axis; (E, C) -> (C,)."""
    return np.asarray(helper.cohens_d(hi, lo), float).reshape(hi.shape[1:])


def hg_envelope(x: np.ndarray, sf: float, n_buf: int) -> np.ndarray:
    """(E, C, T) sub-band Hilbert power, buffer cropped, NOT yet normalized:
    list of per-sub-band arrays."""
    subband_pow = []
    for lo, hi in HG_SUBBANDS:
        analytic = fc.band_analytic(x, sf, lo, hi)
        subband_pow.append((np.abs(analytic) ** 2)[..., n_buf:analytic.shape[-1] - n_buf])
    return subband_pow


def run_session(dfrow: pd.Series, save_root: str, root_dir: str,
                simulation_tag: str | None = None) -> str:
    fc.root_dir = helper.root_dir = root_dir
    sid = fc.ftag(dfrow)
    sync_paths = {b: cond_dir(save_root, BEH, "diff", b) / f"{sid}_fc_mats.pkl" for b in SYNC}
    power_paths = {b: power_dir(save_root, BEH, b) / f"{sid}_power.pkl" for b in ("alpha", "high_gamma", "spectrum")}
    if all(p.exists() for p in [*sync_paths.values(), *power_paths.values()]) and \
            fc.load_pickle(str(sync_paths["alpha"])).get("version") == VERSION:   # small file; written first
        return f"{sid}: cached"

    ev = fc.load_events(dfrow, BEH)
    if ev is None:
        return f"{sid}: no events"
    pairs = helper.get_pairs(dfrow)
    overlap_mask = fc.make_overlap_mask(pairs, n_ch=len(pairs))
    reg_full = np.asarray(helper.regionalize_electrodes_by_type(pairs), dtype=object)
    labels = pairs["label"].astype(str).to_numpy()
    meta = {"sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])), "reg_full": reg_full,
            "version": VERSION, "buffer_ms": fc.REAL_DATA_BUFFER_MS, "subtract_erp": fc.SUBTRACT_ERP,
            "notch_harmonics_up_to_hz": fc.NOTCH_HARMONICS_UP_TO_HZ, "simulation_tag": simulation_tag}

    clips: dict[Any, Any] = {}   # window -> (eeg, mask), each loaded once
    def load(d, events, window, buf, sim):
        win = tuple(window)
        if win not in clips:
            clips[win] = fc.get_beh_eeg(d, events, window, buf, sim)
        return clips[win]

    # synchrony: both bands share the two loaded clips
    for band, metric in SYNC.items():
        sync = fc.compute_prepost_separate(dfrow, BEH, ev, [metric], *fc.bands[band], overlap_mask,
                                           fc.REAL_DATA_BUFFER_MS, simulation_tag=simulation_tag,
                                           load_fn=load)
        out = {**meta, metric: np.squeeze(sync[metric]["diff"]), f"{metric}_epochs": sync[metric]["epochs"]}
        os.makedirs(sync_paths[band].parent, exist_ok=True)
        fc.save_pickle(str(sync_paths[band]), out)

    # power from the same clips
    prepost = fc.PREPOST_SPEC[BEH]
    pre_win, post_win = tuple(prepost["pre_win"]), tuple(prepost["post_win"])
    (pre_eeg, _), (post_eeg, _) = [clip for win, clip in clips.items() if win == pre_win][0], \
        [clip for win, clip in clips.items() if win != pre_win][0]
    sf = float(post_eeg.samplerate)
    n_buf = int(round(fc.REAL_DATA_BUFFER_MS * sf / 1000.0))
    word_ev = ev[np.asarray(ev.attrs["mask"], bool)].reset_index(drop=True)
    blank = _blank_before(word_ev)
    blank_lo = blank[np.asarray(pre_eeg.event, int)]
    blank_hi = blank[np.asarray(post_eeg.event, int)]
    seg_lo, seg_hi = fc.equalize_time_length(_crop(pre_eeg, pre_win), _crop(post_eeg, post_win))
    trial_info = {"labels": labels, "n_lo": len(seg_lo), "n_hi": len(seg_hi), "n_win_samples": seg_lo.shape[-1],
                  "lo_win": pre_win, "hi_win": post_win, "blank_ms_lo": blank_lo, "blank_ms_hi": blank_hi,
                  "recalled_hi": word_ev["correct_recall"].to_numpy(float)[np.asarray(post_eeg.event, int)]}

    def save(band, p_lo, p_hi, extra):
        out = {**meta, **trial_info, "band": band, "fmin": fc.bands[band][0], "fmax": fc.bands[band][1],
               "pow_lo": np.nanmean(p_lo, 0), "pow_hi": np.nanmean(p_hi, 0),
               "cohens_d": _cohens_d(p_hi, p_lo),
               "trial_pow_lo": p_lo.astype(np.float32), "trial_pow_hi": p_hi.astype(np.float32), **extra}
        os.makedirs(power_paths[band].parent, exist_ok=True)
        fc.save_pickle(str(power_paths[band]), out)

    bw = fc.MT_BANDWIDTH
    save("alpha", band_power(seg_lo, sf, *fc.bands["alpha"], bw), band_power(seg_hi, sf, *fc.bands["alpha"], bw),
         {"measure": "power", "fc_mode": "multitaper", "mt_bandwidth": bw})
    (bins_lo, psd_lo, psd_freqs), (bins_hi, psd_hi, _) = band_power_bins(seg_lo, sf, bw), band_power_bins(seg_hi, sf, bw)   # (E, C, n_bins)
    save("spectrum", bins_lo.mean(-1), bins_hi.mean(-1),
         {"measure": "power", "fc_mode": "multitaper", "mt_bandwidth": bw, "freq_edges": FREQ_EDGES,
          "cohens_d_freq": np.stack([_cohens_d(bins_hi[..., k], bins_lo[..., k]) for k in range(bins_lo.shape[-1])], -1),
          "psd_lo": psd_lo.astype(np.float32), "psd_hi": psd_hi.astype(np.float32), "psd_freqs": psd_freqs})   # specparam

    # high gamma: each sub-band / its word-off mean (per channel), then averaged
    t_pre = np.asarray(pre_eeg.time, float)[n_buf:len(pre_eeg.time) - n_buf]
    t_post = np.asarray(post_eeg.time, float)[n_buf:len(post_eeg.time) - n_buf]
    in_pre = (t_pre >= pre_win[0]) & (t_pre < pre_win[1])
    sub_pre, sub_post = hg_envelope(np.asarray(pre_eeg.data, float), sf, n_buf), \
        hg_envelope(np.asarray(post_eeg.data, float), sf, n_buf)
    baseline = [np.nanmean(sub[..., in_pre], axis=(0, 2), keepdims=True) for sub in sub_pre]
    env_pre = np.mean([sub / base for sub, base in zip(sub_pre, baseline)], axis=0)[..., in_pre]
    env_post = np.mean([sub / base for sub, base in zip(sub_post, baseline)], axis=0)
    in_post = (t_post >= post_win[0]) & (t_post < post_win[1])
    p_lo, p_hi = env_pre.mean(-1), env_post[..., in_post].mean(-1)
    bin_ms = fc.TIME_BIN_MS
    bin_edges = np.arange(post_win[0], post_win[1] + 1e-9, bin_ms)
    d_bins = np.stack([_cohens_d(env_post[..., (t_post >= t0) & (t_post < t1)].mean(-1), p_lo)
                       for t0, t1 in zip(bin_edges[:-1], bin_edges[1:])], -1)
    env_step = int(round(ENV_STEP_MS * sf / 1000.0))
    downsample = lambda env: env[..., :env.shape[-1] // env_step * env_step].reshape(*env.shape[:2], -1, env_step).mean(-1).astype(np.float32)
    save("high_gamma", p_lo, p_hi,
         {"measure": "power", "fc_mode": "hilbert_subbands", "subbands": HG_SUBBANDS,
          "cohens_d_bins": d_bins, "bin_centers_ms": (bin_edges[:-1] + bin_edges[1:]) / 2 - post_win[0],
          "time_bin_ms": bin_ms, "env_step_ms": ENV_STEP_MS,
          "env_lo": downsample(env_pre), "env_lo_t0_ms": float(t_pre[in_pre][0]),
          "env_hi": downsample(env_post), "env_hi_t0_ms": float(t_post[0])})
    return f"{sid}: {len(labels)} elec, {len(seg_lo)}+{len(seg_hi)} events"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    fc.add_common_args(parser)
    args = parser.parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    print(f"[setup] buffer {fc.REAL_DATA_BUFFER_MS} ms, notch harmonics up to "
          f"{fc.NOTCH_HARMONICS_UP_TO_HZ} Hz, version {VERSION}")
    fc.run_compute_stage(run_session, desc="compute", root_dir_=root_dir,
                         n_sessions=args.n_sessions, n_subjects=args.n_subjects, workers=args.workers,
                         save_root=save_root, root_dir=root_dir, simulation_tag=args.simulation_tag)


if __name__ == "__main__":
    main()
