"""
build_roi_power.py

Local (single-electrode) spectral power in the same 12 Burke ROIs used by the
phase-connectivity analyses -- the power counterpart to build_roi_synchrony.py.

Same events, same windows, same band, same multitaper front end as the phase
metrics; the only change is that each electrode is treated on its own instead
of as one endpoint of a pair:

    per event, per electrode:  multitaper PSD over the window, averaged in band
    per electrode:             Cohen's d of hi vs lo power ACROSS events
    per subject, per ROI:      mean of d over that ROI's electrodes
    across subjects:           box plot (one box per ROI)

Two stages:

  compute  one pickle per session (--workers N to parallelise)
             <save_root>/<beh>/power/<band>/<ftag>_power.pkl
             {"sid", "labels", "reg_full", "pow_lo", "pow_hi",
              "cohens_d", "n_events"}
  plot     aggregate those pickles -> per-subject-per-ROI table -> box plots

Relationship to Rao et al. 2025 (J Neurosci)
--------------------------------------------
The per-electrode summary is Cohen's d of band power (no log transform), region-averaged, following
that paper's power pipeline (helper.get_power -> comp_elpomx ->
regionalize_electrode_powers). Specifically:

  adopted    `helper.cohens_d` verbatim (pooled-SD,
             the same function vendored in this repo); the effect size -- not raw
             power -- as the per-electrode quantity; mean of d over the
             electrodes in a region as the region value.
  not adopted  Rao's Morlet front end: power uses the same multitaper estimator
             as the phase metrics.
  no-op here  get_power's z-score across events. It is one (mean, std) per
             channel/frequency applied to both samples, and Cohen's d is
             invariant to an affine transform -- so with a single band-averaged
             value per event it changes nothing. (In Rao's pipeline it does
             matter, because it reweights the frequencies before the band
             average; here the band average happens inside the multitaper.)

Note that Rao's contrast (recalled vs not-recalled) is between INDEPENDENT event
groups, while pre vs post word onset are the same trials at two windows. The
pooled-SD formula is kept for consistency; it ignores that pairing and is
therefore somewhat conservative. A paired d_z would be mean(diff)/sd(diff).

Measures. Raw power is not comparable across subjects (amplifier gain,
referencing, and coverage differ), so only window contrasts are used:

  cohens_d       effect size of hi vs lo power, across events, per electrode.
                 Unitless and gain-invariant; the plotted measure.
  change_dB      10 * log10(P_hi / P_lo) of the session-mean powers -- the same
                 contrast in dB. Written to the CSVs only.

Per ROI, a one-sample t of the subject values vs 0 (= paired t), FDR across the
12 ROIs (fc.roi_stats). The figure has one row per band already plotted into
--out-dir (fc.band_contrast_figure).

Timing (power time course) is computed for gamma bands only (fmin >= TIMING_MIN_HZ):
latency is not meaningful for slow oscillations. Long et al. settings (Hilbert)
come from config `longetal_params`, never from flags.

Electrode -> ROI uses the canonical `regionalize_electrodes_by_type` label
(volumetric cascade for depths, surface for grid/strip) mapped through
region_to_burke_lobe.csv + hemisphere, identical to the connectivity scripts.

Usage:
    python build_roi_power.py                         # compute + plot
    python build_roi_power.py --n-sessions 2          # smoke test
    python build_roi_power.py --workers 4             # 4 sessions at a time
    python build_roi_power.py --stage plot            # replot from pickles
"""
from __future__ import annotations

from typing import Any

import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc

MEASURES = ("cohens_d", "change_dB")
TIMING_MIN_HZ = 30          # time course only for gamma (PI: latency meaningless at low freqs)
LONG = fc.LONGETAL or {}


def power_dir(root, beh, band):
    return Path(root) / beh / "power" / fc.band_dirname(band, fc.POWER_MODE)


# ------------------------------ compute stage --------------------------------
def _word_locked_eeg(dfrow: pd.Series, beh: str, win: tuple[float, float],
                     buffer_ms: float = 0.0, simulation_tag: str | None = None) -> Any:
    """EEG for the POST-type events of `beh`, loaded over one window spanning
    both the pre and post analysis windows.

    Same load as `fc.session_eeg` but with `beh` as a parameter. For the PREPOST
    behaviors the PRE_* events are row-for-row copies of the POST events
    (fc.compute_prepost_separate asserts this), so slicing one clip in time is
    equivalent to the two separate loads the FC pipeline does, and guarantees
    pre/post come from the same trials.
    """
    import fc_comparison_functions as fc
    import helper

    events = fc.load_events(dfrow, beh)
    if events is None:
        return None
    post_mask = np.asarray(events.attrs["mask"], bool)
    if not post_mask.any():
        return None
    words = events[post_mask].reset_index(drop=True)
    words.attrs = dict(events.attrs)
    # buffer_ms pads the loaded clip with real data so resample/notch/Hilbert
    # edge effects stay outside the analysis windows
    eeg, _ = fc.get_beh_eeg(dfrow, words, win, buffer_ms, simulation_tag)
    kept = np.asarray(eeg.event, int)
    items = words["item_name"].to_numpy()[kept]   # word of each kept event
    if (fc.LONGETAL or {}).get("baseline") != "full_isi":
        return (np.asarray(eeg.data), np.asarray(eeg.time, float),
                float(eeg.samplerate), items, None)
    # blank screen before each word: previous word's offset -> onset (ms); 750 when
    # there is no previous word in the list (first word follows the countdown)
    bids_events = helper.bids_reader(dfrow, eeg=False).load_events()
    bids_words = bids_events[bids_events["trial_type"] == "WORD"].sort_values("onset")
    blank_by_sample = ((bids_words["onset"] - (bids_words["onset"] + bids_words["duration"]).groupby(bids_words["list"]).shift(1)) * 1000).fillna(750.0)
    blank_by_sample = pd.Series(blank_by_sample.to_numpy(), index=bids_words["sample"].astype(int).to_numpy())
    blank_ms = blank_by_sample.reindex(words["eegoffset"].astype(int).to_numpy()[kept]).fillna(750.0).to_numpy()
    return (np.asarray(eeg.data), np.asarray(eeg.time, float),
            float(eeg.samplerate), items, blank_ms)


def window_slice(data, t, win):
    """(n_events, n_channels, n_win) slice of `data` covering `win`, both ends
    inclusive -- so pre and post can differ by one sample;
    `fc.equalize_time_length` reconciles them before the spectra are taken.
    """
    in_win = (t >= win[0]) & (t <= win[1])
    if in_win.sum() < 2:
        raise ValueError(f"window {win} covers {int(in_win.sum())} samples")
    return data[:, :, in_win]


def band_power(seg, sf, fmin, fmax, bandwidth):
    """Per-event, per-channel multitaper power averaged over [fmin, fmax].

    Mirrors the multitaper front end of the phase metrics (same bandwidth,
    adaptive=False, low_bias=False) -- see fc.mt_event_angles -- but keeps the
    auto-spectrum instead of the cross-spectrum. Returns (n_events, n_channels).
    """
    from mne.time_frequency import psd_array_multitaper

    psds, _ = psd_array_multitaper(
        seg, sf, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
        adaptive=False, low_bias=False, normalization="full", verbose=False)
    return np.asarray(psds).mean(-1)                       # (E, C)


FREQ_EDGES = np.geomspace(5, 100, 11)   # log-spaced bins of the power spectrum (band "spectrum")


def band_power_bins(seg, sf, bandwidth, edges=FREQ_EDGES):
    """(E, C, n_bins) multitaper power averaged within each [edges[k], edges[k+1])
    bin, leaving out the notched mains frequencies (48-52, 58-62 Hz)."""
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(seg, sf, fmin=edges[0], fmax=edges[-1], bandwidth=bandwidth,
                                       adaptive=False, low_bias=False, normalization="full", verbose=False)
    off_mains = (np.abs(freqs - 50) > 2) & (np.abs(freqs - 60) > 2)
    return np.stack([np.asarray(psds)[..., (freqs >= f_lo) & (freqs < f_hi) & off_mains].mean(-1)
                     for f_lo, f_hi in zip(edges[:-1], edges[1:])], axis=-1)


def hilbert_envelope(seg, sf, fmin, fmax, buf_samples, env_hz, n_bands=1):
    """Long et al. 2020 high gamma: Hilbert AMPLITUDE envelope at the native rate,
    trim the buffer, downsample to `env_hz`. (E, C, T). n_bands=1: one band-pass
    (same recipe as fc.compute_aec_buffered). n_bands>1: mean amplitude over
    log-spaced Gaussian bands, Hilbert done in the frequency domain (the
    Mesgarani-lab filterbank convention)."""
    import mne
    import scipy.fft as sfft
    from scipy.signal import hilbert
    from mne.filter import next_fast_len
    n_t = np.shape(seg)[-1]
    if n_bands > 1:
        n_fft = next_fast_len(n_t)
        freqs = sfft.fftfreq(n_fft, 1 / sf)
        spectrum = sfft.fft(np.asarray(seg, np.float32), n_fft, axis=-1)   # complex64 keeps memory down
        edges = np.geomspace(fmin, fmax, n_bands + 1)
        env = np.zeros(np.shape(seg), np.float32)
        for lo, hi in zip(edges[:-1], edges[1:]):
            gauss = 2 * np.exp(-0.5 * ((freqs - np.sqrt(lo * hi)) / ((hi - lo) / 2)) ** 2) * (freqs > 0)
            env += np.abs(sfft.ifft(spectrum * gauss.astype(np.float32), axis=-1)[..., :n_t]) / n_bands
    else:
        filtered = mne.filter.filter_data(np.asarray(seg, float), sf, fmin, fmax, verbose=False)
        env = np.abs(hilbert(filtered, N=next_fast_len(n_t), axis=-1))[..., :n_t]
    if buf_samples:
        env = env[..., buf_samples:n_t - buf_samples]
    return mne.filter.resample(np.asarray(env, float), down=sf / env_hz, verbose=False)


def band_power_windowed(seg, sf, fmin, fmax, bandwidth, bin_ms, window_ms):
    """Sliding-window multitaper power, (E, C, n_bins): windows of window_ms
    centred every bin_ms. Consecutive windows overlap, so the EFFECTIVE temporal
    resolution is window_ms; the axis is only SAMPLED at bin_ms. Bins whose full
    window falls outside the segment are NaN (a shorter window would have a
    different spectral resolution)."""
    n_t = seg.shape[-1]
    dur_ms = 1000.0 * n_t / sf
    n_bins = int(round(dur_ms / float(bin_ms)))

    # Scale bandwidth by dur/window to hold the time-bandwidth product
    # NW = T * bandwidth / 2 (and so the tapers) fixed: 600 ms @ 2 Hz and
    # 200 ms @ 6 Hz are both NW = 0.6, while 2 Hz over 200 ms is NW = 0.2, which
    # MNE rejects (< 0.5). The cost is coarser smoothing in Hz (2 -> 6 Hz).
    win_bandwidth = float(bandwidth) * (dur_ms / float(window_ms))
    centers = (np.arange(n_bins) + 0.5) * float(bin_ms)
    half_win_ms = float(window_ms) / 2.0
    win_samples = int(round(float(window_ms) * sf / 1000.0))

    power = np.full(seg.shape[:2] + (n_bins,), np.nan)
    for k, center_ms in enumerate(centers):
        if center_ms - half_win_ms < 0 or center_ms + half_win_ms > dur_ms:
            continue
        start = int(round((center_ms - half_win_ms) * sf / 1000.0))
        win_seg = seg[..., start:start + win_samples]
        if win_seg.shape[-1] != win_samples:
            continue
        power[..., k] = band_power(win_seg, sf, fmin, fmax, win_bandwidth)
    return power


def run_sess_power(
    dfrow: pd.Series, save_root: str, beh: str, band: str, root_dir: str,
    simulation_tag: str | None = None,
) -> str:
    """Compute per-electrode band power for one session; write one pickle.

    Returns a short status string for the dispatcher's progress bar.
    """
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir

    sid = f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}"
    fc_mode = fc.POWER_MODE
    out_dir = str(power_dir(save_root, beh, band))
    out_path = join(out_dir, f"{fc.ftag(dfrow)}_power.pkl")

    if os.path.exists(out_path):
        try:
            cached = fc.load_pickle(out_path)
            # pickles without n_win_samples predate the pre/post window
            # equalization fix (pre had one extra sample): recompute them
            if all(k in cached for k in ("pow_lo", "pow_hi", "cohens_d",
                                         "reg_full", "n_win_samples")) and (
                    cached.get("buffer_ms") == fc.REAL_DATA_BUFFER_MS) and (
                    fc_mode != "hilbert" or cached.get("env_lo") is not None):
                return f"{sid}: cached"
        except Exception:
            pass

    fmin, fmax = fc.bands[band]
    is_hilbert = fc_mode == "hilbert"   # filter edge lives in the real-data buffer
    buf_ms = fc.REAL_DATA_BUFFER_MS if is_hilbert else 0.0
    prepost = fc.PREPOST_SPEC[beh]
    pre_win, post_win = prepost["pre_win"], prepost["post_win"]
    loaded = _word_locked_eeg(dfrow, beh, (pre_win[0], post_win[1]),
                              buffer_ms=fc.REAL_DATA_BUFFER_MS,   # keeps notch ringing out of both windows
                              simulation_tag=simulation_tag)
    if loaded is None:
        return f"{sid}: no events ({beh})"
    data, t, sf, items, blank_ms = loaded

    pairs = helper.get_pairs(dfrow)
    n_ch = data.shape[1]
    if pairs is None or len(pairs) != n_ch:
        raise ValueError(
            f"{sid}: len(pairs)={None if pairs is None else len(pairs)} but eeg "
            f"n_ch={n_ch}; channel order misaligned for region labels.")

    reg_full = np.asarray(helper.regionalize_electrodes_by_type(pairs), dtype=object)
    labels = pairs["label"].astype(str).to_numpy()

    # must equal the phase metrics' bandwidth, so no local fallback when unset
    bandwidth = fc.MT_BANDWIDTH
    if bandwidth is None:
        raise ValueError(
            "config.yaml mt_bandwidth is unset; power and the phase metrics must "
            "share one multitaper bandwidth (default mt_bandwidth: 2)")

    buf_samples = int(buf_ms * sf / 1000.0)
    widen = lambda w: (w[0] - buf_ms, w[1] + buf_ms)
    mt_window_ms = fc.MT_WINDOW_MS
    time_bin_ms = fc.TIME_BIN_MS if fmin >= TIMING_MIN_HZ else None

    env_hz = (fc.LONGETAL or {}).get("envelope_hz", 100)
    env_cache = {}
    def _envelope(seg):
        if id(seg) not in env_cache:
            env_cache[id(seg)] = hilbert_envelope(seg, sf, fmin, fmax, buf_samples, env_hz,
                                                  (fc.LONGETAL or {}).get("hg_bands", 1))
        return env_cache[id(seg)]

    def _power(seg, bin_ms=None):
        if is_hilbert:
            env = _envelope(seg)
            return helper.timebin_power_timeseries(env, env_hz, bin_width_ms=int(bin_ms)) if bin_ms else np.nanmean(env, -1)
        if bin_ms:
            return band_power_windowed(seg, sf, fmin, fmax, bandwidth,
                                       bin_ms, mt_window_ms)
        return band_power(seg, sf, fmin, fmax, bandwidth)

    # inclusive masks give pre one extra sample; equalize as the FC path does so
    # the frequency bins match between the arms
    lo_seg = window_slice(data, t, widen(pre_win))
    hi_seg = window_slice(data, t, widen(post_win))
    if not is_hilbert:   # Long's windows differ in length (750 vs 1600 ms); a mean amplitude doesn't need them equal
        lo_seg, hi_seg = fc.equalize_time_length(lo_seg, hi_seg)
    n_win = lo_seg.shape[-1] - 2 * buf_samples
    if is_hilbert and LONG.get("baseline") == "full_isi":
        # baseline = each trial's whole blank screen: NaN the pre samples before the previous word's offset
        env = _envelope(lo_seg)
        t_env = pre_win[0] + np.arange(env.shape[-1]) * 1000.0 / env_hz
        env[np.broadcast_to(t_env[None, None, :] < -blank_ms[:, None, None], env.shape)] = np.nan
    pow_lo = _power(lo_seg)
    pow_hi = _power(hi_seg)

    pow_lo[~np.isfinite(pow_lo)] = np.nan
    pow_hi[~np.isfinite(pow_hi)] = np.nan

    d_freq = None
    if band == "spectrum":   # Cohen's d per log-spaced frequency bin, (n_ch, n_bins)
        freq_lo, freq_hi = band_power_bins(lo_seg, sf, bandwidth), band_power_bins(hi_seg, sf, bandwidth)
        d_freq = np.asarray(helper.cohens_d(freq_hi.reshape(len(freq_hi), -1),
                                            freq_lo.reshape(len(freq_lo), -1))).reshape(n_ch, -1)

    # Cohen's d per time bin, (n_ch, n_bins) -> d as a function of latency.
    d_bins = bin_centers = None
    if time_bin_ms:
        # Baseline = mean over the WHOLE pre window, one value per event, not the
        # matching pre bin: bin-to-bin pairing is arbitrary, imports pre-window
        # structure (anticipation, previous-word offset) with flipped sign, and is
        # ~B times noisier; Long et al. also use the whole blank screen. It uses the
        # same binned estimator as the response (the whole-window call would add a
        # constant offset); nanmean drops edge bins that cannot fit a full window.
        binned_lo = _power(lo_seg, bin_ms=time_bin_ms)             # (E, C, B)
        binned_hi = _power(hi_seg, bin_ms=time_bin_ms)             # (E, C, B)
        binned_lo[~np.isfinite(binned_lo)] = np.nan
        baseline = np.nanmean(binned_lo, axis=-1)                  # (E, C)
        binned_hi[~np.isfinite(binned_hi)] = np.nan
        baseline[~np.isfinite(baseline)] = np.nan
        n_bins = binned_hi.shape[-1]
        d_bins = np.full((n_ch, n_bins), np.nan)
        for c in range(n_ch):
            base_vals = baseline[np.isfinite(baseline[:, c]), c]
            if base_vals.size <= 1:
                continue
            for k in range(n_bins):
                post_vals = binned_hi[np.isfinite(binned_hi[:, c, k]), c, k]
                if post_vals.size > 1:
                    d_bins[c, k] = helper.cohens_d(post_vals, base_vals)
        # ms into the post window
        bin_centers = (np.arange(n_bins) + 0.5) * float(time_bin_ms)

    # Rao et al. 2025 per-electrode Cohen's d (pooled SD), hi vs lo across events;
    # per channel because helper.cohens_d is not nan-aware
    d_elec = np.full(n_ch, np.nan)
    for c in range(n_ch):
        hi_vals = pow_hi[np.isfinite(pow_hi[:, c]), c]
        lo_vals = pow_lo[np.isfinite(pow_lo[:, c]), c]
        if hi_vals.size >= 2 and lo_vals.size >= 2:
            d_elec[c] = float(helper.cohens_d(hi_vals, lo_vals))

    # Sample-level contrast (hilbert): every envelope sample is an observation,
    # as Long et al. may have done (--t-unit sample at the plot stage).
    d_samp = n_samp_lo = n_samp_hi = None
    if is_hilbert:
        flat = lambda env: env.transpose(0, 2, 1).reshape(-1, env.shape[1])   # (E*T, C)
        samples_lo, samples_hi = flat(_envelope(lo_seg)), flat(_envelope(hi_seg))
        samples_lo = samples_lo[np.isfinite(samples_lo).all(1)]   # full_isi: drop blanked samples
        d_samp, n_samp_lo, n_samp_hi = helper.cohens_d(samples_hi, samples_lo), len(samples_lo), len(samples_hi)

    result: dict[str, Any] = {
        "sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])),
        "labels": labels,
        "reg_full": reg_full,
        "pow_lo": np.nanmean(pow_lo, axis=0),
        "pow_hi": np.nanmean(pow_hi, axis=0),
        "cohens_d": d_elec,
        "cohens_d_freq": d_freq, "freq_edges": FREQ_EDGES if d_freq is not None else None,
        "n_events": int(data.shape[0]),
        "n_lo": int(pow_lo.shape[0]), "n_hi": int(pow_hi.shape[0]),
        "n_win_samples": int(n_win),
        "beh": beh, "band": band, "fmin": fmin, "fmax": fmax,
        "lo_win": pre_win, "hi_win": post_win,
        "mt_bandwidth": bandwidth,
        "fc_mode": fc_mode,
        "measure": "amplitude" if is_hilbert else "power",   # pow_* hold amplitude under hilbert
        "cohens_d_samples": d_samp, "n_lo_samples": n_samp_lo, "n_hi_samples": n_samp_hi,
        # per-event envelopes + words, for averaging each word over sessions (--combine word_average)
        "env_lo": _envelope(lo_seg).astype(np.float32) if is_hilbert else None,
        "env_hi": _envelope(hi_seg).astype(np.float32) if is_hilbert else None,
        "items": items if is_hilbert else None,
        "simulation_tag": simulation_tag,
        "notch_harmonics_up_to_hz": fc.NOTCH_HARMONICS_UP_TO_HZ,
        "buffer_ms": fc.REAL_DATA_BUFFER_MS,
        # latency axis, gamma only
        "cohens_d_bins": d_bins,          # (n_ch, n_bins)
        "bin_centers_ms": bin_centers,    # ms into the post window
        "time_bin_ms": time_bin_ms,
        "mt_window_ms": mt_window_ms if (time_bin_ms and not is_hilbert) else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    fc.save_pickle(out_path, result)
    return f"{sid}: {n_ch} elec, {result['n_lo']}+{result['n_hi']} events"


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(save_root, beh, band, lobe_of, n_sessions=None):
    """Walk the per-session pickles -> tidy per-(subject, electrode) table.

    Sessions of the same subject are averaged per electrode LABEL first, so a
    subject with 4 sessions does not outweigh one with 1.
    """
    pkl_dir = power_dir(save_root, beh, band)
    files = fc.session_files(pkl_dir, "_power.pkl", n_sessions)
    if not files:
        raise SystemExit(
            f"no pickles in {pkl_dir}\nrun the compute stage first: "
            f"python build_roi_power.py --stage compute --band {band}")

    rows = []
    for f in tqdm(files, desc="load sessions"):
        try:
            pkl = fc.load_pickle(str(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e!r}")
            continue
        if "pow_lo" not in pkl:
            print(f"[skip] {f.name}: written by an older version; "
                  "delete it and rerun the compute stage")
            continue
        sub = str(pkl["sid"][0])
        elec_rois = fc.roi_of_reg_full(pkl["reg_full"], lobe_of)
        for label, roi, lo, hi, d in zip(pkl["labels"], elec_rois, pkl["pow_lo"],
                                         pkl["pow_hi"], pkl["cohens_d"]):
            if roi is None or not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            rows.append((sub, str(label), roi, float(lo), float(hi), float(d)))

    if not rows:
        raise SystemExit("no electrodes with a Burke ROI and finite power")
    df = pd.DataFrame(rows,
                      columns=["sub", "label", "roi", "lo", "hi", "cohens_d"])
    # one row per (subject, electrode): plain mean over that subject's sessions
    return (df.groupby(["sub", "label", "roi"], as_index=False)
              [["lo", "hi", "cohens_d"]].mean())


def subject_roi_table(elec_df, min_electrodes=1, db=10.0):
    """Per-electrode measures -> per-(subject, ROI) means."""
    per_elec = elec_df.assign(change_dB=db * np.log10(elec_df["hi"] / elec_df["lo"]))  # db=20 for amplitude (hilbert)
    return per_elec, fc.subject_roi_means(per_elec, list(MEASURES),
                                          min_electrodes=min_electrodes)


def run_latency_stage(save_root, beh, band, args, lobe_of):
    """Time course of the contrast (gamma only)."""
    if fc.bands[band][0] < TIMING_MIN_HZ:
        return
    bin_df = collect_bin_table(save_root, beh, band, lobe_of, args.n_sessions)
    if len(bin_df) == 0:
        print("[latency] these pickles carry no per-bin data; skipping.")
        return
    tbl = subject_roi_bin_table(bin_df, min_electrodes=args.min_electrodes)
    stats = fc.bin_stats(tbl, "bin_ms", "d")

    # after the min-electrodes cut
    elec_counts = (bin_df[bin_df["roi"].isin(set(tbl["roi"]))]
                   .groupby("roi").agg(n_elec=("label", "nunique"),
                                       n_sub=("sub", "nunique")).reset_index())
    print("\n[latency] electrodes per ROI:")
    print(elec_counts.sort_values("n_elec", ascending=False).to_string(index=False))

    tag = f"{beh}_{band}_{fc.POWER_MODE}_{fc.TIME_BIN_MS}ms"
    os.makedirs(args.out_dir, exist_ok=True)
    elec_counts.to_csv(fc.table_path(join(args.out_dir, f"power_timecourse_{tag}_counts.csv")),
               index=False)
    stats.to_csv(fc.table_path(join(args.out_dir, f"power_timecourse_{tag}_stats.csv")),
                 index=False)
    tbl.to_csv(fc.table_path(join(args.out_dir, f"power_timecourse_{tag}_per_subject.csv")),
               index=False)
    path = fc.roi_curve_figure(stats, "bin_ms", join(args.out_dir, f"power_timecourse_{tag}"),
                               "Time after word onset (ms)", "Cohen's d")
    sig = stats[stats["q"] < 0.05]
    print(f"[latency] {tbl['sub'].nunique()} subjects, "
          f"{stats['bin_ms'].nunique()} bins x {stats['roi'].nunique()} ROIs; "
          f"{len(sig)} cells at q<0.05")
    if len(sig):
        # peak per ROI, not earliest significant cell (that surfaces tiny
        # suppressions, e.g. L-frontal @ 75 ms d=-0.02, as onsets)
        peaks = (sig.loc[sig.groupby("roi")["mean"].idxmax()]
                    .sort_values("mean", ascending=False).head(4))
        print("[latency] strongest ROIs (peak bin):")
        for _, row in peaks.iterrows():
            print(f"          {row['roi']:<14} peak @ {row['bin_ms']:>4.0f} ms  "
                  f"d={row['mean']:+.3f}  q={row['q']:.1e}")
    print(f"[latency] wrote {path}")


# Occipital subregions for the spectrum curves: {fine label: (panel, line)}.
SPECTRUM_SUBREGIONS = {"L fusiform gyrus": ("Fusiform", "L"), "R fusiform gyrus": ("Fusiform", "R"),
                       "L lateral occipital cortex": ("Lateral occipital", "L"),
                       "R lateral occipital cortex": ("Lateral occipital", "R")}


def run_spectrum_stage(save_root, beh, args, lobe_of):
    """t of the per-electrode Cohen's d (word on vs off power) per log-spaced
    frequency bin, across subjects: 12-ROI heatmap + occipital-subregion curves.
    BH-FDR over all cells of each figure (fc.bin_stats)."""
    import matplotlib.pyplot as plt
    rows = []
    for f in fc.session_files(power_dir(save_root, beh, "spectrum"), "_power.pkl", args.n_sessions):
        pkl = fc.load_pickle(str(f))
        if pkl.get("cohens_d_freq") is None:
            continue
        elec_rois = fc.roi_of_reg_full(pkl["reg_full"], lobe_of)
        for label, roi, fine_region, d_by_freq in zip(pkl["labels"], elec_rois, pkl["reg_full"], pkl["cohens_d_freq"]):
            subregion = SPECTRUM_SUBREGIONS.get(fine_region)
            rows += [(str(pkl["sid"][0]), str(label), roi, subregion and f"{subregion[0]}|{subregion[1]}", k, float(d))
                     for k, d in enumerate(d_by_freq)]
    d_df = pd.DataFrame(rows, columns=["sub", "label", "roi", "sub_region", "bin", "d"])
    centres = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
    contrast = fc.contrast(beh)
    t_label = f"t ({contrast['hi_label']} vs. {contrast['lo_label']})"

    def stats(col, min_elec):   # sessions -> electrode, electrodes -> subject, t per cell
        per_elec = d_df.dropna(subset=[col]).groupby(["sub", "label", col, "bin"], as_index=False)["d"].mean()
        per_sub = per_elec.groupby(["sub", col, "bin"]).agg(d=("d", "mean"), n=("label", "nunique")).reset_index()
        per_sub = per_sub[per_sub["n"] >= min_elec].rename(columns={col: "roi"})
        per_sub["freq_hz"] = centres[per_sub["bin"]]
        return fc.bin_stats(per_sub, "freq_hz", "d")

    os.makedirs(args.out_dir, exist_ok=True)
    cell_stats = stats("roi", args.min_electrodes)
    cell_stats.to_csv(fc.table_path(join(args.out_dir, f"power_spectrum_{beh}_rois_stats.csv")), index=False)
    t_mat = cell_stats.pivot(index="roi", columns="freq_hz", values="t").reindex(fc.present_rois(cell_stats))
    q_mat = cell_stats.pivot(index="roi", columns="freq_hz", values="q").reindex(fc.present_rois(cell_stats))
    vmax = float(np.nanmax(np.abs(t_mat.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(t_mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), q in np.ndenumerate(q_mat.to_numpy()):
        if fc.stars(q):
            ax.text(j, i, fc.stars(q), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(t_mat.columns)), [f"{x:.0f}" for x in t_mat.columns], fontsize=13)
    ax.set_yticks(range(len(t_mat.index)), [fc.pretty_roi(x) for x in t_mat.index], fontsize=13)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.colorbar(im, ax=ax).set_label(t_label, fontsize=14)
    fig.tight_layout()
    fig.savefig(join(args.out_dir, f"power_spectrum_{beh}_rois.png"), dpi=300)
    fig.savefig(join(args.out_dir, f"power_spectrum_{beh}_rois.pdf"))
    plt.close(fig)

    cell_stats = stats("sub_region", args.min_electrodes)
    cell_stats.to_csv(fc.table_path(join(args.out_dir, f"power_spectrum_{beh}_occipital_stats.csv")), index=False)
    panels = sorted({x.split("|")[0] for x in cell_stats.dropna(subset=["t"])["roi"]})   # subregions with a test
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for name, curve in cell_stats[cell_stats["roi"].str.startswith(panel + "|")].groupby("roi"):
            curve = curve.sort_values("freq_hz")
            line, = ax.plot(curve["freq_hz"], curve["t"], marker="o", lw=2,
                            label=f"{name.split('|')[1]} (n={int(curve['n'].max())})")
            for x, y, q in zip(curve["freq_hz"], curve["t"], curve["q"]):
                if fc.stars(q):
                    ax.annotate(fc.stars(q), (x, y), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=13, color=line.get_color())
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(centres, [f"{x:.0f}" for x in centres], fontsize=12)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(panel, fontsize=15)
        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        ax.legend(fontsize=12, frameon=False)
    axes[0, 0].set_ylabel(t_label, fontsize=14)
    fig.tight_layout()
    fig.savefig(join(args.out_dir, f"power_spectrum_{beh}_occipital.png"), dpi=300)
    fig.savefig(join(args.out_dir, f"power_spectrum_{beh}_occipital.pdf"))
    plt.close(fig)
    print(f"[saved] power_spectrum_{beh}_rois / _occipital (.png/.pdf) in {args.out_dir}")


def run_plot_stage(save_root, beh, band, args):
    lobe_of = fc.load_burke_maps()
    if band == "spectrum":
        return run_spectrum_stage(save_root, beh, args, lobe_of)
    run_latency_stage(save_root, beh, band, args, lobe_of)
    elec_df = collect_electrode_table(save_root, beh, band, lobe_of, args.n_sessions)
    per_elec, tbl = subject_roi_table(elec_df, db=20.0 if fc.POWER_MODE == "hilbert" else 10.0,
                                      min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")

    stats = {m: fc.roi_stats(tbl, m) for m in MEASURES}
    contrast = fc.contrast(beh)
    fc.print_roi_stats(
        stats["cohens_d"],
        f"{contrast['hi_label']} vs {contrast['lo_label']} {band} power (Cohen's d, mean over "
        f"the ROI's electrodes), per ROI:")
    stem = f"roi_power_{beh}_{{band}}"
    fc.write_roi_csvs(args.out_dir, stem.format(band=band), tbl, per_elec, stats)
    for bars in (False, True):
        fc.band_contrast_figure(args.out_dir, stem, "cohens_d",
                                f"Power, Cohen's d ({contrast['hi_label']} vs. {contrast['lo_label']})", bars=bars)
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    fc.add_common_args(parser)
    parser.add_argument("--out-dir", default=join("figures", "burke_roi_power"))
    parser.add_argument("--min-electrodes", type=int, default=1,
                        help="min electrodes a subject must have in an ROI to enter the group test")
    args = parser.parse_args()
    args = parser.parse_args()

    if fc.POWER_MODE != "multitaper":   # longetal: hilbert figures in their own subfolder
        args.out_dir = join(args.out_dir, fc.POWER_MODE)
    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError("pass only one of --n-sessions / --n-subjects")
    return args


# --------------------------- latency (time-bin) stage -------------------------
def collect_bin_table(save_root, beh, band, lobe_of, n_sessions=None):
    """Per-(subject, electrode, time-bin) Cohen's d -> tidy frame.

    Mirrors collect_electrode_table's aggregation: sessions of one subject are
    averaged per electrode LABEL first, so a 4-session subject does not outweigh
    a 1-session subject.
    """
    files = fc.session_files(power_dir(save_root, beh, band), "_power.pkl", n_sessions)
    rows, centers = [], None
    for f in tqdm(files, desc="load bins"):
        try:
            pkl = fc.load_pickle(str(f))
        except Exception:
            continue
        if pkl.get("cohens_d_bins") is None:
            continue
        if centers is None:
            centers = np.asarray(pkl["bin_centers_ms"], float)
        sub = str(pkl["sid"][0])
        elec_rois = fc.roi_of_reg_full(pkl["reg_full"], lobe_of)
        d_bins = np.asarray(pkl["cohens_d_bins"], float)
        for label, roi, d_row in zip(pkl["labels"], elec_rois, d_bins):
            if roi is None:
                continue
            for k, val in enumerate(d_row):
                if np.isfinite(val):
                    rows.append((sub, str(label), roi, float(centers[k]), float(val)))

    if not rows:
        return pd.DataFrame(columns=["sub", "label", "roi", "bin_ms", "d"])
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "bin_ms", "d"])
    return (df.groupby(["sub", "label", "roi", "bin_ms"], as_index=False)["d"]
              .mean())


def subject_roi_bin_table(bin_df, min_electrodes=3):
    """(subject, ROI, bin) mean d, applying the same electrode floor as the
    collapsed figure so the two describe the same cells."""
    per_sub = (bin_df.groupby(["sub", "roi", "bin_ms"])
                     .agg(d=("d", "mean"), n_elec=("label", "nunique"))
                     .reset_index())
    return per_sub[per_sub["n_elec"] >= min_electrodes].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    contrast = fc.contrast(args.beh)
    prepost = fc.PREPOST_SPEC[args.beh]
    print(f"[setup] beh={args.beh}  band={args.band} {fc.bands[args.band]} Hz  ({fc.POWER_MODE})")
    print(f"[setup] {contrast['lo_label']} (lo) {prepost['pre_win']} vs {contrast['hi_label']} (hi) {prepost['post_win']}")

    if args.stage in ("compute", "both"):
        fc.run_compute_stage(
            run_sess_power, desc="ROI power", root_dir_=root_dir,
            n_sessions=args.n_sessions, n_subjects=args.n_subjects,
            workers=args.workers, save_root=save_root, beh=args.beh, band=args.band, root_dir=root_dir,
            simulation_tag=args.simulation_tag)

    if args.stage in ("plot", "both"):
        run_plot_stage(save_root, args.beh, args.band, args)


if __name__ == "__main__":
    main()
