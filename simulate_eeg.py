"""simulate_eeg.py -- synthetic EEG for the pipeline validity checks.

helper.get_eeg calls replace_w_simulated_EEG when --simulation-tag is set: the
session's real montage, regions and events are kept (so the whole real pipeline
is exercised) and only the signal is replaced. Data-generating processes,
parameterised in config/simulation_config.yaml (each on top of pink noise):

  hg_power     band-limited noise on every channel; on target-lobe channels the
               band part is scaled by hg_gain from word onset on.
  osc_lag      sustained oscillation, phases drawn per event from a wrapped normal
               (target pairs at target_ppc0 pre / target_ppc1 post, all other pairs
               at global_ppc), every other target channel lagged by phase_lag
               -> ciPLV (alpha) should rise.
  hg_envelope  independent band-limited carriers; on post events target
               channels share a slow amplitude envelope -> AEC-c should rise.
  leak         zero-lag common source on target channels, post events only
               (volume conduction) -> ciPLV and AEC-c should stay at 0.

A list of processes (e.g. [osc_lag, hg_envelope]) adds them up. hg_envelope option
`equal_conditions: true` shares the envelope on pre events too (equal coupling, null).

Optional `active_ms: [a, b]` restricts the post-onset effect (osc_lag, hg_envelope,
hg_power) to a <= t < b ms, to test the epoch network's / decoder's timing.

hg_power options: `onset_ms: {lobe: ms}` starts each target lobe's gain at its own time
(latency order); `recall_gain` multiplies later-recalled words by a further gain (the
recall decoder).

Optional `line_amplitude` adds a 60 Hz mains sinusoid (random phase per event,
shared by all channels), to test the notch filter on line noise.

The random seed is a CRC of the real clip, so reruns reproduce the same signal.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]

from helper import get_pairs, regionalize_electrodes_by_type

NDArrayAny = np.ndarray
RAMP_MS = 50   # osc_lag: phase transition at the active_ms edges
_SIM_CONFIG_PATH = Path(__file__).parent / "config" / "simulation_config.yaml"


def _load_simulation_parameters() -> dict[str, dict[str, Any]]:
    """Named parameter sets; each `sweeps` value becomes its own tag `<sweep>_<value>`."""
    with open(_SIM_CONFIG_PATH) as f:
        params = yaml.safe_load(f)
    for name, sweep in params.pop("sweeps", {}).items():
        for value in sweep["values"]:
            params[f"{name}_{value}"] = {**sweep["params"], **{key: value for key in sweep["keys"]},
                                         "sweep": name, "planted": value}
    return params


simulation_parameters = _load_simulation_parameters()
# {sweep name: [its tags in value order]}, for the Snakefile and plot_recovery.py
SWEEPS: dict[str, list[str]] = {}
for _tag, _params in simulation_parameters.items():
    if "sweep" in _params:
        SWEEPS.setdefault(_params["sweep"], []).append(_tag)


def generate_pink_noise(n: int, amplitude: float, exponent: float, n_ref: int = 1000,
                        n_gen: int = 8192) -> NDArrayAny:
    """1/f^exponent noise of length n (global np.random state). Scaled so its spectrum is the
    same for any n (amplitude = total std at n_ref samples): normalising each clip to a fixed
    total variance instead moved power out of the analysed band in longer clips (the word-on
    clip is longer than the word-off clip: a broadband word-on decrease in the null).
    Generated at a fixed length n_gen (>= every clip) and cropped: FFT noise is periodic over
    its length, so per-clip lengths gave the longer clip more low-frequency power (+0.13% at 6 Hz)."""
    n_fft = max(n_gen, n + (n % 2))
    scales = np.linspace(0, 0.5, n_fft // 2 + 1)[1:] ** (-exponent / 2)
    spectrum = np.concatenate([[0], np.random.normal(scale=scales) * np.exp(2j * np.pi * np.random.random(n_fft // 2))])
    ref_scales = np.linspace(0, 0.5, n_ref // 2 + 1)[1:] ** (-exponent / 2)
    sigma = np.sqrt(2 * np.sum(ref_scales ** 2)) / n_ref * np.sqrt(n_ref / n_fft)
    return (amplitude * np.fft.irfft(spectrum) / sigma)[:n]


def _pink(shape: tuple[int, ...], p: dict[str, Any]) -> NDArrayAny:
    """Independent pink noise per (event, channel) row."""
    if not p["pinknoise_amplitude"]:
        return np.zeros(shape)
    rows = [generate_pink_noise(shape[-1], p["pinknoise_amplitude"], p["pinknoise_exponent"])
            for _ in range(int(np.prod(shape[:-1])))]
    return np.reshape(rows, shape)


def _channel_lobes(regionalizations: Sequence[Any]) -> NDArrayAny:
    """Burke lobe of each channel's region ('L frontal pole' -> 'frontal'), None if unmapped."""
    lobe_of = pd.read_csv(Path(__file__).resolve().parent / 'region_to_burke_lobe.csv').set_index('region')['burke_lobe']
    return np.array([lobe_of.get(region.split(' ', 1)[1]) if isinstance(region, str) else None
                     for region in regionalizations], dtype=object)


def _phase_cov(target: NDArrayAny, target_ppc: float, global_ppc: float) -> NDArrayAny:
    """Wrapped-normal covariance whose pairwise PPC (= exp(-sigma^2)) is target_ppc
    between target channels and global_ppc otherwise."""
    ppc = np.where(np.outer(target, target), target_ppc, global_ppc)
    np.fill_diagonal(ppc, 1.0)
    var = np.triu(-np.log(ppc), 1)                    # pair phase-difference variance
    cov = np.full(ppc.shape, var.max()) - (var + var.T) / 2
    np.linalg.cholesky(cov)                           # raises if not positive definite
    return cov


def replace_w_simulated_EEG(eeg: TimeSeries, dfrow: pd.Series, condition_mask: NDArrayAny,
                            simulation_tag: str, events: pd.DataFrame | None = None) -> TimeSeries:
    """Same shape/coords as `eeg` (time in ms), signal from `simulation_tag`.
    condition_mask True = post-onset events (osc_lag: target_ppc1), False = pre; `events`
    (indexed by eeg.event) gives correct_recall for `recall_gain`."""
    params = simulation_parameters[simulation_tag]
    np.random.seed(zlib.crc32(np.ascontiguousarray(eeg.data).tobytes()))
    import fc_comparison_functions as fc   # the analyses' labels (recovered / MNI-neighbour), not the raw atlas
    regions = fc.session_regions(dfrow["sub"], dfrow["exp"], str(dfrow["sess"]))
    lobes = _channel_lobes(regionalize_electrodes_by_type(get_pairs(dfrow)) if regions is None else regions)
    target = np.isin(lobes, params["target_lobes"])
    t_ms = np.asarray(eeg.time, float)
    data = _pink(eeg.shape, params)
    active = ((t_ms >= params["active_ms"][0]) & (t_ms < params["active_ms"][1])) if "active_ms" in params \
        else np.ones(t_ms.size, bool)
    for dgp in np.atleast_1d(params["data_generating_process"]):   # several processes add up
        if dgp == "hg_power":
            import mne
            hg = params["hg_amplitude"] * mne.filter.filter_data(np.random.standard_normal(eeg.shape),
                                                                 float(eeg.samplerate), *params["band"], verbose=False)
            onset_ms = np.array([params.get("onset_ms", {}).get(lb, 0) for lb in lobes])[target]
            gain_on = (t_ms[None, :] >= onset_ms[:, None]) & active[None, :]   # (target ch, time)
            hg[:, target] *= np.where(gain_on, params["hg_gain"], 1.0)
            if params.get("recall_gain"):   # for the recall decoder
                recalled = events.loc[np.asarray(eeg.event), "correct_recall"].fillna(0).to_numpy(bool)
                hg[np.ix_(recalled, target)] *= np.where(gain_on, params["recall_gain"], 1.0)
            data += hg
        elif dgp == "osc_lag":
            # every other target channel lagged: ciPLV sees only the lagged pairs
            is_post = np.asarray(condition_mask, bool)
            lag = np.where(target, np.cumsum(target) % 2 * params["phase_lag"], 0.0)
            # active_ms: target channels move from their outside phases to coupled ones over a
            # RAMP_MS raised-cosine ramp (a phase jump would leak power out of the band), and
            # non-target channels keep theirs (a redraw changed alpha power on every channel)
            ramp = np.ones(t_ms.size)
            if "active_ms" in params:
                n_ramp = max(1, int(round(RAMP_MS * float(eeg.samplerate) / 1000)))
                ramp = np.convolve(np.pad(active.astype(float), n_ramp, mode="edge"), np.hanning(n_ramp) / np.hanning(n_ramp).sum(), "same")[n_ramp:-n_ramp]
            def draw(key, n):   # (n, n_ch, 1) per-event phases
                cov = _phase_cov(target, params[key], params["global_ppc"])
                return (np.random.multivariate_normal(np.zeros(len(cov)), cov, size=n) + lag)[:, :, None]
            for sel, key in ((~is_post, "target_ppc0"), (is_post, "target_ppc1")):
                if sel.any():
                    n = int(sel.sum())
                    outside, coupled = draw("target_ppc0", n), draw(key, n)
                    coupled = np.where(target[None, :, None], coupled, outside)
                    phase = outside + ramp * np.angle(np.exp(1j * (coupled - outside)))
                    data[sel] += params["osc_amplitude"] * np.cos(2 * np.pi * params["oscillation_frequency"]
                                                                  * t_ms[None, None, :] / 1000.0 + phase)
        elif dgp == "hg_envelope":
            import mne
            sf = float(eeg.samplerate)
            carrier = mne.filter.filter_data(np.random.standard_normal(eeg.shape), sf, *params["band"], verbose=False)
            # slow envelope filtered on a 2 s-padded segment, then cropped: the low-pass
            # is longer than the shorter clips, so its edges must not reach the clip
            pad = int(2 * sf)
            env = mne.filter.filter_data(np.random.standard_normal((*eeg.shape[:-1], eeg.shape[-1] + 2 * pad)),
                                         sf, None, params["env_hz"], verbose=False)[..., pad:-pad]
            shared = np.ones(eeg.shape[0], bool) if params.get("equal_conditions") else np.asarray(condition_mask, bool)
            own_env = env[np.ix_(shared, target)]
            env[np.ix_(shared, target)] = np.where(active, own_env[:, :1], own_env)   # first target channel's envelope
            env /= np.sqrt(2 * params["env_hz"] / sf)   # ~unit variance; a fixed scale, not the clip's own SD
            # (per-clip scaling made the longer word-on clip's envelope weaker -> AEC-c biased low)
            data += params["hg_amplitude"] * carrier * np.exp(params["env_depth"] * env)
        elif dgp == "leak":
            is_post = np.asarray(condition_mask, bool)
            source = _pink((int(is_post.sum()), 1, eeg.shape[-1]), params)
            data[np.ix_(is_post, target)] += params["leak_gain"] * source
        else:
            raise ValueError(f"unknown data_generating_process {dgp!r}")
    if params.get("line_amplitude"):   # mains, one phase per event shared by all channels
        phase = np.random.uniform(0, 2 * np.pi, (eeg.shape[0], 1, 1))
        data += params["line_amplitude"] * np.sin(2 * np.pi * 60.0 * t_ms[None, None, :] / 1000.0 + phase)
    return eeg.copy(data=data)
