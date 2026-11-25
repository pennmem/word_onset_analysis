# word_onset_analysis

Word-onset spectral power and phase connectivity in human intracranial EEG,
across 12 ROIs (used in Burke et. al. 2013), from the Penn Computational Memory Lab free-recall
datasets (FR1 / catFR1 / pyFR).

The contrast used is **`word_on`**: activity while a word is on
the screen (0–600 ms after onset) versus the blank inter-stimulus interval
before it (−700 to −100 ms), for the same events. 
---

## Requirements

Python ≥ 3.11 and `pip install -r requirements.txt`. Two of the pins come
from GitHub rather than PyPI: `ptsa` (needs a C++ compiler and FFTW —
`brew install fftw` on macOS, then `CPPFLAGS=-I$(brew --prefix fftw)/include
LDFLAGS=-L$(brew --prefix fftw)/lib pip install ...`) and `bidsreader`, CML's
reader for the BIDS exports of its datasets.

## Data

Everything is read from the lab's public BIDS datasets on OpenNeuro
(FR1 `ds004789`, catFR1 `ds004809`, pyFR `ds004865`); nothing needs the lab
cluster. `cml_data.py` lists a dataset on S3 and downloads what a session needs
on first use — events / channel / electrode tables (~1 MB) and, for the
compute stages, the bipolar recording (300–700 MB per session) — into
`config.yaml paths.bids_cache` (default `~/scratch/word_onset_analysis/bids_data`;
`CML_BIDS_CACHE` overrides). Files are never re-downloaded.
You are asked once per run before anything is fetched; set
`CML_AUTO_APPROVE=1` for unattended runs.

Not every rhino session is on OpenNeuro yet (as of 2026-09: FR1 149/153 of the
original subjects, catFR1 31/33, pyFR 42/80). The session list is whatever
OpenNeuro lists, so the cohort grows as the release is updated.

## To run

```bash
zsh run_project.sh --dry-run          # what would run (add --smokescreen)
zsh run_project.sh --smokescreen      # plot stages on 40 random subjects of the existing cache -> *.smokescreen
zsh run_project.sh                    # full run: sessions -> compute -> proposal figures -> proposal/proposal.pdf
zsh run_project.sh --target timing    # one stage (and what it needs)
snakemake paper / simulations --cores 4   # paper figures (on hold) / validity checks -> figures/simulations/
# add --config longetal=true for the Long et al. 2020 replication (power only) -> *.longetal
```

Proposal stages, in order (Snakefile; each figure's rules in plot_variance_structure.py's docstring):
`plot_boxplots.py --measure power` (1A), `plot_variance_structure.py` (1B, 2A, 3D, S1B, noise ceilings),
`--single` (1C), `plot_timing.py` (2B-D, 3A-C, 3E, S2, S3), `--recall` (4), `plot_variability.py` (S1A),
then the figures are copied to `proposal/img/` and `proposal.tex` is recompiled.

`prepare_sessions.py` writes the session list and per-session artifacts under
`config.yaml paths.scratch_dir` (default `~/scratch/word_onset_analysis/scratch`,
outside the repo; longetal runs use `scratch.longetal`).
`--subjects` / `--n-subjects` restrict it. Sessions whose 100/120/150 Hz mains
harmonics exceed 5x their neighbours are excluded (`max_line_harmonic_ratio`).

Every build script has `--stage compute` (one pickle per session under
`SCRATCH_DIR/word_on/…`, `--workers N` in parallel) and `--stage plot`
(figures + CSVs); `--stage both` is the default. Compute is cached on the file
existing, **not** on the settings that produced it — after changing anything
upstream (band, notch, buffers, windows) delete the output directory.

Simulations (`--simulation-tag`, config/simulation_config.yaml) keep each real
session's montage and events and replace only the signal, so they exercise the
whole real pipeline.

---

## Figures (word on − word off only)

Each plot stage redraws one figure per analysis with a row per band already
plotted into its directory (alpha, high gamma), paired t per ROI, BH-FDR.
---

## Defaults

| parameter | value | note |
|---|---|---|
| `resample_hz` | 500 | Nyquist 250 Hz |
| `min_sample_rate_hz` | 499 | sessions below this are excluded upstream |
| `bands.high_gamma` | 70–150 Hz | Long et al.'s band |
| `bands.gamma` | 70–110 Hz | AEC-c band |
| `notch_harmonics_up_to_hz` | null | mains fundamental only (as riley-thesis); data_check records harmonic peaks |
| `mt_bandwidth` | 2 | NW = 0.6 over a 600 ms window |
| `time_bin_ms` / `mt_window_ms` | 50 / 100 | gamma time course: bin step / sliding window |
| `real_data_buffer_ms` | 50 | real EEG around each window (riley-thesis), cropped before estimation |

---

## Layout

```
prepare_sessions.py         session list, data check, events (run first)
compute.py                  per session, one load: alpha + spectrum power (multitaper), high-gamma
                            power (Hilbert sub-bands, + per-trial envelopes), alpha ciPLV, high-gamma AEC-c
plot_boxplots.py            power + synchrony per ROI (subject and mean +/- CI versions)
plot_spectrum.py            t per ROI x frequency bin
plot_epoch_network.py       200 ms epoch synchrony network (hubs + edges)
plot_latency.py             high-gamma half-max latency / order of activation
plot_anticipation.py        high-gamma response after short vs long waits (paper)
plot_variance_structure.py  proposal: variance components, alpha vs high gamma, carry-over, single subjects, SME;
                            every figure's event rules and power units
plot_timing.py              proposal: waveforms, latency lags, ISI effects, ISI histogram
plot_variability.py         proposal: electrode test-retest
mixed_models.R              lme4 fits for the variance, single-subject and recall models
plot_recovery.py            simulation recovery curves
build_roi_power.py          Long et al. replication only (legacy, longetal=true)

fc_comparison_functions.py  connectivity estimators, ROI stats, figures, session dispatch
helper.py                   BIDS EEG / electrode loading, resample + notch, regionalisation
simulate_eeg.py             synthetic EEG for the validity checks
cml_data.py                 OpenNeuro listing / download / cache
data_check.py               per-session validation -> sess_list_df_data_check.json
load_events.py  match_events.py  word_on events (+ recall matching for the cohort rule)
exclusion_log.py            event/session exclusion counts
misc.py  project_paths.py  ptsa_patches.py   pickles, config constants, PTSA numpy patch
config/config.yaml          all tunable parameters
```
