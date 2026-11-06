# On hold / to do

## On hold
- Done (2026-10-21): region labels for depth contacts without a volumetric label now come from
  Riley's imaging-label database (recover_labels.py, copied to <scratch>/imaging_label_recovery;
  white matter / ventricles stay unlabelled); the MNI-neighbour labels only fill pairs the
  database lacks (13%). Before: ~3,600 neighbour-labelled pairs were white matter per the
  database. All region-based figures replotted.
- **Time-locked (evoked) component in synchrony.** PI: ERP subtraction is not the answer;
  review how the literature handles stimulus-locked effects in phase / envelope coupling.
  Currently `subtract_erp: false`. Also covers the early high-gamma AEC-c hubs (0–200 ms,
  within-region pairs), which may be a shared stimulus-locked envelope rise.

## Paper: references to check
- Edwards et al. 2009: confirm it supports "high gamma is broadband, not an oscillation" (Power Methods).
- Azarion et al. 2014: confirm it is the right citation for volumetric (depth) vs surface (grid/strip) labels.
- Lenth_2025 (emmeans): cited as R package v1.10.3 (installed version); update the year/version if re-run.
- Supplementary table (ADD): anatomical label -> 12-region mapping (region_to_burke_lobe.csv).

## Proposal (variability of the word response): to confirm
Status 2026-11-05. Memory analyses on hold (DeHaan 2026 thesis ch. 3 covers task structure in the SME);
connectivity stays out of the proposal (Mike decides what goes in the paper).
- **Trial-level artifact rejection in the main pipeline** (one R1323T trial makes the supramarginal spike
  in Fig. 4B; affects every analysis) -> rerun all analyses and simulations. Preliminaries already drop
  robust outliers (window mean and whole-clip peak, 10 MAD).
- **Latency reliability** (re-added; was not in the file): does an electrode's half-max latency replicate
  between session halves, as the amplitude pattern does (r up to 0.90)?
- Variance components (plot_variance_structure.py -> mixed_models.R variance, 40 subjects; DeHaan 2026
  ch. 3's random-intercept model on the single-trial response): full cohort; confidence intervals (Riley:
  glmmTMB Wald; or bootstrap over subjects); experiment (FR1 / catFR1 / pyFR) as a term (list length and
  timing differ -> could pose as subject differences); split-half reliability to separate the session
  share from sampling noise; the full time x frequency x level partition needs per-trial time-frequency
  power (not cached -> recompute).
- Response types (alpha x high gamma, up / none / down): full cohort; anatomy (MNI) and latency per type;
  whether a type is stable between session halves.
- Variance components: the full fit (all electrodes; the preliminary uses 5 random per subject x lobe:
  `--refit` after raising LMM_MAX_ELECTRODES); over time and frequency once per-trial time-frequency power exists.
- Alpha vs high gamma trial by trial (r ~ 0): full cohort; check a shared broadband / aperiodic shift
  does not drive it; extends Marks 2021 (site-level, trial-averaged, within-epoch z, no baseline).
- Single-trial latency (half-max per trial, in the variance model): replace with a template / cross-correlation
  estimate; latency reliability between session halves.
- Blank interval (plot_blank_interval.py): occipital high gamma rises with blank length (FR1 gradual,
  pyFR flat 800-1050 then a step up; all 37 pyFR subjects span the range) -> fit hazard-rate
  (expectation) vs recovery-from-adaptation models (Groen 2022). Short blanks: occipital high-gamma
  baseline slightly up and alpha baseline down (previous word's offset in the -700..-100 window)
  -> report; the response measures use the session-mean baseline. Previous word's response predicts
  the next positively (beta 0.08-0.30 with serial position controlled) = slow state, not adaptation;
  control list-level drift before interpreting.
- Task structure now in the mixed model (serial position, list, word, session x list, blank length, same
  category). Still to add: word properties (frequency, concreteness) for the item term, and the same model
  for the blank-interval and previous-word analyses (list-level drift).
- Connectivity in the same framework (decide with Riley / Mike): synchrony is per pair x session (not per
  trial), so a pair-level model: word-on - off synchrony ~ distance (spline) + (1|subject) + (1|lobe pair) +
  (1|electrode A) + (1|electrode B) + (1|pair) + (1|pair:session). Distance and lobe pair are this paper's;
  the pair / session split is FC test-retest (Riley ch. 4, joint project) -> agree the split first. Task
  factors would need per-trial synchrony (high-gamma AEC from the cached envelopes is possible but not
  orthogonalized; alpha ciPLV needs a recompute).
- Paper: cite and differentiate Kucewicz et al. 2019 (same data, 165 subjects, high-gamma latency
  gradient) and Marks et al. 2021.

## Reliability (final analysis of the paper, per PI)
- Done (2026-10-14, plot_reliability.py): test-retest as riley-thesis
  compute_retest_reliability.py (random session halves, >= 2 sessions per half, >= 10 regions,
  Pearson, medians), per anatomical region: power d (alpha 0.63, high gamma 0.79) and synchrony
  node strength (alpha ciPLV 0.16, high-gamma AEC-c 0.13); 91 subjects.
- Identifiability (own half-B profile vs other subjects'): a first try confounded stable
  electrode placement with a subject trait -- needs matched regions.
- Robustness (pick one): subject-subsampling replicability (how many patients to find each
  ROI effect again), or cross-cohort replication (RAM FR1/catFR1 vs pyFR).
- Supplement if asked: split-half (odd/even trials) per measure; power from the stored
  per-trial values, synchrony would need a recompute with trial splits.
- **Individual differences (later)**: subject recall rate vs onset measures (alpha d, high-gamma
  d, latency per lobe). Decide how to handle experiment (FR1 / catFR1 / pyFR recall rates
  differ) and electrode coverage before testing.

## Latency
- Done (2026-10-19): pairwise lobe tests, mixed model with fine regions as random effects
  (mixed_models.R latency). Next: show them on the latency figure (brackets or letters).

## Spectral tilt
- Frontal: low up / beta down; occipital: alpha down / gamma up. Test directly with specparam
  (aperiodic exponent change, word on vs off, per ROI) + periodic peaks (alpha) separately;
  small extra figure. Needs compute.py to store per-electrode word-off / word-on spectra
  (trial-averaged, 1 Hz) -> next recompute.

## Recall decoding (plot_decoding.py)
- Current: encoding contrast, high gamma per lobe x 100 ms window, >= 5 electrodes per lobe,
  >= 3 sessions per montage. On hold with the memory analyses.
- Done (2026-11-03): per-session AUC vs 0.5, nested CV over C, per montage (DeHaan et al. 2025).
- Check whether the small above-chance recall information at word onset (frontal / parietal,
  vs the permutation level) is serial-position / encoding-state drift (add serial position).

## Open
- **Spectrum 64 Hz bin**: the 60 Hz notch ringing came from the 50 ms buffer; recheck the
  64 Hz bin with the 500 ms buffer before dropping this item.
- **Alpha in 200 ms epochs**: one taper, ±3 Hz smoothing; PI caveat on timing at low
  frequencies. Option: Hilbert phase binned per epoch (Rao / Solomon).
- **Frontal alpha power increase**: check evoked contribution and fine-region breakdown.

## Methods notes / deviations to report
- Region-pair floor 100 subjects (MIN_SUBJECTS_PAIR, as riley-thesis).
- Synchrony includes within-region pairs (contact-sharing pairs excluded).
- Localization caveats (Riley): grids/strips before ~R1350 not brain-shift corrected;
  FreeSurfer 5.3; provenance of the "mni" coordinates.
- 9 sessions excluded for non-mains 100 Hz peaks; R1171M_FR1_2 fails event extraction.

## Cleanup
- `match_events.py`: encoding/retrieval matching code now unused.
- ~30 GB raw EEG from the null simulations under `bids_data/` (delete by hand).
