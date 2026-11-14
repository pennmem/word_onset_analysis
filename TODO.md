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

## Gap / prior work (2026-11-11 search; verify before citing)
- Long vs short pre-stimulus period (Fig. 3A) is known in visual cortex: short ISI -> smaller visual-cortex response,
  recovery with longer ISI (Groen et al. 2022 J Neurosci, ECoG broadband; Brands et al. 2024 PLoS Comput Biol, slower
  recovery in higher visual areas); V1 high gamma (80-150 Hz) smaller after shorter ISI, more prior stimuli and lower
  pre-stimulus high gamma ("refractory"; Asano group, 2011 abstract / PubMed 21945696 -- confirm). Contrast: stimulus
  on vs baseline, flashes / images, not words in a memory task.
- Drift / autocorrelation (Fig. 3B neighbours): Manning et al. 2011 PNAS (pattern similarity, context reinstatement at
  recall); Folkerts et al. 2018 Nat Commun (temporal-lobe drift); Kulkarni & Lega (eLife reviewed preprint 90576,
  representational drift and boundaries in free recall). All multivariate pattern drift tied to memory; Fig. 3B is
  univariate response-size carry-over. Riley also mentioned a Broitman paper (scalp EEG context reinstatement found;
  check which one he meant).
- Ongoing vs evoked: He 2013 J Neurosci (fMRI, negative interaction); He & Zempel 2013 PLoS Comput Biol (ECoG,
  inverted-U pre- and post-stimulus vs behaviour); iEEG variability quenching blended with pre-stimulus dynamics
  (Commun Biol 2021). Circularity caution when correlating pre-stimulus and evoked activity of the same signal.
- Memory-relevant brain state: Ezzyat et al. 2017 Curr Biol / 2018 Nat Commun (encoding-state classifier, closed-loop
  stimulation) -- the multivariate precedent for "state predicts recall".
- No single-trial variance decomposition of the iEEG word response found (subject / electrode / event).

## Paper: references to check
- Edwards et al. 2009: confirm it supports "high gamma is broadband, not an oscillation" (Power Methods).
- Azarion et al. 2014: confirm it is the right citation for volumetric (depth) vs surface (grid/strip) labels.
- Lenth_2025 (emmeans): cited as R package v1.10.3 (installed version); update the year/version if re-run.
- Supplementary table (ADD): anatomical label -> 12-region mapping (region_to_burke_lobe.csv).

## Proposal (variability of the word response): to confirm
Status 2026-11-06. Memory analyses on hold (DeHaan 2026 thesis ch. 3 covers task structure in the SME);
connectivity is Aim 4 since 2026-11-08 (paper's synchrony figures; variance split planned). These items are kept here, not
in the proposal text. Every model or analysis choice and its run cost goes to the user before running.

**Data coverage (what does not use all the data yet)**
- Variance model (Figs. 1B, 2B): at most 5 random electrodes per subject x lobe. Power: frontal 19% of electrodes,
  temporal 20%, parietal 36%, limbic 45%, occipital 55%, hippocampus 66%. Latency: 69-93% of responsive electrodes.
  Task effects, coupling and prevalence already use all electrodes.
- Serial positions 13-15 of pyFR (1.6% of events, 21% of pyFR's) are dropped from the model, task effects and coupling:
  decide whether to keep them. Position 1 (no blank baseline) and practice lists stay out on purpose.
- Fig. 3B needs >= 5 electrodes in the lobe (135/254 subjects occipital, 157/262 hippocampus); Fig. 3A needs >= 2
  sessions as well (99-246 subjects). Inherent to the measures unless the minimum changes.

**Model**
- PI (2026-11-09): is (1|electrode) collinear with (1|subject)? Electrode is nested in subject, so the two are
  separable only through subjects with >= 2 electrodes in the ROI. Check: subject-only, electrode-only and both
  (expect electrode-only's electrode share ~ subject + electrode of the full model, same total); one ROI x band on
  the 40-subject pilot first (~10-15 min per fit).
- Planned (Aim 1, no panel; Fig. 5C is now pre-stimulus high gamma long - short, done 2026-11-10): the same variance split for power in the pre-stimulus blank alone ((word-off power - mean
  word-off power) / pooled SD per event; word-off = the existing -700..-100 ms window, as DeHaan ch. 4), control for the word-event component. ~6 h for the 12 power fits; confirm before running.
- Done 2026-11-09 (Fig. 5B, prelim): previous-word model (mixed_models.R blank): prev_resp x blank_len +
  prev2_resp + the 13 random terms, 12 fits ~4 h; kept as the check. Figure since 2026-11-10: per-subject lag-1 / lag-2 r,
  all electrodes (carryover.png); carry-over beyond drift in every ROI and band, no blank-length change.
- Synchrony test-retest (session_test_retest.png; median r 0.19 / 0.14 vs power 0.67 / 0.81) removed from proposal
  Fig. 6 on 2026-11-10; decide later where (if anywhere) it goes.
- Done 2026-11-06: per-event Cohen's d, DeHaan 2026 ch. 3 terms + electrode, electrode in session, word event.
  Single-event d has heavier tails than log power (excess kurtosis alpha 5.1 vs 0.3, high gamma 2.2 vs 0.4): check
  whether the shares depend on the tails (residuals; rank-based or log comparison).
- Confidence intervals for the variance shares (Riley: glmmTMB Wald; or bootstrap over subjects); experiment
  (FR1 / catFR1 / pyFR) as a term (list length and timing differ -> could pose as subject differences).
- The session share includes the sampling noise of each session's average; split-half reliability separates it.
  Over time and frequency the split needs per-event time-frequency power (not cached -> recompute).
- Word properties (frequency, concreteness) for the item term.

- Region within lobe (pilot 2026-11-10, occipital, 52 subjects; proposal Fig. 1B occipital row): full run adds
  `(1 | region)` to all variance fits, only regions with >= MIN_SUBJECTS_ROI (30) subjects, as Fig. 3A (latency:
  subjects with responsive electrodes; one calcarine electrode at 460 ms inflated the first latency pilot). Decide
  the same rule for power (all electrodes) and whether electrodes outside those regions are dropped or pooled. Only 4-5 occipital regions (fusiform most electrodes;
  medial / striate under-sampled). In high gamma region also takes from subject (subregion coverage differs between
  subjects). Pilot subject share (11%) differs from the full run's (8%): update the Aim 1 numbers after the full run.
- Band coupling by timescale (Fig. 2C): word-event r is per electrode, serial-position / list r from lobe-averaged
  means (all electrodes) -> averaging favours slow coupling; matched check: per-electrode slow r (trials.csv). Slow r from 11 noisy
  position means is attenuated by noise: split-half reliability of the curves to correct. MTL positive coupling at
  every timescale: broadband / aperiodic shift? (see Analyses: alpha vs high gamma).
- Carry-over (Fig. 5B): mixed model with Aim 1's terms agrees with the correlations (prev x blank |b| <= 0.003) --
  methods / supplement, not the proposal text.

- Baseline check for Fig. 1 (2026-11-11): the response uses the electrode-session's mean word-off; refit the variance
  model with each event's own word-off (pa / pg, already computed in the pass) -> does the word-event share shrink
  (slow state present before and after the word)? Hours for all regions; confirm before running.
- Single-subject variance fits (Riley): occipital pilot done 2026-11-11 (99 subjects; scratch single_subject/): individual
  splits match the group; high-gamma electrode share 0-50% across subjects; region large in a few. Decide: Fig. 1 panel
  or supplement; other lobes; move the scripts into the repo. Older related figures: variance_partition.png
  (method of moments, stable shares over time / frequency) and carpet_example.png (git bbbc6cc).
- Timing in every magnitude analysis (user 2026-11-11): (1) wait -> latency done (no change); (2) does the momentary
  state change timing too (event-level amplitude vs latency, same electrode, as 2C's word-event column; vision: stronger
  stimuli -> larger and earlier); (3) is timing variability inherited from early visual cortex or added per stage
  (same word, occipital vs temporal / frontal latency) -> pairs with Fig. 1C (latency mostly unexplained) to question
  the feedforward picture. (2) and (3) need the cross-correlation latency first.
  Done 2026-11-11 (scratch timing/): (3) frontal timing ~independent of occipital (corrected r 0.08; temporal 0.30);
  caveats: reliability 0.17-0.34, group-level correction, different subjects per pair. (2) with each event's peak: r 0.01-0.04,
  size and timing nearly independent (the window-mean version, -0.06 to -0.17, was an artefact).
  Waveforms (scratch timing/waveforms.py): words split by occipital latency -> temporal ~50 ms earlier, frontal /
  limbic / hippocampus overlap. Early-occipital words are already elevated at onset (~0.12 vs < 0 for late): part of
  "early" is the pre-stimulus level (state) -> test with the pre-stimulus-level analysis; caveat for the latency
  estimate (template matching may lock onto an elevated start). Move the timing scripts into the repo. Narrative idea: the paper around
  timing / waiting / lengths.
- Word offset effects (user 2026-11-11, raised by Fig. 4A): after short waits occipital high gamma is still elevated at
  -700 ms, decaying by ~-500 ms -- the tail of the previous word's offset response. Plan: occipital high gamma aligned to
  the previous word's offset (time since it disappeared), short vs long waits; size of the offset response and when it
  has decayed vs the next word's onset. section 4 (offset_recovery.png) already re-aligns the word-off window but pools
  waits and has the coverage problem below. Done 2026-11-11 (offset.py): the quartiles collapse onto one offset-aligned
  decay (over ~300 ms after the offset). Open: the small onset-aligned rise in the last ~200 ms (anticipation?).
  2026-11-12 pilot (scratch timing/anticipation.py: offset- + onset-locked + period x onset-locked kernels) suggests
  the rise grows with the period (temporal q = .01, occipital n.s.), but with periods only 800-1000 ms the split from a
  slow offset-locked drift rests on an assumption; parked (user: "maybe not the best"). Also: the
  offset itself and the first ~100 ms after it are not cached (env_hi ends at 1000 ms, words stay 1.6 s).
- Fig. 2A cross-correlation latency (2026-11-12): latency = the electrode-session template's half-max + the best lag,
  so the template's half-max is a constant per electrode-session. The large electrode-in-session share (22-25% in
  temporal, limbic, hippocampus; 0.5-2% in occipital, parietal) may be that half-max varying between sessions where the
  average response is weak. Check: refit on the lag alone, or with one template per electrode across sessions.
- Unifying figure (2026-11-11): every factor as % of single-event variance explained, same units as Fig. 1, per band:
  subregion, serial position, list, pre-stimulus wait length, previous word's response, pre-stimulus level, shared
  word-event component; external vs internal. Most are model terms already; wait length / previous word / pre-stimulus
  level as extra terms.
- Pre-stimulus level as internal state: per event, the pre-stimulus HG / alpha level (cached word-off window) vs
  that event's response (per electrode-session r, as 3B) -> partner panel to Fig. 3A (wait length = external timing).
- 2C as "one state or several": done 2026-11-11 by electrode halves (corrected r -0.13 to 0.19; ceilings.py); the
  BLUP version from the Fig. 1 models is an alternative. Riley: the same pattern across regions in all 3 timescale columns looked odd -> check.
- Final figure: does any of it matter for behaviour -- do the internal-state parts (shared word-event component,
  pre-stimulus level) vs the external parts predict recall?

**Data quality**
- Fig. 5A response peak: parietal mean +0.07 with a large SEM from R1323T (4.4; the known artifact subject below);
  not significant. Recheck after artifact rejection.
- **Artifact rejection per word event in the main pipeline** (one R1323T word event makes the supramarginal spike
  in Fig. 2A; affects every analysis) -> rerun all analyses and simulations. Preliminaries already drop robust
  outliers (window mean and whole-clip peak, 10 MAD).

**Analyses**
- Alpha vs high gamma: |d| r across electrodes is 0.46 in occipital, 0.13-0.21 elsewhere; occipital alpha goes both
  ways where high gamma rises (19% down, 8% up). Check whether this is real or signal quality, and that a shared
  broadband / aperiodic shift does not drive r ~ 0 across word events; extends Marks 2021.
- Response types (alpha x high gamma, up / none / down): anatomy (MNI), latency per type, stability between
  session halves; within-region anatomical order as Pinheiro-Chagas 2024 (arithmetic vs baseline; math vs non-math).
- Latency per word event (half-max; SD ~180 ms): replace with a template / cross-correlation estimate (Woody 1967:
  slide the electrode's session-average curve over each event, best lag; choices: lag range +/-200 ms, Woody
  iterations or not, events with weak best r (< 0.2?) -> NaN). ~15 min (one pass over the cached envelopes in
  plot_variance_structure.py) + ~1 h latency variance refit + region pilot redo; Fig. 3A unchanged; does an
  electrode's latency replicate between session halves, as the amplitude pattern does (r up to 0.90)?
- What the shared word-event component is (arousal, attention, common reference).
- Blank interval (plot_variance_structure.py sections 4-5): dropped from the proposal 2026-11-06. The blank model now
  uses per-event d; the offset curve still uses the envelope and log alpha. Hazard-rate (expectation) vs recovery-from-adaptation fits (Groen 2022); whether short
  blanks contaminate the word-off baseline -> paper's methods. Offset curve: each time bin only has the word
  events whose word-off window covers it (short blanks early, long late); coverage cut or per-design panels.
- Connectivity in the same framework (decide with Riley / Mike): synchrony is per pair x session (not per word
  event), so a pair-level model: word-on - off synchrony ~ distance (spline) + (1|subject) + (1|lobe pair) +
  (1|electrode A) + (1|electrode B) + (1|pair) + (1|pair:session). Distance and lobe pair are this paper's;
  the pair / session split is FC test-retest (Riley ch. 4, joint project) -> agree the split first. Task
  factors would need per-event synchrony (high-gamma AEC from the cached envelopes is possible but not
  orthogonalized; alpha ciPLV needs a recompute).
- Paper: cite and differentiate Kucewicz et al. 2019 (same data, 165 subjects, high-gamma latency
  gradient) and Marks et al. 2021. Check the Methods line "12 trials of 12 words (25 trials of 15 words in the
  variant)": the events show up to 25 lists of 12 words (FR1, catFR1) and up to 16 lists of 15 words (pyFR).

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
- `build_roi_power.py --band spectrum` plot stage: stale (bin count no longer matches FREQ_EDGES; crashes);
  the paper's spectrum comes from `plot_spectrum.py`. Its ROI box plots duplicate `plot_boxplots.py` with another y label.
- ~30 GB raw EEG from the null simulations under `bids_data/` (delete by hand).
