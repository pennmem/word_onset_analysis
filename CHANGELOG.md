# Changelog

Project-level change log for `phase_visualization`. Newest entries first.

Entries before 2026-08-17 predate git and are not reconstructable; anything
earlier than the last entry below is undocumented.

---

## 2026-11-09 — previous-word model (Fig. 5B) on the variance model's terms

- `mixed_models.R blank`: `y ~ prev_resp + prev2_resp + blank_len + prev_resp:blank_len` + the variance model's 13 random intercepts (shared helper `add_terms` / `rand`). Dropped: the blank bins per design, the blank after the word, the old 5 random terms. `prev2_resp` (word two back) separates drift from carry-over; no electrode-in-list term (biases a lagged slope, Nickell 1981). Lags now computed before the blank-range cut. Serial positions 2-3 drop out (no word two back).
- `plot_variance_structure.py` section 5 reads the new term names. Occipital pilot (40 subjects, seed 0): 6.5 min for both bands; full run (all subjects, up to 5 electrodes per subject x ROI) started.
- Full run done (12 fits, 31-133 min each, ~4 h wall; 254-339 subjects, 346k-646k word events; all but occipital alpha singular). Previous word's response: high gamma 0.020-0.055, alpha 0.008-0.029 d per SD (all q < .05); two words back 46-80% of that (drift + carry-over); previous word x blank length |b| <= 0.003, mixed sign (q < .05 only temporal alpha, limbic and hippocampus high gamma). `blank_interval.png` -> proposal Fig. 5B (Prelim), Aim 3 text added.
- 2026-11-10: proposal Fig. 6 drops the session test-retest panel (and the Aim 4 sentence citing it); planned synchrony split is now 6B. Fig. 5A is now a per-subject box plot (`plot_anticipation.py` also saves `hg_anticipation_word_on_box.png`; y axis -0.5..0.5; stats unchanged). Paper Fig. 9 keeps the bars.
- `fc.roi_panel(ylim=)`: crops (never extends) the y axis, stars kept at the edge. Subject box plots (`band_contrast_figure`, paper Fig. S1 / proposal 1A) capped at 1.0 (occipital high gamma ran to 2.8); synchrony (S2) unchanged. Paper and proposal recompiled.
- Fig. 5B by correlation, all electrodes (user: slopes in "d per SD" unclear; box plots + stars like the other figures): `plot_variance_structure.py` computes per electrode-session (serial-position means out) r with the word 1 / 2 back in the same list and r with the previous word after long vs short blanks (session median, as 5A); sessions -> electrode -> subject x lobe, `fc.roi_stats` -> `carryover.png` (high gamma) + `carryover_stats.csv` (both bands, r1 - r2 paired). Rebuild of the per-event tables (361 subjects, 26,762 electrodes) ~10 min, no model refits. High gamma r1 0.056-0.124, r2 0.040-0.105, r1 - r2 0.014-0.024; alpha r1 0.026-0.053, r1 - r2 0.007-0.020; all q < .05 (occipital alpha now carry-over too); long - short |dr| <= 0.008, n.s. everywhere. Mixed-model slope plot (`blank_interval.png`) dropped; its coefficients stay as the check. Proposal 5B (Done) and Aim 3 text updated. Then (user) one axis instead of three box-plot rows: categories on x (previous word, two back, long - short blank), per-lobe mean r +/- 95% CI across subjects, FDR stars above. Third category labelled "long − short wait" (was blank; caption too). Proposal text: "blank" -> "pre-stimulus period".
- Fig. 5C = pre-stimulus high gamma (replaces the planned pre-stimulus variance split, which stays a Planned item in Aim 1): `plot_anticipation.py` also takes the cached word-off envelope (`env_lo`, -700..-100 ms, inside every blank; no recompute), long - short per responsive electrode, as the peak -> `hg_anticipation_word_on_pre.png`, `_pre_stats.csv`. Occipital -0.010 (q = .004; peak +0.058), other lobes n.s.: the long-blank effect is in the response, not a pre-stimulus level. Aim 3 text: 5A/5C sentence; carry-over "does not explain the long-blank effect"; Planned line now adaptation vs expectation. Then (user) panels reordered: B = pre-stimulus (`img/fig5b_prestimulus_box.png`), C = carry-over; text references swapped.
- Fig. 5 now two dot panels, mean +/- SEM (user: SEM, not CI): new `fc.roi_dot_panel` (measures as x categories, lobe dots, FDR stars), used by `plot_anticipation.py` (-> `_prepost.png`: before the word and response peak on one axis, replacing the `_pre` box plot) and `plot_variance_structure.py` (carryover, was CI). Proposal 5A = pre/post (`img/fig5a_prepost.png`), 5B = carry-over; old 5A/5B box images removed. Category label "Pre-stimulus average" (window in the caption).
- Region pilot (anatomical subregion within lobe, hemispheres pooled, `(1 | region)` added to the 13 terms): occipital, 52 random occipital subjects (seed 0), up to 5 electrodes each; scratch scripts, ~20-40 min per fit. High gamma: region 3.9% of the no-region total, out of electrode (23.1 -> 21.8%) and subject (12.6 -> 11.2%); alpha 0.1%; latency first fit inflated by one calcarine electrode (460 ms) -> refit without it. `plot_variance_structure.py` GROUPS gains "Region (within lobe)". Proposal Fig. 1B: occipital row from the pilot with region (52/184), others unchanged; Aim 1 sentence added. Latency (Fig. 1C): Fig. 3A regions only (>= 30 subjects), 52 random subjects per lobe, lobes with >= 2 such regions (frontal 4, temporal 3, occipital 3, parietal 2; limbic / hippocampus 1 -> full-run rows kept), with region only, 4 fits in parallel (5-41 min): region 0-1.7%.
- Proposal restructured, one figure per Aim, cited in order (7 pages): Aim 1 cites Fig. 1A (average response) and gains the latency variance split as Fig. 1C; old Fig. 3A (latency per region, known gradient) and Fig. 4 (task effects) dropped from the proposal; the list-trend numbers move to Aim 2 as the slow side of a slow-vs-fast band coupling sentence with Fig. 2C; Aim 3 = blank + carry-over only ("What links one word to the next?"); closing paragraph labelled Summary.
- `plot_variance_structure.py` section 3: alpha-HG coupling by timescale (`alpha_hg_timescales.png`, `_stats.csv`): word event (same electrode, serial-position and list means out; new `r_fast` in electrode_sessions.csv), serial-position means and list means (subject x lobe, all electrodes, from levels.pkl); lobe dots mean +/- SEM, y -0.3..0.3 (user; -1..1 hid the SEMs). Replaces `alpha_hg_across_events.png` (proposal Fig. 2C). Full run: frontal / parietal / occipital serial-position r -0.12 to -0.20 vs word event -0.01 to 0.04; limbic / hippocampus +0.07 to 0.13 at every timescale; list r near 0 outside MTL. Aim 2 sentence rewritten to that; unused proposal images (alpha_hg_across_events, fig4b_latency_fine, task_effects) removed.
- Hemispheres pooled (PI): `fc.roi_of_reg_full` returns `B-<lobe>` by default (`pool=False` keeps L / R: synchrony box plots and epochs only). ROI helpers (`roi_stats`, `roi_panel`, `roi_bar_panel`, spectrum heatmap) take the ROIs present (`fc.present_rois`); pooled box plots one per region. Re-plotted: power box plots (paper Fig. 1A, S1; proposal 1A), spectrum by ROI (`plot_spectrum.py` rows pooled; Fig. 1B), anticipation (Fig. 9B; proposal 5A), latency per ROI (`plot_latency.py --rois`, text only). Test-retest stays on anatomical regions. Paper text: Methods (six regions; synchrony 12), power, spectrum (40 of 66 cells), latency per ROI and anticipation numbers updated; frontal high gamma now a slight decrease (d -0.008, q = .043). Paper and proposal recompiled.
- `build_roi_power.py --stage plot`: `ci=` -> `bars=` (stale keyword; crashed).

## 2026-11-08 — proposal: Prelim/Planned panels; synchrony as Aim 4

- Panel letters top left with (Done) for paper figures and full-data analyses, (Prelim, reason) for the 5-electrode variance fits, (Planned) as empty boxes: Fig. 5B variance split of the pre-stimulus blank alone, Fig. 6C synchrony variance split.
- Aim 4 (synchrony) from the paper's figures: alpha ciPLV epochs (Fig. 6A) and network replicability by ROI (Fig. 6B, right panel of the paper's Fig. 8 cropped, title removed; session test-retest panel and its sentence dropped).
- Proposal text and captions: "lobe" -> "ROI"; "cortex" -> "regions".
- Fig. 6B (PI, 2026-11-09): subject-half replicability replaced by the paper's session test-retest panel (`session_test_retest.png`, left panel of the paper's Fig. 8); Aim 4 quotes its medians.
- Fig. 5B: planned box for the previous-word (carry-over) model, to be refit with the variance model's random terms + electrode in list; the planned pre-stimulus split is 5C.
- `alpha_hg_across_{electrodes,events}.png`: y label "r (alpha vs HG response)"; signed d orange, |d| purple, across word events green.
- Alpha-high gamma across word events moved from Fig. 4B to 3C (Fig. 4 is task effects only); stable-part figure now before latency (power: Figs. 1-2, latency: Fig. 3).

## 2026-11-07 — full-cohort refit in the proposal

- Refit of `mixed_models.R variance` in Cohen's d with DeHaan's terms finished on the full cohort (up to 5 electrodes per subject x lobe; 6.1 h). Proposal Aim 1 and Figs. 1B, 2B now use it instead of the 40-subject pilot; pilot labels removed. Figures redrawn with `--from-csv` (legend above the axes).

## 2026-11-06 — variance analyses in Cohen's d; DeHaan's model terms

- `plot_variance_structure.py`: power response per word event = (word-on power - mean word-off power) / Cohen's d's pooled SD (mean over a session's events = the cached d, verified to 1e-7); replaces log power. Outlier rule unchanged. `same` category and blank length removed from the model.
- `mixed_models.R variance`: DeHaan 2026 ch. 3's terms (item, sp, list, subject, subject:item, session, subject:sp, list:sp, session:list, session:sp) plus electrode, electrode in session, word event. Electrode x serial position and fixed effects dropped. Log-power tables backed up in scratch (`variance_logpower_2026-11-06`).
- Task effects (`task_effects.png`, `task_trends.csv`): no longer the model's shrunk estimates. All electrodes (latency: responsive); per position / list, deviations from each subject's mean, 95% CI across subjects; per-subject slopes vs 0, BH-FDR.
- Coupling figure split into `alpha_hg_across_electrodes.png` (now with |d|) and `alpha_hg_across_events.png`; `prevalence.csv` from all electrodes (paired t).
- Proposal reorganized: Aim 1 the variance split, Aim 2 the stable part, Aim 3 the word-to-word part; text corrections; open items moved to TODO.md.

## 2026-11-06 — full-cohort variance results in the proposal

- Full run finished (339 subjects for power, 175 for latency; up to 5 electrodes per subject × lobe). Proposal Aims 1 and 3 now quote these numbers and the blank-model result: a larger previous response predicts a larger current one, with no blank-length interaction and a null next-blank control, so carry-over/drift, not adaptation.
- Proposal captions shortened to one line per panel.
- `task_effects.png`: bands changed from +/-1 SE to 95% intervals (1.96 x the model's conditional SD); caption updated. Open issues moved from the proposal to TODO.md, with a data-coverage list.
- Wording: nested terms as "in" ("Electrode in session", "List in session"), "Word event" for the shared component and in the proposal text; residual stays "(electrode, event)".
- Terminology: "trial" (a list in FR) -> "event" for word presentations in the proposal, figure labels ("Event", "Residual (electrode x event)", "Share of variance across events", "Across events (per electrode)") and the docstrings/comments of plot_variance_structure.py, plot_variability.py, mixed_models.R. Column/file names (`trials.csv`, `r_trial`, `trial_pow_*`) unchanged.
- Proposal for the PI: blank-model and offset-recovery figures and text dropped (to be recomputed with the paper's measures, TODO); status tags removed; variance components split into `variance_components_power.png` (next to the power boxplots, Fig. 1B) and `variance_components_latency.png` (next to latency, Fig. 2B); coupling and electrode reliability stacked; open issues now note log power and the missing DeHaan terms. Fixed inverted lobe order with two shared-y panels.
- Proposal figure order: paper power boxplots (figS1), latency, variance components, coupling + electrode reliability combined (A/B), task effects, blank model, offset recovery, anticipation.
- `blank_interval.png`: only the coefficient panels remain (the per-design blank-bin curves are gone; the bins stay in the model). Unused `BLANK_BIN_MS`/`DESIGNS` constants removed.
- `variance_components.png`: legend label shortened to "Word presentation".
- `blank_interval.png`: constrained layout (the coefficient labels had widened every column gap); 4 x ticks on the coefficient axis. The caption now says bins are relative to the shortest bin.

## 2026-11-05 — blank interval and latency in the variance framework; full-cohort run

- `plot_variance_structure.py` sections 4-5 replace `plot_blank_interval.py` (deleted): (4) the word-off window re-aligned to time since the previous word's offset (offset response and recovery; high-gamma envelope at responsive electrodes, alpha window means); (5) `mixed_models.R blank`: response ~ 50 ms blank bins per design + previous word's response x blank length + the blank after the word (control) + random subject / electrode / electrode x session / serial position / session x list; Wald CIs, BH-FDR.
- Design from the experiment (FR1 / catFR1: 750-1000 ms, pyFR: 800-1200 ms), not the longest blank: measured FR1 blanks reach ~1035 ms (screen timing), 40 ms tolerance, longer blanks in the last bin.
- Models: electrode cap applied in Python (`--max-electrodes`, seed 0); the latency model draws its own electrodes among responsive ones (`in_power` flags the power-model electrodes); `--cores`; a fit with < 5 subjects or an lmer error is skipped with a message instead of failing the run.

## 2026-11-05 — variance structure as one mixed model; boxplots with dots

- `mixed_models.R variance` (run by `plot_variance_structure.py`): per lobe and band, DeHaan 2026 ch. 3's random-intercept model on the single-trial response: subject, electrode, electrode x session, serial position, electrode x serial position, list, session x list, word (item), word presentation (shared by a lobe's electrodes), blank length + same category (fixed). Replaces the method-of-moments `partition`, the OLS `task_design` and its chance-fit correction; the time / frequency subject-electrode-session shares use the same model on electrode-session means.
- Serial position is a factor (positions 2-12) everywhere: the 2-4 vs 8-12 split is gone; size vs timing = peak and half-max latency per position, per subject (`serialpos_size_timing.png`). New `task_effects.png` (model estimates per serial position and list) replaces `trial_modulators.png`.
- Removed: the carpet example; the "task structure removed" coupling box; from `plot_variability.py` the alpha-vs-high-gamma d panel (duplicate of the coupling figure), the early/late serial-position time course, electrode direction, variance shares and the trial-modulator section (superseded). It keeps electrode-pattern stability and the (on-hold) memory analyses.
- Quick preliminary fit: at most 5 random electrodes per subject and lobe (`LMM_MAX_ELECTRODES`; the full fit ran > 20 min); single-trial fits rerun only when missing or with `--refit`; `mixed_models.R levels` (cheap) always.
- Variance components shown as stacked shares per lobe (all terms sum to 100%; task terms grouped; n subjects / electrodes per lobe). The separate trial-averaged subject / electrode / session split and `mixed_models.R levels` are removed (its levels are segments of the same bars).
- Blank length in `plot_variance_structure.py` measured on all words before masking, so serial position 2 gets its real blank (after position 1) instead of the session median; the quick model refit.
- Response types figure and code removed (the coupling figure with stats answers whether alpha and high-gamma responses sit at the same electrodes; prevalence numbers kept in the proposal text). Electrode-stability axis relabelled.
- Alpha vs high gamma: per lobe, subjects' mean Fisher z vs 0 (one-sample t) for the across-trial and across-electrode r, BH-FDR over 12 tests; marked above the boxes.
- Size vs timing by serial position removed (its peak panel repeated the task-effects curve): single-trial high-gamma half-max latency (responsive electrodes) is now a third measure in the same mixed model, so its variance components and serial-position / list estimates sit next to alpha and high-gamma power. The model output records n subjects / electrodes per fit.
- Response types: k-means on 5-150 Hz profiles replaced by a rule: alpha and high gamma each up / none / down per electrode (paired t over trials pooled across sessions, BH-FDR 0.05), 3 x 3 fractions per lobe with the ratio to independence. The spectrum pickles are no longer read.
- Proposal: Aims rewritten on the model results; figures reordered (components, trial-averaged split, types, coupling, stability, task effects, size/timing, paper figs); unused images removed.
- New `fc.box_dots`: boxes with every value as a dot; used by the coupling, stability, serial-position and paper reliability (`plot_reliability.py`, fig 8) figures.

## 2026-11-05 — response-type and gain/timing figures redrawn

- `plot_variance_structure.py`: response types: profiles compact (shared y), coloured by type to match a larger lobe-composition panel. Gain vs timing: the per-electrode scatter replaced by boxplots per lobe of each subject's median peak change and half-max latency change (positions 8-12 vs 2-4). Proposal captions updated.

## 2026-11-05 — proposal refocused on the word response; variance-structure preliminaries

- New `plot_variance_structure.py` (40 random subjects, ~1 min; -> `figures/variance_structure/`): nested variance partition (subject > electrode > session > trial) of the high-gamma response over time and of Cohen's d per frequency; k-means response types on spectral + high-gamma time profiles; alpha vs high gamma trial by trial per electrode vs across electrodes; gain vs latency (serial positions 8-12 vs 2-4); a single-trial carpet. Per-trial time-frequency power is not cached, so the full time x frequency x level partition needs a recompute.
- New `plot_blank_interval.py` (full cohort, ~1 min; -> `figures/blank_interval/`): response and own baseline by blank length per design (50 ms bins), high-gamma time course by blank tercile, previous word's response and blank length as predictors (serial position controlled).
- `plot_variance_structure.py`: task structure as its own component (per electrode-session OLS on serial position dummies, list number z and z^2, blank length z, same category as the previous word in catFR1; variance of the fit minus the chance fit of k regressors) -> 3-7% of single-trial variance; alpha vs high gamma trial-by-trial r also on task residuals (unchanged, ~0).
- TODO: new "Proposal: to confirm" section (incl. the latency-reliability item, which was never in the file); decoder TODO updated.
- Proposal rewritten short: memory on hold (DeHaan 2026 ch. 3), gap from Marks 2021 / Kucewicz 2019 / Burke 2014a; figures ordered by relevance with status tags; irrelevant figures removed (subject distributions, up/down bars, SME, decoder, connectivity reliability, network).

## 2026-11-04 — proposal draft; variability / trial-modulator preliminary results

- New `plot_variability.py` (-> `figures/variability/`): electrode direction (up / down per lobe), variance shares (between vs within subjects), trial modulators (serial position, list number, blank interval; per-subject OLS, BH-FDR), recall ~ covariates, SME raw vs covariates removed. Robust outlier trials dropped (10 MAD); catFR1 list -1 (unscored practice) and serial positions > 12 excluded; the blank of serial position 2 (first in the events; position 1 has no blank baseline) set to the session median.
- Found: no trial-level artifact rejection in the pipeline. R1323T session 3 trial 90 (broadband transient ~5000x on 82 of 148 electrodes, 655 ms) makes the supramarginal-gyrus spike in Fig. 4B. Not yet fixed in the main pipeline.
- `plot_variability.py` adds: electrode-pattern reliability across session halves and alpha vs high-gamma electrode coupling (`electrode_reliability.png`), the high-gamma time course for serial positions 2-4 vs 8-12 (`serialpos_timecourse.png`; outlier trials also by their whole-clip peak, which the 0-600 ms mean missed for R1323T's 655 ms transient); no hatching in any figure.
- Proposal restructured: figures under their aims, power summary dropped, latency and network in Aim 1, a conclusion per aim; Aim 3 builds on DeHaan 2026 (thesis ch. 3: SME ~40% attenuated by serial position, item, list) instead of claiming it.
- New `proposal/proposal.tex` (variability, task drivers, memory), reusing paper figures + the three new ones; `proposal/extra.bib` adds Correa 2006, Todorovic 2012, Brown 2006, Pinheiro-Chagas 2024 (check author list).

## 2026-11-03 — decoder: per-session AUC (DeHaan et al. 2025)

- `plot_decoding.py`: AUC within each held-out session, averaged per subject; subjects with >= 3 sessions; one-sided t vs 0.5. Replaces the pooled held-out AUC, whose chance level was ~0.493 (fold intercepts track the training recall rate), and its per-subject permutations (`DECODER_N_PERM` gone). Rerun on the real data and strong / hg_null / sync_equal.
- Then, as DeHaan et al. 2025: C chosen per held-out session by nested leave-one-session-out over 52 values (1e-6 to 1e11; was fixed 7.2e-4), refit on the training sessions (`fit_path`: damped Newton along the C path, matches sklearn to 1e-7); montages (electrode sets) decoded separately and averaged (4 subjects changed electrodes across sessions; missing electrodes had been zero-filled).
- Connectivity recovery sweeps (rec_ciplv, rec_aec_c; 8 tags) rerun on 10 random subjects with >= 5 frontal electrodes (seed 0; Riley: can a planted non-null level be measured?). Stale pre-fix sweep caches moved to `sim/_stale_2026-11-03/`. Then 4 levels each (ciPLV 0.01/0.25/0.5/0.75, AEC-c depth 0/0.5/1/2; briefly 3) on 30 random frontal subjects (the 10 + 20, seed 1); stale rec_power caches moved there too. `figures/simulations/recovery.png` force-added to git so it can be viewed remotely. Recovery x axis for ciPLV / AEC-c is now the true value: the same metric on the planted signal alone (pink noise off; synthetic clips of the real montages, 10 sessions per level; `results/simulations/recovery_truth.csv`). Measured / true: ciPLV ~0.63 at every level; AEC-c 0.50, 0.75, 0.94 (noise attenuates weak envelope coupling most). Fixed a first attempt whose joblib workers changed a pickled copy of the parameters (noise never turned off). Titles and axis labels rewritten.
- Abstract: the user's latest version (early occipital alpha synchronization; "frontal lobe"; "r up to 0.90"). Decoder methods and Figure caption updated; results text pending the rerun.

## 2026-11-03 — simulations plotted from 300 subjects

- The full-cohort simulation run (3 tags) was taking ~35 min per 10-subject batch, mostly downloading; it was stopped after 30 of 44 batches (300 subjects cached; resumable from `stream_done.txt`).
- The three simulations (hg_null, sync_equal, ceiling) are plotted in parallel from those 300 subjects (`logs/plot_sims_2026-11-03.sh`).
- Paper abstract replaced with the user's new version (decoder sentence: every lobe, most persistently frontal; "we characterized"; "only the occipital network replicated strongly"; "reliably strong early visual engagement").
- Simulator bug: target channels were picked from the raw atlas labels, while the analyses use `fc.session_regions` (recovered whole-brain labels + MNI-neighbour labels). In newer subjects (mostly R13xx-R14xx) frontal electrodes counted by the analyses were never targeted: 76 of 358 frontal subject x hemisphere values in the ceiling simulation were ~0. `replace_w_simulated_EEG` now uses `fc.session_regions` (raw atlas as fallback).
- `ceiling` renamed `strong` (config, run_streamed.sh, plot_sim_summary.py). Its cache and those of the other targeted tags (leak_frontal, hg_latency) were wiped; the three rerun on the first 50 subjects (`logs/run_sims_2026-11-03.sh`). The nulls are unaffected (no planted effect).
- Simulation composites: test-retest y label "Test-retest r within subjects" (was "across regions"); replicability now uses plot_reliability.py's boxes over splits; power and synchrony now use the paper figures' per-ROI box plots (fc.roi_panel; one row per band) instead of mean +/- CI points; panels widened.
- Simulation composite titles: "coupling" -> "connectivity".
- `plot_decoding.py`: `N_PERM` can be set with `DECODER_N_PERM` (default 2000, unchanged for real data); simulations use 200 (the real decoder took ~4 h per run).

## 2026-11-02 — simulation set cut again; ceiling fixed

- `ceiling` didn't recover high-gamma power, the decoder or latency. Envelope depth 2
  multiplies high-gamma power by about e^8 (~3000x), which buried the 4x gain. Now
  `env_depth` is 1 (~7x) and `hg_gain` is 10.
- `hg_recall_frontal` dropped. `ceiling` already tests recalled vs. not-recalled words
  and the timing (200-600 ms).
- `hg_latency`: gain steps up (x4) at a known time per lobe: frontal 50, occipital 100,
  parietal 150, limbic 200, hippocampus 250, temporal 300 ms. Previously occipital,
  temporal and frontal at 100/200/300; recovered as 105/206/306 ms.
- `sync_equal` made near-deterministic: phase PPC 0.9999 before and after onset, pink
  noise 1 instead of 10. Alpha power and ciPLV differences should be ~0 with almost
  no scatter.
- Each control plots only its own analyses (`run_streamed.sh`):
  - `leak_frontal`: ROI boxplots and the epoch network.
  - `hg_latency`: latency.
  - Nulls and `ceiling` still run everything.
- Sweeps paused: not run, and the recovery figure isn't in the paper.
- New `plot_sim_summary.py` → `figures/simulations/sim_summary_<analysis>.png`, one figure per analysis: power,
  synchrony, recall decoder, network replicability. Panels: `hg_null`, `sync_equal`,
  `ceiling`, sharing the y-axis.
  `sim_summary_test_retest.png`: the test-retest histograms (power and synchrony rows)
  for the same three tags.
- `plot_reliability.py` no longer crashes when no subject has enough sessions for
  test-retest. With 30 subjects the nulls had none, so no replicability table was
  written.
- Real-data decoder (2000 permutations) started on 2026-11-02 in parallel with the
  simulations instead of after them, for the abstract deadline.
- Test-retest reliability: histograms replaced with box plots across subjects.
  - One r per subject (a single random session split, as in riley-thesis
    `compute_retest_reliability.py`). Opaque boxes, no dots, whiskers at the central
    95% of subjects.
  - Reported as median [IQR] only, no test vs 0: reliability is judged by its size,
    and "above 0" doesn't make a measure reliable. Real data, 91 subjects (the paper
    said 85; that count was stale):

    | Measure | Median r [IQR] |
    |---|---|
    | Alpha d | 0.67 [0.38, 0.85] |
    | High-gamma d | 0.81 [0.58, 0.93] |
    | Alpha ciPLV | 0.19 [-0.05, 0.43] |
    | High-gamma AEC-c | 0.14 [-0.04, 0.32] |

  - Interim versions today (a Wilcoxon/Holm test vs 0, the bootstrap distribution of
    the median) were dropped.
  - Panel title "Test-retest across sessions", y label "r (within subject)". The
    no-coupling simulation's median is drawn as a dotted reference when its table
    exists.
- Network replicability simplified to the plain splits.
  - 1000 random splits of the subjects into two disjoint halves of 127 (no
    replacement). Each split gives r between the halves' mean connection x epoch
    profiles. Box over the 1000 r, median = estimate, whiskers = central 95% of
    splits.
  - Dropped: the within-split bootstrap and the shift to the split median (added
    2026-11-01).
  - Occipital: alpha 0.90 [0.81, 0.95], high gamma 0.81 [0.64, 0.90]. Alpha intervals
    exclude 0 only in parietal (0.42 [0.15, 0.62]) and limbic (0.43 [0.17, 0.65]); all
    high-gamma intervals except occipital include 0.
- The paper (methods, Figure 8 caption, results) and the simulation composites are
  updated to match.
- Paper wording: the abstract gains a reliability sentence (power reliable within
  subjects, synchrony not, only the occipital network replicates). "Power/synchrony
  maps" became "power/synchrony", and "patients" became "subjects" throughout
  (7 places).
- Found: a small positive bias in high-gamma Cohen's d. In the full-cohort no-coupling
  null, mean d is +0.0026 (SE 0.0004, 213 subjects); alpha is 0.0000.
  - Cause: each high-gamma sub-band is divided by its word-off mean, estimated from the
    same word-off trials. So word off averages exactly 1, while word on is divided by a
    noisy estimate, which biases it up (Jensen's inequality).
  - Pure-noise test: +0.0017 (SE 0.00013). Dividing by the mean of both conditions
    gives +0.0001.
  - Not fixed yet: the fix needs the real compute rerun.
  - Null high-gamma test-retest (median r 0.16, 39 subjects) is inconclusive:
    shuffled-region p < .001, but Wilcoxon p = .08 and the bootstrap CI includes 0.
    Waiting for the full-cohort null.
- Real-data decoder rerun with 2000 permutations per subject. Each subject is tested
  against their own permutation chance (mean 0.493), paired one-sided t across
  subjects, BH over lobes x windows.
  - 45 of 60 cells pass (q < .05).
  - Frontal passes in every window from 0 ms (peak AUC 0.525 at 800-900 ms).
    Parietal passes at 0-100 ms and from 200 ms.
  - Occipital passes from 100 ms (peak 0.519 at 500-600 ms), limbic at 200-800 ms,
    temporal from 300 ms, hippocampus at 500-700 and 800-1000 ms.
  - The paper's methods, Figure 7 caption, results and abstract are updated. The
    old text said "no lobe before 200 ms" and tested against 0.5. Added the
    Jamalabadi 2016 reference.
- Quick look (30 subjects; no-coupling null on its ~210 cached subjects):
  - Nulls: about 0 everywhere.
  - Leak: synchrony about 0.
  - Latency recovered exactly: planted 50/100/150/200/250/300 ms came back as
    55/105/155/205/255/305 ms (bin centre +5).
  - `ceiling` synchrony, decoder and epoch hubs: frontal only, at the planted times.
    The decoder spills about 100 ms each side of 200-600 ms; high-gamma estimates
    smear by about 100 ms.
- Simulator fixes for `ceiling`:
  - `osc_lag` with `active_ms` redrew phases on every channel at the window edges.
    The phase jump inside the word-on window cut alpha power everywhere (d about -1.9
    in all regions). Now only target channels change phase, over a 50 ms raised-cosine
    ramp. Non-target power change is 0; a target channel still loses about 10% alpha
    power from the phase change itself (isolated test).
  - `recalled_only` replaced by `recall_gain`: the high-gamma gain (x10) now applies
    to every word, and recalled words get a further x2. Word on vs off power is then
    strong, and recall stays decodable (it was d about 0.3 with recalled-only).
  - Only `ceiling` uses these options; the other tags are unchanged.
- Full-cohort stage restarted (5 workers, `hg_null` `sync_equal` `ceiling`) with
  the `ceiling` cache wiped; runner `logs/run_sims_2026-11-02b.sh`.
- The 2026-11-01 full-cohort null run was stopped at batch 22; the changed tags'
  caches were wiped. Runner: `logs/run_sims_2026-11-02.sh`.

## 2026-11-01 — simulations cut + rerun; decoder permutations; replicability intervals

- Simulator: pink noise generated at a fixed length (8192 samples) and cropped. FFT
  noise is periodic over its length, so per-clip generation gave the longer word-on
  clip (2.0 s vs 1.6 s word off) more low-frequency power: +0.13% at 6 Hz, falling
  to +0.01% at 70 Hz (exact calculation). That was the slight positive 6-8 Hz lean
  in the full-cohort nulls. Real data are unaffected (both clips are cut from one
  recording, and the spectrum uses equal 600 ms crops).
- Simulation set cut from 31 tags to 6 + 3 sweeps:
  - Nulls, full cohort: `hg_null` (no coupling), `sync_equal` (equal coupling before
    and after onset).
  - New `ceiling`: alpha phase coupling (PPC 0.99), a shared high-gamma envelope
    (depth 2) and a 4x high-gamma gain on later-recalled words only, frontal,
    200-600 ms, near noise-free.
  - Controls, first 30 subjects: `leak_frontal`, `hg_latency`, `hg_recall_frontal`;
    sweeps `rec_power`, `rec_ciplv`, `rec_aec_c` (4 levels each).
  - Removed: `hg_power_frontal`, `alpha_lag_frontal`, `hg_env_frontal`,
    `alpha_lag_epoch`, `hg_env_epoch` (merged into ceiling), `hg_null_line`,
    `rec_power_clean`. All simulation outputs (51 GB) were deleted.
  - Runner: `logs/run_sims_2026-11-01.sh`.
- plot_anticipation: no wait-times figure for simulations (wait times come from the
  events, so they are identical in every simulation).
- plot_decoding:
  - Permutations 20 -> 2000 per subject, all label sets fit at once (`fit_predict`:
    the same L2 logistic regression; AUCs within 4e-4 of sklearn's, whose
    convergence tolerance is looser).
  - Per subject: chance level (mean of the permutation AUCs, ~0.49 because held-out
    AUC is biased low) and p.
  - Group test: AUC minus the subject's chance level (paired, one-sided), not AUC
    vs 0.5. The dashed line is the chance level.
  - Fix: `scipy.special` referenced through the module (joblib workers couldn't
    unpickle `expit`).
- plot_reliability, network replicability:
  - Estimate: median r over 1000 disjoint subject splits (unchanged).
  - Interval: each split's halves also bootstrapped (subjects resampled with
    replacement, within each half). Bootstrapped r runs low (occipital alpha 0.83 vs
    0.90), so the distribution is shifted to the split median. Box plots show it;
    whiskers are the 95% interval.
  - Real data: occipital 0.90 [0.64, 0.99] alpha, 0.81 [0.46, 0.98] high gamma.
    Alpha parietal and limbic just exclude 0; every other interval includes 0.
- plot_epoch_network:
  - Hubs are absolute again, as Rao et al. 2025: a region's mean change over its
    connections, t vs 0. Holm over regions x epochs replaces BH/BKY, because each
    hub x epoch is reported on its own.
  - BKY had marked more alpha cells than having no correction at all (69 vs 65);
    BH marked 53 of 108 (most of the map, since all of it rises); Holm marks 18.
  - Real data: alpha hubs move from occipital (0-400 ms) to limbic/temporal,
    parietal (300-700), left frontal and left temporal (400-1000); 4 fine-level
    hubs; high gamma has one (left occipital, 0-200 ms).
  - The "region minus rest of network" hub is supplement only
    (`_hubs_relative.csv`, ROI x epoch t heatmap `_relative.png`, Holm; 12 ROIs).
    9 alpha cells: occipital above the network, temporal and left hippocampus below.
  - plot_spectrum: `heatmap` split into `draw` (one axis, reused) + saving.
  - Paper: methods, statistics, Fig. 4 caption and results rewritten; new
    Supplementary Fig. `fig:hubs-relative`. figS4_recovery dropped from
    sync_figures (no simulation validation in the paper).
- plot_latency: the default regions are anatomical labels within the Burke ROIs,
  hemispheres pooled (the paper's Fig. 3, `hg_latency_word_on_fine_pooled`).
  `--fine` and `--pool-hemispheres` were replaced by `--rois` (the 12 ROIs, unused
  by the paper; dropped from sync_figures).
- Case studies removed: plot_case_studies.py, its pipeline entries and all
  `power_spectrum_word_on_occipital*` outputs. build_roi_power.py (longetal)
  writes its own file of the same name and was left alone.

## 2026-10-30/31 — Figure 2/3 rework; simulation null fixes

- plot_spectrum, as DeHaan et al. 2025 (11 bins kept):
  - Colour is t across subjects per ROI x frequency cell; BH-FDR over all cells;
    outlines around connected significant cells (no cluster statistic; TFCE
    removed).
  - The 12-ROI map is Fig. 2B; the subregion map (`_fine`) is in the supplement.
- Figure labels use "Δ word" (`delta_label`) for word on vs off; larger fonts in
  the ROI bars and the latency figure.
- plot_mixed: the latency mixed model is a lower-triangle heatmap of pairwise lobe
  differences (Holm), now a supplementary figure; Fig. 3B was dropped.
- Simulator:
  - A tag can list several processes, and their signals add up.
  - `hg_envelope` `equal_conditions`: the envelope is shared on pre events too.
  - New null `sync_equal`.
  - Fix: the shared envelope is now scaled by a fixed factor, not each clip's own
    SD. Per-clip scaling weakened it in the longer word-on clip, which biased AEC-c
    low (11/12 ROIs in `sync_equal`).
- run_streamed.sh: `SIM_TAGS` / `SIM_PLOT_TAGS` stream simulation tags.
  - Fix: `--simulation-tag` is built as an array, because zsh doesn't word-split.
    The 10-30 run had failed in all 44 batches.
  - Sweeps run inside the stream, because drop_raw had deleted their EEG.

## 2026-10-29 — hub measure, network replicability, study figure

- Epoch-network hub: a region's mean change over its connections minus the mean
  over all other regions x epochs.
- plot_reliability: network replicability per lobe across subjects (disjoint
  halves, 1000 splits).
- New plot_study_figure.py: electrode coverage and an example high-gamma response
  (Fig. 1B/C).

## 2026-10-23 — spectrum to 150 Hz; simulator pink-noise scaling

- compute.py VERSION single_pass_8: spectrum bins 5-150 Hz (11 log bins, mains
  +/- 2 Hz left out), plus native-resolution trial-mean spectra.
- Simulator: pink noise scaled so its spectrum doesn't depend on clip length.
  Normalising each clip's total variance had produced a broadband word-on decrease
  in the null. Only 3 tags were rerun then; all tags were rerun 2026-10-31/11-01.

## 2026-10-13 to 10-22 — decoder, reliability, mixed models, label recovery

- plot_decoding: recall decoder (L2 logistic regression, ram_utils C,
  leave-one-session-out), per lobe and 100 ms window, 0-1000 ms.
- plot_reliability: test-retest (session halves) of power d and synchrony hub
  strength, as riley-thesis.
- mixed_models.R + plot_mixed.py:
  - Latency: `latency ~ lobe + (1|sub) + (1|sub:lobe) + (1|sub:region)`.
  - Bands: lobe x band interaction (Holm).
- Tables go to results/ (`fc.table_path`), figures to figures/.
- recover_labels.py (copied from Riley's cml_anatomy_standalone): recovers pair
  labels the DB dropped.
- compute.py single_pass_7: per-trial spectrum bins.

## 2026-10-12 — PAC removed (PI decision)

Theta-gamma PAC (compute.py, plot_boxplots, plot_epoch_network, fc.compute_pac,
bands.low, pactools) removed with its results and figures. Trial-shuffled
surrogates reproduced the whole word-onset increase (stimulus-locked, not
within-trial coupling).

---

## 2026-09-30 — word onset, multitaper only: large cut

Outputs checked unchanged on a recomputed session (power, d, time bins, PPC:
max difference 0); cohort unchanged (716/1019 recomputed recall checks match).

- Removed: Morlet everywhere (cwt_morlet mode, `--fc-mode`, cwt_* config,
  wavelet.py, data_check's Morlet phase check, which never excluded a session);
  AEC/AEC-c and GC/GC-TR (metrics now coh/plv/ppc/ciplv/pli/wpli/pac); the
  en/rm/voc/ri/en_all/rm_all contrasts (`--beh`; only word_on events are written,
  but load_events.recall_matching_ok keeps the cohort rule); mirror buffer;
  compare_distance_controls.py, plot_phase_conn_distance*.py,
  burke_roi_connectivity.py, visualize_phase_connectivity.ipynb; cstat.py,
  matrix_operations.py, figure_io.py; ~40 dead fc/helper/misc functions and the
  CFG_FLOW_VERIFY pins; unused bands (theta, narrow_*, theta_6_12).
- Long et al. options are no longer flags: `--t-unit`, `--combine`,
  `--min-responsive-per-region`, `--responsive-alpha`, `--responsive-only`
  come from `longetal_params` (new key `t_unit: sample`); p < 1e-8 is fixed.
  Also removed `--tfce-perm` (TFCE), `--time-bin-ms`, `--mt-window-ms` (config),
  `--fine-top-n` (argument, default 15), `--fc-root`.
- Power time course only for gamma bands (fmin >= 30 Hz).
- New: `roi_synchrony_distance_*` (on - off synchrony per distance bin per ROI);
  fc.bin_stats / fc.roi_curve_figure shared by it and the power time course.
- Synchrony compute: one EEG load per window (the extra union load is gone).
- prepare_sessions flags: `--subjects`, `--n-subjects`, `--workers`,
  `--setup-only` (was 11 flags; setup runs when the listing is missing).
- Simulations: simulate_eeg.py rewritten for the two high-gamma processes
  (hg_power, hg_ppc; ~1600 legacy config lines dropped, the broken AEC process
  removed); seed = CRC of the real clip, so reruns are reproducible (was
  Python's per-process hash). Snakefile recovery sweeps now run on the
  high-gamma run (were on the first run, alpha, so recovery.png came out empty).
- Deleted all smokescreen outputs.

## 2026-09-30 — contrast-only figures, raw synchrony

- Power, synchrony and power-synchrony draw only word on - word off: one figure
  per analysis (a row per band plotted into that directory,
  fc.band_contrast_figure) in its own figure dir. Removed the lo/hi panels and
  measures (z_lo/z_hi, sync_lo/sync_hi, lo/hi power-sync r), `--measures`,
  `--conds`, `--style`, fc.roi_bar_panel, fc.cond_labels, plot_band_contrasts.py
  and figures/band_contrasts/. Synchrony compute writes only fc_mats/diff/.
- Synchrony is raw PPC by default (whole-brain changes show); `--zscore` restores
  the per-distance-bin z-score. Power-synchrony still z-scores.

## 2026-09-30 — longetal replication closed

Final longetal settings (config `longetal_params`): FR1, English only, monopolar
+ CAR, Desikan-Killiany -> Long subregions, 8-band Hilbert high gamma at 100 Hz,
blank -750-0 vs word 0-1600 ms, words averaged over sessions, uncorrected
per-timepoint t at p<1e-8. Reproduces Long's 25.3% (ours 28.6%) and her
per-region |t|, latencies and % enhanced outside the atlas-mismatched regions
(cuneus, lingual, medial frontal). A per-trial / autocorrelation-corrected test
gives ~3%. Open, not pursued: Talairach labels, per-band z-scoring.

## 2026-09-29 — longetal figure names; Long latency / % enhanced figure

- Hilbert responsiveness outputs now say which t-test they used:
  `_pertimepoint` (Long's, uncorrected) / `_pertrial` (was `_samples` / none).
  Juee file names unchanged.
- word_averaged_rows also returns Long's latency (sample of the absolute max of
  the average word response, z-scored to the blank) and its sign; new
  `long_peaks_*.png/csv`: latency and % enhanced of responsive electrodes per
  subregion, ours (mean ± SEM) vs Long's values (`LONG_PEAKS`: paper text for
  occipital, read off thesis Figs 2.6A/C otherwise).
- helper.bids_reader passes `device='ieeg'` (bidsreader only infers it for R*
  subjects, so every pyFR TJ*/UP* session failed the data check).
- run_streamed.sh: retry loop variable made local (it reset the batch index).

## 2026-09-28 — Juee full run: alpha + high gamma, contrast figures, streamed data

- config `runs`: alpha added next to high_gamma (word_on, multitaper, ppc).
- Snakefile: synchrony now waits for power (power -> synchrony ->
  power-synchrony). Juee figures drop the word-on / word-off panels
  (`--measures cohens_d`, `--measures sync_diff`, new
  build_power_synchrony `--conds diff`); longetal figures unchanged. New rule
  `band_contrasts` -> plot_band_contrasts.py: one figure per analysis
  (power, synchrony, power-synchrony), alpha row + high-gamma row, word on -
  word off, paired t via fc.roi_stats (one-sample t on the within-subject
  contrast, BH-FDR over ROIs) -> figures/band_contrasts/.
- run_streamed.sh: bipolar FR1+catFR1+pyFR is ~680 GB (205 GB free), so per
  batch of subjects: download -> prepare -> power, synchrony compute -> delete
  the batch's bipolar EEG; then snakemake plots from the cache. Resumable
  (logs/stream_done.txt); run under `caffeinate -is`.

## 2026-09-28 — longetal: CAR everywhere, English only; t-unit diagnosis

- longetal `depth_reference: car` (PI's reading) and new `english_only: true`
  (data_check records `english` from the WORD stim_file; FR1 has 13 Spanish
  sessions / 6 subjects). Full rerun: 304 sessions, 147 subjects, 13,080
  electrodes; 28.6% responsive per sample, 2.6% per trial (~1 h awake; the
  earlier 11 h was Mac sleep).
- Diagnostic (scratchpad, read-only): envelope autocorrelation inflates the
  per-sample t (variance inflation ~18 blank / 23 word); corrected per-sample
  2.7% ≈ per trial 2.6%. Long's mean |t| (MOG 39.4, fusiform 10.7) matches our
  uncorrected per-sample (40.8, 10.0), so her 25.3% is most likely uncorrected.
  8 vs 1 band: same per trial, ~1.7x larger per-sample t (8 narrow-band
  envelopes are ~3x more autocorrelated). Null simulation: 0.0% per trial,
  3.2% per sample false positives at p<1e-8.

## 2026-09-25 — longetal settings chosen from pilot; full recompute

- Pilot (12 subjects, % responsive per sample vs CAR baseline 15.9%):
  `hg_bands 8` 26.2%, `include_first_word` 19.7%, `include_incomplete_sessions`
  16.8%, `baseline full_isi` 11.2%; bipolar depths 17.3% (earlier pilot).
- longetal now: `depth_reference: bipolar`, `hg_bands: 8`,
  `include_incomplete_sessions: true`, `include_first_word: true`,
  `baseline: fixed`. All previous longetal results (full, smoke, simulations,
  pilots) deleted and regenerated.
- Full recompute (311 sessions, 149 subjects, 11,399 electrodes; ~11 h, run with
  `CML_DATA_SOURCE=local`): 28.4% responsive per sample, 3.5% per trial (Long
  25.3%). Occipital highest (MOG 77%, lingual 52%, fusiform 39%). Simulations:
  hg_null 0.0%, recovery tracks identity.
- Fixes: 8-band envelope cast to float64 before resampling; full_isi mask
  broadcast over channels; helper.get_eeg sample-rate check uses a 1e-9
  relative tolerance (R1006P_FR1_1 failed on float rounding only).

## 2026-09-25 — Scratch and downloads moved out of the repo

- `paths.scratch_dir` → `~/scratch/word_onset_analysis/scratch` (sibling
  `scratch.smokescreen`, `scratch.longetal`, ...); new `paths.bids_cache` →
  `~/scratch/word_onset_analysis/bids_data`, exported as `CML_BIDS_CACHE` by
  project_paths. Existing folders were moved, not recomputed. Figures stay in
  the repo.
- longetal `atlases: [ind.region]`: Desikan–Killiany (surface) labels for every
  contact, as Long did. Depths previously used stein → wb (volumetric), which
  labels many depth contacts "Cerebral White Matter" (dropped); DK gives the
  nearest cortical label.
- longetal `depth_reference: car | bipolar`: `bipolar` pairs adjacent contacts
  on each depth lead and keeps the common average for grids/strips only
  (get_eeg builds contact differences; data_check drops a pair if either
  contact is flat). Diagnostic on depth-only subjects: CAR suppresses depth
  responses, within-lead bipolar recovers them. Juee (default) path verified
  unchanged. Being piloted on 12 subjects before the full recompute.
- longetal settings for choices Long's text leaves open (defaults = current
  behaviour): `hg_bands` (1 = one 70–150 band-pass; 8 = mean amplitude over 8
  log-spaced Gaussian bands, frequency-domain Hilbert, Mesgarani-lab style),
  `baseline` (fixed pre_win; `full_isi` = each trial's whole blank from the
  previous word's offset, pre samples outside it set to NaN and skipped),
  `include_incomplete_sessions` (all sessions of subjects with ≥1 complete
  one), `include_first_word` (keep serial position 1). Piloted one at a time.
- Convention: "juee flags" = the default methods (bipolar, all experiments,
  multitaper, own windows/stats); their results are on hold and never
  overwritten. longetal results are wiped and regenerated on every method
  change. Full-cohort longetal now: 262 sessions / 150 subjects; responsive at
  p<1e-8: 2.7% (per trial), 13.2% (per sample), vs Long 25.3%.

---

## 2026-09-24 — longetal: Long et al. 2020 replication

`snakemake [all|simulations] --config longetal=true` (settings in config
`longetal_params:`; `WOA_LONGETAL` in project_paths; outputs `*.longetal`).
Default runs are unchanged.

- FR1 only; complete sessions only (≥300 WORD events, rule `incomplete_session`).
- Monopolar contacts (one "pair" per contact, distance 0; ECOG/SEEG channels
  only) + common average reference per sample over the kept contacts, applied
  after any simulated EEG so simulations see it too.
- No resample; `--fc-mode hilbert`: band-pass 70–150, Hilbert amplitude at the
  native rate, 500 ms buffer trimmed, envelope → 100 Hz, mean per window (or
  50 ms bins). Pickles store amplitude (`measure: amplitude`); change_dB and
  recovery use 20·log10.
- Windows: blank −750–0 ms vs word 0–1600 ms (not length-equalized).
- Only word_on events required. Responsiveness figure keeps regions with ≥50
  responsive electrodes (CSV keeps all). Power only, no synchrony.
- `exclude_lobes: [limbic, hippocampus, subcortical]` (Long excluded limbic and
  sublobar electrodes): those ROIs map to None under longetal.
- `--t-unit {trial,sample}` (plot stage): Long doesn't say whether the
  responsiveness t-test used trial means or every envelope sample. Hilbert
  pickles now also store `cohens_d_samples` / `n_{lo,hi}_samples`. Smoke:
  6.6% responsive per trial vs 19.3% per sample (Long 25.3%).
- `--combine {d_mean,word_average}` (longetal default word_average): Long
  averaged each word over repeated sessions before testing. Hilbert pickles
  store per-event envelopes (`env_lo`/`env_hi`, float32) and words (`items`);
  `word_averaged_rows` averages per word within subject (labels present in all
  sessions), then d across words or samples. `get_eeg` now labels each epoch
  with its position in the input events (was 0..n-1), so dropped events keep
  their identity. Smoke: 23.1% responsive (word_average, sample) vs Long 25.3%.
- `long_regions.csv`: approximate map from our region names to Long's 16
  subregions (hemispheres pooled); used for longetal `--fine-labels`.
  `exclude_regions: [posterior cingulate gyrus]` (limbic in her atlas).
- `get_eeg` drops events whose clip isn't fully inside the recording before
  epoching (MNE raised instead; truncated OpenNeuro EDFs, e.g. R1156D, R1161E).

---

## 2026-09-24 — Recovery curves

- `sweeps:` in `config/simulation_config.yaml` expand (simulate_eeg loader) to
  one tag per planted value: power gain 1–4 (0–12 dB) and frontal PPC 0–0.99,
  each with and without pink noise. `snakemake simulations` runs them
  compute-only and `plot_recovery.py` draws measured vs planted →
  `<figures>/simulations/recovery.png`.
- Smoke result: noise-free power on the identity line; noisy power below it
  (pink noise shares the band); PPC slightly below identity mid-range (≈0.46 at
  0.5) with or without noise; non-target electrodes 0 throughout.

---

## 2026-09-23 — Snakefile, smokescreen, simulations, no log power

- **Snakefile** runs prepare → ROI power + ROI synchrony → power-synchrony for
  each config `runs:` entry. `--config smokescreen=true` = first 3 subjects,
  outputs under `*.smokescreen` (`WOA_SMOKESCREEN` in project_paths).
  `profiles/default` sets the greedy scheduler (default solver is x86-only).
- **`snakemake simulations`**: `--simulation-tag` on all build scripts; EEG is
  replaced by a `config/simulation_config.yaml` generator, pickles go to
  `<scratch>/sim/<tag>`, figures to `<figures>/simulations/<tag>`. New tags:
  `hg_null`, `hg_power_frontal` (70-150 Hz gain after onset, frontal only),
  `hg_ppc_frontal` (110 Hz phase coupling after onset, frontal only).
  Smoke result: effects recovered in frontal only (power d ≈ +5.8, synchrony
  +1.0), null ≈ 0 everywhere — after the buffer fix below.
- **`real_data_buffer_ms` 50 → 500, and build_roi_power now uses it** (it loaded
  0 ms for multitaper, ~45 ms for Morlet). The 4 Hz-wide 100/120/150 Hz notches
  ring for a few hundred ms; with the windows 0–50 ms from the clip edges, null
  power read d ≈ -0.08 in every ROI. Changes every prepost result.
- All simulation code moved from `helper.py` to `simulate_eeg.py`.
- **Power is no longer log-transformed** (PI): Cohen's d on raw band power.
  Pickle keys `log10_lo/hi` → `pow_lo/hi`, so old pickles recompute.
- `fc.run_sessions` writes `<scratch>/logs/<stage>_errors.log`, one capped line
  per failed session, overwritten each run.

---

## 2026-09-21 — OpenNeuro / BIDS data; no cluster dependency

The pipeline no longer needs rhino. Data comes from CML's public BIDS datasets
on OpenNeuro through `bidsreader` (the BIDS counterpart of cmlreaders) and
`cml_data.py` (S3 listing / download / cache, copied verbatim from
COGS4290_DataMemoryBrainsSolutions `grader-autograder`); parallelism is a local
process pool (`--workers N`) instead of cmldask/SLURM. Verified end to end on
R1111M FR1 (OpenNeuro `ds004789`): prepare_sessions → build_roi_power →
build_roi_synchrony → build_power_synchrony, single- and multi-worker.

### Session key

`(sub, exp, sess)`. BIDS has no localization/montage, so `loc`/`mon` are gone
from `ftag`, `get_dfrow`, `sid` tuples, `sess_list_df*`, `excluded_sessions.csv`,
`unrecoverable_sessions.csv` and the exclusion logs. Old pickles / session
lists are not readable by the new code.

### Data loading (`helper.py`)

- `bids_reader(dfrow)` → `CMLBIDSReader` for the session, fetching its files on
  first use; `prefetch_bids(dfrows)` fetches a list up front (used before any
  multi-worker run so workers never download).
- `get_eeg` epochs the bipolar recording with `mne.Epochs` on the events'
  `eegoffset` (= BIDS `sample`) and returns the same `(event, channel, time)`
  PTSA TimeSeries as before, in microvolts with time in ms, channels in
  `get_pairs` order. Repeated onsets (WORD + its PRE_WORD copy) are epoched
  once and expanded. Events whose clip would run off the recording are dropped
  from both the EEG and the mask (reported), where cmlreaders asserted.
- `load_pairs_table(reader)` builds the pairs table from
  `load_combined_channels(acquisition='bipolar')` in the column vocabulary the
  pipeline reads: `label`, `contact_label_1/2`, `type_1/2` (D/G/S from the BIDS
  `description`), `mni.x/y/z` (pair centroid, MNI152NLin6ASym), `distance`,
  `stein.region` / `wb.region` / `ind.region`, `hemisphere`.
- **A pair takes contact 1's atlas label, or contact 2's when contact 1 has
  none.** cmlreaders' pairs.json looked the atlas up at the pair midpoint;
  BIDS has no pair-level lookup, so pairs straddling two regions can be
  labelled differently from the cmlreaders pipeline. Checked on R1111M FR1
  ses-0 against the cmlreaders test fixtures (2026-09-22): identical 141 pairs
  and per-contact labels (100/100), identical WORD onsets (288/288); Burke ROI
  agrees for 129/141 pairs, fine region for 110/141. The misses are the
  midpoint-vs-contact rule plus main's grid cascade taking `dk` where BIDS
  takes `ind`. Monopolar contacts carry the same labels in both, so a
  monopolar analysis regionalizes identically.
- Regionalization cascades trimmed to the three atlases BIDS carries:
  depths `stein → wb`, grid/strip `stein → ind`; `n/a` is a missing-label
  sentinel. The localization-table merge is gone.
- Removed: the cmlreaders loader + NFS retry, the legacy PTSA/`TalReader`
  path (`/data/eeg`), `get_localization`, `coerce_unhashable_event_fields`,
  `eeg_data_source` plumbing in the simulation hooks.

### Session preparation

- `data_check.py` rewritten: the session list is every (sub, exp, sess)
  OpenNeuro lists for FR1/catFR1/pyFR (`cml_data.session_dataframe`); the
  per-session check loads the pairs table and one WORD-locked clip through the
  production loader, drops flat pairs by scanning the recording, checks the
  phase path, writes `electrode_information/pairs/<ftag>_pairs.json`. Gone:
  the hard-coded 284-row cohort list, the curated `channels_to_drop` table
  (pairs ARE the recording's channels now), the `.mat`/`/protocols` readers,
  the mstime/eegoffset consistency check (BIDS `sample` is derived from onset).
- `load_events.py` and `match_events.py` (the stage-2 event writer this repo
  imported but did not contain) are ported from fc_methods_comparison_cml
  `riley-thesis`. `MatchedEvents.load_events` adapts BIDS events to the CML
  schema (`type`=trial_type, `mstime`=onset·1000, `eegoffset`=sample,
  `rectime`=response_time·1000, `trial`=list, one constant `eegfile`). The
  per-session `R1171M` unix-mstime fix was dropped (BIDS onsets are
  recording-relative); the cmlreaders `correct_retrieval_offsets` /
  `sort_eegfiles` corrections are assumed applied by the BIDS conversion.
- `prepare_sessions.py`: `--workers N`, `--subjects ...`; the new-experiment
  cohort (`--newexps`, ICatFR1/IFR1/RepFR1, `presentation_times.py`,
  `RepFR1_session_presentation_times.csv`, `newexps_overrides`) is removed.

### Dispatch

`fc.run_sessions(fn, items, desc, workers=1, ...)` (process pool) replaces
`run_sessions_local` / `run_sessions_cluster`; `run_compute_stage` and
`prepare_sessions` use it. `dask_client.py`, `cluster.py`, `scripts/env.sh`
and every `--local/--n-workers/--mem/--walltime/--cluster-log-dir/--cluster`
flag are gone; `--workers N` everywhere.

### Geometry

Every contact is in one space, so `pair_xyz_lead(dfrow)` → `(xyz, lead)` from
`mni.*`, and `pair_distance_mask` no longer takes `is_depth` /
`drop_cross_type` (the flag is removed from all scripts). The cmlreaders
contacts-table MNI cache and the duplicate geometry code in
`plot_phase_conn_distance.py` are deleted.

### Config

`config.yaml` / `project_paths.py` reduced to what this repo reads:
`paths.{scratch_dir, exclusion_counts, unrecoverable_sessions_csv}`, bands,
behavior lists, signal-processing constants, `computation_metrics`. Removed:
`python_env_path`, `cluster`, `compute_profiles`, `smokescreen*`,
`newexps_overrides`, `case_study`, the thesis results/intermediate path tree,
`seed_offset`, `behaviors:` per-beh block, and the corresponding constants.
Dependencies are pinned in `requirements.txt` (ptsa and bidsreader from
GitHub).

### Known gaps

- OpenNeuro coverage vs the rhino cohort (2026-09-21): FR1 149/153 subjects,
  catFR1 31/33, pyFR 42/80. The conversion fix is on rhino, not yet released.
- `ri` events fall below the 10-event threshold on many sessions (unchanged
  behaviour; `ri` is not a required behavior).
- `visualize_phase_connectivity.ipynb` still builds `loc`/`mon` dfrows.
- The `aec_envelope` simulation DGP imports `fc_aec_dgp`, which this repo
  never contained (it lives in fc_methods_comparison_cml `sim/`).

---

## 2026-08-17 — High-gamma band + Morlet FC mode

Adds a high-gamma analysis band and an alternate Morlet (time-resolved)
spectral estimator alongside the existing multitaper path.

**Design rule throughout: default behaviour is unchanged.** Every existing
result stays bit-identical and no pickle needs recomputing or migrating. The new
band, the new estimator, and harmonic notching are all opt-in.

---

### 1. Parameters (config/config.yaml)

| key | value | new? | effect when left at default |
|---|---|---|---|
| `bands.high_gamma` | `[70, 150]` | new | none — nothing selects it unless `--band high_gamma` |
| `notch_harmonics_up_to_hz` | `null` | new | none — fundamental-only notch, historical behaviour |
| `fc_mode` | `multitaper` | new | none — existing estimator |
| `cwt_fnum` | `6` | new | Morlet path only |
| `cwt_morlet_reps` | `5` | new | Morlet path only (Rao's value) |
| `cwt_buffer_n_sigma` | `4.0` | new | Morlet path only |

Unchanged: `gamma [70,110]`, `resample_hz 500`, `min_sample_rate_hz 499`,
`mirror_buffer_ms 1000`, `real_data_buffer_ms 50`, `mt_bandwidth 2`.

### Why `high_gamma` is separate from `gamma`

`gamma [70,110]` is a **filter** band — the PAC amplitude band and the AEC/AEC-c
pre-envelope filter — whose 110 Hz upper edge is set by MNE stopband fitting,
not by scientific interest. `high_gamma [70,150]` is an **analysis** band.
Merging them would silently change existing AEC/PAC results.

### Why 70–150

Field-standard high gamma, and the range Long et al. use on this same dataset,
so the two are directly comparable. Geometric centre `sqrt(70*150)` = 102.5 Hz —
i.e. centred at ~100 Hz on the log-frequency axis Morlet wavelets tile, which is
the only sense in which "centred at 100" is meaningful for a wavelet bank.

Nyquist is safe with no new guard: sessions below 499 Hz native are already
excluded upstream (`min_sample_rate_hz` → `data_check` reason `sub_500hz`), so
every retained session has Nyquist ≥ 249.5 Hz and the resample to 500 Hz never
upsamples.

### Why `cwt_buffer_n_sigma = 4.0`

Buffer is derived from the Morlet Gaussian width, σ = `morlet_reps/(2πf)`, so it
scales with frequency instead of being a fixed constant (5σ is 995 ms at 4 Hz
but only 57 ms at 70 Hz — no single number serves both).

At 4σ and fmin = 70 Hz the requirement is **45.5 ms/side**, which the existing
`real_data_buffer_ms: 50` already covers, leaving **9.1 ms of clean separation**
between the buffered `word_on` pre/post windows. So `PREPOST_SPEC` needs no
change: `pre_win (-700,-100)` and `post_win (0,600)` both stay 600 ms,
`equalize_time_length` stays a no-op, and stimulus onset stays at t = 0.

5σ would have needed 113.7 ms against a 100 ms gap, forcing a window change, in
exchange for reducing leakage from 1.1e-7 to 1.4e-11 **in power**. Not worth
23 ms of analysis window.

---

### 2. Methods changed

### `helper.notch_filter` — new `harmonics_up_to_hz` argument

Signature: `notch_filter(eeg, sub, harmonics_up_to_hz=None)`. Backwards
compatible; `None` = fundamental only = historical behaviour.

When set, notches harmonics at ±2 Hz each — the convention already assumed by
`wavelet.Wavelet.MakePlot`, which plots 58–62, 118–122, 178–182.

**Notching is site-uniform.** It removes the union of *both* mains standards'
harmonics for every subject, not just that subject's own fundamental. Per-site
notching would give subjects different effective passbands: a Freiburg (50 Hz)
subject loses 100 and 150 Hz, a US (60 Hz) subject loses 120 Hz. Within
`high_gamma` that is 10% vs 5% of the band, **and the German hole sits at the
band centre**. Group-averaging such subjects means averaging different frequency
content, so an apparent site difference could be a filtering artifact. The union
costs 15% of the band for everyone, but every subject contributes an identical
passband.

For `high_gamma`, set `notch_harmonics_up_to_hz: 150` — that notches exactly
{100, 120, 150}. Larger values only add pointless passes at 180/200 Hz.

> **Not setting this while analysing high gamma is a silent failure mode**: the
> band fills with mains, worst at Freiburg where contamination lands on 100 Hz.

### `helper.get_beh_eeg`

Now passes `NOTCH_HARMONICS_UP_TO_HZ` through to `notch_filter`. No behavioural
change at the default `null`.

### `build_roi_power.run_sess_power` — Morlet estimator

Takes `fc_mode`. `cwt_morlet` selects `band_power_morlet` instead of
`band_power`; both return raw per-event per-channel power `(E, C)`, so they are
drop-in alternatives and the caller's log10 / Cohen's d chain is untouched.

Morlet needs real data either side of the analysis window, so `run_sess_power`
widens each window by `fc.morlet_buffer_ms` before slicing and
`band_power_morlet` trims it back off after the transform — no analysed sample
sits inside the wavelet edge artifact. `n_win_samples` records the *trimmed*
length. For prepost behaviours it also calls `assert_windows_separable`, so a
band or window change that would make the buffered pre/post windows overlap
fails at compute time rather than silently.

`band_power_morlet` uses the same PTSA primitive as `helper.get_power`
(`MorletWaveletFilter(width=5, output='power', complete=True)`) but does not
call it: `get_power` is specialised for Rao's 1000 ms mirror-buffered theta
clips, hardcoding a 1000 ms buffer clip that would erase a 600 ms word_on
window, and folding in log10 and an across-event z-score at the wrong point in
this chain.

Verified on synthetic data with a 100 Hz signal in one channel of six: both
estimators pick the same channel, signal/noise power ratio 26.6 (multitaper)
vs 23.3 (Morlet).

### `fc_comparison_functions.compute_spectral_fc` — cwt grid default

The `cwt_morlet` branch previously defaulted to `np.arange(fmin, fmax+0.5, 1)`.
For 70–150 that is **81 wavelets**, but a 5-rep Morlet at 100 Hz has ~40 Hz
half-power bandwidth, so those estimates are near-duplicates. Now defaults to a
log-spaced bank via `default_cwt_freqs`.

Measured on 16 channels × 40 epochs: **2.83 s → 0.13 s (22× faster)**, grid
`[70, 81.5, 95, 110.6, 128.8, 150]`.

Only affects the `cwt_morlet` branch, which nothing called before this work. The
`multitaper` branch is untouched.

---

### `build_roi_power` — latency bins (`time_bin_ms`, default 50)

`band_power_morlet(..., bin_ms=)` returns `(E, C, n_bins)` instead of `(E, C)`,
binning with Rao's own `helper.timebin_power_timeseries` — the function behind
his 200 ms epochs. `run_sess_power` then computes Cohen's d independently within
each bin, giving **d as a function of latency**, stored as `cohens_d_bins`
`(n_ch, n_bins)` plus `bin_centers_ms`.

50 ms rather than Rao's 200 ms because high gamma affords far more temporal
resolution: a 5-cycle Morlet at 70 Hz has sigma = 11.4 ms, so 50 ms bins are
~4.4 sigma apart and effectively independent. (Rao's 200 ms bins at 3 Hz sit
INSIDE a 265 ms sigma, so his adjacent bins are correlated — his time axis is
oversampled relative to what his wavelets resolve.) 50 ms also separates the
early latencies Long et al. report on this dataset — bitmap 40 ms,
orthographic neighbourhood 80 ms, word frequency 120 ms — which 100 ms bins
would merge. Floor is ~25 ms (~2 sigma at 70 Hz).

Bins apply only to `fc_mode: cwt_morlet` AND prepost behaviours; multitaper has
no time axis, and en/rm contrast two event groups in one window so there is no
pre/post time course to resolve. 600 ms windows divide evenly into 12 bins.

Verified: a 100 Hz burst planted at 200-300 ms peaks in the bins centred at 225
and 275 ms; the noise channel stays flat.

Pickles also gained provenance keys (`fc_mode`, `notch_harmonics_up_to_hz`,
`morlet_reps`, `cwt_fnum`, `buffer_ms`) so a consumer can tell what produced a
file without inferring it from the path.

### `--min-electrodes` unified at 3

`build_roi_power` default 2 -> 3, matching `build_roi_synchrony`, so the power
and synchrony figures admit the same (subject, ROI) cells. Plot-stage only — no
recompute. `build_power_synchrony` keeps `MIN_ELEC_CORR = 5`, which is a
different quantity (electrodes entering a Pearson r across electrodes; an r from
3 points is not meaningful).

### Morlet figures land in `<out-dir>/cwt_morlet/`

All three scripts, mirroring the `<band>__cwt_morlet` split on the compute side.

### Latency stage in `build_roi_power` (Morlet only)

`run_latency_stage` -> `collect_bin_table` -> `subject_roi_bin_table` ->
`bin_stats` -> `plot_time_course`. Reads `cohens_d_bins` from the pickles and
emits a small-multiple figure (one panel per ROI, mean d +/- SEM vs latency,
significant bins marked) plus two CSVs. Runs automatically in the plot stage
when `--fc-mode cwt_morlet`; no-ops for multitaper. Applies the same
`--min-electrodes` floor as the collapsed figure so both describe the same cells.

`bin_stats` is a one-sample t per (ROI, bin) with BH-FDR over the whole
ROI x bin grid. **Interim**: the principled test for a contiguous time grid is
TFCE (Rao's `tfce.py`, the machinery behind his Fig 4B). BH ignores that
neighbouring bins form clusters rather than independent tests, so it will miss
temporally extended-but-weak effects TFCE would find.

Verified on synthetic data: an effect planted at 100-250 ms in one ROI is
recovered at exactly the 125/175/225 ms bins, with no false positives elsewhere.

### Morlet grid no longer lands on a notched line (`nudge_off_notches`)

`default_cwt_freqs(70,150,6)` ended at exactly **150.0 Hz** -- the German 3rd
harmonic, which `notch_harmonics_up_to_hz: 150` removes. One of six wavelets was
centred on a hole. Offending centres now shift to the stopband edge
(150.0 -> 153.5). Any Morlet pickle computed before this is stale.

### Windowed multitaper latency axis (`band_power_windowed`, `mt_window_ms`)

Gives the multitaper path a time course on the SAME bin centres as Morlet, so
the two are comparable bin for bin. Windows are centred on Morlet's centres and
stepped by `time_bin_ms`, not by `mt_window_ms`.

Two constraints surfaced while building it, both consequences of multitaper
being CONSTANT-BANDWIDTH (unlike Morlet, constant-Q — see the buffer note):

1. **50 ms windows are unusable.** Resolution is 1/T and the NW=2 half-bandwidth
   is NW/T: at T=50 ms that is 20 Hz resolution and a 40 Hz half-bandwidth, so
   the whole 80 Hz band collapses into ~one resolution element. 200 ms is the
   floor. Multitaper gains no time resolution by moving up in frequency.
2. **`mt_bandwidth: 2` is illegal below ~500 ms.** MNE's normalized half-
   bandwidth is NW = T*bw/2 and it refuses NW < 0.5; 2 Hz over 200 ms is NW=0.2.
   Fixed by holding the TIME-BANDWIDTH PRODUCT constant instead of the bandwidth
   in Hz: `bw_win = bandwidth * dur/window`, so 600 ms @ 2 Hz and 200 ms @ 6 Hz
   are both NW = 0.6 — identical taper count and estimator shape. The spectral
   smoothing in Hz necessarily differs (2 -> 6); that is the uncertainty
   principle, not a choice.

Bins whose full window falls outside the analysis window return NaN rather than
being truncated (a shorter window has different resolution and is not comparable
to its neighbours). For 600 ms / 50 ms bins / 200 ms windows that is centres 25,
75, 525, 575 — leaving 8 comparable centres at 125-475 ms.

**Effective temporal resolution is `mt_window_ms` (200 ms), not `time_bin_ms`
(50 ms).** The axis is merely SAMPLED at 50 ms; consecutive windows overlap by
150 ms. Morlet's 50 ms bins are ~4.4 sigma apart at 70 Hz and effectively
independent; these are not. This is a cross-check on the Morlet time course, not
an equivalent measurement, and must be labelled as such wherever both are shown.

Verified: a 100 ms burst at 200-300 ms peaks at the 275 ms bin under BOTH
estimators, but Morlet confines it to two bins (SNR 44/51) while multitaper
spreads it over ~250 ms (SNR 4/17/31/31/18/5) and loses peak amplitude to
within-window averaging.

### 3. Additions

All delegate to existing modules — no new Morlet math.

| function | file | delegates to |
|---|---|---|
| `band_dirname(band, fc_mode)` | `fc_comparison_functions.py` | — |
| `default_cwt_freqs(fmin, fmax, fnum, morlet_reps)` | `fc_comparison_functions.py` | `wavelet.Wavelet(...).freqs` |
| `morlet_buffer_ms(fmin, morlet_reps, n_sigma)` | `fc_comparison_functions.py` | `Wavelet.get_morlet_width` |
| `assert_windows_separable(pre_win, post_win, fmin, ...)` | `fc_comparison_functions.py` | `morlet_buffer_ms` |
| `FC_MODES` tuple | `fc_comparison_functions.py` | — |
| `band_power_morlet(seg, sf, fmin, fmax, fnum, morlet_reps, buf_samples)` | `build_roi_power.py` | PTSA `MorletWaveletFilter` |
| `NOTCH_HARMONICS_UP_TO_HZ`, `FC_MODE`, `CWT_FNUM`, `CWT_MORLET_REPS`, `CWT_BUFFER_N_SIGMA` | `project_paths.py` | config |

`assert_windows_separable` raises (not warns) when the buffered pre/post windows
would overlap: overlap means the same samples inform both arms of a contrast,
manufacturing agreement between conditions that must be independent.

### Output-path separation

`band_dirname` keeps the two estimators from colliding:

```
multitaper  ->  <beh>/fc_mats/<cond>/high_gamma/
cwt_morlet  ->  <beh>/fc_mats/<cond>/high_gamma__cwt_morlet/
```

Multitaper returns the bare band name, so **every existing path and pickle stays
where it is**. Without this, a Morlet run would silently overwrite multitaper
results for the same band/behaviour.

---

### 4. CLI

`--fc-mode {multitaper,cwt_morlet}` added to all three scripts, defaulting to
`fc.FC_MODE` (config).

| script | role | status |
|---|---|---|
| `build_roi_power.py` | computes power | **fully live in both modes** |
| `build_roi_synchrony.py` | computes phase FC | flag parsed; `cwt_morlet` **exits with an error** (estimator not threaded) |
| `build_power_synchrony.py` | **consumes only** | flag selects which pickles to read |

`build_roi_synchrony.py` refuses `--fc-mode cwt_morlet` rather than running:
its estimator is still multitaper, so the run would write multitaper numbers
into a directory labelled `cwt_morlet`. Failing loudly beats a mislabelled
pickle.

Note `build_power_synchrony.py` computes nothing — its lines 100/109 are
`SystemExit` messages instructing you to run the other two first. Its flag is a
path *selector*, not a compute option.

---

### 5. Verified

- config loads; `high_gamma (70,150)`, `gamma (70,110)` intact
- `fc_comparison_functions` and `helper` import cleanly
- both FC modes run on synthetic data with a known 100 Hz phase-locked pair:
  multitaper PPC 0.132, Morlet PPC 0.754, both ≈0 for the noise pair
  (Morlet higher because multitaper averages the whole band including
  noise-only frequencies)
- 22× grid speedup measured
- `assert_windows_separable` fires correctly at 5σ, passes at 4σ
- all three `--help` invocations parse

Not run: anything against real EEG (no data access from this session).

---

### 6. Not done yet

1. **`fc_mode` threading for the FC path.** Done for power
   (`run_sess_power`); still needed for `build_roi_synchrony.py`, which goes
   two levels deeper: `run_sess_phase_fc` → `compute_session_fc` →
   `compute_prepost_separate`, plus its five `cond_dir` call sites. Guarded
   until then.
2. **Time-resolved output** — the actual latency payload. `compute_spectral_fc`
   still does `out.mean(axis=-1)`. Planned route reuses existing functions:
   `helper.get_phase` → `helper.timebin_phase_timeseries(bin_width_ms=200)` →
   `fc.event_angles` → `fc.conn_from_z`, i.e. Rao's published method, with the
   200 ms binning supplying the latency axis.
3. **Vendor `tfce.py`** from `RaoEtal25JNeurosci` for time × frequency cluster
   statistics (his Fig. 4B machinery). `tfce_compute` is pure numpy +
   `skimage.measure.label`; the `xarray` / `BehavioralData` imports at the top of
   that module belong to other functions and do not need to come along.
4. **Call `assert_windows_separable` inside `compute_prepost_separate`** so a
   future window or band change cannot silently reintroduce buffer overlap.
