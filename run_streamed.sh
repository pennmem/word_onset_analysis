#!/bin/zsh
# Juee (default) full run when the bipolar EEG (FR1+catFR1+pyFR, ~680 GB) doesn't
# fit on disk: per batch of subjects, download -> data check + events -> compute.py
# (power + synchrony, one load per session) -> delete the batch's raw bipolar EEG.
# Results are cached per session; the final plot stages only read them.
# Resumable: finished batches are listed in <scratch>/logs/stream_done.txt.
#   caffeinate -is zsh run_streamed.sh [batch_size]
# SIM_TAGS="hg_null sync_equal": the same per simulation tag instead of the real data
# (full cohort, as the real analyses); figures/results -> */simulations/<tag>/.
# SIM_PLOT_TAGS: plot only these tags (default: all SIM_TAGS; e.g. sweeps are compute only).
cd "${0:A:h}"
PY=~/.venvs/phase_viz/bin/python; B=${1:-10}
ROOT=$($PY -c "from project_paths import SCRATCH_DIR; print(SCRATCH_DIR)" 2>/dev/null | tail -1)
CACHE=$($PY -c "from project_paths import get; print(get('bids_cache'))" 2>/dev/null | tail -1)
BANDS=($($PY -c "import yaml; print(' '.join(r['band'] for r in yaml.safe_load(open('config/config.yaml'))['runs']))"))
export CML_AUTO_APPROVE=1
mkdir -p $ROOT/logs; DONE=$ROOT/logs/stream_done.txt; touch $DONE
drop_raw () { find $CACHE/ds004789 $CACHE/ds004809 $CACHE/ds004865 -name "*acq-bipolar*_ieeg.*" ! -name "*.json" -delete 2>/dev/null }
retry () { local k; for k in 1 2 3; do "$@" && return 0; echo "[stream] retry $k: $*"; sleep 120; done; return 1 }   # local: the batch loop uses $i

retry $PY prepare_sessions.py --setup-only || exit 1   # session listing (network)
SUBS=(${=STREAM_SUBJECTS:-$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('$ROOT/sess_list_df_initial.csv')['sub'].unique())))")})   # STREAM_SUBJECTS: only these
echo "[stream] ${#SUBS} subjects, batches of $B, bands: $BANDS"
for ((i = 1; i <= ${#SUBS}; i += B)); do
  batch=(${SUBS[$i,$((i + B - 1))]})
  grep -qx "$batch" $DONE && continue
  echo "[stream] $(date +%H:%M) batch $(( (i - 1) / B + 1 )): $batch"
  retry $PY prepare_sessions.py --subjects $batch --workers 3 || continue   # downloads this batch
  for t in ${=SIM_TAGS:-real}; do   # single pass; earlier batches are cached
    S=(); [[ $t != real ]] && S=(--simulation-tag $t)   # array: zsh does not word-split
    CML_DATA_SOURCE=local $PY compute.py --workers 3 $S
  done
  drop_raw; echo "$batch" >> $DONE
done
for t in ${=SIM_PLOT_TAGS:-${SIM_TAGS:-real}}; do
  F=figures; R=results; S=()
  [[ $t != real ]] && F=figures/simulations/$t && R=results/simulations/$t && S=(--simulation-tag $t)
  case $t in   # a control runs only the analyses it tests; real, nulls and ceiling run all
    leak_frontal) only=(plot_boxplots plot_epoch_network) ;;
    hg_latency) only=(plot_latency) ;;
    *) only=() ;;
  esac
  for s d in plot_boxplots . plot_spectrum burke_roi_power \
             plot_timecourse burke_roi_power plot_epoch_network burke_roi_synchrony plot_latency latency \
             plot_anticipation latency plot_decoding decoding plot_reliability reliability; do
    (( ${#only} )) && (( ! ${only[(Ie)$s]} )) && continue
    CML_DATA_SOURCE=local $PY $s.py $S --out-dir $F/$d   # figures from the cache
  done
  (( ${#only} )) && continue
  Rscript mixed_models.R latency $R/latency/hg_latency_word_on_electrodes.csv
  Rscript mixed_models.R bands $R/burke_roi_power/roi_power_word_on_alpha_per_electrode.csv
  [[ $t == real ]] && $PY plot_mixed.py
done
drop_raw
