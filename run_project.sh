#!/bin/zsh
# Top-level wrapper for the Snakefile (as fc_methods_comparison_cml's run_project.sh); no env activation needed.
#   zsh run_project.sh                  # full run: prepare_sessions -> compute -> proposal figures -> proposal.pdf
#   zsh run_project.sh --smokescreen    # plot stages on 40 subjects of the existing cache -> *.smokescreen
#   zsh run_project.sh --dry-run        # show what would run, don't execute (combine with --smokescreen)
#   zsh run_project.sh --target timing  # one rule (and what it needs); --forceall reruns everything
#   zsh run_project.sh --jobs 4         # snakemake --cores (default 6)
# Runs under caffeinate (Mac sleep stretched a ~1 h run to 11 h). If the bipolar EEG does not fit on disk, compute
# with run_streamed.sh first (resumable batches); the rules then find the cache. Logs: logs/run_<time>.log.
cd "${0:A:h}"
PY=${PYTHON:-~/.venvs/phase_viz/bin/python}; SNAKEMAKE=${SNAKEMAKE:-${PY:h}/snakemake}
JOBS=6; TARGET=all; ARGS=(); CONFIG=()
while (( $# )); do
  case $1 in
    --smokescreen) CONFIG=(--config smokescreen=true); shift ;;
    --dry-run|-n)  ARGS+=(-n); shift ;;
    --target|-t)   TARGET=$2; shift 2 ;;
    --forceall|-F) ARGS+=(--forceall); shift ;;
    --jobs|-j)     JOBS=$2; shift 2 ;;
    --help|-h)     sed -n '2,9p' $0; exit 0 ;;
    *)             ARGS+=($1); shift ;;
  esac
done
[[ -x $SNAKEMAKE ]] || { echo "[run_project] snakemake not found at $SNAKEMAKE ($PY -m pip install snakemake)"; exit 1 }
mkdir -p logs; LOG=logs/run_$(date +%Y%m%d_%H%M).log
echo "[run_project] target $TARGET, jobs $JOBS, ${CONFIG:-full run} ${ARGS} -> $LOG"
caffeinate -is $SNAKEMAKE --cores $JOBS --keep-going --printshellcmds --rerun-incomplete $TARGET $ARGS $CONFIG 2>&1 | tee $LOG
