#!/usr/bin/env bash
# Pool member sites (stage 7c) and fit the outcome model on the stack (stage 8, pooled).
#
#   ./run_pooled.sh [--members a,b,c] [--cohort a,b] [--pool-only | --model-only]
#
# --members a,b,c  write sites/pooled.json with these members (order sets Site A, B, ...);
#                  without it, the existing sites/pooled.json is used
# --cohort a,b     stage-8 cohorts to fit, one after another (default: every COHORTS entry in
#                  code/08_outcome_model.R, all,early_nonop,early_nonop_hypoxemia); only early_nonop
#                  gets the sensitivity analyses and covariate blocks
# --pool-only      stop after stage 7c
# --model-only     skip stage 7c and refit stage 8 on the existing pooled data
#
# Every member must already have run stages 0-7 (./run_scores.sh --site <member>).
set -euo pipefail
cd "$(dirname "$0")"

MEMBERS=""
COHORTS="all,early_nonop,early_nonop_hypoxemia"
POOL_ONLY=0
MODEL_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --members)     MEMBERS="$2"; shift 2 ;;
    --members=*)   MEMBERS="${1#*=}"; shift ;;
    --cohort)      COHORTS="$2"; shift 2 ;;
    --cohort=*)    COHORTS="${1#*=}"; shift ;;
    --pool-only)   POOL_ONLY=1; shift ;;
    --model-only)  MODEL_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$POOL_ONLY" -eq 1 && "$MODEL_ONLY" -eq 1 ]]; then
  echo "--pool-only and --model-only cannot be combined" >&2; exit 2
fi
IFS=',' read -r -a COHORT_LIST <<< "$COHORTS"
for c in "${COHORT_LIST[@]}"; do
  if [[ ! "$c" =~ ^[A-Za-z0-9_]+$ ]]; then echo "bad cohort name in --cohort: '$c'" >&2; exit 2; fi
done

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No virtualenv at .venv. Create it with:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

PROFILE="sites/pooled.json"
if [[ -n "$MEMBERS" ]]; then
  IFS=',' read -r -a M <<< "$MEMBERS"
  if [[ ${#M[@]} -lt 2 ]]; then echo "--members needs at least two sites" >&2; exit 2; fi
  for m in "${M[@]}"; do
    if [[ ! "$m" =~ ^[A-Za-z0-9_-]+$ ]]; then echo "bad site name in --members: '$m'" >&2; exit 2; fi
  done
  items="$(printf '"%s", ' "${M[@]}")"
  printf '{\n  "members": [%s]\n}\n' "${items%, }" > "$PROFILE"
  echo "wrote $PROFILE: ${M[*]} (Site A = ${M[0]})"
elif [[ ! -f "$PROFILE" && "$MODEL_ONLY" -eq 0 ]]; then
  echo "No $PROFILE. Pass the members, e.g.: ./run_pooled.sh --members site_one,site_two,site_three" >&2
  exit 1
fi
if [[ "$MODEL_ONLY" -eq 1 && ! -f output/pooled/final_no_phi/unit_panel.parquet ]]; then
  echo "No pooled data to refit (output/pooled/final_no_phi/unit_panel.parquet); run without --model-only" >&2
  exit 1
fi

if [[ "$POOL_ONLY" -eq 0 ]]; then
  if ! command -v Rscript > /dev/null; then
    echo "Rscript not found. Install R (the version renv.lock pins), then: Rscript -e 'renv::restore(prompt = FALSE)'" >&2
    exit 1
  fi
  # Checked before the pooling step so a missing package fails in seconds, not after it.
  if ! Rscript -e 'suppressPackageStartupMessages({library(arrow); library(lme4); library(ggplot2)})' > /dev/null 2>&1; then
    echo "R packages are not installed for this project. Run once (10-30 min):" >&2
    echo "  Rscript -e 'renv::restore(prompt = FALSE)'" >&2
    exit 1
  fi
fi

fmt(){ printf '%dh%02dm%02ds' $((${1}/3600)) $(((${1}%3600)/60)) $((${1}%60)); }
T0=$SECONDS

if [[ "$MODEL_ONLY" -eq 0 ]]; then
  echo ""; echo "========== 07c_pool_sites =========="
  "$PY" code/07c_pool_sites.py || {
    echo ">>> 07c refused or failed; nothing was written. Its log: ./show_results.sh --site pooled" >&2
    exit 1
  }
fi

if [[ "$POOL_ONLY" -eq 1 ]]; then
  echo ""; echo "=== pooled in $(fmt $((SECONDS - T0))) — log: ./show_results.sh --site pooled ==="
  exit 0
fi

# Cohorts are independent fits on the same data, so they run side by side, each to its own log.
LOGS="output/pooled/final_no_phi/logs"
mkdir -p "$LOGS"
PIDS=()
echo ""; echo "========== 08_outcome_model (pooled, cohorts in parallel) =========="
for c in "${COHORT_LIST[@]}"; do
  CLIF_SITE=pooled CLIF_COHORT="$c" Rscript code/08_outcome_model.R > "$LOGS/08_outcome_model_$c.log" 2>&1 &
  PIDS+=($!)
  echo "  started $c -> $LOGS/08_outcome_model_$c.log"
done
FAILED=()
for i in "${!PIDS[@]}"; do
  c="${COHORT_LIST[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "  finished $c ($(fmt $((SECONDS - T0))) since start)"
  else
    FAILED+=("$c")
  fi
done
if [[ ${#FAILED[@]} -gt 0 ]]; then
  for c in "${FAILED[@]}"; do
    echo ">>> stage 8 failed for cohort $c; its log: tail -n 40 $LOGS/08_outcome_model_$c.log" >&2
  done
  exit 1
fi

echo ""
echo "=== done in $(fmt $((SECONDS - T0))) — cohorts: ${COHORT_LIST[*]} ==="
for c in "${COHORT_LIST[@]}"; do
  echo "    summaries to share: ./show_results.sh --site pooled --cohort $c"
done
