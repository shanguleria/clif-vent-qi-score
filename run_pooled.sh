#!/usr/bin/env bash
# Pool member sites (stage 7c) and fit the outcome model on the stack (stage 8, pooled).
#
#   ./run_pooled.sh [--members a,b,c] [--pool-only]
#
# --members a,b,c  write sites/pooled.json with these members (order sets Site A, B, ...);
#                  without it, the existing sites/pooled.json is used
# --pool-only      stop after stage 7c
#
# Every member must already have run stages 0-7 (./run_scores.sh --site <member>).
set -euo pipefail
cd "$(dirname "$0")"

MEMBERS=""
POOL_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --members)    MEMBERS="$2"; shift 2 ;;
    --members=*)  MEMBERS="${1#*=}"; shift ;;
    --pool-only)  POOL_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
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
elif [[ ! -f "$PROFILE" ]]; then
  echo "No $PROFILE. Pass the members, e.g.: ./run_pooled.sh --members site_one,site_two,site_three" >&2
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

echo ""; echo "========== 07c_pool_sites =========="
"$PY" code/07c_pool_sites.py || {
  echo ">>> 07c refused or failed; nothing was written. Its log: ./show_results.sh --site pooled" >&2
  exit 1
}

if [[ "$POOL_ONLY" -eq 1 ]]; then
  echo ""; echo "=== pooled in $(fmt $((SECONDS - T0))) — log: ./show_results.sh --site pooled ==="
  exit 0
fi

echo ""; echo "========== 08_outcome_model (pooled) =========="
CLIF_SITE=pooled Rscript code/08_outcome_model.R || { echo ">>> stage 8 failed" >&2; exit 1; }

echo ""
echo "=== done in $(fmt $((SECONDS - T0))) — results: output/pooled/final_no_phi/08_outcome_model/ ==="
echo "    summaries to share: ./show_results.sh --site pooled"
