#!/usr/bin/env bash
# Print the aggregate, PHI-free outputs for one site or for the pooled run, ready to share.
#
#   ./show_results.sh --site <id>      the site's covariate probe log (and its stage-8
#                                      summaries, if stage 8 has run for that site)
#   ./show_results.sh --site pooled    the pooling log and the pooled model summaries
#   --cohort <name>                    a restricted stage-8 cohort (08_outcome_model_<name>/)
#
# Reads only the files named below, all under final_no_phi/. Per-unit tables
# (unit_summary.csv, unit_effects.csv) are left out: they carry per-unit counts that can fall
# below the small-cell threshold.
set -euo pipefail
cd "$(dirname "$0")"

SITE="${CLIF_SITE:-}"
COHORT="${CLIF_COHORT:-all}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)      SITE="$2"; shift 2 ;;
    --site=*)    SITE="${1#*=}"; shift ;;
    --cohort)    COHORT="$2"; shift 2 ;;
    --cohort=*)  COHORT="${1#*=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$SITE" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "usage: ./show_results.sh --site <id>|pooled [--cohort <name>]" >&2
  exit 2
fi
if [[ ! "$COHORT" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "bad --cohort: '$COHORT'" >&2
  exit 2
fi

SHARE="output/${SITE}/final_no_phi"
MODEL="$SHARE/08_outcome_model"
[[ "$COHORT" == all ]] || MODEL="${MODEL}_${COHORT}"
SUMMARIES=(icc.csv covariate_blocks.csv sensitivity_analyses.csv model_coefficients.csv)

FILES=()
if [[ "$SITE" == pooled ]]; then
  FILES+=("$SHARE/logs/07c_pool_sites.log")
  for f in "${SUMMARIES[@]}"; do FILES+=("$MODEL/$f"); done
else
  FILES+=("$SHARE/logs/07b_covariate_probe.log")
  if [[ -d "$MODEL" ]]; then
    for f in "${SUMMARIES[@]}"; do FILES+=("$MODEL/$f"); done
  fi
fi

for f in "${FILES[@]}"; do
  echo ""
  if [[ -f "$f" ]]; then
    echo "===== $f"
    cat "$f"
  else
    echo "===== $f — not found (has that step run for '$SITE'?)"
  fi
done
