#!/usr/bin/env bash
# Build the ventilator QI score panel and patient rows for one site.
#
#   ./run_scores.sh [--site <id>] [--rebuild] [--fresh] [--from <stage>] [--probe]
#                   [--require-stage] [--skip-verify]
#
# --rebuild        ignore any stage cache and rebuild the cohort from raw CLIF (slow: re-runs
#                  encounter stitching and the clifpy respiratory-support waterfall)
# --fresh          move output/<site> aside to a timestamped backup first; implies --rebuild
# --from <stage>   start at 00, 01, 02, 03, 04 or 07, reusing the earlier stages' outputs
#                  (e.g. --from 07 after a stage-7 fix); --rebuild and --fresh need --from 00
# --probe          run the 07b covariate probe after stage 7
# --require-stage  fail rather than fall back to a cold build (for CI)
set -euo pipefail
cd "$(dirname "$0")"

SITE="${CLIF_SITE:-uchicago}"
PASS=()
SKIP_VERIFY=0
FROM=00
PROBE=0
FRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)          SITE="$2"; shift 2 ;;
    --site=*)        SITE="${1#*=}"; shift ;;
    --from)          FROM="$2"; shift 2 ;;
    --from=*)        FROM="${1#*=}"; shift ;;
    --rebuild|--require-stage) PASS+=("$1"); shift ;;
    --fresh)         FRESH=1; shift ;;
    --probe)         PROBE=1; shift ;;
    --skip-verify)   SKIP_VERIFY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export CLIF_SITE="$SITE"

case "$FROM" in
  00|01|02|03|04|07) ;;
  *) echo "--from must be one of 00 01 02 03 04 07 (got '$FROM')" >&2; exit 2 ;;
esac
if [[ "$FRESH" -eq 1 && " ${PASS[*]-} " != *" --rebuild "* ]]; then PASS+=("--rebuild"); fi
if [[ "$FROM" != 00 && ( "$FRESH" -eq 1 || " ${PASS[*]-} " == *" --rebuild "* ) ]]; then
  echo "--rebuild and --fresh rebuild the cohort in stage 0, so they need --from 00" >&2
  exit 2
fi

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No virtualenv at .venv. Create it with:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -f "sites/${SITE}.json" ]]; then
  echo "No site profile at sites/${SITE}.json. Create it with:" >&2
  echo "  cp sites/uchicago.example.json sites/${SITE}.json   # then edit data_path" >&2
  exit 1
fi

OUT="output/${SITE}"
if [[ "$FRESH" -eq 1 && -d "$OUT" ]]; then
  BACKUP="output/${SITE}_backup_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$BACKUP"
  echo "moved $OUT to $BACKUP (delete it once this run succeeds)"
fi
if [[ "$FROM" != 00 && ! -f "$OUT/intermediate_phi/cohort/source.json" ]]; then
  echo "No earlier run for '$SITE' to resume from ($OUT/intermediate_phi/cohort/source.json)." >&2
  echo "Run from stage 00 first: ./run_scores.sh --site $SITE" >&2
  exit 1
fi

mkdir -p "$OUT"
CSV="$OUT/run_timings.csv"
[[ -f "$CSV" ]] || echo "run_started_utc,site,stage,seconds,hms,git_sha" > "$CSV"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
fmt(){ printf '%dh%02dm%02ds' $((${1}/3600)) $(((${1}%3600)/60)) $((${1}%60)); }

# Each stage is appended to the CSV as it finishes, so an interrupted run keeps what completed.
run_stage(){
  local script="$1"; shift
  local name; name="$(basename "$script" .py)"
  echo ""; echo "========== $name =========="
  local t0=$SECONDS
  # Explicit exit: callers sit in && / || lists, where bash suspends set -e inside the function,
  # so a failed stage would otherwise be logged as done and the next stage would run.
  "$PY" "$script" --site "$SITE" "$@" || { echo ">>> $name FAILED" >&2; exit 1; }
  local d=$((SECONDS - t0))
  echo "$START_UTC,$SITE,$name,$d,$(fmt "$d"),$SHA" >> "$CSV"
  echo ">>> $name: $(fmt "$d")"
}
from_here(){ (( 10#$1 >= 10#$FROM )); }

T0=$SECONDS
# Stage 0 resolves the cohort source (building it cold when needed) and records the result;
# every later stage reuses that record. --rebuild therefore goes ONLY to stage 0 — handing it
# further down would re-run the upstream waterfall build once per stage.
from_here 00 && run_stage code/00_probe_units.py "${PASS[@]+"${PASS[@]}"}"
from_here 01 && run_stage code/01_build_cohort.py
from_here 02 && run_stage code/02_metric_records.py
from_here 03 && run_stage code/03_build_panel.py
from_here 04 && run_stage code/04_exposure_assignment.py
run_stage code/07_analysis_dataset.py
[[ "$PROBE" -eq 0 ]] || run_stage code/07b_covariate_probe.py
[[ "$SKIP_VERIFY" -eq 1 ]] || run_stage code/05_verify_equivalence.py

TOTAL=$((SECONDS - T0))
echo "$START_UTC,$SITE,TOTAL,$TOTAL,$(fmt "$TOTAL"),$SHA" >> "$CSV"
echo ""
echo "=== done in $(fmt "$TOTAL") — panel: $OUT/final_no_phi/unit_period_panel.csv ==="
[[ "$PROBE" -eq 0 ]] || echo "    probe log: ./show_results.sh --site $SITE"
