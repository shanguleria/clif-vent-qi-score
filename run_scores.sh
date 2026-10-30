#!/usr/bin/env bash
# Build the ventilator QI score panel for one site.
#
#   ./run_scores.sh [--site <id>] [--rebuild] [--require-stage] [--skip-verify]
#
# --rebuild        ignore any stage cache and rebuild the cohort from raw CLIF (slow: re-runs
#                  encounter stitching and the clifpy respiratory-support waterfall)
# --require-stage  fail rather than fall back to a cold build (for CI)
set -euo pipefail
cd "$(dirname "$0")"

SITE="${CLIF_SITE:-uchicago}"
PASS=()
SKIP_VERIFY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)          SITE="$2"; shift 2 ;;
    --site=*)        SITE="${1#*=}"; shift ;;
    --rebuild|--require-stage) PASS+=("$1"); shift ;;
    --skip-verify)   SKIP_VERIFY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export CLIF_SITE="$SITE"

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

OUT="output/${SITE}"; mkdir -p "$OUT"
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
  "$PY" "$script" --site "$SITE" "$@"
  local d=$((SECONDS - t0))
  echo "$START_UTC,$SITE,$name,$d,$(fmt "$d"),$SHA" >> "$CSV"
  echo ">>> $name: $(fmt "$d")"
}

T0=$SECONDS
# Stage 0 resolves the cohort source (building it cold when needed) and records the result;
# every later stage reuses that record. --rebuild therefore goes ONLY to stage 0 — handing it
# further down would re-run the upstream waterfall build once per stage.
run_stage code/00_probe_units.py       "${PASS[@]+"${PASS[@]}"}"
run_stage code/01_build_cohort.py
run_stage code/02_metric_records.py
run_stage code/03_build_panel.py
run_stage code/04_exposure_assignment.py
run_stage code/07_analysis_dataset.py
[[ "$SKIP_VERIFY" -eq 1 ]] || run_stage code/05_verify_equivalence.py

TOTAL=$((SECONDS - T0))
echo "$START_UTC,$SITE,TOTAL,$TOTAL,$(fmt "$TOTAL"),$SHA" >> "$CSV"
echo ""
echo "=== done in $(fmt "$TOTAL") — panel: $OUT/final_no_phi/unit_period_panel.csv ==="
