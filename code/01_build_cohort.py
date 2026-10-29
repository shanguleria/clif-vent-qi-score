"""Stage 1 — resolve the cohort source and materialize the frames the metric records need.

Reads a validated clif-dashboards stage cache when one is available, otherwise runs the upstream
respiratory build to create one (see common/cohort_source.py). Either way the cohort is exactly the
dashboard's: adults on IMV, episodes split at a 24 h gap, ICU overlap required, outside-hospital
intubations excluded.

Writes (all row-level -> intermediate_phi/, never shared):
  cohort/episodes.parquet        one row per analysed IMV episode, with its period label
  cohort/runs.parquet            contiguous IMV runs, needed by the SAT/SBT phenotypes
  cohort/encounters.parquet      encounter-level attributes (patient_id, sex, age, discharge)
  cohort/icu_intervals.parquet   ICU stays keyed by location_name, for attribution
  cohort/manifest.json           provenance + which cohort-source tier was used

Prints only aggregate counts.

Run:  .venv/bin/python code/01_build_cohort.py [--site uchicago] [--rebuild|--require-stage]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import attribution as attr                                 # noqa: E402
from common import cohort_source                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore any stage cache and rebuild the cohort from raw CLIF")
    ap.add_argument("--require-stage", action="store_true",
                    help="fail rather than fall back to a cold build (for CI)")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    prof = bc.load_profile(site)
    tz, period = prof["timezone"], prof.get("period", "quarter")

    print(f"=== 01 build cohort: site {site} ===")
    src = (cohort_source.recorded(bc.phi_dir(site) / "cohort" / "source.json")
           or cohort_source.resolve(site, rebuild=args.rebuild,
                                    require_stage=args.require_stage))

    episodes = src.read("episodes_a")
    runs = src.read("runs_a")
    encounters = src.read("encounters")
    adt = src.read("adt")

    # Period label comes from the episode start in SITE-LOCAL time, matching the upstream's
    # time_attribution decision ("month of IMV episode start"); we only widen month -> quarter.
    episodes = episodes.copy()
    episodes["start"] = pd.to_datetime(episodes["start"], utc=True)
    episodes["end"] = pd.to_datetime(episodes["end"], utc=True)
    episodes["period"] = attr.period_label(episodes["start"], tz, period)

    icu = attr.icu_intervals(adt,
                             unit_include=prof.get("unit_include"),
                             unit_collapse=prof.get("unit_collapse"))

    out = bc.phi_dir(site) / "cohort"
    out.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    runs.to_parquet(out / "runs.parquet", index=False)
    encounters.to_parquet(out / "encounters.parquet", index=False)
    icu.to_parquet(out / "icu_intervals.parquet", index=False)

    manifest = {
        "provenance": bc.provenance(site, cohort_source=src.tier),
        "cohort_source": {"tier": src.tier, "reason": src.reason, "stage": str(src.stage)},
        "counts": {
            "imv_episodes": int(len(episodes)),
            "encounter_blocks_with_episode": int(episodes["encounter_block"].nunique()),
            "first_episodes": int((episodes["episode_seq"] == 1).sum()),
            "imv_runs": int(len(runs)),
            "icu_stays": int(len(icu)),
            "icu_units": int(icu["unit_name"].nunique()),
            "waterfall_chunks": len(src.waterfall_chunks()),
        },
        "period": {"grain": period, "n_periods": int(episodes["period"].nunique()),
                   "first": episodes["period"].min(), "last": episodes["period"].max()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cohort_source.record(src, out / "source.json")

    c = manifest["counts"]
    print(f"\n  IMV episodes analysed    : {c['imv_episodes']:,}")
    print(f"  in encounter_blocks      : {c['encounter_blocks_with_episode']:,}")
    print(f"  first episodes           : {c['first_episodes']:,}")
    print(f"  IMV runs                 : {c['imv_runs']:,}")
    print(f"  ICU stays (named units)  : {c['icu_stays']:,} across {c['icu_units']} units")
    print(f"  waterfall chunks to scan : {c['waterfall_chunks']}")
    print(f"  periods ({period})        : {manifest['period']['n_periods']} "
          f"({manifest['period']['first']} .. {manifest['period']['last']})")
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
