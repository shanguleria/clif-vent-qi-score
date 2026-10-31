"""Stage 0 — aggregate-only probe of ICU units and definition integrity.

Answers the questions the panel design depends on, BEFORE anything expensive runs:
  * does this site's adt table carry `location_name` at all (it is OPTIONAL in CLIF 2.1)?
  * how many distinct ICU units does it resolve to, and how much volume does each carry?
  * does each `location_name` nest within exactly one `location_type`?
  * do the shipped definitions still match the pinned upstream commit?

Prints ONLY aggregate counts. No row-level data and no raw CLIF table reaches stdout.

Run:  .venv/bin/python code/00_probe_units.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import cohort_source                                       # noqa: E402

# Optional: only used to re-diff definitions/ against the pinned upstream commit.
# Defaults to a sibling checkout; override with CLIF_DASHBOARDS_DIR.
UPSTREAM_CLONE = Path(os.environ.get(
    "CLIF_DASHBOARDS_DIR",
    Path(__file__).resolve().parents[2] / "clif-dashboards"))


def check_definitions() -> bool:
    """Recompute the definition hashes and, when the upstream clone is reachable, diff them."""
    info = bc.source_info()
    sha = info["upstream_sha"]
    print(f"\n[definitions] pinned upstream {info['upstream_sha_short']} ({info['upstream_repo']})")
    ok = True
    for name, recorded in info["files"].items():
        path = bc.DEFINITIONS / name
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != recorded:
            print(f"  FAIL {name}: hash differs from definitions/SOURCE")
            print(f"       recorded {recorded}")
            print(f"       actual   {actual}")
            ok = False
            continue
        if not (UPSTREAM_CLONE / ".git").is_dir():
            print(f"  ok   {name}: hash matches SOURCE (upstream clone not reachable; no diff)")
            continue
        res = subprocess.run(["git", "-C", str(UPSTREAM_CLONE), "show",
                              f"{sha}:config/decisions/{name}"],
                             capture_output=True)
        if res.returncode != 0:
            print(f"  ok   {name}: hash matches SOURCE (commit {sha[:7]} not in local clone)")
            continue
        if hashlib.sha256(res.stdout).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest():
            print(f"  ok   {name}: byte-identical to upstream @ {sha[:7]}")
        else:
            print(f"  FAIL {name}: DIFFERS from upstream @ {sha[:7]} — definitions have drifted")
            ok = False
    return ok


def check_composite_coverage() -> bool:
    """Every composite metric must name a decision block that exists upstream."""
    print("\n[composite] metric -> upstream decision block")
    comp = bc.load_composite()
    upstream_metrics = bc.decisions()["metrics"]
    ok = True
    for mid, spec in comp["metrics"].items():
        key = spec["upstream_decision_key"]
        if key in upstream_metrics:
            print(f"  ok   {mid:<10} -> decisions.metrics.{key}")
        else:
            print(f"  FAIL {mid:<10} -> decisions.metrics.{key} NOT FOUND upstream")
            ok = False
    # Every metric must be claimed by at least one score, or adding it was a silent no-op.
    claimed = {m for s in comp["scores"].values() for m in s["weights"]}
    for mid in comp["metrics"]:
        if mid not in claimed:
            print(f"  FAIL {mid}: declared but no score references it")
            ok = False
    # Every score weight must name a declared metric.
    for sid, s in comp["scores"].items():
        for m in s["weights"]:
            if m not in comp["metrics"]:
                print(f"  FAIL score {sid} weights unknown metric '{m}'")
                ok = False
    return ok


def check_min_denominator(site) -> bool:
    """Every min_denominator key must be a live composite metric, at every grain.

    Floors are nested by grain (`unit`, `unit_period`) because a quarter-calibrated floor can
    never bind on a seven-year pooled denominator. Both blocks are checked, since a key that is
    live at one grain and dead at the other is still a silent no-op where it is dead.
    """
    print("\n[config] min_denominator coverage")
    declared = set(bc.metric_ids())
    ok = True
    for grain in bc.GRAINS:
        try:
            floors = bc.min_denominator(grain, site)
        except KeyError as exc:
            print(f"  FAIL {grain}: {exc}")
            ok = False
            continue
        for key, val in floors.items():
            if key in declared:
                print(f"  ok   {grain:<12} {key:<10} floor={val}")
            else:
                print(f"  FAIL {grain}: '{key}' is not a composite metric — the floor is a no-op")
                ok = False
        for m in sorted(declared - set(floors)):
            print(f"  WARN {grain}: no floor for '{m}'; cells will never be flagged")
    return ok


def probe_units(src: cohort_source.CohortSource, site) -> bool:
    adt = src.read("adt")
    print(f"\n[adt] staged rows: {len(adt):,}  columns: {len(adt.columns)}")
    if "location_name" not in adt.columns:
        print("  FAIL `location_name` is absent. It is OPTIONAL in CLIF 2.1, so this site cannot be")
        print("       scored at unit granularity. Fall back to location_type, or map names upstream.")
        return False

    icu = adt[adt["location_category"].astype(str).str.lower().eq("icu")]
    named = icu["location_name"].notna() & icu["location_name"].astype(str).str.strip().ne("")
    print(f"  ICU rows                 : {len(icu):,}")
    print(f"  ICU rows with a name     : {named.sum():,} ({100 * named.mean():.1f}%)")
    print(f"  distinct ICU location_name: {icu.loc[named, 'location_name'].nunique()}")
    print(f"  distinct ICU location_type: {icu['location_type'].nunique()}")

    # Nesting invariant: a physical unit belongs to exactly one ICU type.
    nest = icu[named].groupby("location_name")["location_type"].nunique()
    bad = nest[nest > 1]
    print(f"\n[nesting] location_name -> location_type")
    if len(bad):
        print(f"  WARN {len(bad)} name(s) map to >1 location_type; unit_collapse may be needed.")
    else:
        print("  ok   every location_name maps to exactly one location_type")

    # Per-unit volume, by ICU stay count and total ICU days. Aggregate only.
    icu_named = icu[named].copy()
    icu_named["days"] = (icu_named["out_dttm"].fillna(icu_named["in_dttm"])
                         - icu_named["in_dttm"]).dt.total_seconds() / 86400
    per_unit = (icu_named.groupby(["location_type", "location_name"])
                .agg(stays=("location_name", "size"), icu_days=("days", "sum"))
                .sort_values("icu_days", ascending=False))
    per_unit["icu_days"] = per_unit["icu_days"].round(0).astype("Int64")
    print(f"\n[units] ICU stays and ICU-days per location_name  (n={len(per_unit)})")
    print(per_unit.to_string())

    # How many unit x period cells will the panel have?
    ep = src.read("episodes_a")
    tz = bc.load_profile(site)["timezone"]
    period = bc.load_profile(site).get("period", "quarter")
    freq = "Q" if period == "quarter" else "M"
    starts = pd.to_datetime(ep["start"], utc=True).dt.tz_convert(tz)
    # to_period() drops tz; localize away first so the site-local calendar is what gets binned
    n_periods = starts.dt.tz_localize(None).dt.to_period(freq).nunique()
    print(f"\n[panel shape] {len(per_unit)} units x {n_periods} {period}s = "
          f"{len(per_unit) * n_periods:,} potential cells for {len(ep):,} IMV episodes")
    print(f"  mean episodes per cell if evenly spread: {len(ep) / max(len(per_unit) * n_periods, 1):.1f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--require-stage", action="store_true")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    print(f"=== probe: site {site} ===")

    # Stage 0 runs first, so it owns cohort-source resolution — including a cold build when one
    # is needed — and records the result. Every later stage reuses that record, so the upstream
    # build runs at most ONCE per pipeline invocation rather than once per stage.
    src = cohort_source.resolve(site, rebuild=args.rebuild, require_stage=args.require_stage)
    source_file = bc.phi_dir(site) / "cohort" / "source.json"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    cohort_source.record(src, source_file)
    checks = [
        ("definitions", check_definitions()),
        ("composite", check_composite_coverage()),
        ("min_denominator", check_min_denominator(site)),
        ("units", probe_units(src, site)),
    ]
    print("\n=== summary ===")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
