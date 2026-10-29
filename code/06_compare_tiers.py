"""Stage 6 (verification only) — prove the stage cache is an optimization, not a dependency.

Compares a panel built from a validated stage cache against one built cold from raw CLIF. They must
be identical: the cache holds nothing the cold build does not recompute.

Usage:
    .venv/bin/python code/06_compare_tiers.py --snapshot          # before a --rebuild run
    ./run_scores.sh --site <site> --rebuild
    .venv/bin/python code/06_compare_tiers.py --compare           # after it

The snapshot lands in output/<site>/tier_comparison/ (aggregate panels only, no PHI).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402

PANEL = "unit_period_panel.parquet"


def snapshot_dir(site) -> Path:
    d = bc.output_root(site) / "tier_comparison"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    ap.add_argument("--snapshot", action="store_true", help="save the current panel as the baseline")
    ap.add_argument("--compare", action="store_true", help="diff the current panel against it")
    ap.add_argument("--label", default="tier1", help="name for the snapshot")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    current = bc.share_dir(site) / PANEL
    snap = snapshot_dir(site) / f"{args.label}_{PANEL}"

    if args.snapshot:
        if not current.exists():
            print(f"  no panel at {current} to snapshot"); return 1
        shutil.copy2(current, snap)
        provenance = (bc.share_dir(site) / "panel_provenance.json")
        if provenance.exists():
            shutil.copy2(provenance, snapshot_dir(site) / f"{args.label}_panel_provenance.json")
        n = len(pd.read_parquet(snap))
        print(f"  snapshot '{args.label}': {n:,} cells -> {snap}")
        return 0

    if not args.compare:
        ap.error("pass --snapshot or --compare")
    if not snap.exists():
        print(f"  no snapshot at {snap}; run --snapshot first"); return 1

    a = pd.read_parquet(snap).sort_values(["unit_name", "period"]).reset_index(drop=True)
    b = pd.read_parquet(current).sort_values(["unit_name", "period"]).reset_index(drop=True)
    print(f"=== tier comparison: site {site} ===")
    print(f"  baseline ({args.label}) : {len(a):,} cells, {len(a.columns)} columns")
    print(f"  current              : {len(b):,} cells, {len(b.columns)} columns")

    problems = []
    if list(a.columns) != list(b.columns):
        problems.append(f"column sets differ: only-baseline={set(a.columns) - set(b.columns)}, "
                        f"only-current={set(b.columns) - set(a.columns)}")
    if len(a) != len(b):
        problems.append(f"cell counts differ: {len(a):,} vs {len(b):,}")

    if not problems:
        for col in a.columns:
            x, y = a[col], b[col]
            # bool is "numeric" to pandas but does not support `-`; compare it as a value.
            numeric = (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
                       and not pd.api.types.is_bool_dtype(x) and not pd.api.types.is_bool_dtype(y))
            if numeric:
                same = ((x - y).abs() <= 1e-9) | (x.isna() & y.isna())
                detail = f"max |delta| {(x - y).abs().max()}"
            else:
                same = (x.astype("string") == y.astype("string")) | (x.isna() & y.isna())
                detail = "value mismatch"
            bad = int((~same).sum())
            if bad:
                problems.append(f"{col}: {bad} cells differ ({detail})")

    if problems:
        print("\n  FAIL — the cache is NOT equivalent to a cold build:")
        for p in problems:
            print(f"    {p}")
        return 1
    print(f"\n  PASS — all {len(a):,} cells x {len(a.columns)} columns identical.")
    print("  The stage cache holds nothing the cold build does not recompute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
