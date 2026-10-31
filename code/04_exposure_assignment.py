"""Stage 4 — turn each panel into a per-patient exposure.

Two grains mirroring stage 3, and two artifacts per grain because a patient ventilated across
several units has no single unambiguous exposure:

  *_long.parquet     one row per (patient, cell), with the cell score and the patient's own share
  *_primary.parquet  one row per patient: the cell where they accrued the most ventilated time

All patient-level -> intermediate_phi/.

Run:  .venv/bin/python code/04_exposure_assignment.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402

#: Metric whose denominator measures "how much ventilated time this patient spent in this cell".
#: Vent-days, not IMV hours, because they are the grain SAT and SBT are measured on.
PRIMARY_VOLUME_METRIC = "sat"


def build_exposure(contrib, panel, grain, *, log=print):
    """Per-(patient, cell) exposure at one grain, plus the primary-cell assignment."""
    group = list(bc.GROUP_COLS[grain])
    score_cols = bc.score_ids()
    keep = (group + score_cols
            + [c for c in panel.columns if c.endswith("_den")]
            + [c for c in panel.columns if c.endswith("_low")]
            + [c for c in panel.columns if c.endswith("_n_components")]
            + ["unit_id"])
    panel_keys = panel[[c for c in dict.fromkeys(keep) if c in panel.columns]]

    per_cell = (contrib.groupby(group + ["patient_id", "metric"], observed=True)
                [["num", "den"]].sum().reset_index())
    own = per_cell.pivot_table(index=group + ["patient_id"], columns="metric",
                               values=["num", "den"], fill_value=0, observed=True)
    own.columns = [f"own_{metric}_{kind}" for kind, metric in own.columns]
    long = own.reset_index().merge(panel_keys, on=group, how="left")

    vol = f"own_{PRIMARY_VOLUME_METRIC}_den"
    if vol not in long.columns:
        raise KeyError(f"no {vol} column to rank cells by")
    ranked = long.sort_values(["patient_id", vol] + group,
                             ascending=[True, False] + [True] * len(group), kind="stable")
    primary = ranked.drop_duplicates("patient_id").reset_index(drop=True)
    return long, primary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)

    print(f"=== 04 exposure assignment: site {site} ===")
    phi = bc.phi_dir(site)
    contrib = pd.read_parquet(phi / "panel" / "contributions.parquet")
    out = phi / "exposure"
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"provenance": bc.provenance(site),
                "primary_volume_metric": PRIMARY_VOLUME_METRIC, "grains": {}}
    for grain in bc.GRAINS:
        panel = pd.read_parquet(bc.share_dir(site) / f"{grain}_panel.parquet")
        role = "PRIMARY" if grain == "unit" else "secondary"
        print(f"\n[{grain}] {role}")
        long, primary = build_exposure(contrib, panel, grain)

        n_pat = long["patient_id"].nunique()
        multi = long.groupby("patient_id").size()
        print(f"  {n_pat:,} patients, {len(long):,} (patient, cell) rows; "
              f"in >1 cell: {int((multi > 1).sum()):,} ({100 * (multi > 1).mean():.1f}%), "
              f"max {int(multi.max())}")
        for score_id in bc.score_ids():
            assigned = int(primary[score_id].notna().sum())
            low = int(primary[f"{score_id}_low"].fillna(False).sum())
            print(f"  primary {score_id:<12} assigned {assigned:,}/{n_pat:,}  "
                  f"median {primary[score_id].median():.1f}  "
                  f"in a cell below that score's floors: {low:,} ({100 * low / n_pat:.1f}%)")

        long.to_parquet(out / f"exposure_{grain}_long.parquet", index=False)
        primary.to_parquet(out / f"exposure_{grain}_primary.parquet", index=False)
        manifest["grains"][grain] = {
            "role": role.lower(), "patients": int(n_pat),
            "patient_cell_rows": int(len(long)),
            "patients_in_multiple_cells": int((multi > 1).sum()),
            "scores_assigned": {s: int(primary[s].notna().sum()) for s in bc.score_ids()},
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
