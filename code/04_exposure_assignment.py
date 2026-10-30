"""Stage 4 — turn each panel into a per-patient exposure, with leave-one-out variants.

Two grains, mirroring stage 3:

  exposure_unit_*          PRIMARY: each patient assigned their ICU's score, pooled over all time
  exposure_unit_period_*   SECONDARY: the same against the unit x period panel

and two artifacts per grain, because a patient ventilated across several units has no single
unambiguous exposure and the modelling step should get to decide:

  *_long.parquet      one row per (patient, cell) they contributed to, with that cell's score and
                      the patient's own contribution to it
  *_primary.parquet   one row per patient: the cell where they accrued the most ventilated time

Leave-one-out subtracts ALL of a patient's own numerator and denominator from the cell before
re-computing the rate, so the exposure is the care the *rest* of the unit received. Without it the
patient's own care drives their own exposure value and the association is partly mechanical.

All outputs are patient-level -> intermediate_phi/, never shared.

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
from common import scoring as S                                        # noqa: E402

#: Metric whose denominator measures "how much ventilated time this patient spent in this cell".
#: Vent-days, not IMV hours, because they are the grain SAT and SBT are measured on.
PRIMARY_VOLUME_METRIC = "sat"


def build_exposure(contrib, panel, grain, *, scale, log=print):
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

    for score_id in score_cols:
        w = bc.weights(score_id)
        loo = S.leave_one_out(contrib, w, group=tuple(group), scale=scale)
        loo = loo.rename(columns={"loo_score": f"{score_id}_loo",
                                  "loo_n_components": f"{score_id}_loo_n_components"})
        cols = group + ["patient_id", f"{score_id}_loo", f"{score_id}_loo_n_components"]
        long = long.merge(loo[cols], on=group + ["patient_id"], how="left")
        long[f"{score_id}_loo"] = long[f"{score_id}_loo"].round(2)
        shift = (long[f"{score_id}_loo"] - long[score_id]).abs()
        log(f"    {score_id:<12} LOO computable {int(long[f'{score_id}_loo'].notna().sum()):>6,}"
            f"/{len(long):,}   median |score - LOO| {shift.median():.3f}"
            f"   p95 {shift.quantile(0.95):.3f}   max {shift.max():.3f}")

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
    scale = bc.load_composite().get("scale", 100)

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
        long, primary = build_exposure(contrib, panel, grain, scale=scale)

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
