"""Stage 4 — turn the panel into a per-patient exposure, with leave-one-out variants.

Two artifacts, because a patient ventilated across several units or quarters has no single
unambiguous exposure and the modelling step should get to decide:

  exposure_long.parquet      one row per (patient, unit, period) they contributed to, with that
                             cell's score and the patient's own contribution to it
  exposure_primary.parquet   one row per patient: the cell where they accrued the most ventilated
                             time, plus plain and leave-one-out scores

Leave-one-out subtracts ALL of a patient's own numerator and denominator from the cell before
re-computing the rate, so the exposure is the care the *rest* of the unit received. Without it the
patient's own care drives their own exposure value and the exposure-outcome association is partly
mechanical.

Both files are patient-level -> intermediate_phi/, never shared.

Run:  .venv/bin/python code/04_exposure_assignment.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import scoring as S                                        # noqa: E402

GROUP = ["unit_name", "period"]
#: Metric whose denominator defines "how much ventilated time this patient spent in this cell".
#: Vent-days, not IMV hours, because they are the grain SAT and SBT are measured on.
PRIMARY_VOLUME_METRIC = "sat"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    comp = bc.load_composite()
    scale = comp.get("scale", 100)

    print(f"=== 04 exposure assignment: site {site} ===")
    phi = bc.phi_dir(site)
    contrib = pd.read_parquet(phi / "panel" / "contributions.parquet")
    panel = pd.read_parquet(bc.share_dir(site) / "unit_period_panel.parquet")

    score_cols = bc.score_ids()
    low_cols = [c for c in panel.columns if c.endswith("_low")] + ["low_denominator"]
    panel_keys = panel[GROUP + score_cols + [c for c in panel.columns if c.endswith("_den")]
                       + low_cols]

    # ---- long: every (patient, cell) the patient actually contributed to ----
    per_patient_cell = (contrib.groupby(GROUP + ["patient_id", "metric"], observed=True)
                        [["num", "den"]].sum().reset_index())
    own = per_patient_cell.pivot_table(index=GROUP + ["patient_id"], columns="metric",
                                      values=["num", "den"], fill_value=0, observed=True)
    own.columns = [f"own_{metric}_{kind}" for kind, metric in own.columns]
    long = own.reset_index().merge(panel_keys, on=GROUP, how="left")

    # ---- leave-one-out, per score ------------------------------------------
    for score_id in score_cols:
        w = bc.weights(score_id)
        loo = S.leave_one_out(contrib, w, group=tuple(GROUP), scale=scale)
        loo = loo.rename(columns={"loo_score": f"{score_id}_loo",
                                  "loo_n_components": f"{score_id}_loo_n_components"})
        keep = GROUP + ["patient_id", f"{score_id}_loo", f"{score_id}_loo_n_components"]
        long = long.merge(loo[keep], on=GROUP + ["patient_id"], how="left")
        long[f"{score_id}_loo"] = long[f"{score_id}_loo"].round(2)
        shift = (long[f"{score_id}_loo"] - long[score_id]).abs()
        print(f"  {score_id:<12} LOO computable {int(long[f'{score_id}_loo'].notna().sum()):>6,}"
              f"/{len(long):,}   median |score - LOO| = {shift.median():.3f}"
              f"   p95 = {shift.quantile(0.95):.3f}   max = {shift.max():.3f}")

    # ---- primary cell: most ventilated time ---------------------------------
    vol = f"own_{PRIMARY_VOLUME_METRIC}_den"
    if vol not in long.columns:
        raise KeyError(f"no {vol} column to rank cells by")
    ranked = long.sort_values(["patient_id", vol] + GROUP,
                              ascending=[True, False, True, True], kind="stable")
    primary = ranked.drop_duplicates("patient_id").reset_index(drop=True)

    out = phi / "exposure"
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out / "exposure_long.parquet", index=False)
    primary.to_parquet(out / "exposure_primary.parquet", index=False)

    n_pat = long["patient_id"].nunique()
    multi = long.groupby("patient_id").size()
    print(f"\n[exposure] {n_pat:,} patients, {len(long):,} (patient, cell) rows")
    print(f"  patients in >1 cell      : {int((multi > 1).sum()):,} "
          f"({100 * (multi > 1).mean():.1f}%)  max cells for one patient: {int(multi.max())}")
    for score_id in score_cols:
        low = int(primary[f"{score_id}_low"].fillna(False).sum())
        print(f"  primary {score_id:<12} assigned to {int(primary[score_id].notna().sum()):,}"
              f"/{n_pat:,} patients   median {primary[score_id].median():.1f}   "
              f"in a cell below THAT score's floors: {low:,} ({100 * low / n_pat:.1f}%)")

    (out / "manifest.json").write_text(json.dumps({
        "provenance": bc.provenance(site),
        "primary_volume_metric": PRIMARY_VOLUME_METRIC,
        "patients": int(n_pat),
        "patient_cell_rows": int(len(long)),
        "patients_in_multiple_cells": int((multi > 1).sum()),
    }, indent=2, default=str))
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
