"""Stage 3 — roll the metric records up into composite scores, at two grains.

  unit_panel.csv          PRIMARY exposure: one score per ICU, pooled over all available time
  unit_period_panel.csv   SECONDARY: the same scores per ICU x calendar period, for sensitivity

Pooling over all available time is what makes the primary exposure stable, and it fixes two
artifacts the per-period panel cannot: a unit that came online mid-window no longer produces a
score off a handful of patient-days, and the proning denominator becomes large enough for the
four-metric score to be computable at all.

Builds the canonical contribution table (one row per metric x unit-period x patient-episode), then
pools it at each grain. Pooling is sum(num)/sum(den) over the whole cell, so LTVV is hour-weighted
and SAT/SBT are vent-day-weighted — matching the upstream's aggregation rather than averaging
per-patient percentages. Pooling the WHOLE window is not the mean of the quarterly rates: it
re-divides the summed counts, so a thin quarter cannot pull the unit score around.

Writes:
  intermediate_phi/panel/contributions.parquet   row-level, carries patient_id — NEVER shared
  final_no_phi/unit_panel.{csv,parquet}          PRIMARY, one row per ICU
  final_no_phi/unit_period_panel.{csv,parquet}   SECONDARY, one row per ICU x period
  final_no_phi/*_anonymized.csv                  both, with location_name stripped
  final_no_phi/panel_provenance.json

Records that fell in no named ICU stay (IMV on the ward, in the OR, or in the ED) are reported and
then excluded: they are real care, but not care any ICU can be scored for.

Run:  .venv/bin/python code/03_build_panel.py [--site uchicago]
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
from common import scoring as S                                        # noqa: E402

from clif_dash.upstream import metrics as up_metrics                   # noqa: E402

CONTRIB_GROUP = ("unit_name", "period")

#: Row-level identifiers that must never appear in a shareable artifact. Checked by execution on
#: every run, because "the panel is aggregate by construction" is an assumption a future column
#: could quietly break.
FORBIDDEN = ("hospitalization_id", "patient_id", "encounter_block", "episode_id")


def phi_gate(share_dir) -> None:
    """Abort if anything under final_no_phi/ carries a row-level identifier."""
    import pandas as _pd
    leaks = []
    for f in sorted(share_dir.rglob("*")):
        if f.is_dir():
            continue
        if f.suffix in (".csv", ".json"):
            hits = [t for t in FORBIDDEN if t in f.read_text()]
        elif f.suffix == ".parquet":
            hits = [t for t in FORBIDDEN if t in list(_pd.read_parquet(f).columns)]
        else:
            hits = []
        if hits:
            leaks.append(f"{f.name}: {', '.join(hits)}")
    if leaks:
        raise SystemExit("[phi-gate] ABORT: row-level identifiers in shareable output:\n  "
                         + "\n  ".join(leaks))
    print(f"  PHI gate: {len(list(share_dir.rglob('*.csv')) + list(share_dir.rglob('*.parquet')) + list(share_dir.rglob('*.json')))} "
          f"shareable files, no row-level identifiers")


def build_contributions(site: str, ltvv_threshold: float) -> tuple[pd.DataFrame, dict]:
    """One row per (metric, unit-period, encounter_block) with its numerator and denominator."""
    phi = bc.phi_dir(site)
    rec = phi / "records"
    encounters = pd.read_parquet(phi / "cohort" / "encounters.parquet",
                                 columns=["encounter_block", "patient_id"])
    encounters["encounter_block"] = attr.norm_block(encounters["encounter_block"])
    pid = encounters.drop_duplicates("encounter_block").set_index("encounter_block")["patient_id"]

    parts, dropped = [], {}

    def add(metric: str, frame: pd.DataFrame, num: pd.Series, den: pd.Series):
        if not len(frame):
            return
        f = frame[["unit_name", "period", "encounter_block"]].copy()
        f["encounter_block"] = attr.norm_block(f["encounter_block"])
        f["num"], f["den"] = num.to_numpy(), den.to_numpy()
        n_before = len(f)
        f = f[f["unit_name"] != attr.UNATTRIBUTED]
        dropped[metric] = {"records_unattributed": int(n_before - len(f)),
                           "denominator_unattributed": int(den.to_numpy().sum() - f["den"].sum())}
        g = (f.groupby(["unit_name", "period", "encounter_block"], observed=True, dropna=False)
             [["num", "den"]].sum().reset_index())
        g["patient_id"] = g["encounter_block"].map(pid)
        g["metric"] = metric
        parts.append(g[S.CONTRIB_COLUMNS])

    ltvv = pd.read_parquet(rec / "ltvv_hours.parquet")
    key = up_metrics.ltvv_key(ltvv_threshold)
    add("ltvv", ltvv, ltvv[key], pd.Series(1, index=ltvv.index))

    days = pd.read_parquet(rec / "vent_days.parquet")
    add("sat", days, days["sat_num"], days["sat_den"])
    add("sbt", days, days["sbt_num"], days["sbt_den"])

    prone_path = rec / "prone_patients.parquet"
    if prone_path.exists():
        prone = pd.read_parquet(prone_path)
        add("prone_12h", prone, prone["p12"], pd.Series(1, index=prone.index))

    contrib = pd.concat(parts, ignore_index=True)
    missing_pid = int(contrib["patient_id"].isna().sum())
    if missing_pid:
        raise AssertionError(f"{missing_pid} contribution rows have no patient_id; stage 4 could "
                             f"not assign those patients to a primary unit")
    return contrib, dropped


def supplementary_ltvv(site: str, thresholds, primary, group) -> pd.DataFrame:
    """LTVV rates at the other selectable thresholds, for sensitivity. Not part of any score."""
    ltvv = pd.read_parquet(bc.phi_dir(site) / "records" / "ltvv_hours.parquet")
    ltvv = ltvv[ltvv["unit_name"] != attr.UNATTRIBUTED]
    keys = {t: up_metrics.ltvv_key(t) for t in thresholds if t != primary}
    if not keys:
        return pd.DataFrame(columns=list(group))
    agg = (ltvv.groupby(list(group), observed=True)
           .agg(_den=("vt_pbw", "size"), **{f"_n{t}": (k, "sum") for t, k in keys.items()})
           .reset_index())
    for t in keys:
        agg[f"ltvv_rate_at_{str(t).replace('.', '_')}"] = S.pooled_rate(agg[f"_n{t}"], agg["_den"])
    return agg.drop(columns=["_den"] + [f"_n{t}" for t in keys])


def build_panel(contrib, grain, site, *, scale, selectable, primary, unit_id, log=print):
    """Pool the contribution table at one grain and attach composite scores and floor flags."""
    group = list(bc.GROUP_COLS[grain])
    panel = S.cell_totals(contrib, tuple(group))

    # A metric with no records at all (e.g. a site with no `position` table cannot observe proning)
    # must degrade to an empty column, not a KeyError three functions later. Scores that include it
    # then correctly evaluate to NaN, and the gap is reported rather than inferred from a traceback.
    absent = [m for m in bc.metric_ids() if f"{m}_den" not in panel.columns]
    for m in absent:
        panel[f"{m}_num"] = 0
        panel[f"{m}_den"] = 0
        panel[f"{m}_rate"] = float("nan")
    if absent:
        log(f"  NOTE: no records for {', '.join(absent)} at this site — "
            f"any score including them will be null. Check the source tables exist.")

    # Composite scores. Iterate the config's own score keys so adding one is never a no-op.
    for score_id in bc.score_ids():
        w = bc.weights(score_id)
        sc, n = S.composite(panel, w, scale=scale)
        panel[score_id] = sc.round(2)
        panel[f"{score_id}_n_components"] = n

    floors = bc.min_denominator(grain, site)
    panel = S.flag_low_denominator(panel, floors,
                                   scores={s: bc.weights(s) for s in bc.score_ids()})

    panel["unit_id"] = panel["unit_name"].map(unit_id)
    supp = supplementary_ltvv(site, selectable, primary, group)
    if len(supp):
        panel = panel.merge(supp, on=group, how="left")
    for c in [c for c in panel.columns if c.endswith("_rate") or c.startswith("ltvv_rate_at_")]:
        panel[c] = panel[c].round(4)

    front = [c for c in ("unit_id", "unit_name", "period") if c in panel.columns]
    panel = panel[front + [c for c in panel.columns if c not in front]]
    return panel.sort_values(group).reset_index(drop=True), floors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    comp = bc.load_composite()
    scale = comp.get("scale", 100)

    print(f"=== 03 build panel: site {site} ===")
    primary = bc.decision_value("ltvv", "threshold_ml_kg", site)
    selectable = bc.decision_value("ltvv", "thresholds_selectable", site)
    print(f"  LTVV threshold (from definitions/): <= {primary} mL/kg PBW "
          f"(selectable {selectable})")

    contrib, dropped = build_contributions(site, primary)
    print(f"\n[contributions] {len(contrib):,} rows over {contrib['metric'].nunique()} metrics")
    for metric in bc.metric_ids():
        m = contrib[contrib["metric"] == metric]
        d = dropped.get(metric, {})
        print(f"  {metric:<10} den={int(m['den'].sum()):>8,}  num={int(m['num'].sum()):>8,}  "
              f"unattributed_den_dropped={d.get('denominator_unattributed', 0):,}")

    # Anonymized unit labels, numbered by total LTVV volume so the mapping is identical at both
    # grains and deterministic across reruns. The crosswalk stays site-local.
    volume = contrib[contrib["metric"] == "ltvv"].groupby("unit_name", observed=True)["den"].sum()
    order = sorted(volume.index, key=lambda u: (-volume[u], u))
    unit_id = {u: f"Unit {i}" for i, u in enumerate(order, 1)}

    phi_out = bc.phi_dir(site) / "panel"
    phi_out.mkdir(parents=True, exist_ok=True)
    contrib.to_parquet(phi_out / "contributions.parquet", index=False)
    (pd.DataFrame({"unit_id": list(unit_id.values()), "unit_name": list(unit_id.keys())})
     .to_csv(phi_out / "unit_crosswalk.csv", index=False))

    share = bc.share_dir(site)
    written, prov_grains = [], {}
    for grain in bc.GRAINS:
        panel, floors = build_panel(contrib, grain, site, scale=scale, selectable=selectable,
                                    primary=primary, unit_id=unit_id)
        role = "PRIMARY" if grain == "unit" else "secondary"
        shape = (f"{panel['unit_name'].nunique()} units"
                 if grain == "unit" else
                 f"{panel['unit_name'].nunique()} units x {panel['period'].nunique()} periods")
        print(f"\n[{grain}] {role} — {len(panel):,} cells ({shape})")
        # Report effective floors BY EXECUTION: how many cells each one actually flagged.
        for metric, floor in floors.items():
            n = int(panel[f"{metric}_low"].sum())
            print(f"  floor {metric:<10} den < {floor:<5} flagged {n:>4} / {len(panel)} cells")
        for score_id in bc.score_ids():
            ok = int(panel[score_id].notna().sum())
            low = int(panel[f"{score_id}_low"].sum())
            med = panel[score_id].median()
            print(f"  {score_id:<12} computable in {ok}/{len(panel)}  "
                  f"median {med:.1f}  below its own floors: {low}")

        stem = f"{grain}_panel"
        panel.to_parquet(share / f"{stem}.parquet", index=False)
        panel.to_csv(share / f"{stem}.csv", index=False)
        panel.drop(columns=["unit_name"]).to_csv(share / f"{stem}_anonymized.csv", index=False)
        written.append(stem)
        prov_grains[grain] = {
            "role": role.lower(), "group": list(bc.GROUP_COLS[grain]), "cells": int(len(panel)),
            "min_denominator": floors,
            "scores_computable": {s: int(panel[s].notna().sum()) for s in bc.score_ids()},
        }

    (share / "panel_provenance.json").write_text(json.dumps({
        "provenance": bc.provenance(site),
        "composite": {s: comp["scores"][s] for s in bc.score_ids()},
        "ltvv_threshold_ml_kg": primary,
        "grains": prov_grains,
        "unattributed": dropped,
        "units": len(unit_id),
    }, indent=2, default=str))

    phi_gate(share)
    print(f"\n  wrote {share}/")
    for stem in written:
        print(f"    {stem}.csv / .parquet / _anonymized.csv")
    print(f"  unit crosswalk kept site-local in {phi_out}/unit_crosswalk.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
