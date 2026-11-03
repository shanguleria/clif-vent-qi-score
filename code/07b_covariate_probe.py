"""Stage 7b — coverage of candidate case-mix covariates, before any enters the outcome model.

For the stage 8 model sample (complete cases on died, qclif, sofa, cci, age), tabulates each
candidate overall, by anonymized unit, and by intubation year, with crude mortality per level:

  location_at_intubation        adt.location_category spanning intubation
  procedural_within_24h         any procedural ADT stay in the 24 h before intubation
  admission_type_category       hospitalization, first hospitalization of the encounter block
  code_status_before_intubation code_status, last order at or before intubation
  hours_admission_to_intubation binned
  cci::<condition>              clifpy Charlson conditions, all coded diagnoses
  cci_poa::<condition>          the same, present-on-admission diagnoses only

A candidate whose table or column is absent is logged as absent, not dropped silently.

Writes aggregate output only, counts in [1, small_cell_threshold) suppressed:
  final_no_phi/07b_covariate_probe/covariate_coverage.csv, provenance.json
  final_no_phi/logs/07b_covariate_probe.log

Run:  .venv/bin/python code/07b_covariate_probe.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import attribution as attr                                 # noqa: E402
from common import cohort_source                                       # noqa: E402
from common import covariates as cov                                   # noqa: E402

from clif_dash import stage                                            # noqa: E402

warnings.filterwarnings("ignore", message=".*generic.*unit for NumPy timedelta.*")

STEP = "07b_covariate_probe"
MODEL_COMPLETE = ["died", "qclif", "sofa", "cci", "age_at_admission"]
PROCEDURAL_HOURS = cov.PROCEDURAL_HOURS
POA_COLUMNS = ("poa_present", "present_on_admission")


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.fh.write(msg + "\n")


def read_optional(cfg, table: str, log):
    try:
        return stage.read_whole_table(cfg, table)
    except FileNotFoundError:
        log(f"[absent] table {table} not found")
        return None


def has_column(df, col: str, table: str, log) -> bool:
    if df is None:
        return False
    if col not in df.columns:
        log(f"[absent] {table}.{col} not in table")
        return False
    return True


def unit_hospitals(adt: pd.DataFrame, unit_ids: dict, collapse: dict) -> pd.DataFrame:
    """Which hospital each ICU sits in, anonymized: hospitals numbered in sorted `hospital_id`
    order, units by their panel `unit_id`. An ICU name absent from the panel is dropped."""
    icu = adt[adt["location_category"].astype(str).str.lower().eq("icu")]
    names = icu["location_name"].astype(str).map(lambda n: collapse.get(n, n))
    hid = icu["hospital_id"].astype("string").fillna("(missing)")
    pairs = (pd.DataFrame({"unit_id": names.map(unit_ids).to_numpy(), "hospital_id": hid.to_numpy()})
             .dropna(subset=["unit_id"]).drop_duplicates())
    known = sorted(set(pairs["hospital_id"]) - {"(missing)"})
    labels = {h: f"Hospital {i}" for i, h in enumerate(known, 1)} | {"(missing)": "Hospital missing"}
    pairs["hospital"] = pairs["hospital_id"].map(labels)
    return (pairs[["hospital", "unit_id"]].sort_values(["hospital", "unit_id"], kind="stable")
            .reset_index(drop=True))


def cci_components(hd: pd.DataFrame, mapping: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Charlson condition flags per encounter block; a block with no coded diagnosis scores 0,
    as the cci covariate already does in stage 7."""
    from clifpy.utils.comorbidity import calculate_cci
    cols = ["hospitalization_id", "diagnosis_code", "diagnosis_code_format"]
    res = calculate_cci(hd[cols].astype(str))
    conds = [c for c in res.columns if c not in ("hospitalization_id", "cci_score")]
    res["hospitalization_id"] = res["hospitalization_id"].astype(str)
    per_block = res.merge(mapping, on="hospitalization_id").groupby("encounter_block")[conds].max()
    return (cohort[["encounter_block"]].join(per_block, on="encounter_block")[conds]
            .fillna(0).astype(int))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    share = bc.share_dir(site)
    log = Log(share / "logs" / f"{STEP}.log")
    threshold = bc.small_cell_threshold()
    log(f"=== {STEP}: site {site} ===")
    log(f"small-cell threshold {threshold} (from .clif-prepush.json)")

    src = cohort_source.recorded(bc.phi_dir(site) / "cohort" / "source.json", log=log)
    if src is None:
        raise SystemExit("No recorded cohort source; run stages 00-07 first.")
    cfg = src.cfg(site)
    tz = bc.load_profile(site)["timezone"]

    ext = pd.read_parquet(bc.phi_dir(site) / "analysis" / "analysis_dataset_extended.parquet")
    cohort = ext.dropna(subset=MODEL_COMPLETE).reset_index(drop=True)
    cohort["encounter_block"] = attr.norm_block(cohort["encounter_block"])
    log(f"[cohort] model sample {len(cohort):,} of {len(ext):,} patients; "
        f"{cohort['unit_id'].nunique()} units")

    mapping = src.read("mapping")[["hospitalization_id", "encounter_block"]].drop_duplicates()
    mapping["hospitalization_id"] = mapping["hospitalization_id"].astype(str)
    mapping["encounter_block"] = attr.norm_block(mapping["encounter_block"])
    blocks = set(cohort["encounter_block"])
    overlap = len(blocks & set(mapping["encounter_block"]))
    log(f"[keys] cohort blocks found in mapping: {overlap:,}/{len(blocks):,}")
    if overlap == 0:
        raise SystemExit("encounter_block join has zero overlap; check key normalization.")

    cohort["intubation_year"] = attr.as_ns(cohort["intubation_dttm"]).dt.tz_convert(tz).dt.year
    cohort["hours_admission_to_intubation"] = cov.hours_category(
        cohort["intubation_dttm"], cohort["admission_dttm"])
    candidates = ["hours_admission_to_intubation"]
    presence = {}

    adt = read_optional(cfg, "adt", log)
    if has_column(adt, "location_category", "adt", log):
        adt = cov.attach_block(adt, mapping, blocks)
        cohort["location_at_intubation"] = cov.location_at_intubation(cohort, adt)
        cohort[f"procedural_within_{PROCEDURAL_HOURS}h"] = cov.procedural_within(
            cohort, adt, PROCEDURAL_HOURS)
        candidates += ["location_at_intubation", f"procedural_within_{PROCEDURAL_HOURS}h"]
        if {"location_name", "hospital_id"} <= set(adt.columns):
            panel = pd.read_parquet(share / "unit_panel.parquet")[["unit_name", "unit_id"]]
            uh = unit_hospitals(adt, dict(zip(panel["unit_name"], panel["unit_id"])),
                                bc.load_profile(site).get("unit_collapse") or {})
            by_h = {h: sorted(g["unit_id"]) for h, g in uh.groupby("hospital")}
            log(f"\n[hospitals] {len(by_h)} hospital(s) hold {uh['unit_id'].nunique()} ICUs")
            for h, units in by_h.items():
                log(f"    {h:<20} {len(units)} ICU(s): {', '.join(units)}")
            split = sorted(uh.loc[uh["unit_id"].duplicated(keep=False), "unit_id"].unique())
            log(f"    ICUs recorded under more than one hospital: {split or 'none'}")
            presence["hospitals"] = by_h
        else:
            log("[absent] adt.location_name or adt.hospital_id; hospital structure not reported")

    hosp = read_optional(cfg, "hospitalization", log)
    if has_column(hosp, "admission_type_category", "hospitalization", log):
        cohort["admission_type_category"] = cov.admission_type(
            cohort, cov.attach_block(hosp, mapping, blocks))
        candidates.append("admission_type_category")

    cs = read_optional(cfg, "code_status", log)
    if has_column(cs, "code_status_category", "code_status", log):
        cs = cs[cs["patient_id"].astype(str).isin(set(cohort["patient_id"].astype(str)))]
        cohort["code_status_before_intubation"] = cov.code_status_before(cohort, cs)
        cohort["code_status_recorded_before"] = (
            cohort["code_status_before_intubation"] != cov.NONE_RECORDED)
        candidates.append("code_status_before_intubation")

    hd = read_optional(cfg, "hospital_diagnosis", log)
    if hd is not None:
        hd = cov.attach_block(hd, mapping, blocks)
        poa_col = next((c for c in POA_COLUMNS if c in hd.columns), None)
        presence["poa_column"] = poa_col
        comp = cci_components(hd, mapping, cohort)
        cohort = cohort.join(comp.add_prefix("cci::"))
        candidates += [f"cci::{c}" for c in comp.columns]
        if poa_col is None:
            log(f"[absent] hospital_diagnosis has none of {list(POA_COLUMNS)}")
        else:
            poa = pd.to_numeric(hd[poa_col], errors="coerce")
            dist = {str(k): int(v) for k, v in poa.value_counts(dropna=False).items()}
            log(f"[poa] {poa_col} values among cohort diagnosis rows: {dist}")
            presence["poa_distribution"] = dist
        if poa_col is not None and not poa.eq(1).any():
            log(f"[absent] no {poa_col} == 1 rows; POA-restricted conditions not computed")
        elif poa_col is not None:
            comp_poa = cci_components(hd[poa.eq(1).to_numpy()], mapping, cohort)
            cohort = cohort.join(comp_poa.add_prefix("cci_poa::"))
            candidates += [f"cci_poa::{c}" for c in comp_poa.columns]

    tabs = []
    for v in candidates:
        tabs.append(cov.tabulate(cohort, v))
        tabs.append(cov.tabulate(cohort, v, by="unit_id"))
    tabs.append(cov.tabulate(cohort, "intubation_year"))
    tabs.append(cov.tabulate(cohort, "intubation_year", by="unit_id"))
    for v in ("location_at_intubation", "admission_type_category", "code_status_recorded_before"):
        if v in cohort:
            tabs.append(cov.tabulate(cohort, v, by="intubation_year"))
    table = cov.suppress(pd.concat(tabs, ignore_index=True), threshold)

    out = share / STEP
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "covariate_coverage.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "provenance": bc.provenance(site, cohort_source=src.tier),
        "sample": "stage 8 model sample: complete cases on " + ", ".join(MODEL_COMPLETE),
        "n_patients": int(len(cohort)),
        "small_cell_threshold": threshold,
        "candidates": candidates,
        "presence": presence,
    }, indent=2, default=str))

    log("\n[overall] level counts and crude mortality (suppressed cells blank)")
    overall = table[table["by"] == "all"]
    for v, g in overall.groupby("variable", sort=False):
        if v.startswith("cci"):
            continue
        log(f"\n  {v}")
        for r in g.itertuples():
            n = "<supp>" if pd.isna(r.n) else f"{int(r.n):,}"
            pct = "" if pd.isna(r.pct) else f"{r.pct:5.1f}%"
            mort = "" if pd.isna(r.mortality) else f"mortality {r.mortality:5.1f}%"
            log(f"    {str(r.level):<32} {n:>8} {pct:>7}  {mort}")

    log("\n[cci] condition prevalence, all coded vs present-on-admission")
    cc = overall[overall["variable"].str.startswith("cci") & (overall["level"] == "1")]
    for r in cc.itertuples():
        pct = "<supp>" if pd.isna(r.pct) else f"{r.pct:5.1f}%"
        log(f"    {r.variable:<44} {pct:>7}")

    log(f"\n[done] {len(table):,} rows -> {out / 'covariate_coverage.csv'}")
    log(f"       suppressed rows: {int(table['suppressed'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
