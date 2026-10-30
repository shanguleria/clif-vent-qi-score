"""Stage 7 — assemble the patient-level analysis dataset for the outcome model.

One row per patient, using their FIRST qualifying IMV hospitalization. Eight columns:

  patient_id      key
  died            in-hospital death (see below)
  death_dttm      recorded time of death, ALL-CAUSE — see the caveat below
  unit_name       the ICU the patient spent the most time in during that hospitalization
  qclif           that unit's composite ventilator quality score (score_core, unit grain)
  age             age at admission
  cci             Charlson comorbidity index total
  sofa            SOFA over [intubation - 24 h, intubation + 4 h]

`died` = `discharge_category` in {Expired, Hospice} OR `death_dttm` <= discharge + 1 day.
The second clause adds only 8 encounters: CLIF's `death_dttm` is an all-cause date that at this
site falls a median of 206 days AFTER discharge for patients discharged alive, so it must not be
used unqualified as an in-hospital outcome. `death_dttm` is still emitted verbatim as its own
column; it is simply not what defines the outcome.

The ICU is the one holding the most ADT time in the hospitalization, not the most ventilated time,
so a patient intubated in the ED or moved between units is attributed to where they were actually
cared for. SOFA spans 24 h before to 4 h after intubation, so the intubation itself counts toward
the severity it marks.

Writes (patient-level -> intermediate_phi/, never shared):
  analysis/analysis_dataset.parquet / .csv     the eight columns
  analysis/analysis_dataset_extended.parquet   + diagnostics (ids, SOFA components, unit time)
  analysis/manifest.json

Run:  .venv/bin/python code/07_analysis_dataset.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import attribution as attr                                 # noqa: E402
from common import cohort_source                                       # noqa: E402

from clif_dash import stage                                            # noqa: E402
from clif_dash.upstream import clif_io                                 # noqa: E402

warnings.filterwarnings("ignore", message=".*generic.*unit for NumPy timedelta.*")

#: discharge_category values that constitute an in-hospital death. Hospice is included as a death
#: by convention (discharge to hospice is a terminal disposition), which is a modelling choice,
#: not a fact about the data.
DEATH_CATEGORIES = {"expired", "hospice"}

#: How long after discharge a recorded death_dttm may still count as the same hospitalization.
DEATH_DTTM_GRACE = pd.Timedelta(days=1)

#: SOFA window, relative to the first intubation of the encounter.
SOFA_BEFORE = pd.Timedelta(hours=24)
SOFA_AFTER = pd.Timedelta(hours=4)

#: The score that becomes `qclif`.
QCLIF_SCORE = "score_core"


def clifpy_config(site) -> Path:
    """clifpy wants its own config shape; derive it from the site profile, not a second copy."""
    prof = bc.load_profile(site)
    p = bc.upstream_build_dir(site) / "clifpy_config.json"
    p.write_text(json.dumps({"data_directory": prof["data_path"],
                             "filetype": prof.get("file_format", "parquet"),
                             "timezone": prof["timezone"]}, indent=2))
    return p


def first_encounter_per_patient(episodes, encounters) -> pd.DataFrame:
    """Each patient's first qualifying IMV hospitalization, with its first intubation time."""
    ep = episodes.copy()
    ep["start"] = pd.to_datetime(ep["start"], utc=True)
    ep["encounter_block"] = attr.norm_block(ep["encounter_block"])
    first_ep = (ep.sort_values(["encounter_block", "start"], kind="stable")
                .groupby("encounter_block", as_index=False)
                .agg(intubation_dttm=("start", "min")))
    enc = encounters.copy()
    enc["encounter_block"] = attr.norm_block(enc["encounter_block"])
    m = first_ep.merge(enc, on="encounter_block", how="left")
    # Earliest intubation wins; encounter_block breaks any tie deterministically.
    return (m.sort_values(["patient_id", "intubation_dttm", "encounter_block"], kind="stable")
            .drop_duplicates("patient_id").reset_index(drop=True))


def dominant_icu(cohort, icu) -> pd.DataFrame:
    """The ICU holding the most ADT time in each encounter.

    Open stays are clipped to the encounter's discharge: the FAR_FUTURE sentinel that keeps an
    open stay joinable would otherwise make it win every comparison outright.
    """
    disch = cohort.set_index("encounter_block")["discharge_dttm"]
    x = icu.copy()
    x["encounter_block"] = attr.norm_block(x["encounter_block"])
    x = x[x["encounter_block"].isin(cohort["encounter_block"])]
    cap = attr.as_ns(x["encounter_block"].map(disch))
    x["out_dttm"] = attr.as_ns(x["out_dttm"]).where(
        attr.as_ns(x["out_dttm"]) < attr.FAR_FUTURE, cap)
    x["out_dttm"] = x["out_dttm"].fillna(cap)
    x["hours"] = (x["out_dttm"] - attr.as_ns(x["in_dttm"])).dt.total_seconds() / 3600
    x = x[x["hours"] > 0]
    per = (x.groupby(["encounter_block", "unit_name"], observed=True)["hours"]
           .sum().reset_index())
    n_units = per.groupby("encounter_block")["unit_name"].nunique().rename("n_icus")
    per = per.sort_values(["encounter_block", "hours", "unit_name"],
                          ascending=[True, False, True], kind="stable")
    top = per.drop_duplicates("encounter_block").rename(columns={"hours": "icu_hours_in_unit"})
    total = per.groupby("encounter_block")["hours"].sum().rename("icu_hours_total")
    return top.merge(n_units, on="encounter_block").merge(total, on="encounter_block")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    print(f"=== 07 analysis dataset: site {site} ===")

    src = cohort_source.recorded(bc.phi_dir(site) / "cohort" / "source.json")
    if src is None:
        src = cohort_source.resolve(site)
    cfg = src.cfg(site)
    phi = bc.phi_dir(site)

    episodes = pd.read_parquet(phi / "cohort" / "episodes.parquet")
    encounters = pd.read_parquet(phi / "cohort" / "encounters.parquet")
    icu = pd.read_parquet(phi / "cohort" / "icu_intervals.parquet")
    mapping = src.read("mapping")[["hospitalization_id", "encounter_block"]].drop_duplicates()

    # ---- 1. one row per patient --------------------------------------------
    cohort = first_encounter_per_patient(episodes, encounters)
    n_enc = episodes["encounter_block"].nunique()
    print(f"\n[cohort] {len(cohort):,} patients (first IMV encounter of {n_enc:,} total encounters)")

    # ---- 2. outcome ---------------------------------------------------------
    pat = stage.read_whole_table(cfg, "patient")[["patient_id", "death_dttm"]]
    cohort = cohort.merge(pat, on="patient_id", how="left")
    cohort["death_dttm"] = pd.to_datetime(cohort["death_dttm"], utc=True)
    disch = pd.to_datetime(cohort["discharge_dttm"], utc=True)
    cat_dead = cohort["discharge_category"].astype(str).str.lower().isin(DEATH_CATEGORIES)
    dttm_dead = cohort["death_dttm"].notna() & (cohort["death_dttm"] <= disch + DEATH_DTTM_GRACE)
    cohort["died"] = (cat_dead | dttm_dead).astype(int)
    print(f"[outcome] died {int(cohort['died'].sum()):,}/{len(cohort):,} "
          f"({100 * cohort['died'].mean():.1f}%)   "
          f"by category {int(cat_dead.sum()):,}, +{int((dttm_dead & ~cat_dead).sum())} by death_dttm")
    post = int((cohort["death_dttm"].notna() & ~dttm_dead).sum())
    print(f"          {post:,} patients have a death_dttm AFTER discharge — recorded in the "
          f"death_dttm column, NOT counted as died")

    # ---- 3. dominant ICU and its score --------------------------------------
    top = dominant_icu(cohort, icu)
    cohort = cohort.merge(top, on="encounter_block", how="left")
    panel = pd.read_parquet(bc.share_dir(site) / "unit_panel.parquet")
    cohort = cohort.merge(panel[["unit_name", QCLIF_SCORE, "unit_id"]]
                          .rename(columns={QCLIF_SCORE: "qclif"}), on="unit_name", how="left")
    no_unit = int(cohort["unit_name"].isna().sum())
    print(f"[unit] assigned by most ICU time; {no_unit:,} patients with no named ICU stay")
    print(f"       patients in >1 ICU: {int((cohort['n_icus'] > 1).sum()):,} "
          f"({100 * (cohort['n_icus'] > 1).mean():.1f}%)")

    # ---- 4. age and Charlson -------------------------------------------------
    cohort = cohort.rename(columns={"age": "age_at_admission"})
    hd = stage.read_whole_table(cfg, "hospital_diagnosis")
    cci = clif_io.charlson(hd)
    cci["hospitalization_id"] = cci["hospitalization_id"].astype(str)
    mp = mapping.copy()
    mp["hospitalization_id"] = mp["hospitalization_id"].astype(str)
    mp["encounter_block"] = attr.norm_block(mp["encounter_block"])
    cci_block = (cci.merge(mp, on="hospitalization_id")
                 .groupby("encounter_block")["cci_score"].max().rename("cci"))
    cohort = cohort.merge(cci_block, on="encounter_block", how="left")
    cohort["cci"] = cohort["cci"].fillna(0)
    print(f"[covariates] age median {cohort['age_at_admission'].median():.0f}; "
          f"cci median {cohort['cci'].median():.0f} "
          f"({int((cohort['cci'] == 0).sum()):,} with no coded comorbidity)")

    # ---- 5. SOFA over the intubation window ---------------------------------
    from clifpy.utils.sofa2 import calculate_sofa2
    win = cohort[["encounter_block", "intubation_dttm"]].copy()
    win["start_dttm"] = win["intubation_dttm"] - SOFA_BEFORE
    win["end_dttm"] = win["intubation_dttm"] + SOFA_AFTER
    print(f"[sofa] computing over [-{int(SOFA_BEFORE.total_seconds()//3600)} h, "
          f"+{int(SOFA_AFTER.total_seconds()//3600)} h] around intubation for {len(win):,} patients")
    sofa = calculate_sofa2(win[["encounter_block", "start_dttm", "end_dttm"]],
                           clif_config_path=str(clifpy_config(site)),
                           id_name="encounter_block", id_mapping=mapping)
    comp = [c for c in sofa.columns if c.startswith("sofa2_") and c != "sofa2_total"]
    cohort = cohort.merge(
        sofa[["encounter_block", "sofa2_total"] + comp].rename(columns={"sofa2_total": "sofa"}),
        on="encounter_block", how="left")
    print(f"       sofa median {cohort['sofa'].median():.0f} "
          f"(IQR {cohort['sofa'].quantile(.25):.0f}-{cohort['sofa'].quantile(.75):.0f}, "
          f"range {cohort['sofa'].min():.0f}-{cohort['sofa'].max():.0f}); "
          f"missing {int(cohort['sofa'].isna().sum()):,}")
    print("       component completeness (a missing component scores 0, deflating the total):")
    for c in comp:
        print(f"         {c:<14} present {int(cohort[c].notna().sum()):>6,}/{len(cohort):,} "
              f"({100 * cohort[c].notna().mean():>5.1f}%)")

    # ---- 6. write -----------------------------------------------------------
    COLUMNS = ["patient_id", "died", "death_dttm", "unit_name", "qclif",
               "age_at_admission", "cci", "sofa"]
    out = phi / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = cohort[COLUMNS].copy()
    df.to_parquet(out / "analysis_dataset.parquet", index=False)
    df.to_csv(out / "analysis_dataset.csv", index=False)

    extended = ["unit_id", "encounter_block", "hospitalization_id", "intubation_dttm",
                "admission_dttm", "discharge_dttm", "discharge_category",
                "icu_hours_in_unit", "icu_hours_total", "n_icus", "sex_category"] + comp
    cohort[COLUMNS + [c for c in extended if c in cohort.columns]].to_parquet(
        out / "analysis_dataset_extended.parquet", index=False)

    complete = df.dropna(subset=["qclif", "sofa", "cci", "age_at_admission"])
    (out / "manifest.json").write_text(json.dumps({
        "provenance": bc.provenance(site),
        "rows": int(len(df)), "columns": COLUMNS,
        "grain": "one row per patient, first qualifying IMV hospitalization",
        "outcome": {"definition": "discharge_category in {Expired, Hospice} "
                                  "OR death_dttm <= discharge + 1 day",
                    "deaths": int(df["died"].sum()), "rate": round(float(df["died"].mean()), 4)},
        "qclif": {"source": f"unit_panel.{QCLIF_SCORE}", "missing": int(df["qclif"].isna().sum())},
        "sofa": {"window_hours": [-24, 4], "implementation": "clifpy calculate_sofa2 (6 organ)",
                 "missing": int(df["sofa"].isna().sum())},
        "complete_cases": int(len(complete)),
    }, indent=2, default=str))

    print(f"\n[dataset] {len(df):,} rows x {len(COLUMNS)} columns")
    print(f"  complete cases (qclif, sofa, cci, age all present): {len(complete):,} "
          f"({100 * len(complete) / len(df):.1f}%)")
    print(f"  wrote {out}/analysis_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
