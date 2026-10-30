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

from common import sofa_crrt                                           # noqa: E402

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

#: SOFA subscore -> the raw value(s) that feed it. Used to report WHY a component is absent: a
#: missing component is summed as 0, i.e. as normal, so a charting gap is indistinguishable from
#: health in the total unless the inputs are inspected directly.
SOFA_INPUTS = {
    "sofa_cns":    ["gcs_total"],
    "sofa_resp":   ["pf", "sf"],
    "sofa_cv_97":  ["map", "nee"],
    "sofa_liver":  ["bilirubin_total"],
    "sofa_renal":  ["creatinine"],
    "sofa_coag":   ["platelet_count"],
}


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


def missingness_report(df, sofa, comp, out, log=print):
    """Per-covariate and per-SOFA-component missingness, plus the inputs behind each component.

    Written as a table rather than printed only, because the decision this informs (impute,
    restrict to complete components, or accept) needs the numbers to hand.
    """
    rows = []

    def add(group, var, present, n):
        rows.append({"group": group, "variable": var, "n": n, "n_present": int(present),
                     "n_missing": int(n - present), "pct_missing": round(100 * (n - present) / n, 2)})

    n = len(df)
    for c in ("died", "unit_name", "qclif", "age_at_admission", "cci", "sofa"):
        add("covariate", c, df[c].notna().sum(), n)
    # death_dttm is legitimately absent for survivors; report it against decedents only.
    add("covariate", "death_dttm (among died==1)",
        df.loc[df["died"] == 1, "death_dttm"].notna().sum(), int((df["died"] == 1).sum()))

    for c in comp:
        add("sofa_component", c, sofa[c].notna().sum(), n)
    for c, inputs in SOFA_INPUTS.items():
        for i in inputs:
            if i in sofa.columns:
                add(f"sofa_input::{c}", i, sofa[i].notna().sum(), n)

    rep = pd.DataFrame(rows)
    rep.to_csv(out / "missingness_report.csv", index=False)

    log("\n[missingness] covariates")
    for r in rep[rep.group == "covariate"].itertuples():
        log(f"    {r.variable:<28} missing {r.n_missing:>6,}/{r.n:<7,} ({r.pct_missing:>5.2f}%)")

    log("\n[missingness] SOFA components — a MISSING component is scored 0 (normal) in the total")
    for r in rep[rep.group == "sofa_component"].itertuples():
        scored_zero = int((sofa[r.variable] == 0).sum())
        log(f"    {r.variable:<16} absent {r.n_missing:>6,} ({r.pct_missing:>5.2f}%)   "
            f"present-and-0 {scored_zero:>6,}   -> {r.n_missing + scored_zero:>6,} "
            f"({100 * (r.n_missing + scored_zero) / r.n:>5.1f}%) contribute 0")

    log("\n[missingness] inputs behind each component")
    for c in comp:
        ins = [i for i in SOFA_INPUTS.get(c, []) if i in sofa.columns]
        if not ins:
            continue
        parts = ", ".join(f"{i} {100 * sofa[i].notna().mean():.0f}%" for i in ins)
        log(f"    {c:<16} {parts}")

    # Does charting completeness differ BY UNIT? If it does, the unit-level exposure is confounded
    # by documentation quality: a unit that charts less looks less sick and gets under-adjusted.
    both = df[["unit_name"]].join(sofa[comp])
    by_unit = both.groupby("unit_name")[comp].apply(lambda g: 100 * g.isna().mean()).round(1)
    by_unit["n"] = both.groupby("unit_name").size()
    by_unit.to_csv(out / "missingness_by_unit.csv")
    log("\n[missingness] % of patients with the component ABSENT, by unit")
    log(by_unit.to_string())
    spread = {c: float(by_unit[c].max() - by_unit[c].min()) for c in comp}
    worst = max(spread, key=spread.get)
    log(f"\n    widest between-unit spread: {worst} "
        f"({by_unit[worst].min():.1f}% - {by_unit[worst].max():.1f}%, "
        f"{spread[worst]:.1f} points)")

    # How much does treating absent-as-zero deflate the total?
    allpresent = sofa[comp].notna().all(axis=1)
    log(f"\n[impact] all six components present: {int(allpresent.sum()):,}/{n:,} "
        f"({100 * allpresent.mean():.1f}%)")
    log(f"    mean SOFA, complete cases   : {df.loc[allpresent.to_numpy(), 'sofa'].mean():.2f}")
    log(f"    mean SOFA, >=1 absent       : {df.loc[~allpresent.to_numpy(), 'sofa'].mean():.2f}")
    log(f"    (the gap mixes real severity with charting; it is NOT a correction factor)")
    return rep


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
    win = cohort[["encounter_block", "intubation_dttm"]].copy()
    win["start_dttm"] = win["intubation_dttm"] - SOFA_BEFORE
    win["end_dttm"] = win["intubation_dttm"] + SOFA_AFTER
    print(f"[sofa] classic SOFA over [-{int(SOFA_BEFORE.total_seconds()//3600)} h, "
          f"+{int(SOFA_AFTER.total_seconds()//3600)} h] around intubation")
    mp2 = mapping.copy()
    mp2["hospitalization_id"] = mp2["hospitalization_id"].astype(str)
    mp2["encounter_block"] = attr.norm_block(mp2["encounter_block"])
    sofa = sofa_crrt.compute(src, cfg, mp2, win, bc.load_profile(site)["data_path"],
                             bc.load_profile(site)["timezone"])
    comp = sofa_crrt.SUBSCORES
    keep = ["encounter_block", "sofa_total", "sofa_total_raw", "sofa_n_components",
            "sofa_resp_raw", "oxy_source"] + comp + sofa_crrt.COMPONENT_VALUES
    keep = [c for c in dict.fromkeys(keep) if c in sofa.columns]
    cohort = cohort.merge(sofa[keep].rename(columns={"sofa_total": "sofa"}),
                          on="encounter_block", how="left")
    inputs = [c for v in SOFA_INPUTS.values() for c in v if c in cohort.columns]
    print(f"       sofa median {cohort['sofa'].median():.0f} "
          f"(IQR {cohort['sofa'].quantile(.25):.0f}-{cohort['sofa'].quantile(.75):.0f}, "
          f"range {cohort['sofa'].min():.0f}-{cohort['sofa'].max():.0f})")
    if "sofa_resp_raw" in cohort:
        a = cohort["sofa_resp_raw"].notna().mean(); b = cohort["sofa_resp"].notna().mean()
        print(f"       respiratory component: {100*a:.1f}% from raw-table P/F -> "
              f"{100*b:.1f}% using waterfall FiO2 with S/F fallback")
        print(f"       oxygenation source: "
              f"{cohort['oxy_source'].value_counts(dropna=False).to_dict()}")

    # ---- 6. write -----------------------------------------------------------
    COLUMNS = ["patient_id", "died", "death_dttm", "unit_name", "qclif",
               "age_at_admission", "cci", "sofa"]
    out = phi / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = cohort[COLUMNS].copy()
    df.to_parquet(out / "analysis_dataset.parquet", index=False)
    df.to_csv(out / "analysis_dataset.csv", index=False)

    extended = (["unit_id", "encounter_block", "hospitalization_id", "intubation_dttm",
                "admission_dttm", "discharge_dttm", "discharge_category",
                "icu_hours_in_unit", "icu_hours_total", "n_icus", "sex_category",
                "sofa_total_raw", "sofa_resp_raw", "sofa_n_components", "oxy_source"]
               + comp + sofa_crrt.COMPONENT_VALUES + inputs)
    ext_cols = [c for c in dict.fromkeys(extended) if c in cohort.columns]
    missing_ext = [c for c in sofa_crrt.COMPONENT_VALUES if c not in cohort.columns]
    if missing_ext:
        raise AssertionError(f"component covariates absent from the cohort: {missing_ext}")
    cohort[COLUMNS + ext_cols].to_parquet(
        out / "analysis_dataset_extended.parquet", index=False)

    missingness_report(cohort, cohort, comp, out)
    complete = df.dropna(subset=["qclif", "sofa", "cci", "age_at_admission"])
    (out / "manifest.json").write_text(json.dumps({
        "provenance": bc.provenance(site),
        "rows": int(len(df)), "columns": COLUMNS,
        "grain": "one row per patient, first qualifying IMV hospitalization",
        "outcome": {"definition": "discharge_category in {Expired, Hospice} "
                                  "OR death_dttm <= discharge + 1 day",
                    "deaths": int(df["died"].sum()), "rate": round(float(df["died"].mean()), 4)},
        "qclif": {"source": f"unit_panel.{QCLIF_SCORE}", "missing": int(df["qclif"].isna().sum())},
        "sofa": {"window_hours": [-24, 4],
                 "implementation": f"classic SOFA, CLIF-epidemiology-of-CRRT "
                                   f"sofa_calculator.py @ {sofa_crrt.CRRT_COMMIT}",
                 "respiratory": "recomputed from waterfall FiO2, P/F with S/F fallback "
                                "(Rice 2007 equivalents)",
                 "missing": int(df["sofa"].isna().sum()),
                 "component_absent_pct": {c: round(100 * float(cohort[c].isna().mean()), 2)
                                          for c in comp},
                 "all_components_present": int(cohort[comp].notna().all(axis=1).sum()),
                 "note": "a missing subscore is summed as 0, i.e. as normal"},
        "complete_cases": int(len(complete)),
    }, indent=2, default=str))

    print(f"\n[dataset] {len(df):,} rows x {len(COLUMNS)} columns")
    print(f"  complete cases (qclif, sofa, cci, age all present): {len(complete):,} "
          f"({100 * len(complete) / len(df):.1f}%)")
    print(f"  wrote {out}/analysis_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
