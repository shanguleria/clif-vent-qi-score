"""Stage 7 — assemble the patient-level analysis dataset for the outcome model.

One row per patient, using their FIRST qualifying IMV hospitalisation:

  patient_id  died  death_dttm  unit_name  qclif  age_at_admission  cci  sofa

`died` is `discharge_category` in {Expired, Hospice} OR `death_dttm` within a day of discharge.
CLIF's `death_dttm` is an ALL-CAUSE date that often falls long after discharge, so it must not be
used unqualified as an in-hospital outcome; it is still emitted verbatim as its own column.

The ICU is the one holding the most ADT time in the hospitalisation, not the most ventilated time,
so patients intubated in the ED or moved between units are attributed to where they were cared
for. SOFA spans 24 h before to 4 h after intubation, so the intubation counts toward the severity
it marks.

The case-mix covariates in MODEL_COVARIATES (present-on-admission Charlson, admission-to-intubation
time, location at intubation, a procedural stay in the prior 24 h, limited code status, admission
type, race group, BMI category, CRRT in the SOFA window) are built in `common/covariates.py` and
carried in the extended frame, which stage 8 reads.

Writes to intermediate_phi/analysis/: the eight columns, an extended frame with covariates and
diagnostics, and missingness reports per covariate, per SOFA component and per unit.

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
from common import covariates as cov                                   # noqa: E402

from common import sofa_crrt                                           # noqa: E402

from clif_dash import stage                                            # noqa: E402
from clif_dash.upstream import clif_io                                 # noqa: E402
from clif_dash.upstream import metrics as upstream_metrics             # noqa: E402

warnings.filterwarnings("ignore", message=".*generic.*unit for NumPy timedelta.*")

# Hospice counts as a death by convention -- a modelling choice, not a fact about the data.
DEATH_CATEGORIES = {"expired", "hospice"}

DEATH_DTTM_GRACE = pd.Timedelta(days=1)

SOFA_BEFORE = pd.Timedelta(hours=24)
SOFA_AFTER = pd.Timedelta(hours=4)

QCLIF_SCORE = "score_core"

# A missing subscore is summed as 0, i.e. as normal, so a charting gap is indistinguishable from
# health unless the inputs behind it are inspected.
REQUIRED_EXTENDED = ["sex_category", "race_category", "ethnicity_category"]

# The primary adjustment beyond age, sofa and cci, all anchored at the first intubation.
# code/07c_pool_sites.py carries the same list; tests/test_pool_sites.py asserts it.
MODEL_COVARIATES = ["cci_poa", "hours_to_intubation", "location_at_intubation",
                    "procedural_within_24h", "limited_code_status", "admission_type",
                    "race_group", "bmi_category", "crrt_at_intubation"]

POA_COLUMN = "poa_present"

SOFA_INPUTS = {
    "sofa_cns":    ["gcs_total"],
    "sofa_resp":   ["pf", "sf"],
    "sofa_cv_97":  ["map", "nee"],
    "sofa_liver":  ["bilirubin_total"],
    "sofa_renal":  ["creatinine"],
    "sofa_coag":   ["platelet_count"],
}


def clifpy_config(site) -> Path:
    """clifpy's own config shape, derived from the site profile rather than duplicated."""
    prof = bc.load_profile(site)
    p = bc.upstream_build_dir(site) / "clifpy_config.json"
    p.write_text(json.dumps({"data_directory": prof["data_path"],
                             "filetype": prof.get("file_format", "parquet"),
                             "timezone": prof["timezone"]}, indent=2))
    return p


def first_encounter_per_patient(episodes, encounters) -> pd.DataFrame:
    """First qualifying IMV hospitalisation per patient, with its first intubation time and the
    length of that first episode (end - start, so gaps under the episode-gap rule are included)."""
    ep = episodes.copy()
    ep["start"] = pd.to_datetime(ep["start"], utc=True)
    ep["end"] = pd.to_datetime(ep["end"], utc=True)
    ep["encounter_block"] = attr.norm_block(ep["encounter_block"])
    first_ep = (ep.sort_values(["encounter_block", "start", "end"], kind="stable")
                .drop_duplicates("encounter_block").reset_index(drop=True))
    first_ep = pd.DataFrame({
        "encounter_block": first_ep["encounter_block"],
        "intubation_dttm": first_ep["start"],
        "first_episode_hours": (first_ep["end"] - first_ep["start"]).dt.total_seconds() / 3600})
    enc = encounters.copy()
    enc["encounter_block"] = attr.norm_block(enc["encounter_block"])
    m = first_ep.merge(enc, on="encounter_block", how="left")
    # Earliest intubation wins; encounter_block breaks any tie deterministically.
    return (m.sort_values(["patient_id", "intubation_dttm", "encounter_block"], kind="stable")
            .drop_duplicates("patient_id").reset_index(drop=True))


def dominant_icu(cohort, icu) -> pd.DataFrame:
    """The ICU holding the most ADT time. Open stays are clipped to discharge first -- the
    FAR_FUTURE sentinel would otherwise win every comparison."""
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


def height_weight(cfg, mapping, site) -> pd.DataFrame:
    """Per block, the first height and first weight in the stay, from upstream
    `metrics.first_height` / `first_weight`, so BMI uses the height behind predicted body weight.
    The vitals table is too large to read whole; only the two categories for the cohort's
    hospitalizations are scanned. `mapping` carries string ids and normalized blocks."""
    import duckdb
    path = cfg.table_path("vitals")
    if not path.exists():
        raise FileNotFoundError(f"BMI needs {path}")
    con = duckdb.connect()
    con.register("ids", mapping[["hospitalization_id"]].drop_duplicates())
    v = con.execute(f"""
        SELECT CAST(s.hospitalization_id AS VARCHAR) AS hospitalization_id, s.recorded_dttm,
               lower(trim(CAST(s.vital_category AS VARCHAR))) AS vital_category,
               TRY_CAST(s.vital_value AS DOUBLE) AS vital_value
        FROM read_parquet('{path}') s
        JOIN ids ON CAST(s.hospitalization_id AS VARCHAR) = ids.hospitalization_id
        WHERE lower(trim(CAST(s.vital_category AS VARCHAR))) IN ('height_cm', 'weight_kg')
    """).df()
    con.close()
    v = stage.clif_conventions(v)
    lo, hi = bc.decision_value("ltvv", "height_range_cm", site)
    h = upstream_metrics.first_height(v, mapping, lo, hi)["height_cm"]
    w = upstream_metrics.first_weight(v, mapping).rename("weight_kg")
    out = pd.concat([h, w], axis=1).rename_axis("encounter_block").reset_index()
    out["encounter_block"] = attr.norm_block(out["encounter_block"])
    return out


def poa_mask(flag: pd.Series) -> np.ndarray:
    """True where the flag is 1. A missing flag is not present on admission; with a nullable
    integer column, `.eq(1)` alone yields NA there, which pandas refuses as a row mask."""
    return pd.to_numeric(flag, errors="coerce").eq(1).fillna(False).to_numpy(dtype=bool)


def require_columns(cohort, cols, what):
    """The extended export drops a column it cannot find, so anything load-bearing is named
    here and fails the run instead of going missing from the artifact."""
    missing = [c for c in cols if c not in cohort.columns]
    if missing:
        raise AssertionError(f"{what} absent from the cohort: {missing}")


def missingness_report(df, sofa, comp, out, log=print):
    """Per-covariate and per-component missingness, plus the inputs behind each component."""
    rows = []

    def add(group, var, present, n):
        rows.append({"group": group, "variable": var, "n": n, "n_present": int(present),
                     "n_missing": int(n - present), "pct_missing": round(100 * (n - present) / n, 2)})

    n = len(df)
    for c in ["died", "unit_name", "qclif", "age_at_admission", "cci", "sofa"] + MODEL_COVARIATES:
        add("covariate", c, df[c].notna().sum(), n)
    # death_dttm is legitimately absent for survivors; report it against decedents only.
    add("covariate", "death_dttm (among died==1)",
        df.loc[df["died"] == 1, "death_dttm"].notna().sum(), int((df["died"] == 1).sum()))

    # Table 1 variables, not model covariates -- reported separately so the group column does not
    # imply they enter the adjustment.
    for c in ("sex_category", "race_category", "ethnicity_category"):
        add("demographic", c, df[c].notna().sum(), n)

    for c in comp:
        add("sofa_component", c, sofa[c].notna().sum(), n)
    for c, inputs in SOFA_INPUTS.items():
        for i in inputs:
            if i in sofa.columns:
                add(f"sofa_input::{c}", i, sofa[i].notna().sum(), n)

    rep = pd.DataFrame(rows)
    rep.to_csv(out / "missingness_report.csv", index=False)

    log("\n[missingness] covariates")
    for r in rep[rep.group.isin(["covariate", "demographic"])].itertuples():
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

    # Unit-differential charting becomes unit-differential measured severity.
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
    mp = mapping.copy()
    mp["hospitalization_id"] = mp["hospitalization_id"].astype(str)
    mp["encounter_block"] = attr.norm_block(mp["encounter_block"])

    def cci_by_block(diagnoses, name):
        cci = clif_io.charlson(diagnoses)
        cci["hospitalization_id"] = cci["hospitalization_id"].astype(str)
        return (cci.merge(mp, on="hospitalization_id")
                .groupby("encounter_block")["cci_score"].max().rename(name))

    if POA_COLUMN not in hd.columns:
        raise AssertionError(f"hospital_diagnosis.{POA_COLUMN} is required for cci_poa")
    # Reported, because a site whose flag is mostly missing would otherwise get a near-zero cci_poa
    # without any error.
    poa_raw = pd.to_numeric(hd[POA_COLUMN], errors="coerce")
    poa = poa_mask(hd[POA_COLUMN])
    print(f"[poa] {POA_COLUMN}: {int(poa.sum()):,} rows = 1, "
          f"{int(poa_raw.eq(0).fillna(False).sum()):,} = 0, "
          f"{int(poa_raw.isna().sum()):,} missing (treated as not present on admission)")
    for diagnoses, name in ((hd, "cci"), (hd[poa], "cci_poa")):
        cohort = cohort.merge(cci_by_block(diagnoses, name), on="encounter_block", how="left")
        cohort[name] = cohort[name].fillna(0)
    print(f"[covariates] age median {cohort['age_at_admission'].median():.0f}; "
          f"cci median {cohort['cci'].median():.0f} "
          f"({int((cohort['cci'] == 0).sum()):,} with no coded comorbidity); "
          f"cci_poa median {cohort['cci_poa'].median():.0f} "
          f"({100 * poa.mean():.1f}% of diagnosis rows present on admission)")

    # ---- 4b. case mix at intubation ------------------------------------------
    blocks = set(cohort["encounter_block"])
    adt = cov.attach_block(stage.read_whole_table(cfg, "adt"), mp, blocks)
    cohort["location_at_intubation"] = cov.collapse_location(
        cov.location_at_intubation(cohort, adt))
    cohort["procedural_within_24h"] = cov.procedural_within(
        cohort, adt, cov.PROCEDURAL_HOURS).astype(int)
    cohort["hours_to_intubation"] = cov.hours_category(
        cohort["intubation_dttm"], cohort["admission_dttm"])
    hosp = cov.attach_block(stage.read_whole_table(cfg, "hospitalization"), mp, blocks)
    cohort["admission_type"] = cov.admission_type(cohort, hosp)
    cs = stage.read_whole_table(cfg, "code_status")
    cs = cs[cs["patient_id"].astype(str).isin(set(cohort["patient_id"].astype(str)))]
    cohort["code_status_before_intubation"] = cov.code_status_before(cohort, cs)
    cohort["limited_code_status"] = cov.limited_code_status(cohort["code_status_before_intubation"])
    cohort["race_group"] = cov.race_group(cohort["race_category"])
    cohort = cohort.merge(height_weight(cfg, mp, site), on="encounter_block", how="left")
    cohort["bmi"] = cov.bmi(cohort["weight_kg"], cohort["height_cm"])
    cohort["bmi_category"] = cov.bmi_category(cohort["bmi"])
    print(f"       height {100 * cohort['height_cm'].notna().mean():.1f}%, "
          f"weight {100 * cohort['weight_kg'].notna().mean():.1f}%, "
          f"plausible BMI {100 * cohort['bmi'].notna().mean():.1f}%")
    crrt = cov.attach_block(stage.read_whole_table(cfg, "crrt_therapy"), mp, blocks)
    cohort["crrt_at_intubation"] = cov.crrt_in_window(cohort, crrt, SOFA_BEFORE, SOFA_AFTER)
    for c in ("location_at_intubation", "procedural_within_24h", "hours_to_intubation",
              "admission_type", "limited_code_status", "race_group", "bmi_category",
              "crrt_at_intubation"):
        print(f"       {c}: {cohort[c].value_counts(dropna=False).sort_index().to_dict()}")

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
    prof = bc.load_profile(site)
    gcs = sofa_crrt.gcs_before_intubation(prof["data_path"], prof["timezone"], mp2, win,
                                          prof.get("file_format", "parquet"))
    cohort = cohort.merge(gcs, on="encounter_block", how="left")
    print(f"       gcs_pre (worst GCS in the {sofa_crrt.GCS_BEFORE.total_seconds() / 3600:.0f} h "
          f"before intubation): recorded for {100 * cohort['gcs_pre'].notna().mean():.1f}%")
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
    # Every check runs before the first write, so a failed run leaves the previous outputs intact
    # rather than a new short file beside a stale or missing extended one.
    require_columns(cohort, sofa_crrt.COMPONENT_VALUES + ["gcs_pre", "oxy_source"],
                    "physiology covariates")
    require_columns(cohort, REQUIRED_EXTENDED, "Table 1 demographics")
    require_columns(cohort, MODEL_COVARIATES, "model covariates")
    require_columns(cohort, ["first_episode_hours"], "first IMV episode length")
    nulls = {c: int(cohort[c].isna().sum()) for c in MODEL_COVARIATES if cohort[c].isna().any()}
    if nulls:
        raise AssertionError(f"model covariates must be complete by construction: {nulls}")

    out = phi / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = cohort[COLUMNS].copy()
    df.to_parquet(out / "analysis_dataset.parquet", index=False)
    df.to_csv(out / "analysis_dataset.csv", index=False)

    extended = (["unit_id", "encounter_block", "hospitalization_id", "intubation_dttm",
                "admission_dttm", "discharge_dttm", "discharge_category",
                "icu_hours_in_unit", "icu_hours_total", "n_icus", "first_episode_hours",
                "sex_category", "race_category", "ethnicity_category",
                *MODEL_COVARIATES, "code_status_before_intubation",
                "height_cm", "weight_kg", "bmi",
                "sofa_total_raw", "sofa_resp_raw", "sofa_n_components", "oxy_source", "gcs_pre"]
               + comp + sofa_crrt.COMPONENT_VALUES + inputs)
    ext_cols = [c for c in dict.fromkeys(extended) if c in cohort.columns]
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
        "model_covariates": {
            "cci_poa": f"Charlson from hospital_diagnosis rows with {POA_COLUMN} == 1",
            "hours_to_intubation": dict(zip(cov.HOURS_LABELS,
                                            [str(b) for b in cov.HOURS_BINS[1:]])),
            "location_at_intubation": f"adt.location_category spanning intubation; kept "
                                      f"{list(cov.LOCATION_KEPT)}, no covering row -> "
                                      f"{cov.LOCATION_UNKNOWN}, rest -> {cov.LOCATION_OTHER}",
            "procedural_within_24h": f"procedural ADT stay overlapping the "
                                     f"{cov.PROCEDURAL_HOURS} h before intubation",
            "limited_code_status": f"last code_status order at or before intubation in "
                                   f"{sorted(cov.LIMITED_CODE)}",
            "admission_type": "admission_type_category of the block's first hospitalization",
            "race_group": {**cov.RACE_GROUPS, "<missing>": "unknown"},
            "bmi_category": {"bands_left_closed": cov.BMI_LABELS,
                             "plausible_bmi": list(cov.BMI_PLAUSIBLE),
                             "inputs": "first weight_kg (upstream first_weight) and first "
                                       "height_cm within ltvv.height_range_cm (upstream "
                                       "first_height) in the stay",
                             "absent": cov.MISSING},
            "crrt_at_intubation": f"any crrt_therapy row with a mode or flow in "
                                  f"[-{SOFA_BEFORE.total_seconds() / 3600:g} h, "
                                  f"+{SOFA_AFTER.total_seconds() / 3600:g} h] around intubation",
        },
        "complete_cases": int(len(complete)),
    }, indent=2, default=str))

    print(f"\n[dataset] {len(df):,} rows x {len(COLUMNS)} columns")
    print(f"  complete cases (qclif, sofa, cci, age all present): {len(complete):,} "
          f"({100 * len(complete) / len(df):.1f}%)")
    print(f"  wrote {out}/analysis_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
