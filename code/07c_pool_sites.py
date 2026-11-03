"""Stage 7c — stack the member sites' stage-7 patient rows and unit panels into one pooled dataset.

Each member site runs stages 0-7 on its own data. This stage reads, per site,

  output/<site>/intermediate_phi/analysis/analysis_dataset_extended.parquet  (+ manifest.json)
  output/<site>/final_no_phi/unit_panel.parquet                              (+ panel_provenance.json)

and writes one patient-level frame and one unit panel under output/pooled/, which stage 8 reads
with CLIF_SITE=pooled. Sites are labelled Site A, B, ... in the order of `members` in
sites/pooled.json; the letter-to-site key is written to intermediate_phi only. Keys are prefixed
with the letter, so no site's identifiers or name reach the pooled rows.

Refuses to pool when:
  - definition_version, clif_version or the upstream clif-dashboards SHA differ between sites
  - the code that builds patient rows or panels differs between the commits the sites ran
    (checked with git diff over BUILD_PATHS, so commits that touch nothing else do not block)
  - a required column is missing, a model covariate has gaps, a reference level is absent,
    or a key collides after prefixing

Run:  .venv/bin/python code/07c_pool_sites.py [--profile pooled]
"""
from __future__ import annotations

import argparse
import json
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bundle_config as bc                                             # noqa: E402
from common import covariates as cov                                   # noqa: E402

STEP = "07c_pool_sites"

# Everything that shapes a site's patient rows or unit panel. A commit touching none of these
# cannot change what stages 0-7 produced, so sites built at different such commits still pool.
BUILD_PATHS = ["bundle_config.py", "requirements.txt", "common", "vendor", "definitions",
               "code/00_probe_units.py", "code/01_build_cohort.py", "code/02_metric_records.py",
               "code/03_build_panel.py", "code/04_exposure_assignment.py",
               "code/07_analysis_dataset.py"]

MUST_MATCH = ("definition_version", "clif_version", "upstream_sha")
PREFIXED = ("patient_id", "hospitalization_id", "encounter_block", "unit_name")

# What stage 8 reads; a site missing any of these cannot be pooled.
REQUIRED = ["patient_id", "died", "unit_name", "unit_id", "qclif", "age_at_admission", "cci",
            "sofa", "cci_poa", "hours_to_intubation", "location_at_intubation",
            "procedural_within_24h", "limited_code_status", "admission_type", "n_icus",
            "icu_hours_in_unit", "discharge_category", "sex_category", "race_category",
            "ethnicity_category", "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv_97",
            "sofa_cns", "sofa_renal"]
# Same list as MODEL_COVARIATES in code/07_analysis_dataset.py; tests/test_pool_sites.py asserts it.
MODEL_COVARIATES = ["cci_poa", "hours_to_intubation", "location_at_intubation",
                    "procedural_within_24h", "limited_code_status", "admission_type"]
REFERENCE_LEVELS = {"hours_to_intubation": "0-24 h", "location_at_intubation": "icu",
                    "admission_type": "ed"}


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.fh.write(msg + "\n")


def load_pooled_profile(name: str = "pooled") -> dict:
    path = bc.SITES / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No pooled profile at {path}. Create one with:\n"
                                f"  cp sites/pooled.example.json sites/{name}.json")
    prof = json.loads(path.read_text())
    members = prof.get("members") or []
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError(f"{path}: `members` must list at least two distinct sites")
    if len(members) > len(string.ascii_uppercase):
        raise ValueError(f"{path}: more members than site letters")
    return prof


def site_letters(members: list[str]) -> dict[str, str]:
    return {m: f"Site {string.ascii_uppercase[i]}" for i, m in enumerate(members)}


def git_same_code(sha_a: str | None, sha_b: str | None) -> bool:
    if not sha_a or not sha_b:
        raise ValueError("a site's provenance has no code_version; rerun it from a git checkout")
    if sha_a == sha_b:
        return True
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", sha_a, sha_b, "--", *BUILD_PATHS],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git diff {sha_a} {sha_b} failed: {r.stderr.strip()} "
                           f"(git pull so both commits are present)")
    return r.returncode == 0


def check_versions(provs: dict[str, dict], what: str, same_code=git_same_code) -> list[str]:
    """`provs` maps site -> provenance block. Returns problems; empty means poolable."""
    problems = []
    sites = list(provs)
    for key in MUST_MATCH:
        vals = {s: provs[s].get(key) for s in sites}
        if len(set(vals.values())) > 1:
            problems.append(f"{what}: {key} differs across sites: {vals}")
    ref = sites[0]
    for s in sites[1:]:
        a, b = provs[ref].get("code_version"), provs[s].get("code_version")
        if not same_code(a, b):
            problems.append(f"{what}: build code differs between {ref} ({a}) and {s} ({b}); "
                            f"rerun the older site on the current commit")
    return problems


def check_site_frame(df: pd.DataFrame, site: str) -> list[str]:
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"{site}: required columns absent: {missing}")
    gaps = {c: int(df[c].isna().sum()) for c in MODEL_COVARIATES
            if c in df.columns and df[c].isna().any()}
    if gaps:
        problems.append(f"{site}: model covariates with gaps: {gaps}")
    return problems


def prefix_site(df: pd.DataFrame, letter: str, cols=PREFIXED) -> pd.DataFrame:
    """Prefix keys with the site letter ("A::<id>") so ids unique within a site stay unique."""
    code = letter.split()[-1]
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = code + "::" + out[c].astype("string")
    if "unit_id" in out.columns:
        out["unit_id"] = letter + " · " + out["unit_id"].astype("string")
    out.insert(0, "site", letter)
    return out


def check_pooled(rows: pd.DataFrame, panel: pd.DataFrame) -> list[str]:
    problems = []
    dup = int(rows["patient_id"].duplicated().sum())
    if dup:
        problems.append(f"{dup} duplicate patient_id after prefixing")
    if panel["unit_name"].duplicated().any():
        problems.append("duplicate unit_name in the pooled panel")
    orphan = sorted(set(rows["unit_name"]) - set(panel["unit_name"]))
    if orphan:
        problems.append(f"{len(orphan)} unit(s) in the patient rows have no panel row")
    for col, ref in REFERENCE_LEVELS.items():
        if not rows[col].astype(str).eq(ref).any():
            problems.append(f"reference level {col} = {ref!r} absent from the pooled rows")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="pooled")
    args = ap.parse_args()
    prof = load_pooled_profile(args.profile)
    members, name = prof["members"], args.profile
    letters = site_letters(members)
    share, phi = bc.share_dir(name), bc.phi_dir(name)
    log = Log(share / "logs" / f"{STEP}.log")
    threshold = bc.small_cell_threshold()
    log(f"=== {STEP}: {len(members)} sites -> output/{name}/ ===")

    rows, panels, manifests, panel_provs, problems = {}, {}, {}, {}, []
    for m in members:
        ana = bc.output_root(m) / "intermediate_phi" / "analysis"
        shr = bc.output_root(m) / "final_no_phi"
        need = [ana / "analysis_dataset_extended.parquet", ana / "manifest.json",
                shr / "unit_panel.parquet", shr / "panel_provenance.json"]
        absent = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
        if absent:
            problems.append(f"{letters[m]}: missing {absent}; run stages 0-7 for that site")
            continue
        rows[m] = pd.read_parquet(need[0])
        manifests[m] = json.loads(need[1].read_text())
        panels[m] = pd.read_parquet(need[2])
        panel_provs[m] = json.loads(need[3].read_text())["provenance"]
        problems += check_site_frame(rows[m], letters[m])
    if not problems:
        problems += check_versions({letters[m]: manifests[m]["provenance"] for m in members},
                                   "patient rows")
        problems += check_versions({letters[m]: panel_provs[m] for m in members}, "unit panel")
        impl = {letters[m]: manifests[m]["sofa"]["implementation"] for m in members}
        if len(set(impl.values())) > 1:
            problems.append(f"SOFA implementation differs across sites: {impl}")
    if problems:
        for p in problems:
            log(f"[refused] {p}")
        raise SystemExit(f"{STEP}: {len(problems)} problem(s); nothing written. See the log.")

    pooled_rows = pd.concat([prefix_site(rows[m], letters[m]) for m in members],
                            ignore_index=True)
    pooled_panel = pd.concat([prefix_site(panels[m], letters[m]) for m in members],
                             ignore_index=True)
    problems = check_pooled(pooled_rows, pooled_panel)
    if problems:
        for p in problems:
            log(f"[refused] {p}")
        raise SystemExit(f"{STEP}: {len(problems)} problem(s); nothing written. See the log.")

    unshared = sorted(set().union(*[set(r.columns) for r in rows.values()])
                      - set.intersection(*[set(r.columns) for r in rows.values()]))
    if unshared:
        log(f"[note] columns present at some sites only (NaN elsewhere): {unshared}")

    log("\n[sites]")
    for m in members:
        log(f"    {letters[m]}: {len(rows[m]):,} patients, {panels[m]['unit_name'].nunique()} units, "
            f"stage-7 code {manifests[m]['provenance'].get('code_version')}")
    log(f"    pooled: {len(pooled_rows):,} patients, {pooled_panel['unit_name'].nunique()} units")

    tabs = [cov.tabulate(pooled_rows, v, by="site") for v in REFERENCE_LEVELS]
    tabs.append(cov.tabulate(pooled_rows, "limited_code_status", by="site"))
    table = cov.suppress(pd.concat(tabs, ignore_index=True), threshold)
    log(f"\n[covariate levels by site] counts below {threshold} suppressed")
    for (v, g), t in table.groupby(["variable", "group"], sort=False):
        cells = ", ".join(f"{r.level} {'<supp>' if pd.isna(r.n) else int(r.n)}"
                          for r in t.itertuples())
        log(f"    {v:<24} {g}: {cells}")
    one_site = [(v, lv) for (v, lv), t in table.groupby(["variable", "level"])
                if t["n"].fillna(1).gt(0).sum() == 1]
    if one_site:
        log(f"    levels seen at only one site (confounded with its fixed effect): {one_site}")

    (phi / "analysis").mkdir(parents=True, exist_ok=True)
    pooled_rows.to_parquet(phi / "analysis" / "analysis_dataset_extended.parquet", index=False)
    (phi / "site_key.json").write_text(json.dumps(
        {letters[m]: {"site": m, "site_id": manifests[m]["provenance"].get("site_id")}
         for m in members}, indent=2))
    pooled_panel.to_parquet(share / "unit_panel.parquet", index=False)
    pooled_panel.drop(columns=["unit_name"]).to_csv(share / "unit_panel_anonymized.csv", index=False)
    anon = {letters[m]: {k: v for k, v in manifests[m]["provenance"].items() if k != "site_id"}
            for m in members}
    (share / "pooled_provenance.json").write_text(json.dumps({
        "code_version": bc.git_sha(), "members": anon,
        "panel_provenance": {letters[m]: {k: v for k, v in panel_provs[m].items() if k != "site_id"}
                             for m in members},
        "n_patients": int(len(pooled_rows)), "n_units": int(pooled_panel["unit_name"].nunique()),
        "build_paths_compared": BUILD_PATHS}, indent=2, default=str))

    log(f"\n[done] output/{name}/intermediate_phi/analysis/analysis_dataset_extended.parquet")
    log(f"       output/{name}/final_no_phi/unit_panel.parquet (+ _anonymized.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
