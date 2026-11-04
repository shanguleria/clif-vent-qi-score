"""Stage 7c pooling: prefixing, collision and version refusal. Synthetic frames only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pool():
    return _load("stage07c", "code/07c_pool_sites.py")


def _site_rows(pool, n=3, unit="N1"):
    d = {c: ["x"] * n for c in pool.REQUIRED}
    d.update(patient_id=[str(i) for i in range(n)], unit_name=[unit] * n, unit_id=["Unit 1"] * n,
             hours_to_intubation=["0-24 h"] * n, location_at_intubation=["icu"] * n,
             admission_type=["ed"] * n, died=[0] * n, procedural_within_24h=[0] * n,
             limited_code_status=[0] * n, race_group=["white"] * n,
             bmi_category=["18.5-24.9"] * n, crrt_at_intubation=[0] * n)
    return pd.DataFrame(d)


PROV = {"definition_version": "v1", "clif_version": "2.1.0", "upstream_sha": "f55c379",
        "code_version": "abc1234"}


def test_model_covariates_match_stage7(pool):
    stage07 = _load("stage07_for_7c", "code/07_analysis_dataset.py")
    assert pool.MODEL_COVARIATES == stage07.MODEL_COVARIATES


def test_reference_levels_are_real_levels(pool):
    from common import covariates as cov
    assert pool.REFERENCE_LEVELS["race_group"] in set(cov.RACE_GROUPS.values())
    assert pool.REFERENCE_LEVELS["bmi_category"] in cov.BMI_LABELS


def test_letters_follow_member_order(pool):
    assert pool.site_letters(["rush", "uchicago"]) == {"rush": "Site A", "uchicago": "Site B"}


def test_prefix_makes_colliding_ids_unique_and_hides_site_name(pool):
    a = pool.prefix_site(_site_rows(pool), "Site A")
    b = pool.prefix_site(_site_rows(pool), "Site B")
    both = pd.concat([a, b], ignore_index=True)
    assert not both["patient_id"].duplicated().any()
    assert set(both["unit_name"]) == {"A::N1", "B::N1"}
    assert set(both["unit_id"]) == {"Site A · Unit 1", "Site B · Unit 1"}
    assert set(both["site"]) == {"Site A", "Site B"}


def test_check_pooled_flags_duplicates_orphans_and_missing_reference(pool):
    rows = pool.prefix_site(_site_rows(pool), "Site A")
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    rows.loc[:, "admission_type"] = "osh"
    panel = pd.DataFrame({"unit_name": ["A::OTHER"], "unit_id": ["Site A · Unit 9"]})
    problems = " | ".join(pool.check_pooled(rows, panel))
    assert "duplicate patient_id" in problems
    assert "no panel row" in problems
    assert "admission_type" in problems


def test_check_pooled_passes_a_clean_stack(pool):
    rows = pd.concat([pool.prefix_site(_site_rows(pool), "Site A"),
                      pool.prefix_site(_site_rows(pool), "Site B")], ignore_index=True)
    panel = pd.DataFrame({"unit_name": ["A::N1", "B::N1"]})
    assert pool.check_pooled(rows, panel) == []


def test_check_site_frame_reports_missing_columns_and_gaps(pool):
    df = _site_rows(pool).drop(columns=["cci_poa"])
    df.loc[0, "admission_type"] = None
    problems = " | ".join(pool.check_site_frame(df, "Site A"))
    assert "cci_poa" in problems and "admission_type" in problems


def test_versions_refuse_definition_mismatch(pool):
    provs = {"Site A": PROV, "Site B": {**PROV, "definition_version": "v2"}}
    problems = pool.check_versions(provs, "rows", pool.ROW_PATHS, same_code=lambda a, b, paths: True)
    assert any("definition_version" in p for p in problems)


def test_versions_refuse_when_build_code_differs(pool):
    provs = {"Site A": PROV, "Site B": {**PROV, "code_version": "def5678"}}
    problems = pool.check_versions(provs, "rows", pool.ROW_PATHS, same_code=lambda a, b, paths: a == b)
    assert any("build code differs" in p for p in problems)


def test_versions_pass_different_commits_with_same_build_code(pool):
    provs = {"Site A": PROV, "Site B": {**PROV, "code_version": "def5678"}}
    assert pool.check_versions(provs, "rows", pool.ROW_PATHS, same_code=lambda a, b, paths: True) == []


def test_git_same_code_needs_a_version(pool):
    with pytest.raises(ValueError, match="code_version"):
        pool.git_same_code(None, "abc1234", pool.ROW_PATHS)


def test_git_same_code_on_real_history(pool):
    # 8e79c8d changed common/sofa_crrt.py and code/07 (row paths) but nothing stages 0-4 import.
    assert pool.git_same_code("d4fd793", "d4fd793", pool.ROW_PATHS)
    assert not pool.git_same_code("d4fd793", "8e79c8d", pool.ROW_PATHS)
    assert pool.git_same_code("d4fd793", "8e79c8d", pool.PANEL_PATHS)


def test_panel_paths_are_a_subset_of_row_paths(pool):
    assert set(pool.PANEL_PATHS) <= set(pool.ROW_PATHS)


def test_versions_allow_clif_patch_difference(pool):
    provs = {"Site A": PROV, "Site B": {**PROV, "clif_version": "2.1.1"}}
    assert pool.check_versions(provs, "rows", pool.ROW_PATHS, same_code=lambda a, b, p: True) == []


def test_versions_refuse_clif_minor_difference(pool):
    provs = {"Site A": PROV, "Site B": {**PROV, "clif_version": "2.2.0"}}
    problems = pool.check_versions(provs, "rows", pool.ROW_PATHS, same_code=lambda a, b, p: True)
    assert any("clif_version" in p for p in problems)
