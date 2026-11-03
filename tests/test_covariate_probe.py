"""Stage 7b's hospital-structure summary. Synthetic frames only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _probe():
    spec = importlib.util.spec_from_file_location("stage07b", ROOT / "code" / "07b_covariate_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ADT = pd.DataFrame({
    "location_category": ["ICU", "icu", "icu", "icu", "ward", "icu", "icu"],
    "location_name":     ["A",   "A",   "B",   "C",   "W",    "B",   "Z"],
    "hospital_id":       ["h2",  "h2",  "h1",  "h1",  "h9",   "h2",  "h1"]})
UNITS = {"A": "Unit 1", "B": "Unit 2", "C": "Unit 3"}


def test_hospitals_are_anonymized_and_units_mapped():
    uh = _probe().unit_hospitals(ADT, UNITS, {})
    assert set(uh["hospital"]) == {"Hospital 1", "Hospital 2"}
    assert not uh.isin(["h1", "h2", "A", "B", "C"]).any().any()
    assert sorted(uh.loc[uh["hospital"] == "Hospital 1", "unit_id"]) == ["Unit 2", "Unit 3"]


def test_non_icu_and_unpanelled_names_are_dropped():
    uh = _probe().unit_hospitals(ADT, UNITS, {})
    assert "Hospital 3" not in set(uh["hospital"])
    assert set(uh["unit_id"]) == {"Unit 1", "Unit 2", "Unit 3"}


def test_a_unit_under_two_hospitals_appears_twice():
    uh = _probe().unit_hospitals(ADT, UNITS, {})
    assert (uh["unit_id"] == "Unit 2").sum() == 2


def test_collapse_map_is_applied_before_lookup():
    adt = pd.DataFrame({"location_category": ["icu"], "location_name": ["A-old"], "hospital_id": ["h1"]})
    assert _probe().unit_hospitals(adt, UNITS, {"A-old": "A"})["unit_id"].tolist() == ["Unit 1"]


def test_missing_hospital_id_is_labelled_not_numbered():
    adt = pd.DataFrame({"location_category": ["icu"], "location_name": ["A"], "hospital_id": [None]})
    assert _probe().unit_hospitals(adt, UNITS, {})["hospital"].tolist() == ["Hospital missing"]


def test_primary_diagnosis_counts_and_chapter():
    hd = pd.DataFrame({
        "encounter_block":       ["1", "1", "2", "2", "2", "3"],
        "diagnosis_code":        ["J96.01", "E11.9", "I21.4", "A41.9", "N17.9", "428.0"],
        "diagnosis_code_format": ["ICD10CM", "ICD10CM", "icd10cm", "ICD10CM", "ICD10CM", "ICD9CM"],
        "diagnosis_primary":     pd.array([1, 0, 1, 1, None, 1], dtype="Int64")})
    cohort = pd.DataFrame({"encounter_block": ["1", "2", "3", "4"]})
    n, chapter, dist = _probe().primary_diagnosis(hd, cohort)
    assert n.tolist() == ["1", "2+", "1", "0"]
    # block 2 has two primaries; the lowest code (A41.9) decides
    assert chapter.tolist() == ["respiratory", "infectious", "not_icd10cm", "none_flagged"]
    assert dist["1"] == 4 and dist["0"] == 1 and sum(dist.values()) == 6


@pytest.mark.parametrize("code, chapter", [
    ("A41.9", "infectious"), ("J96.01", "respiratory"), ("I21.4", "circulatory"),
    ("C34.90", "neoplasm"), ("D49.9", "neoplasm"), ("D50.0", "blood_immune"),
    ("H59.0", "eye"), ("H60.1", "ear"), ("S06.5X0A", "injury_poisoning"),
    ("T40.2X1A", "injury_poisoning"), ("U07.1", "special_purpose"), ("Z99.11", "health_status_contact"),
    ("o9a.211", "pregnancy"), ("", "unclassified"), ("123.4", "unclassified"), ("DX1", "unclassified")])
def test_icd10_chapter(code, chapter):
    assert _probe().icd10_chapter(code) == chapter
