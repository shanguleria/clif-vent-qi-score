"""The P/F time-unit workaround around the vendored calculator. Synthetic frames only."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sofa_crrt  # noqa: E402

TZ = "America/Chicago"


def _frames(lab_unit, resp_unit):
    labs = pl.DataFrame({
        "hospitalization_id": ["h1", "h1"],
        "lab_result_dttm": [datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 20)],
        "po2_arterial": [80.0, 60.0]}).with_columns(
        pl.col("lab_result_dttm").cast(pl.Datetime(lab_unit)).dt.replace_time_zone(TZ))
    resp = pl.DataFrame({
        "hospitalization_id": ["h1", "h1"],
        "recorded_dttm": [datetime(2024, 1, 1, 9), datetime(2024, 1, 1, 12)],
        "fio2_set": [0.4, 0.8],
        "device_category": ["imv", "imv"]}).with_columns(
        pl.col("recorded_dttm").cast(pl.Datetime(resp_unit)).dt.replace_time_zone(TZ))
    return labs, resp


def test_unpatched_upstream_fails_on_mixed_units():
    calc = sofa_crrt.load_calculator()
    original = calc._calculate_concurrent_pf_ratios.__wrapped__
    labs, resp = _frames("ns", "us")
    with pytest.raises(Exception):
        original(labs, resp)


@pytest.mark.parametrize("units", [("ns", "us"), ("us", "ns"), ("us", "us"), ("ns", "ns")])
def test_patched_pairing_succeeds_and_matches_most_recent_fio2(units):
    calc = sofa_crrt.load_calculator()
    out = calc._calculate_concurrent_pf_ratios(*_frames(*units)).sort("lab_result_dttm")
    # 10:00 pairs with the 09:00 FiO2 (P/F 200); 20:00 has no FiO2 within the 4 h lookback
    assert out["concurrent_pf"].to_list() == [pytest.approx(200.0)]


def _assessments(tmp_path):
    rows = [  # (hospitalization, recorded, category, value)
        ("h1", datetime(2024, 1, 1, 11), "gcs_total", 4.0),    # 25 h before: outside
        ("h1", datetime(2024, 1, 2, 0), "gcs_total", 10.0),    # inside
        ("h1", datetime(2024, 1, 2, 6), "gcs_total", 0.0),     # out of range
        ("h1", datetime(2024, 1, 2, 7), "gcs_motor", 1.0),     # other category
        ("h1", datetime(2024, 1, 2, 12), "gcs_total", 3.0),    # at intubation: excluded
        ("h1", datetime(2024, 1, 2, 13), "gcs_total", 3.0),    # after: excluded
        ("h2", datetime(2024, 1, 2, 12), "gcs_total", 3.0),    # only at intubation -> none
        ("h3", datetime(2024, 1, 1, 12), "gcs_total", 12.0),   # exactly 24 h before: included
        ("h3", datetime(2024, 1, 2, 9), "gcs_total", 14.0),
    ]
    h, t, c, v = zip(*rows)
    pl.DataFrame({"hospitalization_id": list(h), "recorded_dttm": list(t),
                  "assessment_category": list(c), "numerical_value": list(v),
                  "categorical_value": [None] * len(rows)},
                 schema_overrides={"categorical_value": pl.Utf8}).with_columns(
        pl.col("recorded_dttm").dt.replace_time_zone("UTC")
    ).write_parquet(tmp_path / "clif_patient_assessments.parquet")


def test_gcs_before_intubation_window(tmp_path):
    import pandas as pd
    _assessments(tmp_path)
    mapping = pd.DataFrame({"hospitalization_id": ["h1", "h2", "h3"],
                            "encounter_block": ["1", "2", "3"]})
    cohort = pd.DataFrame({"encounter_block": ["1", "2", "3"],
                           "intubation_dttm": pd.to_datetime(["2024-01-02 12:00"] * 3, utc=True)})
    out = sofa_crrt.gcs_before_intubation(tmp_path, "UTC", mapping, cohort)
    got = dict(zip(out["encounter_block"].astype(str), out["gcs_pre"]))
    assert got == {"1": 10.0, "3": 12.0}


def test_gcs_before_intubation_needs_the_table(tmp_path):
    import pandas as pd
    with pytest.raises(FileNotFoundError):
        sofa_crrt.gcs_before_intubation(
            tmp_path, "UTC", pd.DataFrame({"hospitalization_id": [], "encounter_block": []}),
            pd.DataFrame({"encounter_block": [], "intubation_dttm": []}))
