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
