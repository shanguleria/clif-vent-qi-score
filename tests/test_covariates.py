"""Candidate covariates anchored at intubation, and small-cell suppression. Synthetic frames only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import covariates as cov  # noqa: E402


def ts(h):
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)


@pytest.fixture
def cohort():
    return pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                         "encounter_block": ["1", "2", "3", "4"],
                         "admission_dttm": [ts(0)] * 4,
                         "intubation_dttm": [ts(10), ts(5), ts(50), ts(3)]})


@pytest.fixture
def adt():
    return pd.DataFrame({
        "encounter_block": ["1", "1", "2", "2", "3", "4"],
        "in_dttm":  [ts(0), ts(5), ts(0), ts(5), ts(0), ts(0)],
        "out_dttm": [ts(5), pd.NaT, ts(5), ts(40), ts(20), ts(30)],
        "location_category": ["ED", "ICU", "Procedural", "ICU", "Ward", None]})


def test_location_uses_the_spanning_row(cohort, adt):
    loc = cov.location_at_intubation(cohort, adt)
    assert loc.loc[0] == "icu"


def test_location_at_transfer_instant_is_destination(cohort, adt):
    assert cov.location_at_intubation(cohort, adt).loc[1] == "icu"


def test_location_gap_is_not_carried_forward(cohort, adt):
    assert cov.location_at_intubation(cohort, adt).loc[2] == cov.NO_ADT_ROW


def test_location_null_category_is_missing_not_dropped(cohort, adt):
    assert cov.location_at_intubation(cohort, adt).loc[3] == cov.MISSING


def test_location_overlap_resolves_to_latest_start():
    c = pd.DataFrame({"patient_id": ["p"], "encounter_block": ["1"], "intubation_dttm": [ts(6)]})
    a = pd.DataFrame({"encounter_block": ["1", "1"], "in_dttm": [ts(0), ts(4)],
                      "out_dttm": [ts(10), ts(8)], "location_category": ["Ward", "ICU"]})
    assert cov.location_at_intubation(c, a).iloc[0] == "icu"


def test_location_aligns_to_cohort_index(adt):
    c = pd.DataFrame({"patient_id": ["p1"], "encounter_block": ["1"],
                      "intubation_dttm": [ts(10)]}, index=[42])
    assert list(cov.location_at_intubation(c, adt).index) == [42]


def test_procedural_window(cohort, adt):
    flag = cov.procedural_within(cohort, adt, hours=24)
    assert flag.tolist() == [False, True, False, False]


def test_procedural_window_excludes_older_stays():
    c = pd.DataFrame({"patient_id": ["p"], "encounter_block": ["1"], "intubation_dttm": [ts(48)]})
    a = pd.DataFrame({"encounter_block": ["1"], "in_dttm": [ts(0)], "out_dttm": [ts(10)],
                      "location_category": ["procedural"]})
    assert not cov.procedural_within(c, a, hours=24).iloc[0]
    assert cov.procedural_within(c, a, hours=48).iloc[0]


def test_code_status_takes_last_order_before_intubation(cohort):
    cs = pd.DataFrame({"patient_id": ["p1", "p1", "p1", "p2"],
                       "start_dttm": [ts(0), ts(8), ts(11), ts(6)],
                       "code_status_category": ["Full", "DNR", "DNR/DNI", "AND"]})
    out = cov.code_status_before(cohort, cs)
    assert out.tolist() == ["DNR", cov.NONE_RECORDED, cov.NONE_RECORDED, cov.NONE_RECORDED]


def test_code_status_matches_integer_patient_ids():
    c = pd.DataFrame({"patient_id": [7], "encounter_block": ["1"], "admission_dttm": [ts(0)],
                      "intubation_dttm": [ts(5)]})
    cs = pd.DataFrame({"patient_id": ["7"], "start_dttm": [ts(1)], "code_status_category": ["Full"]})
    assert cov.code_status_before(c, cs).iloc[0] == "Full"


def test_code_status_ignores_orders_from_earlier_admissions():
    c = pd.DataFrame({"patient_id": ["p", "q"], "encounter_block": ["1", "2"],
                      "admission_dttm": [ts(100), ts(100)], "intubation_dttm": [ts(110), ts(110)]})
    cs = pd.DataFrame({"patient_id": ["p", "q", "q"],
                       "start_dttm": [ts(-5000), ts(-5000), ts(101)],
                       "code_status_category": ["DNR", "DNR", "Full"]})
    assert cov.code_status_before(c, cs).tolist() == [cov.NONE_RECORDED, "Full"]


def test_code_status_counts_an_order_at_the_admission_instant():
    c = pd.DataFrame({"patient_id": ["p"], "encounter_block": ["1"],
                      "admission_dttm": [ts(100)], "intubation_dttm": [ts(110)]})
    cs = pd.DataFrame({"patient_id": ["p"], "start_dttm": [ts(100)], "code_status_category": ["DNR"]})
    assert cov.code_status_before(c, cs).tolist() == ["DNR"]


def test_tabulate_overall_and_by_group():
    df = pd.DataFrame({"x": ["a", "a", "b", None], "died": [1, 0, 1, 0], "u": [1, 1, 2, 2]})
    all_ = cov.tabulate(df, "x").set_index("level")
    assert all_.loc["a", "n"] == 2 and all_.loc["a", "deaths"] == 1
    assert all_.loc[cov.MISSING, "n"] == 1
    assert (all_["n_total"] == 4).all()
    by = cov.tabulate(df, "x", by="u")
    assert set(by["group"]) == {"1", "2"}
    assert by.groupby("group")["pct"].sum().round(6).eq(100).all()


def _tab(ns, deaths, group="g"):
    n = np.array(ns)
    return pd.DataFrame({"variable": "v", "by": "all", "group": group,
                         "level": [f"l{i}" for i in range(len(ns))], "n": n,
                         "n_total": n.sum(), "pct": 100 * n / n.sum(), "deaths": deaths,
                         "mortality": 100 * np.array(deaths) / n})


def test_suppress_blanks_small_counts_and_a_complement():
    t = cov.suppress(_tab([100, 50, 5], [20, 20, 0]), threshold=11)
    assert t["n"].isna().tolist() == [False, True, True]
    assert t["pct"].isna().tolist() == [False, True, True]


def test_suppress_needs_no_complement_when_two_cells_are_small():
    t = cov.suppress(_tab([100, 5, 3], [20, 0, 0]), threshold=11)
    assert t["n"].isna().tolist() == [False, True, True]


def test_suppress_keeps_zero_counts():
    t = cov.suppress(_tab([100, 50, 0], [20, 20, 0]), threshold=11)
    assert not t["n"].isna().any()


def test_suppress_blanks_mortality_when_survivors_are_small():
    t = cov.suppress(_tab([100, 50, 40], [20, 45, 20]), threshold=11)
    assert pd.isna(t.loc[1, "mortality"])
    assert t.loc[1, "n"] == 50
    assert t["deaths"].isna().sum() >= 2


def test_suppress_never_leaves_a_lone_blank_in_a_group():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ns = rng.integers(0, 40, size=4)
        deaths = [int(rng.integers(0, k + 1)) for k in ns]
        t = cov.suppress(_tab(ns, deaths), threshold=11)
        assert t["n"].isna().sum() != 1
        assert t["deaths"].isna().sum() != 1


def test_hours_category_bins_and_reference_label():
    adm = pd.Series([ts(10)] * 6)
    intub = pd.Series([ts(5), ts(10), ts(30), ts(60), ts(150), ts(400)])
    assert cov.hours_category(intub, adm).tolist() == [
        "<= 0 h", "<= 0 h", "0-24 h", "24-72 h", "72 h-7 d", "> 7 d"]
    assert "0-24 h" in cov.HOURS_LABELS


def test_collapse_location():
    loc = pd.Series(["ed", "icu", "procedural", "ward", "l&d", "stepdown",
                     cov.NO_ADT_ROW, cov.MISSING], index=range(10, 18))
    out = cov.collapse_location(loc)
    assert out.tolist() == ["ed", "icu", "procedural", cov.LOCATION_OTHER, cov.LOCATION_OTHER,
                            cov.LOCATION_OTHER, cov.LOCATION_UNKNOWN, cov.LOCATION_UNKNOWN]
    assert list(out.index) == list(loc.index)


def test_limited_code_status_binary():
    s = pd.Series(["DNAR", "DNAR/DNI", "Full", cov.NONE_RECORDED, "DNR", "Presume Full", "AND"])
    assert cov.limited_code_status(s).tolist() == [1, 1, 0, 0, 1, 0, 1]


@pytest.mark.parametrize("unclassified", [cov.MISSING, "Comfort"])
def test_limited_code_status_refuses_unclassified(unclassified):
    with pytest.raises(ValueError, match="neither"):
        cov.limited_code_status(pd.Series(["Full", unclassified]))


def test_attach_block_keeps_only_requested_blocks():
    mapping = pd.DataFrame({"hospitalization_id": ["10", "11", "12"],
                            "encounter_block": ["1", "1", "2"]})
    df = pd.DataFrame({"hospitalization_id": [10, 11, 12, 13], "x": list("abcd")})
    out = cov.attach_block(df, mapping, {"1"})
    assert out["x"].tolist() == ["a", "b"]
    assert set(out["encounter_block"]) == {"1"}


def test_admission_type_takes_first_hospitalization_and_fills_unknown():
    cohort = pd.DataFrame({"encounter_block": ["1", "2", "3"]})
    hosp = pd.DataFrame({"encounter_block": ["1", "1", "2"],
                         "hospitalization_id": ["11", "10", "20"],
                         "admission_dttm": [ts(5), ts(0), ts(0)],
                         "admission_type_category": ["ED", "OSH", None]})
    assert cov.admission_type(cohort, hosp).tolist() == ["osh", "unknown", "unknown"]


def test_other_code_status_is_not_limited():
    s = pd.Series(["Other", "other ", "DNAR"])
    assert cov.limited_code_status(s).tolist() == [0, 0, 1]


def test_race_group_maps_mcide_and_missing():
    s = pd.Series(["White", "Black or African American", "Asian", " other", "Unknown", None,
                   "American Indian or Alaska Native"])
    assert cov.race_group(s).tolist() == ["white", "black", "other", "other", "unknown",
                                          "unknown", "other"]


def test_race_group_refuses_unclassified():
    with pytest.raises(ValueError, match="RACE_GROUPS"):
        cov.race_group(pd.Series(["White", "Black"]))


def test_bmi_and_who_bands_are_left_closed():
    w = pd.Series([50.0, 25.0, 30.0, 40.0, 80.0, None, 500.0])
    b = cov.bmi(w, pd.Series([100.0] * 6 + [100.0]))
    assert b.iloc[:5].tolist() == [50.0, 25.0, 30.0, 40.0, 80.0]
    assert b.iloc[5:].isna().all()
    cats = cov.bmi_category(pd.Series([18.4, 18.5, 24.99, 25.0, 30.0, 40.0, np.nan]))
    assert cats.tolist() == ["< 18.5", "18.5-24.9", "18.5-24.9", "25-29.9", "30-39.9", ">= 40",
                             cov.MISSING]


def test_bmi_is_missing_without_height_or_weight():
    b = cov.bmi(pd.Series([70.0, None]), pd.Series([None, 170.0]))
    assert b.isna().all()
    assert cov.bmi_category(b).eq(cov.MISSING).all()


def test_crrt_in_window_bounds_inclusive_and_requires_therapy(cohort):
    # intubations at 10, 5, 50, 3 h; window [-24 h, +4 h]
    crrt = pd.DataFrame({
        "encounter_block": ["1", "2", "3", "3", "4"],
        "recorded_dttm": [ts(10 - 24), ts(5 + 4.01), ts(50 + 4), ts(40), ts(2)],
        "crrt_mode_category": ["CVVHDF", "CVVH", None, None, None],
        "dialysate_flow_rate": [None, None, 2000.0, None, None],
    })
    assert cov.crrt_in_window(cohort, crrt, pd.Timedelta(hours=24),
                              pd.Timedelta(hours=4)).tolist() == [1, 0, 1, 0]


def test_crrt_in_window_needs_a_mode_or_flow_column(cohort):
    crrt = pd.DataFrame({"encounter_block": ["1"], "recorded_dttm": [ts(10)]})
    with pytest.raises(ValueError, match="neither"):
        cov.crrt_in_window(cohort, crrt, pd.Timedelta(hours=24), pd.Timedelta(hours=4))
