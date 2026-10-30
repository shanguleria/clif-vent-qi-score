"""Unit tests for pooled rates, the composite and attribution. Synthetic frames, no CLIF data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import attribution as attr          # noqa: E402
from common import scoring as S                 # noqa: E402

W3 = {"ltvv": 1 / 3, "sat": 1 / 3, "sbt": 1 / 3}


def contrib(rows):
    return pd.DataFrame(rows, columns=S.CONTRIB_COLUMNS)


# --------------------------------------------------------------------------- #
# pooled_rate
# --------------------------------------------------------------------------- #
def test_pooled_rate_is_volume_weighted_not_a_mean_of_percentages():
    # 1/1 and 0/99: pooled = 1/100, whereas a mean of per-patient rates would be 0.5
    assert S.pooled_rate([1 + 0], [1 + 99])[0] == pytest.approx(0.01)


def test_pooled_rate_zero_denominator_is_nan_not_zero():
    out = S.pooled_rate([0, 3], [0, 4])
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# cell_totals
# --------------------------------------------------------------------------- #
def test_cell_totals_pools_across_patients_within_a_cell():
    c = contrib([
        ("ltvv", "MICU", "2023-Q1", "b1", "p1", 6, 10),
        ("ltvv", "MICU", "2023-Q1", "b2", "p2", 2, 10),
        ("ltvv", "SICU", "2023-Q1", "b3", "p3", 1, 4),
    ])
    t = S.cell_totals(c).set_index(["unit_name", "period"])
    assert t.loc[("MICU", "2023-Q1"), "ltvv_den"] == 20
    assert t.loc[("MICU", "2023-Q1"), "ltvv_rate"] == pytest.approx(0.4)
    assert t.loc[("SICU", "2023-Q1"), "ltvv_rate"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# composite
# --------------------------------------------------------------------------- #
def test_composite_equal_weight_matches_hand_calculation():
    frame = pd.DataFrame({"ltvv_rate": [0.72], "sat_rate": [0.55], "sbt_rate": [0.61]})
    score, ncomp = S.composite(frame, W3)
    assert score.iloc[0] == pytest.approx(100 * (0.72 + 0.55 + 0.61) / 3)
    assert ncomp.iloc[0] == 3


def test_composite_nulls_the_cell_when_a_component_is_missing():
    frame = pd.DataFrame({"ltvv_rate": [0.8], "sat_rate": [np.nan], "sbt_rate": [0.6]})
    score, ncomp = S.composite(frame, W3)
    assert np.isnan(score.iloc[0]), "a partial score would mean something different per cell"
    assert ncomp.iloc[0] == 2


def test_composite_can_renormalize_when_explicitly_asked():
    frame = pd.DataFrame({"ltvv_rate": [0.8], "sat_rate": [np.nan], "sbt_rate": [0.6]})
    score, _ = S.composite(frame, W3, require_all=False)
    assert score.iloc[0] == pytest.approx(100 * (0.8 + 0.6) / 2)


def test_composite_raises_on_a_weight_with_no_rate_column():
    frame = pd.DataFrame({"ltvv_rate": [0.8]})
    with pytest.raises(KeyError, match="prone_12h"):
        S.composite(frame, {"ltvv": 0.5, "prone_12h": 0.5})


def test_composite_weights_need_not_be_equal():
    frame = pd.DataFrame({"a_rate": [1.0], "b_rate": [0.0]})
    score, _ = S.composite(frame, {"a": 0.75, "b": 0.25})
    assert score.iloc[0] == pytest.approx(75.0)


# --------------------------------------------------------------------------- #
# low denominator flags
# --------------------------------------------------------------------------- #
def test_flag_low_denominator_flags_but_does_not_drop():
    frame = pd.DataFrame({"unit_name": ["A", "B"], "ltvv_den": [5, 500], "sat_den": [50, 50]})
    out = S.flag_low_denominator(frame, {"ltvv": 100, "sat": 20})
    assert len(out) == 2, "thin cells must be kept"
    assert list(out["ltvv_low"]) == [True, False]
    assert list(out["low_denominator"]) == [True, False]


def test_flag_low_denominator_rejects_an_unclaimed_config_key():
    frame = pd.DataFrame({"ltvv_den": [100]})
    with pytest.raises(KeyError, match="silent no-op"):
        S.flag_low_denominator(frame, {"prone_12h": 5})


# --------------------------------------------------------------------------- #
# attribution
# --------------------------------------------------------------------------- #
def _icu():
    T = lambda s: pd.Timestamp(s, tz="UTC")
    return attr.icu_intervals(pd.DataFrame([
        ("b1", "icu", "MICU", "medical_icu", T("2023-01-01 00:00"), T("2023-01-04 00:00")),
        ("b1", "ward", "W5", None, T("2023-01-04 00:00"), T("2023-01-05 00:00")),
        ("b1", "icu", "SICU", "surgical_icu", T("2023-01-05 00:00"), None),
    ], columns=["encounter_block", "location_category", "location_name",
                "location_type", "in_dttm", "out_dttm"]))


@pytest.mark.parametrize("when,expect", [
    ("2023-01-01 12:00", "MICU"),
    ("2023-01-04 12:00", attr.UNATTRIBUTED),      # on the ward
    ("2023-01-06 12:00", "SICU"),                 # open-ended stay
    ("2023-01-04 00:00", attr.UNATTRIBUTED),      # boundary is half-open [in, out)
])
def test_unit_at_covers_the_right_interval(when, expect):
    got = attr.unit_at(pd.DataFrame({"encounter_block": ["b1"],
                                     "t": [pd.Timestamp(when, tz="UTC")]}), _icu())
    assert got["unit_name"].iloc[0] == expect


def test_unit_at_is_row_order_invariant():
    times = ["2023-01-06 12:00", "2023-01-01 12:00", "2023-01-04 12:00"]
    inst = pd.DataFrame({"encounter_block": ["b1"] * 3,
                         "t": [pd.Timestamp(t, tz="UTC") for t in times]})
    forward = list(attr.unit_at(inst, _icu())["unit_name"])
    backward = list(attr.unit_at(inst.iloc[::-1].reset_index(drop=True), _icu())["unit_name"])
    assert forward == backward[::-1]


def test_norm_block_unifies_int_and_null_bearing_float_keys():
    # The documented trap: a float column with a null must not fall back to "1.0" vs "1".
    f = pd.Series([1.0, 2.0, np.nan])
    i = pd.Series([1, 2], dtype="int64")
    assert set(attr.norm_block(f).dropna()) == set(attr.norm_block(i))


def test_hour_midpoint_lands_inside_the_hour_the_scaffold_row_stands_for():
    t = pd.Series([pd.Timestamp("2023-01-01 13:59:59", tz="UTC")])
    assert attr.hour_midpoints(t).iloc[0] == pd.Timestamp("2023-01-01 13:29:59", tz="UTC")


def test_period_label_uses_site_local_calendar_not_utc():
    # 2023-03-31 23:00 UTC is still Q1 in Chicago (18:00 local); 2023-04-01 05:00 UTC is Q2 (00:00).
    ts = pd.Series([pd.Timestamp("2023-03-31 23:00", tz="UTC"),
                    pd.Timestamp("2023-04-01 05:00", tz="UTC")])
    assert list(attr.period_label(ts, "America/Chicago", "quarter")) == ["2023-Q1", "2023-Q2"]


def test_period_label_month_grain():
    ts = pd.Series([pd.Timestamp("2023-07-15 12:00", tz="UTC")])
    assert list(attr.period_label(ts, "America/Chicago", "month")) == ["2023-07"]


def test_period_label_rejects_an_unknown_grain():
    with pytest.raises(ValueError, match="quarter"):
        attr.period_label(pd.Series([pd.Timestamp("2023-01-01", tz="UTC")]), "UTC", "fortnight")

def test_flag_low_denominator_scopes_flags_to_each_scores_own_metrics():
    frame = pd.DataFrame({"ltvv_den": [500], "sat_den": [50], "sbt_den": [50], "prone_12h_den": [1]})
    out = S.flag_low_denominator(
        frame, {"ltvv": 100, "sat": 20, "sbt": 20, "prone_12h": 5},
        scores={"score_core": {"ltvv": 1 / 3, "sat": 1 / 3, "sbt": 1 / 3},
                "score_full": {"ltvv": 0.25, "sat": 0.25, "sbt": 0.25, "prone_12h": 0.25}})
    assert not out["score_core_low"].iloc[0], "the proning floor must not taint a score without it"
    assert out["score_full_low"].iloc[0]
    assert out["low_denominator"].iloc[0], "the global roll-up still ORs every floor"


def test_a_metric_with_no_records_degrades_instead_of_raising():
    """A site with no `position` table cannot observe proning; that must not crash the panel.

    Mirrors the zero-column backfill in code/03_build_panel.py: scores that include the absent
    metric go null, scores that exclude it still compute.
    """
    c = contrib([
        ("ltvv", "MICU", "2023-Q1", "b1", "p1", 8, 10),
        ("sat", "MICU", "2023-Q1", "b1", "p1", 1, 2),
        ("sbt", "MICU", "2023-Q1", "b1", "p1", 1, 2),
    ])
    panel = S.cell_totals(c)
    for m in ("prone_12h",):
        panel[f"{m}_num"], panel[f"{m}_den"], panel[f"{m}_rate"] = 0, 0, np.nan
    W4 = {"ltvv": 0.25, "sat": 0.25, "sbt": 0.25, "prone_12h": 0.25}
    panel = S.flag_low_denominator(panel, {"ltvv": 100, "sat": 20, "sbt": 20, "prone_12h": 5},
                                   scores={"score_core": W3, "score_full": W4})
    core, _ = S.composite(panel, W3)
    full, _ = S.composite(panel, W4)
    assert core.iloc[0] == pytest.approx(100 * (0.8 + 0.5 + 0.5) / 3)
    assert np.isnan(full.iloc[0])


# --------------------------------------------------------------------------- #
# grain
# --------------------------------------------------------------------------- #
def test_pooling_the_window_is_not_the_mean_of_period_rates():
    """The unit score re-divides summed counts; it is NOT the average of the quarterly rates.

    This is what stops a near-empty quarter from pulling a unit's score around, which matters at
    UChicago where one unit came online mid-window with 11 IMV hours across five quarters.
    """
    c = contrib([
        # a huge quarter at 90%, and a 1-hour quarter at 0%
        ("ltvv", "MICU", "2023-Q1", "b1", "p1", 900, 1000),
        ("ltvv", "MICU", "2023-Q2", "b2", "p2", 0, 1),
    ])
    per_period = S.cell_totals(c, ("unit_name", "period"))["ltvv_rate"]
    pooled = S.cell_totals(c, ("unit_name",))["ltvv_rate"].iloc[0]
    assert sorted(per_period.round(4)) == [0.0, 0.9]
    assert pooled == pytest.approx(900 / 1001)              # 0.8991
    assert pooled != pytest.approx(per_period.mean())        # the mean would be 0.45
    assert abs(pooled - per_period.mean()) > 0.4


def test_unit_grain_collapses_periods_but_keeps_units_separate():
    c = contrib([
        ("ltvv", "MICU", "2023-Q1", "b1", "p1", 8, 10),
        ("ltvv", "MICU", "2023-Q2", "b2", "p2", 2, 10),
        ("ltvv", "SICU", "2023-Q1", "b3", "p3", 1, 4),
    ])
    t = S.cell_totals(c, ("unit_name",)).set_index("unit_name")
    assert len(t) == 2
    assert t.loc["MICU", "ltvv_den"] == 20 and t.loc["MICU", "ltvv_rate"] == pytest.approx(0.5)
    assert t.loc["SICU", "ltvv_den"] == 4
