"""Stage 7's extended export must not lose a column silently. Synthetic frames, no CLIF data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stage07():
    """07_* is not a legal module name, so load it by path."""
    spec = importlib.util.spec_from_file_location(
        "stage07", ROOT / "code" / "07_analysis_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demographics_are_required():
    mod = _stage07()
    assert set(mod.REQUIRED_EXTENDED) >= {"race_category", "ethnicity_category", "sex_category"}


def test_require_columns_passes_when_all_present():
    mod = _stage07()
    cohort = pd.DataFrame({c: ["x"] for c in mod.REQUIRED_EXTENDED})
    mod.require_columns(cohort, mod.REQUIRED_EXTENDED, "demographics")


@pytest.mark.parametrize("absent", ["race_category", "ethnicity_category", "sex_category"])
def test_require_columns_fires_on_each_missing_demographic(absent):
    mod = _stage07()
    cohort = pd.DataFrame({c: ["x"] for c in mod.REQUIRED_EXTENDED if c != absent})
    with pytest.raises(AssertionError, match=absent):
        mod.require_columns(cohort, mod.REQUIRED_EXTENDED, "demographics")


def test_poa_mask_handles_nullable_missing_flags():
    mod = _stage07()
    flag = pd.Series([1, 0, None, 1], dtype="Int64")
    assert mod.poa_mask(flag).tolist() == [True, False, False, True]
    df = pd.DataFrame({"x": list("abcd")})
    assert df[mod.poa_mask(flag)]["x"].tolist() == ["a", "d"]


@pytest.mark.parametrize("flag", [pd.Series([1.0, float("nan"), 0.0]), pd.Series(["1", None, "0"])])
def test_poa_mask_handles_float_and_string_flags(flag):
    assert _stage07().poa_mask(flag).tolist() == [True, False, False]


def test_nullable_eq_alone_is_the_bug_poa_mask_avoids():
    flag = pd.Series([1, None], dtype="Int64")
    with pytest.raises(ValueError, match="NA"):
        pd.DataFrame({"x": [1, 2]})[flag.eq(1).to_numpy()]


def _ts(h):
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)


def test_first_episode_length_comes_from_the_first_episode_of_the_first_encounter():
    mod = _stage07()
    episodes = pd.DataFrame({"encounter_block": [1, 1, 2],
                             "start": [_ts(50), _ts(10), _ts(500)],
                             "end":   [_ts(60), _ts(40), _ts(600)]})
    encounters = pd.DataFrame({"encounter_block": [1, 2], "patient_id": ["p", "p"]})
    out = mod.first_encounter_per_patient(episodes, encounters)
    assert len(out) == 1
    assert out.loc[0, "intubation_dttm"] == _ts(10)
    assert out.loc[0, "first_episode_hours"] == 30
    assert str(out["intubation_dttm"].dtype).startswith("datetime64")
