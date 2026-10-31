"""Attribute ventilated care to the ICU that delivered it, by `adt.location_name`.

Upstream credits a whole IMV episode to the ICU it started in, labelled by `location_type`
(`prep.attribute_unit`). Here each record is credited to the ICU covering its own instant:
an IMV hour to the ICU at that hour's midpoint, a vent-day to the ICU at its earliest IMV hour
(SAT and SBT are morning, nursing-driven), proning to the ICU at enrollment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNATTRIBUTED = "__unattributed__"

# Open ADT stays run to this rather than being dropped. Inside the [ns] bound of 2262-04-11.
FAR_FUTURE = pd.Timestamp("2262-01-01", tz="UTC")


def as_ns(s: pd.Series) -> pd.Series:
    """Parquet gives datetime64[us], in-memory pandas gives [ns]; merge_asof won't join across."""
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def norm_block(s: pd.Series) -> pd.Series:
    """Canonical `encounter_block` key.

    Staged frames carry the id as int64 or, where nulls forced a float, float64. A naive
    `astype(str)` then yields "1" against "1.0" and the join misses silently with no error, so
    integral numerics go through int64 first and nulls are preserved as NA rather than dragging
    the column down the raw-string path.
    """
    num = pd.to_numeric(s, errors="coerce")
    present = num.notna()
    if (present | s.isna()).all() and present.any():
        vals = num[present].to_numpy()
        if np.allclose(vals, vals.round()):
            out = pd.Series(pd.NA, index=s.index, dtype="string")
            out.loc[present] = num[present].round().astype("int64").astype("string")
            return out
    return s.astype("string").str.strip()


def icu_intervals(adt: pd.DataFrame, *, unit_include=None, unit_collapse=None) -> pd.DataFrame:
    """ICU stays as [encounter_block, in_dttm, out_dttm, unit_name, unit_type].

    `unit_collapse` renames units, for one split or renamed mid-study that should be pooled.
    """
    icu = adt[adt["location_category"].astype(str).str.lower().eq("icu")].copy()
    name = icu["location_name"].astype("string").str.strip()
    icu = icu[name.notna() & name.ne("")]
    icu["unit_name"] = name[icu.index]
    if unit_include:
        icu = icu[icu["unit_name"].isin(list(unit_include))]
    if unit_collapse:
        icu["unit_name"] = icu["unit_name"].replace(dict(unit_collapse))
    icu["unit_type"] = icu["location_type"].astype("string")
    icu["encounter_block"] = norm_block(icu["encounter_block"])
    icu = icu[icu["encounter_block"].notna()]
    icu["in_dttm"] = as_ns(icu["in_dttm"])
    icu["out_dttm"] = as_ns(icu["out_dttm"]).fillna(FAR_FUTURE)
    icu = icu[icu["out_dttm"] > icu["in_dttm"]]
    cols = ["encounter_block", "in_dttm", "out_dttm", "unit_name", "unit_type"]
    return (icu[cols].sort_values(["encounter_block", "in_dttm"], kind="stable")
            .reset_index(drop=True))


def unit_at(instants: pd.DataFrame, icu: pd.DataFrame, *,
            block_col: str = "encounter_block", time_col: str = "t") -> pd.DataFrame:
    """Label each (block, instant) with the ICU covering it, else UNATTRIBUTED.

    A backward as-of join takes the latest stay starting at or before the instant and then checks
    its end, which is correct only because ADT locations do not overlap in time.
    """
    out = instants.copy()
    if out.empty:
        out["unit_name"] = pd.Series(dtype="string")
        out["unit_type"] = pd.Series(dtype="string")
        return out
    if icu.empty:
        out["unit_name"] = UNATTRIBUTED
        out["unit_type"] = pd.NA
        return out

    left = out.reset_index(names="_row")
    left[block_col] = norm_block(left[block_col])
    left[time_col] = as_ns(left[time_col])
    left = left.sort_values([time_col, "_row"], kind="stable")
    right = icu.copy()
    right["encounter_block"] = norm_block(right["encounter_block"])
    right["in_dttm"] = as_ns(right["in_dttm"])
    right["out_dttm"] = as_ns(right["out_dttm"])
    right = right.sort_values("in_dttm", kind="stable")

    merged = pd.merge_asof(left, right, left_on=time_col, right_on="in_dttm",
                           left_by=block_col, right_by="encounter_block",
                           direction="backward", allow_exact_matches=True)
    covered = merged["out_dttm"].notna() & (merged[time_col] < merged["out_dttm"])
    merged["unit_name"] = merged["unit_name"].where(covered, UNATTRIBUTED)
    merged["unit_type"] = merged["unit_type"].where(covered, pd.NA)
    merged = merged.sort_values("_row", kind="stable").set_index("_row")
    out["unit_name"] = merged["unit_name"].to_numpy()
    out["unit_type"] = merged["unit_type"].to_numpy()
    return out


def hour_midpoints(recorded_dttm: pd.Series) -> pd.Series:
    """Scaffold rows are stamped at HH:59:59 for the hour they end, so test 30 min earlier."""
    return as_ns(recorded_dttm) - pd.Timedelta(minutes=30)


def period_label(ts, tz: str, period: str = "quarter") -> pd.Series:
    local = as_ns(ts).dt.tz_convert(tz)
    if period == "quarter":
        return local.dt.year.astype("string") + "-Q" + local.dt.quarter.astype("string")
    if period == "month":
        return local.dt.strftime("%Y-%m")
    raise ValueError(f"period must be 'quarter' or 'month', got {period!r}")
