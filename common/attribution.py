"""Attribute ventilated care to the ICU that actually delivered it, by `adt.location_name`.

clif-dashboards attributes a whole IMV episode to the ICU where the episode *started*, labelled by
`location_type` (`prep.attribute_unit`). For a unit-level score used as a model exposure that is too
coarse twice over: a transferred patient's entire course is credited to the sending unit, and two
physically distinct units sharing a `location_type` collapse into one. This module replaces both:

  * **LTVV** — each hourly IMV scaffold row is credited to the ICU covering that hour. Scaffold rows
    are stamped at HH:59:59 and stand for the hour they end, so the hour's MIDPOINT (t - 30 min) is
    the instant tested. This mirrors upstream `metrics._in_segments`.
  * **SAT / SBT** — each vent-day is credited to the unit the patient began that ventilated day in,
    i.e. the unit covering the earliest IMV scaffold hour of that site-local calendar day within the
    episode. Clinically, SAT and SBT are morning, nursing-driven processes, so the unit holding the
    trial opportunity is the one the patient enters the day in. The pick is tie-free (there is
    exactly one earliest hour) and therefore order-invariant.
  * **Proning** — credited to the ICU covering the enrollment instant.

ADT locations are mutually exclusive in time, so "the ICU covering instant t" is unambiguous.
Records with no covering ICU stay, or a blank `location_name`, go to an explicit `UNATTRIBUTED`
bucket; they are reported and excluded from the panel, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Bucket for records that fall in no named ICU stay. Reported in the funnel, kept out of the panel.
UNATTRIBUTED = "__unattributed__"

#: Open ADT stays (null out_dttm) run to this sentinel rather than being dropped.
FAR_FUTURE = pd.Timestamp("2262-01-01", tz="UTC")


#: Canonical datetime resolution. Parquet round-trips give datetime64[us, UTC] while in-memory
#: pandas operations give [ns, UTC]; merge_asof refuses to join across the two. FAR_FUTURE
#: (2262-01-01) still fits inside the [ns] range, whose bound is 2262-04-11.
NS_UTC = "datetime64[ns, UTC]"


def as_ns(s: pd.Series) -> pd.Series:
    """UTC datetimes at nanosecond resolution, whatever the source resolution was."""
    return pd.to_datetime(s, utc=True).astype(NS_UTC)


def norm_block(s: pd.Series) -> pd.Series:
    """Canonical `encounter_block` key.

    Encounter stitching leaves the id as int64 in some staged frames and float64 in others (adt
    carries nulls from its left join onto the mapping, which forces float). A join on the raw
    column either raises on dtype or — worse — silently misses after a naive `astype(str)`, where
    1 becomes "1" on one side and "1.0" on the other, yielding a clean-looking zero-overlap join.

    Integral numerics are therefore routed through int64 before stringification, with nulls
    preserved as NA rather than being allowed to force the whole column down the raw-string path.
    Genuinely non-numeric ids are passed through as trimmed strings.
    """
    num = pd.to_numeric(s, errors="coerce")
    present = num.notna()
    # Every value must either coerce to a number or already be null, and the numbers must be whole.
    if (present | s.isna()).all() and present.any():
        vals = num[present].to_numpy()
        if np.allclose(vals, vals.round()):
            out = pd.Series(pd.NA, index=s.index, dtype="string")
            out.loc[present] = num[present].round().astype("int64").astype("string")
            return out
    return s.astype("string").str.strip()


def icu_intervals(adt: pd.DataFrame, *, unit_include=None, unit_collapse=None) -> pd.DataFrame:
    """ICU stays per encounter_block, as [encounter_block, in_dttm, out_dttm, unit_name, unit_type].

    `unit_include` restricts to those `location_name` values; `unit_collapse` renames them (for a
    unit that was split or renamed mid-period and should be pooled).
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
    return (icu[cols]
            .sort_values(["encounter_block", "in_dttm"], kind="stable")
            .reset_index(drop=True))


def unit_at(instants: pd.DataFrame, icu: pd.DataFrame, *,
            block_col: str = "encounter_block", time_col: str = "t") -> pd.DataFrame:
    """Label each (block, instant) with the ICU covering it.

    Returns `instants` with `unit_name` / `unit_type` added, `UNATTRIBUTED` where no ICU stay
    covers the instant. Vectorized: a backward as-of join finds the latest stay beginning at or
    before the instant, then the stay's end is checked. Correct because ADT locations do not
    overlap in time.
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
    merged = pd.merge_asof(
        left, right,
        left_on=time_col, right_on="in_dttm",
        left_by=block_col, right_by="encounter_block",
        direction="backward", allow_exact_matches=True,
    )
    covered = merged["out_dttm"].notna() & (merged[time_col] < merged["out_dttm"])
    merged["unit_name"] = merged["unit_name"].where(covered, UNATTRIBUTED)
    merged["unit_type"] = merged["unit_type"].where(covered, pd.NA)
    merged = merged.sort_values("_row", kind="stable").set_index("_row")
    out["unit_name"] = merged["unit_name"].to_numpy()
    out["unit_type"] = merged["unit_type"].to_numpy()
    return out


def hour_midpoints(recorded_dttm: pd.Series) -> pd.Series:
    """The instant an hourly scaffold row stands for.

    clifpy stamps scaffold rows at HH:59:59 to mean "the hour ending here", so the hour's midpoint
    is 30 minutes earlier. Upstream `metrics._in_segments` tests the same instant.
    """
    return as_ns(recorded_dttm) - pd.Timedelta(minutes=30)


def period_label(ts, tz: str, period: str = "quarter") -> pd.Series:
    """Site-local calendar period label: '2023-Q1' (quarter) or '2023-01' (month)."""
    local = as_ns(ts).dt.tz_convert(tz)
    if period == "quarter":
        return local.dt.year.astype("string") + "-Q" + local.dt.quarter.astype("string")
    if period == "month":
        return local.dt.strftime("%Y-%m")
    raise ValueError(f"period must be 'quarter' or 'month', got {period!r}")


def vent_day_units(scaffold_hours: pd.DataFrame, icu: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Per (encounter_block, episode_id, day): the unit the patient began that ventilated day in.

    `scaffold_hours` must hold one row per hourly IMV scaffold row inside an episode, with columns
    [encounter_block, episode_id, recorded_dttm]. The unit is taken from the EARLIEST such hour of
    the site-local calendar day, so a patient transferred mid-day is credited to the unit that held
    the morning trial opportunity.
    """
    if scaffold_hours.empty:
        return pd.DataFrame(columns=["encounter_block", "episode_id", "day",
                                     "unit_name", "unit_type", "imv_hours"])
    sc = scaffold_hours.copy()
    sc["recorded_dttm"] = as_ns(sc["recorded_dttm"])
    sc["day"] = sc["recorded_dttm"].dt.tz_convert(tz).dt.date
    first = (sc.sort_values(["encounter_block", "episode_id", "day", "recorded_dttm"], kind="stable")
               .groupby(["encounter_block", "episode_id", "day"], as_index=False)
               .agg(first_hour=("recorded_dttm", "first"), imv_hours=("recorded_dttm", "size")))
    first["t"] = hour_midpoints(first["first_hour"])
    labelled = unit_at(first, icu)
    return labelled[["encounter_block", "episode_id", "day", "unit_name", "unit_type", "imv_hours"]]
