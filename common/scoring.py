"""Pooled rates, the equal-weight composite, and leave-one-out exposure.

Everything is derived from one canonical long table, the *contribution* table:

    metric | unit_name | period | encounter_block | patient_id | num | den

One row per (metric, unit-period, patient-contribution). The panel is its sum; leave-one-out is
the panel minus one patient's own rows. Deriving both from the same table means a patient can never
be counted in one and missed in the other.

Rates are POOLED — `sum(num) / sum(den)` — never a mean of per-patient percentages, matching the
upstream's aggregation (`clif_dash/server/aggregate.py`). Scores are a weighted mean of pooled
rates times `scale` (100).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRIB_COLUMNS = ["metric", "unit_name", "period", "encounter_block", "patient_id", "num", "den"]

#: A score needs every one of its weighted components to have a non-zero denominator in that cell.
#: Renormalizing over whatever happens to be present would make the score mean different things in
#: different cells, which is exactly what a regression exposure must not do. Cells failing this are
#: null and counted, never silently partial.
REQUIRE_ALL_COMPONENTS = True


def pooled_rate(num, den):
    """sum(num)/sum(den) with a zero denominator giving NaN rather than 0 or a divide warning."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    return np.divide(num, den, out=np.full_like(den, np.nan, dtype="float64"), where=den > 0)


def cell_totals(contrib: pd.DataFrame, group=("unit_name", "period")) -> pd.DataFrame:
    """Wide num/den/rate per cell per metric: columns `<metric>_num`, `<metric>_den`, `<metric>_rate`."""
    group = list(group)
    agg = (contrib.groupby(group + ["metric"], observed=True, dropna=False)[["num", "den"]]
           .sum().reset_index())
    wide = agg.pivot_table(index=group, columns="metric", values=["num", "den"],
                           fill_value=0, observed=True)
    wide.columns = [f"{metric}_{kind}" for kind, metric in wide.columns]
    wide = wide.reset_index()
    for metric in sorted(contrib["metric"].unique()):
        n, d = f"{metric}_num", f"{metric}_den"
        if n in wide.columns:
            wide[f"{metric}_rate"] = pooled_rate(wide[n], wide[d])
    return wide


def composite(frame: pd.DataFrame, weights: dict[str, float], *, scale: float = 100.0,
              require_all: bool = REQUIRE_ALL_COMPONENTS) -> tuple[pd.Series, pd.Series]:
    """Weighted mean of `<metric>_rate` columns, scaled.

    Returns (score, n_components). `weights` must already be normalized to sum to 1; iterate its
    keys so a metric added to the config is never silently ignored.
    """
    missing = [m for m in weights if f"{m}_rate" not in frame.columns]
    if missing:
        raise KeyError(f"composite needs rate columns for {missing}; "
                       f"have {[c for c in frame.columns if c.endswith('_rate')]}")
    rates = frame[[f"{m}_rate" for m in weights]]
    w = np.array([weights[m] for m in weights], dtype="float64")
    present = rates.notna().to_numpy()
    n_components = pd.Series(present.sum(axis=1), index=frame.index, dtype="int64")

    values = rates.to_numpy(dtype="float64")
    if require_all:
        score = np.where(present.all(axis=1), np.nansum(values * w, axis=1), np.nan)
    else:
        wsum = (present * w).sum(axis=1)
        score = np.divide(np.nansum(np.where(present, values, 0.0) * w, axis=1), wsum,
                          out=np.full(len(frame), np.nan), where=wsum > 0)
    return pd.Series(score * scale, index=frame.index), n_components


def flag_low_denominator(frame: pd.DataFrame, floors: dict[str, float],
                         scores: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Add `<metric>_low` per configured floor, plus a per-score and an overall roll-up.

    Cells are FLAGGED, never dropped: a thin cell is information, and dropping it would bias the
    panel toward high-volume units. Iterates the config's own keys.

    `scores` maps score id -> its weight map, giving `<score>_low` flagged only by the floors of
    the metrics that score actually uses. The global `low_denominator` ORs every floor, so for a
    score that excludes a rare metric it is the wrong flag to filter on — at UChicago the proning
    floor alone marks ~71% of cells, none of which affects score_core.
    """
    out = frame.copy()
    flags = []
    for metric, floor in floors.items():
        den = f"{metric}_den"
        if den not in out.columns:
            raise KeyError(f"min_denominator names '{metric}' but there is no {den} column — "
                           f"the floor would be a silent no-op")
        col = f"{metric}_low"
        out[col] = out[den] < floor
        flags.append(col)
    for score_id, weights in (scores or {}).items():
        cols = [f"{m}_low" for m in weights if f"{m}_low" in out.columns]
        out[f"{score_id}_low"] = out[cols].any(axis=1) if cols else False
    out["low_denominator"] = out[flags].any(axis=1) if flags else False
    return out


def leave_one_out(contrib: pd.DataFrame, weights: dict[str, float], *,
                  group=("unit_name", "period"), scale: float = 100.0,
                  require_all: bool = REQUIRE_ALL_COMPONENTS) -> pd.DataFrame:
    """Per (patient, cell) score with that patient's own contributions removed.

    A unit's measured quality partly reflects the care given to the very patient whose outcome is
    being modelled, so regressing an outcome on the unadjusted cell rate builds in a mechanical
    association. Subtracting the patient's own numerator and denominator removes it.

    Cells where removing the patient leaves no denominator yield NaN for that metric, and a NaN
    score under `require_all`.
    """
    group = list(group)
    totals = (contrib.groupby(group + ["metric"], observed=True, dropna=False)[["num", "den"]]
              .sum().rename(columns={"num": "cell_num", "den": "cell_den"}).reset_index())
    own_agg = (contrib.groupby(group + ["metric", "patient_id"], observed=True, dropna=False)
               [["num", "den"]].sum()
               .rename(columns={"num": "own_num", "den": "own_den"}).reset_index())

    # A patient who contributed NOTHING to a metric must still get that metric's LOO rate — which
    # is simply the cell total, since removing zero changes nothing. Without the explicit grid the
    # pivot leaves NaN there, and under require_all the whole score silently becomes NaN for every
    # patient outside (say) the proning cohort.
    pairs = contrib[group + ["patient_id"]].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"metric": list(weights)}), how="cross")
    own = grid.merge(own_agg, on=group + ["metric", "patient_id"], how="left")
    own[["own_num", "own_den"]] = own[["own_num", "own_den"]].fillna(0.0)

    merged = own.merge(totals, on=group + ["metric"], how="left")
    merged["loo_num"] = merged["cell_num"] - merged["own_num"]
    merged["loo_den"] = merged["cell_den"] - merged["own_den"]
    merged["loo_rate"] = pooled_rate(merged["loo_num"], merged["loo_den"])

    wide = merged.pivot_table(index=group + ["patient_id"], columns="metric",
                              values=["loo_num", "loo_den", "loo_rate"], observed=True,
                              dropna=False)
    wide.columns = [f"{metric}_{kind.replace('loo_', '')}" for kind, metric in wide.columns]
    wide = wide.reset_index()
    for m in weights:
        for kind in ("num", "den", "rate"):
            if f"{m}_{kind}" not in wide.columns:
                wide[f"{m}_{kind}"] = np.nan
    score, ncomp = composite(wide, weights, scale=scale, require_all=require_all)
    wide["loo_score"] = score
    wide["loo_n_components"] = ncomp
    return wide
