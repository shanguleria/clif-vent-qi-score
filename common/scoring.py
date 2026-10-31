"""Pooled rates and the equal-weight composite.

Everything derives from one long *contribution* table, one row per (metric, cell, patient):

    metric | unit_name | period | encounter_block | patient_id | num | den

The panel is its sum at whichever grain the caller groups by; the per-patient rows are what let
stage 4 assign each patient to the cell where they accrued the most ventilated time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRIB_COLUMNS = ["metric", "unit_name", "period", "encounter_block", "patient_id", "num", "den"]

# Renormalizing over whatever components happen to be present would make the score mean different
# things in different cells, which a regression exposure must not do.
REQUIRE_ALL_COMPONENTS = True


def pooled_rate(num, den):
    """Zero denominator gives NaN, not 0."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    return np.divide(num, den, out=np.full_like(den, np.nan, dtype="float64"), where=den > 0)


def cell_totals(contrib: pd.DataFrame, group=("unit_name", "period")) -> pd.DataFrame:
    """Wide `<metric>_num` / `_den` / `_rate` per cell."""
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
    """Weighted mean of `<metric>_rate` columns, scaled. Returns (score, n_components)."""
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
    """Flag cells below each floor; never drop them, which would bias toward high-volume units.

    `<score>_low` uses only the floors of the metrics that score contains. The global
    `low_denominator` ORs every floor and is the wrong filter for a score excluding a rare metric.
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
