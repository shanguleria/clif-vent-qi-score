"""Candidate case-mix covariates anchored at intubation, and small-cell-safe tabulation.

Pure functions over frames; no CLIF I/O. `cohort` is one row per patient with at least
`encounter_block`, `patient_id` and `intubation_dttm`; results align to `cohort.index`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import attribution as attr

MISSING = "Missing"
NO_ADT_ROW = "No ADT row"
NONE_RECORDED = "None recorded"

PROCEDURAL_HOURS = 24
HOURS_BINS = [-np.inf, 0, 24, 72, 168, np.inf]
HOURS_LABELS = ["<= 0 h", "0-24 h", "24-72 h", "72 h-7 d", "> 7 d"]

LOCATION_KEPT = ("ed", "icu", "procedural")
LOCATION_UNKNOWN = "unknown"
LOCATION_OTHER = "ward_other"

# mCIDE code_status_category, lowercased. "Other" does not say whether resuscitation is limited,
# so it is treated as not limited (researcher decision, 2026-11-03). Any category outside both
# sets still stops stage 7.
LIMITED_CODE = {"dnr", "dnar", "udnr", "dnr/dni", "dnar/dni", "and"}
NOT_LIMITED_CODE = {"full", "presume full", "other", NONE_RECORDED.lower()}


def attach_block(df: pd.DataFrame, mapping: pd.DataFrame, blocks) -> pd.DataFrame:
    """Key a hospitalization-level table by `encounter_block`, keeping only `blocks`. `mapping`
    must already carry string hospitalization ids and normalized blocks."""
    d = df.copy()
    d["hospitalization_id"] = d["hospitalization_id"].astype(str)
    d = d.merge(mapping, on="hospitalization_id", how="inner")
    return d[d["encounter_block"].isin(blocks)]


def hours_category(intubation: pd.Series, admission: pd.Series) -> pd.Series:
    h = (attr.as_ns(intubation) - attr.as_ns(admission)).dt.total_seconds() / 3600
    return pd.cut(h, HOURS_BINS, labels=HOURS_LABELS).astype(object)


def collapse_location(location: pd.Series) -> pd.Series:
    s = location.astype(str)
    out = np.where(s.isin(LOCATION_KEPT), s,
                   np.where(s.isin([NO_ADT_ROW, MISSING]), LOCATION_UNKNOWN, LOCATION_OTHER))
    return pd.Series(out, index=location.index, name=location.name)


def limited_code_status(status: pd.Series) -> pd.Series:
    s = status.astype(str).str.strip().str.lower()
    unknown = sorted(set(s) - LIMITED_CODE - NOT_LIMITED_CODE)
    if unknown:
        raise ValueError(f"code_status_category values in neither the limited nor the "
                         f"not-limited set: {unknown}; classify them in common/covariates.py")
    return s.isin(LIMITED_CODE).astype(int)


def admission_type(cohort: pd.DataFrame, hosp: pd.DataFrame) -> pd.Series:
    """`admission_type_category` of the block's first hospitalization, lowercased; a block
    without one is `unknown`, never dropped."""
    h = hosp.assign(admission_dttm=attr.as_ns(hosp["admission_dttm"]))
    first = (h.sort_values(["encounter_block", "admission_dttm", "hospitalization_id"],
                           kind="stable").drop_duplicates("encounter_block")
             .set_index("encounter_block")["admission_type_category"])
    out = cohort["encounter_block"].map(first).astype("string").str.lower()
    return out.fillna("unknown").astype(object).rename("admission_type")


def _anchored(cohort: pd.DataFrame, other: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    c = pd.DataFrame({"_row": np.arange(len(cohort)),
                      key: cohort[key].reset_index(drop=True),
                      "t": attr.as_ns(cohort["intubation_dttm"]).reset_index(drop=True)})
    return c.merge(other[[key] + cols], on=key, how="inner")


def _adt(adt: pd.DataFrame) -> pd.DataFrame:
    a = adt[["encounter_block", "in_dttm", "out_dttm", "location_category"]].copy()
    a["in_dttm"] = attr.as_ns(a["in_dttm"])
    a["out_dttm"] = attr.as_ns(a["out_dttm"])
    a["location_category"] = a["location_category"].astype("string").str.lower()
    return a


def location_at_intubation(cohort: pd.DataFrame, adt: pd.DataFrame) -> pd.Series:
    """`location_category` of the ADT row spanning intubation, on half-open [in, out) intervals,
    so an intubation at the instant of a transfer belongs to the destination. Overlapping rows
    resolve to the latest-starting one. A time no row covers is reported, never carried forward."""
    m = _anchored(cohort, _adt(adt), "encounter_block", ["in_dttm", "out_dttm", "location_category"])
    m = m[(m["in_dttm"] <= m["t"]) & (m["out_dttm"].isna() | (m["t"] < m["out_dttm"]))]
    m = (m.sort_values(["_row", "in_dttm", "location_category"], kind="stable")
         .drop_duplicates("_row", keep="last"))
    out = np.full(len(cohort), NO_ADT_ROW, dtype=object)
    out[m["_row"].to_numpy()] = m["location_category"].fillna(MISSING).to_numpy()
    return pd.Series(out, index=cohort.index, name="location_at_intubation")


def procedural_within(cohort: pd.DataFrame, adt: pd.DataFrame, hours: float) -> pd.Series:
    """Any `procedural` ADT stay overlapping the `hours` before intubation."""
    m = _anchored(cohort, _adt(adt), "encounter_block", ["in_dttm", "out_dttm", "location_category"])
    lo = m["t"] - pd.Timedelta(hours=hours)
    hit = (m["location_category"].eq("procedural").fillna(False)
           & (m["in_dttm"] <= m["t"]) & (m["out_dttm"].isna() | (m["out_dttm"] >= lo)))
    out = np.zeros(len(cohort), dtype=bool)
    out[m.loc[hit, "_row"].unique()] = True
    return pd.Series(out, index=cohort.index, name=f"procedural_within_{hours:g}h")


def code_status_before(cohort: pd.DataFrame, code_status: pd.DataFrame) -> pd.Series:
    """The last `code_status_category` starting during the current admission, at or before
    intubation. Later orders are excluded: a limitation placed after intubation sits on the path
    from care to death. Earlier admissions are excluded too: CLIF keys code_status by patient,
    so without the lower bound an order from a stay years before would still count.
    `cohort` needs `admission_dttm` as well as `intubation_dttm`."""
    cs = code_status[["patient_id", "start_dttm", "code_status_category"]].copy()
    cs["patient_id"] = cs["patient_id"].astype(str)
    cs["start_dttm"] = attr.as_ns(cs["start_dttm"])
    co = cohort.assign(patient_id=cohort["patient_id"].astype(str))
    m = _anchored(co, cs, "patient_id", ["start_dttm", "code_status_category"])
    m["admit"] = attr.as_ns(cohort["admission_dttm"]).array[m["_row"].to_numpy()]
    m = m[(m["start_dttm"] >= m["admit"]) & (m["start_dttm"] <= m["t"])]
    m = (m.sort_values(["_row", "start_dttm", "code_status_category"], kind="stable")
         .drop_duplicates("_row", keep="last"))
    out = np.full(len(cohort), NONE_RECORDED, dtype=object)
    out[m["_row"].to_numpy()] = m["code_status_category"].astype("string").fillna(MISSING).to_numpy()
    return pd.Series(out, index=cohort.index, name="code_status_before_intubation")


def tabulate(df: pd.DataFrame, var: str, by: str | None = None, outcome: str = "died") -> pd.DataFrame:
    """Long counts of `var`'s levels, overall or within each `by` group, with crude mortality."""
    keys = [by] if by else []
    x = df[[var, outcome] + keys].copy()
    x[var] = x[var].astype(object).where(x[var].notna(), MISSING).astype(str)
    g = (x.groupby(keys + [var], observed=True, sort=True)[outcome]
         .agg(n="size", deaths="sum").reset_index())
    g["n_total"] = g.groupby(keys)["n"].transform("sum") if keys else g["n"].sum()
    g["pct"] = 100 * g["n"] / g["n_total"]
    g["mortality"] = 100 * g["deaths"] / g["n"]
    g = g.rename(columns={var: "level"})
    g["group"] = g.pop(by).astype(str) if by else "ALL"
    g.insert(0, "by", by or "all")
    g.insert(0, "variable", var)
    return g[["variable", "by", "group", "level", "n", "n_total", "pct", "deaths", "mortality"]]


def _with_complement(small: pd.Series, values: pd.Series, groups: pd.Series) -> pd.Series:
    """A lone suppressed cell is recoverable from its group's total, so the smallest other cell
    in that group is suppressed with it; a blanked zero protects as well as any other."""
    small = small.copy()
    for idx in groups.groupby(groups).groups.values():
        if small.loc[idx].sum() == 1:
            rest = values.loc[idx][~small.loc[idx]]
            if len(rest):
                small.loc[rest.idxmin()] = True
    return small


def suppress(tab: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Blank counts in [1, threshold) together with what would let them be back-calculated:
    a complementary cell per group, mortality wherever deaths or survivors are small, and the
    whole group when its own total is small."""
    t = tab.reset_index(drop=True).copy()
    groups = t["variable"] + "\x1f" + t["by"] + "\x1f" + t["group"]
    small_group = t["n_total"].between(1, threshold - 1)
    small_n = _with_complement(t["n"].between(1, threshold - 1), t["n"], groups) | small_group
    survivors = t["n"] - t["deaths"]
    small_d = (t["deaths"].between(1, threshold - 1) | survivors.between(1, threshold - 1)) | small_n
    small_d = _with_complement(small_d, t["deaths"], groups)
    for c in ("n", "n_total", "deaths"):
        t[c] = t[c].astype("Int64")
    t.loc[small_group, "n_total"] = pd.NA
    t.loc[small_n, "n"] = pd.NA
    t.loc[small_n, "pct"] = np.nan
    t.loc[small_d, "deaths"] = pd.NA
    t.loc[small_d, "mortality"] = np.nan
    t["suppressed"] = small_n | small_d
    return t
