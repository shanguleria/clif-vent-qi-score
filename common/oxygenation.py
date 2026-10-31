"""Oxygenation for the SOFA respiratory component, computed from the waterfall.

Both SOFA calculators load `respiratory_support` straight from the CLIF directory, where the table
is EVENT-BASED: `fio2_set` appears only on rows where FiO2 was charted, so many IMV rows carry a
null not because the ventilator had no setting but because nothing changed and nothing was
written. Pairing an oxygenation measurement against those raw rows fails far more often than the
clinical reality warrants.

The waterfall already solves this: it forward-fills `fio2_set` within mode blocks and lays down an
hourly scaffold, so every ventilated hour carries the FiO2 then in force. This module pairs PaO2
and SpO2 against that timeline instead.

The second half of the problem is real and not fixable: the window opens 24 h BEFORE intubation,
when there is no ventilator and FiO2 is genuinely absent. SpO2 is recorded throughout, so an S/F
fallback covers the pre-intubation deterioration that P/F cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import attribution as attr

#: How far back an FiO2 may be carried to pair with an oxygenation measurement.
PAIR_TOLERANCE = pd.Timedelta(hours=4)

#: SpO2 above this is uninformative for S/F — the haemoglobin dissociation curve is flat, so a
#: reading of 100% tells you nothing about how much reserve is left. Rice 2007 restricts to <= 97.
SPO2_MAX_FOR_SF = 97.0

#: Devices that satisfy the "with respiratory support" qualifier on SOFA respiratory 3 and 4.
SUPPORT_DEVICES = {"imv", "nippv", "cpap"}

#: S/F cutpoints equivalent to the P/F cutpoints, via Rice 2007: SF = 64 + 0.84 x PF.
#: P/F 400 -> 400, 300 -> 316, 200 -> 232, 100 -> 148.
SF_FOR_PF = {400: 400.0, 300: 316.0, 200: 232.0, 100: 148.0}


def waterfall_fio2(src, windows: pd.DataFrame, pad: pd.Timedelta = PAIR_TOLERANCE) -> pd.DataFrame:
    """FiO2 timeline from the waterfall, clipped to each encounter's window (plus a lead-in)."""
    out = []
    for p in src.waterfall_chunks():
        g = pd.read_parquet(p, columns=["encounter_block", "recorded_dttm", "fio2_set",
                                        "device_category"])
        g["encounter_block"] = attr.norm_block(g["encounter_block"])
        g = g[g["fio2_set"].notna() & (g["fio2_set"] > 0)]
        if g.empty:
            continue
        g["recorded_dttm"] = attr.as_ns(g["recorded_dttm"])
        g = g.join(windows, on="encounter_block")
        g = g[(g["recorded_dttm"] >= g["s"] - pad) & (g["recorded_dttm"] <= g["e"])]
        if len(g):
            out.append(g[["encounter_block", "recorded_dttm", "fio2_set", "device_category"]])
    if not out:
        return pd.DataFrame(columns=["encounter_block", "recorded_dttm", "fio2_set",
                                     "device_category"])
    return pd.concat(out, ignore_index=True).sort_values("recorded_dttm", kind="stable")


def _worst_ratio(measures: pd.DataFrame, fio2: pd.DataFrame, value_col: str,
                 name: str) -> pd.DataFrame:
    """Pair each oxygenation measurement to the FiO2 in force, and keep the worst ratio."""
    if measures.empty or fio2.empty:
        return pd.DataFrame(columns=["encounter_block", name, f"{name}_device"])
    d = measures.sort_values("t", kind="stable")
    j = pd.merge_asof(d, fio2.rename(columns={"recorded_dttm": "t"}), on="t",
                      by="encounter_block", direction="backward", tolerance=PAIR_TOLERANCE)
    j = j[j["fio2_set"].notna()]
    if j.empty:
        return pd.DataFrame(columns=["encounter_block", name, f"{name}_device"])
    j[name] = j[value_col] / j["fio2_set"]
    # The WORST ratio, and the device in force at that moment (SOFA 3/4 need support).
    j = j.sort_values([name, "t"], kind="stable").drop_duplicates("encounter_block", keep="first")
    j = j.rename(columns={"device_category": f"{name}_device"})
    return j[["encounter_block", name, f"{name}_device"]]


def compute(src, windows: pd.DataFrame, pao2: pd.DataFrame, spo2: pd.DataFrame) -> pd.DataFrame:
    """Per encounter: worst P/F, worst S/F, and the ratio actually used.

    `pao2` / `spo2` need columns [encounter_block, t, value].
    """
    fio2 = waterfall_fio2(src, windows)
    sp = spo2[spo2["value"] <= SPO2_MAX_FOR_SF]
    pf = _worst_ratio(pao2.rename(columns={"value": "v"}), fio2, "v", "pf")
    sf = _worst_ratio(sp.rename(columns={"value": "v"}), fio2, "v", "sf")
    out = pf.merge(sf, on="encounter_block", how="outer")
    out["oxy_source"] = np.where(out["pf"].notna(), "P/F",
                                 np.where(out["sf"].notna(), "S/F", None))
    out["oxy_device"] = out["pf_device"].where(out["pf"].notna(), out["sf_device"])
    return out


def resp_subscore(pf, sf, device) -> pd.Series:
    """Classic SOFA respiratory subscore, P/F preferred with an S/F fallback.

    P/F cutpoints are verbatim from CLIF-epidemiology-of-CRRT `_compute_sofa_scores`; the S/F
    cutpoints are their Rice 2007 equivalents. Scores 3 and 4 require respiratory support, as in
    the original definition.

    One deliberate departure from the CRRT implementation: there, a P/F below 200 WITHOUT support
    falls through every branch and yields null. Here it scores 2 — the highest level the original
    definition awards without a support qualifier — so severe hypoxaemia on a non-support device
    is recorded as abnormal rather than as missing.
    """
    pf = pd.to_numeric(pf, errors="coerce")
    sf = pd.to_numeric(sf, errors="coerce")
    dev = device.astype("string").str.lower()
    supported = dev.isin(SUPPORT_DEVICES).fillna(False)

    # Work in P/F space; convert S/F to its P/F equivalent where P/F is unavailable.
    ratio = pf.copy()
    use_sf = pf.isna() & sf.notna()
    # invert Rice: PF = (SF - 64) / 0.84
    ratio[use_sf] = (sf[use_sf] - 64.0) / 0.84

    score = pd.Series(np.nan, index=ratio.index, dtype="float64")
    score[ratio >= 400] = 0
    score[(ratio >= 300) & (ratio < 400)] = 1
    score[(ratio >= 200) & (ratio < 300)] = 2
    score[(ratio >= 100) & (ratio < 200)] = np.where(supported[(ratio >= 100) & (ratio < 200)], 3, 2)
    score[ratio < 100] = np.where(supported[ratio < 100], 4, 2)
    return score
