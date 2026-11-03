"""Oxygenation for the SOFA respiratory component, paired against the waterfall.

Both SOFA calculators read `respiratory_support` raw, where the table is EVENT-BASED: `fio2_set`
appears only on rows where it was charted, so many IMV rows carry a null because nothing changed,
not because the ventilator had no setting. The waterfall forward-fills `fio2_set` within mode
blocks and lays an hourly scaffold, so pairing against it recovers those hours.

The pre-intubation half of the window has no ventilator and genuinely no FiO2; SpO2 is recorded
throughout, so an S/F fallback covers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import attribution as attr

#: How far back an FiO2 may be carried to pair with an oxygenation measurement.
PAIR_TOLERANCE = pd.Timedelta(hours=4)

# Above this the dissociation curve is flat and S/F is uninformative (Rice 2007).
SPO2_MAX_FOR_SF = 97.0

#: Devices that satisfy the "with respiratory support" qualifier on SOFA respiratory 3 and 4.
SUPPORT_DEVICES = {"imv", "nippv", "cpap"}

# Rice 2007: SF = 64 + 0.84 x PF. code/08_outcome_model.R inverts the same line (RICE_SF_*).
SF_FOR_PF = {400: 400.0, 300: 316.0, 200: 232.0, 100: 148.0}


def waterfall_fio2(src, windows: pd.DataFrame, pad: pd.Timedelta = PAIR_TOLERANCE) -> pd.DataFrame:
    """FiO2 timeline clipped to each window, plus a lead-in for the as-of join."""
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
    """Pair each oxygenation measurement to the FiO2 in force; keep the worst ratio."""
    if measures.empty or fio2.empty:
        return pd.DataFrame(columns=["encounter_block", name, f"{name}_device"])
    d = measures.sort_values("t", kind="stable")
    j = pd.merge_asof(d, fio2.rename(columns={"recorded_dttm": "t"}), on="t",
                      by="encounter_block", direction="backward", tolerance=PAIR_TOLERANCE)
    j = j[j["fio2_set"].notna()]
    if j.empty:
        return pd.DataFrame(columns=["encounter_block", name, f"{name}_device"])
    j[name] = j[value_col] / j["fio2_set"]
    # Device at the worst moment too: SOFA 3 and 4 require a support device.
    j = j.sort_values([name, "t"], kind="stable").drop_duplicates("encounter_block", keep="first")
    j = j.rename(columns={"device_category": f"{name}_device"})
    return j[["encounter_block", name, f"{name}_device"]]


def compute(src, windows: pd.DataFrame, pao2: pd.DataFrame, spo2: pd.DataFrame) -> pd.DataFrame:
    """Worst P/F, worst S/F, and which was used. `pao2`/`spo2`: [encounter_block, t, value]."""
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

    P/F cutpoints are verbatim from CLIF-epidemiology-of-CRRT `_compute_sofa_scores`. Departure:
    there, P/F below 200 without a support device falls through every branch and yields null;
    here it scores 2, so severe hypoxaemia off support reads as abnormal rather than missing.
    """
    pf = pd.to_numeric(pf, errors="coerce")
    sf = pd.to_numeric(sf, errors="coerce")
    dev = device.astype("string").str.lower()
    supported = dev.isin(SUPPORT_DEVICES).fillna(False)

    ratio = pf.copy()
    use_sf = pf.isna() & sf.notna()
    ratio[use_sf] = (sf[use_sf] - 64.0) / 0.84      # invert Rice

    score = pd.Series(np.nan, index=ratio.index, dtype="float64")
    score[ratio >= 400] = 0
    score[(ratio >= 300) & (ratio < 400)] = 1
    score[(ratio >= 200) & (ratio < 300)] = 2
    score[(ratio >= 100) & (ratio < 200)] = np.where(supported[(ratio >= 100) & (ratio < 200)], 3, 2)
    score[ratio < 100] = np.where(supported[ratio < 100], 4, 2)
    return score
