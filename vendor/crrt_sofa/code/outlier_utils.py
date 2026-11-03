"""
Single source of truth for outlier handling.

Every outlier bound comes from config/outlier_config.json — a flat map of
{variable: [min, max]}. `apply_outliers()` sets values outside [min, max] to NaN,
for both long/categorical CLIF tables (labs / vitals / meds, clipped per category)
and wide frames (columns whose de-prefixed name is a config variable). No outlier
range is ever hard-coded in a pipeline script; change a bound in one JSON file and
it applies everywhere.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "outlier_config.json"

# long/categorical tables: kind -> (category column, numeric value column)
_LONG_COLS = {
    "lab": ("lab_category", "lab_value_numeric"),
    "vital": ("vital_category", "vital_value"),
    "med": ("med_category", "med_dose"),
}
# wide-column prefixes stripped to recover the config key (longest first)
_WIDE_PREFIXES = ("med_cont_", "crrt_", "resp_", "lab_", "vital_")


def load_outlier_config(path=None) -> dict:
    """Load config/outlier_config.json as {variable: (min, max)}. Raises if absent —
    silently skipping outlier handling would change the analysis without warning."""
    path = Path(path) if path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"Outlier config not found: {path}\n"
            "  Refusing to continue: skipping outlier handling silently changes results."
        )
    with open(path) as f:
        return {k: (float(lo), float(hi)) for k, (lo, hi) in json.load(f).items()}


def _deprefix(col: str) -> str:
    for p in _WIDE_PREFIXES:
        if col.startswith(p):
            return col[len(p):]
    return col


def _clip(series: pd.Series, lo: float, hi: float):
    """Out-of-range -> NaN. Returns (series, n_removed); copies only if it changes."""
    s = pd.to_numeric(series, errors="coerce")
    bad = s.notna() & ((s < lo) | (s > hi))
    n = int(bad.sum())
    if n:
        series = series.copy()
        series[bad.to_numpy()] = np.nan
    return series, n


def apply_outliers(df: pd.DataFrame, *, long: str = None, wide: bool = False,
                   config: dict = None, label: str = None) -> pd.DataFrame:
    """Clip out-of-range values to NaN using config/outlier_config.json.

    long : one of {'lab','vital','med'} — clip the value column per category.
    wide : True — clip every column whose de-prefixed name is a config variable.
    Exactly one of `long` / `wide` must be given. Returns the (clipped) frame.
    """
    cfg = config or load_outlier_config()
    removed = {}
    if long:
        catcol, valcol = _LONG_COLS[long]
        if catcol in df.columns and valcol in df.columns:
            for var in set(df[catcol].dropna().unique()) & cfg.keys():
                lo, hi = cfg[var]
                mask = (df[catcol] == var).to_numpy()
                clipped, n = _clip(df.loc[mask, valcol], lo, hi)
                if n:
                    df.loc[mask, valcol] = clipped
                    removed[var] = n
    elif wide:
        for col in df.columns:
            # Full name first: config keys like `resp_rate_set` start with a prefix
            # ('resp_') and must NOT be de-prefixed to a non-key ('rate_set').
            var = col if col in cfg else _deprefix(col)
            if var in cfg:
                df[col], n = _clip(df[col], *cfg[var])
                if n:
                    removed[col] = n
    else:
        raise ValueError("apply_outliers: pass long=<'lab'|'vital'|'med'> or wide=True")

    tag = label or long or "wide"
    if removed:
        print(f"  outliers [{tag}]: " + ", ".join(f"{k}={v}" for k, v in removed.items()))
    else:
        print(f"  outliers [{tag}]: none out of range")
    return df
