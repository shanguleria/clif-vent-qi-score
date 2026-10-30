"""Site profile + shared definitions resolver.

Three layers collapse into one object:
  * `sites/<site>.json`        — per-site access (data path, timezone, optional stage cache), gitignored
  * `definitions/*.yaml`       — the clif-dashboards decision files, copied verbatim at a pinned SHA
  * `definitions/composite.json` — which metrics make up which score, and at what weight

The decision YAMLs are handed back to clif-dashboards by pointing its `SiteConfig.decisions_dir`
at our `definitions/` folder, so the upstream builder and our scoring code read the *same* files.
Nothing in this repo re-types a threshold that lives in those YAMLs.

Site selection: `CLIF_SITE` env var, or an explicit `site=` argument (default "uchicago").
Reads no CLIF data.
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "definitions"
SITES = ROOT / "sites"

#: Dashboard whose decision file carries the four metrics we score.
UPSTREAM_DASHBOARD = "respiratory"


def active_site(site: str | None = None) -> str:
    return site or os.environ.get("CLIF_SITE") or "uchicago"


@lru_cache(maxsize=None)
def load_profile(site: str | None = None) -> dict:
    site = active_site(site)
    path = SITES / f"{site}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No site profile at {path}. Create one with:\n"
            f"  cp sites/uchicago.example.json sites/{site}.json   # then edit data_path"
        )
    prof = json.loads(path.read_text())
    for required in ("site_id", "data_path", "timezone", "clif_version"):
        if not prof.get(required):
            raise ValueError(f"{path} is missing required key '{required}'")
    return prof


@lru_cache(maxsize=None)
def load_composite() -> dict:
    return json.loads((DEFINITIONS / "composite.json").read_text())


@lru_cache(maxsize=None)
def source_info() -> dict:
    """Pinned upstream provenance recorded in definitions/SOURCE."""
    return yaml.safe_load((DEFINITIONS / "SOURCE").read_text())


#: Panel grains. "unit" pools every available period into one score per ICU and is the PRIMARY
#: exposure; "unit_period" keeps the time dimension and is retained for sensitivity analysis.
GRAINS = ("unit", "unit_period")

#: Grouping columns for each grain.
GROUP_COLS = {"unit": ("unit_name",), "unit_period": ("unit_name", "period")}


def min_denominator(grain: str, site: str | None = None) -> dict[str, float]:
    """Per-metric denominator floors for one grain.

    Floors are stated per grain because a quarter-calibrated floor can never bind on a denominator
    pooled over seven years — and a config key that cannot bind is a policy declaration the next
    reader will trust.
    """
    if grain not in GRAINS:
        raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
    floors = load_profile(site).get("min_denominator") or {}
    if grain not in floors:
        raise KeyError(
            f"sites/{active_site(site)}.json min_denominator has no '{grain}' block; "
            f"found {[k for k in floors if not k.startswith('_')]}"
        )
    return {k: v for k, v in floors[grain].items() if not k.startswith("_")}


def metric_ids() -> list[str]:
    """Every metric the composite spec declares. Iterate THIS, never a hardcoded list."""
    return list(load_composite()["metrics"])


def score_ids() -> list[str]:
    return list(load_composite()["scores"])


def weights(score: str) -> dict[str, float]:
    """Normalized weights for `score`, summing to 1."""
    raw = load_composite()["scores"][score]["weights"]
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError(f"score '{score}' has non-positive total weight")
    return {m: v / total for m, v in raw.items()}


# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #
def output_root(site: str | None = None) -> Path:
    return ROOT / "output" / active_site(site)


def phi_dir(site: str | None = None) -> Path:
    """Row-level working space. NEVER shared, never read with a display tool."""
    d = output_root(site) / "intermediate_phi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def share_dir(site: str | None = None) -> Path:
    """Aggregate-by-construction deliverables. PHI-gated at assembly."""
    d = output_root(site) / "final_no_phi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def upstream_build_dir(site: str | None = None) -> Path:
    """Where a tier-2 (cold) upstream build materializes its own stage cache."""
    d = output_root(site) / "upstream_build"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# clif-dashboards SiteConfig
# --------------------------------------------------------------------------- #
def site_yaml_path(site: str | None = None) -> Path:
    return upstream_build_dir(site) / "site.yaml"


def write_site_yaml(site: str | None = None) -> Path:
    """Render a clif-dashboards site YAML from our profile.

    `decisions_dir` points at our definitions/ folder, so an upstream build reads exactly the
    decision files this repo ships — there is no second copy to drift from.
    """
    site = active_site(site)
    prof = load_profile(site)
    out = {
        "clif_dir": prof["data_path"],
        "filetype": prof.get("file_format", "parquet"),
        "file_prefix": prof.get("file_prefix", "clif_"),
        "site_name": prof["site_id"],
        "site_timezone": prof["timezone"],
        "start_date": prof.get("start_date"),
        "end_date": prof.get("end_date"),
        "decisions_dir": str(DEFINITIONS),
        "output_dir": str(upstream_build_dir(site)),
        "dashboards": [UPSTREAM_DASHBOARD],
        "encounters_per_chunk": prof.get("encounters_per_chunk", 2000),
        "cohort_encounters_per_chunk": prof.get("cohort_encounters_per_chunk", 300),
        "workers": prof.get("workers", 4),
    }
    path = site_yaml_path(site)
    path.write_text(
        "# GENERATED by bundle_config.write_site_yaml() — do not edit.\n"
        f"# Source: sites/{site}.json + definitions/\n"
        + yaml.safe_dump(out, sort_keys=False)
    )
    return path


def upstream_config(site: str | None = None):
    """clif-dashboards SiteConfig bound to the respiratory dashboard."""
    from clif_dash.config import load_config
    return load_config(write_site_yaml(site)).bind(UPSTREAM_DASHBOARD)


def decisions(site: str | None = None) -> dict:
    """The merged platform+respiratory decisions tree, as the upstream resolves it."""
    return upstream_config(site).decisions()


def decision_value(metric: str, decision_id: str, site: str | None = None):
    """One upstream decision value, e.g. decision_value('ltvv', 'threshold_ml_kg') -> 8.

    Reads the shipped YAML rather than restating the number, so a threshold can only be changed
    in one place.
    """
    entries = decisions(site)["metrics"][metric]["decisions"]
    for d in entries:
        if d["id"] == decision_id:
            return d["value"]
    raise KeyError(
        f"decision '{decision_id}' not found for metric '{metric}'; "
        f"available: {[d['id'] for d in entries]}"
    )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance(site: str | None = None, cohort_source: str | None = None) -> dict:
    from datetime import datetime, timezone as _tz
    prof = load_profile(site)
    return {
        "site_id": prof["site_id"],
        "code_version": git_sha(),
        "clif_version": prof["clif_version"],
        "dataset_version": prof.get("dataset_version"),
        "definition_version": load_composite()["definition_version"],
        "upstream_repo": source_info()["upstream_repo"],
        "upstream_sha": source_info()["upstream_sha_short"],
        "cohort_source": cohort_source,
        "period": prof.get("period", "quarter"),
        "generated": datetime.now(_tz.utc).isoformat(timespec="minutes"),
    }
