"""Resolve where the cohort comes from: a validated stage cache, or a cold upstream build.

A clif-dashboards respiratory build leaves behind a `stage/` folder holding the stitched
encounters, the ADT table, the IMV episodes/runs and the clifpy respiratory-support waterfall.
Reusing it skips the expensive re-stitch and re-waterfall. It is strictly an optimization:

  tier 1  stage  — the cache exists, is complete, AND was built under the same decisions
  tier 2  build  — otherwise, materialize it by invoking the upstream builder in-process
  tier 3  fail   — no cache and no readable data path

Tier 2 runs the *same* code that produced the cache, so there is exactly one staging
implementation. Nothing about the chunked DuckDB staging or the waterfall is reimplemented here.

Validation deliberately checks what the cache actually *did* (the decisions tree embedded in its
summary.json) rather than a hash of the files we hope it read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import bundle_config as bc

#: Files a complete respiratory stage cache must contain.
REQUIRED_FILES = ("encounters.parquet", "mapping.parquet", "adt.parquet",
                  "episodes_a.parquet", "runs_a.parquet", "chunks_b.parquet")
#: Per-chunk tables the metric records need, staged under phase_b/<table>/.
PHASE_B_TABLES = ("vitals", "labs", "medication_admin_continuous",
                  "patient_assessments", "position", "code_status")


@dataclass(frozen=True)
class CohortSource:
    tier: str                 # "stage" | "build"
    stage: Path
    reason: str               # why this tier was chosen, for the log and for provenance

    @property
    def summary_path(self) -> Path:
        return self.stage.parent / "summary.json"

    def cfg(self, site: str | None = None):
        """clif-dashboards SiteConfig whose stage_dir is the stage we actually resolved.

        `stage_dir` is derived as output_dir/<dashboard>/stage, so pointing output_dir two
        levels above the stage re-targets the upstream readers at an external cache without
        copying anything.
        """
        from dataclasses import replace
        return replace(bc.upstream_config(site), output_dir=self.stage.parent.parent)

    def load_chunk(self, table: str, chunk: int, site: str | None = None) -> pd.DataFrame:
        """One staged phase-B chunk, read with the upstream's own loader.

        Reusing `stage.load_staged_chunk` keeps source-file ordering and the CLIF-2.1 column
        conventions identical to what the dashboard's own metric code sees.
        """
        from clif_dash.stage import load_staged_chunk
        return load_staged_chunk(self.cfg(site), "phase_b", table, chunk)

    def waterfall_chunks(self) -> list[Path]:
        return sorted((self.stage / "waterfall").glob("chunk_*.parquet"),
                      key=lambda p: int(p.stem.split("_")[1]))

    def phase_b_chunk(self, table: str, chunk: int) -> Path | None:
        p = self.stage / "phase_b" / table / f"chunk_{chunk}.parquet"
        return p if p.exists() else None

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.stage / f"{name}.parquet")


def _completeness(stage: Path) -> str | None:
    """None when complete, else a human reason it is not."""
    if not stage.is_dir():
        return f"no stage directory at {stage}"
    missing = [f for f in REQUIRED_FILES if not (stage / f).exists()]
    if missing:
        return f"missing {', '.join(missing)}"
    chunks = sorted((stage / "waterfall").glob("chunk_*.parquet")) if (stage / "waterfall").is_dir() else []
    if not chunks:
        return "no waterfall chunks"
    try:
        expected = pd.read_parquet(stage / "chunks_b.parquet")["chunk"].nunique()
    except Exception as exc:                                  # unreadable cache is an incomplete cache
        return f"chunks_b.parquet unreadable ({type(exc).__name__})"
    if len(chunks) != expected:
        return f"{len(chunks)} waterfall chunks but chunks_b declares {expected}"
    for table in PHASE_B_TABLES:
        d = stage / "phase_b" / table
        # duckdb COPY ... PARTITION_BY writes hive-style chunk=N/ directories
        if not d.is_dir() or not any(d.glob("chunk=*")):
            return f"phase_b/{table} not staged"
    return None


def _decision_mismatch(stage: Path, site: str | None) -> str | None:
    """None when the cache's decisions match ours, else a reason."""
    summary = stage.parent / "summary.json"
    if not summary.exists():
        return f"no summary.json beside the stage ({summary.name}) to verify decisions against"
    try:
        cached = json.loads(summary.read_text()).get("decisions")
    except Exception as exc:
        return f"summary.json unreadable ({type(exc).__name__})"
    if cached is None:
        return "summary.json carries no decisions tree"
    ours = bc.decisions(site)
    diffs = _diff_decisions(cached, ours)
    if diffs:
        head = "; ".join(diffs[:4])
        more = f" (+{len(diffs) - 4} more)" if len(diffs) > 4 else ""
        return f"decisions differ from definitions/: {head}{more}"
    return None


def _diff_decisions(cached: dict, ours: dict) -> list[str]:
    """Compare only the decision VALUES that change a number; ignore prose and changelog."""
    def flat(tree: dict) -> dict[str, object]:
        out: dict[str, object] = {}
        for d in tree.get("shared", []) or []:
            out[f"shared.{d['id']}"] = d.get("value")
        for metric, spec in (tree.get("metrics") or {}).items():
            for d in spec.get("decisions", []) or []:
                out[f"{metric}.{d['id']}"] = d.get("value")
        return out

    a, b = flat(cached), flat(ours)
    diffs = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        # YAML tuples vs JSON lists are the same decision
        if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
            if list(va) == list(vb):
                continue
        elif va == vb:
            continue
        diffs.append(f"{key}: cache={va!r} vs ours={vb!r}")
    return diffs


def record(src: "CohortSource", path: Path) -> None:
    """Persist which source stage 0 resolved, so later stages reuse it instead of re-resolving.

    Without this, every stage calling `resolve(rebuild=True)` would re-run the whole upstream
    build — paying the waterfall cost once per stage — and a stage that did NOT get the flag would
    silently fall back to a different (stale) cache than the one stage 0 just built.
    """
    path.write_text(json.dumps({"tier": src.tier, "stage": str(src.stage), "reason": src.reason},
                               indent=2))


def recorded(path: Path, log=print) -> "CohortSource | None":
    """The source stage 0 recorded, if it is still complete. None means re-resolve."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        stage = Path(d["stage"])
    except Exception:
        return None
    bad = _completeness(stage)
    if bad:
        log(f"  recorded cohort source at {stage} is no longer usable: {bad}")
        return None
    log(f"  cohort source: reusing what stage 0 resolved ({d['tier']}) — {stage}")
    return CohortSource(d["tier"], stage, d["reason"])


def resolve(site: str | None = None, *, rebuild: bool = False,
            require_stage: bool = False, log=print) -> CohortSource:
    """Pick a cohort source. See module docstring for the tier semantics."""
    site = bc.active_site(site)
    prof = bc.load_profile(site)
    declared = prof.get("stage_dir")

    if rebuild:
        if require_stage:
            raise ValueError("--rebuild and --require-stage are mutually exclusive")
        return _build(site, "forced by --rebuild", log)

    if declared:
        stage = Path(declared)
        reason = _completeness(stage) or _decision_mismatch(stage, site)
        if reason is None:
            log(f"  cohort source: stage cache (tier 1) — {stage}")
            return CohortSource("stage", stage, f"validated stage cache at {stage}")
        log(f"  stage cache at {stage} NOT usable: {reason}")
        if require_stage:
            raise RuntimeError(f"--require-stage was given but the cache is unusable: {reason}")
    else:
        reason = "no stage_dir in the site profile"
        log(f"  {reason}")
        if require_stage:
            raise RuntimeError("--require-stage was given but the site profile declares no stage_dir")

    # A cache we built ourselves on a previous cold run is just as good.
    own = bc.upstream_config(site).stage_dir
    if own != Path(declared or ""):
        own_reason = _completeness(own) or _decision_mismatch(own, site)
        if own_reason is None:
            log(f"  cohort source: previously built stage (tier 1) — {own}")
            return CohortSource("stage", own, f"validated self-built stage cache at {own}")

    return _build(site, reason, log)


def _build(site: str, why: str, log=print) -> CohortSource:
    """Tier 2 — run the upstream respiratory build to materialize a stage cache."""
    prof = bc.load_profile(site)
    data_path = Path(prof["data_path"])
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Cannot build the cohort: data_path does not exist.\n"
            f"  data_path : {data_path}\n"
            f"  reason the cache was not used: {why}\n"
            f"Fix sites/{site}.json, or point stage_dir at a clif-dashboards respiratory build."
        )
    log(f"  cohort source: cold build (tier 2) — {why}")
    log(f"  running the upstream clif-dashboards respiratory build; this re-runs encounter "
        f"stitching and the clifpy waterfall and writes ~1 GB of stage output.")

    from clif_dash.build import build_all
    from clif_dash.dashboards import DASHBOARDS

    cfg = bc.upstream_config(site)
    build_all(cfg, [DASHBOARDS[bc.UPSTREAM_DASHBOARD]])

    stage = cfg.stage_dir
    bad = _completeness(stage)
    if bad:
        raise RuntimeError(f"upstream build finished but its stage is incomplete: {bad}")
    return CohortSource("build", stage, f"cold upstream build ({why})")
