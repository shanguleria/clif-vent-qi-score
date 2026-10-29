"""Tests for stage-cache validation. Synthetic directory trees, no CLIF data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cohort_source as cs           # noqa: E402


def make_stage(tmp_path: Path, *, chunks: int = 2, omit: str | None = None) -> Path:
    stage = tmp_path / "respiratory" / "stage"
    (stage / "waterfall").mkdir(parents=True)
    pd.DataFrame({"encounter_block": list(range(chunks)), "chunk": list(range(chunks))}
                 ).to_parquet(stage / "chunks_b.parquet", index=False)
    for f in cs.REQUIRED_FILES:
        if f == "chunks_b.parquet" or f == omit:
            continue
        pd.DataFrame({"encounter_block": [1]}).to_parquet(stage / f, index=False)
    for c in range(chunks):
        pd.DataFrame({"encounter_block": [1]}).to_parquet(
            stage / "waterfall" / f"chunk_{c}.parquet", index=False)
    for table in cs.PHASE_B_TABLES:
        for c in range(chunks):
            d = stage / "phase_b" / table / f"chunk={c}"
            d.mkdir(parents=True)
            pd.DataFrame({"encounter_block": [1]}).to_parquet(d / "data_0.parquet", index=False)
    return stage


def test_completeness_accepts_a_full_stage(tmp_path):
    assert cs._completeness(make_stage(tmp_path)) is None


def test_completeness_reports_a_missing_required_file(tmp_path):
    stage = make_stage(tmp_path, omit="episodes_a.parquet")
    assert "episodes_a.parquet" in cs._completeness(stage)


def test_completeness_reports_a_missing_directory(tmp_path):
    assert "no stage directory" in cs._completeness(tmp_path / "nope")


def test_completeness_catches_a_waterfall_chunk_count_mismatch(tmp_path):
    stage = make_stage(tmp_path, chunks=3)
    (stage / "waterfall" / "chunk_2.parquet").unlink()
    reason = cs._completeness(stage)
    assert "2 waterfall chunks" in reason and "declares 3" in reason


def test_completeness_catches_an_unstaged_phase_b_table(tmp_path):
    stage = make_stage(tmp_path)
    for p in sorted((stage / "phase_b" / "vitals").glob("chunk=*")):
        for f in p.glob("*"):
            f.unlink()
        p.rmdir()
    assert "phase_b/vitals" in cs._completeness(stage)


# --------------------------------------------------------------------------- #
# decision matching
# --------------------------------------------------------------------------- #
def tree(ltvv_threshold=8, stitch=6):
    return {
        "shared": [{"id": "stitch_hours", "value": stitch}],
        "metrics": {"ltvv": {"decisions": [
            {"id": "threshold_ml_kg", "value": ltvv_threshold},
            {"id": "modes", "value": ["simv", "volume support"]},
        ]}},
    }


def test_identical_decision_trees_do_not_differ():
    assert cs._diff_decisions(tree(), tree()) == []


def test_a_changed_threshold_is_detected():
    diffs = cs._diff_decisions(tree(ltvv_threshold=6), tree(ltvv_threshold=8))
    assert len(diffs) == 1 and "ltvv.threshold_ml_kg" in diffs[0]


def test_a_changed_shared_decision_is_detected():
    diffs = cs._diff_decisions(tree(stitch=12), tree(stitch=6))
    assert any("shared.stitch_hours" in d for d in diffs)


def test_yaml_tuples_and_json_lists_are_the_same_decision():
    """A cache's decisions arrive via JSON (lists); ours via YAML (may be tuples)."""
    a = tree()
    b = tree()
    b["metrics"]["ltvv"]["decisions"][1]["value"] = ("simv", "volume support")
    assert cs._diff_decisions(a, b) == []


def test_prose_changes_do_not_count_as_decision_changes():
    a, b = tree(), tree()
    b["metrics"]["ltvv"]["decisions"][0]["display"] = "rewritten explanation"
    b["changelog"] = [{"date": "2026-01-01", "change": "docs"}]
    assert cs._diff_decisions(a, b) == [], "only values that change a number should matter"


def test_decision_mismatch_requires_a_summary_to_verify_against(tmp_path):
    stage = make_stage(tmp_path)
    reason = cs._decision_mismatch(stage, None)
    assert "summary.json" in reason, "an unverifiable cache must not be trusted"


def test_decision_mismatch_rejects_a_summary_with_no_decisions_tree(tmp_path):
    stage = make_stage(tmp_path)
    (stage.parent / "summary.json").write_text(json.dumps({"meta": {}}))
    assert "no decisions tree" in cs._decision_mismatch(stage, None)
