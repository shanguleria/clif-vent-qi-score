"""The PHI gate must actually fire. Synthetic files only, no CLIF data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gate():
    """Import phi_gate from the numbered stage script (03_* is not a legal module name)."""
    spec = importlib.util.spec_from_file_location("stage03", ROOT / "code" / "03_build_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.phi_gate, mod.FORBIDDEN


def test_gate_passes_on_an_aggregate_panel(tmp_path):
    gate, _ = _gate()
    pd.DataFrame({"unit_id": ["Unit 1"], "period": ["2023-Q1"], "score_core": [54.0]}
                 ).to_csv(tmp_path / "unit_period_panel.csv", index=False)
    gate(tmp_path)  # must not raise


@pytest.mark.parametrize("identifier", ["patient_id", "hospitalization_id",
                                        "encounter_block", "episode_id"])
def test_gate_fires_on_each_forbidden_identifier_in_csv(tmp_path, identifier):
    gate, forbidden = _gate()
    assert identifier in forbidden
    pd.DataFrame({identifier: ["x1"], "score_core": [54.0]}).to_csv(tmp_path / "leak.csv", index=False)
    with pytest.raises(SystemExit, match="ABORT"):
        gate(tmp_path)


def test_gate_fires_on_a_parquet_column(tmp_path):
    gate, _ = _gate()
    pd.DataFrame({"patient_id": ["p1"], "score_core": [54.0]}).to_parquet(
        tmp_path / "leak.parquet", index=False)
    with pytest.raises(SystemExit, match="patient_id"):
        gate(tmp_path)


def test_gate_fires_on_json_content(tmp_path):
    gate, _ = _gate()
    (tmp_path / "prov.json").write_text('{"note": "keyed by encounter_block"}')
    with pytest.raises(SystemExit, match="encounter_block"):
        gate(tmp_path)


def test_gate_names_every_offending_file(tmp_path):
    gate, _ = _gate()
    pd.DataFrame({"patient_id": ["p"]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"episode_id": ["e"]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(SystemExit) as exc:
        gate(tmp_path)
    assert "a.csv" in str(exc.value) and "b.csv" in str(exc.value)
