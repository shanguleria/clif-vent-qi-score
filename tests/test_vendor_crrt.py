"""The vendored CRRT SOFA calculator must stay byte-identical to the commit SOURCE records."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "crrt_sofa"
SOURCE = yaml.safe_load((VENDOR / "SOURCE").read_text())


@pytest.mark.parametrize("rel", sorted(SOURCE["files"]))
def test_vendored_file_matches_recorded_hash(rel):
    digest = hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest()
    assert f"sha256:{digest}" == SOURCE["files"][rel], f"{rel} differs from upstream {SOURCE['upstream_sha_short']}"


def test_every_vendored_file_is_recorded():
    on_disk = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
               if p.is_file() and p.name != "SOURCE" and "__pycache__" not in p.parts}
    assert on_disk == set(SOURCE["files"])


def test_sofa_module_points_at_the_vendored_copy():
    import sys
    sys.path.insert(0, str(VENDOR.parents[1]))
    from common import sofa_crrt
    assert sofa_crrt.CRRT_CODE_DIR == VENDOR / "code"
    assert sofa_crrt.CRRT_COMMIT == SOURCE["upstream_sha_short"]
