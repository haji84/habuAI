from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path("scripts/check_canonical_gpx_inventory.py")
_spec = importlib.util.spec_from_file_location("check_canonical_gpx_inventory", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_current_qc_schema_uses_sha256_and_is_accepted():
    expected = pd.read_csv("reports/v2_actual_gpx_13night_qc.csv")
    assert "sha256" in expected.columns
    assert _module._resolve_hash_column(expected) == "sha256"


def test_legacy_raw_sha256_schema_remains_accepted():
    expected = pd.DataFrame({"operational_night": ["2026-08-28"], "raw_sha256": ["abc"]})
    assert _module._resolve_hash_column(expected) == "raw_sha256"
