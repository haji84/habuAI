from __future__ import annotations

from pathlib import Path


def test_import_script_uses_isolated_compressed_gpx_directory():
    text = Path("scripts/import_raw_bundles.sh").read_text(encoding="utf-8")
    assert "data/raw/compressed_gpx/*.gpx.xz" in text
    assert "data/raw/holdout_2026-08-28.gpx.xz" in text
    assert "rm -f data/raw/gpx/2026-08-28.gpx" not in text


def test_import_script_integrity_and_trackpoint_gates_exist():
    text = Path("scripts/import_raw_bundles.sh").read_text(encoding="utf-8")
    assert 'xz -t "$compressed"' in text
    assert 'xz -dc "$compressed"' in text
    assert 'grep -q \'<trkpt\'' in text
    assert '[ ! -s "$tmp_gpx" ]' in text
