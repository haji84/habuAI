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


def test_import_script_attempts_sha_gated_legacy_recovery():
    text = Path("scripts/import_raw_bundles.sh").read_text(encoding="utf-8")
    assert "bash scripts/recover_legacy_holdout_parts.sh" in text


def test_legacy_recovery_accepts_only_canonical_0828_raw_sha():
    text = Path("scripts/recover_legacy_holdout_parts.sh").read_text(encoding="utf-8")
    assert "d10ef5ec68db9bef4e792ecb0a8cee418e9277872f09d536363b22b49bce7ee5" in text
    assert 'if [ "$sha" = "$TARGET_SHA" ]' in text
    assert "part_00_v2.b64" in text
    assert "part_01_v2.b64" in text
    assert "part_06.b64" in text
    assert "base64 -d" in text
    assert "xz -t" in text
    assert "grep -q '<trkpt'" in text
