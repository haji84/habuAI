from __future__ import annotations

from pathlib import Path

from habuai.session_evidence import (
    canonical_population_from_manifest,
    parse_text_exploration_markers,
)


def test_after_midnight_search_end_belongs_to_previous_operational_night(tmp_path: Path):
    path = tmp_path / "log.txt"
    path.write_text(
        "2026/08/20 2:52\n鹿児島県 瀬戸内町\nエリア探索終了\n",
        encoding="utf-8",
    )
    manifest = parse_text_exploration_markers(path)
    assert len(manifest) == 1
    assert manifest.iloc[0].operational_date_0700 == "2026-08-19"
    assert manifest.iloc[0].evidence_type == "SEARCH_END"


def test_multiple_area_transitions_collapse_to_one_operational_night(tmp_path: Path):
    path = tmp_path / "log.txt"
    path.write_text(
        "2026/06/13 22:02\nエリア探索開始\n"
        "2026/06/14 0:04\nエリア探索終了\n"
        "2026/06/14 0:04\nエリア探索開始\n"
        "2026/06/14 0:25\nエリア探索終了\n",
        encoding="utf-8",
    )
    manifest = parse_text_exploration_markers(path)
    assert len(manifest) == 4
    assert canonical_population_from_manifest(manifest) == ["2026-06-13"]


def test_same_calendar_day_can_contain_two_operational_nights(tmp_path: Path):
    path = tmp_path / "log.txt"
    path.write_text(
        "2026/07/17 3:34\nエリア探索終了\n"
        "2026/07/17 21:36\nエリア探索開始\n",
        encoding="utf-8",
    )
    manifest = parse_text_exploration_markers(path)
    assert canonical_population_from_manifest(manifest) == ["2026-07-16", "2026-07-17"]
