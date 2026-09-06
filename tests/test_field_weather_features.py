from pathlib import Path

import pandas as pd

from habuai.field_weather_features import add_field_rain_candidate_features, parse_field_weather_markers


def test_parse_field_weather_markers_supports_block_and_inline(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text(
        "2026/09/05 23:47\n雨\n2026/09/06 1:21\n目撃\n1匹\nカエル\nなう(2026/09/06 01:25:48)雨\n",
        encoding="utf-8",
    )
    markers = parse_field_weather_markers(p)
    assert len(markers) == 2
    assert markers["marker"].tolist() == ["雨", "雨"]
    assert markers["night_date"].tolist() == ["2026-09-05", "2026-09-05"]


def test_add_field_rain_candidate_features_uses_same_operational_night(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text("2026/09/06 1:25\n雨\n", encoding="utf-8")
    markers = parse_field_weather_markers(p)
    visits = pd.DataFrame({
        "entered_at": pd.to_datetime([
            "2026-09-06T01:33:00+09:00",
            "2026-09-06T01:50:00+09:00",
        ]),
        "segment_id": ["A", "B"],
    })
    out = add_field_rain_candidate_features(visits, markers)
    assert round(float(out.loc[0, "field_minutes_since_rain_marker"]), 1) == 8.0
    assert int(out.loc[0, "field_rain_transition_0_10m"]) == 1
    assert int(out.loc[1, "field_rain_transition_20_30m"]) == 1
