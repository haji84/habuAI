from pathlib import Path

import pandas as pd

from habuai.field_weather_features import add_field_rain_candidate_features, parse_field_weather_markers


def test_parse_field_weather_markers_supports_block_inline_and_recovery(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text(
        "2026/09/05 23:47\n雨\n"
        "2026/09/06 0:23\n雨後\n"
        "2026/09/06 1:21\n目撃\n1匹\nカエル\n"
        "なう(2026/09/06 01:25:48)雨やみ\n",
        encoding="utf-8",
    )
    markers = parse_field_weather_markers(p)
    assert markers["marker"].tolist() == ["雨", "雨後", "雨やみ"]
    assert markers["phase"].tolist() == ["rain_onset", "rain_recovery", "rain_recovery"]
    assert markers["night_date"].tolist() == ["2026-09-05"] * 3


def test_add_field_rain_candidate_features_separates_onset_and_recovery(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text(
        "2026/09/06 0:00\n雨\n"
        "2026/09/06 0:23\n雨後\n",
        encoding="utf-8",
    )
    markers = parse_field_weather_markers(p)
    visits = pd.DataFrame({
        "entered_at": pd.to_datetime([
            "2026-09-06T00:29:00+09:00",
            "2026-09-06T00:45:00+09:00",
        ]),
        "segment_id": ["A", "B"],
    })
    out = add_field_rain_candidate_features(visits, markers)

    assert round(float(out.loc[0, "field_minutes_since_rain_onset"]), 1) == 29.0
    assert round(float(out.loc[0, "field_minutes_since_rain_recovery"]), 1) == 6.0
    assert int(out.loc[0, "field_rain_recovery_0_10m"]) == 1
    assert int(out.loc[0, "field_rain_onset_20_30m"]) == 1

    assert round(float(out.loc[1, "field_minutes_since_rain_recovery"]), 1) == 22.0
    assert int(out.loc[1, "field_rain_recovery_20_30m"]) == 1


def test_candidate_features_never_use_future_marker(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text("2026/09/06 1:25\n雨\n", encoding="utf-8")
    markers = parse_field_weather_markers(p)
    visits = pd.DataFrame({
        "entered_at": pd.to_datetime([
            "2026-09-06T01:20:00+09:00",
            "2026-09-06T01:33:00+09:00",
        ]),
        "segment_id": ["A", "B"],
    })
    out = add_field_rain_candidate_features(visits, markers)

    assert pd.isna(out.loc[0, "field_minutes_since_rain_marker"])
    assert int(out.loc[0, "field_weather_marker_available"]) == 0
    assert round(float(out.loc[1, "field_minutes_since_rain_onset"]), 1) == 8.0
    assert int(out.loc[1, "field_rain_onset_0_10m"]) == 1
