from __future__ import annotations

import pandas as pd

from habuai.gps_history_mapmatch import (
    prepare_gps_history_for_map_match,
    select_operational_nights,
    summarize_history_map_match,
)


def test_prepare_history_uses_0700_operational_night_and_keeps_provenance_distinct():
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-23 06:59:00+09:00",
                "2026-07-23 07:01:00+09:00",
            ],
            "lat": [28.17, 28.171],
            "lon": [129.35, 129.351],
        }
    )
    prepared = prepare_gps_history_for_map_match(history)
    assert prepared["operational_date_0700"].tolist() == ["2026-07-22", "2026-07-23"]
    assert prepared["session_file"].tolist() == [
        "gps_history:2026-07-22",
        "gps_history:2026-07-23",
    ]
    assert prepared["seq"].tolist() == [0, 0]


def test_night_filter_does_not_relabel_history_as_actual_gpx():
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-28 20:00:00+09:00",
                "2026-06-29 20:00:00+09:00",
            ],
            "lat": [28.17, 28.18],
            "lon": [129.35, 129.36],
        }
    )
    prepared = prepare_gps_history_for_map_match(history)
    filtered = select_operational_nights(prepared, {"2026-06-28"})
    assert len(filtered) == 1
    assert filtered.iloc[0]["session_file"] == "gps_history:2026-06-28"


def test_map_match_summary_calculates_ratio_and_distance():
    matched = pd.DataFrame(
        {
            "operational_date_0700": ["2026-06-28"] * 5,
            "timestamp": pd.date_range("2026-06-28 20:00", periods=5, freq="5min"),
            "segment_id": ["a", "b", "c", "d", pd.NA],
            "match_distance_m": [5.0, 10.0, 15.0, 20.0, pd.NA],
        }
    )
    summary = summarize_history_map_match(matched).iloc[0]
    assert summary["anchor_count"] == 5
    assert summary["matched_anchor_count"] == 4
    assert summary["match_ratio"] == 0.8
    assert summary["median_match_distance_m"] == 12.5
