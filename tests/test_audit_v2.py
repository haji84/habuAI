from __future__ import annotations

import pandas as pd

from habuai.audit_v2 import (
    CLASS_ACTUAL_GPX,
    CLASS_RECONSTRUCTED_HIGH,
    CLASS_RECONSTRUCTED_PARTIAL,
    CLASS_SPATIAL_ONLY,
    build_night_audit,
    operational_date_0700,
)


def test_operational_date_0700_boundary():
    assert operational_date_0700("2026-07-23 00:00:00+09:00").isoformat() == "2026-07-22"
    assert operational_date_0700("2026-07-23 05:23:00+09:00").isoformat() == "2026-07-22"
    assert operational_date_0700("2026-07-23 06:59:59+09:00").isoformat() == "2026-07-22"
    assert operational_date_0700("2026-07-23 07:00:00+09:00").isoformat() == "2026-07-23"


def _actual_gpx(matched_points: int, total_points: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2026-08-20 20:00", periods=total_points, freq="1min", tz="Asia/Tokyo")
    return pd.DataFrame(
        {
            "session_file": ["night.gpx"] * total_points,
            "timestamp": ts,
            "lat": [28.17] * total_points,
            "lon": [129.35] * total_points,
            "segment_id": [
                "seg-1" if i < matched_points else pd.NA for i in range(total_points)
            ],
        }
    )


def test_actual_gpx_no_capture_becomes_no_capture_observed_at_full_match():
    gpx = _actual_gpx(120)
    audit = build_night_audit(gpx, pd.DataFrame(), exploration_nights=["2026-08-20"])
    row = audit.iloc[0]
    assert row.classification == CLASS_ACTUAL_GPX
    assert bool(row.can_generate_no_capture_observed)
    assert row.night_observation_label == "NO_CAPTURE_OBSERVED"


def test_actual_gpx_below_80pct_keeps_provenance_but_blocks_strict_labels():
    gpx = _actual_gpx(95, 120)  # 79.17%
    audit = build_night_audit(gpx, pd.DataFrame(), exploration_nights=["2026-08-20"])
    row = audit.iloc[0]
    assert row.classification == CLASS_ACTUAL_GPX
    assert row.usable_road
    assert not row.usable_road_10min_train
    assert not row.usable_road_10min_eval
    assert not row.can_generate_no_capture_observed
    assert row.night_observation_label == "Unknown"
    assert "below 80%" in row.limitation_reason


def test_actual_gpx_at_80pct_becomes_strict_eligible():
    gpx = _actual_gpx(96, 120)  # exactly 80%
    audit = build_night_audit(gpx, pd.DataFrame(), exploration_nights=["2026-08-20"])
    row = audit.iloc[0]
    assert row.classification == CLASS_ACTUAL_GPX
    assert row.usable_road_10min_train
    assert row.usable_road_10min_eval
    assert row.can_generate_no_capture_observed


def test_ordinary_event_only_night_does_not_create_population():
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-10 23:02", tz="Asia/Tokyo")],
            "event_type": ["轢死"],
            "species": ["ハブ"],
            "lat": [28.25],
            "lon": [129.33],
        }
    )
    audit = build_night_audit(pd.DataFrame(), events)
    assert audit.empty


def test_event_is_joined_after_explicit_exploration_population_is_fixed():
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-10 23:02", tz="Asia/Tokyo")],
            "event_type": ["轢死"],
            "species": ["ハブ"],
            "lat": [28.25],
            "lon": [129.33],
        }
    )
    audit = build_night_audit(
        pd.DataFrame(), events, exploration_nights=["2026-05-10"]
    )
    row = audit.iloc[0]
    assert row.classification == CLASS_SPATIAL_ONLY
    assert row.capture_count == 0
    assert row.night_observation_label == "Unknown"


def test_search_end_marker_creates_exploration_night_with_0700_boundary():
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-08-20 02:52", tz="Asia/Tokyo")],
            "event_type": ["探索終了"],
            "species": [""],
            "lat": [pd.NA],
            "lon": [pd.NA],
        }
    )
    audit = build_night_audit(pd.DataFrame(), events)
    assert len(audit) == 1
    assert audit.iloc[0].operational_date_0700 == "2026-08-19"
    assert audit.iloc[0].night_observation_label == "Unknown"


def _dense_history(*, matched: bool, temporal_verified: bool = False) -> pd.DataFrame:
    ts = pd.date_range("2026-06-28 20:00", periods=30, freq="8min", tz="Asia/Tokyo")
    data = {
        "timestamp": ts,
        "lat": [28.17 + i * 0.0001 for i in range(len(ts))],
        "lon": [129.35 + i * 0.0001 for i in range(len(ts))],
        "strict_temporal_route_verified": [temporal_verified] * len(ts),
    }
    if matched:
        data["segment_id"] = [f"seg-{i}" for i in range(len(ts))]
        data["match_distance_m"] = [8.0] * len(ts)
    return pd.DataFrame(data)


def test_dense_gps_history_is_high_candidate_but_not_strict_before_map_match():
    audit = build_night_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        gps_history=_dense_history(matched=False),
        exploration_nights=["2026-06-28"],
    )
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_HIGH
    assert row.usable_road
    assert not row.usable_road_10min_train
    assert not row.usable_road_10min_eval
    assert not row.can_generate_no_capture_observed


def test_good_anchor_map_match_alone_is_still_not_strict():
    audit = build_night_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        gps_history=_dense_history(matched=True, temporal_verified=False),
        exploration_nights=["2026-06-28"],
    )
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_HIGH
    assert not row.usable_road_10min_train
    assert not row.can_generate_no_capture_observed


def test_high_reconstruction_becomes_strict_only_after_temporal_route_verification():
    audit = build_night_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        gps_history=_dense_history(matched=True, temporal_verified=True),
        exploration_nights=["2026-06-28"],
    )
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_HIGH
    assert row.usable_road_10min_train
    assert row.usable_road_10min_eval
    assert row.can_generate_no_capture_observed


def test_high_candidate_with_poor_map_match_remains_non_strict_even_if_temporal_verified():
    hist = _dense_history(matched=True, temporal_verified=True)
    hist.loc[0:14, "segment_id"] = pd.NA
    hist.loc[0:14, "match_distance_m"] = pd.NA
    audit = build_night_audit(pd.DataFrame(), pd.DataFrame(), gps_history=hist)
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_HIGH
    assert not row.usable_road_10min_train


def test_sparse_timed_anchors_are_partial_not_strict():
    ts = pd.to_datetime(
        [
            "2026-07-08 20:00+09:00",
            "2026-07-08 20:20+09:00",
            "2026-07-08 20:40+09:00",
            "2026-07-08 21:10+09:00",
            "2026-07-08 21:30+09:00",
        ]
    )
    hist = pd.DataFrame(
        {
            "timestamp": ts,
            "lat": [28.17, 28.171, 28.172, 28.173, 28.174],
            "lon": [129.35, 129.351, 129.352, 129.353, 129.354],
        }
    )
    audit = build_night_audit(pd.DataFrame(), pd.DataFrame(), gps_history=hist)
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_PARTIAL
    assert not row.usable_road_10min_train
