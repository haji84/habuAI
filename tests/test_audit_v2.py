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


def test_actual_gpx_no_capture_becomes_no_capture_observed():
    ts = pd.date_range("2026-08-20 20:00", periods=120, freq="1min", tz="Asia/Tokyo")
    gpx = pd.DataFrame(
        {
            "session_file": ["night.gpx"] * len(ts),
            "timestamp": ts,
            "lat": [28.17] * len(ts),
            "lon": [129.35] * len(ts),
            "segment_id": ["seg-1"] * len(ts),
        }
    )
    audit = build_night_audit(gpx, pd.DataFrame(), exploration_nights=["2026-08-20"])
    row = audit.iloc[0]
    assert row.classification == CLASS_ACTUAL_GPX
    assert bool(row.can_generate_no_capture_observed)
    assert row.night_observation_label == "NO_CAPTURE_OBSERVED"


def test_event_only_night_is_not_automatically_zero_observed():
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
    row = audit.iloc[0]
    assert row.classification == CLASS_SPATIAL_ONLY
    assert row.capture_count == 0
    assert row.night_observation_label == "Unknown"


def test_dense_gps_history_is_high_reconstruction_candidate():
    ts = pd.date_range("2026-06-28 20:00", periods=30, freq="8min", tz="Asia/Tokyo")
    hist = pd.DataFrame(
        {
            "timestamp": ts,
            "lat": [28.17 + i * 0.0001 for i in range(len(ts))],
            "lon": [129.35 + i * 0.0001 for i in range(len(ts))],
        }
    )
    audit = build_night_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        gps_history=hist,
        exploration_nights=["2026-06-28"],
    )
    row = audit.iloc[0]
    assert row.classification == CLASS_RECONSTRUCTED_HIGH
    assert row.usable_road_10min_train


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
