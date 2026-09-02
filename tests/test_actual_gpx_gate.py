from __future__ import annotations

import pandas as pd

from habuai.actual_gpx_gate import enforce_actual_gpx_map_match_gate
from habuai.audit_v2 import build_night_audit


def _actual_gpx(*, matched_ratio: float) -> pd.DataFrame:
    ts = pd.date_range("2026-09-02 22:00", periods=120, freq="1min", tz="Asia/Tokyo")
    matched = int(len(ts) * matched_ratio)
    segment_id = [f"seg-{i}" if i < matched else pd.NA for i in range(len(ts))]
    return pd.DataFrame(
        {
            "session_file": ["2026-09-02-探索.gpx"] * len(ts),
            "timestamp": ts,
            "lat": [28.17] * len(ts),
            "lon": [129.35] * len(ts),
            "segment_id": segment_id,
        }
    )


def test_actual_gpx_without_map_match_is_not_strict():
    gpx = _actual_gpx(matched_ratio=0.0)
    audit = build_night_audit(gpx, pd.DataFrame())
    gated = enforce_actual_gpx_map_match_gate(audit, gpx)
    row = gated.iloc[0]
    assert row.classification == "ACTUAL_GPX"
    assert not row.usable_road_10min_train
    assert not row.usable_road_10min_eval
    assert not row.can_generate_no_capture_observed
    assert row.night_observation_label == "Unknown"


def test_actual_gpx_below_80_percent_map_match_is_not_strict():
    gpx = _actual_gpx(matched_ratio=0.79)
    audit = build_night_audit(gpx, pd.DataFrame())
    gated = enforce_actual_gpx_map_match_gate(audit, gpx)
    row = gated.iloc[0]
    assert row.actual_gpx_match_ratio < 0.80
    assert not row.usable_road_10min_train


def test_actual_gpx_at_80_percent_map_match_can_be_strict():
    gpx = _actual_gpx(matched_ratio=0.80)
    audit = build_night_audit(gpx, pd.DataFrame())
    gated = enforce_actual_gpx_map_match_gate(audit, gpx)
    row = gated.iloc[0]
    assert row.actual_gpx_match_ratio >= 0.80
    assert row.usable_road_10min_train
    assert row.usable_road_10min_eval
    assert row.can_generate_no_capture_observed
