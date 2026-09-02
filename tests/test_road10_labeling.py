from __future__ import annotations

import pandas as pd

from habuai.road10_labeling import label_verified_road10_intervals


def _interval() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": ["2026-06-18 23:54:00+09:00"],
            "end_time": ["2026-06-18 23:55:00+09:00"],
            "strict_road10_verified": [True],
        }
    )


def test_himehabu_does_not_block_main_habu_no_capture_observed():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-06-18 23:55:00+09:00"],
            "event_type": ["目撃"],
            "species": ["ヒメハブ"],
        }
    )
    row = label_verified_road10_intervals(_interval(), events).iloc[0]
    assert row.label == "NO_CAPTURE_OBSERVED"
    assert row.main_habu_event_count == 0


def test_main_habu_capture_inside_interval_is_positive():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-06-18 23:54:30+09:00"],
            "event_type": ["捕獲"],
            "species": ["ハブ"],
        }
    )
    row = label_verified_road10_intervals(_interval(), events).iloc[0]
    assert row.label == "Positive"
    assert row.main_habu_event_count == 1


def test_main_habu_sighting_inside_interval_blocks_negative():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-06-18 23:54:30+09:00"],
            "event_type": ["目撃"],
            "species": ["ハブ"],
        }
    )
    row = label_verified_road10_intervals(_interval(), events).iloc[0]
    assert row.label == "Positive"


def test_non_strict_interval_is_rejected_before_labeling():
    interval = _interval()
    interval["strict_road10_verified"] = False
    events = pd.DataFrame(
        {
            "timestamp": ["2026-06-18 23:55:00+09:00"],
            "event_type": ["目撃"],
            "species": ["ヒメハブ"],
        }
    )
    try:
        label_verified_road10_intervals(interval, events)
    except ValueError as exc:
        assert "strict_road10_verified" in str(exc)
    else:
        raise AssertionError("non-strict interval must not be labeled")
