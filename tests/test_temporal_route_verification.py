from __future__ import annotations

import pandas as pd

from habuai.temporal_route_verification import verify_reconstructed_intervals


def test_unique_route_inside_same_10min_bin_is_strict():
    intervals = pd.DataFrame(
        {
            "start_time": ["2026-07-26 01:44:00+09:00"],
            "end_time": ["2026-07-26 01:49:00+09:00"],
            "route_class": ["確定一本道区間"],
        }
    )
    row = verify_reconstructed_intervals(intervals).iloc[0]
    assert bool(row.strict_road10_verified)
    assert bool(row.can_generate_no_capture_observed)
    assert row.verification_reason == "unique_route_inside_single_10min_bin"


def test_unique_route_crossing_10min_boundary_is_not_strict():
    intervals = pd.DataFrame(
        {
            "start_time": ["2026-07-26 01:53:00+09:00"],
            "end_time": ["2026-07-26 02:02:00+09:00"],
            "route_class": ["確定一本道区間"],
        }
    )
    row = verify_reconstructed_intervals(intervals).iloc[0]
    assert not bool(row.strict_road10_verified)
    assert row.verification_reason == "crosses_10min_bin"


def test_alternative_route_is_not_strict_even_inside_one_bin():
    intervals = pd.DataFrame(
        {
            "start_time": ["2026-07-26 02:31:00+09:00"],
            "end_time": ["2026-07-26 02:35:00+09:00"],
            "route_class": ["高信頼候補（代替経路あり）"],
        }
    )
    row = verify_reconstructed_intervals(intervals).iloc[0]
    assert not bool(row.strict_road10_verified)
    assert row.verification_reason == "route_not_unique"
