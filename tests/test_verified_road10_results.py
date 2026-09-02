from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_persisted_verified_interval_results_are_consistent():
    intervals = pd.read_csv("reports/v2_reconstructed_road10_verified_intervals.csv")
    summary = json.loads(
        Path("reports/v2_reconstructed_road10_verified_summary.json").read_text(encoding="utf-8")
    )
    assert len(intervals) == summary["strict_single_10min_bin_interval_count"] == 8
    assert intervals["operational_date_0700"].nunique() == summary["strict_night_count"] == 6
    assert intervals["strict_road10_verified"].astype(bool).all()
    assert summary["whole_night_promotion"] is False
