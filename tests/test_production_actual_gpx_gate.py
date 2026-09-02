from __future__ import annotations

import pandas as pd

from habuai.runtime_fixes import enforce_actual_gpx_session_match_gate


def _frame(matched_count: int, total: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_file": ["night.gpx"] * total,
            "segment_id": [f"seg-{i}" if i < matched_count else pd.NA for i in range(total)],
        }
    )


def test_79pct_or_less_session_is_fully_invalidated():
    out = enforce_actual_gpx_session_match_gate(_frame(7, 10))
    assert out["session_map_match_ratio"].iloc[0] == 0.7
    assert not out["strict_session_eligible"].any()
    assert out["segment_id"].isna().all()


def test_80pct_session_remains_available_to_segment_visits():
    out = enforce_actual_gpx_session_match_gate(_frame(8, 10))
    assert out["session_map_match_ratio"].iloc[0] == 0.8
    assert out["strict_session_eligible"].all()
    assert out["segment_id"].notna().sum() == 8


def test_gate_is_per_session_not_global():
    good = _frame(9, 10)
    good["session_file"] = "good.gpx"
    bad = _frame(4, 10)
    bad["session_file"] = "bad.gpx"
    out = enforce_actual_gpx_session_match_gate(pd.concat([good, bad], ignore_index=True))
    assert out.loc[out.session_file == "good.gpx", "segment_id"].notna().sum() == 9
    assert out.loc[out.session_file == "bad.gpx", "segment_id"].isna().all()
