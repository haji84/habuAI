from __future__ import annotations

import pandas as pd

from habuai.runtime_fixes import canonicalize_operational_night


def test_canonical_operational_night_boundary():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-07-23 00:00:00+09:00",
                    "2026-07-23 05:23:00+09:00",
                    "2026-07-23 06:59:59+09:00",
                    "2026-07-23 07:00:00+09:00",
                ]
            ),
            "night_date": ["wrong"] * 4,
        }
    )
    out = canonicalize_operational_night(df)
    assert out["night_date"].tolist() == [
        "2026-07-22",
        "2026-07-22",
        "2026-07-22",
        "2026-07-23",
    ]
    assert out["operational_date_0700"].tolist() == out["night_date"].tolist()


def test_session_start_calendar_date_cannot_override_0700_rule():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-23 05:23:00", tz="Asia/Tokyo")],
            "session_start": [pd.Timestamp("2026-07-23 00:30:00", tz="Asia/Tokyo")],
            "night_date": ["2026-07-23"],
        }
    )
    out = canonicalize_operational_night(df)
    assert out.iloc[0].night_date == "2026-07-22"
