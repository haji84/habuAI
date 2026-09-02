from __future__ import annotations

import pandas as pd

from habuai.evidence_policy import derive_dense_field_sequence_nights


def test_dense_multi_place_sequence_confirms_exploration_night():
    events = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-02 19:46:00+09:00",
                "2026-05-02 20:01:00+09:00",
                "2026-05-02 20:41:00+09:00",
                "2026-05-02 21:07:00+09:00",
                "2026-05-02 21:50:00+09:00",
            ],
            "event_type": ["目撃", "目撃", "目撃", "目撃", "目撃"],
            "place": ["大和村県道79号", "大和村県道79号", "宇検村芦検", "宇検村県道85号", "瀬戸内町県道612号"],
        }
    )
    assert derive_dense_field_sequence_nights(events) == {"2026-05-02"}


def test_single_or_short_sequence_remains_weak():
    events = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-10 23:02:00+09:00",
                "2026-05-10 23:10:00+09:00",
                "2026-05-10 23:20:00+09:00",
                "2026-05-10 23:30:00+09:00",
            ],
            "event_type": ["轢死", "目撃", "目撃", "気象"],
            "place": ["A", "B", "C", "D"],
        }
    )
    assert derive_dense_field_sequence_nights(events) == set()


def test_after_midnight_sequence_uses_previous_operational_night():
    events = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-27 00:58:00+09:00",
                "2026-07-27 01:01:00+09:00",
                "2026-07-27 01:17:00+09:00",
                "2026-07-27 01:29:00+09:00",
                "2026-07-27 01:53:00+09:00",
            ],
            "event_type": ["目撃"] * 5,
            "place": ["嘉徳A", "嘉徳B", "嘉徳C", "嘉徳D", "国道58号"],
        }
    )
    assert derive_dense_field_sequence_nights(events) == {"2026-07-26"}
