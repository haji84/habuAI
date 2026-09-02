from __future__ import annotations

import pandas as pd

from habuai.evidence_policy import derive_exploration_nights_from_events


def test_self_capture_creates_exploration_night_with_0700_boundary():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-07-23 05:23:00+09:00"],
            "event_type": ["捕獲"],
            "species": ["ハブ"],
        }
    )
    assert derive_exploration_nights_from_events(events) == {"2026-07-22"}


def test_roadkill_sighting_and_weather_do_not_create_nights():
    events = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-10 23:02:00+09:00",
                "2026-05-11 00:20:00+09:00",
                "2026-05-11 01:00:00+09:00",
            ],
            "event_type": ["轢死", "目撃", "気象"],
        }
    )
    assert derive_exploration_nights_from_events(events) == set()


def test_ambiguous_capture_or_sighting_does_not_create_night():
    events = pd.DataFrame(
        {
            "timestamp": ["2025-11-07 21:00:00+09:00"],
            "event_type": ["capture_or_sighting"],
        }
    )
    assert derive_exploration_nights_from_events(events) == set()


def test_explicit_zero_is_strong_exploration_evidence():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-04-13 01:20:00+09:00"],
            "event_type": ["no_capture"],
        }
    )
    assert derive_exploration_nights_from_events(events) == {"2026-04-12"}


def test_other_person_capture_is_not_user_exploration_evidence():
    events = pd.DataFrame(
        {
            "timestamp": ["2026-08-20 22:00:00+09:00"],
            "event_type": ["捕獲"],
            "other_capture": [True],
        }
    )
    assert derive_exploration_nights_from_events(events) == set()
