from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from habuai.audit_v2 import operational_date_0700

# Event types that are strong enough, by themselves, to prove that an exploration
# night occurred. Ordinary sightings, roadkill and weather observations are not.
SELF_CAPTURE_EVENT_TYPES = {"捕獲", "capture"}
EXPLICIT_ZERO_EVENT_TYPES = {"no_capture", "捕獲なし", "探索ゼロ", "zero_capture"}


def derive_exploration_nights_from_events(
    events: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    other_capture_col: str | None = "other_capture",
) -> set[str]:
    """Return operational nights proven by strong event evidence.

    Strong event evidence is limited to the user's own confirmed capture or an
    explicit zero-capture/search-night record. A roadkill, sighting, weather event,
    or ambiguous `capture_or_sighting` record cannot create a canonical exploration
    night by itself.

    When `other_capture_col` is present, rows marked as another person's capture
    are excluded from the population evidence.
    """
    if events.empty or timestamp_col not in events.columns:
        return set()

    e = events.copy()
    if event_type_col not in e.columns:
        return set()

    event_type = e[event_type_col].astype(str)
    strong = event_type.isin(SELF_CAPTURE_EVENT_TYPES | EXPLICIT_ZERO_EVENT_TYPES)

    if other_capture_col and other_capture_col in e.columns:
        other = e[other_capture_col]
        if other.dtype == bool:
            strong &= ~other.fillna(False)
        else:
            normalized = other.astype(str).str.strip().str.lower()
            strong &= ~normalized.isin({"true", "1", "y", "yes", "他者捕獲"})

    nights = {
        operational_date_0700(value).isoformat()
        for value in e.loc[strong, timestamp_col].dropna()
    }
    return nights


def merge_exploration_night_sources(*sources: Iterable[str]) -> list[str]:
    """Merge provenance-derived night sets without forcing a historical count."""
    merged: set[str] = set()
    for source in sources:
        merged.update(str(value) for value in source)
    return sorted(merged)
