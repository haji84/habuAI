from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from habuai.audit_v2 import operational_date_0700

# Event types that are strong enough, by themselves, to prove that an exploration
# night occurred. Ordinary sightings, roadkill and weather observations are not.
SELF_CAPTURE_EVENT_TYPES = {"捕獲", "capture"}
EXPLICIT_ZERO_EVENT_TYPES = {"no_capture", "捕獲なし", "探索ゼロ", "zero_capture"}
FIELD_SEQUENCE_EVENT_TYPES = {
    "目撃",
    "sighting",
    "轢死",
    "roadkill_sighting",
    "気象",
    "weather",
}

STATUS_CONFIRMED = "CONFIRMED"
STATUS_SOURCE_CONFLICT = "SOURCE_CONFLICT"
STATUS_DUPLICATE_SUSPECT = "DUPLICATE_SUSPECT"
STATUS_WEAK_EVENT_ONLY = "WEAK_EVENT_ONLY"
QUARANTINE_STATUSES = {
    STATUS_SOURCE_CONFLICT,
    STATUS_DUPLICATE_SUSPECT,
    STATUS_WEAK_EVENT_ONLY,
}


def derive_exploration_nights_from_events(
    events: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    other_capture_col: str | None = "other_capture",
) -> set[str]:
    """Return operational nights proven by strong event evidence.

    Strong event evidence is limited to the user's own confirmed capture or an
    explicit zero-capture/search-night record. A single roadkill, sighting, weather
    event, or ambiguous `capture_or_sighting` record cannot create a canonical
    exploration night by itself.

    Dense multi-event field sequences are handled separately by
    `derive_dense_field_sequence_nights`.
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


def derive_dense_field_sequence_nights(
    events: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    place_col: str | None = "place",
    min_events: int = 5,
    min_span_minutes: float = 45.0,
    min_distinct_places: int = 2,
) -> set[str]:
    """Infer exploration nights from a dense, time-ordered field logging sequence.

    A lone sighting is weak evidence. Repeated field observations over a sustained
    night-time interval, especially across multiple places/roads, are qualitatively
    different: they document the user's movement and active field logging.

    Conservative defaults require at least five eligible field events spanning at
    least 45 minutes. If a place column is available, at least two distinct places
    are required. The operational-night key always uses the canonical 07:00 JST
    boundary.
    """
    if events.empty or timestamp_col not in events.columns or event_type_col not in events.columns:
        return set()

    e = events.copy()
    e = e[e[event_type_col].astype(str).isin(FIELD_SEQUENCE_EVENT_TYPES)].copy()
    if e.empty:
        return set()

    e["_ts"] = e[timestamp_col].map(pd.Timestamp)
    e = e[e["_ts"].notna()].copy()
    if e.empty:
        return set()

    e["operational_date_0700"] = e["_ts"].map(
        lambda value: operational_date_0700(value).isoformat()
    )

    confirmed: set[str] = set()
    for night, group in e.groupby("operational_date_0700"):
        group = group.sort_values("_ts")
        if len(group) < min_events:
            continue
        span_minutes = (group["_ts"].max() - group["_ts"].min()).total_seconds() / 60.0
        if span_minutes < min_span_minutes:
            continue
        if place_col and place_col in group.columns:
            places = {
                str(value).strip()
                for value in group[place_col].dropna()
                if str(value).strip()
            }
            if len(places) < min_distinct_places:
                continue
        confirmed.add(str(night))
    return confirmed


def load_population_conflicts(path: Path | str | None) -> pd.DataFrame:
    """Load a reviewed conflict registry for historical recovered nights."""
    columns = [
        "operational_date_0700",
        "status",
        "include_by_default",
        "reason",
        "source_evidence",
    ]
    if path is None:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"population conflict registry missing columns: {missing}")
    frame["operational_date_0700"] = frame["operational_date_0700"].astype(str)
    frame["status"] = frame["status"].astype(str)
    frame["include_by_default"] = (
        frame["include_by_default"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    return frame[columns].copy()


def quarantine_unresolved_nights(
    nights: Iterable[str], conflicts: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """Remove unresolved recovered nights unless independently resolved for inclusion.

    This is deliberately conservative. It prevents one corrected workbook from
    silently overriding conflicting historical provenance. A night can later be
    promoted by changing the reviewed registry to `include_by_default=true` after
    its source evidence has been resolved.
    """
    night_set = {str(value) for value in nights}
    if conflicts.empty:
        return sorted(night_set), conflicts.copy()

    unresolved = conflicts[
        conflicts["status"].isin(QUARANTINE_STATUSES)
        & ~conflicts["include_by_default"]
        & conflicts["operational_date_0700"].isin(night_set)
    ].copy()
    night_set.difference_update(unresolved["operational_date_0700"].astype(str))
    return sorted(night_set), unresolved


def merge_exploration_night_sources(*sources: Iterable[str]) -> list[str]:
    """Merge provenance-derived night sets without forcing a historical count."""
    merged: set[str] = set()
    for source in sources:
        merged.update(str(value) for value in source)
    return sorted(merged)
