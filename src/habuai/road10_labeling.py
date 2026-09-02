from __future__ import annotations

import pandas as pd

MAIN_HABU = "ハブ"
MAIN_HABU_EVENT_TYPES = {"捕獲", "capture", "目撃", "sighting", "轢死", "roadkill_sighting"}


def _to_jst(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Tokyo")
    return ts.tz_convert("Asia/Tokyo")


def label_verified_road10_intervals(
    intervals: pd.DataFrame,
    events: pd.DataFrame,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
    event_time_col: str = "timestamp",
    event_type_col: str = "event_type",
    species_col: str = "species",
) -> pd.DataFrame:
    """Label already-verified reconstructed intervals for the main Habu target.

    Inputs must already have passed strict route/time verification. Any main-Habu
    capture/sighting/roadkill event inside the inclusive anchor-time interval makes
    the interval Positive. If none exists, the interval is NO_CAPTURE_OBSERVED.

    Himehabu and other species do not block a main-Habu negative label. The result
    remains an observation label, never a claim that no Habu was biologically present.
    """
    required_intervals = {start_col, end_col, "strict_road10_verified"}
    missing = required_intervals.difference(intervals.columns)
    if missing:
        raise ValueError(f"verified intervals missing required columns: {sorted(missing)}")
    if not intervals["strict_road10_verified"].fillna(False).astype(bool).all():
        raise ValueError("all input intervals must already be strict_road10_verified")

    required_events = {event_time_col, event_type_col, species_col}
    missing_events = required_events.difference(events.columns)
    if missing_events:
        raise ValueError(f"events missing required columns: {sorted(missing_events)}")

    out = intervals.copy()
    out[start_col] = out[start_col].map(_to_jst)
    out[end_col] = out[end_col].map(_to_jst)

    e = events.copy()
    e[event_time_col] = e[event_time_col].map(_to_jst)
    main = e[
        e[species_col].astype(str).eq(MAIN_HABU)
        & e[event_type_col].astype(str).isin(MAIN_HABU_EVENT_TYPES)
    ].copy()

    labels: list[str] = []
    counts: list[int] = []
    event_types: list[str] = []
    for _, row in out.iterrows():
        mask = (main[event_time_col] >= row[start_col]) & (main[event_time_col] <= row[end_col])
        hits = main.loc[mask]
        count = int(len(hits))
        counts.append(count)
        if count:
            labels.append("Positive")
            event_types.append(" | ".join(sorted(set(hits[event_type_col].astype(str)))))
        else:
            labels.append("NO_CAPTURE_OBSERVED")
            event_types.append("")

    out["main_habu_event_count"] = counts
    out["label"] = labels
    out["main_habu_event_types"] = event_types
    return out
