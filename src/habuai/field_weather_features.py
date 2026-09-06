from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

JST = "Asia/Tokyo"
OPERATIONAL_BOUNDARY_HOUR = 7

_TIMESTAMP_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_INLINE_RE = re.compile(r"なう\((\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}:\d{2})\)([^\n]*)")
_RAIN_TOKENS = {"小雨", "雨", "大雨", "雨後"}


def _to_jst_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(JST)
    return ts.tz_convert(JST)


def _night_date(ts: pd.Timestamp) -> str:
    return (ts - pd.Timedelta(hours=OPERATIONAL_BOUNDARY_HOUR)).date().isoformat()


def parse_field_weather_markers(path: Path) -> pd.DataFrame:
    """Extract explicit rain-state markers from a field log.

    Supports both timestamp blocks such as `2026/09/05 23:47` followed by `雨`
    and inline observations such as `なう(2026/09/06 01:25:48)雨`.
    The result is observational context only; it is not assumed to be a complete
    precipitation record.
    """
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "marker", "night_date", "source_file"])

    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    rows: list[dict] = []
    current_ts: pd.Timestamp | None = None

    for line in lines:
        m = _TIMESTAMP_RE.match(line)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            current_ts = pd.Timestamp(
                year=int(y), month=int(mo), day=int(d), hour=int(hh), minute=int(mm),
                second=int(ss or 0), tz=JST,
            )
            continue

        if current_ts is not None and line in _RAIN_TOKENS:
            rows.append({
                "timestamp": current_ts,
                "marker": line,
                "night_date": _night_date(current_ts),
                "source_file": path.name,
            })

        for inline in _INLINE_RE.finditer(line):
            suffix = inline.group(2).strip()
            marker = next((token for token in ["大雨", "小雨", "雨後", "雨"] if suffix.startswith(token)), None)
            if marker is None:
                continue
            ts = _to_jst_timestamp(inline.group(1))
            rows.append({
                "timestamp": ts,
                "marker": marker,
                "night_date": _night_date(ts),
                "source_file": path.name,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "marker", "night_date", "source_file"])
    out = out.drop_duplicates(["timestamp", "marker", "source_file"]).sort_values("timestamp").reset_index(drop=True)
    return out


def add_field_rain_candidate_features(visits: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Add non-production candidate rain-transition features to observed visits.

    These columns are intentionally not added to the model feature list yet. They
    exist so complete-GPX nights can be compared consistently before deciding
    whether a rain-transition effect generalizes beyond a single night.
    """
    if visits is None or visits.empty:
        return visits.copy() if visits is not None else pd.DataFrame()

    out = visits.copy()
    entered = pd.to_datetime(out["entered_at"], errors="coerce")

    def to_jst(value):
        if pd.isna(value):
            return pd.NaT
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            return stamp.tz_localize(JST)
        return stamp.tz_convert(JST)

    entered_jst = entered.map(to_jst)
    out["operational_date_0700"] = entered_jst.map(lambda x: _night_date(x) if not pd.isna(x) else None)
    out["field_minutes_since_rain_marker"] = np.nan
    out["field_weather_marker_available"] = 0

    if markers is None or markers.empty:
        for label in ["0_10m", "10_20m", "20_30m"]:
            out[f"field_rain_transition_{label}"] = 0
        return out

    m = markers.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")

    for night, row_idx in out.groupby("operational_date_0700").groups.items():
        night_markers = m[m["night_date"] == night].sort_values("timestamp")
        if night_markers.empty:
            continue
        marker_times = list(night_markers["timestamp"])
        for idx in row_idx:
            t = entered_jst.loc[idx]
            if pd.isna(t):
                continue
            prior = [mt for mt in marker_times if mt <= t]
            if not prior:
                continue
            delta = (t - prior[-1]).total_seconds() / 60.0
            out.loc[idx, "field_minutes_since_rain_marker"] = delta
            out.loc[idx, "field_weather_marker_available"] = 1

    mins = out["field_minutes_since_rain_marker"]
    out["field_rain_transition_0_10m"] = ((mins >= 0) & (mins <= 10)).astype(int)
    out["field_rain_transition_10_20m"] = ((mins > 10) & (mins <= 20)).astype(int)
    out["field_rain_transition_20_30m"] = ((mins > 20) & (mins <= 30)).astype(int)
    return out


def apply_field_weather_candidate_features(pipeline) -> None:
    """Patch the runtime to retain explicit field-rain timing as candidate features."""
    marker_frames: list[pd.DataFrame] = []
    original_parse = pipeline.parse_field_log

    def parse_with_markers(path):
        marker_frames.append(parse_field_weather_markers(Path(path)))
        return original_parse(path)

    pipeline.parse_field_log = parse_with_markers
    original_add = pipeline.add_outcomes_and_bio

    def add_with_field_weather(visits, events, cfg):
        base = original_add(visits, events, cfg)
        markers = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()
        return add_field_rain_candidate_features(base, markers)

    pipeline.add_outcomes_and_bio = add_with_field_weather
