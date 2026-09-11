from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

JST = "Asia/Tokyo"
OPERATIONAL_BOUNDARY_HOUR = 7

_TIMESTAMP_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_INLINE_RE = re.compile(r"なう\((\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}:\d{2})\)([^\n]*)")
_RAIN_ONSET_TOKENS = {"小雨", "雨", "大雨"}
_RAIN_RECOVERY_TOKENS = {"雨後", "雨やみ", "雨止み", "雨やむ", "雨止む"}
_RAIN_TOKENS = _RAIN_ONSET_TOKENS | _RAIN_RECOVERY_TOKENS


def _to_jst_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(JST)
    return ts.tz_convert(JST)


def _night_date(ts: pd.Timestamp) -> str:
    return (ts - pd.Timedelta(hours=OPERATIONAL_BOUNDARY_HOUR)).date().isoformat()


def _marker_phase(marker: str) -> str:
    if marker in _RAIN_ONSET_TOKENS:
        return "rain_onset"
    if marker in _RAIN_RECOVERY_TOKENS:
        return "rain_recovery"
    return "unknown"


def parse_field_weather_markers(path: Path) -> pd.DataFrame:
    """Extract explicit field rain-state markers without inferring missing weather.

    Rain-onset markers (小雨/雨/大雨) and rain-recovery markers
    (雨後/雨やみ/雨止み/雨やむ/雨止む) are retained separately. The result is
    observational context only and is not assumed to be a complete precipitation
    record.
    """
    columns = ["timestamp", "marker", "phase", "night_date", "source_file"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

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
                "phase": _marker_phase(line),
                "night_date": _night_date(current_ts),
                "source_file": path.name,
            })

        for inline in _INLINE_RE.finditer(line):
            suffix = inline.group(2).strip()
            ordered_tokens = ["大雨", "小雨", "雨後", "雨やみ", "雨止み", "雨やむ", "雨止む", "雨"]
            marker = next((token for token in ordered_tokens if suffix.startswith(token)), None)
            if marker is None:
                continue
            ts = _to_jst_timestamp(inline.group(1))
            rows.append({
                "timestamp": ts,
                "marker": marker,
                "phase": _marker_phase(marker),
                "night_date": _night_date(ts),
                "source_file": path.name,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.drop_duplicates(["timestamp", "marker", "source_file"]).sort_values("timestamp").reset_index(drop=True)
    return out[columns]


def add_field_rain_candidate_features(visits: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Add causal, non-production field rain candidate features.

    Only markers at or before each visit's entered_at are used. Rain onset and
    rain recovery are tracked by separate timers. Legacy aggregate transition
    columns are retained for comparison, but production model features remain
    unchanged.
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

    timer_columns = [
        "field_minutes_since_rain_marker",
        "field_minutes_since_rain_onset",
        "field_minutes_since_rain_recovery",
    ]
    for col in timer_columns:
        out[col] = np.nan

    out["field_weather_marker_available"] = 0
    out["field_rain_onset_available"] = 0
    out["field_rain_recovery_available"] = 0

    def add_windows(prefix: str, mins: pd.Series) -> None:
        out[f"{prefix}_0_10m"] = ((mins >= 0) & (mins <= 10)).astype(int)
        out[f"{prefix}_10_20m"] = ((mins > 10) & (mins <= 20)).astype(int)
        out[f"{prefix}_20_30m"] = ((mins > 20) & (mins <= 30)).astype(int)
        out[f"{prefix}_30_60m"] = ((mins > 30) & (mins <= 60)).astype(int)
        out[f"{prefix}_gt_60m"] = (mins > 60).astype(int)

    if markers is None or markers.empty:
        for prefix in ["field_rain_transition", "field_rain_onset", "field_rain_recovery"]:
            add_windows(prefix, pd.Series(np.nan, index=out.index))
        return out

    m = markers.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    if "phase" not in m.columns:
        m["phase"] = m["marker"].map(_marker_phase)

    for night, row_idx in out.groupby("operational_date_0700").groups.items():
        night_markers = m[m["night_date"] == night].sort_values("timestamp")
        if night_markers.empty:
            continue

        all_markers = list(zip(night_markers["timestamp"], night_markers["phase"]))
        for idx in row_idx:
            t = entered_jst.loc[idx]
            if pd.isna(t):
                continue

            prior_all = [(mt, phase) for mt, phase in all_markers if mt <= t]
            if prior_all:
                delta = (t - prior_all[-1][0]).total_seconds() / 60.0
                out.loc[idx, "field_minutes_since_rain_marker"] = delta
                out.loc[idx, "field_weather_marker_available"] = 1

            prior_onset = [mt for mt, phase in prior_all if phase == "rain_onset"]
            if prior_onset:
                out.loc[idx, "field_minutes_since_rain_onset"] = (t - prior_onset[-1]).total_seconds() / 60.0
                out.loc[idx, "field_rain_onset_available"] = 1

            prior_recovery = [mt for mt, phase in prior_all if phase == "rain_recovery"]
            if prior_recovery:
                out.loc[idx, "field_minutes_since_rain_recovery"] = (t - prior_recovery[-1]).total_seconds() / 60.0
                out.loc[idx, "field_rain_recovery_available"] = 1

    add_windows("field_rain_transition", out["field_minutes_since_rain_marker"])
    add_windows("field_rain_onset", out["field_minutes_since_rain_onset"])
    add_windows("field_rain_recovery", out["field_minutes_since_rain_recovery"])
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
