from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_clock(date_value, clock_value):
    if pd.isna(date_value):
        return pd.NaT
    day = pd.to_datetime(str(date_value), errors="coerce")
    if pd.isna(day):
        return pd.NaT
    text = "" if pd.isna(clock_value) else str(clock_value)
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if not m:
        return pd.NaT
    hour, minute = int(m.group(1)), int(m.group(2))
    extra_day, hour = divmod(hour, 24)
    return (day + pd.Timedelta(days=extra_day, hours=hour, minutes=minute)).tz_localize("Asia/Tokyo")


def _size_code(value):
    text = "" if pd.isna(value) else str(value)
    if "極小" in text or "幼体" in text:
        return "極小"
    if "大" in text or "大型" in text:
        return "大"
    if "中" in text or "中型" in text:
        return "中"
    if "小" in text or "小型" in text:
        return "小"
    return None


def load_canonical_capture_master(root: Path) -> pd.DataFrame:
    path = root / "data" / "canonical" / "habu_capture_master_2025-09_2026-08-29.csv"
    if not path.exists():
        return pd.DataFrame()
    src = pd.read_csv(path)
    rows = []
    for r in src.itertuples(index=False):
        event_type = getattr(r, "イベント")
        if event_type not in {"捕獲", "轢死", "sighting", "roadkill_sighting", "no_capture"}:
            continue
        timestamp = _parse_clock(getattr(r, "日付"), getattr(r, "時刻"))
        count = pd.to_numeric(getattr(r, "数"), errors="coerce")
        rows.append({
            "canonical_id": str(getattr(r, "ID")),
            "timestamp": timestamp,
            "session_start": pd.NaT,
            "event_type": event_type,
            "species": str(getattr(r, "種別")),
            "individual_count": int(count) if pd.notna(count) else 0,
            "size": _size_code(getattr(r, "サイズ")),
            "wetness": None,
            "lat": pd.to_numeric(getattr(r, "緯度"), errors="coerce"),
            "lon": pd.to_numeric(getattr(r, "経度"), errors="coerce"),
            "night_date": str(getattr(r, "運用日_7時")),
            "moon_age_observed": pd.to_numeric(getattr(r, "月齢"), errors="coerce"),
            "tide_observed": None if pd.isna(getattr(r, "潮")) else str(getattr(r, "潮")),
            "tide_direction_observed": None if pd.isna(getattr(r, "上げ下げ")) else str(getattr(r, "上げ下げ")),
            "weather_observed": None if pd.isna(getattr(r, "天候")) else str(getattr(r, "天候")),
            "temperature_observed_c": pd.to_numeric(getattr(r, "気温"), errors="coerce"),
            "canonical_source": None if pd.isna(getattr(r, "ソース")) else str(getattr(r, "ソース")),
            "raw_text": f"canonical:{getattr(r, 'ID')} {event_type}",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["individual_count"] = pd.to_numeric(out.individual_count, errors="coerce").fillna(0).astype(int)
    return out


def merge_canonical_capture_master(raw_events: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    if canonical.empty:
        return raw_events.copy()
    if raw_events.empty:
        return canonical.copy()
    raw = raw_events.copy()
    # Canonical capture master is authoritative for all Habu capture outcomes.
    raw = raw[~((raw.species == "ハブ") & (raw.event_type == "捕獲"))].copy()
    cols = sorted(set(raw.columns) | set(canonical.columns))
    return pd.concat([raw.reindex(columns=cols), canonical.reindex(columns=cols)], ignore_index=True)


def match_events_preserve_all(pipeline, events: pd.DataFrame, segs) -> pd.DataFrame:
    """Road-match geolocated events while preserving events without GPS."""
    if events.empty:
        return events.copy()
    x = events.copy().reset_index(drop=True)
    x["_source_row"] = np.arange(len(x))
    lat = pd.to_numeric(x.lat, errors="coerce")
    lon = pd.to_numeric(x.lon, errors="coerce")
    has_gps = lat.notna() & lon.notna()
    with_gps = x[has_gps].copy()
    without_gps = x[~has_gps].copy()
    if not with_gps.empty:
        matched = pipeline.match_events(with_gps.set_index("_source_row"), segs)
        if "_source_row" not in matched.columns:
            matched["_source_row"] = matched.index
    else:
        matched = with_gps
    without_gps["segment_id"] = None
    without_gps["event_match_distance_m"] = np.nan
    out = pd.concat([matched, without_gps], ignore_index=True, sort=False)
    out = out.sort_values("_source_row", kind="stable").drop(columns=["_source_row"], errors="ignore").reset_index(drop=True)
    return out


def add_canonical_audit(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"capture_events": 0, "capture_individuals": 0, "gps_capture_events": 0, "gps_capture_individuals": 0, "gps_missing_capture_events": 0}
    h = events[(events.species == "ハブ") & (events.event_type == "捕獲")].copy()
    count = pd.to_numeric(h.individual_count, errors="coerce").fillna(1)
    has_gps = pd.to_numeric(h.lat, errors="coerce").notna() & pd.to_numeric(h.lon, errors="coerce").notna()
    return {
        "capture_events": int(len(h)),
        "capture_individuals": int(count.sum()),
        "gps_capture_events": int(has_gps.sum()),
        "gps_capture_individuals": int(count[has_gps].sum()),
        "gps_missing_capture_events": int((~has_gps).sum()),
    }


def build_positive_label_audit(events: pd.DataFrame, learning: pd.DataFrame) -> pd.DataFrame:
    """Trace every Habu capture through event -> road -> visit positive label."""
    h = events[(events.species == "ハブ") & (events.event_type == "捕獲")].copy()
    rows = []
    for r in h.itertuples():
        segment = getattr(r, "segment_id", None)
        timestamp = getattr(r, "timestamp", pd.NaT)
        reason = ""
        positives = 0
        nearest_seconds = np.nan
        if pd.isna(getattr(r, "lat", np.nan)) or pd.isna(getattr(r, "lon", np.nan)):
            reason = "missing_gps"
        elif not segment or pd.isna(segment):
            reason = "road_unmatched"
        elif pd.isna(timestamp):
            reason = "missing_exact_time"
        else:
            candidates = learning[learning.segment_id == segment].copy()
            if candidates.empty:
                reason = "segment_not_visited_in_gpx"
            else:
                delta = (pd.to_datetime(candidates.entered_at) - pd.Timestamp(timestamp)).dt.total_seconds().abs()
                nearest_seconds = float(delta.min()) if len(delta) else np.nan
                positives = int(((candidates.habu_capture == 1) & (delta <= 600)).sum())
                if positives == 0:
                    reason = "no_visit_within_10_minutes"
        rows.append({
            "canonical_id": getattr(r, "canonical_id", None),
            "timestamp": timestamp,
            "individual_count": int(getattr(r, "individual_count", 1) or 1),
            "lat": getattr(r, "lat", np.nan),
            "lon": getattr(r, "lon", np.nan),
            "segment_id": segment,
            "event_match_distance_m": getattr(r, "event_match_distance_m", np.nan),
            "positive_learning_rows": positives,
            "nearest_visit_delta_s": nearest_seconds,
            "audit_status": "ok" if positives > 0 else reason,
        })
    return pd.DataFrame(rows)
