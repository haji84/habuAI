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
        rows.append({
            "canonical_id": str(getattr(r, "ID")),
            "timestamp": timestamp,
            "session_start": pd.NaT,
            "event_type": event_type,
            "species": str(getattr(r, "種別")),
            "individual_count": int(pd.to_numeric(getattr(r, "数"), errors="coerce") if pd.notna(getattr(r, "数")) else 0),
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
