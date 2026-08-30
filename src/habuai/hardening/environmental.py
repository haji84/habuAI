from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYNODIC_MONTH_DAYS = 29.53058867
NEW_MOON_EPOCH_UTC = pd.Timestamp("2000-01-06T18:14:00Z")
JMA_TIDE_STATION = "O9"  # 奄美（名瀬小湊）
JMA_TIDE_URL = "https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/O9.txt"


def add_lunar_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "entered_at" not in data:
        return data
    x = data.copy()
    t = pd.to_datetime(x.entered_at, utc=True, errors="coerce")
    days = (t - NEW_MOON_EPOCH_UTC).dt.total_seconds() / 86400.0
    age = np.mod(days, SYNODIC_MONTH_DAYS)
    phase = age / SYNODIC_MONTH_DAYS
    x["moon_age_days"] = age
    x["moon_phase_sin"] = np.sin(2 * np.pi * phase)
    x["moon_phase_cos"] = np.cos(2 * np.pi * phase)
    x["moon_illumination"] = (1 - np.cos(2 * np.pi * phase)) / 2
    return x


def add_weather_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    x = data.copy()
    if "weather_code" in x:
        code = pd.to_numeric(x.weather_code, errors="coerce")
        x["fog_wmo_flag"] = code.isin([45, 48]).astype(int)
    else:
        x["fog_wmo_flag"] = 0
    if {"temperature_c", "dew_point_c"}.issubset(x.columns):
        spread = pd.to_numeric(x.temperature_c, errors="coerce") - pd.to_numeric(x.dew_point_c, errors="coerce")
        x["temp_dewpoint_spread_c"] = spread
        humidity = pd.to_numeric(x.get("humidity_pct"), errors="coerce") if "humidity_pct" in x else pd.Series(np.nan, index=x.index)
        x["fog_proxy_flag"] = ((spread <= 1.5) & (humidity >= 95)).astype(int)
    else:
        x["fog_proxy_flag"] = 0
    if "temperature_c" in x and "session_file" in x:
        x = x.sort_values(["session_file", "entered_at"]).copy()
        x["temperature_change_3visits_c"] = x.groupby("session_file")["temperature_c"].diff(3)
    return x


def _parse_jma_tide_text(text: str, year: int) -> pd.DataFrame:
    """Parse JMA fixed-width tide-table text.

    JMA format: columns 1-72 are 24 hourly tide heights (3 chars each),
    73-78 are YY MM DD, and 79-80 are the station code.
    """
    rows = []
    for line in text.splitlines():
        if len(line) < 80:
            continue
        station = line[78:80]
        if station != JMA_TIDE_STATION:
            continue
        try:
            yy = int(line[72:74]); month = int(line[74:76]); day = int(line[76:78])
        except ValueError:
            continue
        full_year = 2000 + yy if yy < 70 else 1900 + yy
        if full_year != year:
            continue
        for hour in range(24):
            token = line[hour * 3:(hour + 1) * 3]
            try:
                height = int(token)
            except ValueError:
                continue
            ts = pd.Timestamp(full_year, month, day, hour, tz="Asia/Tokyo")
            rows.append({"timestamp": ts, "tide_height_cm": float(height)})
    tide = pd.DataFrame(rows)
    if tide.empty:
        return tide
    tide = tide.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    tide["tide_change_1h_cm"] = tide.tide_height_cm.diff()
    tide["tide_state_code"] = np.sign(tide.tide_change_1h_cm)
    # Distance in time from a change in slope, used as an hourly proxy for nearest high/low tide.
    slope = tide.tide_state_code.replace(0, np.nan).ffill().bfill()
    turning = slope.ne(slope.shift())
    turn_times = tide.loc[turning, "timestamp"]
    if len(turn_times):
        turn_ns = turn_times.astype("int64").to_numpy()
        now_ns = tide.timestamp.astype("int64").to_numpy()
        tide["minutes_to_nearest_turning_tide"] = [float(np.min(np.abs(turn_ns - n)) / 60_000_000_000) for n in now_ns]
    else:
        tide["minutes_to_nearest_turning_tide"] = np.nan
    return tide


def _fetch_jma_tide(years) -> pd.DataFrame:
    parts = []
    for year in sorted(set(int(y) for y in years)):
        try:
            r = requests.get(JMA_TIDE_URL.format(year=year), timeout=30)
            r.raise_for_status()
            parsed = _parse_jma_tide_text(r.text, year)
            if not parsed.empty:
                parts.append(parsed)
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_tide(data: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, str]:
    local = root / "data" / "canonical" / "tide_hourly.csv"
    if local.exists():
        tide = pd.read_csv(local)
        tide["timestamp"] = pd.to_datetime(tide.timestamp, errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
        tide["tide_height_cm"] = pd.to_numeric(tide.get("tide_height_cm"), errors="coerce")
        if "tide_state_code" not in tide:
            if "tide_state" in tide:
                tide["tide_state_code"] = tide.tide_state.map({"上げ": 1.0, "下げ": -1.0, "満潮": 0.0, "干潮": 0.0})
            else:
                tide["tide_state_code"] = np.sign(tide.tide_height_cm.diff())
        if "tide_change_1h_cm" not in tide:
            tide["tide_change_1h_cm"] = tide.tide_height_cm.diff()
        if "minutes_to_nearest_turning_tide" not in tide:
            tide["minutes_to_nearest_turning_tide"] = np.nan
        return tide, "local_authoritative"
    entered = pd.to_datetime(data.entered_at, errors="coerce", utc=True)
    years = entered.dropna().dt.year.unique().tolist()
    return _fetch_jma_tide(years), "jma_O9_predicted"


def join_optional_tide_features(data: pd.DataFrame, root: Path) -> pd.DataFrame:
    if data.empty:
        return data
    x = data.copy()
    tide, source = _load_tide(x, root)
    if tide.empty:
        x["tide_height_cm"] = np.nan
        x["tide_change_1h_cm"] = np.nan
        x["tide_state_code"] = np.nan
        x["minutes_to_nearest_turning_tide"] = np.nan
        x["tide_source_available"] = 0
        x["tide_source"] = "unavailable"
        return x
    x["entered_at"] = pd.to_datetime(x.entered_at, errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    tide = tide.dropna(subset=["timestamp"]).sort_values("timestamp")
    x = pd.merge_asof(
        x.sort_values("entered_at"),
        tide[["timestamp", "tide_height_cm", "tide_change_1h_cm", "tide_state_code", "minutes_to_nearest_turning_tide"]],
        left_on="entered_at",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("40min"),
    ).drop(columns=["timestamp"], errors="ignore")
    x["tide_source_available"] = x.tide_height_cm.notna().astype(int)
    x["tide_source"] = source
    return x
