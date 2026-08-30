from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SYNODIC_MONTH_DAYS = 29.53058867
NEW_MOON_EPOCH_UTC = pd.Timestamp("2000-01-06T18:14:00Z")


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
    if "temperature_c" in x and "session_file" in x:
        x = x.sort_values(["session_file", "entered_at"]).copy()
        x["temperature_change_3visits_c"] = x.groupby("session_file")["temperature_c"].diff(3)
    return x


def join_optional_tide_features(data: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Join measured/authoritative tide data when available.

    Expected optional file: data/canonical/tide_hourly.csv
    Columns: timestamp,tide_height_cm,tide_state
    No astronomical tide proxy is fabricated when this source is absent.
    """
    if data.empty:
        return data
    x = data.copy()
    path = root / "data" / "canonical" / "tide_hourly.csv"
    if not path.exists():
        x["tide_height_cm"] = np.nan
        x["tide_state_code"] = np.nan
        x["tide_source_available"] = 0
        return x
    tide = pd.read_csv(path)
    tide["timestamp"] = pd.to_datetime(tide.timestamp, errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    states = {"上げ": 1.0, "下げ": -1.0, "満潮": 0.0, "干潮": 0.0}
    tide["tide_state_code"] = tide.get("tide_state", pd.Series(index=tide.index, dtype="object")).map(states)
    tide["tide_height_cm"] = pd.to_numeric(tide.get("tide_height_cm"), errors="coerce")
    x["entered_at"] = pd.to_datetime(x.entered_at, errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    x = pd.merge_asof(
        x.sort_values("entered_at"),
        tide[["timestamp", "tide_height_cm", "tide_state_code"]].dropna(subset=["timestamp"]).sort_values("timestamp"),
        left_on="entered_at",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("40min"),
    ).drop(columns=["timestamp"], errors="ignore")
    x["tide_source_available"] = x.tide_height_cm.notna().astype(int)
    return x
