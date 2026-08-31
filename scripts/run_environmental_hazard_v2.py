from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_advanced_time_models import (
    SLOTS,
    _as_jst,
    _circular_minutes,
    _fetch_weather,
    _local_prior_features,
)
from habuai.hardening.environmental import (
    NEW_MOON_EPOCH_UTC,
    SYNODIC_MONTH_DAYS,
    _fetch_jma_tide,
)


def _prepare_weather(w: pd.DataFrame) -> pd.DataFrame:
    x = w.copy().sort_values("time").reset_index(drop=True)
    for c in [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "weather_code",
        "surface_pressure",
        "cloud_cover",
    ]:
        x[c] = pd.to_numeric(x.get(c), errors="coerce")
    x["rain_24h_mm"] = x["precipitation"].rolling(24, min_periods=1).sum()
    x["rain_48h_mm"] = x["precipitation"].rolling(48, min_periods=1).sum()
    spread = x["temperature_2m"] - x["dew_point_2m"]
    x["fog_wmo_flag"] = x["weather_code"].isin([45, 48]).astype(int)
    x["fog_proxy_flag"] = ((spread <= 1.5) & (x["relative_humidity_2m"] >= 95)).astype(int)
    x["fog_any_flag"] = ((x["fog_wmo_flag"] == 1) | (x["fog_proxy_flag"] == 1)).astype(int)

    last_rain_time = None
    hours_since = []
    for row in x.itertuples():
        if pd.notna(row.precipitation) and float(row.precipitation) >= 0.1:
            last_rain_time = row.time
        if last_rain_time is None:
            hours_since.append(np.nan)
        else:
            hours_since.append((row.time - last_rain_time).total_seconds() / 3600.0)
    x["hours_since_rain"] = hours_since
    return x


def _weather_30m(w: pd.DataFrame) -> pd.DataFrame:
    x = _prepare_weather(w).set_index("time").sort_index()
    idx = pd.date_range(x.index.min(), x.index.max(), freq="30min")
    z = x.reindex(x.index.union(idx)).sort_index()
    interp = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "cloud_cover",
        "rain_24h_mm",
        "rain_48h_mm",
        "hours_since_rain",
    ]
    z[interp] = z[interp].interpolate(method="time").ffill().bfill()
    for c in ["precipitation", "weather_code", "fog_wmo_flag", "fog_proxy_flag", "fog_any_flag"]:
        z[c] = z[c].ffill().bfill()
    return z.reindex(idx)


def _moon_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    utc = pd.to_datetime(times, utc=True)
    days = (utc - NEW_MOON_EPOCH_UTC).total_seconds() / 86400.0
    age = np.mod(days, SYNODIC_MONTH_DAYS)
    phase = age / SYNODIC_MONTH_DAYS
    return pd.DataFrame({
        "moon_age_days": age,
        "moon_illumination": (1 - np.cos(2 * np.pi * phase)) / 2,
        "moon_phase_sin": np.sin(2 * np.pi * phase),
        "moon_phase_cos": np.cos(2 * np.pi * phase),
    }, index=times)


def _tide_30m(times: pd.DatetimeIndex) -> pd.DataFrame:
    years = sorted(set(pd.DatetimeIndex(times).year.tolist()))
    tide = _fetch_jma_tide(years)
    if tide.empty:
        return pd.DataFrame(index=times, data={
            "tide_height_cm": np.nan,
            "tide_change_1h_cm": np.nan,
            "tide_state_code": np.nan,
            "minutes_to_nearest_turning_tide": np.nan,
        })
    t = tide.copy()
    t["timestamp"] = pd.to_datetime(t.timestamp, utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    t = t.set_index("timestamp").sort_index()
    idx = pd.DatetimeIndex(times)
    z = t.reindex(t.index.union(idx)).sort_index()
    for c in ["tide_height_cm", "tide_change_1h_cm", "tide_state_code", "minutes_to_nearest_turning_tide"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").interpolate(method="time").ffill().bfill()
    return z.reindex(idx)[["tide_height_cm", "tide_change_1h_cm", "tide_state_code", "minutes_to_nearest_turning_tide"]]


def _risk_rows_for_night(cap: pd.DataFrame, night: str, weather: pd.DataFrame, tide_all: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(str(night), tz="Asia/Tokyo") + pd.Timedelta(hours=7)
    end = start + pd.Timedelta(days=1)
    actual = cap[(cap.timestamp >= start) & (cap.timestamp < end)].copy()
    past = cap[cap.timestamp < start].copy()
    if actual.empty or len(past) < 10:
        return pd.DataFrame()

    times = pd.DatetimeIndex([pd.Timestamp(str(night)) + pd.Timedelta(hours=7, minutes=int(m)) for m in SLOTS])
    ww = weather.reindex(times, method="nearest")
    mm = _moon_features(times)
    tt = tide_all.reindex(times)
    rows = []
    for a in actual.itertuples():
        gp, lp = _local_prior_features(past, start, a.lat, a.lon)
        actual_min = int(a.timestamp.hour) * 60 + int(a.timestamp.minute)
        positive_idx = int(np.argmin([_circular_minutes(actual_min, m) for m in SLOTS]))
        for i, (minute, t) in enumerate(zip(SLOTS, times)):
            wr = ww.iloc[i]
            mr = mm.iloc[i]
            tr = tt.iloc[i]
            hr = minute / 60.0
            doy = int(t.dayofyear)
            rows.append({
                "night": str(night),
                "event_id": str(getattr(a, "canonical_id", "")) + ":" + str(actual_min),
                "slot_minute": int(minute),
                "label": int(i == positive_idx),
                "actual_minute": actual_min,
                "sin_hour": math.sin(2 * math.pi * hr / 24),
                "cos_hour": math.cos(2 * math.pi * hr / 24),
                "sin_doy": math.sin(2 * math.pi * doy / 365.25),
                "cos_doy": math.cos(2 * math.pi * doy / 365.25),
                "temperature_2m": wr.get("temperature_2m", np.nan),
                "relative_humidity_2m": wr.get("relative_humidity_2m", np.nan),
                "dew_point_2m": wr.get("dew_point_2m", np.nan),
                "precipitation": wr.get("precipitation", np.nan),
                "weather_code": wr.get("weather_code", np.nan),
                "surface_pressure": wr.get("surface_pressure", np.nan),
                "cloud_cover": wr.get("cloud_cover", np.nan),
                "rain_24h_mm": wr.get("rain_24h_mm", np.nan),
                "rain_48h_mm": wr.get("rain_48h_mm", np.nan),
                "hours_since_rain": wr.get("hours_since_rain", np.nan),
                "fog_wmo_flag": wr.get("fog_wmo_flag", np.nan),
                "fog_proxy_flag": wr.get("fog_proxy_flag", np.nan),
                "fog_any_flag": wr.get("fog_any_flag", np.nan),
                "moon_age_days": mr.get("moon_age_days", np.nan),
                "moon_illumination": mr.get("moon_illumination", np.nan),
                "moon_phase_sin": mr.get("moon_phase_sin", np.nan),
                "moon_phase_cos": mr.get("moon_phase_cos", np.nan),
                "tide_height_cm": tr.get("tide_height_cm", np.nan),
                "tide_change_1h_cm": tr.get("tide_change_1h_cm", np.nan),
                "tide_state_code": tr.get("tide_state_code", np.nan),
                "minutes_to_nearest_turning_tide": tr.get("minutes_to_nearest_turning_tide", np.nan),
                "global_prior": gp[i],
                "local3km_prior": lp[i],
            })
    return pd.DataFrame(rows)


def _score_walkforward(risk: pd.DataFrame, nights: list[str], features: list[str]) -> pd.DataFrame:
    preds = []
    for night in nights:
        te = risk[risk.night == str(night)].copy()
        tr = risk[risk.night.astype(str) < str(night)].copy()
        if te.empty or int(tr.label.sum()) < 10:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=.2),
        )
        model.fit(tr[features], tr.label.astype(int))
        te["p"] = model.predict_proba(te[features])[:, 1]
        for eid, g in te.groupby("event_id"):
            best = g.loc[g.p.idxmax()]
            actual = int(g.actual_minute.iloc[0])
            pred = int(best.slot_minute)
            preds.append({
                "night": str(night),
                "event_id": eid,
                "actual_minute": actual,
                "predicted_minute": pred,
                "error_min": _circular_minutes(actual, pred),
            })
    return pd.DataFrame(preds)


def _metrics(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"eligible": 0, "nights_scored": 0}
    e = pd.to_numeric(x.error_min, errors="coerce").dropna()
    out = {
        "eligible": int(len(e)),
        "nights_scored": int(x.night.nunique()),
        "median_error_min": float(e.median()),
        "mean_error_min": float(e.mean()),
    }
    for m in [30, 60, 90, 120]:
        h = int((e <= m).sum())
        out[f"within_{m}m_hits"] = h
        out[f"within_{m}m_rate"] = h / len(e)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "processed"
    r = root / "reports"
    r.mkdir(exist_ok=True)
    ev = pd.read_csv(p / "events_matched.csv", low_memory=False)
    ev["timestamp"] = _as_jst(ev.timestamp)
    ev["lat"] = pd.to_numeric(ev.lat, errors="coerce")
    ev["lon"] = pd.to_numeric(ev.lon, errors="coerce")
    cap = ev[(ev.species == "ハブ") & (ev.event_type == "捕獲") & ev.lat.notna() & ev.lon.notna() & ev.timestamp.notna()].copy()
    cap["night"] = (cap.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    nights = sorted(cap.night.unique().tolist())
    split = max(4, int(np.floor(len(nights) * .70)))
    conf = nights[split:]

    weather_start = str((cap.timestamp.min() - pd.Timedelta(days=3)).date())
    weather_end = str(cap.timestamp.max().date())
    cache = p / "cache" / "openmeteo_historical_environment_v2.csv"
    w = _fetch_weather(weather_start, weather_end, cache)
    w30 = _weather_30m(w)
    all_times = pd.date_range(w30.index.min(), w30.index.max(), freq="30min")
    tide30 = _tide_30m(all_times)
    risk = pd.concat([_risk_rows_for_night(cap, n, w30, tide30) for n in nights], ignore_index=True)

    groups = {
        "baseline_clock_priors": ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "global_prior", "local3km_prior"],
        "weather_current": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "weather_code", "surface_pressure", "cloud_cover"],
        "rain_history": ["rain_24h_mm", "rain_48h_mm", "hours_since_rain"],
        "fog": ["fog_wmo_flag", "fog_proxy_flag", "fog_any_flag"],
        "moon": ["moon_age_days", "moon_illumination", "moon_phase_sin", "moon_phase_cos"],
        "tide": ["tide_height_cm", "tide_change_1h_cm", "tide_state_code", "minutes_to_nearest_turning_tide"],
    }
    stages = [
        ("baseline", ["baseline_clock_priors"]),
        ("+weather", ["baseline_clock_priors", "weather_current"]),
        ("+rain24_48_since", ["baseline_clock_priors", "weather_current", "rain_history"]),
        ("+fog", ["baseline_clock_priors", "weather_current", "rain_history", "fog"]),
        ("+moon", ["baseline_clock_priors", "weather_current", "rain_history", "fog", "moon"]),
        ("full_environment_plus_tide", ["baseline_clock_priors", "weather_current", "rain_history", "fog", "moon", "tide"]),
    ]
    rows = []
    for name, include in stages:
        feats = []
        for g in include:
            feats += groups[g]
        pred = _score_walkforward(risk, nights, feats)
        conf_pred = pred[pred.night.isin(set(conf))]
        row = {"model": name, "feature_count": len(feats)}
        for prefix, met in [("all", _metrics(pred)), ("confirmation", _metrics(conf_pred))]:
            for k, v in met.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(r / "environmental_hazard_v2_ablation.csv", index=False)
    best = out.sort_values([
        "confirmation_within_60m_rate",
        "confirmation_within_90m_rate",
        "confirmation_median_error_min",
    ], ascending=[False, False, True]).iloc[0].to_dict()
    summary = {
        "status": "ok",
        "protocol": {
            "gps_timestamp_capture_events": int(len(cap)),
            "capture_nights": int(len(nights)),
            "selection_nights": int(split),
            "confirmation_nights": int(len(conf)),
            "rollover": "07:00 Asia/Tokyo",
            "min_prior_captures": 10,
            "slot_minutes": 30,
        },
        "data_sources": {
            "weather": "Open-Meteo Historical Weather API; hourly temperature, humidity, dew point, precipitation, weather code, surface pressure, cloud cover. Derived rain_24h, rain_48h, hours_since_rain and fog proxies.",
            "moon": "deterministic astronomical phase from synodic month; no external lookup required",
            "tide": "Japan Meteorological Agency tide table station O9, Amami (Naze Kominato), used as regional proxy",
        },
        "feature_groups": groups,
        "best_stage_by_frozen_confirmation": best,
        "guardrails": [
            "same-night capture outcomes are hidden at 07:00",
            "reconstructed historical routes are not assigned fabricated clock times",
            "JMA O9 tide is a regional proxy for Setouchi, not a local tide gauge",
            "Open-Meteo historical weather is gridded/reanalysis data, not a road-level station observation",
        ],
    }
    (r / "environmental_hazard_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
