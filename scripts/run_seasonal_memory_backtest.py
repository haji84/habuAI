from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_full_capture_time_backtest import _as_jst, _circular_minutes, _haversine_m
from run_advanced_time_models import _fetch_weather

SLOTS = np.arange(0, 1440, 30, dtype=int)
BASE = {
    "radius_m": 3000.0,
    "bandwidth_h": 1.0,
    "local_weight": 0.95,
    "min_local": 2,
}


def _kde_predict(past, cutoff, lat, lon, weights):
    if past.empty:
        return None
    dist = _haversine_m(lat, lon, past.lat.to_numpy(float), past.lon.to_numpy(float))
    local = dist <= BASE["radius_m"]
    if int(local.sum()) >= BASE["min_local"]:
        weights = weights * np.where(local, 1.0, 1.0 - BASE["local_weight"])
    h = past.timestamp.dt.hour.to_numpy(float) + past.timestamp.dt.minute.to_numpy(float) / 60.0
    scores = []
    for minute in SLOTS:
        hour = minute / 60.0
        d = np.abs(h - hour)
        d = np.minimum(d, 24.0 - d)
        scores.append(float((weights * np.exp(-0.5 * (d / BASE["bandwidth_h"]) ** 2)).sum()))
    return int(SLOTS[int(np.argmax(scores))])


def _cyclic_doy_distance(a, b):
    d = np.abs(np.asarray(a, float) - float(b))
    return np.minimum(d, 365.25 - d)


def _night_features(weather):
    w = weather.copy().sort_values("time")
    for c in ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure", "cloud_cover"]:
        w[c] = pd.to_numeric(w.get(c), errors="coerce")
    w["rain24"] = w["precipitation"].rolling(24, min_periods=1).sum()
    w["rain48"] = w["precipitation"].rolling(48, min_periods=1).sum()
    rows = []
    dates = pd.date_range(w.time.min().normalize(), w.time.max().normalize(), freq="D")
    for d in dates:
        t = d + pd.Timedelta(hours=18)
        x = w.iloc[(w.time - t).abs().argsort()[:1]]
        if x.empty:
            continue
        r = x.iloc[0]
        rows.append({
            "night": d.date().isoformat(),
            "temp": float(r.temperature_2m) if pd.notna(r.temperature_2m) else np.nan,
            "rh": float(r.relative_humidity_2m) if pd.notna(r.relative_humidity_2m) else np.nan,
            "dew": float(r.dew_point_2m) if pd.notna(r.dew_point_2m) else np.nan,
            "rain24": float(r.rain24) if pd.notna(r.rain24) else np.nan,
            "rain48": float(r.rain48) if pd.notna(r.rain48) else np.nan,
            "pressure": float(r.surface_pressure) if pd.notna(r.surface_pressure) else np.nan,
            "cloud": float(r.cloud_cover) if pd.notna(r.cloud_cover) else np.nan,
        })
    return pd.DataFrame(rows).set_index("night") if rows else pd.DataFrame()


def _env_weight(past_nights, target_night, nf, scale):
    cols = ["temp", "rh", "dew", "rain24", "rain48", "pressure", "cloud"]
    if nf.empty or target_night not in nf.index:
        return np.ones(len(past_nights), float)
    t = nf.loc[target_night, cols].to_numpy(float)
    vals = []
    for n in past_nights:
        if n in nf.index:
            vals.append(nf.loc[n, cols].to_numpy(float))
        else:
            vals.append(np.full(len(cols), np.nan))
    x = np.vstack(vals)
    # Robust scales to put weather dimensions on comparable footing.
    sig = np.array([3.0, 12.0, 3.0, 12.0, 20.0, 8.0, 25.0]) * float(scale)
    z = (x - t) / sig
    z = np.where(np.isfinite(z), z, 0.0)
    d2 = np.sum(z * z, axis=1)
    return np.exp(-0.5 * d2)


def _weights(past, cutoff, target_night, nf, family, params):
    age = (cutoff - past.timestamp).dt.total_seconds().to_numpy(float) / 86400.0
    age = np.maximum(age, 0.0)
    target_doy = cutoff.dayofyear
    doy = past.timestamp.dt.dayofyear.to_numpy(float)
    sd = _cyclic_doy_distance(doy, target_doy)
    season_bw = float(params.get("season_bw", 60.0))
    season = np.exp(-0.5 * (sd / season_bw) ** 2)
    env = _env_weight(((past.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)).tolist(), target_night, nf, params.get("env_scale", 1.0))

    if family == "recency120":
        return np.exp(-age / 120.0)
    if family == "no_decay":
        return np.ones(len(past), float)
    if family == "recency_season":
        return np.exp(-age / float(params["tau_days"])) * season
    if family == "recency_season_weather":
        return np.exp(-age / float(params["tau_days"])) * season * env
    if family == "season_weather_weak_age":
        tau = params.get("tau_days")
        age_w = np.ones(len(past), float) if tau is None else np.exp(-age / float(tau))
        return age_w * season * env
    raise ValueError(family)


def _score(cap, nights, nf, family, params, min_prior=10):
    rows = []
    for night in nights:
        start = pd.Timestamp(str(night), tz="Asia/Tokyo") + pd.Timedelta(hours=7)
        end = start + pd.Timedelta(days=1)
        actual = cap[(cap.timestamp >= start) & (cap.timestamp < end)]
        past = cap[cap.timestamp < start]
        if actual.empty or len(past) < min_prior:
            continue
        w = _weights(past, start, str(night), nf, family, params)
        for a in actual.itertuples():
            pred = _kde_predict(past, start, a.lat, a.lon, w)
            actual_min = int(a.timestamp.hour) * 60 + int(a.timestamp.minute)
            err = _circular_minutes(actual_min, pred)
            rows.append({"night": str(night), "error_min": err, "predicted_minute": pred, "actual_minute": actual_min})
    return pd.DataFrame(rows)


def _metrics(x):
    if x.empty:
        return {"eligible": 0}
    e = pd.to_numeric(x.error_min, errors="coerce").dropna()
    out = {"eligible": int(len(e)), "nights": int(x.night.nunique()), "median_error_min": float(e.median()), "mean_error_min": float(e.mean())}
    for m in [30, 60, 90, 120]:
        hit = int((e <= m).sum())
        out[f"within_{m}m_hits"] = hit
        out[f"within_{m}m_rate"] = hit / len(e)
    out["score"] = 4*out["within_30m_rate"] + 3*out["within_60m_rate"] + 2*out["within_90m_rate"] + out["within_120m_rate"] - out["median_error_min"]/720.0
    return out


def _grid(family):
    if family in {"recency120", "no_decay"}:
        return [{}]
    if family == "recency_season":
        return [{"tau_days": tau, "season_bw": sb} for tau in [90, 120, 180, 240, 365] for sb in [30, 45, 60, 90]]
    if family == "recency_season_weather":
        return [{"tau_days": tau, "season_bw": sb, "env_scale": es} for tau in [90, 120, 180, 240, 365] for sb in [30, 45, 60, 90] for es in [0.75, 1.0, 1.5, 2.0]]
    if family == "season_weather_weak_age":
        return [{"tau_days": tau, "season_bw": sb, "env_scale": es} for tau in [730, 1095, None] for sb in [30, 45, 60, 90] for es in [0.75, 1.0, 1.5, 2.0]]
    raise ValueError(family)


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
    nights = sorted(cap.night.unique())
    split = int(np.floor(len(nights) * .70))
    selection = nights[:split]
    confirmation = nights[split:]

    cache = p / "cache" / "openmeteo_historical_seasonal_memory.csv"
    weather = _fetch_weather(str((cap.timestamp.min()-pd.Timedelta(days=3)).date()), str(cap.timestamp.max().date()), cache)
    nf = _night_features(weather)

    families = ["recency120", "no_decay", "recency_season", "recency_season_weather", "season_weather_weak_age"]
    selected = []
    all_candidates = []
    for family in families:
        cand = []
        for params in _grid(family):
            met = _metrics(_score(cap, selection, nf, family, params))
            row = {"family": family, "params": json.dumps(params, ensure_ascii=False), **met}
            cand.append(row)
            all_candidates.append(row)
        cdf = pd.DataFrame(cand).sort_values(["score", "within_60m_rate", "median_error_min"], ascending=[False, False, True])
        best = cdf.iloc[0]
        params = json.loads(best.params)
        conf = _metrics(_score(cap, confirmation, nf, family, params))
        allm = _metrics(_score(cap, nights, nf, family, params))
        selected.append({
            "family": family,
            "selected_params": best.params,
            **{f"selection_{k}": v for k, v in _metrics(_score(cap, selection, nf, family, params)).items()},
            **{f"confirmation_{k}": v for k, v in conf.items()},
            **{f"all_{k}": v for k, v in allm.items()},
        })

    pd.DataFrame(all_candidates).to_csv(r / "seasonal_memory_selection_grid.csv", index=False)
    out = pd.DataFrame(selected).sort_values(["confirmation_score", "confirmation_within_60m_rate", "confirmation_median_error_min"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "descriptive_confirmation_rank", np.arange(1, len(out)+1))
    out.to_csv(r / "seasonal_memory_strategy_comparison.csv", index=False)
    summary = {
        "status": "ok",
        "protocol": {
            "gps_timestamp_capture_events": int(len(cap)),
            "capture_nights": int(len(nights)),
            "selection_nights": int(len(selection)),
            "confirmation_nights": int(len(confirmation)),
            "confirmation_capture_events": int(sum((cap.night == n).sum() for n in confirmation)),
            "rollover": "07:00 Asia/Tokyo",
            "model_base": BASE,
            "selection_rule": "Each family tunes only on early 70%; later 30% is untouched confirmation.",
            "environment_similarity": "18:00 local conditions plus prior 24/48h rain, using historical weather; no same-night capture outcome is used.",
        },
        "families": {
            "recency120": "current 120-day exponential age decay",
            "no_decay": "all historical captures equal age weight",
            "recency_season": "age decay times cyclic day-of-year similarity",
            "recency_season_weather": "age decay times seasonal similarity times pre-night weather similarity",
            "season_weather_weak_age": "season plus weather similarity with very weak 2-3 year age decay or no age decay",
        },
        "results": out.to_dict("records"),
        "guardrail": "Confirmation ranking is descriptive only. Family hyperparameters were selected exclusively on the early 70% period.",
    }
    (r / "seasonal_memory_strategy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
