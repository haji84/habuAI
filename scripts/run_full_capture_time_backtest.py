from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


def _as_jst(s):
    return pd.to_datetime(s, format="mixed", utc=True).dt.tz_convert("Asia/Tokyo")


def _haversine_m(lat, lon, lats, lons):
    r = 6371000.0
    p1 = np.radians(float(lat))
    p2 = np.radians(np.asarray(lats, dtype=float))
    dp = p2 - p1
    dl = np.radians(np.asarray(lons, dtype=float) - float(lon))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _circular_minutes(a, b):
    d = abs(float(a) - float(b)) % 1440.0
    return min(d, 1440.0 - d)


def _wilson(hits, n, z=1.96):
    if n <= 0:
        return [None, None]
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _predict_minutes(past, cutoff, lat, lon, radius_m, bandwidth_h, local_weight, min_local, tau_days):
    if past.empty:
        return None
    dist = _haversine_m(lat, lon, past.lat.to_numpy(float), past.lon.to_numpy(float))
    local = dist <= float(radius_m)
    local_n = int(local.sum())
    h = past.timestamp.dt.hour.to_numpy(float) + past.timestamp.dt.minute.to_numpy(float) / 60.0
    if local_n >= int(min_local):
        w = np.where(local, 1.0, 1.0 - float(local_weight))
    else:
        w = np.ones(len(past), dtype=float)
    if tau_days is not None:
        age = (cutoff - past.timestamp).dt.total_seconds().to_numpy(float) / 86400.0
        w *= np.exp(-np.maximum(age, 0) / float(tau_days))
    slots = np.arange(0, 1440, 30, dtype=int)
    scores = []
    bw = float(bandwidth_h)
    for minute in slots:
        hour = minute / 60.0
        d = np.abs(h - hour)
        d = np.minimum(d, 24.0 - d)
        scores.append(float((w * np.exp(-0.5 * (d / bw) ** 2)).sum()))
    if not scores:
        return None
    return int(slots[int(np.argmax(scores))])


def _score_nights(cap, nights, params, min_prior=10):
    rows = []
    for night in nights:
        start = pd.Timestamp(str(night), tz="Asia/Tokyo") + pd.Timedelta(hours=7)
        end = start + pd.Timedelta(days=1)
        actual = cap[(cap.timestamp >= start) & (cap.timestamp < end)].copy()
        past = cap[cap.timestamp < start].copy()
        if actual.empty or len(past) < min_prior:
            continue
        for a in actual.itertuples():
            pred = _predict_minutes(
                past,
                start,
                a.lat,
                a.lon,
                params["radius_m"],
                params["bandwidth_h"],
                params["local_weight"],
                params["min_local"],
                params["tau_days"],
            )
            if pred is None:
                continue
            actual_min = int(a.timestamp.hour) * 60 + int(a.timestamp.minute)
            delta = _circular_minutes(actual_min, pred)
            rows.append({
                "night": str(night),
                "canonical_id": getattr(a, "canonical_id", None),
                "actual_timestamp": str(a.timestamp),
                "actual_minute_of_day": actual_min,
                "predicted_minute_of_day": pred,
                "absolute_circular_error_minutes": delta,
                "hit_30m": int(delta <= 30),
                "hit_60m": int(delta <= 60),
                "hit_90m": int(delta <= 90),
                "prior_capture_events": int(len(past)),
            })
    return pd.DataFrame(rows)


def _metrics(scored):
    if scored.empty:
        return {"eligible": 0}
    n = int(len(scored))
    out = {
        "eligible": n,
        "nights_scored": int(scored.night.nunique()),
        "median_absolute_error_minutes": float(scored.absolute_circular_error_minutes.median()),
        "mean_absolute_error_minutes": float(scored.absolute_circular_error_minutes.mean()),
    }
    for m in [30, 60, 90]:
        hits = int(scored[f"hit_{m}m"].sum())
        out[f"within_{m}m_hits"] = hits
        out[f"within_{m}m_rate"] = hits / n
        out[f"within_{m}m_wilson95"] = _wilson(hits, n)
    return out


def _inventory(events, learning):
    habu_capture = events[(events.species == "ハブ") & (events.event_type == "捕獲")].copy()
    habu_capture["operational_night"] = (habu_capture.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    gps_time = habu_capture[habu_capture.lat.notna() & habu_capture.lon.notna() & habu_capture.timestamp.notna()].copy()
    gpx = learning[learning.learning_row_source == "gpx_visit"].copy()
    gpx["operational_night"] = (gpx.entered_at - pd.Timedelta(hours=7)).dt.date.astype(str)
    no_capture = events[(events.species == "ハブ") & (events.event_type == "no_capture")].copy()
    no_capture["operational_night"] = (no_capture.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    month = habu_capture.timestamp.dt.strftime("%Y-%m")
    monthly = habu_capture.assign(month=month).groupby("month").agg(
        capture_events=("event_type", "size"),
        capture_nights=("operational_night", "nunique"),
        gps_capture_events=("lat", lambda x: int(x.notna().sum())),
    ).reset_index().to_dict("records")
    return {
        "capture_events": int(len(habu_capture)),
        "capture_individuals": int(pd.to_numeric(habu_capture.individual_count, errors="coerce").fillna(1).sum()),
        "capture_nights": int(habu_capture.operational_night.nunique()),
        "gps_timestamp_capture_events": int(len(gps_time)),
        "gps_timestamp_capture_nights": int(gps_time.operational_night.nunique()),
        "actual_gpx_nights": int(gpx.operational_night.nunique()),
        "explicit_no_capture_event_nights": int(no_capture.operational_night.nunique()),
        "monthly_capture_inventory": monthly,
        "note": "capture_nights are nights with a recorded Habu capture, not total operating nights. Total operating nights require explicit GPX/session/no-capture evidence.",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    reports = root / "reports"
    events = pd.read_csv(processed / "events_matched.csv", low_memory=False)
    learning = pd.read_csv(processed / "learning_10m_road.csv", low_memory=False)
    events["timestamp"] = _as_jst(events.timestamp)
    learning["entered_at"] = _as_jst(learning.entered_at)
    events["lat"] = pd.to_numeric(events.lat, errors="coerce")
    events["lon"] = pd.to_numeric(events.lon, errors="coerce")

    cap = events[(events.species == "ハブ") & (events.event_type == "捕獲") & events.lat.notna() & events.lon.notna() & events.timestamp.notna()].copy()
    cap["night"] = (cap.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    nights = sorted(cap.night.unique().tolist())
    if len(nights) < 8:
        result = {"status": "insufficient-data", "inventory": _inventory(events, learning)}
        (reports / "full_capture_time_backtest_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(result)
        return

    split_idx = max(4, int(np.floor(len(nights) * 0.70)))
    selection_nights = nights[:split_idx]
    confirmation_nights = nights[split_idx:]

    variants = []
    param_grid = []
    for radius in [500, 1000, 1500, 2000, 3000]:
        for bw in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for local_weight in [0.80, 0.95, 0.99]:
                for min_local in [1, 2, 3]:
                    for tau in [None, 90, 180]:
                        param_grid.append({
                            "radius_m": radius,
                            "bandwidth_h": bw,
                            "local_weight": local_weight,
                            "min_local": min_local,
                            "tau_days": tau,
                        })

    for params in param_grid:
        scored = _score_nights(cap, selection_nights, params)
        m = _metrics(scored)
        if m.get("eligible", 0) == 0:
            continue
        row = dict(params)
        row.update(m)
        row["selection_score"] = 3 * m["within_30m_rate"] + 2 * m["within_60m_rate"] + m["within_90m_rate"] - (m["median_absolute_error_minutes"] / 1440.0)
        variants.append(row)

    tournament = pd.DataFrame(variants).sort_values(
        ["selection_score", "within_30m_rate", "within_60m_rate", "median_absolute_error_minutes"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    tournament.to_csv(reports / "full_capture_time_tournament_selection.csv", index=False)
    best = tournament.iloc[0].to_dict()
    params = {k: best[k] for k in ["radius_m", "bandwidth_h", "local_weight", "min_local", "tau_days"]}
    if pd.isna(params["tau_days"]):
        params["tau_days"] = None

    selection_scored = _score_nights(cap, selection_nights, params)
    confirmation_scored = _score_nights(cap, confirmation_nights, params)
    all_scored = _score_nights(cap, nights, params)
    selection_scored.to_csv(reports / "full_capture_time_selection_predictions.csv", index=False)
    confirmation_scored.to_csv(reports / "full_capture_time_confirmation_predictions.csv", index=False)
    all_scored.to_csv(reports / "full_capture_time_all_walkforward_predictions.csv", index=False)

    hist_recon = {}
    recon_path = reports / "historical_route_reconstruction.json"
    if recon_path.exists():
        hist_recon = json.loads(recon_path.read_text(encoding="utf-8"))

    inventory = _inventory(events, learning)
    inventory.update({
        "may_july_capture_nights_considered_for_reconstruction": int(hist_recon.get("nights_total", 0)),
        "may_july_reconstructed_nights": int(hist_recon.get("nights_reconstructed", 0)),
        "may_july_reconstructed_capture_events": int(hist_recon.get("capture_events", 0)),
        "may_july_road_matched_capture_events": int(hist_recon.get("road_matched_capture_events", 0)),
    })

    summary = {
        "status": "ok",
        "method": "full-history point-conditioned capture-time validation. For each operational night, all same-night captures are hidden at the 07:00 cutoff and only earlier captures are visible. Hyperparameters are selected on the earlier 70% of GPS+timestamp capture nights and frozen for the later 30% chronological confirmation period. 30-minute slots are scored with circular time error.",
        "inventory": inventory,
        "gps_timestamp_capture_nights_total": len(nights),
        "selection_nights": len(selection_nights),
        "confirmation_nights": len(confirmation_nights),
        "best_parameters_selected_on_early_period": params,
        "selection_metrics": _metrics(selection_scored),
        "confirmation_metrics_frozen": _metrics(confirmation_scored),
        "all_walkforward_metrics_descriptive": _metrics(all_scored),
        "interpretation_guardrail": "The confirmation period is the primary reliability estimate for capture-time prediction. GPX-only 20-event testing remains as a separate stricter survey-night check. Reconstructed routes add spatial zero/exposure evidence but do not invent historical pass-through times.",
    }
    (reports / "full_capture_time_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
