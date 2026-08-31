from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_full_capture_time_backtest import (
    _as_jst,
    _circular_minutes,
    _predict_minutes,
)

PARAMS = {
    "radius_m": 3000.0,
    "bandwidth_h": 1.0,
    "local_weight": 0.95,
    "min_local": 2,
    "tau_days": 120.0,
}


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
    for m in (30, 60, 90, 120):
        hits = int((e <= m).sum())
        out[f"within_{m}m_hits"] = hits
        out[f"within_{m}m_rate"] = hits / len(e)
    return out


def _score_strategy(
    cap: pd.DataFrame,
    confirmation_nights: list[str],
    strategy: str,
    freeze_cutoff: pd.Timestamp,
    rolling_days: int | None = None,
) -> pd.DataFrame:
    rows = []
    frozen_pool = cap[cap.timestamp < freeze_cutoff].copy()

    for night in confirmation_nights:
        start = pd.Timestamp(str(night), tz="Asia/Tokyo") + pd.Timedelta(hours=7)
        end = start + pd.Timedelta(days=1)
        actual = cap[(cap.timestamp >= start) & (cap.timestamp < end)].copy()
        if actual.empty:
            continue

        if strategy == "frozen":
            past = frozen_pool.copy()
        elif strategy == "expanding_daily":
            past = cap[cap.timestamp < start].copy()
        elif strategy == "rolling_daily":
            lower = start - pd.Timedelta(days=int(rolling_days))
            past = cap[(cap.timestamp < start) & (cap.timestamp >= lower)].copy()
        else:
            raise ValueError(strategy)

        if len(past) < 10:
            continue

        for a in actual.itertuples():
            pred = _predict_minutes(
                past,
                start,
                a.lat,
                a.lon,
                PARAMS["radius_m"],
                PARAMS["bandwidth_h"],
                PARAMS["local_weight"],
                PARAMS["min_local"],
                PARAMS["tau_days"],
            )
            if pred is None:
                continue
            actual_min = int(a.timestamp.hour) * 60 + int(a.timestamp.minute)
            err = _circular_minutes(actual_min, pred)
            rows.append({
                "strategy": strategy if rolling_days is None else f"rolling_{rolling_days}d",
                "night": str(night),
                "canonical_id": getattr(a, "canonical_id", None),
                "actual_minute": actual_min,
                "predicted_minute": pred,
                "error_min": err,
                "history_events_used": int(len(past)),
            })
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "processed"
    r = root / "reports"
    r.mkdir(exist_ok=True)

    ev = pd.read_csv(p / "events_matched.csv", low_memory=False)
    ev["timestamp"] = _as_jst(ev.timestamp)
    ev["lat"] = pd.to_numeric(ev.lat, errors="coerce")
    ev["lon"] = pd.to_numeric(ev.lon, errors="coerce")
    cap = ev[
        (ev.species == "ハブ")
        & (ev.event_type == "捕獲")
        & ev.lat.notna()
        & ev.lon.notna()
        & ev.timestamp.notna()
    ].copy()
    cap["night"] = (cap.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    nights = sorted(cap.night.unique().tolist())
    split = max(4, int(np.floor(len(nights) * 0.70)))
    confirmation_nights = nights[split:]
    freeze_cutoff = pd.Timestamp(str(confirmation_nights[0]), tz="Asia/Tokyo") + pd.Timedelta(hours=7)

    scored_parts = [
        _score_strategy(cap, confirmation_nights, "frozen", freeze_cutoff),
        _score_strategy(cap, confirmation_nights, "expanding_daily", freeze_cutoff),
    ]
    for days in (30, 60, 90, 120, 180):
        scored_parts.append(
            _score_strategy(cap, confirmation_nights, "rolling_daily", freeze_cutoff, rolling_days=days)
        )

    scored = pd.concat(scored_parts, ignore_index=True)
    scored.to_csv(r / "daily_update_strategy_predictions.csv", index=False)

    rows = []
    for strategy, g in scored.groupby("strategy"):
        m = _metrics(g)
        rows.append({"strategy": strategy, **m})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["score"] = (
            4 * table.within_30m_rate
            + 3 * table.within_60m_rate
            + 2 * table.within_90m_rate
            + table.within_120m_rate
            - table.median_error_min / 720.0
        )
        table = table.sort_values(
            ["score", "within_60m_rate", "median_error_min"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(r / "daily_update_strategy_tournament.csv", index=False)

    summary = {
        "status": "ok",
        "question": "Does adding each completed night's capture data before predicting the next night improve the current best point-conditioned time model?",
        "protocol": {
            "model_parameters": PARAMS,
            "gps_timestamp_capture_events": int(len(cap)),
            "capture_nights": int(len(nights)),
            "confirmation_nights": int(len(confirmation_nights)),
            "confirmation_capture_events": int(len(cap[cap.night.isin(set(confirmation_nights))])),
            "freeze_cutoff": str(freeze_cutoff),
            "rollover": "07:00 Asia/Tokyo",
            "same_night_outcomes_hidden": True,
        },
        "strategies": {
            "frozen": "Use only data available before the first confirmation night for every later night.",
            "expanding_daily": "After each night finishes, add that night's captures before predicting the following night.",
            "rolling_30d_to_180d": "Add nightly outcomes but discard history older than the rolling window.",
        },
        "results": table.to_dict("records"),
        "interpretation_guardrail": "Hyperparameters are held fixed. This isolates the effect of the update policy itself rather than re-tuning the model on the confirmation period.",
    }
    (r / "daily_update_strategy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
