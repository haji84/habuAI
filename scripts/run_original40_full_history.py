from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_full_capture_time_backtest import _as_jst, _score_nights


def _metrics(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"eligible": 0, "nights_scored": 0}
    err = pd.to_numeric(scored["absolute_circular_error_minutes"], errors="coerce").dropna()
    n = int(len(err))
    out = {
        "eligible": n,
        "nights_scored": int(scored["night"].nunique()),
        "median_error_min": float(err.median()),
        "mean_error_min": float(err.mean()),
    }
    for m in (30, 60, 90, 120):
        hits = int((err <= m).sum())
        out[f"within_{m}m_hits"] = hits
        out[f"within_{m}m_rate"] = hits / n if n else np.nan
    out["selection_score"] = (
        4 * out["within_30m_rate"]
        + 3 * out["within_60m_rate"]
        + 2 * out["within_90m_rate"]
        + out["within_120m_rate"]
        - out["median_error_min"] / 720.0
    )
    return out


def evaluate(cap: pd.DataFrame, nights: list[str], params: dict) -> dict:
    scored = _score_nights(cap, nights, params, min_prior=10)
    return _metrics(scored)


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(processed / "events_matched.csv", low_memory=False)
    events["timestamp"] = _as_jst(events["timestamp"])
    events["lat"] = pd.to_numeric(events["lat"], errors="coerce")
    events["lon"] = pd.to_numeric(events["lon"], errors="coerce")
    cap = events[
        (events["species"] == "ハブ")
        & (events["event_type"] == "捕獲")
        & events["lat"].notna()
        & events["lon"].notna()
        & events["timestamp"].notna()
    ].copy()
    cap["night"] = (cap["timestamp"] - pd.Timedelta(hours=7)).dt.date.astype(str)
    nights = sorted(cap["night"].unique().tolist())
    split_idx = max(4, int(np.floor(len(nights) * 0.70)))
    selection_nights = nights[:split_idx]
    confirmation_nights = nights[split_idx:]

    variants = []
    rank = 0
    for radius in [250, 500, 1000, 2000, 3000]:
        for bw in [0.75, 1.0, 1.5, 2.0]:
            for tau in [None, 120]:
                rank += 1
                params = {
                    "radius_m": radius,
                    "bandwidth_h": bw,
                    "local_weight": 0.95,
                    "min_local": 2,
                    "tau_days": tau,
                }
                all_m = evaluate(cap, nights, params)
                sel_m = evaluate(cap, selection_nights, params)
                conf_m = evaluate(cap, confirmation_nights, params)
                row = {
                    "radius_m": radius,
                    "bandwidth_h": bw,
                    "recency_tau_days": tau,
                    "local_weight": 0.95,
                    "min_local": 2,
                }
                for prefix, metrics in (("all", all_m), ("selection", sel_m), ("confirmation", conf_m)):
                    for k, v in metrics.items():
                        row[f"{prefix}_{k}"] = v
                variants.append(row)

    out = pd.DataFrame(variants)
    out = out.sort_values(
        ["confirmation_selection_score", "confirmation_within_60m_rate", "confirmation_median_error_min", "all_selection_score"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.to_csv(reports / "original40_full_history_tournament.csv", index=False)

    top = out.iloc[0].to_dict() if not out.empty else {}
    summary = {
        "status": "ok" if top else "insufficient-data",
        "method": "Exact original 40 point-conditioned time-model settings re-evaluated with the maximum GPS+timestamp Habu capture history. Every operational night is cut at 07:00 JST; same-night captures are hidden; only prior captures are visible. The original grid is fixed: radii 250/500/1000/2000/3000m x bandwidth 0.75/1.0/1.5/2.0h x recency none/120d, with local_weight=.95 and min_local=2.",
        "gps_timestamp_capture_events": int(len(cap)),
        "gps_timestamp_capture_nights": int(len(nights)),
        "selection_nights": int(len(selection_nights)),
        "confirmation_nights": int(len(confirmation_nights)),
        "variants": int(len(out)),
        "ranking_primary": "frozen confirmation-period score",
        "best": top,
        "guardrail": "The all-history walk-forward results are descriptive. The later 30% chronological confirmation metrics are the primary reliability check because the 40 variants are not allowed to tune on that period.",
    }
    (reports / "original40_full_history_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
