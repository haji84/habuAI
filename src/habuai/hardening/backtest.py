from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import _feature_columns


def _new_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=.2),
    )


def _survey_night(ts: pd.Series, rollover_hour: int) -> pd.Series:
    return (pd.to_datetime(ts) - pd.Timedelta(hours=rollover_hour)).dt.date.astype(str)


def _night_bounds(night: str, rollover_hour: int, tz) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(night, tz=tz)
    start = day + pd.Timedelta(hours=rollover_hour)
    return start, start + pd.Timedelta(days=1)


def _actual_captures(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        (events.species == "ハブ")
        & (events.event_type == "捕獲")
        & (events.timestamp >= start)
        & (events.timestamp < end)
    ].copy()


def run_walk_forward_backtest(root: Path, model_data: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> dict:
    """Replay each GPX survey night using only information available before that night.

    Candidate roads are that night's actual GPX visit rows. Capture GPS anchors are outcomes,
    never candidate visits. Historical training can include anchors only when their timestamp is
    strictly before the replay-night boundary.
    """
    out_dir = root / "reports"
    rollover = int(cfg.get("night_rollover_hour", 7))
    top_k = int(cfg.get("walk_forward_backtest", {}).get("top_k_segments", 30))
    min_train_positives = int(cfg.get("walk_forward_backtest", {}).get("min_train_positives", 5))
    min_train_negatives = int(cfg.get("walk_forward_backtest", {}).get("min_train_negatives", 20))

    data = model_data.copy()
    data["entered_at"] = pd.to_datetime(data.entered_at)
    if data.empty or "learning_row_source" not in data:
        result = {"status": "no-data", "nights": 0}
        (out_dir / "walk_forward_backtest_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    gpx = data[data.learning_row_source == "gpx_visit"].copy()
    if gpx.empty:
        result = {"status": "no-gpx-visits", "nights": 0}
        (out_dir / "walk_forward_backtest_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    gpx["survey_night"] = _survey_night(gpx.entered_at, rollover)
    tz = gpx.entered_at.iloc[0].tz
    rows = []
    ranked_rows = []

    for night in sorted(gpx.survey_night.unique()):
        start, end = _night_bounds(night, rollover, tz)
        train = data[data.entered_at < start].copy()
        test = gpx[(gpx.entered_at >= start) & (gpx.entered_at < end)].copy()
        actual = _actual_captures(events, start, end)
        if test.empty:
            continue

        train_pos = int(train.habu_capture.sum()) if "habu_capture" in train else 0
        train_neg = int((train.habu_capture == 0).sum()) if "habu_capture" in train else 0
        base = {
            "night": night,
            "train_rows": int(len(train)),
            "train_positives": train_pos,
            "gpx_visit_rows": int(len(test)),
            "actual_capture_events": int(len(actual)),
            "actual_individuals": int(pd.to_numeric(actual.get("individual_count", pd.Series(dtype=float)), errors="coerce").fillna(1).sum()) if not actual.empty else 0,
        }
        if train_pos < min_train_positives or train_neg < min_train_negatives:
            base["status"] = "insufficient-training-history"
            rows.append(base)
            continue

        feats = _feature_columns(train)
        Xtr = train[feats].replace([np.inf, -np.inf], np.nan)
        ytr = train.habu_capture.astype(int)
        model = _new_model(); model.fit(Xtr, ytr)
        Xte = test.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
        test["pred_prob"] = model.predict_proba(Xte)[:, 1]

        # Count proxy: expected positive GPX passages. It is an auditable model-derived count proxy,
        # not a claim that each passage is an independent snake.
        predicted_count = float(test.pred_prob.sum())
        predicted_count_rounded = int(np.rint(predicted_count))

        seg_rank = test.groupby("segment_id", dropna=True).agg(
            pred_prob=("pred_prob", "max"),
            visits=("segment_id", "size"),
        ).reset_index().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        seg_rank["rank"] = np.arange(1, len(seg_rank) + 1)
        seg_rank["night"] = night
        ranked_rows.append(seg_rank[["night", "segment_id", "rank", "pred_prob", "visits"]])

        top_ids = set(seg_rank.head(top_k).segment_id.astype(str))
        actual_matched = actual[actual.segment_id.notna()].copy() if not actual.empty and "segment_id" in actual else pd.DataFrame()
        actual_ids = actual_matched.segment_id.astype(str).tolist() if not actual_matched.empty else []
        actual_ids_unique = set(actual_ids)
        location_hits = sum(1 for sid in actual_ids if sid in top_ids)
        location_hit_rate = None if not actual_ids else location_hits / len(actual_ids)
        unique_location_hit_rate = None if not actual_ids_unique else len(actual_ids_unique & top_ids) / len(actual_ids_unique)

        test["hour"] = test.entered_at.dt.hour
        hour_scores = test.groupby("hour").pred_prob.sum().sort_values(ascending=False)
        peak_hour = int(hour_scores.index[0]) if len(hour_scores) else None
        if actual.empty or peak_hour is None:
            time_hits = 0
            time_hit_rate = None if actual.empty else 0.0
        else:
            actual_hours = actual.timestamp.dt.hour
            time_hits = int((actual_hours == peak_hour).sum())
            time_hit_rate = time_hits / len(actual)

        actual_count = int(len(actual))
        base.update({
            "status": "ok",
            "predicted_capture_count_proxy": predicted_count,
            "predicted_capture_count_rounded": predicted_count_rounded,
            "count_absolute_error": abs(predicted_count_rounded - actual_count),
            "candidate_segments": int(len(seg_rank)),
            "top_k_segments": top_k,
            "actual_road_matched_capture_events": int(len(actual_ids)),
            "location_hits_top_k": int(location_hits),
            "location_hit_rate_top_k": location_hit_rate,
            "unique_location_hit_rate_top_k": unique_location_hit_rate,
            "predicted_peak_hour_start": peak_hour,
            "predicted_peak_hour_end": None if peak_hour is None else (peak_hour + 1) % 24,
            "time_hits_peak_hour": int(time_hits),
            "time_hit_rate_peak_hour": time_hit_rate,
        })
        rows.append(base)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "walk_forward_backtest_nightly.csv", index=False)
    if ranked_rows:
        pd.concat(ranked_rows, ignore_index=True).to_csv(out_dir / "walk_forward_backtest_segment_ranks.csv", index=False)

    ok = detail[detail.status == "ok"].copy() if not detail.empty and "status" in detail else pd.DataFrame()
    summary = {
        "status": "ok" if not ok.empty else "insufficient-data",
        "method": "nightly walk-forward replay; train strictly before 07:00 survey-night boundary; score that night's GPX visits only",
        "nights_total": int(len(detail)),
        "nights_scored": int(len(ok)),
        "nights_skipped_insufficient_history": int((detail.status == "insufficient-training-history").sum()) if not detail.empty else 0,
        "top_k_segments": top_k,
    }
    if not ok.empty:
        summary.update({
            "actual_capture_events": int(ok.actual_capture_events.sum()),
            "predicted_capture_count_rounded_total": int(ok.predicted_capture_count_rounded.sum()),
            "count_mae_per_night": float(ok.count_absolute_error.mean()),
            "count_exact_nights": int((ok.predicted_capture_count_rounded == ok.actual_capture_events).sum()),
            "count_exact_rate": float((ok.predicted_capture_count_rounded == ok.actual_capture_events).mean()),
            "location_hit_events_top_k": int(ok.location_hits_top_k.sum()),
            "location_eligible_events": int(ok.actual_road_matched_capture_events.sum()),
            "location_hit_rate_top_k": None if int(ok.actual_road_matched_capture_events.sum()) == 0 else float(ok.location_hits_top_k.sum() / ok.actual_road_matched_capture_events.sum()),
            "time_hit_events_peak_hour": int(ok.time_hits_peak_hour.sum()),
            "time_eligible_events": int(ok.actual_capture_events.sum()),
            "time_hit_rate_peak_hour": None if int(ok.actual_capture_events.sum()) == 0 else float(ok.time_hits_peak_hour.sum() / ok.actual_capture_events.sum()),
        })
    (out_dir / "walk_forward_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
