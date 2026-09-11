#!/usr/bin/env python3
"""Export canonical OSM segment_id × 10-minute training rows.

This script consumes the *canonical* observation table produced by
src/habuai/pipeline.py. It does not perform fallback spatial matching and
does not create negative labels outside actual surveyed visits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def operational_night(ts: pd.Series, rollover_hour: int = 7) -> pd.Series:
    t = pd.to_datetime(ts)
    shifted = t - pd.to_timedelta(rollover_hour, unit="h")
    return shifted.dt.date.astype(str)


def build(df: pd.DataFrame, night: str, rollover_hour: int = 7) -> pd.DataFrame:
    required = {"segment_id", "entered_at", "exited_at", "habu_capture"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing canonical columns: {sorted(missing)}")

    x = df.dropna(subset=["segment_id", "entered_at"]).copy()
    x["entered_at"] = pd.to_datetime(x["entered_at"])
    x["exited_at"] = pd.to_datetime(x["exited_at"])
    x["night_date"] = operational_night(x["entered_at"], rollover_hour)
    x = x[x["night_date"] == night].copy()
    if x.empty:
        return x

    x["bin_start"] = x["entered_at"].dt.floor("10min")
    x["duration_s"] = (
        x["exited_at"] - x["entered_at"]
    ).dt.total_seconds().clip(lower=0)

    # A row exists only if the segment was actually observed by GPX.
    # This is an opportunity-aware negative: habu_capture=0 means
    # "no user capture during this observed visit", never biological absence.
    agg_map = {
        "entered_at": "min",
        "exited_at": "max",
        "duration_s": "sum",
        "habu_capture": "max",
    }
    for c in [
        "habu_individuals", "mean_speed_mps", "mean_match_distance_m",
        "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm",
        "rain_24h_mm", "rain_48h_mm", "temperature_c", "humidity_pct",
        "dew_point_c", "hours_since_rain", "curvature_deg",
        "segment_prior_visits",
    ]:
        if c in x.columns:
            agg_map[c] = "mean" if c not in {"habu_individuals"} else "sum"

    bio_cols = [c for c in x.columns if c.startswith("bio_")]
    for c in bio_cols:
        agg_map[c] = "max"

    out = (
        x.groupby(["night_date", "bin_start", "segment_id"], as_index=False)
        .agg(agg_map)
        .rename(columns={"habu_capture": "target_user_capture"})
    )
    out["survey_opportunity"] = 1
    out["target_user_capture"] = out["target_user_capture"].astype(int)
    out["observed_negative"] = (out["target_user_capture"] == 0).astype(int)
    return out.sort_values(["bin_start", "segment_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/observations.parquet")
    ap.add_argument("--night", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rollover-hour", type=int, default=7)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    out = build(df, args.night, args.rollover_hour)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(
        f"night={args.night} rows={len(out)} "
        f"positives={int(out.target_user_capture.sum()) if len(out) else 0}"
    )


if __name__ == "__main__":
    main()
