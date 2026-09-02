from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from habuai.gps_history_mapmatch import (
    map_match_gps_history,
    prepare_gps_history_for_map_match,
    select_operational_nights,
    summarize_history_map_match,
)
from habuai.pipeline import load_config


def _read_night_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    frame = pd.read_csv(path, dtype=str)
    for column in ("operational_date_0700", "operational_date", "date"):
        if column in frame.columns:
            return set(frame[column].dropna().astype(str))
    raise ValueError("night filter CSV requires operational_date_0700/date column")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map-match reconstructed GPS-history anchors to the Habu AI road graph."
    )
    parser.add_argument("--gps-history", type=Path, required=True)
    parser.add_argument(
        "--road-segments",
        type=Path,
        default=Path("data/processed/road_segments_10m.geojson"),
    )
    parser.add_argument("--night-filter", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/gps_history_matched.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/v2_gps_history_mapmatch_summary.csv"),
    )
    args = parser.parse_args()

    history = pd.read_csv(args.gps_history)
    prepared = prepare_gps_history_for_map_match(history)
    nights = _read_night_filter(args.night_filter)
    prepared = select_operational_nights(prepared, nights)

    # map_match_gps_history performs preparation itself, so pass only the canonical
    # timestamp/lat/lon rows after filtering to avoid changing provenance semantics.
    input_columns = [c for c in history.columns if c in prepared.columns]
    filtered = prepared[input_columns].copy() if input_columns else prepared[["timestamp", "lat", "lon"]].copy()

    cfg = load_config(Path("."))
    matched = map_match_gps_history(filtered, args.road_segments, cfg)
    summary = summarize_history_map_match(matched)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"anchors={len(matched)}")
    print(f"nights={len(summary)}")
    if not summary.empty:
        passed = (
            (summary["match_ratio"] >= 0.80)
            & (summary["median_match_distance_m"] <= 25.0)
        )
        print(f"strict_mapmatch_pass={int(passed.sum())}")


if __name__ == "__main__":
    main()
