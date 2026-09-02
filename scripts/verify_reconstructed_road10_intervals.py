from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from habuai.temporal_route_verification import (
    summarize_verified_intervals,
    verify_reconstructed_intervals,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify interval-censored reconstructed routes for strict Road×10min use."
    )
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/reconstructed_road10_verified.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/v2_reconstructed_road10_interval_summary.csv"),
    )
    args = parser.parse_args()

    intervals = pd.read_csv(args.intervals)
    verified = verify_reconstructed_intervals(intervals)
    summary = summarize_verified_intervals(verified)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"intervals={len(verified)}")
    print(f"strict_intervals={int(verified['strict_road10_verified'].sum()) if len(verified) else 0}")
    print(f"nights_with_intervals={len(summary)}")
    if not summary.empty:
        print(f"nights_with_strict_intervals={int((summary['strict_interval_count'] > 0).sum())}")


if __name__ == "__main__":
    main()
