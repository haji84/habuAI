from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from habuai.audit_v2 import build_night_audit, write_audit_outputs
from habuai.evidence_policy import (
    derive_exploration_nights_from_events,
    load_population_conflicts,
    merge_exploration_night_sources,
    quarantine_unresolved_nights,
)


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_nights(path: Path | None) -> list[str]:
    if path is None:
        return []
    df = pd.read_csv(path)
    for column in ("operational_date_0700", "operational_date", "date"):
        if column in df.columns:
            return sorted({str(v) for v in df[column].dropna()})
    if len(df.columns) == 1:
        return sorted({str(v) for v in df.iloc[:, 0].dropna()})
    raise ValueError("exploration nights CSV needs operational_date_0700/date column")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the Habu AI v2 audit from exploration evidence first."
    )
    parser.add_argument("--gpx-points", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--gps-history", type=Path)
    parser.add_argument("--exploration-nights", type=Path)
    parser.add_argument(
        "--population-conflicts",
        type=Path,
        default=Path("config/v2_population_conflicts.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/v2_audit"))
    parser.add_argument(
        "--expected-nights",
        type=int,
        default=None,
        help=(
            "Optional validation only. Do not use this to force the historical 89-night count; "
            "canonical population is derived from exploration evidence."
        ),
    )
    args = parser.parse_args()

    gpx = _read_csv(args.gpx_points)
    events = _read_csv(args.events)
    gps_history = _read_csv(args.gps_history)
    explicit_nights = _read_nights(args.exploration_nights)

    # A confirmed self-capture proves that the user was exploring that night.
    # Roadkill/sighting/weather events do not create exploration nights by themselves.
    strong_event_nights = derive_exploration_nights_from_events(events)
    proposed_nights = merge_exploration_night_sources(explicit_nights, strong_event_nights)

    conflicts = load_population_conflicts(args.population_conflicts)
    nights, quarantined = quarantine_unresolved_nights(proposed_nights, conflicts)

    audit = build_night_audit(
        gpx_points=gpx,
        events=events,
        gps_history=gps_history,
        exploration_nights=nights,
    )

    if args.expected_nights is not None and len(audit) != args.expected_nights:
        raise SystemExit(
            f"audit population mismatch: expected {args.expected_nights}, got {len(audit)}. "
            "Fix exploration evidence population instead of padding/removing nights."
        )

    if not audit.empty and audit["operational_date_0700"].duplicated().any():
        duplicated = audit.loc[
            audit["operational_date_0700"].duplicated(), "operational_date_0700"
        ].tolist()
        raise SystemExit(f"duplicate operational nights: {duplicated}")

    write_audit_outputs(audit, args.out_dir)
    if not quarantined.empty:
        quarantined.to_csv(args.out_dir / "v2_population_quarantine.csv", index=False)

    print(f"strong_event_nights={len(strong_event_nights)}")
    print(f"quarantined_nights={len(quarantined)}")
    if not audit.empty:
        print(audit["classification"].value_counts().sort_index().to_string())
        print(f"road_10min_eval={int(audit['usable_road_10min_eval'].sum())}")
        print(
            "no_capture_observed="
            f"{int((audit['night_observation_label'] == 'NO_CAPTURE_OBSERVED').sum())}"
        )
    print(f"nights={len(audit)}")


if __name__ == "__main__":
    main()
