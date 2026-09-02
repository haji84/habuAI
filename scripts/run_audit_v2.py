from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from habuai.audit_v2 import build_night_audit, write_audit_outputs
from habuai.evidence_policy import (
    derive_dense_field_sequence_nights,
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


def _merge_quarantine_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    merged = pd.concat(nonempty, ignore_index=True)
    if "operational_date_0700" in merged.columns:
        merged = merged.drop_duplicates(subset=["operational_date_0700"], keep="first")
        merged = merged.sort_values("operational_date_0700").reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the Habu AI v2 audit from exploration evidence first."
    )
    parser.add_argument("--gpx-points", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--gps-history", type=Path)
    parser.add_argument(
        "--exploration-nights",
        type=Path,
        help=(
            "Optional historical/reviewed candidate-night CSV. Entries still pass through "
            "the population-conflict quarantine; independent GPX/GPS/session evidence can "
            "re-establish a quarantined night inside build_night_audit."
        ),
    )
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
    imported_nights = _read_nights(args.exploration_nights)
    conflicts = load_population_conflicts(args.population_conflicts)

    # Imported night lists are candidates, not authority. This prevents the historical
    # 89-night workbook (or another derived list) from bypassing reviewed provenance conflicts.
    safe_imported_nights, quarantined_imported = quarantine_unresolved_nights(
        imported_nights, conflicts
    )

    # Confirmed self-captures / explicit zero rows are strong event evidence, but
    # historical recovered versions may still be quarantined when provenance conflicts.
    strong_event_nights = derive_exploration_nights_from_events(events)
    safe_event_nights, quarantined_events = quarantine_unresolved_nights(
        strong_event_nights, conflicts
    )

    # Dense time-ordered field sequences are independent exploration evidence.
    # They can therefore establish a night even when a recovered capture row is weak.
    dense_field_nights = derive_dense_field_sequence_nights(events)
    nights = merge_exploration_night_sources(
        safe_imported_nights,
        safe_event_nights,
        dense_field_nights,
    )

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
    quarantined = _merge_quarantine_frames(quarantined_imported, quarantined_events)
    if not quarantined.empty:
        quarantined.to_csv(args.out_dir / "v2_population_quarantine.csv", index=False)

    print(f"imported_candidate_nights={len(imported_nights)}")
    print(f"strong_event_nights={len(strong_event_nights)}")
    print(f"dense_field_nights={len(dense_field_nights)}")
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
