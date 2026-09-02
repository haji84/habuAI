from __future__ import annotations

import pandas as pd

from habuai.audit_v2 import CLASS_ACTUAL_GPX, add_operational_date_0700

ACTUAL_GPX_STRICT_MIN_MATCH_RATIO = 0.80


def actual_gpx_match_quality(gpx_points: pd.DataFrame) -> pd.DataFrame:
    """Summarize road-map-match coverage for timestamped actual GPX nights.

    Missing ``segment_id`` means the raw GPX has not been road-map-matched yet and
    therefore cannot generate strict Road x 10 min negatives.
    """
    columns = [
        "operational_date_0700",
        "actual_gpx_total_points_for_gate",
        "actual_gpx_matched_points_for_gate",
        "actual_gpx_match_ratio",
        "actual_gpx_map_match_ready",
    ]
    if gpx_points is None or gpx_points.empty:
        return pd.DataFrame(columns=columns)

    g = add_operational_date_0700(gpx_points, "timestamp")
    if "segment_id" not in g.columns:
        g["segment_id"] = pd.NA

    rows = []
    for night, x in g.groupby("operational_date_0700"):
        total = int(len(x))
        matched = int(x["segment_id"].notna().sum())
        ratio = float(matched / total) if total else 0.0
        rows.append(
            {
                "operational_date_0700": str(night),
                "actual_gpx_total_points_for_gate": total,
                "actual_gpx_matched_points_for_gate": matched,
                "actual_gpx_match_ratio": ratio,
                "actual_gpx_map_match_ready": ratio >= ACTUAL_GPX_STRICT_MIN_MATCH_RATIO,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def enforce_actual_gpx_map_match_gate(
    audit: pd.DataFrame,
    gpx_points: pd.DataFrame,
) -> pd.DataFrame:
    """Block strict actual-GPX use until road map matching reaches the threshold.

    Classification remains ``ACTUAL_GPX`` because provenance is unchanged. Only
    Road x 10 min train/eval and NO_CAPTURE_OBSERVED eligibility are gated.
    """
    if audit is None or audit.empty:
        return audit.copy()

    quality = actual_gpx_match_quality(gpx_points)
    out = audit.copy()
    out = out.merge(quality, on="operational_date_0700", how="left")
    out["actual_gpx_match_ratio"] = out["actual_gpx_match_ratio"].fillna(0.0)
    out["actual_gpx_map_match_ready"] = out["actual_gpx_map_match_ready"].fillna(False)

    actual = out["classification"].eq(CLASS_ACTUAL_GPX)
    blocked = actual & ~out["actual_gpx_map_match_ready"]
    out.loc[blocked, "usable_road_10min_train"] = False
    out.loc[blocked, "usable_road_10min_eval"] = False
    out.loc[blocked, "can_generate_no_capture_observed"] = False

    if "night_observation_label" in out.columns:
        no_capture = pd.to_numeric(out.get("capture_count", 0), errors="coerce").fillna(0).eq(0)
        out.loc[blocked & no_capture, "night_observation_label"] = "Unknown"

    if "limitation_reason" in out.columns:
        reason = (
            "actual GPX continuity is valid, but strict Road x 10 min and "
            f"NO_CAPTURE_OBSERVED require >= {ACTUAL_GPX_STRICT_MIN_MATCH_RATIO:.0%} "
            "road-map-matched GPX points"
        )
        out.loc[blocked, "limitation_reason"] = reason

    return out
