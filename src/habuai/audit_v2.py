from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
import json

import pandas as pd

JST = timezone(timedelta(hours=9))
OPERATIONAL_BOUNDARY_HOUR = 7


CLASS_ACTUAL_GPX = "ACTUAL_GPX"
CLASS_RECONSTRUCTED_HIGH = "RECONSTRUCTED_GPS_HIGH"
CLASS_RECONSTRUCTED_PARTIAL = "RECONSTRUCTED_PARTIAL"
CLASS_SPATIAL_ONLY = "SPATIAL_ONLY_RECONSTRUCTION"
CLASS_UNRECONSTRUCTABLE = "UNRECONSTRUCTABLE"

ALL_CLASSES = {
    CLASS_ACTUAL_GPX,
    CLASS_RECONSTRUCTED_HIGH,
    CLASS_RECONSTRUCTED_PARTIAL,
    CLASS_SPATIAL_ONLY,
    CLASS_UNRECONSTRUCTABLE,
}


@dataclass(frozen=True)
class NightAudit:
    operational_date_0700: str
    classification: str
    evidence_summary: str
    actual_gpx_points: int = 0
    gps_anchor_count: int = 0
    route_confidence: float = 0.0
    time_confidence: float = 0.0
    usable_1km_area: bool = False
    usable_road: bool = False
    usable_road_10min_train: bool = False
    usable_road_10min_eval: bool = False
    can_generate_no_capture_observed: bool = False
    capture_count: int = 0
    event_count: int = 0
    limitation_reason: str = ""


def _to_jst_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        # Pipeline timestamps are intended to be local/JST once parsed.
        ts = ts.tz_localize("Asia/Tokyo")
    else:
        ts = ts.tz_convert("Asia/Tokyo")
    return ts


def operational_date_0700(value) -> date:
    """Return the canonical exploration-night date using a 07:00 boundary.

    00:00-06:59 belongs to the previous operational night.
    07:00-23:59 belongs to the calendar date.
    """
    ts = _to_jst_timestamp(value)
    shifted = ts - pd.Timedelta(hours=OPERATIONAL_BOUNDARY_HOUR)
    return shifted.date()


def add_operational_date_0700(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    out_col: str = "operational_date_0700",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype="object")
        return out
    out[out_col] = out[timestamp_col].map(lambda x: operational_date_0700(x).isoformat())
    return out


def _bool_thresholds(route_confidence: float, time_confidence: float) -> dict[str, bool]:
    usable_1km = route_confidence >= 0.40
    usable_road = route_confidence >= 0.65
    strict = route_confidence >= 0.85 and time_confidence >= 0.80
    return {
        "usable_1km_area": usable_1km,
        "usable_road": usable_road,
        "usable_road_10min_train": strict,
        "usable_road_10min_eval": strict,
        "can_generate_no_capture_observed": strict,
    }


def _actual_gpx_summary(gpx_points: pd.DataFrame) -> pd.DataFrame:
    """Aggregate actual timestamped GPX into canonical operational nights.

    Expected columns: session_file, timestamp. Optional: segment_id.
    Multiple copied GPX files can still be deduplicated upstream by SHA; this
    function intentionally does not guess whether two different files are copies.
    """
    if gpx_points.empty:
        return pd.DataFrame(
            columns=[
                "operational_date_0700",
                "actual_gpx_points",
                "gpx_session_files",
                "gpx_start",
                "gpx_end",
                "gpx_duration_minutes",
                "matched_segment_points",
            ]
        )

    g = add_operational_date_0700(gpx_points, "timestamp")
    g["timestamp"] = g["timestamp"].map(_to_jst_timestamp)
    if "session_file" not in g.columns:
        g["session_file"] = "unknown.gpx"
    if "segment_id" not in g.columns:
        g["segment_id"] = pd.NA

    summary = (
        g.groupby("operational_date_0700", as_index=False)
        .agg(
            actual_gpx_points=("timestamp", "size"),
            gpx_session_files=("session_file", lambda s: " | ".join(sorted(set(map(str, s))))),
            gpx_start=("timestamp", "min"),
            gpx_end=("timestamp", "max"),
            matched_segment_points=("segment_id", lambda s: int(s.notna().sum())),
        )
    )
    summary["gpx_duration_minutes"] = (
        summary["gpx_end"] - summary["gpx_start"]
    ).dt.total_seconds() / 60.0
    return summary


def _event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "operational_date_0700",
                "event_count",
                "capture_count",
                "gps_anchor_count",
            ]
        )

    e = add_operational_date_0700(events, "timestamp")
    if "event_type" not in e.columns:
        e["event_type"] = "unknown"
    if "species" not in e.columns:
        e["species"] = "unknown"
    if "lat" not in e.columns:
        e["lat"] = pd.NA
    if "lon" not in e.columns:
        e["lon"] = pd.NA

    def capture_rows(x: pd.DataFrame) -> int:
        event = x["event_type"].astype(str)
        species = x["species"].astype(str)
        is_capture = event.isin(["捕獲", "capture"])
        is_habu = species.eq("ハブ")
        return int((is_capture & is_habu).sum())

    rows = []
    for night, x in e.groupby("operational_date_0700"):
        gps_ok = x["lat"].notna() & x["lon"].notna()
        rows.append(
            {
                "operational_date_0700": night,
                "event_count": int(len(x)),
                "capture_count": capture_rows(x),
                "gps_anchor_count": int(gps_ok.sum()),
            }
        )
    return pd.DataFrame(rows)


def _optional_gps_history_summary(gps_history: pd.DataFrame) -> pd.DataFrame:
    """Summarize optional device GPS/location history.

    Expected columns: timestamp, lat, lon. Dense history is useful for
    reconstruction. This function does not invent route geometry or passage time.
    """
    if gps_history.empty:
        return pd.DataFrame(
            columns=[
                "operational_date_0700",
                "history_anchor_count",
                "history_start",
                "history_end",
                "history_span_minutes",
                "history_median_gap_minutes",
            ]
        )

    h = add_operational_date_0700(gps_history, "timestamp")
    h["timestamp"] = h["timestamp"].map(_to_jst_timestamp)
    h = h.sort_values(["operational_date_0700", "timestamp"])
    rows = []
    for night, x in h.groupby("operational_date_0700"):
        gps_ok = x.get("lat", pd.Series(index=x.index, dtype=float)).notna() & x.get(
            "lon", pd.Series(index=x.index, dtype=float)
        ).notna()
        t = x.loc[gps_ok, "timestamp"].sort_values()
        gaps = t.diff().dt.total_seconds().div(60.0).dropna()
        rows.append(
            {
                "operational_date_0700": night,
                "history_anchor_count": int(len(t)),
                "history_start": t.min() if len(t) else pd.NaT,
                "history_end": t.max() if len(t) else pd.NaT,
                "history_span_minutes": (
                    float((t.max() - t.min()).total_seconds() / 60.0) if len(t) >= 2 else 0.0
                ),
                "history_median_gap_minutes": float(gaps.median()) if len(gaps) else None,
            }
        )
    return pd.DataFrame(rows)


def classify_night(row: pd.Series) -> NightAudit:
    """Classify a night without conflating actual GPX and reconstruction.

    Conservative rules intentionally prefer lower-confidence classes over
    pretending that event anchors form a continuous trajectory.
    """
    night = str(row["operational_date_0700"])
    actual_points = int(row.get("actual_gpx_points", 0) or 0)
    event_gps = int(row.get("gps_anchor_count", 0) or 0)
    history_anchors = int(row.get("history_anchor_count", 0) or 0)
    event_count = int(row.get("event_count", 0) or 0)
    capture_count = int(row.get("capture_count", 0) or 0)

    gpx_duration = float(row.get("gpx_duration_minutes", 0.0) or 0.0)
    matched_segment_points = int(row.get("matched_segment_points", 0) or 0)
    median_gap = row.get("history_median_gap_minutes", None)
    history_span = float(row.get("history_span_minutes", 0.0) or 0.0)

    if actual_points >= 100 and gpx_duration >= 30:
        matched_ratio = min(1.0, matched_segment_points / max(actual_points, 1))
        route_conf = max(0.90, 0.90 + 0.10 * matched_ratio)
        time_conf = 0.99
        classification = CLASS_ACTUAL_GPX
        evidence = f"timestamped actual GPX: {actual_points} points / {gpx_duration:.0f} min"
        limitation = ""

    elif history_anchors >= 20 and history_span >= 90 and median_gap is not None and median_gap <= 10:
        route_conf = 0.85
        time_conf = 0.82
        classification = CLASS_RECONSTRUCTED_HIGH
        evidence = (
            f"device GPS history: {history_anchors} anchors / {history_span:.0f} min / "
            f"median gap {median_gap:.1f} min"
        )
        limitation = "requires road-network map matching before strict use"

    elif history_anchors >= 4 or event_gps >= 4:
        route_conf = 0.65
        time_conf = 0.55
        classification = CLASS_RECONSTRUCTED_PARTIAL
        evidence = f"partial timed anchors: history={history_anchors}, event={event_gps}"
        limitation = "only high-confidence reconstructed subsegments may be used temporally"

    elif history_anchors >= 1 or event_gps >= 1:
        route_conf = 0.45
        time_conf = 0.20
        classification = CLASS_SPATIAL_ONLY
        evidence = f"spatial anchors only/insufficient temporal coverage: history={history_anchors}, event={event_gps}"
        limitation = "not valid for strict Road×10min temporal negatives"

    else:
        route_conf = 0.0
        time_conf = 0.0
        classification = CLASS_UNRECONSTRUCTABLE
        evidence = "no usable trajectory/GPS anchors found in supplied evidence"
        limitation = "remain Unknown for Road×Time until additional evidence is supplied"

    flags = _bool_thresholds(route_conf, time_conf)
    return NightAudit(
        operational_date_0700=night,
        classification=classification,
        evidence_summary=evidence,
        actual_gpx_points=actual_points,
        gps_anchor_count=history_anchors + event_gps,
        route_confidence=round(route_conf, 3),
        time_confidence=round(time_conf, 3),
        capture_count=capture_count,
        event_count=event_count,
        limitation_reason=limitation,
        **flags,
    )


def build_night_audit(
    gpx_points: pd.DataFrame,
    events: pd.DataFrame,
    gps_history: pd.DataFrame | None = None,
    exploration_nights: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build an exploration-evidence-first audit table.

    `exploration_nights` is the canonical session/evidence population when it is
    available. GPX/GPS/event evidence may add nights, but capture records are never
    used as the sole definition of what counts as an exploration night.
    """
    gps_history = gps_history if gps_history is not None else pd.DataFrame()
    gpx = _actual_gpx_summary(gpx_points)
    ev = _event_summary(events)
    hist = _optional_gps_history_summary(gps_history)

    explicit = {str(x) for x in (exploration_nights or [])}
    evidence_nights = set(gpx.get("operational_date_0700", [])) | set(
        hist.get("operational_date_0700", [])
    )

    # Event-only dates are included as evidence candidates so old records are not
    # lost, but they are NOT automatically interpreted as confirmed full searches.
    event_nights = set(ev.get("operational_date_0700", []))
    nights = sorted(explicit | evidence_nights | event_nights)

    base = pd.DataFrame({"operational_date_0700": nights})
    for part in [gpx, ev, hist]:
        base = base.merge(part, on="operational_date_0700", how="left")

    numeric_zero = [
        "actual_gpx_points",
        "matched_segment_points",
        "event_count",
        "capture_count",
        "gps_anchor_count",
        "history_anchor_count",
        "history_span_minutes",
        "gpx_duration_minutes",
    ]
    for c in numeric_zero:
        if c in base.columns:
            base[c] = base[c].fillna(0)

    audits = pd.DataFrame([asdict(classify_night(row)) for _, row in base.iterrows()])
    audits["capture_outcome"] = audits["capture_count"].map(lambda n: "CAPTURED" if n > 0 else "NO_CAPTURE_RECORDED")

    # Night-level zero outcome is only observationally strong when the trajectory is
    # sufficiently known. Otherwise the correct label is Unknown, not absence.
    audits["night_observation_label"] = audits.apply(
        lambda r: (
            "Positive"
            if r.capture_count > 0
            else "NO_CAPTURE_OBSERVED"
            if r.can_generate_no_capture_observed
            else "Unknown"
        ),
        axis=1,
    )
    return audits


def write_audit_outputs(audit: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v2_89_night_audit.csv"
    json_path = out_dir / "v2_89_night_audit_summary.json"
    audit.to_csv(csv_path, index=False)

    counts = audit["classification"].value_counts().reindex(sorted(ALL_CLASSES), fill_value=0)
    summary = {
        "night_count": int(len(audit)),
        "classification_counts": {k: int(v) for k, v in counts.items()},
        "road_10min_train_usable": int(audit["usable_road_10min_train"].sum()),
        "road_10min_eval_usable": int(audit["usable_road_10min_eval"].sum()),
        "no_capture_observed_nights": int((audit["night_observation_label"] == "NO_CAPTURE_OBSERVED").sum()),
        "unknown_nights": int((audit["night_observation_label"] == "Unknown").sum()),
        "note": "Population must be fixed from exploration/session evidence; event-only nights remain candidates unless search evidence confirms them.",
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
