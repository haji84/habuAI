from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from habuai.audit_v2 import operational_date_0700
from habuai.pipeline import _add_track_dynamics, map_match_gpx


def prepare_gps_history_for_map_match(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamped device/history anchors for the existing GPX matcher.

    Reconstructed GPS remains provenance-distinct from actual GPX: each operational
    night is assigned a synthetic `gps_history:<YYYY-MM-DD>` session_file.
    """
    required = {"timestamp", "lat", "lon"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"GPS history missing required columns: {sorted(missing)}")

    h = history.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce")
    h["lat"] = pd.to_numeric(h["lat"], errors="coerce")
    h["lon"] = pd.to_numeric(h["lon"], errors="coerce")
    h = h.dropna(subset=["timestamp", "lat", "lon"]).copy()
    if h.empty:
        return h

    h["operational_date_0700"] = h["timestamp"].map(
        lambda value: operational_date_0700(value).isoformat()
    )
    h["session_file"] = h["operational_date_0700"].map(
        lambda value: f"gps_history:{value}"
    )
    h = h.sort_values(["operational_date_0700", "timestamp"]).reset_index(drop=True)
    h["seq"] = h.groupby("session_file").cumcount()
    if "elevation_m" not in h.columns:
        h["elevation_m"] = pd.NA

    return _add_track_dynamics(h)


def select_operational_nights(
    history: pd.DataFrame,
    nights: set[str] | list[str] | tuple[str, ...] | None,
) -> pd.DataFrame:
    if nights is None or history.empty:
        return history.copy()
    allowed = {str(value) for value in nights}
    return history[history["operational_date_0700"].astype(str).isin(allowed)].copy()


def map_match_gps_history(
    history: pd.DataFrame,
    road_segments_path: Path,
    cfg: dict,
) -> pd.DataFrame:
    prepared = prepare_gps_history_for_map_match(history)
    if prepared.empty:
        return prepared

    segments = gpd.read_file(road_segments_path)
    if segments.crs is None:
        raise ValueError("road segment layer must have a CRS")
    return map_match_gpx(prepared, segments, cfg)


def summarize_history_map_match(matched: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "operational_date_0700",
        "anchor_count",
        "matched_anchor_count",
        "match_ratio",
        "median_match_distance_m",
    ]
    if matched.empty:
        return pd.DataFrame(columns=columns)

    m = matched.copy()
    if "operational_date_0700" not in m.columns:
        m["operational_date_0700"] = m["timestamp"].map(
            lambda value: operational_date_0700(value).isoformat()
        )
    if "segment_id" not in m.columns:
        m["segment_id"] = pd.NA
    if "match_distance_m" not in m.columns:
        m["match_distance_m"] = pd.NA

    rows = []
    for night, group in m.groupby("operational_date_0700"):
        anchor_count = int(len(group))
        matched_mask = group["segment_id"].notna()
        matched_count = int(matched_mask.sum())
        distances = pd.to_numeric(
            group.loc[matched_mask, "match_distance_m"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "operational_date_0700": str(night),
                "anchor_count": anchor_count,
                "matched_anchor_count": matched_count,
                "match_ratio": matched_count / anchor_count if anchor_count else 0.0,
                "median_match_distance_m": (
                    float(distances.median()) if len(distances) else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("operational_date_0700").reset_index(drop=True)
