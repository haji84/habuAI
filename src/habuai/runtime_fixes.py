from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .hardening import apply_hardening

OPERATIONAL_BOUNDARY_HOUR = 7
ACTUAL_GPX_STRICT_MIN_MATCH_RATIO = 0.80
CAPTURE_LABEL_FALLBACK_MAX_DISTANCE_M = 50.0
CAPTURE_LABEL_FALLBACK_MAX_TIME_MINUTES = 10


def canonicalize_operational_night(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the single canonical 07:00 Asia/Tokyo operational-night rule."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")

    def to_jst(value):
        if pd.isna(value):
            return pd.NaT
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            return stamp.tz_localize("Asia/Tokyo")
        return stamp.tz_convert("Asia/Tokyo")

    jst = ts.map(to_jst)
    out["night_date"] = jst.map(
        lambda value: (
            (value - pd.Timedelta(hours=OPERATIONAL_BOUNDARY_HOUR)).date().isoformat()
            if not pd.isna(value)
            else None
        )
    )
    out["operational_date_0700"] = out["night_date"]
    return out


def species_from_text_specific_first(text: str) -> str:
    """Parse species without letting a short name swallow a longer one."""
    keys = [
        "ヒメハブ", "ガラスヒバァ", "ガラスヒヴァ", "リュウキュウアオヘビ",
        "アカマタ", "ヒャン", "ハブ", "オットンガエル", "イシカワガエル",
        "アマミハナサキガエル", "カエル", "ヤマシギ", "クロウサギ", "ネズミ",
    ]
    for key in keys:
        if key in text:
            return key
    return "その他"


def mark_hindsight_weather_evaluation(score: dict) -> dict:
    """Prevent archive-weather holdout scores from being reported as strict v2 accuracy."""
    out = dict(score or {})
    out["evaluation_mode"] = "DIAGNOSTIC_HINDSIGHT_WEATHER"
    out["strict_no_leakage_eligible"] = False
    out["weather_feature_source"] = "open_meteo_archive_actual_or_reanalysis"
    out["strict_blocker"] = (
        "holdout features use archive weather; official v2 accuracy requires a frozen "
        "forecast snapshot available at prediction generation time"
    )
    return out


def enforce_actual_gpx_session_match_gate(matched: pd.DataFrame) -> pd.DataFrame:
    """Invalidate entire GPX sessions whose road-map coverage is below 80%."""
    if matched is None or matched.empty or "session_file" not in matched.columns:
        return matched.copy() if matched is not None else pd.DataFrame()

    out = matched.copy()
    if "segment_id" not in out.columns:
        out["segment_id"] = pd.NA

    ratios = out.groupby("session_file")["segment_id"].apply(lambda s: float(s.notna().mean()))
    out["session_map_match_ratio"] = out["session_file"].map(ratios)
    out["strict_session_eligible"] = (
        out["session_map_match_ratio"] >= ACTUAL_GPX_STRICT_MIN_MATCH_RATIO
    )
    bad = ~out["strict_session_eligible"]
    out.loc[bad, "segment_id"] = pd.NA
    return out


def _rescue_capture_labels_to_observed_visits(
    visits: pd.DataFrame,
    events: pd.DataFrame,
    segs: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Attach confirmed Habu captures to an actually observed GPX visit.

    The base pipeline first tries exact 10 m segment-id + ±10 min matching. Field-log
    coordinates/timestamps can lag the track by a few minutes, so an exact segment id
    can drop a real positive and turn it into a false negative. For confirmed user
    Habu captures only, rescue unmatched events to the nearest visited road segment
    within 50 m and ±10 min. This is a training-label rule, not the official
    100 m ±20 min evaluation KPI.
    """
    if visits is None or visits.empty:
        return visits

    out = visits.copy()
    out["outcome_label_method"] = "surveyed_non_capture"
    out.loc[out.get("habu_capture", 0).astype(int) > 0, "outcome_label_method"] = "exact_segment_10m"
    out["label_event_distance_m"] = pd.NA
    out["label_time_offset_s"] = pd.NA

    if events is None or events.empty or segs is None or segs.empty:
        return out

    habu = events[(events.species == "ハブ") & (events.event_type == "捕獲")].copy()
    if habu.empty:
        return out

    seg_geom = segs[["segment_id", "geometry"]].drop_duplicates("segment_id").set_index("segment_id")
    metric_crs = segs.crs

    for event in habu.itertuples():
        exact = (
            (out.segment_id == getattr(event, "segment_id", None))
            & (abs((out.entered_at - event.timestamp).dt.total_seconds()) <= 600)
            & (out.habu_capture.astype(int) > 0)
        )
        if exact.any():
            continue
        if pd.isna(event.lat) or pd.isna(event.lon):
            continue

        time_mask = abs((out.entered_at - event.timestamp).dt.total_seconds()) <= (
            CAPTURE_LABEL_FALLBACK_MAX_TIME_MINUTES * 60
        )
        candidates = out.loc[time_mask].copy()
        if candidates.empty:
            continue

        event_geom = gpd.GeoSeries(
            gpd.points_from_xy([event.lon], [event.lat]), crs="EPSG:4326"
        ).to_crs(metric_crs).iloc[0]
        distances = []
        for segment_id in candidates.segment_id:
            if segment_id not in seg_geom.index:
                distances.append(float("nan"))
            else:
                distances.append(float(event_geom.distance(seg_geom.loc[segment_id].geometry)))
        candidates["_event_distance_m"] = distances
        candidates = candidates.dropna(subset=["_event_distance_m"])
        if candidates.empty:
            continue

        best_idx = candidates["_event_distance_m"].idxmin()
        best_distance = float(candidates.loc[best_idx, "_event_distance_m"])
        if best_distance > CAPTURE_LABEL_FALLBACK_MAX_DISTANCE_M:
            continue

        out.loc[best_idx, "habu_capture"] = 1
        out.loc[best_idx, "habu_individuals"] = int(out.loc[best_idx, "habu_individuals"] or 0) + int(event.individual_count or 1)
        out.loc[best_idx, "outcome_label_method"] = "spatiotemporal_fallback_50m_10min"
        out.loc[best_idx, "label_event_distance_m"] = best_distance
        out.loc[best_idx, "label_time_offset_s"] = float(
            (out.loc[best_idx, "entered_at"] - event.timestamp).total_seconds()
        )

    return out


def apply_runtime_fixes(pipeline) -> None:
    """Apply canonical IDs, label guards, map-match gates, and no-leakage guards."""
    apply_hardening(pipeline)

    pipeline._species_from_text = species_from_text_specific_first
    original_parse_field_log = pipeline.parse_field_log

    def parse_field_log_0700(path):
        return canonicalize_operational_night(original_parse_field_log(path))

    pipeline.parse_field_log = parse_field_log_0700

    original_map_match_gpx = pipeline.map_match_gpx

    def map_match_gpx_strict(points, segs, cfg):
        return enforce_actual_gpx_session_match_gate(original_map_match_gpx(points, segs, cfg))

    pipeline.map_match_gpx = map_match_gpx_strict

    segment_context = {}
    original_match_events = pipeline.match_events

    def match_events_with_segment_context(events, segs, max_m=50.0):
        segment_context["segs"] = segs
        return original_match_events(events, segs, max_m=max_m)

    pipeline.match_events = match_events_with_segment_context

    original_add_outcomes_and_bio = pipeline.add_outcomes_and_bio

    def add_outcomes_and_bio_exposure_safe(visits, events, cfg):
        base = original_add_outcomes_and_bio(visits, events, cfg)
        return _rescue_capture_labels_to_observed_visits(base, events, segment_context.get("segs"))

    pipeline.add_outcomes_and_bio = add_outcomes_and_bio_exposure_safe

    original_score_holdout = pipeline.score_holdout

    def score_holdout_no_leakage(root, data, cfg):
        return mark_hindsight_weather_evaluation(original_score_holdout(root, data, cfg))

    pipeline.score_holdout = score_holdout_no_leakage
