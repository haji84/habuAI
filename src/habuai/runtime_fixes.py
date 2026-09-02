from __future__ import annotations

import pandas as pd

from .hardening import apply_hardening

OPERATIONAL_BOUNDARY_HOUR = 7
ACTUAL_GPX_STRICT_MIN_MATCH_RATIO = 0.80


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
    """Invalidate entire GPX sessions whose road-map coverage is below 80%.

    Using only the matched subset of a poor session would create selective Road×Time
    exposure and false negatives. Keep the raw matched rows for QC, but clear their
    `segment_id` so the downstream `segment_visits()` step cannot use that session for
    strict training/evaluation. The match ratio is attached to every row for audit.
    """
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


def apply_runtime_fixes(pipeline) -> None:
    """Apply canonical IDs, label guards, map-match gates, and no-leakage guards."""
    apply_hardening(pipeline)

    pipeline._species_from_text = species_from_text_specific_first
    original_parse_field_log = pipeline.parse_field_log

    def parse_field_log_0700(path):
        return canonicalize_operational_night(original_parse_field_log(path))

    pipeline.parse_field_log = parse_field_log_0700

    # Enforce the same ACTUAL_GPX strict map-match rule in the production dataset
    # builder that the v2 audit already uses. Poor sessions cannot contribute a
    # cherry-picked matched subset as Road×10min exposure/negatives.
    original_map_match_gpx = pipeline.map_match_gpx

    def map_match_gpx_strict(points, segs, cfg):
        return enforce_actual_gpx_session_match_gate(original_map_match_gpx(points, segs, cfg))

    pipeline.map_match_gpx = map_match_gpx_strict

    original_score_holdout = pipeline.score_holdout

    def score_holdout_no_leakage(root, data, cfg):
        return mark_hindsight_weather_evaluation(original_score_holdout(root, data, cfg))

    pipeline.score_holdout = score_holdout_no_leakage
