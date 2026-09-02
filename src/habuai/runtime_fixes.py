from __future__ import annotations

import pandas as pd

from .hardening import apply_hardening

OPERATIONAL_BOUNDARY_HOUR = 7


def canonicalize_operational_night(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the single canonical 07:00 Asia/Tokyo operational-night rule.

    00:00:00 through 06:59:59 belong to the previous operational night. The
    timestamp column is authoritative; an earlier session-start date must not
    silently override the shared audit/training boundary.
    """
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
        "ヒメハブ",
        "ガラスヒバァ",
        "ガラスヒヴァ",
        "リュウキュウアオヘビ",
        "アカマタ",
        "ヒャン",
        "ハブ",
        "オットンガエル",
        "イシカワガエル",
        "アマミハナサキガエル",
        "カエル",
        "ヤマシギ",
        "クロウサギ",
        "ネズミ",
    ]
    for key in keys:
        if key in text:
            return key
    return "その他"


def mark_hindsight_weather_evaluation(score: dict) -> dict:
    """Prevent archive-weather holdout scores from being reported as strict v2 accuracy.

    Historical archive/reanalysis weather can be useful for retrospective model
    development, but it was not necessarily available at the time the prediction
    would have been generated. Strict v2 evaluation requires a frozen forecast
    snapshot whose issued/acquired timestamps precede prediction generation.
    """
    out = dict(score or {})
    out["evaluation_mode"] = "DIAGNOSTIC_HINDSIGHT_WEATHER"
    out["strict_no_leakage_eligible"] = False
    out["weather_feature_source"] = "open_meteo_archive_actual_or_reanalysis"
    out["strict_blocker"] = (
        "holdout features use archive weather; official v2 accuracy requires a frozen "
        "forecast snapshot available at prediction generation time"
    )
    return out


def apply_runtime_fixes(pipeline) -> None:
    """Apply hardening, canonical IDs, species parsing, and no-leakage guards."""
    apply_hardening(pipeline)

    pipeline._species_from_text = species_from_text_specific_first
    original_parse_field_log = pipeline.parse_field_log

    def parse_field_log_0700(path):
        return canonicalize_operational_night(original_parse_field_log(path))

    pipeline.parse_field_log = parse_field_log_0700

    # The current pipeline joins Open-Meteo archive weather to historical visits.
    # Keep those results for retrospective diagnostics, but never let latest_score
    # masquerade as the strict pre-search forecast evaluation required by Habu AI v2.
    original_score_holdout = pipeline.score_holdout

    def score_holdout_no_leakage(root, data, cfg):
        return mark_hindsight_weather_evaluation(original_score_holdout(root, data, cfg))

    pipeline.score_holdout = score_holdout_no_leakage
