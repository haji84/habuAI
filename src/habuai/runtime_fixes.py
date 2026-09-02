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


def apply_runtime_fixes(pipeline) -> None:
    """Apply hardening plus the canonical operational-night policy."""
    apply_hardening(pipeline)

    original_parse_field_log = pipeline.parse_field_log

    def parse_field_log_0700(path):
        return canonicalize_operational_night(original_parse_field_log(path))

    pipeline.parse_field_log = parse_field_log_0700
