from __future__ import annotations

import pandas as pd

UNIQUE_ROUTE_CLASSES = {"確定一本道区間", "UNIQUE_ROUTE"}


def canonical_10min_bin(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Tokyo")
    else:
        ts = ts.tz_convert("Asia/Tokyo")
    return ts.floor("10min")


def verify_reconstructed_intervals(
    intervals: pd.DataFrame,
    *,
    start_col: str = "start_time",
    end_col: str = "end_time",
    route_class_col: str = "route_class",
) -> pd.DataFrame:
    """Conservatively mark reconstructed route intervals usable for Road×10min.

    A reconstructed interval is strict only when:
    1. the road route between timed anchors is unique,
    2. both anchor times are valid and ordered,
    3. the entire interval lies inside one canonical 10-minute bin.

    This is interval-censored evidence. It does not invent a precise passage time.
    Candidate routes, alternative-route intervals and intervals crossing a 10-minute
    boundary remain non-strict and cannot generate `NO_CAPTURE_OBSERVED` labels.
    """
    required = {start_col, end_col, route_class_col}
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"interval table missing required columns: {sorted(missing)}")

    out = intervals.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")
    valid_time = out[start_col].notna() & out[end_col].notna() & (out[end_col] >= out[start_col])

    start_bin = out[start_col].map(lambda x: canonical_10min_bin(x) if pd.notna(x) else pd.NaT)
    end_bin = out[end_col].map(lambda x: canonical_10min_bin(x) if pd.notna(x) else pd.NaT)
    unique_route = out[route_class_col].astype(str).isin(UNIQUE_ROUTE_CLASSES)
    same_bin = start_bin.eq(end_bin)

    out["interval_minutes"] = (
        (out[end_col] - out[start_col]).dt.total_seconds() / 60.0
    )
    out["road10_bin_start"] = start_bin
    out["strict_road10_verified"] = valid_time & unique_route & same_bin
    out["can_generate_no_capture_observed"] = out["strict_road10_verified"]
    out["verification_reason"] = ""
    out.loc[~valid_time, "verification_reason"] = "invalid_or_missing_anchor_time"
    out.loc[valid_time & ~unique_route, "verification_reason"] = "route_not_unique"
    out.loc[valid_time & unique_route & ~same_bin, "verification_reason"] = "crosses_10min_bin"
    out.loc[out["strict_road10_verified"], "verification_reason"] = "unique_route_inside_single_10min_bin"
    return out


def summarize_verified_intervals(
    verified: pd.DataFrame,
    *,
    night_col: str = "operational_date_0700",
) -> pd.DataFrame:
    if verified.empty:
        return pd.DataFrame(
            columns=[
                night_col,
                "interval_count",
                "strict_interval_count",
                "strict_interval_ratio",
            ]
        )
    if night_col not in verified.columns:
        raise ValueError(f"verified interval table missing {night_col}")

    rows = []
    for night, group in verified.groupby(night_col):
        total = int(len(group))
        strict = int(group["strict_road10_verified"].fillna(False).sum())
        rows.append(
            {
                night_col: str(night),
                "interval_count": total,
                "strict_interval_count": strict,
                "strict_interval_ratio": strict / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(night_col).reset_index(drop=True)
