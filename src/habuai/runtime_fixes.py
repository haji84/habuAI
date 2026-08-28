from __future__ import annotations

import pandas as pd


def apply_runtime_fixes(pipeline) -> None:
    """Small safety fixes kept separate while the foundation PR is being hardened."""

    def species_from_text(text: str) -> str:
        # Specific names must precede the substring "ハブ" or ヒメハブ is contaminated as main Habu.
        keys = [
            "ヒメハブ",
            "ハブ",
            "アカマタ",
            "ガラスヒバァ",
            "ガラスヒヴァ",
            "リュウキュウアオヘビ",
            "ヒャン",
            "ネズミ",
            "オットンガエル",
            "イシカワガエル",
            "アマミハナサキガエル",
            "カエル",
            "ヤマシギ",
            "クロウサギ",
        ]
        for key in keys:
            if key in text:
                return key
        return "その他"

    original_static = pipeline.add_segment_static_features

    def add_segment_static_features(df, segs):
        if df.empty:
            cols = [
                "segment_id",
                "highway",
                "length_m",
                "bearing_deg",
                "curvature_deg",
            ]
            out = df.copy()
            for col in cols:
                if col not in out.columns:
                    out[col] = pd.Series(dtype="object" if col in {"segment_id", "highway"} else "float64")
            return out
        if "segment_id" not in df.columns:
            raise ValueError("non-empty learning rows are missing required segment_id")
        return original_static(df, segs)

    def join_weather(visits, weather):
        if visits.empty or weather.empty:
            return visits
        v = visits.copy()
        w = weather.copy()
        # GPX timestamps arrive as a fixed +09:00 offset while Open-Meteo uses
        # the named Asia/Tokyo zone. Pandas merge_asof requires identical tz dtypes.
        # Normalize both through UTC, then return them to Asia/Tokyo so local-hour
        # features remain correct.
        v["entered_at"] = pd.to_datetime(v["entered_at"], utc=True).dt.tz_convert("Asia/Tokyo")
        v["exited_at"] = pd.to_datetime(v["exited_at"], utc=True).dt.tz_convert("Asia/Tokyo")
        w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
        v = v.sort_values("entered_at")
        w = w.sort_values("timestamp")
        return pd.merge_asof(
            v,
            w,
            left_on="entered_at",
            right_on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("40min"),
        )

    pipeline._species_from_text = species_from_text
    pipeline.add_segment_static_features = add_segment_static_features
    pipeline.join_weather = join_weather
