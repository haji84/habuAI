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

    pipeline._species_from_text = species_from_text
    pipeline.add_segment_static_features = add_segment_static_features
