from __future__ import annotations

import json
from pathlib import Path


def test_working_five_class_summary_totals_to_population():
    summary = json.loads(
        Path("reports/v2_working_five_class_summary.json").read_text(encoding="utf-8")
    )
    assert sum(summary["classification_counts"].values()) == summary["working_canonical_nights"]
    assert summary["working_canonical_nights"] == 87
    assert summary["classification_counts"]["ACTUAL_GPX"] == 12
    assert summary["classification_counts"]["RECONSTRUCTED_GPS_HIGH"] == 21
    assert summary["final"] is False
