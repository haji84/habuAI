from __future__ import annotations

from pathlib import Path

import pandas as pd

from habuai.evidence_policy import load_population_conflicts, quarantine_unresolved_nights


def _conflicts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "operational_date_0700": [
                "2025-11-07",
                "2025-11-18",
                "2026-08-21",
            ],
            "status": [
                "SOURCE_CONFLICT",
                "LIKELY_DATE_DUPLICATE",
                "SOURCE_CONFLICT",
            ],
            "include_by_default": [False, False, True],
            "reason": [
                "version conflict",
                "strong date-duplicate evidence",
                "independent GPX resolved",
            ],
            "source_evidence": ["db", "db", "actual gpx"],
        }
    )


def test_unresolved_conflicts_are_removed_from_proposed_population():
    kept, quarantine = quarantine_unresolved_nights(
        ["2025-11-07", "2025-11-18", "2026-08-21", "2026-08-31"],
        _conflicts(),
    )
    assert kept == ["2026-08-21", "2026-08-31"]
    assert set(quarantine["operational_date_0700"]) == {"2025-11-07", "2025-11-18"}


def test_reviewed_conflict_can_be_promoted_after_independent_evidence():
    kept, quarantine = quarantine_unresolved_nights(["2026-08-21"], _conflicts())
    assert kept == ["2026-08-21"]
    assert quarantine.empty


def test_every_configured_default_exclusion_is_actually_quarantined():
    conflicts = load_population_conflicts(Path("config/v2_population_conflicts.csv"))
    excluded = conflicts.loc[
        ~conflicts["include_by_default"], "operational_date_0700"
    ].astype(str).tolist()
    kept, quarantine = quarantine_unresolved_nights(excluded, conflicts)
    assert kept == []
    assert set(quarantine["operational_date_0700"].astype(str)) == set(excluded)
