from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from habuai.evidence_policy import load_population_conflicts


def test_working_population_manifest_is_internally_consistent():
    population = pd.read_csv("config/v2_canonical_population_working.csv", dtype=str)
    summary = json.loads(Path("reports/v2_population_working_summary.json").read_text(encoding="utf-8"))

    assert population["operational_date_0700"].is_unique
    assert len(population) == summary["working_canonical_nights"]

    counts = population["evidence_class"].value_counts().to_dict()
    breakdown = summary["working_source_breakdown"]
    assert counts.get("UNAMBIGUOUS_SELF_CAPTURE", 0) == breakdown["unambiguous_self_capture_nights"]
    assert counts.get("EXPLICIT_ZERO", 0) == breakdown["explicit_zero_nights"]
    assert counts.get("INDEPENDENT_OMITTED_EVIDENCE", 0) == breakdown["independent_omitted_evidence_nights"]


def test_working_population_does_not_include_default_quarantine_dates():
    population = pd.read_csv("config/v2_canonical_population_working.csv", dtype=str)
    conflicts = load_population_conflicts("config/v2_population_conflicts.csv")
    excluded = set(
        conflicts.loc[~conflicts["include_by_default"], "operational_date_0700"].astype(str)
    )
    working = set(population["operational_date_0700"].astype(str))
    assert working.isdisjoint(excluded)
