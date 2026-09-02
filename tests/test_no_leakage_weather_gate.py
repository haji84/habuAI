from __future__ import annotations

from habuai.runtime_fixes import mark_hindsight_weather_evaluation


def test_archive_weather_score_is_diagnostic_not_strict():
    score = mark_hindsight_weather_evaluation({"status": "ok", "brier": 0.12})
    assert score["status"] == "ok"
    assert score["brier"] == 0.12
    assert score["evaluation_mode"] == "DIAGNOSTIC_HINDSIGHT_WEATHER"
    assert score["strict_no_leakage_eligible"] is False
    assert score["weather_feature_source"] == "open_meteo_archive_actual_or_reanalysis"
    assert "frozen forecast snapshot" in score["strict_blocker"]
