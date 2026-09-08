from datetime import datetime

import pytest

from habuai.field_contracts import (
    Capture,
    FieldOptionDefinition,
    ForecastSnapshot,
    GenAIInfluence,
    HybridPredictionMode,
    RoadEvent,
    Sighting,
    resolve_night_date,
)


def test_night_date_before_cutoff_belongs_to_previous_day() -> None:
    assert str(resolve_night_date(datetime(2026, 9, 8, 6, 59, 59))) == "2026-09-07"


def test_night_date_at_cutoff_belongs_to_current_day() -> None:
    assert str(resolve_night_date(datetime(2026, 9, 8, 7, 0, 0))) == "2026-09-08"


def test_capture_accepts_habu_only() -> None:
    capture = Capture("c1", datetime(2026, 9, 8, 1, 0), "ハブ", 28.1, 129.3)
    assert str(capture.night_date) == "2026-09-07"

    with pytest.raises(ValueError, match="only ハブ"):
        Capture("c2", datetime(2026, 9, 8, 1, 0), "ヒメハブ", 28.1, 129.3)


def test_sighting_accepts_himehabu_without_becoming_capture() -> None:
    sighting = Sighting("s1", datetime(2026, 9, 8, 1, 0), "ヒメハブ", 28.1, 129.3)
    assert sighting.species == "ヒメハブ"


def test_road_event_discovery_time_is_not_activity_time() -> None:
    discovered = datetime(2026, 9, 8, 0, 1)
    event = RoadEvent("r1", discovered, "ハブ", 28.1, 129.3)
    assert event.activity_time_for_training is None


def test_road_event_requires_provenance_for_actual_presence_time() -> None:
    with pytest.raises(ValueError, match="presence_time_source"):
        RoadEvent(
            "r1",
            datetime(2026, 9, 8, 0, 1),
            "ハブ",
            28.1,
            129.3,
            actual_presence_at=datetime(2026, 9, 7, 23, 55),
        )


def test_field_option_identity_survives_label_change() -> None:
    before = FieldOptionDefinition("behavior:waiting", "behavior", "待機中", sort_order=2)
    after = FieldOptionDefinition("behavior:waiting", "behavior", "待機", sort_order=1, active=False)
    assert before.option_id == after.option_id
    assert before.label != after.label


def test_point_prediction_is_exactly_one_integer() -> None:
    ForecastSnapshot(
        forecast_id="f1",
        exploration_night=resolve_night_date(datetime(2026, 9, 8, 1, 0)),
        created_at=datetime(2026, 9, 7, 21, 0),
        point_prediction=2,
        primary_window="22:30-23:30",
    )

    with pytest.raises(ValueError, match="integer"):
        ForecastSnapshot(
            forecast_id="f2",
            exploration_night=resolve_night_date(datetime(2026, 9, 8, 1, 0)),
            created_at=datetime(2026, 9, 7, 21, 0),
            point_prediction=2.0,  # type: ignore[arg-type]
            primary_window="22:30-23:30",
        )


def test_ml_only_cannot_claim_genai_prediction_influence() -> None:
    with pytest.raises(ValueError, match="ML_ONLY"):
        ForecastSnapshot(
            forecast_id="f3",
            exploration_night=resolve_night_date(datetime(2026, 9, 8, 1, 0)),
            created_at=datetime(2026, 9, 7, 21, 0),
            point_prediction=2,
            primary_window="22:30-23:30",
            prediction_mode=HybridPredictionMode.ML_ONLY,
            genai_influence=GenAIInfluence.PREDICTION_INFLUENCE,
        )
