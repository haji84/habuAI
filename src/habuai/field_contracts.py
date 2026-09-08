from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1
NIGHT_CUTOFF = time(7, 0)


class EventKind(StrEnum):
    CAPTURE = "capture"
    SIGHTING = "sighting"
    ROAD_EVENT = "road_event"


class HybridPredictionMode(StrEnum):
    ML_ONLY = "ML_ONLY"
    ML_PLUS_GENAI = "ML_PLUS_GENAI"


class GenAIInfluence(StrEnum):
    EXPLANATION_ONLY = "explanation_only"
    PREDICTION_INFLUENCE = "prediction_influence"


def resolve_night_date(observed_at: datetime) -> date:
    """Map timestamps before 07:00 to the previous exploration night."""
    if observed_at.timetz().replace(tzinfo=None) < NIGHT_CUTOFF:
        return observed_at.date() - timedelta(days=1)
    return observed_at.date()


@dataclass(frozen=True, slots=True)
class FieldOptionDefinition:
    option_id: str
    category: str
    label: str
    sort_order: int = 0
    active: bool = True
    favorite: bool = False
    built_in: bool = False
    model_mapping: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.option_id.strip():
            raise ValueError("option_id must be a stable non-empty identifier")
        if not self.category.strip():
            raise ValueError("category must be non-empty")
        if not self.label.strip():
            raise ValueError("label must be non-empty")


@dataclass(frozen=True, slots=True)
class Capture:
    event_id: str
    observed_at: datetime
    species: str
    latitude: float
    longitude: float
    length_cm: float | None = None
    sex_option_id: str | None = None
    location_option_id: str | None = None
    behavior_option_id: str | None = None
    surface_option_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.species != "ハブ":
            raise ValueError("Capture domain accepts only ハブ")

    @property
    def night_date(self) -> date:
        return resolve_night_date(self.observed_at)


@dataclass(frozen=True, slots=True)
class Sighting:
    event_id: str
    observed_at: datetime
    species: str
    latitude: float
    longitude: float
    count: int = 1
    location_option_id: str | None = None
    behavior_option_id: str | None = None
    surface_option_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.species.strip():
            raise ValueError("species must be non-empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")

    @property
    def night_date(self) -> date:
        return resolve_night_date(self.observed_at)


@dataclass(frozen=True, slots=True)
class RoadEvent:
    event_id: str
    discovered_at: datetime
    species: str
    latitude: float
    longitude: float
    count: int = 1
    state_option_id: str | None = None
    freshness_option_id: str | None = None
    actual_presence_at: datetime | None = None
    presence_time_source: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.species.strip():
            raise ValueError("species must be non-empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.actual_presence_at is not None and not self.presence_time_source:
            raise ValueError("presence_time_source is required when actual_presence_at is set")

    @property
    def night_date(self) -> date:
        return resolve_night_date(self.discovered_at)

    @property
    def activity_time_for_training(self) -> datetime | None:
        """Never substitute discovered_at for unknown actual presence time."""
        return self.actual_presence_at


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    forecast_id: str
    exploration_night: date
    created_at: datetime
    point_prediction: int
    primary_window: str
    secondary_window: str | None = None
    prediction_mode: HybridPredictionMode = HybridPredictionMode.ML_ONLY
    genai_influence: GenAIInfluence = GenAIInfluence.EXPLANATION_ONLY
    genai_evidence: tuple[str, ...] = ()
    frozen: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.point_prediction) is not int or self.point_prediction < 0:
            raise ValueError("point_prediction must be one non-negative integer")
        if (
            self.prediction_mode == HybridPredictionMode.ML_ONLY
            and self.genai_influence == GenAIInfluence.PREDICTION_INFLUENCE
        ):
            raise ValueError("ML_ONLY cannot declare GenAI prediction influence")


@dataclass(slots=True)
class ExplorationSession:
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    forecast_snapshot_id: str | None = None
    selected_route_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def night_date(self) -> date:
        return resolve_night_date(self.started_at)

    def end(self, ended_at: datetime) -> None:
        if ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        self.ended_at = ended_at
