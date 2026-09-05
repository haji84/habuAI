from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .route_planning import SelectedRoute


_ROUTE_LABELS = {
    "A_CAPTURE_MAX": "A 本命・捕獲期待値最大",
    "B_EFFICIENCY": "B 効率重視",
    "C_ALTERNATIVE": "C 別戦略",
}



def build_route_plan_payload(
    exploration_night: str,
    point_prediction: int,
    primary_window: str,
    secondary_window: str | None,
    selected_routes: Iterable[SelectedRoute],
    weather: dict[str, object] | None = None,
    constraints: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create the mobile-facing contract for the nightly route-plan screen."""
    routes = []
    for item in selected_routes:
        candidate = item.candidate
        routes.append(
            {
                "kind": item.kind,
                "label": _ROUTE_LABELS[item.kind],
                "explanation": item.explanation,
                "route_id": candidate.route_id,
                "expected_captures": candidate.expected_captures,
                "efficiency_per_hour": candidate.efficiency * 60.0,
                "distance_km": candidate.distance_km,
                "duration_min": candidate.duration_min,
                "risk_score": candidate.risk_score,
                "large_habu_score": candidate.large_habu_score,
                "novelty_score": candidate.novelty_score,
                "contains_forest_road": candidate.contains_forest_road,
                "start_time": candidate.start_time,
                "end_time": candidate.end_time,
                "areas": list(candidate.areas),
                "roads": list(candidate.roads),
                "turnaround_label": candidate.turnaround_label,
                "segment_ids": list(candidate.segment_ids),
                "metadata": candidate.metadata,
            }
        )

    return {
        "exploration_night": exploration_night,
        "point_prediction": point_prediction,
        "primary_window": primary_window,
        "secondary_window": secondary_window,
        "weather": weather or {},
        "constraints": constraints or {},
        "routes": routes,
    }



def write_route_plan_json(payload: dict[str, object], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path



def write_route_geojson(
    selected_routes: Iterable[SelectedRoute],
    segment_features: dict[str, dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Write exact GIS route geometry from pre-built 10 m road-segment features.

    `segment_features` must map segment_id to a valid GeoJSON Feature. Geometry is never
    synthesized by AI. Missing segments are skipped and reported in feature properties.
    """
    features: list[dict[str, object]] = []

    for route in selected_routes:
        missing: list[str] = []
        for order, segment_id in enumerate(route.candidate.segment_ids, start=1):
            source = segment_features.get(segment_id)
            if source is None:
                missing.append(segment_id)
                continue
            feature = {
                "type": "Feature",
                "geometry": source.get("geometry"),
                "properties": {
                    **dict(source.get("properties") or {}),
                    "segment_id": segment_id,
                    "route_kind": route.kind,
                    "route_label": _ROUTE_LABELS[route.kind],
                    "route_id": route.candidate.route_id,
                    "route_order": order,
                    "expected_captures": route.candidate.expected_captures,
                    "risk_score": route.candidate.risk_score,
                },
            }
            features.append(feature)

        if missing:
            features.append(
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {
                        "route_kind": route.kind,
                        "route_id": route.candidate.route_id,
                        "warning": "missing_segment_geometry",
                        "missing_segment_ids": missing,
                    },
                }
            )

    collection = {"type": "FeatureCollection", "features": features}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path



def route_summary(selected: SelectedRoute) -> dict[str, object]:
    """Small serializable summary for logging, scoring and post-run comparison."""
    payload = asdict(selected.candidate)
    payload.update(
        {
            "kind": selected.kind,
            "label": _ROUTE_LABELS[selected.kind],
            "explanation": selected.explanation,
            "efficiency_per_hour": selected.candidate.efficiency * 60.0,
        }
    )
    return payload
