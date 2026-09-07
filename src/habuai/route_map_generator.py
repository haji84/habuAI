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
    routes = []
    for item in selected_routes:
        candidate = item.candidate
        routes.append({
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
        })
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
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _valid_linestring_geometry(feature: dict[str, object] | None) -> bool:
    if not feature:
        return False
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return False
    if geometry.get("type") not in {"LineString", "MultiLineString"}:
        return False
    coordinates = geometry.get("coordinates")
    return isinstance(coordinates, list) and len(coordinates) > 0


def route_geometry_quality(
    route: SelectedRoute,
    segment_features: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Audit whether every planned 10 m segment has drawable GIS road geometry."""
    requested = list(route.candidate.segment_ids)
    missing = [sid for sid in requested if sid not in segment_features]
    invalid = [
        sid for sid in requested
        if sid in segment_features and not _valid_linestring_geometry(segment_features[sid])
    ]
    valid_count = len(requested) - len(missing) - len(invalid)
    coverage = valid_count / len(requested) if requested else 0.0
    return {
        "route_id": route.candidate.route_id,
        "requested_segments": len(requested),
        "valid_segments": valid_count,
        "coverage": coverage,
        "missing_segment_ids": missing,
        "invalid_geometry_segment_ids": invalid,
    }


def write_route_geojson(
    selected_routes: Iterable[SelectedRoute],
    segment_features: dict[str, dict[str, object]],
    output_path: str | Path,
    *,
    min_geometry_coverage: float = 1.0,
) -> Path:
    """Render only exact existing GIS road geometry and reject low-quality maps.

    The default requires 100% planned-segment geometry coverage. This intentionally fails
    closed: an incomplete route is better reported as unusable than rendered as a misleading
    map. Segment order is preserved through route_order.
    """
    routes = list(selected_routes)
    audits = [route_geometry_quality(route, segment_features) for route in routes]
    bad = [audit for audit in audits if float(audit["coverage"]) < min_geometry_coverage]
    if bad:
        detail = "; ".join(
            f"{a['route_id']}: coverage={float(a['coverage']):.3f}, "
            f"missing={len(a['missing_segment_ids'])}, invalid={len(a['invalid_geometry_segment_ids'])}"
            for a in bad
        )
        raise ValueError(f"Route map GIS quality gate failed: {detail}")

    features: list[dict[str, object]] = []
    for route, audit in zip(routes, audits, strict=True):
        for order, segment_id in enumerate(route.candidate.segment_ids, start=1):
            source = segment_features[segment_id]
            features.append({
                "type": "Feature",
                "geometry": source["geometry"],
                "properties": {
                    **dict(source.get("properties") or {}),
                    "segment_id": segment_id,
                    "route_kind": route.kind,
                    "route_label": _ROUTE_LABELS[route.kind],
                    "route_id": route.candidate.route_id,
                    "route_order": order,
                    "expected_captures": route.candidate.expected_captures,
                    "risk_score": route.candidate.risk_score,
                    "geometry_coverage": audit["coverage"],
                    "geometry_source": "canonical_10m_gis_segment",
                },
            })

    collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "geometry_policy": "exact_10m_gis_only",
            "min_geometry_coverage": min_geometry_coverage,
            "route_geometry_audit": audits,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def route_summary(selected: SelectedRoute) -> dict[str, object]:
    payload = asdict(selected.candidate)
    payload.update({
        "kind": selected.kind,
        "label": _ROUTE_LABELS[selected.kind],
        "explanation": selected.explanation,
        "efficiency_per_hour": selected.candidate.efficiency * 60.0,
    })
    return payload
