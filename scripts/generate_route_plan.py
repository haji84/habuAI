from __future__ import annotations

import argparse
import json
from pathlib import Path

from habuai.route_map_generator import build_route_plan_payload, write_route_geojson, write_route_plan_json
from habuai.route_map_html import write_leaflet_route_map
from habuai.route_planning import RouteCandidate, RouteConstraints, select_top3_routes


def _load_json(path: str | Path) -> dict[str, object] | list[object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_from_dict(row: dict[str, object]) -> RouteCandidate:
    return RouteCandidate(
        route_id=str(row["route_id"]),
        segment_ids=tuple(str(x) for x in row.get("segment_ids", [])),
        expected_captures=float(row.get("expected_captures", 0.0)),
        distance_km=float(row.get("distance_km", 0.0)),
        duration_min=float(row.get("duration_min", 0.0)),
        risk_score=float(row.get("risk_score", 0.0)),
        large_habu_score=float(row.get("large_habu_score", 0.0)),
        novelty_score=float(row.get("novelty_score", 0.0)),
        contains_forest_road=bool(row.get("contains_forest_road", False)),
        start_time=row.get("start_time") if isinstance(row.get("start_time"), str) else None,
        end_time=row.get("end_time") if isinstance(row.get("end_time"), str) else None,
        areas=tuple(str(x) for x in row.get("areas", [])),
        roads=tuple(str(x) for x in row.get("roads", [])),
        turnaround_label=(
            row.get("turnaround_label") if isinstance(row.get("turnaround_label"), str) else None
        ),
        metadata=dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {},
    )


def _segment_feature_index(geojson: dict[str, object]) -> dict[str, dict[str, object]]:
    features = geojson.get("features", [])
    if not isinstance(features, list):
        raise ValueError("road segment GeoJSON must contain a features list")

    index: dict[str, dict[str, object]] = {}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        raw_id = properties.get("segment_id") or properties.get("seg_id") or properties.get("id")
        if raw_id is not None:
            index[str(raw_id)] = feature
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Habu AI top-3 nightly route map")
    parser.add_argument("--candidates", required=True, help="Route Optimizer candidate JSON")
    parser.add_argument("--segments", required=True, help="10 m road segment GeoJSON")
    parser.add_argument("--night", required=True, help="Exploration night YYYY-MM-DD")
    parser.add_argument("--point-prediction", required=True, type=int)
    parser.add_argument("--primary-window", required=True)
    parser.add_argument("--secondary-window")
    parser.add_argument("--output-dir", default="reports/nightly_route")
    parser.add_argument("--forest-road-allowed", action="store_true")
    parser.add_argument("--max-distance-km", type=float)
    parser.add_argument("--max-duration-min", type=float)
    parser.add_argument("--max-risk-score", type=float)
    parser.add_argument("--max-overlap", type=float, default=0.70)
    args = parser.parse_args()

    raw_candidates = _load_json(args.candidates)
    if isinstance(raw_candidates, dict):
        rows = raw_candidates.get("routes", raw_candidates.get("candidates", []))
    else:
        rows = raw_candidates
    if not isinstance(rows, list):
        raise ValueError("candidate JSON must be a list or contain routes/candidates list")

    candidates = [_candidate_from_dict(row) for row in rows if isinstance(row, dict)]
    constraints = RouteConstraints(
        forest_road_allowed=args.forest_road_allowed,
        max_distance_km=args.max_distance_km,
        max_duration_min=args.max_duration_min,
        max_risk_score=args.max_risk_score,
        max_pairwise_overlap=args.max_overlap,
    )
    selected = select_top3_routes(candidates, constraints)
    if not selected:
        raise SystemExit("No route candidates satisfy tonight's constraints")

    segments_raw = _load_json(args.segments)
    if not isinstance(segments_raw, dict):
        raise ValueError("segment input must be GeoJSON object")
    segment_features = _segment_feature_index(segments_raw)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.night

    plan = build_route_plan_payload(
        exploration_night=args.night,
        point_prediction=args.point_prediction,
        primary_window=args.primary_window,
        secondary_window=args.secondary_window,
        selected_routes=selected,
        constraints={
            "forest_road_allowed": args.forest_road_allowed,
            "max_distance_km": args.max_distance_km,
            "max_duration_min": args.max_duration_min,
            "max_risk_score": args.max_risk_score,
            "max_pairwise_overlap": args.max_overlap,
        },
    )
    plan_path = write_route_plan_json(plan, output_dir / f"{stem}_route_plan.json")
    geo_path = write_route_geojson(
        selected,
        segment_features,
        output_dir / f"{stem}_routes.geojson",
    )
    geo_payload = json.loads(geo_path.read_text(encoding="utf-8"))
    html_path = write_leaflet_route_map(
        plan,
        geo_payload,
        output_dir / f"{stem}_route_map.html",
    )

    print(plan_path)
    print(geo_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
