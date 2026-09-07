import pytest

from habuai.route_map_generator import route_geometry_quality, write_route_geojson
from habuai.route_planning import RouteCandidate, SelectedRoute


def selected(segment_ids: tuple[str, ...]) -> SelectedRoute:
    candidate = RouteCandidate(
        route_id="route-a",
        segment_ids=segment_ids,
        expected_captures=2.0,
        distance_km=10.0,
        duration_min=60.0,
    )
    return SelectedRoute(
        kind="A_CAPTURE_MAX",
        candidate=candidate,
        explanation="test",
    )


def line_feature(segment_id: str) -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[129.30, 28.10], [129.3001, 28.1001]],
        },
        "properties": {"segment_id": segment_id},
    }


def test_geometry_quality_requires_every_planned_segment() -> None:
    route = selected(("1", "2"))
    audit = route_geometry_quality(route, {"1": line_feature("1")})
    assert audit["coverage"] == 0.5
    assert audit["missing_segment_ids"] == ["2"]


def test_geojson_fails_closed_when_segment_geometry_missing(tmp_path) -> None:
    route = selected(("1", "2"))
    with pytest.raises(ValueError, match="GIS quality gate failed"):
        write_route_geojson(
            [route],
            {"1": line_feature("1")},
            tmp_path / "route.geojson",
        )


def test_geojson_preserves_route_order_and_exact_source(tmp_path) -> None:
    route = selected(("1", "2"))
    path = write_route_geojson(
        [route],
        {"1": line_feature("1"), "2": line_feature("2")},
        tmp_path / "route.geojson",
    )
    text = path.read_text(encoding="utf-8")
    assert '"route_order": 1' in text
    assert '"route_order": 2' in text
    assert '"geometry_source": "canonical_10m_gis_segment"' in text
