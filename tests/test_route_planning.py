from habuai.route_planning import RouteCandidate, RouteConstraints, segment_overlap, select_top3_routes


def route(
    route_id: str,
    segments: list[str],
    captures: float,
    minutes: float,
    *,
    distance: float = 20.0,
    large: float = 0.0,
    novelty: float = 0.0,
    forest: bool = False,
) -> RouteCandidate:
    return RouteCandidate(
        route_id=route_id,
        segment_ids=tuple(segments),
        expected_captures=captures,
        distance_km=distance,
        duration_min=minutes,
        large_habu_score=large,
        novelty_score=novelty,
        contains_forest_road=forest,
    )


def test_segment_overlap_uses_jaccard() -> None:
    a = route("a", ["1", "2", "3"], 1, 60)
    b = route("b", ["2", "3", "4"], 1, 60)
    assert segment_overlap(a, b) == 0.5


def test_selects_capture_efficiency_and_alternative_routes() -> None:
    candidates = [
        route("A", ["1", "2", "3"], 2.4, 180, large=0.2, novelty=0.1),
        route("B", ["4", "5", "6"], 1.8, 90, large=0.1, novelty=0.2),
        route("C", ["7", "8", "9"], 1.5, 150, large=1.0, novelty=0.9),
        route("D", ["1", "2", "10"], 2.2, 160, large=0.5, novelty=0.4),
    ]

    selected = select_top3_routes(
        candidates,
        RouteConstraints(max_pairwise_overlap=0.70),
    )

    assert [item.candidate.route_id for item in selected] == ["A", "B", "C"]
    assert [item.kind for item in selected] == [
        "A_CAPTURE_MAX",
        "B_EFFICIENCY",
        "C_ALTERNATIVE",
    ]


def test_forest_road_constraint_applies_to_all_routes() -> None:
    candidates = [
        route("forest-best", ["1"], 9.0, 30, forest=True),
        route("A", ["2"], 2.0, 90),
        route("B", ["3"], 1.5, 60),
        route("C", ["4"], 1.2, 70, large=1.0, novelty=1.0),
    ]
    selected = select_top3_routes(
        candidates,
        RouteConstraints(forest_road_allowed=False),
    )
    assert all(not item.candidate.contains_forest_road for item in selected)
    assert "forest-best" not in {item.candidate.route_id for item in selected}


def test_strict_overlap_can_fallback_without_breaking_safety_constraints() -> None:
    candidates = [
        route("A", ["1", "2", "3"], 3.0, 120),
        route("B", ["1", "2", "4"], 2.5, 100),
        route("C", ["1", "2", "5"], 2.0, 90),
    ]
    selected = select_top3_routes(
        candidates,
        RouteConstraints(max_pairwise_overlap=0.10),
    )
    assert len(selected) == 3
    assert any("候補不足" in item.explanation for item in selected[1:])
