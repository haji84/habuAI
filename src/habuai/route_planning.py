from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


RouteKind = Literal["A_CAPTURE_MAX", "B_EFFICIENCY", "C_ALTERNATIVE"]


@dataclass(frozen=True)
class RouteConstraints:
    forest_road_allowed: bool = True
    max_distance_km: float | None = None
    max_duration_min: float | None = None
    max_risk_score: float | None = None
    max_pairwise_overlap: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_pairwise_overlap <= 1.0:
            raise ValueError("max_pairwise_overlap must be between 0 and 1")


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    segment_ids: tuple[str, ...]
    expected_captures: float
    distance_km: float
    duration_min: float
    risk_score: float = 0.0
    large_habu_score: float = 0.0
    novelty_score: float = 0.0
    contains_forest_road: bool = False
    start_time: str | None = None
    end_time: str | None = None
    areas: tuple[str, ...] = ()
    roads: tuple[str, ...] = ()
    turnaround_label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        if self.duration_min <= 0:
            return 0.0
        return self.expected_captures / self.duration_min


@dataclass(frozen=True)
class SelectedRoute:
    kind: RouteKind
    candidate: RouteCandidate
    explanation: str


def segment_overlap(a: RouteCandidate, b: RouteCandidate) -> float:
    """Return symmetric overlap using Jaccard similarity on 10 m segment IDs."""
    a_ids = set(a.segment_ids)
    b_ids = set(b.segment_ids)
    if not a_ids and not b_ids:
        return 1.0
    union = a_ids | b_ids
    if not union:
        return 0.0
    return len(a_ids & b_ids) / len(union)


def _allowed(candidate: RouteCandidate, constraints: RouteConstraints) -> bool:
    if not constraints.forest_road_allowed and candidate.contains_forest_road:
        return False
    if constraints.max_distance_km is not None and candidate.distance_km > constraints.max_distance_km:
        return False
    if constraints.max_duration_min is not None and candidate.duration_min > constraints.max_duration_min:
        return False
    if constraints.max_risk_score is not None and candidate.risk_score > constraints.max_risk_score:
        return False
    return True


def _passes_overlap(
    candidate: RouteCandidate,
    selected: list[SelectedRoute],
    max_overlap: float,
) -> bool:
    return all(segment_overlap(candidate, item.candidate) <= max_overlap for item in selected)


def _pick_first(
    ranked: Iterable[RouteCandidate],
    selected: list[SelectedRoute],
    constraints: RouteConstraints,
) -> RouteCandidate | None:
    selected_ids = {item.candidate.route_id for item in selected}
    for candidate in ranked:
        if candidate.route_id in selected_ids:
            continue
        if _passes_overlap(candidate, selected, constraints.max_pairwise_overlap):
            return candidate
    return None


def select_top3_routes(
    candidates: Iterable[RouteCandidate],
    constraints: RouteConstraints | None = None,
) -> list[SelectedRoute]:
    """Select three deliberately different routes from optimizer-produced candidates.

    The upstream Route Optimizer is responsible for producing road-valid candidate paths.
    This function applies field constraints first, then chooses:
      A: maximum expected captures
      B: maximum capture efficiency per minute
      C: alternative strategy using large-habu and novelty value

    The overlap rule prevents the three choices from becoming cosmetic variants of one route.
    """
    constraints = constraints or RouteConstraints()
    allowed = [candidate for candidate in candidates if _allowed(candidate, constraints)]
    if not allowed:
        return []

    selected: list[SelectedRoute] = []

    ranked_a = sorted(
        allowed,
        key=lambda c: (c.expected_captures, -c.risk_score, -c.duration_min),
        reverse=True,
    )
    a = ranked_a[0]
    selected.append(
        SelectedRoute(
            kind="A_CAPTURE_MAX",
            candidate=a,
            explanation="捕獲期待値を最優先した本命ルート",
        )
    )

    ranked_b = sorted(
        allowed,
        key=lambda c: (c.efficiency, c.expected_captures, -c.distance_km),
        reverse=True,
    )
    b = _pick_first(ranked_b, selected, constraints)
    if b is not None:
        selected.append(
            SelectedRoute(
                kind="B_EFFICIENCY",
                candidate=b,
                explanation="探索時間あたりの捕獲期待値を優先した効率ルート",
            )
        )

    ranked_c = sorted(
        allowed,
        key=lambda c: (
            0.45 * c.large_habu_score
            + 0.35 * c.novelty_score
            + 0.20 * c.expected_captures,
            -c.risk_score,
        ),
        reverse=True,
    )
    c = _pick_first(ranked_c, selected, constraints)
    if c is not None:
        selected.append(
            SelectedRoute(
                kind="C_ALTERNATIVE",
                candidate=c,
                explanation="大型狙いと未探索高期待区間を混ぜた別戦略ルート",
            )
        )

    # If strict diversity leaves fewer than three routes, keep constraints intact and
    # relax only the diversity rule. This is surfaced in the explanation so the UI can warn.
    if len(selected) < 3:
        selected_ids = {item.candidate.route_id for item in selected}
        fallback = sorted(
            allowed,
            key=lambda c: (c.expected_captures, c.efficiency, -c.risk_score),
            reverse=True,
        )
        for candidate in fallback:
            if candidate.route_id in selected_ids:
                continue
            selected.append(
                SelectedRoute(
                    kind="C_ALTERNATIVE" if len(selected) == 2 else "B_EFFICIENCY",
                    candidate=candidate,
                    explanation="重複率条件を満たす候補不足のため、次善候補を表示",
                )
            )
            selected_ids.add(candidate.route_id)
            if len(selected) == 3:
                break

    return selected[:3]
