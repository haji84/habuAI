from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatIntentKind(StrEnum):
    NAVIGATE = "navigate"
    EXPLAIN = "explain"
    REQUEST_RECORD = "request_record"
    REQUEST_ROUTE_CHANGE = "request_route_change"
    REQUEST_SESSION_ACTION = "request_session_action"
    NONE = "none"


class AppTarget(StrEnum):
    TONIGHT = "tonight"
    MAP = "map"
    RECORD = "record"
    DASHBOARD = "dashboard"
    SETTINGS = "settings"
    CHAT = "chat"


class AuthorizationLevel(StrEnum):
    READ_ONLY = "read_only"
    CONFIRM_REQUIRED = "confirm_required"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    message_id: str
    thread_id: str
    role: ChatRole
    text: str
    created_at: datetime
    context_snapshot_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChatContextSnapshot:
    """Structured, auditable context supplied to the Habu specialist.

    The model must answer operational-state questions from this object rather than
    inventing current values. `forecast` is expected to be the frozen pre-search
    ForecastSnapshot projection when a search is active.
    """

    snapshot_id: str
    created_at: datetime
    exploration_night: str | None = None
    forecast: Mapping[str, Any] = field(default_factory=dict)
    route: Mapping[str, Any] = field(default_factory=dict)
    session: Mapping[str, Any] = field(default_factory=dict)
    dashboard: Mapping[str, Any] = field(default_factory=dict)
    sync: Mapping[str, Any] = field(default_factory=dict)
    model: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    is_stale: bool = False
    is_online: bool = True


@dataclass(frozen=True, slots=True)
class ChatActionIntent:
    intent_id: str
    kind: ChatIntentKind
    authorization: AuthorizationLevel
    target: AppTarget | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChatAuditEvent:
    audit_id: str
    created_at: datetime
    thread_id: str
    message_id: str
    context_snapshot_id: str | None
    action: ChatActionIntent | None
    provider: str
    model_name: str | None
    used_online_genai: bool


@dataclass(frozen=True, slots=True)
class ChatResponseEnvelope:
    text: str
    context_snapshot_id: str
    action: ChatActionIntent | None = None
    cached_fallback: bool = False
    warnings: tuple[str, ...] = ()


def authorize_intent(intent: ChatActionIntent) -> AuthorizationLevel:
    """Hard gate for v1 chat actions.

    Chat may navigate or explain immediately. Any state-changing request must be
    explicitly confirmed by the app. Direct historical-record mutation is not a
    chat capability and remains blocked behind dedicated domain workflows.
    """

    if intent.kind in {ChatIntentKind.NAVIGATE, ChatIntentKind.EXPLAIN, ChatIntentKind.NONE}:
        return AuthorizationLevel.READ_ONLY
    if intent.kind in {
        ChatIntentKind.REQUEST_RECORD,
        ChatIntentKind.REQUEST_ROUTE_CHANGE,
        ChatIntentKind.REQUEST_SESSION_ACTION,
    }:
        return AuthorizationLevel.CONFIRM_REQUIRED
    return AuthorizationLevel.BLOCKED


def validate_context(snapshot: ChatContextSnapshot) -> tuple[str, ...]:
    warnings: list[str] = []
    if snapshot.is_stale:
        warnings.append("context_stale")
    if not snapshot.is_online:
        warnings.append("offline_no_live_genai")

    forecast = snapshot.forecast
    if forecast:
        point_prediction = forecast.get("point_prediction")
        if isinstance(point_prediction, bool) or (
            point_prediction is not None and not isinstance(point_prediction, int)
        ):
            warnings.append("invalid_forecast_point_prediction")

    return tuple(warnings)


def offline_fallback_response(
    snapshot: ChatContextSnapshot,
    prompt: str,
) -> ChatResponseEnvelope:
    """Deterministic offline response for a small set of operational questions.

    This is intentionally not a pretend LLM. It only reflects values already in
    the structured local snapshot.
    """

    warnings = list(validate_context(snapshot))
    normalized = prompt.strip()

    if "今日の予想" in normalized or "今夜の予想" in normalized:
        forecast = snapshot.forecast
        if not forecast:
            text = "保存済みの今夜予想はありません。"
        else:
            count = forecast.get("point_prediction")
            primary = forecast.get("primary_window")
            secondary = forecast.get("secondary_window")
            text = f"今夜の予想は{count}匹。主時間帯は{primary}"
            if secondary:
                text += f"、副時間帯は{secondary}"
            text += "です。"
    elif "今季" in normalized and "何匹" in normalized:
        total = snapshot.dashboard.get("season_capture_count")
        text = (
            f"今季の捕獲数は{total}匹です。"
            if total is not None
            else "今季捕獲数の保存済み集計がありません。"
        )
    elif "今どこ" in normalized or "次" in normalized and "回" in normalized:
        next_road = snapshot.route.get("next_road")
        text = (
            f"現在の保存済みルートでは、次は{next_road}です。"
            if next_road
            else "次に回る道路の保存済み情報がありません。"
        )
    else:
        text = "現在はオフラインです。保存済みの予想・ルート・実績は確認できます。"

    return ChatResponseEnvelope(
        text=text,
        context_snapshot_id=snapshot.snapshot_id,
        cached_fallback=True,
        warnings=tuple(warnings),
    )
