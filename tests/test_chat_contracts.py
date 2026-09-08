from datetime import datetime, timezone

from habuai.chat_contracts import (
    AppTarget,
    AuthorizationLevel,
    ChatActionIntent,
    ChatContextSnapshot,
    ChatIntentKind,
    authorize_intent,
    offline_fallback_response,
    validate_context,
)


def _snapshot(**overrides):
    base = dict(
        snapshot_id="ctx-1",
        created_at=datetime(2026, 9, 9, 1, 0, tzinfo=timezone.utc),
        exploration_night="2026-09-08",
        forecast={
            "point_prediction": 2,
            "primary_window": "22:20-23:30",
            "secondary_window": "00:20-01:20",
        },
        route={"next_road": "県道79号・候補区間"},
        dashboard={"season_capture_count": 208},
        is_online=False,
    )
    base.update(overrides)
    return ChatContextSnapshot(**base)


def test_read_only_navigation_needs_no_confirmation():
    intent = ChatActionIntent(
        intent_id="i1",
        kind=ChatIntentKind.NAVIGATE,
        authorization=AuthorizationLevel.READ_ONLY,
        target=AppTarget.MAP,
    )
    assert authorize_intent(intent) == AuthorizationLevel.READ_ONLY


def test_state_changing_chat_actions_require_confirmation():
    intent = ChatActionIntent(
        intent_id="i2",
        kind=ChatIntentKind.REQUEST_RECORD,
        authorization=AuthorizationLevel.CONFIRM_REQUIRED,
        target=AppTarget.RECORD,
    )
    assert authorize_intent(intent) == AuthorizationLevel.CONFIRM_REQUIRED


def test_offline_today_forecast_uses_exact_snapshot_values():
    response = offline_fallback_response(_snapshot(), "今日の予想は？")
    assert response.cached_fallback is True
    assert "2匹" in response.text
    assert "22:20-23:30" in response.text
    assert "00:20-01:20" in response.text
    assert response.context_snapshot_id == "ctx-1"


def test_offline_season_count_uses_dashboard_snapshot():
    response = offline_fallback_response(_snapshot(), "今季何匹？")
    assert response.text == "今季の捕獲数は208匹です。"


def test_offline_next_road_uses_route_snapshot():
    response = offline_fallback_response(_snapshot(), "今どこを回る？")
    assert "県道79号・候補区間" in response.text


def test_stale_and_offline_context_are_explicit():
    warnings = validate_context(_snapshot(is_stale=True))
    assert "context_stale" in warnings
    assert "offline_no_live_genai" in warnings


def test_non_integer_point_prediction_is_flagged():
    warnings = validate_context(
        _snapshot(forecast={"point_prediction": 2.5, "primary_window": "22:00-23:00"})
    )
    assert "invalid_forecast_point_prediction" in warnings


def test_bool_is_not_accepted_as_integer_prediction():
    warnings = validate_context(
        _snapshot(forecast={"point_prediction": True, "primary_window": "22:00-23:00"})
    )
    assert "invalid_forecast_point_prediction" in warnings
