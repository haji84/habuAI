import pandas as pd

from habuai.hardening.events import dedupe_events, species_from_text, strict_holdout_score


def test_himehabu_not_contaminated_as_habu():
    assert species_from_text("ヒメハブ捕獲") == "ヒメハブ"
    assert species_from_text("ハブ捕獲大") == "ハブ"


def test_dedupe_preserves_multi_individual_count():
    row = {
        "timestamp": pd.Timestamp("2026-08-20T01:44:00+09:00"),
        "species": "ハブ",
        "event_type": "捕獲",
        "lat": 28.1,
        "lon": 129.3,
        "individual_count": 2,
        "raw_text": "ハブ捕獲小2匹",
    }
    out, removed = dedupe_events(pd.DataFrame([row, row]))
    assert len(out) == 1
    assert removed == 1
    assert int(out.iloc[0].individual_count) == 2


def test_strict_holdout_windows_do_not_move():
    rows = []
    for ts in ["2026-08-28T22:06:00+09:00", "2026-08-28T22:45:00+09:00", "2026-08-29T01:22:00+09:00", "2026-08-29T02:26:00+09:00"]:
        rows.append({"timestamp": pd.Timestamp(ts), "species": "ハブ", "event_type": "捕獲", "individual_count": 1})
    score = strict_holdout_score(pd.DataFrame(rows))
    assert score["main_window"]["actual_capture_events"] == 0
    assert score["secondary_window"]["actual_capture_events"] == 1
    assert score["actual_individuals"] == 4
    assert score["range_hit"] is True
