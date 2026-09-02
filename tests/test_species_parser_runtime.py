from __future__ import annotations

from habuai.runtime_fixes import species_from_text_specific_first


def test_himehabu_is_not_swallowed_by_habu_substring():
    assert species_from_text_specific_first("目撃\n1匹\nヒメハブ\nウエット") == "ヒメハブ"


def test_main_habu_stays_main_habu():
    assert species_from_text_specific_first("捕獲\nハブ捕獲大\n道路中央") == "ハブ"


def test_specific_frog_name_wins_over_generic_frog():
    assert species_from_text_specific_first("1匹\nイシカワガエル") == "イシカワガエル"
