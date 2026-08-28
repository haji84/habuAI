from habuai.pipeline import _angle_diff, _count_from_text


def test_angle_diff_wraps():
    assert _angle_diff(359, 1) == 2


def test_count_parser():
    assert _count_from_text("2匹") == 2
    assert _count_from_text("ハブ捕獲大") == 1
