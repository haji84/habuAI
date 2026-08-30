import pandas as pd

from habuai.hardening.environmental import _parse_jma_tide_text,add_lunar_features


def test_jma_o9_fixed_width_tide_parser():
    hourly="".join(f"{v:3d}" for v in range(24))
    line=hourly+"260829"+"O9"
    df=_parse_jma_tide_text(line,2026)
    assert len(df)==24
    assert df.iloc[0].timestamp==pd.Timestamp("2026-08-29T00:00:00+09:00")
    assert df.iloc[0].tide_height_cm==0
    assert df.iloc[-1].tide_height_cm==23
    assert df.iloc[1].tide_state_code==1


def test_lunar_features_are_bounded():
    data=pd.DataFrame({"entered_at":[pd.Timestamp("2026-08-29T22:00:00+09:00")]})
    out=add_lunar_features(data)
    assert 0<=out.iloc[0].moon_age_days<29.53058867
    assert 0<=out.iloc[0].moon_illumination<=1
