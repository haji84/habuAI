from pathlib import Path

from habuai.hardening.canonical import load_canonical_capture_master


def test_canonical_capture_master_totals():
    root=Path(__file__).resolve().parents[1]
    df=load_canonical_capture_master(root)
    assert len(df)==206
    assert int(df.individual_count.sum())==208
    gps=df.lat.notna()&df.lon.notna()
    assert int(gps.sum())==150
    assert int(df.loc[gps,"individual_count"].sum())==151
    assert int((~gps).sum())==56
    assert set(df.event_type.unique())=={"捕獲"}


def test_night_rollover_is_0700():
    root=Path(__file__).resolve().parents[1]
    df=load_canonical_capture_master(root)
    r=df[df.canonical_id=="11"].iloc[0]
    assert str(r.night_date)=="2025-10-12"
