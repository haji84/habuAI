from pathlib import Path

import numpy as np
import pandas as pd

from habuai.hardening.canonical import load_canonical_capture_master,load_canonical_habu_events,match_events_preserve_all
from habuai.hardening.events import dedupe_events


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


def test_all_canonical_habu_outcomes_are_present():
    root=Path(__file__).resolve().parents[1]
    df=load_canonical_habu_events(root)
    assert len(df)==247
    assert int((df.event_type=="捕獲").sum())==206
    assert int((df.event_type=="no_capture").sum())==2
    assert int(df.event_type.isin(["轢死","roadkill_sighting"]).sum())==33
    assert int(df.event_type.isin(["目撃","sighting"]).sum())==6


def test_night_rollover_is_0700():
    root=Path(__file__).resolve().parents[1]
    df=load_canonical_capture_master(root)
    r=df[df.canonical_id=="11"].iloc[0]
    assert str(r.night_date)=="2025-10-12"


def test_missing_gps_canonical_events_survive_dedupe():
    root=Path(__file__).resolve().parents[1]
    df=load_canonical_capture_master(root)
    before=df[df.lat.isna()|df.lon.isna()].copy()
    assert len(before)==56
    out,removed=dedupe_events(df)
    after=out[out.lat.isna()|out.lon.isna()]
    assert len(out)==206
    assert len(after)==56
    assert removed==0
    assert after.canonical_id.nunique()==56


def test_dedupe_removes_true_duplicate_but_not_distinct_missing_values():
    rows=pd.DataFrame([
        {"canonical_id":"a","timestamp":pd.NaT,"species":"ハブ","event_type":"捕獲","lat":np.nan,"lon":np.nan,"raw_text":"canonical:a 捕獲"},
        {"canonical_id":"b","timestamp":pd.NaT,"species":"ハブ","event_type":"捕獲","lat":np.nan,"lon":np.nan,"raw_text":"canonical:b 捕獲"},
        {"canonical_id":"a","timestamp":pd.NaT,"species":"ハブ","event_type":"捕獲","lat":np.nan,"lon":np.nan,"raw_text":"canonical:a 捕獲"},
    ])
    out,removed=dedupe_events(rows)
    assert len(out)==2
    assert removed==1
    assert set(out.canonical_id)=={"a","b"}


def test_match_preserves_missing_gps_and_named_source_index():
    class FakePipeline:
        @staticmethod
        def match_events(df,segs):
            out=df.copy()
            out["segment_id"]="OSM_test"
            out["event_match_distance_m"]=1.0
            return out
    events=pd.DataFrame([
        {"canonical_id":"a","lat":28.1,"lon":129.3},
        {"canonical_id":"b","lat":np.nan,"lon":np.nan},
    ])
    out=match_events_preserve_all(FakePipeline(),events,None)
    assert len(out)==2
    assert out.loc[0,"segment_id"]=="OSM_test"
    assert pd.isna(out.loc[1,"segment_id"])
