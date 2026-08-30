from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from habuai.hardening.spatial_history import add_historical_spatial_features


def test_spatial_history_uses_only_pre_night_evidence(tmp_path:Path):
    segs=gpd.GeoDataFrame([
        {"segment_id":"A","geometry":LineString([(-5,0),(5,0)])},
    ],crs="EPSG:3857")
    data=pd.DataFrame([{"night":"2026-05-02","segment_id":"A","spatial_label":0}])
    events=pd.DataFrame([
        {"timestamp":"2026-05-01T20:00:00+09:00","species":"ハブ","event_type":"捕獲","individual_count":1,"size":"大","lat":0.0,"lon":0.0,"raw_text":"大型ハブ"},
        {"timestamp":"2026-05-02T01:00:00+09:00","species":"ネズミ","event_type":"目撃","individual_count":2,"size":None,"lat":0.0,"lon":0.0,"raw_text":"ネズミ2匹"},
        # Same operational night after 07:00 cutoff. This is the outcome and must be invisible.
        {"timestamp":"2026-05-02T20:00:00+09:00","species":"ハブ","event_type":"捕獲","individual_count":9,"size":"大","lat":0.0,"lon":0.0,"raw_text":"未来の捕獲"},
    ])
    out,audit=add_historical_spatial_features(data,events,segs,{},root=tmp_path)
    r=out.iloc[0]
    assert r["hist_capture_count_50m"]==1
    assert r["hist_large_capture_count_50m"]==1
    assert r["hist_bio_30d_count_50m"]==2
    assert r["hist_bio_ネズミ_30d_count_50m"]==2
    assert 0 < r["days_since_capture_50m"] < 1
    assert r["large_female_history_available"]==0
    assert audit["large_female_history_available"] is False


def test_large_does_not_imply_female(tmp_path:Path):
    segs=gpd.GeoDataFrame([{"segment_id":"A","geometry":LineString([(-5,0),(5,0)])}],crs="EPSG:3857")
    data=pd.DataFrame([{"night":"2026-05-02","segment_id":"A","spatial_label":0}])
    events=pd.DataFrame([{"timestamp":"2026-05-01T20:00:00+09:00","species":"ハブ","event_type":"捕獲","individual_count":1,"size":"大","lat":0.0,"lon":0.0,"raw_text":"大"}])
    out,audit=add_historical_spatial_features(data,events,segs,{},root=tmp_path)
    assert out.iloc[0]["hist_large_capture_count_50m"]==1
    assert out.iloc[0]["large_female_history_available"]==0
    assert audit["large_female_history_available"] is False
