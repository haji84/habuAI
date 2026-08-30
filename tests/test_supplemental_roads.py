import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from habuai.hardening.roads import _stable_road_id, recover_supplemental_roads


def _osm_segment():
    g=gpd.GeoDataFrame([{"segment_id":"OSM_A","way_id":"A","geometry":LineString([(129.3500,28.1500),(129.3502,28.1500)])}],crs="EPSG:4326")
    g=g.to_crs("EPSG:6669")
    g["part_index"]=0;g["segment_index"]=0;g["length_m"]=g.geometry.length;g["highway"]="residential";g["name"]="mapped"
    return g


def _gpx_run(spike=False):
    rows=[]
    t=pd.Timestamp("2026-08-30T21:00:00+09:00")
    for i in range(25):
        lon=129.35015 + i*0.000035
        lat=28.15000
        if spike and i==12:
            lon+=0.003
        rows.append({"session_file":"test.gpx","seq":i,"timestamp":t+pd.Timedelta(seconds=i*3),"lat":lat,"lon":lon,"segment_id":None})
    return pd.DataFrame(rows)


def test_unmatched_gpx_connected_to_osm_becomes_supplemental_road():
    osm=_osm_segment();m=_gpx_run();cfg={"segment_length_m":10.0,"supplemental_roads":{"enabled":True,"min_unmatched_points":20,"min_length_m":30.0,"max_point_gap_m":45.0,"endpoint_connect_m":50.0,"duplicate_hausdorff_m":20.0,"event_corroboration_m":25.0}}
    supplemental,audit=recover_supplemental_roads(m,m,pd.DataFrame(),osm,cfg)
    assert not supplemental.empty
    assert len(audit)==1
    assert audit[0]["accepted"] is True
    assert audit[0]["connected_to_osm"] is True
    assert set(supplemental.road_source)=={"gpx_recovered"}


def test_gpx_spike_is_rejected_not_promoted_to_road():
    osm=_osm_segment();m=_gpx_run(spike=True);cfg={"segment_length_m":10.0,"supplemental_roads":{"enabled":True,"min_unmatched_points":20,"min_length_m":30.0,"max_point_gap_m":45.0,"endpoint_connect_m":50.0}}
    supplemental,audit=recover_supplemental_roads(m,m,pd.DataFrame(),osm,cfg)
    assert supplemental.empty


def test_stable_road_id_is_direction_independent():
    line=LineString([(100.0,200.0),(120.0,205.0),(150.0,210.0)])
    reverse=LineString(list(reversed(line.coords)))
    assert _stable_road_id(line)==_stable_road_id(reverse)
