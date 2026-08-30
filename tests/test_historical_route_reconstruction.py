from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from habuai.hardening.reconstruct import reconstruct_historical_routes


def _dirs(tmp_path:Path):
    (tmp_path/"data"/"processed").mkdir(parents=True);(tmp_path/"reports").mkdir()


def test_reconstruction_prefers_contiguous_multi_gpx_supported_route(tmp_path:Path):
    _dirs(tmp_path);t=pd.Timestamp("2026-06-10T22:00:00+09:00")
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":t,"segment_id":"C1"},{"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(hours=1),"segment_id":"C2"}])
    rows=[]
    for session,seq in {"2026-08-13.gpx":["X","A","C1","B","C2","D","Y"],"2026-08-15.gpx":["Q","A","C1","B","C2","D","R"],"2026-08-17.gpx":["Z","C1","K","Z2"]}.items():
        for i,seg in enumerate(seq):rows.append({"session_file":session,"visit_index":i,"segment_id":seg,"entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")+pd.Timedelta(minutes=i)})
    visits=pd.DataFrame(rows);cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"candidate_coverage_delta":0.15,"min_candidate_coverage":0.3,"high_confidence_coverage":1.0,"high_confidence_min_support_sessions":2}}
    out,report=reconstruct_historical_routes(tmp_path,events,visits,cfg)
    assert report["nights_reconstructed"]==1;assert report["nights_coverage_100pct"]==1;assert set(out.segment_id)=={"C1","B","C2"};assert out.reconstruction_confidence.iloc[0]=="B-high";assert not out.historical_time_reconstructed.any()


def test_reconstruction_does_not_copy_future_timestamps(tmp_path:Path):
    _dirs(tmp_path);events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":pd.Timestamp("2026-07-01T23:00:00+09:00"),"segment_id":"C1"}]);visits=pd.DataFrame([{"session_file":"2026-08-13.gpx","visit_index":0,"segment_id":"C1","entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")}]);cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"min_candidate_coverage":0.3}}
    out,_=reconstruct_historical_routes(tmp_path,events,visits,cfg);assert "entered_at" not in out.columns;assert out.iloc[0].template_session_file=="2026-08-13.gpx"


def test_c_low_is_rescued_when_capture_anchors_are_split_across_gpx_sessions(tmp_path:Path):
    _dirs(tmp_path);t=pd.Timestamp("2026-06-20T22:00:00+09:00")
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(minutes=10*i),"segment_id":seg} for i,seg in enumerate(["C1","C2","C3"])])
    seqs={"2026-08-13.gpx":["C1","A","J"],"2026-08-15.gpx":["J","B","C2"],"2026-08-17.gpx":["J","D","C3"]};rows=[]
    for session,seq in seqs.items():
        for i,seg in enumerate(seq):rows.append({"session_file":session,"visit_index":i,"segment_id":seg,"entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")+pd.Timedelta(minutes=i)})
    cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"min_candidate_coverage":0.3,"rescue_min_anchor_coverage":1.0,"rescue_high_median_edge_support":2.0}}
    out,report=reconstruct_historical_routes(tmp_path,events,pd.DataFrame(rows),cfg)
    assert report["rescued_nights"]==1;assert report["remaining_c_low_nights"]==0;assert set(["C1","C2","C3"]).issubset(set(out.segment_id));assert out.reconstruction_source.eq("multi_gpx_union_graph_rescue").all();assert out.reconstruction_confidence.eq("B-medium").all();assert not out.historical_time_reconstructed.any()


def test_c_low_stays_low_without_network_when_anchor_absent_from_all_later_gpx(tmp_path:Path):
    _dirs(tmp_path);t=pd.Timestamp("2026-07-10T22:00:00+09:00")
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":t,"segment_id":"C1"},{"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(minutes=20),"segment_id":"MISSING"},{"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(minutes=40),"segment_id":"C3"}]);rows=[]
    for session,seq in {"2026-08-13.gpx":["C1","A","J"],"2026-08-15.gpx":["J","B","C3"]}.items():
        for i,seg in enumerate(seq):rows.append({"session_file":session,"visit_index":i,"segment_id":seg,"entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")+pd.Timedelta(minutes=i)})
    cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"min_candidate_coverage":0.3,"rescue_min_anchor_coverage":1.0}}
    out,report=reconstruct_historical_routes(tmp_path,events,pd.DataFrame(rows),cfg)
    assert report["rescued_nights"]==0;assert report["remaining_c_low_nights"]==1;assert out.reconstruction_confidence.eq("C-low").all()


def test_road_network_rescue_promotes_anchor_missing_from_later_gpx(tmp_path:Path):
    _dirs(tmp_path);t=pd.Timestamp("2026-07-12T22:00:00+09:00")
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(minutes=20*i),"segment_id":seg} for i,seg in enumerate(["C1","C2","C3"])])
    rows=[]
    for session,seq in {"2026-08-13.gpx":["C1","S1"],"2026-08-15.gpx":["S3","C3"]}.items():
        for i,seg in enumerate(seq):rows.append({"session_file":session,"visit_index":i,"segment_id":seg,"entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")+pd.Timedelta(minutes=i)})
    ids=["C1","S1","C2","S3","C3"];geoms=[LineString([(i*10,0),((i+1)*10,0)]) for i in range(len(ids))];segs=gpd.GeoDataFrame({"segment_id":ids},geometry=geoms,crs="EPSG:6669")
    cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"min_candidate_coverage":0.3,"rescue_min_anchor_coverage":1.0,"road_rescue_max_detour_ratio":1.6,"road_rescue_unseen_penalty":2.5}}
    out,report=reconstruct_historical_routes(tmp_path,events,pd.DataFrame(rows),cfg,segs=segs)
    assert report["road_network_rescued_nights"]==1;assert report["remaining_c_low_nights"]==0;assert set(["C1","C2","C3"]).issubset(set(out.segment_id));assert out.reconstruction_source.eq("road_network_gpx_prior_rescue").all();assert out.reconstruction_confidence.isin(["B-high","B-medium"]).all();assert not out.historical_time_reconstructed.any()
