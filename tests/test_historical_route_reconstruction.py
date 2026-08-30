from pathlib import Path
import pandas as pd
from habuai.hardening.reconstruct import reconstruct_historical_routes


def test_reconstruction_prefers_contiguous_multi_gpx_supported_route(tmp_path:Path):
    (tmp_path/"data"/"processed").mkdir(parents=True);(tmp_path/"reports").mkdir()
    t=pd.Timestamp("2026-06-10T22:00:00+09:00")
    events=pd.DataFrame([
        {"species":"ハブ","event_type":"捕獲","timestamp":t,"segment_id":"C1"},
        {"species":"ハブ","event_type":"捕獲","timestamp":t+pd.Timedelta(hours=1),"segment_id":"C2"},
    ])
    rows=[]
    for session,seq in {
        "2026-08-13.gpx":["X","A","C1","B","C2","D","Y"],
        "2026-08-15.gpx":["Q","A","C1","B","C2","D","R"],
        "2026-08-17.gpx":["Z","C1","K","Z2"],
    }.items():
        for i,seg in enumerate(seq):
            rows.append({"session_file":session,"visit_index":i,"segment_id":seg,"entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")+pd.Timedelta(minutes=i)})
    visits=pd.DataFrame(rows)
    cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"candidate_coverage_delta":0.15,"min_candidate_coverage":0.3,"high_confidence_coverage":1.0,"high_confidence_min_support_sessions":2}}
    out,report=reconstruct_historical_routes(tmp_path,events,visits,cfg)
    assert report["nights_reconstructed"]==1
    assert report["nights_coverage_100pct"]==1
    assert set(out.segment_id)=={"C1","B","C2"}
    assert out.reconstruction_confidence.iloc[0]=="B-high"
    assert not out.historical_time_reconstructed.any()


def test_reconstruction_does_not_copy_future_timestamps(tmp_path:Path):
    (tmp_path/"data"/"processed").mkdir(parents=True);(tmp_path/"reports").mkdir()
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":pd.Timestamp("2026-07-01T23:00:00+09:00"),"segment_id":"C1"}])
    visits=pd.DataFrame([{"session_file":"2026-08-13.gpx","visit_index":0,"segment_id":"C1","entered_at":pd.Timestamp("2026-08-13T20:00:00+09:00")}])
    cfg={"night_rollover_hour":7,"historical_route_reconstruction":{"enabled":True,"start":"2026-05-01","end":"2026-08-01","template_start":"2026-08-13","padding_segments":0,"min_candidate_coverage":0.3}}
    out,_=reconstruct_historical_routes(tmp_path,events,visits,cfg)
    assert "entered_at" not in out.columns
    assert out.iloc[0].template_session_file=="2026-08-13.gpx"
