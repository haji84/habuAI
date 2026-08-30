from pathlib import Path
import pandas as pd
from habuai.hardening.reconstruct import build_reconstructed_spatial_learning
from habuai.hardening.spatial_backtest import run_reconstructed_spatial_backtest


def test_trusted_reconstruction_becomes_spatial_labels(tmp_path:Path):
    (tmp_path/"data"/"processed").mkdir(parents=True);(tmp_path/"reports").mkdir()
    r=pd.DataFrame([
        {"night":"2026-05-01","segment_id":"A","capture_anchor_segment":True,"reconstruction_confidence":"B-high","segment_support_fraction":1.0,"support_sessions":3,"template_session_file":"x"},
        {"night":"2026-05-01","segment_id":"B","capture_anchor_segment":False,"reconstruction_confidence":"B-high","segment_support_fraction":1.0,"support_sessions":3,"template_session_file":"x"},
        {"night":"2026-05-02","segment_id":"C","capture_anchor_segment":False,"reconstruction_confidence":"C-low","segment_support_fraction":0.2,"support_sessions":1,"template_session_file":"y"},
    ])
    out,audit=build_reconstructed_spatial_learning(tmp_path,r)
    assert len(out)==2
    assert int(out.spatial_label.sum())==1
    assert int((out.spatial_label==0).sum())==1
    assert set(out.reconstruction_confidence)=={"B-high"}
    assert audit["c_low_excluded"]==1


def test_spatial_backtest_walks_forward(tmp_path:Path):
    (tmp_path/"reports").mkdir()
    rows=[]
    for d in range(1,5):
        night=f"2026-05-0{d}"
        for i in range(20):
            rows.append({"night":night,"segment_id":f"{d}-{i}","spatial_label":1 if i==0 else 0,"sample_weight":1.0,"reconstruction_confidence":"B-high","length_m":10.0,"bearing_deg":float(i),"curvature_deg":0.0,"road_class_code":1.0,"junction_distance_m":float(i),"stream_distance_m":float(i+1),"coast_distance_m":100.0,"forest_distance_m":20.0,"farmland_distance_m":30.0,"residential_distance_m":40.0})
    data=pd.DataFrame(rows)
    result=run_reconstructed_spatial_backtest(tmp_path,data,{"historical_spatial_backtest":{"min_train_positives":2,"min_train_negatives":30,"top_k_segments":5}})
    assert result["nights_scored"]>=2
    assert (tmp_path/"reports"/"historical_spatial_backtest_nightly.csv").exists()
    assert (tmp_path/"reports"/"historical_spatial_backtest_segment_ranks.csv").exists()
