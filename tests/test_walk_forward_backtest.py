from pathlib import Path
import numpy as np
import pandas as pd
from habuai.hardening.backtest import _prior_correct_balanced_probability,run_walk_forward_backtest


def test_prior_correction_reduces_balanced_probability_for_rare_event():
    p=_prior_correct_balanced_probability(np.array([0.5]),0.01)
    assert 0 < p[0] < 0.02


def test_walk_forward_uses_only_pre_night_training_rows(tmp_path:Path):
    (tmp_path/"reports").mkdir()
    rows=[]
    for i in range(30):
        rows.append({"entered_at":pd.Timestamp("2026-08-20T20:00:00+09:00")+pd.Timedelta(minutes=i),"habu_capture":1 if i<5 else 0,"learning_row_source":"gpx_visit","segment_id":f"S{i%4}","sin_hour":float(i%2)})
    rows += [
        {"entered_at":pd.Timestamp("2026-08-21T20:00:00+09:00"),"habu_capture":0,"learning_row_source":"gpx_visit","segment_id":"A","sin_hour":0.1},
        {"entered_at":pd.Timestamp("2026-08-21T21:00:00+09:00"),"habu_capture":1,"learning_row_source":"gpx_visit","segment_id":"B","sin_hour":0.9},
    ]
    rows.append({"entered_at":pd.Timestamp("2026-08-22T20:00:00+09:00"),"habu_capture":1,"learning_row_source":"capture_gps_anchor","segment_id":"FUTURE","sin_hour":1.0})
    data=pd.DataFrame(rows)
    events=pd.DataFrame([{"species":"ハブ","event_type":"捕獲","timestamp":pd.Timestamp("2026-08-21T21:00:00+09:00"),"segment_id":"B","individual_count":1}])
    cfg={"night_rollover_hour":7,"walk_forward_backtest":{"top_k_segments":1,"min_train_positives":5,"min_train_negatives":20}}
    result=run_walk_forward_backtest(tmp_path,data,events,cfg)
    nightly=pd.read_csv(tmp_path/"reports"/"walk_forward_backtest_nightly.csv")
    r=nightly[nightly.night=="2026-08-21"].iloc[0]
    assert r.train_rows==30
    assert r.train_positives==5
    assert pd.notna(r.pre_night_predicted_capture_count_rounded)
    assert result["route_replay_nights_scored"]>=1
    assert result["pre_night_count_nights_scored"]>=1
    assert (tmp_path/"reports"/"walk_forward_backtest_segment_ranks.csv").exists()
