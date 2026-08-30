from pathlib import Path
import joblib
import pandas as pd
from habuai.hardening.modeling import fit_model,fit_production_model,score_holdout


def _data():
    return pd.DataFrame([
        {"entered_at":pd.Timestamp("2026-08-27T20:00:00+09:00"),"habu_capture":0,"sin_hour":0.0},
        {"entered_at":pd.Timestamp("2026-08-27T21:00:00+09:00"),"habu_capture":1,"sin_hour":1.0},
        {"entered_at":pd.Timestamp("2026-08-28T20:00:00+09:00"),"habu_capture":1,"sin_hour":0.5},
    ])


def test_evaluation_excludes_holdout_but_production_uses_it(tmp_path:Path):
    (tmp_path/"models").mkdir();cfg={"baseline_cutoff":"2026-08-28T07:00:00+09:00","holdout_start":"2026-08-28T07:00:00+09:00","night_rollover_hour":7}
    data=_data();evaluation=fit_model(tmp_path,data,cfg);holdout=score_holdout(tmp_path,data,cfg);production=fit_production_model(tmp_path,data,cfg)
    assert evaluation["rows"]==2
    assert evaluation["positives"]==1
    assert holdout["rows"]==1
    assert holdout["positive_rows"]==1
    assert production["rows"]==3
    assert production["positives"]==2
    assert (tmp_path/"models"/"habu_occurrence_evaluation.joblib").exists()
    assert (tmp_path/"models"/"habu_occurrence_production.joblib").exists()
    assert joblib.load(tmp_path/"models"/"habu_occurrence.joblib")["model_role"]=="production"
