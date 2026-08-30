from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _feature_columns(data):
    base=[
        "sin_hour","cos_hour","mean_speed_mps","elevation_m",
        "rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_48h_mm",
        "temperature_c","temperature_change_3visits_c","humidity_pct","dew_point_c","temp_dewpoint_spread_c","hours_since_rain",
        "weather_code","surface_pressure","cloud_cover","fog_wmo_flag","fog_proxy_flag",
        "moon_age_days","moon_phase_sin","moon_phase_cos","moon_illumination",
        "tide_height_cm","tide_change_1h_cm","tide_state_code","minutes_to_nearest_turning_tide","tide_source_available",
        "curvature_deg","segment_prior_visits","slope_observed_deg","local_relief_proxy_m","road_class_code",
        "junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"
    ]
    return [c for c in base+[c for c in data if c.startswith("bio_") and (c.endswith("_count") or c.endswith("_nearest_m"))] if c in data]


def _new_model():
    return make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",C=.2))


def fit_model(root,data,cfg):
    """Fit the frozen-cutoff evaluation model. Holdout rows never enter this model."""
    train=data[data.entered_at<pd.Timestamp(cfg["baseline_cutoff"])].copy()
    if train.empty or train.habu_capture.nunique()<2:return {"status":"insufficient-data"}
    train["survey_night"]=(pd.to_datetime(train.entered_at)-pd.Timedelta(hours=cfg.get("night_rollover_hour",7))).dt.date.astype(str)
    feats=_feature_columns(train);X=train[feats].replace([np.inf,-np.inf],np.nan);y=train.habu_capture.astype(int)
    oy,op,folds=[],[],[]
    for night in sorted(train.survey_night.unique()):
        te=train.survey_night==night;tr=~te
        if y[tr].nunique()<2:folds.append({"night":night,"status":"skipped"});continue
        m=_new_model();m.fit(X.loc[tr],y[tr]);pr=m.predict_proba(X.loc[te])[:,1];oy+=y[te].tolist();op+=pr.tolist();f={"night":night,"status":"ok","rows":int(te.sum()),"positives":int(y[te].sum()),"brier":float(brier_score_loss(y[te],pr))}
        if y[te].nunique()>1:f["pr_auc"]=float(average_precision_score(y[te],pr))
        folds.append(f)
    m=_new_model();m.fit(X,y);ptr=m.predict_proba(X)[:,1];r={"status":"ok","model_role":"evaluation","rows":len(train),"positives":int(y.sum()),"features":feats,"brier_train":float(brier_score_loss(y,ptr)),"cv_method":"leave-one-survey-night-out","cv_folds":folds}
    if len(set(oy))>1:r.update(brier_loono=float(brier_score_loss(oy,op)),pr_auc_loono=float(average_precision_score(oy,op)),oof_rows=len(oy),oof_positives=int(sum(oy)))
    joblib.dump({"model":m,"features":feats,"model_role":"evaluation","cutoff":cfg["baseline_cutoff"]},root/"models"/"habu_occurrence_evaluation.joblib")
    return r


def fit_production_model(root,data,cfg):
    """Fit the operational model on every currently available learning row."""
    train=data.copy()
    if train.empty or train.habu_capture.nunique()<2:return {"status":"insufficient-data"}
    feats=_feature_columns(train);X=train[feats].replace([np.inf,-np.inf],np.nan);y=train.habu_capture.astype(int)
    m=_new_model();m.fit(X,y);pr=m.predict_proba(X)[:,1]
    payload={"model":m,"features":feats,"model_role":"production","trained_through":str(pd.to_datetime(train.entered_at).max())}
    joblib.dump(payload,root/"models"/"habu_occurrence.joblib")
    joblib.dump(payload,root/"models"/"habu_occurrence_production.joblib")
    return {"status":"ok","model_role":"production","rows":len(train),"positives":int(y.sum()),"features":feats,"brier_train":float(brier_score_loss(y,pr)),"trained_through":payload["trained_through"]}


def score_holdout(root,data,cfg):
    p=root/"models"/"habu_occurrence_evaluation.joblib";hold=data[data.entered_at>=pd.Timestamp(cfg["holdout_start"])].copy()
    if not p.exists() or hold.empty:return {"status":"no-holdout"}
    o=joblib.load(p);X=hold.reindex(columns=o["features"]).replace([np.inf,-np.inf],np.nan);pr=o["model"].predict_proba(X)[:,1];r={"status":"ok","model_role":"evaluation","rows":len(hold),"positive_rows":int(hold.habu_capture.sum()),"mean_pred_prob":float(pr.mean()),"max_pred_prob":float(pr.max())}
    if hold.habu_capture.nunique()>1:r["brier"]=float(brier_score_loss(hold.habu_capture,pr))
    return r
