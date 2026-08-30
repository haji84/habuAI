from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import _feature_columns
from .spatial_backtest import STATIC_SPATIAL_FEATURES


def _as_jst(s):
    return pd.to_datetime(s, format="mixed", utc=True).dt.tz_convert("Asia/Tokyo")


def _night(s, rollover=7):
    return (_as_jst(s)-pd.Timedelta(hours=rollover)).dt.date.astype(str)


def _estimator(name: str):
    if name == "logistic_c005":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=.05))
    if name == "logistic_c02":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=.2))
    if name == "logistic_c1":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), ExtraTreesClassifier(n_estimators=350, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=84, n_jobs=-1))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestClassifier(n_estimators=350, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", random_state=84, n_jobs=-1))
    if name == "hist_gb":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), HistGradientBoostingClassifier(max_iter=180, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=84))
    raise KeyError(name)


def _fit(model, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y); return
    last=list(model.named_steps)[-1] if hasattr(model, "named_steps") else None
    try:
        model.fit(X, y, **({f"{last}__sample_weight":sample_weight} if last else {"sample_weight":sample_weight}))
    except (TypeError, ValueError):
        model.fit(X, y)


def _spatial_feature_sets(data: pd.DataFrame) -> dict[str,list[str]]:
    all_feats=[c for c in data.columns if c in STATIC_SPATIAL_FEATURES or c.startswith("hist_") or c.startswith("days_since_capture_")]
    groups={
        "road_geometry": {"length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m"},
        "terrain": {"stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"},
        "recency": {c for c in all_feats if c.startswith("days_since_capture_")},
        "capture30": {c for c in all_feats if c.startswith("hist_capture_30d_")},
        "capture90": {c for c in all_feats if c.startswith("hist_capture_90d_")},
        "capture_all": {c for c in all_feats if c.startswith("hist_capture_count_")},
        "large": {c for c in all_feats if c.startswith("hist_large_capture_")},
        "bio": {c for c in all_feats if c.startswith("hist_bio_")},
    }
    def keep_without(*names):
        drop=set().union(*(groups[n] for n in names)); return [c for c in all_feats if c not in drop]
    return {
        "full58": all_feats,
        "no_recency": keep_without("recency"),
        "no_road": keep_without("road_geometry"),
        "no_recency_road": keep_without("recency","road_geometry"),
        "no_recency_road_capture30": keep_without("recency","road_geometry","capture30"),
        "terrain_large": [c for c in all_feats if c in groups["terrain"]|groups["large"]],
        "terrain_large_capture": [c for c in all_feats if c in groups["terrain"]|groups["large"]|groups["capture_all"]|groups["capture30"]|groups["capture90"]],
        "terrain_large_no_bio": [c for c in all_feats if c not in groups["bio"]|groups["recency"]|groups["road_geometry"]],
    }


def _spatial_score(row):
    # Operational priority: exact short-list hits first, then top-10%, then broad ranking stability.
    return 4*row["top30_rate"] + 2*row["top10_rate"] + row["top5_rate"] - .5*row["median_rank_pct"]


def run_spatial_tournament(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,dict]:
    rollover=int(cfg.get("night_rollover_hour",7)); top_k=int(cfg.get("historical_spatial_backtest",{}).get("top_k_segments",30))
    min_pos=int(cfg.get("historical_spatial_backtest",{}).get("min_train_positives",3));min_neg=int(cfg.get("historical_spatial_backtest",{}).get("min_train_negatives",30))
    estimators=["logistic_c005","logistic_c02","logistic_c1","extra_trees","random_forest","hist_gb"]
    sets=_spatial_feature_sets(data); rows=[]
    for fs_name, feats in sets.items():
        if not feats: continue
        for est_name in estimators:
            hit30=hit1=hit5=hit10=eligible=0; pcts=[]; nights=0
            for night in sorted(data.night.astype(str).unique()):
                tr=data[data.night.astype(str)<night];te=data[data.night.astype(str)==night].copy()
                if int(tr.spatial_label.sum())<min_pos or int((tr.spatial_label==0).sum())<min_neg or te.empty: continue
                Xtr=tr.reindex(columns=feats).replace([np.inf,-np.inf],np.nan); y=tr.spatial_label.astype(int);m=_estimator(est_name);w=pd.to_numeric(tr.sample_weight,errors="coerce").fillna(1.0).to_numpy()
                _fit(m,Xtr,y,w);te["p"]=m.predict_proba(te.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];te=te.sort_values("p",ascending=False).reset_index(drop=True);te["rank"]=np.arange(1,len(te)+1);te["pct"]=te["rank"]/len(te);a=te[te.spatial_label==1]
                eligible+=len(a);hit30+=int((a["rank"]<=top_k).sum());hit1+=int((a.pct<=.01).sum());hit5+=int((a.pct<=.05).sum());hit10+=int((a.pct<=.10).sum());pcts+=a.pct.tolist();nights+=1
            if eligible:
                r={"feature_set":fs_name,"estimator":est_name,"feature_count":len(feats),"nights_scored":nights,"eligible":eligible,"top30_hits":hit30,"top30_rate":hit30/eligible,"top1_hits":hit1,"top1_rate":hit1/eligible,"top5_hits":hit5,"top5_rate":hit5/eligible,"top10_hits":hit10,"top10_rate":hit10/eligible,"median_rank_pct":float(np.median(pcts)) if pcts else np.nan};r["selection_score"]=_spatial_score(r);rows.append(r)
    out=pd.DataFrame(rows).sort_values(["selection_score","top30_hits","top10_hits"],ascending=False).reset_index(drop=True)
    best=out.iloc[0].to_dict() if not out.empty else {}
    return out,{"status":"ok" if best else "insufficient-data","candidates":int(len(out)),"best":best,"selection_rule":"4*top30_rate + 2*top10_rate + top5_rate - 0.5*median_rank_percentile"}


def _temporal_feature_sets(data: pd.DataFrame) -> dict[str,list[str]]:
    all_feats=_feature_columns(data)
    clock=[c for c in ["sin_hour","cos_hour"] if c in all_feats]
    env=[c for c in all_feats if c not in clock and not c.startswith("bio_") and c not in {"curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m","segment_prior_visits","slope_observed_deg","local_relief_proxy_m","mean_speed_mps","elevation_m"}]
    bio=[c for c in all_feats if c.startswith("bio_")]
    spatial=[c for c in all_feats if c in {"curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m","segment_prior_visits","slope_observed_deg","local_relief_proxy_m"}]
    return {"clock":clock,"clock_env":list(dict.fromkeys(clock+env)),"clock_env_bio":list(dict.fromkeys(clock+env+bio)),"all_temporal":all_feats,"clock_spatial":list(dict.fromkeys(clock+spatial)),"clock_env_spatial":list(dict.fromkeys(clock+env+spatial))}


def _capture_hour_prior(train_events: pd.DataFrame, candidate_hours: list[int], bandwidth: float=1.0):
    cap=train_events[(train_events.species=="ハブ")&(train_events.event_type=="捕獲")].copy()
    if cap.empty:return None
    h=_as_jst(cap.timestamp).dt.hour.to_numpy(float)+_as_jst(cap.timestamp).dt.minute.to_numpy(float)/60
    scores=[]
    for ch in candidate_hours:
        d=np.abs(h-ch);d=np.minimum(d,24-d);scores.append(float(np.exp(-.5*(d/bandwidth)**2).sum()))
    return int(candidate_hours[int(np.argmax(scores))]) if scores else None


def run_temporal_tournament(model_data: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,dict]:
    rollover=int(cfg.get("night_rollover_hour",7)); min_pos=5;min_neg=20
    d=model_data.copy();d["entered_at"]=_as_jst(d.entered_at);e=events.copy();e["timestamp"]=_as_jst(e.timestamp);g=d[d.learning_row_source=="gpx_visit"].copy();g["night"]=_night(g.entered_at,rollover)
    estimators=["logistic_c005","logistic_c02","logistic_c1","extra_trees","random_forest","hist_gb"]
    feature_sets=_temporal_feature_sets(d); rows=[]
    # Positive-only historical clock prior is a deliberately different baseline.
    for bw in [.5,1.0,1.5,2.0]:
        hits=hits_pm1=eligible=nights=0
        for night in sorted(g.night.unique()):
            start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=rollover);end=start+pd.Timedelta(days=1);test=g[(g.entered_at>=start)&(g.entered_at<end)];actual=e[(e.species=="ハブ")&(e.event_type=="捕獲")&(e.timestamp>=start)&(e.timestamp<end)];past=e[e.timestamp<start]
            if test.empty or actual.empty: continue
            hours=sorted(test.entered_at.dt.hour.unique().tolist());peak=_capture_hour_prior(past,hours,bw)
            if peak is None:continue
            ah=actual.timestamp.dt.hour.to_numpy();hits+=int((ah==peak).sum());delta=np.minimum((ah-peak)%24,(peak-ah)%24);hits_pm1+=int((delta<=1).sum());eligible+=len(actual);nights+=1
        if eligible: rows.append({"feature_set":"capture_clock_prior","estimator":f"cyclic_kde_bw{bw}","feature_count":1,"nights_scored":nights,"eligible":eligible,"peak_hour_hits":hits,"peak_hour_rate":hits/eligible,"within_1h_hits":hits_pm1,"within_1h_rate":hits_pm1/eligible})
    for fs_name,feats in feature_sets.items():
        if not feats:continue
        for est_name in estimators:
            hits=hits_pm1=eligible=nights=0
            for night in sorted(g.night.unique()):
                start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=rollover);end=start+pd.Timedelta(days=1);tr=d[d.entered_at<start];test=g[(g.entered_at>=start)&(g.entered_at<end)].copy();actual=e[(e.species=="ハブ")&(e.event_type=="捕獲")&(e.timestamp>=start)&(e.timestamp<end)]
                if test.empty or actual.empty or int(tr.habu_capture.sum())<min_pos or int((tr.habu_capture==0).sum())<min_neg:continue
                m=_estimator(est_name);X=tr.reindex(columns=feats).replace([np.inf,-np.inf],np.nan);y=tr.habu_capture.astype(int);_fit(m,X,y);test["p"]=m.predict_proba(test.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];hs=test.groupby(test.entered_at.dt.hour).p.sum();peak=int(hs.idxmax());ah=actual.timestamp.dt.hour.to_numpy();hits+=int((ah==peak).sum());delta=np.minimum((ah-peak)%24,(peak-ah)%24);hits_pm1+=int((delta<=1).sum());eligible+=len(actual);nights+=1
            if eligible:rows.append({"feature_set":fs_name,"estimator":est_name,"feature_count":len(feats),"nights_scored":nights,"eligible":eligible,"peak_hour_hits":hits,"peak_hour_rate":hits/eligible,"within_1h_hits":hits_pm1,"within_1h_rate":hits_pm1/eligible})
    out=pd.DataFrame(rows)
    if not out.empty:
        out["selection_score"]=2*out.peak_hour_rate+out.within_1h_rate;out=out.sort_values(["selection_score","peak_hour_hits","within_1h_hits"],ascending=False).reset_index(drop=True)
    best=out.iloc[0].to_dict() if not out.empty else {}
    return out,{"status":"ok" if best else "insufficient-data","candidates":int(len(out)),"best":best,"selection_rule":"2*exact_peak_hour_rate + within_1h_rate","note":"temporal supervised evaluation uses only actual GPX survey nights; reconstructed May-Jul routes are never treated as clock-time negatives"}


def run_model_tournament(root: Path, cfg: dict) -> dict:
    p=root/"data"/"processed";r=root/"reports";r.mkdir(parents=True,exist_ok=True)
    spatial=pd.read_csv(p/"reconstructed_spatial_learning_2026-05_07.csv",low_memory=False);model=pd.read_csv(p/"learning_10m_road.csv",low_memory=False);events=pd.read_csv(p/"events_matched.csv",low_memory=False)
    st,ss=run_spatial_tournament(spatial,cfg);tt,ts=run_temporal_tournament(model,events,cfg);st.to_csv(r/"model_tournament_spatial.csv",index=False);tt.to_csv(r/"model_tournament_temporal.csv",index=False)
    summary={"status":"ok","spatial":ss,"temporal":ts,"integration_next":"score each actual GPX visit with selected spatial and temporal winners, then rank segment-time pairs; keep exact point and time-window metrics separate"}
    (r/"model_tournament_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");return summary
