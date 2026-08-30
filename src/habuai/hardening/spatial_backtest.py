from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPATIAL_FEATURES=[
    "length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m",
    "stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m",
    "residential_distance_m",
]


def _model():
    return make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",C=.2))


def run_reconstructed_spatial_backtest(root:Path,data:pd.DataFrame,cfg:dict)->dict:
    """Walk forward over trusted May-July reconstructed routes using static road context only."""
    out_dir=root/"reports";out_dir.mkdir(parents=True,exist_ok=True)
    if data.empty:
        result={"status":"empty","nights":0};(out_dir/"historical_spatial_backtest_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
    bcfg=cfg.get("historical_spatial_backtest",{});top_k=int(bcfg.get("top_k_segments",30));min_pos=int(bcfg.get("min_train_positives",3));min_neg=int(bcfg.get("min_train_negatives",30))
    feats=[c for c in SPATIAL_FEATURES if c in data.columns]
    rows=[];ranks=[]
    for night in sorted(data.night.astype(str).unique()):
        train=data[data.night.astype(str)<night].copy();test=data[data.night.astype(str)==night].copy();pos=int(train.spatial_label.sum());neg=int((train.spatial_label==0).sum());base={"night":night,"train_rows":int(len(train)),"train_positive_segments":pos,"train_negative_segments":neg,"candidate_segments":int(len(test)),"actual_positive_segments":int(test.spatial_label.sum())}
        if pos<min_pos or neg<min_neg or not feats:
            base["status"]="insufficient-training-history";rows.append(base);continue
        m=_model();m.fit(train[feats].replace([np.inf,-np.inf],np.nan),train.spatial_label.astype(int),logisticregression__sample_weight=pd.to_numeric(train.sample_weight,errors="coerce").fillna(1.0))
        test=test.copy();test["pred_prob"]=m.predict_proba(test[feats].replace([np.inf,-np.inf],np.nan))[:,1];test=test.sort_values("pred_prob",ascending=False).reset_index(drop=True);test["rank"]=np.arange(1,len(test)+1);test["rank_percentile"]=test["rank"]/max(len(test),1);test["night"]=night
        actual=test[test.spatial_label==1];hits30=int((actual["rank"]<=top_k).sum());hits10=int((actual.rank_percentile<=0.10).sum());n_actual=int(len(actual));base.update({"status":"ok","location_hits_top_k":hits30,"location_hit_rate_top_k":None if n_actual==0 else hits30/n_actual,"location_hits_top_10pct":hits10,"location_hit_rate_top_10pct":None if n_actual==0 else hits10/n_actual,"actual_capture_median_rank":None if n_actual==0 else float(actual["rank"].median()),"actual_capture_median_rank_percentile":None if n_actual==0 else float(actual.rank_percentile.median()),"random_expected_top_k_rate":min(1.0,top_k/max(len(test),1)),"random_expected_top_10pct_rate":0.10});rows.append(base)
        ranks.append(test[["night","segment_id","spatial_label","reconstruction_confidence","sample_weight","rank","rank_percentile","pred_prob"]])
    detail=pd.DataFrame(rows);detail.to_csv(out_dir/"historical_spatial_backtest_nightly.csv",index=False)
    if ranks:pd.concat(ranks,ignore_index=True).to_csv(out_dir/"historical_spatial_backtest_segment_ranks.csv",index=False)
    ok=detail[detail.status=="ok"] if not detail.empty else pd.DataFrame();summary={"status":"ok" if not ok.empty else "insufficient-data","method":"walk-forward by operational night using B-high/B-medium reconstructed route negatives and static road/terrain features only; no fabricated historical clock time","nights_total":int(len(detail)),"nights_scored":int(len(ok)),"nights_skipped":int(len(detail)-len(ok)),"top_k_segments":top_k,"features":feats}
    if not ok.empty:
        eligible=int(ok.actual_positive_segments.sum());summary.update({"location_eligible_positive_segments":eligible,"location_hits_top_k":int(ok.location_hits_top_k.sum()),"location_hit_rate_top_k":None if eligible==0 else float(ok.location_hits_top_k.sum()/eligible),"location_hits_top_10pct":int(ok.location_hits_top_10pct.sum()),"location_hit_rate_top_10pct":None if eligible==0 else float(ok.location_hits_top_10pct.sum()/eligible),"median_actual_rank_percentile":float(ok.actual_capture_median_rank_percentile.dropna().median()) if ok.actual_capture_median_rank_percentile.notna().any() else None})
    (out_dir/"historical_spatial_backtest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");return summary
