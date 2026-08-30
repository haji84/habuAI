from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .spatial_backtest import STATIC_SPATIAL_FEATURES, run_reconstructed_spatial_backtest


def _group_columns(data:pd.DataFrame)->dict[str,list[str]]:
    cols=set(data.columns)
    def present(xs):return [x for x in xs if x in cols]
    groups={
        "road_geometry":present(["length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m"]),
        "terrain_context":present(["stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"]),
        "capture_alltime":sorted(c for c in cols if c.startswith("hist_capture_count_")),
        "capture_30d":sorted(c for c in cols if c.startswith("hist_capture_30d_count_")),
        "capture_90d":sorted(c for c in cols if c.startswith("hist_capture_90d_count_")),
        "capture_recency":sorted(c for c in cols if c.startswith("days_since_capture_")),
        "large_capture_history":sorted(c for c in cols if c.startswith("hist_large_capture_count_")),
        "bio_alltime":sorted(c for c in cols if c.startswith("hist_bio_count_")),
        "bio_30d_total":sorted(c for c in cols if c.startswith("hist_bio_30d_count_")),
        "bio_species_30d":sorted(c for c in cols if c.startswith("hist_bio_") and "_30d_count_" in c and not c.startswith("hist_bio_30d_count_")),
    }
    return {k:v for k,v in groups.items() if v}


def _full_features(data:pd.DataFrame)->list[str]:
    static=[c for c in STATIC_SPATIAL_FEATURES if c in data.columns]
    hist=sorted(c for c in data.columns if c.startswith("hist_") or c.startswith("days_since_capture_"))
    return static+hist


def _metric(summary:dict,key:str):
    v=summary.get(key)
    return None if v is None else float(v)


def run_spatial_ablation_suite(root:Path,data:pd.DataFrame,cfg:dict,full_summary:dict|None=None)->dict:
    """Leave-one-feature-group-out ablation on the exact same walk-forward nights.

    Positive delta_top30/delta_top10 means removing the group improves the score, so that group
    was hurting that metric. Positive contribution_top30/contribution_top10 means the group helps
    the full model. For median rank percentile, lower is better; positive contribution_median means
    the group improves the median capture rank.
    """
    out_dir=root/"reports";out_dir.mkdir(parents=True,exist_ok=True)
    if data.empty:
        result={"status":"empty"};(out_dir/"historical_spatial_ablation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
    full_features=_full_features(data);groups=_group_columns(data)
    if full_summary is None:
        full_summary=run_reconstructed_spatial_backtest(root,data,cfg,feature_mode="enhanced",feature_columns=full_features,output_tag="ablation_full")
    rows=[]
    for name,gcols in groups.items():
        kept=[c for c in full_features if c not in set(gcols)]
        s=run_reconstructed_spatial_backtest(root,data,cfg,feature_mode="enhanced",feature_columns=kept,output_tag=f"ablation_without_{name}")
        row={
            "group":name,
            "removed_feature_count":len(gcols),
            "removed_features":"|".join(gcols),
            "remaining_feature_count":len(kept),
            "nights_scored":s.get("nights_scored"),
            "top30_hits_without":s.get("location_hits_top_k"),
            "top1_hits_without":s.get("location_hits_top_1pct"),
            "top5_hits_without":s.get("location_hits_top_5pct"),
            "top10_hits_without":s.get("location_hits_top_10pct"),
            "median_rank_percentile_without":s.get("median_actual_rank_percentile"),
        }
        for metric,out_name in [
            ("location_hits_top_k","top30"),("location_hits_top_1pct","top1"),
            ("location_hits_top_5pct","top5"),("location_hits_top_10pct","top10")]:
            full_v=_metric(full_summary,metric);without_v=_metric(s,metric)
            row[f"contribution_{out_name}"]=None if full_v is None or without_v is None else full_v-without_v
        fm=_metric(full_summary,"median_actual_rank_percentile");wm=_metric(s,"median_actual_rank_percentile")
        row["contribution_median_rank_percentile"]=None if fm is None or wm is None else wm-fm
        rows.append(row)
    detail=pd.DataFrame(rows)
    if not detail.empty:
        detail=detail.sort_values(["contribution_top30","contribution_top10","contribution_median_rank_percentile"],ascending=[False,False,False],kind="stable")
    detail.to_csv(out_dir/"historical_spatial_ablation.csv",index=False)
    result={
        "status":"ok",
        "method":"leave one semantic feature group out; exact same leakage-safe 33-night walk-forward, estimator and labels as enhanced model",
        "full_model":{
            "feature_count":len(full_features),
            "top30_hits":full_summary.get("location_hits_top_k"),
            "top1_hits":full_summary.get("location_hits_top_1pct"),
            "top5_hits":full_summary.get("location_hits_top_5pct"),
            "top10_hits":full_summary.get("location_hits_top_10pct"),
            "median_rank_percentile":full_summary.get("median_actual_rank_percentile"),
        },
        "groups_tested":len(groups),
        "group_definitions":groups,
        "results":detail.to_dict(orient="records"),
        "interpretation":{
            "positive_contribution_hit_metric":"group helps the full model because removing it loses hits",
            "negative_contribution_hit_metric":"group hurts that hit metric because removing it gains hits",
            "positive_contribution_median_rank_percentile":"group helps median rank because removing it makes percentile larger/worse",
            "negative_contribution_median_rank_percentile":"group hurts median rank because removing it makes percentile smaller/better",
        },
    }
    (out_dir/"historical_spatial_ablation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
