from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import (
    _as_jst,_estimator,_fit,_night,_spatial_feature_sets,_spatial_score,_temporal_feature_sets,_capture_hour_prior
)


def spatial_eval(data, cfg, fs_name, feats, est_name):
    top_k=int(cfg.get("historical_spatial_backtest",{}).get("top_k_segments",30));min_pos=3;min_neg=30
    h30=h1=h5=h10=eligible=nights=0;pcts=[]
    for night in sorted(data.night.astype(str).unique()):
        tr=data[data.night.astype(str)<night];te=data[data.night.astype(str)==night].copy()
        if int(tr.spatial_label.sum())<min_pos or int((tr.spatial_label==0).sum())<min_neg or te.empty:continue
        m=_estimator(est_name);X=tr.reindex(columns=feats).replace([np.inf,-np.inf],np.nan);y=tr.spatial_label.astype(int);w=pd.to_numeric(tr.sample_weight,errors="coerce").fillna(1.0).to_numpy();_fit(m,X,y,w)
        te["p"]=m.predict_proba(te.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];te=te.sort_values("p",ascending=False).reset_index(drop=True);te["rank"]=np.arange(1,len(te)+1);te["pct"]=te["rank"]/len(te);a=te[te.spatial_label==1]
        eligible+=len(a);h30+=int((a["rank"]<=top_k).sum());h1+=int((a.pct<=.01).sum());h5+=int((a.pct<=.05).sum());h10+=int((a.pct<=.10).sum());pcts+=a.pct.tolist();nights+=1
    if not eligible:return None
    r={"feature_set":fs_name,"estimator":est_name,"feature_count":len(feats),"nights_scored":nights,"eligible":eligible,"top30_hits":h30,"top30_rate":h30/eligible,"top1_hits":h1,"top1_rate":h1/eligible,"top5_hits":h5,"top5_rate":h5/eligible,"top10_hits":h10,"top10_rate":h10/eligible,"median_rank_pct":float(np.median(pcts))};r["selection_score"]=_spatial_score(r);return r


def temporal_eval(d,e,cfg,fs_name,feats,est_name):
    rollover=int(cfg.get("night_rollover_hour",7));g=d[d.learning_row_source=="gpx_visit"].copy();g["night"]=_night(g.entered_at,rollover);hits=hits1=eligible=nights=0
    for night in sorted(g.night.unique()):
        start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=rollover);end=start+pd.Timedelta(days=1);tr=d[d.entered_at<start];te=g[(g.entered_at>=start)&(g.entered_at<end)].copy();a=e[(e.species=="ハブ")&(e.event_type=="捕獲")&(e.timestamp>=start)&(e.timestamp<end)]
        if te.empty or a.empty or int(tr.habu_capture.sum())<5 or int((tr.habu_capture==0).sum())<20:continue
        m=_estimator(est_name);_fit(m,tr.reindex(columns=feats).replace([np.inf,-np.inf],np.nan),tr.habu_capture.astype(int));te["p"]=m.predict_proba(te.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];hs=te.groupby(te.entered_at.dt.hour).p.sum();peak=int(hs.idxmax());ah=a.timestamp.dt.hour.to_numpy();hits+=int((ah==peak).sum());delta=np.minimum((ah-peak)%24,(peak-ah)%24);hits1+=int((delta<=1).sum());eligible+=len(a);nights+=1
    if not eligible:return None
    r={"feature_set":fs_name,"estimator":est_name,"feature_count":len(feats),"nights_scored":nights,"eligible":eligible,"peak_hour_hits":hits,"peak_hour_rate":hits/eligible,"within_1h_hits":hits1,"within_1h_rate":hits1/eligible};r["selection_score"]=2*r["peak_hour_rate"]+r["within_1h_rate"];return r


def main():
    root=Path(__file__).resolve().parents[1];cfg=pipeline.load_config(root);p=root/"data"/"processed";reports=root/"reports"
    spatial=pd.read_csv(p/"reconstructed_spatial_learning_2026-05_07.csv",low_memory=False);d=pd.read_csv(p/"learning_10m_road.csv",low_memory=False);e=pd.read_csv(p/"events_matched.csv",low_memory=False);d["entered_at"]=_as_jst(d.entered_at);e["timestamp"]=_as_jst(e.timestamp)
    ssets=_spatial_feature_sets(spatial);phase1=[]
    for name,feats in ssets.items():
        r=spatial_eval(spatial,cfg,name,feats,"logistic_c02")
        if r:phase1.append(r)
    p1=pd.DataFrame(phase1).sort_values(["selection_score","top30_hits","top10_hits"],ascending=False);top_s=p1.feature_set.head(3).tolist();phase2=[]
    for name in top_s:
        for est in ["logistic_c005","logistic_c02","logistic_c1","extra_trees","random_forest","hist_gb"]:
            r=spatial_eval(spatial,cfg,name,ssets[name],est)
            if r:phase2.append(r)
    sres=pd.DataFrame(phase2).sort_values(["selection_score","top30_hits","top10_hits"],ascending=False).drop_duplicates(["feature_set","estimator"])

    tsets=_temporal_feature_sets(d);tp1=[]
    for name,feats in tsets.items():
        r=temporal_eval(d,e,cfg,name,feats,"logistic_c02")
        if r:tp1.append(r)
    tp1df=pd.DataFrame(tp1).sort_values(["selection_score","peak_hour_hits","within_1h_hits"],ascending=False);top_t=tp1df.feature_set.head(3).tolist();tp2=[]
    for name in top_t:
        for est in ["logistic_c005","logistic_c02","logistic_c1","extra_trees","random_forest","hist_gb"]:
            r=temporal_eval(d,e,cfg,name,tsets[name],est)
            if r:tp2.append(r)
    # Add genuinely different positive-only clock-density baselines.
    rollover=int(cfg.get("night_rollover_hour",7));g=d[d.learning_row_source=="gpx_visit"].copy();g["night"]=_night(g.entered_at,rollover)
    for bw in [.5,1.0,1.5,2.0]:
        hits=hits1=eligible=nights=0
        for night in sorted(g.night.unique()):
            start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=rollover);end=start+pd.Timedelta(days=1);te=g[(g.entered_at>=start)&(g.entered_at<end)];a=e[(e.species=="ハブ")&(e.event_type=="捕獲")&(e.timestamp>=start)&(e.timestamp<end)];past=e[e.timestamp<start]
            if te.empty or a.empty:continue
            peak=_capture_hour_prior(past,sorted(te.entered_at.dt.hour.unique().tolist()),bw)
            if peak is None:continue
            ah=a.timestamp.dt.hour.to_numpy();hits+=int((ah==peak).sum());delta=np.minimum((ah-peak)%24,(peak-ah)%24);hits1+=int((delta<=1).sum());eligible+=len(a);nights+=1
        if eligible:tp2.append({"feature_set":"capture_clock_prior","estimator":f"cyclic_kde_bw{bw}","feature_count":1,"nights_scored":nights,"eligible":eligible,"peak_hour_hits":hits,"peak_hour_rate":hits/eligible,"within_1h_hits":hits1,"within_1h_rate":hits1/eligible,"selection_score":2*hits/eligible+hits1/eligible})
    tres=pd.DataFrame(tp2).sort_values(["selection_score","peak_hour_hits","within_1h_hits"],ascending=False).drop_duplicates(["feature_set","estimator"])
    p1.to_csv(reports/"model_tournament_spatial_feature_stage.csv",index=False);sres.to_csv(reports/"model_tournament_spatial.csv",index=False);tp1df.to_csv(reports/"model_tournament_temporal_feature_stage.csv",index=False);tres.to_csv(reports/"model_tournament_temporal.csv",index=False)
    summary={"status":"ok","method":"two-stage leakage-safe tournament: feature-set screen with logistic C=.2, then estimator tournament on top 3 sets","spatial":{"feature_sets_screened":len(p1),"finalists":top_s,"best":sres.iloc[0].to_dict()},"temporal":{"feature_sets_screened":len(tp1df),"finalists":top_t,"best":tres.iloc[0].to_dict(),"note":"time scoring uses actual GPX survey nights only; May-Jul reconstructed routes never become time negatives"}}
    (reports/"model_tournament_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");print(summary)

if __name__=="__main__":main()
