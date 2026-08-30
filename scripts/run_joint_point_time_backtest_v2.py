from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import _as_jst,_estimator,_fit,_spatial_feature_sets
from habuai.hardening.spatial_history import add_historical_spatial_features


def hav(lat,lon,lats,lons):
    R=6371000.;p1=np.radians(lat);p2=np.radians(lats);dp=p2-p1;dl=np.radians(lons-lon);a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(a))

def curve24(past,cutoff,lat,lon,radius,bw,tau=None):
    c=past[(past.species=="ハブ")&(past.event_type=="捕獲")&past.lat.notna()&past.lon.notna()].copy()
    if c.empty:return np.ones(24,dtype=float)
    dist=hav(float(lat),float(lon),c.lat.to_numpy(float),c.lon.to_numpy(float));local=dist<=radius;h=c.timestamp.dt.hour.to_numpy(float)+c.timestamp.dt.minute.to_numpy(float)/60;w=np.where(local,1.,.05) if int(local.sum())>=2 else np.ones(len(c))
    if tau is not None:
        age=(cutoff-c.timestamp).dt.total_seconds().to_numpy(float)/86400.;w*=np.exp(-np.maximum(age,0)/tau)
    out=[]
    for hr in range(24):
        d=np.minimum(np.abs(h-hr),24-np.abs(h-hr));out.append(float((w*np.exp(-.5*(d/bw)**2)).sum()))
    arr=np.asarray(out,float);mx=float(arr.max());return arr/mx if mx>0 else np.ones(24,dtype=float)

def main():
    root=Path(__file__).resolve().parents[1];cfg=pipeline.load_config(root);p=root/"data"/"processed";r=root/"reports";tour=json.loads((r/"model_tournament_summary.json").read_text(encoding="utf-8"));pt=json.loads((r/"point_conditioned_time_summary.json").read_text(encoding="utf-8"));sw=tour["spatial"]["best"];tw=pt["best"]
    recon=pd.read_csv(p/"reconstructed_spatial_learning_2026-05_07.csv",low_memory=False);md=pd.read_csv(p/"learning_10m_road.csv",low_memory=False);ev=pd.read_csv(p/"events_matched.csv",low_memory=False);segs=gpd.read_file(p/"road_segments_10m.geojson");segs=segs.to_crs("EPSG:6669")
    md["entered_at"]=_as_jst(md.entered_at);ev["timestamp"]=_as_jst(ev.timestamp);ev["lat"]=pd.to_numeric(ev.lat,errors="coerce");ev["lon"]=pd.to_numeric(ev.lon,errors="coerce");md["night"]=(md.entered_at-pd.Timedelta(hours=7)).dt.date.astype(str);gpx=md[md.learning_row_source=="gpx_visit"].copy()
    static=[c for c in ["length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"] if c in md]
    agg={c:(c,"first") for c in static};agg["spatial_label"]=("habu_capture","max");aug=md.groupby(["night","segment_id"],as_index=False).agg(**agg);aug["sample_weight"]=1.;aug["reconstruction_confidence"]="actual_august";aug=aug.drop(columns=[c for c in aug if c.startswith("hist_") or c.startswith("days_since_capture_") or c in {"segment_x_m","segment_y_m"}],errors="ignore");aug,_=add_historical_spatial_features(aug,ev,segs,cfg,root=None)
    combined=pd.concat([recon,aug],ignore_index=True,sort=False);fs=str(sw["feature_set"]);est=str(sw["estimator"]);feats=_spatial_feature_sets(combined)[fs]
    coord=segs[["segment_id","geometry"]].copy().to_crs("EPSG:4326");coord["lat"]=coord.geometry.apply(lambda z:z.centroid.y);coord["lon"]=coord.geometry.apply(lambda z:z.centroid.x);coord=coord.drop_duplicates("segment_id").set_index("segment_id")[["lat","lon"]]
    radius=float(tw["radius_m"]);bw=float(tw["bandwidth_h"]);tau=tw.get("recency_tau_days");tau=None if tau is None or (isinstance(tau,float) and np.isnan(tau)) else float(tau);variants=[(1.,1.),(2.,1.),(1.,2.),(.5,1.),(1.,.5)];rows=[]
    for alpha,beta in variants:
        hits={50:0,100:0,250:0};eligible=exact=top10hit=nsc=0
        for night in sorted(gpx.night.unique()):
            start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=7);end=start+pd.Timedelta(days=1);train=combined[combined.night.astype(str)<night];vis=gpx[(gpx.entered_at>=start)&(gpx.entered_at<end)].copy();actual=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&(ev.timestamp>=start)&(ev.timestamp<end)&ev.lat.notna()&ev.lon.notna()].copy()
            if vis.empty or actual.empty or train.spatial_label.nunique()<2:continue
            m=_estimator(est);_fit(m,train.reindex(columns=feats).replace([np.inf,-np.inf],np.nan),train.spatial_label.astype(int),pd.to_numeric(train.sample_weight,errors="coerce").fillna(1.).to_numpy());ts=aug[aug.night.astype(str)==night].copy();ts["spatial_p"]=m.predict_proba(ts.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];vis["spatial_p"]=vis.segment_id.map(ts.set_index("segment_id").spatial_p).fillna(0.)
            past=ev[ev.timestamp<start];curves={}
            for sid in vis.segment_id.dropna().unique():
                if sid not in coord.index:continue
                la,lo=coord.loc[sid,["lat","lon"]];curves[sid]=curve24(past,start,la,lo,radius,bw,tau)
            vis["time_p"]=[float(curves.get(sid,np.ones(24))[int(t.hour)]) for sid,t in zip(vis.segment_id,vis.entered_at)];vis["joint"]=(np.clip(vis.spatial_p,1e-9,1)**alpha)*(np.clip(vis.time_p,1e-9,1)**beta);vis=vis.sort_values("joint",ascending=False).reset_index(drop=True);vis["rank"]=np.arange(1,len(vis)+1);vis["pct"]=vis["rank"]/len(vis);top=vis.head(30).merge(coord,left_on="segment_id",right_index=True,how="left");top10=vis[vis.pct<=.10].merge(coord,left_on="segment_id",right_index=True,how="left")
            for a in actual.itertuples():
                eligible+=1;ah=int(a.timestamp.hour);dt=np.minimum((top.entered_at.dt.hour.to_numpy()-ah)%24,(ah-top.entered_at.dt.hour.to_numpy())%24);dist=hav(a.lat,a.lon,top.lat.to_numpy(float),top.lon.to_numpy(float));
                if getattr(a,"segment_id",None) is not None:exact+=int(((top.segment_id.astype(str)==str(a.segment_id))&(dt<=1)).any())
                for rad in hits:hits[rad]+=int(((dist<=rad)&(dt<=1)).any())
                if not top10.empty:
                    d10=hav(a.lat,a.lon,top10.lat.to_numpy(float),top10.lon.to_numpy(float));t10=np.minimum((top10.entered_at.dt.hour.to_numpy()-ah)%24,(ah-top10.entered_at.dt.hour.to_numpy())%24);top10hit+=int(((d10<=100)&(t10<=1)).any())
            nsc+=1
        if eligible:rows.append({"spatial_alpha":alpha,"time_beta":beta,"nights_scored":nsc,"eligible_gps_captures":eligible,"exact_segment_plusminus1h_top30":exact,"hit_50m_plusminus1h_top30":hits[50],"hit_100m_plusminus1h_top30":hits[100],"hit_250m_plusminus1h_top30":hits[250],"hit_100m_plusminus1h_top10pct":top10hit,"rate_100m_plusminus1h_top30":hits[100]/eligible,"rate_250m_plusminus1h_top30":hits[250]/eligible,"rate_100m_plusminus1h_top10pct":top10hit/eligible})
    out=pd.DataFrame(rows);out["selection_score"]=2*out.rate_100m_plusminus1h_top30+out.rate_250m_plusminus1h_top30+out.rate_100m_plusminus1h_top10pct;out=out.sort_values("selection_score",ascending=False).reset_index(drop=True);out.to_csv(r/"joint_point_time_backtest.csv",index=False);summary={"status":"ok","spatial_winner":sw,"point_time_winner":tw,"best_joint":out.iloc[0].to_dict(),"method":"leakage-safe August actual-GPX route replay; spatial model updates only from prior nights; each candidate road gets a 24h location-conditioned capture-time curve; joint hits reported at 50/100/250m and ±1h"};(r/"joint_point_time_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");print(summary)

if __name__=="__main__":main()
