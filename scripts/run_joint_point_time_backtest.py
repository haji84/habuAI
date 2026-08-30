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

def time_density(past,cutoff,lat,lon,hours,radius,bw,tau=None):
    c=past[(past.species=="ハブ")&(past.event_type=="捕獲")&past.lat.notna()&past.lon.notna()].copy()
    if c.empty:return np.ones(len(hours),dtype=float)
    dist=hav(float(lat),float(lon),c.lat.to_numpy(float),c.lon.to_numpy(float));local=dist<=radius;h=c.timestamp.dt.hour.to_numpy(float)+c.timestamp.dt.minute.to_numpy(float)/60;w=np.where(local,1.,.05) if int(local.sum())>=2 else np.ones(len(c))
    if tau is not None:
        age=(cutoff-c.timestamp).dt.total_seconds().to_numpy(float)/86400.;w*=np.exp(-np.maximum(age,0)/tau)
    out=[]
    for hr in hours:
        d=np.minimum(np.abs(h-hr),24-np.abs(h-hr));out.append(float((w*np.exp(-.5*(d/bw)**2)).sum()))
    arr=np.asarray(out,float);return arr/(arr.max() if arr.max()>0 else 1.)

def main():
    root=Path(__file__).resolve().parents[1];cfg=pipeline.load_config(root);p=root/"data"/"processed";r=root/"reports"
    tour=json.loads((r/"model_tournament_summary.json").read_text(encoding="utf-8"));pt=json.loads((r/"point_conditioned_time_summary.json").read_text(encoding="utf-8"));spatial_winner=tour["spatial"]["best"];time_winner=pt["best"]
    recon=pd.read_csv(p/"reconstructed_spatial_learning_2026-05_07.csv",low_memory=False);md=pd.read_csv(p/"learning_10m_road.csv",low_memory=False);ev=pd.read_csv(p/"events_matched.csv",low_memory=False);segs=gpd.read_file(p/"road_segments_10m.geojson");segs=segs.to_crs("EPSG:6669") if str(segs.crs)!="EPSG:6669" else segs
    md["entered_at"]=_as_jst(md.entered_at);ev["timestamp"]=_as_jst(ev.timestamp);ev["lat"]=pd.to_numeric(ev.lat,errors="coerce");ev["lon"]=pd.to_numeric(ev.lon,errors="coerce")
    md["night"]=(md.entered_at-pd.Timedelta(hours=7)).dt.date.astype(str);gpx=md[md.learning_row_source=="gpx_visit"].copy()
    # One spatial observation per segment/night. Anchors make road-matched captures positive without inventing exposure.
    static_cols=[c for c in ["length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"] if c in md]
    aug=md.groupby(["night","segment_id"],as_index=False).agg(**{c:(c,"first") for c in static_cols},spatial_label=("habu_capture","max"));aug["sample_weight"]=1.0;aug["reconstruction_confidence"]="actual_august"
    drop=[c for c in aug if c.startswith("hist_") or c.startswith("days_since_capture_") or c in {"segment_x_m","segment_y_m"}];aug=aug.drop(columns=drop,errors="ignore");aug,_=add_historical_spatial_features(aug,ev,segs,cfg,root=None)
    combined=pd.concat([recon,aug],ignore_index=True,sort=False);fs_name=str(spatial_winner["feature_set"]);est_name=str(spatial_winner["estimator"]);feats=_spatial_feature_sets(combined)[fs_name]
    cent=segs[["segment_id","geometry"]].copy();cent["x"]=cent.geometry.centroid.x;cent["y"]=cent.geometry.centroid.y;cent=cent.to_crs("EPSG:4326");cent["lat"]=cent.geometry.centroid.y;cent["lon"]=cent.geometry.centroid.x;coord=cent.set_index("segment_id")[["lat","lon"]]
    radius=float(time_winner["radius_m"]);bw=float(time_winner["bandwidth_h"]);tau=time_winner.get("recency_tau_days");tau=None if tau is None or (isinstance(tau,float) and np.isnan(tau)) else float(tau)
    variants=[(1.,1.),(2.,1.),(1.,2.),(.5,1.),(1.,.5)];rows=[]
    for alpha,beta in variants:
        hits={50:0,100:0,250:0};eligible=0;exact_seg_time=0;top10_joint=0;nights_scored=0
        for night in sorted(gpx.night.unique()):
            start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=7);end=start+pd.Timedelta(days=1);train=combined[combined.night.astype(str)<night].copy();test_vis=gpx[(gpx.entered_at>=start)&(gpx.entered_at<end)].copy();actual=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&(ev.timestamp>=start)&(ev.timestamp<end)&ev.lat.notna()&ev.lon.notna()].copy()
            if test_vis.empty or actual.empty or train.spatial_label.nunique()<2:continue
            m=_estimator(est_name);_fit(m,train.reindex(columns=feats).replace([np.inf,-np.inf],np.nan),train.spatial_label.astype(int),pd.to_numeric(train.sample_weight,errors="coerce").fillna(1.).to_numpy())
            test_seg=aug[aug.night.astype(str)==night].copy();test_seg["spatial_p"]=m.predict_proba(test_seg.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];smap=test_seg.set_index("segment_id").spatial_p.to_dict();test_vis["spatial_p"]=test_vis.segment_id.map(smap).fillna(0.)
            past=ev[ev.timestamp<start];td=[]
            for rr in test_vis.itertuples():
                if rr.segment_id not in coord.index:td.append(0.);continue
                la,lo=coord.loc[rr.segment_id,["lat","lon"]];td.append(float(time_density(past,start,la,lo,[int(rr.entered_at.hour)],radius,bw,tau)[0]))
            test_vis["time_p"]=td;test_vis["joint"]=(np.clip(test_vis.spatial_p,1e-9,1)**alpha)*(np.clip(test_vis.time_p,1e-9,1)**beta);test_vis=test_vis.sort_values("joint",ascending=False).reset_index(drop=True);test_vis["rank"]=np.arange(1,len(test_vis)+1);test_vis["pct"]=test_vis["rank"]/len(test_vis);top=test_vis.head(30);top10=test_vis[test_vis.pct<=.10]
            tc=top.merge(coord,left_on="segment_id",right_index=True,how="left");t10=top10.merge(coord,left_on="segment_id",right_index=True,how="left")
            for a in actual.itertuples():
                eligible+=1;ah=int(a.timestamp.hour)
                if getattr(a,"segment_id",None) is not None:
                    exact=((top.segment_id.astype(str)==str(a.segment_id))&(np.minimum((top.entered_at.dt.hour-ah)%24,(ah-top.entered_at.dt.hour)%24)<=1)).any();exact_seg_time+=int(exact)
                if not t10.empty:
                    dist=hav(a.lat,a.lon,t10.lat.to_numpy(float),t10.lon.to_numpy(float));dt=np.minimum((t10.entered_at.dt.hour.to_numpy()-ah)%24,(ah-t10.entered_at.dt.hour.to_numpy())%24);top10_joint+=int(((dist<=100)&(dt<=1)).any())
                if not tc.empty:
                    dist=hav(a.lat,a.lon,tc.lat.to_numpy(float),tc.lon.to_numpy(float));dt=np.minimum((tc.entered_at.dt.hour.to_numpy()-ah)%24,(ah-tc.entered_at.dt.hour.to_numpy())%24)
                    for rad in hits:hits[rad]+=int(((dist<=rad)&(dt<=1)).any())
            nights_scored+=1
        if eligible:rows.append({"spatial_alpha":alpha,"time_beta":beta,"nights_scored":nights_scored,"eligible_gps_captures":eligible,"exact_segment_plusminus1h_top30":exact_seg_time,"hit_50m_plusminus1h_top30":hits[50],"hit_100m_plusminus1h_top30":hits[100],"hit_250m_plusminus1h_top30":hits[250],"hit_100m_plusminus1h_top10pct":top10_joint,"rate_100m_plusminus1h_top30":hits[100]/eligible,"rate_250m_plusminus1h_top30":hits[250]/eligible,"rate_100m_plusminus1h_top10pct":top10_joint/eligible})
    out=pd.DataFrame(rows);out["selection_score"]=2*out.rate_100m_plusminus1h_top30+out.rate_250m_plusminus1h_top30+out.rate_100m_plusminus1h_top10pct;out=out.sort_values("selection_score",ascending=False).reset_index(drop=True);out.to_csv(r/"joint_point_time_backtest.csv",index=False);summary={"status":"ok","spatial_winner":spatial_winner,"point_time_winner":time_winner,"best_joint":out.iloc[0].to_dict(),"method":"August actual-GPX route replay; every night trains only on earlier nights plus May-Jul reconstructed spatial evidence; joint candidates are actual visited segment-hours; 50/100/250m spatial tolerance and ±1h time tolerance reported"};(r/"joint_point_time_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");print(summary)

if __name__=="__main__":main()
