from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _as_jst(s):
    return pd.to_datetime(s,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo")


def _haversine_m(lat,lon,lats,lons):
    R=6371000.0;p1=np.radians(lat);p2=np.radians(lats);dp=p2-p1;dl=np.radians(lons-lon);a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(a))


def _predict(past,cutoff,lat,lon,hours,radius,bw,local_weight=.95,min_local=2,tau_days=None):
    cap=past[(past.species=="ハブ")&(past.event_type=="捕獲")&past.lat.notna()&past.lon.notna()].copy()
    if cap.empty:return None
    dist=_haversine_m(float(lat),float(lon),cap.lat.to_numpy(float),cap.lon.to_numpy(float));local=dist<=radius;h=cap.timestamp.dt.hour.to_numpy(float)+cap.timestamp.dt.minute.to_numpy(float)/60
    w=np.where(local,1.0,1-local_weight) if int(local.sum())>=min_local else np.ones(len(cap))
    if tau_days is not None:
        age=(cutoff-cap.timestamp).dt.total_seconds().to_numpy(float)/86400.0;w*=np.exp(-np.maximum(age,0)/tau_days)
    scores=[]
    for hour in hours:
        d=np.abs(h-hour);d=np.minimum(d,24-d);scores.append(float((w*np.exp(-.5*(d/bw)**2)).sum()))
    return int(hours[int(np.argmax(scores))]) if scores else None


def main():
    root=Path(__file__).resolve().parents[1];p=root/"data"/"processed";r=root/"reports";d=pd.read_csv(p/"learning_10m_road.csv",low_memory=False);e=pd.read_csv(p/"events_matched.csv",low_memory=False);d["entered_at"]=_as_jst(d.entered_at);e["timestamp"]=_as_jst(e.timestamp);e["lat"]=pd.to_numeric(e.lat,errors="coerce");e["lon"]=pd.to_numeric(e.lon,errors="coerce");g=d[d.learning_row_source=="gpx_visit"].copy();g["night"]=(g.entered_at-pd.Timedelta(hours=7)).dt.date.astype(str)
    variants=[]
    for radius in [250,500,1000,2000,3000]:
        for bw in [.75,1.0,1.5,2.0]:
            for tau in [None,120]:
                hits=hits1=eligible=nights=0
                for night in sorted(g.night.unique()):
                    start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=7);end=start+pd.Timedelta(days=1);te=g[(g.entered_at>=start)&(g.entered_at<end)];actual=e[(e.species=="ハブ")&(e.event_type=="捕獲")&(e.timestamp>=start)&(e.timestamp<end)&e.lat.notna()&e.lon.notna()];past=e[e.timestamp<start]
                    if te.empty or actual.empty:continue
                    hours=sorted(te.entered_at.dt.hour.unique().tolist());nights+=1
                    for a in actual.itertuples():
                        peak=_predict(past,start,a.lat,a.lon,hours,radius,bw,.95,2,tau)
                        if peak is None:continue
                        ah=int(a.timestamp.hour);hits+=int(ah==peak);delta=min((ah-peak)%24,(peak-ah)%24);hits1+=int(delta<=1);eligible+=1
                if eligible:
                    score=2*hits/eligible+hits1/eligible;variants.append({"radius_m":radius,"bandwidth_h":bw,"recency_tau_days":tau,"nights_scored":nights,"eligible_gps_captures":eligible,"exact_hour_hits":hits,"exact_hour_rate":hits/eligible,"within_1h_hits":hits1,"within_1h_rate":hits1/eligible,"selection_score":score})
    out=pd.DataFrame(variants).sort_values(["selection_score","exact_hour_hits","within_1h_hits"],ascending=False).reset_index(drop=True);out.to_csv(r/"point_conditioned_time_tournament.csv",index=False);best=out.iloc[0].to_dict() if not out.empty else {};summary={"status":"ok" if best else "insufficient-data","method":"for each candidate point, use only prior capture times; captures within radius receive full weight and global captures a 5% background weight; evaluated only on actual GPX survey nights","best":best,"candidates":len(out),"operational_use":"assign a separate peak hour/window to each spatial candidate point, then combine point score and time score"};(r/"point_conditioned_time_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8");print(summary)

if __name__=="__main__":main()
