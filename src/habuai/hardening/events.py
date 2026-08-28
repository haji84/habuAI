from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

SPECIES_KEYS=["ヒメハブ","ハブ","アカマタ","ガラスヒバァ","ガラスヒヴァ","リュウキュウアオヘビ","ヒャン","ネズミ","オットンガエル","イシカワガエル","アマミハナサキガエル","カエル","ヤマシギ","クロウサギ"]

def species_from_text(text:str)->str:
    for k in SPECIES_KEYS:
        if k in text:return k
    return "その他"

def dedupe_events(df:pd.DataFrame)->tuple[pd.DataFrame,int]:
    if df.empty:return df.copy(),0
    x=df.copy(); ts=pd.to_datetime(x.timestamp,errors="coerce",utc=True).astype("string")
    lat=pd.to_numeric(x.lat,errors="coerce").round(7).astype("string"); lon=pd.to_numeric(x.lon,errors="coerce").round(7).astype("string")
    x["event_signature"]=ts+"|"+x.species.astype("string")+"|"+x.event_type.astype("string")+"|"+lat+"|"+lon+"|"+x.raw_text.fillna("").astype("string")
    n=len(x); x=x.drop_duplicates("event_signature",keep="first").reset_index(drop=True); return x,n-len(x)

def join_weather(visits,weather):
    if visits.empty or weather.empty:return visits
    v,w=visits.copy(),weather.copy()
    for c in ["entered_at","exited_at"]:v[c]=pd.to_datetime(v[c],utc=True).dt.tz_convert("Asia/Tokyo")
    w["timestamp"]=pd.to_datetime(w.timestamp,utc=True).dt.tz_convert("Asia/Tokyo")
    return pd.merge_asof(v.sort_values("entered_at"),w.sort_values("timestamp"),left_on="entered_at",right_on="timestamp",direction="nearest",tolerance=pd.Timedelta("40min"))

def slope_features(visits):
    if visits.empty:return visits
    x=visits.sort_values(["session_file","entered_at"]).copy(); prev=x.groupby("session_file").elevation_m.shift(); x["elevation_delta_m"]=x.elevation_m-prev
    x["slope_observed_deg"]=np.degrees(np.arctan2(x.elevation_delta_m.clip(-20,20),10.0)); med=x.groupby("session_file").elevation_m.transform(lambda s:s.rolling(11,center=True,min_periods=3).median()); x["local_relief_proxy_m"]=x.elevation_m-med; return x

def project_events(events,crs):
    if events.empty:return events.copy()
    e=events.dropna(subset=["lat","lon"]).copy()
    if e.empty:return e
    g=gpd.GeoDataFrame(e,geometry=gpd.points_from_xy(e.lon,e.lat),crs="EPSG:4326").to_crs(crs); g["x_m"],g["y_m"]=g.geometry.x,g.geometry.y; return pd.DataFrame(g.drop(columns="geometry"))

def strict_holdout_score(events):
    if events.empty:return {"status":"no-holdout-events"}
    e=events[(events.species=="ハブ")&(events.event_type=="捕獲")].copy(); e["timestamp"]=pd.to_datetime(e.timestamp,utc=True).dt.tz_convert("Asia/Tokyo")
    e=e[(e.timestamp>=pd.Timestamp("2026-08-28T07:00:00+09:00"))&(e.timestamp<pd.Timestamp("2026-08-29T07:00:00+09:00"))]
    if e.empty:return {"status":"no-holdout-events"}
    a,b=pd.Timestamp("2026-08-29T00:20:00+09:00"),pd.Timestamp("2026-08-29T01:20:00+09:00"); c,d=pd.Timestamp("2026-08-29T01:35:00+09:00"),pd.Timestamp("2026-08-29T02:35:00+09:00")
    main=e[(e.timestamp>=a)&(e.timestamp<=b)]; sec=e[(e.timestamp>=c)&(e.timestamp<=d)]; actual=int(pd.to_numeric(e.individual_count,errors="coerce").fillna(1).sum())
    return {"status":"ok","forecast_frozen":True,"night":"2026-08-28_to_2026-08-29","main_window":{"start":a.isoformat(),"end":b.isoformat(),"actual_capture_events":len(main),"hit":bool(len(main))},"secondary_window":{"start":c.isoformat(),"end":d.isoformat(),"actual_capture_events":len(sec),"hit":bool(len(sec))},"forecast_point":3,"forecast_range":[2,4],"actual_capture_events":len(e),"actual_individuals":actual,"range_hit":2<=actual<=4,"point_error":actual-3,"event_times":[t.isoformat() for t in e.timestamp.sort_values()],"note":"Strict windows frozen; 01:22 remains outside the main window."}
