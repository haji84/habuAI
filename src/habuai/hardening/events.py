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
    """Remove exact duplicate events without collapsing rows that contain missing values.

    Pandas nullable string concatenation propagates NA.  The previous implementation
    therefore assigned the same NA signature to every event with missing GPS and
    silently collapsed distinct historical captures.  Every signature component is
    now normalized to a concrete sentinel before concatenation.  raw_text/canonical_id
    remains part of the signature, so distinct canonical events are preserved.
    """
    if df.empty:return df.copy(),0
    x=df.copy()
    ts=pd.to_datetime(x.get("timestamp"),errors="coerce",utc=True).astype("string").fillna("<missing_timestamp>")
    species=x.get("species",pd.Series(index=x.index,dtype="object")).astype("string").fillna("<missing_species>")
    event_type=x.get("event_type",pd.Series(index=x.index,dtype="object")).astype("string").fillna("<missing_event_type>")
    lat=pd.to_numeric(x.get("lat"),errors="coerce").round(7).astype("string").fillna("<missing_lat>")
    lon=pd.to_numeric(x.get("lon"),errors="coerce").round(7).astype("string").fillna("<missing_lon>")
    raw=x.get("raw_text",pd.Series(index=x.index,dtype="object")).astype("string").fillna("")
    canonical=x.get("canonical_id",pd.Series(index=x.index,dtype="object")).astype("string").fillna("")
    x["event_signature"]=ts+"|"+species+"|"+event_type+"|"+lat+"|"+lon+"|"+canonical+"|"+raw
    n=len(x); x=x.drop_duplicates("event_signature",keep="first").reset_index(drop=True); return x,n-len(x)

def collapse_nearest_event_matches(matched:pd.DataFrame)->pd.DataFrame:
    """Collapse only rows duplicated by a tied nearest-road join.

    GeoPandas sjoin_nearest keeps the source index and may return more than
    one row for one source event when several road geometries are tied at the
    same minimum distance. Distinct source events are never deduplicated here.
    """
    if matched.empty:return matched
    x=matched.copy(); x["_source_row"]=x.index.to_numpy()
    if not x["_source_row"].duplicated().any():return x.drop(columns="_source_row").reset_index(drop=True)
    x["_distance_sort"]=pd.to_numeric(x.get("event_match_distance_m"),errors="coerce").fillna(np.inf)
    x["_segment_sort"]=x.get("segment_id",pd.Series(index=x.index,dtype="object")).fillna("").astype(str)
    x=x.sort_values(["_source_row","_distance_sort","_segment_sort"],kind="stable")
    x=x.drop_duplicates("_source_row",keep="first")
    return x.drop(columns=["_source_row","_distance_sort","_segment_sort"]).reset_index(drop=True)

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
