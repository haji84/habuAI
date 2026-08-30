from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RADII_M=(50,100,250,500)
BIO_SPECIES=("ネズミ","オットンガエル","カエル","ヤマシギ","クロウサギ")


def _night_cutoff(night:str)->pd.Timestamp:
    # Operational night YYYY-MM-DD starts after the 07:00 rollover on that date.
    return pd.Timestamp(str(night),tz="Asia/Tokyo")+pd.Timedelta(hours=7)


def _project_events(events:pd.DataFrame,crs)->gpd.GeoDataFrame:
    e=events.dropna(subset=["lat","lon","timestamp"]).copy()
    if e.empty:return gpd.GeoDataFrame(e,geometry=[],crs=crs)
    e["timestamp"]=pd.to_datetime(e.timestamp,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo")
    g=gpd.GeoDataFrame(e,geometry=gpd.points_from_xy(pd.to_numeric(e.lon,errors="coerce"),pd.to_numeric(e.lat,errors="coerce")),crs="EPSG:4326")
    return g.to_crs(crs)


def _explicit_female_mask(e:pd.DataFrame)->pd.Series:
    # Never infer sex from size. Only explicit structured/text evidence is accepted.
    out=pd.Series(False,index=e.index)
    for c in ("sex","gender"):
        if c in e.columns:
            s=e[c].astype("string").fillna("")
            out|=s.isin(["メス","♀","雌","female","F"])
    if "raw_text" in e.columns:
        s=e.raw_text.astype("string").fillna("")
        out|=s.str.contains(r"メス|♀|雌|female",case=False,regex=True)
    return out


def _counts_within(dx:np.ndarray,dy:np.ndarray,weights:np.ndarray,radius:float)->np.ndarray:
    if len(dx)==0:return np.zeros(dx.shape[0] if dx.ndim else 0,dtype=float)
    return weights[(dx*dx+dy*dy)<=radius*radius].sum(axis=1)


def add_historical_spatial_features(data:pd.DataFrame,events:pd.DataFrame,segs,cfg:dict,root:Path|None=None)->tuple[pd.DataFrame,dict]:
    """Attach point-in-time spatial history using only evidence before each night's 07:00 cutoff.

    This is intentionally night-level rather than event-time-level because reconstructed May-July
    rows have no fabricated clock time. The same pre-night feature snapshot is used for every road
    candidate on a given operational night.
    """
    if data.empty:
        return data.copy(),{"status":"empty","feature_count":0}
    x=data.copy()
    s=segs[["segment_id","geometry"]].dropna(subset=["segment_id","geometry"]).copy()
    s["segment_x_m"]=s.geometry.centroid.x;s["segment_y_m"]=s.geometry.centroid.y
    x=x.merge(s[["segment_id","segment_x_m","segment_y_m"]],on="segment_id",how="left")
    ge=_project_events(events,segs.crs)
    if ge.empty:
        return x,{"status":"no-events","feature_count":0}
    ge["x_m"]=ge.geometry.x;ge["y_m"]=ge.geometry.y
    explicit_female=_explicit_female_mask(ge)
    ge["explicit_female"]=explicit_female.to_numpy(bool)
    ge["is_large"]=(ge.get("size",pd.Series(index=ge.index,dtype=object)).astype("string").fillna("")=="大")
    ge["count_w"]=pd.to_numeric(ge.get("individual_count",1),errors="coerce").fillna(1).clip(lower=1)
    feature_cols=[]
    for r in RADII_M:
        feature_cols += [f"hist_capture_count_{r}m",f"hist_capture_30d_count_{r}m",f"hist_capture_90d_count_{r}m",f"days_since_capture_{r}m",f"hist_large_capture_count_{r}m"]
        feature_cols += [f"hist_bio_count_{r}m",f"hist_bio_30d_count_{r}m"]
        for sp in BIO_SPECIES:feature_cols.append(f"hist_bio_{sp}_30d_count_{r}m")
    for c in feature_cols:x[c]=0.0
    for r in RADII_M:x[f"days_since_capture_{r}m"]=np.nan
    x["large_female_history_available"]=1.0 if bool(ge.explicit_female.any()) else 0.0

    for night,idx in x.groupby(x.night.astype(str),sort=True).groups.items():
        cutoff=_night_cutoff(night);past=ge[ge.timestamp<cutoff].copy();rows=np.asarray(list(idx),dtype=int)
        if past.empty:continue
        px=x.loc[rows,"segment_x_m"].to_numpy(float);py=x.loc[rows,"segment_y_m"].to_numpy(float)
        valid=np.isfinite(px)&np.isfinite(py)
        if not valid.any():continue
        p=past
        ex=p.x_m.to_numpy(float);ey=p.y_m.to_numpy(float);w=p.count_w.to_numpy(float)
        # Distance matrix is bounded by one reconstructed night x prior events, not all 31k x all events.
        d2=(px[:,None]-ex[None,:])**2+(py[:,None]-ey[None,:])**2
        age_days=(cutoff-p.timestamp).dt.total_seconds().to_numpy(float)/86400.0
        is_cap=((p.species=="ハブ")&(p.event_type=="捕獲")).to_numpy(bool)
        is_large=(is_cap&p.is_large.to_numpy(bool))
        excluded={"ハブ","ヒメハブ","アカマタ","ガラスヒバァ","ガラスヒヴァ","リュウキュウアオヘビ","ヒャン"}
        is_bio=(~p.species.isin(excluded)).to_numpy(bool)
        for r in RADII_M:
            near=d2<=float(r*r)
            def cnt(mask):return (near[:,mask]*w[mask][None,:]).sum(axis=1) if mask.any() else np.zeros(len(rows))
            x.loc[rows,f"hist_capture_count_{r}m"]=cnt(is_cap)
            x.loc[rows,f"hist_capture_30d_count_{r}m"]=cnt(is_cap&(age_days<=30))
            x.loc[rows,f"hist_capture_90d_count_{r}m"]=cnt(is_cap&(age_days<=90))
            x.loc[rows,f"hist_large_capture_count_{r}m"]=cnt(is_large)
            x.loc[rows,f"hist_bio_count_{r}m"]=cnt(is_bio)
            x.loc[rows,f"hist_bio_30d_count_{r}m"]=cnt(is_bio&(age_days<=30))
            cap_idx=np.where(is_cap)[0]
            if len(cap_idx):
                near_cap=near[:,cap_idx]
                ages=age_days[cap_idx]
                last=np.where(near_cap,ages[None,:],np.inf).min(axis=1);last[~np.isfinite(last)]=np.nan
                x.loc[rows,f"days_since_capture_{r}m"]=last
            for sp in BIO_SPECIES:
                m=(p.species.astype(str).to_numpy()==sp)&(age_days<=30)
                x.loc[rows,f"hist_bio_{sp}_30d_count_{r}m"]=cnt(m)
    x[feature_cols]=x[feature_cols].replace([np.inf,-np.inf],np.nan)
    audit={
        "status":"ok","cutoff_rule":"for operational night YYYY-MM-DD, only events strictly before YYYY-MM-DD 07:00 Asia/Tokyo are visible",
        "radii_m":list(RADII_M),"bio_species":list(BIO_SPECIES),"feature_count":len(feature_cols)+1,
        "history_features":feature_cols+['large_female_history_available'],
        "large_female_history_available":bool(ge.explicit_female.any()),
        "large_female_note":"sex is never inferred from size; current master has no explicit female observations, so large-female counts are not model features until explicit sex data exists",
    }
    if root is not None:
        p=root/"reports";p.mkdir(parents=True,exist_ok=True);(p/"historical_spatial_feature_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return x,audit
