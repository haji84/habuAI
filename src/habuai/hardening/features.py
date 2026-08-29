from __future__ import annotations
import geopandas as gpd
import numpy as np
import pandas as pd
from .events import project_events

def add_outcomes_bio(visits,events,segs,cfg):
    x=visits.copy(); x["habu_capture"],x["habu_individuals"]=0,0
    habu=events[(events.species=="ハブ")&(events.event_type=="捕獲")] if not events.empty else pd.DataFrame()
    for r in habu.itertuples():
        if pd.isna(getattr(r,"segment_id",None)):continue
        m=(x.segment_id==r.segment_id)&(abs((x.entered_at-r.timestamp).dt.total_seconds())<=600); x.loc[m,"habu_capture"]=1; x.loc[m,"habu_individuals"]+=int(r.individual_count or 1)
    c=segs[["segment_id","geometry"]].copy(); c["segment_x_m"],c["segment_y_m"]=c.geometry.centroid.x,c.geometry.centroid.y; x=x.merge(c[["segment_id","segment_x_m","segment_y_m"]],on="segment_id",how="left")
    excluded={"ハブ","ヒメハブ","アカマタ","ガラスヒバァ","ガラスヒヴァ","リュウキュウアオヘビ","ヒャン"}; bio=project_events(events[~events.species.isin(excluded)],segs.crs) if not events.empty else pd.DataFrame()
    windows=[int(v) for v in cfg.get("biological_windows_minutes",[5,10,15,30])]; distances=[float(v) for v in cfg.get("biological_distance_m",[50,100,250,500])]
    t=pd.to_datetime(x.entered_at,utc=True).astype("int64").to_numpy(); vx=x.segment_x_m.to_numpy(float); vy=x.segment_y_m.to_numpy(float)
    feature_cols={}
    for sp in ["ネズミ","オットンガエル","カエル","ヤマシギ","クロウサギ"]:
        b=bio[bio.species==sp].copy() if not bio.empty else pd.DataFrame()
        if not b.empty:b["u"]=pd.to_datetime(b.timestamp,utc=True); b=b.sort_values("u"); bt=b.u.astype("int64").to_numpy(); bx=b.x_m.to_numpy(float); by=b.y_m.to_numpy(float); bc=pd.to_numeric(b.individual_count,errors="coerce").fillna(1).to_numpy(float)
        else:bt=np.array([],dtype=np.int64); bx=by=bc=np.array([],dtype=float)
        for mins in windows:
            cnt={d:np.zeros(len(x),dtype=np.int32) for d in distances}; nearest=np.full(len(x),np.nan); delta=int(pd.Timedelta(minutes=mins).value)
            for i,now in enumerate(t):
                if not len(bt) or not np.isfinite(vx[i]):continue
                lo=np.searchsorted(bt,now-delta,"left"); hi=np.searchsorted(bt,now,"right")
                if hi<=lo:continue
                dd=np.hypot(bx[lo:hi]-vx[i],by[lo:hi]-vy[i]); nearest[i]=float(dd.min())
                for d in distances:
                    mm=dd<=d
                    if mm.any():cnt[d][i]=int(bc[lo:hi][mm].sum())
            base=f"bio_{sp}_{mins}m"; feature_cols[base]=cnt[max(distances)]; feature_cols[f"{base}_nearest_m"]=nearest
            for d in distances:feature_cols[f"{base}_{int(d)}m_count"]=cnt[d]
    if feature_cols:
        x=pd.concat([x,pd.DataFrame(feature_cols,index=x.index)],axis=1)
    return x

def _nearest(points,target):
    """Return exactly one nearest distance per input point.

    geopandas.sjoin_nearest can emit multiple rows for one source point when
    several target geometries are tied at the same minimum distance.  Reduce
    those ties back to the source index so callers always receive len(points)
    values in the original order.
    """
    out=np.full(len(points),np.nan,dtype=float)
    if target is None or target.empty:return out
    target=target[["geometry"]].dropna().copy()
    target=target[~target.geometry.is_empty]
    if target.empty:return out
    src=points[["geometry"]].copy().reset_index(drop=True)
    src["_source_row"]=np.arange(len(src),dtype=int)
    j=gpd.sjoin_nearest(src,target,how="left",distance_col="d")
    d=pd.to_numeric(j["d"],errors="coerce")
    reduced=pd.DataFrame({"_source_row":j["_source_row"].to_numpy(),"d":d.to_numpy()}).groupby("_source_row",sort=False)["d"].min()
    idx=reduced.index.to_numpy(dtype=int)
    out[idx]=reduced.to_numpy(dtype=float)
    return out

def add_static_context(data,segs,root):
    s=segs[["segment_id","highway","length_m","bearing_deg","curvature_deg","geometry"]].copy(); s["road_class_code"]=s.highway.astype("category").cat.codes.astype(float); p=gpd.GeoDataFrame(s[["segment_id"]].copy(),geometry=s.geometry.centroid,crs=segs.crs)
    ep=[]
    for g in segs.geometry:
        if g is None or g.is_empty:continue
        a,b=list(g.coords)[0],list(g.coords)[-1]; ep += [(round(a[0],1),round(a[1],1)),(round(b[0],1),round(b[1],1))]
    e=pd.DataFrame(ep,columns=["x","y"]); j=e.value_counts().reset_index(name="degree"); j=j[j.degree>=3]; jg=gpd.GeoDataFrame(j,geometry=gpd.points_from_xy(j.x,j.y),crs=segs.crs) if not j.empty else None; s["junction_distance_m"]=_nearest(p,jg)
    path=root/"data"/"osm"/"setouchi-context.geojson"; ctx=None
    if path.exists():
        try:ctx=gpd.read_file(path); ctx=ctx.set_crs("EPSG:4326") if ctx.crs is None else ctx; ctx=ctx.to_crs(segs.crs)
        except Exception:ctx=None
    for col in ["stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"]:s[col]=np.nan
    if ctx is not None and not ctx.empty:
        tag=lambda k:ctx[k].astype("string").fillna("") if k in ctx else pd.Series("",index=ctx.index,dtype="string"); w,n,l=tag("waterway"),tag("natural"),tag("landuse")
        s["stream_distance_m"]=_nearest(p,ctx[w!=""]); s["coast_distance_m"]=_nearest(p,ctx[n=="coastline"]); s["forest_distance_m"]=_nearest(p,ctx[(n=="wood")|l.isin(["forest","orchard"])]); s["farmland_distance_m"]=_nearest(p,ctx[l.isin(["farmland","farmyard","meadow"])]); s["residential_distance_m"]=_nearest(p,ctx[l=="residential"])
    cols=["segment_id","highway","length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"]; return data.merge(pd.DataFrame(s[cols]),on="segment_id",how="left")
