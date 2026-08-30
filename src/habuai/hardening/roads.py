from __future__ import annotations
import hashlib
import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring


def _empty(crs):
    return gpd.GeoDataFrame(columns=["segment_id","way_id","part_index","segment_index","length_m","highway","name","road_source","geometry"],geometry="geometry",crs=crs)


def _point_metric(lon,lat,crs):
    return gpd.GeoSeries(gpd.points_from_xy([lon],[lat]),crs="EPSG:4326").to_crs(crs).iloc[0]


def _event_hits(ev,track,crs,spatial_m=25.0,time_pad_min=20.0):
    if pd.isna(ev.get("lat")) or pd.isna(ev.get("lon")) or pd.isna(ev.get("timestamp")):return False
    pt=_point_metric(ev.lon,ev.lat,crs);t=pd.Timestamp(ev.timestamp)
    if t.tzinfo is None:t=t.tz_localize("Asia/Tokyo")
    delta=(pd.to_datetime(track.timestamp)-t).abs().dt.total_seconds();near=track.loc[delta<=time_pad_min*60]
    return (not near.empty) and float(near.geometry.distance(pt).min())<=spatial_m


def _split(line,road_id,n=10.0):
    out=[]
    for i in range(max(1,math.ceil(line.length/n))):
        seg=substring(line,i*n,min((i+1)*n,line.length))
        if not seg.is_empty and seg.length>0:out.append({"segment_id":f"{road_id}_{i}","way_id":road_id,"part_index":0,"segment_index":i,"length_m":float(seg.length),"highway":"track","name":"GPX-confirmed supplemental road","road_source":"gpx_recovered","geometry":seg})
    return out


def _stable_road_id(line):
    coords=[(round(float(x),1),round(float(y),1)) for x,y,*_ in line.coords]
    rev=list(reversed(coords));canonical=min(coords,rev)
    digest=hashlib.sha1(repr(canonical).encode("utf-8")).hexdigest()[:12]
    return f"GPXROAD_{digest}"


def _step_quality(block,max_gap_m):
    if len(block)<2:return False,0.0
    xy=np.array([(p.x,p.y) for p in block.geometry],dtype=float)
    gaps=np.hypot(np.diff(xy[:,0]),np.diff(xy[:,1]))
    max_gap=float(np.nanmax(gaps)) if len(gaps) else 0.0
    return bool(np.isfinite(max_gap) and max_gap<=max_gap_m),max_gap


def _connected_to_osm(line,osm_segments,connect_m):
    if osm_segments.empty:return False
    endpoints=[gpd.GeoSeries([line.interpolate(0)],crs=osm_segments.crs),gpd.GeoSeries([line.interpolate(line.length)],crs=osm_segments.crs)]
    for ep in endpoints:
        pt=ep.iloc[0]
        idx=list(osm_segments.sindex.query(pt.buffer(connect_m),predicate="intersects"))
        if idx and float(osm_segments.iloc[idx].geometry.distance(pt).min())<=connect_m:return True
    return False


def recover_supplemental_roads(points,osm_matches,events_osm_matched,osm_segments,cfg,root:Path|None=None):
    settings=cfg.get("supplemental_roads",{})
    if settings.get("enabled",True) is False or points.empty or osm_matches.empty:return _empty(osm_segments.crs),[]
    crs=osm_segments.crs
    p=gpd.GeoDataFrame(osm_matches.copy(),geometry=gpd.points_from_xy(osm_matches.lon,osm_matches.lat),crs="EPSG:4326").to_crs(crs)
    events=events_osm_matched.copy() if events_osm_matched is not None else pd.DataFrame()
    unmatched_events=events[events.segment_id.isna()].copy() if (not events.empty and "segment_id" in events) else pd.DataFrame()
    min_points=int(settings.get("min_unmatched_points",20));min_length=float(settings.get("min_length_m",30.0));max_gap=float(settings.get("max_point_gap_m",45.0));connect_m=float(settings.get("endpoint_connect_m",50.0));duplicate_m=float(settings.get("duplicate_hausdorff_m",20.0));event_m=float(settings.get("event_corroboration_m",25.0))
    recovered=[];audit=[];canonical_lines=[]
    for session_file,g in p.groupby("session_file",sort=False):
        g=g.sort_values("seq").copy();mask=g.segment_id.isna().to_numpy();start=0
        while start<len(g):
            if not mask[start]:start+=1;continue
            end=start
            while end+1<len(g) and mask[end+1]:end+=1
            block=g.iloc[start:end+1].copy();start=end+1
            if len(block)<min_points:continue
            quality,max_observed_gap=_step_quality(block,max_gap)
            if not quality:continue
            entry=block.geometry.iloc[0];exit_pt=block.geometry.iloc[-1];used=block
            if entry.distance(exit_pt)<=100:
                dist=block.geometry.distance(entry).to_numpy()
                if len(dist) and np.isfinite(dist).any():
                    turn=int(np.nanargmax(dist))
                    if turn>=10:used=block.iloc[:turn+1]
            line=LineString([(x.x,x.y) for x in used.geometry]).simplify(1.5,preserve_topology=False)
            if line.length<min_length:continue
            duplicate=next(((a,l) for a,l in canonical_lines if line.hausdorff_distance(l)<=duplicate_m),None)
            if duplicate is not None:
                duplicate[0]["support_sessions"].append(str(session_file));continue
            hits=[]
            if not unmatched_events.empty:
                for ei,ev in unmatched_events.iterrows():
                    if pd.isna(ev.get("lat")) or pd.isna(ev.get("lon")):continue
                    d=float(line.distance(_point_metric(ev.lon,ev.lat,crs)))
                    if d<=event_m:hits.append({"event_index":int(ei),"timestamp":str(ev.get("timestamp")),"species":str(ev.get("species","")),"event_type":str(ev.get("event_type","")),"distance_to_recovered_road_m":d})
            connected=_connected_to_osm(line,osm_segments,connect_m)
            temporally_corroborated=any(_event_hits(ev,used,crs,spatial_m=event_m) for _,ev in unmatched_events.iterrows()) if not unmatched_events.empty else False
            # A first traversal may be sufficient when the GPX run connects back to a mapped road.
            # Isolated runs are kept out unless an event corroborates them; repeated sessions are
            # merged later and can become evidence without fabricating geometry.
            accepted=bool(connected or temporally_corroborated)
            item={"road_id":_stable_road_id(line),"session_file":str(session_file),"support_sessions":[str(session_file)],"source_seq_start":int(used.seq.iloc[0]),"source_seq_end":int(used.seq.iloc[-1]),"source_time_start":str(used.timestamp.iloc[0]),"source_time_end":str(used.timestamp.iloc[-1]),"length_m":float(line.length),"max_point_gap_m":max_observed_gap,"connected_to_osm":connected,"temporally_corroborated_event":temporally_corroborated,"accepted":accepted,"corroborated_events":hits}
            audit.append(item)
            if not accepted:continue
            canonical_lines.append((item,line));recovered.extend(_split(line,item["road_id"],float(cfg.get("segment_length_m",10.0))))
    supplemental=gpd.GeoDataFrame(recovered,geometry="geometry",crs=crs) if recovered else _empty(crs)
    if not supplemental.empty:supplemental=supplemental.drop_duplicates("segment_id").reset_index(drop=True)
    if root is not None:
        out=Path(root)/"data"/"processed"/"supplemental_gpx_roads.geojson"
        if supplemental.empty:out.write_text('{"type":"FeatureCollection","features":[]}',encoding="utf-8")
        else:supplemental.to_crs("EPSG:4326").to_file(out,driver="GeoJSON")
    return supplemental,audit


def combine_roads(osm_segments,supplemental):
    osm=osm_segments.copy()
    if "road_source" not in osm.columns:osm["road_source"]="osm"
    if supplemental.empty:return osm
    supplemental=supplemental.copy();cols=sorted(set(osm.columns)|set(supplemental.columns))
    for c in cols:
        if c not in osm.columns:osm[c]=np.nan
        if c not in supplemental.columns:supplemental[c]=np.nan
    combined=pd.concat([osm[cols],supplemental[cols]],ignore_index=True)
    combined=combined.drop_duplicates("segment_id",keep="first").reset_index(drop=True)
    return gpd.GeoDataFrame(combined,geometry="geometry",crs=osm.crs)
