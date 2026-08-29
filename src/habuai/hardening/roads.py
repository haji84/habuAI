from __future__ import annotations
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
    if pd.isna(ev.get("lat")) or pd.isna(ev.get("lon")):return False
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


def recover_supplemental_roads(points,osm_matches,events_osm_matched,osm_segments,cfg,root:Path|None=None):
    if points.empty or osm_matches.empty or events_osm_matched.empty:return _empty(osm_segments.crs),[]
    crs=osm_segments.crs;p=gpd.GeoDataFrame(osm_matches.copy(),geometry=gpd.points_from_xy(osm_matches.lon,osm_matches.lat),crs="EPSG:4326").to_crs(crs)
    all_unmatched=events_osm_matched[events_osm_matched.segment_id.isna()].copy()
    triggers=all_unmatched[(all_unmatched.species=="ハブ")&(all_unmatched.event_type=="捕獲")].copy()
    if triggers.empty:return _empty(crs),[]
    recovered=[];audit=[];canonical_lines=[];road_no=0
    for session_file,g in p.groupby("session_file",sort=False):
        g=g.sort_values("seq").copy();mask=g.segment_id.isna().to_numpy();start=0
        while start<len(g):
            if not mask[start]:start+=1;continue
            end=start
            while end+1<len(g) and mask[end+1]:end+=1
            block=g.iloc[start:end+1].copy();start=end+1
            if len(block)<20 or not any(_event_hits(ev,block,crs) for _,ev in triggers.iterrows()):continue
            entry=block.geometry.iloc[0];exit_pt=block.geometry.iloc[-1];used=block
            if entry.distance(exit_pt)<=100:
                turn=int(np.nanargmax(block.geometry.distance(entry).to_numpy()))
                if turn>=10:used=block.iloc[:turn+1]
            line=LineString([(x.x,x.y) for x in used.geometry]).simplify(1.5,preserve_topology=False)
            if line.length<30:continue
            duplicate=next((a for a,l in canonical_lines if line.hausdorff_distance(l)<=20.0),None)
            if duplicate is not None:
                duplicate["support_sessions"].append(str(session_file));continue
            road_no+=1;road_id=f"GPXROAD_{pd.Timestamp(used.timestamp.iloc[0]).strftime('%Y%m%d')}_{road_no:02d}"
            hits=[]
            for ei,ev in all_unmatched.iterrows():
                d=float(line.distance(_point_metric(ev.lon,ev.lat,crs)))
                if d<=25: hits.append({"event_index":int(ei),"timestamp":str(ev.timestamp),"species":str(ev.get("species","")),"event_type":str(ev.get("event_type","")),"distance_to_recovered_road_m":d})
            item={"road_id":road_id,"session_file":str(session_file),"support_sessions":[str(session_file)],"source_seq_start":int(used.seq.iloc[0]),"source_seq_end":int(used.seq.iloc[-1]),"source_time_start":str(used.timestamp.iloc[0]),"source_time_end":str(used.timestamp.iloc[-1]),"length_m":float(line.length),"corroborated_events":hits}
            audit.append(item);canonical_lines.append((item,line));recovered.extend(_split(line,road_id,float(cfg.get("segment_length_m",10.0))))
    supplemental=gpd.GeoDataFrame(recovered,geometry="geometry",crs=crs) if recovered else _empty(crs)
    if root is not None:
        out=Path(root)/"data"/"processed"/"supplemental_gpx_roads.geojson"
        if supplemental.empty:out.write_text('{"type":"FeatureCollection","features":[]}',encoding="utf-8")
        else:supplemental.to_crs("EPSG:4326").to_file(out,driver="GeoJSON")
    return supplemental,audit


def combine_roads(osm_segments,supplemental):
    osm=osm_segments.copy()
    if "road_source" not in osm.columns:osm["road_source"]="osm"
    if supplemental.empty:return osm
    cols=sorted(set(osm.columns)|set(supplemental.columns))
    for c in cols:
        if c not in osm.columns:osm[c]=np.nan
        if c not in supplemental.columns:supplemental[c]=np.nan
    return gpd.GeoDataFrame(pd.concat([osm[cols],supplemental[cols]],ignore_index=True),geometry="geometry",crs=osm.crs)
