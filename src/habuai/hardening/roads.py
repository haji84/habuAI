from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring


def _empty(crs):
    return gpd.GeoDataFrame(
        columns=["segment_id", "way_id", "part_index", "segment_index", "length_m", "highway", "name", "road_source", "geometry"],
        geometry="geometry",
        crs=crs,
    )


def _point_metric(lon: float, lat: float, metric_crs):
    return gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(metric_crs).iloc[0]


def _event_hits_track(event_row, track_metric: gpd.GeoDataFrame, metric_crs, spatial_m: float, time_pad_min: float) -> bool:
    if pd.isna(event_row.get("lat")) or pd.isna(event_row.get("lon")):
        return False
    pt = _point_metric(event_row.lon, event_row.lat, metric_crs)
    t = pd.Timestamp(event_row.timestamp)
    if t.tzinfo is None:
        t = t.tz_localize("Asia/Tokyo")
    delta = (pd.to_datetime(track_metric.timestamp) - t).abs().dt.total_seconds()
    nearby_time = track_metric.loc[delta <= time_pad_min * 60]
    if nearby_time.empty:
        return False
    return float(nearby_time.geometry.distance(pt).min()) <= spatial_m


def _split_10m(line: LineString, road_id: str, segment_length_m: float):
    rows = []
    n = max(1, math.ceil(line.length / segment_length_m))
    for idx in range(n):
        a = idx * segment_length_m
        b = min((idx + 1) * segment_length_m, line.length)
        seg = substring(line, a, b)
        if seg.is_empty or seg.length <= 0:
            continue
        rows.append({"segment_id":f"{road_id}_{idx}","way_id":road_id,"part_index":0,"segment_index":idx,"length_m":float(seg.length),"highway":"track","name":"GPX-confirmed supplemental road","road_source":"gpx_recovered","geometry":seg})
    return rows


def recover_supplemental_roads(points: pd.DataFrame, osm_matches: pd.DataFrame, events_osm_matched: pd.DataFrame, osm_segments: gpd.GeoDataFrame, cfg: dict, root: Path | None = None) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Recover roads absent from OSM only when a field event is corroborated by a continuous GPX traversal."""
    if points.empty or osm_matches.empty or events_osm_matched.empty:
        return _empty(osm_segments.crs), []
    metric_crs=osm_segments.crs
    p=gpd.GeoDataFrame(osm_matches.copy(),geometry=gpd.points_from_xy(osm_matches.lon,osm_matches.lat),crs="EPSG:4326").to_crs(metric_crs)
    unmatched_events=events_osm_matched[events_osm_matched.segment_id.isna()].copy()
    if unmatched_events.empty:return _empty(metric_crs),[]
    recovered_rows=[];audit=[];road_no=0;seg_len=float(cfg.get("segment_length_m",10.0))
    for session_file,g in p.groupby("session_file",sort=False):
        g=g.sort_values("seq").copy();is_unmatched=g.segment_id.isna().to_numpy();start=0
        while start<len(g):
            if not is_unmatched[start]:start+=1;continue
            end=start
            while end+1<len(g) and is_unmatched[end+1]:end+=1
            block=g.iloc[start:end+1].copy();start=end+1
            if len(block)<20:continue
            corroborated=[]
            for ei,ev in unmatched_events.iterrows():
                if _event_hits_track(ev,block,metric_crs,25.0,20.0):corroborated.append((ei,ev))
            if not corroborated:continue
            coords=[(geom.x,geom.y) for geom in block.geometry]
            if len(coords)<2:continue
            line=LineString(coords)
            if line.length<30:continue
            entry=block.geometry.iloc[0];exit_pt=block.geometry.iloc[-1];used=block
            if entry.distance(exit_pt)<=100.0:
                radial=block.geometry.distance(entry).to_numpy();turn_pos=int(np.nanargmax(radial))
                if turn_pos>=10:used=block.iloc[:turn_pos+1]
            line=LineString([(geom.x,geom.y) for geom in used.geometry]).simplify(1.5,preserve_topology=False)
            if line.length<30:continue
            road_no+=1;date_token=pd.Timestamp(used.timestamp.iloc[0]).strftime("%Y%m%d");road_id=f"GPXROAD_{date_token}_{road_no:02d}";recovered_rows.extend(_split_10m(line,road_id,seg_len))
            hit_events=[]
            for ei,ev in corroborated:
                d=float(line.distance(_point_metric(ev.lon,ev.lat,metric_crs)))
                if d<=25.0:hit_events.append({"event_index":int(ei),"timestamp":str(ev.timestamp),"species":str(ev.get("species","")),"event_type":str(ev.get("event_type","")),"distance_to_recovered_road_m":d})
            audit.append({"road_id":road_id,"session_file":str(session_file),"source_seq_start":int(used.seq.iloc[0]),"source_seq_end":int(used.seq.iloc[-1]),"source_time_start":str(used.timestamp.iloc[0]),"source_time_end":str(used.timestamp.iloc[-1]),"length_m":float(line.length),"corroborated_events":hit_events})
    supplemental=gpd.GeoDataFrame(recovered_rows,geometry="geometry",crs=metric_crs) if recovered_rows else _empty(metric_crs)
    if root is not None:
        out=Path(root)/"data"/"processed"/"supplemental_gpx_roads.geojson"
        if supplemental.empty:out.write_text('{"type":"FeatureCollection","features":[]}',encoding="utf-8")
        else:supplemental.to_crs("EPSG:4326").to_file(out,driver="GeoJSON")
    return supplemental,audit


def combine_roads(osm_segments: gpd.GeoDataFrame, supplemental: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    osm=osm_segments.copy()
    if "road_source" not in osm.columns:osm["road_source"]="osm"
    if supplemental.empty:return osm
    cols=sorted(set(osm.columns)|set(supplemental.columns))
    for c in cols:
        if c not in osm.columns:osm[c]=np.nan
        if c not in supplemental.columns:supplemental[c]=np.nan
    return gpd.GeoDataFrame(pd.concat([osm[cols],supplemental[cols]],ignore_index=True),geometry="geometry",crs=osm.crs)
