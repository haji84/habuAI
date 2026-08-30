from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_clock(date_value, clock_value):
    if pd.isna(date_value): return pd.NaT
    day=pd.to_datetime(str(date_value),errors="coerce")
    if pd.isna(day): return pd.NaT
    text="" if pd.isna(clock_value) else str(clock_value)
    m=re.search(r"(\d{1,2}):(\d{2})",text)
    if not m:return pd.NaT
    hour,minute=int(m.group(1)),int(m.group(2));extra_day,hour=divmod(hour,24)
    return (day+pd.Timedelta(days=extra_day,hours=hour,minutes=minute)).tz_localize("Asia/Tokyo")


def _size_code(value):
    text="" if pd.isna(value) else str(value)
    if "極小" in text or "幼体" in text:return "極小"
    if "大" in text or "大型" in text:return "大"
    if "中" in text or "中型" in text:return "中"
    if "小" in text or "小型" in text:return "小"
    return None


def load_canonical_capture_master(root:Path)->pd.DataFrame:
    files=sorted((root/"data"/"canonical").glob("habu_capture_master_part_*.csv"))
    if not files:return pd.DataFrame()
    src=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    rows=[]
    for r in src.to_dict("records"):
        event_type=r.get("イベント")
        if event_type!="捕獲":continue
        timestamp=_parse_clock(r.get("日付"),r.get("時刻"));count=pd.to_numeric(r.get("数"),errors="coerce")
        night_date=(timestamp-pd.Timedelta(hours=7)).date().isoformat() if pd.notna(timestamp) else str(r.get("日付"))
        rows.append({"canonical_id":str(r.get("ID")),"timestamp":timestamp,"session_start":pd.NaT,"event_type":"捕獲","species":str(r.get("種別")),"individual_count":int(count) if pd.notna(count) else 1,"size":_size_code(r.get("サイズ")),"wetness":None,"lat":pd.to_numeric(r.get("緯度"),errors="coerce"),"lon":pd.to_numeric(r.get("経度"),errors="coerce"),"night_date":night_date,"moon_age_observed":pd.to_numeric(r.get("月齢"),errors="coerce"),"tide_observed":None if pd.isna(r.get("潮")) else str(r.get("潮")),"tide_direction_observed":None if pd.isna(r.get("上げ下げ")) else str(r.get("上げ下げ")),"weather_observed":None if pd.isna(r.get("天候")) else str(r.get("天候")),"temperature_observed_c":pd.to_numeric(r.get("気温"),errors="coerce"),"canonical_source":None if pd.isna(r.get("ソース")) else str(r.get("ソース")),"raw_text":f"canonical:{r.get('ID')} 捕獲"})
    out=pd.DataFrame(rows)
    if not out.empty:out["individual_count"]=pd.to_numeric(out.individual_count,errors="coerce").fillna(1).astype(int)
    return out


def merge_canonical_capture_master(raw_events,canonical):
    if canonical.empty:return raw_events.copy()
    if raw_events.empty:return canonical.copy()
    raw=raw_events.copy();cols=sorted(set(raw.columns)|set(canonical.columns))
    return pd.concat([raw.reindex(columns=cols),canonical.reindex(columns=cols)],ignore_index=True)


def match_events_preserve_all(pipeline,events,segs):
    if events.empty:return events.copy()
    x=events.copy().reset_index(drop=True);x["_source_row"]=np.arange(len(x));lat=pd.to_numeric(x.lat,errors="coerce");lon=pd.to_numeric(x.lon,errors="coerce");has_gps=lat.notna()&lon.notna();with_gps=x[has_gps].copy();without_gps=x[~has_gps].copy()
    if not with_gps.empty:
        matched=pipeline.match_events(with_gps.set_index("_source_row"),segs);matched["_source_row"]=matched.index;matched["_distance_sort"]=pd.to_numeric(matched.get("event_match_distance_m"),errors="coerce").fillna(np.inf);matched=matched.sort_values(["_source_row","_distance_sort"],kind="stable").drop_duplicates("_source_row",keep="first").drop(columns="_distance_sort")
    else:matched=with_gps
    without_gps["segment_id"]=None;without_gps["event_match_distance_m"]=np.nan
    return pd.concat([matched,without_gps],ignore_index=True,sort=False).sort_values("_source_row",kind="stable").drop(columns="_source_row",errors="ignore").reset_index(drop=True)


def add_canonical_audit(events):
    if events.empty:return {"capture_events":0,"capture_individuals":0,"gps_capture_events":0,"gps_capture_individuals":0,"gps_missing_capture_events":0}
    h=events[(events.species=="ハブ")&(events.event_type=="捕獲")].copy();count=pd.to_numeric(h.individual_count,errors="coerce").fillna(1);has_gps=pd.to_numeric(h.lat,errors="coerce").notna()&pd.to_numeric(h.lon,errors="coerce").notna()
    return {"capture_events":int(len(h)),"capture_individuals":int(count.sum()),"gps_capture_events":int(has_gps.sum()),"gps_capture_individuals":int(count[has_gps].sum()),"gps_missing_capture_events":int((~has_gps).sum())}


def build_positive_label_audit(events,learning):
    h=events[(events.species=="ハブ")&(events.event_type=="捕獲")].copy();rows=[]
    for r in h.itertuples():
        segment=getattr(r,"segment_id",None);timestamp=getattr(r,"timestamp",pd.NaT);reason="";positives=0;nearest_seconds=np.nan
        if pd.isna(getattr(r,"lat",np.nan)) or pd.isna(getattr(r,"lon",np.nan)):reason="missing_gps"
        elif not segment or pd.isna(segment):reason="road_unmatched"
        elif pd.isna(timestamp):reason="missing_exact_time"
        else:
            c=learning[learning.segment_id==segment].copy()
            if c.empty:reason="segment_not_visited_in_gpx"
            else:
                delta=(pd.to_datetime(c.entered_at)-pd.Timestamp(timestamp)).dt.total_seconds().abs();nearest_seconds=float(delta.min()) if len(delta) else np.nan;positives=int(((c.habu_capture==1)&(delta<=600)).sum());reason="" if positives else "no_visit_within_10_minutes"
        rows.append({"canonical_id":getattr(r,"canonical_id",None),"timestamp":timestamp,"individual_count":int(getattr(r,"individual_count",1) or 1),"lat":getattr(r,"lat",np.nan),"lon":getattr(r,"lon",np.nan),"segment_id":segment,"event_match_distance_m":getattr(r,"event_match_distance_m",np.nan),"positive_learning_rows":positives,"nearest_visit_delta_s":nearest_seconds,"audit_status":"ok" if positives>0 else reason})
    return pd.DataFrame(rows)
