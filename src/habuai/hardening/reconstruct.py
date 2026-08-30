from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _operational_night(ts: pd.Series, rollover_hour: int) -> pd.Series:
    t=pd.to_datetime(ts,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo")
    return (t-pd.Timedelta(hours=rollover_hour)).dt.date.astype(str)


def _ordered_session_segments(visits: pd.DataFrame) -> dict[str,list[str]]:
    out={}
    for session,df in visits.sort_values(["session_file","visit_index"]).groupby("session_file",sort=True):
        out[str(session)]=df.segment_id.astype(str).tolist()
    return out


def _coverage(capture_segments:set[str],route:list[str])->float:
    if not capture_segments:return 0.0
    return len(capture_segments & set(route))/len(capture_segments)


def _minimal_span(route:list[str],capture_segments:set[str],padding:int)->tuple[int,int]|None:
    pos=[i for i,s in enumerate(route) if s in capture_segments]
    if not pos:return None
    return max(0,min(pos)-padding),min(len(route)-1,max(pos)+padding)


def reconstruct_historical_routes(root:Path,events:pd.DataFrame,visits:pd.DataFrame,cfg:dict)->tuple[pd.DataFrame,dict]:
    """Reconstruct 2026-05..07 exploration roads from later GPX route geometry.

    The reconstruction uses historical capture GPS road matches as anchors and only later GPX
    road sequences as route templates. It never pretends the later GPX timestamps are historical.
    Output rows therefore contain route order and provenance, but no fabricated entered_at time.
    """
    rcfg=cfg.get("historical_route_reconstruction",{})
    if not rcfg.get("enabled",True) or events.empty or visits.empty:
        return pd.DataFrame(),{"status":"disabled-or-empty"}
    rollover=int(cfg.get("night_rollover_hour",7));start=pd.Timestamp(rcfg.get("start","2026-05-01"),tz="Asia/Tokyo");end=pd.Timestamp(rcfg.get("end","2026-08-01"),tz="Asia/Tokyo");padding=int(rcfg.get("padding_segments",40));candidate_delta=float(rcfg.get("candidate_coverage_delta",0.15));min_candidate_cov=float(rcfg.get("min_candidate_coverage",0.30));high_cov=float(rcfg.get("high_confidence_coverage",1.0));high_support=int(rcfg.get("high_confidence_min_support_sessions",2))
    ev=events.copy();ev["timestamp"]=pd.to_datetime(ev.timestamp,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo");hist=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&(ev.timestamp>=start)&(ev.timestamp<end)].copy();hist["reconstruction_night"]=_operational_night(hist.timestamp,rollover)
    later=visits.copy();later["entered_at"]=pd.to_datetime(later.entered_at,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo");template_start=pd.Timestamp(rcfg.get("template_start","2026-08-13"),tz="Asia/Tokyo");later=later[later.entered_at>=template_start].copy();sessions=_ordered_session_segments(later)
    rows=[];audit=[]
    for night,ndf in hist.groupby("reconstruction_night",sort=True):
        matched=ndf[ndf.segment_id.notna()].copy();cap_segments=set(matched.segment_id.astype(str));total_caps=int(len(ndf));matched_caps=int(len(matched))
        if not cap_segments:
            audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"status":"no-road-matched-capture","best_coverage":0.0});continue
        scored=[]
        for session,route in sessions.items():
            cov=_coverage(cap_segments,route);hit=len(cap_segments&set(route));scored.append((session,cov,hit,route))
        scored.sort(key=lambda x:(x[1],x[2]),reverse=True);best_cov=scored[0][1] if scored else 0.0
        candidates=[x for x in scored if x[1]>=max(min_candidate_cov,best_cov-candidate_delta) and x[2]>0]
        if not candidates:
            audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"status":"no-template-match","best_coverage":best_cov});continue
        spans=[];support=Counter()
        for session,cov,hit,route in candidates:
            span=_minimal_span(route,cap_segments,padding)
            if span is None:continue
            a,b=span;piece=route[a:b+1];spans.append((session,cov,hit,piece,a,b));support.update(set(piece))
        if not spans:
            audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"status":"no-contiguous-span","best_coverage":best_cov});continue
        # Prefer a high-coverage template whose contiguous span is most corroborated by other GPX sessions.
        def quality(item):
            session,cov,hit,piece,a,b=item;common=np.mean([support[s]/len(spans) for s in set(piece)]) if piece else 0.0;return (cov,common,hit,-len(piece))
        primary=max(spans,key=quality);session,cov,hit,piece,a,b=primary
        support_sessions=len(candidates);confidence="B-high" if cov>=high_cov and support_sessions>=high_support else ("B-medium" if cov>=0.5 else "C-low")
        cap_set=set(cap_segments)
        for order,seg in enumerate(piece):
            rows.append({"night":night,"route_order":order,"segment_id":seg,"reconstruction_source":"later_gpx_template","template_session_file":session,"template_span_start_visit":a,"template_span_end_visit":b,"capture_anchor_segment":seg in cap_set,"capture_coverage":cov,"support_sessions":support_sessions,"segment_support_sessions":support[seg],"segment_support_fraction":support[seg]/len(spans),"reconstruction_confidence":confidence,"historical_time_reconstructed":False})
        audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"unique_capture_segments":len(cap_segments),"status":"reconstructed","best_template":session,"best_coverage":cov,"candidate_sessions":support_sessions,"reconstructed_segment_rows":len(piece),"confidence":confidence})
    out=pd.DataFrame(rows);report={"status":"ok","method":"capture GPS -> later GPX match -> candidate route extraction -> multi-GPX support -> minimum contiguous span containing capture anchors","nights_total":int(hist.reconstruction_night.nunique()),"capture_events":int(len(hist)),"road_matched_capture_events":int(hist.segment_id.notna().sum()),"nights_reconstructed":int(sum(a.get("status")=="reconstructed" for a in audit)),"nights_coverage_100pct":int(sum(a.get("status")=="reconstructed" and a.get("best_coverage",0)>=1.0 for a in audit)),"nights_coverage_ge_50pct":int(sum(a.get("status")=="reconstructed" and a.get("best_coverage",0)>=0.5 for a in audit)),"confidence_counts":dict(Counter(a.get("confidence") for a in audit if a.get("confidence"))),"nightly":audit,"warning":"Routes are reconstructed spatial exposure only. Later GPX timestamps are never copied into May-July, so exact historical traversal time is not fabricated."}
    p=root/"data"/"processed";r=root/"reports";p.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    out.to_csv(p/"reconstructed_routes_2026-05_07.csv",index=False);(r/"historical_route_reconstruction.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return out,report
