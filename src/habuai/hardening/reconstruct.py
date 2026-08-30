from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
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
    return 0.0 if not capture_segments else len(capture_segments & set(route))/len(capture_segments)


def _minimal_span(route:list[str],capture_segments:set[str],padding:int)->tuple[int,int]|None:
    pos=[i for i,s in enumerate(route) if s in capture_segments]
    if not pos:return None
    return max(0,min(pos)-padding),min(len(route)-1,max(pos)+padding)


def _union_graph(sessions:dict[str,list[str]]):
    node_support=Counter();edge_support=Counter();adj=defaultdict(dict)
    for _,route in sessions.items():
        node_support.update(set(route))
        for a,b in zip(route,route[1:]):
            if a==b:continue
            e=tuple(sorted((a,b)));edge_support[e]+=1
    for (a,b),support in edge_support.items():
        cost=1.0/max(support,1)
        adj[a][b]=min(adj[a].get(b,float("inf")),cost);adj[b][a]=min(adj[b].get(a,float("inf")),cost)
    return adj,node_support,edge_support


def _shortest_path(adj,start,goal):
    if start==goal:return [start]
    q=[(0.0,start)];best={start:0.0};prev={}
    while q:
        cost,node=heapq.heappop(q)
        if cost!=best.get(node):continue
        if node==goal:break
        for nxt,w in adj.get(node,{}).items():
            nc=cost+w
            if nc<best.get(nxt,float("inf")):
                best[nxt]=nc;prev[nxt]=node;heapq.heappush(q,(nc,nxt))
    if goal not in best:return None
    path=[goal]
    while path[-1]!=start:path.append(prev[path[-1]])
    path.reverse();return path


def _rescue_with_union_graph(cap_segments:set[str],sessions:dict[str,list[str]],min_anchor_coverage:float):
    """Join capture anchors across different later GPX sessions using only observed GPX adjacencies."""
    adj,node_support,edge_support=_union_graph(sessions)
    present=[s for s in cap_segments if s in node_support]
    coverage=len(present)/len(cap_segments) if cap_segments else 0.0
    if coverage<min_anchor_coverage or not present:return None
    root=max(present,key=lambda s:node_support[s]);ordered=[root];used_edges=[]
    for target in sorted((s for s in present if s!=root),key=lambda s:(-node_support[s],s)):
        path=_shortest_path(adj,root,target)
        if not path:return None
        for a,b in zip(path,path[1:]):
            used_edges.append(tuple(sorted((a,b))))
        for seg in path:
            if seg not in ordered:ordered.append(seg)
    edge_supports=[edge_support[e] for e in used_edges] or [node_support[root]]
    support_sessions=max(node_support[s] for s in present)
    return {
        "piece":ordered,"coverage":coverage,"node_support":node_support,"edge_supports":edge_supports,
        "support_sessions":support_sessions,"median_edge_support":float(np.median(edge_supports)),
        "min_edge_support":int(min(edge_supports)),"present_capture_segments":len(present),
    }


def reconstruct_historical_routes(root:Path,events:pd.DataFrame,visits:pd.DataFrame,cfg:dict)->tuple[pd.DataFrame,dict]:
    """Reconstruct May-July spatial exposure from later GPX without inventing historical times."""
    rcfg=cfg.get("historical_route_reconstruction",{})
    if not rcfg.get("enabled",True) or events.empty or visits.empty:return pd.DataFrame(),{"status":"disabled-or-empty"}
    rollover=int(cfg.get("night_rollover_hour",7));start=pd.Timestamp(rcfg.get("start","2026-05-01"),tz="Asia/Tokyo");end=pd.Timestamp(rcfg.get("end","2026-08-01"),tz="Asia/Tokyo");padding=int(rcfg.get("padding_segments",40));candidate_delta=float(rcfg.get("candidate_coverage_delta",0.15));min_candidate_cov=float(rcfg.get("min_candidate_coverage",0.30));high_cov=float(rcfg.get("high_confidence_coverage",1.0));high_support=int(rcfg.get("high_confidence_min_support_sessions",2));rescue_cov=float(rcfg.get("rescue_min_anchor_coverage",1.0));rescue_high_edge=float(rcfg.get("rescue_high_median_edge_support",2.0))
    ev=events.copy();ev["timestamp"]=pd.to_datetime(ev.timestamp,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo");hist=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&(ev.timestamp>=start)&(ev.timestamp<end)].copy();hist["reconstruction_night"]=_operational_night(hist.timestamp,rollover)
    later=visits.copy();later["entered_at"]=pd.to_datetime(later.entered_at,format="mixed",utc=True).dt.tz_convert("Asia/Tokyo");template_start=pd.Timestamp(rcfg.get("template_start","2026-08-13"),tz="Asia/Tokyo");later=later[later.entered_at>=template_start].copy();sessions=_ordered_session_segments(later)
    rows=[];audit=[]
    for night,ndf in hist.groupby("reconstruction_night",sort=True):
        matched=ndf[ndf.segment_id.notna()].copy();cap_segments=set(matched.segment_id.astype(str));total_caps=int(len(ndf));matched_caps=int(len(matched))
        if not cap_segments:audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"status":"no-road-matched-capture","best_coverage":0.0});continue
        scored=sorted(((session,_coverage(cap_segments,route),len(cap_segments&set(route)),route) for session,route in sessions.items()),key=lambda x:(x[1],x[2]),reverse=True);best_cov=scored[0][1] if scored else 0.0;candidates=[x for x in scored if x[1]>=max(min_candidate_cov,best_cov-candidate_delta) and x[2]>0]
        spans=[];support=Counter()
        for session,cov,hit,route in candidates:
            span=_minimal_span(route,cap_segments,padding)
            if span is None:continue
            a,b=span;piece=route[a:b+1];spans.append((session,cov,hit,piece,a,b));support.update(set(piece))
        primary=None
        if spans:
            def quality(item):
                session,cov,hit,piece,a,b=item;common=np.mean([support[s]/len(spans) for s in set(piece)]) if piece else 0.0;return (cov,common,hit,-len(piece))
            primary=max(spans,key=quality)
        rescued=None;original_confidence=None
        if primary:
            session,cov,hit,piece,a,b=primary;support_sessions=len(candidates);original_confidence="B-high" if cov>=high_cov and support_sessions>=high_support else ("B-medium" if cov>=0.5 else "C-low")
            if original_confidence=="C-low":rescued=_rescue_with_union_graph(cap_segments,sessions,rescue_cov)
        else:
            rescued=_rescue_with_union_graph(cap_segments,sessions,rescue_cov)
        if rescued:
            piece=rescued["piece"];cov=rescued["coverage"];support_sessions=rescued["support_sessions"];confidence="B-high" if rescued["median_edge_support"]>=rescue_high_edge else "B-medium";session="MULTI_GPX_UNION";a=b=-1;support=rescued["node_support"];source="multi_gpx_union_graph_rescue";rescue_status="rescued-from-"+(original_confidence or "unreconstructed")
        elif primary:
            session,cov,hit,piece,a,b=primary;support_sessions=len(candidates);confidence=original_confidence;source="later_gpx_template";rescue_status="not-needed" if confidence!="C-low" else "rescue-failed"
        else:
            audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"status":"no-template-match","best_coverage":best_cov,"rescue_status":"failed"});continue
        cap_set=set(cap_segments)
        for order,seg in enumerate(piece):
            rows.append({"night":night,"route_order":order,"segment_id":seg,"reconstruction_source":source,"template_session_file":session,"template_span_start_visit":a,"template_span_end_visit":b,"capture_anchor_segment":seg in cap_set,"capture_coverage":cov,"support_sessions":support_sessions,"segment_support_sessions":support.get(seg,0),"segment_support_fraction":support.get(seg,0)/max(len(sessions),1),"reconstruction_confidence":confidence,"rescue_status":rescue_status,"historical_time_reconstructed":False})
        audit.append({"night":night,"capture_events":total_caps,"matched_capture_events":matched_caps,"unique_capture_segments":len(cap_segments),"status":"reconstructed","best_template":session,"best_coverage":cov,"candidate_sessions":support_sessions,"reconstructed_segment_rows":len(piece),"confidence":confidence,"original_confidence":original_confidence,"rescue_status":rescue_status,"rescue_median_edge_support":None if not rescued else rescued["median_edge_support"],"rescue_min_edge_support":None if not rescued else rescued["min_edge_support"]})
    out=pd.DataFrame(rows);report={"status":"ok","method":"capture GPS -> single-template match; C-low/no-template -> multi-GPX observed-adjacency graph rescue -> trusted contiguous spatial route","nights_total":int(hist.reconstruction_night.nunique()),"capture_events":int(len(hist)),"road_matched_capture_events":int(hist.segment_id.notna().sum()),"nights_reconstructed":int(sum(a.get("status")=="reconstructed" for a in audit)),"nights_coverage_100pct":int(sum(a.get("status")=="reconstructed" and a.get("best_coverage",0)>=1.0 for a in audit)),"nights_coverage_ge_50pct":int(sum(a.get("status")=="reconstructed" and a.get("best_coverage",0)>=0.5 for a in audit)),"confidence_counts":dict(Counter(a.get("confidence") for a in audit if a.get("confidence"))),"rescued_nights":int(sum(str(a.get("rescue_status","")).startswith("rescued-") for a in audit)),"remaining_c_low_nights":int(sum(a.get("confidence")=="C-low" for a in audit)),"nightly":audit,"warning":"Spatial exposure only. Later GPX timestamps are never copied into May-July."}
    p=root/"data"/"processed";r=root/"reports";p.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True);out.to_csv(p/"reconstructed_routes_2026-05_07.csv",index=False);(r/"historical_route_reconstruction.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return out,report


def build_reconstructed_spatial_learning(root:Path,reconstructed:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    """Use only routes promoted to B-high/B-medium after reconstruction/rescue."""
    out_path=root/"data"/"processed"/"reconstructed_spatial_learning_2026-05_07.csv";report_path=root/"reports"/"reconstructed_spatial_learning.json"
    if reconstructed.empty:
        out=pd.DataFrame();report={"status":"empty","rows":0};out.to_csv(out_path,index=False);report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return out,report
    trusted=reconstructed[reconstructed.reconstruction_confidence.isin(["B-high","B-medium"])].copy()
    if trusted.empty:
        out=pd.DataFrame();report={"status":"no-trusted-routes","rows":0};out.to_csv(out_path,index=False);report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return out,report
    trusted["spatial_label"]=trusted.capture_anchor_segment.astype(bool).astype(int);trusted["sample_weight"]=np.where(trusted.reconstruction_confidence.eq("B-high"),1.0,0.65)
    agg=trusted.groupby(["night","segment_id"],as_index=False).agg(spatial_label=("spatial_label","max"),reconstruction_confidence=("reconstruction_confidence",lambda s:"B-high" if (s=="B-high").any() else "B-medium"),sample_weight=("sample_weight","max"),segment_support_fraction=("segment_support_fraction","max"),support_sessions=("support_sessions","max"),template_session_file=("template_session_file","first"),reconstruction_source=("reconstruction_source","first"))
    agg["learning_row_source"]="historical_reconstructed_spatial_exposure";agg["historical_time_reconstructed"]=False;agg.to_csv(out_path,index=False)
    report={"status":"ok","rows":int(len(agg)),"nights":int(agg.night.nunique()),"positive_segment_rows":int(agg.spatial_label.sum()),"negative_segment_rows":int((agg.spatial_label==0).sum()),"confidence_counts":agg.reconstruction_confidence.value_counts().to_dict(),"rescued_rows":int(agg.reconstruction_source.eq("multi_gpx_union_graph_rescue").sum()),"rescued_nights":int(agg.loc[agg.reconstruction_source.eq("multi_gpx_union_graph_rescue"),"night"].nunique()),"usage":"spatial/location model only; excluded from temporal/weather/time-of-night model","remaining_c_low_excluded":int((reconstructed.reconstruction_confidence=="C-low").sum())}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return agg,report
