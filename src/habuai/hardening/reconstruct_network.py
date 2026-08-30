from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np


def _later_support(sessions:dict[str,list[str]])->Counter:
    c=Counter()
    for route in sessions.values():c.update(set(route))
    return c


def _segment_graph(segs,snap_m:float):
    """Build segment adjacency from endpoint proximity using an exact metric-distance gate.

    OSM ways that meet at the same physical junction are not always emitted with numerically
    identical endpoints after 10 m segmentation. The former rounded-bin implementation at 0.5 m
    fragmented real roads. We bucket endpoints only for candidate lookup, then require true
    Euclidean endpoint distance <= snap_m before connecting segments.
    """
    cell=max(float(snap_m),0.01);endpoint_cells=defaultdict(list);lengths={}
    for r in segs[["segment_id","geometry"]].dropna(subset=["segment_id","geometry"]).itertuples():
        sid=str(r.segment_id);g=r.geometry
        if g is None or g.is_empty:continue
        coords=list(g.coords)
        if len(coords)<2:continue
        lengths[sid]=float(g.length)
        for x,y in (coords[0],coords[-1]):
            x=float(x);y=float(y);endpoint_cells[(int(np.floor(x/cell)),int(np.floor(y/cell)))].append((sid,x,y))
    adj=defaultdict(set);seen_pairs=set()
    for (cx,cy),members in endpoint_cells.items():
        nearby=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):nearby.extend(endpoint_cells.get((cx+dx,cy+dy),()))
        for a,x1,y1 in members:
            for b,x2,y2 in nearby:
                if a==b:continue
                pair=tuple(sorted((a,b)))
                if pair in seen_pairs:continue
                if float(np.hypot(x1-x2,y1-y2))<=cell:
                    adj[a].add(b);adj[b].add(a);seen_pairs.add(pair)
    return adj,lengths


def _path(adj,lengths,support,start,goal,unseen_penalty:float,support_bonus:float,use_support:bool):
    if start==goal:return [start],0.0
    if start not in lengths or goal not in lengths:return None,None
    q=[(0.0,start)];best={start:0.0};prev={}
    while q:
        cost,node=heapq.heappop(q)
        if cost!=best.get(node):continue
        if node==goal:break
        for nxt in adj.get(node,()):
            base=max(lengths.get(nxt,10.0),1.0)
            if use_support:
                s=support.get(nxt,0);w=base*((1.0 if s else unseen_penalty)/(1.0+support_bonus*min(s,3)))
            else:w=base
            nc=cost+w
            if nc<best.get(nxt,float("inf")):
                best[nxt]=nc;prev[nxt]=node;heapq.heappush(q,(nc,nxt))
    if goal not in best:return None,None
    p=[goal]
    while p[-1]!=start:p.append(prev[p[-1]])
    p.reverse();return p,float(sum(lengths.get(s,10.0) for s in p[1:]))


def rescue_with_road_network(capture_segments_ordered:list[str],sessions:dict[str,list[str]],segs,cfg:dict):
    """Conservative rescue for anchors not all present in later GPX.

    Connects the actual capture-road anchors in chronological order on the mapped road network.
    Repeated later-GPX roads are cheaper, but unseen roads remain available with a penalty. A route
    is accepted only when it stays reasonably close to the pure shortest road-network path.
    """
    rcfg=cfg.get("historical_route_reconstruction",{})
    snap=float(rcfg.get("road_rescue_endpoint_snap_m",5.0));unseen=float(rcfg.get("road_rescue_unseen_penalty",2.5));bonus=float(rcfg.get("road_rescue_support_bonus",0.35));max_detour=float(rcfg.get("road_rescue_max_detour_ratio",1.6));max_gap=float(rcfg.get("road_rescue_max_gap_path_m",8000.0));high_support_fraction=float(rcfg.get("road_rescue_high_support_fraction",0.75));high_detour=float(rcfg.get("road_rescue_high_max_detour_ratio",1.25));min_supported=float(rcfg.get("road_rescue_min_supported_fraction",0.65))
    anchors=list(dict.fromkeys(str(s) for s in capture_segments_ordered if s and str(s)!="nan"))
    if not anchors or segs is None or len(anchors)<2:return None
    adj,lengths=_segment_graph(segs,snap);support=_later_support(sessions);combined=[];weighted_len=0.0;baseline_len=0.0
    for a,b in zip(anchors,anchors[1:]):
        p,plen=_path(adj,lengths,support,a,b,unseen,bonus,True);base,b_len=_path(adj,lengths,support,a,b,unseen,bonus,False)
        if not p or not base or plen is None or b_len is None or plen>max_gap:return None
        weighted_len+=plen;baseline_len+=max(b_len,1.0)
        if combined and combined[-1]==p[0]:combined.extend(p[1:])
        else:combined.extend(p)
    detour=weighted_len/max(baseline_len,1.0)
    if detour>max_detour:return None
    support_vals=[support.get(s,0) for s in combined];supported_fraction=float(np.mean([v>0 for v in support_vals])) if support_vals else 0.0;median_support=float(np.median(support_vals)) if support_vals else 0.0
    if supported_fraction<min_supported:return None
    confidence="B-high" if supported_fraction>=high_support_fraction and detour<=high_detour and median_support>=1.0 else "B-medium"
    return {"piece":combined,"coverage":1.0,"support":support,"support_sessions":max(support_vals) if support_vals else 0,"confidence":confidence,"detour_ratio":detour,"supported_fraction":supported_fraction,"median_support":median_support,"weighted_path_m":weighted_len,"baseline_shortest_path_m":baseline_len}
