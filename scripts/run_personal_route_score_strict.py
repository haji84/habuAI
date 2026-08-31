from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import _as_jst, _estimator, _fit, _spatial_feature_sets
from habuai.hardening.spatial_history import add_historical_spatial_features
from run_joint_point_time_backtest_v2 import hav, curve24

SALT='habuai-private-route-v2'

def pid(s): return 'R-'+hashlib.sha256(f'{SALT}:{s}'.encode()).hexdigest()[:10]

def cmin(a,b):
    d=abs((a-b).total_seconds())/60.; return float(min(d,1440.-(d%1440.)))

def hit(a,z,rad,mins):
    if z.empty:return 0
    dt=np.array([cmin(a.timestamp,t) for t in z.entered_at])
    ds=hav(a.lat,a.lon,z.lat.to_numpy(float),z.lon.to_numpy(float))
    return int(((ds<=rad)&(dt<=mins)).any())

def summarize(rows):
    if not rows:return {'eligible_captures':0}
    d=pd.DataFrame(rows);o={'nights_scored':int(d.night.nunique()),'eligible_captures':int(len(d))}
    for scope in ['top1','top3','top5','top10','top10pct','top20pct']:
      for rad in [50,100,250]:
       for mins in [10,20,30,60]:
        k=f'{scope}_{rad}m_{mins}min';o[k+'_hits']=int(d[k].sum());o[k+'_rate']=float(d[k].mean())
    return o

def main():
    root=Path(__file__).resolve().parents[1];cfg=pipeline.load_config(root);p=root/'data'/'processed';r=root/'reports';r.mkdir(exist_ok=True)
    tour=json.loads((r/'model_tournament_summary.json').read_text(encoding='utf-8'));o40=json.loads((r/'original40_full_history_summary.json').read_text(encoding='utf-8'))
    sw=tour['spatial']['best'];tw=o40['best_selected_on_early_period']
    recon=pd.read_csv(p/'reconstructed_spatial_learning_2026-05_07.csv',low_memory=False);md=pd.read_csv(p/'learning_10m_road.csv',low_memory=False);ev=pd.read_csv(p/'events_matched.csv',low_memory=False);segs=gpd.read_file(p/'road_segments_10m.geojson').to_crs('EPSG:6669')
    md['entered_at']=_as_jst(md.entered_at);ev['timestamp']=_as_jst(ev.timestamp);ev['lat']=pd.to_numeric(ev.lat,errors='coerce');ev['lon']=pd.to_numeric(ev.lon,errors='coerce');md['night']=(md.entered_at-pd.Timedelta(hours=7)).dt.date.astype(str);gpx=md[md.learning_row_source=='gpx_visit'].copy()
    static=[c for c in ['length_m','bearing_deg','curvature_deg','road_class_code','junction_distance_m','stream_distance_m','coast_distance_m','forest_distance_m','farmland_distance_m','residential_distance_m'] if c in md]
    agg={c:(c,'first') for c in static};agg['spatial_label']=('habu_capture','max');aug=md.groupby(['night','segment_id'],as_index=False).agg(**agg);aug['sample_weight']=1.;aug['reconstruction_confidence']='actual_august';aug=aug.drop(columns=[c for c in aug if c.startswith('hist_') or c.startswith('days_since_capture_') or c in {'segment_x_m','segment_y_m'}],errors='ignore');aug,_=add_historical_spatial_features(aug,ev,segs,cfg,root=None)
    combined=pd.concat([recon,aug],ignore_index=True,sort=False);feats=_spatial_feature_sets(combined)[str(sw['feature_set'])]
    coord=segs[['segment_id','geometry']].copy().to_crs('EPSG:4326');coord['lat']=coord.geometry.apply(lambda z:z.centroid.y);coord['lon']=coord.geometry.apply(lambda z:z.centroid.x);coord=coord.drop_duplicates('segment_id').set_index('segment_id')[['lat','lon']]
    radius=float(tw['radius_m']);bw=float(tw['bandwidth_h']);tau=tw.get('recency_tau_days');tau=None if tau is None else float(tau)
    weights=[(1.,1.),(1.,2.),(2.,1.),(.5,1.),(1.,.5)];nights=sorted(gpx.night.unique());split=max(4,int(np.floor(len(nights)*.60)));selection=set(nights[:split]);confirmation=set(nights[split:]);variants=[]
    for alpha,beta in weights:
      rows=[]
      for night in nights:
        start=pd.Timestamp(night,tz='Asia/Tokyo')+pd.Timedelta(hours=7);end=start+pd.Timedelta(days=1);train=combined[combined.night.astype(str)<night];vis=gpx[(gpx.entered_at>=start)&(gpx.entered_at<end)].copy();actual=ev[(ev.species=='ハブ')&(ev.event_type=='捕獲')&(ev.timestamp>=start)&(ev.timestamp<end)&ev.lat.notna()&ev.lon.notna()].copy()
        if vis.empty or actual.empty or train.spatial_label.nunique()<2:continue
        m=_estimator(str(sw['estimator']));_fit(m,train.reindex(columns=feats).replace([np.inf,-np.inf],np.nan),train.spatial_label.astype(int),pd.to_numeric(train.sample_weight,errors='coerce').fillna(1.).to_numpy());ts=aug[aug.night.astype(str)==night].copy();ts['spatial_p']=m.predict_proba(ts.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];vis['spatial_p']=vis.segment_id.map(ts.set_index('segment_id').spatial_p).fillna(0.)
        past=ev[ev.timestamp<start];curves={}
        for sid in vis.segment_id.dropna().unique():
          if sid in coord.index:
            la,lo=coord.loc[sid,['lat','lon']];curves[sid]=curve24(past,start,la,lo,radius,bw,tau)
        vis['time_p']=[float(curves.get(sid,np.ones(24))[int(t.hour)]) for sid,t in zip(vis.segment_id,vis.entered_at)];vis['score']=(np.clip(vis.spatial_p,1e-9,1)**alpha)*(np.clip(vis.time_p,1e-9,1)**beta);vis=vis.sort_values('score',ascending=False).reset_index(drop=True);vis['rank']=np.arange(1,len(vis)+1);vis['pct']=vis['rank']/len(vis);vc=vis.merge(coord,left_on='segment_id',right_index=True,how='left')
        scopes={'top1':vc.head(1),'top3':vc.head(3),'top5':vc.head(5),'top10':vc.head(10),'top10pct':vc[vc.pct<=.10],'top20pct':vc[vc.pct<=.20]}
        for a in actual.itertuples():
          rr={'night':night,'phase':'selection' if night in selection else 'confirmation'}
          for sn,z in scopes.items():
            for rad in [50,100,250]:
              for mins in [10,20,30,60]:rr[f'{sn}_{rad}m_{mins}min']=hit(a,z,rad,mins)
          rows.append(rr)
      d=pd.DataFrame(rows);sel=d[d.phase=='selection'] if not d.empty else d;conf=d[d.phase=='confirmation'] if not d.empty else d;sm=summarize(sel.to_dict('records'));cm=summarize(conf.to_dict('records'));am=summarize(d.to_dict('records'))
      score=4*sm.get('top5_100m_10min_rate',0)+3*sm.get('top5_100m_20min_rate',0)+2*sm.get('top5_100m_30min_rate',0)+2*sm.get('top10pct_100m_30min_rate',0)+sm.get('top10_250m_30min_rate',0)
      variants.append({'alpha':alpha,'beta':beta,'selection_score':score,'selection':sm,'confirmation':cm,'all':am})
    variants.sort(key=lambda x:x['selection_score'],reverse=True);best=variants[0]
    summary={'status':'ok','profile_mode':'owner_personal','privacy':{'secret_hotspot_policy':'No raw road name, segment_id, coordinates, or hotspot ranking exported. Other-user profiles remain isolated.'},'protocol':{'actual_gpx_nights':len(nights),'selection_nights':len(selection),'confirmation_nights':len(confirmation),'candidate_unit':'actual owner GPX visits at actual passage times','strict_evaluation':'Only top-ranked route visits count; full-route coverage is not treated as model accuracy','distance_thresholds_m':[50,100,250],'time_thresholds_min':[10,20,30,60],'leakage_guard':'target-night captures hidden from training'},'score_definition':'route_score = spatial_occupancy^alpha * point_conditioned_time^beta','variants':variants,'best_selected_on_early_gpx_nights':best,'guardrail':'Retrospective owner-route replay only. It evaluates prioritization within routes actually driven, not discovery of roads never visited and not general-user accuracy.'}
    (r/'personal_route_score_strict_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
