from __future__ import annotations

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_zone_models_89night_pdca as b
from habuai.hardening.model_tournament import _as_jst
from run_advanced_time_models import _fetch_weather

MAX_UNITS=1500


def bounded_candidates(n,gpx,cap,ctx):
    start=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(hours=7)
    pv=gpx[gpx.entered_at<start]; pc=cap[cap.timestamp<start]
    if pc.empty:return pd.DataFrame(),pv,pc
    vc=pv.unit_id.value_counts(); cc=pc.unit_id.value_counts()
    units=sorted(set(vc.index)|set(cc.index))
    # Purely pre-night prescreen: favor verified capture units while retaining explored units.
    rank=[]
    for u in units:
        v=float(vc.get(u,0)); c=float(cc.get(u,0)); rank.append((u,(c+1.0)/(v+c+20.0),c,-v))
    units=[x[0] for x in sorted(rank,key=lambda q:(q[1],q[2],q[3]),reverse=True)[:MAX_UNITS]]
    rows=[]
    for uid in units:
        for t in b.slot_times(n):
            q=b.feat(uid,t,n,pv,pc,ctx); q.update({'unit_id':uid,'slot_time':t}); rows.append(q)
    return pd.DataFrame(rows),pv,pc


def main():
    root=Path(__file__).resolve().parents[1]; p=root/'data'/'processed'; r=root/'reports'; r.mkdir(exist_ok=True)
    md=pd.read_csv(p/'learning_10m_road.csv',low_memory=False); ev=pd.read_csv(p/'events_matched.csv',low_memory=False)
    md['entered_at']=_as_jst(md.entered_at); ev['timestamp']=_as_jst(ev.timestamp); md['night']=b.op_night(md.entered_at); ev['night']=b.op_night(ev.timestamp)
    ev['lat']=pd.to_numeric(ev.lat,errors='coerce'); ev['lon']=pd.to_numeric(ev.lon,errors='coerce')
    gpx0=md[md.learning_row_source=='gpx_visit'][['segment_id','entered_at','night']].dropna().copy()
    cap0=ev[(ev.species=='ハブ')&(ev.event_type=='捕獲')&ev.timestamp.notna()&ev.segment_id.notna()&ev.lat.notna()&ev.lon.notna()][['segment_id','timestamp','lat','lon','night']].copy()
    field_nights=sorted(ev.night.dropna().unique().tolist()); gps_nights=set(cap0.night.unique())
    seg=gpd.read_file(p/'road_segments_10m.geojson').to_crs('EPSG:6669'); seg['x']=seg.geometry.centroid.x; seg['y']=seg.geometry.centroid.y; segxy=seg[['segment_id','x','y']].drop_duplicates('segment_id')
    cent=gpd.GeoSeries(seg.geometry.centroid,crs='EPSG:6669').to_crs('EPSG:4326'); segll=pd.DataFrame({'segment_id':seg.segment_id.to_numpy(),'lat':cent.y.to_numpy(),'lon':cent.x.to_numpy()}).drop_duplicates('segment_id').set_index('segment_id')
    weather=_fetch_weather(str((pd.Timestamp(field_nights[0])-pd.Timedelta(days=3)).date()),str(pd.Timestamp(field_nights[-1]).date()),p/'cache'/'openmeteo_zone_pdca89.csv'); ctx=b.weather_contexts(weather,field_nights)
    results=[]; availability={}; progression={}
    for scheme,size in b.SCHEMES:
        gpx=b.assign_units(gpx0,segxy,size); cap=b.assign_units(cap0,segxy,size); members=b.unit_members(segxy,size); hist=b.build_observations(gpx,cap,ctx,field_nights)
        fcols=[c for c in hist.columns if c not in {'night','unit_id','slot','label','source'}]; ens=[]
        for idx,n in enumerate(field_nights,1):
            actual=cap[cap.night==n]
            if actual.empty:continue
            start=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(hours=7); pc0=cap[cap.timestamp<start]
            if len(pc0)<b.MIN_PRIOR_CAPTURES:continue
            cand,pv,pc=bounded_candidates(n,gpx,cap,ctx)
            if cand.empty:continue
            train=hist[hist.night<n]
            if not train.empty and train.label.nunique()>=2:
                m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.3)); m.fit(train[fcols],train.label.astype(int)); cand['score']=m.predict_proba(cand[fcols])[:,1]
            else:cand['score']=(cand.occupancy_proxy+1e-6)*(cand.global_time_prior+1e-6)*(cand.unit_time_prior+1e-6)
            cand=cand.sort_values('score',ascending=False).drop_duplicates(['unit_id','slot_time']).reset_index(drop=True); ens.append(idx)
            for count in b.COUNTS:
                top=cand.head(count)
                for a in actual.itertuples():
                    row={'scheme':scheme,'night_index':idx,'n_predictions':count}
                    for rad in b.RADII:
                        for mins in b.WINDOWS:row[f'hit_{rad}m_{mins}min']=b.hit(a,top,members,segll,rad,mins)
                    results.append(row)
        availability[scheme]={'strict_scored_nights':len(set(ens)),'strict_first_night_index':min(ens) if ens else None,'strict_last_night_index':max(ens) if ens else None}
    d=pd.DataFrame(results); overall=b.summarize(d)
    for scheme,_ in b.SCHEMES:
        ns=sorted(d[d.scheme==scheme].night_index.unique()) if not d.empty else []
        blocks=[list(x) for x in np.array_split(np.array(ns,dtype=int),3)] if ns else []
        progression[scheme]={f'block_{i}':b.summarize(d[(d.scheme==scheme)&(d.night_index.isin(v))]) for i,v in enumerate(blocks,1)}
    compare={}; best={}
    for count in b.COUNTS:
        compare[str(count)]={x['scheme']:x['coverage']['100m_10min'] for x in overall if x['n_predictions']==count}
        best[str(count)]=max(compare[str(count)],key=lambda k:(-1 if compare[str(count)][k] is None else compare[str(count)][k]))
    out={'status':'ok','method':'89-night PDCA walk-forward, bounded pre-night candidate prescreen, preserved segment plus 100/250/500/1000m zone x 10-minute models','field_evidence_nights_total':len(field_nights),'gps_timestamp_capture_events_total':int(len(cap0)),'gps_timestamp_capture_nights_total':int(cap0.night.nunique()),'protocol':{'cycle':'predict -> reveal -> score -> add verified night -> retrain -> next prediction','schemes':[x[0] for x in b.SCHEMES],'prediction_counts':b.COUNTS,'distance_m':b.RADII,'time_windows_min':b.WINDOWS,'slot_minutes':10,'min_prior_gps_captures':b.MIN_PRIOR_CAPTURES,'max_preselected_units':MAX_UNITS,'prescreen':'prior verified captures/visits only; no target-night data','no_future_gpx':True,'no_target_outcome_before_prediction':True},'model_availability':availability,'overall':overall,'progression_thirds':progression,'strict_100m_10min_comparison':compare,'best_scheme_by_prediction_count_100m_10min':best,'privacy':'aggregate metrics only; no dates, raw route/unit ids, coordinates, road names, or hotspot rankings exported','guardrail':'All 89 nights advance the PDCA clock, but strict road-time accuracy is scored only where GPS+timestamp capture outcomes exist. Historical GPX is never fabricated.'}
    (r/'zone_models_89night_pdca_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
