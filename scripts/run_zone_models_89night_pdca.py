from __future__ import annotations

import json, math
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from habuai.hardening.model_tournament import _as_jst
from run_advanced_time_models import _fetch_weather
from run_environmental_hazard_v2 import _prepare_weather
from run_joint_point_time_backtest_v2 import hav

SCHEMES=[('segment',None),('zone100',100),('zone250',250),('zone500',500),('zone1000',1000)]
COUNTS=[30,50,100]
RADII=[50,100,250]
WINDOWS=[10,20,30]
SLOT_MIN=10
MIN_PRIOR_CAPTURES=10


def op_night(ts): return (ts-pd.Timedelta(hours=7)).dt.date.astype(str)

def slot_times(n):
    d=pd.Timestamp(n,tz='Asia/Tokyo')
    return pd.date_range(d+pd.Timedelta(hours=18),d+pd.Timedelta(days=1,hours=7),freq='10min',inclusive='left')

def slot_minute(ts): return int((round((ts.hour*60+ts.minute)/10)*10)%1440)

def circ_minutes(a,b):
    d=abs((a-b).total_seconds())/60.0%1440.0
    return float(min(d,1440-d))

def weather_contexts(weather,nights):
    w=_prepare_weather(weather).copy(); w['time']=pd.to_datetime(w['time'])
    if getattr(w['time'].dt,'tz',None) is not None: w['time']=w['time'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    cols=['temperature_2m','relative_humidity_2m','rain_24h_mm','rain_48h_mm','hours_since_rain','fog_any_flag']
    out=[]
    for n in nights:
        t=pd.Timestamp(n)+pd.Timedelta(hours=18); i=(w.time-t).abs().idxmin(); row={'night':n}
        for c in cols: row[c]=float(w.loc[i,c]) if pd.notna(w.loc[i,c]) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index('night')

def assign_units(df,segxy,size):
    x=df.merge(segxy,on='segment_id',how='left'); x=x[x.x.notna()&x.y.notna()].copy()
    if size is None: x['unit_id']=x.segment_id.astype(str)
    else:
        x['zx']=(np.floor(x.x/size)*size).astype(int); x['zy']=(np.floor(x.y/size)*size).astype(int)
        x['unit_id']=x.zx.astype(str)+':'+x.zy.astype(str)
    return x

def unit_members(segxy,size):
    z=segxy.copy()
    if size is None: z['unit_id']=z.segment_id.astype(str)
    else:
        z['zx']=(np.floor(z.x/size)*size).astype(int); z['zy']=(np.floor(z.y/size)*size).astype(int)
        z['unit_id']=z.zx.astype(str)+':'+z.zy.astype(str)
    return {k:v.segment_id.to_numpy() for k,v in z.groupby('unit_id')}

def time_prior(cap,minute):
    if cap.empty:return 1e-3
    cm=(cap.timestamp.dt.hour*60+cap.timestamp.dt.minute).to_numpy(float); d=np.abs(cm-minute); d=np.minimum(d,1440-d)
    return float(np.mean(np.exp(-.5*(d/60.)**2))+1e-6)

def feat(uid,t,night,pastv,pastc,ctx):
    zv=pastv[pastv.unit_id==uid]; zc=pastc[pastc.unit_id==uid]; visits=len(zv); captures=len(zc)
    last=9999. if zc.empty else max(0.,(t-zc.timestamp.max()).total_seconds()/86400.)
    minute=t.hour*60+t.minute; hr=minute/60.; doy=t.dayofyear
    r={'unit_visits':visits,'unit_captures':captures,'capture30':int((zc.timestamp>=t-pd.Timedelta(days=30)).sum()),'capture90':int((zc.timestamp>=t-pd.Timedelta(days=90)).sum()),'days_since_capture':last,'detection_proxy':(captures+1.)/(visits+12.),'occupancy_proxy':(captures+1.)/(captures+visits+20.),'sin_hour':math.sin(2*math.pi*hr/24),'cos_hour':math.cos(2*math.pi*hr/24),'sin_doy':math.sin(2*math.pi*doy/365.25),'cos_doy':math.cos(2*math.pi*doy/365.25),'global_time_prior':time_prior(pastc,minute),'unit_time_prior':time_prior(zc,minute)}
    if night in ctx.index:
        for c in ctx.columns:r[c]=float(ctx.loc[night,c]) if pd.notna(ctx.loc[night,c]) else np.nan
    return r

def build_observations(gpx,cap,ctx,all_nights):
    rows=[]
    for n in all_nights:
        start=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(hours=7); pastv=gpx[gpx.entered_at<start]; pastc=cap[cap.timestamp<start]
        seen=gpx[gpx.night==n].copy(); acts=cap[cap.night==n].copy()
        if not seen.empty:
            seen['slot']=seen.entered_at.map(slot_minute); seen=seen.drop_duplicates(['unit_id','slot'])
            acts2=acts.copy(); acts2['slot']=acts2.timestamp.map(slot_minute); pos=set(zip(acts2.unit_id,acts2.slot))
            for z in seen.itertuples():
                t=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(minutes=int(z.slot));
                if int(z.slot)<420:t+=pd.Timedelta(days=1)
                q=feat(z.unit_id,t,n,pastv,pastc,ctx); q.update({'night':n,'unit_id':z.unit_id,'slot':int(z.slot),'label':int((z.unit_id,int(z.slot)) in pos),'source':'gpx'}) ; rows.append(q)
        # Every verified capture becomes a positive anchor for later nights, even if that night had no GPX track.
        for a in acts.itertuples():
            sl=slot_minute(a.timestamp); q=feat(a.unit_id,a.timestamp,n,pastv,pastc,ctx); q.update({'night':n,'unit_id':a.unit_id,'slot':sl,'label':1,'source':'capture_anchor'}); rows.append(q)
    return pd.DataFrame(rows)

def candidates(n,gpx,cap,ctx):
    start=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(hours=7); pv=gpx[gpx.entered_at<start]; pc=cap[cap.timestamp<start]
    units=sorted(set(pv.unit_id)|set(pc.unit_id)); rows=[]
    for uid in units:
        for t in slot_times(n):
            q=feat(uid,t,n,pv,pc,ctx); q.update({'unit_id':uid,'slot_time':t}); rows.append(q)
    return pd.DataFrame(rows),pv,pc

def hit(a,top,members,segll,rad,mins):
    for c in top.itertuples():
        if circ_minutes(a.timestamp,c.slot_time)>mins:continue
        ids=members.get(c.unit_id); pts=segll.reindex(ids).dropna() if ids is not None else pd.DataFrame()
        if pts.empty:continue
        if float(np.min(hav(a.lat,a.lon,pts.lat.to_numpy(float),pts.lon.to_numpy(float))))<=rad:return 1
    return 0

def summarize(d):
    out=[]
    for scheme in [x[0] for x in SCHEMES]:
        for count in COUNTS:
            z=d[(d.scheme==scheme)&(d.n_predictions==count)]
            item={'scheme':scheme,'n_predictions':count,'eligible_capture_events':int(len(z)),'eligible_nights':int(z.night_index.nunique()) if not z.empty else 0,'coverage':{}}
            for rad in RADII:
                for mins in WINDOWS:item['coverage'][f'{rad}m_{mins}min']=float(z[f'hit_{rad}m_{mins}min'].mean()) if not z.empty else None
            out.append(item)
    return out

def main():
    root=Path(__file__).resolve().parents[1]; p=root/'data'/'processed'; r=root/'reports'; r.mkdir(exist_ok=True)
    md=pd.read_csv(p/'learning_10m_road.csv',low_memory=False); ev=pd.read_csv(p/'events_matched.csv',low_memory=False)
    md['entered_at']=_as_jst(md.entered_at); ev['timestamp']=_as_jst(ev.timestamp); md['night']=op_night(md.entered_at); ev['night']=op_night(ev.timestamp)
    ev['lat']=pd.to_numeric(ev.lat,errors='coerce'); ev['lon']=pd.to_numeric(ev.lon,errors='coerce')
    gpx0=md[md.learning_row_source=='gpx_visit'][['segment_id','entered_at','night']].dropna().copy()
    cap0=ev[(ev.species=='ハブ')&(ev.event_type=='捕獲')&ev.timestamp.notna()&ev.segment_id.notna()&ev.lat.notna()&ev.lon.notna()][['segment_id','timestamp','lat','lon','night']].copy()
    field_nights=sorted(ev.night.dropna().unique().tolist())
    seg=gpd.read_file(p/'road_segments_10m.geojson').to_crs('EPSG:6669'); seg['x']=seg.geometry.centroid.x; seg['y']=seg.geometry.centroid.y; segxy=seg[['segment_id','x','y']].drop_duplicates('segment_id')
    # avoid geographic-CRS centroid warning: use projected centroid transformed back to lat/lon
    cent=gpd.GeoSeries(seg.geometry.centroid,crs='EPSG:6669').to_crs('EPSG:4326'); segll=pd.DataFrame({'segment_id':seg.segment_id.to_numpy(),'lat':cent.y.to_numpy(),'lon':cent.x.to_numpy()}).drop_duplicates('segment_id').set_index('segment_id')
    weather=_fetch_weather(str((pd.Timestamp(field_nights[0])-pd.Timedelta(days=3)).date()),str(pd.Timestamp(field_nights[-1]).date()),p/'cache'/'openmeteo_zone_pdca89.csv'); ctx=weather_contexts(weather,field_nights)
    results=[]; model_availability={}; progression={}
    for scheme,size in SCHEMES:
        gpx=assign_units(gpx0,segxy,size); cap=assign_units(cap0,segxy,size); members=unit_members(segxy,size); hist=build_observations(gpx,cap,ctx,field_nights)
        fcols=[c for c in hist.columns if c not in {'night','unit_id','slot','label','source'}]
        eligible_nights=[]
        for idx,n in enumerate(field_nights,1):
            train=hist[hist.night<n]; actual=cap[cap.night==n]
            cand,pv,pc=candidates(n,gpx,cap,ctx)
            # Strict pre-night prediction requires enough prior GPS captures and at least one known road unit.
            if len(pc)<MIN_PRIOR_CAPTURES or cand.empty or actual.empty:continue
            if not train.empty and train.label.nunique()>=2:
                m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.3)); m.fit(train[fcols],train.label.astype(int)); cand['score']=m.predict_proba(cand[fcols])[:,1]
            else:
                # Pre-GPX fallback uses only already-known capture/visit priors, never target-night outcome.
                cand['score']=(cand.occupancy_proxy+1e-6)*(cand.global_time_prior+1e-6)*(cand.unit_time_prior+1e-6)
            cand=cand.sort_values('score',ascending=False).drop_duplicates(['unit_id','slot_time']).reset_index(drop=True); eligible_nights.append(idx)
            for count in COUNTS:
                top=cand.head(count)
                for a in actual.itertuples():
                    row={'scheme':scheme,'night_index':idx,'n_predictions':count}
                    for rad in RADII:
                        for mins in WINDOWS:row[f'hit_{rad}m_{mins}min']=hit(a,top,members,segll,rad,mins)
                    results.append(row)
        model_availability[scheme]={'strict_scored_nights':len(set(eligible_nights)),'strict_first_night_index':min(eligible_nights) if eligible_nights else None,'strict_last_night_index':max(eligible_nights) if eligible_nights else None}
    d=pd.DataFrame(results); overall=summarize(d)
    # Progression thirds are based on each scheme's own chronologically eligible scored nights.
    for scheme,_ in SCHEMES:
        ns=sorted(d[d.scheme==scheme].night_index.unique()) if not d.empty else []
        blocks=[list(x) for x in np.array_split(np.array(ns,dtype=int),3)] if ns else []
        progression[scheme]={}
        for i,b in enumerate(blocks,1): progression[scheme][f'block_{i}']=summarize(d[(d.scheme==scheme)&(d.night_index.isin(b))])
    target='100m_10min'; compare={}
    for count in COUNTS:
        compare[str(count)]={x['scheme']:x['coverage'][target] for x in overall if x['n_predictions']==count}
    best={str(count):max(compare[str(count)],key=lambda k:(-1 if compare[str(count)][k] is None else compare[str(count)][k])) for count in COUNTS}
    out={'status':'ok','method':'89-night outer PDCA walk-forward for preserved 10m-segment and 100/250/500/1000m direct zone x 10-minute models','field_evidence_nights_total':len(field_nights),'gps_timestamp_capture_events_total':int(len(cap0)),'gps_timestamp_capture_nights_total':int(cap0.night.nunique()),'protocol':{'cycle':'predict using prior nights only -> reveal target result -> score -> add verified capture/GPX rows -> retrain for next night','schemes':[x[0] for x in SCHEMES],'prediction_counts':COUNTS,'distance_m':RADII,'time_windows_min':WINDOWS,'slot_minutes':10,'min_prior_gps_captures':MIN_PRIOR_CAPTURES,'candidate_units':'verified prior GPX-traversed or prior capture road units only','no_future_gpx':True,'no_target_outcome_before_prediction':True},'model_availability':model_availability,'overall':overall,'progression_thirds':progression,'strict_100m_10min_comparison':compare,'best_scheme_by_prediction_count_100m_10min':best,'privacy':'aggregate metrics only; no dates, raw route/unit ids, coordinates, road names, or hotspot rankings exported','guardrail':'All 89 field-evidence nights advance the PDCA clock. Strict road-time accuracy is scored only on nights with GPS+timestamp captures and enough prior verified road/capture history; missing historical GPX is never fabricated.'}
    (r/'zone_models_89night_pdca_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
