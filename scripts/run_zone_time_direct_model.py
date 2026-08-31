from __future__ import annotations

import json
import math
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

ZONE_SIZES_M = [100, 250, 500, 1000]
COUNTS = [30, 50, 100]
WINDOWS_MIN = [10, 20, 30]
RADII_M = [100, 250]
SLOT_MIN = 10
START_HOUR = 18
END_HOUR = 7


def slot_times(night: str) -> pd.DatetimeIndex:
    d = pd.Timestamp(night, tz='Asia/Tokyo')
    return pd.date_range(d + pd.Timedelta(hours=START_HOUR), d + pd.Timedelta(days=1, hours=END_HOUR), freq='10min', inclusive='left')


def op_night(ts: pd.Series) -> pd.Series:
    return (ts - pd.Timedelta(hours=7)).dt.date.astype(str)


def slot_minute(ts: pd.Timestamp) -> int:
    return int((round((ts.hour * 60 + ts.minute) / SLOT_MIN) * SLOT_MIN) % 1440)


def circ_minutes(a: pd.Timestamp, b: pd.Timestamp) -> float:
    d = abs((a - b).total_seconds()) / 60.0 % 1440.0
    return float(min(d, 1440.0 - d))


def weather_contexts(weather: pd.DataFrame, nights: list[str]) -> pd.DataFrame:
    w = _prepare_weather(weather).copy()
    w['time'] = pd.to_datetime(w['time'])
    if getattr(w['time'].dt, 'tz', None) is not None:
        w['time'] = w['time'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    cols = ['temperature_2m','relative_humidity_2m','rain_24h_mm','rain_48h_mm','hours_since_rain','fog_any_flag']
    out=[]
    for n in nights:
        t=pd.Timestamp(n)+pd.Timedelta(hours=18)
        i=(w['time']-t).abs().idxmin()
        row={'night':n}
        for c in cols: row[c]=float(w.loc[i,c]) if pd.notna(w.loc[i,c]) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index('night')


def add_zone(df: pd.DataFrame, seg_xy: pd.DataFrame, size: int) -> pd.DataFrame:
    x=df.merge(seg_xy,on='segment_id',how='left')
    x=x[x.x.notna() & x.y.notna()].copy()
    x['zx']=(np.floor(x.x/size)*size).astype(int)
    x['zy']=(np.floor(x.y/size)*size).astype(int)
    x['zone_id']=x.zx.astype(str)+':'+x.zy.astype(str)
    return x


def priors(past_vis: pd.DataFrame, past_cap: pd.DataFrame, zone_id: str, t: pd.Timestamp) -> dict:
    zv=past_vis[past_vis.zone_id==zone_id]
    zc=past_cap[past_cap.zone_id==zone_id]
    visits=int(len(zv)); captures=int(len(zc))
    last_days=9999.0 if zc.empty else max(0.0,(t-zc.timestamp.max()).total_seconds()/86400.0)
    recent30=int((zc.timestamp>=t-pd.Timedelta(days=30)).sum())
    recent90=int((zc.timestamp>=t-pd.Timedelta(days=90)).sum())
    detection=(captures+1.0)/(visits+12.0)
    occ=(captures+1.0)/(captures+visits+20.0)
    return {'zone_visits':visits,'zone_captures':captures,'zone_capture30':recent30,'zone_capture90':recent90,'days_since_zone_capture':last_days,'detection_proxy':detection,'occupancy_proxy':occ}


def time_prior(past_cap: pd.DataFrame, minute: int) -> float:
    if past_cap.empty:return 1e-3
    cm=(past_cap.timestamp.dt.hour*60+past_cap.timestamp.dt.minute).to_numpy(float)
    d=np.abs(cm-minute); d=np.minimum(d,1440-d)
    return float(np.mean(np.exp(-0.5*(d/60.0)**2))+1e-6)


def feature_row(zone_id: str, t: pd.Timestamp, night: str, past_vis: pd.DataFrame, past_cap: pd.DataFrame, contexts: pd.DataFrame) -> dict:
    p=priors(past_vis,past_cap,zone_id,t)
    minute=t.hour*60+t.minute; hr=minute/60.0; doy=t.dayofyear
    zpast=past_cap[past_cap.zone_id==zone_id]
    p.update({
        'sin_hour':math.sin(2*math.pi*hr/24),'cos_hour':math.cos(2*math.pi*hr/24),
        'sin_doy':math.sin(2*math.pi*doy/365.25),'cos_doy':math.cos(2*math.pi*doy/365.25),
        'global_time_prior':time_prior(past_cap,minute),'zone_time_prior':time_prior(zpast,minute),
    })
    if night in contexts.index:
        for c in contexts.columns:p[c]=float(contexts.loc[night,c]) if pd.notna(contexts.loc[night,c]) else np.nan
    return p


def build_history_rows(gpx: pd.DataFrame, cap: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    nights=sorted(set(gpx.night))
    for n in nights:
        start=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(hours=7)
        pastv=gpx[gpx.entered_at<start]; pastc=cap[cap.timestamp<start]
        if pastv.empty: continue
        seen=gpx[gpx.night==n].copy()
        seen['slot']=seen.entered_at.map(slot_minute)
        seen=seen.drop_duplicates(['zone_id','slot'])
        acts=cap[cap.night==n].copy(); acts['slot']=acts.timestamp.map(slot_minute)
        positives=set(zip(acts.zone_id,acts.slot))
        for z in seen.itertuples():
            t=pd.Timestamp(n,tz='Asia/Tokyo')+pd.Timedelta(minutes=int(z.slot))
            if int(z.slot)<7*60:t+=pd.Timedelta(days=1)
            r=feature_row(z.zone_id,t,n,pastv,pastc,contexts); r.update({'night':n,'zone_id':z.zone_id,'slot':int(z.slot),'label':int((z.zone_id,int(z.slot)) in positives)})
            rows.append(r)
        # Positive anchors ensure captures not exactly represented by a GPX sample are retained.
        existing={(x['zone_id'],x['slot']) for x in rows if x['night']==n}
        for a in acts.itertuples():
            key=(a.zone_id,int(a.slot))
            if key in existing: continue
            t=a.timestamp
            r=feature_row(a.zone_id,t,n,pastv,pastc,contexts); r.update({'night':n,'zone_id':a.zone_id,'slot':int(a.slot),'label':1}); rows.append(r)
    return pd.DataFrame(rows)


def candidate_rows(night: str, gpx: pd.DataFrame, cap: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    start=pd.Timestamp(night,tz='Asia/Tokyo')+pd.Timedelta(hours=7)
    pastv=gpx[gpx.entered_at<start]; pastc=cap[cap.timestamp<start]
    zones=sorted(pastv.zone_id.unique())
    rows=[]
    for z in zones:
        for t in slot_times(night):
            r=feature_row(z,t,night,pastv,pastc,contexts); r.update({'zone_id':z,'slot_time':t}); rows.append(r)
    return pd.DataFrame(rows)


def event_hit(a, cand: pd.DataFrame, zone_members: dict[str,np.ndarray], seg_latlon: pd.DataFrame, rad: int, mins: int) -> int:
    for c in cand.itertuples():
        if circ_minutes(a.timestamp,c.slot_time)>mins: continue
        members=zone_members.get(c.zone_id)
        if members is None or len(members)==0: continue
        pts=seg_latlon.reindex(members).dropna()
        if pts.empty: continue
        d=hav(a.lat,a.lon,pts.lat.to_numpy(float),pts.lon.to_numpy(float))
        if float(np.min(d))<=rad:return 1
    return 0


def main():
    root=Path(__file__).resolve().parents[1]; p=root/'data'/'processed'; r=root/'reports'; r.mkdir(exist_ok=True)
    md=pd.read_csv(p/'learning_10m_road.csv',low_memory=False); ev=pd.read_csv(p/'events_matched.csv',low_memory=False)
    md['entered_at']=_as_jst(md.entered_at); ev['timestamp']=_as_jst(ev.timestamp); md['night']=op_night(md.entered_at)
    ev['lat']=pd.to_numeric(ev.lat,errors='coerce'); ev['lon']=pd.to_numeric(ev.lon,errors='coerce')
    gpx0=md[md.learning_row_source=='gpx_visit'][['segment_id','entered_at','night']].dropna().copy()
    cap0=ev[(ev.species=='ハブ')&(ev.event_type=='捕獲')&ev.timestamp.notna()&ev.segment_id.notna()&ev.lat.notna()&ev.lon.notna()][['segment_id','timestamp','lat','lon']].copy(); cap0['night']=op_night(cap0.timestamp)
    seg=gpd.read_file(p/'road_segments_10m.geojson').to_crs('EPSG:6669'); seg['x']=seg.geometry.centroid.x; seg['y']=seg.geometry.centroid.y
    segxy=seg[['segment_id','x','y']].drop_duplicates('segment_id')
    ll=seg.to_crs('EPSG:4326'); ll['lat']=ll.geometry.centroid.y; ll['lon']=ll.geometry.centroid.x; segll=ll[['segment_id','lat','lon']].drop_duplicates('segment_id').set_index('segment_id')
    alln=sorted(set(gpx0.night)|set(cap0.night)); ws=str((cap0.timestamp.min()-pd.Timedelta(days=3)).date()); we=str(cap0.timestamp.max().date())
    weather=_fetch_weather(ws,we,p/'cache'/'openmeteo_zone_time_direct.csv'); contexts=weather_contexts(weather,alln)
    results=[]; summaries=[]
    for size in ZONE_SIZES_M:
        gpx=add_zone(gpx0,segxy,size); cap=add_zone(cap0,segxy,size)
        zone_members=segxy.assign(zx=(np.floor(segxy.x/size)*size).astype(int),zy=(np.floor(segxy.y/size)*size).astype(int)); zone_members['zone_id']=zone_members.zx.astype(str)+':'+zone_members.zy.astype(str); zm={k:v.segment_id.to_numpy() for k,v in zone_members.groupby('zone_id')}
        hist=build_history_rows(gpx,cap,contexts); nights=sorted(gpx.night.unique()); split=max(4,int(np.floor(len(nights)*.60))); selection=set(nights[:split]); confirmation=set(nights[split:])
        feature_cols=[c for c in hist.columns if c not in {'night','zone_id','slot','label'}]
        for n in nights:
            train=hist[hist.night<n]; actual=cap[cap.night==n]
            if actual.empty or train.empty or train.label.nunique()<2: continue
            cand=candidate_rows(n,gpx,cap,contexts)
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.3))
            model.fit(train[feature_cols],train.label.astype(int)); cand['p']=model.predict_proba(cand[feature_cols])[:,1]
            cand=cand.sort_values('p',ascending=False).drop_duplicates(['zone_id','slot_time']).reset_index(drop=True)
            phase='selection' if n in selection else 'confirmation'
            for count in COUNTS:
                top=cand.head(count)
                for a in actual.itertuples():
                    row={'zone_size_m':size,'night':n,'phase':phase,'n_predictions':count}
                    for rad in RADII_M:
                        for mins in WINDOWS_MIN: row[f'hit_{rad}m_{mins}min']=event_hit(a,top,zm,segll,rad,mins)
                    results.append(row)
        d=pd.DataFrame([x for x in results if x['zone_size_m']==size])
        for phase in ['selection','confirmation','all']:
            q=d if phase=='all' else d[d.phase==phase]
            for count in COUNTS:
                z=q[q.n_predictions==count]; item={'zone_size_m':size,'phase':phase,'n_predictions':count,'eligible_captures':int(len(z)),'nights_scored':int(z.night.nunique()) if not z.empty else 0,'coverage':{}}
                for rad in RADII_M:
                    for mins in WINDOWS_MIN:
                        k=f'hit_{rad}m_{mins}min'; item['coverage'][f'{rad}m_{mins}min']=float(z[k].mean()) if not z.empty else None
                summaries.append(item)
    # Choose scale only from selection nights, prioritizing strict 100m/10min at 30,50,100 candidates.
    scores={}
    for size in ZONE_SIZES_M:
        ss=[x for x in summaries if x['zone_size_m']==size and x['phase']=='selection']; by={x['n_predictions']:x for x in ss}
        def g(n,k):
            v=by.get(n,{}).get('coverage',{}).get(k); return 0.0 if v is None else float(v)
        scores[str(size)]=5*g(30,'100m_10min')+4*g(50,'100m_10min')+3*g(100,'100m_10min')+2*g(50,'100m_20min')+g(100,'250m_30min')
    best=int(max(scores,key=scores.get)); conf=[x for x in summaries if x['zone_size_m']==best and x['phase']=='confirmation']
    target90={}
    for key in ['100m_10min','100m_20min','100m_30min','250m_10min','250m_20min','250m_30min']:
        reached=[x['n_predictions'] for x in conf if x['coverage'].get(key) is not None and x['coverage'][key]>=.90]; target90[key]=min(reached) if reached else None
    out={'status':'ok','model':'zone_time_direct_experimental','legacy_model_preserved':True,'protocol':{'zone_sizes_m':ZONE_SIZES_M,'prediction_counts':COUNTS,'slot_minutes':10,'actual_gpx_nights':len(sorted(gpx0.night.unique())),'selection_rule':'zone scale selected only on early GPX nights; later confirmation untouched','candidate_zones':'owner-traversed zones known before target night only','privacy':'aggregate metrics only; no raw zone ids, road names, coordinates or hotspot rankings exported'},'feature_groups':['zone occupancy proxy','personal detection proxy','zone capture recency 30/90d','global and zone-specific time prior','season cyclic terms','18:00 temperature/RH/rain24/rain48/hours-since-rain/fog'],'selection_scores':scores,'best_zone_size_selected_on_early_nights':best,'results':summaries,'best_confirmation':conf,'smallest_confirmation_prediction_count_reaching_90pct':target90,'guardrail':'Experimental additional model. It does not replace the existing segment-based model. Zone size can improve ranking efficiency but may lose fine spatial detail.'}
    (r/'zone_time_direct_model_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
