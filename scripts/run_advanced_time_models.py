from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_full_capture_time_backtest import _as_jst, _circular_minutes, _haversine_m

SLOTS = np.arange(0, 1440, 30, dtype=int)


def _kde_curve(times_h, weights, bw):
    if len(times_h) == 0:
        return np.ones(len(SLOTS), dtype=float)
    out=[]
    for minute in SLOTS:
        h=minute/60.0
        d=np.abs(times_h-h); d=np.minimum(d,24-d)
        out.append(float((weights*np.exp(-.5*(d/float(bw))**2)).sum()))
    a=np.asarray(out,float)
    if float(a.max())>0:a=a/float(a.max())
    return a


def _season_distance_days(ts, cutoff):
    a=ts.dt.dayofyear.to_numpy(float); b=float(cutoff.dayofyear)
    d=np.abs(a-b); return np.minimum(d,365-d)


def _hier_curve(past, cutoff, lat, lon, radius, bw, k_local, k_season, tau_days):
    h=past.timestamp.dt.hour.to_numpy(float)+past.timestamp.dt.minute.to_numpy(float)/60
    age=(cutoff-past.timestamp).dt.total_seconds().to_numpy(float)/86400.0
    base_w=np.exp(-np.maximum(age,0)/float(tau_days)) if tau_days else np.ones(len(past))
    global_c=_kde_curve(h,base_w,bw)
    sd=_season_distance_days(past.timestamp,cutoff); sm=sd<=35
    season_c=_kde_curve(h[sm],base_w[sm],bw) if sm.any() else global_c
    dist=_haversine_m(lat,lon,past.lat.to_numpy(float),past.lon.to_numpy(float)); lm=dist<=float(radius)
    local_c=_kde_curve(h[lm],base_w[lm],bw) if lm.any() else global_c
    wl=float(lm.sum())/(float(lm.sum())+float(k_local))
    ws=float(sm.sum())/(float(sm.sum())+float(k_season))
    # local first, then season borrows the remaining mass, then global.
    c=wl*local_c+(1-wl)*(ws*season_c+(1-ws)*global_c)
    return c/(float(c.max()) if float(c.max())>0 else 1.0)


def _score_hier(cap,nights,params,min_prior=10):
    rows=[]
    for night in nights:
        start=pd.Timestamp(str(night),tz='Asia/Tokyo')+pd.Timedelta(hours=7); end=start+pd.Timedelta(days=1)
        actual=cap[(cap.timestamp>=start)&(cap.timestamp<end)]; past=cap[cap.timestamp<start]
        if actual.empty or len(past)<min_prior: continue
        for a in actual.itertuples():
            c=_hier_curve(past,start,a.lat,a.lon,**params); pred=int(SLOTS[int(np.argmax(c))])
            amin=int(a.timestamp.hour)*60+int(a.timestamp.minute); err=_circular_minutes(amin,pred)
            rows.append({'night':str(night),'actual_minute':amin,'predicted_minute':pred,'error_min':err})
    return pd.DataFrame(rows)


def _metrics(x):
    if x.empty:return {'eligible':0,'nights_scored':0}
    e=pd.to_numeric(x.error_min,errors='coerce').dropna(); n=len(e)
    out={'eligible':int(n),'nights_scored':int(x.night.nunique()),'median_error_min':float(e.median()),'mean_error_min':float(e.mean())}
    for m in (30,60,90,120):
        hit=int((e<=m).sum());out[f'within_{m}m_hits']=hit;out[f'within_{m}m_rate']=hit/n
    out['selection_score']=4*out['within_30m_rate']+3*out['within_60m_rate']+2*out['within_90m_rate']+out['within_120m_rate']-out['median_error_min']/720.0
    return out


def _fetch_weather(start_date,end_date,cache):
    if cache.exists():
        w=pd.read_csv(cache);w['time']=pd.to_datetime(w.time,errors='coerce');return w
    hourly='temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,weather_code,surface_pressure,cloud_cover'
    q=urllib.parse.urlencode({'latitude':28.1456,'longitude':129.3200,'start_date':start_date,'end_date':end_date,'hourly':hourly,'timezone':'Asia/Tokyo'})
    url='https://archive-api.open-meteo.com/v1/archive?'+q
    with urllib.request.urlopen(url,timeout=60) as r: obj=json.loads(r.read().decode('utf-8'))
    h=obj['hourly'];w=pd.DataFrame(h);w['time']=pd.to_datetime(w.time,errors='coerce')
    cache.parent.mkdir(parents=True,exist_ok=True);w.to_csv(cache,index=False);return w


def _weather_30m(w):
    w=w.copy().set_index('time').sort_index(); idx=pd.date_range(w.index.min(),w.index.max(),freq='30min')
    x=w.reindex(w.index.union(idx)).sort_index()
    numeric=[c for c in x.columns if c!='weather_code']
    x[numeric]=x[numeric].interpolate(method='time').ffill().bfill()
    x['weather_code']=pd.to_numeric(x['weather_code'],errors='coerce').ffill().bfill()
    return x.reindex(idx)


def _slot_times(night):
    start=pd.Timestamp(str(night))+pd.Timedelta(hours=7)
    return pd.DatetimeIndex([start+pd.Timedelta(minutes=int(m)) for m in SLOTS])


def _local_prior_features(past,cutoff,lat,lon):
    if past.empty:return np.ones(len(SLOTS)),np.ones(len(SLOTS))
    h=past.timestamp.dt.hour.to_numpy(float)+past.timestamp.dt.minute.to_numpy(float)/60
    age=(cutoff-past.timestamp).dt.total_seconds().to_numpy(float)/86400.0;rw=np.exp(-np.maximum(age,0)/120.0)
    dist=_haversine_m(lat,lon,past.lat.to_numpy(float),past.lon.to_numpy(float))
    g=_kde_curve(h,rw,1.5)
    lm=dist<=3000
    l=_kde_curve(h[lm],rw[lm],1.5) if lm.any() else g
    return g,l


def _risk_rows_for_night(cap,night,weather):
    start=pd.Timestamp(str(night),tz='Asia/Tokyo')+pd.Timedelta(hours=7);end=start+pd.Timedelta(days=1)
    actual=cap[(cap.timestamp>=start)&(cap.timestamp<end)].copy();past=cap[cap.timestamp<start].copy()
    if actual.empty or len(past)<10:return pd.DataFrame()
    # weather index is timezone-naive local time.
    times=pd.DatetimeIndex([pd.Timestamp(str(night))+pd.Timedelta(hours=7,minutes=int(m)) for m in SLOTS])
    ww=weather.reindex(times,method='nearest').copy()
    rows=[]
    for a in actual.itertuples():
        gp,lp=_local_prior_features(past,start,a.lat,a.lon); actual_min=int(a.timestamp.hour)*60+int(a.timestamp.minute)
        positive_idx=int(np.argmin([_circular_minutes(actual_min,m) for m in SLOTS]))
        for i,(minute,t) in enumerate(zip(SLOTS,times)):
            wr=ww.iloc[i]
            hr=minute/60.0;doy=int(t.dayofyear)
            rows.append({'night':str(night),'event_id':str(getattr(a,'canonical_id',''))+':'+str(a.Index if hasattr(a,'Index') else actual_min),
                'slot_minute':int(minute),'label':int(i==positive_idx),'sin_hour':math.sin(2*math.pi*hr/24),'cos_hour':math.cos(2*math.pi*hr/24),
                'sin_doy':math.sin(2*math.pi*doy/365.25),'cos_doy':math.cos(2*math.pi*doy/365.25),
                'temperature_2m':wr.get('temperature_2m',np.nan),'relative_humidity_2m':wr.get('relative_humidity_2m',np.nan),
                'dew_point_2m':wr.get('dew_point_2m',np.nan),'precipitation':wr.get('precipitation',np.nan),'weather_code':wr.get('weather_code',np.nan),
                'surface_pressure':wr.get('surface_pressure',np.nan),'cloud_cover':wr.get('cloud_cover',np.nan),'global_prior':gp[i],'local3km_prior':lp[i],
                'actual_minute':actual_min})
    return pd.DataFrame(rows)


def _hazard_walkforward(cap,nights,weather):
    all_risk=pd.concat([_risk_rows_for_night(cap,n,weather) for n in nights],ignore_index=True)
    feats=['sin_hour','cos_hour','sin_doy','cos_doy','temperature_2m','relative_humidity_2m','dew_point_2m','precipitation','weather_code','surface_pressure','cloud_cover','global_prior','local3km_prior']
    preds=[]
    for night in nights:
        te=all_risk[all_risk.night==str(night)].copy();tr=all_risk[all_risk.night.astype(str)<str(night)].copy()
        if te.empty or tr.label.sum()<10:continue
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.2))
        model.fit(tr[feats],tr.label.astype(int))
        te['p']=model.predict_proba(te[feats])[:,1]
        for eid,g in te.groupby('event_id'):
            best=g.loc[g.p.idxmax()];actual=int(g.actual_minute.iloc[0]);pred=int(best.slot_minute);err=_circular_minutes(actual,pred)
            preds.append({'night':str(night),'event_id':eid,'actual_minute':actual,'predicted_minute':pred,'error_min':err})
    return pd.DataFrame(preds),all_risk


def main():
    root=Path(__file__).resolve().parents[1];p=root/'data'/'processed';r=root/'reports';r.mkdir(exist_ok=True)
    ev=pd.read_csv(p/'events_matched.csv',low_memory=False);ev['timestamp']=_as_jst(ev.timestamp);ev['lat']=pd.to_numeric(ev.lat,errors='coerce');ev['lon']=pd.to_numeric(ev.lon,errors='coerce')
    cap=ev[(ev.species=='ハブ')&(ev.event_type=='捕獲')&ev.lat.notna()&ev.lon.notna()&ev.timestamp.notna()].copy();cap['night']=(cap.timestamp-pd.Timedelta(hours=7)).dt.date.astype(str)
    nights=sorted(cap.night.unique());split=max(4,int(np.floor(len(nights)*.70)));sel=nights[:split];conf=nights[split:]

    variants=[]
    for radius in [1000,2000,3000]:
      for bw in [.75,1.0,1.5,2.0]:
       for kl in [2,5,10]:
        for ks in [5,10,20]:
         for tau in [120,240,None]:
          pa={'radius':radius,'bw':bw,'k_local':kl,'k_season':ks,'tau_days':tau};m=_metrics(_score_hier(cap,sel,pa));variants.append({**pa,**m})
    tv=pd.DataFrame(variants).sort_values(['selection_score','within_60m_rate','median_error_min'],ascending=[False,False,True]).reset_index(drop=True);tv.to_csv(r/'hierarchical_time_tournament.csv',index=False)
    bp={k:tv.iloc[0][k] for k in ['radius','bw','k_local','k_season','tau_days']};bp['tau_days']=None if pd.isna(bp['tau_days']) else float(bp['tau_days'])
    h_all=_score_hier(cap,nights,bp);h_conf=_score_hier(cap,conf,bp);h_all.to_csv(r/'hierarchical_time_predictions.csv',index=False)

    cache=p/'cache'/'openmeteo_historical_2026-05_08.csv';w=_fetch_weather(str(cap.timestamp.min().date()),str(cap.timestamp.max().date()),cache);w30=_weather_30m(w)
    hazard,risk=_hazard_walkforward(cap,nights,w30);hazard.to_csv(r/'environmental_hazard_time_predictions.csv',index=False);risk.to_csv(r/'environmental_hazard_risksets.csv',index=False)
    hz_conf=hazard[hazard.night.isin(set(conf))]

    summary={'status':'ok','common_protocol':{'gps_timestamp_capture_events':int(len(cap)),'gps_timestamp_capture_nights':int(len(nights)),'selection_nights':int(len(sel)),'confirmation_nights':int(len(conf)),'rollover':'07:00 Asia/Tokyo','min_prior_captures':10,'slot_minutes':30},
      'hierarchical_empirical_bayes':{'description':'partial pooling of global, seasonal +-35d, and local capture-time KDE; local/season weights shrink automatically when evidence is sparse','best_selected_on_early70pct':bp,'all_walkforward':_metrics(h_all),'frozen_confirmation':_metrics(h_conf)},
      'environmental_hazard':{'description':'discrete 30-min hazard ranking using clock, season, temperature, humidity, dew point, precipitation, weather code, pressure, cloud cover, and leakage-safe local/global capture-time priors; historical hourly weather from Open-Meteo archive','all_walkforward':_metrics(hazard),'frozen_confirmation':_metrics(hz_conf)},
      'guardrail':'Environmental hazard uses historical archive weather only. Same-night capture outcomes are hidden at 07:00. The later 30% of nights is the primary confirmation period.'}
    (r/'advanced_time_models_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
