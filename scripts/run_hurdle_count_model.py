from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_full_capture_time_backtest import _as_jst


def _fetch_weather(start_date,end_date,cache):
    if cache.exists():
        w=pd.read_csv(cache);w['time']=pd.to_datetime(w.time,errors='coerce');return w
    hourly='temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,weather_code,surface_pressure,cloud_cover'
    q=urllib.parse.urlencode({'latitude':28.1456,'longitude':129.3200,'start_date':start_date,'end_date':end_date,'hourly':hourly,'timezone':'Asia/Tokyo'})
    with urllib.request.urlopen('https://archive-api.open-meteo.com/v1/archive?'+q,timeout=60) as r:obj=json.loads(r.read().decode('utf-8'))
    w=pd.DataFrame(obj['hourly']);w['time']=pd.to_datetime(w.time,errors='coerce');cache.parent.mkdir(parents=True,exist_ok=True);w.to_csv(cache,index=False);return w


def _night_weather(w,night):
    start=pd.Timestamp(str(night))+pd.Timedelta(hours=18);end=start+pd.Timedelta(hours=13)
    x=w[(w.time>=start)&(w.time<end)]
    if x.empty:return {}
    return {'temp_mean':pd.to_numeric(x.temperature_2m,errors='coerce').mean(),'temp_min':pd.to_numeric(x.temperature_2m,errors='coerce').min(),'humidity_mean':pd.to_numeric(x.relative_humidity_2m,errors='coerce').mean(),'dewpoint_mean':pd.to_numeric(x.dew_point_2m,errors='coerce').mean(),'precip_sum':pd.to_numeric(x.precipitation,errors='coerce').sum(),'pressure_mean':pd.to_numeric(x.surface_pressure,errors='coerce').mean(),'cloud_mean':pd.to_numeric(x.cloud_cover,errors='coerce').mean(),'weather_max':pd.to_numeric(x.weather_code,errors='coerce').max()}


def _build_nightly(events,weather):
    e=events.copy();e['night']=(e.timestamp-pd.Timedelta(hours=7)).dt.date.astype(str)
    cap=e[(e.species=='ハブ')&(e.event_type=='捕獲')].groupby('night').agg(capture_events=('event_type','size'),capture_count=('individual_count',lambda s:int(pd.to_numeric(s,errors='coerce').fillna(1).sum()))).reset_index()
    # Any logged event is evidence that the user was collecting field observations that night. This is broader than GPX and is flagged as weak survey evidence.
    nights=sorted(e.loc[e.timestamp.notna(),'night'].dropna().unique().tolist())
    c=cap.set_index('night').capture_count.to_dict();rows=[]
    for n in nights:
        ts=pd.Timestamp(n);month=ts.month;doy=ts.dayofyear
        prior=[c.get(p,0) for p in nights if p<n]
        prior_dates=[p for p in nights if p<n]
        d30=[c.get(p,0) for p in prior_dates if (ts-pd.Timestamp(p)).days<=30]
        d14=[c.get(p,0) for p in prior_dates if (ts-pd.Timestamp(p)).days<=14]
        row={'night':n,'capture_count':int(c.get(n,0)),'positive':int(c.get(n,0)>0),'multi2':int(c.get(n,0)>=2),'multi3':int(c.get(n,0)>=3),'multi5':int(c.get(n,0)>=5),
             'sin_month':math.sin(2*math.pi*month/12),'cos_month':math.cos(2*math.pi*month/12),'sin_doy':math.sin(2*math.pi*doy/365.25),'cos_doy':math.cos(2*math.pi*doy/365.25),
             'prior_mean_count':float(np.mean(prior[-10:])) if prior else 0.0,'prior30_mean_count':float(np.mean(d30)) if d30 else 0.0,'prior14_mean_count':float(np.mean(d14)) if d14 else 0.0,'prior30_active_nights':len(d30),'prior14_active_nights':len(d14)}
        row.update(_night_weather(weather,n));rows.append(row)
    return pd.DataFrame(rows)


def _fit_logistic(tr,features,target):
    if tr[target].nunique()<2:return None
    m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.2));m.fit(tr[features],tr[target].astype(int));return m


def _walkforward(d,nights,features,min_prior_nights=12):
    rows=[]
    for n in nights:
        tr=d[d.night.astype(str)<str(n)];te=d[d.night.astype(str)==str(n)]
        if len(tr)<min_prior_nights or te.empty:continue
        pos=_fit_logistic(tr,features,'positive');m2=_fit_logistic(tr[tr.positive==1],features,'multi2');m3=_fit_logistic(tr[tr.positive==1],features,'multi3');m5=_fit_logistic(tr[tr.positive==1],features,'multi5')
        pr=PoissonRegressor(alpha=.5,max_iter=1000);ptr=tr[tr.positive==1]
        if len(ptr)>=8:pr.fit(SimpleImputer(strategy='median').fit_transform(ptr[features]),ptr.capture_count)
        x=te[features];imp=SimpleImputer(strategy='median');
        # Poisson imputer fit only on prior positives to avoid future leakage.
        lam=float('nan')
        if len(ptr)>=8:
            ip=SimpleImputer(strategy='median');xp=ip.fit_transform(ptr[features]);pr=PoissonRegressor(alpha=.5,max_iter=1000).fit(xp,ptr.capture_count);lam=float(pr.predict(ip.transform(x))[0])
        ppos=float(pos.predict_proba(x)[0,1]) if pos is not None else float(tr.positive.mean())
        def pp(model,target):return float(model.predict_proba(x)[0,1]) if model is not None else float(ptr[target].mean()) if len(ptr) else 0.0
        rows.append({'night':n,'actual_count':int(te.capture_count.iloc[0]),'p_positive':ppos,'p_zero':1-ppos,'p_multi2_given_positive':pp(m2,'multi2'),'p_multi3_given_positive':pp(m3,'multi3'),'p_multi5_given_positive':pp(m5,'multi5'),'expected_count_given_positive':lam})
    return pd.DataFrame(rows)


def _metrics(x):
    if x.empty:return {'eligible_nights':0}
    a=x.actual_count.to_numpy(int);pred=np.where(np.isfinite(x.expected_count_given_positive),x.p_positive*x.expected_count_given_positive,x.p_positive)
    out={'eligible_nights':int(len(x)),'zero_nights':int((a==0).sum()),'positive_nights':int((a>0).sum()),'actual_total_captures':int(a.sum()),'predicted_total_expected':float(np.nansum(pred)),'count_mae':float(np.nanmean(np.abs(a-pred)))}
    y0=(a==0).astype(float);out['zero_brier']=float(np.mean((y0-x.p_zero.to_numpy(float))**2))
    for k,col in [(2,'p_multi2_given_positive'),(3,'p_multi3_given_positive'),(5,'p_multi5_given_positive')]:
        mask=a>0;y=(a[mask]>=k).astype(float);p=x.loc[mask,col].to_numpy(float);out[f'multi{k}_eligible_positive_nights']=int(mask.sum());out[f'multi{k}_actual_rate']=float(y.mean()) if len(y) else None;out[f'multi{k}_brier']=float(np.mean((y-p)**2)) if len(y) else None
    return out


def main():
    root=Path(__file__).resolve().parents[1];p=root/'data'/'processed';r=root/'reports';r.mkdir(exist_ok=True)
    e=pd.read_csv(p/'events_matched.csv',low_memory=False);e['timestamp']=_as_jst(e.timestamp);e=e[e.timestamp.notna()].copy()
    start=str(e.timestamp.min().date());end=str(e.timestamp.max().date());w=_fetch_weather(start,end,p/'cache'/'openmeteo_historical_full_period.csv')
    d=_build_nightly(e,w);d.to_csv(r/'hurdle_count_nightly_dataset.csv',index=False);nights=sorted(d.night.unique());split=max(10,int(np.floor(len(nights)*.70)));sel=nights[:split];conf=nights[split:]
    features=[c for c in ['sin_month','cos_month','sin_doy','cos_doy','prior_mean_count','prior30_mean_count','prior14_mean_count','prior30_active_nights','prior14_active_nights','temp_mean','temp_min','humidity_mean','dewpoint_mean','precip_sum','pressure_mean','cloud_mean','weather_max'] if c in d]
    pred=_walkforward(d,nights,features);pred.to_csv(r/'hurdle_count_walkforward_predictions.csv',index=False)
    summary={'status':'ok','method':'night-level hurdle family: probability of >=1 capture, conditional probabilities of >=2/3/5 captures, and positive-night Poisson expected count. Training is chronological only. Weather and prior-count features are visible before each target night.','inventory':{'field_evidence_nights':int(len(d)),'capture_positive_nights':int((d.capture_count>0).sum()),'zero_field_evidence_nights':int((d.capture_count==0).sum()),'selection_nights':int(len(sel)),'confirmation_nights':int(len(conf))},'all_walkforward':_metrics(pred),'frozen_confirmation':_metrics(pred[pred.night.isin(set(conf))]),'guardrail':'Zero-night evidence is currently sparse and includes nights inferred from any logged field event, not only full GPX-confirmed survey sessions. Therefore zero-probability calibration is provisional; multi-capture conditional probabilities have substantially more positive-night data.'}
    (r/'hurdle_count_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
