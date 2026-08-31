from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_full_capture_time_backtest import _as_jst

FEATURES=['sin_month','cos_month','sin_doy','cos_doy','prior5_mean','prior10_mean','prior30_mean','prior5_positive_rate','prior10_positive_rate','prior30_positive_rate','days_since_last_capture','prior_last_count']

def _clf(tr,target):
    if tr[target].nunique()<2:return None
    m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=.2));m.fit(tr[FEATURES],tr[target].astype(int));return m

def _prob(m,x,fallback):return float(m.predict_proba(x[FEATURES])[0,1]) if m is not None else float(fallback)

def main():
    root=Path(__file__).resolve().parents[1];p=root/'data'/'processed';r=root/'reports';r.mkdir(exist_ok=True)
    e=pd.read_csv(p/'events_matched.csv',low_memory=False);e['timestamp']=_as_jst(e.timestamp);e=e[e.timestamp.notna()].copy();e['night']=(e.timestamp-pd.Timedelta(hours=7)).dt.date.astype(str)
    cap=e[(e.species=='ハブ')&(e.event_type=='捕獲')].copy();cap['individual_count']=pd.to_numeric(cap.individual_count,errors='coerce').fillna(1).clip(lower=1)
    counts=cap.groupby('night').individual_count.sum().to_dict();nights=sorted(e.night.dropna().unique().tolist());rows=[]
    for i,n in enumerate(nights):
        ts=pd.Timestamp(n);hist=np.array([counts.get(q,0) for q in nights[:i]],float);lastpos=np.where(hist>0)[0]
        def tail(k):return hist[-k:] if len(hist) else hist
        row={'night':n,'capture_count':int(counts.get(n,0)),'positive':int(counts.get(n,0)>0),'multi2':int(counts.get(n,0)>=2),'multi3':int(counts.get(n,0)>=3),'multi5':int(counts.get(n,0)>=5),'sin_month':math.sin(2*math.pi*ts.month/12),'cos_month':math.cos(2*math.pi*ts.month/12),'sin_doy':math.sin(2*math.pi*ts.dayofyear/365.25),'cos_doy':math.cos(2*math.pi*ts.dayofyear/365.25)}
        for k in (5,10,30):
            h=tail(k);row[f'prior{k}_mean']=float(h.mean()) if len(h) else 0.;row[f'prior{k}_positive_rate']=float((h>0).mean()) if len(h) else 0.
        row['prior_last_count']=float(hist[-1]) if len(hist) else 0.;row['days_since_last_capture']=float((ts-pd.Timestamp(nights[int(lastpos[-1])])).days) if len(lastpos) else 999.
        rows.append(row)
    d=pd.DataFrame(rows);split=max(10,int(np.floor(len(nights)*.7)));conf=set(nights[split:]);pred=[]
    for n in nights:
        tr=d[d.night<n];te=d[d.night==n]
        if len(tr)<12 or te.empty:continue
        pos=_clf(tr,'positive');ptr=tr[tr.positive==1];m2=_clf(ptr,'multi2') if len(ptr) else None;m3=_clf(ptr,'multi3') if len(ptr) else None;m5=_clf(ptr,'multi5') if len(ptr) else None
        ppos=_prob(pos,te,tr.positive.mean());p2=_prob(m2,te,ptr.multi2.mean() if len(ptr) else 0);p3=_prob(m3,te,ptr.multi3.mean() if len(ptr) else 0);p5=_prob(m5,te,ptr.multi5.mean() if len(ptr) else 0)
        lam=float(ptr.capture_count.mean()) if len(ptr) else 1.;
        if len(ptr)>=8:
            imp=SimpleImputer(strategy='median');X=imp.fit_transform(ptr[FEATURES]);pm=PoissonRegressor(alpha=.5,max_iter=500).fit(X,ptr.capture_count);lam=float(pm.predict(imp.transform(te[FEATURES]))[0])
        pred.append({'night':n,'actual_count':int(te.capture_count.iloc[0]),'p_zero':1-ppos,'p_positive':ppos,'p_ge2':ppos*p2,'p_ge3':ppos*p3,'p_ge5':ppos*p5,'expected_count':ppos*lam})
    o=pd.DataFrame(pred);o.to_csv(r/'hurdle_count_fast_predictions.csv',index=False)
    def met(x):
        if x.empty:return {'eligible_nights':0}
        a=x.actual_count.to_numpy(int);res={'eligible_nights':int(len(x)),'zero_nights':int((a==0).sum()),'positive_nights':int((a>0).sum()),'actual_total':int(a.sum()),'predicted_expected_total':float(x.expected_count.sum()),'count_mae':float(np.mean(np.abs(a-x.expected_count.to_numpy(float)))),'zero_brier':float(np.mean(((a==0).astype(float)-x.p_zero.to_numpy(float))**2))}
        for k in (2,3,5):res[f'ge{k}_actual_rate']=float((a>=k).mean());res[f'ge{k}_brier']=float(np.mean(((a>=k).astype(float)-x[f'p_ge{k}'].to_numpy(float))**2))
        return res
    summary={'status':'ok','method':'fast chronological hurdle baseline using only pre-night count/season history; any event-log night is weak field-evidence for a zero night','field_evidence_nights':len(nights),'capture_positive_nights':int((d.capture_count>0).sum()),'zero_field_evidence_nights':int((d.capture_count==0).sum()),'all_walkforward':met(o),'frozen_confirmation':met(o[o.night.isin(conf)]),'guardrail':'Zero field-evidence nights are not all GPX-confirmed full surveys, so p_zero is provisional. The weather-enhanced hurdle model is evaluated separately.'}
    (r/'hurdle_count_fast_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
