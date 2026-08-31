from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from habuai.hardening.model_tournament import _spatial_feature_sets


def _model(c=1.0):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=c))


def _add_prior_exposure(d):
    x=d.copy();x['prior_exposure_nights']=0.0
    seen={}
    for night in sorted(x.night.astype(str).unique()):
        idx=x.index[x.night.astype(str)==night]
        x.loc[idx,'prior_exposure_nights']=[seen.get(str(s),0) for s in x.loc[idx,'segment_id']]
        for s in x.loc[idx,'segment_id'].astype(str).unique():seen[s]=seen.get(s,0)+1
    x['log_prior_exposure']=np.log1p(x.prior_exposure_nights)
    return x


def _rank_eval(data,features,correct_exposure=False):
    hit30=hit10=hit5=hit1=eligible=0;pcts=[];nights=0
    for night in sorted(data.night.astype(str).unique()):
        tr=data[data.night.astype(str)<night].copy();te=data[data.night.astype(str)==night].copy()
        if tr.spatial_label.sum()<5 or (tr.spatial_label==0).sum()<20 or te.empty:continue
        feats=list(features)+(['log_prior_exposure'] if correct_exposure and 'log_prior_exposure' not in features else [])
        m=_model(1.0);w=pd.to_numeric(tr.sample_weight,errors='coerce').fillna(1.0).to_numpy()
        if correct_exposure:w=w/np.sqrt(1.0+pd.to_numeric(tr.prior_exposure_nights,errors='coerce').fillna(0).to_numpy())
        try:m.fit(tr[feats],tr.spatial_label.astype(int),logisticregression__sample_weight=w)
        except TypeError:m.fit(tr[feats],tr.spatial_label.astype(int))
        te['p']=m.predict_proba(te[feats])[:,1]
        if correct_exposure:
            # probability per opportunity, not raw historical exposure volume
            te['p']=te.p/np.sqrt(1.0+pd.to_numeric(te.prior_exposure_nights,errors='coerce').fillna(0))
        te=te.sort_values('p',ascending=False).reset_index(drop=True);te['rank']=np.arange(1,len(te)+1);te['pct']=te['rank']/len(te);a=te[te.spatial_label==1]
        eligible+=len(a);hit30+=int((a['rank']<=30).sum());hit1+=int((a.pct<=.01).sum());hit5+=int((a.pct<=.05).sum());hit10+=int((a.pct<=.10).sum());pcts+=a.pct.tolist();nights+=1
    return {'nights_scored':nights,'eligible_positive_segments':eligible,'top30_hits':hit30,'top30_rate':hit30/eligible if eligible else None,'top1pct_hits':hit1,'top1pct_rate':hit1/eligible if eligible else None,'top5pct_hits':hit5,'top5pct_rate':hit5/eligible if eligible else None,'top10pct_hits':hit10,'top10pct_rate':hit10/eligible if eligible else None,'median_rank_pct':float(np.median(pcts)) if pcts else None}


def _detection_eval(learning):
    g=learning[learning.learning_row_source=='gpx_visit'].copy();g['entered_at']=pd.to_datetime(g.entered_at,format='mixed',utc=True).dt.tz_convert('Asia/Tokyo');g['night']=(g.entered_at-pd.Timedelta(hours=7)).dt.date.astype(str)
    feats=[c for c in ['sin_hour','cos_hour','duration_s','mean_speed_mps','mean_match_distance_m','temperature_c','humidity_pct','dew_point_c','precip_mm','weather_code','surface_pressure','cloud_cover','rain_1h_mm','rain_3h_mm','rain_6h_mm','hours_since_rain','moon_age_days','moon_illumination','fog_wmo_flag','fog_proxy_flag','tide_height_cm','tide_change_1h_cm','minutes_to_nearest_turning_tide','segment_prior_visits'] if c in g]
    rows=[]
    for night in sorted(g.night.unique()):
        tr=g[g.night.astype(str)<night];te=g[g.night.astype(str)==night].copy()
        if tr.habu_capture.sum()<5 or (tr.habu_capture==0).sum()<20 or te.empty:continue
        m=_model(.2);m.fit(tr[feats],tr.habu_capture.astype(int));te['p']=m.predict_proba(te[feats])[:,1]
        te=te.sort_values('p',ascending=False).reset_index(drop=True);te['rank']=np.arange(1,len(te)+1);te['pct']=te['rank']/len(te);a=te[te.habu_capture==1]
        rows.append({'night':night,'positive_visit_rows':int(len(a)),'top30_hits':int((a['rank']<=30).sum()),'top10pct_hits':int((a.pct<=.10).sum()),'candidate_visit_rows':int(len(te))})
    o=pd.DataFrame(rows)
    if o.empty:return {'nights_scored':0,'positive_visit_rows':0,'note':'insufficient prior GPX positive visits'}
    n=int(o.positive_visit_rows.sum());return {'nights_scored':int(len(o)),'positive_visit_rows':n,'top30_hits':int(o.top30_hits.sum()),'top30_rate':float(o.top30_hits.sum()/n) if n else None,'top10pct_hits':int(o.top10pct_hits.sum()),'top10pct_rate':float(o.top10pct_hits.sum()/n) if n else None,'candidate_visit_rows':int(o.candidate_visit_rows.sum())}


def main():
    root=Path(__file__).resolve().parents[1];p=root/'data'/'processed';r=root/'reports';r.mkdir(exist_ok=True)
    d=pd.read_csv(p/'reconstructed_spatial_learning_2026-05_07.csv',low_memory=False);d=_add_prior_exposure(d)
    fs=_spatial_feature_sets(d);base=fs.get('no_recency_road_capture30') or fs.get('full58') or []
    naive=_rank_eval(d,base,False);corrected=_rank_eval(d,base,True)
    learning=pd.read_csv(p/'learning_10m_road.csv',low_memory=False);detect=_detection_eval(learning)
    summary={'status':'ok','common_rule':'chronological walk-forward only; each target night is scored from prior nights','occupancy_component':{'training_rows':int(len(d)),'reconstructed_nights':int(d.night.nunique()),'positive_segment_rows':int(d.spatial_label.sum()),'spatial_features':int(len(base)),'naive_occupancy_ranking':naive},'detection_component':{'strict_actual_gpx_only':detect,'guardrail':'Detection requires actual timed visits. Reconstructed May-Jul routes have no fabricated passage times, so they improve occupancy but are not used as timed detection negatives.'},'exposure_bias_test':{'naive_ranking':naive,'exposure_corrected_ranking':corrected,'correction':'add prior exposure count as a feature, down-weight repeatedly exposed training rows, and divide test priority by sqrt(1+prior exposure). This tests whether ranking was merely rewarding roads searched more often.'}}
    (r/'occupancy_detection_ranking_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
