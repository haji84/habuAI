from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from habuai.hardening.model_tournament import _spatial_feature_sets
from run_full_capture_time_backtest import _as_jst, _circular_minutes
from run_hurdle_count_model import _build_nightly, _fetch_weather as fetch_hurdle_weather, _metrics as hurdle_metrics, _walkforward as hurdle_walkforward, _fit_logistic
from run_environmental_hazard_v2 import _risk_rows_for_night, _weather_30m, _tide_30m
from run_advanced_time_models import _fetch_weather as fetch_time_weather


def _model(c=1.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=c),
    )


def _rank_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"nights_scored": 0, "eligible_positive_segments": 0}
    d = pd.DataFrame(rows)
    n = int(d.positive_segments.sum())
    return {
        "nights_scored": int(d.night.nunique()),
        "eligible_positive_segments": n,
        "top30_hits": int(d.top30_hits.sum()),
        "top30_rate": float(d.top30_hits.sum() / n) if n else None,
        "top10pct_hits": int(d.top10pct_hits.sum()),
        "top10pct_rate": float(d.top10pct_hits.sum() / n) if n else None,
        "median_rank_pct": float(np.nanmedian(d.median_rank_pct)) if n else None,
    }


def _score_occ_night(model, te, features):
    z = te.copy()
    z["p"] = model.predict_proba(z[features])[:, 1]
    z = z.sort_values("p", ascending=False).reset_index(drop=True)
    z["rank"] = np.arange(1, len(z) + 1)
    z["pct"] = z["rank"] / len(z)
    a = z[z.spatial_label == 1]
    return {
        "positive_segments": int(len(a)),
        "top30_hits": int((a["rank"] <= 30).sum()),
        "top10pct_hits": int((a.pct <= .10).sum()),
        "median_rank_pct": float(a.pct.median()) if len(a) else np.nan,
    }


def occupancy_compare(recon: pd.DataFrame) -> dict:
    fs = _spatial_feature_sets(recon)
    features = fs.get("no_recency_road_capture30") or fs.get("full58") or []
    nights = sorted(recon.night.astype(str).unique())
    split = max(5, int(np.floor(len(nights) * .70)))
    sel, conf = nights[:split], nights[split:]
    cutoff = conf[0]
    base_tr = recon[recon.night.astype(str) < cutoff].copy()
    frozen = _model(1.0)
    w = pd.to_numeric(base_tr.sample_weight, errors="coerce").fillna(1.0).to_numpy()
    try:
        frozen.fit(base_tr[features], base_tr.spatial_label.astype(int), logisticregression__sample_weight=w)
    except TypeError:
        frozen.fit(base_tr[features], base_tr.spatial_label.astype(int))

    frozen_rows, expanding_rows = [], []
    for night in conf:
        te = recon[recon.night.astype(str) == str(night)].copy()
        if te.empty:
            continue
        fr = _score_occ_night(frozen, te, features); fr["night"] = night; frozen_rows.append(fr)
        tr = recon[recon.night.astype(str) < str(night)].copy()
        if tr.spatial_label.sum() < 5 or (tr.spatial_label == 0).sum() < 20:
            continue
        m = _model(1.0)
        ww = pd.to_numeric(tr.sample_weight, errors="coerce").fillna(1.0).to_numpy()
        try:
            m.fit(tr[features], tr.spatial_label.astype(int), logisticregression__sample_weight=ww)
        except TypeError:
            m.fit(tr[features], tr.spatial_label.astype(int))
        er = _score_occ_night(m, te, features); er["night"] = night; expanding_rows.append(er)
    return {
        "protocol": {"reconstructed_nights": len(nights), "selection_nights": len(sel), "confirmation_nights": len(conf), "features": len(features)},
        "frozen_model": _rank_metrics(frozen_rows),
        "expanding_daily_refit": _rank_metrics(expanding_rows),
        "guardrail": "Historical reconstructed routes are spatial-only. No fabricated passage times are used.",
    }


def _time_metrics(rows):
    if not rows:
        return {"eligible": 0, "nights_scored": 0}
    d = pd.DataFrame(rows); e = d.error_min.astype(float)
    out = {"eligible": int(len(d)), "nights_scored": int(d.night.nunique()), "median_error_min": float(e.median()), "mean_error_min": float(e.mean())}
    for m in [30, 60, 90, 120]:
        h = int((e <= m).sum()); out[f"within_{m}m_hits"] = h; out[f"within_{m}m_rate"] = h / len(d)
    return out


def _hazard_predict(model, te, features):
    z = te.copy(); z["p"] = model.predict_proba(z[features])[:, 1]
    rows=[]
    for eid,g in z.groupby("event_id"):
        best=g.loc[g.p.idxmax()]; actual=int(g.actual_minute.iloc[0]); pred=int(best.slot_minute)
        rows.append({"night":str(g.night.iloc[0]),"event_id":eid,"error_min":_circular_minutes(actual,pred)})
    return rows


def hazard_compare(events: pd.DataFrame, processed: Path) -> dict:
    ev=events.copy(); ev["lat"]=pd.to_numeric(ev.lat,errors="coerce"); ev["lon"]=pd.to_numeric(ev.lon,errors="coerce")
    cap=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&ev.lat.notna()&ev.lon.notna()&ev.timestamp.notna()].copy()
    cap["night"]=(cap.timestamp-pd.Timedelta(hours=7)).dt.date.astype(str)
    nights=sorted(cap.night.unique().tolist()); split=max(4,int(np.floor(len(nights)*.70))); sel=nights[:split]; conf=nights[split:]
    ws=str((cap.timestamp.min()-pd.Timedelta(days=3)).date()); we=str(cap.timestamp.max().date())
    w=fetch_time_weather(ws,we,processed/"cache"/"openmeteo_historical_environment_v2.csv"); w30=_weather_30m(w)
    all_times=pd.date_range(w30.index.min(),w30.index.max(),freq="30min"); tide=_tide_30m(all_times)
    risk=pd.concat([_risk_rows_for_night(cap,n,w30,tide) for n in nights],ignore_index=True)
    features=["sin_hour","cos_hour","sin_doy","cos_doy","global_prior","local3km_prior","temperature_2m","relative_humidity_2m","dew_point_2m","precipitation","weather_code","surface_pressure","cloud_cover","rain_24h_mm","rain_48h_mm","hours_since_rain","fog_wmo_flag","fog_proxy_flag","fog_any_flag"]
    base=risk[risk.night.isin(set(sel))].copy(); frozen=_model(.2); frozen.fit(base[features],base.label.astype(int))
    frozen_rows=[]
    for n in conf:
        te=risk[risk.night==str(n)].copy()
        if not te.empty: frozen_rows += _hazard_predict(frozen,te,features)
    expanding_rows=[]
    for n in conf:
        tr=risk[risk.night.astype(str)<str(n)].copy(); te=risk[risk.night==str(n)].copy()
        if te.empty or int(tr.label.sum())<10: continue
        m=_model(.2);m.fit(tr[features],tr.label.astype(int));expanding_rows += _hazard_predict(m,te,features)
    return {
        "protocol":{"capture_nights":len(nights),"selection_nights":len(sel),"confirmation_nights":len(conf),"confirmation_capture_events":int(cap[cap.night.isin(set(conf))].shape[0]),"feature_stage":"+fog (weather + rain24/48 + hours_since_rain + fog)"},
        "frozen_coefficients":_time_metrics(frozen_rows),
        "expanding_daily_refit":_time_metrics(expanding_rows),
        "guardrail":"Both strategies use only features available before/scorable for the target night; same-night capture labels are outcomes only. Frozen coefficients still receive target-night environmental inputs.",
    }


def _hurdle_fit_bundle(tr, features):
    pos=_fit_logistic(tr,features,"positive")
    ptr=tr[tr.positive==1].copy()
    models={"positive":pos}
    for target in ["multi2","multi3","multi5"]: models[target]=_fit_logistic(ptr,features,target)
    imp=None;pois=None
    if len(ptr)>=8:
        imp=SimpleImputer(strategy="median");xp=imp.fit_transform(ptr[features]);pois=PoissonRegressor(alpha=.5,max_iter=1000).fit(xp,ptr.capture_count)
    return models,ptr,imp,pois


def _hurdle_predict_row(bundle, te, features):
    models,ptr,imp,pois=bundle;x=te[features]
    ppos=float(models["positive"].predict_proba(x)[0,1]) if models["positive"] is not None else float(ptr.positive.mean()) if len(ptr) else 0.0
    def pp(target):
        m=models[target]
        return float(m.predict_proba(x)[0,1]) if m is not None else float(ptr[target].mean()) if len(ptr) else 0.0
    lam=float(pois.predict(imp.transform(x))[0]) if pois is not None else float("nan")
    return {"night":str(te.night.iloc[0]),"actual_count":int(te.capture_count.iloc[0]),"p_positive":ppos,"p_zero":1-ppos,"p_multi2_given_positive":pp("multi2"),"p_multi3_given_positive":pp("multi3"),"p_multi5_given_positive":pp("multi5"),"expected_count_given_positive":lam}


def hurdle_compare(events: pd.DataFrame, processed: Path) -> dict:
    start=str(events.timestamp.min().date()); end=str(events.timestamp.max().date()); w=fetch_hurdle_weather(start,end,processed/"cache"/"openmeteo_historical_full_period.csv")
    d=_build_nightly(events,w); nights=sorted(d.night.unique()); split=max(10,int(np.floor(len(nights)*.70))); sel=nights[:split]; conf=nights[split:]
    features=[c for c in ["sin_month","cos_month","sin_doy","cos_doy","prior_mean_count","prior30_mean_count","prior14_mean_count","prior30_active_nights","prior14_active_nights","temp_mean","temp_min","humidity_mean","dewpoint_mean","precip_sum","pressure_mean","cloud_mean","weather_max"] if c in d]
    base=d[d.night.isin(set(sel))].copy(); bundle=_hurdle_fit_bundle(base,features)
    frozen=[]
    for n in conf:
        te=d[d.night==str(n)]
        if not te.empty: frozen.append(_hurdle_predict_row(bundle,te,features))
    expanding=hurdle_walkforward(d,conf,features,min_prior_nights=12)
    return {
        "protocol":{"field_evidence_nights":len(nights),"selection_nights":len(sel),"confirmation_nights":len(conf),"zero_field_evidence_nights":int((d.capture_count==0).sum())},
        "frozen_coefficients":hurdle_metrics(pd.DataFrame(frozen)),
        "expanding_daily_refit":hurdle_metrics(expanding),
        "guardrail":"Zero-night evidence remains sparse and includes weak field-evidence nights, so P(0) calibration is provisional.",
    }


def main():
    root=Path(__file__).resolve().parents[1];p=root/"data"/"processed";r=root/"reports";r.mkdir(exist_ok=True)
    events=pd.read_csv(p/"events_matched.csv",low_memory=False);events["timestamp"]=_as_jst(events.timestamp);events=events[events.timestamp.notna()].copy()
    recon=pd.read_csv(p/"reconstructed_spatial_learning_2026-05_07.csv",low_memory=False)
    summary={
        "status":"ok",
        "purpose":"Compare fixed coefficients versus refitting after each completed night while keeping each target night's outcome hidden until after prediction.",
        "occupancy":occupancy_compare(recon),
        "hazard_v2":hazard_compare(events,p),
        "hurdle_count":hurdle_compare(events,p),
        "integration_readiness":{
            "recommended_daily_cycle":["freeze target-night outcome at 07:00","predict count/location/time","after the night, append verified observations","score prior prediction","refit only the model families proven to benefit from daily refit","issue next-night integrated recommendation"],
            "warning":"Do not claim a single integrated accuracy yet because occupancy, time, and count are evaluated on different observational units. A joint route-level prospective score is the next required validation."
        }
    }
    (r/"daily_update_extended_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
