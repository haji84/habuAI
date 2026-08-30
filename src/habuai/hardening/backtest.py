from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import _feature_columns


def _new_model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=.2))


def _prior_correct_balanced_probability(prob: np.ndarray, observed_positive_rate: float) -> np.ndarray:
    p=np.clip(np.asarray(prob,dtype=float),1e-9,1-1e-9);q=float(np.clip(observed_positive_rate,1e-9,1-1e-9));odds=p/(1-p);adj=odds*(q/(1-q));return adj/(1+adj)


def _survey_night(ts: pd.Series, rollover_hour: int) -> pd.Series:
    return (pd.to_datetime(ts)-pd.Timedelta(hours=rollover_hour)).dt.date.astype(str)


def _night_bounds(night: str, rollover_hour: int, tz) -> tuple[pd.Timestamp,pd.Timestamp]:
    day=pd.Timestamp(night,tz=tz);start=day+pd.Timedelta(hours=rollover_hour);return start,start+pd.Timedelta(days=1)


def _actual_captures(events: pd.DataFrame,start: pd.Timestamp,end: pd.Timestamp)->pd.DataFrame:
    if events.empty:return events.copy()
    return events[(events.species=="ハブ")&(events.event_type=="捕獲")&(events.timestamp>=start)&(events.timestamp<end)].copy()


def run_walk_forward_backtest(root: Path,model_data: pd.DataFrame,events: pd.DataFrame,cfg: dict)->dict:
    """Leakage-safe nightly replay.

    Pre-night count uses only previous GPX-night capture/exposure history and previous typical effort.
    Location/time are route-replay metrics: model is frozen at 07:00, then scores that night's GPX
    visits as they occurred. Capture anchors from the replay night never enter training or candidates.
    """
    out_dir=root/"reports";rollover=int(cfg.get("night_rollover_hour",7));bcfg=cfg.get("walk_forward_backtest",{});top_k=int(bcfg.get("top_k_segments",30));min_pos=int(bcfg.get("min_train_positives",5));min_neg=int(bcfg.get("min_train_negatives",20));count_prior_exposure=float(bcfg.get("count_prior_exposure_rows",1000));count_prior_events=float(bcfg.get("count_prior_events",1))
    data=model_data.copy()
    if data.empty or "learning_row_source" not in data:
        result={"status":"no-data","nights":0};(out_dir/"walk_forward_backtest_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
    data["entered_at"]=pd.to_datetime(data.entered_at);events=events.copy();events["timestamp"]=pd.to_datetime(events.timestamp);gpx=data[data.learning_row_source=="gpx_visit"].copy()
    if gpx.empty:
        result={"status":"no-gpx-visits","nights":0};(out_dir/"walk_forward_backtest_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
    gpx["survey_night"]=_survey_night(gpx.entered_at,rollover);tz=gpx.entered_at.iloc[0].tz;rows=[];ranked=[];prior_night_rows=[];prior_night_captures=[]
    for night in sorted(gpx.survey_night.unique()):
        start,end=_night_bounds(night,rollover,tz);train=data[data.entered_at<start].copy();test=gpx[(gpx.entered_at>=start)&(gpx.entered_at<end)].copy();actual=_actual_captures(events,start,end)
        if test.empty:continue
        actual_count=int(len(actual));train_pos=int(train.habu_capture.sum());train_neg=int((train.habu_capture==0).sum())
        # Pure pre-night count estimate. Current-night effort is not used. Planned effort is proxied by
        # the median number of GPX visit rows from previous survey nights only.
        if prior_night_rows:
            expected_effort=float(np.median(prior_night_rows));historical_rate=(sum(prior_night_captures)+count_prior_events)/(sum(prior_night_rows)+count_prior_exposure);pre_count=float(historical_rate*expected_effort);pre_round=int(np.rint(pre_count))
        else:
            expected_effort=np.nan;historical_rate=np.nan;pre_count=np.nan;pre_round=np.nan
        # Exposure-normalized diagnostic asks: with the actual amount of searching, how many captures
        # would the historical rate imply? It is not a pre-night forecast and is reported separately.
        if prior_night_rows:
            exposure_count=float(historical_rate*len(test));exposure_round=int(np.rint(exposure_count))
        else:
            exposure_count=np.nan;exposure_round=np.nan
        base={"night":night,"train_rows":int(len(train)),"train_positives":train_pos,"gpx_visit_rows":int(len(test)),"actual_capture_events":actual_count,"actual_individuals":int(pd.to_numeric(actual.get("individual_count",pd.Series(dtype=float)),errors="coerce").fillna(1).sum()) if not actual.empty else 0,"pre_night_expected_effort_rows":expected_effort,"pre_night_historical_capture_rate_per_visit":historical_rate,"pre_night_predicted_capture_count":pre_count,"pre_night_predicted_capture_count_rounded":pre_round,"pre_night_count_absolute_error":np.nan if pd.isna(pre_round) else abs(int(pre_round)-actual_count),"exposure_normalized_predicted_capture_count":exposure_count,"exposure_normalized_predicted_capture_count_rounded":exposure_round,"exposure_normalized_count_absolute_error":np.nan if pd.isna(exposure_round) else abs(int(exposure_round)-actual_count)}
        if train_pos<min_pos or train_neg<min_neg:
            base["status"]="insufficient-training-history-for-route-model";rows.append(base);prior_night_rows.append(int(len(test)));prior_night_captures.append(actual_count);continue
        feats=_feature_columns(train);Xtr=train[feats].replace([np.inf,-np.inf],np.nan);ytr=train.habu_capture.astype(int);model=_new_model();model.fit(Xtr,ytr);raw=model.predict_proba(test.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1];prior=float(ytr.mean());test["pred_prob"]=_prior_correct_balanced_probability(raw,prior)
        seg_rank=test.groupby("segment_id",dropna=True).agg(pred_prob=("pred_prob","max"),visits=("segment_id","size")).reset_index().sort_values("pred_prob",ascending=False).reset_index(drop=True);seg_rank["rank"]=np.arange(1,len(seg_rank)+1);seg_rank["rank_percentile"]=seg_rank["rank"]/max(len(seg_rank),1);seg_rank["night"]=night;ranked.append(seg_rank[["night","segment_id","rank","rank_percentile","pred_prob","visits"]])
        rank_map=dict(zip(seg_rank.segment_id.astype(str),seg_rank["rank"]));pct_map=dict(zip(seg_rank.segment_id.astype(str),seg_rank.rank_percentile));top_ids=set(seg_rank.head(top_k).segment_id.astype(str));actual_matched=actual[actual.segment_id.notna()].copy() if not actual.empty and "segment_id" in actual else pd.DataFrame();actual_ids=actual_matched.segment_id.astype(str).tolist() if not actual_matched.empty else [];loc_ranks=[rank_map.get(s) for s in actual_ids];loc_pcts=[pct_map.get(s) for s in actual_ids];location_hits=sum((r is not None and r<=top_k) for r in loc_ranks);top10_hits=sum((p is not None and p<=0.10) for p in loc_pcts);ranked_actual=[r for r in loc_ranks if r is not None];pct_actual=[p for p in loc_pcts if p is not None]
        test["hour"]=test.entered_at.dt.hour;hour_scores=test.groupby("hour").pred_prob.sum().sort_values(ascending=False);peak=int(hour_scores.index[0]) if len(hour_scores) else None
        if actual.empty or peak is None:time_hits=0;time_rate=None if actual.empty else 0.0
        else:time_hits=int((actual.timestamp.dt.hour==peak).sum());time_rate=time_hits/len(actual)
        base.update({"status":"ok","training_positive_rate":prior,"candidate_segments":int(len(seg_rank)),"top_k_segments":top_k,"actual_road_matched_capture_events":int(len(actual_ids)),"actual_capture_segments_present_in_route_candidates":int(len(ranked_actual)),"location_hits_top_k":int(location_hits),"location_hit_rate_top_k":None if not actual_ids else location_hits/len(actual_ids),"location_hits_top_10pct":int(top10_hits),"location_hit_rate_top_10pct":None if not actual_ids else top10_hits/len(actual_ids),"actual_capture_median_rank":None if not ranked_actual else float(np.median(ranked_actual)),"actual_capture_median_rank_percentile":None if not pct_actual else float(np.median(pct_actual)),"predicted_peak_hour_start":peak,"predicted_peak_hour_end":None if peak is None else (peak+1)%24,"time_hits_peak_hour":int(time_hits),"time_hit_rate_peak_hour":time_rate});rows.append(base);prior_night_rows.append(int(len(test)));prior_night_captures.append(actual_count)
    detail=pd.DataFrame(rows);detail.to_csv(out_dir/"walk_forward_backtest_nightly.csv",index=False)
    if ranked:pd.concat(ranked,ignore_index=True).to_csv(out_dir/"walk_forward_backtest_segment_ranks.csv",index=False)
    route_ok=detail[detail.status=="ok"].copy() if not detail.empty and "status" in detail else pd.DataFrame();count_ok=detail[detail.pre_night_predicted_capture_count_rounded.notna()].copy() if not detail.empty else pd.DataFrame();exposure_ok=detail[detail.exposure_normalized_predicted_capture_count_rounded.notna()].copy() if not detail.empty else pd.DataFrame();summary={"status":"ok" if not detail.empty else "insufficient-data","method":"07:00 leakage wall; pre-night count from prior survey-night exposure/captures and prior median effort; location/time from frozen-model route replay","nights_total":int(len(detail)),"route_replay_nights_scored":int(len(route_ok)),"route_replay_nights_skipped":int(len(detail)-len(route_ok)),"pre_night_count_nights_scored":int(len(count_ok)),"top_k_segments":top_k}
    if not count_ok.empty:summary.update({"pre_night_actual_capture_events":int(count_ok.actual_capture_events.sum()),"pre_night_predicted_capture_count_rounded_total":int(count_ok.pre_night_predicted_capture_count_rounded.sum()),"pre_night_count_mae_per_night":float(count_ok.pre_night_count_absolute_error.mean()),"pre_night_count_exact_nights":int((count_ok.pre_night_predicted_capture_count_rounded==count_ok.actual_capture_events).sum()),"pre_night_count_exact_rate":float((count_ok.pre_night_predicted_capture_count_rounded==count_ok.actual_capture_events).mean())})
    if not exposure_ok.empty:summary.update({"exposure_normalized_count_mae_per_night":float(exposure_ok.exposure_normalized_count_absolute_error.mean()),"exposure_normalized_count_exact_nights":int((exposure_ok.exposure_normalized_predicted_capture_count_rounded==exposure_ok.actual_capture_events).sum()),"exposure_normalized_count_exact_rate":float((exposure_ok.exposure_normalized_predicted_capture_count_rounded==exposure_ok.actual_capture_events).mean())})
    if not route_ok.empty:
        eligible=int(route_ok.actual_road_matched_capture_events.sum());present=int(route_ok.actual_capture_segments_present_in_route_candidates.sum());time_eligible=int(route_ok.actual_capture_events.sum());summary.update({"route_replay_actual_capture_events":time_eligible,"route_replay_location_eligible_events":eligible,"route_replay_capture_segments_present_in_candidates":present,"route_replay_location_hits_top_k":int(route_ok.location_hits_top_k.sum()),"route_replay_location_hit_rate_top_k":None if eligible==0 else float(route_ok.location_hits_top_k.sum()/eligible),"route_replay_location_hits_top_10pct":int(route_ok.location_hits_top_10pct.sum()),"route_replay_location_hit_rate_top_10pct":None if eligible==0 else float(route_ok.location_hits_top_10pct.sum()/eligible),"route_replay_time_hits_peak_hour":int(route_ok.time_hits_peak_hour.sum()),"route_replay_time_hit_rate_peak_hour":None if time_eligible==0 else float(route_ok.time_hits_peak_hour.sum()/time_eligible)})
    (out_dir/"walk_forward_backtest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");return summary
