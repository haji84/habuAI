from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .backtest import run_walk_forward_backtest
from .canonical import add_canonical_audit,build_positive_label_audit,load_canonical_habu_events,match_events_preserve_all,merge_canonical_capture_master
from .environmental import add_lunar_features,add_weather_derived_features,join_optional_tide_features
from .events import dedupe_events,join_weather,slope_features,species_from_text,strict_holdout_score
from .features import add_outcomes_bio,add_static_context,apply_anchor_prior_visits,build_capture_anchor_visits
from .modeling import fit_model,fit_production_model,score_holdout
from .output import write_map,write_qa
from .roads import combine_roads,recover_supplemental_roads

EXPECTED_CANONICAL={"capture_events":206,"capture_individuals":208,"gps_capture_events":150,"gps_capture_individuals":151,"gps_missing_capture_events":56,"canonical_habu_events_total":247,"no_capture_events":2,"roadkill_events":33,"sighting_events":6}

def _full_canonical_audit(events):
    audit=add_canonical_audit(events);audit["canonical_habu_events_total"]=int((events.species=="ハブ").sum());audit["no_capture_events"]=int(((events.species=="ハブ")&(events.event_type=="no_capture")).sum());audit["roadkill_events"]=int(((events.species=="ハブ")&events.event_type.isin(["轢死","roadkill_sighting"])).sum());audit["sighting_events"]=int(((events.species=="ハブ")&events.event_type.isin(["目撃","sighting"])).sum());return audit

def _require_expected_canonical(audit,stage):
    bad={k:{"expected":v,"actual":audit.get(k)} for k,v in EXPECTED_CANONICAL.items() if audit.get(k)!=v}
    if bad:raise RuntimeError(f"canonical master gate failed at {stage}: {json.dumps(bad,ensure_ascii=False)}")

def apply_hardening(pipeline):
    pipeline._species_from_text=species_from_text;original_parse=pipeline.parse_field_log
    def hardened_run(root):
        cfg=pipeline.load_config(root);paths=pipeline.Paths(root);pipeline.ensure_dirs(paths);osm_segs=pipeline.build_10m_segments(root,cfg);points=pipeline.read_gpx_files(root);osm_mp=pipeline.map_match_gpx(points,osm_segs,cfg)
        files=sorted((root/"data"/"raw"/"logs").glob("*.txt"));raw_logs=pd.concat([original_parse(p) for p in files],ignore_index=True) if files else pd.DataFrame();canonical=load_canonical_habu_events(root);_require_expected_canonical(_full_canonical_audit(canonical),"canonical-load")
        if not raw_logs.empty:raw_logs=raw_logs[raw_logs.species!="ハブ"].copy()
        raw=merge_canonical_capture_master(raw_logs,canonical);events,removed=dedupe_events(raw);pre_match_rows=len(events);_require_expected_canonical(_full_canonical_audit(events[events.species=="ハブ"].copy()),"post-dedupe")
        events_osm=match_events_preserve_all(pipeline,events,osm_segs) if not events.empty else events
        if len(events_osm)>pre_match_rows:raise RuntimeError(f"event road matching expanded rows: {pre_match_rows} -> {len(events_osm)}")
        gps_events_osm=events_osm[events_osm.lat.notna()&events_osm.lon.notna()] if not events_osm.empty else events_osm;supplemental,recovery_audit=recover_supplemental_roads(points,osm_mp,gps_events_osm,osm_segs,cfg,root=root);segs=combine_roads(osm_segs,supplemental)
        if not supplemental.empty:segs.to_crs("EPSG:4326").to_file(paths.processed/"road_segments_10m.geojson",driver="GeoJSON")
        mp=pipeline.map_match_gpx(points,segs,cfg);visits=slope_features(pipeline.segment_visits(mp));events=match_events_preserve_all(pipeline,events,segs) if not events.empty else events
        if len(events)>pre_match_rows:raise RuntimeError(f"event road matching expanded rows after supplemental recovery: {pre_match_rows} -> {len(events)}")
        if not events.empty:
            missing_gps=events.lat.isna()|events.lon.isna();events["unmatched_reason"]=np.where(missing_gps,"missing GPS",np.where(events.segment_id.isna(),"nearest mapped road exceeds match threshold",""))
        canonical_audit=_full_canonical_audit(events[events.species=="ハブ"].copy());_require_expected_canonical(canonical_audit,"post-road-match");(paths.reports/"canonical_master_audit.json").write_text(json.dumps(canonical_audit,ensure_ascii=False,indent=2),encoding="utf-8");(paths.reports/"gpx_road_recovery.json").write_text(json.dumps({"recovered_segment_count":int(len(supplemental)),"recovered_roads":recovery_audit},ensure_ascii=False,indent=2),encoding="utf-8")
        weather_events=events.dropna(subset=["timestamp"]) if not events.empty else events;weather=pipeline.fetch_weather(root,weather_events,visits,cfg)
        data=join_weather(visits,weather);data=add_outcomes_bio(data,events,segs,cfg);data=add_static_context(data,segs,root);data=pipeline.add_exposure_features(data);data=add_lunar_features(data);data=add_weather_derived_features(data);data=join_optional_tide_features(data,root);data["learning_row_source"]="gpx_visit"
        anchor_base=build_capture_anchor_visits(events,data);anchors=pd.DataFrame()
        if not anchor_base.empty:
            anchors=join_weather(anchor_base,weather);anchors=add_outcomes_bio(anchors,events,segs,cfg);anchors["habu_capture"]=1;anchors["habu_individuals"]=pd.to_numeric(anchors["anchor_individual_count"],errors="coerce").fillna(1).astype(int);anchors=add_static_context(anchors,segs,root);anchors=pipeline.add_exposure_features(anchors);anchors=apply_anchor_prior_visits(anchors,data);anchors=add_lunar_features(anchors);anchors=add_weather_derived_features(anchors);anchors=join_optional_tide_features(anchors,root);anchors["learning_row_source"]="capture_gps_anchor"
        model_data=pd.concat([data,anchors],ignore_index=True,sort=False) if not anchors.empty else data.copy();model_data=model_data.sort_values("entered_at",kind="stable").reset_index(drop=True);model_data.to_csv(paths.processed/"learning_10m_road.csv",index=False);model_data.to_parquet(paths.processed/"learning_10m_road.parquet",index=False);mp.to_csv(paths.processed/"gpx_points_matched.csv",index=False);visits.to_csv(paths.processed/"segment_visits.csv",index=False)
        if not anchors.empty:anchors.to_csv(paths.processed/"capture_gps_anchor_rows.csv",index=False)
        if not events.empty:events.to_csv(paths.processed/"events_matched.csv",index=False);events[events.segment_id.isna()].to_csv(paths.processed/"events_unmatched.csv",index=False)
        label_audit=build_positive_label_audit(events,model_data);label_audit.to_csv(paths.reports/"positive_label_audit.csv",index=False);label_ok=int((label_audit.audit_status=="ok").sum()) if not label_audit.empty else 0;road_matched_capture_events=int(((events.species=="ハブ")&(events.event_type=="捕獲")&events.segment_id.notna()).sum())
        if label_ok!=road_matched_capture_events:raise RuntimeError(f"GPS capture learning gate failed: road_matched={road_matched_capture_events}, labeled_ok={label_ok}")
        backtest=run_walk_forward_backtest(root,model_data,events,cfg) if cfg.get("walk_forward_backtest",{}).get("enabled",True) else {"status":"disabled"}
        # Evaluation model is frozen before the cutoff. Production is trained only after holdout scoring,
        # then uses every currently available learning row for operational forecasts.
        metrics=fit_model(root,model_data,cfg);hold=score_holdout(root,model_data,cfg);production=fit_production_model(root,model_data,cfg);strict=strict_holdout_score(events);(paths.reports/"strict_holdout_2026-08-28.json").write_text(json.dumps(strict,ensure_ascii=False,indent=2),encoding="utf-8");forecast=pipeline.make_forecast(root,data,cfg);(paths.reports/"model_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8");(paths.reports/"production_model_metrics.json").write_text(json.dumps(production,ensure_ascii=False,indent=2),encoding="utf-8");(paths.reports/"latest_score.json").write_text(json.dumps(hold,ensure_ascii=False,indent=2),encoding="utf-8");write_map(root,segs,data,events,points,cfg);qa=write_qa(root,pipeline,cfg,points,raw,events,model_data,removed,strict)
        qa["canonical_master"]=canonical_audit;qa["positive_label_audit"]={"capture_events":int(len(label_audit)),"road_matched_capture_events":road_matched_capture_events,"labeled_ok":label_ok,"not_labeled":int(len(label_audit)-label_ok),"status_counts":label_audit.audit_status.value_counts(dropna=False).to_dict() if not label_audit.empty else {},"gpx_positive_rows":int(((model_data.habu_capture==1)&(model_data.learning_row_source=="gpx_visit")).sum()),"gps_anchor_positive_rows":int(((model_data.habu_capture==1)&(model_data.learning_row_source=="capture_gps_anchor")).sum())};qa["foundation_steps"]["2_full_capture_master_road_match_and_label_audit"]="complete" if canonical_audit==EXPECTED_CANONICAL and label_ok==road_matched_capture_events else "partial"
        env_cols=["moon_age_days","moon_illumination","moon_phase_sin","moon_phase_cos","fog_wmo_flag","temp_dewpoint_spread_c","fog_proxy_flag","temperature_change_3visits_c","tide_height_cm","tide_change_1h_cm","tide_state_code","minutes_to_nearest_turning_tide","tide_source_available"];qa["environmental_feature_missing_rate"]={c:(None if c not in model_data else float(model_data[c].isna().mean())) for c in env_cols};qa["feature_sources"]={"lunar":"deterministic synodic calculation","fog":"Open-Meteo WMO code 45/48 plus temperature-dewpoint proxy","temperature":"Open-Meteo hourly","tide":"JMA Amami O9 hourly predicted tide table; local authoritative tide_hourly.csv overrides when supplied"};qa["learning_sources"]={"gpx_visit_rows":int((model_data.learning_row_source=="gpx_visit").sum()),"capture_gps_anchor_rows":int((model_data.learning_row_source=="capture_gps_anchor").sum()),"note":"capture anchors are model evidence only and do not count as GPX exploration visits"};qa["model_roles"]={"evaluation":{"artifact":"models/habu_occurrence_evaluation.joblib","cutoff":cfg["baseline_cutoff"],"positives":metrics.get("positives")},"production":{"artifact":"models/habu_occurrence_production.joblib","alias":"models/habu_occurrence.joblib","positives":production.get("positives"),"uses_all_current_learning_rows":True}};qa["walk_forward_backtest"]=backtest
        (paths.reports/"qa_summary.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8");return {"segments":len(segs),"supplemental_segments":len(supplemental),"gpx_points":len(points),"visits":len(visits),"events":len(events),"duplicates_removed":removed,"canonical_master":canonical_audit,"positive_label_audit":qa["positive_label_audit"],"learning_sources":qa["learning_sources"],"metrics":metrics,"production_metrics":production,"holdout":hold,"strict_holdout":strict,"backtest":backtest,"qa":qa,"forecast":forecast}
    pipeline.run=hardened_run
