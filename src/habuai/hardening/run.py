from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .canonical import add_canonical_audit,build_positive_label_audit,load_canonical_habu_events,match_events_preserve_all,merge_canonical_capture_master
from .environmental import add_lunar_features,add_weather_derived_features,join_optional_tide_features
from .events import dedupe_events,join_weather,slope_features,species_from_text,strict_holdout_score
from .features import add_outcomes_bio,add_static_context
from .modeling import fit_model,score_holdout
from .output import write_map,write_qa
from .roads import combine_roads,recover_supplemental_roads

def apply_hardening(pipeline):
    pipeline._species_from_text=species_from_text; original_parse=pipeline.parse_field_log
    def hardened_run(root):
        cfg=pipeline.load_config(root);paths=pipeline.Paths(root);pipeline.ensure_dirs(paths);osm_segs=pipeline.build_10m_segments(root,cfg);points=pipeline.read_gpx_files(root);osm_mp=pipeline.map_match_gpx(points,osm_segs,cfg)
        files=sorted((root/"data"/"raw"/"logs").glob("*.txt"));raw_logs=pd.concat([original_parse(p) for p in files],ignore_index=True) if files else pd.DataFrame()
        canonical=load_canonical_habu_events(root)
        # The integrated workbook is authoritative for every Habu outcome. Keep raw logs
        # only for non-Habu biological reactions to avoid duplicate Habu events.
        if not raw_logs.empty:raw_logs=raw_logs[raw_logs.species!="ハブ"].copy()
        raw=merge_canonical_capture_master(raw_logs,canonical);events,removed=dedupe_events(raw);pre_match_rows=len(events)
        events_osm=match_events_preserve_all(pipeline,events,osm_segs) if not events.empty else events
        if len(events_osm)>pre_match_rows:raise RuntimeError(f"event road matching expanded rows: {pre_match_rows} -> {len(events_osm)}")
        gps_events_osm=events_osm[events_osm.lat.notna()&events_osm.lon.notna()] if not events_osm.empty else events_osm
        supplemental,recovery_audit=recover_supplemental_roads(points,osm_mp,gps_events_osm,osm_segs,cfg,root=root);segs=combine_roads(osm_segs,supplemental)
        if not supplemental.empty:segs.to_crs("EPSG:4326").to_file(paths.processed/"road_segments_10m.geojson",driver="GeoJSON")
        mp=pipeline.map_match_gpx(points,segs,cfg);visits=slope_features(pipeline.segment_visits(mp));events=match_events_preserve_all(pipeline,events,segs) if not events.empty else events
        if len(events)>pre_match_rows:raise RuntimeError(f"event road matching expanded rows after supplemental recovery: {pre_match_rows} -> {len(events)}")
        if not events.empty:
            missing_gps=events.lat.isna()|events.lon.isna();events["unmatched_reason"]=np.where(missing_gps,"missing GPS",np.where(events.segment_id.isna(),"nearest mapped road exceeds match threshold",""))
        canonical_audit=add_canonical_audit(events)
        canonical_audit["canonical_habu_events_total"]=int((events.species=="ハブ").sum())
        canonical_audit["no_capture_events"]=int(((events.species=="ハブ")&(events.event_type=="no_capture")).sum())
        canonical_audit["roadkill_events"]=int(((events.species=="ハブ")&events.event_type.isin(["轢死","roadkill_sighting"])).sum())
        canonical_audit["sighting_events"]=int(((events.species=="ハブ")&events.event_type.isin(["目撃","sighting"])).sum())
        (paths.reports/"canonical_master_audit.json").write_text(json.dumps(canonical_audit,ensure_ascii=False,indent=2),encoding="utf-8")
        (paths.reports/"gpx_road_recovery.json").write_text(json.dumps({"recovered_segment_count":int(len(supplemental)),"recovered_roads":recovery_audit},ensure_ascii=False,indent=2),encoding="utf-8")
        weather_events=events.dropna(subset=["timestamp"]) if not events.empty else events
        weather=pipeline.fetch_weather(root,weather_events,visits,cfg);data=join_weather(visits,weather);data=add_outcomes_bio(data,events,segs,cfg);data=add_static_context(data,segs,root);data=pipeline.add_exposure_features(data);data=add_lunar_features(data);data=add_weather_derived_features(data);data=join_optional_tide_features(data,root)
        data.to_csv(paths.processed/"learning_10m_road.csv",index=False);data.to_parquet(paths.processed/"learning_10m_road.parquet",index=False);mp.to_csv(paths.processed/"gpx_points_matched.csv",index=False);visits.to_csv(paths.processed/"segment_visits.csv",index=False)
        if not events.empty:events.to_csv(paths.processed/"events_matched.csv",index=False);events[events.segment_id.isna()].to_csv(paths.processed/"events_unmatched.csv",index=False)
        label_audit=build_positive_label_audit(events,data);label_audit.to_csv(paths.reports/"positive_label_audit.csv",index=False)
        metrics=fit_model(root,data,cfg);hold=score_holdout(root,data,cfg);strict=strict_holdout_score(events);(paths.reports/"strict_holdout_2026-08-28.json").write_text(json.dumps(strict,ensure_ascii=False,indent=2),encoding="utf-8");cut=pd.Timestamp(cfg["baseline_cutoff"]);forecast=pipeline.make_forecast(root,data[data.entered_at<cut].copy(),cfg);(paths.reports/"model_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8");(paths.reports/"latest_score.json").write_text(json.dumps(hold,ensure_ascii=False,indent=2),encoding="utf-8");write_map(root,segs,data,events,points,cfg);qa=write_qa(root,pipeline,cfg,points,raw,events,data,removed,strict)
        label_ok=int((label_audit.audit_status=="ok").sum()) if not label_audit.empty else 0
        qa["canonical_master"]=canonical_audit;qa["positive_label_audit"]={"capture_events":int(len(label_audit)),"labeled_ok":label_ok,"not_labeled":int(len(label_audit)-label_ok),"status_counts":label_audit.audit_status.value_counts(dropna=False).to_dict() if not label_audit.empty else {}}
        qa.setdefault("foundation_steps",{}).pop("2_40_habu_same_osm_10m",None);qa["foundation_steps"]["2_full_capture_master_road_match_and_label_audit"]="complete" if canonical_audit.get("capture_events")==206 and canonical_audit.get("capture_individuals")==208 and label_ok>0 else "partial"
        qa["feature_sources"]={"lunar":"deterministic synodic calculation","fog":"Open-Meteo WMO code 45/48 plus temperature-dewpoint proxy","temperature":"Open-Meteo hourly","tide":"JMA Amami O9 hourly predicted tide table; local authoritative tide_hourly.csv overrides when supplied"}
        (paths.reports/"qa_summary.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
        return {"segments":len(segs),"supplemental_segments":len(supplemental),"gpx_points":len(points),"visits":len(visits),"events":len(events),"duplicates_removed":removed,"canonical_master":canonical_audit,"positive_label_audit":qa["positive_label_audit"],"metrics":metrics,"holdout":hold,"strict_holdout":strict,"qa":qa,"forecast":forecast}
    pipeline.run=hardened_run
