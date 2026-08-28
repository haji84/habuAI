from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .events import dedupe_events,join_weather,slope_features,species_from_text,strict_holdout_score
from .features import add_outcomes_bio,add_static_context
from .modeling import fit_model,score_holdout
from .output import write_map,write_qa

def apply_hardening(pipeline):
    pipeline._species_from_text=species_from_text; original_parse=pipeline.parse_field_log
    def hardened_run(root):
        cfg=pipeline.load_config(root);paths=pipeline.Paths(root);pipeline.ensure_dirs(paths);segs=pipeline.build_10m_segments(root,cfg);points=pipeline.read_gpx_files(root);mp=pipeline.map_match_gpx(points,segs,cfg);visits=slope_features(pipeline.segment_visits(mp))
        files=sorted((root/"data"/"raw"/"logs").glob("*.txt"));raw=pd.concat([original_parse(p) for p in files],ignore_index=True) if files else pd.DataFrame();events,removed=dedupe_events(raw);events=pipeline.match_events(events,segs) if not events.empty else events
        if not events.empty:events["unmatched_reason"]=np.where(events.segment_id.isna(),"nearest OSM highway exceeds match threshold","")
        weather=pipeline.fetch_weather(root,events,visits,cfg);data=join_weather(visits,weather);data=add_outcomes_bio(data,events,segs,cfg);data=add_static_context(data,segs,root);data=pipeline.add_exposure_features(data)
        data.to_csv(paths.processed/"learning_10m_road.csv",index=False);data.to_parquet(paths.processed/"learning_10m_road.parquet",index=False);mp.to_csv(paths.processed/"gpx_points_matched.csv",index=False);visits.to_csv(paths.processed/"segment_visits.csv",index=False)
        if not events.empty:events.to_csv(paths.processed/"events_matched.csv",index=False);events[events.segment_id.isna()].to_csv(paths.processed/"events_unmatched.csv",index=False)
        metrics=fit_model(root,data,cfg);hold=score_holdout(root,data,cfg);strict=strict_holdout_score(events);(paths.reports/"strict_holdout_2026-08-28.json").write_text(json.dumps(strict,ensure_ascii=False,indent=2),encoding="utf-8");cut=pd.Timestamp(cfg["baseline_cutoff"]);forecast=pipeline.make_forecast(root,data[data.entered_at<cut].copy(),cfg);(paths.reports/"model_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8");(paths.reports/"latest_score.json").write_text(json.dumps(hold,ensure_ascii=False,indent=2),encoding="utf-8");write_map(root,segs,data,events,points,cfg);qa=write_qa(root,pipeline,cfg,points,raw,events,data,removed,strict)
        return {"segments":len(segs),"gpx_points":len(points),"visits":len(visits),"events":len(events),"duplicates_removed":removed,"metrics":metrics,"holdout":hold,"strict_holdout":strict,"qa":qa,"forecast":forecast}
    pipeline.run=hardened_run
