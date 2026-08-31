from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import _as_jst, _estimator, _fit, _spatial_feature_sets
from habuai.hardening.spatial_history import add_historical_spatial_features
from run_joint_point_time_backtest_v2 import hav, curve24

OWNER_PROFILE = "owner"
PRIVATE_SALT = "habuai-private-route-v1"


def _private_id(segment_id: object) -> str:
    raw = f"{PRIVATE_SALT}:{segment_id}".encode("utf-8")
    return "R-" + hashlib.sha256(raw).hexdigest()[:10]


def _circular_minutes(a: pd.Timestamp, b: pd.Timestamp) -> float:
    d = abs((a - b).total_seconds()) / 60.0
    return float(min(d, 1440.0 - d % 1440.0))


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"eligible_captures": 0}
    d = pd.DataFrame(rows)
    out = {"nights_scored": int(d.night.nunique()), "eligible_captures": int(len(d))}
    for mins in [10, 20, 30, 60]:
        for rad in [50, 100, 250]:
            k = f"hit_{rad}m_{mins}m"
            out[k] = int(d[k].sum())
            out[k + "_rate"] = float(d[k].mean())
    for k in ["top1_hit_100m_30m", "top3_hit_100m_30m", "top5_hit_100m_30m", "top10pct_hit_100m_30m"]:
        out[k] = int(d[k].sum()); out[k + "_rate"] = float(d[k].mean())
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = pipeline.load_config(root)
    p = root / "data" / "processed"; r = root / "reports"; r.mkdir(exist_ok=True)
    tour = json.loads((r / "model_tournament_summary.json").read_text(encoding="utf-8"))
    original40 = json.loads((r / "original40_full_history_summary.json").read_text(encoding="utf-8"))
    sw = tour["spatial"]["best"]
    tw = original40["best_selected_on_early_period"]

    recon = pd.read_csv(p / "reconstructed_spatial_learning_2026-05_07.csv", low_memory=False)
    md = pd.read_csv(p / "learning_10m_road.csv", low_memory=False)
    ev = pd.read_csv(p / "events_matched.csv", low_memory=False)
    segs = gpd.read_file(p / "road_segments_10m.geojson").to_crs("EPSG:6669")
    md["entered_at"] = _as_jst(md.entered_at); ev["timestamp"] = _as_jst(ev.timestamp)
    ev["lat"] = pd.to_numeric(ev.lat, errors="coerce"); ev["lon"] = pd.to_numeric(ev.lon, errors="coerce")
    md["night"] = (md.entered_at - pd.Timedelta(hours=7)).dt.date.astype(str)
    gpx = md[md.learning_row_source == "gpx_visit"].copy()

    static = [c for c in ["length_m","bearing_deg","curvature_deg","road_class_code","junction_distance_m","stream_distance_m","coast_distance_m","forest_distance_m","farmland_distance_m","residential_distance_m"] if c in md]
    agg = {c:(c,"first") for c in static}; agg["spatial_label"]=("habu_capture","max")
    aug = md.groupby(["night","segment_id"],as_index=False).agg(**agg)
    aug["sample_weight"] = 1.; aug["reconstruction_confidence"] = "actual_august"
    aug = aug.drop(columns=[c for c in aug if c.startswith("hist_") or c.startswith("days_since_capture_") or c in {"segment_x_m","segment_y_m"}], errors="ignore")
    aug,_ = add_historical_spatial_features(aug, ev, segs, cfg, root=None)
    combined = pd.concat([recon, aug], ignore_index=True, sort=False)
    feats = _spatial_feature_sets(combined)[str(sw["feature_set"])]

    coord = segs[["segment_id","geometry"]].copy().to_crs("EPSG:4326")
    coord["lat"] = coord.geometry.apply(lambda z:z.centroid.y); coord["lon"] = coord.geometry.apply(lambda z:z.centroid.x)
    coord = coord.drop_duplicates("segment_id").set_index("segment_id")[["lat","lon"]]

    radius=float(tw["radius_m"]); bw=float(tw["bandwidth_h"]); tau=tw.get("recency_tau_days")
    tau=None if tau is None else float(tau)
    weights = [(1.,1.),(1.,2.),(2.,1.),(.5,1.),(1.,.5)]
    all_variants=[]; prediction_rows=[]

    nights=sorted(gpx.night.unique())
    split=max(4,int(np.floor(len(nights)*.60)))
    selection=set(nights[:split]); confirmation=set(nights[split:])

    for alpha,beta in weights:
        rows=[]
        for night in nights:
            start=pd.Timestamp(night,tz="Asia/Tokyo")+pd.Timedelta(hours=7); end=start+pd.Timedelta(days=1)
            train=combined[combined.night.astype(str)<night]
            vis=gpx[(gpx.entered_at>=start)&(gpx.entered_at<end)].copy()
            actual=ev[(ev.species=="ハブ")&(ev.event_type=="捕獲")&(ev.timestamp>=start)&(ev.timestamp<end)&ev.lat.notna()&ev.lon.notna()].copy()
            if vis.empty or actual.empty or train.spatial_label.nunique()<2: continue
            m=_estimator(str(sw["estimator"])); _fit(m,train.reindex(columns=feats).replace([np.inf,-np.inf],np.nan),train.spatial_label.astype(int),pd.to_numeric(train.sample_weight,errors="coerce").fillna(1.).to_numpy())
            ts=aug[aug.night.astype(str)==night].copy(); ts["spatial_p"]=m.predict_proba(ts.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1]
            vis["spatial_p"]=vis.segment_id.map(ts.set_index("segment_id").spatial_p).fillna(0.)
            past=ev[ev.timestamp<start]; curves={}
            for sid in vis.segment_id.dropna().unique():
                if sid in coord.index:
                    la,lo=coord.loc[sid,["lat","lon"]]; curves[sid]=curve24(past,start,la,lo,radius,bw,tau)
            # Score every observed route visit at its real passage time. This is owner-personal route replay.
            vis["time_p"]=[float(curves.get(sid,np.ones(24))[int(t.hour)]) for sid,t in zip(vis.segment_id,vis.entered_at)]
            vis["route_score"]=(np.clip(vis.spatial_p,1e-9,1)**alpha)*(np.clip(vis.time_p,1e-9,1)**beta)
            vis=vis.sort_values("route_score",ascending=False).reset_index(drop=True); vis["rank"]=np.arange(1,len(vis)+1); vis["pct"]=vis["rank"]/len(vis)
            vc=vis.merge(coord,left_on="segment_id",right_index=True,how="left")
            for a in actual.itertuples():
                rr={"night":night,"phase":"selection" if night in selection else "confirmation"}
                for mins in [10,20,30,60]:
                    for rad in [50,100,250]:
                        dt=np.array([_circular_minutes(a.timestamp,t) for t in vc.entered_at])
                        dist=hav(a.lat,a.lon,vc.lat.to_numpy(float),vc.lon.to_numpy(float))
                        rr[f"hit_{rad}m_{mins}m"]=int(((dist<=rad)&(dt<=mins)).any())
                for n,k in [(1,"top1_hit_100m_30m"),(3,"top3_hit_100m_30m"),(5,"top5_hit_100m_30m")]:
                    z=vc.head(n); dt=np.array([_circular_minutes(a.timestamp,t) for t in z.entered_at]); dist=hav(a.lat,a.lon,z.lat.to_numpy(float),z.lon.to_numpy(float)); rr[k]=int(((dist<=100)&(dt<=30)).any())
                z=vc[vc.pct<=.10]; dt=np.array([_circular_minutes(a.timestamp,t) for t in z.entered_at]); dist=hav(a.lat,a.lon,z.lat.to_numpy(float),z.lon.to_numpy(float)); rr["top10pct_hit_100m_30m"]=int(((dist<=100)&(dt<=30)).any())
                rows.append(rr)
            # Public-safe prediction rows: hashed road IDs only, no coordinates or road names.
            for z in vc.head(10).itertuples():
                prediction_rows.append({"night":night,"alpha":alpha,"beta":beta,"private_route_id":_private_id(z.segment_id),"passage_time":z.entered_at.isoformat(),"route_score":float(z.route_score),"rank":int(z.rank)})
        d=pd.DataFrame(rows)
        sel=d[d.phase=="selection"] if not d.empty else d; conf=d[d.phase=="confirmation"] if not d.empty else d
        sm=_metrics(sel.to_dict("records")); cm=_metrics(conf.to_dict("records")); am=_metrics(d.to_dict("records"))
        score=3*sm.get("top3_hit_100m_30m_rate",0)+2*sm.get("hit_100m_30m_rate",0)+sm.get("hit_250m_30m_rate",0)
        all_variants.append({"alpha":alpha,"beta":beta,"selection_score":score,"selection":sm,"confirmation":cm,"all":am})

    all_variants.sort(key=lambda x:x["selection_score"],reverse=True); best=all_variants[0]
    safe=pd.DataFrame(prediction_rows); safe=safe[(safe.alpha==best["alpha"])&(safe.beta==best["beta"])].copy(); safe.to_csv(r/"personal_route_score_predictions_private_ids.csv",index=False)
    summary={
      "status":"ok","profile_mode":"owner_personal","privacy":{
        "secret_hotspot_policy":"Never export raw segment_id, road name, latitude, longitude, or a ranked hotspot list in shareable reports.",
        "shareable_identifier":"salted SHA-256 private_route_id only",
        "other_user_policy":"Each other user gets an isolated profile_id and learns from that user's own GPX/search/capture history. Owner private hotspot labels and owner route rankings are never transferred or exposed. Population/ecology priors may be shared only after removing owner-specific route identity and exact hotspot leakage."
      },
      "protocol":{"actual_gpx_nights":len(nights),"selection_nights":len(selection),"confirmation_nights":len(confirmation),"time_resolution_target_min":10,"distance_thresholds_m":[50,100,250],"time_thresholds_min":[10,20,30,60],"candidate_unit":"actual GPX road visit at actual passage time","leakage_guard":"target-night captures are outcomes only; models train on prior nights"},
      "score_definition":"route_score = spatial_occupancy^alpha * point-conditioned_time^beta. Night count is intentionally not used for within-night ranking because it is a night-wide scalar. Detection/environmental components remain separate until enough timed GPX evidence supports leakage-safe calibration.",
      "variants":all_variants,"best_selected_on_early_gpx_nights":best,
      "guardrail":"This is an owner-personal retrospective route replay on roads actually visited. It measures how well the score prioritizes the owner's route, not general-public road accuracy and not unvisited-road discovery."
    }
    (r/"personal_route_score_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
