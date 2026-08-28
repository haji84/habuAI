from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import gpxpy
import joblib
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import substring
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

JST = timezone(timedelta(hours=9))
GEOD = Geod(ellps="WGS84")


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def models(self) -> Path:
        return self.root / "models"


def load_config(root: Path) -> dict:
    return json.loads((root / "config" / "pipeline.json").read_text(encoding="utf-8"))


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.raw, paths.processed, paths.reports, paths.docs, paths.models]:
        p.mkdir(parents=True, exist_ok=True)


def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            yield g


def build_10m_segments(root: Path, cfg: dict) -> gpd.GeoDataFrame:
    src = root / "data" / "osm" / "setouchi-roads.geojson"
    if not src.exists():
        raise FileNotFoundError("OSM road extract missing; run scripts/fetch_osm.sh first")
    roads = gpd.read_file(src)
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    roads = roads.to_crs(cfg["crs_metric"])
    out = []
    for row_i, row in roads.iterrows():
        osm_id = row.get("@id") or row.get("id") or row.get("osm_id") or f"row{row_i}"
        way_id = str(osm_id).replace("way/", "").replace("w", "")
        highway = row.get("highway")
        name = row.get("name")
        for part_i, line in enumerate(_iter_lines(row.geometry) or []):
            if line.length <= 0:
                continue
            n = max(1, math.ceil(line.length / cfg["segment_length_m"]))
            for idx in range(n):
                a = idx * cfg["segment_length_m"]
                b = min((idx + 1) * cfg["segment_length_m"], line.length)
                seg = substring(line, a, b)
                if seg.is_empty or seg.length == 0:
                    continue
                out.append({
                    "segment_id": f"OSM_{way_id}_{part_i}_{idx}",
                    "way_id": way_id,
                    "part_index": part_i,
                    "segment_index": idx,
                    "length_m": float(seg.length),
                    "highway": highway,
                    "name": name,
                    "geometry": seg,
                })
    segs = gpd.GeoDataFrame(out, crs=roads.crs)
    segs["bearing_deg"] = segs.geometry.apply(_bearing_metric)
    segs["curvature_deg"] = segs.geometry.apply(_curvature)
    segs.to_file(root / "data" / "processed" / "road_segments_10m.geojson", driver="GeoJSON")
    return segs


def _bearing_metric(line: LineString) -> float:
    if len(line.coords) < 2:
        return np.nan
    x1, y1 = line.coords[0][:2]
    x2, y2 = line.coords[-1][:2]
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360) % 360


def _curvature(line: LineString) -> float:
    if len(line.coords) < 3:
        return 0.0
    b1 = _bearing_metric(LineString([line.coords[0], line.coords[len(line.coords)//2]]))
    b2 = _bearing_metric(LineString([line.coords[len(line.coords)//2], line.coords[-1]]))
    return float(min(abs(b1-b2), 360-abs(b1-b2)))


def read_gpx_files(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "data" / "raw" / "gpx").glob("*.gpx")):
        if path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            gpx = gpxpy.parse(f)
        seq = 0
        for track in gpx.tracks:
            for seg in track.segments:
                for p in seg.points:
                    dt = p.time
                    if dt is None:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(JST)
                    rows.append({
                        "session_file": path.name,
                        "seq": seq,
                        "timestamp": dt,
                        "lat": p.latitude,
                        "lon": p.longitude,
                        "elevation_m": p.elevation,
                    })
                    seq += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _add_track_dynamics(df)


def _add_track_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("session_file", sort=False):
        g = g.sort_values("seq").copy()
        prev_lat = g.lat.shift()
        prev_lon = g.lon.shift()
        prev_t = g.timestamp.shift()
        dist = []
        bearing = []
        for la1, lo1, la2, lo2 in zip(prev_lat, prev_lon, g.lat, g.lon):
            if pd.isna(la1):
                dist.append(np.nan); bearing.append(np.nan); continue
            az, _, m = GEOD.inv(lo1, la1, lo2, la2)
            dist.append(m); bearing.append((az + 360) % 360)
        g["step_m"] = dist
        g["heading_deg"] = bearing
        dt_s = (g.timestamp - prev_t).dt.total_seconds()
        g["speed_mps"] = g.step_m / dt_s.replace(0, np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _angle_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    d = abs(float(a) - float(b)) % 360
    return min(d, 360-d)


def map_match_gpx(points: pd.DataFrame, segs: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    p = gpd.GeoDataFrame(points.copy(), geometry=gpd.points_from_xy(points.lon, points.lat), crs="EPSG:4326")
    p = p.to_crs(segs.crs)
    sindex = segs.sindex
    matched = []
    last_seg_by_session = {}
    for row in p.itertuples():
        candidates_idx = list(sindex.query(row.geometry.buffer(cfg["map_match_max_distance_m"]), predicate="intersects"))
        if not candidates_idx:
            matched.append({**row._asdict(), "segment_id": None, "match_distance_m": np.nan, "match_score": np.nan})
            continue
        best = None
        prev_id = last_seg_by_session.get(row.session_file)
        for ci in candidates_idx:
            sr = segs.iloc[ci]
            dist = row.geometry.distance(sr.geometry)
            if dist > cfg["map_match_max_distance_m"]:
                continue
            angle = _angle_diff(row.heading_deg, sr.bearing_deg)
            angle = min(angle, abs(180-angle))
            heading_pen = cfg["heading_penalty_m"] * (angle / 90.0)
            continuity_pen = 0 if prev_id in (None, sr.segment_id) else cfg["continuity_penalty_m"]
            score = dist + heading_pen + continuity_pen
            if best is None or score < best[0]:
                best = (score, sr.segment_id, dist)
        if best is None:
            matched.append({**row._asdict(), "segment_id": None, "match_distance_m": np.nan, "match_score": np.nan})
        else:
            last_seg_by_session[row.session_file] = best[1]
            matched.append({**row._asdict(), "segment_id": best[1], "match_distance_m": best[2], "match_score": best[0]})
    out = pd.DataFrame(matched).drop(columns=["geometry", "Index"], errors="ignore")
    return out


def segment_visits(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched
    m = matched.dropna(subset=["segment_id"]).copy()
    m["prev_segment"] = m.groupby("session_file").segment_id.shift()
    m["new_visit"] = (m.segment_id != m.prev_segment).astype(int)
    m["visit_index"] = m.groupby("session_file").new_visit.cumsum()
    visits = m.groupby(["session_file", "visit_index", "segment_id"], as_index=False).agg(
        entered_at=("timestamp", "min"), exited_at=("timestamp", "max"),
        point_count=("seq", "count"), mean_speed_mps=("speed_mps", "mean"),
        mean_match_distance_m=("match_distance_m", "mean"), elevation_m=("elevation_m", "median")
    )
    visits["duration_s"] = (visits.exited_at - visits.entered_at).dt.total_seconds().clip(lower=0)
    return visits


def parse_field_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    dt_re = re.compile(r"^(2026)/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")
    coord_re = re.compile(r"^(28\.\d+)\s*$")
    lon_re = re.compile(r"^(129\.\d+)\s*$")
    rows = []
    current_session_start = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = dt_re.match(line)
        if not m:
            i += 1; continue
        y, mo, d, hh, mm, ss = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0), tzinfo=JST)
        block = [line]
        j = i + 1
        while j < len(lines) and not dt_re.match(lines[j].strip()):
            block.append(lines[j].strip()); j += 1
        text = "\n".join(block)
        if "エリア探索開始" in text:
            current_session_start = dt
        event_type = next((x for x in ["捕獲", "轢死", "目撃"] if re.search(rf"(^|\n){x}(\n|$)", text)), None)
        if event_type:
            lats = [float(x) for x in (coord_re.match(z).group(1) for z in block if coord_re.match(z))]
            lons = [float(x) for x in (lon_re.match(z).group(1) for z in block if lon_re.match(z))]
            lat = lats[-1] if lats else np.nan
            lon = lons[-1] if lons else np.nan
            species = _species_from_text(text)
            individual_count = _count_from_text(text)
            size = "大" if "ハブ捕獲大" in text else "中" if "ハブ捕獲中" in text else "小" if "ハブ捕獲小" in text else "極小" if "極小" in text else None
            wetness = next((w for w in ["ソークド(水跳ね)", "ウェット", "ウエット", "湿り", "ドライ"] if w in text), None)
            rows.append({
                "timestamp": dt,
                "session_start": current_session_start,
                "event_type": event_type,
                "species": species,
                "individual_count": individual_count,
                "size": size,
                "wetness": wetness,
                "lat": lat,
                "lon": lon,
                "raw_text": text,
            })
        i = j
    out = pd.DataFrame(rows)
    if not out.empty:
        out["night_date"] = out.apply(lambda r: (r.session_start or r.timestamp).date().isoformat(), axis=1)
    return out


def _species_from_text(text: str) -> str:
    keys = ["ハブ", "ヒメハブ", "アカマタ", "ガラスヒバァ", "ガラスヒヴァ", "リュウキュウアオヘビ", "ヒャン", "ネズミ", "オットンガエル", "イシカワガエル", "アマミハナサキガエル", "カエル", "ヤマシギ", "クロウサギ"]
    for k in keys:
        if k in text:
            return k
    return "その他"


def _count_from_text(text: str) -> int:
    m = re.search(r"(\d+)匹", text)
    if m:
        return int(m.group(1))
    if "5匹以上" in text:
        return 5
    return 1


def match_events(events: pd.DataFrame, segs: gpd.GeoDataFrame, max_m: float = 50.0) -> pd.DataFrame:
    e = events.dropna(subset=["lat", "lon"]).copy()
    if e.empty:
        return e
    eg = gpd.GeoDataFrame(e, geometry=gpd.points_from_xy(e.lon, e.lat), crs="EPSG:4326").to_crs(segs.crs)
    joined = gpd.sjoin_nearest(eg, segs[["segment_id", "geometry"]], how="left", distance_col="event_match_distance_m")
    joined.loc[joined.event_match_distance_m > max_m, "segment_id"] = None
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def fetch_weather(root: Path, events: pd.DataFrame, visits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    times = []
    if not events.empty: times.extend(events.timestamp.tolist())
    if not visits.empty: times.extend(visits.entered_at.tolist())
    if not times:
        return pd.DataFrame()
    start = min(times).date().isoformat(); end = max(times).date().isoformat()
    wcfg = cfg["weather"]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": wcfg["reference_lat"], "longitude": wcfg["reference_lon"],
        "start_date": start, "end_date": end, "timezone": "Asia/Tokyo",
        "hourly": ",".join(wcfg["hourly"]),
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Tokyo")
    df = df.rename(columns={"temperature_2m":"temperature_c", "relative_humidity_2m":"humidity_pct", "dew_point_2m":"dew_point_c", "precipitation":"precip_mm"})
    df["rain_1h_mm"] = df.precip_mm
    for hrs in [3,6,12,24,48]:
        df[f"rain_{hrs}h_mm"] = df.precip_mm.rolling(hrs, min_periods=1).sum()
    last_rain = None; since = []
    for t, p in zip(df.timestamp, df.precip_mm):
        if p > 0: last_rain = t
        since.append(np.nan if last_rain is None else (t-last_rain).total_seconds()/3600)
    df["hours_since_rain"] = since
    return df


def join_weather(visits: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if visits.empty or weather.empty:
        return visits
    v = visits.sort_values("entered_at").copy()
    w = weather.sort_values("timestamp").copy()
    return pd.merge_asof(v, w, left_on="entered_at", right_on="timestamp", direction="nearest", tolerance=pd.Timedelta("40min"))


def add_outcomes_and_bio(visits: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if visits.empty:
        return visits
    out = visits.copy()
    habu = events[(events.species == "ハブ") & (events.event_type == "捕獲")].copy() if not events.empty else pd.DataFrame()
    out["habu_capture"] = 0
    out["habu_individuals"] = 0
    if not habu.empty:
        for r in habu.itertuples():
            mask = (out.segment_id == r.segment_id) & (abs((out.entered_at - r.timestamp).dt.total_seconds()) <= 600)
            out.loc[mask, "habu_capture"] = 1
            out.loc[mask, "habu_individuals"] += int(r.individual_count or 1)
    bio = events[~events.species.isin(["ハブ", "ヒメハブ", "アカマタ", "ガラスヒバァ", "ガラスヒヴァ", "リュウキュウアオヘビ", "ヒャン"])].copy() if not events.empty else pd.DataFrame()
    for mins in cfg["biological_windows_minutes"]:
        for species in ["ネズミ", "オットンガエル", "カエル", "ヤマシギ", "クロウサギ"]:
            col = f"bio_{species}_{mins}m"
            vals = []
            for vr in out.itertuples():
                if bio.empty:
                    vals.append(0); continue
                b = bio[(bio.species == species) & (bio.timestamp <= vr.entered_at) & (bio.timestamp >= vr.entered_at - pd.Timedelta(minutes=mins))]
                vals.append(int(b.individual_count.sum()) if not b.empty else 0)
            out[col] = vals
    return out


def add_segment_static_features(df: pd.DataFrame, segs: gpd.GeoDataFrame) -> pd.DataFrame:
    static = segs[["segment_id", "highway", "length_m", "bearing_deg", "curvature_deg"]].copy()
    return df.merge(static, on="segment_id", how="left")


def add_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    x["hour"] = x.entered_at.dt.hour + x.entered_at.dt.minute / 60.0
    x["sin_hour"] = np.sin(2*np.pi*x.hour/24)
    x["cos_hour"] = np.cos(2*np.pi*x.hour/24)
    x["segment_prior_visits"] = x.sort_values("entered_at").groupby("segment_id").cumcount()
    return x


def fit_model(root: Path, data: pd.DataFrame, cfg: dict) -> dict:
    cutoff = pd.Timestamp(cfg["baseline_cutoff"])
    train = data[data.entered_at < cutoff].copy()
    if train.empty or train.habu_capture.nunique() < 2:
        return {"status":"insufficient-data"}
    numeric = [c for c in ["sin_hour","cos_hour","mean_speed_mps","elevation_m","rain_1h_mm","rain_3h_mm","rain_6h_mm","rain_12h_mm","rain_24h_mm","rain_48h_mm","temperature_c","humidity_pct","dew_point_c","hours_since_rain","curvature_deg","segment_prior_visits","bio_ネズミ_5m","bio_ネズミ_10m","bio_オットンガエル_10m","bio_カエル_10m","bio_ヤマシギ_10m"] if c in train.columns]
    X = train[numeric].replace([np.inf,-np.inf], np.nan).fillna(0)
    y = train.habu_capture.astype(int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    model.fit(X, y)
    p = model.predict_proba(X)[:,1]
    metrics = {"status":"ok","rows":len(train),"positives":int(y.sum()),"brier_train":float(brier_score_loss(y,p)),"features":numeric}
    joblib.dump({"model":model,"features":numeric}, root/"models"/"habu_occurrence.joblib")
    return metrics


def score_holdout(root: Path, data: pd.DataFrame, cfg: dict) -> dict:
    model_path = root/"models"/"habu_occurrence.joblib"
    if not model_path.exists(): return {"status":"no-model"}
    obj = joblib.load(model_path); model=obj["model"]; feats=obj["features"]
    hold = data[data.entered_at >= pd.Timestamp(cfg["holdout_start"])].copy()
    if hold.empty: return {"status":"no-holdout"}
    X = hold[feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    hold["pred_prob"] = model.predict_proba(X)[:,1]
    score = {"status":"ok","rows":len(hold),"actual_captures":int(hold.habu_capture.sum()),"mean_pred_prob":float(hold.pred_prob.mean())}
    if hold.habu_capture.nunique() > 1:
        score["brier"] = float(brier_score_loss(hold.habu_capture, hold.pred_prob))
    return score


def make_forecast(root: Path, data: pd.DataFrame, cfg: dict) -> dict:
    # Conservative v1: rank recent traversed segments by smoothed historical capture rate.
    if data.empty: return {"status":"no-data"}
    agg = data.groupby("segment_id").agg(visits=("segment_id","size"),captures=("habu_capture","sum"),individuals=("habu_individuals","sum")).reset_index()
    global_rate = (agg.captures.sum()+1)/(agg.visits.sum()+100)
    agg["smoothed_rate"] = (agg.captures + global_rate*20)/(agg.visits+20)
    top = agg.sort_values("smoothed_rate", ascending=False).head(30)
    by_hour = data.groupby(data.entered_at.dt.hour).agg(visits=("segment_id","size"),captures=("habu_capture","sum"))
    by_hour["rate"] = (by_hour.captures+0.5)/(by_hour.visits+5)
    hour = int(by_hour.rate.idxmax()) if len(by_hour) else 0
    result = {
        "status":"ok",
        "generated_at": datetime.now(JST).isoformat(),
        "method":"exposure-corrected segment passages with Bayesian-smoothed segment ranking; provisional time peak",
        "provisional_peak_start": f"{hour:02d}:00",
        "provisional_peak_end": f"{(hour+1)%24:02d}:00",
        "top_segment_ids": top.segment_id.tolist(),
        "warning":"Time peak remains provisional until exposure correction is validated across more nights."
    }
    (root/"reports"/"latest_prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_map(root: Path, segs: gpd.GeoDataFrame, data: pd.DataFrame, events: pd.DataFrame) -> None:
    docs = root/"docs"; (docs/"data").mkdir(parents=True, exist_ok=True)
    stats = data.groupby("segment_id").agg(visits=("segment_id","size"),captures=("habu_capture","sum"),individuals=("habu_individuals","sum")).reset_index() if not data.empty else pd.DataFrame(columns=["segment_id","visits","captures","individuals"])
    m = segs.merge(stats, on="segment_id", how="left").fillna({"visits":0,"captures":0,"individuals":0})
    m["risk"] = (m.captures + 0.5)/(m.visits + 5)
    m.to_crs("EPSG:4326").to_file(docs/"data"/"segments.geojson", driver="GeoJSON")
    if not events.empty:
        eg = gpd.GeoDataFrame(events.copy(), geometry=gpd.points_from_xy(events.lon, events.lat), crs="EPSG:4326")
        eg.to_file(docs/"data"/"events.geojson", driver="GeoJSON")
    html = '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>habuAI Map</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>html,body,#map{height:100%;margin:0}.panel{position:absolute;z-index:999;background:white;padding:8px;top:10px;right:10px;border-radius:8px}</style></head><body><div id="map"></div><div class="panel">habuAI 10m risk map</div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const map=L.map('map').setView([28.17,129.36],12);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:20,attribution:'© OpenStreetMap'}).addTo(map);function color(r){return r>.2?'#7f0000':r>.1?'#d7301f':r>.05?'#fc8d59':r>.02?'#fdcc8a':'#d9f0a3'}fetch('data/segments.geojson').then(r=>r.json()).then(g=>L.geoJSON(g,{style:f=>({color:color(f.properties.risk||0),weight:4,opacity:.8}),onEachFeature:(f,l)=>l.bindPopup(`segment: ${f.properties.segment_id}<br>visits: ${f.properties.visits}<br>captures: ${f.properties.captures}<br>risk: ${(f.properties.risk||0).toFixed(3)}`)}).addTo(map));fetch('data/events.geojson').then(r=>r.ok?r.json():null).then(g=>{if(g)L.geoJSON(g,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:f.properties.species==='ハブ'?6:3})}).addTo(map)});</script></body></html>'''
    (docs/"index.html").write_text(html, encoding="utf-8")


def run(root: Path) -> dict:
    cfg=load_config(root); paths=Paths(root); ensure_dirs(paths)
    segs=build_10m_segments(root,cfg)
    points=read_gpx_files(root)
    matched=map_match_gpx(points,segs,cfg)
    visits=segment_visits(matched)
    log_files=sorted((root/"data"/"raw"/"logs").glob("*.txt"))
    events=pd.concat([parse_field_log(p) for p in log_files],ignore_index=True) if log_files else pd.DataFrame()
    events=match_events(events,segs) if not events.empty else events
    weather=fetch_weather(root,events,visits,cfg)
    data=join_weather(visits,weather)
    data=add_outcomes_and_bio(data,events,cfg)
    data=add_segment_static_features(data,segs)
    data=add_exposure_features(data)
    data.to_csv(paths.processed/"learning_10m_road.csv",index=False)
    data.to_parquet(paths.processed/"learning_10m_road.parquet",index=False)
    matched.to_csv(paths.processed/"gpx_points_matched.csv",index=False)
    visits.to_csv(paths.processed/"segment_visits.csv",index=False)
    if not events.empty: events.to_csv(paths.processed/"events_matched.csv",index=False)
    metrics=fit_model(root,data,cfg)
    holdout=score_holdout(root,data,cfg)
    forecast=make_forecast(root,data,cfg)
    (paths.reports/"model_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    (paths.reports/"latest_score.json").write_text(json.dumps(holdout,ensure_ascii=False,indent=2),encoding="utf-8")
    write_map(root,segs,data,events)
    return {"segments":len(segs),"gpx_points":len(points),"visits":len(visits),"events":len(events),"metrics":metrics,"holdout":holdout,"forecast":forecast}
