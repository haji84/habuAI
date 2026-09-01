from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

import run_zone_models_89night_pdca as b
from habuai.hardening.model_tournament import _as_jst
from run_advanced_time_models import _fetch_weather

FINAL_ZONE_M = 250
PARENT_ZONE_M = 1000
COUNTS = [30, 50, 100]
RADII = [50, 100, 250]
WINDOWS = [10, 20, 30]
MIN_PRIOR = 10
MAX_UNITS = 500
MODELS = [
    'hierarchical_rank',
    'mixture_experts',
    'graph_time',
    'hawkes_st',
    'conformal_ensemble',
]


def parse_xy(uid: str) -> tuple[float, float]:
    a, z = str(uid).split(':')
    return float(a), float(z)


def unit_center(uid: str, size: int = FINAL_ZONE_M) -> tuple[float, float]:
    x, y = parse_xy(uid)
    return x + size / 2.0, y + size / 2.0


def parent_id(uid: str) -> str:
    x, y = parse_xy(uid)
    px = int(np.floor(x / PARENT_ZONE_M) * PARENT_ZONE_M)
    py = int(np.floor(y / PARENT_ZONE_M) * PARENT_ZONE_M)
    return f'{px}:{py}'


def time_prior_vec(cap: pd.DataFrame, mins: np.ndarray, bw_min: float = 60.0) -> np.ndarray:
    if cap.empty:
        return np.full(len(mins), 1e-4, dtype=float)
    cm = (cap.timestamp.dt.hour * 60 + cap.timestamp.dt.minute).to_numpy(float)
    d = np.abs(cm[:, None] - mins[None, :])
    d = np.minimum(d, 1440.0 - d)
    return np.mean(np.exp(-0.5 * (d / bw_min) ** 2), axis=0) + 1e-6


def normalize_score(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.nanpercentile(x[finite], [1, 99])
    if hi <= lo:
        return np.full_like(x, 0.5)
    y = (x - lo) / (hi - lo)
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method='average', pct=True).to_numpy(float)


def preselect_units(pv: pd.DataFrame, pc: pd.DataFrame) -> list[str]:
    vc = pv.unit_id.value_counts()
    cc = pc.unit_id.value_counts()
    units = sorted(set(vc.index) | set(cc.index))
    ranked = []
    for u in units:
        v = float(vc.get(u, 0))
        c = float(cc.get(u, 0))
        occ = (c + 1.0) / (v + c + 20.0)
        ranked.append((u, occ, c, -v))
    return [z[0] for z in sorted(ranked, key=lambda z: (z[1], z[2], z[3]), reverse=True)[:MAX_UNITS]]


def make_candidates(night: str, gpx: pd.DataFrame, cap: pd.DataFrame, ctx: pd.DataFrame):
    start = pd.Timestamp(night, tz='Asia/Tokyo') + pd.Timedelta(hours=7)
    pv = gpx[gpx.entered_at < start].copy()
    pc = cap[cap.timestamp < start].copy()
    if len(pc) < MIN_PRIOR:
        return pd.DataFrame(), pv, pc
    units = preselect_units(pv, pc)
    if not units:
        return pd.DataFrame(), pv, pc

    times = b.slot_times(night)
    mins = (times.hour * 60 + times.minute).to_numpy(float)
    gt = time_prior_vec(pc, mins)
    weather = {c: (float(ctx.loc[night, c]) if night in ctx.index and pd.notna(ctx.loc[night, c]) else np.nan) for c in ctx.columns}

    pvg = pv.assign(parent=pv.unit_id.map(parent_id))
    pcg = pc.assign(parent=pc.unit_id.map(parent_id))
    frames = []
    for uid in units:
        zv = pv[pv.unit_id == uid]
        zc = pc[pc.unit_id == uid]
        pid = parent_id(uid)
        pzv = pvg[pvg.parent == pid]
        pzc = pcg[pcg.parent == pid]
        visits = len(zv)
        captures = len(zc)
        pvisits = len(pzv)
        pcaptures = len(pzc)
        child_occ = (captures + 1.0) / (captures + visits + 20.0)
        parent_occ = (pcaptures + 1.0) / (pcaptures + pvisits + 30.0)
        child_det = (captures + 1.0) / (visits + 12.0)
        parent_det = (pcaptures + 1.0) / (pvisits + 20.0)
        ct = time_prior_vec(zc, mins)
        pt = time_prior_vec(pzc, mins, bw_min=75.0)
        x, y = unit_center(uid)
        data = {
            'unit_id': np.repeat(uid, len(times)),
            'slot_time': times,
            'x': x,
            'y': y,
            'child_occ': child_occ,
            'parent_occ': parent_occ,
            'child_det': child_det,
            'parent_det': parent_det,
            'child_time': ct,
            'parent_time': pt,
            'global_time': gt,
            'captures': captures,
            'parent_captures': pcaptures,
        }
        for k, v in weather.items():
            data[k] = v
        frames.append(pd.DataFrame(data))
    cand = pd.concat(frames, ignore_index=True)
    return cand, pv, pc


def hierarchical_score(cand: pd.DataFrame) -> np.ndarray:
    return (
        np.power(cand.parent_occ.to_numpy(float) + 1e-6, 0.75)
        * np.power(cand.child_occ.to_numpy(float) + 1e-6, 0.65)
        * np.power(cand.parent_time.to_numpy(float) + 1e-6, 0.65)
        * np.power(cand.child_time.to_numpy(float) + 1e-6, 0.8)
        * np.power(cand.global_time.to_numpy(float) + 1e-6, 0.35)
        * np.power(cand.child_det.to_numpy(float) + 1e-6, 0.20)
    )


def context_subset(pc: pd.DataFrame, target_ctx: pd.Series | None) -> pd.DataFrame:
    if pc.empty or target_ctx is None:
        return pc
    q = pc.copy()
    if 'rain_24h_mm' in q.columns and pd.notna(target_ctx.get('rain_24h_mm', np.nan)):
        wet = float(target_ctx['rain_24h_mm']) >= 1.0
        qq = q[(pd.to_numeric(q.rain_24h_mm, errors='coerce').fillna(0) >= 1.0) == wet]
        if len(qq) >= 8:
            q = qq
    if 'relative_humidity_2m' in q.columns and pd.notna(target_ctx.get('relative_humidity_2m', np.nan)):
        humid = float(target_ctx['relative_humidity_2m']) >= 85.0
        qq = q[(pd.to_numeric(q.relative_humidity_2m, errors='coerce').fillna(0) >= 85.0) == humid]
        if len(qq) >= 8:
            q = qq
    return q


def mixture_expert_score(cand: pd.DataFrame, pc: pd.DataFrame, night: str, ctx: pd.DataFrame) -> np.ndarray:
    target_ctx = ctx.loc[night] if night in ctx.index else None
    # Expert 1: global context-matched capture-time expert.
    selected = context_subset(pc, target_ctx)
    mins = (cand.slot_time.dt.hour * 60 + cand.slot_time.dt.minute).to_numpy(float)
    # compute once per unique slot, then map back
    unique_mins, inv = np.unique(mins, return_inverse=True)
    tcurve = time_prior_vec(selected, unique_mins, bw_min=50.0)[inv]

    # Expert 2: season-memory expert, recent 60d gets higher mass.
    if pc.empty:
        recent_occ = np.ones(len(cand)) * 1e-4
    else:
        start = cand.slot_time.iloc[0]
        recent = pc[pc.timestamp >= start - pd.Timedelta(days=60)]
        rc = recent.unit_id.value_counts()
        recent_occ = np.array([(float(rc.get(u, 0)) + 0.5) for u in cand.unit_id], dtype=float)

    # Expert 3: route-personal occupancy/detection expert.
    route = np.sqrt((cand.child_occ.to_numpy(float) + 1e-6) * (cand.child_det.to_numpy(float) + 1e-6))

    wet_gate = 0.5
    humid_gate = 0.5
    if target_ctx is not None:
        rv = target_ctx.get('rain_24h_mm', np.nan)
        hv = target_ctx.get('relative_humidity_2m', np.nan)
        if pd.notna(rv):
            wet_gate = min(1.0, max(0.0, float(rv) / 10.0))
        if pd.notna(hv):
            humid_gate = min(1.0, max(0.0, (float(hv) - 70.0) / 25.0))
    w_time = 0.35 + 0.20 * wet_gate
    w_recent = 0.25 + 0.15 * humid_gate
    w_route = max(0.15, 1.0 - w_time - w_recent)
    return w_time * normalize_score(tcurve) + w_recent * normalize_score(recent_occ) + w_route * normalize_score(route)


def graph_unit_intensity(units: list[str], pc: pd.DataFrame, target: pd.Timestamp) -> dict[str, float]:
    if not units:
        return {}
    ux = np.array([unit_center(u)[0] for u in units], dtype=float)
    uy = np.array([unit_center(u)[1] for u in units], dtype=float)
    out = {u: 0.0 for u in units}
    if pc.empty:
        return out
    cp = pc.copy()
    cp['cx'] = cp.unit_id.map(lambda u: unit_center(u)[0])
    cp['cy'] = cp.unit_id.map(lambda u: unit_center(u)[1])
    ages = np.maximum(0.0, (target - cp.timestamp).dt.total_seconds().to_numpy(float) / 86400.0)
    agew = np.exp(-ages / 45.0)
    cx = cp.cx.to_numpy(float)
    cy = cp.cy.to_numpy(float)
    for i, u in enumerate(units):
        dist = np.sqrt((cx - ux[i]) ** 2 + (cy - uy[i]) ** 2)
        out[u] = float(np.sum(agew * np.exp(-dist / 500.0)))
    return out


def graph_score(cand: pd.DataFrame, pc: pd.DataFrame) -> np.ndarray:
    target = cand.slot_time.iloc[0]
    units = cand.unit_id.drop_duplicates().tolist()
    gi = graph_unit_intensity(units, pc, target)
    spatial = np.array([gi.get(u, 0.0) for u in cand.unit_id], dtype=float)
    return (
        0.45 * normalize_score(spatial)
        + 0.25 * normalize_score(cand.child_occ.to_numpy(float))
        + 0.15 * normalize_score(cand.parent_occ.to_numpy(float))
        + 0.15 * normalize_score(cand.global_time.to_numpy(float))
    )


def hawkes_score(cand: pd.DataFrame, pc: pd.DataFrame) -> np.ndarray:
    if pc.empty:
        return np.zeros(len(cand), dtype=float)
    cp = pc.copy()
    cp['cx'] = cp.unit_id.map(lambda u: unit_center(u)[0])
    cp['cy'] = cp.unit_id.map(lambda u: unit_center(u)[1])
    cx = cp.cx.to_numpy(float)
    cy = cp.cy.to_numpy(float)
    cm = (cp.timestamp.dt.hour * 60 + cp.timestamp.dt.minute).to_numpy(float)
    out = np.empty(len(cand), dtype=float)
    # vector per candidate; bounded candidate set keeps this tractable.
    for i, z in enumerate(cand.itertuples()):
        age = np.maximum(0.0, (z.slot_time - cp.timestamp).dt.total_seconds().to_numpy(float) / 86400.0)
        dist = np.sqrt((cx - float(z.x)) ** 2 + (cy - float(z.y)) ** 2)
        m = float(z.slot_time.hour * 60 + z.slot_time.minute)
        td = np.abs(cm - m)
        td = np.minimum(td, 1440.0 - td)
        out[i] = float(np.sum(np.exp(-age / 60.0) * np.exp(-dist / 750.0) * np.exp(-0.5 * (td / 75.0) ** 2)))
    return out


def score_all(cand: pd.DataFrame, pc: pd.DataFrame, night: str, ctx: pd.DataFrame) -> dict[str, np.ndarray]:
    h = hierarchical_score(cand)
    m = mixture_expert_score(cand, pc, night, ctx)
    g = graph_score(cand, pc)
    hw = hawkes_score(cand, pc)
    ens = (rank01(h) + rank01(m) + rank01(g) + rank01(hw)) / 4.0
    return {
        'hierarchical_rank': h,
        'mixture_experts': m,
        'graph_time': g,
        'hawkes_st': hw,
        'conformal_ensemble': ens,
    }


def first_rank_hit(a, ranked: pd.DataFrame, members, segll, rad=100, mins=30, max_rank=500):
    limit = min(max_rank, len(ranked))
    for k in range(1, limit + 1):
        row = ranked.iloc[[k - 1]]
        if b.hit(a, row, members, segll, rad, mins):
            return k
    return None


def summarize(d: pd.DataFrame):
    out = []
    for model in MODELS:
        for count in COUNTS:
            z = d[(d.model == model) & (d.n_predictions == count)]
            item = {
                'model': model,
                'n_predictions': count,
                'eligible_capture_events': int(len(z)),
                'eligible_nights': int(z.night_index.nunique()) if not z.empty else 0,
                'coverage': {},
            }
            for rad in RADII:
                for mins in WINDOWS:
                    key = f'hit_{rad}m_{mins}min'
                    item['coverage'][f'{rad}m_{mins}min'] = float(z[key].mean()) if not z.empty else None
            out.append(item)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    p = root / 'data' / 'processed'
    r = root / 'reports'
    r.mkdir(exist_ok=True)

    md = pd.read_csv(p / 'learning_10m_road.csv', low_memory=False)
    ev = pd.read_csv(p / 'events_matched.csv', low_memory=False)
    md['entered_at'] = _as_jst(md.entered_at)
    ev['timestamp'] = _as_jst(ev.timestamp)
    md['night'] = b.op_night(md.entered_at)
    ev['night'] = b.op_night(ev.timestamp)
    ev['lat'] = pd.to_numeric(ev.lat, errors='coerce')
    ev['lon'] = pd.to_numeric(ev.lon, errors='coerce')

    gpx0 = md[md.learning_row_source == 'gpx_visit'][['segment_id', 'entered_at', 'night']].dropna().copy()
    cap0 = ev[(ev.species == 'ハブ') & (ev.event_type == '捕獲') & ev.timestamp.notna() & ev.segment_id.notna() & ev.lat.notna() & ev.lon.notna()][['segment_id', 'timestamp', 'lat', 'lon', 'night']].copy()
    field_nights = sorted(ev.night.dropna().unique().tolist())

    seg = gpd.read_file(p / 'road_segments_10m.geojson').to_crs('EPSG:6669')
    seg['x'] = seg.geometry.centroid.x
    seg['y'] = seg.geometry.centroid.y
    segxy = seg[['segment_id', 'x', 'y']].drop_duplicates('segment_id')
    cent = gpd.GeoSeries(seg.geometry.centroid, crs='EPSG:6669').to_crs('EPSG:4326')
    segll = pd.DataFrame({'segment_id': seg.segment_id.to_numpy(), 'lat': cent.y.to_numpy(), 'lon': cent.x.to_numpy()}).drop_duplicates('segment_id').set_index('segment_id')

    gpx = b.assign_units(gpx0, segxy, FINAL_ZONE_M)
    cap = b.assign_units(cap0, segxy, FINAL_ZONE_M)
    members = b.unit_members(segxy, FINAL_ZONE_M)

    weather = _fetch_weather(
        str((pd.Timestamp(field_nights[0]) - pd.Timedelta(days=3)).date()),
        str(pd.Timestamp(field_nights[-1]).date()),
        p / 'cache' / 'openmeteo_five_models_pdca89.csv',
    )
    ctx = b.weather_contexts(weather, field_nights)

    # Attach the known 18:00 context to historical captures so expert gating only uses information available after that night is completed.
    for c in ctx.columns:
        cap[c] = cap.night.map(ctx[c].to_dict())

    results = []
    first_ranks = {m: [] for m in MODELS}
    scored_nights = []

    for idx, night in enumerate(field_nights, 1):
        actual = cap[cap.night == night]
        if actual.empty:
            continue
        cand, pv, pc = make_candidates(night, gpx, cap, ctx)
        if cand.empty:
            continue
        scores = score_all(cand, pc, night, ctx)
        scored_nights.append(idx)

        for model, score in scores.items():
            ranked = cand[['unit_id', 'slot_time']].copy()
            ranked['score'] = score
            ranked = ranked.sort_values('score', ascending=False).drop_duplicates(['unit_id', 'slot_time']).reset_index(drop=True)

            for a in actual.itertuples():
                fr = first_rank_hit(a, ranked, members, segll, 100, 30, 500)
                if fr is not None:
                    first_ranks[model].append(fr)

            for count in COUNTS:
                top = ranked.head(count)
                for a in actual.itertuples():
                    row = {'model': model, 'night_index': idx, 'n_predictions': count}
                    for rad in RADII:
                        for mins in WINDOWS:
                            row[f'hit_{rad}m_{mins}min'] = b.hit(a, top, members, segll, rad, mins)
                    results.append(row)

    d = pd.DataFrame(results)
    overall = summarize(d)

    progression = {}
    for model in MODELS:
        ns = sorted(d[d.model == model].night_index.unique()) if not d.empty else []
        blocks = [list(x) for x in np.array_split(np.array(ns, dtype=int), 3)] if ns else []
        progression[model] = {}
        for i, block in enumerate(blocks, 1):
            progression[model][f'block_{i}'] = summarize(d[(d.model == model) & (d.night_index.isin(block))])

    strict = {}
    for count in COUNTS:
        strict[str(count)] = {
            x['model']: x['coverage']['100m_10min']
            for x in overall
            if x['n_predictions'] == count
        }

    adaptive90 = {}
    for model in MODELS:
        ranks = sorted(first_ranks[model])
        adaptive90[model] = {
            'calibration_hits_with_rank': len(ranks),
            'empirical_rank_needed_for_90pct_100m_30min': int(np.quantile(ranks, 0.90, method='higher')) if ranks else None,
            'note': 'empirical rolling-style calibration only; not a formal iid conformal coverage guarantee for dependent time-series data',
        }

    out = {
        'status': 'ok',
        'method': '89-night sequential PDCA tournament of five advanced road-time model families on a common 250m-zone x 10-minute output space',
        'field_evidence_nights_total': len(field_nights),
        'gps_timestamp_capture_events_total': int(len(cap0)),
        'gps_timestamp_capture_nights_total': int(cap0.night.nunique()),
        'protocol': {
            'cycle': 'predict using prior verified data only -> reveal result -> score -> add verified capture/GPX -> next night',
            'models': MODELS,
            'common_final_zone_m': FINAL_ZONE_M,
            'prediction_counts': COUNTS,
            'distance_m': RADII,
            'time_windows_min': WINDOWS,
            'slot_minutes': 10,
            'min_prior_gps_captures': MIN_PRIOR,
            'max_preselected_units': MAX_UNITS,
            'no_future_gpx': True,
            'no_target_outcome_before_prediction': True,
            'fairness': 'all five models rank the same 250m-zone x 10-minute candidate space so coarse-zone area cannot inflate one model by itself',
        },
        'implementations': {
            'hierarchical_rank': '1km parent occupancy/time fused with 250m child occupancy/detection/time and global time prior',
            'mixture_experts': 'context-gated wet/dry, humidity, recent-season and route-personal expert scores combined from pre-night evidence',
            'graph_time': '250m road-zone neighborhood capture intensity with spatial decay plus occupancy and global time prior',
            'hawkes_st': 'direct spatiotemporal self-exciting intensity with 60d temporal, 750m spatial and 75min time-of-day kernels',
            'conformal_ensemble': 'rank ensemble of the other four plus empirical 90% candidate-set calibration; formal conformal guarantee is not claimed',
        },
        'strict_scored_nights': len(set(scored_nights)),
        'overall': overall,
        'progression_thirds': progression,
        'strict_100m_10min_comparison': strict,
        'empirical_90pct_candidate_need_100m_30min': adaptive90,
        'privacy': 'aggregate metrics only; no dates, coordinates, road names, raw unit ids or hotspot rankings exported',
        'guardrails': [
            'All 89 field-evidence nights advance the PDCA clock.',
            'Strict road-time scoring occurs only when GPS+timestamp captures and sufficient prior verified captures exist.',
            'Missing historical GPX is never fabricated.',
            'All five models share the same final candidate-zone size for fair candidate-count comparison.',
            'The conformal-style empirical 90% rank is diagnostic, not a guaranteed future 90% coverage bound.',
        ],
    }
    path = r / 'five_models_89night_pdca_summary.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
