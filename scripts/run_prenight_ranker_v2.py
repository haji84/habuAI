from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import _as_jst, _estimator, _fit, _spatial_feature_sets
from habuai.hardening.spatial_history import add_historical_spatial_features
from run_advanced_time_models import _fetch_weather
from run_environmental_hazard_v2 import _prepare_weather
from run_joint_point_time_backtest_v2 import hav, curve24

COUNTS = [10, 20, 30, 50, 100, 200, 300, 500]
RADII_M = [50, 100, 250]
WINDOWS_MIN = [10, 20, 30]
SLOT_MIN = 10
NIGHT_START_HOUR = 18
NIGHT_END_HOUR = 7
SEASON_SIGMA_DAYS = 45.0
MEMORY_TIME_BW_MIN = 60.0
MEMORY_RECENCY_TAU_DAYS = 365.0

VARIANTS = [
    {'name': 'baseline', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 0.0, 'd_detection': 0.0, 'e_exposure': 0.0},
    {'name': 'memory', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 1.0, 'd_detection': 0.0, 'e_exposure': 0.0},
    {'name': 'detection', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 0.0, 'd_detection': 0.5, 'e_exposure': 0.0},
    {'name': 'memory_detection', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 1.0, 'd_detection': 0.5, 'e_exposure': 0.0},
    {'name': 'full_balanced', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 1.0, 'd_detection': 0.5, 'e_exposure': 0.25},
    {'name': 'full_time_heavy', 'a_spatial': 1.0, 'b_local_time': 2.0, 'c_memory': 1.0, 'd_detection': 0.5, 'e_exposure': 0.25},
    {'name': 'full_spatial_heavy', 'a_spatial': 2.0, 'b_local_time': 1.0, 'c_memory': 1.0, 'd_detection': 0.5, 'e_exposure': 0.25},
    {'name': 'full_memory_heavy', 'a_spatial': 1.0, 'b_local_time': 1.0, 'c_memory': 2.0, 'd_detection': 0.5, 'e_exposure': 0.25},
]


def _slot_times(night: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(night, tz='Asia/Tokyo')
    start = day + pd.Timedelta(hours=NIGHT_START_HOUR)
    end = day + pd.Timedelta(days=1, hours=NIGHT_END_HOUR)
    return pd.date_range(start, end, freq=f'{SLOT_MIN}min', inclusive='left')


def _circular_min(a: pd.Timestamp, b: pd.Timestamp) -> float:
    d = abs((a - b).total_seconds()) / 60.0
    return float(min(d, 1440.0 - (d % 1440.0)))


def _circular_minute_values(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 1440.0 - d)


def _circular_doy(a: int, b: np.ndarray) -> np.ndarray:
    d = np.abs(b.astype(float) - float(a))
    return np.minimum(d, 365.25 - d)


def _event_hit(a, cand: pd.DataFrame, rad: float, mins: int) -> int:
    if cand.empty:
        return 0
    dist = hav(a.lat, a.lon, cand.lat.to_numpy(float), cand.lon.to_numpy(float))
    dt = np.array([_circular_min(a.timestamp, t) for t in cand.slot_time])
    return int(((dist <= rad) & (dt <= mins)).any())


def _greedy_diverse(x: pd.DataFrame, n: int) -> pd.DataFrame:
    kept = []
    for row in x.itertuples(index=False):
        if len(kept) >= n:
            break
        accept = True
        for k in kept:
            dist = hav(row.lat, row.lon, np.array([k.lat]), np.array([k.lon]))[0]
            dt = _circular_min(row.slot_time, k.slot_time)
            if dist <= 40.0 and dt <= 20.0:
                accept = False
                break
        if accept:
            kept.append(row)
    if not kept:
        return x.head(0).copy()
    return pd.DataFrame([r._asdict() for r in kept])


def _weather_contexts(weather: pd.DataFrame, nights: list[str]) -> pd.DataFrame:
    w = _prepare_weather(weather).copy()
    w['time'] = pd.to_datetime(w['time'])
    if getattr(w['time'].dt, 'tz', None) is not None:
        w['time'] = w['time'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    cols = ['temperature_2m', 'relative_humidity_2m', 'rain_24h_mm', 'rain_48h_mm', 'hours_since_rain', 'fog_any_flag']
    out = []
    for n in nights:
        t = pd.Timestamp(n) + pd.Timedelta(hours=18)
        idx = (w['time'] - t).abs().idxmin()
        row = w.loc[idx]
        d = {'night': n}
        for c in cols:
            d[c] = float(row[c]) if pd.notna(row[c]) else np.nan
        out.append(d)
    return pd.DataFrame(out).set_index('night')


def _robust_scale(v: pd.Series, default: float) -> float:
    x = pd.to_numeric(v, errors='coerce').dropna()
    if len(x) < 3:
        return default
    q = float(x.quantile(.75) - x.quantile(.25))
    return q if q > 1e-6 else default


def _memory_curve(past_cap: pd.DataFrame, target_night: str, contexts: pd.DataFrame, slots: pd.DatetimeIndex) -> np.ndarray:
    if past_cap.empty:
        return np.ones(len(slots), dtype=float)
    target_ctx = contexts.loc[target_night] if target_night in contexts.index else pd.Series(dtype=float)
    pc = past_cap.copy()
    pc['cap_night'] = (pc.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)
    pc = pc[pc.cap_night.isin(contexts.index)].copy()
    if pc.empty:
        return np.ones(len(slots), dtype=float)
    ctx = contexts.reindex(pc.cap_night).reset_index(drop=True)
    target_day = pd.Timestamp(target_night)
    target_doy = int(target_day.dayofyear)
    cap_doy = pd.to_datetime(pc.cap_night).dt.dayofyear.to_numpy(int)
    season_w = np.exp(-0.5 * (_circular_doy(target_doy, cap_doy) / SEASON_SIGMA_DAYS) ** 2)
    age_days = np.maximum(0.0, np.array([(target_day - pd.Timestamp(n)).days for n in pc.cap_night], dtype=float))
    recency_w = np.exp(-age_days / MEMORY_RECENCY_TAU_DAYS)

    env_logw = np.zeros(len(pc), dtype=float)
    specs = [
        ('temperature_2m', 3.0),
        ('relative_humidity_2m', 10.0),
        ('rain_24h_mm', 5.0),
        ('rain_48h_mm', 10.0),
        ('hours_since_rain', 12.0),
    ]
    for c, fallback in specs:
        tv = float(target_ctx.get(c, np.nan)) if len(target_ctx) else np.nan
        vals = pd.to_numeric(ctx[c], errors='coerce') if c in ctx else pd.Series(np.nan, index=range(len(pc)))
        scale = _robust_scale(contexts[c] if c in contexts else vals, fallback)
        if np.isfinite(tv):
            diff = (vals.to_numpy(float) - tv) / max(scale, 1e-6)
            diff = np.where(np.isfinite(diff), diff, 0.0)
            env_logw += -0.5 * np.clip(diff, -4, 4) ** 2
    if 'fog_any_flag' in contexts and len(target_ctx):
        tf = float(target_ctx.get('fog_any_flag', 0.0))
        fv = pd.to_numeric(ctx['fog_any_flag'], errors='coerce').fillna(0).to_numpy(float)
        env_logw += np.where(fv == tf, 0.0, -0.7)
    env_w = np.exp(np.clip(env_logw, -12, 0))
    base_w = np.maximum(season_w * recency_w * env_w, 1e-12)

    cap_minutes = (pc.timestamp.dt.hour * 60 + pc.timestamp.dt.minute).to_numpy(float)
    result = []
    for t in slots:
        sm = float(t.hour * 60 + t.minute)
        dm = _circular_minute_values(cap_minutes, sm)
        tw = np.exp(-0.5 * (dm / MEMORY_TIME_BW_MIN) ** 2)
        result.append(float(np.sum(base_w * tw)))
    y = np.asarray(result, dtype=float)
    if not np.isfinite(y).any() or float(np.nanmax(y)) <= 0:
        return np.ones(len(slots), dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    y = (y + 1e-9) / (float(y.max()) + 1e-9)
    return np.clip(y, 1e-6, 1.0)


def _summary_rows(d: pd.DataFrame, phase: str, variant: str) -> list[dict]:
    q = d[(d.phase == phase) & (d.variant == variant)]
    out = []
    for n in COUNTS:
        z = q[q.n_predictions == n]
        item = {'phase': phase, 'variant': variant, 'n_predictions': n, 'nights_scored': int(z.night.nunique()) if not z.empty else 0, 'eligible_captures': int(len(z)), 'coverage': {}}
        for rad in RADII_M:
            for mins in WINDOWS_MIN:
                k = f'hit_{rad}m_{mins}min'
                item['coverage'][f'{rad}m_{mins}min'] = float(z[k].mean()) if not z.empty else None
        out.append(item)
    return out


def _selection_score(items: list[dict]) -> float:
    byn = {x['n_predictions']: x for x in items}
    def g(n: int, k: str) -> float:
        x = byn.get(n, {}).get('coverage', {}).get(k)
        return 0.0 if x is None else float(x)
    # Reward useful strict coverage at smaller lists, while still crediting broader 250m/30m recovery.
    return (
        4.0 * g(30, '100m_10min') +
        4.0 * g(50, '100m_10min') +
        3.0 * g(50, '100m_20min') +
        3.0 * g(100, '100m_10min') +
        2.0 * g(100, '250m_30min') +
        1.0 * g(200, '250m_30min')
    )


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = pipeline.load_config(root)
    p = root / 'data' / 'processed'
    r = root / 'reports'
    r.mkdir(exist_ok=True)

    tour = json.loads((r / 'model_tournament_summary.json').read_text(encoding='utf-8'))
    o40 = json.loads((r / 'original40_full_history_summary.json').read_text(encoding='utf-8'))
    sw = tour['spatial']['best']
    tw = o40['best_selected_on_early_period']

    recon = pd.read_csv(p / 'reconstructed_spatial_learning_2026-05_07.csv', low_memory=False)
    md = pd.read_csv(p / 'learning_10m_road.csv', low_memory=False)
    ev = pd.read_csv(p / 'events_matched.csv', low_memory=False)
    segs = gpd.read_file(p / 'road_segments_10m.geojson').to_crs('EPSG:6669')

    md['entered_at'] = _as_jst(md.entered_at)
    ev['timestamp'] = _as_jst(ev.timestamp)
    ev['lat'] = pd.to_numeric(ev.lat, errors='coerce')
    ev['lon'] = pd.to_numeric(ev.lon, errors='coerce')
    md['night'] = (md.entered_at - pd.Timedelta(hours=7)).dt.date.astype(str)
    gpx = md[md.learning_row_source == 'gpx_visit'].copy()
    cap = ev[(ev.species == 'ハブ') & (ev.event_type == '捕獲') & ev.timestamp.notna() & ev.lat.notna() & ev.lon.notna()].copy()

    static_cols = [c for c in [
        'length_m','bearing_deg','curvature_deg','road_class_code','junction_distance_m',
        'stream_distance_m','coast_distance_m','forest_distance_m','farmland_distance_m','residential_distance_m'
    ] if c in md]
    agg = {c: (c, 'first') for c in static_cols}
    agg['spatial_label'] = ('habu_capture', 'max')
    aug = md.groupby(['night', 'segment_id'], as_index=False).agg(**agg)
    aug['sample_weight'] = 1.0
    aug['reconstruction_confidence'] = 'actual_august'
    aug = aug.drop(columns=[c for c in aug if c.startswith('hist_') or c.startswith('days_since_capture_') or c in {'segment_x_m','segment_y_m'}], errors='ignore')
    aug, _ = add_historical_spatial_features(aug, ev, segs, cfg, root=None)
    combined = pd.concat([recon, aug], ignore_index=True, sort=False)
    feats = _spatial_feature_sets(combined)[str(sw['feature_set'])]

    coords = segs[['segment_id','geometry']].copy().to_crs('EPSG:4326')
    coords['lat'] = coords.geometry.apply(lambda z: z.centroid.y)
    coords['lon'] = coords.geometry.apply(lambda z: z.centroid.x)
    coords = coords.drop_duplicates('segment_id').set_index('segment_id')[['lat','lon']]

    static = md.sort_values('entered_at').groupby('segment_id', as_index=False).first()[['segment_id'] + static_cols]
    first_seen = gpx.groupby('segment_id', as_index=False).entered_at.min().rename(columns={'entered_at':'first_seen_at'})
    static = static.merge(first_seen, on='segment_id', how='inner').merge(coords, left_on='segment_id', right_index=True, how='left')

    radius = float(tw['radius_m'])
    bw = float(tw['bandwidth_h'])
    tau = tw.get('recency_tau_days')
    tau = None if tau is None else float(tau)

    nights = sorted(gpx.night.unique())
    split = max(4, int(np.floor(len(nights) * 0.60)))
    selection = set(nights[:split])
    confirmation = set(nights[split:])

    weather_start = str((cap.timestamp.min() - pd.Timedelta(days=3)).date())
    weather_end = str(cap.timestamp.max().date())
    weather = _fetch_weather(weather_start, weather_end, p / 'cache' / 'openmeteo_historical_ranker_v2.csv')
    all_context_nights = sorted(set([(t - pd.Timedelta(hours=7)).date().isoformat() for t in cap.timestamp] + nights))
    contexts = _weather_contexts(weather, all_context_nights)

    rows = []
    public_audit = []

    for night in nights:
        start = pd.Timestamp(night, tz='Asia/Tokyo') + pd.Timedelta(hours=7)
        end = start + pd.Timedelta(days=1)
        actual = cap[(cap.timestamp >= start) & (cap.timestamp < end)].copy()
        if actual.empty:
            continue
        train = combined[combined.night.astype(str) < night].copy()
        if train.empty or train.spatial_label.nunique() < 2:
            continue
        cand = static[static.first_seen_at < start].copy()
        if cand.empty:
            continue
        cand['night'] = night
        cand['spatial_label'] = 0
        cand['sample_weight'] = 1.0
        cand['reconstruction_confidence'] = 'prenight_candidate'
        past_ev = ev[ev.timestamp < start].copy()
        cand, _ = add_historical_spatial_features(cand, past_ev, segs, cfg, root=None)

        model = _estimator(str(sw['estimator']))
        _fit(model, train.reindex(columns=feats).replace([np.inf,-np.inf],np.nan), train.spatial_label.astype(int), pd.to_numeric(train.sample_weight, errors='coerce').fillna(1.0).to_numpy())
        cand['spatial_p'] = model.predict_proba(cand.reindex(columns=feats).replace([np.inf,-np.inf],np.nan))[:,1]

        prior_gpx = gpx[gpx.entered_at < start]
        visit_counts = prior_gpx.groupby('segment_id').size()
        positive_counts = prior_gpx.groupby('segment_id').habu_capture.sum() if 'habu_capture' in prior_gpx.columns else pd.Series(dtype=float)
        cand['prior_visits'] = cand.segment_id.map(visit_counts).fillna(0).astype(float)
        cand['prior_positive_visits'] = cand.segment_id.map(positive_counts).fillna(0).astype(float)
        # Smoothed personal detection proxy; conservative prior prevents one lucky visit from dominating.
        cand['detection_p'] = (cand.prior_positive_visits + 1.0) / (cand.prior_visits + 12.0)
        # Penalize pure familiarity so frequently traversed roads do not win merely from exposure.
        cand['exposure_factor'] = 1.0 / np.sqrt(1.0 + cand.prior_visits)

        slots = _slot_times(night)
        past_cap = cap[cap.timestamp < start].copy()
        memory_curve = _memory_curve(past_cap, night, contexts, slots)
        expanded = []
        for z in cand.itertuples(index=False):
            local_curve = curve24(past_ev, start, float(z.lat), float(z.lon), radius, bw, tau)
            for i,t in enumerate(slots):
                expanded.append({
                    'segment_id': z.segment_id,
                    'lat': float(z.lat),
                    'lon': float(z.lon),
                    'slot_time': t,
                    'spatial_p': max(float(z.spatial_p), 1e-9),
                    'local_time_p': max(float(local_curve[int(t.hour)]), 1e-9),
                    'memory_p': max(float(memory_curve[i]), 1e-9),
                    'detection_p': max(float(z.detection_p), 1e-9),
                    'exposure_factor': max(float(z.exposure_factor), 1e-9),
                })
        base = pd.DataFrame(expanded)
        phase = 'selection' if night in selection else 'confirmation'

        for v in VARIANTS:
            ranked = base.copy()
            ranked['score'] = (
                ranked.spatial_p ** v['a_spatial'] *
                ranked.local_time_p ** v['b_local_time'] *
                ranked.memory_p ** v['c_memory'] *
                ranked.detection_p ** v['d_detection'] *
                ranked.exposure_factor ** v['e_exposure']
            )
            ranked = ranked.sort_values('score', ascending=False).reset_index(drop=True)
            chosen = _greedy_diverse(ranked, max(COUNTS)).sort_values('score', ascending=False).reset_index(drop=True)
            chosen['rank'] = np.arange(1, len(chosen)+1)
            for n in COUNTS:
                zc = chosen.head(n)
                for a in actual.itertuples():
                    rr = {'night': night, 'phase': phase, 'variant': v['name'], 'n_predictions': n}
                    for rad in RADII_M:
                        for mins in WINDOWS_MIN:
                            rr[f'hit_{rad}m_{mins}min'] = _event_hit(a, zc, rad, mins)
                    rows.append(rr)
            # Public audit contains only rank/slot/score, never road IDs or coordinates.
            if v['name'] == 'baseline':
                for z in chosen.head(max(COUNTS)).itertuples(index=False):
                    public_audit.append({'night': night, 'phase': phase, 'variant': v['name'], 'rank': int(z.rank), 'slot_time': z.slot_time.isoformat(), 'score': float(z.score)})

    d = pd.DataFrame(rows)
    selection_summaries = {}
    for v in VARIANTS:
        items = _summary_rows(d, 'selection', v['name'])
        selection_summaries[v['name']] = {'score': _selection_score(items), 'items': items}
    best_name = sorted(selection_summaries, key=lambda n: (-selection_summaries[n]['score'], n))[0]

    best_selection = _summary_rows(d, 'selection', best_name)
    best_confirmation = _summary_rows(d, 'confirmation', best_name)
    baseline_confirmation = _summary_rows(d, 'confirmation', 'baseline')
    all_best = _summary_rows(d.assign(phase='all'), 'all', best_name)

    target90 = {}
    for rad in RADII_M:
        for mins in WINDOWS_MIN:
            k = f'{rad}m_{mins}min'
            reached = [x for x in best_confirmation if x['coverage'].get(k) is not None and x['coverage'][k] >= .90]
            target90[k] = min((x['n_predictions'] for x in reached), default=None)

    summary = {
        'status': 'ok',
        'protocol': {
            'mode': 'pre-night fixed 10-minute schedule ranking tournament v2',
            'actual_gpx_nights': len(nights),
            'selection_nights': len(selection),
            'confirmation_nights': len(confirmation),
            'prediction_counts': COUNTS,
            'distance_thresholds_m': RADII_M,
            'time_windows_min': WINDOWS_MIN,
            'candidate_roads': 'owner-traversed road segments known before target night only',
            'weather_leakage_guard': 'only 18:00 target-night context and prior 24/48h rainfall/history are used; future realized target-night weather is not used',
            'outcome_leakage_guard': 'target-night captures and GPX passage times are hidden when schedule is generated',
            'privacy': 'raw road names, coordinates, segment IDs and hotspot rankings are not exported',
        },
        'components': {
            'spatial': 'daily-updated occupancy features',
            'local_time': 'point-conditioned KDE using only prior captures',
            'season_weather_memory': '45-day cyclic season similarity + 18:00 weather/rain/fog context + weak 365-day recency + 60-minute capture-time kernel',
            'personal_detection_proxy': '(prior positive GPX visits + 1)/(prior GPX visits + 12)',
            'exposure_bias_control': '1/sqrt(1 + prior GPX visits)',
        },
        'variants': VARIANTS,
        'selection_scores': {k: v['score'] for k,v in selection_summaries.items()},
        'best_selected_on_early_gpx_nights': best_name,
        'best_selection': best_selection,
        'best_confirmation': best_confirmation,
        'baseline_confirmation': baseline_confirmation,
        'best_all_scored': all_best,
        'target': {
            'coverage_goal': .90,
            'smallest_confirmation_prediction_count_reaching_90pct': target90,
        },
        'guardrail': 'Variant selection uses early GPX nights only. Confirmation results are not used to choose the variant. Candidate count alone must not be treated as model improvement.',
    }
    (r/'prenight_ranker_v2_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(public_audit).to_csv(r/'prenight_ranker_v2_public_audit.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
