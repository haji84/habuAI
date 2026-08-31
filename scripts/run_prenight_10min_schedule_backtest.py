from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from habuai import pipeline
from habuai.hardening.model_tournament import _as_jst, _estimator, _fit, _spatial_feature_sets
from habuai.hardening.spatial_history import add_historical_spatial_features
from run_joint_point_time_backtest_v2 import hav, curve24

COUNTS = [10, 20, 30, 50, 100, 200]
RADII_M = [50, 100, 250]
WINDOWS_MIN = [10, 20, 30]
SLOT_MIN = 10
NIGHT_START_HOUR = 18
NIGHT_END_HOUR = 7


def _slot_times(night: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(night, tz='Asia/Tokyo')
    start = day + pd.Timedelta(hours=NIGHT_START_HOUR)
    end = day + pd.Timedelta(days=1, hours=NIGHT_END_HOUR)
    return pd.date_range(start, end, freq=f'{SLOT_MIN}min', inclusive='left')


def _circular_min(a: pd.Timestamp, b: pd.Timestamp) -> float:
    d = abs((a - b).total_seconds()) / 60.0
    return float(min(d, 1440.0 - (d % 1440.0)))


def _event_hit(a, cand: pd.DataFrame, rad: float, mins: int) -> int:
    if cand.empty:
        return 0
    dist = hav(a.lat, a.lon, cand.lat.to_numpy(float), cand.lon.to_numpy(float))
    dt = np.array([_circular_min(a.timestamp, t) for t in cand.slot_time])
    return int(((dist <= rad) & (dt <= mins)).any())


def _greedy_diverse(x: pd.DataFrame, n: int) -> pd.DataFrame:
    # Avoid filling the list with nearly identical adjacent 10 m segments / adjacent slots.
    # A candidate is suppressed only when it is both spatially very close and within 20 min.
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

    static_cols = [c for c in [
        'length_m','bearing_deg','curvature_deg','road_class_code','junction_distance_m',
        'stream_distance_m','coast_distance_m','forest_distance_m','farmland_distance_m',
        'residential_distance_m'
    ] if c in md]

    # August actual-GPX rows are used only as historical training evidence before each target night.
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

    # Static candidate template. Only segments personally traversed before target night are eligible.
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

    rows = []
    schedule_rows = []

    for night in nights:
        start = pd.Timestamp(night, tz='Asia/Tokyo') + pd.Timedelta(hours=7)
        end = start + pd.Timedelta(days=1)
        actual = ev[(ev.species == 'ハブ') & (ev.event_type == '捕獲') & (ev.timestamp >= start) & (ev.timestamp < end) & ev.lat.notna() & ev.lon.notna()].copy()
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
        cand, _ = add_historical_spatial_features(cand, ev[ev.timestamp < start], segs, cfg, root=None)

        model = _estimator(str(sw['estimator']))
        _fit(
            model,
            train.reindex(columns=feats).replace([np.inf, -np.inf], np.nan),
            train.spatial_label.astype(int),
            pd.to_numeric(train.sample_weight, errors='coerce').fillna(1.0).to_numpy(),
        )
        cand['spatial_p'] = model.predict_proba(cand.reindex(columns=feats).replace([np.inf, -np.inf], np.nan))[:, 1]

        past = ev[ev.timestamp < start]
        slots = _slot_times(night)
        expanded = []
        for z in cand.itertuples(index=False):
            curve = curve24(past, start, float(z.lat), float(z.lon), radius, bw, tau)
            for t in slots:
                tp = float(curve[int(t.hour)])
                expanded.append({
                    'segment_id': z.segment_id,
                    'lat': float(z.lat),
                    'lon': float(z.lon),
                    'slot_time': t,
                    'spatial_p': float(z.spatial_p),
                    'time_p': tp,
                    'score': max(float(z.spatial_p), 1e-12) * max(tp, 1e-12),
                })
        ranked = pd.DataFrame(expanded).sort_values('score', ascending=False).reset_index(drop=True)
        # Build one fixed, diverse pre-night list up to the largest requested count.
        chosen = _greedy_diverse(ranked, max(COUNTS))
        chosen = chosen.sort_values('score', ascending=False).reset_index(drop=True)
        chosen['rank'] = np.arange(1, len(chosen) + 1)

        phase = 'selection' if night in selection else 'confirmation'
        for n in COUNTS:
            z = chosen.head(n)
            for a in actual.itertuples():
                rr = {'night': night, 'phase': phase, 'n_predictions': n}
                for rad in RADII_M:
                    for mins in WINDOWS_MIN:
                        rr[f'hit_{rad}m_{mins}min'] = _event_hit(a, z, rad, mins)
                rows.append(rr)

        # Public-safe audit: no coordinates or raw road IDs are exported.
        for z in chosen.head(max(COUNTS)).itertuples(index=False):
            schedule_rows.append({
                'night': night,
                'phase': phase,
                'rank': int(z.rank),
                'slot_time': z.slot_time.isoformat(),
                'score': float(z.score),
            })

    d = pd.DataFrame(rows)
    variants = []
    for phase in ['selection', 'confirmation', 'all']:
        q = d if phase == 'all' else d[d.phase == phase]
        for n in COUNTS:
            z = q[q.n_predictions == n]
            item = {
                'phase': phase,
                'n_predictions': n,
                'nights_scored': int(z.night.nunique()) if not z.empty else 0,
                'eligible_captures': int(len(z)),
                'coverage': {},
            }
            for rad in RADII_M:
                for mins in WINDOWS_MIN:
                    k = f'hit_{rad}m_{mins}min'
                    item['coverage'][f'{rad}m_{mins}min'] = float(z[k].mean()) if not z.empty else None
            variants.append(item)

    # Determine the smallest candidate count reaching the 90% target, without changing parameters on holdout.
    confirmation_items = [x for x in variants if x['phase'] == 'confirmation']
    target90 = {}
    for rad in RADII_M:
        for mins in WINDOWS_MIN:
            key = f'{rad}m_{mins}min'
            reached = [x for x in confirmation_items if x['coverage'].get(key) is not None and x['coverage'][key] >= 0.90]
            target90[key] = min((x['n_predictions'] for x in reached), default=None)

    summary = {
        'status': 'ok',
        'protocol': {
            'mode': 'pre-night fixed schedule',
            'actual_gpx_nights': len(nights),
            'selection_nights': len(selection),
            'confirmation_nights': len(confirmation),
            'slot_minutes': SLOT_MIN,
            'prediction_counts': COUNTS,
            'distance_thresholds_m': RADII_M,
            'time_windows_min': WINDOWS_MIN,
            'candidate_roads': 'owner-traversed road segments known before target night only',
            'schedule_window_local': '18:00-07:00 JST',
            'leakage_guard': 'target-night GPX passage times and target-night captures are not used to generate the schedule',
            'diversity_guard': 'near-duplicate candidates within 40m and 20min are suppressed',
        },
        'target': {
            'coverage_goal': 0.90,
            'meaning': 'fraction of actual captures covered by the pre-night candidate list; not Top-1 precision',
            'smallest_confirmation_prediction_count_reaching_90pct': target90,
        },
        'results': variants,
        'guardrail': 'Increasing candidate count can raise coverage mechanically. Production selection should minimize candidate count subject to coverage, travel-time, and route-length constraints.',
    }
    (r / 'prenight_10min_schedule_backtest_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(schedule_rows).to_csv(r / 'prenight_10min_schedule_public_audit.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
