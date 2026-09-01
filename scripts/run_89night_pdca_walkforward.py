from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_full_capture_time_backtest import _as_jst

FEATURES = [
    'sin_month','cos_month','sin_doy','cos_doy',
    'prior5_mean','prior10_mean','prior30_mean',
    'prior5_positive_rate','prior10_positive_rate','prior30_positive_rate',
    'days_since_last_capture','prior_last_count'
]
MIN_HISTORY_NIGHTS = 12
TIME_MIN_PRIOR = 10
TIME_BWS_H = [0.75, 1.0, 1.5, 2.0]
TIME_TAUS_D = [None, 120.0]
TIME_SLOT_MIN = 10


def _clf(tr: pd.DataFrame, target: str):
    if tr.empty or tr[target].nunique() < 2:
        return None
    m = make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1500, class_weight='balanced', C=.2),
    )
    m.fit(tr[FEATURES], tr[target].astype(int))
    return m


def _prob(m, x: pd.DataFrame, fallback: float) -> float:
    if m is None:
        return float(fallback)
    return float(m.predict_proba(x[FEATURES])[0, 1])


def _count_feature_table(nights: list[str], counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for i, n in enumerate(nights):
        ts = pd.Timestamp(n)
        hist = np.array([counts.get(q, 0) for q in nights[:i]], dtype=float)
        lastpos = np.where(hist > 0)[0]
        def tail(k: int):
            return hist[-k:] if len(hist) else hist
        row = {
            'night': n,
            'capture_count': int(counts.get(n, 0)),
            'positive': int(counts.get(n, 0) > 0),
            'multi2': int(counts.get(n, 0) >= 2),
            'multi3': int(counts.get(n, 0) >= 3),
            'multi5': int(counts.get(n, 0) >= 5),
            'sin_month': math.sin(2 * math.pi * ts.month / 12),
            'cos_month': math.cos(2 * math.pi * ts.month / 12),
            'sin_doy': math.sin(2 * math.pi * ts.dayofyear / 365.25),
            'cos_doy': math.cos(2 * math.pi * ts.dayofyear / 365.25),
        }
        for k in (5, 10, 30):
            h = tail(k)
            row[f'prior{k}_mean'] = float(h.mean()) if len(h) else 0.0
            row[f'prior{k}_positive_rate'] = float((h > 0).mean()) if len(h) else 0.0
        row['prior_last_count'] = float(hist[-1]) if len(hist) else 0.0
        row['days_since_last_capture'] = (
            float((ts - pd.Timestamp(nights[int(lastpos[-1])])).days) if len(lastpos) else 999.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _count_candidates(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, float]:
    actual_hist = tr.capture_count.to_numpy(float)
    prior5 = float(actual_hist[-5:].mean()) if len(actual_hist) else 0.0
    prior10 = float(actual_hist[-10:].mean()) if len(actual_hist) else 0.0
    prior30 = float(actual_hist[-30:].mean()) if len(actual_hist) else 0.0

    pos = _clf(tr, 'positive')
    ptr = tr[tr.positive == 1].copy()
    ppos = _prob(pos, te, tr.positive.mean())
    lam = float(ptr.capture_count.mean()) if len(ptr) else max(prior10, 1.0)
    if len(ptr) >= 8:
        imp = SimpleImputer(strategy='median')
        X = imp.fit_transform(ptr[FEATURES])
        pm = PoissonRegressor(alpha=.5, max_iter=500).fit(X, ptr.capture_count)
        lam = float(pm.predict(imp.transform(te[FEATURES]))[0])
    hurdle = max(0.0, ppos * lam)

    return {
        'prior5': max(0.0, prior5),
        'prior10': max(0.0, prior10),
        'prior30': max(0.0, prior30),
        'hurdle_ml': hurdle,
        'blend_hurdle_prior10': max(0.0, 0.65 * hurdle + 0.35 * prior10),
    }


def _choose_by_past_mae(history: list[dict], candidate_names: list[str]) -> str:
    if not history:
        return 'blend_hurdle_prior10' if 'blend_hurdle_prior10' in candidate_names else candidate_names[0]
    recent = history[-20:]
    scores = {}
    for name in candidate_names:
        errs = [abs(float(r['actual']) - float(r['candidates'][name])) for r in recent if name in r['candidates']]
        scores[name] = float(np.mean(errs)) if errs else float('inf')
    return min(scores, key=scores.get)


def _circular_minute_delta(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 1440.0 - d)


def _time_curve(past_cap: pd.DataFrame, target_start: pd.Timestamp, bw_h: float, tau_days: float | None) -> tuple[np.ndarray, np.ndarray]:
    slots = np.arange(18 * 60, 24 * 60 + 7 * 60, TIME_SLOT_MIN, dtype=float) % 1440.0
    if past_cap.empty:
        return slots, np.ones(len(slots))
    mins = (past_cap.timestamp.dt.hour * 60 + past_cap.timestamp.dt.minute).to_numpy(float)
    ages = np.maximum(0.0, (target_start - past_cap.timestamp).dt.total_seconds().to_numpy(float) / 86400.0)
    age_w = np.ones(len(past_cap), dtype=float) if tau_days is None else np.exp(-ages / tau_days)
    vals = []
    bw_min = bw_h * 60.0
    for s in slots:
        dm = _circular_minute_delta(mins, s)
        vals.append(float(np.sum(age_w * np.exp(-0.5 * (dm / bw_min) ** 2))))
    return slots, np.asarray(vals, dtype=float)


def _time_prediction(past_cap: pd.DataFrame, target_start: pd.Timestamp, bw_h: float, tau_days: float | None) -> int:
    slots, score = _time_curve(past_cap, target_start, bw_h, tau_days)
    return int(slots[int(np.argmax(score))])


def _circ_err_min(actual_min: int, pred_min: int) -> float:
    d = abs(float(actual_min) - float(pred_min)) % 1440.0
    return float(min(d, 1440.0 - d))


def _select_time_params(past_eval: list[dict]) -> tuple[float, float | None]:
    if not past_eval:
        return 2.0, 120.0
    recent = past_eval[-50:]
    best = None
    best_key = None
    for bw in TIME_BWS_H:
        for tau in TIME_TAUS_D:
            key = f'bw{bw}_tau{tau}'
            errs = [float(r['errors'][key]) for r in recent if key in r['errors']]
            if not errs:
                continue
            obj = (float(np.mean(np.array(errs) <= 30.0)), -float(np.median(errs)), -float(np.mean(errs)))
            if best is None or obj > best:
                best = obj
                best_key = (bw, tau)
    return best_key if best_key is not None else (2.0, 120.0)


def main():
    root = Path(__file__).resolve().parents[1]
    p = root / 'data' / 'processed'
    r = root / 'reports'
    r.mkdir(exist_ok=True)

    e = pd.read_csv(p / 'events_matched.csv', low_memory=False)
    e['timestamp'] = _as_jst(e.timestamp)
    e = e[e.timestamp.notna()].copy()
    e['night'] = (e.timestamp - pd.Timedelta(hours=7)).dt.date.astype(str)

    cap = e[(e.species == 'ハブ') & (e.event_type == '捕獲')].copy()
    cap['individual_count'] = pd.to_numeric(cap.individual_count, errors='coerce').fillna(1).clip(lower=1)
    counts = cap.groupby('night').individual_count.sum().astype(int).to_dict()
    nights = sorted(e.night.dropna().unique().tolist())
    d = _count_feature_table(nights, counts)

    gps_cap = cap[cap.timestamp.notna() & pd.to_numeric(cap.lat, errors='coerce').notna() & pd.to_numeric(cap.lon, errors='coerce').notna()].copy()

    count_history: list[dict] = []
    time_history: list[dict] = []
    nightly_internal = []

    for i, n in enumerate(nights):
        te = d[d.night == n]
        tr = d[d.night < n]
        actual_count = int(te.capture_count.iloc[0])
        target_start = pd.Timestamp(n, tz='Asia/Tokyo') + pd.Timedelta(hours=7)

        rec = {
            'night_index': i + 1,
            'night': n,
            'history_nights_before_prediction': i,
            'actual_count': actual_count,
            'count_prediction_available': False,
            'time_prediction_available': False,
        }

        if len(tr) >= MIN_HISTORY_NIGHTS:
            candidates = _count_candidates(tr, te)
            selected_name = _choose_by_past_mae(count_history, list(candidates))
            pred = float(candidates[selected_name])
            rec.update({
                'count_prediction_available': True,
                'count_selected_model': selected_name,
                'count_expected': pred,
                'count_abs_error': abs(actual_count - pred),
            })

        past_gps = gps_cap[gps_cap.timestamp < target_start].copy()
        actual_gps = gps_cap[gps_cap.night == n].copy()
        if len(past_gps) >= TIME_MIN_PRIOR and not actual_gps.empty:
            bw, tau = _select_time_params(time_history)
            pred_min = _time_prediction(past_gps, target_start, bw, tau)
            event_errors = []
            all_variant_errors = []
            for a in actual_gps.itertuples():
                amin = int(a.timestamp.hour) * 60 + int(a.timestamp.minute)
                event_errors.append(_circ_err_min(amin, pred_min))
                errs = {}
                for vbw in TIME_BWS_H:
                    for vtau in TIME_TAUS_D:
                        vp = _time_prediction(past_gps, target_start, vbw, vtau)
                        errs[f'bw{vbw}_tau{vtau}'] = _circ_err_min(amin, vp)
                all_variant_errors.append({'errors': errs})
            rec.update({
                'time_prediction_available': True,
                'time_selected_bw_h': bw,
                'time_selected_tau_days': tau,
                'time_predicted_minute_of_day': pred_min,
                'time_events_scored': len(event_errors),
                'time_median_error_min': float(np.median(event_errors)),
                'time_within_10m': int(np.sum(np.array(event_errors) <= 10)),
                'time_within_20m': int(np.sum(np.array(event_errors) <= 20)),
                'time_within_30m': int(np.sum(np.array(event_errors) <= 30)),
            })

        if rec['count_prediction_available']:
            count_history.append({'actual': actual_count, 'candidates': candidates, 'selected': rec['count_selected_model']})
        if rec['time_prediction_available']:
            time_history.extend(all_variant_errors)
        nightly_internal.append(rec)

    out = pd.DataFrame(nightly_internal)
    scored = out[out.count_prediction_available == True].copy()
    time_scored = out[out.time_prediction_available == True].copy()

    def block_metrics(x: pd.DataFrame) -> dict:
        if x.empty:
            return {'nights': 0}
        return {
            'nights': int(len(x)),
            'actual_total': int(x.actual_count.sum()),
            'predicted_total': float(x.count_expected.sum()),
            'mae': float(x.count_abs_error.mean()),
            'median_abs_error': float(x.count_abs_error.median()),
            'within_1_capture_rate': float((x.count_abs_error <= 1.0).mean()),
            'within_2_capture_rate': float((x.count_abs_error <= 2.0).mean()),
        }

    base = scored.reset_index(drop=True)
    cut1 = len(base) // 3
    cut2 = (2 * len(base)) // 3
    thirds = [base.iloc[:cut1].copy(), base.iloc[cut1:cut2].copy(), base.iloc[cut2:].copy()] if not base.empty else []
    count_blocks = {f'block_{i+1}': block_metrics(x) for i, x in enumerate(thirds)}

    time_events = int(time_scored.time_events_scored.sum()) if not time_scored.empty else 0
    time_summary = {
        'eligible_nights': int(len(time_scored)),
        'eligible_capture_events': time_events,
        'night_median_error_min_median': float(time_scored.time_median_error_min.median()) if not time_scored.empty else None,
        'within_10m_hits': int(time_scored.time_within_10m.sum()) if not time_scored.empty else 0,
        'within_20m_hits': int(time_scored.time_within_20m.sum()) if not time_scored.empty else 0,
        'within_30m_hits': int(time_scored.time_within_30m.sum()) if not time_scored.empty else 0,
    }
    if time_events:
        time_summary['within_10m_rate'] = time_summary['within_10m_hits'] / time_events
        time_summary['within_20m_rate'] = time_summary['within_20m_hits'] / time_events
        time_summary['within_30m_rate'] = time_summary['within_30m_hits'] / time_events

    selected_counts = scored.count_selected_model.value_counts().to_dict() if not scored.empty else {}
    summary = {
        'status': 'ok',
        'method': 'night-by-night PDCA walk-forward: predict -> reveal result -> score -> add verified night -> adapt model choice -> next-night predict',
        'field_evidence_nights_total': int(len(nights)),
        'capture_positive_nights': int((d.capture_count > 0).sum()),
        'zero_field_evidence_nights': int((d.capture_count == 0).sum()),
        'warmup_nights': MIN_HISTORY_NIGHTS,
        'count_scored_nights': int(len(scored)),
        'count_overall': block_metrics(scored),
        'count_progression_thirds': count_blocks,
        'count_selected_model_frequency': {str(k): int(v) for k, v in selected_counts.items()},
        'time_walkforward': time_summary,
        'adaptation': {
            'count': 'each night chooses the lowest trailing-20-night MAE among prior5/prior10/prior30/hurdle/blend using only already-scored nights',
            'time': 'each eligible night chooses bandwidth/recency from prior scored GPS capture events only, optimizing <=30min hit rate then median/mean error',
        },
        'route_zone_note': 'All 89 field-evidence nights are processed in the outer PDCA loop. Road-zone ranking is not fabricated for nights lacking prior route/GPX support; route-specific validation remains limited to nights with valid route history.',
        'privacy': 'Only aggregate metrics are committed. Per-night dates, route identities, coordinates, and hotspot rankings are intentionally not exported from this public repository.',
        'guardrails': [
            '07:00 Asia/Tokyo operational-day rollover',
            'target-night outcome is not used before prediction',
            'each completed night becomes available only to later nights',
            'field-evidence zero nights are weaker than GPX-confirmed complete zero surveys',
            'time evaluation uses only GPS+timestamp captures and a pre-night global time forecast, not the target capture location',
        ],
    }
    (r / 'pdca_89night_walkforward_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
