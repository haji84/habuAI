from __future__ import annotations

import numpy as np
import pandas as pd

import run_five_models_89night_pdca as m

_SPATIAL_CACHE: dict[tuple[float, float, str], float] = {}


def hawkes_score_fast(cand: pd.DataFrame, pc: pd.DataFrame) -> np.ndarray:
    if pc.empty or cand.empty:
        return np.zeros(len(cand), dtype=float)

    units = cand.unit_id.drop_duplicates().tolist()
    times = pd.DatetimeIndex(cand.slot_time.drop_duplicates().tolist())
    uindex = {u: i for i, u in enumerate(units)}
    tindex = {pd.Timestamp(t): i for i, t in enumerate(times)}

    ux = np.array([m.unit_center(u)[0] for u in units], dtype=float)
    uy = np.array([m.unit_center(u)[1] for u in units], dtype=float)

    cp = pc.copy()
    cx = cp.unit_id.map(lambda u: m.unit_center(u)[0]).to_numpy(float)
    cy = cp.unit_id.map(lambda u: m.unit_center(u)[1]).to_numpy(float)
    cm = (cp.timestamp.dt.hour * 60 + cp.timestamp.dt.minute).to_numpy(float)

    dx = ux[:, None] - cx[None, :]
    dy = uy[:, None] - cy[None, :]
    spatial = np.exp(-np.sqrt(dx * dx + dy * dy) / 750.0)

    tmins = (times.hour * 60 + times.minute).to_numpy(float)
    td = np.abs(tmins[:, None] - cm[None, :])
    td = np.minimum(td, 1440.0 - td)
    tod = np.exp(-0.5 * (td / 75.0) ** 2)

    ts_ns = times.asi8.astype(np.float64)
    cp_ns = cp.timestamp.astype('int64').to_numpy(dtype=np.float64)
    age_days = np.maximum(0.0, (ts_ns[:, None] - cp_ns[None, :]) / 1e9 / 86400.0)
    temporal = np.exp(-age_days / 60.0) * tod

    mat = spatial @ temporal.T
    out = np.empty(len(cand), dtype=float)
    for i, z in enumerate(cand.itertuples()):
        out[i] = mat[uindex[z.unit_id], tindex[pd.Timestamp(z.slot_time)]]
    return out


def _min_distance(a, uid: str, members, segll) -> float:
    key = (round(float(a.lat), 7), round(float(a.lon), 7), str(uid))
    if key in _SPATIAL_CACHE:
        return _SPATIAL_CACHE[key]
    ids = members.get(uid)
    if ids is None or len(ids) == 0:
        d = float('inf')
    else:
        pts = segll.reindex(ids).dropna()
        if pts.empty:
            d = float('inf')
        else:
            d = float(np.min(m.b.hav(float(a.lat), float(a.lon), pts.lat.to_numpy(float), pts.lon.to_numpy(float))))
    _SPATIAL_CACHE[key] = d
    return d


def hit_fast(a, top: pd.DataFrame, members, segll, rad: int, mins: int) -> int:
    for c in top.itertuples():
        if m.b.circ_minutes(a.timestamp, c.slot_time) > mins:
            continue
        if _min_distance(a, c.unit_id, members, segll) <= rad:
            return 1
    return 0


def first_rank_hit_fast(a, ranked: pd.DataFrame, members, segll, rad=100, mins=30, max_rank=500):
    for k, c in enumerate(ranked.head(max_rank).itertuples(), 1):
        if m.b.circ_minutes(a.timestamp, c.slot_time) > mins:
            continue
        if _min_distance(a, c.unit_id, members, segll) <= rad:
            return k
    return None


m.hawkes_score = hawkes_score_fast
m.b.hit = hit_fast
m.first_rank_hit = first_rank_hit_fast

if __name__ == '__main__':
    m.main()
