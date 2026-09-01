from __future__ import annotations

import numpy as np
import pandas as pd

import run_five_models_89night_pdca as m


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

    # Spatial kernel: U x captures.
    dx = ux[:, None] - cx[None, :]
    dy = uy[:, None] - cy[None, :]
    spatial = np.exp(-np.sqrt(dx * dx + dy * dy) / 750.0)

    # Temporal kernel: T x captures. Same formula as the original candidate loop.
    tmins = (times.hour * 60 + times.minute).to_numpy(float)
    td = np.abs(tmins[:, None] - cm[None, :])
    td = np.minimum(td, 1440.0 - td)
    tod = np.exp(-0.5 * (td / 75.0) ** 2)

    ts_ns = times.asi8.astype(np.float64)
    cp_ns = cp.timestamp.astype('int64').to_numpy(dtype=np.float64)
    age_days = np.maximum(0.0, (ts_ns[:, None] - cp_ns[None, :]) / 1e9 / 86400.0)
    age = np.exp(-age_days / 60.0)
    temporal = age * tod

    # U x T intensity.
    mat = spatial @ temporal.T
    out = np.empty(len(cand), dtype=float)
    for i, z in enumerate(cand.itertuples()):
        out[i] = mat[uindex[z.unit_id], tindex[pd.Timestamp(z.slot_time)]]
    return out


m.hawkes_score = hawkes_score_fast

if __name__ == '__main__':
    m.main()
