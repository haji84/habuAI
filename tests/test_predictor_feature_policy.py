from pathlib import Path

import pandas as pd

from habuai.pipeline import fit_model


def test_fit_model_excludes_post_outcome_speed(tmp_path: Path):
    (tmp_path / "models").mkdir(parents=True)
    cfg = {"baseline_cutoff": "2026-09-01T00:00:00+09:00"}
    data = pd.DataFrame({
        "entered_at": pd.to_datetime([
            "2026-08-01T22:00:00+09:00",
            "2026-08-01T22:10:00+09:00",
            "2026-08-02T22:00:00+09:00",
            "2026-08-02T22:10:00+09:00",
        ]),
        "habu_capture": [0, 1, 0, 1],
        "sin_hour": [0.0, 0.1, 0.0, 0.1],
        "cos_hour": [1.0, 0.9, 1.0, 0.9],
        "mean_speed_mps": [5.0, 0.2, 5.2, 0.3],
    })
    metrics = fit_model(tmp_path, data, cfg)
    assert metrics["status"] == "ok"
    assert "mean_speed_mps" not in metrics["features"]
    assert metrics["feature_policy"] == "predictor-only; post-outcome movement speed excluded"
