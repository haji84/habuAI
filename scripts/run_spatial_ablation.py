from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from habuai.hardening.spatial_ablation import run_spatial_ablation_suite


root=Path(__file__).resolve().parents[1]
data_path=root/"data"/"processed"/"reconstructed_spatial_learning_2026-05_07.csv"
cfg=json.loads((root/"config"/"pipeline.json").read_text(encoding="utf-8"))
if not data_path.exists():
    raise SystemExit(f"missing reconstructed spatial learning data: {data_path}")
data=pd.read_csv(data_path)
result=run_spatial_ablation_suite(root,data,cfg)
print(json.dumps(result,ensure_ascii=False,indent=2))
