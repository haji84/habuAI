from pathlib import Path

from habuai import pipeline
from habuai.hardening.model_tournament import run_model_tournament

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    cfg=pipeline.load_config(root)
    out=run_model_tournament(root,cfg)
    print(out)
