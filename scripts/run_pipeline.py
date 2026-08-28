from pathlib import Path
import json

from habuai.pipeline import run

if __name__ == "__main__":
    result = run(Path(__file__).resolve().parents[1])
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
