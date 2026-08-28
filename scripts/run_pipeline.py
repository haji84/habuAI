from pathlib import Path
import json

import habuai.pipeline as pipeline
from habuai.runtime_fixes import apply_runtime_fixes

if __name__ == "__main__":
    apply_runtime_fixes(pipeline)
    result = pipeline.run(Path(__file__).resolve().parents[1])
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
