from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CSV = ROOT / "reports" / "v2_actual_gpx_13night_qc.csv"
GPX_DIR = ROOT / "data" / "raw" / "gpx"
OUT_JSON = ROOT / "reports" / "v2_actual_gpx_pipeline_inventory_status.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    expected = pd.read_csv(EXPECTED_CSV)
    expected_hashes = {
        str(row.raw_sha256): str(row.operational_night)
        for row in expected.itertuples(index=False)
    }

    actual_files = sorted(GPX_DIR.glob("*.gpx")) if GPX_DIR.exists() else []
    actual_by_hash = {sha256(path): path.name for path in actual_files if path.stat().st_size > 0}

    found = []
    missing = []
    for raw_hash, night in expected_hashes.items():
        if raw_hash in actual_by_hash:
            found.append(
                {
                    "operational_night": night,
                    "sha256": raw_hash,
                    "repo_runtime_filename": actual_by_hash[raw_hash],
                }
            )
        else:
            missing.append({"operational_night": night, "sha256": raw_hash})

    complete = len(missing) == 0
    status = {
        "expected_actual_gpx_nights": len(expected_hashes),
        "found_expected_actual_gpx_nights": len(found),
        "missing_expected_actual_gpx_nights": len(missing),
        "canonical_inventory_complete": complete,
        "canonical_model_publish_allowed": complete,
        "found": found,
        "missing": missing,
        "rule": (
            "Raw GPX identity is matched by SHA-256, not filename. Pipeline outputs may run "
            "with an incomplete inventory, but they must not be published as canonical latest "
            "until every expected ACTUAL_GPX original is present."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"CANONICAL_GPX_COMPLETE={'true' if complete else 'false'}")


if __name__ == "__main__":
    main()
