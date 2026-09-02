from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import re

import pandas as pd

from habuai.audit_v2 import operational_date_0700

DATE_TIME_RE = re.compile(r"^(?P<date>20\d{2}/\d{1,2}/\d{1,2})\s+(?P<time>\d{1,2}:\d{2})\s*$")
START_MARKERS = ("エリア探索開始", "探索開始")
END_MARKERS = ("エリア探索終了", "探索終了")


@dataclass(frozen=True)
class ExplorationEvidence:
    operational_date_0700: str
    evidence_type: str
    source_file: str
    source_timestamp: str
    source_hash: str
    strength: str
    notes: str = ""


def _marker_type(line: str) -> str | None:
    if any(marker in line for marker in START_MARKERS):
        return "SEARCH_START"
    if any(marker in line for marker in END_MARKERS):
        return "SEARCH_END"
    return None


def parse_text_exploration_markers(path: Path) -> pd.DataFrame:
    """Extract explicit exploration start/end markers from a raw text log.

    The timestamp immediately preceding a marker block is used. Multiple area
    transitions in one operational night remain separate evidence rows but collapse
    to one night when the canonical population is constructed.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    digest = sha256(raw).hexdigest()
    lines = text.splitlines()

    current_ts: pd.Timestamp | None = None
    rows: list[dict] = []
    for line_no, line in enumerate(lines, start=1):
        m = DATE_TIME_RE.match(line.strip())
        if m:
            current_ts = pd.Timestamp(f"{m.group('date')} {m.group('time')}", tz="Asia/Tokyo")
            continue

        marker_type = _marker_type(line.strip())
        if marker_type is None or current_ts is None:
            continue

        rows.append(
            asdict(
                ExplorationEvidence(
                    operational_date_0700=operational_date_0700(current_ts).isoformat(),
                    evidence_type=marker_type,
                    source_file=path.name,
                    source_timestamp=current_ts.isoformat(),
                    source_hash=digest,
                    strength="STRONG_SESSION_MARKER",
                    notes=f"line={line_no}; marker={line.strip()}",
                )
            )
        )

    return pd.DataFrame(rows)


def canonical_population_from_manifest(manifest: pd.DataFrame) -> list[str]:
    if manifest.empty:
        return []
    return sorted({str(v) for v in manifest["operational_date_0700"].dropna()})


def combine_manifests(*manifests: pd.DataFrame) -> pd.DataFrame:
    frames = [m for m in manifests if m is not None and not m.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "operational_date_0700",
                "evidence_type",
                "source_file",
                "source_timestamp",
                "source_hash",
                "strength",
                "notes",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates().sort_values(
        ["operational_date_0700", "source_timestamp", "source_file"]
    ).reset_index(drop=True)
