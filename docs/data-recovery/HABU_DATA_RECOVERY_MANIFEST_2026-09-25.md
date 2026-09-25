# Habu AI data recovery manifest — 2026-09-25

Purpose: track artifacts recovered from Chat / Work / Codex / Library against the GitHub canonical repository before promotion.

## Classification rules
- Canonical: newest verified artifact that supersedes older versions and passes the promotion gate.
- Archive: older or superseded artifact retained for provenance.
- Raw: original observation/input that must not be regenerated or silently modified.
- Reproducible: can be regenerated from preserved Raw + code + frozen configuration.
- Non-reproducible: source evidence or historical output cannot be reconstructed exactly from currently preserved inputs/code.

## Byte-verified recovery identities

| Artifact | Source | SHA-256 | Classification | Status |
|---|---|---|---|---|
| Habu_AI_Audit_Walkforward_2026-09-24.xlsx | Library v4 | d8c4bcd6f75aaefad941fb89af6690f2d3c904d2885c8abcba4f1cf5fc1c7c17 | Canonical audit | promoted in recovery inventory |
| Habu_AI_Audit_Model_Package_2026-09-24.zip | Library v4 | bd03b62e7dfb4885ef11d2e8cd83b9b015079d455408d049e7ab13710668169a | Canonical package | promoted in recovery inventory; keep outside ordinary Git blob |
| final_report.md | Library v2 | 70441a00e144b5d6b62727134f2fdc824f5a313d955c052e7b5450778b3443a7 | Canonical report | byte-verified; copy text to repo only with provenance |
| Habu_AI_v2_89夜_GPS_GPX復元監査.xlsx | Library v1 | 07388b47fd65bea047b004ecc411ce2b7bb906f283434d243a06fbba6580f949 | Archive | historical audit only |

## Canonical 2026-09-24 audit state
- Raw input audit: 36 files with SHA-256 recorded; 3 rechecks were identical SHA.
- GPX audit: 25 files = 24 eligible search GPX + 1 non-search context GPX.
- Audited GPX points: 435,513 total; 2 anomaly points excluded; 25 segment breaks.
- Capture matching: 68 captures matched at 100 m ±10 min.
- Exposure: 18,531 rows = 68 Positive + 18,463 Negative. Unsearched space is not materialized as Negative.
- Terrain source counts: DEM5B 203, DEM5C 194, DEM10B 286.
- Road GIS: 2,177 tiles / 62,247 features.
- Official weather: 7,896 observations.
- Tide: JMA Amami O9, 365 days.
- Movement Front eligible/observed: 1,268 / 1,268.
- Walk-forward: 23 evaluation nights in the rebuilt series.
- Current rebuilt-series best: M1RS, mean ROC-AUC 0.6426903, capture percentile 0.6423709.
- Formal KPI remains unmet. Maximum mean Top10 Recall 0.033; maximum Top5% Recall 0.116.
- Historical formal M0-M5 fixed specifications remain unrecovered. M0R-M5R / RW / RF series must not be relabeled as those historical models.

## Prospective freeze
- M6P_BASE spec SHA-256: 6c5285ff24b026a72eb797d203625820606e1e3997ee94c3049b880847970973
- M6P_ELAPSED spec SHA-256: dce7913c19a4b9e3e8e7958927f7dd4c5406dca6d974ca3b189bea7b1dc67297
- M6P_SURFACE_ELAPSED spec SHA-256: 69dda6b8482736a99c5360c091ead3181dd0713d322a62d04c26a04d93a22b30
- Adoption gate remains M6P_BASE until 10 future positive nights and 30 captures, with the frozen paired-CI decision rule.

## GPX deduplication result
The historical 89-night audit records filename-copy groups by SHA-256. Examples:
- 2026-08-12: four filename copies collapse to one content identity, SHA-256 1b390060aa8f553e55cbd9f6d2e8f45dbb9859a5fe6f22edc9e8daf0aa00cded.
- 2026-08-13: three filename copies collapse to one content identity, SHA-256 966adf290ef0390961b37b84d564b3e3328cfea85ee0376acb91c8dcf93d1645.
- 2026-08-19: two filename copies collapse to one content identity, SHA-256 1af6d99292a12b483153af4b8f1550a582a783e3f830096e3420f1fee187689c.
The 2026-09-24 canonical package uses one audited GPX per night and lists all 25 canonical filenames in 05_night_registry/gpx_file_audit.csv. Raw copies are not separate observations.

## Capture DB resolution
Do not promote an older Library workbook merely because its filename says "latest". The 2026-09-24 audit package contains the newer audited analytical capture table:
- 05_night_registry/capture_db_audited.csv: 232 audited rows.
- audit_summary.json: raw_log_captures=186, excel_capture_rows=231, excel_capture_individuals=233.
- Control mismatch: excel_capture_individuals_through_2026_08=209 versus user-stated control=190, difference=19.
Therefore capture_db_audited.csv is the current canonical analytical capture table for the recovered audit lineage. The pre-audit Excel capture DB lineage remains source/archive evidence until the 19-individual discrepancy is resolved. The Chat-referenced 09-12 workbook has not been recovered as an exact byte object and must not be fabricated or promoted.

## Archive lineage
- Habu_AI_v2_89夜_GPS_GPX復元監査.xlsx.
- ハブAI_100m×10分_Exposure_2026-09-23.csv and its copy candidate.
- complete_walkforward_results.csv.
- ハブAI_再現試験_PDCA_2026-09-23.xlsx and copy candidate.
- 2026-08-31 model comparison/validation reports.
- Older GitHub canonical-audit lineage, including the earlier 88-night state.
- 2026-09-04 gpx10m/retrain artifacts if exact bytes are recovered.

## Reproducibility
- Raw GPX and original field/log observations: non-reproducible. Preserve exact bytes and hashes.
- Exposure, environment features, model predictions, walk-forward metrics and ablations: derived and reproducible only when the exact Raw set, source code, frozen specs and preprocessing rules are preserved.
- The 2026-09-24 model package is the recovery capsule for those derived artifacts.
- Historical outputs whose exact source set/specification is missing remain non-reproducible historical evidence, even if a similar pipeline can be rerun.

## Remaining gaps
- Exact bytes for the Chat-referenced ハブDB_2025-2026_09-12更新版 workbook are not recovered.
- Historical formal M0-M5 definitions remain unrecovered.
- Some Work/Codex-only outputs may never have been exported and cannot be assumed to exist.
- The 2026-09-24 raw GPX is indexed in Library and audited inside the package, but its separate raw-byte materialization path is unavailable from this Project context; preserve its Library identity and package audit record rather than inventing a hash.
- The package records that authenticated GSI DEM ZIP/XML originals were not obtained.

This file is the recovery control document. It distinguishes verified Canonical, Archive, Raw, reproducible derivatives, and unresolved evidence; it does not claim missing bytes exist.
