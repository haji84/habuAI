# Habu AI data recovery manifest — 2026-09-25

Purpose: track artifacts recovered from Chat / Work / Codex / Library against the GitHub canonical repository before any artifact is promoted.

## Classification rules
- Canonical: newest verified artifact that supersedes older versions.
- Archive: older or superseded artifact retained for provenance.
- Raw: original observation/input that must not be regenerated or silently modified.
- Reproducible: can be regenerated from preserved Raw + code + frozen configuration.
- Non-reproducible: source evidence or historical output cannot be reconstructed exactly from currently preserved inputs/code.

## Verified inventory

| Artifact | Observed source | Date/version | Classification | Reproducibility | GitHub action |
|---|---|---|---|---|---|
| Habu_AI_Audit_Walkforward_2026-09-24.xlsx | Library | 2026-09-24 | Canonical audit candidate | Partly reproducible; contains current audit/evaluation state | Preserve externally; do not overwrite old M0-M5 definitions |
| Habu_AI_Audit_Model_Package_2026-09-24.zip | Library | 2026-09-24 | Canonical package candidate | Package-dependent | Preserve externally / LFS or release storage, not ordinary Git blob |
| final_report.md | Library | 2026-09-24 | Canonical report candidate | Reproducible only if package inputs remain | Copy verified text into repo after provenance check |
| 2026-09-24-探索.gpx | Library | 2026-09-24 | Raw | Non-reproducible original observation | Preserve Raw; hash before import |
| 2026-09-22-探索.gpx and earlier actual GPX files | Library | through 2026-09-22 | Raw | Non-reproducible original observations | Deduplicate by SHA-256; preserve Raw |
| ハブAI_100m×10分_Exposure_2026-09-23.csv | Library | 2026-09-23 | Archive | Derived/reproducible if exact pipeline/config preserved | Archive |
| ハブAI_100m×10分_Exposure_2026-09-23(1).csv | Library | 2026-09-23 | Archive duplicate candidate | Derived | Hash-compare before keeping one |
| complete_walkforward_results.csv | Library | 2026-09-23 | Archive | Derived | Archive |
| ハブAI_再現試験_PDCA_2026-09-23.xlsx | Library | 2026-09-23 | Archive | Derived | Archive |
| Habu_AI_v2_89夜_GPS_GPX復元監査.xlsx | Library | 2026-09-01 | Archive / historical audit | Partly reproducible; historical source set matters | Archive, never treat old 89 as current canonical population |
| 2026-08-31 model comparison reports | Library | 2026-08-31 | Archive | Historical evaluation | Archive |
| 2026-09-04 gpx10m learning CSV / retrain bundle | prior Chat/Codex evidence | 2026-09-04 | Archive candidate | Derived | Recover exact bytes if available; otherwise mark missing |
| Route Map Generator Top-3 implementation | prior Codex/GitHub evidence | PR-era 2026-09-05 | Code/history candidate | Reproducible from Git history if commit/PR retained | Recover through Git history, not data import |

## Current canonical facts recovered from the 2026-09-24 audit
- GPX inventory: 25 files in the audited source set.
- Exposure: 18,531 rows = 68 Positive + 18,463 Negative.
- Terrain source counts recorded: DEM5B 203, DEM5C 194, DEM10B 286.
- Road GIS audit recorded: 2,177 tiles / 62,247 features.
- Official-weather audit recorded: 7,896 observations.
- Tide audit recorded: Amami O9, 365 days.
- Movement Front eligible/observed: 1,268 / 1,268.
- Walk-forward: 23 evaluation nights in the rebuilt series.
- Current rebuilt-series best: M1RS, mean ROC-AUC 0.643, capture percentile 0.642.
- Formal KPI remains unmet. Maximum mean Top10 Recall recorded as 0.033; maximum Top5% Recall 0.116.
- M6P prospective candidates are frozen but remain unscored on future nights; gate requires 10 future positive nights and 30 captures.
- The workbook explicitly says the historical formal M0-M5 fixed specifications were not recovered. M0R-M5R / RW / RF series must not be relabeled as those old models.

## Important version conflict
GitHub history already contains an older canonical-audit lineage, including an 88-night state on 2026-09-02. The later Library audit supersedes that evaluation dataset for current analysis, but old history must remain Archive, not be deleted.

## Promotion gate
Nothing in this manifest is promoted to Canonical merely because it is newer. Promotion requires:
1. exact file identity/hash where bytes are available,
2. provenance recorded,
3. no leakage/hindsight violation,
4. reproducibility status recorded,
5. raw observations preserved separately from derived artifacts,
6. current KPI/spec naming verified,
7. PR/CI review before merge.

## Known unresolved recovery gaps
- Exact bytes of some Chat/Work/Codex-generated artifacts are not yet located.
- Historical formal M0-M5 definitions are explicitly unrecovered in the 2026-09-24 audit.
- Current capture DB exact latest byte-version still needs resolution against older Library workbooks and the 09-12 update referenced in Chat.
- Work-session-only outputs that were never exported cannot be assumed to exist.
- GitHub code search is not currently indexed for this repository; history/file-path verification must use direct repository operations.

This file is an inventory/control document, not a claim that all listed artifacts have already been copied into GitHub.
