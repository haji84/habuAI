# Habu AI v2 2025 canonical-population conflict audit

Canonical operational boundary: Asia/Tokyo 07:00. This audit concerns whether a night is proven to be the user's exploration night, not whether a historical row can be relabeled as a biological capture.

## Decisions

### 2025-10-12: include
A separate confirmed self-capture exists at 2025-10-13 01:40 in 油井林道. Under the canonical 07:00 boundary it belongs to operational night 2025-10-12. Therefore the night is population-proven independently of the ambiguous 2025-10-12 20:00 capture_or_sighting row.

### 2025-11-18: exclude by default as likely date duplicate
Recovered 11/18 rows form a sequence at 嘉鉄 18:30, 嘉鉄付近 immediately afterward, and 小名瀬〜阿室釜 around 23:00. The recovered 10/18 records contain the same sequence: 嘉鉄 capture at 18:30, 嘉鉄付近 roadkill immediately afterward, then 小名瀬〜阿室釜 capture around 23:00. The recovery-candidate sheet itself flags the 11/18 records as similar to 10/18 / possible date duplication. No independent raw/session/GPX evidence for 11/18 was found in the Library scan. Keep quarantined unless new independent evidence appears.

### 2025-10-15: exclude by default pending independent evidence
The recovery-candidate table rates the 嘉徳林道 capture as medium confidence and explicitly says it must be checked against 10/17 and other 嘉徳 logs. The reconstructed DB later labels it confirmed, but this is derived from the same recovery lineage. No independent raw/session/GPX evidence was found. Do not let the derived confirmed label establish the population by itself.

### 2025-11-07: exclude by default pending provenance resolution
The main/reconstructed lineage retains capture_or_sighting / 捕獲候補 and describes the record as detail-insufficient. A capture-label-corrected derivative promotes it to confirmed capture. Because the source lineages conflict and no independent raw/session/GPX evidence was found, the corrected derivative is not sufficient to establish the exploration night.

### 2025-09-22, 09-25, 09-26, 09-28: exclude by default pending independent evidence
The reconstructed/source lineage labels these as capture_or_sighting / 捕獲候補. A later capture-label-corrected derivative promotes them to confirmed captures, but no independent raw/session/GPX evidence was found in this audit. Keep them quarantined rather than inheriting the derivative relabeling.

## Important broader September note
The same source lineage also contains 2025-09-17, 09-18 and 09-20 as capture_or_sighting / 捕獲候補. Any canonical-population builder must apply the same evidence rule to those dates too, even if an older workbook happened to include them without a conflict flag.

## Rule
A derived workbook that changes `capture_or_sighting` to `捕獲` cannot, by itself, upgrade population provenance. Population inclusion requires independent exploration evidence or an unambiguous source-level self-capture / explicit-zero record.
