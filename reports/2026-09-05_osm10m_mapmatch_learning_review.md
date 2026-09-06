# 2026-09-05 OSM 10 m map-match learning review

## Scope
Operational night `2026-09-05` using the project 07:00 rollover rule.

Evidence used for this diagnostic:
- actual GPX: `/mnt/data/2026-09-05-探索.gpx`
- GPX SHA256: `396a3c2c29b2c3daa3c4616410c2002faf880fdc3abfc778f8f7cb5c6c6dfddc`
- official 10 m OSM road segments recovered from GitHub Actions artifact `9968569395`, workflow run `33962615953`
- same map-match parameters as the production pipeline: 35 m max distance, 12 m heading penalty, 8 m continuity penalty

The raw GPX file itself is not yet promoted as a canonical repository original. This report is therefore a diagnostic learning artifact, not permission to bypass the canonical ACTUAL_GPX inventory gate.

## GPX road-map match QC
- GPX points: **18,007**
- points matched to OSM 10 m road segments: **17,993**
- map-match ratio: **99.9223%**
- median map-match distance: **6.36 m**
- p95 map-match distance: **16.37 m**
- strict session threshold: **80%**
- strict session eligible: **YES**
- derived segment visits: **3,160**
- unique 10 m road segments visited: **2,322**

This is high-quality exposure evidence. Surveyed non-capture labels may be derived from the observed passages, subject to the confirmed-capture labeling guard below.

## Confirmed user Habu captures and OSM road ids

| capture time JST | size / sex | nearest event OSM segment | road metadata | event-road distance |
|---|---|---|---|---:|
| 2026-09-05 23:54 | 130-140 cm female | `OSM_ro440_0_74` | unclassified | 5.57 m |
| 2026-09-05 23:58 | 150-160 cm female | `OSM_ro42_0_21` | `林道節子線`, unclassified | 0.63 m |
| 2026-09-06 01:33 | 100-120 cm, sex unknown | `OSM_ro280_0_4` | unclassified | 5.60 m |
| 2026-09-06 01:59 | 130-140 cm male | `OSM_ro249_0_52` | unclassified | 13.14 m |

## Critical labeling defect found
The base production labeler requires the capture event's snapped 10 m segment id to be identical to a visited GPX segment id within ±10 minutes. On this night that exact-id rule labeled only **2/4** confirmed captures as positives.

The missing two are not biological negatives. They are logging / map-snapping mismatch cases:

- 23:54 capture: the actually visited road segment nearest the logged capture coordinate was only **6.56 m** away and was entered about **235 s before** the log timestamp.
- 01:59 capture: the actually visited road segment nearest the logged capture coordinate was **13.26 m** away and was entered about **140 s before** the log timestamp.

Therefore an exact 10 m segment-id join would incorrectly convert two confirmed captures into surveyed non-capture negatives.

## Fix adopted
For **confirmed user Habu captures only**:
1. keep the existing exact 10 m segment + ±10 min join as the primary label;
2. if no exact positive is assigned, search the actually observed GPX visits within ±10 min;
3. assign the event to the nearest visited road segment only when its geometric distance is ≤50 m;
4. mark the rescued label as `spatiotemporal_fallback_50m_10min` and retain event-distance and time-offset audit columns;
5. never apply this rescue to Himehabu or other species.

This is a **training-ground-truth alignment rule**, not the formal v2 evaluation KPI. The official evaluation target remains 100 m ±20 min.

## 10-minute exposure around positives
- 23:50 bin: about **755.68 m** surveyed, **36** unique 10 m segments, **2 captures**.
- 01:30 bin: about **1,248.58 m** surveyed, **69** unique 10 m segments, **1 capture**.
- 01:50 bin: about **1,100.58 m** surveyed, **58** unique 10 m segments, **1 capture**.

These bins should be compared against all zero-capture 10-minute bins with the same exposure denominator. Raw capture totals alone are not sufficient.

## Learning evaluation
Learning value: **HIGH**.

Reasons:
- complete high-frequency actual GPX;
- 99.92% OSM road-map coverage;
- four confirmed user captures;
- multiple positive 10-minute bins;
- enough surveyed non-capture exposure to build hard negatives;
- a concrete label-alignment failure was discovered and protected by a regression test.

## Production gate
Do **not** publish a new canonical production model merely from this report. The full canonical ACTUAL_GPX original inventory must still pass. The 2026-09-05 raw GPX must be ingested with the SHA above, and the pipeline must reproduce four confirmed Habu positives before promotion.
