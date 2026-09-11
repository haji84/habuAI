# 2026-09-06 OSM 10m map-match learning review

## Scope
Diagnostic matching of the complete 2026-09-06 operational-night ACTUAL_GPX against the exact `road_segments_10m.geojson` recovered from the successful GitHub Actions pipeline artifact (run 34019660231). This is a diagnostic reuse of the canonical road geometry, not a claim that the 2026-09-06 raw GPX has already passed the repository canonical-input gate.

## GPX
- Operational night: 2026-09-06 (events through 07:00 belong to 9/6)
- Trackpoints: 17,248
- Distance: 53.50 km
- Time: 20:57:52 to 01:47:53 JST
- Confirmed self-captured Habu: 2

## OSM 10 m diagnostic match
Using the pipeline artifact's 10 m OSM road segments and the configured 35 m distance gate:
- matched points: 17,244 / 17,248
- match ratio: 99.9768%
- median nearest-segment distance: 4.98 m
- P95 nearest-segment distance: 13.47 m
- maximum nearest-segment distance: 42.23 m
- diagnostic visits: 4,934
- unique traversed 10 m segments: 3,283

This is comfortably above the 80% strict-session threshold and indicates that the night is suitable for surveyed/non-capture learning after canonical provenance ingestion.

## Capture assignment
Both confirmed captures align cleanly with the surveyed GPX, without requiring a large field-entry-lag rescue:
1. 23:35 Katoku capture -> `OSM_ro599_0_22`; nearest surveyed GPX point 0.002 m spatially and 57 s after the logged minute.
2. 00:36 capture -> `OSM_ro327_0_31`; nearest surveyed GPX point 0.005 m spatially and 13 s after the logged minute.

The apparent millimetre-scale distances reflect the capture coordinates lying effectively on the recorded GPX path after projection and should not be interpreted as GPS physical accuracy.

## Positive Road x 10 min bins
- 23:30-23:39: 1 capture, about 1.230 km travelled.
- 00:30-00:39: 1 capture, about 1.182 km travelled.

## Biological context
Event-block counts immediately before capture:
- 23:35 capture: 9 biological event blocks in prior 10 min; 14 in prior 30 min.
- 00:36 capture: 4 biological event blocks in prior 10 min; 11 in prior 30 min.

This supports testing a within-night biological-activity surge feature. It does not establish that raw biological count is causal, because biological observations were already present much earlier in the night without Habu capture.

## Rain / wet-state interpretation
- Both captures were recorded on wet surface.
- The first capture is not explained by a simple post-rain-20-minute rule; a light-rain marker was recorded two minutes after capture.
- The second capture occurred about 12 minutes after the 00:23:56 rain-stop note.
- Keep antecedent wetness and rain-transition timing as separate candidate signals.

## Learning decision
1. Accept the night as high-value diagnostic evidence.
2. Do not yet publish its surveyed passages as production negatives until the 2026-09-06 ACTUAL_GPX original is in the canonical repository inventory and verified.
3. After canonical ingestion, export OSM `segment_id x 10min` observations and compare 9/4, 9/5 and 9/6 with identical exposure definitions.
4. Keep the formal forecast KPI (100 m ±20 min) separate from capture-label assignment.
5. Candidate interaction remains: wet/antecedent moisture × rain transition × within-night bio surge × road × time.

## Status
`DIAGNOSTIC_OSM_MATCH_PASS`, not `PRODUCTION_TRAINED`.
