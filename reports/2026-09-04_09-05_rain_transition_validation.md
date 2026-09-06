# 2026-09-04 / 2026-09-05 field rain-transition validation

## Purpose
Validate whether the strong 2026-09-05 observation that all four user Habu captures occurred within 20 minutes of an explicit rain/light-rain field marker generalizes to another complete-GPX night before promoting `field_minutes_since_rain_marker` into the production prediction model.

## Evidence policy
- Operational-night boundary: 07:00 Asia/Tokyo.
- Positive label: user Habu capture only.
- Distance denominator: complete ACTUAL_GPX track distance, assigned at each track step to elapsed time since the latest explicit active precipitation marker (`小雨`, `雨`, `大雨`).
- `雨後` is not counted as active precipitation in the primary calculation because it describes a transition/result state rather than observed rain onset/intensity.
- No future weather information is used.
- This is a candidate-feature validation, not official v2 forecast accuracy scoring.

## 2026-09-04
Complete GPX: 83.762 km. User captures: 5.

Explicit precipitation markers begin at 22:28. Three early captures (21:45, 21:56, 22:12) therefore occurred before any explicit rain marker while their field log still records wet road/surface. The 23:54 capture occurred 64 minutes after the previous active-rain marker (22:50). The 01:32 capture occurred 5 minutes after the 01:27 light-rain marker.

Exposure / positives:
- no prior active-rain marker: 11.092 km, 3 captures, 2.705 captures / 10 km
- 0-10 min: 23.484 km, 1 capture, 0.426 / 10 km
- 10-20 min: 15.963 km, 0 captures
- 20-30 min: 9.415 km, 0 captures
- >30 min: 23.807 km, 1 capture, 0.420 / 10 km

Interpretation: 2026-09-04 does **not** support a hard rule that Habu activity is confined to the first 20 minutes after an explicit rain marker. Wet-surface activity clearly existed before the first logged precipitation transition.

## 2026-09-05
Complete GPX: 61.641 km. User captures: 4.

Capture elapsed times from the latest explicit active-rain marker:
- 23:54: 7 min after 23:47 rain
- 23:58: 11 min after 23:47 rain
- 01:33: 7.2 min after 01:25:48 rain
- 01:59: 18 min after 01:41 light rain

Exposure / positives:
- no prior active-rain marker: 1.529 km, 0 captures
- 0-10 min: 10.998 km, 2 captures, 1.819 / 10 km
- 10-20 min: 8.838 km, 2 captures, 2.263 / 10 km
- 20-30 min: 8.662 km, 0 captures
- >30 min: 31.615 km, 0 captures

Interpretation: the 2026-09-05 signal is strong within that night, but it is not sufficient by itself for promotion to production.

## Combined two-night exposure test
Combining 2026-09-04 and 2026-09-05:
- <=20 min after active-rain marker: 59.283 km, 5 captures, 0.843 / 10 km
- all other exposure (no prior marker, 20-30 min, >30 min): 86.121 km, 4 captures, 0.464 / 10 km
- crude rate ratio: 1.82
- Fisher exact test using 10 m exposure units, one-sided: p = 0.283

Result: directionally positive but statistically weak with the currently available complete-GPX + second-level field-rain evidence.

## Important contradictory evidence
The 2026-08-28 field log contains four user Habu captures on wet road/surface but no explicit rain marker was found in that log. Its canonical ACTUAL_GPX original is still missing from the production inventory, so it cannot yet be included in the exposure-normalized denominator. Qualitatively, however, it is another warning against turning `minutes_since_rain_marker <= 20` into a hard activity rule.

## Model decision
**Decision: KEEP AS CANDIDATE, DO NOT PROMOTE YET.**

The stronger working hypothesis is now:

`wet / antecedent moisture state` × `recent rain transition` × `road / microhabitat` × `time of night`

rather than `recent rain marker` alone.

Recommended next features/tests:
1. `surface_wet_state` from field log, separated from precipitation occurrence.
2. `antecedent_rain_6h`, `12h`, `24h`, `48h` and their interaction with wet surface.
3. `field_minutes_since_rain_marker` as a continuous candidate, not a hard threshold.
4. Interaction terms for 0-10 / 10-20 / 20-30 min with road segment history and biological reactions.
5. Re-run this exact exposure-normalized test when 8/28, 8/31, 9/2 and other complete original GPX + timestamped weather logs are canonicalized.
6. Never label pre-rain wet-road captures as counterexamples to moisture; they are counterexamples only to the narrow explicit-rain-marker hypothesis.

## Overall evaluation
- 2026-09-05-only rain-transition signal: HIGH within-night value.
- Cross-night generalization strength: LOW-MODERATE.
- Production readiness of <=20 min rule: NOT READY.
- Candidate value of moisture/rain-transition interaction: HIGH.
