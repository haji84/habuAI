# 2026-09-05 Road×10min positive-vs-negative analysis

## Scope
Operational night 2026-09-05 (07:00 rollover).
This is an exploratory within-night analysis using the SHA-verified GPX, OSM 10m map matching, fixed user-capture labels, field-log biological observations, and explicit rain markers.

## Dataset
- GPX road-match ratio: 99.9223%
- Segment visits: 3,160
- Unique visited 10m segments: 2,322
- 10-minute exposure bins: 32
- Positive 10-minute bins: 3
- Confirmed user captures: 4
- Negative 10-minute bins: 29
- Important: a negative means surveyed in that bin with no confirmed user capture, not biological absence.

## Positive bins
| bin | distance_m | unique_segments | captures | bio previous 10m | frogs previous 10m | bio previous 30m | frogs previous 30m | minutes since latest explicit rain marker at bin end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-05 23:50:00+09:00 | 755.7 | 36 | 2 | 3 | 2 | 12 | 10 | 13.0 |
| 2026-09-06 01:30:00+09:00 | 1248.6 | 69 | 1 | 3 | 3 | 7 | 5 | 14.2 |
| 2026-09-06 01:50:00+09:00 | 1100.6 | 58 | 1 | 4 | 1 | 10 | 7 | 19.0 |

## Rain-transition candidate
All three positive 10-minute bins ended within 20 minutes of the latest explicit rain/light-rain marker in the field log.

Exposure split:
- <=20 min after rain marker: 14 bins, 20.438 km exposure, 4 captures, 1.957 captures / 10 km.
- >20 min after rain marker: 17 bins, 39.326 km exposure, 0 captures.
- One pre-rain bin is excluded from this split because there was no prior explicit marker.
- Fisher exact exploratory test on positive-bin membership: p=0.0810 (one-sided). This is suggestive, not confirmatory.

Capture-level offsets from the latest explicit rain marker:
- 23:54 capture: 7.0 min
- 23:58 capture: 11.0 min
- 01:33 capture: 7.2 min
- 01:59 capture: 18.0 min

Interpretation:
- `minutes_since_field_rain_marker` is a strong candidate interaction for Road×10min.
- Do not hard-code a 20-minute rule. Rain markers are manually logged and not guaranteed to represent every rain onset/ending.
- Validate the threshold on multiple complete-GPX nights with exposure correction before promotion.

## Biological-reaction candidate
The simple raw density signal is weaker than expected.

Mean preceding biological counts by 10-minute bin:
- Positive bins: bio 10m mean 3.33; frog 10m mean 2.00; bio 30m mean 9.67; frog 30m mean 7.33.
- Negative bins: bio 10m mean 4.83; frog 10m mean 4.21; bio 30m mean 14.45; frog 30m mean 12.48.

This means "more frogs = capture" is not supported by this night alone. Biological observations are probably useful only as time-distance interactions, local prey/activity context, or change-from-baseline features rather than a raw count threshold.

## Timing-model implication
The old fixed-window miss should not be learned as a pure time-of-night failure:
- Main window began 22:10, but GPX began 22:36:36, so the first ~26.6 min had no survey opportunity.
- Captures occurred at 23:54, 23:58, 01:33, and 01:59.
- Activity appears clustered after logged moisture transitions rather than at a single universal clock hour on this night.

Recommended candidate features:
1. `minutes_since_field_rain_marker`
2. `rain_transition_0_10m`, `rain_transition_10_20m`, `rain_transition_20_30m`
3. `wet_surface`
4. interaction `rain_transition × road_segment_history`
5. interaction `rain_transition × frog/local_bio_density`
6. `survey_opportunity_fraction` for forecast-window scoring

## Model policy
- Keep this analysis as exploratory learning evidence.
- Do not alter the official KPI or widen fixed windows retrospectively.
- Do not promote the 20-minute threshold to production until multi-night validation.
- Preserve canonical GPX gate and no-leakage rules.
