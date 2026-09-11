# 2026-09-05 operational night: learning analysis

## Status
- Operational night: 2026-09-05 (07:00 rollover rule)
- Evidence: ACTUAL_GPX + field log
- GPX audit class: ① complete real GPX
- User Habu captures: 4
- Fixed pre-survey point forecast: 3
- Forecast error: +1 actual vs forecast (underprediction)
- Fixed main window: 22:10-23:40
- Fixed counter window: 00:00-01:10
- Strict capture hits inside either fixed window: 0/4
- Important rule: do not widen windows after observing captures.

## GPX QA
- Trackpoints: 18007
- Start: 2026-09-05T22:36:36.047000+09:00
- End: 2026-09-06T03:41:12.998000+09:00
- Duration: 5.076931 h
- Distance: 61.641444 km
- Median sampling gap: 1.000 s
- P95 sampling gap: 1.000 s
- Max sampling gap: 70.070 s
- Gaps >300 s: 0
- SHA256: `396a3c2c29b2c3daa3c4616410c2002faf880fdc3abfc778f8f7cb5c6c6dfddc`
- BBOX lon/lat: [129.319201994, 28.1304473425, 129.4015619879, 28.1947174291]
- Assessment: continuous enough to derive surveyed/non-capture observations. A non-capture segment means no user capture during that observed passage, not biological absence.

## Confirmed user captures
1. 23:54, Setsuko, 130-140 cm female, coiled, road edge, wet, 28.1645522210766, 129.3905931162695.
2. 23:58, Setsuko, 150-160 cm female, coiled, road edge, wet, 28.16395791789445, 129.392508488569.
3. 01:33, Katoku, 100-120 cm, sex unknown, moving, road center, wet, 28.18594378571125, 129.3969946611967.
4. 01:59, 130-140 cm male, coiled, mountain-foot, wet. Coordinate associated with the same log block: 28.17179711407563, 129.3982904224454. Area label should be revalidated during canonical parsing because the capture block itself omits a fresh area token.

Size-label note: ranges crossing a project size threshold must not be forced into a single class without exact length. 150-160 is within project large class; 130-140 is medium. 100-120 touches the 120 cm boundary and remains unresolved until exact length is known.

## Weather / surface transitions observed in the field log
All four captures were recorded on wet road/surface conditions.
Logged transitions include:
- 00:01:33 fog
- 00:04:44 fog ended
- 00:12:07 fog
- 00:24:25 rain
- 00:45:57 fog
- 00:55:47 fog ended
- 01:03:23 light rain
- 01:25:48 rain
- 01:36:54 fog
- 01:41:00 light rain + fog ended
- 02:14:02 rain

Capture timing relative to logged transitions:
- 23:54 and 23:58 occurred before the first explicit fog marker, both on wet surface.
- 01:33 occurred 7m12s after the 01:25:48 rain marker.
- 01:59 occurred after the 01:41 light-rain/fog-end marker and before the 02:14 rain marker.
These are associations, not causal effects.

## Biological reaction signal
The normalized log contains many prey/other-animal observations, especially frogs. Simple event-block counts found frog-family labels repeatedly (カエル, オットンガエル, イシカワガエル), plus Yamashigi, Amami rabbit, Akamata, Ryukyu green snake, rat, and one Himehabu record. These should be converted to time-distance features rather than treated as independent causal predictors.

Recommended derived features:
- bio_count_10m / 20m / 30m
- bio_count_10min / 20min / 30min
- frog_count_10min / 30min
- minutes_since_last_frog
- road_network_distance_to_recent_bio
- rain_state_transition
- minutes_since_rain_start / rain_end
- fog_state
- wet_surface
- recent_positive_segment_7d / 14d / 30d

## Forecast evaluation
Point count:
- predicted = 3
- actual = 4
- signed error = +1 actual
- absolute error = 1
- ratio actual/predicted = 1.333

Efficiency:
- 61.6414435 km / 4 captures = 15.4104 km per capture
- 5.0769308 h / 4 captures = 1.26923 h = 76.15 min per capture

Strict fixed-window scoring:
- Main 22:10-23:40: 0/4
- Counter 00:00-01:10: 0/4
- Combined strict recall: 0%
- 23:54 capture was 14 min after main-window end.
- 23:58 capture was 18 min after main-window end.
- 01:33 capture was 23 min after counter-window end.
This indicates the count model was close but the time-of-night model was materially miscalibrated for this night.

## Spatial interpretation
- Two captures occurred only 4 minutes apart in Setsuko. This is a strong local positive cluster for this passage, but it must be exposure-corrected before calling the road intrinsically high probability.
- The 01:33 Katoku capture extends the positive period later than the fixed counter window.
- The 01:59 capture adds another late-night positive and should be map-matched before assigning a canonical road segment.
- Compare these positives with all GPX passages through the same 50/100/250 m road-network neighborhoods and with previous Setsuko positives. Do not compare raw capture totals without denominator/exposure.

## External research/context
JMA recent-rainfall tables show Koniya had substantial antecedent rainfall around this period: 48-hour maximum precipitation 53.0 mm as of 2026-09-04 20:50, and a 72-hour maximum 57.5 mm as of 2026-09-05 06:00. This supports treating antecedent moisture as a serious candidate feature, but does not prove rainfall caused the captures.

Tide/lunar state should remain candidate features only. The surrounding period was near the last-quarter phase. Exact Koniya tide values must be joined from the Koniya/O9 tide table in the canonical enrichment step; do not substitute another station as ground truth.

## Learning decisions
1. Include all 4 user captures as positive labels.
2. Exclude Himehabu from Habu positives.
3. Keep other species/reactions as biological-context features.
4. Generate surveyed non-capture observations from the complete GPX.
5. Add rain/fog transition features and antecedent precipitation.
6. Add micro-position categories: road_edge, road_center, mountain_foot.
7. Preserve sex and size-range fields without inventing exact lengths.
8. Strengthen time model evaluation because strict window recall was 0% despite count MAE=1.
9. Compare 2026-09-04 vs 2026-09-05 using exposure-normalized road×time observations, not raw totals.
10. Do not promote any tide effect to a causal feature until multi-night exposure-controlled validation supports it.

## Overall evaluation
Learning value: HIGH.
Reason: complete 1-second-class GPX, four confirmed user captures, dense biological observations, explicit wet-surface labels, multiple rain/fog transitions, and a useful forecast failure pattern. The most informative failure is temporal: the activity level was underestimated only modestly, while both fixed time windows missed all captures. This night should therefore carry high value for recalibrating Road×10min timing and weather-transition interactions.
