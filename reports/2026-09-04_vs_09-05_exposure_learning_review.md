# 2026-09-04 vs 2026-09-05 Exposure-Normalized Learning Review

## Scope
Purpose: explain why the 2026-09-05 fixed time windows missed all user captures even though the point count forecast (3) was close to the actual count (4), and convert the failure into training rules.

Operational-night rule: events through 07:00 belong to the previous night.

Evidence:
- 2026-09-04: complete ACTUAL_GPX + field log
- 2026-09-05: complete ACTUAL_GPX + field log
- User captures only are positive capture labels.
- Other-person captures, roadkill and Himehabu are not user-capture positives.

## Exposure-normalized comparison

| Metric | 2026-09-04 | 2026-09-05 |
|---|---:|---:|
| GPX points | 17,751 | 18,007 |
| Distance | 83.76 km | 61.64 km |
| Duration | 4.94 h | 5.08 h |
| User captures | 5 | 4 |
| Captures / 10 km | 0.597 | 0.649 |
| Captures / hour | 1.011 | 0.788 |
| Biological event blocks | 35 | 68 |
| Biological event blocks / hour | 7.08 | 13.39 |

Interpretation:
- Raw captures fell from 5 to 4, but distance also fell sharply.
- Distance-normalized capture efficiency was slightly higher on 09-05 (0.649 vs 0.597 captures/10 km).
- Time-normalized capture rate was lower on 09-05 (0.788 vs 1.011 captures/hour).
- Biological observation density was much higher on 09-05. This is consistent with a biologically active wet night, but must be modeled as a detection/context signal rather than proof of causality.

## Critical scoring correction: predicted window vs survey opportunity

### 2026-09-04
Fixed main window: 21:30-23:10
- GPX coverage: 97.0%
- Distance surveyed inside window: 21.65 km
- Captures inside window: 3

Fixed counter window: 00:20-01:20
- GPX coverage: 100.0%
- Distance surveyed inside window: 20.24 km
- Captures inside window: 0

### 2026-09-05
Fixed main window: 22:10-23:40
- GPX actually began at 22:36:36
- Coverage of the fixed main window: 70.4%
- The first ~26 minutes of the predicted main window were not surveyed.
- Distance surveyed inside window: 19.16 km
- Captures inside window: 0

Fixed counter window: 00:00-01:10
- GPX coverage: 100.0%
- Distance surveyed inside window: 8.72 km
- Captures inside window: 0

Learning rule:
Keep strict forecast scoring unchanged, but separate:
1. prediction-window hit/miss,
2. survey-opportunity coverage,
3. conditional hit/miss given actual survey exposure.
Otherwise an unobserved predicted interval is incorrectly taught to the model as a biological negative.

## 10-minute capture concentration
2026-09-04 capture-positive bins:
- 21:40: 1 capture, 3.56 km exposure
- 21:50: 1 capture, 0.73 km exposure
- 22:10: 1 capture, 1.15 km exposure
- 23:50: 1 capture, 2.34 km exposure
- 01:30: 1 capture, 3.16 km exposure

2026-09-05 capture-positive bins:
- 23:50: 2 captures, 0.76 km exposure
- 01:30: 1 capture, 1.25 km exposure
- 01:50: 1 capture, 1.10 km exposure

The 23:50 bin on 09-05 was the strongest observed short-window cluster: 2 captures during only ~0.76 km of GPX exposure. Treat this as a high-value passage signal, but shrink it with Bayesian smoothing because it is one night and one local passage.

## Route independence
Only about 0.61% of sampled 09-05 GPX points were within 100 m of the 09-04 route, and about 1.30% were within 250 m. Median nearest distance from the 09-05 route to the 09-04 route was 7.24 km.

This means the two nights are largely different route exposures. Raw 5-vs-4 capture totals therefore cannot be interpreted as a simple weather-only comparison. Road/area selection is a major confounder.

The nearest 09-04 user-capture point to each 09-05 capture was still roughly 9.0-9.8 km away. The 09-05 positives are not merely repeated observations of the exact 09-04 capture cluster.

## Biological-context signal before captures

For 09-05, all four captures had biological observations in the preceding 10 minutes:
- 23:54: 2 bio event blocks in prior 10 min
- 23:58: 2
- 01:33: 1
- 01:59: 4

In the preceding 30 minutes, 09-05 captures had 7-8 biological event blocks each, with repeated frog-family observations.

By contrast, early 09-04 captures at 21:45, 21:56 and 22:12 had much weaker prior biological-event density; later 09-04 captures had stronger biological context.

Inference:
- recent biological density may be useful for Road×10min re-ranking during a live survey,
- but it should not replace weather/road/exposure features,
- and it must be evaluated against non-capture passages with similarly high biological density.

## Weather-transition interpretation
09-05 field log recorded repeated fog/rain/light-rain transitions. All four user captures were recorded on wet surface.
- 01:33 capture was ~7 minutes after the 01:25:48 rain marker.
- 01:59 followed the 01:41 light-rain/fog-end marker.

Historical Habu literature supports retaining weather variables as plausible activity predictors. Nishimura (2000) found temperature, humidity and precipitation important in year-round climate analyses, while summer relationships could be weaker when temperature/humidity were already high. Earlier Amami behavioral work reported transient emergence associated with rainfall. These findings justify candidate features, not a causal label for any individual capture.

## Why the 09-05 peak shifted later
Supported by the data:
1. The 09-05 route was largely spatially different from 09-04, so road selection changed the exposure distribution.
2. The predicted main window was only ~70% actually surveyed.
3. Dense biological reactions and wet conditions continued well after midnight.
4. Strong capture clusters occurred at 23:54-23:58 and 01:33-01:59, later than the fixed windows.
5. Therefore the error is not purely a clock-time error. It is a joint Road × Time × Weather-transition × Survey-exposure problem.

Not supported:
- claiming tide caused the delay,
- claiming rain alone caused the delay,
- shifting the fixed window after seeing the outcomes,
- treating unsurveyed 22:10-22:36 as a negative.

## Model changes approved from this review
1. Add `survey_opportunity` / `window_coverage_fraction`.
2. Add `distance_exposure_10min` and `time_exposure_10min`.
3. Train negatives only from actually traversed road-time passages.
4. Add `rain_state_transition` and `minutes_since_rain_start/end`.
5. Add `fog_state` and `wet_surface`.
6. Add `bio_events_10m/20m/30m/60m`, with frog-specific derivatives.
7. Add `minutes_since_last_bio` and road-network distance to recent bio reaction.
8. Add Road×10min interaction features instead of relying on a global hour peak.
9. Maintain two timing metrics: strict fixed-window recall and exposure-conditioned recall.
10. Compare 09-04/09-05 only after OSM map matching; the current comparison is GPX-exposure normalized but not yet canonical OSM 10 m road-segment normalized.

## Evaluation
Learning value: VERY HIGH.

Most valuable finding:
The 09-05 failure should not teach "22:10-23:40 is a bad time." It should teach that the model needs to know whether the predicted interval was actually surveyed, which road was being surveyed, and whether local wet/bio conditions were active at that time.

Production status:
This report is suitable as a learning specification and diagnostic input. It is not a replacement for the canonical OSM 10 m × 10 min pipeline, and it must not bypass the ACTUAL_GPX provenance gate.
