# 2026-09-04 to 2026-09-06 three-night learning review

## Purpose
Compare three consecutive wet operational nights without converting unobserved time or non-canonical GPX into production negatives. The goal is to decide which Road×10min feature hypotheses deserve formal cross-night testing.

## Exposure summary
| operational night | GPX distance km | duration h | self captures | captures/10km | captures/h |
|---|---:|---:|---:|---:|---:|
| 2026-09-04 | 83.762 | 4.943 | 5 | 0.597 | 1.011 |
| 2026-09-05 | 61.641 | 5.077 | 4 | 0.649 | 0.788 |
| 2026-09-06 | 53.50 | 4.83 | 2 | 0.374 | 0.414 |
| **total** | **198.903** | **14.850** | **11** | **0.553** | **0.741** |

The raw capture count falls 5 → 4 → 2, but route, timing and exposure differ. Do not interpret this sequence as a weather-only effect.

## What survives the three-night review
### 1. Wet/antecedent-moisture state remains important
All four 9/5 captures and both 9/6 captures were recorded on wet surface. On 9/4, early captures also occurred under wet-road conditions. This makes surface/antecedent moisture a more stable candidate than a narrow rain-after-20-minute rule.

### 2. Rain transition is a modifier, not a deterministic trigger
9/5 strongly clustered captures near explicit rain markers. 9/6 is mixed: the 00:36 capture is compatible with a recent rain-stop transition, while the 23:35 capture precedes the next explicit light-rain marker. Therefore `minutes_since_rain_marker` should remain a candidate interaction feature, not a standalone rule.

### 3. Raw biological counts are unstable
9/5 and 9/6 contain dense frog/other biological activity, but biological reactions also occur for long periods without Habu capture. Existing multi-night review also showed that raw frog-count enrichment can be driven by individual nights.

### 4. Test within-night biological surge
The 9/6 positives are preceded by locally dense biological event sequences. The correct candidate is not `frog_count` alone but a causal-time-safe relative surge, for example:
- recent 10/20/30 min event rate versus trailing 60–120 min night baseline;
- frog-specific recent rate versus trailing baseline;
- minutes since last biological reaction;
- local-road-network distance to recent reactions;
- interactions with wet surface, rain transition, road and time.

All features must use only information available before the Road×10min prediction timestamp.

## Current hypothesis ranking
1. **Wet/antecedent moisture × road × time**: KEEP, high priority.
2. **Within-night biological surge × road × time**: KEEP, high priority, requires formal negative comparison.
3. **Rain transition × wet state**: KEEP as interaction candidate.
4. **Raw frog count**: DOWNGRADE as standalone feature.
5. **Rain-after-20min deterministic rule**: REJECT as standalone rule.

## Next formal test
After canonical ingestion of the raw GPX originals, build one common Road×10min table for 9/4, 9/5 and 9/6 and compare the 11 confirmed self-captures against only actually surveyed non-capture exposure. Use leave-one-night-out validation and report both discrimination and calibration. Keep the formal 100m ±20min forecast score separate from training diagnostics.

## Production gate
No new production model is claimed here. Missing canonical ACTUAL_GPX originals still block promotion. PR remains draft until provenance, OSM match, labels, tests and candidate-vs-current metrics all pass.
