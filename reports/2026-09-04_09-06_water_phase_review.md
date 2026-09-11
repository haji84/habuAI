# 2026-09-04 to 2026-09-06 causal water-phase review

## Scope
Three operational nights, 92 surveyed 10-minute bins, 10 positive bins, 11 self-captured Habu. All water-state features are frozen at the **start of each 10-minute bin**. No observations from later in the bin are used.

## Main finding
The strongest pooled screening signal is a **weather transition within the previous 10 minutes**, but it is not stable enough to become a deterministic rule.

- transition <=10 min: 44.196 km, 6 captures, 1.358 captures/10 km
- all other exposure: 154.709 km, 5 captures, 0.323 captures/10 km
- crude rate ratio: 4.20
- one-sided Fisher test on positive/negative 10-minute bins: p=0.0665

This is suggestive, not established.

## By-night stability
- 2026-09-04: recent transition <=10 min underperformed other exposure (0.384 vs 0.693 captures/10 km).
- 2026-09-05: all 4 captures fell in recent-onset <=10 min exposure; 4.501 captures/10 km in that exposure vs 0 outside.
- 2026-09-06: recent transition <=10 min was enriched (1.079 vs 0.226 captures/10 km), but the two captures split between a recent rain-stop state and a long-post-rain state.

Therefore the pooled effect is partly driven by 9/5 and should not be hard-coded.

## Surface state
Prior surface state was available in 83/92 bins. Nine of ten positive bins had a prior surface observation and all nine were wet. However, the three nights were overwhelmingly wet overall, so this dataset cannot estimate a clean wet-vs-dry causal contrast.

## Leave-one-night-out screening
Baseline = time-of-night + distance traversed in the bin. Candidate models add only information known by bin start.

| model | mean AUC | mean AP | mean Brier | mean log loss |
|---|---:|---:|---:|---:|
| baseline time+distance | 0.573 | 0.261 | 0.243 | 0.692 |
| + simple water state | **0.653** | 0.226 | **0.212** | **0.640** |
| + detailed water phase | 0.583 | 0.243 | 0.215 | 0.652 |

Interpretation: simple water-state information improved mean discrimination and calibration, especially on 9/5 and 9/6, but hurt 9/4. The detailed phase encoding did not clearly outperform the simpler representation. With only three nights, this is a screening result, not production evidence.

## Decision
1. KEEP `recent_weather_transition_10m` as a candidate interaction feature.
2. KEEP prior wet-state availability/state with explicit missingness.
3. DO NOT hard-code `rain_after_20m` or any fixed rain window.
4. Prefer a compact water representation over many fine-grained phase buckets for now.
5. Next test should add road/spatial context and evaluate on more nights, while preserving the same causal cutoff.

## Production gate
No production retrain or model promotion is claimed. Formal forecast scoring remains separate from this exploratory 10-minute-bin screening.
