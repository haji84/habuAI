# Habu AI required validation final review — 2026-09-08

## Scope
This review deliberately separates facts supported by current reliable data from hypotheses that remain unproven. No candidate is promoted to production solely because it looked strong on one or three nights.

## Reliable-data boundary
- 89-night audit population exists.
- 10 nights are classified as complete actual GPX.
- 21 additional nights are only high-precision reconstruction candidates and must not be treated as equivalent to actual GPX until road-network map matching is completed.
- The current processed training artifact contains 8 actual-GPX sessions, 26,135 road-visit rows, and 14 positive visit labels.
- The current artifact reports `status: no-holdout`; therefore there is no completed independent production holdout for that artifact.

## Confirmed findings

### 1. Post-outcome speed leakage is real and must be excluded
In the 8-night processed artifact:
- positive visits: mean speed 0.664 m/s, median 0.432 m/s
- negative visits: mean speed 5.375 m/s, median 4.243 m/s
- Mann–Whitney two-sided p = 3.77e-10
- every positive night shows much lower speed at positive visits than the night’s negative median

Leave-one-night-out using only time + speed produced pooled AUC ≈ 0.98 and 14/14 positives in the top 10% score region. This is not credible forecasting skill; it is consistent with slowing/stopping after detecting/capturing a snake. `mean_speed_mps` has therefore been removed from the predictor feature list on the retraining branch and a guard test was added.

### 2. Raw biological counts are not a stable production predictor
On 8 actual-GPX nights aggregated to 10-minute bins:
- time + exposure baseline LONO AUC ≈ 0.634
- adding raw 30-minute counts for rat, Otton frog, frog, woodcock, and rabbit reduced LONO AUC to ≈ 0.515
- per-night direction changes repeatedly across nights

Conclusion: raw animal counts such as “more frogs = more Habu” are not supported as a stable main predictor. Biological observations may remain candidate contextual features, but not as a simple monotonic production rule.

### 3. Exact-road revisit history is not stable
On the same 8-night bin analysis:
- adding `segment_prior_visits` did not improve the baseline; LONO AUC ≈ 0.592 vs ≈ 0.634 baseline
- prior-visit deltas changed direction by night

On 2026-09-04 through 2026-09-06, same-10m prior visits and exact prior-capture segment also failed to generalize. Exact segment revisit history should not be treated as a main predictor.

### 4. Simple static road/environment context is not stable enough
Adding curvature and distances to stream/forest/farmland/residential/junction did not improve 8-night LONO performance (≈0.591 vs ≈0.634 baseline). Direction changed by night. These remain candidate interaction/context features, not established main effects.

### 5. Wet surface alone is not a valid rule
Field logs contain repeated multi-capture dry nights as well as multi-capture wet nights. Therefore `wet = active` and `dry = inactive` are contradicted by observed captures. Wetness may matter only as an interaction/state-transition variable.

### 6. Rain/water transition is promising but not proven
For 2026-09-04 through 2026-09-06, recent weather transition exposure showed higher capture rate, but:
- only three nights are available for this exact causal Road×10min feature
- one night moves in the opposite direction
- the pooled comparison is suggestive, not conclusive (previous Fisher one-sided p ≈ 0.0665)

Therefore rain onset/stop/post-rain timing must remain candidate-only until it generalizes across more actual-GPX nights.

### 7. Current archived-weather evaluation is not strict prospective evidence
The pipeline uses archived/reanalysis weather and has an explicit hindsight-weather diagnostic gate. In addition, the current join implementation uses nearest hourly weather within 40 minutes, which can associate a future hourly record with an earlier road visit. Archived-weather results must not be used as strict prospective evidence until a causal/prediction-time weather source and timestamp policy are enforced.

### 8. Current trained logistic model is not the current forecast generator
`fit_model()` trains a logistic regression artifact, but `make_forecast()` does not consume its probabilities. The current forecast path ranks historically traversed segments by smoothed capture rate and selects a provisional hourly peak from historical exposure/captures. Thus statements that the trained logistic model is directly generating the live forecast would be incorrect for this branch.

### 9. Current artifact is not production-validated
The processed artifact reports 8 GPX sessions, 14 positive visit labels and `no-holdout`. Its LONO PR-AUC is extremely low in the stored metrics. Production promotion must remain blocked until canonical GPX ingestion, causal-feature policy, and independent night-level evaluation are complete.

## Not confirmed and therefore not promoted
- exact minutes-after-rain effect
- wet/dry main effect
- raw frog/animal-count effect
- exact 10m historical hotspot effect
- 100/250/500m hotspot distance as a stable production effect
- static road class/curvature as a standalone effect
- full-season generalization from the current August–early September actual-GPX cohort
- formal 100m ±20min KPI target attainment

## Decisions
1. Exclude post-outcome movement speed from predictive features.
2. Do not use actual distance travelled inside a 10-minute outcome bin as a forecasting predictor; it may be affected by stopping for a capture. Keep distance as exposure/denominator or planned-route quantity only.
3. Keep raw bio counts, wetness, rain transitions, hotspot history and static road context as candidate/interaction features until multi-night out-of-sample validation passes.
4. Keep production model publication blocked.
5. Next evidence expansion should prioritize converting remaining reliable GPX/reconstructed nights into causal Road×10min exposure rows, then rerun night-level out-of-sample tests. Do not weaken the canonical GPX gate.
