# Existing complete-GPX nights: biological/weather enrichment review

## Scope
Diagnostic review of the currently runnable canonical subset in the GitHub Actions artifact:
- 8 ACTUAL_GPX sessions: 2026-08-13, 08-15, 08-17, 08-19, 08-21, 08-23, 08-24, 08-27
- 26,135 Road×10m visit rows
- 14 positive visit rows
- 26,121 surveyed non-capture rows

This is not the final 90-night population and must not be reported as production accuracy.

## Rain archive features
Simple archive-weather enrichment was weak. Positive rows were not consistently concentrated at larger recent-rain values:
- rain_1h_mm: positive mean 0.064 vs negative 0.135
- rain_3h_mm: 0.207 vs 0.416
- rain_6h_mm: 0.814 vs 1.129
- rain_24h_mm: 5.779 vs 5.827
- rain_48h_mm: 15.614 vs 23.815
- hours_since_rain: 4.857 vs 3.990

These values do not support a simple monotonic rule such as “more recent rain = more Habu” across this subset. They also use coarse archive weather, not second-level field transitions.

## Biological reaction features
Several univariate features appear enriched if all 26k rows are pooled, but night-by-night stability is poor.

### Amami rabbit, previous 10 min within 100 m
- positive mean: 0.286
- negative mean: 0.096
- pooled Mann-Whitney p≈0.0039

However the positive-minus-negative direction is positive in only 4 of the 7 nights with captures and negative in 3. This is not stable enough for a standalone production rule and multiple-feature screening inflates false-discovery risk.

### Frog, previous 30 min within 100 m
- positive mean: 3.500
- negative mean: 2.013
- pooled Mann-Whitney p≈0.0169

Night-level direction is positive in only 3 of 7 capture nights and negative in 4. The pooled result is strongly influenced by 2026-08-27, where the single positive row had a count of 28 against a night negative mean of about 2.05.

### Yamashigi, previous 30 min within 100 m
- positive mean: 1.571
- negative mean: 0.794
- pooled p≈0.085
- direction positive in 3 of 7 capture nights, negative in 4.

## Interpretation
Raw biological counts are highly night-dependent. Pooling all Road×10m rows creates pseudo-replication because thousands of rows from the same night share weather, observer effort, route choice and local animal-density conditions.

The correct next hypothesis is not “frog/rabbit count is high”. It is whether **recent biological activity is unusually high relative to that night / recent route baseline**, using only observations available before the Road×Time prediction point.

## Model decision
- Do not promote any simple frog/rabbit/Yamashigi threshold from this diagnostic screen.
- Keep existing biological features as candidates.
- Add no-future-leakage relative-intensity candidates, e.g. recent 10/30 min activity divided by or differenced from a trailing 60/120 min baseline available at that time.
- Evaluate with leave-one-survey-night-out or night-blocked validation, not random row CV.
- Treat zero biological reports carefully because logging intensity is observer-dependent.

## Priority interaction candidates
1. wet/antecedent-moisture × field rain transition
2. recent bio surge vs trailing-night baseline
3. recent bio surge × wet state
4. recent bio surge × road segment prior positives
5. time-of-night × road × moisture interaction

## Overall evaluation
The current evidence argues against simple one-factor ecology rules. The strongest direction for Habu AI remains a hierarchical Road×Time model with night-level activity state plus within-night spatial/temporal deviations.
