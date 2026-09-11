# 2026-09-04 to 2026-09-06 road-history / prior-hotspot review

## Scope
Leakage-safe diagnostic analysis on the 92 Road×10min bins from operational nights 2026-09-04 through 2026-09-06. There are 10 positive bins containing 11 confirmed self-captures and 82 surveyed non-capture bins. All dynamic features use only information available before the current operational night / bin.

## Exact 10 m road-history features
Positive vs negative bin means:
- distance-weighted prior visit count: 4.149 vs 4.404, Mann-Whitney p=0.730
- fraction of route on previously seen 10 m segments: 0.690 vs 0.629, p=0.615
- fraction of route on previously captured 10 m segments: 0.0031 vs 0.0027, p=0.993
- time since 20:00: 3.817 h vs 4.140 h, p=0.629

A leave-one-night-out logistic model using exact road-history + time performed poorly:
- overall ROC AUC 0.263
- average precision 0.077
- per-night AUC: 9/4 0.272, 9/5 0.356, 9/6 0.161
- top 20% scored bins captured 0/10 positive bins

Conclusion: exact 10 m revisit frequency / exact prior-capture-segment identity is not useful as a production predictor on these three nights and should be downgraded.

## Prior capture neighborhood features
To avoid over-specific 10 m identity, each current traversed segment was compared with capture segments from prior eligible nights only.

Positive vs negative means:
- mean distance to prior capture segments: 2,894 m vs 3,227 m, p=0.674
- fraction within 100 m: 0.058 vs 0.079, p=0.994
- fraction within 250 m: 0.180 vs 0.189, p=0.831
- fraction within 500 m: 0.447 vs 0.307, p=0.461

The 500 m neighborhood shows a directional enrichment in positives, but it is weak and not statistically established.

Leave-one-night-out hotspot-neighborhood + time model:
- overall ROC AUC 0.478
- average precision 0.113
- per-night AUC: 9/4 0.456, 9/5 0.644, 9/6 0.482
- top 20% scored bins captured 2/10 positive bins

Conclusion: prior hotspot proximity is at most a weak interaction feature. It does not justify a standalone hotspot rule yet.

## Static road context
Distance-weighted road-context comparison:
- curvature: positives 4.581 vs negatives 4.103, p=0.426
- secondary-road fraction: 0.100 vs 0.176, p=0.230
- tertiary-road fraction: 0.000 vs 0.012, p=0.225
- unclassified-road fraction: 0.216 vs 0.321, p=0.236
- residential fraction: 0.080 vs 0.027, p=0.494
- service/track fraction: 0.090 vs 0.031, p=0.890

Leave-one-night-out static road context + time:
- overall ROC AUC 0.560
- average precision 0.212
- per-night AUC: 9/4 0.576, 9/5 0.563, 9/6 0.232
- top 20% scored bins captured 3/10 positive bins

Static road context is slightly better than simple road history, but it is not stable across nights, especially 9/6.

## Surface-state coverage
Using only the latest independent surface observation available before each 10-minute bin, with a 60-minute freshness limit:
- surface state available: 83/92 bins
- positive bins with available state: 9/10
- every surface-available positive bin had a wet / soaked state

However nearly all surface-available bins were wet, so these three nights cannot estimate a clean wet-vs-dry effect size. Wet state remains a necessary context candidate, not a proven discriminator within this wet-night subset.

## Decision
1. DOWNGRADE exact 10 m revisit count as a standalone feature.
2. DOWNGRADE exact previous-capture-segment identity as a standalone feature.
3. KEEP 500 m prior-hotspot proximity only as a weak interaction candidate.
4. KEEP static road context, but require more nights before promotion.
5. KEEP antecedent wet/moisture state as context; do not claim it discriminates within these three mostly-wet nights.
6. Next priority should shift toward rain-transition timing, moisture history and road-time interaction rather than simple road-history frequency.

No production model retrain or promotion is claimed by this review.