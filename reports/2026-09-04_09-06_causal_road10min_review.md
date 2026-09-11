# 2026-09-04 to 2026-09-06 causal-safe Road×10min review

## Scope
Exploratory three-night comparison using the three user-provided complete GPX tracks. This review deliberately freezes biological features at the **start of each 10-minute bin** so no observation occurring later in the bin, including after a capture, can leak into the prediction feature vector.

## Exposure
- 92 actually surveyed 10-minute bins
- 198.905 km total GPX exposure
- 10 positive bins
- 11 confirmed self-captures (two 2026-09-05 captures share the 23:50 bin)
- 82 surveyed non-capture bins

Night totals:
- 2026-09-04: 30 bins, 83.762 km, 5 positive bins / 5 captures
- 2026-09-05: 32 bins, 61.641 km, 3 positive bins / 4 captures
- 2026-09-06: 30 bins, 53.501 km, 2 positive bins / 2 captures

## Critical leakage correction
A prior exploratory calculation counted biological observations through the **end** of the positive 10-minute bin. That can include observations occurring after the capture and therefore cannot be used as a pre-survey prediction feature. The causal-safe calculation below uses only observations timestamped before the bin start.

This changes the conclusion materially.

## Causal-safe biological features at bin start
Positive vs negative bins:
- bio events previous 10 min: mean 1.20 vs 1.29; AUC 0.462; Mann-Whitney p=0.681
- bio events previous 30 min: mean 4.10 vs 3.79; AUC 0.507; p=0.945
- 10-min recent/baseline surge ratio: mean 2.54 vs 3.22; AUC 0.508; p=0.940
- 30-min recent/baseline surge ratio: mean 2.86 vs 3.56; AUC 0.526; p=0.797
- minutes since last biological observation: positive mean 5.0 min vs negative 9.44 min; lower values are directionally favorable, but p=0.138 and the signal is not stable enough to promote.

Leave-one-night-out top-quartile surge thresholds also fail to reproduce on 9/5 and 9/6. Therefore the previously suspected `within-night biological surge` is **not validated as a standalone pre-bin predictor** on these three nights.

## Positive-bin details
At the start of the two 9/6 positive bins, biological activity is genuinely recent (last reaction about 1 min earlier for both bins), but this does not generalize cleanly across 9/4 and 9/5. Early 9/4 positive bins also have incomplete/absent prior biological logging, limiting inference.

## Revised feature decision
1. wet / antecedent moisture × road × time: KEEP, highest priority
2. rain transition × wet state: KEEP as interaction candidate
3. minutes since last biological reaction: KEEP as exploratory candidate only
4. raw bio/frog count: DOWNGRADE
5. within-night biological surge ratio: DOWNGRADE until more nights reproduce it causally
6. deterministic rain-after-20min rule: REJECT

## Method rule going forward
For Road×10min prediction, every dynamic feature must be frozen at the prediction/bin-start timestamp. No same-bin future observation may enter the feature vector. Capture-time retrospective analysis can still be reported separately, but must be labeled retrospective and never mixed with prospective model metrics.

## Production status
This is exploratory analysis, not a production retrain. Formal model promotion remains gated by canonical raw-GPX ingestion/provenance and full pipeline validation.
