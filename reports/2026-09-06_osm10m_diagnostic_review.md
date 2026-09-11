# 2026-09-06 OSM 10m diagnostic review

## Scope
Diagnostic map match for operational night 2026-09-06 using the complete uploaded ACTUAL_GPX and the project's recovered official OSM 10 m road-segment artifact. This is analysis evidence only. The raw GPX original is not yet in the repository canonical raw inventory, so production retraining remains blocked.

## GPX
- points: 17,248
- distance: 53.50 km
- time: 20:57:52 JST to 01:47:53 JST
- duration: 4.83 h
- gaps >300 s: 0
- SHA-256: `93c8338a21517c53e9dd6335b78efa33fa7734666faa41683e2fdbfbd6e9cd1b`
- audit class: ① complete ACTUAL_GPX

## OSM 10 m diagnostic match
- matched points: 17,244 / 17,248
- match ratio: 99.9768%
- median point-to-road distance: 4.98 m
- p95 point-to-road distance: 13.47 m
- segment visits: 4,934
- unique visited 10 m road segments: 3,283
- session eligibility against the 80% match gate: PASS

## Confirmed self-captures
1. 23:35 JST, Katoku, Habu ~90 cm juvenile, coiled, road edge, wet. Diagnostic road segment `OSM_ro599_0_22`; nearest observed GPX timing difference about 57 s.
2. 00:36 JST, operational 9/6, Habu ~90 cm juvenile, moving, mountain-foot, wet. Diagnostic road segment `OSM_ro327_0_31`; nearest observed GPX timing difference about 13 s.

## Road × 10 min positive windows
- 23:30 bin: about 1.230 km traversed, 1 confirmed capture.
- 00:30 bin: about 1.182 km traversed, 1 confirmed capture.

## Pre-capture biological context
- 23:35 capture: 9 biological event blocks in preceding 10 min; 14 in preceding 30 min.
- 00:36 capture: 4 biological event blocks in preceding 10 min; 11 in preceding 30 min.

Raw frog count is not sufficient as a rule: biological activity was already present much earlier in the night without a Habu capture. The candidate to test is a within-night surge relative to that night's own recent baseline, interacting with wet surface, rain-transition state, road and time.

## Gate
Do not convert all 3,283 visited segments into production negatives until the 2026-09-06 raw GPX original is canonical-ingested and SHA-verified. Formal forecast KPI remains 100 m ±20 min and is separate from training-label rescue logic.
