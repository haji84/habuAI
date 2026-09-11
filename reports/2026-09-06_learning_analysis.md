# 2026-09-06 operational-night learning analysis

## Status
- Operational night: 2026-09-06 (07:00 rollover rule)
- Sources: complete ACTUAL_GPX + field log
- GPX audit class: ① complete real GPX
- Self-captured Habu: 2

## GPX QC
- Trackpoints: 17,248
- Distance: 53.50 km
- Start: 2026-09-06 20:57:52 JST
- End: 2026-09-07 01:47:53 JST
- Duration: 4.83 h
- Median GPX gap: 1.000 s
- P95 gap: 1.000 s
- Max gap: 63.672 s
- Gaps >300 s: 0
- SHA-256: `93c8338a21517c53e9dd6335b78efa33fa7734666faa41683e2fdbfbd6e9cd1b`

## Confirmed self-captures
1. 2026-09-06 23:35 JST, Katoku, Habu ~90 cm juvenile, coiled, road edge, wet, 28.19282259182148, 129.3971515267262.
2. 2026-09-07 00:36 JST (belongs to 9/6 operational night), Habu ~90 cm juvenile, moving, mountain-foot, wet, 28.17333698841352, 129.3989442727244.

Both are small-class under the project size rule and both were on wet surface.

## Field-state observations
- 21:42 light rain.
- 22:09 rain-after marker.
- A free-text marker at 22:29:22 records light rain.
- 22:56:52 field note says rain stopped about 20 minutes earlier.
- 23:37 light rain, two minutes after the first capture.
- 00:23:56 field note says rain stopped.
- The second capture at 00:36 is about 12 minutes after the 00:23:56 rain-stop note.

The first capture is therefore not explained by a simple `within 20 minutes after rain` rule, while the second is compatible with a recent rain-transition state.

## Biological-reaction pattern
The first capture was preceded by a dense local sequence of biological observations in Katoku: repeated frog observations at 23:25, 23:28, 23:29, 23:30 and 23:34, plus other reactions. The capture occurred at 23:35.

The second capture was preceded by frogs (5+) at 00:29 and rabbit/woodcock reaction at 00:32, then capture at 00:36.

This supports testing a within-night biological-activity surge feature, but does not prove causality. Frogs were already observed from 21:35 onward without a Habu capture for roughly two hours, so raw frog count alone is not sufficient.

## Learning decisions
1. Keep wet-surface / antecedent-moisture state separate from rain-transition timing.
2. Do not promote `rain_after_20m` as a deterministic production rule.
3. Test relative biological-surge features against each night’s own baseline rather than raw counts alone.
4. Convert this complete GPX into canonical OSM 10m road exposure and Road×10min observations before strict non-capture labeling.
5. Preserve capture-label rescue for realistic field-entry lag while keeping the formal forecast KPI (100m ±20min) separate.
6. Compare 9/4, 9/5 and 9/6 as three consecutive wet operational nights with different timing and route patterns.

## Learning value
HIGH. Complete real GPX, two confirmed self-captures, extensive wet-surface biological observations, and explicit rain-transition notes.

## Production gate
This report is learning evidence only. It does not authorize production-model publication. Canonical ACTUAL_GPX inventory and formal OSM map-match gates remain mandatory.
