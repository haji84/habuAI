# 2026-09-05 Road×10min preparation and canonical OSM gate

## Purpose
Prepare the 2026-09-05 operational night for the production Road×10min layer without weakening the canonical OSM map-match gate.

## Evidence status
- ACTUAL_GPX: present locally and QC class ①.
- GPX SHA256: `396a3c2c29b2c3daa3c4616410c2002faf880fdc3abfc778f8f7cb5c6c6dfddc`
- Trackpoints: 18,007
- Distance: 61.641 km
- GPX time: 22:36:36.047 to 03:41:12.998 JST
- Confirmed user captures: 4
- Existing canonical OSM 10m geometry exists in branch as `data/processed/road_segments_10m.geojson`, blob SHA `4e93e298263535ed4f2870d54738c524cc338696`, but its 23.6 MB geometry could not be transported through the current connector session.
- Therefore the local GPX-chain 10m output is ANALYSIS-ONLY and must not be promoted to production `segment_id`.

## Survey opportunity correction
Fixed main window: 22:10-23:40.
Actual GPX starts 22:36:36.
Main-window observed coverage = 70.44%.
The unobserved 22:10-22:36:36 period must be `NO_OPPORTUNITY`, not a negative label.

Fixed counter window: 00:00-01:10.
Counter-window observed coverage = 100.00%.
This window is fully surveyed and may contribute genuine surveyed non-capture observations.

Strict forecast score remains unchanged:
- Main fixed-window captures: 0/4
- Counter fixed-window captures: 0/4
- No post-hoc widening.

## GPX-chain 10m × 10min candidate dataset
For analysis only, the actual track was resampled at 10m:
- candidate 10m units: 6,165
- 10-minute clock bins: 32
- represented distance: 61.65 km
- positive bins:
  - 23:50: 2 captures / 0.76 km
  - 01:30: 1 capture / 1.25 km
  - 01:50: 1 capture / 1.10 km

These rates are descriptive within-night exposure rates, not intrinsic road risk.

## Capture-to-GPX QC
C1 23:54:
- capture coordinate lies 2.49m from the route spatially, but the route passed that point about 237s before the recorded capture time.
- at the recorded timestamp the GPX point is ~44.50m away.
- Both fall within the same 23:50-23:59 Road×10min bin.
- Classification: PASS_FOR_10MIN / REVIEW_FOR_FINE_TEMPORAL_ALIGNMENT.

C2 23:58:
- nearest route distance 1.88m; at-capture-time distance 4.82m.
- PASS.

C3 01:33:
- nearest route distance 0.91m; at-capture-time distance 3.41m.
- PASS.

C4 01:59:
- nearest route distance 0.01m; at-capture-time distance 1.58m.
- PASS.

## Canonical production transformation
After the raw GPX is ingested into the canonical repo inventory, the existing pipeline must:
1. parse the GPX,
2. map-match GPS points to canonical 10m OSM segments using the configured 35m distance threshold plus heading/continuity penalties,
3. collapse successive matched points into visits,
4. attach Habu captures to OSM segments,
5. attach weather/biological features,
6. aggregate only observed visits into `segment_id × 10min`,
7. create surveyed non-capture rows only where actual GPX exposure exists.

The new helper `scripts/export_road10min_training.py` performs step 6 from the pipeline observation table. It never invents negatives for unobserved time.

## Learning evaluation
The 2026-09-05 failure should be decomposed into:
- count calibration error: forecast 3 vs actual 4, MAE 1,
- fixed-window scoring failure: 0/4,
- opportunity mismatch: main window only 70.44% observed,
- late activity: all four captures after 23:40,
- route/weather/bio interaction: must be evaluated after canonical OSM IDs are available.

### Decision
Status = `READY_FOR_CANONICAL_OSM_MATCH`, not `PRODUCTION_TRAINED`.

No model artifact should be published from the GPX-chain candidate data.
