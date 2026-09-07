# Habu AI Field v1 Product Specification

Status: FROZEN FOR IMPLEMENTATION
Owner intent: maximize personal Habu captures with accurate nightly prediction, road-level routing, fast field logging, offline operation, and continuous learning.

## 1. Product goal

Build one field app for iPhone and Android that integrates:

Prediction AI + offline map + route optimization + exploration GPS + field input + results DB + income/cost management + learning.

Primary usage loop:

`Open app -> Tonight prediction -> Compare A/B/C routes -> Start exploration -> GPS track + field logging -> live re-score/reroute -> End exploration -> auto-score -> dashboard update -> learning data update`

The product must work in weak-signal and no-signal mountain roads. Network access improves weather/sync but must not be required for core field operation.

## 2. Platforms and architecture

Mobile client: Flutter, one codebase for iOS and Android.

Recommended local stack:
- Flutter UI
- MapLibre for map rendering
- OpenStreetMap-derived road data
- MBTiles or PMTiles for offline basemap/road package
- SQLite with Drift for local relational data
- device GPS for exploration tracking
- ONNX Runtime or equivalent mobile inference path for local model execution where practical

Backend/cloud is optional for field operation but required for sync, heavy retraining, dataset governance, model publishing, and backup.

## 3. Non-negotiable data semantics

### 3.1 Capture

`capture` means only a Habu personally captured by the user.

- Himehabu must never be stored as a capture.
- Capture count, season total, monthly total, revenue, and capture-model labels use Habu captures only.
- A capture event is a high-confidence positive observation with precise GPS and event time.

### 3.2 Sighting

`sighting` is a direct observation of any species, including Habu and non-Habu species.

- A Habu sighting is a positive occurrence observation but not a capture.
- Non-Habu sightings may be used as biological/context features.
- Sighting must never increase personal Habu capture totals or revenue.

### 3.3 Road event / roadkill observation

`road_event` records an organism found on or immediately associated with the road in a dead, injured, struck, run-over, or otherwise abnormal state. It applies to Habu and all other species.

Important temporal rule:
- `discovered_at` is the time the user found the organism.
- `presence_time` is unknown unless independently observed.
- The model must NOT use `discovered_at` as the organism's actual activity/presence time.
- The valid inference is only that the organism was present at or near the recorded location at some unknown time before or up to discovery.

Suggested state values:
- struck_dead
- struck_alive
- injured
- dead_cause_unknown
- road_dead
- unknown

Freshness may be recorded as an observation (`very_fresh`, `fresh`, `old`, `unknown`) but must not be converted automatically into a fabricated exact event time.

### 3.4 Night-date rule

Any exploration and observations occurring before 07:00 belong to the previous exploration night.

Example: 2026-09-08 01:30 belongs to exploration night 2026-09-07.

## 4. Field input UX

The field-entry UX must preserve the user's current Shortcut interaction pattern:

`Tap one choice -> immediately reveal/advance to the next required choice`.

No long forms while driving/stopped roadside. The user should be able to complete a standard capture record rapidly with large tap targets.

Main record buttons:
- Capture
- Sighting
- Road event

### 4.1 Capture flow

Capture is Habu only. Do not ask species first.

Recommended flow:
1. Tap Capture
2. Size / length class or measured length
3. Sex: female / male / unknown / juvenile if appropriate
4. Discovery location
5. Behavior
6. Road surface
7. Optional note/photo
8. Save

Automatically recorded without manual input:
- event timestamp
- GPS
- exploration night
- active exploration session
- current/nearest road segment
- available environmental snapshot

Current size classes for analytics:
- <=119 cm: small
- 120-149 cm: medium
- 150-169 cm: large
- 170-189 cm: very large
- >=190 cm: super large
- >=200 cm: out-of-standard class

### 4.2 Sighting flow

Recommended flow:
1. Species
2. Count
3. State/behavior where relevant
4. Discovery location
5. Road surface where relevant
6. Optional note/photo
7. Save

GPS/time/session are automatic.

### 4.3 Road event flow

Recommended flow:
1. Species
2. Count
3. State (`struck_dead`, `struck_alive`, `injured`, `dead_cause_unknown`, `road_dead`, `unknown`)
4. Freshness if observable
5. Road position/location
6. Optional note/photo
7. Save

Store discovery time but do not infer exact occurrence time.

## 5. Tonight screen

The default entry screen must answer the user's normal question, effectively: "今日の予想は？"

Required outputs:
- single integer capture forecast, never a range
- primary fixed time window
- secondary fixed time window
- confidence/uncertainty indicator
- current environment summary
- recommended route A/B/C
- one clearly marked recommended route
- start exploration button

Forecast windows are fixed before exploration and must never be widened after results are known.

Capture quantity is the primary route objective. Large-Habu targeting is not the default objective.

## 6. Prediction hierarchy

Required model hierarchy:

`Tonight Activity -> 1 km Area -> 500/250/100 m Road Zone -> 10 m Road Segment -> Road x 10 min -> Route Optimizer -> Top 3 Routes`

Official evaluation KPI:
- 100 m within +/-20 min

Supporting KPIs:
- 50 m +/-10 min
- 100 m +/-10 min
- 100 m +/-30 min
- 250 m +/-30 min

Stage 1 target:
- 1 km Area Top10 Recall >=90%

Long-term target:
- 100 m +/-20 min >=90%

## 7. Zero / negative observations

The user does not manually record "0" at every road.

If GPS/search logs prove a road/time interval was searched and no capture was recorded, create a non-capture observation for that passage.

Interpretation:
- "not captured during this passage"
- NOT "no Habu existed there"

Detection probability/observer process must remain distinct from ecological presence probability.

## 8. Offline map

The app must own its field map. Google Maps must not be the route-generation engine.

Principle:

`GIS decides geometry; AI scores roads.`

Requirements:
- accurate OSM-derived roads stored offline for the target area
- real road geometry only, no AI-invented streets or straight-line fake routing
- roads segmented into 10 m analysis segments where required
- display current GPS position
- display planned A/B/C route
- display searched vs unsearched road segments
- display capture, sighting, and road-event points with layer toggles
- display excluded roads and user-verified road type
- allow map pan/zoom while offline

The map package must be downloadable/updatable while online and fully usable offline afterward.

## 9. Road classification and constraints

Each road/segment should support:
- road_environment
- is_forest_road
- user_verified_road_type
- risk/exclusion state

Priority of truth:
1. user verification
2. verified GPX/search evidence
3. OSM metadata

Nightly operational exclusions, such as `forest_road_allowed=false`, are constraints and must not be interpreted as low ecological probability.

## 10. Route optimizer

Generate real, connected, driveable/searchable routes over the road network.

Every nightly plan returns three meaningfully different routes:

A. Capture maximum
- primary objective: maximize expected Habu captures

B. Efficiency maximum
- maximize expected capture value per time/distance

C. Alternative/diversification
- prioritize high-value unsearched roads and strategic diversity while still respecting capture value

Large-Habu targeting is only a secondary feature unless explicitly requested.

All three must obey the same safety/road constraints.

Required route metadata:
- expected capture value
- route score
- distance
- expected duration
- recommended start/end
- areas/roads/order
- turnaround point
- forest-road presence
- searched/unsearched ratio
- risk
- pairwise overlap

A default diversity threshold around 70% overlap may be used initially but must be configurable and validated empirically.

The optimizer must score routes using arrival time because `Road x 10 min` probability is time-dependent.

## 11. Exploration mode

When Start Exploration is pressed:
- create exploration session
- start GPS tracking
- mark traversed road segments as searched
- preserve current prediction and fixed forecast windows for later scoring
- show current route and next recommended road

During exploration, new observations can trigger an online-in-session re-score:
- current location
- current time
- remaining time
- already searched roads
- new capture/sighting/road event

The app may propose a reroute, but original forecast values must remain preserved for evaluation.

## 12. Learning model

Use two layers of updating:

### 12.1 In-session adaptation

Lightweight recalculation/reweighting using the night's newest observations to update remaining-road priorities.

This is not allowed to rewrite the original pre-search prediction used for scoring.

### 12.2 Post-session learning

After End Exploration:
- finalize GPX
- finalize captures/sightings/road events
- generate searched non-capture passages
- join environmental/tide/road features
- score predictions
- append canonical training data
- publish model-update candidate

Heavy retraining may run in cloud/desktop infrastructure. Mobile receives a versioned validated model package afterward.

## 13. Dashboard

Dashboard must support season, month, and metric switching.

### 13.1 Season definition

Season is grouped by year. User selects year, e.g. 2025 / 2026 / 2027.

Core cards:
- season Habu captures
- current-month Habu captures
- exploration nights
- distance traveled
- fuel cost
- capture revenue
- other costs
- profit/net

### 13.2 Month drill-down

User can select a month and see at minimum:
- captures
- exploration nights
- distance
- fuel cost
- revenue
- profit
- captures per night
- km per capture
- captures per 100 km
- time per capture if session duration is available

Optional breakdowns:
- size class
- sex
- area
- road type
- time window

Dashboard map filters should support season/month selection and layer filtering.

## 14. Vehicle and fuel-cost model

Support one or more vehicle profiles.

Vehicle fields:
- vehicle name
- fuel type
- average fuel economy km/L
- optional measured fuel economy history
- default/current fuel price per L

Estimated fuel cost:

`session_distance_km / fuel_economy_km_per_l * fuel_price_per_l`

Prefer measured real-world fuel economy when sufficient data exists.

Allow fuel price and fuel economy history so old sessions are not silently recalculated using a future price unless the user explicitly requests recalculation.

## 15. Revenue and profit

Capture revenue must use Habu captures only.

Store capture unit price as a configurable value with effective dates.

Session profit:

`capture revenue - fuel cost - other recorded session costs`

Dashboard should also be able to calculate:
- profit per exploration night
- profit per hour
- profit per km
- revenue per capture

## 16. Offline behavior

Must work offline for:
- previously downloaded map
- GPS recording
- field input
- local DB
- latest downloaded model inference
- route display/recalculation that uses locally available data
- tide/moon/sunset calculations when locally calculable or preloaded

Latest live weather cannot be guaranteed offline. Show data age clearly, for example:

`Weather last updated: 20:14, using cached forecast`

When connectivity returns:
- sync local records
- fetch updated weather/model/map packages
- resolve conflicts without losing local field records

Offline-first rule: local field data is authoritative until successfully synced.

## 17. GPX quality and historical data

Historical exploration nights remain classified:
1. complete real GPX
2. high-precision reconstructed GPS
3. partial reconstruction
4. spatial-only reconstruction
5. unreconstructable

Do not discard non-complete nights automatically. Their reliability level must be carried into training/evaluation weighting.

## 18. Data integrity rules

Hard invariants:
- Himehabu cannot increment Habu capture totals.
- Sighting cannot increment Habu capture totals.
- Road event cannot increment Habu capture totals.
- Road-event discovery time cannot become exact presence/activity time unless independently verified.
- Forecast quantity must be one integer.
- Forecast windows are immutable after search begins.
- Real map geometry must come from GIS/road data.
- A route must be topologically connected and must not teleport between roads.
- Offline field records must never be deleted because cloud sync failed.

## 19. Acceptance criteria for v1

v1 is technically complete when a user can, on both iPhone and Android:
1. install the app
2. download the Setouchi offline map package
3. see a stored/latest nightly forecast
4. view A/B/C real-road routes
5. start an exploration and record GPS offline
6. record Habu capture using sequential tap flow
7. record sightings for any species
8. record road events for any species without inventing event time
9. end exploration and retain GPX + observations locally
10. see season/month dashboard totals
11. see distance, fuel cost, revenue, and profit
12. sync successfully after connectivity returns
13. preserve pre-search forecast for scoring
14. create training-ready searched/non-capture observations from GPS passages

Prediction accuracy KPI improvement is an ongoing model objective and is not faked as a binary app-release gate.

## 20. Implementation principle for AI employees

Do not rebuild the existing Habu AI model/data pipeline from zero. Reuse and extend the existing `haji84/habuAI` Python/GIS assets.

Build in small dependency-ordered PRs. No monolithic app rewrite.

Human approval should be required only for genuinely irreversible/high-impact choices, secrets, store publishing, paid services, destructive migrations, or unresolved product ambiguity. Normal implementation, tests, documentation, refactors, and reversible architecture work should proceed autonomously.
