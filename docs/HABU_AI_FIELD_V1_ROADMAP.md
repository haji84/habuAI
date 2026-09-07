# Habu AI Field v1 Implementation Roadmap

This roadmap is dependency-ordered. AI employees should take the next unblocked issue, implement it in a small PR, run tests, update docs/state, then continue.

## Phase 0: contracts and guardrails

Goal: lock schemas, invariants, and interfaces before mobile implementation.

Deliverables:
- canonical entities: ExplorationSession, Capture, Sighting, RoadEvent, VehicleProfile, FuelPrice, Expense, ForecastSnapshot, RoutePlan, RouteSegmentObservation, ModelPackage
- exploration-night date resolver with 07:00 cutoff
- capture-only-Habu invariant
- road-event temporal uncertainty fields
- versioned JSON/schema contracts between Python predictor and mobile app
- unit tests for all hard invariants

Exit criteria:
- schemas documented and tested
- no ambiguous use of discovered_at as road-event occurrence time
- no Himehabu path can increment Habu capture totals

## Phase 1: Flutter offline-first app shell

Goal: iOS/Android app starts and persists local data offline.

Deliverables:
- Flutter project
- bottom navigation: Tonight / Map / Record / Dashboard / Settings
- SQLite/Drift local DB
- repository/service layer
- migration strategy
- local event queue for later sync
- basic app-state tests

Exit criteria:
- app runs on iOS simulator/device and Android emulator/device
- records survive app restart without network

## Phase 2: sequential field-entry UX

Goal: reproduce the fast Shortcut interaction pattern.

Deliverables:
- three primary buttons: Capture / Sighting / Road Event
- one-choice-then-next-step flow
- large touch targets
- automatic timestamp/GPS/session binding where available
- edit/cancel/review-before-save path
- capture flow without species selection because Capture is Habu-only

Exit criteria:
- standard capture can be completed rapidly without long form scrolling
- Himehabu cannot be entered through Capture
- sighting/road-event support all configured species

## Phase 3: exploration GPS and session lifecycle

Goal: reliable offline exploration recording.

Deliverables:
- Start Exploration / End Exploration
- background-capable GPS tracking as platform policy permits
- GPX/export representation
- distance and duration calculation
- persisted track points
- crash/restart recovery for active session
- night-date assignment

Exit criteria:
- offline session records route, distance, timestamps, and observations
- session can recover from foreground/background transitions without silent data loss

## Phase 4: offline map package

Goal: accurate field map without Google routing dependency.

Deliverables:
- MapLibre rendering
- downloadable Setouchi offline package
- OSM-derived roads
- current GPS position
- capture/sighting/road-event layers
- searched/unsearched visualization support
- user-verified road classification overlay

Exit criteria:
- map pans/zooms offline
- no invented straight-line road geometry
- observed points align with real GIS coordinates

## Phase 5: prediction package integration

Goal: mobile Tonight screen consumes existing Habu AI outputs.

Deliverables:
- versioned ForecastSnapshot schema
- single-integer nightly forecast
- immutable primary/secondary windows
- environment summary and data-age display
- latest validated model package/version display
- cached/offline prediction support

Exit criteria:
- app can display a precomputed prediction entirely offline
- search start freezes the forecast snapshot for later scoring

## Phase 6: road graph and route optimizer

Goal: generate real connected exploration routes from scored road segments.

Deliverables:
- road-network graph built from GIS geometry
- connectivity validation
- time-dependent segment scoring
- constraints: forest-road allowed, risk, max time/distance, exclusions
- candidate route generation
- A Capture Max
- B Efficiency Max
- C Alternative/Diversity
- overlap calculation and configurable diversity threshold

Exit criteria:
- no teleporting routes
- route geometry follows real road network
- A objective prioritizes capture count by default
- all route candidates respect hard constraints

## Phase 7: live exploration rerouting

Goal: respond to new field observations without corrupting evaluation.

Deliverables:
- mark traversed road segments searched
- in-session lightweight rescoring
- reroute using current location/time/remaining time
- capture/sighting/road-event update hooks
- preserve original pre-search forecast and route for scoring

Exit criteria:
- rerouting changes remaining plan only
- original forecast remains immutable and auditable

## Phase 8: dashboard and economics

Goal: make actual performance visible by season and month.

Deliverables:
- year/season selector
- month selector
- season Habu captures
- current-month Habu captures
- exploration nights
- distance
- fuel cost
- capture revenue
- other expenses
- profit
- efficiency metrics
- optional size/sex/area/time breakdowns

Exit criteria:
- changing year/month changes all relevant metrics consistently
- only Habu captures contribute to capture totals/revenue

## Phase 9: vehicle, fuel, and cost automation

Goal: compute session economics automatically.

Deliverables:
- multiple vehicle profiles
- fuel type
- km/L
- dated fuel-price history
- measured fuel-economy history optional
- session vehicle selection/default
- automatic estimated fuel cost from GPS distance
- manual other expenses

Exit criteria:
- historical session cost is reproducible from values effective at that time
- future fuel-price changes do not silently rewrite old records

## Phase 10: scoring and training-data generation

Goal: close the learning loop.

Deliverables:
- prediction vs actual scoring
- fixed-window HIT/MISS
- Area Recall
- Road Zone Recall
- official 100 m +/-20 min KPI
- searched passage -> non-capture observation generation
- reliability weighting for GPX reconstruction classes
- road-event temporal uncertainty preserved in features
- canonical training export

Exit criteria:
- session end produces auditable training-ready data
- discovery-time leakage from road events is tested and prevented

## Phase 11: sync and model publishing

Goal: safe online/offline synchronization and validated model updates.

Deliverables:
- local-first sync queue
- conflict policy
- cloud backup
- heavy retraining pipeline integration
- versioned validated model package publication
- mobile update/download flow
- rollback to previous model package

Exit criteria:
- failed sync never deletes local records
- model version is auditable and rollback-capable

## Phase 12: field hardening and v1 release

Goal: prove the entire loop on real iPhone and Android hardware.

Test scenarios:
- full online
- fully offline
- network disappears mid-session
- app background/foreground
- device restart/recovery
- GPS interruption
- long session battery test
- map package missing/stale
- model package stale
- duplicate sync attempt
- conflicting edit

v1 release gate:
- all product-spec acceptance criteria pass on both platforms
- no known data-loss path
- no route geometry fabrication
- no Himehabu capture-count contamination
- no road-event discovery-time leakage

## AI employee execution rules

1. Work phase-by-phase unless an item is independently unblocked.
2. Prefer 1 concern per PR.
3. Add tests with every invariant/bug fix.
4. Do not add paid services without owner approval.
5. Do not publish to App Store/Google Play without owner approval.
6. Do not destructively migrate canonical Habu data without explicit migration plan and backup.
7. Reuse existing Python/GIS/model assets rather than duplicate them in Dart.
8. Keep mobile offline-first and make cloud functionality additive.
9. Update implementation state after every merged PR.
10. If a decision is reversible and covered by this spec, proceed without a Human Gate.
