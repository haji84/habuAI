# Habu AI Field v1 Architecture

## System shape

### Mobile app

Flutter app is the field surface and remains usable offline.

Modules:
- Tonight
- Map
- Record
- Dashboard
- Settings
- Exploration session manager
- Local database
- Sync queue
- Local inference adapter
- Route planner adapter

### Existing Habu AI core

Reuse current Python/GIS pipeline for:
- GPX ingestion
- OSM 10 m segmentation
- map matching
- exposure/non-capture generation
- feature engineering
- model training/evaluation
- nightly prediction
- route scoring inputs

Do not duplicate canonical model logic in Dart unless required for mobile inference. Prefer exported/versioned model artifacts and shared schemas.

### Cloud/desktop services

Non-field critical workloads:
- heavy model retraining
- canonical dataset QA
- model validation
- model package publishing
- map-package generation/update
- backup/sync

The mobile app must continue operating if these services are temporarily unavailable.

## Data flow

Pre-search:

`canonical data -> training/evaluation -> validated model -> nightly features -> forecast snapshot -> road scores -> route optimizer -> A/B/C route plan -> mobile cache`

During search:

`GPS + field observations -> local DB -> searched-road state -> lightweight re-score -> remaining-route update`

Post-search:

`session close -> GPX finalize -> non-capture passages -> prediction scoring -> dashboard metrics -> sync -> canonical QA -> retraining candidate`

## Source-of-truth rules

- Local unsynced field records are authoritative until confirmed synced.
- Canonical historical/training data remains under the Python/GIS data pipeline.
- User-verified road classification overrides inferred/OSM classification.
- Pre-search forecast snapshot becomes immutable when exploration starts.
- Model packages are versioned and rollback-capable.

## Offline packages

A mobile offline bundle should contain:
- base/road map package for target area
- road-network graph or equivalent routing representation
- latest validated model/inference artifact where supported
- feature metadata/version
- cached environmental forecast
- tide/moon/sunset data or calculation support
- schema versions

Every package must expose version and last-updated time.

## Sync design

Use an append-safe local outbox pattern.

Each local record has:
- stable UUID
- created_at
- updated_at
- device_id
- sync_state
- schema_version

Sync states:
- local_only
- pending
- synced
- conflict
- failed_retryable

Never delete a local record merely because remote sync failed.

## Route optimization boundary

Map geometry and connectivity are GIS concerns. Prediction and route value are model concerns.

The optimizer consumes:
- graph geometry/connectivity
- segment IDs
- time-dependent scores
- current location/time
- searched state
- constraints
- remaining time/distance budget

The optimizer outputs:
- connected segment sequence
- route geometry reference
- route metrics
- route class A/B/C

It must never synthesize arbitrary coordinates between disconnected road segments.

## Temporal semantics

Capture and direct sighting timestamps are observation times.

Road event `discovered_at` is discovery time only. Its actual presence/activity time is interval-censored/unknown unless separately observed. This distinction must survive storage, feature engineering, training, scoring, and export.

## Security/privacy baseline

- No secrets committed to repository.
- Device identifiers should be app-scoped, not hardware-global where avoidable.
- Location data is sensitive operational data and should not be exposed publicly by default.
- Production sync endpoints require authenticated access.
- Debug logs must avoid dumping full private location history unnecessarily.

## Cross-platform acceptance

Any platform-specific capability must have equivalent behavior or explicit degradation on iOS and Android. A feature is not complete because it works on only one platform unless the roadmap marks it platform-specific.
