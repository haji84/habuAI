# Habu AI Field v1 Spec Addendum: Configurable Field Options

This addendum is normative and extends `HABU_AI_FIELD_V1_SPEC.md`.

## Requirement

Field-entry selection catalogs must be configurable from the app Settings screen instead of being permanently hard-coded.

The app must allow adding, reordering, hiding/unhiding, favoriting, and restoring selectable values for field workflows while preserving historical data integrity and model semantics.

The detailed contract is defined in `HABU_AI_FIELD_V1_CONFIGURABLE_FIELD_OPTIONS.md` and must be treated as part of the Habu AI Field v1 acceptance criteria.

## Non-negotiable guardrails

Configurable labels/options must never allow the user to redefine the following domain invariants:

- Capture counts/revenue include Habu only.
- Himehabu never becomes a Habu capture through configuration.
- Sighting supports all species.
- Road events support all species.
- Road-event discovery time is not exact presence/activity time.
- Historical observations must never be orphaned when an option is renamed, hidden, or reordered.
- Unknown custom values must not be silently mapped to an existing model feature class.
