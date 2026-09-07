# Habu AI Field v1 - Configurable Field Options

## Goal

Allow the owner to extend and tune field-entry choices from the app Settings screen without requiring an app release or code change.

The field workflow remains fast and cascading, but selectable values are data-driven rather than hard-coded wherever practical.

## Scope

Settings must allow the user to manage selectable values used by field entry, including at minimum:

- species
- sighting status / road-event status
- discovery location
- behavior
- road-surface condition
- count presets
- size quick-select presets
- optional notes/templates where applicable

## Required capabilities

For configurable option groups, the Settings UI must support:

1. Add a new option.
2. Rename a user-created option.
3. Reorder options.
4. Hide/unhide options from field entry without deleting historical data.
5. Mark frequently used options so they can appear first.
6. Restore the default ordering/options.
7. Export/sync configuration as part of normal app backup/sync.

## Data integrity rules

Configuration must not break historical records or model semantics.

Each option must use a stable internal ID. Display labels may change, but existing observations must continue to reference the same stable ID.

Default/canonical options and user-created options must be distinguishable.

Recommended fields:

- `option_id`
- `group_key`
- `canonical_key` nullable
- `display_label`
- `is_builtin`
- `is_enabled`
- `sort_order`
- `is_favorite`
- `created_at`
- `updated_at`

Deleting an option that is already referenced by observations is prohibited. The UI should offer Hide instead.

## Canonical-vs-custom semantics

Some values have model or accounting meaning and therefore cannot be freely redefined.

Hard invariants remain fixed:

- Capture event type means Habu captured by the user only.
- Himehabu never counts as Habu capture revenue/count.
- Sighting may include any species.
- Road event may include any species.
- Road-event discovery time is not exact activity/presence time.
- The exploration-night 07:00 boundary is not configurable from this option manager.

Canonical values used directly by model features should retain a `canonical_key`. A custom value may either:

- map to an existing canonical key, or
- remain `custom/unmapped` until explicitly mapped.

The model pipeline must not silently reinterpret an unmapped custom option as a known canonical class.

## Field UX

Field entry should continue to advance immediately after a selection when the next choice is deterministic.

Settings changes should take effect locally immediately and must work offline.

The user should be able to prioritize the most frequently used choices so the field UI stays fast even as the option catalog grows.

Long catalogs should support favorites/recent values and search where useful.

## Acceptance criteria

- A user can add a new selectable field value from Settings and use it in a new observation without code changes.
- Reordering changes field-entry order.
- Hiding removes an option from new entry but historical records still render correctly.
- Renaming a custom label does not orphan historical rows.
- Built-in semantic invariants cannot be changed into contradictory meanings.
- Custom unmapped values do not silently corrupt model training features.
- All operations work offline and sync later.
