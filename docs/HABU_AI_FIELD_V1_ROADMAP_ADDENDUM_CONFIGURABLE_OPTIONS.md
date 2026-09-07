# Habu AI Field v1 Roadmap Addendum: Configurable Field Options

This addendum extends `HABU_AI_FIELD_V1_ROADMAP.md`.

## Phase placement

Implement the configurable field-option system as part of the early offline-first app/data foundation, before the field-entry workflow is considered complete.

Required implementation work:

1. Add a local option-catalog schema with stable IDs, built-in/custom distinction, ordering, enabled state, favorites, and optional canonical mapping.
2. Seed the canonical default option sets.
3. Build Settings UI for add, rename custom option, reorder, hide/unhide, favorite, and restore defaults.
4. Make field-entry sheets read from the local option catalog rather than hard-coded arrays.
5. Preserve historical rendering when options change.
6. Add sync/export support for option configuration.
7. Add tests preventing destructive deletion of referenced values and silent canonical remapping.

## Definition of done

The field-entry phase is not complete until a new custom selectable value can be created entirely from Settings, used offline in a field record, survive app restart, remain readable after hide/rename, and sync without altering model/accounting invariants.
