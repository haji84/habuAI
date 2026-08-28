from __future__ import annotations

from .hardening import apply_hardening


def apply_runtime_fixes(pipeline) -> None:
    """Apply the hardened OSM 10 m pipeline before execution."""
    apply_hardening(pipeline)
