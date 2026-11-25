"""Protocol constants shared by the dashboard text builders and the web exporter."""

from __future__ import annotations

from typing import Final

# ADR of the deep-learning comparison (docs/adr/0006-...).
ADR_NUMBER: Final = 6

# Fallback for the "primary_completeness" key of delta.json; ADR 0006 sec. 4.4 (F5) fixes C* = 0,80.
FIXED_COMPLETENESS: Final = 0.80

# Release the exporter opens by default (the full-sample evaluation).
DEFAULT_RELEASE_NAME: Final = "full_v0"

# Error rate above which a bar, point or sky region counts as "high" (highlight colour, "passes").
ERROR_HIGH: Final = 0.10
