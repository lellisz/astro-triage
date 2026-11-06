"""Protocol constants and plot config shared by the dashboard views (no Streamlit import)."""

from __future__ import annotations

from typing import Final

# Plotly config that hides the mode bar; one object for every st.plotly_chart call.
NO_BAR: Final = {"displayModeBar": False}

# ADR of the deep-learning comparison (docs/adr/0006-...).
ADR_NUMBER: Final = 6

# Fallback for the "primary_completeness" key of delta.json; ADR 0006 sec. 4.4 (F5) fixes C* = 0,80.
FIXED_COMPLETENESS: Final = 0.80

# Release the dashboard opens by default (the full-sample evaluation).
DEFAULT_RELEASE_NAME: Final = "full_v0"

# Error rate above which a bar, point or sky region counts as "high" (highlight colour, "passes").
ERROR_HIGH: Final = 0.10
