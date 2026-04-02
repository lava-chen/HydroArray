"""
Hydrological Observation Module

Handles observed hydrological data including:
- Water level processing
- Discharge calculations
- Frequency analysis

Example:
    >>> from HydroArray.domain.observation import to_daily, frequency
"""

from HydroArray.domain.observation.waterlevel import (
    to_daily,
    frequency,
    ARITHMETIC_DAILY_VARIATION_THRESHOLD,
)

__all__ = [
    "to_daily",
    "frequency",
    "ARITHMETIC_DAILY_VARIATION_THRESHOLD",
]
