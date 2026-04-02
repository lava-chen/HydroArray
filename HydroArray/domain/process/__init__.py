"""
Hydrological Process Module

Implements core hydrological processes including:
- Runoff generation models (saturation excess, source separation)
- Evaporation models (three-layer evaporation)
- Cross-section calculations (channel geometry)

Example:
    >>> from HydroArray.domain.process import saturation_excess_runoff, three_layer_evaporation

    >>> # Calculate runoff
    >>> result = saturation_excess_runoff(data_df, WUM=10, WLM=80, WDM=100, ...)

    >>> # Calculate evaporation
    >>> EU, EL, ED = three_layer_evaporation(Ep=5.0, WU=5.0, WL=50.0, P=10.0, WLM=80.0)
"""

from HydroArray.domain.process.runoff import (
    saturation_excess_runoff,
    two_source_runoff_separation,
    three_source_runoff_separation,
)

from HydroArray.domain.process.evaporation import (
    three_layer_evaporation,
)

from HydroArray.domain.process.crosssection import (
    calculate_channel_section_detailed,
    calculate_cross_section_area,
)

__all__ = [
    "saturation_excess_runoff",
    "two_source_runoff_separation",
    "three_source_runoff_separation",
    "three_layer_evaporation",
    "calculate_channel_section_detailed",
    "calculate_cross_section_area",
]
