"""
Basin Dataset Module for HydroArray

Provides dataset classes for watershed/basin data including:
- CAMELS-US (Contiguous USA, 671 basins)
- CAMELS-CN (China)
- CAMELS-AUS (Australia)
- LamaH (Central Europe)
- Caravan (Global)

Example:
    >>> from HydroArray.datasets.basin import BasinDataset

    >>> # List available datasets
    >>> print(BasinDataset.list_available())

    >>> # Create a dataset
    >>> camels = BasinDataset.create("camels_us", data_dir="path/to/camels")

    >>> # Get basin data
    >>> basin = camels.get_basin("01013500")
    >>> print(basin.area_km2)
    >>> print(basin.forcings.keys())
"""

from HydroArray.datasets.basin.base import BasinDataset, BasinData

__all__ = [
    "BasinDataset",
    "BasinData",
]
