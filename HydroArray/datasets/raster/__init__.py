"""
Raster Dataset Module for HydroArray

Provides dataset classes for gridded/raster data sources including:
- GPM (Global Precipitation Measurement)
- TRMM (Tropical Rainfall Measuring Mission)
- ERA5 (ECMWF Reanalysis 5th Generation)
- MERRA2 (Modern-Era Retrospective Analysis for Research and Applications)
- CLDAS (China Land Data Assimilation System)

Example:
    >>> from HydroArray.datasets.raster import RasterDataset

    >>> # List available datasets
    >>> print(RasterDataset.list_available())

    >>> # Create a dataset
    >>> gpm = RasterDataset.create("gpm", data_dir="path/to/gpm")

    >>> # Extract basin data
    >>> precip = gpm.extract_basin(geometry, variables=["precipitation"])
"""

from HydroArray.datasets.raster.base import RasterDataset

__all__ = [
    "RasterDataset",
]
