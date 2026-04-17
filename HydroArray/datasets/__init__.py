"""
HydroArray Datasets Module

Provides dataset classes for hydrological data:

Raster Datasets (gridded/spatial data):
    - GPM, TRMM (satellite precipitation)
    - ERA5, MERRA2 (reanalysis)
    - CLDAS (China Land Data Assimilation)

Basin Datasets (watershed/point data):
    - CAMELS-US (671 USA basins)
    - CAMELS-CN (China basins)
    - CAMELS-AUS (Australia basins)
    - LamaH (Central Europe)
    - Caravan (Global)

Multi-Source Dataset:
    - Combines multiple sources for ML training

Example:
    >>> from HydroArray.datasets import BasinDataset, RasterDataset

    >>> # List available datasets
    >>> print(BasinDataset.list_available())
    >>> print(RasterDataset.list_available())

    >>> # Create datasets
    >>> camels = BasinDataset.create("camels_us", data_dir="path/to/camels")
    >>> gpm = RasterDataset.create("gpm", data_dir="path/to/gpm")

    >>> # Get basin data
    >>> basin = camels.get_basin("01013500")
"""

# Base classes
from HydroArray.datasets.base import (
    HydroDataset,
    Subset,
    create_dataloader,
    DataTransformer,
)

# Raster datasets
from HydroArray.datasets.raster import RasterDataset
from HydroArray.datasets.raster.base import RasterDataset

# Basin datasets
from HydroArray.datasets.basin import BasinDataset, BasinData
from HydroArray.datasets.basin.base import BasinDataset, BasinData

# CAMELS-US
from HydroArray.datasets.basin.camels_us import (
    CAMELSUSDataset,
    CAMELS_US_FORCING_VARS,
    CAMELS_US_TARGET_VAR,
    get_camels_us_basins,
    load_camels_us_attributes,
)

# River dataset for streamflow forecasting
from HydroArray.datasets.river import (
    RiverDataset,
    Seq2SeqLSTM,
)

# Multi-source dataset
from HydroArray.datasets.multi_source import MultiSourceDataset

# Moving MNIST (for video prediction benchmarking)
from HydroArray.datasets.moving_mnist import MovingMNISTDataset

__all__ = [
    # Base classes
    "HydroDataset",
    "Subset",
    "create_dataloader",
    "DataTransformer",
    # Raster
    "RasterDataset",
    # Basin
    "BasinDataset",
    "BasinData",
    # CAMELS-US
    "CAMELSUSDataset",
    "CAMELS_US_FORCING_VARS",
    "CAMELS_US_TARGET_VAR",
    "get_camels_us_basins",
    "load_camels_us_attributes",
    # River dataset
    "RiverDataset",
    "Seq2SeqLSTM",
    # Multi-source
    "MultiSourceDataset",
    # MovingMNIST
    "MovingMNISTDataset",
]
