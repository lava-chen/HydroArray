"""
Raster Dataset Base Class for HydroArray

Provides base functionality for raster/gridded datasets such as:
- GPM (Global Precipitation Measurement)
- TRMM (Tropical Rainfall Measuring Mission)
- ERA5 (ECMWF Reanalysis 5)
- CLDAS (China Land Data Assimilation System)

These datasets contain spatial continuous data on regular grids,
independent of basin boundaries.

Example:
    >>> from HydroArray.datasets.raster import RasterDataset, RasterDataset

    >>> # Create from registry
    >>> gpm = RasterDataset.create("gpm", data_dir="path/to/gpm")

    >>> # Check available variables
    >>> print(gpm.variables)

    >>> # Extract basin mean
    >>> basin_precip = gpm.extract_basin(geometry, variables=["precipitation"])
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


class RasterDataset(ABC):
    """
    Abstract base class for raster/gridded datasets.

    Raster datasets are spatial continuous data on regular grids,
    such as satellite precipitation or reanalysis data. They are
    independent of basin boundaries and can be used to extract
    forcing data for any watershed.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to the dataset directory
    mode : str
        Data loading mode ('local', 'opendap', etc.)

    Attributes
    ----------
    data_dir : Path
        Path to the dataset directory
    variables : List[str]
        Available variables in this dataset
    spatial_bounds : tuple
        (lon_min, lon_max, lat_min, lat_max)
    temporal_bounds : tuple
        (start_date, end_date)
    resolution : tuple
        (lon_resolution, lat_resolution)
    """

    _registry: Dict[str, type] = {}

    def __init__(
        self,
        data_dir: Union[str, Path],
        mode: str = "local",
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self._data = None
        self._load_metadata(**kwargs)

    @abstractmethod
    def _load_metadata(self, **kwargs):
        """Load dataset metadata (variables, bounds, resolution)."""
        pass

    @abstractmethod
    def _load_data(
        self,
        variables: List[str],
        time_range: Optional[tuple] = None,
        bounding_box: Optional[tuple] = None,
    ) -> "xr.Dataset":
        """
        Load data for specified variables and conditions.

        Parameters
        ----------
        variables : List[str]
            Variables to load
        time_range : tuple, optional
            (start_date, end_date) tuple
        bounding_box : tuple, optional
            (lon_min, lon_max, lat_min, lat_max)

        Returns
        -------
        xr.Dataset
            Loaded data
        """
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """Return list of available variables."""
        pass

    @property
    @abstractmethod
    def spatial_bounds(self) -> tuple:
        """Return spatial bounds (lon_min, lon_max, lat_min, lat_max)."""
        pass

    @property
    @abstractmethod
    def temporal_bounds(self) -> tuple:
        """Return temporal bounds (start_date, end_date)."""
        pass

    def extract_basin(
        self,
        geometry: Any,
        variables: List[str],
        method: str = "area_weighted",
        time_range: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Extract basin-averaged data from raster dataset.

        Parameters
        ----------
        geometry : Union[Polygon, List[Point]]
            Basin geometry (shapely Polygon or list of boundary points)
        variables : List[str]
            Variables to extract
        method : str
            Extraction method: 'area_weighted', 'mean', 'median'
        time_range : tuple, optional
            (start_date, end_date) tuple

        Returns
        -------
        np.ndarray
            Extracted data of shape (time, variables)

        Example:
            >>> from shapely.geometry import Polygon
            >>> basin_geom = Polygon([(-72, 42), (-71, 42), (-71, 43), (-72, 43)])
            >>> precip = gpm.extract_basin(
            ...     geometry=basin_geom,
            ...     variables=["precipitation"],
            ...     method="area_weighted"
            ... )
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for extract_basin. Install with: pip install xarray")

        data = self._load_data(variables, time_range)

        if method == "area_weighted":
            return self._extract_area_weighted(data, geometry, variables)
        elif method == "mean":
            return self._extract_mean(data, geometry, variables)
        elif method == "median":
            return self._extract_median(data, geometry, variables)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def _extract_area_weighted(
        self,
        data: "xr.Dataset",
        geometry: Any,
        variables: List[str],
    ) -> np.ndarray:
        """Extract using area-weighted mean."""
        # This is a placeholder - actual implementation would use rasterio/rioxarray
        # for proper masking and area calculation
        raise NotImplementedError("Area-weighted extraction requires proper geometry support")

    def _extract_mean(
        self,
        data: "xr.Dataset",
        geometry: Any,
        variables: List[str],
    ) -> np.ndarray:
        """Extract using simple mean over geometry bounds."""
        raise NotImplementedError("Mean extraction requires proper geometry support")

    def _extract_median(
        self,
        data: "xr.Dataset",
        geometry: Any,
        variables: List[str],
    ) -> np.ndarray:
        """Extract using median over geometry bounds."""
        raise NotImplementedError("Median extraction requires proper geometry support")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{self.data_dir}', "
            f"variables={self.variables}, "
            f"bounds={self.spatial_bounds}"
            f")"
        )

    @classmethod
    def register(cls, name: str):
        """
        Register a raster dataset class.

        Example:
            >>> @RasterDataset.register("gpm")
            >>> class GPMSDataset(RasterDataset):
            ...     pass
        """
        def decorator(dataset_cls):
            cls._registry[name.lower()] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        data_dir: Union[str, Path],
        **kwargs
    ) -> "RasterDataset":
        """
        Create a raster dataset instance from registry.

        Parameters
        ----------
        name : str
            Dataset name (e.g., 'gpm', 'era5', 'trmm', 'cldas')
        data_dir : Union[str, Path]
            Path to the dataset directory
        **kwargs
            Additional arguments passed to dataset constructor

        Returns
        -------
        RasterDataset
            Instance of the requested dataset class

        Example:
            >>> gpm = RasterDataset.create("gpm", data_dir="path/to/gpm")
            >>> era5 = RasterDataset.create("era5", data_dir="path/to/era5")
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown raster dataset: '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name_lower](data_dir=data_dir, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of available raster dataset names."""
        return list(cls._registry.keys())
