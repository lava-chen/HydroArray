"""
Basin Dataset Base Class for HydroArray

Provides base functionality for basin/watershed datasets such as:
- CAMELS-US (Contiguous USA)
- CAMELS-CN (China)
- CAMELS-AUS (Australia)
- LamaH (Central Europe)
- Caravan (Global)

These datasets contain discrete watershed data with:
- Meteorological forcing data
- Streamflow observations
- Catchment attributes

Example:
    >>> from HydroArray.datasets.basin import BasinDataset

    >>> # Create from registry
    >>> camels = BasinDataset.create("camels_us", data_dir="path/to/camels")

    >>> # Get basin data
    >>> basin = camels.get_basin("01013500")
    >>> print(basin.forcings.keys())
    >>> print(basin.discharge)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class BasinData:
    """
    Container for single basin data.

    Attributes
    ----------
    basin_id : str
        Unique basin identifier
    geometry : Any
        Basin geometry (shapely Polygon)
    area_km2 : float
        Basin area in square kilometers
    forcings : Dict[str, np.ndarray]
        Meteorological forcing data arrays
    forcing_vars : List[str]
        List of forcing variable names
    discharge : np.ndarray
        Observed discharge/time series
    discharge_dates : np.ndarray
        Dates for discharge observations
    attributes : Dict[str, float]
        Static basin attributes
    """
    basin_id: str
    geometry: Any
    area_km2: float
    forcings: Dict[str, np.ndarray]
    forcing_vars: List[str]
    discharge: np.ndarray
    discharge_dates: np.ndarray
    attributes: Dict[str, float]


class BasinDataset(ABC):
    """
    Abstract base class for basin/watershed datasets.

    Basin datasets contain data for discrete watersheds, each with
    its own forcing data, discharge observations, and attributes.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to the dataset directory
    basins : List[str], optional
        List of basin IDs to load. If None, load all available basins.

    Attributes
    ----------
    data_dir : Path
        Path to the dataset directory
    basin_ids : List[str]
        List of available basin IDs
    forcing_vars : List[str]
        List of available forcing variable names
    target_var : str
        Name of the target variable (usually discharge)
    """

    _registry: Dict[str, type] = {}

    def __init__(
        self,
        data_dir: Union[str, Path],
        basins: Optional[List[str]] = None,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.basins = basins
        self._basin_cache: Dict[str, BasinData] = {}
        self._load_metadata(**kwargs)

    @abstractmethod
    def _load_metadata(self, **kwargs):
        """Load dataset metadata (available basins, variables, attributes)."""
        pass

    @abstractmethod
    def _load_basin(self, basin_id: str) -> BasinData:
        """
        Load data for a single basin.

        Parameters
        ----------
        basin_id : str
            Basin identifier

        Returns
        -------
        BasinData
            Container with basin data
        """
        pass

    def get_basin(self, basin_id: str, use_cache: bool = True) -> BasinData:
        """
        Get data for a single basin.

        Parameters
        ----------
        basin_id : str
            Basin identifier
        use_cache : bool
            Whether to use cached data (default: True)

        Returns
        -------
        BasinData
            Container with basin data

        Example:
            >>> basin = camels.get_basin("01013500")
            >>> print(basin.area_km2)
            >>> print(basin.forcings.keys())
        """
        if basin_id not in self.basin_ids:
            raise ValueError(f"Unknown basin ID: {basin_id}")

        if use_cache and basin_id in self._basin_cache:
            return self._basin_cache[basin_id]

        basin_data = self._load_basin(basin_id)
        if use_cache:
            self._basin_cache[basin_id] = basin_data

        return basin_data

    def get_basins(
        self,
        basin_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, BasinData]:
        """
        Get data for multiple basins.

        Parameters
        ----------
        basin_ids : List[str], optional
            List of basin IDs. If None, returns all basins.
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        Dict[str, BasinData]
            Dictionary mapping basin_id to BasinData

        Example:
            >>> basins = camels.get_basins(["01013500", "01015000", "01020000"])
        """
        if basin_ids is None:
            basin_ids = self.basin_ids

        return {
            bid: self.get_basin(bid, use_cache=use_cache)
            for bid in basin_ids
        }

    def get_forcing(
        self,
        basin_id: str,
        forcing_source: Optional["RasterDataset"] = None,
        variables: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get forcing data for a basin.

        Parameters
        ----------
        basin_id : str
            Basin identifier
        forcing_source : RasterDataset, optional
            External forcing source to extract from (e.g., GPM)
        variables : List[str], optional
            Variables to get. If None, returns all available.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forcing variable names to data arrays
        """
        basin = self.get_basin(basin_id)

        if forcing_source is not None and variables is not None:
            # Extract from raster source
            return forcing_source.extract_basin(
                geometry=basin.geometry,
                variables=variables
            )

        if variables is None:
            return basin.forcings

        return {v: basin.forcings[v] for v in variables if v in basin.forcings}

    @property
    @abstractmethod
    def basin_ids(self) -> List[str]:
        """Return list of available basin IDs."""
        pass

    @property
    @abstractmethod
    def forcing_vars(self) -> List[str]:
        """Return list of available forcing variable names."""
        pass

    @property
    @abstractmethod
    def target_var(self) -> str:
        """Return name of the target variable."""
        pass

    @property
    @abstractmethod
    def attribute_vars(self) -> List[str]:
        """Return list of available basin attribute names."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{self.data_dir}', "
            f"n_basins={len(self.basin_ids)}, "
            f"forcings={self.forcing_vars}"
            f")"
        )

    @classmethod
    def register(cls, name: str):
        """
        Register a basin dataset class.

        Example:
            >>> @BasinDataset.register("camels_us")
            >>> class CAMELSUSDataset(BasinDataset):
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
    ) -> "BasinDataset":
        """
        Create a basin dataset instance from registry.

        Parameters
        ----------
        name : str
            Dataset name (e.g., 'camels_us', 'camels_cn', 'camels_aus')
        data_dir : Union[str, Path]
            Path to the dataset directory
        **kwargs
            Additional arguments passed to dataset constructor

        Returns
        -------
        BasinDataset
            Instance of the requested dataset class

        Example:
            >>> camels_us = BasinDataset.create("camels_us", data_dir="path/to/camels")
            >>> camels_cn = BasinDataset.create("camels_cn", data_dir="path/to/camels_cn")
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown basin dataset: '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name_lower](data_dir=data_dir, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of available basin dataset names."""
        return list(cls._registry.keys())
