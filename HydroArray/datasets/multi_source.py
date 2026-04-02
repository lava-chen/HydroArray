"""
Multi-Source Dataset for HydroArray

Provides functionality to combine multiple data sources:
- Basin datasets (CAMELS, etc.) for watershed data
- Raster datasets (GPM, ERA5, etc.) for meteorological forcing
- External forcing sources

This enables:
- Multi-source data fusion
- Downscaling/counseling learning
- Gap-filling with external data

Example:
    >>> from HydroArray.datasets import MultiSourceDataset, BasinDataset, RasterDataset

    >>> # Create data sources
    >>> camels = BasinDataset.create("camels_us", data_dir="path/to/camels")
    >>> gpm = RasterDataset.create("gpm", data_dir="path/to/gpm")
    >>> era5 = RasterDataset.create("era5", data_dir="path/to/era5")

    >>> # Create multi-source dataset
    >>> dataset = MultiSourceDataset(
    ...     basins=["01013500", "01015000"],
    ...     forcing_sources={
    ...         "precipitation": gpm,
    ...         "temperature": era5,
    ...     },
    ...     target_source=camels,
    ...     variables=["prcp", "tmax", "tmin"],
    ...     target_var="QObs",
    ...     seq_len=365,
    ...     pred_len=1
    ... )

    >>> # Use with PyTorch DataLoader
    >>> from HydroArray.datasets import create_dataloader
    >>> loader = create_dataloader(dataset, batch_size=32)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from HydroArray.datasets.basin import BasinDataset
    from HydroArray.datasets.raster import RasterDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class MultiSourceDataset(Dataset):
    """
    Dataset that combines multiple data sources for ML training.

    This dataset allows combining:
    - Basin datasets (CAMELS, etc.) for observation data
    - Raster datasets (GPM, ERA5, etc.) for forcing data
    - External data sources

    Parameters
    ----------
    basins : List[str]
        List of basin IDs to include
    forcing_sources : Dict[str, RasterDataset]
        Dictionary mapping variable names to raster datasets
    target_source : BasinDataset
        Basin dataset providing target observations
    variables : List[str]
        List of forcing variables to use
    target_var : str
        Target variable name
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction sequence length
    period : str
        Data period ('train', 'val', 'test')

    Attributes
    ----------
    basins : List[str]
        List of basin IDs
    variables : List[str]
        List of forcing variable names
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction sequence length

    Example:
        >>> dataset = MultiSourceDataset(
        ...     basins=["01013500", "01015000"],
        ...     forcing_sources={"precipitation": gpm, "temperature": era5},
        ...     target_source=camels,
        ...     variables=["prcp", "tmax", "tmin"],
        ...     target_var="QObs",
        ...     seq_len=365,
        ...     pred_len=1
        ... )
        >>> x, y = dataset[0]
        >>> print(x.shape)  # (365, 3) - 3 variables
    """

    def __init__(
        self,
        basins: List[str],
        forcing_sources: Dict[str, "RasterDataset"],
        target_source: "BasinDataset",
        variables: List[str],
        target_var: str,
        seq_len: int = 365,
        pred_len: int = 1,
        period: str = 'train',
    ):
        if not HAS_DATASETS:
            raise ImportError("HydroArray datasets module not available")

        self.basins = basins
        self.forcing_sources = forcing_sources
        self.target_source = target_source
        self.variables = variables
        self.target_var = target_var
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period

        # Build samples index
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[str, int]]:
        """
        Build index of samples for efficient access.

        Returns:
            List of (basin_id, start_idx) tuples
        """
        samples = []

        for basin_id in self.basins:
            try:
                basin_data = self.target_source.get_basin(basin_id)

                if basin_data.discharge is None:
                    continue

                n_timesteps = len(basin_data.discharge)
                n_samples = n_timesteps - self.seq_len - self.pred_len + 1

                if n_samples > 0:
                    for start_idx in range(n_samples):
                        samples.append((basin_id, start_idx))

            except Exception as e:
                print(f"Warning: Could not load basin {basin_id}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (input_sequence, target_sequence)
        """
        basin_id, start_idx = self.samples[idx]

        basin_data = self.target_source.get_basin(basin_id)

        # Build forcing input
        forcing_data = []
        for var in self.variables:
            if var in basin_data.forcings:
                var_data = basin_data.forcings[var][start_idx:start_idx + self.seq_len]
            else:
                # Try to get from raster sources
                var_data = self._get_from_raster(basin_id, var, start_idx)

            forcing_data.append(var_data)

        forcing_array = np.stack(forcing_data, axis=-1)  # (seq_len, n_vars)

        # Build target
        target_data = basin_data.discharge[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]

        return (
            torch.FloatTensor(forcing_array),
            torch.FloatTensor(target_data).unsqueeze(-1)
        )

    def _get_from_raster(
        self,
        basin_id: str,
        var: str,
        start_idx: int
    ) -> np.ndarray:
        """
        Get variable from raster source.

        This is a placeholder - actual implementation would
        extract from the raster dataset based on basin geometry.
        """
        return np.zeros(self.seq_len)

    def get_basin(self, idx: int) -> str:
        """Get basin ID for a sample."""
        return self.samples[idx][0]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for normalization."""
        all_x = []
        all_y = []

        for i in range(min(len(self), 1000)):  # Sample for efficiency
            x, y = self[i]
            all_x.append(x.numpy())
            all_y.append(y.numpy())

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return {
            'inputs': {
                'mean': all_x.mean(axis=0).tolist(),
                'std': all_x.std(axis=0).tolist(),
            },
            'targets': {
                'mean': float(all_y.mean()),
                'std': float(all_y.std()),
            }
        }

    def __repr__(self) -> str:
        return (
            f"MultiSourceDataset("
            f"n_basins={len(self.basins)}, "
            f"n_samples={len(self)}, "
            f"variables={self.variables}, "
            f"seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}"
            f")"
        )
